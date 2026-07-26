"""
rerunning/ensembles/context_v1_confidence_inline.py

PHASE 1 of 2: Context-aware V1 (hand-written rules) + word-level
confidence flagging using INLINE markers.

CONCURRENT VERSION: same Phase A (fast sequential prep) / Phase B
(concurrent Ollama calls via thread pool) split as the other confidence
scripts. See src/concurrent_ollama.py's docstring for the required
OLLAMA_NUM_PARALLEL server-side setting.

NOTE: kept as a separate file (context_v1_confidence_inline_concurrent.py)
rather than overwriting the original while your local CommonVoice run is
still in progress - swap this in as the real context_v1_confidence_inline.py
once that run finishes, before using it for English Dialects/EdAcc on the HPC.

NOT CHANGED - STILL NEEDS YOUR REVIEW: the named-entity trust rule
("WhisperX is more reliable on named entities") is unchanged, per your
earlier decision to run with it as-is and revisit only if results look
off for that specific rule.

Usage:
    python rerunning/ensembles/context_v1_confidence_inline.py --dataset commonvoice --split dev --percentile 20
"""

import json
import os
import re
import argparse
from jiwer import wer
from ollama import Client

from src.judge import normalise, is_tag_only
from src.selector import (
    find_canonical_file, OLLAMA_MODELS, check_selector_available, load_samples,
)
from src.splits import get_indices_for_split
from src.concurrent_ollama import run_concurrent

NEW_OUTPUT_DIR = "writeup_results/ensembles/context_v1_confidence_inline"
OLD_OUTPUT_DIR = "results/combinations_v2judge/context_v1_confidence_inline"
OLLAMA_HOST    = "http://localhost:11434"
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]
CONFIDENCE_MODELS  = ["qwen", "whisperx", "parakeet"]   # qwen added
DEFAULT_PERCENTILE = 20
DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))
# tune via --max-workers or the ENSEMBLE_MAX_WORKERS env var (roughly match
# OLLAMA_NUM_PARALLEL on the server) - NOT a hardcoded constant, since your
# Mac (limited unified memory) and the HPC (dedicated GPU memory) need very
# different values, and this file is shared between both via git.

# NOTE: rule content unchanged (named-entity trust etc.) - review per
# context_v1.py's discussion before editing. The confidence-weighting
# rules ARE new, making Qwen's confidence do something now it's available.
SELECTOR_PROMPT = """You are correcting an ASR transcript. You are given three transcripts of the same audio from different models.

Some transcripts include inline [LOW-CONF: word] markers — these indicate words the model itself was uncertain about. Treat words marked this way as MORE likely to be wrong. Do NOT include the [LOW-CONF: ...] markers in your final output — write the plain word only.

TRANSCRIPT A (base — use this as your starting point):
{qwen}

TRANSCRIPT B (WhisperX):
{whisperx}

TRANSCRIPT C (Parakeet):
{parakeet}

Your task: return Transcript A with targeted corrections where needed.

GENERAL RULE — applies to all words except named entities:
Only change a word if BOTH B and C disagree with A and agree with each other on the same alternative. If only one of B or C disagrees with A, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

CONFIDENCE WEIGHTING:
- If a disagreeing word in B or C is marked [LOW-CONF: ...], treat that disagreement as WEAKER evidence.
- If the word in A itself is marked [LOW-CONF: ...], treat disagreement from B and C as STRONGER evidence (A is less trustworthy at that specific word).

NAMED ENTITY RULE — applies to people's names, place names, organisations:
WhisperX (B) is more reliable on named entities. If B has a different named entity than A, consider switching — BUT only if C does not agree with A (case-insensitive). If C agrees with A on the named entity, keep A.

ADDITIONAL KNOWN ERROR PATTERNS:
- PROFANITY AND INFORMAL EXPRESSIONS: A sometimes self-censors mild profanity (e.g. "shit-scared"→"scared", "bloody"→"body", "sweet F all"→"sweetie fall") — if B and C have the original expression, restore it.
- NEGATIONS: If A drops or changes a negation and B and C preserve it — this is critical, switch.
- NUMBERS: If A has a different number than B and C agree on — switch.
- SCOTTISH DIALECT WORDS: If B or C preserve a Scottish dialect word that A has normalised (e.g. "wee", "wisnae", "dinnae", "cannae", "braw", "aboot", "carry-out", "noo") and both agree — preserve the dialect word.
- PRONOUNS: If A changes a pronoun (I/you/we/she/they) and B and C agree on the original — switch.

Do NOT paraphrase, reorder, or restructure sentences.
Do NOT add words that appear in none of the three transcripts.
Do NOT include any [LOW-CONF: ...] markers in your output.

Return ONLY the corrected transcript. No explanation, no labels, no preamble."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def compute_percentile_thresholds(dataset: str, percentile: int) -> dict:
    thresholds = {}
    for model in CONFIDENCE_MODELS:
        path = find_canonical_file(model, dataset)
        samples = load_samples(path)
        scores = [
            seg["confidence"]
            for s in samples
            for seg in (s.get("segments") or [])
            if seg.get("confidence") is not None
        ]
        if not scores:
            thresholds[model] = 0.5
            continue
        scores.sort()
        idx = min(int(len(scores) * percentile / 100), len(scores) - 1)
        thresholds[model] = scores[idx]
        print(f"  Calibrated threshold for {model} (p{percentile}): {thresholds[model]:.3f} "
              f"(from {len(scores)} word scores)")
    return thresholds


def build_inline_transcript(hyp: str, segments: list, threshold: float) -> str:
    if not segments:
        return hyp
    parts = []
    for seg in segments:
        word = seg.get("word", "")
        conf = seg.get("confidence")
        if not word:
            continue
        if conf is not None and conf < threshold:
            parts.append(f"[LOW-CONF: {word}]")
        else:
            parts.append(word)
    return " ".join(parts) if parts else hyp


def strip_lowconf_markers(text: str) -> str:
    return re.sub(r'\[LOW-CONF:\s*([^\]]+)\]', r'\1', text)


def compute_num_predict(hyps: list) -> int:
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, qwen_text, whisperx_text, parakeet_text, num_predict, retries=2):
    prompt = SELECTOR_PROMPT.format(qwen=qwen_text, whisperx=whisperx_text, parakeet=parakeet_text)
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
                keep_alive="30m",
                think=False,
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (select): {e}")
                return None
    return None


def run_dataset(dataset, selector_key, client, percentile, max_samples=None,
                 rerun=False, split="dev", max_workers=None):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | context V1 + confidence INLINE (PHASE 1: selector only, CONCURRENT) "
          f"selector={selector_key} percentile={percentile} split={split} ──")

    qwen_samples     = get_indexed_samples("qwen", dataset)
    whisperx_samples = get_indexed_samples("whisperx", dataset)
    parakeet_samples = get_indexed_samples("parakeet", dataset)
    thresholds = compute_percentile_thresholds(dataset, percentile)

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"context_v1confinline_{dataset}_{selector_key}_p{percentile}_{split}.json"
    new_output_path = os.path.join(NEW_OUTPUT_DIR, filename)
    old_output_path = os.path.join(OLD_OUTPUT_DIR, filename)

    if os.path.exists(old_output_path) and not rerun:
        with open(old_output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results    = []
        start_from = 0

    def save_progress():
        payload = {"progress": len(results), "samples": results}
        with open(new_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(old_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ── PHASE A: fast sequential prep - build work items, append skips directly ──
    remaining_indices = indices[start_from:]
    work_items = []   # each: (idx, ref, qwen_text, whisperx_text, parakeet_text, num_predict)
    skip_slots = {}

    for idx in remaining_indices:
        qwen_sample     = qwen_samples.get(idx)
        whisperx_sample = whisperx_samples.get(idx)
        parakeet_sample = parakeet_samples.get(idx)

        if not all([qwen_sample, whisperx_sample, parakeet_sample]):
            skip_slots[idx] = {"ref": None, "hyp": None, "severity": None,
                               "sample_WER": None, "error": True,
                               "error_reason": "missing sample from one or more models",
                               "dataset_index": idx}
            continue

        ref = qwen_sample["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            skip_slots[idx] = {"ref": ref, "hyp": None, "severity": None,
                               "sample_WER": None, "skipped": True,
                               "skip_reason": "ignore_time_segment", "dataset_index": idx}
            continue

        if is_tag_only(ref):
            skip_slots[idx] = {"ref": ref, "hyp": None, "severity": None,
                               "sample_WER": None, "skipped": True,
                               "skip_reason": "tag_only_reference", "dataset_index": idx}
            continue

        qwen_hyp     = qwen_sample["hyp"]
        whisperx_hyp = whisperx_sample["hyp"]
        parakeet_hyp = parakeet_sample["hyp"]

        qwen_text     = build_inline_transcript(qwen_hyp, qwen_sample.get("segments"), thresholds["qwen"])
        whisperx_text = build_inline_transcript(whisperx_hyp, whisperx_sample.get("segments"), thresholds["whisperx"])
        parakeet_text = build_inline_transcript(parakeet_hyp, parakeet_sample.get("segments"), thresholds["parakeet"])
        num_predict = compute_num_predict([qwen_text, whisperx_text, parakeet_text])

        work_items.append((idx, ref, qwen_text, whisperx_text, parakeet_text, num_predict))

    print(f"  {len(work_items)} samples queued for concurrent Ollama calls "
          f"({len(skip_slots)} skipped without needing a call)")

    # ── PHASE B: fire all Ollama calls concurrently ──
    def _worker(item):
        _, _, qwen_text, whisperx_text, parakeet_text, num_predict = item
        return ollama_select(client, selector_model, qwen_text, whisperx_text, parakeet_text, num_predict)

    raw_results = run_concurrent(work_items, _worker, max_workers=max_workers, progress_every=10)

    # ── Reassemble in original index order ──
    call_results = {item[0]: (item, raw) for item, raw in zip(work_items, raw_results)}

    for idx in remaining_indices:
        if idx in skip_slots:
            results.append(skip_slots[idx])
            continue

        item, raw = call_results[idx]
        _, ref, qwen_text, whisperx_text, parakeet_text, _ = item

        if raw is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        best_hyp = strip_lowconf_markers(raw)
        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":                  ref,
            "hyp":                  best_hyp,
            "qwen_hyp_flagged":     qwen_text,
            "whisperx_hyp_flagged": whisperx_text,
            "parakeet_hyp_flagged": parakeet_text,
            "thresholds_used":      thresholds,
            "sample_WER":           sample_wer_val,
            "severity":             None,
            "dataset_index":        idx,
        })

        if len(results) % 10 == 0:
            save_progress()

    save_progress()

    valid = [r for r in results if not r.get("skipped") and not r.get("error")
             and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None

    output = {
        "selector":         selector_key,
        "approach":         "context_v1_confidence_inline",
        "phase":            "selector_only - severity not yet judged",
        "dataset":          dataset,
        "split":            split,
        "percentile":       percentile,
        "thresholds_used":  thresholds,
        "subset_indices":   indices,
        "corpus_wer":       corpus_wer,
        "num_samples":      len(valid),
        "samples":          results,
    }

    with open(new_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(old_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2:")
    print(f"  python rerunning/add_severity_to_existing.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="gemma4",      choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--percentile",  type=int, default=DEFAULT_PERCENTILE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Concurrent Ollama calls (default from ENSEMBLE_MAX_WORKERS "
                             "env var, or 8 if unset). Lower this on memory-limited machines "
                             "(e.g. a 16GB Mac) - try 1-2. Higher is fine on a dedicated GPU "
                             "node with real VRAM headroom.")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} "
              f"percentile={args.percentile} split={args.split}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client, args.percentile,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split,
                max_workers=args.max_workers)


if __name__ == "__main__":
    main()