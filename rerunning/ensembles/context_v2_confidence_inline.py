"""
rerunning/ensembles/context_v2_confidence_inline.py

PHASE 1 of 2: Context-aware V2 (auto-generated rules) + word-level
confidence flagging using INLINE markers - built specifically to compare
against context_v2_confidence.py's SEPARATE/structured metadata approach.

WHY BOTH VARIANTS EXIST: the original feedback favoured separate metadata
over inline markers (shorter prompt, less risk of disrupting sentence
structure). But inline markers have a real advantage the separate design
lost: an inline marker sits at the EXACT occurrence of the flagged word in
the running sentence, so repeated words (e.g. two occurrences of "the")
are naturally disambiguated by position in the text itself. The separate
design needed an explicit position-index fix (see context_v2_confidence.py)
to recover that same disambiguation. This script exists so you can run
both and compare empirically rather than assuming one is better.

Once this finishes, run PHASE 2:
    python rerunning/add_severity_to_existing.py --files <output file from this script>

Same underlying fixes as context_v2_confidence.py: WhisperX properly
wired, per-model percentile thresholds, two-pass split, tag-only skip,
--split {dev,test,full} replaces --full (defaults to "dev" for
iteration), dual write, dynamic num_predict, keep_alive.

*** SAME UNRESOLVED DEPENDENCY: build_rules_text() reads error_profiles.json,
likely still reflecting old Whisper's error data. Regenerate from
WhisperX's error profiles before trusting the auto-generated rules. ***

Usage:
    python rerunning/ensembles/context_v2_confidence_inline.py --dataset commonvoice --split dev --percentile 20
    python rerunning/ensembles/context_v2_confidence_inline.py --dataset edacc --split full --percentile 20
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
from src.rules import build_rules_text
from src.splits import get_indices_for_split

NEW_OUTPUT_DIR = "writeup_results/ensembles/context_v2_confidence_inline"
OLD_OUTPUT_DIR = "results/combinations_v2judge/context_v2_confidence_inline"
OLLAMA_HOST    = "http://localhost:11434"
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]
CONFIDENCE_MODELS  = ["qwen", "whisperx", "parakeet"]   # qwen added
DEFAULT_PERCENTILE = 20

SELECTOR_PROMPT_TEMPLATE = """You are correcting an ASR transcript. You are given three transcripts of the same audio from different models.

Some transcripts include inline [LOW-CONF: word] markers — these indicate words that model itself was uncertain about. Treat words marked this way as MORE likely to be wrong. Do NOT include the [LOW-CONF: ...] markers in your final output — write the plain word only.

TRANSCRIPT A (base — use this as your starting point, model: qwen):
{qwen}

TRANSCRIPT B (model: whisperx):
{whisperx}

TRANSCRIPT C (model: parakeet):
{parakeet}

Your task: return Transcript A with targeted corrections where needed. Do NOT include the [LOW-CONF: ...] markers in your final output — write the plain word only.

GENERAL RULE — applies to all words except where a specific rule below overrides it:
Only change a word if BOTH B and C disagree with A and agree with each other on the same alternative. If only one of B or C disagrees with A, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

CONFIDENCE WEIGHTING:
- If a disagreeing word in B or C is marked [LOW-CONF: ...], treat that disagreement as WEAKER evidence.
- If the word in A itself is marked [LOW-CONF: ...], treat disagreement from B and C as STRONGER evidence (A is less trustworthy at that specific word).

MODEL-SPECIFIC RELIABILITY RULES (derived from measured error rates across all benchmark datasets):
{auto_rules}

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
    """
    Builds the transcript with [LOW-CONF: word] markers inserted at the
    EXACT position of each flagged word - this is what gives inline
    markers their positional-disambiguation advantage over a separate list.
    Falls back to the plain hyp if segments don't cover it (e.g. Qwen,
    which has no confidence data in this comparison run).
    """
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


def ollama_select(client, model_name, qwen_text, whisperx_text, parakeet_text,
                   auto_rules, num_predict, retries=2):
    prompt = SELECTOR_PROMPT_TEMPLATE.format(
        qwen=qwen_text, whisperx=whisperx_text, parakeet=parakeet_text, auto_rules=auto_rules,
    )
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
                keep_alive="30m",
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (select): {e}")
                return None
    return None


def run_dataset(dataset, selector_key, client, auto_rules, percentile,
                 max_samples=None, rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | context V2 + confidence INLINE (PHASE 1: selector only) "
          f"selector={selector_key} percentile={percentile} split={split} ──")

    qwen_samples     = get_indexed_samples("qwen", dataset)
    whisperx_samples = get_indexed_samples("whisperx", dataset)
    parakeet_samples = get_indexed_samples("parakeet", dataset)
    thresholds = compute_percentile_thresholds(dataset, percentile)

    indices = get_indices_for_split(dataset, split)
    print(f"  Split: {split} ({len(indices)} samples)")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"context_v2confinline_{dataset}_{selector_key}_p{percentile}_{split}.json"
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

    for pos in range(start_from, len(indices)):
        idx = indices[pos]

        qwen_sample     = qwen_samples.get(idx)
        whisperx_sample = whisperx_samples.get(idx)
        parakeet_sample = parakeet_samples.get(idx)

        if not all([qwen_sample, whisperx_sample, parakeet_sample]):
            results.append({"ref": None, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "missing sample from one or more models",
                            "dataset_index": idx})
            continue

        ref = qwen_sample["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "ignore_time_segment", "dataset_index": idx})
            continue

        if is_tag_only(ref):
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "tag_only_reference", "dataset_index": idx})
            continue

        qwen_hyp     = qwen_sample["hyp"]
        whisperx_hyp = whisperx_sample["hyp"]
        parakeet_hyp = parakeet_sample["hyp"]

        qwen_text     = build_inline_transcript(qwen_hyp, qwen_sample.get("segments"), thresholds["qwen"])
        whisperx_text = build_inline_transcript(whisperx_hyp, whisperx_sample.get("segments"), thresholds["whisperx"])
        parakeet_text = build_inline_transcript(parakeet_hyp, parakeet_sample.get("segments"), thresholds["parakeet"])
        num_predict = compute_num_predict([qwen_text, whisperx_text, parakeet_text])

        raw = ollama_select(client, selector_model, qwen_text, whisperx_text, parakeet_text,
                             auto_rules, num_predict)

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
            "auto_rules_used":      auto_rules,
            "dataset_index":        idx,
        })

        if (pos + 1) % 10 == 0:
            save_progress()
            print(f"  {pos+1}/{len(indices)} done")

    valid = [r for r in results if not r.get("skipped") and not r.get("error")
             and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None

    output = {
        "selector":         selector_key,
        "approach":         "context_v2_confidence_inline",
        "confidence_models": CONFIDENCE_MODELS,
        "phase":            "selector_only - severity not yet judged",
        "dataset":          dataset,
        "split":            split,
        "percentile":       percentile,
        "thresholds_used":  thresholds,
        "auto_rules_used":  auto_rules,
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
    parser.add_argument("--selector",    default="qwen",        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--percentile",  type=int,              default=DEFAULT_PERCENTILE)
    parser.add_argument("--gap",         type=float,            default=1.0)
    parser.add_argument("--max-samples", type=int,              default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    auto_rules = build_rules_text(min_gap_pp=args.gap, save=False)

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} "
              f"percentile={args.percentile} split={args.split}")
        print(f"Auto-generated rules:\n{auto_rules}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client, auto_rules, args.percentile,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()