"""
rerunning/ensembles/context_v1_confidence.py

PHASE 1 of 2: Context-aware V1 (hand-written rules) + word-level confidence
flagging - now using ALL 3 models' confidence (Qwen, WhisperX, Parakeet),
not just WhisperX/Parakeet. Selector calls only - no severity judging here.

Once this finishes, run PHASE 2:
    python rerunning/add_severity_to_existing.py --files <output file from this script>

CHANGES from the previous version:
  - Qwen now gets confidence flagging too (its confidence extraction was
    recently added - see src/models.py). Qwen remains the structural
    "base/anchor" (the transcript being corrected), but its own confidence
    now modulates how much weight to give B/C's disagreement: a NEW rule
    says if a word in A itself is low-confidence, treat B/C's disagreement
    as STRONGER evidence, not just "weaker if B/C's word is low-confidence"
    as before. Without this, Qwen's confidence data would be extracted but
    never actually used for anything.
  - wav2vec2 still NOT included - Context V1/V2 only ever used 3 models
    (Qwen, WhisperX, Parakeet) as input, unlike naive/naive_confidence
    which use all 4. Adding wav2vec2 here would change the technique's
    architecture, not just add a confidence signal.

Everything else unchanged from the previous version: WhisperX swap,
per-model percentile thresholds, two-pass split, tag-only skip, dual
write, dynamic num_predict, keep_alive. --split {dev,test,full} replaces
--full - defaults to "dev" for iteration.

NOT CHANGED - NEEDS YOUR REVIEW: the named-entity trust rule etc. is
still unchanged - edit SELECTOR_PROMPT once you've decided how to handle
the WhisperX-vs-Qwen dataset-dependent reliability question.

Usage:
    python rerunning/ensembles/context_v1_confidence.py --dataset commonvoice --split dev --percentile 20
"""

import json
import os
import argparse
from jiwer import wer
from ollama import Client

from src.judge import normalise, is_tag_only
from src.selector import (
    find_canonical_file, OLLAMA_MODELS, check_selector_available, load_samples,
)
from src.splits import get_indices_for_split

NEW_OUTPUT_DIR = "writeup_results/ensembles/context_v1_confidence"
OLD_OUTPUT_DIR = "results/combinations_v2judge/context_v1_confidence"
OLLAMA_HOST    = "http://localhost:11434"
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]
CONFIDENCE_MODELS  = ["qwen", "whisperx", "parakeet"]   # qwen added
DEFAULT_PERCENTILE = 20

# NOTE: rule content unchanged (named-entity trust etc.) - review per the
# discussion in context_v1.py before editing. The confidence-weighting
# rules below ARE new, added specifically to make Qwen's confidence do
# something now that it's available.
SELECTOR_PROMPT = """You are correcting an ASR transcript. You are given three transcripts of the same audio from different models.

Each transcript is followed by a list of words that model flagged as low-confidence - i.e. words the model itself was uncertain about.

TRANSCRIPT A (base — use this as your starting point):
{qwen}
Low-confidence words: {qwen_lowconf}

TRANSCRIPT B (WhisperX):
{whisperx}
Low-confidence words: {whisperx_lowconf}

TRANSCRIPT C (Parakeet):
{parakeet}
Low-confidence words: {parakeet_lowconf}

Your task: return Transcript A with targeted corrections where needed.

GENERAL RULE — applies to all words except named entities:
Only change a word if BOTH B and C disagree with A and agree with each other on the same alternative. If only one of B or C disagrees with A, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

CONFIDENCE WEIGHTING:
- If a disagreeing word in B or C is on that transcript's low-confidence list, treat that disagreement as WEAKER evidence (less likely B/C are actually right).
- If the word in A itself is on A's low-confidence list, treat disagreement from B and C as STRONGER evidence (A is less trustworthy at that specific word, so it should be easier to override).

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


def get_low_conf_words(segments: list, threshold: float) -> list:
    """Returns (position, word) pairs, 1-indexed by the word's position in
    this transcript's own word sequence - disambiguates repeated words."""
    if not segments:
        return []
    return [
        (i + 1, seg["word"])
        for i, seg in enumerate(segments)
        if seg.get("confidence") is not None and seg["confidence"] < threshold
    ]


def format_low_conf(low_conf_list: list) -> str:
    if not low_conf_list:
        return "none"
    return ", ".join(f'word {pos}: "{word}"' for pos, word in low_conf_list)


def compute_num_predict(hyps: list) -> int:
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, qwen_hyp, whisperx_hyp, parakeet_hyp,
                   qwen_lowconf, whisperx_lowconf, parakeet_lowconf, num_predict, retries=2):
    prompt = SELECTOR_PROMPT.format(
        qwen=qwen_hyp, whisperx=whisperx_hyp, parakeet=parakeet_hyp,
        qwen_lowconf=format_low_conf(qwen_lowconf),
        whisperx_lowconf=format_low_conf(whisperx_lowconf),
        parakeet_lowconf=format_low_conf(parakeet_lowconf),
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


def run_dataset(dataset, selector_key, client, percentile, max_samples=None,
                 rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | context V1 + confidence (PHASE 1: selector only) "
          f"selector={selector_key} percentile={percentile} ──")

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

    filename = f"context_v1conf_{dataset}_{selector_key}_p{percentile}_{split}.json"
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

        qwen_lowconf     = get_low_conf_words(qwen_sample.get("segments"), thresholds["qwen"])
        whisperx_lowconf = get_low_conf_words(whisperx_sample.get("segments"), thresholds["whisperx"])
        parakeet_lowconf = get_low_conf_words(parakeet_sample.get("segments"), thresholds["parakeet"])
        num_predict = compute_num_predict([qwen_hyp, whisperx_hyp, parakeet_hyp])

        best_hyp = ollama_select(client, selector_model, qwen_hyp, whisperx_hyp, parakeet_hyp,
                                  qwen_lowconf, whisperx_lowconf, parakeet_lowconf, num_predict)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":                ref,
            "hyp":                best_hyp,
            "qwen_base":          qwen_hyp,
            "whisperx_hyp":       whisperx_hyp,
            "parakeet_hyp":       parakeet_hyp,
            "qwen_lowconf":       qwen_lowconf,
            "whisperx_lowconf":   whisperx_lowconf,
            "parakeet_lowconf":   parakeet_lowconf,
            "thresholds_used":    thresholds,
            "sample_WER":         sample_wer_val,
            "severity":           None,
            "dataset_index":      idx,
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
        "approach":         "context_v1_confidence",
        "confidence_models": CONFIDENCE_MODELS,
        "phase":            "selector_only - severity not yet judged",
        "dataset":          dataset,
        "full_dataset":     full,
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
    parser.add_argument("--selector",    default="qwen",        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--percentile",  type=int, default=DEFAULT_PERCENTILE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--full",        action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} "
              f"percentile={args.percentile} full={args.full}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client, args.percentile,
                max_samples=args.max_samples, rerun=args.rerun, full=args.full)


if __name__ == "__main__":
    main()