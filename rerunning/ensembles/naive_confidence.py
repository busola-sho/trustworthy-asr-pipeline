"""
rerunning/ensembles/naive_confidence.py

PHASE 1 of 2: Naive combination + word-level confidence flagging.
Selector calls only - no severity judging here.

CONCURRENT VERSION: the sequential per-sample loop is split into a fast
Phase A (skip-checks, threshold-based confidence flagging, prompt block
building - all cheap, no network I/O) and a concurrent Phase B (actually
firing the Ollama selector calls via a thread pool, instead of one at a
time). See src/concurrent_ollama.py's docstring for what this requires
on the Ollama SERVER side (OLLAMA_NUM_PARALLEL) - client-side
concurrency alone does nothing without it.

Usage:
    python rerunning/ensembles/naive_confidence.py --dataset commonvoice --split dev --percentile 20
"""

import json
import os
import random
import argparse
from jiwer import wer
from ollama import Client

from src.judge import normalise, is_tag_only
from src.selector import (
    find_canonical_file, OLLAMA_MODELS, check_selector_available, load_samples,
)
from src.splits import get_indices_for_split
from src.concurrent_ollama import run_concurrent

NEW_OUTPUT_DIR = "writeup_results/ensembles/naive_confidence"
OLD_OUTPUT_DIR = "results/combinations_v2judge/naive_confidence"
OLLAMA_HOST    = "http://localhost:11434"
ASR_MODELS     = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]
CONFIDENCE_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]   # all 4 now produce confidence scores
DEFAULT_PERCENTILE = 20
MAX_WORKERS = 8   # tune to roughly match OLLAMA_NUM_PARALLEL on the server

SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Each transcript is followed by a list of words that model flagged as
low-confidence - i.e. words the model itself was uncertain about. Treat
those words as MORE likely to be wrong when deciding what to combine.
Each entry is labeled with its position in that transcript (counting
from 1) - use this position to identify the specific occurrence flagged,
especially when a word appears more than once in the transcript.

Your task is to construct the most accurate transcript by selecting the best
words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is
  clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps

Return only the final transcript, nothing else. Do not include any
"Low-confidence words" text in your output."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def compute_percentile_thresholds(dataset: str, percentile: int) -> dict:
    """
    Calibrate a separate confidence threshold per model, as the Nth
    percentile of THAT model's own confidence score distribution across
    the dataset (dev data only - never Shetland). Words scoring below
    their own model's threshold are flagged as low-confidence. This
    replaces a single shared raw threshold, since WhisperX and Parakeet
    scores aren't guaranteed to be on the same scale.
    """
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
    this transcript's own word sequence - disambiguates repeated words
    (e.g. two occurrences of "the" where only one is actually flagged)."""
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


def get_rotated_order(models: list, seed_key: int) -> list:
    """Deterministic per-sample shuffle, seeded by dataset_index, so
    transcript position doesn't correlate with model identity across the
    dataset, while staying reproducible run to run."""
    order = list(models)
    rng = random.Random(seed_key)
    rng.shuffle(order)
    return order


def build_prompt_block(model_order: list, hyp_by_model: dict, low_conf_by_model: dict) -> str:
    lines = []
    for i, model in enumerate(model_order):
        lines.append(f"Transcript {i+1}: {hyp_by_model[model]}")
        lines.append(f"Low-confidence words: {format_low_conf(low_conf_by_model[model])}")
        lines.append("")
    return "\n".join(lines)


def compute_num_predict(hyp_by_model: dict) -> int:
    """
    Size num_predict dynamically per sample, based on the longest input
    transcript - a fixed cap risks truncating long EdAcc/English Dialects
    monologues (some run 200-300+ words), which would silently corrupt
    both WER and downstream severity for that sample. ~1.3 tokens/word is
    a reasonable English estimate; the +50 buffer covers punctuation and
    tokenization variance. Floor of 300 keeps short samples from getting
    an unnecessarily tiny cap; ceiling of 2048 guards against a pathological
    outlier blowing up generation time.
    """
    max_words = max(len(h.split()) for h in hyp_by_model.values())
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, prompt_block, num_predict, retries=2):
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SELECTOR_PROMPT},
                    {"role": "user",   "content": prompt_block},
                ],
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
                 rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | naive + confidence (PHASE 1: selector only, CONCURRENT) "
          f"selector={selector_key} percentile={percentile} split={split} ──")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}
    thresholds = compute_percentile_thresholds(dataset, percentile)

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"naive_conf_{dataset}_{selector_key}_p{percentile}_{split}.json"
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
    work_items = []   # each: (idx, ref, hyp_by_model, low_conf_by_model, prompt_block, num_predict)
    skip_slots = {}   # idx -> pre-built skip/error result dict

    for idx in remaining_indices:
        samples_by_model = {m: model_samples[m].get(idx) for m in ASR_MODELS}

        if not all(samples_by_model.values()):
            skip_slots[idx] = {"ref": None, "hyp": None, "severity": None,
                               "sample_WER": None, "error": True,
                               "error_reason": "missing sample from one or more models",
                               "dataset_index": idx}
            continue

        ref = samples_by_model["qwen"]["ref"]

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

        hyp_by_model = {m: samples_by_model[m]["hyp"] for m in ASR_MODELS}
        low_conf_by_model = {
            m: get_low_conf_words(samples_by_model[m].get("segments"), thresholds[m])
               if m in CONFIDENCE_MODELS else []
            for m in ASR_MODELS
        }

        model_order = get_rotated_order(ASR_MODELS, seed_key=idx)
        prompt_block = build_prompt_block(model_order, hyp_by_model, low_conf_by_model)
        num_predict = compute_num_predict(hyp_by_model)

        work_items.append((idx, ref, hyp_by_model, low_conf_by_model, model_order, prompt_block, num_predict))

    print(f"  {len(work_items)} samples queued for concurrent Ollama calls "
          f"({len(skip_slots)} skipped without needing a call)")

    # ── PHASE B: fire all Ollama calls concurrently ──
    def _worker(item):
        _, _, _, _, _, prompt_block, num_predict = item
        return ollama_select(client, selector_model, prompt_block, num_predict)

    best_hyps = run_concurrent(work_items, _worker, max_workers=MAX_WORKERS, progress_every=10)

    # ── Reassemble in original index order ──
    call_results = {item[0]: (item, best_hyp) for item, best_hyp in zip(work_items, best_hyps)}

    for idx in remaining_indices:
        if idx in skip_slots:
            results.append(skip_slots[idx])
            continue

        item, best_hyp = call_results[idx]
        _, ref, hyp_by_model, low_conf_by_model, model_order, _, _ = item

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":               ref,
            "hyp":               best_hyp,
            "source_hyps":       hyp_by_model,
            "model_order":       model_order,
            "low_conf_words":    low_conf_by_model,
            "thresholds_used":   thresholds,
            "sample_WER":        sample_wer_val,
            "severity":          None,
            "dataset_index":     idx,
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
        "selector":             selector_key,
        "approach":             "naive_confidence",
        "asr_models":           ASR_MODELS,
        "phase":                "selector_only - severity not yet judged",
        "dataset":              dataset,
        "split":                split,
        "percentile":           percentile,
        "thresholds_used":      thresholds,
        "subset_indices":       indices,
        "corpus_wer":           corpus_wer,
        "num_samples":          len(valid),
        "samples":              results,
    }

    with open(new_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(old_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2 to add severity scores:")
    print(f"  python rerunning/add_severity_to_existing.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="gemma4",      choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--percentile",  type=int,              default=DEFAULT_PERCENTILE,
                        help="Percentile of each model's own confidence distribution "
                             "used as its flagging threshold (default: 20)")
    parser.add_argument("--max-samples", type=int,              default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"],
                        help="'dev' for iteration (default), 'full' only for a final "
                             "confirmatory run, 'test' to check the in-domain split")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
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
                max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()