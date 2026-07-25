"""
rerunning/ensembles/naive.py

PHASE 1 of 2: Naive combination — passes all 4 ASR model transcripts
(WhisperX, Qwen, Parakeet, wav2vec2) to a selector LLM. Selector calls
only - no severity judging here (two-pass split, same reasoning as
naive_confidence.py: alternating between the selector and Phi-4 in memory
was causing reload-thrashing on limited GPU memory).

Once this finishes, run PHASE 2:
    python rerunning/add_severity_to_existing.py --files <output file from this script>

CHANGES from the original run_naive_combination_v3.py:
  - "whisper" replaced with "whisperx" throughout (standing decision)
  - Two-pass execution - this script does ONLY selector calls; severity
    judging is a separate pass, so only one model is ever loaded at a time
  - num_predict sized dynamically per sample (based on the longest input
    transcript) instead of a fixed cap - avoids truncating long
    EdAcc/English Dialects monologues, which would silently corrupt WER
  - keep_alive="30m" added; unnecessary sleep removed
  - Order rotation - which transcript is labelled "Transcript 1-4" is
    deterministically rotated per sample (seeded by dataset_index), to
    avoid position/model-order bias in the selector's choices
  - Added tag-only reference skip (e.g. "<OVERLAP>")
  - Added --split {dev,test,full} - defaults to "dev" so iteration only
    scores the dev subset (excludes calibration + test indices), cutting
    compute ~30-40% per run. Use --split full only for the final
    confirmatory run once a configuration is locked in.
  - Writes to BOTH writeup_results/ensembles/naive/ (new) and
    results/combinations_v2judge/ (old, kept for continuity)

NOTE: --selector default is gemma4, per selector_ablation.py's locked
result (gemma4:12b: mean_severity=0.85, mean_wer=9.26%, compliance=100%,
best of the 5 candidates tested).

Usage:
    python rerunning/ensembles/naive.py --dataset commonvoice --split dev
    python rerunning/ensembles/naive.py --dataset edacc --split full --selector gemma4
    python rerunning/ensembles/naive.py --dry-run
"""

import json
import os
import random
import argparse
from jiwer import wer
from ollama import Client
from dotenv import load_dotenv

from src.judge import normalise, is_tag_only
from src.selector import (
    find_canonical_file, OLLAMA_MODELS, check_selector_available, load_samples,
)
from src.splits import get_indices_for_split

load_dotenv()

NEW_OUTPUT_DIR = "writeup_results/ensembles/naive"
OLD_OUTPUT_DIR = "results/combinations_v2judge"
OLLAMA_HOST    = "http://localhost:11434"
ASR_MODELS     = ["qwen", "whisperx", "parakeet", "wav2vec2"]   # whisper -> whisperx
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps

Return only the final transcript, nothing else."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def get_rotated_order(models: list, seed_key: int) -> list:
    """Deterministic per-sample shuffle, seeded by dataset_index, so
    transcript position doesn't correlate with model identity across the
    dataset, while staying reproducible run to run."""
    order = list(models)
    rng = random.Random(seed_key)
    rng.shuffle(order)
    return order


def compute_num_predict(hyps: list) -> int:
    """
    Size num_predict dynamically per sample, based on the longest input
    transcript - a fixed cap risks truncating long EdAcc/English Dialects
    monologues, silently corrupting WER for that sample. Floor of 300,
    ceiling of 2048.
    """
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, model_order, hyp_by_model, num_predict, retries=2):
    hyp_block = "\n".join(
        f"Transcript {i+1}: {hyp_by_model[m]}" for i, m in enumerate(model_order)
    )
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SELECTOR_PROMPT},
                    {"role": "user",   "content": hyp_block},
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


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | naive (PHASE 1: selector only) selector={selector_key} split={split} ──")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"naive_{dataset}_{selector_key}sel_{split}.json"
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
        samples_by_model = {m: model_samples[m].get(idx) for m in ASR_MODELS}

        if not all(samples_by_model.values()):
            results.append({"ref": None, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "missing sample from one or more models",
                            "dataset_index": idx})
            continue

        ref = samples_by_model["qwen"]["ref"]

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

        hyp_by_model = {m: samples_by_model[m]["hyp"] for m in ASR_MODELS}
        model_order = get_rotated_order(ASR_MODELS, seed_key=idx)
        num_predict = compute_num_predict(list(hyp_by_model.values()))

        best_hyp = ollama_select(client, selector_model, model_order, hyp_by_model, num_predict)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":            ref,
            "hyp":            best_hyp,   # matches add_severity_to_existing.py's expected schema
            "source_hyps":    hyp_by_model,
            "model_order":    model_order,
            "sample_WER":     sample_wer_val,
            "severity":       None,   # filled in by phase 2 (add_severity_to_existing.py)
            "dataset_index":  idx,
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
        "selector":       selector_key,
        "approach":       "naive",
        "asr_models":     ASR_MODELS,
        "phase":          "selector_only - severity not yet judged",
        "dataset":        dataset,
        "split":          split,
        "subset_indices": indices,
        "corpus_wer":     corpus_wer,
        "num_samples":    len(valid),
        "samples":        results,
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"],
                        help="'dev' for iteration (default, excludes calibration+test), "
                             "'full' only for a final confirmatory run, 'test' to check "
                             "the in-domain test split specifically")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} split={args.split}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()