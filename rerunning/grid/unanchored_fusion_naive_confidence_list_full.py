"""
rerunning/grid/unanchored_fusion_naive_confidence_list_full.py

Strategy: UNANCHORED FUSION. Context condition: ASR CONFIDENCE,
SEPARATE LIST FORMAT, UNFILTERED - every word from every model is
listed with its position and its ACTUAL confidence value (not a
binary flag from a percentile cutoff), letting the model see the full
continuous gradient and judge for itself. The no-threshold counterpart
to unanchored_fusion_naive_confidence_list.py.

Part of the 4-condition ablation on the locked winning strategy
(Unanchored Fusion + Naive):
  Baseline               -> naive.py (no confidence)
  List, thresholded p20  -> unanchored_fusion_naive_confidence_list.py
  Inline, thresholded p20 -> unanchored_fusion_naive_confidence_inline.py
  List, unfiltered       -> THIS SCRIPT
  Inline, unfiltered     -> unanchored_fusion_naive_confidence_inline_full.py

Uses concurrent execution (src.concurrent_ollama.run_concurrent), same
as the thresholded versions. No threshold to calibrate, so no
threshold-leakage concern here.

Writes to writeup_results/grid/unanchored_fusion_naive_confidence_list_full/.

Usage:
    python rerunning/grid/unanchored_fusion_naive_confidence_list_full.py --dataset commonvoice --split dev --selector gemma4
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

OUTPUT_DIR = "writeup_results/grid/unanchored_fusion_naive_confidence_list_full"
OLLAMA_HOST = "http://localhost:11434"
ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]
DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))

SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Each transcript is followed by a list of ALL its words with their confidence scores (0.00-1.00, higher = more confident). Use these scores as evidence when deciding which words to trust - lower scores indicate the model itself was less certain about that word.

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


def format_full_confidence_list(segments: list) -> str:
    if not segments:
        return "none"
    parts = []
    for i, seg in enumerate(segments):
        word = seg.get("word", "")
        conf = seg.get("confidence")
        if not word:
            continue
        conf_str = f"{conf:.2f}" if conf is not None else "?"
        parts.append(f'word {i+1}: "{word}" (conf={conf_str})')
    return ", ".join(parts) if parts else "none"


def get_rotated_order(models: list, seed_key: int) -> list:
    order = list(models)
    rng = random.Random(seed_key)
    rng.shuffle(order)
    return order


def compute_num_predict(hyps: list) -> int:
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, model_order, hyp_by_model, conf_list_by_model, num_predict, retries=2):
    hyp_block = "\n".join(
        f"Transcript {i+1} (model: {m}): {hyp_by_model[m]}\n"
        f"Word confidences: {conf_list_by_model[m]}"
        for i, m in enumerate(model_order)
    )
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SELECTOR_PROMPT},
                    {"role": "user", "content": hyp_block},
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


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False, split="dev", max_workers=None):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    selector_model = OLLAMA_MODELS[selector_key]
    print(f"\n-- {dataset} | unanchored_fusion_naive_confidence_list_full (CONCURRENT) "
          f"selector={selector_key} split={split} --")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"unanchored_fusion_naive_confidence_list_full_{dataset}_{selector_key}_{split}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results = []
        start_from = 0

    def save_progress():
        payload = {"progress": len(results), "samples": results}
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    remaining_indices = indices[start_from:]
    work_items = []
    skip_slots = {}

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
        conf_list_by_model = {
            m: format_full_confidence_list(samples_by_model[m].get("segments")) for m in ASR_MODELS
        }
        model_order = get_rotated_order(ASR_MODELS, seed_key=idx)
        num_predict = compute_num_predict(list(hyp_by_model.values()))

        work_items.append((idx, ref, hyp_by_model, conf_list_by_model, model_order, num_predict))

    print(f"  {len(work_items)} samples queued for concurrent Ollama calls "
          f"({len(skip_slots)} skipped without needing a call)")

    def _worker(item):
        _, _, hyp_by_model, conf_list_by_model, model_order, num_predict = item
        return ollama_select(client, selector_model, model_order, hyp_by_model, conf_list_by_model, num_predict)

    raw_results = run_concurrent(work_items, _worker, max_workers=max_workers, progress_every=10)
    call_results = {item[0]: (item, raw) for item, raw in zip(work_items, raw_results)}

    for idx in remaining_indices:
        if idx in skip_slots:
            results.append(skip_slots[idx])
            continue

        item, best_hyp = call_results[idx]
        _, ref, hyp_by_model, conf_list_by_model, model_order, _ = item

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref": ref,
            "hyp": best_hyp,
            "source_hyps": hyp_by_model,
            "model_order": model_order,
            "sample_WER": sample_wer_val,
            "severity": None,
            "dataset_index": idx,
        })

        if len(results) % 10 == 0:
            save_progress()

    save_progress()

    valid = [r for r in results if not r.get("skipped") and not r.get("error") and r.get("sample_WER") is not None]
    corpus_wer = wer([normalise(r["ref"]) for r in valid], [normalise(r["hyp"]) for r in valid]) if valid else None

    output = {
        "selector": selector_key,
        "approach": "unanchored_fusion_naive_confidence_list_full",
        "strategy": "unanchored_fusion",
        "context_condition": "confidence_list_no_threshold",
        "asr_models": ASR_MODELS,
        "phase": "selector_only - severity not yet judged",
        "dataset": dataset,
        "split": split,
        "subset_indices": indices,
        "corpus_wer": corpus_wer,
        "num_samples": len(valid),
        "samples": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "-"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")
    print(f"\n  Now run severity: python rerunning/add_severity_to_existing_concurrent.py --files {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector", default="gemma4", choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split,
                max_workers=args.max_workers)


if __name__ == "__main__":
    main()
