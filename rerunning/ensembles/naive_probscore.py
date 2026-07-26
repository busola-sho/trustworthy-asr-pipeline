"""
rerunning/ensembles/naive_probscore.py

Naive fusion (your locked best technique: gemma4 selector, all 4 ASR
models - qwen/whisperx/parakeet/wav2vec2, rotated/anonymized transcript
presentation) EXTENDED to also emit a per-sentence PROBABILITY score
(0.00-1.00) that each sentence of the combined output correctly
preserves the spoken meaning.

This replaces the old sentence-confidence pipeline's Method 1
(verbalized confidence), which was built around context_v2 + qwen2.5 as
selector - now superseded since naive is the confirmed best technique.
Uses PROBSCORE (probability 0.0-1.0), not the older CONFSCORE (1-5
scale) - Yang et al. (2024) found probscore calibrates better for both
small and large LLMs, and since this whole pipeline needs rebuilding
around naive anyway, it made sense to carry that finding over rather
than reproduce the older, worse-calibrated format.

DOWNSTREAM: Methods 2/3 (compute_crossmodel_agreement.py,
compute_acoustic_confidence.py) both read "sentence_confidences" from
this file's output to know how to segment/anchor sentences - their
COMBO_FILES dict will need updating to point at this script's output
instead of the old context_v2_whisperx_confidence_sentences files, as
the next step once this is running.

CONCURRENT: same Phase A (fast sequential prep) / Phase B (concurrent
Ollama calls via thread pool) pattern as naive.py and the rest of your
pipeline. MAX_WORKERS reads from ENSEMBLE_MAX_WORKERS env var.

Usage:
    python rerunning/ensembles/naive_probscore.py --dataset commonvoice --split dev
    python rerunning/ensembles/naive_probscore.py --dataset shetland --split full
"""

import json
import os
import re
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
from src.concurrent_ollama import run_concurrent

load_dotenv()

NEW_OUTPUT_DIR = "writeup_results/ensembles/naive_probscore"
OLD_OUTPUT_DIR = "results/combinations_v2judge/naive_probscore"
OLLAMA_HOST    = "http://localhost:11434"
ASR_MODELS     = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]

DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))

# Naive's original task (select best words/phrases across 4 anonymized
# transcripts, no rules/reliability profiles - that's a context_v1/v2
# thing, not naive's) + probscore's per-sentence probability request
# layered on top. Uses PROBABILITY (0.00-1.00), matching Yang et al.
# (2024)'s better-calibrated formulation, not the older 1-5 CONFIDENCE scale.
SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps

After producing the combined transcript, break it into sentences and, for each sentence, provide the probability from 0.00 to 1.00 that it correctly preserves the spoken meaning - based on how much the four transcripts agreed at that point. Use the full range - do not default to values near 1.00. Consider whether names, numbers, dates, negations, or dialect words were disputed between the transcripts.

Return ONLY this format, nothing else - no explanation, no preamble:

TRANSCRIPT:
<your combined transcript here>

PROBABILITY:
1 | <probability 0.00-1.00> | <sentence 1>
2 | <probability 0.00-1.00> | <sentence 2>
3 | <probability 0.00-1.00> | <sentence 3>
..."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def get_rotated_order(models: list, seed_key: int) -> list:
    order = list(models)
    rng = random.Random(seed_key)
    rng.shuffle(order)
    return order


def compute_num_predict(hyps: list) -> int:
    """
    Larger buffer than naive.py's plain version, since the response now
    also includes the per-sentence PROBABILITY block on top of the
    transcript itself - roughly transcript length again, plus overhead
    for the "N | prob | sentence" formatting per line.
    """
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 2.2) + 150
    return max(400, min(estimated, 3000))


def parse_selector_response(raw: str):
    """
    Parses "TRANSCRIPT:\\n<text>\\nPROBABILITY:\\n1 | 0.85 | sentence"
    into (transcript, sentence_confidences). sentence_confidences items
    match the schema Methods 2/3 already expect: "sentence" (anchor
    text), "confidence" (0-1 probability), plus "idx"/"score" kept for
    schema consistency with the older probscore output.
    """
    transcript = raw
    sentence_confidences = []

    if "TRANSCRIPT:" in raw and "PROBABILITY:" in raw:
        parts = raw.split("PROBABILITY:")
        transcript = parts[0].replace("TRANSCRIPT:", "").strip()
        prob_part = parts[1].strip() if len(parts) > 1 else ""

        sent_pos = 0
        for line in prob_part.split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(.+)$", line)
            if match:
                sent_pos += 1
                prob = float(match.group(2))
                prob = max(0.0, min(1.0, prob))
                sentence_confidences.append({
                    "idx":        sent_pos,
                    "score":      None,
                    "confidence": round(prob, 4),
                    "sentence":   match.group(3).strip(),
                })
    elif "TRANSCRIPT:" in raw:
        transcript = raw.replace("TRANSCRIPT:", "").strip()

    return transcript, sentence_confidences


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


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False,
                 split="dev", max_workers=None):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | naive_probscore (PHASE 1: selector + probscore, CONCURRENT) "
          f"selector={selector_key} split={split} ──")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"naive_probscore_{dataset}_{selector_key}sel_{split}.json"
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
        model_order = get_rotated_order(ASR_MODELS, seed_key=idx)
        num_predict = compute_num_predict(list(hyp_by_model.values()))

        work_items.append((idx, ref, model_order, hyp_by_model, num_predict))

    print(f"  {len(work_items)} samples queued for concurrent Ollama calls "
          f"({len(skip_slots)} skipped without needing a call)")

    # ── PHASE B: fire all Ollama calls concurrently ──
    def _worker(item):
        idx, ref, model_order, hyp_by_model, num_predict = item
        return ollama_select(client, selector_model, model_order, hyp_by_model, num_predict)

    raw_responses = run_concurrent(work_items, _worker, max_workers=max_workers, progress_every=10)

    # ── Reassemble in original index order, parsing the probscore response ──
    call_results = {item[0]: (item, raw) for item, raw in zip(work_items, raw_responses)}

    for idx in remaining_indices:
        if idx in skip_slots:
            results.append(skip_slots[idx])
            continue

        item, raw = call_results[idx]
        _, ref, model_order, hyp_by_model, _ = item

        if raw is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        best_hyp, sentence_confidences = parse_selector_response(raw)

        if not best_hyp:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "could not parse TRANSCRIPT block",
                            "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":                  ref,
            "hyp":                  best_hyp,
            "source_hyps":          hyp_by_model,
            "model_order":          model_order,
            "sample_WER":           sample_wer_val,
            "sentence_confidences": sentence_confidences,
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

    n_missing_probs = sum(1 for r in valid if not r.get("sentence_confidences"))

    output = {
        "selector":         selector_key,
        "approach":         "naive_probscore",
        "prompt_variant":   "probscore",
        "asr_models":       ASR_MODELS,
        "phase":            "selector_only - severity not yet judged",
        "dataset":          dataset,
        "split":            split,
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
    if n_missing_probs:
        print(f"  WARNING: {n_missing_probs}/{len(valid)} samples had no parsed "
              f"sentence_confidences (selector didn't follow the PROBABILITY format)")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2 to add severity scores:")
    print(f"  python rerunning/add_severity_to_existing_concurrent.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="gemma4",      choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
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
                max_samples=args.max_samples, rerun=args.rerun, split=args.split,
                max_workers=args.max_workers)


if __name__ == "__main__":
    main()