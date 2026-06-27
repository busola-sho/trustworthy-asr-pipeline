"""
run_naive_combination_v3.py

Naive combination — passes all 4 ASR model transcripts to a selector LLM
(word-level combiner, prompt p3) and judges the output with the v2 MAR judge.

Restricted to the 150-sample subset (seed=42) for direct comparability with
context-aware variants. Output saved to results/combinations_v2judge/.

Usage:
    python scripts/combination/run_naive_combination_v3.py --dataset commonvoice
    python scripts/combination/run_naive_combination_v3.py --dataset edacc --selector gemma2
    python scripts/combination/run_naive_combination_v3.py --dry-run
"""

import json
import os
import argparse
import time
from jiwer import wer
from ollama import Client
from dotenv import load_dotenv

from src.judge import normalise, ollama_mar
from src.selector import (
    get_subset_indices, CANONICAL_FILES, OLLAMA_MODELS,
    DATASET_SIZES, check_selector_available,
)

load_dotenv()

OUTPUT_DIR = "results/combinations_v2judge"
OLLAMA_HOST = "http://localhost:11434"
ASR_MODELS  = ["qwen", "whisper", "parakeet", "wav2vec2"]
DATASETS    = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps

Return only the final transcript, nothing else."""


def ollama_select(client, model_name, hyps):
    hyp_block = "\n".join([f"Transcript {i+1}: {h}" for i, h in enumerate(hyps)])
    try:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SELECTOR_PROMPT},
                {"role": "user",   "content": hyp_block},
            ],
            options={"temperature": 0, "num_ctx": 4096},
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"  ERROR (select): {e}")
        return None


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | naive selector={selector_key} ──")

    model_samples = {}
    for m in ASR_MODELS:
        path = CANONICAL_FILES[(m, dataset)]
        with open(path) as f:
            model_samples[m] = json.load(f)["samples"]

    indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"naive_{dataset}_{selector_key}sel_p3_qwenjud_sub150.json"
    )

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results    = []
        start_from = 0

    for pos in range(start_from, len(indices)):
        idx = indices[pos]
        ref = model_samples["qwen"][idx]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "skipped": True, "dataset_index": idx})
            continue

        hyps     = [model_samples[m][idx]["hyp"] for m in ASR_MODELS]
        best_hyp = ollama_select(client, selector_model, hyps)
        time.sleep(0.1)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))
        verdict        = ollama_mar(client, ref, best_hyp, sample_wer_val)
        time.sleep(0.1)

        results.append({
            "ref":             ref,
            "hyp":             best_hyp,
            "source_hyps":     {m: model_samples[m][idx]["hyp"] for m in ASR_MODELS},
            "qwen_verdict_p2": verdict,
            "sample_WER":      sample_wer_val,
            "dataset_index":   idx,
        })

        if (pos + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump({"progress": len(results), "samples": results}, f,
                          indent=2, ensure_ascii=False)
            print(f"  {pos+1}/{len(indices)} done")

    valid      = [r for r in results if not r.get("skipped") and not r.get("error")
                  and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None
    mar = sum(1 for r in valid if r.get("qwen_verdict_p2")) / len(valid) if valid else None

    output = {
        "selector":                selector_key,
        "approach":                "naive",
        "dataset":                 dataset,
        "subset_indices":          indices,
        "corpus_wer":              corpus_wer,
        "meaning_alteration_rate": mar,
        "num_samples":             len(valid),
        "samples":                 results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    mar_str = f"{mar*100:.2f}%"        if mar        is not None else "—"
    print(f"  WER: {wer_str}  MAR: {mar_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")
    return {"dataset": dataset, "selector": selector_key,
            "wer": corpus_wer, "mar": mar, "n": len(valid)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="qwen",        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client,
                max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()