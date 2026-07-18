"""
scripts/evaluation/sentence_confidence/run_utterance_severity_ensemble.py

Runs utterance-level severity judge on the ensemble (Context V2) output
to match the format of utterance_severity_{model}_{dataset}.json files.

Output: results/sentence_confidence/utterance_severity_ensemble_{dataset}.json

Usage:
    python scripts/evaluation/sentence_confidence/run_utterance_severity_ensemble.py
    python scripts/evaluation/sentence_confidence/run_utterance_severity_ensemble.py --dataset commonvoice
"""

import json
import os
import argparse
import time
from ollama import Client
from src.judge import ollama_severity

OLLAMA_HOST = "http://localhost:11434"
OUTPUT_DIR  = "results/sentence_confidence"

COMBO_FILES = {
    "commonvoice":      "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_commonvoice_qwen_sub150.json",
    "edacc":            "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_edacc_qwen_sub150.json",
    "english_dialects": "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_english_dialects_qwen_sub150.json",
    "shetland":         "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_shetland_qwen_sub150.json",
}

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]


def run_dataset(dataset, client, rerun=False):
    print(f"\n── ensemble / {dataset} ──")

    output_path = os.path.join(OUTPUT_DIR, f"utterance_severity_ensemble_{dataset}.json")

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            rows = json.load(f)
        done = {r["dataset_index"] for r in rows}
        print(f"  Resuming — {len(rows)} already done")
    else:
        rows = []
        done = set()

    with open(COMBO_FILES[dataset]) as f:
        combo = json.load(f)

    new_rows   = 0
    start_time = time.time()

    for s in combo.get("samples", []):
        if s.get("skipped") or s.get("error"):
            continue

        di  = s.get("dataset_index")
        ref = s.get("ref", "")
        hyp = s.get("hyp", "")

        if di is None or not ref or not hyp:
            continue
        if di in done:
            continue
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        severity = ollama_severity(client, ref, hyp)

        rows.append({
            "dataset_index": di,
            "ref":           ref,
            "hyp":           hyp,
            "severity":      severity,
        })
        done.add(di)
        new_rows += 1

        if new_rows % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  {len(rows)} done ({elapsed:.0f}s)...")
            _save(output_path, rows)

    _save(output_path, rows)

    from collections import Counter
    valid = [r for r in rows if r.get("severity") is not None]
    dist  = Counter(r["severity"] for r in valid)
    print(f"  Total: {len(valid)} | Dist: {dict(sorted(dist.items()))}")


def _save(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--rerun",   action="store_true")
    args = parser.parse_args()

    client   = Client(host=OLLAMA_HOST)
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        run_dataset(dataset, client, rerun=args.rerun)

    print("\nDone. Now update plot_severity_distribution.py to include ensemble utterance files.")


if __name__ == "__main__":
    main()