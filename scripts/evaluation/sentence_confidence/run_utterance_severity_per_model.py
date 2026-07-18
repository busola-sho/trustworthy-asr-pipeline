"""
scripts/evaluation/sentence_confidence/run_utterance_severity_per_model.py

Runs sentence severity judge on individual model outputs (WhisperX, Qwen, Parakeet)
to get utterance-level severity scores for comparison against the ensemble.

Output: results/sentence_confidence/utterance_severity_{model}_{dataset}.json
  [{"dataset_index": int, "ref": str, "hyp": str, "severity": int}, ...]

Usage:
    python scripts/evaluation/sentence_confidence/run_utterance_severity_per_model.py
    python scripts/evaluation/sentence_confidence/run_utterance_severity_per_model.py --dataset commonvoice
    python scripts/evaluation/sentence_confidence/run_utterance_severity_per_model.py --model whisperx
"""

import json
import os
import argparse
import time
from ollama import Client
from src.judge import ollama_severity

OLLAMA_HOST = "http://localhost:11434"
OUTPUT_DIR  = "results/sentence_confidence"

SUBSET_FILES = {
    ("whisperx", "commonvoice"):      "results/benchmarks/subsets/whisperx_commonvoice_sub150.json",
    ("whisperx", "edacc"):            "results/benchmarks/subsets/whisperx_edacc_sub150.json",
    ("whisperx", "english_dialects"): "results/benchmarks/subsets/whisperx_english_dialects_sub150.json",
    ("whisperx", "shetland"):         "results/benchmarks/subsets/shetland/whisperx_shetland_sub100.json",
    ("qwen",     "commonvoice"):      "results/benchmarks/subsets/qwen3asr_commonvoice_sub150.json",
    ("qwen",     "edacc"):            "results/benchmarks/subsets/qwen3asr_edacc_sub150.json",
    ("qwen",     "english_dialects"): "results/benchmarks/subsets/qwen3asr_english_dialects_sub150.json",
    ("qwen",     "shetland"):         "results/benchmarks/subsets/shetland/qwen3asr_shetland_sub100.json",
    ("parakeet", "commonvoice"):      "results/benchmarks/subsets/parakeet_commonvoice_sub150.json",
    ("parakeet", "edacc"):            "results/benchmarks/subsets/parakeet_edacc_sub150.json",
    ("parakeet", "english_dialects"): "results/benchmarks/subsets/parakeet_english_dialects_sub150.json",
    ("parakeet", "shetland"):         "results/benchmarks/subsets/shetland/parakeet_shetland_sub100.json",
}

# ensemble severity comes from existing combo results
ENSEMBLE_FILES = {
    "commonvoice":      "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_commonvoice_qwen_sub150.json",
    "edacc":            "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_edacc_qwen_sub150.json",
    "english_dialects": "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_english_dialects_qwen_sub150.json",
    "shetland":         "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_shetland_qwen_sub150.json",
}

MODELS   = ["whisperx", "qwen", "parakeet"]
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]


def run_model_dataset(model, dataset, client, rerun=False):
    print(f"\n── {model} / {dataset} ──")

    output_path = os.path.join(OUTPUT_DIR, f"utterance_severity_{model}_{dataset}.json")
    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        done = {r["dataset_index"] for r in existing}
        print(f"  Resuming — {len(done)} already done")
    else:
        existing = []
        done     = set()

    path = SUBSET_FILES.get((model, dataset))
    if not path or not os.path.exists(path):
        print(f"  ERROR: subset file not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    samples  = data.get("samples", [])
    new_rows = 0

    for s in samples:
        if s.get("skipped") or s.get("error"):
            continue
        di  = s.get("sample_index")
        ref = s.get("ref", "")
        hyp = s.get("hyp", "")

        if di is None or not ref or not hyp:
            continue
        if di in done:
            continue
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        severity = ollama_severity(client, ref, hyp)
        existing.append({
            "dataset_index": di,
            "ref":           ref,
            "hyp":           hyp,
            "severity":      severity,
        })
        done.add(di)
        new_rows += 1

        if new_rows % 20 == 0:
            print(f"  {len(existing)} done...")
            _save(output_path, existing)

    _save(output_path, existing)
    valid = [r for r in existing if r.get("severity") is not None]
    if valid:
        from collections import Counter
        dist = Counter(r["severity"] for r in valid)
        print(f"  Total: {len(valid)} | Severity dist: {dict(sorted(dist.items()))}")


def _save(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="all", choices=MODELS + ["all"])
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--rerun",   action="store_true")
    args = parser.parse_args()

    client   = Client(host=OLLAMA_HOST)
    models   = MODELS   if args.model   == "all" else [args.model]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for model in models:
        for dataset in datasets:
            run_model_dataset(model, dataset, client, rerun=args.rerun)

    print("\nDone. Now plot with:")
    print("  python scripts/evaluation/sentence_confidence/plot_severity_distribution.py")


if __name__ == "__main__":
    main()