#!/bin/bash
# check_sample_counts.sh
#
# Counts samples, skipped, and scored across all 4 models' canonical
# benchmark files for a given dataset - quick sanity check that every
# model actually covers the same sample_index range before trusting them
# together in an ensemble script.
#
# Usage:
#   ./check_sample_counts.sh edacc
#   ./check_sample_counts.sh commonvoice
#   ./check_sample_counts.sh english_dialects

set -e

DATASET="${1:?Usage: ./check_sample_counts.sh <dataset>}"
SEARCH_DIRS=("writeup_results/benchmarks/main" "results/benchmarks/main")
MODELS=("qwen" "whisperx" "parakeet" "wav2vec2")

python3 - "$DATASET" "${SEARCH_DIRS[@]}" << 'EOF'
import json
import sys
import glob
import os

dataset = sys.argv[1]
search_dirs = sys.argv[2:]
models = ["qwen", "whisperx", "parakeet", "wav2vec2"]

def find_latest(model, dataset):
    for base_dir in search_dirs:
        pattern = os.path.join(base_dir, f"{model}_{dataset}_*.json")
        matches = sorted(glob.glob(pattern))
        matches = [m for m in matches if "sub150" not in m and "sub100" not in m]
        if matches:
            return matches[-1]
    return None

print(f"{'Model':<12} {'File':<55} {'samples':>8} {'scored':>8} {'skipped':>8} {'sample_index range':>20}")
print("-" * 120)

for model in models:
    path = find_latest(model, dataset)
    if path is None:
        print(f"{model:<12} {'(not found)':<55} {'-':>8} {'-':>8} {'-':>8} {'-':>20}")
        continue

    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    n = len(samples)
    skipped = sum(1 for s in samples if s.get("skipped"))
    scored = n - skipped

    indices = [s["sample_index"] for s in samples if s.get("sample_index") is not None]
    idx_range = f"{min(indices)}-{max(indices)}" if indices else "n/a"

    fname = os.path.basename(path)
    print(f"{model:<12} {fname:<55} {n:>8} {scored:>8} {skipped:>8} {idx_range:>20}")

print()
print("If 'samples' counts differ across models, or the sample_index range")
print("differs, ensemble scripts that inner-join on sample_index (get_indexed_samples)")
print("will silently produce fewer combined results than the full dataset size —")
print("worth checking WHICH indices are missing (not just the count) if numbers don't match.")
EOF