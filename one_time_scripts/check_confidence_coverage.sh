#!/bin/bash
# check_confidence_coverage.sh
#
# Checks whether a model's canonical benchmark file for a dataset actually
# has segments/confidence populated - or whether compute_percentile_thresholds
# is silently falling back to 0.5 because the scores list came back empty.
#
# Usage:
#   ./check_confidence_coverage.sh edacc
#   ./check_confidence_coverage.sh commonvoice

set -e

DATASET="${1:?Usage: ./check_confidence_coverage.sh <dataset>}"
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

print(f"{'Model':<12} {'File':<50} {'samples_w/segs':>16} {'total_words_w/conf':>20} {'min':>6} {'max':>6} {'p20':>6}")
print("-" * 120)

for model in models:
    path = find_latest(model, dataset)
    if path is None:
        print(f"{model:<12} (not found)")
        continue

    with open(path) as f:
        data = json.load(f)
    samples = data.get("samples", [])

    samples_with_segs = sum(1 for s in samples if s.get("segments"))
    scores = [
        seg["confidence"]
        for s in samples
        for seg in (s.get("segments") or [])
        if seg.get("confidence") is not None
    ]

    fname = os.path.basename(path)
    if not scores:
        print(f"{model:<12} {fname:<50} {samples_with_segs:>16} {'0 (FALLBACK TO 0.5)':>20} {'-':>6} {'-':>6} {'-':>6}")
        continue

    scores.sort()
    p20_idx = min(int(len(scores) * 20 / 100), len(scores) - 1)
    print(f"{model:<12} {fname:<50} {samples_with_segs:>16} {len(scores):>20} "
          f"{scores[0]:>6.3f} {scores[-1]:>6.3f} {scores[p20_idx]:>6.3f}")

print()
print("'0 (FALLBACK TO 0.5)' means this model's file has NO usable confidence")
print("scores at all - compute_percentile_thresholds silently defaulted to 0.5,")
print("which is why get_low_conf_words returns [] for every sample regardless")
print("of actual transcript quality. If min/max/p20 all look identical or oddly")
print("narrow (e.g. everything clustered near 1.0), that's also worth a second look.")
EOF