#!/bin/bash
# estimate_audio_hours.sh
#
# Estimates total audio duration (in hours) available per dataset, using
# word-level segment timestamps ("start"/"end") already present in the
# WhisperX canonical benchmark files - no need to touch the raw audio
# files directly.
#
# For each sample, duration is approximated as the END time of its LAST
# word segment (assuming each sample's timestamps are relative to that
# clip's own start at t=0). This slightly UNDERESTIMATES true clip length
# if there's trailing silence after the last word, but is a solid
# approximation for deciding fine-tuning feasibility.
#
# Also breaks down by dev/test split (mirroring src/splits.py's logic:
# excludes the 100 calibration IDs, then 70/30 dev/test on the
# remainder) so you can see how much data you'd actually be ABLE to
# fine-tune on if you correctly hold back the test split - and reminds
# you that Shetland is never touched at all.
#
# Usage:
#   ./estimate_audio_hours.sh commonvoice
#   ./estimate_audio_hours.sh english_dialects
#   ./estimate_audio_hours.sh edacc
#   ./estimate_audio_hours.sh all          # sums across commonvoice + english_dialects + edacc

set -e

TARGET="${1:?Usage: ./estimate_audio_hours.sh <dataset|all>}"
SEARCH_DIRS=("writeup_results/benchmarks/main" "results/benchmarks/main")
CANDIDATE_POOL_PATH="writeup_results/candidate_pool.json"
SEED=42
DEV_FRACTION=0.7

python3 - "$TARGET" "$CANDIDATE_POOL_PATH" "$SEED" "$DEV_FRACTION" "${SEARCH_DIRS[@]}" << 'EOF'
import json
import sys
import glob
import os
import random

target = sys.argv[1]
candidate_pool_path = sys.argv[2]
seed = int(sys.argv[3])
dev_fraction = float(sys.argv[4])
search_dirs = sys.argv[5:]

DATASET_SIZES = {
    "commonvoice":      680,
    "edacc":            198,
    "shetland":         100,
    "english_dialects": 2543,
}

DATASETS_TO_CHECK = ["commonvoice", "english_dialects", "edacc"] if target == "all" else [target]


def find_latest(model, dataset):
    for base_dir in search_dirs:
        pattern = os.path.join(base_dir, f"{model}_{dataset}_*.json")
        matches = sorted(glob.glob(pattern))
        matches = [m for m in matches if "sub150" not in m and "sub100" not in m]
        if matches:
            return matches[-1]
    return None


def sample_duration_seconds(sample):
    segments = sample.get("segments") or []
    if not segments:
        return 0.0
    ends = [seg.get("end") for seg in segments if seg.get("end") is not None]
    return max(ends) if ends else 0.0


def get_split_assignment(dataset):
    """Mirrors src/splits.py's get_split_assignments - excludes calibration
    IDs, splits the remainder 70/30 dev/test, seeded for reproducibility."""
    n_total = DATASET_SIZES[dataset]
    try:
        with open(candidate_pool_path) as f:
            pool = json.load(f)
        calibration_indices = {
            entry["sample_index"] for entry in pool
            if entry.get("dataset") == dataset and entry.get("sample_index") is not None
        }
    except FileNotFoundError:
        calibration_indices = set()

    remaining = sorted(set(range(n_total)) - calibration_indices)
    rng = random.Random(seed)
    shuffled = remaining[:]
    rng.shuffle(shuffled)
    n_dev = int(len(shuffled) * dev_fraction)
    return set(shuffled[:n_dev]), set(shuffled[n_dev:]), calibration_indices


grand_total_hours = 0.0
grand_dev_hours = 0.0
grand_test_hours = 0.0

print(f"{'Dataset':<18} {'Total (h)':>10} {'Dev (h)':>10} {'Test (h)':>10} {'Calib (h)':>10} {'N samples':>10}")
print("-" * 75)

for dataset in DATASETS_TO_CHECK:
    path = find_latest("whisperx", dataset)
    if path is None:
        print(f"{dataset:<18} (no whisperx canonical file found)")
        continue

    with open(path) as f:
        data = json.load(f)
    samples = data.get("samples", [])

    dev_idx, test_idx, calib_idx = get_split_assignment(dataset)

    total_secs = 0.0
    dev_secs = 0.0
    test_secs = 0.0
    calib_secs = 0.0

    for s in samples:
        idx = s.get("sample_index")
        dur = sample_duration_seconds(s)
        total_secs += dur
        if idx in dev_idx:
            dev_secs += dur
        elif idx in test_idx:
            test_secs += dur
        elif idx in calib_idx:
            calib_secs += dur

    total_h = total_secs / 3600
    dev_h = dev_secs / 3600
    test_h = test_secs / 3600
    calib_h = calib_secs / 3600

    grand_total_hours += total_h
    grand_dev_hours += dev_h
    grand_test_hours += test_h

    print(f"{dataset:<18} {total_h:>10.2f} {dev_h:>10.2f} {test_h:>10.2f} {calib_h:>10.2f} {len(samples):>10}")

if target == "all":
    print("-" * 75)
    print(f"{'TOTAL':<18} {grand_total_hours:>10.2f} {grand_dev_hours:>10.2f} {grand_test_hours:>10.2f}")

print()
print("Notes:")
print("- Duration per sample = end time of its LAST word segment (assumes")
print("  timestamps are relative to that clip's own start at t=0). This")
print("  slightly UNDERESTIMATES true clip length if there's trailing")
print("  silence, so treat these as a lower-bound estimate.")
print("- 'Dev'/'Test'/'Calib' mirror src/splits.py's split logic - if you")
print("  fine-tune, you should only train on 'Dev' hours (or Dev+Test if")
print("  you're not planning to use the test split for anything else),")
print("  and NEVER touch Shetland (100 samples, held-out, not shown here")
print("  since it's excluded from all ensemble/fine-tuning work by design).")
EOF