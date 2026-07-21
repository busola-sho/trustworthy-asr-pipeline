"""
src/splits.py

Shared dev/test split logic - used by both compute_dev_test_splits.py
(post-hoc analysis) and every ensemble technique script (to restrict a
RUN to just the dev indices during iteration, rather than scoring the
full dataset - including test and calibration samples - on every tweak).

Split, per dataset (never Shetland):
  1. Exclude the 100 judge-calibration sample IDs.
  2. Split the remainder 70% dev / 30% in-domain test (seeded, so the
     same split is reproduced every time this is called).
"""

import json
import random

from src.selector import DATASET_SIZES

CANDIDATE_POOL_PATH = "writeup_results/candidate_pool.json"
SEED = 42
DEV_FRACTION = 0.7

_split_cache = {}


def get_split_assignments(dataset: str) -> dict:
    """Returns {"dev": set, "test": set, "excluded_calibration": set} for
    a dataset, cached after first computation. Deterministic given SEED."""
    if dataset in _split_cache:
        return _split_cache[dataset]

    if dataset not in DATASET_SIZES:
        raise ValueError(f"Unknown dataset '{dataset}' - not in DATASET_SIZES")

    n_total = DATASET_SIZES[dataset]
    all_indices = set(range(n_total))

    with open(CANDIDATE_POOL_PATH) as f:
        pool = json.load(f)

    calibration_indices = {
        entry["sample_index"] for entry in pool
        if entry.get("dataset") == dataset and entry.get("sample_index") is not None
    }

    remaining = sorted(all_indices - calibration_indices)
    rng = random.Random(SEED)
    shuffled = remaining[:]
    rng.shuffle(shuffled)

    n_dev = int(len(shuffled) * DEV_FRACTION)
    assignment = {
        "dev":  set(shuffled[:n_dev]),
        "test": set(shuffled[n_dev:]),
        "excluded_calibration": calibration_indices,
    }
    _split_cache[dataset] = assignment
    return assignment


def get_indices_for_split(dataset: str, split: str) -> list:
    """
    split: "dev", "test", or "full" (full = every index, including
    calibration - use this only for the final confirmatory run, not
    while iterating).
    """
    if split == "full":
        return list(range(DATASET_SIZES[dataset]))
    assignment = get_split_assignments(dataset)
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev', 'test', or 'full', got '{split}'")
    return sorted(assignment[split])