"""
src/finetune_splits.py

Train/validation/test split for ASR FINE-TUNING - separate purpose from
src/splits.py's dev/test split (which is used for iterating on and
scoring ensemble/selector techniques). This module answers a different
question: "what audio can I fine-tune the base ASR model on, and what do
I hold out to check the fine-tuned model afterward?"

SPLIT DESIGN (conservative version - nothing previously labelled "test"
ever enters training):
  - Same calibration-exclusion discipline as src/splits.py: the 100
    judge-calibration sample IDs are excluded first, same as always.
  - The existing 30% TEST pool from src/splits.py (reproduced here with
    the identical seed/algorithm) is left COMPLETELY UNTOUCHED - it
    becomes this module's "test" directly, unchanged. Nothing is carved
    out of it, nothing is added to it.
  - The existing 70% DEV pool is subdivided into:
      - train: 80% of dev
      - val:   20% of dev
  - Overall, this works out to roughly 56% train / 14% val / 30% test
    of the total non-calibration pool (exact numbers vary slightly per
    dataset depending on rounding).
  - Shetland is NEVER included here - same standing decision as
    everywhere else: it's the untouched external held-out test set.

Usage:
    Train:      fine-tune the ASR model.
    Validation: choose LoRA settings, epochs, and checkpoint.
    Test:       evaluate every final pipeline and model - identical to
                the "test" split already used for ensemble evaluation.
    Shetland:   test external generalisation (handled elsewhere, not by
                this module).

    from src.finetune_splits import get_finetune_split, get_finetune_indices

    split = get_finetune_split("commonvoice")
    train_indices = sorted(split["train"])
    val_indices   = sorted(split["val"])
    test_indices  = sorted(split["test"])

    # or directly:
    train_indices = get_finetune_indices("commonvoice", "train")
"""

import json
import random

from src.selector import DATASET_SIZES

CANDIDATE_POOL_PATH = "writeup_results/candidate_pool.json"
SEED = 42                  # same seed as src/splits.py, so the reproduced
                            # dev/test split is IDENTICAL to the existing one
TRAIN_VAL_SEED = 44        # separate seed for splitting dev into train/val

DEV_FRACTION = 0.7             # matches src/splits.py exactly
TRAIN_FRACTION_OF_DEV = 0.8    # 80% of dev -> train, 20% of dev -> val

_finetune_split_cache = {}


def _reproduce_old_dev_test_split(dataset: str) -> dict:
    """
    Exactly reproduces src/splits.py's get_split_assignments for a
    dataset: exclude calibration IDs, then 70/30 dev/test on the
    remainder, seeded with SEED=42. Duplicated here (rather than
    imported) so this module has no dependency on src/splits.py's
    internal cache state, and so the seed/algorithm are explicit and
    visible in one place for anyone auditing the fine-tuning split.
    """
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
    return {
        "old_dev":  set(shuffled[:n_dev]),
        "old_test": set(shuffled[n_dev:]),
        "excluded_calibration": calibration_indices,
    }


def get_finetune_split(dataset: str) -> dict:
    """
    Returns {"train": set, "val": set, "test": set, "excluded_calibration": set}
    for a dataset. "test" is IDENTICAL to src/splits.py's existing test
    pool - untouched. "train"/"val" are an 80/20 split of the existing
    dev pool. Cached after first computation.

    Never call this for "shetland" - it's excluded by design (raises
    ValueError): Shetland has no calibration file entries and should
    never be fine-tuned on or used to pick checkpoints.
    """
    if dataset == "shetland":
        raise ValueError(
            "Shetland is the untouched external held-out test set and must "
            "never be included in fine-tuning train/val/test splits."
        )

    if dataset in _finetune_split_cache:
        return _finetune_split_cache[dataset]

    old_split = _reproduce_old_dev_test_split(dataset)
    old_dev = old_split["old_dev"]
    old_test = old_split["old_test"]   # left completely untouched

    dev_sorted = sorted(old_dev)
    rng_tv = random.Random(TRAIN_VAL_SEED)
    shuffled_dev = dev_sorted[:]
    rng_tv.shuffle(shuffled_dev)
    n_train = int(len(shuffled_dev) * TRAIN_FRACTION_OF_DEV)
    train = set(shuffled_dev[:n_train])
    val = set(shuffled_dev[n_train:])

    assignment = {
        "train": train,
        "val":   val,
        "test":  set(old_test),
        "excluded_calibration": old_split["excluded_calibration"],
    }
    _finetune_split_cache[dataset] = assignment
    return assignment


def get_finetune_indices(dataset: str, split: str) -> list:
    """
    split: "train", "val", or "test". Returns a sorted list of
    sample_index values for that split.
    """
    assignment = get_finetune_split(dataset)
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
    return sorted(assignment[split])