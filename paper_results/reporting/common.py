"""Frozen reporting split lookup and shared text-scoring rules."""

import json
from pathlib import Path
from asr_pipeline.text_normalization import (
    normalise as normalise,
    is_tag_only as is_tag_only,
)


def load_split_assignments(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assignments = data["current_assignments"]
    for dataset, size in (
        ("commonvoice", 680),
        ("edacc", 198),
        ("english_dialects", 2543),
    ):
        parts = assignments[dataset]
        sets = [set(parts[key]) for key in ("dev", "test", "excluded_calibration")]
        if any(
            len(values) != len(parts[key])
            for values, key in zip(sets, ("dev", "test", "excluded_calibration"))
        ):
            raise ValueError(f"Duplicate split IDs for {dataset}")
        if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"Overlapping split IDs for {dataset}")
        if set.union(*sets) != set(range(size)):
            raise ValueError(
                f"Split IDs do not partition the recorded dataset size for {dataset}"
            )
    return assignments


def get_indices_for_split(dataset, split, paths):
    if paths.split_assignments is None:
        raise RuntimeError("Supply explicit frozen splits with --splits-file")
    return paths.split_assignments[dataset][split]
