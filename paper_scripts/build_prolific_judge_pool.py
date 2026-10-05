#!/usr/bin/env python3
"""Build the balanced 200-item pool for the Prolific judge validation.

The pool contains held-out test examples only:

    100 Unanchored Fusion outputs
     25 Qwen3-ASR outputs
     25 WhisperX outputs
     25 Parakeet outputs
     25 Wav2Vec2.0 outputs

No underlying utterance is selected more than once, even across systems.
The script also tries to balance the locked judge's severity labels (0--4)
as far as the system/dataset quotas and available data permit.

Run from the repository root:

    python build_prolific_judge_pool.py

Outputs are written to prolific_study/ by default. This script builds only
the master pool; assignment to 15 sets is a separate step.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.judge import is_tag_only
from src.splits import get_indices_for_split


DATASETS = ("commonvoice", "edacc", "english_dialects")

SYSTEM_PATHS = {
    "Unanchored Fusion": {
        "commonvoice": Path(
            "writeup_results/clean_grid/unanchored_fusion_naive/"
            "unanchored_fusion_naive_commonvoice_gemma4_test.json"
        ),
        "edacc": Path(
            "writeup_results/clean_grid/unanchored_fusion_naive/"
            "unanchored_fusion_naive_edacc_gemma4_test.json"
        ),
        "english_dialects": Path(
            "writeup_results/clean_grid/unanchored_fusion_naive/"
            "unanchored_fusion_naive_english_dialects_gemma4_test.json"
        ),
    },
    "Qwen3-ASR": {
        "commonvoice": Path(
            "writeup_results/benchmarks/main/qwen_commonvoice_20260722_merged.json"
        ),
        "edacc": Path(
            "writeup_results/benchmarks/main/qwen_edacc_20260722_merged.json"
        ),
        "english_dialects": Path(
            "writeup_results/benchmarks/main/qwen_english_dialects_20260722_merged.json"
        ),
    },
    "WhisperX": {
        "commonvoice": Path(
            "writeup_results/benchmarks/main/whisperx_commonvoice_20260720_162322.json"
        ),
        "edacc": Path(
            "writeup_results/benchmarks/main/whisperx_edacc_20260719_025227.json"
        ),
        "english_dialects": Path(
            "writeup_results/benchmarks/main/whisperx_english_dialects_20260720_234027.json"
        ),
    },
    "Parakeet": {
        "commonvoice": Path(
            "writeup_results/benchmarks/main/parakeet_commonvoice_merged.json"
        ),
        "edacc": Path(
            "writeup_results/benchmarks/main/parakeet_edacc_20260719_033319.json"
        ),
        "english_dialects": Path(
            "writeup_results/benchmarks/main/parakeet_english_dialects_merged.json"
        ),
    },
    "Wav2Vec2.0": {
        "commonvoice": Path(
            "writeup_results/benchmarks/main/wav2vec2_commonvoice_merged.json"
        ),
        "edacc": Path(
            "writeup_results/benchmarks/main/wav2vec2_edacc_20260719_033606.json"
        ),
        "english_dialects": Path(
            "writeup_results/benchmarks/main/wav2vec2_english_dialects_merged.json"
        ),
    },
}

# Rows are systems; columns are CommonVoice, EdAcc, English Dialects.
QUOTAS = {
    "Unanchored Fusion": {
        "commonvoice": 35,
        "edacc": 30,
        "english_dialects": 35,
    },
    "Qwen3-ASR": {"commonvoice": 9, "edacc": 7, "english_dialects": 9},
    "WhisperX": {"commonvoice": 9, "edacc": 7, "english_dialects": 9},
    "Parakeet": {"commonvoice": 9, "edacc": 7, "english_dialects": 9},
    "Wav2Vec2.0": {"commonvoice": 9, "edacc": 7, "english_dialects": 9},
}

# Allocate the most constrained dataset first and leave fusion until last so
# the larger fusion pool can fill utterances not used by individual systems.
CELL_ORDER = [
    (system, dataset)
    for dataset in ("edacc", "commonvoice", "english_dialects")
    for system in ("Qwen3-ASR", "WhisperX", "Parakeet", "Wav2Vec2.0")
] + [
    ("Unanchored Fusion", dataset)
    for dataset in ("edacc", "commonvoice", "english_dialects")
]


def load_candidates(system, dataset, path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"No samples list in {path}")

    test_indices = set(get_indices_for_split(dataset, "test"))
    candidates = []
    seen_indices = set()

    for position, sample in enumerate(samples):
        index = sample.get("sample_index")
        if index is None:
            index = sample.get("dataset_index")
        if index is None:
            index = position

        if index not in test_indices:
            continue
        if index in seen_indices:
            raise ValueError(f"Duplicate index {index} in {path}")
        seen_indices.add(index)

        if sample.get("skipped") or sample.get("error"):
            continue

        reference = sample.get("ref")
        hypothesis = sample.get("hyp")
        severity = sample.get("severity")

        if not isinstance(reference, str) or not reference.strip():
            continue
        if is_tag_only(reference):
            continue
        if hypothesis is None:
            # A genuine empty ASR output should normally be stored as "".
            # Null instead denotes an unavailable/unscoreable output here.
            continue
        if not isinstance(hypothesis, str):
            hypothesis = str(hypothesis)
        if severity not in (0, 1, 2, 3, 4):
            continue

        candidates.append(
            {
                "system": system,
                "dataset": dataset,
                "dataset_index": index,
                "reference": reference.strip(),
                "hypothesis": hypothesis.strip(),
                "judge_severity": int(severity),
                "source_file": str(path),
            }
        )

    required = QUOTAS[system][dataset]
    if len(candidates) < required:
        raise ValueError(
            f"Only {len(candidates)} usable candidates for {system}/{dataset}; "
            f"need {required}."
        )
    return candidates


def choose_cell(candidates, quota, used_utterances, severity_counts, rng):
    available = [
        row
        for row in candidates
        if (row["dataset"], row["dataset_index"]) not in used_utterances
    ]
    if len(available) < quota:
        return None

    chosen = []
    for _ in range(quota):
        remaining = [row for row in available if row not in chosen]
        minimum_count = min(severity_counts[row["judge_severity"]] for row in remaining)
        preferred = [
            row
            for row in remaining
            if severity_counts[row["judge_severity"]] == minimum_count
        ]
        row = rng.choice(preferred)
        chosen.append(row)
        severity_counts[row["judge_severity"]] += 1
    return chosen


def attempt_selection(candidate_map, seed):
    rng = random.Random(seed)
    selected = []
    used_utterances = set()
    severity_counts = Counter({level: 0 for level in range(5)})

    for system, dataset in CELL_ORDER:
        chosen = choose_cell(
            candidate_map[(system, dataset)],
            QUOTAS[system][dataset],
            used_utterances,
            severity_counts,
            rng,
        )
        if chosen is None:
            return None
        for row in chosen:
            used_utterances.add((dataset, row["dataset_index"]))
        selected.extend(chosen)

    return selected


def imbalance_score(selected):
    counts = Counter(row["judge_severity"] for row in selected)
    target = len(selected) / 5
    # Prefer coverage first, then the smallest squared departure from an even
    # five-level distribution. Exact balance may not be possible.
    missing_penalty = sum(1 for level in range(5) if counts[level] == 0) * 1_000_000
    return missing_penalty + sum((counts[level] - target) ** 2 for level in range(5))


def build_pool(candidate_map, seed, attempts):
    best = None
    best_score = None
    for attempt in range(attempts):
        selection = attempt_selection(candidate_map, seed + attempt)
        if selection is None:
            continue
        score = imbalance_score(selection)
        if best is None or score < best_score:
            best = selection
            best_score = score

    if best is None:
        raise RuntimeError(
            "Could not satisfy all quotas without reusing an utterance. "
            "Check source coverage or increase --attempts."
        )
    return best


def verify_pool(rows):
    if len(rows) != 200:
        raise AssertionError(f"Expected 200 rows, found {len(rows)}")

    keys = [(row["dataset"], row["dataset_index"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise AssertionError("The selected pool contains duplicate utterances")

    actual = Counter((row["system"], row["dataset"]) for row in rows)
    for system, dataset_quotas in QUOTAS.items():
        for dataset, expected in dataset_quotas.items():
            found = actual[(system, dataset)]
            if found != expected:
                raise AssertionError(
                    f"Quota mismatch for {system}/{dataset}: {found} != {expected}"
                )

    severities = {row["judge_severity"] for row in rows}
    if severities != {0, 1, 2, 3, 4}:
        raise AssertionError(f"Severity coverage is incomplete: {sorted(severities)}")


def nested_counts(rows, field):
    counts = Counter(row[field] for row in rows)
    return dict(sorted(counts.items(), key=lambda item: str(item[0])))


def write_outputs(rows, output_dir, seed):
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "pool_200_manifest.csv"
    qualtrics_path = output_dir / "pool_200_qualtrics.csv"
    summary_path = output_dir / "pool_200_summary.json"

    manifest_fields = [
        "study_id",
        "dataset",
        "system",
        "dataset_index",
        "reference",
        "hypothesis",
        "judge_severity",
        "source_file",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(rows)

    with qualtrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["study_id", "reference", "hypothesis"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "study_id": row["study_id"],
                    "reference": row["reference"],
                    "hypothesis": row["hypothesis"] or "[EMPTY TRANSCRIPTION]",
                }
            )

    cell_counts = defaultdict(dict)
    for system in QUOTAS:
        for dataset in DATASETS:
            cell_counts[system][dataset] = sum(
                row["system"] == system and row["dataset"] == dataset for row in rows
            )

    summary = {
        "seed": seed,
        "total_items": len(rows),
        "unique_utterances": len(
            {(row["dataset"], row["dataset_index"]) for row in rows}
        ),
        "counts_by_system": nested_counts(rows, "system"),
        "counts_by_dataset": nested_counts(rows, "dataset"),
        "counts_by_severity": nested_counts(rows, "judge_severity"),
        "counts_by_system_and_dataset": dict(cell_counts),
        "files": {
            "manifest": str(manifest_path),
            "qualtrics": str(qualtrics_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return manifest_path, qualtrics_path, summary_path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--attempts", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("prolific_study"))
    args = parser.parse_args()

    candidate_map = {}
    print("Loading held-out test candidates...")
    for system, paths in SYSTEM_PATHS.items():
        for dataset, path in paths.items():
            rows = load_candidates(system, dataset, path)
            candidate_map[(system, dataset)] = rows
            print(f"  {system:<20} {dataset:<18} {len(rows):>4} usable")

    selected = build_pool(candidate_map, args.seed, args.attempts)
    verify_pool(selected)

    # This is the master-pool order only. The later assignment script should
    # randomise presentation order independently for each participant/set.
    random.Random(args.seed + 1_000_000).shuffle(selected)
    for number, row in enumerate(selected, start=1):
        row["study_id"] = f"J{number:03d}"

    manifest, qualtrics, summary_path, summary = write_outputs(
        selected, args.output_dir, args.seed
    )

    print("\nPool verified successfully.")
    print(f"  Items:             {summary['total_items']}")
    print(f"  Unique utterances: {summary['unique_utterances']}")
    print(f"  By system:         {summary['counts_by_system']}")
    print(f"  By dataset:        {summary['counts_by_dataset']}")
    print(f"  By severity:       {summary['counts_by_severity']}")
    print(f"\nSaved: {manifest}")
    print(f"Saved: {qualtrics}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

