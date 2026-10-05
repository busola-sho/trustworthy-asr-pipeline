#!/usr/bin/env python3
"""Create 15 balanced 40-item rating sets from the 200-item master pool.

Every study item appears exactly three times, once in each of three different
sets. Each set contains 40 unique items and has its own random presentation
order.

Run from the repository root after build_prolific_judge_pool.py:

    python build_prolific_sets.py
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


N_SETS = 15
SET_SIZE = 40
REPEATS_PER_ITEM = 3


def read_pool(path):
    if not path.exists():
        raise FileNotFoundError(f"Pool manifest not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 200:
        raise ValueError(f"Expected 200 pool items, found {len(rows)} in {path}")

    required = {
        "study_id",
        "dataset",
        "system",
        "dataset_index",
        "reference",
        "hypothesis",
        "judge_severity",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Pool manifest is missing columns: {sorted(missing)}")

    ids = [row["study_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("study_id values are not unique")

    for row in rows:
        row["judge_severity"] = int(row["judge_severity"])
    return rows


def round_capacities(round_number, rng):
    """Give five sets 14 items and ten sets 13 items per round.

    Across three rounds, every set receives 14 exactly once and 13 twice:
    14 + 13 + 13 = 40.
    """
    groups = list(range(N_SETS))
    rng.shuffle(groups)
    high_sets_by_round = [set(groups[i * 5:(i + 1) * 5]) for i in range(3)]
    high = high_sets_by_round[round_number]
    return {set_id: (14 if set_id in high else 13) for set_id in range(N_SETS)}


def assign_one_round(rows, previous_sets, capacities, rng):
    """Assign every item once in this round, respecting prior assignments."""
    remaining = list(rows)
    assignments = defaultdict(list)

    # Fill larger and more constrained buckets first. Random tie-breaking keeps
    # repeated construction attempts different.
    set_order = list(range(N_SETS))
    rng.shuffle(set_order)
    set_order.sort(key=lambda s: capacities[s], reverse=True)

    for set_id in set_order:
        needed = capacities[set_id]
        eligible = [row for row in remaining if set_id not in previous_sets[row["study_id"]]]
        if len(eligible) < needed:
            return None

        # Prefer categories currently underrepresented within this set.
        chosen = []
        category_counts = Counter()
        for _ in range(needed):
            available = [row for row in eligible if row not in chosen]
            if not available:
                return None
            scores = []
            for row in available:
                category = (
                    row["system"],
                    row["dataset"],
                    row["judge_severity"],
                )
                scores.append((category_counts[category], row))
            best_score = min(score for score, _ in scores)
            best = [row for score, row in scores if score == best_score]
            row = rng.choice(best)
            chosen.append(row)
            category_counts[(row["system"], row["dataset"], row["judge_severity"])] += 1

        assignments[set_id].extend(chosen)
        chosen_ids = {row["study_id"] for row in chosen}
        remaining = [row for row in remaining if row["study_id"] not in chosen_ids]

    if remaining:
        return None
    return assignments


def assignment_score(sets):
    """Lower is better: encourages similar compositions across sets."""
    fields = ("system", "dataset", "judge_severity")
    score = 0.0
    for field in fields:
        all_values = sorted({row[field] for rows in sets.values() for row in rows}, key=str)
        for value in all_values:
            counts = [sum(row[field] == value for row in sets[s]) for s in range(N_SETS)]
            mean = sum(counts) / N_SETS
            score += sum((count - mean) ** 2 for count in counts)
    return score


def build_sets(rows, seed, attempts):
    best_sets = None
    best_score = None

    for attempt in range(attempts):
        rng = random.Random(seed + attempt)

        # Partition which sets receive 14 items in each of the three rounds.
        set_ids = list(range(N_SETS))
        rng.shuffle(set_ids)
        high_sets = [set(set_ids[i * 5:(i + 1) * 5]) for i in range(3)]

        previous_sets = defaultdict(set)
        combined = defaultdict(list)
        successful = True

        for round_number in range(3):
            capacities = {
                set_id: (14 if set_id in high_sets[round_number] else 13)
                for set_id in range(N_SETS)
            }
            result = assign_one_round(rows, previous_sets, capacities, rng)
            if result is None:
                successful = False
                break

            for set_id, assigned_rows in result.items():
                combined[set_id].extend(assigned_rows)
                for row in assigned_rows:
                    previous_sets[row["study_id"]].add(set_id)

        if not successful:
            continue

        score = assignment_score(combined)
        if best_sets is None or score < best_score:
            best_sets = combined
            best_score = score

    if best_sets is None:
        raise RuntimeError(
            "Could not construct the balanced sets. Increase --attempts and try again."
        )
    return best_sets


def verify_sets(sets, pool_rows):
    appearances = Counter()
    appearance_sets = defaultdict(set)

    if set(sets) != set(range(N_SETS)):
        raise AssertionError("Not all 15 sets were created")

    for set_id, rows in sets.items():
        if len(rows) != SET_SIZE:
            raise AssertionError(f"Set {set_id + 1:02d} has {len(rows)} items, not 40")
        ids = [row["study_id"] for row in rows]
        if len(set(ids)) != SET_SIZE:
            raise AssertionError(f"Set {set_id + 1:02d} contains a duplicate item")
        for study_id in ids:
            appearances[study_id] += 1
            appearance_sets[study_id].add(set_id)

    expected_ids = {row["study_id"] for row in pool_rows}
    if set(appearances) != expected_ids:
        raise AssertionError("The set assignments do not cover the complete pool")

    for study_id in expected_ids:
        if appearances[study_id] != REPEATS_PER_ITEM:
            raise AssertionError(
                f"{study_id} appears {appearances[study_id]} times instead of 3"
            )
        if len(appearance_sets[study_id]) != REPEATS_PER_ITEM:
            raise AssertionError(f"{study_id} is repeated within the same set")


def composition(rows, field):
    return dict(sorted(Counter(row[field] for row in rows).items(), key=lambda x: str(x[0])))


def write_outputs(sets, output_dir, seed):
    sets_dir = output_dir / "qualtrics_sets"
    sets_dir.mkdir(parents=True, exist_ok=True)

    master_path = output_dir / "set_assignments_manifest.csv"
    summary_path = output_dir / "set_assignments_summary.json"
    master_fields = [
        "set_id",
        "presentation_order",
        "study_id",
        "dataset",
        "system",
        "dataset_index",
        "reference",
        "hypothesis",
        "judge_severity",
        "source_file",
    ]

    master_rows = []
    summary = {"seed": seed, "sets": {}}

    for zero_based_set_id in range(N_SETS):
        set_number = zero_based_set_id + 1
        set_id = f"{set_number:02d}"
        rows = list(sets[zero_based_set_id])
        random.Random(seed + 100_000 + set_number).shuffle(rows)

        set_path = sets_dir / f"qualtrics_set_{set_id}.csv"
        with set_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "set_id",
                    "presentation_order",
                    "study_id",
                    "reference",
                    "hypothesis",
                ],
            )
            writer.writeheader()
            for position, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "set_id": set_id,
                        "presentation_order": position,
                        "study_id": row["study_id"],
                        "reference": row["reference"],
                        "hypothesis": row["hypothesis"] or "[EMPTY TRANSCRIPTION]",
                    }
                )

        for position, row in enumerate(rows, start=1):
            master_rows.append(
                {
                    "set_id": set_id,
                    "presentation_order": position,
                    **row,
                }
            )

        summary["sets"][set_id] = {
            "items": len(rows),
            "by_system": composition(rows, "system"),
            "by_dataset": composition(rows, "dataset"),
            "by_judge_severity": composition(rows, "judge_severity"),
            "qualtrics_file": str(set_path),
        }

    with master_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=master_fields)
        writer.writeheader()
        writer.writerows(master_rows)

    summary["total_presentations"] = len(master_rows)
    summary["appearances_per_item"] = REPEATS_PER_ITEM
    summary["participants"] = N_SETS
    summary["items_per_participant"] = SET_SIZE
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return master_path, summary_path, sets_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("prolific_study/pool_200_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("prolific_study")
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--attempts", type=int, default=1000)
    args = parser.parse_args()

    rows = read_pool(args.pool)
    sets = build_sets(rows, args.seed, args.attempts)
    verify_sets(sets, rows)
    master, summary, sets_dir = write_outputs(sets, args.output_dir, args.seed)

    print("Assignment verified successfully.")
    print("  15 sets x 40 items = 600 presentations")
    print("  200 items x 3 different sets = 600 presentations")
    print(f"Saved master assignment: {master}")
    print(f"Saved summary:           {summary}")
    print(f"Saved blinded set files: {sets_dir}/qualtrics_set_01.csv ... _15.csv")


if __name__ == "__main__":
    main()

