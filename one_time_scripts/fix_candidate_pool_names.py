"""
fix_candidate_pool_names.py

Permanently fixes the dataset-naming mismatch in
writeup_results/candidate_pool.json that has been silently defeating
the calibration-sample exclusion in get_split_assignments() (src/splits.py)
for every dataset. Renames:
    common_voice                      -> commonvoice
    edinburgh_international_accents   -> edacc
    english_dialects_scots            -> english_dialects

Backs up the original file first (candidate_pool.json.bak), then
overwrites candidate_pool.json in place with corrected names. After
this runs, get_split_assignments() will correctly exclude the 100
calibration samples going forward - for baseline models (recomputed
on the fly every time build_leaderboard.py runs) and for any future
ensemble/grid scripts that call get_indices_for_split().

This does NOT retroactively fix already-saved ensemble/grid result
files - those need patch_calibration_leakage.py, run separately.

Usage:
    python fix_candidate_pool_names.py
    python fix_candidate_pool_names.py --pool-path writeup_results/candidate_pool.json
"""

import json
import shutil
import argparse

DATASET_NAME_ALIASES = {
    "common_voice": "commonvoice",
    "edinburgh_international_accents": "edacc",
    "english_dialects_scots": "english_dialects",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-path", default="writeup_results/candidate_pool.json")
    args = parser.parse_args()

    backup_path = args.pool_path + ".bak"
    shutil.copy2(args.pool_path, backup_path)
    print(f"Backed up original -> {backup_path}")

    with open(args.pool_path) as f:
        pool = json.load(f)

    changed = 0
    counts_before = {}
    for entry in pool:
        raw = entry.get("dataset")
        counts_before[raw] = counts_before.get(raw, 0) + 1
        if raw in DATASET_NAME_ALIASES:
            entry["dataset"] = DATASET_NAME_ALIASES[raw]
            changed += 1

    print(f"\nDataset names found before fix:")
    for name, count in sorted(counts_before.items()):
        arrow = f" -> {DATASET_NAME_ALIASES[name]}" if name in DATASET_NAME_ALIASES else " (already canonical)"
        print(f"  {name!r}: {count} entries{arrow}")

    with open(args.pool_path, "w") as f:
        json.dump(pool, f, indent=2)

    print(f"\nFixed {changed} entries. {args.pool_path} updated in place.")
    print(f"Original preserved at {backup_path} if you need to roll back.")


if __name__ == "__main__":
    main()
