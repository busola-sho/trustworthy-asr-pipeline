"""
build_grid_leaderboard.py

Compares ONLY the 9-cell strategy x context grid
(writeup_results/grid/) - separate from build_leaderboard.py, which
covers the full historical ensemble comparison.

For the 3 cells that also have an original/legacy copy elsewhere
(naive.py -> writeup_results/ensembles/naive/, context_v1.py ->
.../context_v1/, context_v2.py -> .../context_v2/ - copied into
writeup_results/grid/unanchored_fusion_naive/,
.../anchored_correction_v1/, .../anchored_correction_v2/ earlier),
this CROSS-REFERENCES the grid copy against the legacy original and
FLAGS any mismatch loudly - this is exactly the kind of silent
discrepancy that caused a real error earlier (two sources disagreeing,
one manually copied into a table without noticing).

Usage:
    python build_grid_leaderboard.py
"""

import json
import glob
from collections import defaultdict

DATASETS = ["commonvoice", "edacc", "english_dialects"]

GRID_FOLDERS = {
    "selection_naive":              ("selection", "naive"),
    "selection_context_v1":         ("selection", "v1"),
    "selection_context_v2":         ("selection", "v2"),
    "unanchored_fusion_naive":      ("unanchored_fusion", "naive"),
    "unanchored_fusion_context_v1": ("unanchored_fusion", "v1"),
    "unanchored_fusion_context_v2": ("unanchored_fusion", "v2"),
    "anchored_correction_naive":    ("anchored_correction", "naive"),
    "anchored_correction_v1":       ("anchored_correction", "v1"),
    "anchored_correction_v2":       ("anchored_correction", "v2"),
}

# (grid folder, legacy folder, legacy filename pattern) - for cross-checking
CROSS_CHECK = {
    "unanchored_fusion_naive": ("writeup_results/ensembles/naive", "naive_{d}_gemma4sel_dev.json"),
    "anchored_correction_v1":  ("writeup_results/ensembles/context_v1", "context_{d}_gemma4_dev.json"),
    "anchored_correction_v2":  ("writeup_results/ensembles/context_v2", "context_v2_{d}_gemma4_dev.json"),
}


def normalize_dataset(name):
    if not name:
        return "unknown"
    name = name.lower()
    if "common" in name: return "commonvoice"
    if "english" in name or "dialect" in name: return "english_dialects"
    if "edacc" in name: return "edacc"
    return name


def load_grid_data():
    """Returns {folder_name: {dataset: {mean_severity, corpus_wer}}}"""
    data = {}
    for folder in GRID_FOLDERS:
        data[folder] = {}
        for path in glob.glob(f"writeup_results/grid/{folder}/*.json"):
            try:
                d = json.load(open(path))
            except Exception:
                continue
            if d.get("split") not in ("dev", "full") or "mean_severity" not in d:
                continue
            ds = normalize_dataset(d.get("dataset"))
            data[folder][ds] = {
                "mean_severity": d.get("mean_severity"),
                "corpus_wer": d.get("corpus_wer"),
                "path": path,
            }
    return data


def cross_check(grid_data):
    print(f"\n{'='*90}")
    print(f"  CROSS-CHECK: grid copy vs legacy original")
    print(f"{'='*90}")
    any_mismatch = False

    for grid_folder, (legacy_dir, pattern) in CROSS_CHECK.items():
        for d in DATASETS:
            legacy_path = f"{legacy_dir}/{pattern.format(d=d)}"
            try:
                legacy = json.load(open(legacy_path))
            except FileNotFoundError:
                print(f"  {grid_folder}/{d}: legacy file not found at {legacy_path} - skipping")
                continue

            legacy_sev = legacy.get("mean_severity")
            grid_sev = grid_data.get(grid_folder, {}).get(d, {}).get("mean_severity")

            if grid_sev is None:
                print(f"  {grid_folder}/{d}: no grid copy found - skipping")
                continue

            if legacy_sev is None or grid_sev is None:
                continue

            if abs(legacy_sev - grid_sev) > 0.0005:
                any_mismatch = True
                print(f"  MISMATCH: {grid_folder}/{d}")
                print(f"    grid copy:        {grid_sev:.4f}  ({grid_data[grid_folder][d]['path']})")
                print(f"    legacy original:  {legacy_sev:.4f}  ({legacy_path})")
            else:
                print(f"  OK: {grid_folder}/{d} - both sources agree ({grid_sev:.4f})")

    if not any_mismatch:
        print(f"\n  All cross-checked cells agree between grid copy and legacy original.")
    else:
        print(f"\n  WARNING: mismatches found above - the legacy original is the ground truth")
        print(f"  (it's the actual file the script wrote to; the grid copy is a cp of it).")
        print(f"  Re-copy the legacy file into the grid folder to fix, or investigate why")
        print(f"  the two disagree if the legacy file was modified after copying.")


def print_grid_table(grid_data):
    print(f"\n{'='*90}")
    print(f"  9-CELL GRID (mean severity)")
    print(f"{'='*90}")
    header = f"{'Strategy':<22}{'Context':<10}" + "".join(f"{d:>18}" for d in DATASETS) + f"{'Avg':>10}"
    print(header)
    print("-" * len(header))

    for folder, (strategy, context) in GRID_FOLDERS.items():
        sevs = [grid_data.get(folder, {}).get(d, {}).get("mean_severity") for d in DATASETS]
        row = f"{strategy:<22}{context:<10}"
        for sev in sevs:
            row += f"{(f'{sev:.3f}' if sev is not None else '-'):>18}"
        valid = [s for s in sevs if s is not None]
        avg = sum(valid) / len(valid) if valid else None
        row += f"{(f'{avg:.3f}' if avg is not None else '-'):>10}"
        print(row)


def print_strategy_comparison(grid_data):
    print(f"\n{'='*90}")
    print(f"  STRATEGY COMPARISON (best context condition per strategy)")
    print(f"{'='*90}")

    by_strategy = defaultdict(dict)
    for folder, (strategy, context) in GRID_FOLDERS.items():
        sevs = [grid_data.get(folder, {}).get(d, {}).get("mean_severity") for d in DATASETS]
        valid = [s for s in sevs if s is not None]
        avg = sum(valid) / len(valid) if valid and len(valid) == len(DATASETS) else None
        if avg is not None:
            by_strategy[strategy][context] = (avg, sevs)

    header = f"{'Strategy':<22}{'Best condition':<12}" + "".join(f"{d:>18}" for d in DATASETS) + f"{'Avg':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for strategy, conditions in by_strategy.items():
        best_context = min(conditions, key=lambda c: conditions[c][0])
        avg, sevs = conditions[best_context]
        results.append((strategy, best_context, sevs, avg))

    results.sort(key=lambda x: x[3])
    for strategy, best_context, sevs, avg in results:
        row = f"{strategy:<22}{best_context:<12}"
        for sev in sevs:
            row += f"{sev:>18.3f}"
        row += f"{avg:>10.3f}"
        print(row)


def main():
    grid_data = load_grid_data()
    cross_check(grid_data)
    print_grid_table(grid_data)
    print_strategy_comparison(grid_data)


if __name__ == "__main__":
    main()
