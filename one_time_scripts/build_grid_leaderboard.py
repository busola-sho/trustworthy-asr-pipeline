"""
build_grid_leaderboard.py

Compares ONLY the 9-cell strategy x context grid
(writeup_results/grid/) - separate from build_leaderboard.py, which
covers the full historical ensemble comparison.

CROSS_CHECK: for cells that also have an original/legacy copy elsewhere
this cross-references the grid copy against the legacy original and
FLAGS any mismatch loudly. NOTE: anchored_correction_v1/v2 were
REMOVED from this check - their prompts were deliberately edited after
the grid copy was first made, so the grid version and the legacy
context_v1/context_v2 files are now genuinely different experiments,
not duplicates. A mismatch there is expected, not a bug - cross-
checking them would produce a false-positive "mismatch" every time.

Usage:
    python build_grid_leaderboard.py
    python build_grid_leaderboard.py --verify
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

# (grid folder, legacy folder, legacy filename pattern) - for cross-checking.
# anchored_correction_v1/v2 deliberately excluded - see module docstring.
CROSS_CHECK = {
    "unanchored_fusion_naive": ("writeup_results/ensembles/naive/gemma4", "naive_{d}_gemma4sel_dev.json"),
}


def normalize_dataset(name):
    if not name:
        return "unknown"
    name = name.lower()
    if "common" in name: return "commonvoice"
    if "english" in name or "dialect" in name: return "english_dialects"
    if "edacc" in name: return "edacc"
    return name


def load_grid_data(scan_dir):
    """Returns {folder_name: {dataset: {mean_severity, corpus_wer}}}"""
    data = {}
    for folder in GRID_FOLDERS:
        data[folder] = {}
        for path in glob.glob(f"{scan_dir}/{folder}/*.json"):
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
    print(f"  (anchored_correction_v1/v2 excluded - prompts were deliberately")
    print(f"   edited after the grid copy, so those are now separate experiments)")
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


def print_grid_table(grid_data):
    print(f"\n{'='*100}")
    print(f"  9-CELL GRID (severity / WER)")
    print(f"{'='*100}")
    header = f"{'Strategy':<22}{'Context':<10}" + "".join(f"{d:>22}" for d in DATASETS) + f"{'Avg Sev':>10}{'Avg WER':>10}"
    print(header)
    print("-" * len(header))

    for folder, (strategy, context) in GRID_FOLDERS.items():
        cells = [grid_data.get(folder, {}).get(d, {}) for d in DATASETS]
        row = f"{strategy:<22}{context:<10}"
        for c in cells:
            sev = c.get("mean_severity")
            wer = c.get("corpus_wer")
            sev_str = f"{sev:.3f}" if sev is not None else "-"
            wer_str = f"{wer*100:.2f}%" if wer is not None else "-"
            row += f"{f'{sev_str} / {wer_str}':>22}"
        sevs = [c.get("mean_severity") for c in cells if c.get("mean_severity") is not None]
        wers = [c.get("corpus_wer") for c in cells if c.get("corpus_wer") is not None]
        avg_sev = sum(sevs) / len(sevs) if sevs else None
        avg_wer = sum(wers) / len(wers) if wers else None
        row += f"{(f'{avg_sev:.3f}' if avg_sev is not None else '-'):>10}"
        row += f"{(f'{avg_wer*100:.2f}%' if avg_wer is not None else '-'):>10}"
        print(row)


def print_strategy_comparison(grid_data):
    print(f"\n{'='*100}")
    print(f"  STRATEGY COMPARISON (best context condition per strategy, by severity)")
    print(f"{'='*100}")

    by_strategy = defaultdict(dict)
    for folder, (strategy, context) in GRID_FOLDERS.items():
        cells = [grid_data.get(folder, {}).get(d, {}) for d in DATASETS]
        sevs = [c.get("mean_severity") for c in cells]
        wers = [c.get("corpus_wer") for c in cells]
        valid_sevs = [s for s in sevs if s is not None]
        avg_sev = sum(valid_sevs) / len(valid_sevs) if valid_sevs and len(valid_sevs) == len(DATASETS) else None
        if avg_sev is not None:
            by_strategy[strategy][context] = (avg_sev, sevs, wers)

    header = f"{'Strategy':<22}{'Best condition':<16}" + "".join(f"{d:>22}" for d in DATASETS) + f"{'Avg Sev':>10}{'Avg WER':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for strategy, conditions in by_strategy.items():
        best_context = min(conditions, key=lambda c: conditions[c][0])
        avg_sev, sevs, wers = conditions[best_context]
        valid_wers = [w for w in wers if w is not None]
        avg_wer = sum(valid_wers) / len(valid_wers) if valid_wers else None
        results.append((strategy, best_context, sevs, wers, avg_sev, avg_wer))

    results.sort(key=lambda x: x[4])
    for strategy, best_context, sevs, wers, avg_sev, avg_wer in results:
        row = f"{strategy:<22}{best_context:<16}"
        for sev, wer in zip(sevs, wers):
            sev_str = f"{sev:.3f}" if sev is not None else "-"
            wer_str = f"{wer*100:.2f}%" if wer is not None else "-"
            row += f"{f'{sev_str} / {wer_str}':>22}"
        row += f"{avg_sev:>10.3f}"
        row += f"{(f'{avg_wer*100:.2f}%' if avg_wer is not None else '-'):>10}"
        print(row)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", default="writeup_results/grid",
                        help="Directory to read grid results from (default: "
                             "writeup_results/grid - the original, unpatched folder). "
                             "Pass writeup_results/grid_calib_fixed to use the "
                             "calibration-leakage-patched results instead.")
    parser.add_argument("--verify", action="store_true",
                        help="Also run the cross-check against legacy original files "
                             "(off by default - only meaningful when --scan-dir is the "
                             "original grid folder, since the patched mirror's files "
                             "won't byte-match their legacy originals by design)")
    args = parser.parse_args()

    grid_data = load_grid_data(args.scan_dir)
    print(f"Reading grid results from: {args.scan_dir}\n")
    if args.verify:
        cross_check(grid_data)
    print_grid_table(grid_data)
    print_strategy_comparison(grid_data)


if __name__ == "__main__":
    main()