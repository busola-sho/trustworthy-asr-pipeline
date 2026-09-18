"""
build_grid_leaderboard.py

Compares only the 9-cell strategy x context grid stored under
writeup_results/clean_grid/. Development and test results are reported
separately using the --split argument.

This is separate from build_leaderboard.py, which covers the full historical
ensemble comparison.

CROSS-CHECK: for cells that also have an original/legacy copy elsewhere,
this cross-references the development grid copy against the legacy original
and flags any mismatch loudly. Anchored correction v1/v2 are deliberately
excluded because their prompts were edited after the grid copy was first
made; those files now represent different experiments rather than duplicates.

Usage:
    python build_grid_leaderboard.py
    python build_grid_leaderboard.py --split dev
    python build_grid_leaderboard.py --split test
    python build_grid_leaderboard.py --split dev --verify
    python build_grid_leaderboard.py --scan-dir writeup_results/clean_grid --split test
"""

import argparse
import glob
import json
from collections import defaultdict


DATASETS = ["commonvoice", "edacc", "english_dialects"]

GRID_FOLDERS = {
    "selection_naive": ("selection", "naive"),
    "selection_context_v1": ("selection", "v1"),
    "selection_context_v2": ("selection", "v2"),
    "unanchored_fusion_naive": ("unanchored_fusion", "naive"),
    "unanchored_fusion_context_v1": ("unanchored_fusion", "v1"),
    "unanchored_fusion_context_v2": ("unanchored_fusion", "v2"),
    "anchored_correction_naive": ("anchored_correction", "naive"),
    "anchored_correction_v1": ("anchored_correction", "v1"),
    "anchored_correction_v2": ("anchored_correction", "v2"),
}

# (grid folder, legacy folder, legacy filename pattern) for development-only
# cross-checking. Anchored correction v1/v2 are deliberately excluded; see
# the module docstring.
CROSS_CHECK = {
    "unanchored_fusion_naive": (
        "writeup_results/ensembles/naive/gemma4",
        "naive_{d}_gemma4sel_dev.json",
    ),
}


def normalize_dataset(name):
    if not name:
        return "unknown"
    name = name.lower()
    if "common" in name:
        return "commonvoice"
    if "english" in name or "dialect" in name:
        return "english_dialects"
    if "edacc" in name:
        return "edacc"
    return name


def load_grid_data(scan_dir, split):
    """Return {folder_name: {dataset: result_data}} for one split only."""
    data = {}

    for folder in GRID_FOLDERS:
        data[folder] = {}
        pattern = f"{scan_dir}/{folder}/*.json"

        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, encoding="utf-8") as file:
                    result = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"WARNING: could not read {path}: {exc}")
                continue

            if result.get("split") != split:
                continue
            if "mean_severity" not in result:
                continue

            dataset = normalize_dataset(result.get("dataset"))
            if dataset not in DATASETS:
                continue

            if dataset in data[folder]:
                previous = data[folder][dataset]["path"]
                raise RuntimeError(
                    f"Duplicate {split} result for {folder}/{dataset}:\n"
                    f"  {previous}\n"
                    f"  {path}"
                )

            data[folder][dataset] = {
                "mean_severity": result.get("mean_severity"),
                "corpus_wer": result.get("corpus_wer"),
                "path": path,
            }

    return data


def cross_check(grid_data):
    """Cross-check development grid copies against selected legacy files."""
    print(f"\n{'=' * 90}")
    print("  CROSS-CHECK: development grid copy vs legacy original")
    print("  (anchored_correction_v1/v2 excluded because their prompts changed)")
    print(f"{'=' * 90}")

    any_mismatch = False

    for grid_folder, (legacy_dir, pattern) in CROSS_CHECK.items():
        for dataset in DATASETS:
            legacy_path = f"{legacy_dir}/{pattern.format(d=dataset)}"

            try:
                with open(legacy_path, encoding="utf-8") as file:
                    legacy = json.load(file)
            except FileNotFoundError:
                print(
                    f"  {grid_folder}/{dataset}: legacy file not found at "
                    f"{legacy_path} - skipping"
                )
                continue
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  {grid_folder}/{dataset}: could not read legacy file: {exc}")
                continue

            legacy_severity = legacy.get("mean_severity")
            grid_result = grid_data.get(grid_folder, {}).get(dataset, {})
            grid_severity = grid_result.get("mean_severity")

            if grid_severity is None:
                print(f"  {grid_folder}/{dataset}: no development grid copy - skipping")
                continue
            if legacy_severity is None:
                print(f"  {grid_folder}/{dataset}: legacy severity missing - skipping")
                continue

            if abs(legacy_severity - grid_severity) > 0.0005:
                any_mismatch = True
                print(f"  MISMATCH: {grid_folder}/{dataset}")
                print(
                    f"    grid copy:       {grid_severity:.4f}  "
                    f"({grid_result['path']})"
                )
                print(
                    f"    legacy original: {legacy_severity:.4f}  "
                    f"({legacy_path})"
                )
            else:
                print(
                    f"  OK: {grid_folder}/{dataset} - both sources agree "
                    f"({grid_severity:.4f})"
                )

    if any_mismatch:
        print("\n  WARNING: mismatches found above.")
        print("  The legacy original is the ground truth for cross-checked files.")
    else:
        print("\n  All available cross-checked cells agree.")


def print_grid_table(grid_data, split):
    print(f"\n{'=' * 100}")
    print(f"  9-CELL GRID - {split.upper()} SPLIT (severity / WER)")
    print(f"{'=' * 100}")

    header = (
        f"{'Strategy':<22}{'Context':<10}"
        + "".join(f"{dataset:>22}" for dataset in DATASETS)
        + f"{'Avg Sev':>10}{'Avg WER':>10}"
    )
    print(header)
    print("-" * len(header))

    for folder, (strategy, context) in GRID_FOLDERS.items():
        cells = [grid_data.get(folder, {}).get(dataset, {}) for dataset in DATASETS]
        row = f"{strategy:<22}{context:<10}"

        for cell in cells:
            severity = cell.get("mean_severity")
            wer = cell.get("corpus_wer")
            severity_text = f"{severity:.3f}" if severity is not None else "-"
            wer_text = f"{wer * 100:.2f}%" if wer is not None else "-"
            row += f"{f'{severity_text} / {wer_text}':>22}"

        severities = [
            cell.get("mean_severity")
            for cell in cells
            if cell.get("mean_severity") is not None
        ]
        wers = [
            cell.get("corpus_wer")
            for cell in cells
            if cell.get("corpus_wer") is not None
        ]

        avg_severity = (
            sum(severities) / len(severities)
            if len(severities) == len(DATASETS)
            else None
        )
        avg_wer = sum(wers) / len(wers) if len(wers) == len(DATASETS) else None

        row += f"{f'{avg_severity:.3f}' if avg_severity is not None else '-':>10}"
        row += f"{f'{avg_wer * 100:.2f}%' if avg_wer is not None else '-':>10}"
        print(row)


def print_strategy_comparison(grid_data, split):
    print(f"\n{'=' * 100}")
    print(
        f"  STRATEGY COMPARISON - {split.upper()} SPLIT "
        "(best context condition per strategy, by severity)"
    )
    print(f"{'=' * 100}")

    by_strategy = defaultdict(dict)

    for folder, (strategy, context) in GRID_FOLDERS.items():
        cells = [grid_data.get(folder, {}).get(dataset, {}) for dataset in DATASETS]
        severities = [cell.get("mean_severity") for cell in cells]
        wers = [cell.get("corpus_wer") for cell in cells]

        if any(severity is None for severity in severities):
            continue

        avg_severity = sum(severities) / len(severities)
        by_strategy[strategy][context] = (avg_severity, severities, wers)

    header = (
        f"{'Strategy':<22}{'Best condition':<16}"
        + "".join(f"{dataset:>22}" for dataset in DATASETS)
        + f"{'Avg Sev':>10}{'Avg WER':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for strategy, conditions in by_strategy.items():
        best_context = min(conditions, key=lambda context: conditions[context][0])
        avg_severity, severities, wers = conditions[best_context]
        valid_wers = [wer for wer in wers if wer is not None]
        avg_wer = (
            sum(valid_wers) / len(valid_wers)
            if len(valid_wers) == len(DATASETS)
            else None
        )
        results.append(
            (strategy, best_context, severities, wers, avg_severity, avg_wer)
        )

    results.sort(key=lambda result: result[4])

    for strategy, best_context, severities, wers, avg_severity, avg_wer in results:
        row = f"{strategy:<22}{best_context:<16}"

        for severity, wer in zip(severities, wers):
            severity_text = f"{severity:.3f}" if severity is not None else "-"
            wer_text = f"{wer * 100:.2f}%" if wer is not None else "-"
            row += f"{f'{severity_text} / {wer_text}':>22}"

        row += f"{avg_severity:>10.3f}"
        row += f"{f'{avg_wer * 100:.2f}%' if avg_wer is not None else '-':>10}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-dir",
        default="writeup_results/clean_grid",
        help=(
            "Directory containing the nine grid folders "
            "(default: writeup_results/clean_grid)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "full"),
        default="dev",
        help="Dataset split to report (default: dev).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Cross-check selected development grid files against their legacy "
            "originals. This option is valid only with --split dev."
        ),
    )
    args = parser.parse_args()

    if args.verify and args.split != "dev":
        parser.error("--verify is development-only; use it with --split dev")

    grid_data = load_grid_data(args.scan_dir, args.split)
    print(f"Reading {args.split} grid results from: {args.scan_dir}")

    if args.verify:
        cross_check(grid_data)

    print_grid_table(grid_data, args.split)
    print_strategy_comparison(grid_data, args.split)


if __name__ == "__main__":
    main()
