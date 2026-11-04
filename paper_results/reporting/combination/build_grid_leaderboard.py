"""Compare saved fusion runs by dataset, method and split."""

import argparse
import glob
import json
from collections import defaultdict

from paper_results.reporting.common import is_tag_only


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


def calculate_mar(samples):
    severities = []
    for sample in samples or []:
        if sample.get("skipped") or sample.get("error"):
            continue

        reference = (sample.get("ref") or "").strip()
        if not reference or is_tag_only(reference):
            continue

        hypothesis = (sample.get("hyp") or "").strip()
        severity = sample.get("severity")
        if severity is None and not hypothesis:
            severity = 4
        if severity is not None:
            severities.append(severity)

    if not severities:
        return None
    return sum(severity >= 2 for severity in severities) / len(severities)


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

            if result.get("split") != split or "mean_severity" not in result:
                continue

            dataset = normalize_dataset(result.get("dataset"))
            if dataset not in DATASETS:
                continue
            if dataset in data[folder]:
                previous = data[folder][dataset]["path"]
                raise RuntimeError(
                    f"Duplicate {split} result for {folder}/{dataset}:\n"
                    f"  {previous}\n  {path}"
                )

            mar = calculate_mar(result.get("samples"))
            if mar is None:
                print(f"WARNING: could not compute MAR from {path}")

            data[folder][dataset] = {
                "mean_severity": result.get("mean_severity"),
                "corpus_wer": result.get("corpus_wer"),
                "mar": mar,
                "path": path,
            }
    return data


def complete_average(values):
    if len(values) != len(DATASETS) or any(value is None for value in values):
        return None
    return sum(values) / len(values)


def format_severity(value):
    return f"{value:.3f}" if value is not None else "-"


def format_percentage(value):
    return f"{value * 100:.2f}%" if value is not None else "-"


def print_grid_table(grid_data, split):
    print(f"\n{'=' * 112}")
    print(f"  9-CELL GRID - {split.upper()} SPLIT (severity / WER)")
    print("  MAR = percentage of scored samples with severity >= 2")
    print(f"{'=' * 112}")

    header = (
        f"{'Strategy':<22}{'Context':<10}"
        + "".join(f"{dataset:>22}" for dataset in DATASETS)
        + f"{'Avg Sev':>10}{'Avg WER':>10}{'Avg MAR':>10}"
    )
    print(header)
    print("-" * len(header))

    for folder, (strategy, context) in GRID_FOLDERS.items():
        cells = [grid_data.get(folder, {}).get(dataset, {}) for dataset in DATASETS]
        row = f"{strategy:<22}{context:<10}"
        for cell in cells:
            text = (
                f"{format_severity(cell.get('mean_severity'))} / "
                f"{format_percentage(cell.get('corpus_wer'))}"
            )
            row += f"{text:>22}"

        avg_severity = complete_average([c.get("mean_severity") for c in cells])
        avg_wer = complete_average([c.get("corpus_wer") for c in cells])
        avg_mar = complete_average([c.get("mar") for c in cells])
        row += f"{format_severity(avg_severity):>10}"
        row += f"{format_percentage(avg_wer):>10}"
        row += f"{format_percentage(avg_mar):>10}"
        print(row)


def print_strategy_comparison(grid_data, split):
    print(f"\n{'=' * 112}")
    print(
        f"  STRATEGY COMPARISON - {split.upper()} SPLIT "
        "(best context condition per strategy, by severity)"
    )
    print(f"{'=' * 112}")

    by_strategy = defaultdict(dict)
    for folder, (strategy, context) in GRID_FOLDERS.items():
        cells = [grid_data.get(folder, {}).get(dataset, {}) for dataset in DATASETS]
        severities = [cell.get("mean_severity") for cell in cells]
        wers = [cell.get("corpus_wer") for cell in cells]
        mars = [cell.get("mar") for cell in cells]
        avg_severity = complete_average(severities)
        if avg_severity is not None:
            by_strategy[strategy][context] = (
                avg_severity,
                severities,
                wers,
                mars,
            )

    header = (
        f"{'Strategy':<22}{'Best condition':<16}"
        + "".join(f"{dataset:>22}" for dataset in DATASETS)
        + f"{'Avg Sev':>10}{'Avg WER':>10}{'Avg MAR':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for strategy, conditions in by_strategy.items():
        best_context = min(conditions, key=lambda context: conditions[context][0])
        avg_severity, severities, wers, mars = conditions[best_context]
        results.append(
            (
                strategy,
                best_context,
                severities,
                wers,
                avg_severity,
                complete_average(wers),
                complete_average(mars),
            )
        )

    results.sort(key=lambda result: result[4])
    for strategy, context, severities, wers, avg_sev, avg_wer, avg_mar in results:
        row = f"{strategy:<22}{context:<16}"
        for severity, wer in zip(severities, wers):
            text = f"{format_severity(severity)} / {format_percentage(wer)}"
            row += f"{text:>22}"
        row += f"{format_severity(avg_sev):>10}"
        row += f"{format_percentage(avg_wer):>10}"
        row += f"{format_percentage(avg_mar):>10}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Print the saved nine-cell grid")
    parser.add_argument("--scan-dir", required=True)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    args = parser.parse_args()
    data = load_grid_data(args.scan_dir, args.split)
    print(f"Reading {args.split} grid results from: {args.scan_dir}")
    print_grid_table(data, args.split)
    print_strategy_comparison(data, args.split)


if __name__ == "__main__":
    raise SystemExit("Use python -m asr_pipeline report with a separate --output-dir")
