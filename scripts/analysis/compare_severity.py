"""
compare_severity.py

Pulls MAR (computed from the binary qwen_verdict_p2 field, preserved in
every severity file) and mean severity from all results/mar_severity/
severity_*.json files. Prints ONE TABLE PER DATASET, three clean columns:
N, MAR, Mean Severity — every row is the binary verdict and severity score
computed on the SAME 150-sample subset, so rows are directly comparable.

Usage:
    python scripts/analysis/compare_severity.py
    python scripts/analysis/compare_severity.py --dataset commonvoice
"""

import json
import os
import glob
import argparse
import re

SEVERITY_DIR = "results/mar_severity"

SOURCES_ORDER = ["baseline", "naive", "context_v1", "context_v2",
                 "context_v1_confidence", "context_v2_confidence", "naive_confidence"]
DATASETS_ORDER = ["commonvoice", "edacc", "english_dialects", "shetland"]


def find_severity_files():
    """
    Scan results/mar_severity/ for severity_*.json files and parse out
    (dataset, source, threshold) from the filename. Threshold-tagged files
    look like severity_{dataset}_{source}_t070_sub150.json; non-confidence
    files look like severity_{dataset}_{source}_sub150.json.
    """
    pattern = os.path.join(SEVERITY_DIR, "severity_*.json")
    files = {}

    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        name = fname[len("severity_"):-len(".json")]
        name = name.replace("_sub150", "")

        threshold = None
        thresh_match = re.search(r'_t(\d{3})$', name)
        if thresh_match:
            digits = thresh_match.group(1)
            threshold = float(f"{digits[0]}.{digits[1:]}")
            name = name[:thresh_match.start()]

        matched = False
        for dataset in DATASETS_ORDER:
            prefix = f"{dataset}_"
            if name.startswith(prefix):
                source = name[len(prefix):]
                if source in SOURCES_ORDER:
                    key = (dataset, source, threshold)
                    files[key] = path
                    matched = True
                    break
        if not matched:
            print(f"  WARNING: could not parse filename: {fname}")

    return files


def load_summary(path):
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])

    binary_valid = [s for s in samples if s.get("qwen_verdict_p2") is not None]
    binary_mar = (sum(1 for s in binary_valid if s["qwen_verdict_p2"]) / len(binary_valid)
                  if binary_valid else None)

    severity_valid = [s for s in samples if s.get("mar_severity") is not None]
    severity_mar_ge3 = (sum(1 for s in severity_valid if s["mar_severity"] >= 3) / len(severity_valid)
                        if severity_valid else None)

    return {
        "n": data.get("num_samples"),
        "mar": binary_mar,
        "mean_severity": data.get("mean_severity"),
        "severity_mar_ge3": severity_mar_ge3,
        "severity_counts": data.get("severity_counts"),
    }


def fmt_pct(val):
    return f"{val*100:.1f}%" if val is not None else "—"


def fmt_mean(val):
    return f"{val:.3f}" if val is not None else "—"


def fmt_dist(counts):
    if not counts:
        return "—"
    total = sum(counts.values())
    if total == 0:
        return "—"
    return "/".join(f"{counts.get(str(i), counts.get(i, 0))}" for i in range(5))


def row_label(source, threshold):
    if threshold is not None:
        return f"{source} (t={threshold})"
    return source


def print_dataset_table(dataset, files):
    rows = []
    for source in SOURCES_ORDER:
        # collect every threshold variant found for this source, sorted
        matches = sorted(
            [(k, v) for k, v in files.items() if k[0] == dataset and k[1] == source],
            key=lambda kv: (kv[0][2] is None, kv[0][2])
        )
        for (_, _, threshold), path in matches:
            rows.append((row_label(source, threshold), load_summary(path)))

    if not rows:
        return False

    print(f"\n{'='*68}")
    print(f"  {dataset.upper()}")
    print(f"{'='*68}")
    header = f"{'Source':<28} {'N':>4} {'MAR':>8} {'Mean Sev':>9} {'Sev MAR(>=3)':>13}"
    print(header)
    print("-" * len(header))
    for label, summary in rows:
        print(f"{label:<28} {summary['n']:>4} "
              f"{fmt_pct(summary['mar']):>8} "
              f"{fmt_mean(summary['mean_severity']):>9} "
              f"{fmt_pct(summary['severity_mar_ge3']):>13}")

    print(f"\n  Severity distribution (0/1/2/3/4):")
    for label, summary in rows:
        print(f"    {label:<28} {fmt_dist(summary['severity_counts'])}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=DATASETS_ORDER + ["all"])
    args = parser.parse_args()

    files = find_severity_files()

    if not files:
        print(f"No severity files found in {SEVERITY_DIR}/")
        return

    datasets = DATASETS_ORDER if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        print_dataset_table(dataset, files)


if __name__ == "__main__":
    main()