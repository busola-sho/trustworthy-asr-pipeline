"""
summarise_by_selector.py

Prints one table per selector (gemma4, qwen, phi4) - rows are ensemble
techniques, columns are the 4 datasets, values are mean severity. Lets
you check at a glance whether naive stays the best technique across all
three selectors, or whether something switches.

Techniques with no "selector" field (rover, mbr_consensus - purely
mechanical, no LLM involved) are shown once in a separate reference
table at the end, since they're identical regardless of selector.

Usage:
    python summarise_by_selector.py
"""

import json
import glob
import os
from collections import defaultdict

SEARCH_DIRS = [
    "writeup_results/ensembles/*/*.json",
    "results/ensembles/*/*.json",
]

DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]
SELECTORS = ["gemma4", "qwen", "phi4"]

EXCLUDED_APPROACHES = {
    "naive_confscore", "naive_probscore",
    "naive_confscore_meaning", "naive_probscore_meaning",
}


def normalize_dataset_name(name):
    if not name:
        return "unknown"
    name = name.lower()
    if "common" in name:
        return "commonvoice"
    if "english" in name or "dialect" in name:
        return "english_dialects"
    if "edacc" in name:
        return "edacc"
    if "shetland" in name:
        return "shetland"
    return name


def load_all_rows():
    rows = []
    for pattern in SEARCH_DIRS:
        for path in glob.glob(pattern):
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:
                continue
            if "approach" not in data:
                continue   # skip baseline (individual ASR model) files

            approach = data.get("approach")
            if approach in EXCLUDED_APPROACHES:
                continue

            selector = data.get("selector")   # None for mechanical techniques
            dataset = normalize_dataset_name(data.get("dataset"))
            percentile = data.get("percentile")
            mean_severity = data.get("mean_severity")
            corpus_wer = data.get("corpus_wer")
            split = data.get("split")

            # only dev/full splits (never test - test stays out of technique comparison)
            if split not in ("dev", "full"):
                continue

            label = approach
            if percentile is not None:
                label = f"{approach} (p{percentile})"

            rows.append({
                "path": path,
                "label": label,
                "selector": selector,
                "dataset": dataset,
                "mean_severity": mean_severity,
                "corpus_wer": corpus_wer,
            })
    return rows


def dedupe(rows):
    """Keeps one entry per (label, selector, dataset) - true duplicates
    only (same technique, same selector, same dataset), NOT different
    selectors of the same technique, which was the bug being fixed."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["label"], r["selector"], r["dataset"])
        groups[key].append(r)

    deduped = []
    for key, group in groups.items():
        deduped.append(group[0])
        if len(group) > 1:
            print(f"  NOTE: {len(group)} files for {key} - kept {group[0]['path']}")
    return deduped


def fmt_sev(sev):
    return f"{sev:.3f}" if sev is not None else "-"


def print_selector_table(selector, rows):
    matching = [r for r in rows if r["selector"] == selector]
    if not matching:
        print(f"\n(no data found for selector={selector})")
        return

    by_label = defaultdict(dict)
    for r in matching:
        by_label[r["label"]][r["dataset"]] = r["mean_severity"]

    def avg_key(label):
        sevs = [by_label[label].get(d) for d in ["commonvoice", "english_dialects", "edacc"]]
        sevs = [s for s in sevs if s is not None]
        return sum(sevs) / len(sevs) if sevs else float("inf")

    labels = sorted(by_label.keys(), key=avg_key)

    print(f"\n{'='*100}")
    print(f"  SELECTOR: {selector}")
    print(f"{'='*100}")
    header = f"{'Technique':<32}" + "".join(f"{d:>18}" for d in DATASETS) + f"{'Avg(3)':>10}"
    print(header)
    print("-" * len(header))
    for label in labels:
        row = f"{label:<32}"
        for d in DATASETS:
            row += f"{fmt_sev(by_label[label].get(d)):>18}"
        avg = avg_key(label)
        row += f"{avg:>10.3f}" if avg != float("inf") else f"{'-':>10}"
        print(row)


def print_reference_table(rows):
    matching = [r for r in rows if r["selector"] is None]
    if not matching:
        return

    by_label = defaultdict(dict)
    for r in matching:
        by_label[r["label"]][r["dataset"]] = r["mean_severity"]

    print(f"\n{'='*100}")
    print(f"  SELECTOR-INDEPENDENT (mechanical - same regardless of selector)")
    print(f"{'='*100}")
    header = f"{'Technique':<32}" + "".join(f"{d:>18}" for d in DATASETS)
    print(header)
    print("-" * len(header))
    for label, ds in sorted(by_label.items()):
        row = f"{label:<32}"
        for d in DATASETS:
            row += f"{fmt_sev(ds.get(d)):>18}"
        print(row)


def main():
    print("Scanning result files...")
    rows = load_all_rows()
    print(f"Found {len(rows)} row(s) before dedup")
    rows = dedupe(rows)
    print(f"{len(rows)} row(s) after dedup")

    for selector in SELECTORS:
        print_selector_table(selector, rows)

    print_reference_table(rows)


if __name__ == "__main__":
    main()
