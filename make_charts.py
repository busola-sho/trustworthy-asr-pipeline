"""
make_charts.py

Generates the presentation charts for your dissertation results, reusing
build_leaderboard.py's data-loading logic (same file scanning, dataset/model
name normalization, split-restricted baseline recomputation) so the charts
and the printed tables are always built from identical numbers.

Produces:
  1. naive_vs_baselines.png - grouped bar chart: naive vs each baseline
     model, dev and test side by side, on mean severity (lower = better)
  2. ensemble_trend_dev.png / ensemble_trend_test.png - multi-line trend
     chart: one line per ensemble technique, x-axis = dataset, y-axis =
     mean severity - shows whether a technique's relative ranking holds
     steady or swings across datasets
  3. If shetland data is present under any of DEFAULT_ROOTS, it's
     automatically included as an extra x-axis category on the trend
     charts and picked up in the baseline chart too - no code changes
     needed once you've run your winning technique(s) on it.

Usage:
    python make_charts.py
    python make_charts.py --out-dir charts
"""

import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")   # no display needed - just save PNG files
import matplotlib.pyplot as plt

from build_leaderboard import (
    DEFAULT_ROOTS, DATASET_ORDER, find_result_files, extract_row,
    dedupe_rows, recompute_baseline_for_split, best_baseline_per_dataset,
)


def load_all_rows(roots, split):
    found = find_result_files(roots)
    rows = [extract_row(path, data) for path, data in found]

    ensemble_rows = [r for r in rows if not r["is_baseline"]]
    baseline_rows = [r for r in rows if r["is_baseline"]]

    ensemble_rows = [r for r in ensemble_rows if r["split"] == split]
    baseline_rows = [recompute_baseline_for_split(r, split) for r in baseline_rows]

    ensemble_rows = dedupe_rows(ensemble_rows)
    baseline_rows = dedupe_rows(baseline_rows)

    return ensemble_rows, baseline_rows


def get_present_datasets(*row_lists):
    """Returns DATASET_ORDER plus any extra datasets found (e.g. shetland,
    once you've run something on it) - so charts automatically extend
    once new data shows up, no code changes needed."""
    found = set()
    for rows in row_lists:
        for r in rows:
            found.add(r["dataset"])
    extra = sorted(d for d in found if d not in DATASET_ORDER)
    return DATASET_ORDER + extra


def chart_naive_vs_baselines(out_dir):
    """Grouped bar chart: naive technique vs each baseline model, dev and
    test shown side by side (if test data is present) - the headline
    'does my winning technique actually beat the individual models' chart."""
    dev_ensemble, dev_baseline = load_all_rows(DEFAULT_ROOTS, "dev")
    try:
        test_ensemble, test_baseline = load_all_rows(DEFAULT_ROOTS, "test")
    except Exception:
        test_ensemble, test_baseline = [], []

    def avg_severity(rows, approach):
        sevs = [r["mean_severity"] for r in rows if r["approach"] == approach and r["mean_severity"] is not None]
        return sum(sevs) / len(sevs) if sevs else None

    baseline_models = sorted({r["approach"] for r in dev_baseline})
    categories = ["naive"] + baseline_models

    dev_values = [avg_severity(dev_ensemble, "naive")] + [avg_severity(dev_baseline, m) for m in baseline_models]
    test_values_raw = [avg_severity(test_ensemble, "naive")] + [avg_severity(test_baseline, m) for m in baseline_models]

    # only show test bars if there's at least one GENUINELY usable value -
    # baseline rows always come back as *something* (never split-filtered
    # to empty), even when there's no real test-split ensemble data yet,
    # so an empty-list check alone isn't enough here
    has_test = any(v is not None for v in test_values_raw)

    # matplotlib's bar() handles NaN by leaving a gap rather than crashing
    # (unlike None, which errors) - convert missing values so a partially
    # incomplete comparison still renders instead of failing outright
    nan = float("nan")
    dev_values = [v if v is not None else nan for v in dev_values]
    test_values = [v if v is not None else nan for v in test_values_raw] if has_test else None

    x = range(len(categories))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if has_test:
        width = 0.35
        ax.bar([i - width/2 for i in x], dev_values, width, label="Dev", color="#4C72B0")
        ax.bar([i + width/2 for i in x], test_values, width, label="Test", color="#DD8452")
    else:
        ax.bar(x, dev_values, color="#4C72B0", label="Dev")

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Mean Severity (lower = better)")
    ax.set_title("Naive Ensemble vs. Individual Baseline Models\n(average across all datasets)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax.get_xticklabels()[0].set_fontweight("bold")

    fig.tight_layout()
    out_path = Path(out_dir) / "naive_vs_baselines.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def chart_ensemble_trend(out_dir, split):
    """Multi-line trend chart: one line per ensemble technique, x-axis =
    dataset, y-axis = mean severity."""
    try:
        ensemble_rows, baseline_rows = load_all_rows(DEFAULT_ROOTS, split)
    except Exception as e:
        print(f"Skipping {split} trend chart - could not load data: {e}")
        return

    if not ensemble_rows:
        print(f"Skipping {split} trend chart - no ensemble data found for split='{split}'")
        return

    datasets = get_present_datasets(ensemble_rows)

    by_approach = defaultdict(dict)
    for r in ensemble_rows:
        by_approach[r["approach"]][r["dataset"]] = r["mean_severity"]

    fig, ax = plt.subplots(figsize=(9, 6))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    cmap = plt.get_cmap("tab10")

    for i, approach in enumerate(sorted(by_approach.keys())):
        y_values = [by_approach[approach].get(ds) for ds in datasets]
        xs = [j for j, v in enumerate(y_values) if v is not None]
        ys = [v for v in y_values if v is not None]
        if not ys:
            continue
        is_naive = (approach == "naive")
        ax.plot(
            xs, ys,
            marker=markers[i % len(markers)],
            color=cmap(i % 10),
            linewidth=3 if is_naive else 1.5,
            markersize=9 if is_naive else 6,
            label=approach + ("  (winner)" if is_naive else ""),
            zorder=10 if is_naive else 5,
        )

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean Severity (lower = better)")
    ax.set_title(f"Ensemble Technique Performance Across Datasets ({split})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_dir) / f"ensemble_trend_{split}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="charts")
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chart_naive_vs_baselines(out_dir)
    chart_ensemble_trend(out_dir, "dev")
    chart_ensemble_trend(out_dir, "test")


if __name__ == "__main__":
    main()