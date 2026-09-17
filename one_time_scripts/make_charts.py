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

FAMILY_COLORS = {
    "baseline": "#4C72B0",     # blue
    "llm_fusion": "#DD8452",   # orange
    "rover": "#55A868",        # green
}


def get_family(row) -> str:
    if row["is_baseline"]:
        return "baseline"
    if row["approach"] == "rover":
        return "rover"
    return "llm_fusion"


def macro_avg(rows, approach):
    """Mean severity/WER for one approach, averaged across whichever
    datasets it has data for (equal weight per dataset)."""
    matching = [r for r in rows if r["approach"] == approach]
    sevs = [r["mean_severity"] for r in matching if r["mean_severity"] is not None]
    wers = [r["corpus_wer"] for r in matching if r["corpus_wer"] is not None]
    avg_sev = sum(sevs) / len(sevs) if sevs else None
    avg_wer = sum(wers) / len(wers) if wers else None
    return avg_sev, avg_wer


def severity_histogram(rows_for_one_technique):
    """Pooled severity histogram (counts at 0/1/2/3/4) across whichever
    datasets are provided for one technique/model - reads real per-sample
    severity values, same source data as compute_pooled_metrics."""
    counts = [0, 0, 0, 0, 0]
    for row in rows_for_one_technique:
        samples = row["raw_data"].get("samples")
        if not samples:
            continue
        for s in samples:
            if s.get("skipped") or s.get("error"):
                continue
            sev = s.get("severity")
            if sev is not None and 0 <= int(sev) <= 4:
                counts[int(sev)] += 1
    return counts


def load_dev_test_shetland(roots=DEFAULT_ROOTS):
    """Loads ensemble+baseline rows for dev, test, and shetland in one
    call - shetland rows are already always kept regardless of split
    filtering (see build_leaderboard.py), so requesting any split still
    surfaces shetland's own rows; this just also explicitly separates
    them out for panels/charts that need shetland on its own."""
    dev_ensemble, dev_baseline = load_all_rows(roots, "dev")
    try:
        test_ensemble, test_baseline = load_all_rows(roots, "test")
    except Exception:
        test_ensemble, test_baseline = [], []

    shetland_ensemble = [r for r in dev_ensemble if r["dataset"] == "shetland"]
    shetland_baseline = [r for r in dev_baseline if r["dataset"] == "shetland"]
    # exclude shetland from the dev/test macro-average pools - it's a
    # separate held-out panel, never folded into dev/test averages
    dev_ensemble = [r for r in dev_ensemble if r["dataset"] != "shetland"]
    dev_baseline = [r for r in dev_baseline if r["dataset"] != "shetland"]
    test_ensemble = [r for r in test_ensemble if r["dataset"] != "shetland"]
    test_baseline = [r for r in test_baseline if r["dataset"] != "shetland"]

    return {
        "dev": (dev_ensemble, dev_baseline),
        "test": (test_ensemble, test_baseline),
        "shetland": (shetland_ensemble, shetland_baseline),
    }



def load_all_rows(roots, split):
    found = find_result_files(roots)
    rows = [extract_row(path, data) for path, data in found]

    ensemble_rows = [r for r in rows if not r["is_baseline"]]
    baseline_rows = [r for r in rows if r["is_baseline"]]

    # Shetland ensemble rows are always split='full' (never dev/test) -
    # keep them regardless of the requested split, same fix already
    # applied in build_leaderboard.py's main(). Without this, a row like
    # naive's Shetland result (split='full') gets silently dropped
    # whenever this is called with split='dev' or 'test', since 'full'
    # never equals either - this was a real bug (confirmed via a real
    # data check: the file has 100 genuine samples with real severity
    # values, they just never reached this far).
    ensemble_rows = [r for r in ensemble_rows if r["split"] == split or r["dataset"] == "shetland"]
    baseline_rows = [recompute_baseline_for_split(r, split) for r in baseline_rows]

    ensemble_rows = dedupe_rows(ensemble_rows)
    baseline_rows = dedupe_rows(baseline_rows)

    return ensemble_rows, baseline_rows


def chart_wer_vs_severity_scatter(out_dir):
    """
    CHART 1 (main figure): WER vs Mean Severity scatter, 3 panels
    (Dev / Test / Shetland). Each point = one method (one technique or
    baseline model), colored by family (baseline / LLM-fusion / rover).
    The ideal method sits bottom-left (low WER, low severity) - this
    shows the WER/severity trade-off directly, rather than requiring the
    reader to cross-reference two separate tables.
    """
    panels = load_dev_test_shetland()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)

    for ax, panel_name in zip(axes, ["dev", "test", "shetland"]):
        ensemble_rows, baseline_rows = panels[panel_name]
        all_rows = ensemble_rows + baseline_rows

        if not all_rows:
            ax.set_title(f"{panel_name.capitalize()} (no data)")
            ax.axis("off")
            continue

        approaches = sorted({r["approach"] for r in all_rows})
        seen_families = set()

        for approach in approaches:
            matching = [r for r in all_rows if r["approach"] == approach]
            family = get_family(matching[0])

            if panel_name == "shetland":
                # shetland is a single dataset - use its value directly,
                # no averaging needed
                sevs = [r["mean_severity"] for r in matching if r["mean_severity"] is not None]
                wers = [r["corpus_wer"] for r in matching if r["corpus_wer"] is not None]
                sev = sevs[0] if sevs else None
                wer_val = wers[0] if wers else None
            else:
                sev, wer_val = macro_avg(all_rows, approach)

            if sev is None or wer_val is None:
                continue

            label = family if family not in seen_families else None
            seen_families.add(family)

            is_naive = (approach == "naive")
            ax.scatter(
                wer_val * 100, sev,
                color=FAMILY_COLORS[family],
                s=130 if is_naive else 70,
                marker="*" if is_naive else "o",
                edgecolors="black" if is_naive else "none",
                linewidths=1.2 if is_naive else 0,
                label=label,
                zorder=10 if is_naive else 5,
            )
            # annotate naive and rover explicitly since they're referenced
            # by name throughout the dissertation text
            if approach in ("naive", "rover"):
                ax.annotate(approach, (wer_val * 100, sev), fontsize=8,
                            xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("Corpus WER (%)")
        ax.set_ylabel("Mean Severity")
        ax.set_title(panel_name.capitalize())
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    # merge legend entries across panels (some families may only appear
    # in one panel, e.g. rover missing from a panel with no rover data)
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in all_labels:
                all_handles.append(hh)
                all_labels.append(ll)
    fig.legend(all_handles, all_labels, loc="upper center", ncol=len(all_labels),
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("WER vs. Mean Severity by Method (lower-left = better)", y=1.12)
    fig.tight_layout()
    out_path = Path(out_dir) / "wer_vs_severity_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def chart_dataset_comparison_bars(out_dir, split="dev", context_technique="context_v1"):
    """
    CHART 2: grouped horizontal bar chart. Datasets on the y-axis
    (CommonVoice, English Dialects, EdAcc, Shetland), bars for exactly 4
    curated methods per dataset: best baseline, naive, context_technique
    (FIXED across every dataset - default context_v1 - for a clean,
    consistent comparison rather than a different context variant per
    dataset), and rover. Severity is the bar length (the thesis's
    primary metric); WER is annotated as text on each bar.
    """
    panels = load_dev_test_shetland()
    ensemble_rows, baseline_rows = panels[split]
    shetland_ensemble, shetland_baseline = panels["shetland"]

    all_ensemble = ensemble_rows + shetland_ensemble
    all_baseline = baseline_rows + shetland_baseline
    datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in all_ensemble)] + ["shetland"]
    datasets = [d for i, d in enumerate(datasets) if d not in datasets[:i]]  # dedupe, keep order

    CATEGORY_KEYS = ["best_baseline", "naive", "context", "rover"]
    method_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    data = {k: [] for k in CATEGORY_KEYS}
    wer_annotations = {k: [] for k in CATEGORY_KEYS}
    # legend labels can legitimately vary per dataset (e.g. "best baseline"
    # is a different actual model per dataset) - kept separate from the
    # fixed CATEGORY_KEYS used purely for consistent grouping/plotting
    legend_labels = {k: k for k in CATEGORY_KEYS}
    legend_labels["context"] = context_technique

    for ds in datasets:
        methods = _select_curated_methods(all_ensemble, all_baseline, ds, context_technique)
        for key, (label, rows) in zip(CATEGORY_KEYS, methods):
            row = rows[0] if rows else None
            data[key].append(row["mean_severity"] if row else float("nan"))
            wer_annotations[key].append(row["corpus_wer"] if row else None)

    method_labels = [legend_labels[k] for k in CATEGORY_KEYS]

    fig, ax = plt.subplots(figsize=(10, 1.6 * len(datasets) + 2))

    bar_height = 0.2
    y_positions = range(len(datasets))

    for i, (key, label) in enumerate(zip(CATEGORY_KEYS, method_labels)):
        offsets = [y + (i - 1.5) * bar_height for y in y_positions]
        bars = ax.barh(offsets, data[key], height=bar_height, label=label, color=method_colors[i])
        for bar, wer_val in zip(bars, wer_annotations[key]):
            if wer_val is not None:
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{wer_val*100:.1f}% WER", va="center", fontsize=7.5)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Mean Severity (lower = better) — WER annotated on each bar")
    ax.set_title(f"Best Method Comparison by Dataset ({split})")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_dir) / f"dataset_comparison_bars_{split}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _select_curated_methods(ensemble_rows, baseline_rows, dataset, context_technique="context_v1"):
    """
    Picks the same 4 curated methods for one dataset: best baseline
    (lowest severity baseline model for THIS dataset), naive,
    context_technique (FIXED across every panel - default context_v1,
    your other near-winner on test - deliberately NOT re-picked per
    dataset, so every panel compares the literal same 4 methods rather
    than a different context variant each time), rover. Returns a list
    of (label, rows) pairs - rows is empty for any method that has no
    data for this dataset (e.g. Shetland has no context/rover runs),
    which callers render as an empty/missing group rather than crashing.
    """
    ds_ensemble = [r for r in ensemble_rows if r["dataset"] == dataset]
    ds_baseline = [r for r in baseline_rows if r["dataset"] == dataset]

    best_base = min(
        (r for r in ds_baseline if r["mean_severity"] is not None),
        key=lambda r: r["mean_severity"], default=None
    )
    naive_row = next((r for r in ds_ensemble if r["approach"] == "naive"), None)
    context_row = next((r for r in ds_ensemble if r["approach"] == context_technique), None)
    rover_row = next((r for r in ds_ensemble if r["approach"] == "rover"), None)

    def rows_for(row):
        return [row] if row else []

    labels = [
        f"best baseline\n({best_base['approach']})" if best_base else "best baseline\n(n/a)",
        "naive",
        context_technique,
        "rover",
    ]
    return list(zip(labels, [rows_for(best_base), rows_for(naive_row),
                              rows_for(context_row), rows_for(rover_row)]))



def chart_severity_grouped_bars(out_dir, split="dev", context_technique="context_v1"):
    """
    CHART 3 (revised): grouped (side-by-side) bar chart of severity
    distribution. Bars show severity proportions only (0-100%) - no
    per-bar percentage labels, since that would duplicate what the bar
    height already shows and clutter badly across 4 methods x 5 bars x
    4 panels. Instead, the two headline summary numbers (mean severity,
    WER) live in the x-axis tick label under each method name, and a
    delta vs. best baseline is shown for naive/context_technique
    specifically - this lets a reader connect the shape of the
    distribution to the two summary metrics without hunting for them
    elsewhere. Each panel's sample size (n=) is in its title.
    """
    panels = load_dev_test_shetland()
    ensemble_rows, baseline_rows = panels[split]
    shetland_ensemble, shetland_baseline = panels["shetland"]

    all_ensemble = ensemble_rows + shetland_ensemble
    all_baseline = baseline_rows + shetland_baseline

    datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in all_ensemble + all_baseline)]
    datasets += ["shetland"] if any(r["dataset"] == "shetland" for r in all_ensemble + all_baseline) else []

    severity_colors = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]  # green -> red, 0 to 4

    n_panels = len(datasets)
    n_cols = 2
    n_rows = (n_panels + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 5.6 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    bar_width = 0.15
    DELTA_ELIGIBLE = {"naive", context_technique}

    for panel_idx, dataset in enumerate(datasets):
        ax = axes_flat[panel_idx]
        methods = _select_curated_methods(all_ensemble, all_baseline, dataset, context_technique)

        # best baseline's row (first entry) - used to compute delta for
        # naive/context_technique specifically
        best_base_row = methods[0][1][0] if methods[0][1] else None

        tick_labels = []
        n_for_title = None

        for m_idx, (label, rows) in enumerate(methods):
            counts = severity_histogram(rows)
            total = sum(counts)
            pcts = [(c / total * 100) if total else 0 for c in counts]

            if total and n_for_title is None:
                n_for_title = total  # same scored population size for every
                                      # method in a dataset - just need one

            x_base = m_idx
            for sev_level in range(5):
                x = x_base + (sev_level - 2) * bar_width
                ax.bar(x, pcts[sev_level], width=bar_width * 0.9,
                       color=severity_colors[sev_level],
                       label=f"Severity {sev_level}" if (panel_idx == 0 and m_idx == 0) else None)

            # build the tick label: method name, then mu/WER, then delta
            # (only for naive/context_technique, only when both this row
            # and the best-baseline row have real values)
            row = rows[0] if rows else None
            if row and row["mean_severity"] is not None and row["corpus_wer"] is not None:
                stats_line = f"μ={row['mean_severity']:.3f} | WER={row['corpus_wer']*100:.2f}%"
                extra_line = ""
                base_label = label.split("\n")[0]  # strip any existing "(model)" sub-label
                if base_label in DELTA_ELIGIBLE and best_base_row and best_base_row["mean_severity"] is not None:
                    delta = row["mean_severity"] - best_base_row["mean_severity"]
                    extra_line = f"\nΔ={delta:+.3f}"
                tick_labels.append(f"{label}\n{stats_line}{extra_line}")
            else:
                tick_labels.append(f"{label}\n(no data)")

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(tick_labels, fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of samples")
        n_str = f" (n={n_for_title:,})" if n_for_title else ""
        ax.set_title(f"{dataset}{n_str}")
        ax.grid(axis="y", alpha=0.3)

    # hide any unused trailing subplot (odd number of datasets)
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels_ = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, title="Severity", loc="upper center",
               ncol=5, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"Severity Distribution by Method, per Dataset ({split})", y=1.10)
    fig.text(0.5, -0.01, "Note: lower mean severity (μ) and lower WER are better.",
              ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    out_path = Path(out_dir) / f"severity_grouped_bars_{split}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # pooled (micro-average) severity summary, ready to copy into the
    # dissertation text below the figure - computed the same way as
    # build_leaderboard.py's pooled table (concatenate every dataset's
    # samples for a technique, take one mean over all of them), NOT a
    # re-average of per-dataset means
    summary_methods = ["naive", context_technique, "rover"]
    pooled_sevs = {}
    for approach in summary_methods:
        rows_for_approach = [r for r in all_ensemble if r["approach"] == approach]
        counts = severity_histogram(rows_for_approach)
        total = sum(counts)
        pooled_sevs[approach] = (sum(i * c for i, c in enumerate(counts)) / total) if total else None

    if all(v is not None for v in pooled_sevs.values()):
        verb = "achieved" if split == "dev" else "retained"
        print(f"\nPooled severity summary ({split}) - paste into dissertation text:")
        print(f'  "Naive {verb} the best pooled mean severity: {pooled_sevs["naive"]:.3f}, '
              f'compared with {pooled_sevs[context_technique]:.3f} for {context_technique} '
              f'and {pooled_sevs["rover"]:.3f} for ROVER."')


def chart_dev_test_slope(out_dir, top_n=5):
    """
    CHART 4: dev -> test slope/dumbbell chart for the top N methods by
    dev severity - tells the generalisation/stability story directly:
    lines that stay flat and close together show the ranking held up
    out-of-sample; a line that swings a lot flags a method whose dev
    result may have been partly noise. Shetland is NOT included here
    (it's the true external holdout, shown separately in other charts) -
    mixing it into a dev/test slope would misrepresent it as just
    another split.
    """
    panels = load_dev_test_shetland()
    dev_ensemble, dev_baseline = panels["dev"]
    test_ensemble, test_baseline = panels["test"]

    all_dev = dev_ensemble + dev_baseline
    approaches = sorted({r["approach"] for r in all_dev})

    dev_sevs = {a: macro_avg(all_dev, a)[0] for a in approaches}
    test_sevs = {a: macro_avg(test_ensemble + test_baseline, a)[0] for a in approaches}

    ranked = sorted(
        [a for a in approaches if dev_sevs[a] is not None],
        key=lambda a: dev_sevs[a]
    )[:top_n]

    fig, ax = plt.subplots(figsize=(7, 6))

    for approach in ranked:
        dev_val = dev_sevs[approach]
        test_val = test_sevs.get(approach)
        is_naive = (approach == "naive")
        color = "#DD8452" if is_naive else "#4C72B0"

        if test_val is not None:
            ax.plot([0, 1], [dev_val, test_val], marker="o",
                    color=color, linewidth=2.5 if is_naive else 1.5,
                    markersize=9 if is_naive else 6, zorder=10 if is_naive else 5)
            ax.annotate(approach, (1.02, test_val), fontsize=9, va="center")
        else:
            ax.scatter([0], [dev_val], color=color, s=60)
            ax.annotate(f"{approach} (no test data)", (0.02, dev_val), fontsize=8, va="center")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Dev", "Test"])
    ax.set_xlim(-0.15, 1.5)
    ax.set_ylabel("Mean Severity (lower = better)")
    ax.set_title(f"Dev -> Test Generalisation (top {top_n} methods by dev severity)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_dir) / "dev_test_slope.png"
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

    chart_wer_vs_severity_scatter(out_dir)
    chart_dataset_comparison_bars(out_dir, "dev")
    chart_severity_grouped_bars(out_dir, "dev")
    chart_severity_grouped_bars(out_dir, "test")
    chart_dev_test_slope(out_dir, top_n=5)


if __name__ == "__main__":
    main()