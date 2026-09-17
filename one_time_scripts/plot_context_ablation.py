"""
plot_context_ablation.py

Severity distribution comparison across context variants (naive,
context_v1, context_v2) for a given strategy - showing the effect of
adding dataset-derived guidance to the fusion prompt. Built honestly:
the actual clean_grid numbers show a MIXED result for unanchored_fusion
(naive=1.146, v1=1.155 WORSE, v2=1.129 slightly better but NOT
statistically reliable per paired bootstrap, 95% CI [-0.051, +0.014]).
This script reports what's actually there, not a cherry-picked
"guidance helps" story.

Covers all 3 canonical strategies (selection, anchored_correction,
unanchored_fusion) - pass --strategy to pick one, defaults to
unanchored_fusion.

Usage:
    python plot_context_ablation.py --strategy unanchored_fusion
    python plot_context_ablation.py --strategy anchored_correction
    python plot_context_ablation.py --strategy selection
"""

import json
import os
import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "writeup_results/figures"
DATASETS = ["commonvoice", "edacc", "english_dialects"]
DATASET_DISPLAY = {
    "commonvoice": "CommonVoice Scottish",
    "edacc": "EdAcc",
    "english_dialects": "English Dialects",
}
CONTEXTS = ["naive", "v1", "v2"]
CONTEXT_DISPLAY = {"naive": "Naive\n(no guidance)", "v1": "Context v1", "v2": "Context v2"}

SEVERITY_STACK_COLOURS = ["#a8e6a3", "#4caf50", "#f4b942", "#e67e22", "#c0392b"]
SEVERITY_STACK_LABELS = ["0 - no change", "1 - trivial", "2 - ambiguous", "3 - factual", "4 - critical"]

FONT_TITLE = 15
FONT_SUPTITLE = 17
FONT_AXIS_LABEL = 13
FONT_TICK_LABEL = 12
FONT_LEGEND = 13
FONT_BAR_LABEL = 10


def round_percentages_to_100(values):
    floors = [int(v) for v in values]
    remainders = [v - f for v, f in zip(values, floors)]
    deficit = round(100 - sum(floors))
    order = sorted(range(len(values)), key=lambda i: remainders[i], reverse=True)
    result = floors[:]
    for i in order[:max(deficit, 0)]:
        result[i] += 1
    return result


# Explicit folder-name mapping, matching full_results_report.py's own
# GRID_FOLDERS constant exactly - NOT a fragile string-construction
# pattern. Confirmed inconsistency: anchored_correction's v1/v2
# folders have NO "context" in the name, while selection and
# unanchored_fusion DO. Assuming one pattern for all 3 strategies
# silently produced missing data for anchored_correction.
FOLDER_NAMES = {
    "selection": {
        "naive": "selection_naive",
        "v1": "selection_context_v1",
        "v2": "selection_context_v2",
    },
    "anchored_correction": {
        "naive": "anchored_correction_naive",
        "v1": "anchored_correction_v1",
        "v2": "anchored_correction_v2",
    },
    "unanchored_fusion": {
        "naive": "unanchored_fusion_naive",
        "v1": "unanchored_fusion_context_v1",
        "v2": "unanchored_fusion_context_v2",
    },
}


def load_condition(strategy, context, dataset):
    folder = FOLDER_NAMES[strategy][context]
    path = f"writeup_results/clean_grid/{folder}/{folder}_{dataset}_gemma4_dev.json"
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    samples = data.get("samples", [])
    severities = [s["severity"] for s in samples if s.get("severity") is not None]
    if not severities:
        return None
    counts = Counter(severities)
    total = len(severities)
    dist = [counts.get(i, 0) / total * 100 for i in range(5)]
    mean_sev = sum(severities) / total
    return {"dist": dist, "n": total, "mean_sev": mean_sev}


STRATEGY_DISPLAY = {
    "selection": "Selection",
    "anchored_correction": "Anchored Fusion",
    "unanchored_fusion": "Unanchored Fusion",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="unanchored_fusion",
                        choices=["selection", "anchored_correction", "unanchored_fusion"])
    args = parser.parse_args()
    strategy = args.strategy
    strategy_display = STRATEGY_DISPLAY[strategy]

    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(9, 6.5 * len(DATASETS)))

    print(f"\n{'='*80}")
    print(f"  CONTEXT ABLATION: {strategy_display} - naive vs. context_v1 vs. context_v2")
    print(f"{'='*80}")
    header = f"{'Dataset':<20}{'Naive':>14}{'Context v1':>14}{'Context v2':>14}"
    print(header)
    print("-" * len(header))

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        row_data = {}
        for context in CONTEXTS:
            result = load_condition(strategy, context, dataset)
            if result:
                row_data[context] = result

        row = f"{DATASET_DISPLAY[dataset]:<20}"
        for context in CONTEXTS:
            r = row_data.get(context)
            cell = f"{r['mean_sev']:.3f} (N={r['n']})" if r else "-"
            row += f"{cell:>14}"
        print(row)

        labels = [CONTEXT_DISPLAY[c] for c in CONTEXTS if c in row_data]
        x = np.arange(len(labels))

        rounded_per_label = {c: round_percentages_to_100(row_data[c]["dist"]) for c in CONTEXTS if c in row_data}

        bottoms = np.zeros(len(labels))
        for level in range(5):
            values = [row_data[c]["dist"][level] for c in CONTEXTS if c in row_data]
            ax.bar(x, values, bottom=bottoms, color=SEVERITY_STACK_COLOURS[level],
                  edgecolor="black", linewidth=0.6,
                  label=SEVERITY_STACK_LABELS[level] if i == 0 else None)
            for xi, (v, b) in enumerate(zip(values, bottoms)):
                if v >= 4:
                    c = [c for c in CONTEXTS if c in row_data][xi]
                    rv = rounded_per_label[c][level]
                    ax.text(xi, b + v / 2, f"{rv}%", ha="center", va="center",
                            fontsize=FONT_BAR_LABEL, color="black" if level < 3 else "white",
                            fontweight="bold")
            bottoms += np.array(values)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONT_TICK_LABEL)
        ax.tick_params(axis="y", labelsize=FONT_TICK_LABEL)
        ax.set_ylabel("% of samples", fontsize=FONT_AXIS_LABEL)
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=FONT_TITLE, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    legend = fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=FONT_LEGEND,
                        frameon=True, markerscale=2.0, title="Severity", title_fontsize=FONT_LEGEND + 1)
    for handle in legend.legend_handles:
        handle.set_edgecolor("black")
        handle.set_linewidth(1.5)

    fig.suptitle(f"Context Guidance Ablation: {strategy_display}\n(dev split)",
                fontsize=FONT_SUPTITLE, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0, 0.85, 0.98])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f"context_ablation_{strategy}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
