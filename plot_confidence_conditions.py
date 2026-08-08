"""
plot_confidence_conditions.py

Same visual design/improvements as plot_strategy_severity.py (larger
fonts throughout, vertical panel stacking, bold side-positioned
legend), but x-axis = CONFIDENCE CONDITION instead of strategy, all on
the locked winning strategy (Unanchored Fusion + Naive).

Conditions (left to right):
  Baseline (no confidence)
  List (best threshold: p5 - confirmed via full p5/p10/p20 sweep,
        avg severity 1.214 vs 1.220 at p10, 1.237 at p20)
  Inline (best threshold: p10 - avg severity 1.212 vs 1.228 at p5,
        1.224 at p20)
  List (unfiltered, v2 compliance-checked)
  Inline (unfiltered, v2 compliance-checked)

Only the best-performing threshold per format is featured, rather than
all 3 (largely redundant once the sweep confirmed the winner) - the
full p5/p10/p20 comparison remains available separately if needed.

Confidence experiment files were all run AFTER the candidate_pool.json
naming fix, so they already correctly exclude calibration samples -
no _calib_fixed mirror needed for these specifically. The baseline
(no confidence) condition uses the grid_calib_fixed mirror, same as
plot_strategy_severity.py.

FIX: labels are already multi-line wrapped (embedded \\n) - rotating
them on top of that was redundant and made the chart less space-
efficient (awkward diagonal multi-line stacking). Labels now sit
horizontal (rotation=0, center-aligned), which is more compact and
readable given they're already broken into short wrapped lines.

Usage:
    python plot_confidence_conditions.py
"""

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "writeup_results/figures"
DATASETS = ["commonvoice", "edacc", "english_dialects"]

DATASET_DISPLAY = {
    "commonvoice":      "CommonVoice Scottish",
    "edacc":            "EdAcc",
    "english_dialects": "English Dialects",
}

FONT_TITLE = 28
FONT_SUPTITLE = 30
FONT_AXIS_LABEL = 24
FONT_TICK_LABEL = 22
FONT_LEGEND = 24
FONT_BAR_LABEL = 18


def condition_file_paths(dataset):
    """Returns {label: path}.

    Featured conditions, per the full p5/p10/p20 sweep comparison:
      List:   p5 was the best-performing threshold (avg severity 1.214,
              vs 1.220 at p10 and 1.237 at p20 - p20 was actually the
              WORST of the three for List format).
      Inline: p10 was the best-performing threshold (avg severity 1.212,
              vs 1.228 at p5 and 1.224 at p20).
    Rather than clutter the figure with all 3 thresholds per format
    (which are largely redundant once the sweep confirmed the winner),
    this shows only the best of each - the full sweep numbers remain
    available via the printed summary table for anyone who wants the
    detail."""
    paths = {
        "Baseline\n(no confidence)": f"writeup_results/grid_calib_fixed/unanchored_fusion_naive/unanchored_fusion_naive_{dataset}_gemma4_dev.json",
        "List\n(best: p5)": f"writeup_results/grid/unanchored_fusion_naive_confidence_list/unanchored_fusion_naive_confidence_list_{dataset}_gemma4_p5_thr-dev_dev.json",
        "Inline\n(best: p10)": f"writeup_results/grid/unanchored_fusion_naive_confidence_inline/unanchored_fusion_naive_confidence_inline_{dataset}_gemma4_p10_thr-dev_dev.json",
        "List\n(unfiltered,\nfrom p20)": f"writeup_results/grid/unanchored_fusion_naive_confidence_list_full_v2/unanchored_fusion_naive_confidence_list_full_v2_{dataset}_gemma4_dev.json",
        "Inline\n(unfiltered,\nfrom p20)": f"writeup_results/grid/unanchored_fusion_naive_confidence_inline_full_v2/unanchored_fusion_naive_confidence_inline_full_v2_{dataset}_gemma4_dev.json",
    }
    return paths


def load_severities(path):
    if not path or not os.path.exists(path):
        return []
    try:
        data = json.load(open(path))
    except Exception:
        return []
    samples = data.get("samples", [])
    return [s["severity"] for s in samples if s.get("severity") is not None]


def get_distribution(severities):
    if not severities:
        return None
    counts = Counter(severities)
    total = len(severities)
    return [counts.get(i, 0) / total * 100 for i in range(5)]


def get_wer_and_mean(path):
    if not path or not os.path.exists(path):
        return None, None
    try:
        data = json.load(open(path))
    except Exception:
        return None, None
    return data.get("corpus_wer"), data.get("mean_severity")


def collect_dataset_data(dataset):
    paths = condition_file_paths(dataset)
    results = {}
    for label, path in paths.items():
        severities = load_severities(path)
        dist = get_distribution(severities)
        wer, mean_sev = get_wer_and_mean(path)
        if dist is not None:
            results[label] = {"dist": dist, "wer": wer, "mean_sev": mean_sev, "n": len(severities)}
        else:
            print(f"  (skipping '{label.strip()}' for {dataset} - file not ready yet: {path})")
    return results


SEVERITY_STACK_COLOURS = ["#a8e6a3", "#4caf50", "#f4b942", "#e67e22", "#c0392b"]
SEVERITY_STACK_LABELS = ["0 - no change", "1 - trivial", "2 - ambiguous", "3 - factual", "4 - critical"]


def round_percentages_to_100(values):
    """Largest-remainder (Hare-Niemeyer) rounding: rounds a list of
    floats that sum to ~100 into integers that sum to EXACTLY 100 -
    fixes bars whose individually-rounded labels summed to 98-102%
    (confirmed real: English Dialects List(p20) summed to 101%,
    Inline(unfiltered) summed to 101%)."""
    floors = [int(v) for v in values]
    remainders = [v - f for v, f in zip(values, floors)]
    deficit = round(100 - sum(floors))
    order = sorted(range(len(values)), key=lambda i: remainders[i], reverse=True)
    result = floors[:]
    for i in order[:max(deficit, 0)]:
        result[i] += 1
    return result


def plot_severity_distribution():
    nrows = len(DATASETS)
    fig, axes = plt.subplots(nrows, 1, figsize=(16, 10 * nrows))
    if nrows == 1:
        axes = [axes]

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        data = collect_dataset_data(dataset)
        labels = list(data.keys())
        x = np.arange(len(labels))

        rounded_per_label = {l: round_percentages_to_100(data[l]["dist"]) for l in labels}

        bottoms = np.zeros(len(labels))
        for level in range(5):
            values = [data[l]["dist"][level] for l in labels]
            ax.bar(x, values, bottom=bottoms, color=SEVERITY_STACK_COLOURS[level],
                   edgecolor="black", linewidth=0.6,
                   label=SEVERITY_STACK_LABELS[level] if i == 0 else None)
            for xi, (v, b) in enumerate(zip(values, bottoms)):
                if v >= 4:
                    rounded_v = rounded_per_label[labels[xi]][level]
                    ax.text(xi, b + v / 2, f"{rounded_v}%", ha="center", va="center",
                            fontsize=FONT_BAR_LABEL, color="black" if level < 3 else "white",
                            fontweight="bold")
            bottoms += np.array(values)

        ax.set_xticks(x)
        # labels already multi-line wrapped (embedded \n) - horizontal,
        # center-aligned is more compact than rotating already-wrapped text
        ax.set_xticklabels(labels, fontsize=FONT_TICK_LABEL, rotation=0, ha="center")
        ax.tick_params(axis="y", labelsize=FONT_TICK_LABEL)
        ax.set_ylabel("% of samples", fontsize=FONT_AXIS_LABEL)
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=FONT_TITLE, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    legend = fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=FONT_LEGEND,
                        frameon=True, markerscale=3.0, handlelength=2.2, handleheight=2.2,
                        borderpad=1.0, labelspacing=1.4, title="Severity",
                        title_fontsize=FONT_LEGEND + 1)
    for handle in legend.legend_handles:
        handle.set_edgecolor("black")
        handle.set_linewidth(1.5)

    fig.suptitle("Unanchored Fusion: Severity Distribution\nby Confidence Condition\n"
                 "(lighter green = fully preserved, darker red = critical)",
                 fontsize=FONT_SUPTITLE, fontweight="bold", y=1.03)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "confidence_conditions_severity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_summary_table():
    print(f"\n{'='*110}")
    print(f"  SUMMARY TABLE: WER and mean severity by confidence condition, per dataset")
    print(f"{'='*110}")
    header = f"{'Condition':<22}" + "".join(f"{d:>24}" for d in ["CommonVoice", "EdAcc", "English Dialects"])
    print(header)
    print("-" * len(header))

    all_labels = []
    per_dataset = {}
    for dataset in DATASETS:
        per_dataset[dataset] = collect_dataset_data(dataset)
        for l in per_dataset[dataset]:
            if l not in all_labels:
                all_labels.append(l)

    for label in all_labels:
        row = f"{label.replace(chr(10), ' '):<22}"
        for dataset in DATASETS:
            d = per_dataset[dataset].get(label)
            if d is None:
                row += f"{'-':>24}"
            else:
                cell = f"WER={d['wer']*100:.1f}% sev={d['mean_sev']:.3f}"
                row += f"{cell:>24}"
        print(row)


def main():
    plot_severity_distribution()
    print_summary_table()


if __name__ == "__main__":
    main()