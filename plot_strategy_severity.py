"""
plot_strategy_severity.py

Three figures, addressing supervisor feedback:
  1. Method (x-axis) labels enlarged for legibility.
  2. Severity colour-code legend enlarged.
  3. Shetland REMOVED from the dev-set figures (in-domain vs
     out-of-domain distinction matters - Shetland gets its own
     standalone figure instead, correctly framed as out-of-domain).
  4. All text sizes increased throughout.
  5. "MBR" relabeled "MBR-Style Consensus" (this is NOT literal
     Minimum Bayes Risk decoding - a WER-minimizing consensus
     procedure styled after MBR, not the real thing).
  6. "Best Baseline" relabeled "Best Single Model" - ROVER and
     MBR-Style Consensus are ALSO baselines (non-LLM baselines,
     specifically), not a separate category from the single-model
     baseline.

Figure 1 (dev, 3 datasets): severity distribution (%) stacked bars,
x-axis ordered Unanchored Fusion -> Anchored Correction -> Best Single
Model -> MBR-Style Consensus -> ROVER.

Figure 2 (dev, 3 datasets): simple binary meaning-preserved vs
meaning-altered, same ordering.

Figure 3 (Shetland only, standalone): same stacked-severity design as
Figure 1, clearly labeled as the out-of-domain held-out set.

Usage:
    python plot_strategy_severity.py
"""

import json
import os
import glob
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "writeup_results/figures"
DEV_DATASETS = ["commonvoice", "edacc", "english_dialects"]

DATASET_DISPLAY = {
    "commonvoice":      "CommonVoice Scottish",
    "edacc":            "EdAcc",
    "english_dialects": "English Dialects",
    "shetland":         "Shetland (out-of-domain, held-out)",
}

BEST_MODEL_PER_DATASET = {
    "commonvoice": "parakeet",
    "edacc": "qwen",
    "english_dialects": "whisperx",
    "shetland": "qwen",
}

# label text used consistently across all figures/tables
LABEL_UNANCHORED = "Unanchored\nFusion"
LABEL_ANCHORED = "Anchored\nCorrection"
LABEL_MBR = "MBR-Style\nConsensus\n(baseline)"
LABEL_ROVER = "ROVER\n(baseline)"


def _dev_or_full(dataset):
    return "full" if dataset == "shetland" else "dev"


def strategy_file_paths(dataset):
    split = _dev_or_full(dataset)
    best_model = BEST_MODEL_PER_DATASET[dataset]

    # Shetland is NEVER actually patched by patch_calibration_leakage.py
    # (only dev/test get contamination removed - Shetland's split="full"
    # always just gets copied through unchanged). So for Shetland, read
    # directly from the TRUE ORIGINAL locations, not the _calib_fixed
    # mirrors - mirrors are point-in-time snapshots and go stale.
    if dataset == "shetland":
        paths = {
            LABEL_UNANCHORED: "writeup_results/ensembles/naive/gemma4/naive_shetland_gemma4sel_full.json",
            LABEL_ANCHORED:   "writeup_results/ensembles/context_v1/gemma4/context_shetland_gemma4_full.json",
            LABEL_MBR:        "writeup_results/ensembles/mbr_consensus/mbr_shetland_full.json",
            LABEL_ROVER:      "writeup_results/voting/rover/rover_shetland_full.json",
        }
    else:
        paths = {
            LABEL_UNANCHORED: f"writeup_results/grid_calib_fixed/unanchored_fusion_naive/unanchored_fusion_naive_{dataset}_gemma4_{split}.json",
            LABEL_ANCHORED:   f"writeup_results/grid_calib_fixed/anchored_correction_v1/anchored_correction_v1_{dataset}_gemma4_{split}.json",
            LABEL_MBR:        f"writeup_results/ensembles_calib_fixed/mbr_consensus/mbr_{dataset}_{split}.json",
            LABEL_ROVER:      f"writeup_results/voting_calib_fixed/rover/rover_{dataset}_{split}.json",
        }
    # Selection excluded by request - confirmed the weakest of the
    # three strategies, left out to keep the chart focused.
    paths[f"Best Single\nModel ({best_model})"] = None  # resolved separately below

    return paths, best_model


def find_baseline_path(model, dataset):
    if dataset == "shetland":
        shetland_files = {
            "qwen":     "results/benchmarks/shetland/shetland_qwen3asr_20260603_150124.json",
            "whisperx": "results/benchmarks/shetland/shetland_whisper_20260603_123115.json",
            "parakeet": "results/benchmarks/shetland/shetland_parakeet_20260606_134131.json",
            "wav2vec2": "results/benchmarks/shetland/shetland_wav2vec2_20260606_134507.json",
        }
        return shetland_files.get(model)

    matches = sorted(glob.glob(f"writeup_results/benchmarks/main/{model}_{dataset}_*.json"))
    matches = [m for m in matches if "sub150" not in m and "sub100" not in m]
    return matches[-1] if matches else None


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
    paths, best_model = strategy_file_paths(dataset)
    baseline_path = find_baseline_path(best_model, dataset)
    paths[f"Best Single\nModel ({best_model})"] = baseline_path

    results = {}
    for label, path in paths.items():
        severities = load_severities(path)
        dist = get_distribution(severities)
        wer, mean_sev = get_wer_and_mean(path)
        if dist is not None:
            results[label] = {"dist": dist, "wer": wer, "mean_sev": mean_sev, "n": len(severities)}
    return results


SEVERITY_STACK_COLOURS = ["#a8e6a3", "#4caf50", "#f4b942", "#e67e22", "#c0392b"]
SEVERITY_STACK_LABELS = ["0 - no change", "1 - trivial", "2 - ambiguous", "3 - factual", "4 - critical"]

# ── enlarged text sizes throughout, per supervisor feedback ──
FONT_TITLE = 16
FONT_SUPTITLE = 17
FONT_AXIS_LABEL = 14
FONT_TICK_LABEL = 13
FONT_LEGEND = 14
FONT_BAR_LABEL = 10


def round_percentages_to_100(values):
    """Largest-remainder (Hare-Niemeyer) rounding: rounds a list of
    floats that sum to ~100 into integers that sum to EXACTLY 100 -
    fixes bars whose individually-rounded labels summed to 98-102%
    (same bug confirmed and fixed in plot_confidence_conditions.py)."""
    floors = [int(v) for v in values]
    remainders = [v - f for v, f in zip(values, floors)]
    deficit = round(100 - sum(floors))
    order = sorted(range(len(values)), key=lambda i: remainders[i], reverse=True)
    result = floors[:]
    for i in order[:max(deficit, 0)]:
        result[i] += 1
    return result


def _plot_severity_grid(datasets, filename, suptitle, nrows):
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 6.5 * nrows))
    if nrows == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        data = collect_dataset_data(dataset)
        labels = list(data.keys())
        x = np.arange(len(labels))

        # round each bar's 5 severity percentages together so the
        # printed labels always sum to exactly 100 (bar HEIGHTS still
        # use the raw, unrounded values - only the text labels change)
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
        ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=FONT_TICK_LABEL,
                           rotation=20, ha="right")
        ax.tick_params(axis="y", labelsize=FONT_TICK_LABEL)
        ax.set_ylabel("% of samples", fontsize=FONT_AXIS_LABEL)
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=FONT_TITLE, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    # bold, side-positioned severity legend - vertical stack, centered
    # on the whole figure's right edge, using bold-outlined swatches
    # for visibility per supervisor feedback
    legend = fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=FONT_LEGEND,
                        frameon=True, markerscale=2.2, handlelength=1.8, handleheight=1.8,
                        borderpad=1.0, labelspacing=1.4, title="Severity",
                        title_fontsize=FONT_LEGEND + 1)
    for handle in legend.legend_handles:
        handle.set_edgecolor("black")
        handle.set_linewidth(1.5)

    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0, 0.85, 0.98])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_severity_distribution_dev():
    _plot_severity_grid(
        DEV_DATASETS, "severity_distribution_by_strategy_dev.png",
        "Severity Distribution by Strategy, Development Datasets",
        nrows=3,
    )


def plot_severity_distribution_shetland():
    _plot_severity_grid(
        ["shetland"], "severity_distribution_shetland.png",
        "Severity Distribution by Strategy, Shetland (Out-of-Domain Held-Out Set)",
        nrows=1,
    )


def print_summary_table():
    print(f"\n{'='*110}")
    print(f"  SUMMARY TABLE: WER and mean severity by strategy, per dataset (calibration-corrected)")
    print(f"{'='*110}")
    all_datasets = DEV_DATASETS + ["shetland"]
    header = f"{'Strategy':<26}" + "".join(f"{d:>21}" for d in ["CommonVoice", "EdAcc", "English Dialects", "Shetland"])
    print(header)
    print("-" * len(header))

    all_labels = []
    per_dataset = {}
    for dataset in all_datasets:
        per_dataset[dataset] = collect_dataset_data(dataset)
        for l in per_dataset[dataset]:
            if l not in all_labels:
                all_labels.append(l)

    for label in all_labels:
        row = f"{label.replace(chr(10), ' '):<26}"
        for dataset in all_datasets:
            d = per_dataset[dataset].get(label)
            if d is None:
                row += f"{'-':>21}"
            else:
                cell = f"WER={d['wer']*100:.1f}% u={d['mean_sev']:.2f}"
                row += f"{cell:>21}"
        print(row)


def _plot_binary_grid(datasets, filename, suptitle, nrows):
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 6.5 * nrows))
    if nrows == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        data = collect_dataset_data(dataset)
        labels = list(data.keys())

        preserved = [sum(data[l]["dist"][0:2]) for l in labels]
        altered = [sum(data[l]["dist"][2:5]) for l in labels]

        x = np.arange(len(labels))
        ax.bar(x, preserved, color="#2ecc71", edgecolor="black", linewidth=0.6,
               label="Meaning preserved (0-1)" if i == 0 else None)
        ax.bar(x, altered, bottom=preserved, color="#c0392b", edgecolor="black", linewidth=0.6,
               label="Meaning altered (2-4)" if i == 0 else None)

        for xi, (p, a) in enumerate(zip(preserved, altered)):
            p_rounded, a_rounded = round_percentages_to_100([p, a])
            ax.text(xi, p / 2, f"{p_rounded}%", ha="center", va="center",
                    fontsize=FONT_BAR_LABEL + 2, color="white", fontweight="bold")
            ax.text(xi, p + a / 2, f"{a_rounded}%", ha="center", va="center",
                    fontsize=FONT_BAR_LABEL + 2, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=FONT_TICK_LABEL,
                           rotation=20, ha="right")
        ax.tick_params(axis="y", labelsize=FONT_TICK_LABEL)
        ax.set_ylabel("% of samples", fontsize=FONT_AXIS_LABEL)
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=FONT_TITLE, fontweight="bold")
        ax.set_ylim(0, 105)

    legend = fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=FONT_LEGEND,
                        frameon=True, markerscale=2.2, handlelength=1.8, handleheight=1.8,
                        borderpad=1.0, labelspacing=1.4)
    for handle in legend.legend_handles:
        handle.set_edgecolor("black")
        handle.set_linewidth(1.5)

    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0, 0.85, 0.98])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_binary_preserved_altered_dev():
    _plot_binary_grid(
        DEV_DATASETS, "meaning_preservation_binary_dev.png",
        "Meaning Preservation Rate by Strategy, Development Datasets",
        nrows=3,
    )


def plot_binary_preserved_altered_shetland():
    _plot_binary_grid(
        ["shetland"], "meaning_preservation_binary_shetland.png",
        "Meaning Preservation Rate by Strategy, Shetland (Out-of-Domain Held-Out Set)",
        nrows=1,
    )


def main():
    plot_severity_distribution_dev()
    plot_severity_distribution_shetland()
    plot_binary_preserved_altered_dev()
    plot_binary_preserved_altered_shetland()
    print_summary_table()


if __name__ == "__main__":
    main()