"""
plot_strategy_severity.py

Six figures total (dev + test, mirrored), addressing supervisor feedback:
  1. Method (x-axis) labels enlarged for legibility.
  2. Severity colour-code legend enlarged.
  3. Shetland REMOVED from the dev/test-set figures (in-domain vs
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

TEST SPLIT SUPPORT (added): strategy_file_paths()/collect_dataset_data()
now take an explicit split argument ("dev" or "test") instead of only
ever reading dev. Also fixed a real gap this surfaced: baseline model
severities were being loaded WITHOUT split-restriction at all (just
whatever the file happened to contain) - unlike full_results_report.py,
which correctly calls get_indices_for_split() to restrict a baseline's
samples to the requested split. Fixed here to match that same, more
rigorous convention - otherwise a "test" figure could silently include
dev samples in its "Best Single Model" bar. Since most grid cells only
have dev results so far (per full_results_report.py's own note),
missing test data for a strategy is handled gracefully - shown as a
dash, not a crash.

Figures 1-2 (dev, 3 datasets): severity distribution + binary,
x-axis ordered Unanchored Fusion -> Anchored Fusion -> Best Single
Model -> MBR-Style Consensus -> ROVER.

Figures 3-4 (test, 3 datasets): same, test split.

Figures 5-6 (Shetland only, standalone): same stacked-severity design,
clearly labeled as the out-of-domain held-out set - unaffected by the
dev/test split distinction, since Shetland only has "full".

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

from src.splits import get_indices_for_split

FIGURES_DIR = "writeup_results/figures"
DEV_DATASETS = ["commonvoice", "edacc", "english_dialects"]

DATASET_DISPLAY = {
    "commonvoice":      "CommonVoice Scottish",
    "edacc":            "EdAcc",
    "english_dialects": "English Dialects",
    "shetland":         "Shetland (Held-out transfer set)",
}

BEST_MODEL_PER_DATASET = {
    "commonvoice": "parakeet",
    "edacc": "qwen",
    "english_dialects": "whisperx",
    "shetland": "qwen",
}

# label text used consistently across all figures/tables
LABEL_UNANCHORED = "Unanchored\nFusion"
LABEL_ANCHORED = "Anchored\nFusion"
LABEL_SELECTION = "Selection"
LABEL_MBR = "MBR-Style\nConsensus\n(baseline)"
LABEL_ROVER = "ROVER\n(baseline)"


def strategy_file_paths(dataset, split):
    """split: "dev" or "test" for the 3 main datasets; Shetland only
    has "full" regardless of what's passed, since it has no dev/test
    distinction at all."""
    file_split = "full" if dataset == "shetland" else split
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
            LABEL_UNANCHORED: f"writeup_results/clean_grid/unanchored_fusion_naive/unanchored_fusion_naive_{dataset}_gemma4_{file_split}.json",
            LABEL_ANCHORED:   f"writeup_results/clean_grid/anchored_correction_v1/anchored_correction_v1_{dataset}_gemma4_{file_split}.json",
            LABEL_SELECTION:  f"writeup_results/clean_grid/selection_naive/selection_naive_{dataset}_gemma4_{file_split}.json",
            LABEL_MBR:        f"writeup_results/ensembles/mbr_consensus/mbr_{dataset}_{file_split}.json",
            LABEL_ROVER:      f"writeup_results/voting/rover/rover_{dataset}_{file_split}.json",
        }
    paths[f"Best Single\nModel ({best_model})"] = None  # resolved separately below

    return paths, best_model


def find_baseline_path(model, dataset):
    """Returns the baseline's FULL file (contains all samples, both
    dev and test together) - split-restriction happens separately in
    load_baseline_severities(), not by picking a different file."""
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
    """For strategy/mechanical files (unanchored fusion, anchored
    fusion, MBR, ROVER) - these files are already split-specific by
    filename, so no additional filtering needed here."""
    if not path or not os.path.exists(path):
        return []
    try:
        data = json.load(open(path))
    except Exception:
        return []
    samples = data.get("samples", [])
    return [s["severity"] for s in samples if s.get("severity") is not None]


def load_baseline_severities(path, dataset, split):
    """For baseline model files specifically - these contain ALL
    samples (dev+test combined) in one file, so the split must be
    applied explicitly via get_indices_for_split(), same convention
    full_results_report.py already uses. Without this, "test" figures
    would silently include dev samples in the baseline bar. Shetland
    has no dev/test distinction, so it's returned unrestricted."""
    if not path or not os.path.exists(path):
        return []
    try:
        data = json.load(open(path))
    except Exception:
        return []
    samples = data.get("samples", [])

    if dataset == "shetland":
        return [s["severity"] for s in samples if s.get("severity") is not None]

    for i, s in enumerate(samples):
        if s.get("sample_index") is None:
            s["sample_index"] = i
    target_indices = set(get_indices_for_split(dataset, split))
    return [s["severity"] for s in samples
           if s.get("sample_index") in target_indices and s.get("severity") is not None]


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


def collect_dataset_data(dataset, split):
    paths, best_model = strategy_file_paths(dataset, split)
    baseline_path = find_baseline_path(best_model, dataset)

    results = {}
    for label, path in paths.items():
        if label.startswith("Best Single"):
            severities = load_baseline_severities(baseline_path, dataset, split)
            wer, mean_sev = None, None
            if severities:
                mean_sev = sum(severities) / len(severities)
        else:
            severities = load_severities(path)
            wer, mean_sev = get_wer_and_mean(path)
        dist = get_distribution(severities)
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


def _plot_severity_grid(datasets, split, filename, suptitle, nrows):
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 6.5 * nrows))
    if nrows == 1:
        axes = [axes]

    any_data = False
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        data = collect_dataset_data(dataset, split)
        labels = list(data.keys())

        if not labels:
            ax.text(0.5, 0.5, f"No {split} data available yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(DATASET_DISPLAY[dataset], fontsize=FONT_TITLE, fontweight="bold")
            continue
        any_data = True

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

    if not any_data:
        print(f"  Skipped: {filename} (no {split} data available for any dataset)")
        plt.close()
        return

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
        DEV_DATASETS, "dev", "severity_distribution_by_strategy_dev.png",
        "Severity Distribution by Strategy, Development Datasets",
        nrows=3,
    )


def plot_severity_distribution_test():
    _plot_severity_grid(
        DEV_DATASETS, "test", "severity_distribution_by_strategy_test.png",
        "Severity Distribution by Strategy, Test Datasets",
        nrows=3,
    )


def plot_severity_distribution_shetland():
    _plot_severity_grid(
        ["shetland"], "full", "severity_distribution_shetland.png",
        "Severity Distribution by Strategy, Shetland ",
        nrows=1,
    )


def print_summary_table():
    print(f"\n{'='*110}")
    print(f"  SUMMARY TABLE: WER and mean severity by strategy, per dataset+split (calibration-corrected)")
    print(f"{'='*110}")
    columns = [("commonvoice", "dev"), ("commonvoice", "test"),
              ("edacc", "dev"), ("edacc", "test"),
              ("english_dialects", "dev"), ("english_dialects", "test"),
              ("shetland", "full")]
    col_labels = ["CV-dev", "CV-test", "EdAcc-dev", "EdAcc-test", "EngDial-dev", "EngDial-test", "Shetland"]
    header = f"{'Strategy':<26}" + "".join(f"{c:>16}" for c in col_labels)
    print(header)
    print("-" * len(header))

    all_labels = []
    per_col = {}
    for dataset, split in columns:
        per_col[(dataset, split)] = collect_dataset_data(dataset, split)
        for l in per_col[(dataset, split)]:
            if l not in all_labels:
                all_labels.append(l)

    for label in all_labels:
        row = f"{label.replace(chr(10), ' '):<26}"
        for dataset, split in columns:
            d = per_col[(dataset, split)].get(label)
            if d is None or d.get("mean_sev") is None:
                row += f"{'-':>16}"
            else:
                wer_str = f"{d['wer']*100:.1f}%" if d.get("wer") is not None else "-"
                cell = f"W={wer_str} u={d['mean_sev']:.2f}"
                row += f"{cell:>16}"
        print(row)


def _plot_binary_grid(datasets, split, filename, suptitle, nrows):
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 6.5 * nrows))
    if nrows == 1:
        axes = [axes]

    any_data = False
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        data = collect_dataset_data(dataset, split)
        labels = list(data.keys())

        if not labels:
            ax.text(0.5, 0.5, f"No {split} data available yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(DATASET_DISPLAY[dataset], fontsize=FONT_TITLE, fontweight="bold")
            continue
        any_data = True

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

    if not any_data:
        print(f"  Skipped: {filename} (no {split} data available for any dataset)")
        plt.close()
        return

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
        DEV_DATASETS, "dev", "meaning_preservation_binary_dev.png",
        "Meaning Alteration Rate by Strategy, Development Datasets",
        nrows=3,
    )


def plot_binary_preserved_altered_test():
    _plot_binary_grid(
        DEV_DATASETS, "test", "meaning_preservation_binary_test.png",
        "Meaning Alteration Rate by Strategy, Test Datasets",
        nrows=3,
    )


def plot_binary_preserved_altered_shetland():
    _plot_binary_grid(
        ["shetland"], "full", "meaning_preservation_binary_shetland.png",
        "Meaning Alteration Rate by Strategy, Shetland",
        nrows=1,
    )


def main():
    plot_severity_distribution_dev()
    plot_severity_distribution_test()
    plot_severity_distribution_shetland()
    plot_binary_preserved_altered_dev()
    plot_binary_preserved_altered_test()
    plot_binary_preserved_altered_shetland()
    print_summary_table()


if __name__ == "__main__":
    main()