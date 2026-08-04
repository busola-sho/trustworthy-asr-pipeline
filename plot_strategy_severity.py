"""
plot_strategy_severity.py

Two figures, grouped by STRATEGY name (not raw technique labels like
"naive"/"context_v1"), using each strategy's confirmed best condition:
  Selection            -> naive   (selection_naive)
  Unanchored Fusion     -> naive   (naive.py)
  Anchored Correction  -> v1      (context_v1.py)
plus ROVER, MBR consensus, and that dataset's best baseline model.

Figure 1: severity distribution (%) per level 0-4, severity 0-1 visually
distinct from 2-4 (lighter vs darker colour family), WER/mean severity
moved out of x-axis labels into a companion caption line per bar.

Figure 2: simple binary meaning-preserved (severity 0-1) vs
meaning-altered (severity 2-4) - the headline-communicating figure.

NOTE: Selection has no Shetland run (selection_*.py scripts were only
ever run on the 3 dev datasets) - Shetland's panel omits it, not an
oversight in the chart itself.

Usage:
    python plot_strategy_severity.py
"""

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "writeup_results/figures"
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]

DATASET_DISPLAY = {
    "commonvoice":      "CommonVoice Scottish",
    "edacc":            "EdAcc",
    "english_dialects": "English Dialects",
    "shetland":         "Shetland (held-out)",
}

BEST_BASELINE_PER_DATASET = {
    "commonvoice": "parakeet",
    "edacc": "qwen",
    "english_dialects": "whisperx",
    "shetland": "qwen",
}

# severity 0-1 = lighter/greener, 2-4 = darker/redder - visually distinct
SEVERITY_COLOURS = ["#2ecc71", "#95d44e", "#f4b942", "#e67e22", "#c0392b"]
SEVERITY_LABELS = ["0", "1", "2", "3", "4"]


def _dev_or_full(dataset):
    return "full" if dataset == "shetland" else "dev"


def strategy_file_paths(dataset):
    split = _dev_or_full(dataset)
    baseline = BEST_BASELINE_PER_DATASET[dataset]

    # order chosen deliberately: strongest to weakest performer, left to
    # right, so the chart reads as a visual slope (shrinking green,
    # growing red) rather than an arbitrary arrangement
    paths = {
        "Unanchored\nFusion":   f"writeup_results/ensembles/naive/gemma4/naive_{dataset}_gemma4sel_{split}.json",
        "Anchored\nCorrection": f"writeup_results/ensembles/context_v1/gemma4/context_{dataset}_gemma4_{split}.json",
    }
    if dataset != "shetland":
        paths["Selection"] = f"writeup_results/grid/selection_naive/selection_naive_{dataset}_gemma4_{split}.json"
    paths[f"Best Baseline\n({baseline})"] = None  # resolved separately below
    paths["MBR"] = f"writeup_results/ensembles/mbr_consensus/mbr_{dataset}_{split}.json"
    paths["ROVER"] = f"writeup_results/voting/rover/rover_{dataset}_{split}.json"

    return paths, baseline


def find_baseline_path(model, dataset):
    """Baseline files live under writeup_results/benchmarks/main/ or
    have a dedicated Shetland path - matches find_canonical_file()'s
    own lookup convention."""
    if dataset == "shetland":
        shetland_files = {
            "qwen":     "results/benchmarks/shetland/shetland_qwen3asr_20260603_150124.json",
            "whisperx": "results/benchmarks/shetland/shetland_whisper_20260603_123115.json",
            "parakeet": "results/benchmarks/shetland/shetland_parakeet_20260606_134131.json",
            "wav2vec2": "results/benchmarks/shetland/shetland_wav2vec2_20260606_134507.json",
        }
        return shetland_files.get(model)

    import glob
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
    paths, baseline = strategy_file_paths(dataset)
    baseline_path = find_baseline_path(baseline, dataset)
    paths[f"Best Baseline\n({baseline})"] = baseline_path

    results = {}
    for label, path in paths.items():
        severities = load_severities(path)
        dist = get_distribution(severities)
        wer, mean_sev = get_wer_and_mean(path)
        if dist is not None:
            results[label] = {"dist": dist, "wer": wer, "mean_sev": mean_sev, "n": len(severities)}
    return results


# two green shades for meaning-preserving (0,1), three warm/red shades
# for meaning-altered (2,3,4) - keeps the semantic grouping visible
# within each stacked bar
SEVERITY_STACK_COLOURS = ["#a8e6a3", "#4caf50", "#f4b942", "#e67e22", "#c0392b"]
SEVERITY_STACK_LABELS = ["0 - no change", "1 - trivial", "2 - ambiguous", "3 - factual", "4 - critical"]


def plot_severity_distribution():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        data = collect_dataset_data(dataset)
        labels = list(data.keys())
        x = np.arange(len(labels))

        bottoms = np.zeros(len(labels))
        for level in range(5):
            values = [data[l]["dist"][level] for l in labels]
            ax.bar(x, values, bottom=bottoms, color=SEVERITY_STACK_COLOURS[level],
                   edgecolor="black", linewidth=0.4,
                   label=SEVERITY_STACK_LABELS[level] if i == 0 else None)
            for xi, (v, b) in enumerate(zip(values, bottoms)):
                if v >= 4:  # only label segments big enough to read
                    ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=7, color="black" if level < 3 else "white")
            bottoms += np.array(values)

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("% of samples", fontsize=10)
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    fig.legend(loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.0), frameon=True)
    fig.suptitle("Severity Distribution by Strategy, per Dataset\n"
                 "(each bar stacked by severity level - lighter green = fully preserved, darker red = critical)",
                 fontsize=12, fontweight="bold", y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "severity_distribution_by_strategy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_summary_table():
    """WER and mean severity per strategy per dataset - the companion
    table for the figures above, kept separate rather than crammed
    into the plot itself."""
    print(f"\n{'='*100}")
    print(f"  SUMMARY TABLE: WER and mean severity by strategy, per dataset")
    print(f"{'='*100}")
    header = f"{'Strategy':<22}" + "".join(f"{d:>20}" for d in ["CommonVoice", "EdAcc", "English Dialects", "Shetland"])
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
                row += f"{'-':>20}"
            else:
                cell = f"WER={d['wer']*100:.1f}% u={d['mean_sev']:.2f}"
                row += f"{cell:>20}"
        print(row)


def plot_binary_preserved_altered():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        data = collect_dataset_data(dataset)
        labels = list(data.keys())

        preserved = [sum(data[l]["dist"][0:2]) for l in labels]
        altered = [sum(data[l]["dist"][2:5]) for l in labels]

        x = np.arange(len(labels))
        ax.bar(x, preserved, color="#2ecc71", label="Meaning preserved (0-1)")
        ax.bar(x, altered, bottom=preserved, color="#c0392b", label="Meaning altered (2-4)")

        for xi, (p, a) in enumerate(zip(preserved, altered)):
            ax.text(xi, p / 2, f"{p:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            ax.text(xi, p + a / 2, f"{a:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("% of samples", fontsize=10)
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 105)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Meaning Preservation Rate by Strategy, per Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "meaning_preservation_binary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    plot_severity_distribution()
    plot_binary_preserved_altered()
    print_summary_table()


if __name__ == "__main__":
    main()