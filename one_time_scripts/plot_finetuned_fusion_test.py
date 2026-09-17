"""
plot_finetuned_fusion_test.py

Same visual design as the other plotting scripts tonight, comparing
Unanchored Fusion's fine-tuned-model integration on the CLEAN,
uncontaminated TEST split (not dev - the fine-tuned model was trained
on data drawn from dev, so dev results were flagged as unreliable for
this specific comparison):

  Baseline Unanchored Fusion (4 original models)
  5-model (fine-tuned added as a 5th voice)
  Replace WhisperX (fine-tuned swaps in for WhisperX)
  Best Single Model (per-dataset best baseline, test-restricted)

ROVER/MBR deliberately excluded from this diagram - they belong to a
separate, already-established comparison (LLM strategies vs mechanical
baselines). This diagram makes one focused argument: does fine-tuning
integration improve Unanchored Fusion beyond its own baseline and the
best single model. The final results TABLE may still include ROVER/MBR
for completeness - just not this figure.

Per supervisor feedback (already applied to the strategy diagrams):
Shetland (out-of-domain) gets its OWN standalone figure, separate from
the 3 in-domain test datasets - never mixed into the same panel grid.

Figure 1: severity distribution, 3 in-domain test datasets (vertical).
Figure 2: severity distribution, Shetland standalone.
Figure 3: binary meaning-preservation, 3 in-domain test datasets.
Figure 4: binary meaning-preservation, Shetland standalone.

Usage:
    python plot_finetuned_fusion_test.py
"""

import json
import os
import glob
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.text_normalise import normalise
from jiwer import wer as compute_wer
from src.splits import get_indices_for_split

FIGURES_DIR = "writeup_results/figures"
IN_DOMAIN_DATASETS = ["commonvoice", "edacc", "english_dialects"]

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

LABEL_BASELINE = "Baseline\nUnanchored\nFusion"
LABEL_5MODEL = "5-model\n(+ ft)"
LABEL_REPLACE = "Replace\nWhisperX\n(ft)"


def condition_paths_in_domain(dataset):
    return {
        LABEL_5MODEL:   f"writeup_results/grid/unanchored_fusion_naive_5model/unanchored_fusion_naive_5model_{dataset}_gemma4_test.json",
        LABEL_REPLACE:  f"writeup_results/grid/unanchored_fusion_naive_replace_whisperx/unanchored_fusion_naive_replace_whisperx_{dataset}_gemma4_test.json",
        # TODO: swap grid_calib_fixed -> clean_grid once unanchored_fusion_naive job finishes
        LABEL_BASELINE: f"writeup_results/clean_grid/unanchored_fusion_naive/unanchored_fusion_naive_{dataset}_gemma4_test.json",
    }


def condition_paths_shetland():
    return {
        LABEL_5MODEL:   "writeup_results/grid/unanchored_fusion_naive_5model/unanchored_fusion_naive_5model_shetland_gemma4_full.json",
        LABEL_REPLACE:  "writeup_results/grid/unanchored_fusion_naive_replace_whisperx/unanchored_fusion_naive_replace_whisperx_shetland_gemma4_full.json",
        LABEL_BASELINE: "writeup_results/ensembles/naive/gemma4/naive_shetland_gemma4sel_full.json",
    }


def load_severities_from_file(path):
    if not path or not os.path.exists(path):
        return []
    try:
        data = json.load(open(path))
    except Exception:
        return []
    samples = data.get("samples", [])
    return [s["severity"] for s in samples if s.get("severity") is not None]


def find_baseline_model_path(model, dataset):
    if dataset == "shetland":
        shetland_files = {
            "qwen":     "results/benchmarks/shetland/shetland_qwen3asr_20260603_150124.json",
            "whisperx": "results/benchmarks/shetland/shetland_whisper_20260603_123115.json",
            "parakeet": "results/benchmarks/shetland/shetland_parakeet_20260606_134131.json",
            "wav2vec2": "results/benchmarks/shetland/shetland_wav2vec2_20260606_134507.json",
        }
        return shetland_files.get(model)
    matches = sorted(glob.glob(f"writeup_results/benchmarks/main/{model}_{dataset}_*.json"))
    matches = [m for m in matches if "sub150" not in m and "sub100" not in m
               and "whisper_ft_chunked" not in m]
    return matches[-1] if matches else None


def load_best_model_severities(dataset):
    """Best single model - restricted to TEST split (Shetland is
    already full, no restriction needed)."""
    model = BEST_MODEL_PER_DATASET[dataset]
    path = find_baseline_model_path(model, dataset)
    if not path:
        return [], model

    data = json.load(open(path))
    samples = data.get("samples", [])
    for i, s in enumerate(samples):
        if s.get("sample_index") is None:
            s["sample_index"] = i

    if dataset == "shetland":
        subset = samples
    else:
        test_indices = set(get_indices_for_split(dataset, "test"))
        subset = [s for s in samples if s.get("sample_index") in test_indices]

    severities = [s["severity"] for s in subset if s.get("severity") is not None]
    return severities, model


def get_distribution(severities):
    if not severities:
        return None
    counts = Counter(severities)
    total = len(severities)
    return [counts.get(i, 0) / total * 100 for i in range(5)]


def collect_dataset_data(dataset):
    is_shetland = dataset == "shetland"
    paths = condition_paths_shetland() if is_shetland else condition_paths_in_domain(dataset)

    results = {}
    for label, path in paths.items():
        severities = load_severities_from_file(path)
        dist = get_distribution(severities)
        if dist is not None:
            results[label] = {"dist": dist, "n": len(severities)}

    best_severities, best_model = load_best_model_severities(dataset)
    best_dist = get_distribution(best_severities)
    if best_dist is not None:
        results[f"Best Single\nModel ({best_model})"] = {"dist": best_dist, "n": len(best_severities)}

    return results


SEVERITY_STACK_COLOURS = ["#a8e6a3", "#4caf50", "#f4b942", "#e67e22", "#c0392b"]
SEVERITY_STACK_LABELS = ["0 - no change", "1 - trivial", "2 - ambiguous", "3 - factual", "4 - critical"]

FONT_TITLE = 26
FONT_SUPTITLE = 28
FONT_AXIS_LABEL = 22
FONT_TICK_LABEL = 20
FONT_LEGEND = 22
FONT_BAR_LABEL = 16


def round_percentages_to_100(values):
    floors = [int(v) for v in values]
    remainders = [v - f for v, f in zip(values, floors)]
    deficit = round(100 - sum(floors))
    order = sorted(range(len(values)), key=lambda i: remainders[i], reverse=True)
    result = floors[:]
    for i in order[:max(deficit, 0)]:
        result[i] += 1
    return result


def _plot_severity_grid(datasets, filename, suptitle, nrows):
    fig_width = 12 if nrows == 1 else 16
    fig, axes = plt.subplots(nrows, 1, figsize=(fig_width, 9 * nrows))
    if nrows == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
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
        ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=FONT_TICK_LABEL,
                           rotation=15, ha="right")
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

    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0, 0.85, 0.98])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _plot_binary_grid(datasets, filename, suptitle, nrows):
    fig_width = 10 if nrows == 1 else 13
    fig, axes = plt.subplots(nrows, 1, figsize=(fig_width, 9 * nrows))
    if nrows == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        data = collect_dataset_data(dataset)
        labels = list(data.keys())

        preserved = [sum(data[l]["dist"][0:2]) for l in labels]
        altered = [sum(data[l]["dist"][2:5]) for l in labels]

        x = np.arange(len(labels))
        ax.bar(x, preserved, color="#2ecc71", label="Meaning preserved (0-1)")
        ax.bar(x, altered, bottom=preserved, color="#c0392b", label="Meaning altered (2-4)")

        for xi, (p, a) in enumerate(zip(preserved, altered)):
            p_rounded, a_rounded = round_percentages_to_100([p, a])
            ax.text(xi, p / 2, f"{p_rounded}%", ha="center", va="center",
                    fontsize=FONT_BAR_LABEL + 3, color="white", fontweight="bold")
            ax.text(xi, p + a / 2, f"{a_rounded}%", ha="center", va="center",
                    fontsize=FONT_BAR_LABEL + 3, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=FONT_TICK_LABEL,
                           rotation=15, ha="right")
        ax.tick_params(axis="y", labelsize=FONT_TICK_LABEL)
        ax.set_ylabel("% of samples", fontsize=FONT_AXIS_LABEL)
        ax.set_title(DATASET_DISPLAY[dataset], fontsize=FONT_TITLE, fontweight="bold")
        ax.set_ylim(0, 105)
        if i == 0:
            ax.legend(fontsize=FONT_LEGEND, loc="upper right", markerscale=2.4)

    fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    _plot_severity_grid(
        IN_DOMAIN_DATASETS, "finetuned_fusion_test_severity.png",
        "Unanchored Fusion: Fine-Tuned Model Integration\nSeverity (Held-Out Test)",
        nrows=3,
    )
    _plot_severity_grid(
        ["shetland"], "finetuned_fusion_shetland_severity.png",
        "Unanchored Fusion: Fine-Tuned Model Integration\nSeverity (Shetland)",
        nrows=1,
    )
    _plot_binary_grid(
        IN_DOMAIN_DATASETS, "finetuned_fusion_test_binary.png",
        "Unanchored Fusion: Fine-Tuned Model Integration\nMeaning Alteration (Held-Out Test)",
        nrows=3,
    )
    _plot_binary_grid(
        ["shetland"], "finetuned_fusion_shetland_binary.png",
        "Unanchored Fusion: Fine-Tuned Model Integration\nMeaning Alteration (Shetland)",
        nrows=1,
    )


if __name__ == "__main__":
    main()