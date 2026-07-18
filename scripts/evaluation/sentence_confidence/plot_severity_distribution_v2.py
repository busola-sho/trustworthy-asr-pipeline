"""
scripts/evaluation/sentence_confidence/plot_severity_distribution.py

Plots sentence-level severity distribution per model (WhisperX, Qwen, Parakeet)
and ensemble (Context V2), as grouped bar charts.

Shows whether the ensemble shifts distribution leftward (more sentences at
severity 0, fewer at severity 3-4) compared to individual models.

Usage:
    python scripts/evaluation/sentence_confidence/plot_severity_distribution.py
    python scripts/evaluation/sentence_confidence/plot_severity_distribution.py --dataset commonvoice
"""

import json
import os
import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "results/sentence_confidence/figures"

INDIVIDUAL_FILES = {
    ("whisperx", "commonvoice"):      "results/sentence_confidence/sentence_severity_whisperx_commonvoice.json",
    ("whisperx", "edacc"):            "results/sentence_confidence/sentence_severity_whisperx_edacc.json",
    ("whisperx", "english_dialects"): "results/sentence_confidence/sentence_severity_whisperx_english_dialects.json",
    ("whisperx", "shetland"):         "results/sentence_confidence/sentence_severity_whisperx_shetland.json",
    ("qwen",     "commonvoice"):      "results/sentence_confidence/sentence_severity_qwen_commonvoice.json",
    ("qwen",     "edacc"):            "results/sentence_confidence/sentence_severity_qwen_edacc.json",
    ("qwen",     "english_dialects"): "results/sentence_confidence/sentence_severity_qwen_english_dialects.json",
    ("qwen",     "shetland"):         "results/sentence_confidence/sentence_severity_qwen_shetland.json",
    ("parakeet", "commonvoice"):      "results/sentence_confidence/sentence_severity_parakeet_commonvoice.json",
    ("parakeet", "edacc"):            "results/sentence_confidence/sentence_severity_parakeet_edacc.json",
    ("parakeet", "english_dialects"): "results/sentence_confidence/sentence_severity_parakeet_english_dialects.json",
    ("parakeet", "shetland"):         "results/sentence_confidence/sentence_severity_parakeet_shetland.json",
}

ENSEMBLE_LABEL_FILES = {
    "commonvoice":      "results/sentence_confidence/sentence_labels_commonvoice.json",
    "edacc":            "results/sentence_confidence/sentence_labels_edacc.json",
    "english_dialects": "results/sentence_confidence/sentence_labels_english_dialects.json",
    "shetland":         "results/sentence_confidence/sentence_labels_shetland.json",
}

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
MODELS   = ["whisperx", "qwen", "parakeet"]

COLOURS = {
    "whisperx": "#3498db",
    "qwen":     "#e67e22",
    "parakeet": "#9b59b6",
    "ensemble": "#2ecc71",
}

LABELS = {
    "whisperx": "WhisperX",
    "qwen":     "Qwen3-ASR",
    "parakeet": "Parakeet",
    "ensemble": "Context V2 Ensemble",
}

DATASET_DISPLAY = {
    "commonvoice":      "CommonVoice Scottish",
    "edacc":            "EdAcc",
    "english_dialects": "English Dialects",
    "shetland":         "Shetland (held-out)",
}

SEVERITY_LABELS = ["0\nNo error", "1\nTrivial", "2\nAmbiguous", "3\nFactual", "4\nCritical"]


def load_individual_severities(model, dataset):
    path = INDIVIDUAL_FILES.get((model, dataset))
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        d = json.load(f)
    return [r["severity"] for r in d.get("rows", []) if r.get("severity") is not None]


def load_ensemble_severities(dataset):
    path = ENSEMBLE_LABEL_FILES.get(dataset)
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        d = json.load(f)
    return [r["severity"] for r in d.get("rows", []) if r.get("severity") is not None]


def get_distribution(severities):
    if not severities:
        return [0] * 5
    counts = Counter(severities)
    total  = len(severities)
    return [counts.get(i, 0) / total * 100 for i in range(5)]


def plot_dataset(dataset, ax):
    data = {}
    for model in MODELS:
        sevs = load_individual_severities(model, dataset)
        if sevs:
            data[model] = get_distribution(sevs)

    ensemble_sevs = load_ensemble_severities(dataset)
    if ensemble_sevs:
        data["ensemble"] = get_distribution(ensemble_sevs)

    if not data:
        ax.text(0.5, 0.5, "No data yet\n(run run_sentence_severity_per_model.py first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return

    x        = np.arange(5)
    n_models = len(data)
    width    = 0.75 / n_models

    for i, (model, dist) in enumerate(data.items()):
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(
            x + offset, dist, width=width * 0.92,
            color=COLOURS[model], alpha=0.82,
            label=LABELS[model],
            linewidth=0.8,
            edgecolor="white" if model != "ensemble" else "black",
        )
        if model == "ensemble":
            for bar in bars:
                bar.set_linewidth(1.8)
                bar.set_edgecolor("black")

    ax.set_xticks(x)
    ax.set_xticklabels(SEVERITY_LABELS, fontsize=9)
    ax.set_xlabel("Severity level", fontsize=10)
    ax.set_ylabel("% of sentences", fontsize=10)
    ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.legend(fontsize=8.5, loc="upper right")

    ymax = max(max(d) for d in data.values())
    ax.set_ylim(0, ymax * 1.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes      = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        plot_dataset(dataset, axes[i])

    fig.suptitle(
        "Sentence-Level Severity Distribution\nIndividual Models vs Context V2 Ensemble",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout(h_pad=3.5, w_pad=2.5)

    path = os.path.join(FIGURES_DIR, "severity_distribution_models_vs_ensemble.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()