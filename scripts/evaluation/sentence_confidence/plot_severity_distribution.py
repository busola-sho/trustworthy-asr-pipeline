"""
scripts/evaluation/sentence_confidence/plot_severity_distribution.py

Plots the distribution of utterance-level severity scores per model
and ensemble, as overlapping KDE curves or grouped bar charts.

Shows whether the ensemble shifts the distribution leftward (fewer errors)
compared to individual models.

Usage:
    python scripts/evaluation/sentence_confidence/plot_severity_distribution.py
    python scripts/evaluation/sentence_confidence/plot_severity_distribution.py --dataset commonvoice
    python scripts/evaluation/sentence_confidence/plot_severity_distribution.py --style bar
"""

import json
import os
import argparse
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR  = "results/sentence_confidence"
FIGURES_DIR = "results/sentence_confidence/figures"

INDIVIDUAL_FILES = {
    ("whisperx", "commonvoice"):      "results/sentence_confidence/utterance_severity_whisperx_commonvoice.json",
    ("whisperx", "edacc"):            "results/sentence_confidence/utterance_severity_whisperx_edacc.json",
    ("whisperx", "english_dialects"): "results/sentence_confidence/utterance_severity_whisperx_english_dialects.json",
    ("whisperx", "shetland"):         "results/sentence_confidence/utterance_severity_whisperx_shetland.json",
    ("qwen",     "commonvoice"):      "results/sentence_confidence/utterance_severity_qwen_commonvoice.json",
    ("qwen",     "edacc"):            "results/sentence_confidence/utterance_severity_qwen_edacc.json",
    ("qwen",     "english_dialects"): "results/sentence_confidence/utterance_severity_qwen_english_dialects.json",
    ("qwen",     "shetland"):         "results/sentence_confidence/utterance_severity_qwen_shetland.json",
    ("parakeet", "commonvoice"):      "results/sentence_confidence/utterance_severity_parakeet_commonvoice.json",
    ("parakeet", "edacc"):            "results/sentence_confidence/utterance_severity_parakeet_edacc.json",
    ("parakeet", "english_dialects"): "results/sentence_confidence/utterance_severity_parakeet_english_dialects.json",
    ("parakeet", "shetland"):         "results/sentence_confidence/utterance_severity_parakeet_shetland.json",
}

UTTERANCE_ENSEMBLE_FILES = {
    "commonvoice":      "results/sentence_confidence/utterance_severity_ensemble_commonvoice.json",
    "edacc":            "results/sentence_confidence/utterance_severity_ensemble_edacc.json",
    "english_dialects": "results/sentence_confidence/utterance_severity_ensemble_english_dialects.json",
    "shetland":         "results/sentence_confidence/utterance_severity_ensemble_shetland.json",
}

ENSEMBLE_FILES = {
    "commonvoice":      "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_commonvoice_qwen_sub150.json",
    "edacc":            "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_edacc_qwen_sub150.json",
    "english_dialects": "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_english_dialects_qwen_sub150.json",
    "shetland":         "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_shetland_qwen_sub150.json",
}

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
MODELS   = ["whisperx", "qwen", "parakeet", "ensemble"]

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


def load_individual(model, dataset):
    path = INDIVIDUAL_FILES.get((model, dataset))
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        rows = json.load(f)
    return [r["severity"] for r in rows if r.get("severity") is not None]


def load_ensemble(dataset):
    """Load utterance-level ensemble severity scores."""
    path = UTTERANCE_ENSEMBLE_FILES.get(dataset)
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        rows = json.load(f)
    return [r["severity"] for r in rows if r.get("severity") is not None]


def get_distribution(severities):
    """Return normalised counts for severity 0-4."""
    if not severities:
        return [0] * 5
    counts = Counter(severities)
    total  = len(severities)
    return [counts.get(i, 0) / total * 100 for i in range(5)]


def plot_dataset(dataset, ax, style="bar"):
    data = {}
    for model in ["whisperx", "qwen", "parakeet"]:
        sevs = load_individual(model, dataset)
        if sevs:
            data[model] = get_distribution(sevs)

    ensemble_sevs = load_ensemble(dataset)
    if ensemble_sevs:
        data["ensemble"] = get_distribution(ensemble_sevs)

    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(5)
    n_models = len(data)
    width    = 0.8 / n_models

    for i, (model, dist) in enumerate(data.items()):
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(
            x + offset, dist, width=width * 0.9,
            color=COLOURS[model], alpha=0.8,
            label=LABELS[model],
            linewidth=0.5, edgecolor="white",
        )
        # bold ensemble bars
        if model == "ensemble":
            for bar in bars:
                bar.set_linewidth(1.5)
                bar.set_edgecolor("black")

    ax.set_xticks(x)
    ax.set_xticklabels(SEVERITY_LABELS, fontsize=9)
    ax.set_xlabel("Severity level", fontsize=10)
    ax.set_ylabel("% of utterances", fontsize=10)
    ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(max(d) for d in data.values()) * 1.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--style",   default="bar", choices=["bar"])
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes      = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        plot_dataset(dataset, axes[i])

    fig.suptitle(
        "Severity Score Distribution — Individual Models vs Context V2 Ensemble",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout(h_pad=3.5, w_pad=2.5)

    path = os.path.join(FIGURES_DIR, "utterance_severity_distribution_models_vs_ensemble.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()