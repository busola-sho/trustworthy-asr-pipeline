"""
scripts/evaluation/sentence_confidence/plot_confidence_by_severity.py

For each method, plots the distribution of confidence scores grouped by
severity level (0-4) as box plots across all datasets.

Usage:
    python scripts/evaluation/sentence_confidence/plot_confidence_by_severity.py
    python scripts/evaluation/sentence_confidence/plot_confidence_by_severity.py --method crossmodel_mean
"""

import json
import os
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIGURES_DIR = "results/sentence_confidence/figures"

LABEL_FILES = {
    "commonvoice":      "results/sentence_confidence/sentence_labels_commonvoice.json",
    "edacc":            "results/sentence_confidence/sentence_labels_edacc.json",
    "english_dialects": "results/sentence_confidence/sentence_labels_english_dialects.json",
    "shetland":         "results/sentence_confidence/sentence_labels_shetland.json",
}

CONF_FILES = {
    "confscore": {
        "commonvoice":      "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_commonvoice_qwen_sub150.json",
        "edacc":            "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_edacc_qwen_sub150.json",
        "english_dialects": "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_english_dialects_qwen_sub150.json",
        "shetland":         "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_shetland_qwen_sub150.json",
    },
    "probscore": {
        "commonvoice":      "results/combinations_v2judge/context_v2_whisperx_probscore_sentences/context_v2_probscore_commonvoice_qwen_sub150.json",
        "edacc":            "results/combinations_v2judge/context_v2_whisperx_probscore_sentences/context_v2_probscore_edacc_qwen_sub150.json",
        "english_dialects": "results/combinations_v2judge/context_v2_whisperx_probscore_sentences/context_v2_probscore_english_dialects_qwen_sub150.json",
        "shetland":         "results/combinations_v2judge/context_v2_whisperx_probscore_sentences/context_v2_probscore_shetland_qwen_sub150.json",
    },
    "confscore_meaning": {
        "commonvoice":      "results/combinations_v2judge/context_v2_whisperx_confscore_meaning/context_v2_commonvoice_qwen_sub150.json",
        "edacc":            "results/combinations_v2judge/context_v2_whisperx_confscore_meaning/context_v2_edacc_qwen_sub150.json",
        "english_dialects": "results/combinations_v2judge/context_v2_whisperx_confscore_meaning/context_v2_english_dialects_qwen_sub150.json",
        "shetland":         "results/combinations_v2judge/context_v2_whisperx_confscore_meaning/context_v2_shetland_qwen_sub150.json",
    },
    "probscore_meaning": {
        "commonvoice":      "results/combinations_v2judge/context_v2_whisperx_probscore_meaning/context_v2_probscore_commonvoice_qwen_sub150.json",
        "edacc":            "results/combinations_v2judge/context_v2_whisperx_probscore_meaning/context_v2_probscore_edacc_qwen_sub150.json",
        "english_dialects": "results/combinations_v2judge/context_v2_whisperx_probscore_meaning/context_v2_probscore_english_dialects_qwen_sub150.json",
        "shetland":         "results/combinations_v2judge/context_v2_whisperx_probscore_meaning/context_v2_probscore_shetland_qwen_sub150.json",
    },
    "crossmodel_mean": {
        "commonvoice":      "results/sentence_confidence/crossmodel_agreement_commonvoice.json",
        "edacc":            "results/sentence_confidence/crossmodel_agreement_edacc.json",
        "english_dialects": "results/sentence_confidence/crossmodel_agreement_english_dialects.json",
        "shetland":         "results/sentence_confidence/crossmodel_agreement_shetland.json",
    },
    "acoustic_mean": {
        "commonvoice":      "results/sentence_confidence/acoustic_confidence_commonvoice.json",
        "edacc":            "results/sentence_confidence/acoustic_confidence_edacc.json",
        "english_dialects": "results/sentence_confidence/acoustic_confidence_english_dialects.json",
        "shetland":         "results/sentence_confidence/acoustic_confidence_shetland.json",
    },
    "proxy_model": {
        "commonvoice":      "results/sentence_confidence/proxy_model_commonvoice.json",
        "edacc":            "results/sentence_confidence/proxy_model_edacc.json",
        "english_dialects": "results/sentence_confidence/proxy_model_english_dialects.json",
        "shetland":         "results/sentence_confidence/proxy_model_shetland.json",
    },
}

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
METHODS  = list(CONF_FILES.keys())

DATASET_DISPLAY = {
    "commonvoice":      "CommonVoice Scottish",
    "edacc":            "EdAcc",
    "english_dialects": "English Dialects",
    "shetland":         "Shetland (held-out)",
}

METHOD_DISPLAY = {
    "confscore":          "Verbalized Confidence (confscore)",
    "probscore":          "Verbalized Confidence (probscore)",
    "confscore_meaning":  "Verbalized Confidence — Meaning framing (confscore)",
    "probscore_meaning":  "Verbalized Confidence — Meaning framing (probscore)",
    "crossmodel_mean":    "Cross-model Pairwise Agreement",
    "acoustic_mean":      "Acoustic Confidence (mean)",
    "proxy_model":        "Proxy Model (Ridge Regression)",
}

SEVERITY_LABELS = ["0\nNo error", "1\nTrivial", "2\nAmbiguous", "3\nFactual", "4\nCritical"]
COLOURS         = ["#2ecc71", "#95d44e", "#f39c12", "#e67e22", "#e74c3c"]


def load_labels(dataset):
    with open(LABEL_FILES[dataset]) as f:
        d = json.load(f)
    return {
        (r["dataset_index"], r["sent_pos"]): r
        for r in d.get("rows", [])
        if r.get("severity") is not None
    }


def load_confidences(method, dataset):
    path = CONF_FILES[method][dataset]
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    conf_map = {}
    for samp in d.get("samples", []):
        if samp.get("skipped") or samp.get("error"):
            continue
        di        = samp.get("dataset_index")
        sent_list = samp.get("sentence_confidences") or samp.get("sentences", [])
        for pos, sc in enumerate(sent_list):
            sent_pos = sc.get("sent_pos", pos)
            conf     = sc.get("confidence")
            if conf is not None:
                conf_map[(di, sent_pos)] = conf
    return conf_map


def get_data_by_severity(method, dataset):
    labels   = load_labels(dataset)
    conf_map = load_confidences(method, dataset)
    by_sev   = defaultdict(list)
    for key, label in labels.items():
        conf = conf_map.get(key)
        if conf is not None:
            by_sev[label["severity"]].append(conf)
    return by_sev


def plot_one_method_all_datasets(method, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes      = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        ax         = axes[i]
        by_sev     = get_data_by_severity(method, dataset)
        severities = list(range(5))
        data       = [by_sev.get(s, []) for s in severities]

        # box plot
        bp = ax.boxplot(
            [d if d else [np.nan] for d in data],
            positions=severities,
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2.5),
            whiskerprops=dict(linewidth=1.3, color="#444"),
            capprops=dict(linewidth=1.3, color="#444"),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.35, color="#888"),
            boxprops=dict(linewidth=1.2),
        )

        for patch, colour in zip(bp["boxes"], COLOURS):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)

        # mean trend line
        means = [np.mean(d) if d else np.nan for d in data]
        valid_x = [s for s, m in zip(severities, means) if not np.isnan(m)]
        valid_m = [m for m in means if not np.isnan(m)]
        if len(valid_x) > 1:
            ax.plot(valid_x, valid_m, "k--", linewidth=1.5, alpha=0.6,
                    marker="D", markersize=4, label="Mean")

        # n labels inside plot at fixed y=0.06 (always visible, never overlaps)
        for s, d_list in zip(severities, data):
            n = len(d_list)
            ax.text(s, 0.06, f"n={n}", ha="center", va="bottom",
                    fontsize=8, color="#444",
                    transform=ax.get_xaxis_transform(),
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.7, pad=1))

        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(severities)
        ax.set_xticklabels(SEVERITY_LABELS, fontsize=9)
        ax.set_ylabel("Confidence score", fontsize=10)
        ax.set_xlabel("Severity level", fontsize=10)
        ax.axhline(0.5, color="#aaa", linestyle=":", linewidth=1.0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset),
                     fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(axis="x", pad=14)

        if valid_x:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"{METHOD_DISPLAY.get(method, method)}\nConfidence Score Distribution by Severity",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout(h_pad=3.5, w_pad=2.5)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"boxplot_{method}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="all", choices=METHODS + ["all"])
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    methods = METHODS if args.method == "all" else [args.method]

    print(f"Generating box plots (one per method, all datasets)...")
    for method in methods:
        print(f"  {method}...")
        plot_one_method_all_datasets(method, FIGURES_DIR)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()