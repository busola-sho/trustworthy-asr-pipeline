"""
rerunning/sentence_confidence/plot_reliability_diagrams.py

Reliability diagrams: bins predicted confidence into buckets, plots the
ACTUAL observed fraction of severity=0 (no error) sentences in each bin
against the diagonal (perfect calibration). Unlike Spearman correlation
(which only cares about rank order), this shows whether the raw
confidence VALUE itself means what it claims to mean - useful for
deciding whether a signal is "real but needs calibrating" (points near
but not on the diagonal, still monotonic) versus "not much real signal"
(flat or non-monotonic, no calibration would fix this).

One figure per technique, all 4 datasets overlaid as separate lines.

Usage:
    python rerunning/sentence_confidence/plot_reliability_diagrams.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "writeup_results/sentence_confidence/figures"
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
N_BINS = 10

DATASET_COLOURS = {
    "commonvoice":      "#3498db",
    "edacc":            "#e67e22",
    "english_dialects": "#2ecc71",
    "shetland":         "#9b59b6",
}
DATASET_DISPLAY = {
    "commonvoice":      "CommonVoice Scottish",
    "edacc":            "EdAcc",
    "english_dialects": "English Dialects",
    "shetland":         "Shetland (held-out)",
}


def _combo_path(dataset, variant):
    split = "full" if dataset == "shetland" else "dev"
    return f"writeup_results/ensembles/naive_{variant}/naive_{variant}_{dataset}_gemma4sel_{split}.json"


def _label_path(dataset, variant):
    return f"results/sentence_confidence/sentence_labels_{dataset}_{variant}.json"


TECHNIQUES = {
    "confscore": {
        "title": "Verbalized Confidence (confscore)",
        "labels": lambda d: _label_path(d, "confscore"),
        "conf": lambda d: _combo_path(d, "confscore"),
        "field": "confidence",
    },
    "probscore": {
        "title": "Verbalized Confidence (probscore)",
        "labels": lambda d: _label_path(d, "probscore"),
        "conf": lambda d: _combo_path(d, "probscore"),
        "field": "confidence",
    },
    "crossmodel_mean": {
        "title": "Cross-model Agreement (mean, probscore-anchored)",
        "labels": lambda d: _label_path(d, "probscore"),
        "conf": lambda d: f"results/sentence_confidence/crossmodel_agreement_probscore_{d}.json",
        "field": "confidence",
    },
    "acoustic_mean": {
        "title": "Acoustic Confidence (mean, probscore-anchored)",
        "labels": lambda d: _label_path(d, "probscore"),
        "conf": lambda d: f"results/sentence_confidence/acoustic_confidence_probscore_{d}.json",
        "field": "confidence",
    },
}


def load_labels(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return {(r["dataset_index"], r["sent_pos"]): r for r in d.get("rows", []) if r.get("severity") is not None}


def load_confidences(path, field):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    conf_map = {}
    for samp in d.get("samples", []):
        if samp.get("skipped") or samp.get("error"):
            continue
        di = samp.get("dataset_index")
        sent_list = samp.get("sentence_confidences") or samp.get("sentences", [])
        for pos, sc in enumerate(sent_list):
            sent_pos = sc.get("sent_pos", pos)
            conf = sc.get(field)
            if conf is not None:
                conf_map[(di, sent_pos)] = conf
    return conf_map


def compute_reliability(technique_key, dataset):
    tech = TECHNIQUES[technique_key]
    labels = load_labels(tech["labels"](dataset))
    conf_map = load_confidences(tech["conf"](dataset), tech["field"])

    pairs = []
    for key, label in labels.items():
        conf = conf_map.get(key)
        if conf is not None:
            is_correct = 1.0 if label["severity"] == 0 else 0.0
            pairs.append((conf, is_correct))

    if not pairs:
        return None, None, None

    confs = np.array([p[0] for p in pairs])
    correct = np.array([p[1] for p in pairs])

    bin_edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_centers, observed, counts = [], [], []
    for i in range(N_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= lo) & (confs <= hi if i == N_BINS - 1 else confs < hi)
        n = mask.sum()
        if n > 0:
            bin_centers.append(confs[mask].mean())
            observed.append(correct[mask].mean())
            counts.append(int(n))

    return bin_centers, observed, counts


def plot_technique(technique_key, output_dir):
    tech = TECHNIQUES[technique_key]
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")

    for dataset in DATASETS:
        centers, observed, counts = compute_reliability(technique_key, dataset)
        if centers is None:
            continue
        ax.plot(centers, observed, marker="o", markersize=5, linewidth=1.8,
                color=DATASET_COLOURS[dataset], label=DATASET_DISPLAY[dataset])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence (binned)", fontsize=11)
    ax.set_ylabel("Observed fraction with severity = 0 (no error)", fontsize=11)
    ax.set_title(f"Reliability Diagram\n{tech['title']}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"reliability_{technique_key}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    for technique_key in TECHNIQUES:
        plot_technique(technique_key, OUTPUT_DIR)


if __name__ == "__main__":
    main()