"""
rerunning/sentence_confidence/plot_confidence_by_severity.py

Box plots of confidence score distribution grouped by severity level
(0-4), one figure per technique, each figure showing all 4 datasets as
a 2x2 panel. Covers all 6 relevant sentence-confidence techniques on
naive: confscore, probscore, crossmodel_mean, crossmodel_min,
acoustic_mean, proxy_model.

crossmodel_mean/crossmodel_min/proxy_model default to their
probscore-anchored versions (matching the primary comparison table) -
confscore-anchored equivalents also exist on disk if needed instead.

Usage:
    python rerunning/sentence_confidence/plot_confidence_by_severity.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]

DATASET_DISPLAY = {
    "commonvoice":      "CommonVoice Scottish",
    "edacc":            "EdAcc",
    "english_dialects": "English Dialects",
    "shetland":         "Shetland (held-out)",
}

SEVERITY_LABELS = ["0\nNo error", "1\nTrivial", "2\nAmbiguous", "3\nFactual", "4\nCritical"]
COLOURS         = ["#2ecc71", "#95d44e", "#f39c12", "#e67e22", "#e74c3c"]


def _combo_path(dataset, variant):
    split = "full" if dataset == "shetland" else "dev"
    return f"writeup_results/ensembles/naive_{variant}/naive_{variant}_{dataset}_gemma4sel_{split}.json"


def _label_path(dataset, variant):
    return f"results/sentence_confidence/sentence_labels_{dataset}_{variant}.json"


def build_techniques(variant):
    """
    Builds the full TECHNIQUES dict for ONE variant, fully self-consistent -
    crossmodel_mean/min, acoustic_mean, and proxy_model are all anchored
    to the SAME variant as the verbalized score itself (not mixed, per
    the confound fix - see build_leaderboard.py / summarise_final.py).
    "confscore"/"probscore" always shows both raw scores for reference,
    but the other's own OWN row uses whichever variant this function
    was called for.
    """
    return {
        "confscore": {
            "title": "Naive - Verbalized Confidence (confscore)",
            "labels": lambda d: _label_path(d, "confscore"),
            "conf": lambda d: _combo_path(d, "confscore"),
            "field": "confidence",
        },
        "probscore": {
            "title": "Naive - Verbalized Confidence (probscore)",
            "labels": lambda d: _label_path(d, "probscore"),
            "conf": lambda d: _combo_path(d, "probscore"),
            "field": "confidence",
        },
        "crossmodel_mean": {
            "title": f"Naive - Cross-model Agreement (mean, {variant}-anchored)",
            "labels": lambda d: _label_path(d, variant),
            "conf": lambda d: f"results/sentence_confidence/crossmodel_agreement_{variant}_{d}.json",
            "field": "confidence",
        },
        "crossmodel_min": {
            "title": f"Naive - Cross-model Agreement (min, {variant}-anchored)",
            "labels": lambda d: _label_path(d, variant),
            "conf": lambda d: f"results/sentence_confidence/crossmodel_agreement_{variant}_{d}.json",
            "field": "min_agreement",
        },
        "acoustic_mean": {
            "title": f"Naive - Acoustic Confidence (mean, {variant}-anchored)",
            "labels": lambda d: _label_path(d, variant),
            "conf": lambda d: f"results/sentence_confidence/acoustic_confidence_{variant}_{d}.json",
            "field": "confidence",
        },
        "proxy_model": {
            "title": f"Naive - Proxy Model (Ridge Regression, {variant}-trained)",
            "labels": lambda d: _label_path(d, variant),
            "conf": lambda d: f"results/sentence_confidence/proxy_model_{d}_{variant}.json",
            "field": "confidence",
        },
    }


def load_labels(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return {
        (r["dataset_index"], r["sent_pos"]): r
        for r in d.get("rows", [])
        if r.get("severity") is not None
    }


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


def get_data_by_severity(technique_key, dataset, techniques):
    tech = techniques[technique_key]
    labels = load_labels(tech["labels"](dataset))
    conf_map = load_confidences(tech["conf"](dataset), tech["field"])
    by_sev = defaultdict(list)
    for key, label in labels.items():
        conf = conf_map.get(key)
        if conf is not None:
            by_sev[label["severity"]].append(conf)
    return by_sev


def plot_technique(technique_key, output_dir, techniques):
    tech = techniques[technique_key]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        by_sev = get_data_by_severity(technique_key, dataset, techniques)
        severities = list(range(5))
        data = [by_sev.get(s, []) for s in severities]

        if not any(data):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=11, fontweight="bold")
            continue

        bp = ax.boxplot(
            [d if d else [np.nan] for d in data],
            positions=severities, widths=0.45, patch_artist=True,
            medianprops=dict(color="black", linewidth=2.5),
            whiskerprops=dict(linewidth=1.3, color="#444"),
            capprops=dict(linewidth=1.3, color="#444"),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.35, color="#888"),
            boxprops=dict(linewidth=1.2),
        )
        for patch, colour in zip(bp["boxes"], COLOURS):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)

        means = [np.mean(d) if d else np.nan for d in data]
        valid_x = [s for s, m in zip(severities, means) if not np.isnan(m)]
        valid_m = [m for m in means if not np.isnan(m)]
        if len(valid_x) > 1:
            ax.plot(valid_x, valid_m, "k--", linewidth=1.5, alpha=0.6,
                    marker="D", markersize=4, label="Mean")

        y_min = min((min(d) for d in data if d), default=0)
        label_y = max(0.02, y_min - 0.08)
        for s, d_list in zip(severities, data):
            n = len(d_list)
            ax.text(s, label_y, f"n={n}", ha="center", va="bottom",
                    fontsize=8, color="#444",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

        ax.set_xlim(-0.6, 4.6)
        ax.set_xticks(severities)
        ax.set_xticklabels(SEVERITY_LABELS, fontsize=9)
        ax.set_ylabel("Confidence / agreement score", fontsize=10)
        ax.set_xlabel("Severity level", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(axis="x", pad=14)
        if valid_x:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"{tech['title']}\nConfidence Score Distribution by Severity",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=3.5, w_pad=2.5)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"boxplot_{technique_key}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    for variant in ["probscore", "confscore"]:
        techniques = build_techniques(variant)
        output_dir = f"writeup_results/sentence_confidence/figures_{variant}"
        print(f"\n── {variant} ──")
        for technique_key in techniques:
            plot_technique(technique_key, output_dir, techniques)


if __name__ == "__main__":
    main()