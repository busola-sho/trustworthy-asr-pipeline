"""
v2_evaluation.py

Produces the v2 equivalents of your v1 artifacts, from the same
underlying (confidence, severity) data the AUC/ECE script already
uses - no separate data collection needed:

  1. Spearman correlation table (matching your original v1 table
     format: Method, Dataset, WER, MeanConf, MeanSev, MedSev,
     Spearman, p-val)
  2. Box plots of confidence-by-severity (matching plot_confidence_
     by_severity.py's exact visual style - 2x2 dataset panels per
     method)
  3. AUC/ECE (reuses calibration_and_discrimination.py directly)

Loads from the actual v2 method output files:
  - verbalized_confidence/verbalized_{variant}_{dataset}_{split}.json
  - model_internal_confidence/model_internal_{dataset}_{split}.json
  - cross_model_agreement/cross_model_agreement_{dataset}_{split}.json
  - learned_proxy/learned_severity_proxy_{variant}.json

Usage:
    python v2_evaluation.py --variant probscore --split test
"""

import json
import os
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration_and_discrimination import evaluate_method

SEGMENTS_DIR = "writeup_results/sentence_confidence/segments"
VERBALIZED_DIR = "writeup_results/sentence_confidence/verbalized_confidence"
MODEL_INTERNAL_DIR = "writeup_results/sentence_confidence/model_internal_confidence"
CROSSMODEL_DIR = "writeup_results/sentence_confidence/cross_model_agreement"
LEARNED_PROXY_DIR = "writeup_results/sentence_confidence/learned_proxy"
FIGURES_DIR = "writeup_results/sentence_confidence/figures_v2"

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
DATASET_DISPLAY = {
    "commonvoice": "CommonVoice Scottish", "edacc": "EdAcc",
    "english_dialects": "English Dialects", "shetland": "Shetland (held-out)",
}
SEVERITY_LABELS = ["0\nNo error", "1\nTrivial", "2\nAmbiguous", "3\nFactual", "4\nCritical"]
COLOURS = ["#2ecc71", "#95d44e", "#f39c12", "#e67e22", "#e74c3c"]


def _split_for(dataset):
    return "full" if dataset == "shetland" else "test"


def load_v2_samples(method, dataset, variant):
    """Returns a list of {"confidence": ..., "severity": ...} for one
    method+dataset+variant, reading from the actual v2 output schemas."""
    split = _split_for(dataset)
    samples = []

    if method == "verbalized":
        path = os.path.join(VERBALIZED_DIR, f"verbalized_{variant}_{dataset}_{split}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        for t in data.get("transcripts", []):
            for seg in t.get("segments", []):
                if seg.get("severity") is not None and seg.get("verbalized_score") is not None:
                    score = seg["verbalized_score"]
                    conf = score / 100 if variant == "confscore" else score  # normalize confscore's 0-100 to 0-1
                    samples.append({"confidence": conf, "severity": seg["severity"]})

    elif method == "model_internal":
        path = os.path.join(MODEL_INTERNAL_DIR, f"model_internal_{dataset}_{split}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        for t in data.get("transcripts", []):
            for seg in t.get("segments", []):
                if seg.get("severity") is not None and seg.get("model_internal_confidence") is not None:
                    samples.append({"confidence": seg["model_internal_confidence"], "severity": seg["severity"]})

    elif method in ("crossmodel_mean", "crossmodel_min"):
        path = os.path.join(CROSSMODEL_DIR, f"cross_model_agreement_{dataset}_{split}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        field = "crossmodel_mean" if method == "crossmodel_mean" else "crossmodel_min"
        for t in data.get("transcripts", []):
            for seg in t.get("segments", []):
                if seg.get("severity") is not None and seg.get(field) is not None:
                    samples.append({"confidence": seg[field], "severity": seg["severity"]})

    elif method == "proxy_model":
        path = os.path.join(LEARNED_PROXY_DIR, f"learned_severity_proxy_{variant}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        fold = data.get("lodo_folds", {}).get(dataset) if dataset != "shetland" else data.get("shetland")
        if not fold:
            return []
        for conf, sev in zip(fold["confidence_predictions"], fold["actual_severity"]):
            samples.append({"confidence": conf, "severity": sev})

    return samples


METHODS = ["verbalized", "model_internal", "crossmodel_mean", "crossmodel_min", "proxy_model"]
METHOD_DISPLAY = {
    "verbalized": lambda v: v,  # will show as "confscore"/"probscore"/"confprobscore"
    "model_internal": "model_internal",
    "crossmodel_mean": "crossmodel_mean",
    "crossmodel_min": "crossmodel_min",
    "proxy_model": "proxy_model",
}


def print_spearman_table(variant):
    print(f"\n{'='*100}")
    print(f"  V2 TABLE: {variant.upper()}-anchored (fixed SaT segmentation, all methods share this ground truth)")
    print(f"{'='*100}")
    header = f"{'Method':<20}{'Dataset':<20}{'N':>8}{'MeanConf':>12}{'MeanSev':>10}{'Spearman':>12}{'p-val':>10}"
    print(header)
    print("-" * len(header))

    for dataset in DATASETS:
        for method in METHODS:
            label = variant if method == "verbalized" else method
            samples = load_v2_samples(method, dataset, variant)
            if not samples:
                print(f"{label:<20}{dataset:<20}{'NO DATA':>8}")
                continue
            confs = [s["confidence"] for s in samples]
            sevs = [s["severity"] for s in samples]
            rho, p = spearmanr(confs, sevs)
            sig = "*" if p < 0.05 else ""
            print(f"{label:<20}{dataset:<20}{len(samples):>8}{np.mean(confs):>12.3f}"
                  f"{np.mean(sevs):>10.3f}{rho:>11.3f}{sig}{p:>10.4f}")
        print()


def print_monotonicity_summary(variant):
    """Stricter than Spearman alone: does mean confidence decrease at
    EVERY step (0>1>2>3>4), with no local reversals? A method can have
    a strongly negative Spearman rho while still having one bump (e.g.
    severity 2's mean confidence slightly ABOVE severity 1's) that
    wouldn't clearly show up in a single correlation number but would
    visibly break "consistently drops" in the box plot. This directly
    answers "which method does that most consistently"."""
    print(f"\n{'='*100}")
    print(f"  V2 MONOTONICITY CHECK: {variant.upper()} - does mean confidence strictly")
    print(f"  decrease at every severity step (0>1>2>3>4)? Direct test of 'consistently'.")
    print(f"{'='*100}")
    header = f"{'Method':<20}{'Dataset':<20}{'Means (sev 0-4)':<40}{'Fully monotonic?':<18}{'Violations'}"
    print(header)
    print("-" * len(header))

    method_scores = {}  # label -> [n_datasets_fully_monotonic, n_datasets_checked]

    for method in METHODS:
        label = variant if method == "verbalized" else method
        method_scores[label] = [0, 0]
        for dataset in DATASETS:
            samples = load_v2_samples(method, dataset, variant)
            if not samples:
                continue
            by_sev = defaultdict(list)
            for s in samples:
                by_sev[s["severity"]].append(s["confidence"])
            means = [np.mean(by_sev[s]) if by_sev.get(s) else None for s in range(5)]
            present = [(i, m) for i, m in enumerate(means) if m is not None]

            violations = []
            for (i1, m1), (i2, m2) in zip(present, present[1:]):
                if m2 > m1:  # confidence went UP as severity increased - a reversal
                    violations.append(f"{i1}\u2192{i2}")

            fully_monotonic = len(violations) == 0
            method_scores[label][1] += 1
            if fully_monotonic:
                method_scores[label][0] += 1

            means_str = " > ".join(f"{m:.2f}" if m is not None else "-" for m in means)
            print(f"{label:<20}{dataset:<20}{means_str:<40}{'YES' if fully_monotonic else 'no':<18}"
                  f"{', '.join(violations) if violations else '-'}")
        print()

    print(f"{'='*100}")
    print(f"  SUMMARY: fraction of datasets where each method was FULLY monotonic")
    print(f"{'='*100}")
    ranked = sorted(method_scores.items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else 0))
    for label, (n_ok, n_total) in ranked:
        frac = n_ok / n_total if n_total else 0
        print(f"  {label:<20} {n_ok}/{n_total} datasets fully monotonic ({frac*100:.0f}%)")


def plot_boxplot(method, variant, output_dir):
    label = variant if method == "verbalized" else method
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        samples = load_v2_samples(method, dataset, variant)
        by_sev = defaultdict(list)
        for s in samples:
            by_sev[s["severity"]].append(s["confidence"])

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

    fig.suptitle(f"V2 (Fixed SaT Segmentation) - {label}\nConfidence Score Distribution by Severity",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=3.5, w_pad=2.5)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"boxplot_v2_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def run_auc_ece(variant):
    print(f"\n{'='*100}")
    print(f"  V2 AUC/ECE: {variant.upper()}")
    print(f"{'='*100}")
    for dataset in DATASETS:
        for method in METHODS:
            label = variant if method == "verbalized" else method
            samples = load_v2_samples(method, dataset, variant)
            if not samples:
                continue
            evaluate_method(samples, method_name=label, dataset_name=dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="probscore", choices=["confscore", "confprobscore", "probscore"])
    args = parser.parse_args()

    print_spearman_table(args.variant)
    print_monotonicity_summary(args.variant)

    for method in METHODS:
        plot_boxplot(method, args.variant, FIGURES_DIR)

    run_auc_ece(args.variant)


if __name__ == "__main__":
    main()
