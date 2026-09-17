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

CALIBRATION FIX: model_internal's raw confidence is an UNBOUNDED
z-score (can be negative or >1), but compute_calibration() assumes
[0,1] and bins accordingly - values outside that range were silently
falling into no bin at all, while still being counted in ECE's
denominator (confirmed on real data: ~65% of model_internal samples
excluded from the numerator this way, given its realistic mean~-0.2
z-score distribution). AUC/Spearman/monotonicity are UNAFFECTED (rank-
based, invariant to monotonic transforms) - only ECE needed fixing.

FIX: Platt scaling (logistic regression: P(preserved|z) =
sigmoid(a*z+b)), fit ONCE on POOLED dev data across the 3 in-domain
datasets (same pooling convention as Method 2's own normalization
stats), frozen and applied to model_internal's samples specifically
before ECE computation - matches the dev/test discipline used
throughout this project for every other fitted parameter. A raw
unfitted sigmoid was considered and rejected: it maps into [0,1] but
has no reason to equal the true P(preserved|z) unless actually learned
from labelled data. Box plots continue to show model_internal's RAW
z-score (unchanged) - Platt scaling is applied only where calibration
is being measured (ECE), not to the visualization of the raw signal.

LABEL FIX: box plot y-axis was hardcoded "Confidence / agreement
score" for ALL 5 methods, even though only crossmodel_mean/
crossmodel_min are genuinely agreement scores - verbalized,
model_internal, and proxy_model are all confidence scores, not
agreement. Now conditional per method.

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
from fit_platt_scaling import fit_platt, apply_platt, save_platt_params, load_platt_params

SEGMENTS_DIR = "writeup_results/sentence_confidence/segments"
VERBALIZED_DIR = "writeup_results/sentence_confidence/verbalized_confidence"
MODEL_INTERNAL_DIR = "writeup_results/sentence_confidence/model_internal_confidence"
CROSSMODEL_DIR = "writeup_results/sentence_confidence/cross_model_agreement"
LEARNED_PROXY_DIR = "writeup_results/sentence_confidence/learned_proxy"
FIGURES_DIR = "writeup_results/sentence_confidence/figures_v2"
PLATT_PARAMS_PATH = "writeup_results/sentence_confidence/model_internal_platt_params.json"

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
IN_DOMAIN_DATASETS = ["commonvoice", "edacc", "english_dialects"]
DATASET_DISPLAY = {
    "commonvoice": "CommonVoice Scottish", "edacc": "EdAcc",
    "english_dialects": "English Dialects", "shetland": "Shetland (held-out)",
}
SEVERITY_LABELS = ["0\nNo error", "1\nTrivial", "2\nAmbiguous", "3\nFactual", "4\nCritical"]
COLOURS = ["#2ecc71", "#95d44e", "#f39c12", "#e67e22", "#e74c3c"]


def _split_for(dataset):
    return "full" if dataset == "shetland" else "test"


def load_model_internal_dev(dataset):
    """Loads model_internal's RAW z-score samples from the DEV split
    directly - variant-independent (model_internal doesn't depend on
    verbalized variant at all), used only for fitting Platt scaling."""
    path = os.path.join(MODEL_INTERNAL_DIR, f"model_internal_{dataset}_dev.json")
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    samples = []
    for t in data.get("transcripts", []):
        for seg in t.get("segments", []):
            if seg.get("severity") is not None and seg.get("model_internal_confidence") is not None:
                samples.append({"confidence": seg["model_internal_confidence"], "severity": seg["severity"]})
    return samples


def get_model_internal_platt_params():
    """Fits Platt scaling ONCE on pooled dev data (cached to disk -
    model_internal is variant-independent, so this never needs
    refitting per variant). Returns frozen (a, b)."""
    if os.path.exists(PLATT_PARAMS_PATH):
        return load_platt_params(PLATT_PARAMS_PATH)

    pooled_dev = []
    for dataset in IN_DOMAIN_DATASETS:
        pooled_dev += load_model_internal_dev(dataset)
    if not pooled_dev:
        raise FileNotFoundError("No model_internal dev data found to fit Platt scaling - "
                                "check model_internal_{dataset}_dev.json files exist")

    a, b = fit_platt(pooled_dev)
    os.makedirs(os.path.dirname(PLATT_PARAMS_PATH) or ".", exist_ok=True)
    save_platt_params(a, b, PLATT_PARAMS_PATH)
    print(f"  Fitted Platt scaling for model_internal (pooled dev, N={len(pooled_dev)}): a={a:.4f}, b={b:.4f}")
    return a, b


def load_v2_samples_keyed(method, dataset, variant):
    """Like load_v2_samples, but ALSO returns a (dataset_index, position)
    key per sample where available - needed to compute a genuine
    intersection across methods. proxy_model's saved output does NOT
    carry position keys (learned_proxy_confidence.py only saved a flat
    prediction list) - returns None as the key for those, meaning
    proxy_model cannot be joined into the exact intersection without
    modifying that script to save keys and rerunning it. Returns a list
    of (key, confidence, severity) tuples."""
    split = _split_for(dataset)
    samples = []

    if method == "verbalized":
        path = os.path.join(VERBALIZED_DIR, f"verbalized_{variant}_{dataset}_{split}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        for t in data.get("transcripts", []):
            for pos, seg in enumerate(t.get("segments", [])):
                if seg.get("severity") is not None and seg.get("verbalized_score") is not None:
                    score = seg["verbalized_score"]
                    conf = score / 100 if variant == "confscore" else score
                    samples.append(((t["dataset_index"], pos), conf, seg["severity"]))

    elif method == "model_internal":
        path = os.path.join(MODEL_INTERNAL_DIR, f"model_internal_{dataset}_{split}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        for t in data.get("transcripts", []):
            for pos, seg in enumerate(t.get("segments", [])):
                if seg.get("severity") is not None and seg.get("model_internal_confidence") is not None:
                    samples.append(((t["dataset_index"], pos), seg["model_internal_confidence"], seg["severity"]))

    elif method in ("crossmodel_mean", "crossmodel_min"):
        path = os.path.join(CROSSMODEL_DIR, f"cross_model_agreement_{dataset}_{split}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        field = "crossmodel_mean" if method == "crossmodel_mean" else "crossmodel_min"
        for t in data.get("transcripts", []):
            for pos, seg in enumerate(t.get("segments", [])):
                if seg.get("severity") is not None and seg.get(field) is not None:
                    samples.append(((t["dataset_index"], pos), seg[field], seg["severity"]))

    elif method == "proxy_model":
        path = os.path.join(LEARNED_PROXY_DIR, f"learned_severity_proxy_{variant}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        fold = data.get("lodo_folds", {}).get(dataset) if dataset != "shetland" else data.get("shetland")
        if not fold:
            return []
        raw_keys = fold.get("keys")  # present after the learned_proxy_confidence.py fix;
                                      # falls back to None (unkeyed) for older files
        if raw_keys:
            for key_pair, conf, sev in zip(raw_keys, fold["confidence_predictions"], fold["actual_severity"]):
                samples.append((tuple(key_pair), conf, sev))  # JSON saves [idx, pos] as a
                                                                # list - tuple() makes it
                                                                # hashable for set operations
        else:
            for conf, sev in zip(fold["confidence_predictions"], fold["actual_severity"]):
                samples.append((None, conf, sev))

    return samples


def compute_intersection_keys(dataset, variant, keyable_methods=("verbalized", "model_internal", "crossmodel_mean", "crossmodel_min", "proxy_model")):
    """Returns the set of (dataset_index, position) keys present with a
    valid value in EVERY keyable method - the common, fully-aligned
    sample set for a genuinely fair cross-method comparison. proxy_model
    is excluded from this intersection (see load_v2_samples_keyed) and
    reported separately with its own existing N, clearly labeled."""
    key_sets = []
    for method in keyable_methods:
        keyed = load_v2_samples_keyed(method, dataset, variant)
        keys = {k for k, conf, sev in keyed if k is not None}
        key_sets.append(keys)
    if not key_sets:
        return set()
    intersection = key_sets[0]
    for ks in key_sets[1:]:
        intersection &= ks
    return intersection


def load_v2_samples_intersected(method, dataset, variant, intersection_keys):
    """Filters a method's samples down to only the common intersection
    keys - use this instead of load_v2_samples() for a genuinely fair,
    identical-N cross-method comparison. All 5 methods (including
    proxy_model, now that learned_proxy_confidence.py saves position
    keys) are filtered identically. Falls back to proxy_model's full,
    unfiltered set only if an older file without keys is encountered."""
    keyed = load_v2_samples_keyed(method, dataset, variant)

    if method == "proxy_model" and keyed and keyed[0][0] is None:
        return [{"confidence": conf, "severity": sev} for _, conf, sev in keyed]

    return [{"confidence": conf, "severity": sev} for k, conf, sev in keyed if k in intersection_keys]


def load_v2_samples(method, dataset, variant):
    """Returns a list of {"confidence": ..., "severity": ...} for one
    method+dataset+variant, reading from the actual v2 output schemas.
    This is the ORIGINAL per-method version (each method's own full
    available sample set, not restricted to a common intersection) -
    use load_v2_samples_intersected() instead for a fair cross-method
    comparison; this one remains useful for reporting each method's
    real-world usable coverage on its own."""
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
    ylabel = "Confidence score"

    platt_a, platt_b = (get_model_internal_platt_params() if method == "model_internal" else (None, None))

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        intersection_keys = compute_intersection_keys(dataset, variant)
        samples = load_v2_samples_intersected(method, dataset, variant, intersection_keys)
        if method == "model_internal":
            samples = apply_platt(samples, platt_a, platt_b)
        by_sev = defaultdict(list)
        for s in samples:
            by_sev[s["severity"]].append(s["confidence"])

        severities = list(range(5))
        data = [by_sev.get(s, []) for s in severities]

        if not any(data):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=24, fontweight="bold")
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

        for s, d_list in zip(severities, data):
            n = len(d_list)
            ax.text(s, 0.04, f"n={n}", ha="center", va="bottom",
                    fontsize=18, color="#444", transform=ax.get_xaxis_transform(),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

        ax.set_xlim(-0.6, 4.6)
        ax.set_xticks(severities)
        ax.set_xticklabels(SEVERITY_LABELS, fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel("Severity level", fontsize=18)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=24, fontweight="bold", pad=8)
        ax.tick_params(axis="x", pad=14)
        if valid_x:
            ax.legend(fontsize=14, loc="upper right")

    ylabel_title = "Confidence Score"
    fig.suptitle(f" {label}\n{ylabel_title} Distribution by Severity",
                 fontsize=26, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=3.5, w_pad=2.5)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"boxplot_v2_{label}_{variant}.png"
                        if method == "proxy_model" else f"boxplot_v2_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def run_auc_ece(variant):
    print(f"\n{'='*100}")
    print(f"  V2 AUC/ECE: {variant.upper()} (restricted to common intersection - matches")
    print(f"  the FAIR COMPARISON table and box plots, not each method's own raw coverage)")
    print(f"  NOTE: model_internal's confidence is Platt-scaled (fit on pooled dev) before")
    print(f"  ECE here, since its native z-score is unbounded and was silently mis-binned")
    print(f"  by compute_calibration()'s [0,1] assumption. AUC is unaffected (rank-based).")
    print(f"{'='*100}")

    platt_a, platt_b = get_model_internal_platt_params()

    per_method_aucs = {method: [] for method in METHODS}
    per_method_eces = {method: [] for method in METHODS}

    for dataset in DATASETS:
        intersection_keys = compute_intersection_keys(dataset, variant)
        for method in METHODS:
            label = variant if method == "verbalized" else method
            samples = load_v2_samples_intersected(method, dataset, variant, intersection_keys)
            if not samples:
                continue

            if method == "model_internal":
                samples = apply_platt(samples, platt_a, platt_b)

            result = evaluate_method(samples, method_name=label, dataset_name=dataset)
            if result.get("auc") is not None:
                per_method_aucs[method].append(result["auc"])
            if result.get("ece") is not None:
                per_method_eces[method].append(result["ece"])

    print(f"\n{'='*100}")
    print(f"  MEAN AUC / MEAN ECE per method, averaged across the {len(DATASETS)} datasets")
    print(f"  (fair comparison - same intersected N feeding every number above)")
    print(f"{'='*100}")
    header = f"{'Method':<20}{'Mean AUC':>12}{'Mean ECE':>12}{'N datasets':>12}"
    print(header)
    print("-" * len(header))
    for method in METHODS:
        label = variant if method == "verbalized" else method
        aucs = per_method_aucs[method]
        eces = per_method_eces[method]
        mean_auc = sum(aucs) / len(aucs) if aucs else None
        mean_ece = sum(eces) / len(eces) if eces else None
        auc_str = f"{mean_auc:.3f}" if mean_auc is not None else "-"
        ece_str = f"{mean_ece:.3f}" if mean_ece is not None else "-"
        print(f"{label:<20}{auc_str:>12}{ece_str:>12}{len(aucs):>12}")


def print_intersected_comparison(variant):
    print(f"\n{'='*100}")
    print(f"  V2 FAIR COMPARISON: {variant.upper()} - all 5 methods restricted to the common")
    print(f"  intersection of segments scorable by every method (genuinely identical N throughout).")
    print(f"{'='*100}")
    header = f"{'Method':<20}{'Dataset':<20}{'N':>8}{'MeanConf':>12}{'MeanSev':>10}{'Spearman':>12}{'p-val':>10}"
    print(header)
    print("-" * len(header))

    for dataset in DATASETS:
        intersection_keys = compute_intersection_keys(dataset, variant)
        for method in METHODS:
            label = variant if method == "verbalized" else method
            samples = load_v2_samples_intersected(method, dataset, variant, intersection_keys)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="probscore", choices=["confscore", "confprobscore", "probscore"])
    args = parser.parse_args()

    print_spearman_table(args.variant)
    print_monotonicity_summary(args.variant)
    print_intersected_comparison(args.variant)

    for method in METHODS:
        plot_boxplot(method, args.variant, FIGURES_DIR)

    run_auc_ece(args.variant)


if __name__ == "__main__":
    main()