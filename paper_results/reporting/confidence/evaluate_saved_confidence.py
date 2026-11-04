"""Read saved confidence scores and severity labels for the paper.

Make comparisons and confidence plots. Use saved Platt parameters for
model-internal calibration measurements. Keep raw scores for box plots."""

from paper_results.reporting.paths import PaperReportFiles

import json
import os
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .confidence_metrics import evaluate_method
from .platt_scaling import apply_platt, load_platt_params


DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
IN_DOMAIN_DATASETS = ["commonvoice", "edacc", "english_dialects"]
DATASET_DISPLAY = {
    "commonvoice": "CommonVoice Scottish",
    "edacc": "EdAcc",
    "english_dialects": "English Dialects",
    "shetland": "Shetland (held-out)",
}
SEVERITY_LABELS = [
    "0\nNo error",
    "1\nTrivial",
    "2\nAmbiguous",
    "3\nFactual",
    "4\nCritical",
]
COLOURS = ["#2ecc71", "#95d44e", "#f39c12", "#e67e22", "#e74c3c"]


def _split_for(dataset):
    return "full" if dataset == "shetland" else "test"


def load_asr_calibration_parameters(paths=None):
    """Read the saved Platt parameters. Raise an error if the file is missing."""
    paths = paths or PaperReportFiles()
    if os.path.exists(paths.platt_params):
        return load_platt_params(paths.platt_params)

    raise FileNotFoundError(
        f"Missing frozen Platt parameters: {paths.platt_params}; reporting does not fit or write input parameters"
    )


def load_saved_scores_by_sentence(method, dataset, variant, paths=None):
    """Read scores with sentence keys where available.

    A key is (transcript index, sentence position). Older proxy files have no
    sentence positions, so those scores cannot be joined by an exact sentence key."""
    paths = paths or PaperReportFiles()
    split = _split_for(dataset)
    samples = []

    if method == "verbalized":
        path = os.path.join(
            paths.verbalized_dir, f"verbalized_{variant}_{dataset}_{split}.json"
        )
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        for t in data.get("transcripts", []):
            for pos, segment in enumerate(t.get("segments", [])):
                if (
                    segment.get("severity") is not None
                    and segment.get("verbalized_score") is not None
                ):
                    score = segment["verbalized_score"]
                    conf = score / 100 if variant == "confscore" else score
                    samples.append(
                        ((t["dataset_index"], pos), conf, segment["severity"])
                    )

    elif method == "model_internal":
        path = os.path.join(
            paths.internal_dir, f"model_internal_{dataset}_{split}.json"
        )
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        for t in data.get("transcripts", []):
            for pos, segment in enumerate(t.get("segments", [])):
                if (
                    segment.get("severity") is not None
                    and segment.get("model_internal_confidence") is not None
                ):
                    samples.append(
                        (
                            (t["dataset_index"], pos),
                            segment["model_internal_confidence"],
                            segment["severity"],
                        )
                    )

    elif method in ("crossmodel_mean", "crossmodel_min"):
        path = os.path.join(
            paths.agreement_dir, f"cross_model_agreement_{dataset}_{split}.json"
        )
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        field = "crossmodel_mean" if method == "crossmodel_mean" else "crossmodel_min"
        for t in data.get("transcripts", []):
            for pos, segment in enumerate(t.get("segments", [])):
                if (
                    segment.get("severity") is not None
                    and segment.get(field) is not None
                ):
                    samples.append(
                        ((t["dataset_index"], pos), segment[field], segment["severity"])
                    )

    elif method == "proxy_model":
        path = os.path.join(paths.proxy_dir, f"learned_severity_proxy_{variant}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        fold = (
            data.get("lodo_folds", {}).get(dataset)
            if dataset != "shetland"
            else data.get("shetland")
        )
        if not fold:
            return []
        raw_keys = fold.get(
            "keys"
        )  # present after the learned_proxy_confidence.py fix;
        # falls back to None (unkeyed) for older files
        if raw_keys:
            for key_pair, conf, sev in zip(
                raw_keys, fold["confidence_predictions"], fold["actual_severity"]
            ):
                samples.append(
                    (tuple(key_pair), conf, sev)
                )  # JSON saves [idx, pos] as a
                # list - tuple() makes it
                # hashable for set operations
        else:
            for conf, sev in zip(
                fold["confidence_predictions"], fold["actual_severity"]
            ):
                samples.append((None, conf, sev))

    return samples


def shared_sentence_keys(
    dataset,
    variant,
    keyable_methods=(
        "verbalized",
        "model_internal",
        "crossmodel_mean",
        "crossmodel_min",
        "proxy_model",
    ),
    paths=None,
):
    """Returns the set of (dataset_index, position) keys present with a
    valid value in EVERY keyable method - the common, fully-aligned
    sample set for a genuinely fair cross-method comparison. proxy_model
    is excluded from this intersection (see load_v2_samples_keyed) and
    reported separately with its own existing N, clearly labeled."""
    paths = paths or PaperReportFiles()
    key_sets = []
    for method in keyable_methods:
        keyed = load_saved_scores_by_sentence(method, dataset, variant, paths=paths)
        keys = {k for k, conf, sev in keyed if k is not None}
        key_sets.append(keys)
    if not key_sets:
        return set()
    intersection = key_sets[0]
    for ks in key_sets[1:]:
        intersection &= ks
    return intersection


def load_scores_for_shared_sentences(
    method, dataset, variant, intersection_keys, paths=None
):
    """Filters a method's samples down to only the common intersection
    keys - use this instead of load_v2_samples() for a genuinely fair,
    identical-N cross-method comparison. All 5 methods (including
    proxy_model, now that learned_proxy_confidence.py saves position
    keys) are filtered identically. Falls back to proxy_model's full,
    unfiltered set only if an older file without keys is encountered."""
    paths = paths or PaperReportFiles()
    keyed = load_saved_scores_by_sentence(method, dataset, variant, paths=paths)

    if method == "proxy_model" and keyed and keyed[0][0] is None:
        return [{"confidence": conf, "severity": sev} for _, conf, sev in keyed]

    return [
        {"confidence": conf, "severity": sev}
        for k, conf, sev in keyed
        if k in intersection_keys
    ]


def load_saved_scores(method, dataset, variant, paths=None):
    """Read confidence and severity values for one method and dataset.

    Use load_scores_for_shared_sentences when comparing methods on the same sentences."""
    paths = paths or PaperReportFiles()
    split = _split_for(dataset)
    samples = []

    if method == "verbalized":
        path = os.path.join(
            paths.verbalized_dir, f"verbalized_{variant}_{dataset}_{split}.json"
        )
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        for t in data.get("transcripts", []):
            for segment in t.get("segments", []):
                if (
                    segment.get("severity") is not None
                    and segment.get("verbalized_score") is not None
                ):
                    score = segment["verbalized_score"]
                    conf = (
                        score / 100 if variant == "confscore" else score
                    )  # normalize confscore's 0-100 to 0-1
                    samples.append(
                        {"confidence": conf, "severity": segment["severity"]}
                    )

    elif method == "model_internal":
        path = os.path.join(
            paths.internal_dir, f"model_internal_{dataset}_{split}.json"
        )
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        for t in data.get("transcripts", []):
            for segment in t.get("segments", []):
                if (
                    segment.get("severity") is not None
                    and segment.get("model_internal_confidence") is not None
                ):
                    samples.append(
                        {
                            "confidence": segment["model_internal_confidence"],
                            "severity": segment["severity"],
                        }
                    )

    elif method in ("crossmodel_mean", "crossmodel_min"):
        path = os.path.join(
            paths.agreement_dir, f"cross_model_agreement_{dataset}_{split}.json"
        )
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        field = "crossmodel_mean" if method == "crossmodel_mean" else "crossmodel_min"
        for t in data.get("transcripts", []):
            for segment in t.get("segments", []):
                if (
                    segment.get("severity") is not None
                    and segment.get(field) is not None
                ):
                    samples.append(
                        {"confidence": segment[field], "severity": segment["severity"]}
                    )

    elif method == "proxy_model":
        path = os.path.join(paths.proxy_dir, f"learned_severity_proxy_{variant}.json")
        if not os.path.exists(path):
            return []
        data = json.load(open(path))
        fold = (
            data.get("lodo_folds", {}).get(dataset)
            if dataset != "shetland"
            else data.get("shetland")
        )
        if not fold:
            return []
        for conf, sev in zip(fold["confidence_predictions"], fold["actual_severity"]):
            samples.append({"confidence": conf, "severity": sev})

    return samples


METHODS = [
    "verbalized",
    "model_internal",
    "crossmodel_mean",
    "crossmodel_min",
    "proxy_model",
]
METHOD_DISPLAY = {
    "verbalized": lambda v: v,  # will show as "confscore"/"probscore"/"confprobscore"
    "model_internal": "model_internal",
    "crossmodel_mean": "crossmodel_mean",
    "crossmodel_min": "crossmodel_min",
    "proxy_model": "proxy_model",
}


def print_spearman_table(variant, paths=None):
    paths = paths or PaperReportFiles()
    print(f"\n{'=' * 100}")
    print(
        f"  V2 TABLE: {variant.upper()}-anchored (fixed SaT segmentation, all methods share this ground truth)"
    )
    print(f"{'=' * 100}")
    header = f"{'Method':<20}{'Dataset':<20}{'N':>8}{'MeanConf':>12}{'MeanSev':>10}{'Spearman':>12}{'p-val':>10}"
    print(header)
    print("-" * len(header))

    for dataset in DATASETS:
        for method in METHODS:
            label = variant if method == "verbalized" else method
            samples = load_saved_scores(method, dataset, variant, paths=paths)
            if not samples:
                print(f"{label:<20}{dataset:<20}{'NO DATA':>8}")
                continue
            confs = [s["confidence"] for s in samples]
            sevs = [s["severity"] for s in samples]
            rho, p = spearmanr(confs, sevs)
            sig = "*" if p < 0.05 else ""
            print(
                f"{label:<20}{dataset:<20}{len(samples):>8}{np.mean(confs):>12.3f}"
                f"{np.mean(sevs):>10.3f}{rho:>11.3f}{sig}{p:>10.4f}"
            )
        print()


def print_monotonicity_summary(variant, paths=None):
    """Show whether mean confidence decreases at every step of the severity scale.

    A strict decrease is a different check from Spearman correlation."""
    paths = paths or PaperReportFiles()
    print(f"\n{'=' * 100}")
    print(f"  V2 MONOTONICITY CHECK: {variant.upper()} - does mean confidence strictly")
    print(
        f"  decrease at every severity step (0>1>2>3>4)? Direct test of 'consistently'."
    )
    print(f"{'=' * 100}")
    header = f"{'Method':<20}{'Dataset':<20}{'Means (sev 0-4)':<40}{'Fully monotonic?':<18}{'Violations'}"
    print(header)
    print("-" * len(header))

    method_scores = {}  # label -> [n_datasets_fully_monotonic, n_datasets_checked]

    for method in METHODS:
        label = variant if method == "verbalized" else method
        method_scores[label] = [0, 0]
        for dataset in DATASETS:
            samples = load_saved_scores(method, dataset, variant, paths=paths)
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
            print(
                f"{label:<20}{dataset:<20}{means_str:<40}{'YES' if fully_monotonic else 'no':<18}"
                f"{', '.join(violations) if violations else '-'}"
            )
        print()

    print(f"{'=' * 100}")
    print(f"  SUMMARY: fraction of datasets where each method was FULLY monotonic")
    print(f"{'=' * 100}")
    ranked = sorted(
        method_scores.items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else 0)
    )
    for label, (n_ok, n_total) in ranked:
        frac = n_ok / n_total if n_total else 0
        print(
            f"  {label:<20} {n_ok}/{n_total} datasets fully monotonic ({frac * 100:.0f}%)"
        )


def plot_boxplot(method, variant, output_dir, paths=None):
    paths = paths or PaperReportFiles()
    label = variant if method == "verbalized" else method
    ylabel = "Confidence score"

    platt_a, platt_b = (
        load_asr_calibration_parameters(paths=paths)
        if method == "model_internal"
        else (None, None)
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        intersection_keys = shared_sentence_keys(dataset, variant, paths=paths)
        samples = load_scores_for_shared_sentences(
            method, dataset, variant, intersection_keys, paths=paths
        )
        if method == "model_internal":
            samples = apply_platt(samples, platt_a, platt_b)
        by_sev = defaultdict(list)
        for s in samples:
            by_sev[s["severity"]].append(s["confidence"])

        severities = list(range(5))
        data = [by_sev.get(s, []) for s in severities]

        if not any(data):
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(
                DATASET_DISPLAY.get(dataset, dataset), fontsize=24, fontweight="bold"
            )
            continue

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

        means = [np.mean(d) if d else np.nan for d in data]
        valid_x = [s for s, m in zip(severities, means) if not np.isnan(m)]
        valid_m = [m for m in means if not np.isnan(m)]
        if len(valid_x) > 1:
            ax.plot(
                valid_x,
                valid_m,
                "k--",
                linewidth=1.5,
                alpha=0.6,
                marker="D",
                markersize=4,
                label="Mean",
            )

        for s, d_list in zip(severities, data):
            n = len(d_list)
            ax.text(
                s,
                0.04,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=18,
                color="#444",
                transform=ax.get_xaxis_transform(),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
            )

        ax.set_xlim(-0.6, 4.6)
        ax.set_xticks(severities)
        ax.set_xticklabels(SEVERITY_LABELS, fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel("Severity level", fontsize=18)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_title(
            DATASET_DISPLAY.get(dataset, dataset), fontsize=24, fontweight="bold", pad=8
        )
        ax.tick_params(axis="x", pad=14)
        if valid_x:
            ax.legend(fontsize=14, loc="upper right")

    ylabel_title = "Confidence Score"
    fig.suptitle(
        f" {label}\n{ylabel_title} Distribution by Severity",
        fontsize=26,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(h_pad=3.5, w_pad=2.5)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(
        output_dir,
        f"boxplot_v2_{label}_{variant}.png"
        if method == "proxy_model"
        else f"boxplot_v2_{label}.png",
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    if variant == "confscore" and method in {"verbalized", "proxy_model"}:
        plot_boxplots_by_dataset(method, variant, output_dir, paths=paths)


def plot_boxplots_by_dataset(method, variant, output_dir, paths=None):
    """Save each dataset panel as an independent figure."""
    paths = paths or PaperReportFiles()
    label = "confscore" if method == "verbalized" else "proxy_confscore"

    individual_dir = os.path.join(
        output_dir,
        "by_dataset",
    )
    os.makedirs(individual_dir, exist_ok=True)

    for dataset in DATASETS:
        intersection_keys = shared_sentence_keys(dataset, variant, paths=paths)
        samples = load_scores_for_shared_sentences(
            method, dataset, variant, intersection_keys, paths=paths
        )

        by_sev = defaultdict(list)
        for sample in samples:
            by_sev[sample["severity"]].append(sample["confidence"])

        severities = list(range(5))
        data = [by_sev.get(severity, []) for severity in severities]

        fig, ax = plt.subplots(figsize=(9, 6.5))

        bp = ax.boxplot(
            [values if values else [np.nan] for values in data],
            positions=severities,
            widths=0.45,
            patch_artist=True,
            medianprops=dict(
                color="black",
                linewidth=2.5,
            ),
            whiskerprops=dict(
                linewidth=1.3,
                color="#444",
            ),
            capprops=dict(
                linewidth=1.3,
                color="#444",
            ),
            flierprops=dict(
                marker="o",
                markersize=2.5,
                alpha=0.35,
                color="#888",
            ),
            boxprops=dict(linewidth=1.2),
        )

        for patch, colour in zip(bp["boxes"], COLOURS):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)

        means = [np.mean(values) if values else np.nan for values in data]
        valid_x = [
            severity for severity, mean in zip(severities, means) if not np.isnan(mean)
        ]
        valid_means = [mean for mean in means if not np.isnan(mean)]

        if len(valid_x) > 1:
            ax.plot(
                valid_x,
                valid_means,
                "k--",
                linewidth=1.5,
                alpha=0.6,
                marker="D",
                markersize=4,
                label="Mean",
            )

        for severity, values in zip(severities, data):
            ax.text(
                severity,
                0.04,
                f"n={len(values)}",
                ha="center",
                va="bottom",
                fontsize=14,
                color="#444",
                transform=ax.get_xaxis_transform(),
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    pad=1,
                ),
            )

        ax.set_xlim(-0.6, 4.6)
        ax.set_xticks(severities)
        ax.set_xticklabels(
            SEVERITY_LABELS,
            fontsize=14,
        )
        ax.tick_params(
            axis="y",
            labelsize=14,
        )
        ax.tick_params(
            axis="x",
            pad=12,
        )
        ax.set_ylabel(
            "Confidence score",
            fontsize=17,
        )
        ax.set_xlabel(
            "Severity level",
            fontsize=17,
        )
        ax.set_title(
            DATASET_DISPLAY.get(dataset, dataset),
            fontsize=22,
            fontweight="bold",
            pad=10,
        )
        ax.grid(
            axis="y",
            alpha=0.25,
            linewidth=0.7,
        )

        if valid_x:
            ax.legend(
                fontsize=13,
                loc="upper right",
            )

        plt.tight_layout()

        output_path = os.path.join(
            individual_dir,
            f"{label}_{dataset}.png",
        )
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {output_path}")


def run_auc_ece(variant, paths=None):
    paths = paths or PaperReportFiles()
    print(f"\n{'=' * 100}")
    print(
        f"  V2 AUC/ECE: {variant.upper()} (restricted to common intersection - matches"
    )
    print(
        f"  the FAIR COMPARISON table and box plots, not each method's own raw coverage)"
    )
    print(
        f"  NOTE: model_internal's confidence is Platt-scaled (fit on pooled dev) before"
    )
    print(
        f"  ECE here, since its native z-score is unbounded and was silently mis-binned"
    )
    print(
        f"  by compute_calibration()'s [0,1] assumption. AUC is unaffected (rank-based)."
    )
    print(f"{'=' * 100}")

    platt_a, platt_b = load_asr_calibration_parameters(paths=paths)

    per_method_aucs = {method: [] for method in METHODS}
    per_method_eces = {method: [] for method in METHODS}

    for dataset in DATASETS:
        intersection_keys = shared_sentence_keys(dataset, variant, paths=paths)
        for method in METHODS:
            label = variant if method == "verbalized" else method
            samples = load_scores_for_shared_sentences(
                method, dataset, variant, intersection_keys, paths=paths
            )
            if not samples:
                continue

            if method == "model_internal":
                samples = apply_platt(samples, platt_a, platt_b)

            result = evaluate_method(samples, method_name=label, dataset_name=dataset)
            if result.get("auc") is not None:
                per_method_aucs[method].append(result["auc"])
            if result.get("ece") is not None:
                per_method_eces[method].append(result["ece"])

    print(f"\n{'=' * 100}")
    print(
        f"  MEAN AUC / MEAN ECE per method, averaged across the {len(DATASETS)} datasets"
    )
    print(f"  (fair comparison - same intersected N feeding every number above)")
    print(f"{'=' * 100}")
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


def print_intersected_comparison(variant, paths=None):
    paths = paths or PaperReportFiles()
    print(f"\n{'=' * 100}")
    print(
        f"  V2 FAIR COMPARISON: {variant.upper()} - all 5 methods restricted to the common"
    )
    print(
        f"  intersection of segments scorable by every method (genuinely identical N throughout)."
    )
    print(f"{'=' * 100}")
    header = f"{'Method':<20}{'Dataset':<20}{'N':>8}{'MeanConf':>12}{'MeanSev':>10}{'Spearman':>12}{'p-val':>10}"
    print(header)
    print("-" * len(header))

    for dataset in DATASETS:
        intersection_keys = shared_sentence_keys(dataset, variant, paths=paths)
        for method in METHODS:
            label = variant if method == "verbalized" else method
            samples = load_scores_for_shared_sentences(
                method, dataset, variant, intersection_keys, paths=paths
            )
            if not samples:
                print(f"{label:<20}{dataset:<20}{'NO DATA':>8}")
                continue
            confs = [s["confidence"] for s in samples]
            sevs = [s["severity"] for s in samples]
            rho, p = spearmanr(confs, sevs)
            sig = "*" if p < 0.05 else ""
            print(
                f"{label:<20}{dataset:<20}{len(samples):>8}{np.mean(confs):>12.3f}"
                f"{np.mean(sevs):>10.3f}{rho:>11.3f}{sig}{p:>10.4f}"
            )
        print()


def main(paths=None):
    paths = paths or PaperReportFiles()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default="probscore",
        choices=["confscore", "confprobscore", "probscore"],
    )
    args = parser.parse_args()

    print_spearman_table(args.variant, paths=paths)
    print_monotonicity_summary(args.variant, paths=paths)
    print_intersected_comparison(args.variant, paths=paths)

    for method in METHODS:
        plot_boxplot(method, args.variant, paths.confidence_figures, paths=paths)

    run_auc_ece(args.variant, paths=paths)


if __name__ == "__main__":
    raise SystemExit("Use python -m asr_pipeline report with a separate --output-dir")
