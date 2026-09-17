"""
rerunning/sentence_confidence/plot_confidence_by_severity.py

Boxplots for the CURRENT sentence-confidence pipeline.

Plots confidence / agreement distributions across severity levels 0-4
for:
  1. verbalized confidence
  2. model-internal ASR confidence
  3. cross-model mean agreement
  4. cross-model minimum agreement
  5. learned proxy confidence

Each figure contains the four evaluation datasets:
CommonVoice, EdAcc, English Dialects, and Shetland.

Unlike the reliability-diagram script, this file CAN plot the raw
model-internal score because boxplots only inspect ranking/distribution
against severity; the score does not need to be a calibrated probability.

Usage:
    python rerunning/sentence_confidence/plot_confidence_by_severity.py

Optional:
    python rerunning/sentence_confidence/plot_confidence_by_severity.py \
        --variant confscore
"""

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VERBALIZED_DIR = "writeup_results/sentence_confidence/verbalized_confidence"
MODEL_INTERNAL_DIR = "writeup_results/sentence_confidence/model_internal_confidence"
CROSSMODEL_DIR = "writeup_results/sentence_confidence/cross_model_agreement"
PROXY_DIR = "writeup_results/sentence_confidence/learned_proxy"

OUTPUT_ROOT = "writeup_results/sentence_confidence"

DATASETS = [
    "commonvoice",
    "edacc",
    "english_dialects",
    "shetland",
]

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

COLOURS = [
    "#2ecc71",
    "#95d44e",
    "#f39c12",
    "#e67e22",
    "#e74c3c",
]

VARIANTS = ["confscore", "confprobscore", "probscore"]


def split_for(dataset):
    return "full" if dataset == "shetland" else "test"


def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_verbalized_score(score, variant):
    if score is None:
        return None

    score = float(score)

    if variant == "confscore":
        score = score / 100.0

    return score


def load_verbalized(dataset, variant):
    split = split_for(dataset)

    path = os.path.join(
        VERBALIZED_DIR,
        f"verbalized_{variant}_{dataset}_{split}.json",
    )

    data = load_json(path)
    if data is None:
        return []

    rows = []

    for transcript in data.get("transcripts", []):
        idx = transcript.get("dataset_index")

        for pos, seg in enumerate(
            transcript.get("segments", [])
        ):
            severity = seg.get("severity")
            score = normalize_verbalized_score(
                seg.get("verbalized_score"),
                variant,
            )

            if severity is None or score is None:
                continue

            rows.append(
                {
                    "dataset_index": idx,
                    "position": pos,
                    "severity": int(severity),
                    "score": float(score),
                }
            )

    return rows


def load_model_internal(dataset):
    split = split_for(dataset)

    path = os.path.join(
        MODEL_INTERNAL_DIR,
        f"model_internal_{dataset}_{split}.json",
    )

    data = load_json(path)
    if data is None:
        return []

    rows = []

    for transcript in data.get("transcripts", []):
        idx = transcript.get("dataset_index")

        for pos, seg in enumerate(
            transcript.get("segments", [])
        ):
            severity = seg.get("severity")
            score = seg.get(
                "model_internal_confidence"
            )

            if severity is None or score is None:
                continue

            rows.append(
                {
                    "dataset_index": idx,
                    "position": pos,
                    "severity": int(severity),
                    "score": float(score),
                }
            )

    return rows


def load_crossmodel(dataset, field):
    split = split_for(dataset)

    path = os.path.join(
        CROSSMODEL_DIR,
        f"cross_model_agreement_{dataset}_{split}.json",
    )

    data = load_json(path)
    if data is None:
        return []

    rows = []

    for transcript in data.get("transcripts", []):
        idx = transcript.get("dataset_index")

        for pos, seg in enumerate(
            transcript.get("segments", [])
        ):
            severity = seg.get("severity")
            score = seg.get(field)

            if severity is None or score is None:
                continue

            rows.append(
                {
                    "dataset_index": idx,
                    "position": pos,
                    "severity": int(severity),
                    "score": float(score),
                }
            )

    return rows


def _proxy_key_to_tuple(key):
    if isinstance(key, dict):
        return (
            key.get("dataset_index"),
            key.get("segment_position"),
        )

    if isinstance(key, (list, tuple)) and len(key) >= 2:
        return (
            key[0],
            key[1],
        )

    return (None, None)


def load_proxy(dataset, variant):
    path = os.path.join(
        PROXY_DIR,
        f"learned_severity_proxy_{variant}.json",
    )

    data = load_json(path)
    if data is None:
        return []

    if dataset == "shetland":
        block = data.get("shetland")
    else:
        block = (
            data.get("lodo_folds", {})
            .get(dataset)
        )

    if not block:
        return []

    keys = block.get("keys", [])
    confidences = block.get(
        "confidence_predictions",
        [],
    )
    severities = block.get(
        "actual_severity",
        [],
    )

    rows = []

    for key, confidence, severity in zip(
        keys,
        confidences,
        severities,
    ):
        idx, pos = _proxy_key_to_tuple(key)

        if (
            idx is None
            or pos is None
            or confidence is None
            or severity is None
        ):
            continue

        rows.append(
            {
                "dataset_index": idx,
                "position": pos,
                "severity": int(severity),
                "score": float(confidence),
            }
        )

    return rows


def get_rows(method, dataset, variant):
    if method == "verbalized":
        return load_verbalized(
            dataset,
            variant,
        )

    if method == "model_internal":
        return load_model_internal(
            dataset,
        )

    if method == "crossmodel_mean":
        return load_crossmodel(
            dataset,
            "crossmodel_mean",
        )

    if method == "crossmodel_min":
        return load_crossmodel(
            dataset,
            "crossmodel_min",
        )

    if method == "proxy":
        return load_proxy(
            dataset,
            variant,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


def data_by_severity(rows):
    grouped = defaultdict(list)

    for row in rows:
        severity = row["severity"]

        if severity not in range(5):
            continue

        grouped[severity].append(
            row["score"]
        )

    return grouped


def plot_method(
    method,
    title,
    variant,
    output_dir,
    y_label,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    axes = axes.flatten()
    any_data = False

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]

        rows = get_rows(
            method,
            dataset,
            variant,
        )

        grouped = data_by_severity(rows)

        severities = list(range(5))
        data = [
            grouped.get(severity, [])
            for severity in severities
        ]

        if not any(data):
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            ax.set_title(
                DATASET_DISPLAY[dataset],
                fontsize=11,
                fontweight="bold",
            )
            continue

        any_data = True

        bp = ax.boxplot(
            [
                values if values else [np.nan]
                for values in data
            ],
            positions=severities,
            widths=0.45,
            patch_artist=True,
            medianprops={
                "color": "black",
                "linewidth": 2.5,
            },
            whiskerprops={
                "linewidth": 1.3,
                "color": "#444",
            },
            capprops={
                "linewidth": 1.3,
                "color": "#444",
            },
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "alpha": 0.35,
                "color": "#888",
            },
            boxprops={
                "linewidth": 1.2,
            },
        )

        for patch, colour in zip(
            bp["boxes"],
            COLOURS,
        ):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)

        means = [
            np.mean(values)
            if values
            else np.nan
            for values in data
        ]

        valid_x = [
            severity
            for severity, mean in zip(
                severities,
                means,
            )
            if not np.isnan(mean)
        ]

        valid_means = [
            mean
            for mean in means
            if not np.isnan(mean)
        ]

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

        # Put n labels at the bottom of each panel using axes coordinates
        # so this works for both bounded 0-1 scores and unbounded z-scores.
        for severity, values in zip(
            severities,
            data,
        ):
            ax.text(
                severity,
                0.02,
                f"n={len(values)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444",
                transform=ax.get_xaxis_transform(),
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.7,
                    "pad": 1,
                },
            )

        ax.set_xlim(-0.6, 4.6)
        ax.set_xticks(severities)
        ax.set_xticklabels(
            SEVERITY_LABELS,
            fontsize=9,
        )

        ax.set_ylabel(
            y_label,
            fontsize=10,
        )
        ax.set_xlabel(
            "Severity level",
            fontsize=10,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
            linewidth=0.7,
        )

        ax.set_title(
            DATASET_DISPLAY[dataset],
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

        ax.tick_params(
            axis="x",
            pad=14,
        )

        if valid_x:
            ax.legend(
                fontsize=8,
                loc="upper right",
            )

    if not any_data:
        plt.close(fig)
        return

    fig.suptitle(
        f"{title}\nScore Distribution by Severity",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout(
        h_pad=3.5,
        w_pad=2.5,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    path = os.path.join(
        output_dir,
        f"boxplot_{method}_{variant}.png",
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        default="all",
        choices=VARIANTS + ["all"],
    )

    args = parser.parse_args()

    variants = (
        VARIANTS
        if args.variant == "all"
        else [args.variant]
    )

    for variant in variants:
        output_dir = os.path.join(
            OUTPUT_ROOT,
            f"figures_{variant}",
        )

        print(
            f"\n-- Confidence by severity: "
            f"{variant} --"
        )

        plot_method(
            "verbalized",
            f"Verbalized confidence ({variant})",
            variant,
            output_dir,
            "Confidence",
        )

        plot_method(
            "model_internal",
            "Model-internal ASR confidence",
            variant,
            output_dir,
            "Normalized confidence score",
        )

        plot_method(
            "crossmodel_mean",
            "Cross-model agreement (mean)",
            variant,
            output_dir,
            "Agreement score",
        )

        plot_method(
            "crossmodel_min",
            "Cross-model agreement (minimum)",
            variant,
            output_dir,
            "Agreement score",
        )

        plot_method(
            "proxy",
            f"Learned proxy confidence ({variant})",
            variant,
            output_dir,
            "Derived confidence",
        )


if __name__ == "__main__":
    main()
