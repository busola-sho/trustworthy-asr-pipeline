"""
rerunning/sentence_confidence/plot_reliability_diagrams.py

Reliability diagrams for the CURRENT sentence-confidence pipeline.

The target is meaning preservation:
    y = 1  if severity < 2
    y = 0  if severity >= 2

This matches the project's locked binary convention:
    severity >= 2 -> meaning-altering / flagged
    severity < 2  -> meaning-preserved

Important:
- Reliability diagrams are only meaningful for scores naturally expressed
  on a 0-1 confidence scale.
- Therefore this script plots:
    * verbalized confidence variants (confscore/confprobscore/probscore)
    * cross-model agreement mean/min
    * learned proxy's derived confidence score
- Raw model-internal confidence is intentionally NOT plotted here because
  it is a z-scored signal, not a calibrated probability. It remains useful
  for ranking/discrimination analyses and is plotted by
  plot_confidence_by_severity.py.

Current input directories:
    writeup_results/sentence_confidence/verbalized_confidence/
    writeup_results/sentence_confidence/cross_model_agreement/
    writeup_results/sentence_confidence/learned_proxy/

Usage:
    python rerunning/sentence_confidence/plot_reliability_diagrams.py

Optional:
    python rerunning/sentence_confidence/plot_reliability_diagrams.py \
        --variant confscore
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VERBALIZED_DIR = "writeup_results/sentence_confidence/verbalized_confidence"
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

DATASET_COLOURS = {
    "commonvoice": "#3498db",
    "edacc": "#e67e22",
    "english_dialects": "#2ecc71",
    "shetland": "#9b59b6",
}

VARIANTS = ["confscore", "confprobscore", "probscore"]
N_BINS = 10
FLAG_THRESHOLD = 2


def split_for(dataset):
    return "full" if dataset == "shetland" else "test"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_verbalized_score(score, variant):
    """
    Convert verbalized scores to 0-1 where needed.

    confscore historically uses a 0-100 scale.
    confprobscore/probscore use 0-1.
    """
    if score is None:
        return None

    score = float(score)

    if variant == "confscore":
        score = score / 100.0

    return min(1.0, max(0.0, score))


def load_verbalized(dataset, variant):
    split = split_for(dataset)
    path = os.path.join(
        VERBALIZED_DIR,
        f"verbalized_{variant}_{dataset}_{split}.json",
    )
    data = load_json(path)
    if data is None:
        return {}

    pairs = {}
    for transcript in data.get("transcripts", []):
        idx = transcript.get("dataset_index")
        if idx is None:
            continue

        for pos, seg in enumerate(transcript.get("segments", [])):
            severity = seg.get("severity")
            score = normalize_verbalized_score(
                seg.get("verbalized_score"),
                variant,
            )
            if severity is None or score is None:
                continue

            pairs[(idx, pos)] = (score, severity)

    return pairs


def load_crossmodel(dataset, field):
    split = split_for(dataset)
    path = os.path.join(
        CROSSMODEL_DIR,
        f"cross_model_agreement_{dataset}_{split}.json",
    )
    data = load_json(path)
    if data is None:
        return {}

    pairs = {}
    for transcript in data.get("transcripts", []):
        idx = transcript.get("dataset_index")
        if idx is None:
            continue

        for pos, seg in enumerate(transcript.get("segments", [])):
            severity = seg.get("severity")
            score = seg.get(field)

            if severity is None or score is None:
                continue

            score = min(1.0, max(0.0, float(score)))
            pairs[(idx, pos)] = (score, severity)

    return pairs


def _proxy_key_to_tuple(key):
    """
    Supports both:
      old/current key format: [dataset_index, position]
      improved key format: {"dataset_index": ..., "segment_position": ...}
    """
    if isinstance(key, dict):
        return (
            key.get("dataset_index"),
            key.get("segment_position"),
        )

    if isinstance(key, (list, tuple)) and len(key) >= 2:
        return (key[0], key[1])

    return (None, None)


def load_proxy(dataset, variant):
    path = os.path.join(
        PROXY_DIR,
        f"learned_severity_proxy_{variant}.json",
    )
    data = load_json(path)
    if data is None:
        return {}

    if dataset == "shetland":
        block = data.get("shetland")
    else:
        block = data.get("lodo_folds", {}).get(dataset)

    if not block:
        return {}

    keys = block.get("keys", [])
    confidences = block.get("confidence_predictions", [])
    severities = block.get("actual_severity", [])

    pairs = {}

    for key, confidence, severity in zip(keys, confidences, severities):
        idx, pos = _proxy_key_to_tuple(key)

        if idx is None or pos is None:
            continue

        if confidence is None or severity is None:
            continue

        confidence = min(
            1.0,
            max(0.0, float(confidence)),
        )

        pairs[(idx, pos)] = (
            confidence,
            float(severity),
        )

    return pairs


def get_method_pairs(method, dataset, variant):
    if method == "verbalized":
        return load_verbalized(dataset, variant)

    if method == "crossmodel_mean":
        return load_crossmodel(dataset, "crossmodel_mean")

    if method == "crossmodel_min":
        return load_crossmodel(dataset, "crossmodel_min")

    if method == "proxy":
        return load_proxy(dataset, variant)

    raise ValueError(f"Unknown method: {method}")


def compute_reliability(pairs, n_bins=N_BINS):
    if not pairs:
        return None

    confidences = np.array(
        [score for score, _ in pairs.values()],
        dtype=float,
    )

    preserved = np.array(
        [
            1.0 if severity < FLAG_THRESHOLD else 0.0
            for _, severity in pairs.values()
        ],
        dtype=float,
    )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    mean_confidences = []
    observed_preservation = []
    counts = []

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (confidences >= lo)
                & (confidences <= hi)
            )
        else:
            mask = (
                (confidences >= lo)
                & (confidences < hi)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_confidences.append(
            float(confidences[mask].mean())
        )
        observed_preservation.append(
            float(preserved[mask].mean())
        )
        counts.append(count)

    return {
        "mean_confidence": mean_confidences,
        "observed_preservation": observed_preservation,
        "counts": counts,
        "n": len(confidences),
    }


def expected_calibration_error(pairs, n_bins=N_BINS):
    reliability = compute_reliability(
        pairs,
        n_bins=n_bins,
    )

    if reliability is None:
        return None

    total = sum(reliability["counts"])
    if total == 0:
        return None

    ece = 0.0

    for conf, obs, count in zip(
        reliability["mean_confidence"],
        reliability["observed_preservation"],
        reliability["counts"],
    ):
        ece += (
            count / total
        ) * abs(obs - conf)

    return float(ece)


def plot_method(method, title, variant, output_dir):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        alpha=0.4,
        label="Perfect calibration",
    )

    any_data = False

    for dataset in DATASETS:
        pairs = get_method_pairs(
            method,
            dataset,
            variant,
        )

        reliability = compute_reliability(pairs)

        if reliability is None:
            print(
                f"  WARNING: no data for "
                f"{method}/{variant}/{dataset}"
            )
            continue

        any_data = True
        ece = expected_calibration_error(pairs)

        label = (
            f"{DATASET_DISPLAY[dataset]} "
            f"(ECE={ece:.3f}, n={reliability['n']})"
        )

        ax.plot(
            reliability["mean_confidence"],
            reliability["observed_preservation"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=DATASET_COLOURS[dataset],
            label=label,
        )

    if not any_data:
        plt.close(fig)
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(
        "Predicted confidence",
        fontsize=11,
    )
    ax.set_ylabel(
        "Observed fraction meaning-preserved (severity < 2)",
        fontsize=11,
    )

    ax.set_title(
        f"Reliability Diagram\n{title}",
        fontsize=12,
        fontweight="bold",
    )

    ax.legend(
        fontsize=8,
        loc="upper left",
    )
    ax.grid(alpha=0.3)

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    path = os.path.join(
        output_dir,
        f"reliability_{method}_{variant}.png",
    )

    plt.tight_layout()
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

        print(f"\n-- Reliability diagrams: {variant} --")

        plot_method(
            "verbalized",
            f"Verbalized confidence ({variant})",
            variant,
            output_dir,
        )

        plot_method(
            "crossmodel_mean",
            "Cross-model agreement (mean)",
            variant,
            output_dir,
        )

        plot_method(
            "crossmodel_min",
            "Cross-model agreement (minimum)",
            variant,
            output_dir,
        )

        plot_method(
            "proxy",
            f"Learned proxy confidence ({variant})",
            variant,
            output_dir,
        )


if __name__ == "__main__":
    main()