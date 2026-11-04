"""Plot saved Shetland and Police Scotland severity distributions.

The caller supplies the extra input files and a new output folder."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGSIZE = (13.29, 10.16)  # matches the supplied 1329 x 1016 binary figure
DPI = 100

SEVERITY_COLORS = {
    0: "#A6DFA0",  # no change
    1: "#4CAF50",  # trivial
    2: "#F9BD45",  # ambiguous
    3: "#EF8019",  # factual
    4: "#C7372A",  # critical
}

SEVERITY_LABELS = {
    0: "0 - no change",
    1: "1 - trivial",
    2: "2 - ambiguous",
    3: "3 - factual",
    4: "4 - critical",
}

BINARY_COLORS = {"preserved": "#2ECC71", "altered": "#C5392D"}


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def normalise_distribution(raw):
    """Return severity counts keyed by integers 0..4."""
    return {
        severity: int(raw.get(str(severity), raw.get(severity, 0)))
        for severity in range(5)
    }


def distribution_from_result(path):
    """Read a standard result JSON and return its scored severity counts."""
    data = load_json(path)

    if data.get("severity_distribution"):
        return normalise_distribution(data["severity_distribution"])

    samples = data.get("samples", [])
    counts = {severity: 0 for severity in range(5)}
    for sample in samples:
        if sample.get("skipped") or sample.get("error"):
            continue
        severity = sample.get("severity")
        if severity is not None:
            counts[int(severity)] += 1

    if not sum(counts.values()):
        raise ValueError(f"No scored severity values found in {path}")
    return counts


def police_distributions(individual_path, fusion_path):
    individual = load_json(individual_path)
    fusion = load_json(fusion_path)

    raw_individual = individual.get("severity_distribution")
    if not isinstance(raw_individual, dict):
        raise ValueError(
            f"Expected top-level severity_distribution in {individual_path}"
        )

    fusion_distribution = fusion.get("summary", {}).get(
        "severity_distribution"
    ) or fusion.get("severity_distribution")
    if not fusion_distribution:
        raise ValueError(f"No fusion severity distribution found in {fusion_path}")

    return {
        "Unanchored Fusion": normalise_distribution(fusion_distribution),
        "WhisperX": normalise_distribution(raw_individual["whisperx"]),
        "Parakeet": normalise_distribution(raw_individual["parakeet"]),
        "Qwen3-ASR": normalise_distribution(raw_individual["qwen"]),
        "Wav2Vec2.0": normalise_distribution(raw_individual["wav2vec2"]),
    }


def percentages(counts):
    total = sum(counts.values())
    if total == 0:
        return {severity: 0.0 for severity in range(5)}
    return {severity: counts[severity] * 100.0 / total for severity in range(5)}


def configure_axes(ax, panel_title):
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("% of samples", fontsize=26)
    ax.tick_params(axis="y", labelsize=22)
    ax.tick_params(axis="x", labelsize=21)
    ax.set_title(panel_title, fontsize=34, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.28, linewidth=1)
    ax.set_axisbelow(True)


def new_figure(main_title, panel_title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.suptitle(main_title, fontsize=30, fontweight="bold", y=0.82)
    configure_axes(ax, panel_title)
    # Large top area for the legend and title; generous bottom for rotated labels.
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.225, top=0.62)
    return fig, ax


def label_segment(ax, x, bottom, height, severity=None, binary=False):
    if height < 3.0:
        return

    # Tiny slices need a slightly smaller label but remain readable.
    fontsize = 25 if height >= 7 else 17
    if binary or severity in {3, 4}:
        color = "white"
    else:
        color = "black"

    ax.text(
        x,
        bottom + height / 2,
        f"{height:.0f}%",
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        clip_on=True,
    )


def plot_binary(distributions, main_title, panel_title, output_path):
    fig, ax = new_figure(main_title, panel_title)
    labels = list(distributions)
    x = np.arange(len(labels))
    preserved = []
    altered = []

    for label in labels:
        pct = percentages(distributions[label])
        preserved.append(pct[0] + pct[1])
        altered.append(pct[2] + pct[3] + pct[4])

    width = 0.80
    ax.bar(
        x,
        preserved,
        width,
        color=BINARY_COLORS["preserved"],
        edgecolor="black",
        linewidth=1.2,
        label="Meaning preserved (0-1)",
    )
    ax.bar(
        x,
        altered,
        width,
        bottom=preserved,
        color=BINARY_COLORS["altered"],
        edgecolor="black",
        linewidth=1.2,
        label="Meaning altered (2-4)",
    )

    for i, (preserved_value, altered_value) in enumerate(zip(preserved, altered)):
        label_segment(ax, i, 0, preserved_value, binary=True)
        label_segment(ax, i, preserved_value, altered_value, binary=True)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=23, ha="right")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=25,
        frameon=True,
        handlelength=1.8,
        handleheight=1.6,
        columnspacing=1.5,
    )
    fig.savefig(output_path, dpi=DPI, facecolor="white")
    plt.close(fig)


def plot_severity(distributions, main_title, panel_title, output_path):
    fig, ax = new_figure(main_title, panel_title)
    labels = list(distributions)
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    width = 0.80

    all_percentages = [percentages(distributions[label]) for label in labels]

    for severity in range(5):
        values = np.array([pct[severity] for pct in all_percentages])
        ax.bar(
            x,
            values,
            width,
            bottom=bottom,
            color=SEVERITY_COLORS[severity],
            edgecolor="black",
            linewidth=1.2,
            label=SEVERITY_LABELS[severity],
        )
        for i, value in enumerate(values):
            label_segment(ax, i, bottom[i], value, severity=severity)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=23, ha="right")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        fontsize=18,
        frameon=True,
        handlelength=1.4,
        handleheight=1.4,
        columnspacing=1.2,
    )
    fig.savefig(output_path, dpi=DPI, facecolor="white")
    plt.close(fig)
