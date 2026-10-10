import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
FIGURES_DIR = PACKAGE_ROOT / "figures"
TABLES_DIR = PACKAGE_ROOT / "tables"
POLICE_DIR = PACKAGE_ROOT / "data" / "police_scotland"

sys.path.insert(0, str(SCRIPTS_DIR))

spec = importlib.util.spec_from_file_location(
    "confidence_evaluation",
    SCRIPTS_DIR / "v2_evaluation_v2.py",
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


METHODS = [
    {
        "label": "Verbalised\n(confscore)",
        "public_method": "verbalized",
        "variant": "confscore",
        "police_file": "method_confscore_ps.json",
    },
    {
        "label": "Verbalised\n(probscore)",
        "public_method": "verbalized",
        "variant": "probscore",
        "police_file": "method_probscore_ps.json",
    },
    {
        "label": "Learned proxy\n(confscore)",
        "public_method": "proxy_model",
        "variant": "confscore",
        "police_file": "method_proxy_confscore_ps.json",
    },
    {
        "label": "Learned proxy\n(probscore)",
        "public_method": "proxy_model",
        "variant": "probscore",
        "police_file": "method_proxy_probscore_ps.json",
    },
    {
        "label": "Cross-model\nmean",
        "public_method": "crossmodel_mean",
        "variant": "confscore",
        "police_file": "method_crossmodel_mean_ps.json",
    },
    {
        "label": "Cross-model\nminimum",
        "public_method": "crossmodel_min",
        "variant": "confscore",
        "police_file": "method_crossmodel_min_ps.json",
    },
    {
        "label": "Model-internal",
        "public_method": "model_internal",
        "variant": "confscore",
        "police_file": "method_model_internal_ps.json",
    },
]

PUBLIC_DATASETS = [
    "commonvoice",
    "edacc",
    "english_dialects",
    "shetland",
]

DATASET_LABELS = {
    "commonvoice": "Common Voice",
    "edacc": "EdAcc",
    "english_dialects": "English Dialects",
    "shetland": "Shetland",
    "police_scotland": "Police Scotland",
}

DATASET_COLOURS = {
    "commonvoice": "#3274A1",
    "edacc": "#E1812C",
    "english_dialects": "#3A923A",
    "shetland": "#C03D3E",
    "police_scotland": "#7A5AA6",
}


def public_correlation(dataset, method, variant):
    keys = evaluation.compute_intersection_keys(
        dataset,
        variant,
    )

    samples = evaluation.load_v2_samples_intersected(
        method,
        dataset,
        variant,
        keys,
    )

    confidences = [
        sample["confidence"]
        for sample in samples
    ]
    severities = [
        sample["severity"]
        for sample in samples
    ]

    correlation, p_value = spearmanr(
        confidences,
        severities,
    )

    return float(correlation), float(p_value), len(samples)


def load_police_data():
    keyed = {}

    for method in METHODS:
        path = POLICE_DIR / method["police_file"]

        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)

        rows = {}

        for row in data.get("rows", []):
            key = (
                row.get("dataset_index"),
                row.get("sent_pos"),
            )
            confidence = row.get("confidence")
            severity = row.get("severity")

            if (
                key[0] is None
                or key[1] is None
                or confidence is None
                or severity is None
            ):
                continue

            rows[key] = (
                float(confidence),
                float(severity),
            )

        keyed[method["label"]] = rows

    common_keys = set.intersection(
        *(set(rows) for rows in keyed.values())
    )

    return keyed, common_keys


def police_correlation(method_label, keyed, common_keys):
    values = [
        keyed[method_label][key]
        for key in sorted(common_keys)
    ]

    confidences = [
        confidence
        for confidence, _ in values
    ]
    severities = [
        severity
        for _, severity in values
    ]

    correlation, p_value = spearmanr(
        confidences,
        severities,
    )

    return (
        float(correlation),
        float(p_value),
        len(values),
    )


def collect_results():
    results = []

    for method in METHODS:
        for dataset in PUBLIC_DATASETS:
            correlation, p_value, n = public_correlation(
                dataset,
                method["public_method"],
                method["variant"],
            )

            results.append({
                "method": method["label"].replace("\n", " "),
                "dataset": dataset,
                "spearman": correlation,
                "p_value": p_value,
                "n": n,
            })

    police_rows, police_keys = load_police_data()

    for method in METHODS:
        correlation, p_value, n = police_correlation(
            method["label"],
            police_rows,
            police_keys,
        )

        results.append({
            "method": method["label"].replace("\n", " "),
            "dataset": "police_scotland",
            "spearman": correlation,
            "p_value": p_value,
            "n": n,
        })

    return results


def save_table(results):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "spearman_correlations.csv"

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "dataset",
                "spearman",
                "p_value",
                "n",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {path}")


def result_lookup(results):
    return {
        (row["method"], row["dataset"]): row["spearman"]
        for row in results
    }


def plot_combined(results):
    lookup = result_lookup(results)

    datasets = PUBLIC_DATASETS + ["police_scotland"]

    # Preserve the ordering used in the original paper figure.
    ordered_methods = [
        METHODS[0],
        METHODS[1],
        METHODS[2],
        METHODS[3],
        METHODS[4],
        METHODS[6],
        METHODS[5],
    ]

    display_labels = [
        "Confscore",
        "Probscore",
        "Proxy (confscore)",
        "Proxy (probscore)",
        "Cross-model mean",
        "Model-internal",
        "Cross-model minimum",
    ]

    matrix = np.array([
        [
            lookup[
                (
                    method["label"].replace("\n", " "),
                    dataset,
                )
            ]
            for dataset in datasets
        ]
        for method in ordered_methods
    ])

    fig, ax = plt.subplots(figsize=(16, 8.5))

    image = ax.imshow(
        matrix,
        cmap="Blues_r",
        vmin=-0.43,
        vmax=-0.09,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(
        [DATASET_LABELS[dataset] for dataset in datasets],
        fontsize=15,
    )

    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_yticklabels(
        display_labels,
        fontsize=15,
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]

            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=14,
                color="white" if value <= -0.28 else "black",
            )

    ax.set_title(
        "Spearman Correlation by Method and Dataset",
        fontsize=24,
        pad=18,
    )

    colourbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.04,
    )
    colourbar.set_label(
        "More negative = stronger relationship",
        fontsize=16,
        labelpad=16,
    )
    colourbar.ax.tick_params(labelsize=13)

    ax.set_xticks(
        np.arange(-0.5, len(datasets), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(display_labels), 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color="white",
        linewidth=2,
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    output_path = (
        FIGURES_DIR / "spearman_correlations_heatmap.png"
    )
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {output_path}")


def plot_individual_datasets(results):
    lookup = result_lookup(results)
    output_dir = FIGURES_DIR / "by_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = PUBLIC_DATASETS + ["police_scotland"]

    for dataset in datasets:
        labels = [
            method["label"]
            for method in METHODS
        ]
        values = [
            lookup[
                (
                    method["label"].replace("\n", " "),
                    dataset,
                )
            ]
            for method in METHODS
        ]

        y = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(9, 6.5))
        bars = ax.barh(
            y,
            values,
            color=DATASET_COLOURS[dataset],
            alpha=0.88,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(
            labels,
            fontsize=12,
        )
        ax.invert_yaxis()
        ax.set_xlabel(
            "Spearman correlation with severity",
            fontsize=14,
        )
        ax.tick_params(axis="x", labelsize=12)
        ax.axvline(
            0,
            color="black",
            linewidth=0.8,
        )
        ax.grid(
            axis="x",
            alpha=0.25,
        )
        ax.set_title(
            DATASET_LABELS[dataset],
            fontsize=18,
            fontweight="bold",
        )

        for bar, value in zip(bars, values):
            ax.text(
                value - 0.006,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="right",
                color="white",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()

        path = output_dir / f"spearman_{dataset}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved: {path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = collect_results()
    save_table(results)
    plot_combined(results)
    plot_individual_datasets(results)


if __name__ == "__main__":
    main()
