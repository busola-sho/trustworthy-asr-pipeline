"""Paper confidence summary using the shared saved-score evaluator."""

from paper_results.reporting.paths import PaperReportFiles
import json
import numpy as np
from . import evaluate_saved_confidence as evaluation
from .confidence_metrics import evaluate_method
from .platt_scaling import apply_platt

DATASETS = evaluation.DATASETS
load_asr_calibration_parameters = evaluation.load_asr_calibration_parameters
shared_sentence_keys = evaluation.shared_sentence_keys
load_scores_for_shared_sentences = evaluation.load_scores_for_shared_sentences


def _evaluate_quietly(samples, label, dataset):
    """Run the existing evaluator without printing reliability-bin details."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_method(
            samples,
            method_name=label,
            dataset_name=dataset,
        )


def _is_monotonic(samples):
    """Check whether mean confidence decreases as severity increases."""
    means = []

    for severity in range(5):
        values = [
            sample["confidence"]
            for sample in samples
            if int(sample["severity"]) == severity
        ]

        if values:
            means.append(float(np.mean(values)))

    return len(means) >= 2 and all(
        left > right for left, right in zip(means, means[1:])
    )


def public_results(variant, method, paths=None):
    paths = paths or PaperReportFiles()
    platt_a, platt_b = load_asr_calibration_parameters(paths=paths)

    aucs = []
    eces = []
    monotonic_count = 0

    for dataset in DATASETS:
        intersection_keys = shared_sentence_keys(dataset, variant, paths=paths)

        samples = load_scores_for_shared_sentences(
            method, dataset, variant, intersection_keys, paths=paths
        )

        if not samples:
            continue

        if _is_monotonic(samples):
            monotonic_count += 1

        evaluation_samples = samples

        if method == "model_internal":
            evaluation_samples = apply_platt(
                samples,
                platt_a,
                platt_b,
            )

        label = variant if method == "verbalized" else method

        result = _evaluate_quietly(
            evaluation_samples,
            label,
            dataset,
        )

        if result.get("auc") is not None:
            aucs.append(result["auc"])

        if result.get("ece") is not None:
            eces.append(result["ece"])

    return {
        "monotonic": f"{monotonic_count}/{len(DATASETS)}",
        "auc": float(np.mean(aucs)) if aucs else None,
        "ece": float(np.mean(eces)) if eces else None,
    }


def load_police_rows(paths=None):
    paths = paths or PaperReportFiles()

    police_dir = paths.police_dir

    file_map = {
        "confscore": "method_confscore_ps.json",
        "probscore": "method_probscore_ps.json",
        "proxy_confscore": "method_proxy_confscore_ps.json",
        "proxy_probscore": "method_proxy_probscore_ps.json",
        "crossmodel_mean": "method_crossmodel_mean_ps.json",
        "crossmodel_min": "method_crossmodel_min_ps.json",
        "model_internal": "method_model_internal_ps.json",
    }

    keyed = {}

    for method, filename in file_map.items():
        path = police_dir / filename

        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)

        method_rows = {}

        for row in data.get("rows", []):
            dataset_index = row.get("dataset_index")
            sent_pos = row.get("sent_pos")
            confidence = row.get("confidence")
            severity = row.get("severity")

            if (
                dataset_index is None
                or sent_pos is None
                or confidence is None
                or severity is None
            ):
                continue

            confidence = float(confidence)

            if method == "confscore":
                confidence /= 100.0

            method_rows[(dataset_index, sent_pos)] = {
                "confidence": confidence,
                "severity": float(severity),
            }

        keyed[method] = method_rows

    common_keys = set.intersection(*(set(rows) for rows in keyed.values()))

    return keyed, common_keys


def police_results(method, keyed, common_keys, paths=None):
    paths = paths or PaperReportFiles()
    samples = [keyed[method][key] for key in sorted(common_keys)]

    if method == "model_internal":
        platt_a, platt_b = load_asr_calibration_parameters(paths=paths)
        samples = apply_platt(samples, platt_a, platt_b)

    result = _evaluate_quietly(
        samples,
        method,
        "police_scotland",
    )

    return {
        "auc": result.get("auc"),
        "ece": result.get("ece"),
        "n": len(samples),
    }


def build_paper_table(paths=None):
    paths = paths or PaperReportFiles()
    import csv

    specifications = [
        (
            "Verbalised (confscore)",
            "confscore",
            "verbalized",
            "confscore",
        ),
        (
            "Verbalised (probscore)",
            "probscore",
            "verbalized",
            "probscore",
        ),
        (
            "Learned proxy (confscore)",
            "confscore",
            "proxy_model",
            "proxy_confscore",
        ),
        (
            "Learned proxy (probscore)",
            "probscore",
            "proxy_model",
            "proxy_probscore",
        ),
        (
            "Cross-model mean",
            "confscore",
            "crossmodel_mean",
            "crossmodel_mean",
        ),
        (
            "Cross-model minimum",
            "confscore",
            "crossmodel_min",
            "crossmodel_min",
        ),
        (
            "Model-internal",
            "confscore",
            "model_internal",
            "model_internal",
        ),
    ]

    police_rows, police_keys = load_police_rows(paths=paths)

    rows = []

    for (
        display_name,
        variant,
        public_method,
        police_method,
    ) in specifications:
        public = public_results(variant, public_method, paths=paths)

        police = police_results(police_method, police_rows, police_keys, paths=paths)

        rows.append(
            {
                "Method": display_name,
                "Monotonic": public["monotonic"],
                "Public AUROC": public["auc"],
                "Public ECE": public["ece"],
                "Police AUROC": police["auc"],
                "Police ECE": police["ece"],
                "Police N": police["n"],
            }
        )

    output_dir = paths.confidence_tables
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "confidence_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    txt_path = output_dir / "confidence_summary.txt"

    header = (
        f"{'Method':<29}"
        f"{'Mono.':>8}"
        f"{'Public AUC':>13}"
        f"{'Public ECE':>13}"
        f"{'Police AUC':>13}"
        f"{'Police ECE':>13}"
    )

    lines = [
        "SENTENCE-LEVEL CONFIDENCE SUMMARY",
        "",
        header,
        "-" * len(header),
    ]

    for row in rows:
        lines.append(
            f"{row['Method']:<29}"
            f"{row['Monotonic']:>8}"
            f"{row['Public AUROC']:>13.3f}"
            f"{row['Public ECE']:>13.3f}"
            f"{row['Police AUROC']:>13.3f}"
            f"{row['Police ECE']:>13.3f}"
        )

    lines.extend(
        [
            "",
            (
                "Public results are averaged across Common Voice, EdAcc, "
                "English Dialects and Shetland."
            ),
            (
                "All methods are evaluated on their common set of scorable "
                "segments within each comparison."
            ),
            f"Police Scotland common-intersection N = {len(police_keys)}.",
        ]
    )

    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {txt_path}")


def main(paths=None):
    paths = paths or PaperReportFiles()
    build_paper_table(paths=paths)


if __name__ == "__main__":
    raise SystemExit("Use python -m asr_pipeline report with a separate --output-dir")
