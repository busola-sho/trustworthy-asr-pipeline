"""Post-hoc calibration of dissertation-era verbalised confidence.

Fits candidate calibrators on pooled public development data only, selects
between Platt scaling and isotonic regression using grouped cross-validation,
then freezes the selected calibrator and evaluates it on public test data,
Shetland, and Police Scotland.

Target: 1 = meaning preserved (severity 0--1), 0 = meaning altered (2--4).

Usage:
    python calibrate_verbalized_confidence.py
"""

import csv
import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

OLD_ROOT = PACKAGE_ROOT / "data" / "verbalized"
POLICE_PATH = (
    PACKAGE_ROOT
    / "data"
    / "police_scotland"
    / "method_confscore_ps.json"
)
OUTPUT_DIR = PACKAGE_ROOT / "calibration"

PUBLIC_DATASETS = ["commonvoice", "edacc", "english_dialects"]
DISPLAY = {
    "commonvoice_test": "Common Voice",
    "edacc_test": "EdAcc",
    "english_dialects_test": "English Dialects",
    "shetland_full": "Shetland",
    "police_scotland": "Police Scotland",
}

N_BINS = 10
RANDOM_SEED = 42


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalise_confidence(value):
    if value is None:
        return None
    value = float(value)
    if value > 1.0:
        value /= 100.0
    return float(np.clip(value, 0.0, 1.0))


def load_verbalized(path, dataset_name):
    """Read both old transcript/segment JSON and flattened rows JSON."""
    data = load_json(path)
    records = []

    if isinstance(data.get("rows"), list):
        for row in data["rows"]:
            confidence = normalise_confidence(row.get("confidence"))
            severity = row.get("severity")
            if confidence is None or severity is None:
                continue
            sample_id = str(row.get("dataset_index"))
            sent_pos = row.get("sent_pos")
            records.append({
                "dataset": dataset_name,
                "sample_id": sample_id,
                "segment_id": sent_pos,
                "confidence": confidence,
                "preserved": int(float(severity) < 2),
                "severity": float(severity),
            })
        return records

    transcripts = data.get("transcripts", data.get("samples", []))
    for transcript_pos, transcript in enumerate(transcripts):
        sample_id = str(transcript.get("dataset_index", transcript_pos))
        segments = transcript.get("segments", transcript.get("sentences", []))
        for segment_pos, segment in enumerate(segments):
            confidence = normalise_confidence(
                segment.get("verbalized_score", segment.get("confidence"))
            )
            severity = segment.get("severity")
            if confidence is None or severity is None:
                continue
            records.append({
                "dataset": dataset_name,
                "sample_id": sample_id,
                "segment_id": segment.get("sent_pos", segment_pos),
                "confidence": confidence,
                "preserved": int(float(severity) < 2),
                "severity": float(severity),
            })
    return records


class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(random_state=RANDOM_SEED)

    def fit(self, x, y):
        self.model.fit(np.asarray(x).reshape(-1, 1), y)
        return self

    def predict(self, x):
        return self.model.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )

    def fit(self, x, y):
        self.model.fit(np.asarray(x), y)
        return self

    def predict(self, x):
        return self.model.predict(np.asarray(x))


CALIBRATORS = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}


def expected_calibration_error(y, probabilities, n_bins=N_BINS):
    y = np.asarray(y, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0

    for index in range(n_bins):
        lo, hi = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (probabilities >= lo) & (probabilities <= hi)
        else:
            mask = (probabilities >= lo) & (probabilities < hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / total) * abs(
            probabilities[mask].mean() - y[mask].mean()
        )
    return float(ece)


def metrics(y, probabilities):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    auroc = roc_auc_score(y, probabilities) if len(np.unique(y)) == 2 else None
    return {
        "n": int(len(y)),
        "preservation_rate": float(y.mean()),
        "auroc": None if auroc is None else float(auroc),
        "ece": expected_calibration_error(y, probabilities),
        "brier": float(brier_score_loss(y, probabilities)),
    }


def arrays(records):
    x = np.asarray([row["confidence"] for row in records], dtype=float)
    y = np.asarray([row["preserved"] for row in records], dtype=int)
    groups = np.asarray(
        [f'{row["dataset"]}::{row["sample_id"]}' for row in records]
    )
    return x, y, groups


def cross_validated_candidate(name, x, y, groups):
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    predictions = np.full(len(y), np.nan, dtype=float)

    for train_indices, valid_indices in splitter.split(x, y, groups):
        calibrator = CALIBRATORS[name]().fit(x[train_indices], y[train_indices])
        predictions[valid_indices] = calibrator.predict(x[valid_indices])

    if np.isnan(predictions).any():
        raise RuntimeError(f"Missing cross-validation predictions for {name}")
    return predictions, metrics(y, predictions)


def reliability_points(y, probabilities, n_bins=5):
    """Five fixed-width bins, omitting bins with inadequate support."""
    y = np.asarray(y, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    min_count = max(5, int(np.ceil(0.01 * len(y))))
    mean_confidence, observed, counts = [], [], []

    for index in range(n_bins):
        lo, hi = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (probabilities >= lo) & (probabilities <= hi)
        else:
            mask = (probabilities >= lo) & (probabilities < hi)

        count = int(mask.sum())
        if count < min_count:
            continue

        mean_confidence.append(float(probabilities[mask].mean()))
        observed.append(float(y[mask].mean()))
        counts.append(count)

    return mean_confidence, observed, counts


def plot_reliability(eval_outputs, selected_name):
    public_keys = [
        "commonvoice_test",
        "edacc_test",
        "english_dialects_test",
    ]
    plot_outputs = {
        "public_pooled": {
            "y": np.concatenate([eval_outputs[key]["y"] for key in public_keys]),
            "raw": np.concatenate([eval_outputs[key]["raw"] for key in public_keys]),
            "calibrated": np.concatenate(
                [eval_outputs[key]["calibrated"] for key in public_keys]
            ),
        },
        "shetland_full": eval_outputs["shetland_full"],
        "police_scotland": eval_outputs["police_scotland"],
    }
    plot_display = {
        "public_pooled": "Public test sets (pooled)",
        "shetland_full": "Shetland",
        "police_scotland": "Police Scotland",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)

    for axis, (dataset, output) in zip(axes, plot_outputs.items()):
        y = output["y"]
        raw = output["raw"]
        calibrated = output["calibrated"]
        raw_x, raw_y, _ = reliability_points(y, raw)
        cal_x, cal_y, _ = reliability_points(y, calibrated)

        axis.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1.3)
        axis.plot(raw_x, raw_y, "o-", label="Uncalibrated", color="#c0392b")
        axis.plot(cal_x, cal_y, "o-", label="Calibrated", color="#2471a3")
        axis.set_title(plot_display[dataset], fontsize=12, fontweight="bold")
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)

    fig.supxlabel("Predicted probability of meaning preservation", fontsize=12)
    fig.supylabel("Observed meaning-preservation rate", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(
        f"Verbalised confidence before and after {selected_name} calibration",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.03, 0.08, 1, 0.84])
    path = OUTPUT_DIR / "verbalized_confscore_reliability_calibrated.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    development = []
    for dataset in PUBLIC_DATASETS:
        path = OLD_ROOT / f"verbalized_confscore_{dataset}_dev.json"
        development.extend(load_verbalized(path, f"{dataset}_dev"))

    if not development:
        raise SystemExit("No development records were loaded.")

    x_dev, y_dev, groups_dev = arrays(development)
    raw_dev_metrics = metrics(y_dev, x_dev)

    cv_results = {}
    for name in CALIBRATORS:
        _, candidate_metrics = cross_validated_candidate(
            name, x_dev, y_dev, groups_dev
        )
        cv_results[name] = candidate_metrics

    # Brier score is a proper scoring rule and is the primary selection metric.
    # ECE is used only as a deterministic tie-breaker.
    selected_name = min(
        cv_results,
        key=lambda name: (cv_results[name]["brier"], cv_results[name]["ece"]),
    )
    calibrator = CALIBRATORS[selected_name]().fit(x_dev, y_dev)

    evaluation_sets = {}
    for dataset in PUBLIC_DATASETS:
        key = f"{dataset}_test"
        path = OLD_ROOT / f"verbalized_confscore_{dataset}_test.json"
        evaluation_sets[key] = load_verbalized(path, key)

    evaluation_sets["shetland_full"] = load_verbalized(
        OLD_ROOT / "verbalized_confscore_shetland_full.json",
        "shetland_full",
    )
    evaluation_sets["police_scotland"] = load_verbalized(
        POLICE_PATH,
        "police_scotland",
    )

    rows = []
    eval_outputs = {}
    for dataset, records in evaluation_sets.items():
        x, y, _ = arrays(records)
        calibrated = np.clip(calibrator.predict(x), 0.0, 1.0)
        raw_metrics = metrics(y, x)
        calibrated_metrics = metrics(y, calibrated)
        eval_outputs[dataset] = {
            "y": y,
            "raw": x,
            "calibrated": calibrated,
        }
        rows.append({
            "dataset": dataset,
            "n": len(y),
            "raw_auroc": raw_metrics["auroc"],
            "calibrated_auroc": calibrated_metrics["auroc"],
            "raw_ece": raw_metrics["ece"],
            "calibrated_ece": calibrated_metrics["ece"],
            "raw_brier": raw_metrics["brier"],
            "calibrated_brier": calibrated_metrics["brier"],
        })

    payload = {
        "target": "meaning_preserved (severity < 2)",
        "training_data": "pooled public development sets only",
        "selection_metric": "grouped-CV Brier score; ECE tie-breaker",
        "n_development_segments": len(y_dev),
        "uncalibrated_development_metrics": raw_dev_metrics,
        "candidate_cross_validation": cv_results,
        "selected_calibrator": selected_name,
        "evaluation": rows,
    }

    with (OUTPUT_DIR / "calibration_results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with (OUTPUT_DIR / "calibration_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (OUTPUT_DIR / "verbalized_confscore_calibrator.pkl").open("wb") as f:
        pickle.dump(calibrator, f)

    figure_path = plot_reliability(eval_outputs, selected_name)

    print(f"Development segments: {len(y_dev)}")
    print("Grouped cross-validation:")
    for name, result in cv_results.items():
        print(
            f"  {name:<8} Brier={result['brier']:.4f} "
            f"ECE={result['ece']:.4f} AUROC={result['auroc']:.4f}"
        )
    print(f"Selected calibrator: {selected_name}\n")
    print(
        f"{'Dataset':<26} {'N':>6} {'AUROC raw/cal':>18} "
        f"{'ECE raw/cal':>18} {'Brier raw/cal':>18}"
    )
    print("-" * 92)
    for row in rows:
        print(
            f"{DISPLAY[row['dataset']]:<26} {row['n']:>6} "
            f"{row['raw_auroc']:.3f}/{row['calibrated_auroc']:.3f} "
            f"{row['raw_ece']:.3f}/{row['calibrated_ece']:.3f} "
            f"{row['raw_brier']:.3f}/{row['calibrated_brier']:.3f}"
        )
    print(f"\nSaved results to: {OUTPUT_DIR}")
    print(f"Saved reliability figure: {figure_path}")


if __name__ == "__main__":
    main()
