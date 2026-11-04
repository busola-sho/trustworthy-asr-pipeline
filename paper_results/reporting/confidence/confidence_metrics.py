"""Calculate AUROC, calibration error and reliability bins from saved scores."""

import numpy as np
from sklearn.metrics import roc_auc_score

FLAG_THRESHOLD = 2  # locked convention: severity >= 2 -> flagged


def derive_flagged(severity):
    return 1 if severity >= FLAG_THRESHOLD else 0


def compute_auc(samples):
    """AUC-ROC: does confidence rank preserved-meaning segments above
    flagged ones? y=flagged (1=bad), score=(1-confidence) so a HIGH
    confidence maps to a LOW risk score - meaning high AUC = high
    confidence correctly predicts low flagged-probability."""
    y = np.array([derive_flagged(s["severity"]) for s in samples])
    risk_scores = np.array([1 - s["confidence"] for s in samples])

    if len(set(y)) < 2:
        return (
            None,
            "Cannot compute AUC - only one class present (all flagged or all preserved)",
        )

    auc = roc_auc_score(y, risk_scores)
    return auc, None


def compute_calibration(samples, n_bins=10):
    """Group confidence values into bins and calculate expected calibration error.

    Return bin midpoints, mean confidence, observed accuracy, counts and ECE."""
    confidences = np.array([s["confidence"] for s in samples])
    preserved = np.array(
        [1 - derive_flagged(s["severity"]) for s in samples]
    )  # 1 = preserved

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    n_total = len(samples)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_data.append(
                {
                    "bin_range": (lo, hi),
                    "mean_confidence": None,
                    "actual_preserved_rate": None,
                    "n_samples": 0,
                }
            )
            continue

        mean_conf = confidences[mask].mean()
        actual_rate = preserved[mask].mean()
        bin_data.append(
            {
                "bin_range": (lo, hi),
                "mean_confidence": round(mean_conf, 4),
                "actual_preserved_rate": round(actual_rate, 4),
                "n_samples": int(n_in_bin),
            }
        )

        ece += (n_in_bin / n_total) * abs(mean_conf - actual_rate)

    return bin_data, round(ece, 4)


def evaluate_method(samples, method_name="", dataset_name="", n_bins=10):
    """Runs both evaluations and prints a clean summary. samples must
    be a list of dicts with "confidence" (float 0-1) and "severity"
    (int 0-4) keys."""
    print(f"\n{'=' * 70}")
    print(f"  {method_name} - {dataset_name}  (N={len(samples)})")
    print(f"{'=' * 70}")

    auc, auc_error = compute_auc(samples)
    if auc_error:
        print(f"  AUC: {auc_error}")
    else:
        print(f"  AUC-ROC: {auc:.3f}  (0.5=no better than chance, 1.0=perfect ranking)")

    bin_data, ece = compute_calibration(samples, n_bins=n_bins)
    print(f"  Expected Calibration Error (ECE): {ece:.4f}  (0=perfectly calibrated)")
    print(f"\n  Reliability diagram data:")
    print(f"  {'Confidence bin':<18}{'Mean conf':>12}{'Actual preserved%':>20}{'N':>8}")
    for b in bin_data:
        if b["n_samples"] == 0:
            continue
        lo, hi = b["bin_range"]
        print(
            f"  [{lo:.1f}, {hi:.1f}){'':<10}{b['mean_confidence']:>12.3f}"
            f"{b['actual_preserved_rate'] * 100:>19.1f}%{b['n_samples']:>8}"
        )

    return {"auc": auc, "ece": ece, "bin_data": bin_data}
