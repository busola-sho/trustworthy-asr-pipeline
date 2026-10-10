"""
calibration_and_discrimination.py

Two evaluations for existing v1 confidence results, used AS-IS (no
rebuild) - both need per-sentence (confidence, severity) pairs, not
just the aggregate summary table.

1. DISCRIMINATION (AUC-ROC): does the confidence score rank
   meaning-preserved segments above meaning-altered ones? Doesn't
   care whether the confidence NUMBERS themselves are meaningful,
   only the ranking/ordering.

2. CALIBRATION (reliability diagram + Expected Calibration Error):
   if confidence says 0.9, is that segment actually correct ~90% of
   the time? A method can have great AUC (good ranking) while being
   badly calibrated (systematically over/under-confident) - these
   are genuinely different questions.

BINARY TARGET (matching the locked convention used throughout this
project): flagged = 1 if severity >= 2 (meaning-altering), flagged = 0
if severity < 2 (meaning-preserved).

CONFIDENCE ORIENTATION: confidence is treated as P(meaning preserved)
- i.e. higher confidence = more trustworthy = LESS likely to be
flagged. This matches how confscore/probscore were originally framed
("how confident are you this is correct"). AUC is therefore computed
with y=flagged, score=(1-confidence) - equivalently, a HIGH AUC here
means high confidence correctly predicts LOW flagged-probability.

Usage (as a library - adapt load_your_data() to your actual storage
format, then call evaluate_method()):

    from calibration_and_discrimination import evaluate_method

    # samples: list of dicts, each with "confidence" (0-1 float) and
    # "severity" (0-4 int)
    samples = load_your_data(...)
    evaluate_method(samples, method_name="confscore", dataset_name="commonvoice")
"""

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

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
        return None, "Cannot compute AUC - only one class present (all flagged or all preserved)"

    auc = roc_auc_score(y, risk_scores)
    return auc, None


def compute_calibration(samples, n_bins=10):
    """Reliability diagram data + Expected Calibration Error (ECE).
    Confidence is treated as predicted P(preserved) - bins samples by
    confidence, and for each bin checks: does the FRACTION actually
    preserved in that bin match the AVERAGE confidence claimed?

    Returns (bin_data, ece) - bin_data is a list of dicts with
    bin_range, mean_confidence, actual_preserved_rate, n_samples -
    ready for plotting a reliability diagram directly."""
    confidences = np.array([s["confidence"] for s in samples])
    preserved = np.array([1 - derive_flagged(s["severity"]) for s in samples])  # 1 = preserved

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
            bin_data.append({"bin_range": (lo, hi), "mean_confidence": None,
                             "actual_preserved_rate": None, "n_samples": 0})
            continue

        mean_conf = confidences[mask].mean()
        actual_rate = preserved[mask].mean()
        bin_data.append({"bin_range": (lo, hi), "mean_confidence": round(mean_conf, 4),
                         "actual_preserved_rate": round(actual_rate, 4), "n_samples": int(n_in_bin)})

        ece += (n_in_bin / n_total) * abs(mean_conf - actual_rate)

    return bin_data, round(ece, 4)


def evaluate_method(samples, method_name="", dataset_name="", n_bins=10):
    """Runs both evaluations and prints a clean summary. samples must
    be a list of dicts with "confidence" (float 0-1) and "severity"
    (int 0-4) keys."""
    print(f"\n{'='*70}")
    print(f"  {method_name} - {dataset_name}  (N={len(samples)})")
    print(f"{'='*70}")

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
        print(f"  [{lo:.1f}, {hi:.1f}){'':<10}{b['mean_confidence']:>12.3f}"
              f"{b['actual_preserved_rate']*100:>19.1f}%{b['n_samples']:>8}")

    return {"auc": auc, "ece": ece, "bin_data": bin_data}


if __name__ == "__main__":
    # smoke test with synthetic data
    import random
    random.seed(42)
    fake_samples = []
    for _ in range(200):
        severity = random.choices([0, 1, 2, 3, 4], weights=[30, 30, 25, 10, 5])[0]
        # confidence roughly (but not perfectly) tracks severity - good ranking, imperfect calibration
        base_conf = max(0.0, min(1.0, 0.95 - severity * 0.15 + random.gauss(0, 0.15)))
        fake_samples.append({"confidence": base_conf, "severity": severity})

    evaluate_method(fake_samples, method_name="confscore (synthetic test)", dataset_name="test")


# ── Real data loaders, matching the exact schema confirmed from ────────────────
# ── plot_confidence_by_severity.py / compute_crossmodel_agreement.py / ────────
# ── compute_acoustic_confidence.py ─────────────────────────────────────────────

def _label_path(dataset, variant):
    return f"results/sentence_confidence/sentence_labels_{dataset}_{variant}.json"


def load_labels(path):
    """Matches plot_confidence_by_severity.py's load_labels() exactly -
    keyed by (dataset_index, sent_pos)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return {
        (r["dataset_index"], r["sent_pos"]): r
        for r in d.get("rows", [])
        if r.get("severity") is not None
    }


def load_confidences(path, field):
    """Matches plot_confidence_by_severity.py's load_confidences()
    exactly - keyed by (dataset_index, sent_pos), reading the given
    field (usually "confidence", or "min_agreement" for crossmodel_min)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    conf_map = {}
    for samp in d.get("samples", []):
        if samp.get("skipped") or samp.get("error"):
            continue
        di = samp.get("dataset_index")
        sent_list = samp.get("sentence_confidences") or samp.get("sentences", [])
        for pos, sc in enumerate(sent_list):
            sent_pos = sc.get("sent_pos", pos)
            conf = sc.get(field)
            if conf is not None:
                conf_map[(di, sent_pos)] = conf
    return conf_map


def load_joined_samples(labels_path, conf_path, field="confidence"):
    """Joins labels + confidence by (dataset_index, sent_pos), returns
    a list of {"confidence": ..., "severity": ...} dicts ready for
    evaluate_method()."""
    labels = load_labels(labels_path)
    conf_map = load_confidences(conf_path, field)
    joined = []
    for key, label in labels.items():
        conf = conf_map.get(key)
        if conf is not None:
            joined.append({"confidence": conf, "severity": label["severity"]})
    return joined


def _combo_path(dataset, variant):
    split = "full" if dataset == "shetland" else "dev"
    return f"writeup_results/ensembles/naive_{variant}/naive_{variant}_{dataset}_gemma4sel_{split}.json"


def evaluate_all_methods(dataset, variant, n_bins=10):
    """Evaluates all 5 methods for one dataset+variant combination,
    matching the exact same file paths used in your existing table-
    generation scripts. Returns a dict of results, one per method."""
    method_paths = {
        variant: (_combo_path(dataset, variant), "confidence"),
        "crossmodel_mean": (f"results/sentence_confidence/crossmodel_agreement_{variant}_{dataset}.json", "confidence"),
        "crossmodel_min": (f"results/sentence_confidence/crossmodel_agreement_{variant}_{dataset}.json", "min_agreement"),
        "acoustic_mean": (f"results/sentence_confidence/acoustic_confidence_{variant}_{dataset}.json", "confidence"),
        "proxy_model": (f"results/sentence_confidence/proxy_model_{dataset}_{variant}.json", "confidence"),
    }

    labels_path = _label_path(dataset, variant)
    results = {}
    for method_name, (conf_path, field) in method_paths.items():
        samples = load_joined_samples(labels_path, conf_path, field)
        if not samples:
            print(f"\n  {method_name} - {dataset}: NO JOINED DATA (check paths: "
                  f"labels={labels_path}, conf={conf_path})")
            continue
        results[method_name] = evaluate_method(samples, method_name=method_name, dataset_name=dataset, n_bins=n_bins)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        # usage: python calibration_and_discrimination.py --real commonvoice probscore
        dataset = sys.argv[2] if len(sys.argv) > 2 else "commonvoice"
        variant = sys.argv[3] if len(sys.argv) > 3 else "probscore"
        evaluate_all_methods(dataset, variant)
