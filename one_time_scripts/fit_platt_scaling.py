"""
fit_platt_scaling.py

Fits Platt scaling (logistic regression) to convert model_internal's
raw z-score into a genuine calibrated probability:
    P(preserved | z) = sigmoid(a*z + b)
Fit on POOLED dev data across the 3 in-domain datasets (same pooling
convention as Method 2's own normalization stats), frozen and applied
unchanged to test/Shetland - matches the dev/test discipline used
throughout this project for every other fitted parameter.

Only model_internal needs this - every other method's confidence is
already genuinely bounded [0,1] by construction, so this bug is
specific to the one method using an unbounded z-score as "confidence".

AUC/Spearman/monotonicity are UNAFFECTED by this fix (any monotonic
transform with a>0 preserves rank order) - only ECE/calibration
numbers for model_internal need recomputing with this applied.

Usage:
    from fit_platt_scaling import fit_platt, apply_platt

    a, b = fit_platt(dev_samples)  # dev_samples: [{"confidence": z, "severity": s}, ...]
    calibrated_samples = apply_platt(test_samples, a, b)
    # now safe to pass calibrated_samples into evaluate_method()
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression

FLAG_THRESHOLD = 2


def derive_preserved(severity):
    return 1 if severity < FLAG_THRESHOLD else 0


def fit_platt(dev_samples):
    """Fits P(preserved|z) = sigmoid(a*z + b) via logistic regression
    on dev data. Returns (a, b) - freeze these, never refit on
    test/Shetland."""
    z = np.array([[s["confidence"]] for s in dev_samples])
    y = np.array([derive_preserved(s["severity"]) for s in dev_samples])

    if len(set(y.tolist())) < 2:
        raise ValueError("Cannot fit Platt scaling - dev data has only one class")

    model = LogisticRegression()
    model.fit(z, y)
    a = float(model.coef_[0][0])
    b = float(model.intercept_[0])
    return a, b


def apply_platt(samples, a, b):
    """Applies the FROZEN (a, b) to a new set of samples - returns a
    new list with "confidence" replaced by the calibrated probability.
    Original z-score is kept under "raw_z_score" for reference."""
    calibrated = []
    for s in samples:
        z = s["confidence"]
        p = 1 / (1 + np.exp(-(a * z + b)))
        calibrated.append({**s, "confidence": float(p), "raw_z_score": z})
    return calibrated


def save_platt_params(a, b, path):
    with open(path, "w") as f:
        json.dump({"a": a, "b": b}, f, indent=2)


def load_platt_params(path):
    with open(path) as f:
        d = json.load(f)
    return d["a"], d["b"]


if __name__ == "__main__":
    # smoke test: does fitting Platt scaling on a KNOWN synthetic
    # relationship recover something close to the true relationship,
    # and does applying it produce properly calibrated output?
    import random
    random.seed(7)

    # synthetic ground truth: P(preserved|z) genuinely follows
    # sigmoid(1.5*z + 0.3) - check if fit_platt recovers something close
    true_a, true_b = 1.5, 0.3
    dev_samples = []
    for _ in range(2000):
        z = random.gauss(-0.2, 0.6)  # matches real model_internal's distribution shape
        p_true = 1 / (1 + np.exp(-(true_a * z + true_b)))
        preserved = 1 if random.random() < p_true else 0
        severity = 0 if preserved else 3  # crude severity stand-in matching preserved/flagged
        dev_samples.append({"confidence": z, "severity": severity})

    a, b = fit_platt(dev_samples)
    print(f"True (a,b): ({true_a}, {true_b})")
    print(f"Fitted (a,b): ({a:.3f}, {b:.3f})")
    assert abs(a - true_a) < 0.3 and abs(b - true_b) < 0.3, "fit should recover something close to the true relationship"
    print("PASS - Platt scaling correctly recovers the underlying relationship\n")

    # confirm calibrated output is genuinely bounded [0,1]
    test_samples = [{"confidence": random.gauss(-0.2, 0.6), "severity": random.choice([0,1,2,3,4])}
                    for _ in range(100)]
    calibrated = apply_platt(test_samples, a, b)
    confs = [s["confidence"] for s in calibrated]
    assert all(0 <= c <= 1 for c in confs), "all calibrated values must be in [0,1]"
    print(f"PASS - all {len(confs)} calibrated values correctly bounded in [0,1]")
    print(f"  range: [{min(confs):.3f}, {max(confs):.3f}]")
