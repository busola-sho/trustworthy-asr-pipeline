"""Read/apply frozen model-internal calibration for saved paper reports."""

import json
import numpy as np


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


def load_platt_params(path):
    with open(path) as f:
        d = json.load(f)
    return d["a"], d["b"]
