"""
compute_ps_auc_ece.py

Computes AUC-ROC and ECE for Police Scotland's returned results,
reusing calibration_and_discrimination.py's core functions (same
binary target convention, same AUC orientation). Reads from the
public per-row method files (results_v2/sentence_confidence/public/
method/*.json) - these already have exactly what's needed (confidence,
severity), no raw text required.

model_internal gets Platt-scaled using your ALREADY-FROZEN (a, b) from
the dissertation's own pooled dev data (model_internal_platt_params.json)
- NOT refit on PS data. PS's pipeline has no genuine dev/test split to
fit calibration on without leakage, so reusing the frozen dissertation
parameters is the consistent choice, matching how Method 2's pooled
normalization stats and Method 4's Ridge coefficients were already
transferred to PS as frozen values, never refit there.

Usage:
    python compute_ps_auc_ece.py --input_dir results_v2/sentence_confidence/public/method
"""

import json
import argparse
from pathlib import Path

from calibration_and_discrimination import evaluate_method
from fit_platt_scaling import apply_platt, load_platt_params

PLATT_PARAMS_PATH = "writeup_results/sentence_confidence/model_internal_platt_params.json"


def load_ps_method_file(path):
    """PS's evaluate_sentence_conf_and_labels.py saves {"summary":...,
    "rows": [{"confidence":..., "severity":..., "mar_verdict":..., ...}]}
    - already exactly the shape evaluate_method() needs."""
    data = json.load(open(path))
    rows = data.get("rows", [])
    samples = [{"confidence": r["confidence"], "severity": r["severity"]}
              for r in rows if r.get("confidence") is not None and r.get("severity") is not None]
    method_name = data.get("summary", {}).get("method", Path(path).stem)
    return samples, method_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="results_v2/sentence_confidence/public/method")
    args = parser.parse_args()

    platt_a, platt_b = load_platt_params(PLATT_PARAMS_PATH)
    print(f"Using FROZEN Platt params (fit on your own pooled dev data): a={platt_a:.4f}, b={platt_b:.4f}\n")

    for path in sorted(Path(args.input_dir).glob("*.json")):
        samples, method_name = load_ps_method_file(path)
        if not samples:
            print(f"  {path.name}: no usable rows, skipping")
            continue

        if "model_internal" in method_name.lower():
            samples = apply_platt(samples, platt_a, platt_b)

        evaluate_method(samples, method_name=method_name, dataset_name="police_scotland")


if __name__ == "__main__":
    main()
