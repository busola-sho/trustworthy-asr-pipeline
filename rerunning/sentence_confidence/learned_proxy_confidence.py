"""
rerunning/sentence_confidence/learned_proxy_confidence.py

Method 4: Learned SEVERITY proxy (not "confidence" - see orientation
note below). Ridge-regression predictor combining Methods 1-3's
confidence signals as features, predicting segment-level SEVERITY
(continuous, 0-4). Framework motivated by the "LPP" paper, adapted
with LODO specific to this project.

TRAIN/EVAL SPLIT (resolved after review flagged this as unlocked):
LODO training happens on DEV (not test) - matching the historical
design, and consistent with the reasoning that dev-split LABELS
remain usable for training even though dev-split ASR QUALITY
evaluation is contaminated by fine-tuning. Method 4 isn't evaluating
ASR quality; it's learning to combine independent confidence SIGNALS,
so training on dev's (feature, severity) pairs doesn't leak test
information the way evaluating on dev would. Evaluation then happens
on TEST - the same clean, confirmatory split Methods 1-3 are judged
on - so Method 4 is finally held to the identical standard as every
other method, not a looser one. Shetland remains the final true
holdout: trained on all 3 dev sets combined, evaluated on Shetland,
unchanged from the historical design.

FEATURE SCALING: Ridge's regularization is scale-dependent, and the 4
features are NOT on comparable scales (Method 2 is z-scored, roughly
centered at 0; Methods 1/3 are roughly 0-1 or 0-100 bounded). Fixed
via sklearn Pipeline(StandardScaler(), Ridge()) - the scaler is fit
ONLY on each fold's own training data, never on test/Shetland, so no
information leaks across the fold boundary.

ORIENTATION: this model predicts SEVERITY (higher = more severe/less
trustworthy) - the OPPOSITE direction from Methods 1-3, which predict
CONFIDENCE (higher = more trustworthy). Described here as a "learned
severity proxy", not folded into the same "higher=better" framing as
the other 3 methods. A derived, confidence-oriented score is also
saved (C_proxy = 1 - clip(y_hat, 0, 4)/4) for direct comparability
in cross-method tables (e.g. AUC alongside Methods 1-3), but the raw
severity prediction is the primary, correctly-labeled output.

Per-variant design (matching historical precedent): trained SEPARATELY
per verbalized-confidence variant (confscore/confprobscore/probscore).

Usage:
    python learned_proxy_confidence.py --variant confscore
"""

import json
import os
import argparse

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

VERBALIZED_DIR = "writeup_results/sentence_confidence/verbalized_confidence"
MODEL_INTERNAL_DIR = "writeup_results/sentence_confidence/model_internal_confidence"
CROSSMODEL_DIR = "writeup_results/sentence_confidence/cross_model_agreement"
OUTPUT_DIR = "writeup_results/sentence_confidence/learned_proxy"

IN_DOMAIN_DATASETS = ["commonvoice", "edacc", "english_dialects"]
VARIANTS = ["confscore", "confprobscore", "probscore"]

FEATURE_NAMES = ["verbalized_score", "model_internal_confidence", "crossmodel_mean", "crossmodel_min"]


def load_method_file(directory, filename_pattern, dataset, split):
    path = os.path.join(directory, filename_pattern.format(dataset=dataset, split=split))
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def build_feature_matrix(dataset, split, variant):
    verbalized_pattern = f"verbalized_{variant}_{{dataset}}_{{split}}.json"
    verbalized = load_method_file(VERBALIZED_DIR, verbalized_pattern, dataset, split)
    model_internal = load_method_file(MODEL_INTERNAL_DIR, "model_internal_{dataset}_{split}.json", dataset, split)
    crossmodel = load_method_file(CROSSMODEL_DIR, "cross_model_agreement_{dataset}_{split}.json", dataset, split)

    if not all([verbalized, model_internal, crossmodel]):
        missing = [name for name, d in [("verbalized", verbalized), ("model_internal", model_internal),
                                        ("crossmodel", crossmodel)] if d is None]
        print(f"  WARNING: missing method output(s) for {dataset}/{split}: {missing} - skipping")
        return None, None, None, None

    verbalized_by_idx = {t["dataset_index"]: t["segments"] for t in verbalized["transcripts"]}
    internal_by_idx = {t["dataset_index"]: t["segments"] for t in model_internal["transcripts"]}
    cross_by_idx = {t["dataset_index"]: t["segments"] for t in crossmodel["transcripts"]}

    X_rows, y_rows, flagged_rows, keys = [], [], [], []

    common_indices = set(verbalized_by_idx) & set(internal_by_idx) & set(cross_by_idx)
    for idx in common_indices:
        v_segs = verbalized_by_idx[idx]
        i_segs = internal_by_idx[idx]
        c_segs = cross_by_idx[idx]

        n = min(len(v_segs), len(i_segs), len(c_segs))
        if len(v_segs) != len(i_segs) or len(i_segs) != len(c_segs):
            print(f"  WARNING: segment count mismatch at dataset_index={idx} "
                  f"(verbalized={len(v_segs)}, internal={len(i_segs)}, cross={len(c_segs)}) - "
                  f"using first {n} common positions only")

        for pos in range(n):
            v, i, c = v_segs[pos], i_segs[pos], c_segs[pos]
            severity = i.get("severity")
            if severity is None:
                continue

            verbalized_score = v.get("verbalized_score")
            internal_score = i.get("model_internal_confidence")
            cross_mean = c.get("crossmodel_mean")
            cross_min = c.get("crossmodel_min")

            if any(f is None for f in [verbalized_score, internal_score, cross_mean, cross_min]):
                continue

            X_rows.append([verbalized_score, internal_score, cross_mean, cross_min])
            y_rows.append(severity)
            flagged_rows.append(i.get("flagged"))
            keys.append([idx, pos])  # (dataset_index, position) - lets downstream analysis
                                      # join proxy_model's predictions back to the exact
                                      # segment they came from, same as the other 3 methods

    if not X_rows:
        return None, None, None, None

    return np.array(X_rows), np.array(y_rows), np.array(flagged_rows), keys


def severity_to_confidence(y_pred):
    """Derived, confidence-oriented view of the severity prediction -
    for direct comparability with Methods 1-3 in cross-method tables
    only. The primary output remains the raw severity prediction."""
    clipped = np.clip(y_pred, 0, 4)
    return 1 - (clipped / 4)


def run_lodo(variant, train_split="dev", eval_split="test"):
    print(f"\n{'='*70}")
    print(f"Method 4: Learned Severity Proxy - variant={variant}")
    print(f"Training split: {train_split}  |  Evaluation split: {eval_split}")
    print(f"{'='*70}")

    train_data = {}
    eval_data = {}
    for dataset in IN_DOMAIN_DATASETS:
        X_tr, y_tr, _, _ = build_feature_matrix(dataset, train_split, variant)
        X_ev, y_ev, flagged_ev, keys_ev = build_feature_matrix(dataset, eval_split, variant)
        if X_tr is not None:
            train_data[dataset] = (X_tr, y_tr)
            print(f"  {dataset} [{train_split}]: {len(X_tr)} usable segments (training pool)")
        if X_ev is not None:
            eval_data[dataset] = (X_ev, y_ev, flagged_ev, keys_ev)
            print(f"  {dataset} [{eval_split}]: {len(X_ev)} usable segments (evaluation)")

    results = {"variant": variant, "train_split": train_split, "eval_split": eval_split,
               "lodo_folds": {}, "shetland": None}

    # LODO: train on 2 of 3 in-domain TRAIN-split datasets, evaluate on
    # the 3rd dataset's EVAL-split (test) - not the same dataset's
    # train split, so a held-out fold never touches its own training data.
    for held_out in IN_DOMAIN_DATASETS:
        train_datasets = [d for d in IN_DOMAIN_DATASETS if d != held_out]
        if not all(d in train_data for d in train_datasets) or held_out not in eval_data:
            print(f"\n  Skipping fold (held_out={held_out}) - missing data")
            continue

        X_train = np.concatenate([train_data[d][0] for d in train_datasets])
        y_train = np.concatenate([train_data[d][1] for d in train_datasets])
        X_test, y_test, flagged_test, keys_test = eval_data[held_out]

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        confidence_pred = severity_to_confidence(y_pred)

        r, p_value = pearsonr(y_pred, y_test)
        ridge_step = model.named_steps["ridge"]
        coefs = dict(zip(FEATURE_NAMES, ridge_step.coef_.tolist()))
        scaler = model.named_steps["standardscaler"]

        print(f"\n  Held out: {held_out}")
        print(f"    Trained on [{train_split}]: {train_datasets}  (N={len(X_train)})")
        print(f"    Evaluated on [{eval_split}]: {held_out}  (N={len(X_test)})")
        print(f"    Pearson r={r:.3f}  p={p_value:.4f}")
        print(f"    Coefficients (standardized-feature scale): {coefs}")

        results["lodo_folds"][held_out] = {
            "train_datasets": train_datasets,
            "train_split": train_split,
            "eval_split": eval_split,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "pearson_r": r,
            "p_value": p_value,
            "coefficients": coefs,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "intercept": ridge_step.intercept_,
            "severity_predictions": y_pred.tolist(),
            "confidence_predictions": confidence_pred.tolist(),
            "actual_severity": y_test.tolist(),
            "actual_flagged": flagged_test.tolist(),
            "keys": keys_test,  # [(dataset_index, position), ...] - same order as the
                                 # predictions above, lets downstream analysis join proxy_model
                                 # back to the exact segments it scored, same as the other 3 methods
        }

    # Final: train on ALL 3 in-domain TRAIN-split (dev) sets, evaluate
    # on Shetland (true holdout, never used in training at all)
    if len(train_data) == 3:
        X_all = np.concatenate([train_data[d][0] for d in IN_DOMAIN_DATASETS])
        y_all = np.concatenate([train_data[d][1] for d in IN_DOMAIN_DATASETS])

        X_shetland, y_shetland, flagged_shetland, keys_shetland = build_feature_matrix("shetland", "full", variant)
        if X_shetland is not None:
            model_final = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model_final.fit(X_all, y_all)
            y_pred_shetland = model_final.predict(X_shetland)
            confidence_pred_shetland = severity_to_confidence(y_pred_shetland)
            r_shetland, p_shetland = pearsonr(y_pred_shetland, y_shetland)
            ridge_final = model_final.named_steps["ridge"]
            coefs_shetland = dict(zip(FEATURE_NAMES, ridge_final.coef_.tolist()))
            scaler_final = model_final.named_steps["standardscaler"]

            print(f"\n  SHETLAND (true holdout, trained on all 3 in-domain [{train_split}] sets):")
            print(f"    N_train={len(X_all)}  N_shetland={len(X_shetland)}")
            print(f"    Pearson r={r_shetland:.3f}  p={p_shetland:.4f}")
            print(f"    Coefficients: {coefs_shetland}")

            results["shetland"] = {
                "n_train": len(X_all),
                "n_shetland": len(X_shetland),
                "pearson_r": r_shetland,
                "p_value": p_shetland,
                "coefficients": coefs_shetland,
                "scaler_mean": scaler_final.mean_.tolist(),
                "scaler_scale": scaler_final.scale_.tolist(),
                "intercept": ridge_final.intercept_,
                "severity_predictions": y_pred_shetland.tolist(),
                "confidence_predictions": confidence_pred_shetland.tolist(),
                "actual_severity": y_shetland.tolist(),
                "actual_flagged": flagged_shetland.tolist(),
                "keys": keys_shetland,
            }
        else:
            print("\n  SHETLAND: no usable data - skipping final holdout evaluation")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"learned_severity_proxy_{variant}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--train-split", default="dev", choices=["dev"])
    parser.add_argument("--eval-split", default="test", choices=["test"])
    args = parser.parse_args()

    run_lodo(args.variant, train_split=args.train_split, eval_split=args.eval_split)


if __name__ == "__main__":
    main()
