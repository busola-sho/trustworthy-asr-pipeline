"""
rerunning/sentence_confidence/learned_proxy_confidence.py

Method 4: Learned severity proxy.

Combines the three sentence-confidence families:
  1. verbalized confidence,
  2. model-internal ASR confidence,
  3. cross-model agreement,

and uses Ridge Regression to predict sentence-level meaning-alteration
severity on the 0-4 scale.

The model is trained on development data and evaluated on held-out test
data using leave-one-dataset-out evaluation. Shetland remains the final
out-of-domain holdout.

Important robustness change:
- The previous version joined method outputs by list position and silently
  truncated to the shortest list when segment counts differed.
- This version joins segments using a stable key built from normalized
  segment text plus its occurrence number within the transcript.
- If a method is missing a segment, that segment is skipped explicitly
  rather than allowing later positions to become misaligned.

Usage:
    python rerunning/sentence_confidence/learned_proxy_confidence.py \
        --variant confscore
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERBALIZED_DIR = "writeup_results/sentence_confidence/verbalized_confidence"
MODEL_INTERNAL_DIR = "writeup_results/sentence_confidence/model_internal_confidence"
CROSSMODEL_DIR = "writeup_results/sentence_confidence/cross_model_agreement"
OUTPUT_DIR = "writeup_results/sentence_confidence/learned_proxy"

IN_DOMAIN_DATASETS = [
    "commonvoice",
    "edacc",
    "english_dialects",
]

VARIANTS = [
    "confscore",
    "confprobscore",
    "probscore",
]

FEATURE_NAMES = [
    "verbalized_score",
    "model_internal_confidence",
    "crossmodel_mean",
    "crossmodel_min",
]


def load_method_file(directory, filename_pattern, dataset, split):
    path = os.path.join(
        directory,
        filename_pattern.format(
            dataset=dataset,
            split=split,
        ),
    )

    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_segment_text(text):
    """
    Normalize only for joining method outputs.

    This does not alter any confidence feature or severity label.
    """
    tokens = re.findall(r"[\w']+", (text or "").lower())
    return " ".join(tokens)


def build_segment_map(segments):
    """
    Build a stable per-transcript mapping using:

        (normalized segment text, occurrence index)

    Occurrence index disambiguates repeated identical sentence-like segments.

    Returns:
        {
            (normalized_text, occurrence_index): {
                "segment": original_segment_dict,
                "position": original_list_position,
            }
        }
    """
    occurrence_counts = defaultdict(int)
    mapping = {}

    for position, segment in enumerate(segments):
        text = normalize_segment_text(segment.get("segment", ""))
        occurrence_index = occurrence_counts[text]
        occurrence_counts[text] += 1

        key = (text, occurrence_index)

        mapping[key] = {
            "segment": segment,
            "position": position,
        }

    return mapping


def build_feature_matrix(dataset, split, variant):
    verbalized_pattern = (
        f"verbalized_{variant}_{{dataset}}_{{split}}.json"
    )

    verbalized = load_method_file(
        VERBALIZED_DIR,
        verbalized_pattern,
        dataset,
        split,
    )

    model_internal = load_method_file(
        MODEL_INTERNAL_DIR,
        "model_internal_{dataset}_{split}.json",
        dataset,
        split,
    )

    crossmodel = load_method_file(
        CROSSMODEL_DIR,
        "cross_model_agreement_{dataset}_{split}.json",
        dataset,
        split,
    )

    method_files = {
        "verbalized": verbalized,
        "model_internal": model_internal,
        "crossmodel": crossmodel,
    }

    missing = [
        name
        for name, data in method_files.items()
        if data is None
    ]

    if missing:
        print(
            f"  WARNING: missing method output(s) for "
            f"{dataset}/{split}: {missing} - skipping"
        )
        return None, None, None, None

    verbalized_by_idx = {
        t["dataset_index"]: t["segments"]
        for t in verbalized.get("transcripts", [])
    }

    internal_by_idx = {
        t["dataset_index"]: t["segments"]
        for t in model_internal.get("transcripts", [])
    }

    cross_by_idx = {
        t["dataset_index"]: t["segments"]
        for t in crossmodel.get("transcripts", [])
    }

    common_indices = (
        set(verbalized_by_idx)
        & set(internal_by_idx)
        & set(cross_by_idx)
    )

    X_rows = []
    y_rows = []
    flagged_rows = []
    keys = []

    n_missing_join = 0
    n_incomplete_features = 0

    for idx in sorted(common_indices):
        v_map = build_segment_map(
            verbalized_by_idx[idx]
        )
        i_map = build_segment_map(
            internal_by_idx[idx]
        )
        c_map = build_segment_map(
            cross_by_idx[idx]
        )

        all_keys = (
            set(v_map)
            | set(i_map)
            | set(c_map)
        )

        common_segment_keys = (
            set(v_map)
            & set(i_map)
            & set(c_map)
        )

        missing_here = len(all_keys - common_segment_keys)

        if missing_here:
            n_missing_join += missing_here
            print(
                f"  WARNING: dataset_index={idx} has "
                f"{missing_here} segment key(s) missing from at "
                f"least one confidence method; skipping those keys"
            )

        # Preserve the original verbalized ordering for deterministic output.
        ordered_keys = sorted(
            common_segment_keys,
            key=lambda key: v_map[key]["position"],
        )

        for segment_key in ordered_keys:
            v = v_map[segment_key]["segment"]
            i = i_map[segment_key]["segment"]
            c = c_map[segment_key]["segment"]

            severity = i.get("severity")

            if severity is None:
                continue

            verbalized_score = v.get(
                "verbalized_score"
            )
            internal_score = i.get(
                "model_internal_confidence"
            )
            cross_mean = c.get(
                "crossmodel_mean"
            )
            cross_min = c.get(
                "crossmodel_min"
            )

            features = [
                verbalized_score,
                internal_score,
                cross_mean,
                cross_min,
            ]

            if any(value is None for value in features):
                n_incomplete_features += 1
                continue

            X_rows.append(features)
            y_rows.append(severity)
            flagged_rows.append(
                i.get("flagged")
            )

            keys.append(
                {
                    "dataset_index": idx,
                    "segment_position": (
                        v_map[segment_key]["position"]
                    ),
                    "segment_text": (
                        v.get("segment", "")
                    ),
                    "segment_occurrence": (
                        segment_key[1]
                    ),
                }
            )

    if n_missing_join:
        print(
            f"  Join audit: {n_missing_join} unmatched segment "
            f"key(s) skipped for {dataset}/{split}"
        )

    if n_incomplete_features:
        print(
            f"  Feature audit: {n_incomplete_features} aligned "
            f"segment(s) skipped because at least one confidence "
            f"feature was None"
        )

    if not X_rows:
        return None, None, None, None

    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
        np.asarray(flagged_rows),
        keys,
    )


def severity_to_confidence(y_pred):
    """
    Convert predicted severity to a confidence-oriented score:

        C_proxy = 1 - clip(severity, 0, 4) / 4

    Higher confidence therefore corresponds to lower predicted severity.
    """
    clipped = np.clip(y_pred, 0, 4)
    return 1 - (clipped / 4)


def fit_proxy(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_proxy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    confidence_pred = severity_to_confidence(y_pred)

    r, p_value = pearsonr(
        y_pred,
        y_test,
    )

    ridge_step = model.named_steps["ridge"]
    scaler = model.named_steps["standardscaler"]

    coefficients = dict(
        zip(
            FEATURE_NAMES,
            ridge_step.coef_.tolist(),
        )
    )

    return {
        "pearson_r": float(r),
        "p_value": float(p_value),
        "coefficients": coefficients,
        "severity_predictions": y_pred,
        "confidence_predictions": confidence_pred,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "intercept": float(ridge_step.intercept_),
    }


def run_lodo(
    variant,
    train_split="dev",
    eval_split="test",
):
    print("\n" + "=" * 70)
    print(
        f"Method 4: Learned Severity Proxy - "
        f"variant={variant}"
    )
    print(
        f"Training split: {train_split} | "
        f"Evaluation split: {eval_split}"
    )
    print("=" * 70)

    train_data = {}
    eval_data = {}

    for dataset in IN_DOMAIN_DATASETS:
        X_tr, y_tr, _, _ = build_feature_matrix(
            dataset,
            train_split,
            variant,
        )

        X_ev, y_ev, flagged_ev, keys_ev = (
            build_feature_matrix(
                dataset,
                eval_split,
                variant,
            )
        )

        if X_tr is not None:
            train_data[dataset] = (
                X_tr,
                y_tr,
            )

            print(
                f"  {dataset} [{train_split}]: "
                f"{len(X_tr)} usable segments "
                f"(training pool)"
            )

        if X_ev is not None:
            eval_data[dataset] = (
                X_ev,
                y_ev,
                flagged_ev,
                keys_ev,
            )

            print(
                f"  {dataset} [{eval_split}]: "
                f"{len(X_ev)} usable segments "
                f"(evaluation)"
            )

    results = {
        "variant": variant,
        "train_split": train_split,
        "eval_split": eval_split,
        "feature_names": FEATURE_NAMES,
        "join_strategy": (
            "normalized segment text + within-transcript "
            "occurrence index"
        ),
        "lodo_folds": {},
        "shetland": None,
    }

    # Leave-one-dataset-out:
    # train on the other two datasets' dev splits,
    # evaluate on the held-out dataset's test split.
    for held_out in IN_DOMAIN_DATASETS:
        train_datasets = [
            d
            for d in IN_DOMAIN_DATASETS
            if d != held_out
        ]

        if (
            not all(
                d in train_data
                for d in train_datasets
            )
            or held_out not in eval_data
        ):
            print(
                f"\n  Skipping fold "
                f"(held_out={held_out}) - missing data"
            )
            continue

        X_train = np.concatenate(
            [
                train_data[d][0]
                for d in train_datasets
            ],
            axis=0,
        )

        y_train = np.concatenate(
            [
                train_data[d][1]
                for d in train_datasets
            ],
            axis=0,
        )

        (
            X_test,
            y_test,
            flagged_test,
            keys_test,
        ) = eval_data[held_out]

        print(
            f"\n  Train: {train_datasets} [{train_split}]"
            f" -> Test: {held_out} [{eval_split}]"
        )
        print(
            f"  Train: {len(X_train)} segments | "
            f"Test: {len(X_test)} segments"
        )

        model = fit_proxy(
            X_train,
            y_train,
        )

        evaluation = evaluate_proxy(
            model,
            X_test,
            y_test,
        )

        print(
            f"  Pearson r="
            f"{evaluation['pearson_r']:.3f} "
            f"(p={evaluation['p_value']:.4f})"
        )

        print(
            "  Coefficients: "
            + "  ".join(
                f"{name}={value:+.3f}"
                for name, value
                in evaluation[
                    "coefficients"
                ].items()
            )
        )

        results["lodo_folds"][held_out] = {
            "train_datasets": train_datasets,
            "train_split": train_split,
            "eval_split": eval_split,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "pearson_r": (
                evaluation["pearson_r"]
            ),
            "p_value": (
                evaluation["p_value"]
            ),
            "coefficients": (
                evaluation["coefficients"]
            ),
            "scaler_mean": (
                evaluation["scaler_mean"]
            ),
            "scaler_scale": (
                evaluation["scaler_scale"]
            ),
            "intercept": (
                evaluation["intercept"]
            ),
            "severity_predictions": (
                evaluation[
                    "severity_predictions"
                ].tolist()
            ),
            "confidence_predictions": (
                evaluation[
                    "confidence_predictions"
                ].tolist()
            ),
            "actual_severity": (
                y_test.tolist()
            ),
            "actual_flagged": (
                flagged_test.tolist()
            ),
            "keys": keys_test,
        }

    # Final out-of-domain evaluation:
    # train on all three in-domain dev sets,
    # evaluate on Shetland full.
    if len(train_data) == len(IN_DOMAIN_DATASETS):
        X_all = np.concatenate(
            [
                train_data[d][0]
                for d in IN_DOMAIN_DATASETS
            ],
            axis=0,
        )

        y_all = np.concatenate(
            [
                train_data[d][1]
                for d in IN_DOMAIN_DATASETS
            ],
            axis=0,
        )

        (
            X_shetland,
            y_shetland,
            flagged_shetland,
            keys_shetland,
        ) = build_feature_matrix(
            "shetland",
            "full",
            variant,
        )

        if X_shetland is not None:
            model_final = fit_proxy(
                X_all,
                y_all,
            )

            evaluation = evaluate_proxy(
                model_final,
                X_shetland,
                y_shetland,
            )

            print(
                "\n  SHETLAND "
                "(true holdout; trained on all "
                "three in-domain dev sets)"
            )

            print(
                f"  Train: {len(X_all)} segments | "
                f"Test: {len(X_shetland)} segments"
            )

            print(
                f"  Pearson r="
                f"{evaluation['pearson_r']:.3f} "
                f"(p={evaluation['p_value']:.4f})"
            )

            results["shetland"] = {
                "n_train": len(X_all),
                "n_shetland": len(X_shetland),
                "pearson_r": (
                    evaluation["pearson_r"]
                ),
                "p_value": (
                    evaluation["p_value"]
                ),
                "coefficients": (
                    evaluation["coefficients"]
                ),
                "scaler_mean": (
                    evaluation["scaler_mean"]
                ),
                "scaler_scale": (
                    evaluation["scaler_scale"]
                ),
                "intercept": (
                    evaluation["intercept"]
                ),
                "severity_predictions": (
                    evaluation[
                        "severity_predictions"
                    ].tolist()
                ),
                "confidence_predictions": (
                    evaluation[
                        "confidence_predictions"
                    ].tolist()
                ),
                "actual_severity": (
                    y_shetland.tolist()
                ),
                "actual_flagged": (
                    flagged_shetland.tolist()
                ),
                "keys": keys_shetland,
            }

        else:
            print(
                "\n  SHETLAND: no usable data - "
                "skipping holdout evaluation"
            )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"learned_severity_proxy_{variant}.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        required=True,
        choices=VARIANTS,
    )

    parser.add_argument(
        "--train-split",
        default="dev",
        choices=["dev"],
    )

    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["test"],
    )

    args = parser.parse_args()

    run_lodo(
        args.variant,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )


if __name__ == "__main__":
    main()
