"""
scripts/evaluation/sentence_confidence/train_proxy_model.py

Method 4: Proxy/meta-model (Ridge Regression).

Design: Leave-one-dataset-out (LODO) evaluation.
  - Train on 2 datasets, test on the 3rd. Repeat for each dataset.
  - Train on all 3 -> test on Shetland (held-out, unseen dialect).

Following Bachar et al. (2026) LPP framework.

Features:
  - verbalized_conf : selector LLM confidence (0.2-1.0)
  - crossmodel_mean : pairwise model agreement (0-1)
  - acoustic_mean   : mean ASR word confidence (0-1)

Target: severity (0-4)

Usage:
    python scripts/evaluation/sentence_confidence/train_proxy_model.py
"""

import json
import os
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

OUTPUT_DIR     = "results/sentence_confidence"
TRAIN_DATASETS = ["commonvoice", "edacc", "english_dialects"]
HELD_OUT       = "shetland"
FEATURE_NAMES  = ["verbalized_conf", "crossmodel_mean", "acoustic_mean"]

LABEL_FILES = {
    "commonvoice":      "results/sentence_confidence/sentence_labels_commonvoice.json",
    "edacc":            "results/sentence_confidence/sentence_labels_edacc.json",
    "english_dialects": "results/sentence_confidence/sentence_labels_english_dialects.json",
    "shetland":         "results/sentence_confidence/sentence_labels_shetland.json",
}

CROSSMODEL_FILES = {
    "commonvoice":      "results/sentence_confidence/crossmodel_agreement_commonvoice.json",
    "edacc":            "results/sentence_confidence/crossmodel_agreement_edacc.json",
    "english_dialects": "results/sentence_confidence/crossmodel_agreement_english_dialects.json",
    "shetland":         "results/sentence_confidence/crossmodel_agreement_shetland.json",
}

ACOUSTIC_FILES = {
    "commonvoice":      "results/sentence_confidence/acoustic_confidence_commonvoice.json",
    "edacc":            "results/sentence_confidence/acoustic_confidence_edacc.json",
    "english_dialects": "results/sentence_confidence/acoustic_confidence_english_dialects.json",
    "shetland":         "results/sentence_confidence/acoustic_confidence_shetland.json",
}

CORPUS_WER = {
    "commonvoice":      0.18328,
    "edacc":            0.19747,
    "english_dialects": 0.03809,
    "shetland":         0.1150,
}

DATASET_OFFSET = {
    "commonvoice":      0,
    "edacc":            10000,
    "english_dialects": 20000,
    "shetland":         30000,
}


def load_features(dataset):
    with open(LABEL_FILES[dataset]) as f:
        labels_data = json.load(f)

    label_map = {
        (r["dataset_index"], r["sent_pos"]): r
        for r in labels_data.get("rows", [])
        if r.get("severity") is not None
        and r.get("verbalized_conf") is not None
    }

    with open(CROSSMODEL_FILES[dataset]) as f:
        cm_data = json.load(f)
    cm_map = {}
    for samp in cm_data.get("samples", []):
        di = samp.get("dataset_index")
        for sent in samp.get("sentences", []):
            cm_map[(di, sent["sent_pos"])] = sent.get("confidence")

    with open(ACOUSTIC_FILES[dataset]) as f:
        ac_data = json.load(f)
    ac_map = {}
    for samp in ac_data.get("samples", []):
        di = samp.get("dataset_index")
        for sent in samp.get("sentences", []):
            ac_map[(di, sent["sent_pos"])] = sent.get("confidence")

    offset = DATASET_OFFSET[dataset]
    rows = []
    for key, label in label_map.items():
        cm = cm_map.get(key)
        ac = ac_map.get(key)
        if cm is None or ac is None:
            continue
        rows.append({
            "dataset":         dataset,
            "dataset_index":   key[0] + offset,
            "orig_index":      key[0],
            "sent_pos":        key[1],
            "verbalized_conf": label["verbalized_conf"],
            "crossmodel_mean": cm,
            "acoustic_mean":   ac,
            "severity":        label["severity"],
            "hyp_sentence":    label.get("hyp_sentence", ""),
        })
    return rows


def to_arrays(rows):
    X      = np.array([[r["verbalized_conf"], r["crossmodel_mean"], r["acoustic_mean"]] for r in rows])
    y      = np.array([r["severity"] for r in rows])
    groups = np.array([r["dataset_index"] for r in rows])
    return X, y, groups


def train_and_predict(train_rows, test_rows):
    X_train, y_train, groups_train = to_arrays(train_rows)
    X_test,  y_test,  _            = to_arrays(test_rows)

    pipeline   = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    gkf        = GroupKFold(n_splits=min(3, len(set(groups_train))))
    param_grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}

    grid_search = GridSearchCV(pipeline, param_grid, cv=gkf, scoring="r2", refit=True)
    grid_search.fit(X_train, y_train, groups=groups_train)

    best_alpha = grid_search.best_params_["ridge__alpha"]
    best_model = grid_search.best_estimator_
    y_pred     = best_model.predict(X_test)
    corr, pval = spearmanr(y_pred, y_test)
    coefs      = best_model.named_steps["ridge"].coef_

    return {
        "best_alpha": best_alpha,
        "spearman":   round(float(corr), 4),
        "pvalue":     round(float(pval), 4),
        "coefs":      dict(zip(FEATURE_NAMES, coefs.tolist())),
        "y_pred":     y_pred,
        "model":      best_model,
    }


def save_confidence_file(dataset, rows, model):
    X, y, _ = to_arrays(rows)
    preds    = model.predict(X)

    pred_min = preds.min()
    pred_max = preds.max()
    conf_scores = (
        1 - (preds - pred_min) / (pred_max - pred_min)
        if pred_max > pred_min
        else np.ones_like(preds) * 0.5
    )

    by_utterance = defaultdict(list)
    for row, conf in zip(rows, conf_scores):
        by_utterance[row["orig_index"]].append({
            "sent_pos":   row["sent_pos"],
            "confidence": round(float(conf), 4),
            "sentence":   row.get("hyp_sentence", ""),
        })

    samples   = [{"dataset_index": di, "sentences": sents} for di, sents in by_utterance.items()]
    conf_path = os.path.join(OUTPUT_DIR, f"proxy_model_{dataset}.json")
    with open(conf_path, "w") as f:
        json.dump({
            "method":     "proxy_model",
            "dataset":    dataset,
            "corpus_wer": CORPUS_WER.get(dataset),
            "samples":    samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {conf_path}")


def main():
    print("Loading features...")
    all_data = {d: load_features(d) for d in TRAIN_DATASETS + [HELD_OUT]}
    for d, rows in all_data.items():
        n_utt = len(set(r["dataset_index"] for r in rows))
        print(f"  {d}: {len(rows)} sentences, {n_utt} utterances")

    lodo_results = {}

    print("\n── Leave-one-dataset-out evaluation ──")
    for test_dataset in TRAIN_DATASETS:
        train_datasets = [d for d in TRAIN_DATASETS if d != test_dataset]
        train_rows     = []
        for d in train_datasets:
            train_rows.extend(all_data[d])
        test_rows = all_data[test_dataset]

        print(f"\n  Train: {train_datasets} -> Test: {test_dataset}")
        print(f"  Train: {len(train_rows)} sentences | Test: {len(test_rows)} sentences")

        result = train_and_predict(train_rows, test_rows)
        lodo_results[test_dataset] = result

        print(f"  Best alpha: {result['best_alpha']}")
        print(f"  Spearman:   {result['spearman']:+.3f} (p={result['pvalue']:.4f})")
        print(f"  Coefs: " + "  ".join(f"{k}={v:+.3f}" for k, v in result["coefs"].items()))

        save_confidence_file(test_dataset, test_rows, result["model"])

    print("\n── Held-out test: Shetland ──")
    train_rows = []
    for d in TRAIN_DATASETS:
        train_rows.extend(all_data[d])
    test_rows = all_data[HELD_OUT]

    print(f"  Train: {TRAIN_DATASETS} -> Test: {HELD_OUT}")
    print(f"  Train: {len(train_rows)} sentences | Test: {len(test_rows)} sentences")

    shetland_result = train_and_predict(train_rows, test_rows)
    lodo_results[HELD_OUT] = shetland_result

    print(f"  Best alpha: {shetland_result['best_alpha']}")
    print(f"  Spearman:   {shetland_result['spearman']:+.3f} (p={shetland_result['pvalue']:.4f})")
    print(f"  Coefs: " + "  ".join(f"{k}={v:+.3f}" for k, v in shetland_result["coefs"].items()))

    save_confidence_file(HELD_OUT, test_rows, shetland_result["model"])

    print("\n── Summary ──")
    print(f"{'Dataset':<20} {'Spearman':>10} {'p-val':>8} {'alpha':>6}")
    print("-" * 48)
    for dataset, res in lodo_results.items():
        sig = "✓" if res["pvalue"] < 0.05 else " "
        print(f"  {dataset:<18} {res['spearman']:>+10.3f}{sig} {res['pvalue']:>8.4f} {res['best_alpha']:>6}")

    summary = {
        "method":   "proxy_model_lodo",
        "model":    "Ridge Regression",
        "features": FEATURE_NAMES,
        "design":   "leave-one-dataset-out + shetland held-out",
        "results": {
            d: {
                "spearman":   r["spearman"],
                "pvalue":     r["pvalue"],
                "best_alpha": r["best_alpha"],
                "coefs":      r["coefs"],
            }
            for d, r in lodo_results.items()
        },
    }

    out = os.path.join(OUTPUT_DIR, "method4_proxy_model_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")

    print("\nNow run:")
    print("  for dataset in commonvoice edacc english_dialects shetland; do")
    print("    python scripts/evaluation/sentence_confidence/evaluate_sentence_conf_and_labels.py \\")
    print("      --labels results/sentence_confidence/sentence_labels_${dataset}.json \\")
    print("      --confidences results/sentence_confidence/proxy_model_${dataset}.json \\")
    print("      --method proxy_model \\")
    print("      --output results/sentence_confidence/method4_proxy_model_${dataset}.json")
    print("  done")


if __name__ == "__main__":
    main()