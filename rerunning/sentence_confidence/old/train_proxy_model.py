"""
rerunning/sentence_confidence/train_proxy_model.py

Method 4: Proxy/meta-model (Ridge Regression).

FIXED: --variant now selects matching crossmodel_agreement and
acoustic_confidence files too, not just verbalized_conf's source labels.
The previous version always read the probscore-anchored crossmodel/
acoustic files regardless of --variant, meaning --variant confscore
was training on a hybrid (confscore's own score + probscore's
crossmodel/acoustic) - a real inconsistency, now fixed so each variant
is fully self-consistent.

Design: Leave-one-dataset-out (LODO) evaluation.

Usage:
    python rerunning/sentence_confidence/train_proxy_model.py --variant probscore
    python rerunning/sentence_confidence/train_proxy_model.py --variant confscore
"""

import json
import os
import argparse
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

DATASET_OFFSET = {
    "commonvoice":      0,
    "edacc":            10000,
    "english_dialects": 20000,
    "shetland":         30000,
}


def label_file_for(dataset, variant):
    return f"results/sentence_confidence/sentence_labels_{dataset}_{variant}.json"


def crossmodel_file_for(dataset, variant):
    return f"results/sentence_confidence/crossmodel_agreement_{variant}_{dataset}.json"


def acoustic_file_for(dataset, variant):
    return f"results/sentence_confidence/acoustic_confidence_{variant}_{dataset}.json"


def load_features(dataset, variant):
    with open(label_file_for(dataset, variant)) as f:
        labels_data = json.load(f)

    label_map = {
        (r["dataset_index"], r["sent_pos"]): r
        for r in labels_data.get("rows", [])
        if r.get("severity") is not None and r.get("verbalized_conf") is not None
    }

    with open(crossmodel_file_for(dataset, variant)) as f:
        cm_data = json.load(f)
    cm_map = {}
    for samp in cm_data.get("samples", []):
        di = samp.get("dataset_index")
        for sent in samp.get("sentences", []):
            cm_map[(di, sent["sent_pos"])] = sent.get("confidence")

    with open(acoustic_file_for(dataset, variant)) as f:
        ac_data = json.load(f)
    ac_map = {}
    for samp in ac_data.get("samples", []):
        di = samp.get("dataset_index")
        for sent in samp.get("sentences", []):
            ac_map[(di, sent["sent_pos"])] = sent.get("confidence")

    corpus_wer = ac_data.get("corpus_wer")

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
    return rows, corpus_wer


def to_arrays(rows):
    X = np.array([[r["verbalized_conf"], r["crossmodel_mean"], r["acoustic_mean"]] for r in rows])
    y = np.array([r["severity"] for r in rows])
    groups = np.array([r["dataset_index"] for r in rows])
    return X, y, groups


def train_and_predict(train_rows, test_rows):
    X_train, y_train, groups_train = to_arrays(train_rows)
    X_test, y_test, _ = to_arrays(test_rows)

    pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    gkf = GroupKFold(n_splits=min(3, len(set(groups_train))))
    param_grid = {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]}

    grid_search = GridSearchCV(pipeline, param_grid, cv=gkf, scoring="r2", refit=True)
    grid_search.fit(X_train, y_train, groups=groups_train)

    best_alpha = grid_search.best_params_["ridge__alpha"]
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    corr, pval = spearmanr(y_pred, y_test)
    coefs = best_model.named_steps["ridge"].coef_

    return {
        "best_alpha": best_alpha,
        "spearman": round(float(corr), 4),
        "pvalue": round(float(pval), 4),
        "coefs": dict(zip(FEATURE_NAMES, coefs.tolist())),
        "y_pred": y_pred,
        "model": best_model,
    }


def save_confidence_file(dataset, rows, model, corpus_wer, variant):
    X, y, _ = to_arrays(rows)
    preds = model.predict(X)

    pred_min, pred_max = preds.min(), preds.max()
    conf_scores = (1 - (preds - pred_min) / (pred_max - pred_min)
                   if pred_max > pred_min else np.ones_like(preds) * 0.5)

    by_utterance = defaultdict(list)
    for row, conf in zip(rows, conf_scores):
        by_utterance[row["orig_index"]].append({
            "sent_pos": row["sent_pos"],
            "confidence": round(float(conf), 4),
            "sentence": row.get("hyp_sentence", ""),
        })

    samples = [{"dataset_index": di, "sentences": sents} for di, sents in by_utterance.items()]
    conf_path = os.path.join(OUTPUT_DIR, f"proxy_model_{dataset}_{variant}.json")
    with open(conf_path, "w") as f:
        json.dump({"method": "proxy_model", "dataset": dataset, "variant": variant,
                    "corpus_wer": corpus_wer, "samples": samples}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {conf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["confscore", "probscore"], required=True)
    args = parser.parse_args()
    variant = args.variant

    print(f"Loading features (variant: {variant}, fully self-consistent - "
          f"verbalized_conf, crossmodel_mean, and acoustic_mean all anchored on {variant})...")
    all_data = {}
    corpus_wer_by_dataset = {}
    for d in TRAIN_DATASETS + [HELD_OUT]:
        rows, corpus_wer = load_features(d, variant)
        all_data[d] = rows
        corpus_wer_by_dataset[d] = corpus_wer
        n_utt = len(set(r["dataset_index"] for r in rows))
        print(f"  {d}: {len(rows)} sentences, {n_utt} utterances")

    lodo_results = {}

    print("\n-- Leave-one-dataset-out evaluation --")
    for test_dataset in TRAIN_DATASETS:
        train_datasets = [d for d in TRAIN_DATASETS if d != test_dataset]
        train_rows = []
        for d in train_datasets:
            train_rows.extend(all_data[d])
        test_rows = all_data[test_dataset]

        print(f"\n  Train: {train_datasets} -> Test: {test_dataset}")
        print(f"  Train: {len(train_rows)} sentences | Test: {len(test_rows)} sentences")

        result = train_and_predict(train_rows, test_rows)
        lodo_results[test_dataset] = result

        print(f"  Best alpha: {result['best_alpha']}")
        print(f"  Spearman: {result['spearman']:+.3f} (p={result['pvalue']:.4f})")
        print(f"  Coefs: " + "  ".join(f"{k}={v:+.3f}" for k, v in result["coefs"].items()))

        save_confidence_file(test_dataset, test_rows, result["model"], corpus_wer_by_dataset[test_dataset], variant)

    print("\n-- Held-out test: Shetland --")
    train_rows = []
    for d in TRAIN_DATASETS:
        train_rows.extend(all_data[d])
    test_rows = all_data[HELD_OUT]

    print(f"  Train: {TRAIN_DATASETS} -> Test: {HELD_OUT}")
    print(f"  Train: {len(train_rows)} sentences | Test: {len(test_rows)} sentences")

    shetland_result = train_and_predict(train_rows, test_rows)
    lodo_results[HELD_OUT] = shetland_result

    print(f"  Best alpha: {shetland_result['best_alpha']}")
    print(f"  Spearman: {shetland_result['spearman']:+.3f} (p={shetland_result['pvalue']:.4f})")
    print(f"  Coefs: " + "  ".join(f"{k}={v:+.3f}" for k, v in shetland_result["coefs"].items()))

    save_confidence_file(HELD_OUT, test_rows, shetland_result["model"], corpus_wer_by_dataset[HELD_OUT], variant)

    import joblib
    model_path = os.path.join(OUTPUT_DIR, f"proxy_model_fitted_{variant}.joblib")
    joblib.dump(shetland_result["model"], model_path)
    print(f"Saved fitted model (trained on all 3 dev datasets): {model_path}")
    print(f"  Use this to apply the proxy model to new, unseen data (e.g. real PS audio) - "
          f"see rerunning/sentence_confidence/apply_proxy_model_ps.py")

    print("\n-- Summary --")
    print(f"{'Dataset':<20} {'Spearman':>10} {'p-val':>8} {'alpha':>6}")
    print("-" * 48)
    for dataset, res in lodo_results.items():
        sig = "check" if res["pvalue"] < 0.05 else " "
        print(f"  {dataset:<18} {res['spearman']:>+10.3f}{sig} {res['pvalue']:>8.4f} {res['best_alpha']:>6}")

    summary = {
        "method": "proxy_model_lodo",
        "variant": variant,
        "model": "Ridge Regression",
        "features": FEATURE_NAMES,
        "results": {
            d: {"spearman": r["spearman"], "pvalue": r["pvalue"],
                "best_alpha": r["best_alpha"], "coefs": r["coefs"]}
            for d, r in lodo_results.items()
        },
    }
    out = os.path.join(OUTPUT_DIR, f"method4_proxy_model_summary_{variant}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()