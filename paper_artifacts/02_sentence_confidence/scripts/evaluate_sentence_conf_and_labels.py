"""
rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py

Evaluates a sentence-level confidence method against ground truth labels.
Adds --confidence-field so crossmodel_agreement's file can be evaluated
under either "confidence" (mean_agreement) or "min_agreement" without
needing a separate saved variant file.
"""

import json
import os
import argparse
import statistics
from scipy.stats import spearmanr


def load_labels(labels_path):
    with open(labels_path) as f:
        data = json.load(f)
    return {
        (r["dataset_index"], r["sent_pos"]): r
        for r in data.get("rows", [])
        if r.get("mar_verdict") is not None and r.get("severity") is not None
    }


def load_confidences_from_selector(conf_path, confidence_field="confidence"):
    with open(conf_path) as f:
        data = json.load(f)

    corpus_wer = data.get("corpus_wer")
    conf_map = {}

    for sample in data.get("samples", []):
        if sample.get("skipped") or sample.get("error"):
            continue
        dataset_index = sample.get("dataset_index")
        if dataset_index is None:
            continue
        sample_wer = sample.get("sample_WER")

        sent_list = sample.get("sentence_confidences", [])
        if not sent_list:
            sent_list = sample.get("sentences", [])

        for sent_pos, sc in enumerate(sent_list):
            confidence = sc.get(confidence_field)
            pos = sc.get("sent_pos", sent_pos)
            if confidence is not None:
                conf_map[(dataset_index, pos)] = {
                    "confidence": confidence,
                    "sentence": sc.get("sentence", ""),
                    "sample_wer": sample_wer,
                }

    return conf_map, corpus_wer


def evaluate(labels_path, conf_path, method, output_path=None, confidence_field="confidence"):
    print(f"\n-- Method: {method} --")

    if not os.path.exists(labels_path):
        print(f"  ERROR: labels file not found: {labels_path}")
        return None
    if not os.path.exists(conf_path):
        print(f"  ERROR: confidences file not found: {conf_path}")
        return None

    labels = load_labels(labels_path)
    conf_map, corpus_wer = load_confidences_from_selector(conf_path, confidence_field)

    print(f"  Labels: {len(labels)} sentences")
    print(f"  Confidences: {len(conf_map)} sentences (field: {confidence_field})")

    matched = []
    for key, label in labels.items():
        conf_entry = conf_map.get(key)
        if conf_entry is not None:
            matched.append({
                "confidence": conf_entry["confidence"],
                "severity": label["severity"],
                "mar_verdict": label["mar_verdict"],
                "sentence": conf_entry["sentence"],
                "sample_wer": conf_entry.get("sample_wer"),
            })

    if not matched:
        print(f"  ERROR: no matching rows between labels and confidences.")
        return None

    confidences = [r["confidence"] for r in matched]
    severities = [r["severity"] for r in matched]
    verdicts = [r["mar_verdict"] for r in matched]

    n = len(matched)
    mean_conf = sum(confidences) / n
    mean_sev = sum(severities) / n
    median_sev = statistics.median(severities)
    mar_rate = sum(1 for v in verdicts if v) / n
    wers = [r["sample_wer"] for r in matched if r.get("sample_wer") is not None]
    mean_wer = sum(wers) / len(wers) if wers else None

    corr, pvalue = spearmanr(confidences, severities)

    if corpus_wer is not None:
        print(f"  Corpus WER: {corpus_wer*100:.1f}%")
    print(f"  Matched: {n} sentences")
    print(f"  MAR rate: {mar_rate*100:.1f}%")
    print(f"  Mean confidence: {mean_conf:.3f}")
    print(f"  Mean severity: {mean_sev:.3f}")
    print(f"  Median severity: {median_sev:.1f}")
    if mean_wer is not None:
        print(f"  Mean sample WER: {mean_wer*100:.1f}%")
    print(f"  Spearman corr: {corr:+.3f} (p={pvalue:.4f})")

    result = {
        "method": method,
        "labels_file": labels_path,
        "conf_file": conf_path,
        "n_sentences": n,
        "corpus_wer": round(corpus_wer, 4) if corpus_wer is not None else None,
        "mar_rate": round(mar_rate, 4),
        "mean_conf": round(mean_conf, 4),
        "mean_severity": round(mean_sev, 4),
        "median_severity": float(median_sev),
        "spearman_corr": round(corr, 4),
        "spearman_pvalue": round(pvalue, 4),
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--confidences", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--confidence-field", default="confidence")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evaluate(args.labels, args.confidences, args.method, args.output, args.confidence_field)


if __name__ == "__main__":
    main()
