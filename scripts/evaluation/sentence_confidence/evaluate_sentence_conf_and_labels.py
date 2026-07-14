"""
scripts/evaluation/sentence_confidence/evaluate_sentence_conf_and_labels.py

Evaluates a sentence-level confidence method against ground truth labels.

Reads confidence scores directly from selector output files
(context_v2_whisperx_confidence_sentences/ or context_v2_whisperx_probscore/).

Usage:
    # Method 1a: confscore (1-5 scale mapped to 0.2-1.0)
    python scripts/evaluation/sentence_confidence/evaluate_sentence_conf_and_labels.py \
        --labels      results/sentence_confidence/sentence_labels_commonvoice.json \
        --confidences results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_commonvoice_qwen_sub150.json \
        --method      confscore \
        --output      results/sentence_confidence/method1_confscore_commonvoice.json

    # Method 1b: probscore (0.0-1.0 float)
    python scripts/evaluation/sentence_confidence/evaluate_sentence_conf_and_labels.py \
        --labels      results/sentence_confidence/sentence_labels_commonvoice.json \
        --confidences results/combinations_v2judge/context_v2_whisperx_probscore/context_v2_probscore_commonvoice_qwen_sub150.json \
        --method      probscore \
        --output      results/sentence_confidence/method1_probscore_commonvoice.json
"""

import json
import os
import argparse
import statistics
from scipy.stats import spearmanr


def load_labels(labels_path: str) -> dict:
    """Load labels keyed by (dataset_index, sent_pos)."""
    with open(labels_path) as f:
        data = json.load(f)
    return {
        (r["dataset_index"], r["sent_pos"]): r
        for r in data.get("rows", [])
        if r.get("mar_verdict") is not None
        and r.get("severity")   is not None
    }


def load_confidences_from_selector(conf_path: str) -> dict:
    """
    Load confidence scores from any confidence file.
    Handles two formats:
      1. Selector output: sentence_confidences per sample (confscore/probscore)
      2. Agreement/proxy output: sentences per sample (crossmodel, proxy)
    Keyed by (dataset_index, sent_pos).
    """
    with open(conf_path) as f:
        data = json.load(f)

    corpus_wer = data.get("corpus_wer")
    conf_map   = {}

    for sample in data.get("samples", []):
        if sample.get("skipped") or sample.get("error"):
            continue
        dataset_index = sample.get("dataset_index")
        if dataset_index is None:
            continue
        sample_wer = sample.get("sample_WER")

        # format 1: selector output uses sentence_confidences
        sent_list = sample.get("sentence_confidences", [])
        # format 2: agreement/proxy output uses sentences
        if not sent_list:
            sent_list = sample.get("sentences", [])

        for sent_pos, sc in enumerate(sent_list):
            confidence = sc.get("confidence")
            # use explicit sent_pos field if available
            pos = sc.get("sent_pos", sent_pos)
            if confidence is not None:
                conf_map[(dataset_index, pos)] = {
                    "confidence": confidence,
                    "sentence":   sc.get("sentence", ""),
                    "sample_wer": sample_wer,
                }

    return conf_map, corpus_wer


def evaluate(labels_path: str, conf_path: str, method: str, output_path: str = None):
    print(f"\n── Method: {method} ──")

    if not os.path.exists(labels_path):
        print(f"  ERROR: labels file not found: {labels_path}")
        return None
    if not os.path.exists(conf_path):
        print(f"  ERROR: confidences file not found: {conf_path}")
        return None

    labels   = load_labels(labels_path)
    conf_map, corpus_wer = load_confidences_from_selector(conf_path)

    print(f"  Labels:      {len(labels)} sentences")
    print(f"  Confidences: {len(conf_map)} sentences")

    # match by (dataset_index, sent_pos)
    matched = []
    for key, label in labels.items():
        conf_entry = conf_map.get(key)
        if conf_entry is not None:
            matched.append({
                "confidence":  conf_entry["confidence"],
                "severity":    label["severity"],
                "mar_verdict": label["mar_verdict"],
                "sentence":    conf_entry["sentence"],
            })

    if not matched:
        print(f"  ERROR: no matching rows between labels and confidences.")
        print(f"  Check that both files are from the same dataset.")
        return None

    confidences = [r["confidence"] for r in matched]
    severities  = [r["severity"]   for r in matched]
    verdicts    = [r["mar_verdict"] for r in matched]

    n          = len(matched)
    mean_conf  = sum(confidences) / n
    mean_sev   = sum(severities)  / n
    median_sev = statistics.median(severities)
    mar_rate   = sum(1 for v in verdicts if v) / n
    wers       = [r["sample_wer"] for r in matched if r.get("sample_wer") is not None]
    mean_wer   = sum(wers) / len(wers) if wers else None

    corr, pvalue = spearmanr(confidences, severities)

    if corpus_wer is not None:
        print(f"  Corpus WER:               {corpus_wer*100:.1f}%")
    print(f"  Matched:                  {n} sentences")
    print(f"  MAR rate:                 {mar_rate*100:.1f}%")
    print(f"  Mean confidence:          {mean_conf:.3f}")
    print(f"  Mean severity:            {mean_sev:.3f}")
    print(f"  Median severity:          {median_sev:.1f}")
    if mean_wer is not None:
        print(f"  Mean sample WER:          {mean_wer*100:.1f}%")
    print(f"  Spearman corr:            {corr:+.3f}  (p={pvalue:.4f})")

    if corr < -0.3 and pvalue < 0.05:
        print(f"  → Strong negative correlation ✓")
    elif corr < 0 and pvalue < 0.05:
        print(f"  → Weak but significant negative correlation")
    elif pvalue >= 0.05:
        print(f"  → No significant correlation (p={pvalue:.4f})")
    else:
        print(f"  → Positive correlation — confidence does not predict severity")

    # mean confidence per severity level
    print(f"\n  Mean confidence per severity level:")
    for sev in range(5):
        sev_confs = [r["confidence"] for r in matched if r["severity"] == sev]
        if sev_confs:
            print(f"    Severity {sev}: mean conf = {sum(sev_confs)/len(sev_confs):.3f}  (n={len(sev_confs)})")

    result = {
        "method":          method,
        "labels_file":     labels_path,
        "conf_file":       conf_path,
        "n_sentences":     n,
        "corpus_wer":      round(corpus_wer, 4) if corpus_wer is not None else None,
        "mar_rate":        round(mar_rate,  4),
        "mean_conf":       round(mean_conf, 4),
        "mean_severity":   round(mean_sev,  4),
        "median_severity": float(median_sev),
        "spearman_corr":   round(corr,      4),
        "spearman_pvalue": round(pvalue,    4),
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sentence confidence method against ground truth labels"
    )
    parser.add_argument("--labels",      required=True,
                        help="sentence_labels_{dataset}.json from label_sentence_meanings.py")
    parser.add_argument("--confidences", required=True,
                        help="selector output JSON (context_v2_whisperx_confidence_sentences/ or probscore/)")
    parser.add_argument("--method",      required=True,
                        help="method name: confscore, probscore, proxy_model etc.")
    parser.add_argument("--output",      default=None,
                        help="optional path to save result JSON")
    args = parser.parse_args()

    evaluate(args.labels, args.confidences, args.method, args.output)


if __name__ == "__main__":
    main()