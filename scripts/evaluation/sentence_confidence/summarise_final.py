"""
scripts/evaluation/sentence_confidence/summarise_final.py

Two fully symmetric tables, one per variant (probscore, confscore) -
each internally self-consistent: verbalized score, crossmodel_mean,
crossmodel_min, acoustic_mean, and proxy_model all anchored on that
SAME variant's transcript, so severity and WER are directly comparable
within a table.

Usage:
    python scripts/evaluation/sentence_confidence/summarise_final.py
"""

import json
import os
import glob
import argparse

DATASET_ORDER = ["commonvoice", "edacc", "english_dialects", "shetland"]

METHOD_FILE_PATTERNS = {
    "probscore":       "method1_probscore_{d}.json",
    "crossmodel_mean": "method2a_probscore_{d}.json",
    "crossmodel_min":  "method2b_probscore_{d}.json",
    "acoustic_mean":   "method3_probscore_{d}.json",
    "proxy_model":     "method4_probscore_{d}.json",
}

METHOD_FILE_PATTERNS_CONF = {
    "confscore":       "method1_confscore_{d}.json",
    "crossmodel_mean": "method2a_confscore_{d}.json",
    "crossmodel_min":  "method2b_confscore_{d}.json",
    "acoustic_mean":   "method3_confscore_{d}.json",
    "proxy_model":     "method4_confscore_{d}.json",
}


def load(results_dir, filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_row(method, dataset, r):
    wer = r.get("corpus_wer")
    mean_c = r.get("mean_conf")
    mean_s = r.get("mean_severity")
    med_s = r.get("median_severity")
    corr = r.get("spearman_corr")
    pval = r.get("spearman_pvalue")

    wer_str = f"{wer*100:.1f}%" if wer is not None else "-"
    mc_str = f"{mean_c:.3f}" if mean_c is not None else "-"
    ms_str = f"{mean_s:.3f}" if mean_s is not None else "-"
    med_str = f"{med_s:.1f}" if med_s is not None else "-"
    corr_str = f"{corr:+.3f}" if corr is not None else "-"
    pval_str = "<.001" if (pval is not None and pval < 0.001) else (f"{pval:.4f}" if pval is not None else "-")
    sig = "*" if (pval is not None and pval < 0.05) else " "

    return f"{method:<16} {dataset:<18} {wer_str:>6} {mc_str:>8} {ms_str:>7} {med_str:>6} {corr_str:>7}{sig} {pval_str:>7}"


def print_table(title, results_dir, patterns):
    print(f"\n{'=' * 85}")
    print(f"  {title}")
    print(f"{'=' * 85}")
    print(f"{'Method':<16} {'Dataset':<18} {'WER':>6} {'MeanConf':>8} {'MeanSev':>7} {'MedSev':>6} {'Spearman':>8} {'p-val':>7}")
    print("-" * 85)

    prev_dataset = None
    for d in DATASET_ORDER:
        for method, pattern in patterns.items():
            r = load(results_dir, pattern.format(d=d))
            if r is None:
                continue
            if prev_dataset and d != prev_dataset:
                print()
            prev_dataset = d
            print(fmt_row(method, d, r))

    print(f"\n  * = p < 0.05 (statistically significant)")
    print(f"  Spearman: negative = high confidence predicts low severity (good)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/sentence_confidence")
    args = parser.parse_args()

    print_table("TABLE 1: PROBSCORE-anchored (all signals share this ground truth)",
                args.results_dir, METHOD_FILE_PATTERNS)
    print_table("TABLE 2: CONFSCORE-anchored (all signals share this ground truth)",
                args.results_dir, METHOD_FILE_PATTERNS_CONF)


if __name__ == "__main__":
    main()