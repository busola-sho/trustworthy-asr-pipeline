"""
scripts/evaluation/sentence_confidence/summarise_split.py

Splits the sentence-confidence results into two focused tables instead
of one combined one:

  TABLE A: signal comparison against ONE fixed ground truth (naive's
  probscore run) - acoustic_mean, crossmodel_mean, crossmodel_min,
  probscore, proxy_model, proxy_model_confscore. Severity is identical
  across every row within a dataset here, since only the confidence
  SIGNAL changes, not the underlying transcript being judged - this is
  the "which signal predicts severity best" comparison.

  TABLE B: confscore vs probscore, isolated. These two use DIFFERENT
  underlying transcripts (naive_confscore vs naive_probscore are
  separate runs, with a small but real WER difference between them),
  so mixing them into table A would conflate "which signal is better"
  with "which run happened to have lower severity" - kept separate so
  that distinction stays visible rather than causing confusion.

Usage:
    python scripts/evaluation/sentence_confidence/summarise_split.py
"""

import json
import os
import glob
import argparse

TABLE_A_METHODS = ["acoustic_mean", "crossmodel_mean", "crossmodel_min",
                   "probscore", "proxy_model", "proxy_model_confscore"]
TABLE_B_METHODS = ["confscore", "probscore"]
DATASET_ORDER = ["commonvoice", "edacc", "english_dialects", "shetland"]


def load_results(results_dir):
    results = []
    for f in sorted(glob.glob(os.path.join(results_dir, "method*.json"))):
        with open(f) as fp:
            data = json.load(fp)
        if isinstance(data, list) or "method" not in data:
            continue
        if "results" in data and "labels_file" not in data:
            continue
        results.append(data)
    return results


def get_dataset(r):
    labels = r.get("labels_file", "") or r.get("conf_file", "")
    for d in DATASET_ORDER:
        if d in labels:
            return d
    return "?"


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

    return f"{method:<22} {dataset:<18} {wer_str:>6} {mc_str:>8} {ms_str:>7} {med_str:>6} {corr_str:>7}{sig} {pval_str:>7}"


def print_table(title, results, methods):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"{'Method':<22} {'Dataset':<18} {'WER':>6} {'MeanConf':>8} {'MeanSev':>7} {'MedSev':>6} {'Spearman':>8} {'p-val':>7}")
    print("-" * 90)

    prev_dataset = None
    for d in DATASET_ORDER:
        for m in methods:
            matching = [r for r in results if r.get("method") == m and get_dataset(r) == d]
            if not matching:
                continue
            if prev_dataset and d != prev_dataset:
                print()
            prev_dataset = d
            print(fmt_row(m, d, matching[0]))

    print(f"\n  * = p < 0.05 (statistically significant)")
    print(f"  Spearman: negative = high confidence predicts low severity (good)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/sentence_confidence")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
        return

    print(f"Found {len(results)} result files")

    print_table(
        "TABLE A: Signal comparison (fixed ground truth - naive's probscore run)",
        results, TABLE_A_METHODS
    )
    print_table(
        "TABLE B: confscore vs probscore (different underlying transcripts - not directly comparable to Table A)",
        results, TABLE_B_METHODS
    )


if __name__ == "__main__":
    main()