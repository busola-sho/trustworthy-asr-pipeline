"""
scripts/evaluation/sentence_confidence/summarise_sentence_conf_results.py

Reads all method evaluation JSON files and prints a clean summary table.

Usage:
    python scripts/evaluation/sentence_confidence/summarise_sentence_conf_results.py
"""

import json
import os
import argparse
import glob


def load_results(results_dir: str) -> list:
    pattern = os.path.join(results_dir, "method*.json")
    files   = sorted(glob.glob(pattern))
    results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        if isinstance(data, list):
            continue
        if "method" not in data:
            continue
        results.append(data)
    return results


def get_dataset(r: dict) -> str:
    labels = r.get("labels_file", "") or r.get("conf_file", "")
    for d in ["commonvoice", "edacc", "english_dialects", "shetland"]:
        if d in labels:
            return d
    return "?"


def print_table(results: list):
    # sort by dataset then method
    dataset_order = {"commonvoice": 0, "edacc": 1, "english_dialects": 2, "shetland": 3}
    results = sorted(results, key=lambda r: (
        dataset_order.get(get_dataset(r), 9),
        r.get("method", "")
    ))

    # fixed column widths
    print(f"\n{'Method':<20} {'Dataset':<20} {'WER':>6} {'MeanConf':>9} {'MeanSev':>8} {'MedSev':>7} {'Spearman':>9} {'p-val':>7}")
    print(f"{'─'*82}")

    prev_dataset = None
    for r in results:
        method  = r.get("method", "?")
        dataset = get_dataset(r)

        # blank line between datasets
        if prev_dataset and dataset != prev_dataset:
            print()
        prev_dataset = dataset

        wer     = r.get("corpus_wer")
        mean_c  = r.get("mean_conf")
        mean_s  = r.get("mean_severity")
        med_s   = r.get("median_severity")
        corr    = r.get("spearman_corr")
        pval    = r.get("spearman_pvalue")

        wer_str  = f"{wer*100:.1f}%"  if wer    is not None else "—"
        mc_str   = f"{mean_c:.3f}"    if mean_c  is not None else "—"
        ms_str   = f"{mean_s:.3f}"    if mean_s  is not None else "—"
        med_str  = f"{med_s:.1f}"     if med_s   is not None else "—"
        corr_str = f"{corr:+.3f}"     if corr    is not None else "—"
        if pval is None:
            pval_str = "—"
        elif pval < 0.001:
            pval_str = "<.001"
        else:
            pval_str = f"{pval:.4f}"
        sig      = "✓" if (pval is not None and pval < 0.05) else " "

        print(f"{method:<20} {dataset:<20} {wer_str:>6} {mc_str:>9} {ms_str:>8} {med_str:>7} {corr_str:>8}{sig} {pval_str:>7}")

    print(f"\n  ✓ = p < 0.05 (statistically significant)")
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
    print_table(results)


if __name__ == "__main__":
    main()