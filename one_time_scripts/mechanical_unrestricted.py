"""
mechanical_unrestricted.py

ROVER/MBR consensus, computed on their OWN full sample set - no
grid-intersection restriction. Same methodology as full_results_
report.py's baseline table: pooled WER across all 3 datasets combined,
macro-averaged severity (mean of per-dataset means), Alt% macro-
averaged too. Directly comparable to the individual-model baseline
table, which is also computed on each model's own full N.

Usage:
    python mechanical_unrestricted.py
"""

import json
import glob
from jiwer import wer as compute_wer
from src.text_normalise import normalise

DATASETS = ["commonvoice", "edacc", "english_dialects"]
FLAG_THRESHOLD = 2


def find_mechanical_path(technique, dataset, split):
    if technique == "rover":
        matches = glob.glob(f"writeup_results/voting/rover/rover_{dataset}_{split}.json")
    elif technique == "mbr_consensus":
        matches = glob.glob(f"writeup_results/ensembles/mbr_consensus/mbr_{dataset}_{split}.json")
    else:
        return None
    return matches[0] if matches else None


from src.splits import get_indices_for_split


def load_mechanical_unrestricted(technique, dataset, split):
    path = find_mechanical_path(technique, dataset, split)
    if not path:
        return None
    data = json.load(open(path))
    samples = data.get("samples", [])

    # STILL filter to the correct dev/test indices - "unrestricted" here
    # means "no grid-intersection narrowing", NOT "no split filtering at
    # all". An earlier version of this script wrongly dropped both,
    # letting stale/out-of-split samples leak in (confirmed: edacc dev
    # showed N=136, exceeding its own true dev size of 133 entirely).
    correct_indices = set(get_indices_for_split(dataset, split))
    samples = [s for s in samples if s.get("dataset_index") in correct_indices]

    refs, hyps, severities = [], [], []
    for s in samples:
        if s.get("skipped") or s.get("error"):
            continue
        ref, hyp = s.get("ref"), s.get("hyp")
        if ref and hyp:
            refs.append(normalise(ref))
            hyps.append(normalise(hyp))
        if s.get("severity") is not None:
            severities.append(s["severity"])
    return {"refs": refs, "hyps": hyps, "severities": severities}


def main():
    for split in ["dev", "test"]:
        print(f"\n=== {split.upper()} SPLIT - ROVER/MBR on their OWN full N (no grid-intersection restriction) ===")
        header = f"{'Technique':<16}" + "".join(f"{d:>34}" for d in DATASETS) + f"{'Avg Sev':>10}{'Pooled WER':>12}{'Alt%':>10}{'Total N':>10}"
        print(header)
        print("-" * len(header))

        for technique in ["rover", "mbr_consensus"]:
            dataset_sevs, dataset_flags = [], []
            all_refs, all_hyps, all_sevs = [], [], []
            row = f"{technique:<16}"

            for d in DATASETS:
                loaded = load_mechanical_unrestricted(technique, d, split)
                if loaded and loaded["severities"]:
                    sev = sum(loaded["severities"]) / len(loaded["severities"])
                    n = len(loaded["severities"])
                    dataset_sevs.append(sev)
                    cell_wer = compute_wer(loaded["refs"], loaded["hyps"]) if loaded["refs"] else None
                    cell_flag = sum(1 for x in loaded["severities"] if x >= FLAG_THRESHOLD) / n
                    dataset_flags.append(cell_flag)
                    wer_part = f", WER={cell_wer*100:.2f}%" if cell_wer is not None else ""
                    row += f"{f'{sev:.3f} (N={n}{wer_part})':>34}"
                    all_refs.extend(loaded["refs"])
                    all_hyps.extend(loaded["hyps"])
                    all_sevs.extend(loaded["severities"])
                else:
                    row += f"{'-':>34}"

            avg_sev = sum(dataset_sevs) / len(dataset_sevs) if dataset_sevs else None
            pooled_wer = compute_wer(all_refs, all_hyps) if all_refs else None
            overall_flag = sum(dataset_flags) / len(dataset_flags) if dataset_flags else None

            row += f"{(f'{avg_sev:.3f}' if avg_sev is not None else '-'):>10}"
            row += f"{(f'{pooled_wer*100:.2f}%' if pooled_wer is not None else '-'):>12}"
            row += f"{(f'{overall_flag*100:.1f}%' if overall_flag is not None else '-'):>10}"
            row += f"{len(all_sevs):>10}"
            print(row)


if __name__ == "__main__":
    main()
