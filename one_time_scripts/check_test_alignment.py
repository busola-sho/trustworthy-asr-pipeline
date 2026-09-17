"""
full_results_report.py

Four tables, all using the correct pooled/micro WER (concatenate every
dataset's ref/hyp pairs, compute WER once) rather than a simple average
of each dataset's own WER percentage - averaging WER percentages across
datasets with very different sample counts (English Dialects=1780 vs
EdAcc=138) distorts the true combined error rate, since WER's edit-
distance-over-word-count ratio isn't linear. Severity's average is left
as a simple macro-average (mean of per-dataset means) since that's the
standard, defensible way to report it - but pooled severity is also
shown for reference, matching build_leaderboard.py's own convention.

1. Baseline models - dev split
2. Baseline models - test split
3. Grid - severity (macro-avg) + pooled corpus WER
4. Grid strategy comparison (best condition per strategy)

Usage:
    python full_results_report.py
    python full_results_report.py --grid-dir writeup_results/grid_calib_fixed
"""

import json
import glob
import argparse
from collections import defaultdict

from jiwer import wer as compute_wer
from src.text_normalise import normalise
from src.splits import get_indices_for_split

DATASETS = ["commonvoice", "edacc", "english_dialects"]
BASELINE_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]

DATASET_NAME_ALIASES = {
    "common_voice": "commonvoice",
    "edinburgh_international_accents": "edacc",
    "english_dialects_scots": "english_dialects",
}

MODEL_NAME_ALIASES = {
    "qwen3-asr": "qwen", "qwen3asr": "qwen",
    "whisperx": "whisperx", "parakeet": "parakeet", "wav2vec2": "wav2vec2",
}


def normalize_dataset(name):
    if not name:
        return "unknown"
    name = DATASET_NAME_ALIASES.get(name, name)
    name = name.lower()
    if "common" in name: return "commonvoice"
    if "english" in name or "dialect" in name: return "english_dialects"
    if "edacc" in name: return "edacc"
    return name


def normalize_model(name):
    lowered = name.lower()
    for key, clean in MODEL_NAME_ALIASES.items():
        if key in lowered:
            return clean
    return name


def find_baseline_path(model, dataset):
    if dataset == "shetland":
        return None
    matches = sorted(glob.glob(f"writeup_results/benchmarks/main/{model}_{dataset}_*.json"))
    matches = [m for m in matches if "sub150" not in m and "sub100" not in m]
    return matches[-1] if matches else None


def load_baseline_restricted(model, dataset, split):
    """Loads a baseline's full-dataset file, restricts to the given
    split's sample indices, returns per-sample refs/hyps/severities."""
    path = find_baseline_path(model, dataset)
    if not path:
        return None
    data = json.load(open(path))
    samples = data.get("samples", [])
    for i, s in enumerate(samples):
        if s.get("sample_index") is None:
            s["sample_index"] = i

    split_indices = set(get_indices_for_split(dataset, split))
    subset = [s for s in samples if s.get("sample_index") in split_indices]

    refs, hyps, severities = [], [], []
    for s in subset:
        if s.get("skipped") or s.get("error"):
            continue
        ref, hyp = s.get("ref"), s.get("hyp")
        if ref and hyp:
            refs.append(normalise(ref))
            hyps.append(normalise(hyp))
        if s.get("severity") is not None:
            severities.append(s["severity"])
    return {"refs": refs, "hyps": hyps, "severities": severities}


def print_baseline_table(split):
    print(f"\n{'='*100}")
    print(f"  BASELINE MODELS - {split.upper()} SPLIT (mean severity (N), pooled corpus WER)")
    print(f"  Calibration pool exclusion: ALWAYS applied here - get_indices_for_split()")
    print(f"  is called fresh each time, using the corrected candidate_pool.json.")
    print(f"{'='*100}")
    header = f"{'Model':<12}" + "".join(f"{d:>24}" for d in DATASETS) + f"{'Avg Sev':>10}{'Pooled WER':>12}{'Total N':>10}"
    print(header)
    print("-" * len(header))

    for model in BASELINE_MODELS:
        per_dataset = {}
        all_refs, all_hyps, all_sevs = [], [], []
        for d in DATASETS:
            loaded = load_baseline_restricted(model, d, split)
            if loaded is None:
                continue
            per_dataset[d] = loaded
            all_refs.extend(loaded["refs"])
            all_hyps.extend(loaded["hyps"])
            all_sevs.extend(loaded["severities"])

        row = f"{model:<12}"
        dataset_sevs = []
        for d in DATASETS:
            loaded = per_dataset.get(d)
            if loaded and loaded["severities"]:
                sev = sum(loaded["severities"]) / len(loaded["severities"])
                n = len(loaded["severities"])
                dataset_sevs.append(sev)
                row += f"{f'{sev:.3f} (N={n})':>24}"
            else:
                row += f"{'-':>24}"

        avg_sev = sum(dataset_sevs) / len(dataset_sevs) if dataset_sevs else None
        pooled_wer = compute_wer(all_refs, all_hyps) if all_refs else None

        row += f"{(f'{avg_sev:.3f}' if avg_sev is not None else '-'):>10}"
        row += f"{(f'{pooled_wer*100:.2f}%' if pooled_wer is not None else '-'):>12}"
        row += f"{len(all_sevs):>10}"
        print(row)


def load_grid_data(scan_dir):
    """Returns {folder: {dataset: {"refs":.., "hyps":.., "severities":..}}}
    - keeps per-sample data (not just the file's own summary stats) so
    pooled WER can be computed correctly across datasets."""
    GRID_FOLDERS = {
        "selection_naive":              ("selection", "naive"),
        "selection_context_v1":         ("selection", "v1"),
        "selection_context_v2":         ("selection", "v2"),
        "unanchored_fusion_naive":      ("unanchored_fusion", "naive"),
        "unanchored_fusion_context_v1": ("unanchored_fusion", "v1"),
        "unanchored_fusion_context_v2": ("unanchored_fusion", "v2"),
        "anchored_correction_naive":    ("anchored_correction", "naive"),
        "anchored_correction_v1":       ("anchored_correction", "v1"),
        "anchored_correction_v2":       ("anchored_correction", "v2"),
    }

    data = {}
    for folder, (strategy, context) in GRID_FOLDERS.items():
        data[folder] = {"strategy": strategy, "context": context, "by_dataset": {}}
        for path in glob.glob(f"{scan_dir}/{folder}/*.json"):
            try:
                d = json.load(open(path))
            except Exception:
                continue
            if d.get("split") not in ("dev", "full"):
                continue
            ds = normalize_dataset(d.get("dataset"))
            samples = d.get("samples", [])
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
            data[folder]["by_dataset"][ds] = {"refs": refs, "hyps": hyps, "severities": severities}
    return data


def print_grid_tables(grid_data, scan_dir):
    print(f"\n{'='*100}")
    print(f"  9-CELL GRID (mean severity (N) per dataset, pooled corpus WER across all 3)")
    calib_status = ("PATCHED - calibration samples excluded" if "calib_fixed" in scan_dir
                     else "NOT PATCHED - calibration samples may still be included "
                          "(this reads whatever's stored in each file; use "
                          "--grid-dir writeup_results/grid_calib_fixed for the corrected version)")
    print(f"  Calibration pool exclusion status for '{scan_dir}': {calib_status}")
    print(f"{'='*100}")
    header = f"{'Strategy':<22}{'Context':<10}" + "".join(f"{d:>22}" for d in DATASETS) + f"{'Avg Sev':>10}{'Pooled WER':>12}{'Total N':>10}"
    print(header)
    print("-" * len(header))

    strategy_results = defaultdict(dict)

    for folder, info in grid_data.items():
        by_dataset = info["by_dataset"]
        row = f"{info['strategy']:<22}{info['context']:<10}"
        sevs = []
        all_refs, all_hyps, all_sevs_flat = [], [], []
        for d in DATASETS:
            entry = by_dataset.get(d)
            if entry and entry["severities"]:
                sev = sum(entry["severities"]) / len(entry["severities"])
                n = len(entry["severities"])
                sevs.append(sev)
                row += f"{f'{sev:.3f} (N={n})':>22}"
                all_refs.extend(entry["refs"])
                all_hyps.extend(entry["hyps"])
                all_sevs_flat.extend(entry["severities"])
            else:
                row += f"{'-':>22}"

        avg_sev = sum(sevs) / len(sevs) if len(sevs) == len(DATASETS) else None
        pooled_wer = compute_wer(all_refs, all_hyps) if all_refs else None

        row += f"{(f'{avg_sev:.3f}' if avg_sev is not None else '-'):>10}"
        row += f"{(f'{pooled_wer*100:.2f}%' if pooled_wer is not None else '-'):>12}"
        row += f"{len(all_sevs_flat):>10}"
        print(row)

        if avg_sev is not None:
            strategy_results[info["strategy"]][info["context"]] = (avg_sev, pooled_wer, sevs)

    print(f"\n{'='*100}")
    print(f"  STRATEGY COMPARISON (best context condition per strategy, by severity)")
    print(f"{'='*100}")
    header = f"{'Strategy':<22}{'Best condition':<16}{'Avg Sev':>10}{'Pooled WER':>12}"
    print(header)
    print("-" * len(header))

    results = []
    for strategy, conditions in strategy_results.items():
        best_context = min(conditions, key=lambda c: conditions[c][0])
        avg_sev, pooled_wer, sevs = conditions[best_context]
        results.append((strategy, best_context, avg_sev, pooled_wer))

    results.sort(key=lambda x: x[2])
    for strategy, best_context, avg_sev, pooled_wer in results:
        wer_str = f"{pooled_wer*100:.2f}%" if pooled_wer is not None else "-"
        print(f"{strategy:<22}{best_context:<16}{avg_sev:>10.3f}{wer_str:>12}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", default="writeup_results/grid",
                        help="Directory to read grid results from")
    args = parser.parse_args()

    print_baseline_table("dev")
    print_baseline_table("test")

    grid_data = load_grid_data(args.grid_dir)
    print_grid_tables(grid_data, args.grid_dir)


if __name__ == "__main__":
    main()
