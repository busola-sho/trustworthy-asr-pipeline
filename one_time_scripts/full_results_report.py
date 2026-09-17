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

PER-DATASET WER: each individual dataset cell now also shows that
dataset's OWN pooled WER (in brackets alongside severity/N), not just
the single aggregate WER column at the end. This is a different
calculation from the aggregate cross-dataset pooling the docstring
above warns against - a single dataset's own refs/hyps are already one
coherent group, so pooling WER within just that group has no
distortion concern; it's the same math the aggregate Pooled WER column
already uses, just applied to one dataset at a time instead of all
three combined.

ALT%: per-dataset cells and the aggregate column both show % of
severity>=2 (meaning-altering) samples, matching the locked
FLAG_THRESHOLD=2 convention used throughout this project. The
aggregate is MACRO-averaged (mean of each dataset's own %), same
convention as Avg Sev - not pooled, so a large dataset like English
Dialects doesn't dominate the aggregate purely by sample count.

GRID FOLDER WHITELIST BUG (fixed): compute_grid_index_intersection()
previously globbed EVERY subfolder under scan_dir indiscriminately,
including unrelated experiments that happen to live alongside the real
9-cell grid (confidence-threshold ablations with p5/p10/p20 filenames,
5-model variants, whisperx-replacement runs). Those experiments only
score a filtered SUBSET of samples by design, so including them in the
intersection was silently narrowing it far more than the "1-2 samples"
the function's docstring assumed (confirmed on real data: a 456->315,
31% drop for CommonVoice dev, traced to the confidence-threshold
folders being swept in). Fixed by restricting to the same explicit
GRID_FOLDERS whitelist load_grid_data() already used - now pulled out
to a single shared module-level constant so the two functions can't
drift out of sync with each other again.

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

# SINGLE SOURCE OF TRUTH for which folders count as "the real 9-cell
# grid" - used by BOTH load_grid_data() and compute_grid_index_
# intersection(), so they can never again disagree about which
# folders belong to the grid vs. some other, unrelated experiment
# sitting in a sibling folder under the same scan_dir.
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


def cell_wer_str(refs, hyps):
    """Computes pooled WER for ONE dataset's own refs/hyps - safe to
    call per-cell since it's not an average across datasets, just this
    one dataset's own micro-averaged WER."""
    if not refs:
        return None
    return compute_wer(refs, hyps)


FLAG_THRESHOLD = 2  # locked convention: severity >= 2 -> meaning-altering


def flag_rate(severities):
    """% of severities >= 2 (meaning-altering), matching the locked
    convention used throughout this project."""
    if not severities:
        return None
    return sum(1 for s in severities if s >= FLAG_THRESHOLD) / len(severities)


def find_baseline_path(model, dataset):
    if dataset == "shetland":
        return None
    matches = sorted(glob.glob(f"writeup_results/benchmarks/main/{model}_{dataset}_*.json"))
    matches = [m for m in matches if "sub150" not in m and "sub100" not in m]
    return matches[-1] if matches else None


def compute_grid_index_intersection(scan_dir, target_split, grid_folders=None):
    """For each dataset, returns the SET of dataset_index values that
    were successfully scored (severity is not None) in EVERY grid
    strategy file found for that dataset+split. Restricted to the
    GRID_FOLDERS whitelist (same one load_grid_data() uses) - NOT a
    blind glob of every subfolder under scan_dir, since that directory
    also holds other, unrelated experiments (confidence-threshold
    ablations, 5-model variants, whisperx-replacement) that only cover
    a filtered subset of samples by design. Including those in the
    intersection was silently narrowing it far more than the "1-2
    samples" this docstring originally assumed - confirmed and fixed."""
    if grid_folders is None:
        grid_folders = GRID_FOLDERS.keys()

    accepted_splits = ("dev", "full") if target_split == "dev" else ("test",)
    per_dataset_per_file_indices = defaultdict(list)  # dataset -> list of sets (one per file found)

    for folder in grid_folders:
        for path in glob.glob(f"{scan_dir}/{folder}/*.json"):
            try:
                d = json.load(open(path))
            except Exception:
                continue
            if d.get("split") not in accepted_splits:
                continue
            ds = normalize_dataset(d.get("dataset"))
            samples = d.get("samples", [])
            scored_indices = {s.get("dataset_index") for s in samples if s.get("severity") is not None}
            if scored_indices:
                per_dataset_per_file_indices[ds].append(scored_indices)

    intersection_by_dataset = {}
    for ds, index_sets in per_dataset_per_file_indices.items():
        if not index_sets:
            continue
        common = index_sets[0]
        for s in index_sets[1:]:
            common = common & s
        intersection_by_dataset[ds] = common
    return intersection_by_dataset


def load_baseline_restricted(model, dataset, split, explicit_indices=None):
    """Loads a baseline's full-dataset file, restricts to either the
    given split's own sample indices (default) or an explicit index
    set (e.g. the grid's intersection, for direct comparability),
    returns per-sample refs/hyps/severities."""
    path = find_baseline_path(model, dataset)
    if not path:
        return None
    data = json.load(open(path))
    samples = data.get("samples", [])
    for i, s in enumerate(samples):
        if s.get("sample_index") is None:
            s["sample_index"] = i

    if explicit_indices is not None:
        target_indices = explicit_indices
    else:
        target_indices = set(get_indices_for_split(dataset, split))

    subset = [s for s in samples if s.get("sample_index") in target_indices]

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


def print_baseline_table(split, grid_indices=None):
    print(f"\n{'='*100}")
    if grid_indices:
        print(f"  BASELINE MODELS - {split.upper()} SPLIT, RESTRICTED TO GRID'S EXACT SAMPLE SET")
        print(f"  (mean severity (N, WER%), pooled corpus WER)")
        print(f"  Each dataset restricted to the INTERSECTION of sample indices actually")
        print(f"  scored across every grid strategy for that dataset+split - so these numbers")
        print(f"  are directly, exactly comparable to every row in the grid table below.")
    else:
        print(f"  BASELINE MODELS - {split.upper()} SPLIT (mean severity (N, WER%), pooled corpus WER)")
        print(f"  Calibration pool exclusion: ALWAYS applied here - get_indices_for_split()")
        print(f"  is called fresh each time, using the corrected candidate_pool.json.")
    print(f"{'='*100}")
    header = f"{'Model':<12}" + "".join(f"{d:>44}" for d in DATASETS) + f"{'Avg Sev':>10}{'Pooled WER':>12}{'Alt%':>10}{'Total N':>10}"
    print(header)
    print("-" * len(header))

    for model in BASELINE_MODELS:
        per_dataset = {}
        all_refs, all_hyps, all_sevs = [], [], []
        for d in DATASETS:
            explicit = grid_indices.get(d) if grid_indices else None
            loaded = load_baseline_restricted(model, d, split, explicit_indices=explicit)
            if loaded is None:
                continue
            per_dataset[d] = loaded
            all_refs.extend(loaded["refs"])
            all_hyps.extend(loaded["hyps"])
            all_sevs.extend(loaded["severities"])

        row = f"{model:<12}"
        dataset_sevs = []
        dataset_flags = []
        for d in DATASETS:
            loaded = per_dataset.get(d)
            if loaded and loaded["severities"]:
                sev = sum(loaded["severities"]) / len(loaded["severities"])
                n = len(loaded["severities"])
                dataset_sevs.append(sev)
                cell_wer = cell_wer_str(loaded["refs"], loaded["hyps"])
                cell_flag = flag_rate(loaded["severities"])
                dataset_flags.append(cell_flag)
                wer_part = f", WER={cell_wer*100:.2f}%" if cell_wer is not None else ""
                flag_part = f", Alt={cell_flag*100:.1f}%" if cell_flag is not None else ""
                row += f"{f'{sev:.3f} (N={n}{wer_part}{flag_part})':>44}"
            else:
                row += f"{'-':>44}"

        avg_sev = sum(dataset_sevs) / len(dataset_sevs) if dataset_sevs else None
        pooled_wer = compute_wer(all_refs, all_hyps) if all_refs else None
        overall_flag = sum(dataset_flags) / len(dataset_flags) if dataset_flags else None

        row += f"{(f'{avg_sev:.3f}' if avg_sev is not None else '-'):>10}"
        row += f"{(f'{pooled_wer*100:.2f}%' if pooled_wer is not None else '-'):>12}"
        row += f"{(f'{overall_flag*100:.1f}%' if overall_flag is not None else '-'):>10}"
        row += f"{len(all_sevs):>10}"
        print(row)


def load_grid_data(scan_dir, target_split="dev"):
    """Returns {folder: {dataset: {"refs":.., "hyps":.., "severities":..}}}
    - keeps per-sample data (not just the file's own summary stats) so
    pooled WER can be computed correctly across datasets.
    target_split: "dev" (also matches "full", for Shetland-style files)
    or "test". Folders/files with no data for the requested split are
    simply absent from the result - most grid cells only have dev so
    far, only your confirmed top strategies have test."""
    accepted_splits = ("dev", "full") if target_split == "dev" else ("test",)

    data = {}
    for folder, (strategy, context) in GRID_FOLDERS.items():
        data[folder] = {"strategy": strategy, "context": context, "by_dataset": {}}
        for path in glob.glob(f"{scan_dir}/{folder}/*.json"):
            try:
                d = json.load(open(path))
            except Exception:
                continue
            if d.get("split") not in accepted_splits:
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


def print_grid_tables(grid_data, scan_dir, label="DEV"):
    print(f"\n{'='*100}")
    print(f"  9-CELL GRID - {label} SPLIT (mean severity (N, WER%) per dataset, pooled corpus WER across all 3)")
    if label == "TEST":
        print(f"  NOTE: only strategies you've explicitly run on test will show rows here -")
        print(f"  most grid cells only have dev results so far.")
    calib_status = ("PATCHED - calibration samples excluded" if "calib_fixed" in scan_dir
                     else "NOT PATCHED - calibration samples may still be included "
                          "(this reads whatever's stored in each file; use "
                          "--grid-dir writeup_results/grid_calib_fixed for the corrected version)")
    print(f"  Calibration pool exclusion status for '{scan_dir}': {calib_status}")
    print(f"{'='*100}")
    header = f"{'Strategy':<22}{'Context':<10}" + "".join(f"{d:>42}" for d in DATASETS) + f"{'Avg Sev':>10}{'Pooled WER':>12}{'Alt%':>10}{'Total N':>10}"
    print(header)
    print("-" * len(header))

    strategy_results = defaultdict(dict)
    any_rows = False

    for folder, info in grid_data.items():
        by_dataset = info["by_dataset"]
        if not by_dataset:
            continue  # no data at all for this split - skip the row entirely, don't print an all-dash line
        any_rows = True
        row = f"{info['strategy']:<22}{info['context']:<10}"
        sevs = []
        flags = []
        all_refs, all_hyps, all_sevs_flat = [], [], []
        for d in DATASETS:
            entry = by_dataset.get(d)
            if entry and entry["severities"]:
                sev = sum(entry["severities"]) / len(entry["severities"])
                n = len(entry["severities"])
                sevs.append(sev)
                cell_wer = cell_wer_str(entry["refs"], entry["hyps"])
                cell_flag = flag_rate(entry["severities"])
                flags.append(cell_flag)
                wer_part = f", WER={cell_wer*100:.2f}%" if cell_wer is not None else ""
                flag_part = f", Alt={cell_flag*100:.1f}%" if cell_flag is not None else ""
                row += f"{f'{sev:.3f} (N={n}{wer_part}{flag_part})':>42}"
                all_refs.extend(entry["refs"])
                all_hyps.extend(entry["hyps"])
                all_sevs_flat.extend(entry["severities"])
            else:
                row += f"{'-':>42}"

        avg_sev = sum(sevs) / len(sevs) if len(sevs) == len(DATASETS) else None
        pooled_wer = compute_wer(all_refs, all_hyps) if all_refs else None
        overall_flag = sum(flags) / len(flags) if len(flags) == len(DATASETS) else None

        row += f"{(f'{avg_sev:.3f}' if avg_sev is not None else '-'):>10}"
        row += f"{(f'{pooled_wer*100:.2f}%' if pooled_wer is not None else '-'):>12}"
        row += f"{(f'{overall_flag*100:.1f}%' if overall_flag is not None else '-'):>10}"
        row += f"{len(all_sevs_flat):>10}"
        print(row)

        if avg_sev is not None:
            strategy_results[info["strategy"]][info["context"]] = (avg_sev, pooled_wer, sevs)

    if not any_rows:
        print("  (no results found for this split)")
        return

    print(f"\n{'='*100}")
    print(f"  STRATEGY COMPARISON - {label} SPLIT (best context condition per strategy, by severity)")
    print(f"{'='*100}")
    header = f"{'Strategy':<22}{'Best condition':<16}{'Avg Sev':>10}{'Pooled WER':>12}"
    print(header)
    print("-" * len(header))

    results = []
    for strategy, conditions in strategy_results.items():
        best_context = min(conditions, key=lambda c: conditions[c][0])
        avg_sev, pooled_wer, sevs = conditions[best_context]
        results.append((strategy, best_context, avg_sev, pooled_wer))

    if not results:
        print("  (no strategy has complete 3-dataset results for this split yet)")
        return

    results.sort(key=lambda x: x[2])
    for strategy, best_context, avg_sev, pooled_wer in results:
        wer_str = f"{pooled_wer*100:.2f}%" if pooled_wer is not None else "-"
        print(f"{strategy:<22}{best_context:<16}{avg_sev:>10.3f}{wer_str:>12}")


def find_mechanical_path(technique, dataset, split):
    """rover/mbr_consensus files - no selector, purely mechanical."""
    if technique == "rover":
        matches = glob.glob(f"writeup_results/voting/rover/rover_{dataset}_{split}.json")
    elif technique == "mbr_consensus":
        matches = glob.glob(f"writeup_results/ensembles/mbr_consensus/mbr_{dataset}_{split}.json")
    else:
        return None
    return matches[0] if matches else None


def load_mechanical_restricted(technique, dataset, split, explicit_indices):
    """Loads rover/mbr_consensus, restricted to an explicit index set
    (the grid's intersection), same treatment as the baseline models -
    so ROVER/MBR are compared on the EXACT same samples as the grid,
    not their own independently-sized full run."""
    path = find_mechanical_path(technique, dataset, split)
    if not path:
        return None
    data = json.load(open(path))
    samples = data.get("samples", [])

    subset = [s for s in samples if s.get("dataset_index") in explicit_indices]

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


def print_mechanical_table(split, grid_indices, grid_data):
    print(f"\n{'='*100}")
    print(f"  ROVER / MBR CONSENSUS vs GRID - {split.upper()} SPLIT, SAME EXACT SAMPLE SET")
    print(f"  Both restricted to the same grid-intersection indices as the baseline comparison above,")
    print(f"  so this is a fair, apples-to-apples check against the grid's winning strategies.")
    print(f"{'='*100}")
    header = f"{'Technique':<26}" + "".join(f"{d:>44}" for d in DATASETS) + f"{'Avg Sev':>10}{'Pooled WER':>12}{'Alt%':>10}{'Total N':>10}"
    print(header)
    print("-" * len(header))

    rows = []

    for technique in ["rover", "mbr_consensus"]:
        dataset_sevs = []
        dataset_flags = []
        all_refs, all_hyps, all_sevs = [], [], []
        row = f"{technique:<26}"
        for d in DATASETS:
            explicit = grid_indices.get(d, set())
            loaded = load_mechanical_restricted(technique, d, split, explicit)
            if loaded and loaded["severities"]:
                sev = sum(loaded["severities"]) / len(loaded["severities"])
                n = len(loaded["severities"])
                dataset_sevs.append(sev)
                cell_wer = cell_wer_str(loaded["refs"], loaded["hyps"])
                cell_flag = flag_rate(loaded["severities"])
                dataset_flags.append(cell_flag)
                wer_part = f", WER={cell_wer*100:.2f}%" if cell_wer is not None else ""
                flag_part = f", Alt={cell_flag*100:.1f}%" if cell_flag is not None else ""
                row += f"{f'{sev:.3f} (N={n}{wer_part}{flag_part})':>44}"
                all_refs.extend(loaded["refs"])
                all_hyps.extend(loaded["hyps"])
                all_sevs.extend(loaded["severities"])
            else:
                row += f"{'-':>44}"

        avg_sev = sum(dataset_sevs) / len(dataset_sevs) if dataset_sevs else None
        pooled_wer = compute_wer(all_refs, all_hyps) if all_refs else None
        overall_flag = sum(dataset_flags) / len(dataset_flags) if dataset_flags else None
        row += f"{(f'{avg_sev:.3f}' if avg_sev is not None else '-'):>10}"
        row += f"{(f'{pooled_wer*100:.2f}%' if pooled_wer is not None else '-'):>12}"
        row += f"{(f'{overall_flag*100:.1f}%' if overall_flag is not None else '-'):>10}"
        row += f"{len(all_sevs):>10}"
        print(row)
        if avg_sev is not None:
            rows.append((technique, avg_sev, pooled_wer))

    if rows:
        best_grid_label, best_grid_sev = None, None
        for folder, info in grid_data.items():
            by_dataset = info["by_dataset"]
            sevs = [sum(by_dataset[d]["severities"]) / len(by_dataset[d]["severities"])
                    for d in DATASETS if by_dataset.get(d) and by_dataset[d]["severities"]]
            if len(sevs) == len(DATASETS):
                avg = sum(sevs) / len(sevs)
                if best_grid_sev is None or avg < best_grid_sev:
                    best_grid_sev = avg
                    best_grid_label = f"{info['strategy']} + {info['context']}"

        print(f"\n  Best grid strategy overall (unrestricted comparison, see grid table above): "
              f"{best_grid_label} (avg severity {best_grid_sev:.3f})")
        for technique, avg_sev, pooled_wer in sorted(rows, key=lambda x: x[1]):
            verdict = "grid wins" if best_grid_sev < avg_sev else "mechanical wins"
            print(f"  {technique}: {avg_sev:.3f} vs grid's {best_grid_sev:.3f} -> {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", default="writeup_results/grid",
                        help="Directory to read grid results from")
    args = parser.parse_args()

    print_baseline_table("dev")
    print_baseline_table("test")

    grid_data_dev = load_grid_data(args.grid_dir, target_split="dev")
    print_grid_tables(grid_data_dev, args.grid_dir, label="DEV")

    grid_data_test = load_grid_data(args.grid_dir, target_split="test")
    print_grid_tables(grid_data_test, args.grid_dir, label="TEST")

    # baseline restricted to the EXACT same sample indices as the grid,
    # so the two are finally directly comparable rather than each
    # independently deriving a slightly different split
    dev_grid_indices = compute_grid_index_intersection(args.grid_dir, "dev")
    if dev_grid_indices:
        print_baseline_table("dev", grid_indices=dev_grid_indices)
        print_mechanical_table("dev", dev_grid_indices, grid_data_dev)

    test_grid_indices = compute_grid_index_intersection(args.grid_dir, "test")
    if test_grid_indices:
        print_baseline_table("test", grid_indices=test_grid_indices)
        print_mechanical_table("test", test_grid_indices, grid_data_test)


if __name__ == "__main__":
    main()