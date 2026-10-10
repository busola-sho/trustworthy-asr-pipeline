"""
patch_calibration_leakage.py

Fixes the calibration-leakage bug WITHOUT rerunning any experiments.
The bug: candidate_pool.json stores dataset names using stale aliases
(common_voice, edinburgh_international_accents, english_dialects_scots)
that never matched the canonical names (commonvoice, edacc,
english_dialects) used everywhere else - so get_split_assignments()'s
exclusion check silently matched ZERO calibration samples, for every
dataset, meaning all ~100 judge-calibration samples have been sitting
inside dev/test the whole time.

This script does NOT re-run any selector/fusion calls. It mirrors an
entire directory tree (e.g. writeup_results/grid/) into a SIBLING
folder with "_calib_fixed" appended (e.g.
writeup_results/grid_calib_fixed/), preserving the full structure -
every file is copied across, and any file whose samples include
genuine calibration indices gets those samples removed and its
mean_severity/corpus_wer recomputed before being written to the new
tree. Files with no contamination are copied unchanged. The original
--scan-dir is never touched.

This does NOT reproduce the exact dev/test split a from-scratch
corrected run would produce (the shuffle would have operated on a
different, smaller "remaining" pool) - it removes contamination from
your EXISTING results, which is a legitimate, far cheaper fix.

Usage:
    python patch_calibration_leakage.py --scan-dir writeup_results/grid
    python patch_calibration_leakage.py --scan-dir writeup_results/ensembles
"""

import json
import os
import shutil
import glob
import argparse
from pathlib import Path

from jiwer import wer as compute_wer
from src.text_normalise import normalise

DATASET_NAME_ALIASES = {
    "common_voice": "commonvoice",
    "edinburgh_international_accents": "edacc",
    "english_dialects_scots": "english_dialects",
}


def normalize_dataset_name(raw_name: str) -> str:
    return DATASET_NAME_ALIASES.get(raw_name, raw_name)


def load_calibration_indices(pool_path="writeup_results/candidate_pool.json"):
    with open(pool_path) as f:
        pool = json.load(f)
    by_dataset = {}
    for entry in pool:
        dataset = normalize_dataset_name(entry.get("dataset"))
        idx = entry.get("sample_index")
        if idx is not None:
            by_dataset.setdefault(dataset, set()).add(idx)
    return by_dataset


def patch_or_copy_file(src_path, dest_path, calibration_by_dataset):
    """Returns a change-report dict if the file was patched (contamination
    found and removed), or None if it was just copied unchanged."""
    try:
        data = json.load(open(src_path))
    except Exception:
        shutil.copy2(src_path, dest_path)
        return None

    dataset = normalize_dataset_name(data.get("dataset", "unknown"))
    split = data.get("split")
    calibration_indices = calibration_by_dataset.get(dataset, set())

    if split not in ("dev", "test") or not calibration_indices:
        shutil.copy2(src_path, dest_path)
        return None

    samples = data.get("samples", [])
    contaminated = [s for s in samples if s.get("dataset_index") in calibration_indices]

    if not contaminated:
        shutil.copy2(src_path, dest_path)
        return None

    n_before = len(samples)
    clean_samples = [s for s in samples if s.get("dataset_index") not in calibration_indices]

    valid = [s for s in clean_samples if not s.get("skipped") and not s.get("error")
             and s.get("sample_WER") is not None]
    new_wer = compute_wer([normalise(s["ref"]) for s in valid], [normalise(s["hyp"]) for s in valid]) if valid else None
    severities = [s["severity"] for s in valid if s.get("severity") is not None]
    new_mean_sev = sum(severities) / len(severities) if severities else None

    old_wer = data.get("corpus_wer")
    old_mean_sev = data.get("mean_severity")

    data["samples"] = clean_samples
    data["corpus_wer"] = new_wer
    data["mean_severity"] = new_mean_sev
    data["num_samples"] = len(valid)
    data["_calibration_leakage_patch"] = {
        "removed_count": len(contaminated),
        "removed_indices": sorted(s.get("dataset_index") for s in contaminated),
        "old_corpus_wer": old_wer,
        "old_mean_severity": old_mean_sev,
    }

    with open(dest_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {
        "path": src_path,
        "removed": len(contaminated),
        "n_before": n_before,
        "n_after": len(clean_samples),
        "old_wer": old_wer, "new_wer": new_wer,
        "old_sev": old_mean_sev, "new_sev": new_mean_sev,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", required=True,
                        help="Directory to mirror (e.g. writeup_results/grid)")
    parser.add_argument("--pool-path", default="writeup_results/candidate_pool.json")
    args = parser.parse_args()

    scan_dir = args.scan_dir.rstrip("/")
    dest_dir = scan_dir + "_calib_fixed"

    calibration_by_dataset = load_calibration_indices(args.pool_path)
    print("Calibration sample counts per dataset (corrected matching):")
    for ds, indices in calibration_by_dataset.items():
        print(f"  {ds}: {len(indices)}")

    print(f"\nMirroring {scan_dir} -> {dest_dir} ...")

    changes = []
    n_copied_unchanged = 0

    for src_path in glob.glob(os.path.join(scan_dir, "**", "*"), recursive=True):
        if os.path.isdir(src_path):
            continue
        rel_path = os.path.relpath(src_path, scan_dir)
        dest_path = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if src_path.endswith(".json"):
            result = patch_or_copy_file(src_path, dest_path, calibration_by_dataset)
            if result:
                changes.append(result)
            else:
                n_copied_unchanged += 1
        else:
            shutil.copy2(src_path, dest_path)

    print(f"\n{'='*100}")
    print(f"  Mirror complete: {dest_dir}")
    print(f"  {len(changes)} file(s) patched, {n_copied_unchanged} file(s) copied unchanged")
    print(f"{'='*100}")
    for c in changes:
        old_wer_str = f"{c['old_wer']*100:.2f}%" if c['old_wer'] is not None else "-"
        new_wer_str = f"{c['new_wer']*100:.2f}%" if c['new_wer'] is not None else "-"
        old_sev_str = f"{c['old_sev']:.3f}" if c['old_sev'] is not None else "-"
        new_sev_str = f"{c['new_sev']:.3f}" if c['new_sev'] is not None else "-"
        print(f"\n{c['path']}")
        print(f"  removed {c['removed']} contaminated sample(s) ({c['n_before']} -> {c['n_after']})")
        print(f"  WER:      {old_wer_str} -> {new_wer_str}")
        print(f"  Severity: {old_sev_str} -> {new_sev_str}")


if __name__ == "__main__":
    main()
