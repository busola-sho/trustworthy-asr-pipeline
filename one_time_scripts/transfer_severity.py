"""
transfer_severity.py

Transfers severity scores (and judge_raw_response) from an OLD qwen
benchmark file (already severity-judged, but no confidence/segments) onto
a NEW qwen3asr file (confidence-carrying, not yet severity-judged) -
matched by sample_index - so you don't have to re-run the Phi-4 judge
pass on samples that are actually unchanged.

SAFETY: severity is only transferred for a sample_index if ref AND hyp
match EXACTLY between the old and new file for that index. If either
differs (e.g. the new run produced a slightly different transcript for
that sample), the new file's severity is left as None/unset instead of
silently carrying over a score that no longer applies to the actual hyp -
that sample will need re-judging via add_severity_to_existing.py.

Usage:
    python transfer_severity.py \\
        --old results/benchmarks/main/qwen_edacc_20260525_204314.json \\
        --new results/benchmarks/main/qwen_edacc_20260722_merged.json \\
        --output results/benchmarks/main/qwen_edacc_20260722_merged.json
"""

import json
import argparse


def load_samples(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    samples = data.get("samples", data) if isinstance(data, dict) else data

    # Older benchmark runs (pre-sample_index tracking) don't carry a
    # "sample_index" field on each sample - without this backfill, EVERY
    # lookup against this file's samples fails (old_by_index ends up
    # empty), so every sample in the new file gets reported as "no
    # matching old entry" even when one genuinely exists at the same
    # position. Verified (compare_index_alignment.sh, 50 samples each)
    # that for commonvoice, edacc, and english_dialects, list position in
    # these older files lines up 1:1 with the sample_index used by newer
    # runs for the same dataset - same fix already applied to
    # src/selector.py's load_samples.
    for i, s in enumerate(samples):
        if s.get("sample_index") is None:
            s["sample_index"] = i

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old",    required=True, help="Old file WITH severity scores")
    parser.add_argument("--new",    required=True, help="New file (confidence-carrying), severity to be filled in")
    parser.add_argument("--output", required=True, help="Where to write the merged result (can be same as --new)")
    args = parser.parse_args()

    with open(args.new) as f:
        new_data = json.load(f)
    new_samples = new_data.get("samples", new_data) if isinstance(new_data, dict) else new_data

    old_samples = load_samples(args.old)
    old_by_index = {
        s["sample_index"]: s for s in old_samples if s.get("sample_index") is not None
    }

    transferred = 0
    mismatched = 0
    no_old_entry = 0
    already_had_severity = 0

    for sample in new_samples:
        idx = sample.get("sample_index")
        if idx is None:
            continue

        if sample.get("severity") is not None:
            already_had_severity += 1
            continue

        old_sample = old_by_index.get(idx)
        if old_sample is None:
            no_old_entry += 1
            continue

        old_ref = (old_sample.get("ref") or "").strip()
        old_hyp = (old_sample.get("hyp") or "").strip()
        new_ref = (sample.get("ref") or "").strip()
        new_hyp = (sample.get("hyp") or "").strip()

        if old_ref == new_ref and old_hyp == new_hyp and old_sample.get("severity") is not None:
            sample["severity"] = old_sample["severity"]
            if "judge_raw_response" in old_sample:
                sample["judge_raw_response"] = old_sample["judge_raw_response"]
            transferred += 1
        else:
            mismatched += 1

    if isinstance(new_data, dict):
        new_data["samples"] = new_samples
        output = new_data
    else:
        output = new_samples

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Transferred severity for {transferred} samples")
    print(f"Skipped (already had severity): {already_had_severity}")
    print(f"Skipped (ref/hyp mismatch - needs re-judging): {mismatched}")
    print(f"Skipped (no matching old entry - needs judging): {no_old_entry}")
    print(f"Saved: {args.output}")

    if mismatched or no_old_entry:
        print(f"\n{mismatched + no_old_entry} samples still need severity judging. Run:")
        print(f"  python rerunning/add_severity_to_existing.py --files {args.output}")


if __name__ == "__main__":
    main()