"""
bucket_by_selector.py

Reorganizes writeup_results/ensembles/{approach}/*.json into
writeup_results/ensembles/{approach}/{selector}/*.json subfolders,
based on each file's OWN "selector" field read from its JSON content -
NOT parsed from the filename, since filenames have proven unreliable
(some encode selector as a suffix like "_gemma4sel_", others don't
encode it consistently at all).

Files with no "selector" field (rover, mbr_consensus - purely
mechanical, no LLM selector) go into a "_no_selector" subfolder
instead, kept separate so they're never confused with an actual
selector bucket.

DRY RUN by default - prints exactly what would move, moves nothing.
Pass --execute to actually perform the moves.

Usage:
    python bucket_by_selector.py                # dry run, shows the plan
    python bucket_by_selector.py --execute       # actually moves files
"""

import json
import os
import shutil
import argparse
from pathlib import Path

TARGET_DIRS = [
    "writeup_results/ensembles/naive",
    "writeup_results/ensembles/context_v1",
    "writeup_results/ensembles/context_v2",
    "writeup_results/ensembles/naive_confidence",
    "writeup_results/ensembles/naive_confidence_inline",
    "writeup_results/ensembles/context_v1_confidence",
    "writeup_results/ensembles/context_v1_confidence_inline",
    "writeup_results/ensembles/context_v2_confidence",
    "writeup_results/ensembles/context_v2_confidence_inline",
    "writeup_results/ensembles/whole_transcript_selection",
]


def plan_moves():
    moves = []
    for target_dir in TARGET_DIRS:
        dir_path = Path(target_dir)
        if not dir_path.exists():
            print(f"  (skip - not found: {target_dir})")
            continue

        for path in sorted(dir_path.glob("*.json")):
            try:
                data = json.load(open(path))
            except Exception as e:
                print(f"  WARNING: could not read {path}: {e}")
                continue

            selector = data.get("selector")
            bucket = selector if selector else "_no_selector"
            dest_dir = dir_path / bucket
            dest_path = dest_dir / path.name

            moves.append((path, dest_path, selector, data.get("dataset"), data.get("mean_severity")))

    return moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the moves (default: dry run only)")
    args = parser.parse_args()

    moves = plan_moves()

    print(f"\n{'='*90}")
    print(f"  {'EXECUTING' if args.execute else 'DRY RUN - would move'} {len(moves)} file(s)")
    print(f"{'='*90}")

    for src, dest, selector, dataset, sev in moves:
        sev_str = f"{sev:.3f}" if sev is not None else "-"
        print(f"  [{selector or 'NO SELECTOR':<12}] {dataset:<18} sev={sev_str:>7}  "
              f"{src.name}  ->  {dest.relative_to(src.parent.parent)}")

        if args.execute:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))

    if not args.execute:
        print(f"\nDRY RUN complete - no files were moved. Re-run with --execute to actually move them.")
    else:
        print(f"\nDone - {len(moves)} file(s) moved into selector-bucketed subfolders.")


if __name__ == "__main__":
    main()
