#!/bin/bash
# compare_index_alignment.sh
# Checks whether position-in-list in an old (no sample_index) file lines
# up with the "sample_index" field in a new file, for the same dataset.
#
# Usage: ./compare_index_alignment.sh <old_file> <new_file_with_sample_index>

set -e
OLD="${1:?old file path}"
NEW="${2:?new file path}"

python3 - "$OLD" "$NEW" << 'EOF'
import json, sys

old_path, new_path = sys.argv[1], sys.argv[2]

with open(old_path) as f:
    old_data = json.load(f)
old_samples = old_data.get("samples", old_data) if isinstance(old_data, dict) else old_data

with open(new_path) as f:
    new_data = json.load(f)
new_samples = new_data.get("samples", new_data) if isinstance(new_data, dict) else new_data

new_by_index = {s["sample_index"]: s for s in new_samples if s.get("sample_index") is not None}

matches, mismatches, checked = 0, 0, 0
for i, old_s in enumerate(old_samples[:50]):
    new_s = new_by_index.get(i)
    if new_s is None:
        continue
    checked += 1
    if old_s.get("ref", "").strip() == new_s.get("ref", "").strip():
        matches += 1
    else:
        mismatches += 1
        if mismatches <= 3:
            print(f"  MISMATCH at position {i}:")
            print(f"    old ref: {old_s.get('ref','')[:80]}")
            print(f"    new ref (idx={i}): {new_s.get('ref','')[:80]}")

print(f"\nChecked {checked} positions: {matches} match, {mismatches} mismatch")
EOF
</parameter>