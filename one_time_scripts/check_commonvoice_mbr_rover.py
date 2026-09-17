import json
import sys
sys.path.insert(0, ".")
from src.splits import get_indices_for_split

correct_indices = set(get_indices_for_split("commonvoice", "test"))
print(f"Correct test-split indices for commonvoice: {len(correct_indices)} total")

for label, path in [
    ("mbr_consensus", "writeup_results/ensembles/mbr_consensus/mbr_commonvoice_test.json"),
    ("rover", "writeup_results/voting/rover/rover_commonvoice_test.json"),
]:
    try:
        data = json.load(open(path))
    except FileNotFoundError:
        print(f"\n{label}: FILE NOT FOUND at {path}")
        continue

    samples = data.get("samples", [])
    actual_indices = {s.get("dataset_index") for s in samples}
    valid = [s for s in samples if s.get("sample_WER") is not None]

    print(f"\n{label}: {len(samples)} total entries, {len(valid)} with valid sample_WER")
    print(f"  Indices match correct set: {actual_indices == correct_indices}")
    if actual_indices != correct_indices:
        print(f"  Extra: {sorted(actual_indices - correct_indices)[:10]}{'...' if len(actual_indices - correct_indices) > 10 else ''}")
        print(f"  Missing: {sorted(correct_indices - actual_indices)[:10]}{'...' if len(correct_indices - actual_indices) > 10 else ''}")
