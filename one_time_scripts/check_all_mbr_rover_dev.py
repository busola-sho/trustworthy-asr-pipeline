import json
import sys
sys.path.insert(0, ".")
from src.splits import get_indices_for_split

for dataset in ["commonvoice", "edacc", "english_dialects"]:
    correct_indices = set(get_indices_for_split(dataset, "dev"))
    print(f"\n=== {dataset} | correct dev N = {len(correct_indices)} ===")

    for label, path in [
        ("mbr_consensus", f"writeup_results/ensembles/mbr_consensus/mbr_{dataset}_dev.json"),
        ("rover", f"writeup_results/voting/rover/rover_{dataset}_dev.json"),
    ]:
        try:
            data = json.load(open(path))
        except FileNotFoundError:
            print(f"  {label}: FILE NOT FOUND")
            continue

        samples = data.get("samples", [])
        actual_indices = {s.get("dataset_index") for s in samples}
        match = actual_indices == correct_indices
        print(f"  {label}: {len(samples)} entries, indices match: {match}")
        if not match:
            extra = actual_indices - correct_indices
            missing = correct_indices - actual_indices
            print(f"    extra={len(extra)}  missing={len(missing)}")
