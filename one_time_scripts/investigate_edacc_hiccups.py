import json
import glob

GRID_FOLDERS = [
    "selection_naive", "selection_context_v1", "selection_context_v2",
    "unanchored_fusion_naive", "unanchored_fusion_context_v1", "unanchored_fusion_context_v2",
    "anchored_correction_naive", "anchored_correction_v1", "anchored_correction_v2",
]

for folder in GRID_FOLDERS:
    matches = glob.glob(f"writeup_results/grid_calib_fixed/{folder}/*edacc*dev.json")
    if not matches:
        print(f"{folder}: no dev file found")
        continue
    path = matches[0]
    data = json.load(open(path))
    samples = data.get("samples", [])

    scored = 0
    skip_reasons = {}
    error_details = []

    for s in samples:
        if s.get("skipped"):
            reason = s.get("skip_reason", "unknown")
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        elif s.get("error"):
            error_details.append({
                "dataset_index": s.get("dataset_index"),
                "error_reason": s.get("error_reason", "(no reason recorded)"),
            })
        elif s.get("severity") is not None:
            scored += 1

    print(f"\n{'='*80}")
    print(f"{folder}")
    print(f"{'='*80}")
    print(f"  Total samples: {len(samples)}  Scored: {scored}  Skipped: {skip_reasons}  Errors: {len(error_details)}")
    if error_details:
        print(f"  Error details:")
        for e in error_details:
            print(f"    dataset_index={e['dataset_index']}: {e['error_reason']}")
