import json
import glob

FOLDERS = [
    "writeup_results/grid/unanchored_fusion_naive_confidence_list",
    "writeup_results/grid/unanchored_fusion_naive_confidence_inline",
    "writeup_results/grid/unanchored_fusion_naive_confidence_list_full",
    "writeup_results/grid/unanchored_fusion_naive_confidence_inline_full",
    "writeup_results/grid/unanchored_fusion_naive_confidence_list_full_v2",
    "writeup_results/grid/unanchored_fusion_naive_confidence_inline_full_v2",
]

for folder in FOLDERS:
    files = sorted(glob.glob(f"{folder}/*.json"))
    print(f"\n{folder}:")
    if not files:
        print("  (no files found)")
        continue
    for path in files:
        try:
            d = json.load(open(path))
        except Exception as e:
            print(f"  {path.split('/')[-1]}: ERROR reading - {e}")
            continue
        samples = d.get("samples", [])
        scored = sum(1 for s in samples if s.get("severity") is not None)
        mean_sev = d.get("mean_severity")
        wer = d.get("corpus_wer")
        comp = d.get("compliance_first_try_rate")
        comp_str = f"  compliance={comp*100:.1f}%" if comp is not None else ""
        print(f"  {path.split('/')[-1]}: {scored}/{len(samples)} scored  "
              f"sev={mean_sev}  wer={wer}{comp_str}")
