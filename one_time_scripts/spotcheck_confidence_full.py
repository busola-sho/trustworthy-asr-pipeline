import json
import random

path = "writeup_results/grid/unanchored_fusion_naive_confidence_list_full/unanchored_fusion_naive_confidence_list_full_commonvoice_gemma4_dev.json"
data = json.load(open(path))
samples = [s for s in data["samples"] if s.get("severity") is not None]

# bias toward the worst cases, since that's where a defect would show up most clearly
worst = sorted(samples, key=lambda s: s["severity"], reverse=True)[:8]

for s in worst:
    print("=" * 90)
    print(f"REF:  {s['ref']}")
    print(f"HYP:  {s['hyp']}")
    print(f"Severity: {s['severity']}   WER: {s.get('sample_WER')}")
    print()
