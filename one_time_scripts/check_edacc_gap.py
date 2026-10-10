import json

path = "writeup_results/grid_calib_fixed/unanchored_fusion_naive/unanchored_fusion_naive_edacc_gemma4_dev.json"
data = json.load(open(path))
samples = data.get("samples", [])

print(f"Total samples in file: {len(samples)}")

skip_reasons = {}
error_reasons = {}
scored = 0

for s in samples:
    if s.get("skipped"):
        reason = s.get("skip_reason", "unknown")
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    elif s.get("error"):
        reason = s.get("error_reason", "unknown")
        error_reasons[reason] = error_reasons.get(reason, 0) + 1
    elif s.get("severity") is not None:
        scored += 1

print(f"\nScored (has severity): {scored}")
print(f"\nSkipped, by reason:")
for reason, count in skip_reasons.items():
    print(f"  {reason}: {count}")
print(f"\nErrored, by reason:")
for reason, count in error_reasons.items():
    print(f"  {reason}: {count}")

print(f"\nSanity check: {scored} + {sum(skip_reasons.values())} + {sum(error_reasons.values())} = "
      f"{scored + sum(skip_reasons.values()) + sum(error_reasons.values())} (should equal {len(samples)})")
