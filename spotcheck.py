import json
import random

path = "writeup_results/grid/selection_naive/selection_naive_english_dialects_gemma4_dev.json"
data = json.load(open(path))
samples = [s for s in data["samples"] if s.get("severity") is not None]

random.seed(1)
sample_check = random.sample(samples, min(10, len(samples)))

for s in sample_check:
    print("=" * 80)
    print(f"REF:  {s['ref']}")
    print(f"HYP:  {s['hyp']}")
    print(f"Severity: {s['severity']}")
    print(f"Judge said: {s.get('judge_raw_response', '(no raw response saved)')}")
    print()
