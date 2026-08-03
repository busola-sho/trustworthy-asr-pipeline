import json

path = "writeup_results/grid/selection_naive/selection_naive_english_dialects_gemma4_dev.json"
data = json.load(open(path))
samples = [s for s in data["samples"] if s.get("severity") is not None]

# sort by severity descending, take the top 10 highest-severity cases
high_sev = sorted(samples, key=lambda s: s["severity"], reverse=True)[:10]

for s in high_sev:
    print("=" * 80)
    print(f"REF:  {s['ref']}")
    print(f"HYP:  {s['hyp']}")
    print(f"WER:  {s.get('sample_WER')}")
    print(f"Severity: {s['severity']}")
    print(f"Judge said: {s.get('judge_raw_response', '(no raw response saved)')}")
    print()
