import json

path = "writeup_results/clean_grid/selection_naive/selection_naive_edacc_gemma4_dev.json"
data = json.load(open(path))
samples = data.get("samples", [])

total = len(samples)
severities = [s["severity"] for s in samples if s.get("severity") is not None]
n_scored = len(severities)

manual_mean = sum(severities) / n_scored if n_scored else None
stored_mean = data.get("mean_severity")

print(f"Total samples: {total}")
print(f"Scored samples (severity is not None): {n_scored}")
print(f"Manually recomputed mean (sum/n_scored): {manual_mean}")
print(f"Stored mean_severity field in file: {stored_mean}")
print()

wrong_mean_if_divided_by_total = sum(severities) / total
print(f"What it WOULD be if wrongly divided by total ({total}) instead: {wrong_mean_if_divided_by_total}")
print()

match = round(manual_mean, 3) == round(stored_mean, 3) if manual_mean and stored_mean else False
print(f"Manual recomputation matches stored value: {match}")
