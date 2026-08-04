import json

with open("writeup_results/candidate_pool.json") as f:
    pool = json.load(f)

# what dataset name strings actually appear in the pool file, verbatim?
raw_names = set(entry.get("dataset") for entry in pool)
print("Raw dataset names found in candidate_pool.json:")
for name in sorted(raw_names):
    count = sum(1 for e in pool if e.get("dataset") == name)
    print(f"  {name!r}: {count} entries")

print()

# does this match what get_split_assignments() checks against?
CANONICAL = ["commonvoice", "edacc", "english_dialects"]
for canon in CANONICAL:
    matched = sum(1 for e in pool if e.get("dataset") == canon)
    print(f"Entries matching canonical '{canon}' via exact string equality: {matched}")
