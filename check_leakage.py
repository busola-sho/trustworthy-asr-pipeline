import json
from src.splits import get_indices_for_split

# same aliases build_leaderboard.py already handles - older benchmark
# files predate the project's naming standardization
DATASET_NAME_ALIASES = {
    "common_voice": "commonvoice",
    "edinburgh_international_accents": "edacc",
    "english_dialects_scots": "english_dialects",
}


def normalize_dataset_name(raw_name: str) -> str:
    return DATASET_NAME_ALIASES.get(raw_name, raw_name)


# adjust this path if your candidate pool file is named/located differently
with open("candidate_pool.json") as f:
    pool = json.load(f)

print(f"Candidate pool size: {len(pool)}")

by_dataset = {}
for r in pool:
    normalized = normalize_dataset_name(r["dataset"])
    by_dataset.setdefault(normalized, []).append(r["sample_index"])

for dataset, indices in by_dataset.items():
    dev_indices = set(get_indices_for_split(dataset, "dev"))
    pool_indices = set(indices)
    overlap = pool_indices & dev_indices
    print(f"\n{dataset}:")
    print(f"  pool samples: {len(pool_indices)}")
    print(f"  dev split size: {len(dev_indices)}")
    print(f"  overlap: {len(overlap)}  ({len(overlap)/len(pool_indices)*100:.1f}% of this dataset's pool)")
