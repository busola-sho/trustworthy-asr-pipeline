import json
import os

DATASETS = ["commonvoice", "edacc", "english_dialects"]
SPLIT = "dev"

for dataset in DATASETS:
    path = f"writeup_results/sentence_confidence/segments/sentence_segments_{dataset}_{SPLIT}.json"
    if not os.path.exists(path):
        print(f"{dataset}: NOT STARTED")
        continue
    data = json.load(open(path))
    n_transcripts = data.get("num_transcripts", len(data.get("transcripts", [])))
    n_segments = data.get("num_segments")
    n_parsed = data.get("num_segments_parsed")
    n_flagged = data.get("num_segments_flagged")
    print(f"{dataset}: transcripts={n_transcripts}  segments={n_segments}  "
          f"parsed={n_parsed}  flagged={n_flagged}")

