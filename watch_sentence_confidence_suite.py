import json
import os

RUNS = [
    ("commonvoice", "dev"),
    ("commonvoice", "test"),
    ("edacc", "dev"),
    ("edacc", "test"),
    ("english_dialects", "dev"),
    ("english_dialects", "test"),
    ("shetland", "full"),
]

for dataset, split in RUNS:
    path = f"writeup_results/sentence_confidence/segments/sentence_segments_{dataset}_{split}.json"
    if not os.path.exists(path):
        print(f"{dataset} ({split}): NOT STARTED")
        continue
    data = json.load(open(path))
    n_transcripts = data.get("num_transcripts", len(data.get("transcripts", [])))
    n_segments = data.get("num_segments")
    n_parsed = data.get("num_segments_parsed")
    n_flagged = data.get("num_segments_flagged")
    n_align_failed = data.get("num_segments_alignment_failed")
    print(f"{dataset} ({split}): transcripts={n_transcripts}  segments={n_segments}  "
          f"parsed={n_parsed}  flagged={n_flagged}  align_failed={n_align_failed}")
