import json
import os

DATASETS = ["commonvoice", "edacc", "english_dialects"]
VARIANTS = ["confscore", "confprobscore", "probscore"]

VERBALIZED_DIR = "writeup_results/sentence_confidence/verbalized_confidence"
MODEL_INTERNAL_DIR = "writeup_results/sentence_confidence/model_internal_confidence"
CROSSMODEL_DIR = "writeup_results/sentence_confidence/cross_model_agreement"


def check(path, label):
    if not os.path.exists(path):
        print(f"  {label}: NOT STARTED")
        return None
    try:
        data = json.load(open(path))
        n_transcripts = len(data.get("transcripts", []))
        n_segments = sum(len(t.get("segments", [])) for t in data.get("transcripts", []))
        print(f"  {label}: transcripts={n_transcripts}  segments={n_segments}")
        return n_segments
    except Exception as e:
        print(f"  {label}: ERROR reading file ({e})")
        return None


for dataset in DATASETS + ["shetland"]:
    splits = ["full"] if dataset == "shetland" else ["dev", "test"]
    for split in splits:
        print(f"\n{dataset} ({split}):")
        verbalized_counts = []
        for variant in VARIANTS:
            path = os.path.join(VERBALIZED_DIR, f"verbalized_{variant}_{dataset}_{split}.json")
            n = check(path, f"verbalized[{variant}]")
            if n is not None:
                verbalized_counts.append(n)

        n_internal = check(os.path.join(MODEL_INTERNAL_DIR, f"model_internal_{dataset}_{split}.json"), "model_internal")
        n_cross = check(os.path.join(CROSSMODEL_DIR, f"cross_model_agreement_{dataset}_{split}.json"), "crossmodel_agreement")

        # flag mismatch explicitly - this is the exact signal that
        # confirms whether the verbalized-confidence alignment fix
        # actually took effect for this dataset/split
        all_counts = verbalized_counts + [n for n in [n_internal, n_cross] if n is not None]
        if len(set(all_counts)) > 1:
            print(f"  \u26a0 MISMATCH: segment counts differ across methods {all_counts} - not yet consistent")
        elif all_counts:
            print(f"  \u2713 all methods agree: {all_counts[0]} segments")
