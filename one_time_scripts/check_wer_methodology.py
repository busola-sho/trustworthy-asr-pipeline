import sys
sys.path.insert(0, ".")
from jiwer import wer
from src.judge import normalise
from src.selector import find_canonical_file, load_samples
from src.splits import get_indices_for_split

def get_indexed_samples(model, dataset):
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}

for dataset in ["commonvoice", "edacc", "english_dialects"]:
    for split in ["dev", "test"]:
        indexed = get_indexed_samples("qwen", dataset)
        indices = get_indices_for_split(dataset, split)
        subset = [indexed[i] for i in indices if i in indexed]
        valid = [s for s in subset if s.get("ref") and s.get("hyp")]

        # method 1: my fresh pooled jiwer.wer() call
        my_corpus_wer = wer(
            [normalise(s["ref"]) for s in valid],
            [normalise(s["hyp"]) for s in valid]
        )

        # method 2: average of each sample's ALREADY-STORED sample_WER field
        stored_wers = [s["sample_WER"] for s in valid if s.get("sample_WER") is not None]
        avg_stored_wer = sum(stored_wers) / len(stored_wers) if stored_wers else None

        print(f"{dataset}/{split}: my_pooled_wer={my_corpus_wer*100:.2f}%  avg_stored_sample_wer={avg_stored_wer*100:.2f}%  (n_stored={len(stored_wers)}/{len(valid)})")
