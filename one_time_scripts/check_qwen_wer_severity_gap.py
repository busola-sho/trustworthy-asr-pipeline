import sys
sys.path.insert(0, ".")
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

        valid = [s for s in subset if s.get("severity") is not None and s.get("ref") and s.get("hyp")]
        wer_valid = [s for s in subset if s.get("ref") and s.get("hyp")]

        n_valid = len(valid)
        n_wer_valid = len(wer_valid)

        print(f"{dataset}/{split}: valid(severity)={n_valid}  wer_valid={n_wer_valid}  diff={n_wer_valid - n_valid}")

        if n_wer_valid != n_valid:
            valid_indices = {s["sample_index"] for s in valid}
            extra = [s for s in wer_valid if s["sample_index"] not in valid_indices]
            for s in extra[:5]:
                print(f"    EXTRA (in wer_valid but not valid): idx={s['sample_index']}  ref={s.get('ref')[:40]!r}  hyp={s.get('hyp')[:40]!r}  severity={s.get('severity')}")

