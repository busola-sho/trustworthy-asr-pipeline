import sys
sys.path.insert(0, ".")
from src.selector import find_canonical_file, load_samples

for model in ["qwen", "whisperx", "parakeet", "wav2vec2"]:
    path = find_canonical_file(model, "edacc")
    samples = load_samples(path)
    indexed = {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}

    for idx in [33, 68]:
        s = indexed.get(idx)
        if s:
            print(f"{model} idx={idx}: ref={s.get('ref')!r}  hyp={s.get('hyp')!r}  severity={s.get('severity')}")
