import json
import os
import glob

benchmarks_dir = "benchmarks"
naive_files = glob.glob(os.path.join(benchmarks_dir, "naive_*.json"))

print(f"{'File':<50} {'Selector':<10} {'Judge':<10} {'WER':>8} {'MAR':>8} {'N':>6}")
print("-" * 95)

for path in sorted(naive_files):
    with open(path) as f:
        data = json.load(f)
    
    selector = data.get("selector", "—")
    judge    = data.get("judge", "—")
    wer      = data.get("corpus_wer")
    mar      = data.get("meaning_alteration_rate")
    n        = data.get("num_samples", "—")
    fname    = os.path.basename(path)

    if judge != "qwen":
        continue

    wer_str = f"{wer*100:.2f}%" if wer is not None else "—"
    mar_str = f"{mar*100:.2f}%" if mar is not None else "—"

    print(f"{fname:<50} {selector:<10} {judge:<10} {wer_str:>8} {mar_str:>8} {n:>6}")