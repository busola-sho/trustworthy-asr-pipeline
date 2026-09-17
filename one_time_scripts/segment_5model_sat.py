"""
segment_5model_sat.py

Step 1 of the sentence-confidence rebuild: applies SaT (Segment Any
Text / wtpsplit) segmentation to the 5model fused transcript - the
new fixed base for all confidence methods, per the locked redesign
(fixed transcript -> fixed segmentation -> all methods on same
segments).

This script does NOT freeze anything yet - it's for manual inspection
first, per the plan: "SaT sentence segmentation -> manually inspect a
sample -> freeze boundaries". Run this, look at the actual output on
real transcripts, and only then build the full pipeline around it.

Usage:
    python segment_5model_sat.py --dataset commonvoice --n-samples 5
"""

import json
import argparse

from wtpsplit import SaT

DATASETS = ["commonvoice", "edacc", "english_dialects"]


def load_5model_transcripts(dataset, split="dev"):
    path = f"writeup_results/grid/unanchored_fusion_naive_5model/unanchored_fusion_naive_5model_{dataset}_gemma4_{split}.json"
    data = json.load(open(path))
    samples = data.get("samples", [])
    return [s for s in samples if s.get("hyp") and not s.get("skipped") and not s.get("error")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    print("Loading SaT model...")
    sat = SaT("sat-3l")  # small, fast variant - good for a first inspection pass

    samples = load_5model_transcripts(args.dataset, args.split)
    print(f"Loaded {len(samples)} valid transcripts from {args.dataset} ({args.split})\n")

    for i, s in enumerate(samples[:args.n_samples]):
        hyp = s["hyp"]
        print(f"{'='*90}")
        print(f"SAMPLE {i} (dataset_index={s.get('dataset_index')})")
        print(f"{'='*90}")
        print(f"FULL TRANSCRIPT:\n  {hyp}\n")

        sentences = sat.split(hyp)
        print(f"SaT SEGMENTATION ({len(sentences)} sentences):")
        for j, sent in enumerate(sentences):
            print(f"  [{j}] {sent!r}")
        print()


if __name__ == "__main__":
    main()
