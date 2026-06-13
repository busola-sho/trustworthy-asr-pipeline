"""
scripts/benchmarking/inference/run_benchmark.py

Runs a model on a dataset with confidence score extraction.
Supports full dataset or a fixed random subset.

Usage:
    # full dataset
    python scripts/benchmarking/inference/run_benchmark.py --model whisper --dataset commonvoice

    # 150-sample subset (reproducible via seed)
    python scripts/benchmarking/inference/run_benchmark.py --model whisper --dataset commonvoice --subset 150

    # resume
    python scripts/benchmarking/inference/run_benchmark.py --model whisper --dataset commonvoice --subset 150 --output_path results/benchmarks/subsets/whisper_commonvoice_sub150.json
"""

import argparse
import random
import json
import os
from datetime import datetime

from src.benchmark import run_benchmark
from src.models import Whisper, Wav2Vec2, Parakeet, Qwen3ASR
from src.datasets import EnglishDialectsScots, CommonVoiceScots, EdAcc

MODELS = {
    "whisper":   Whisper,
    "wav2vec2":  Wav2Vec2,
    "parakeet":  Parakeet,
    "qwen3asr":  Qwen3ASR,
}

DATASETS = {
    "english_dialects": EnglishDialectsScots,
    "commonvoice":      CommonVoiceScots,
    "edacc":            EdAcc,
}

DATASET_SIZES = {
    "commonvoice":      680,
    "english_dialects": 2543,
    "edacc":            198,
}

parser = argparse.ArgumentParser()
parser.add_argument("--model",       required=True, choices=MODELS.keys())
parser.add_argument("--dataset",     required=True, choices=DATASETS.keys())
parser.add_argument("--subset",      type=int, default=None,
                    help="Number of samples for subset run (e.g. 150)")
parser.add_argument("--seed",        type=int, default=42,
                    help="Random seed for subset selection (default: 42)")
parser.add_argument("--max_samples", type=int, default=None,
                    help="Max samples for full run (ignored if --subset given)")
parser.add_argument("--start_from",  type=int, default=0)
parser.add_argument("--output_path", type=str, default=None)
args = parser.parse_args()

# build output path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.output_path:
    output_path = args.output_path
elif args.subset:
    os.makedirs("results/benchmarks/subsets", exist_ok=True)
    output_path = f"results/benchmarks/subsets/{args.model}_{args.dataset}_sub{args.subset}.json"
else:
    os.makedirs("results/benchmarks/main", exist_ok=True)
    output_path = f"results/benchmarks/main/{args.model}_{args.dataset}_{timestamp}.json"

# compute subset indices if needed
subset_indices = None
if args.subset:
    n_total = DATASET_SIZES.get(args.dataset, 10000)
    n = min(args.subset, n_total)
    random.seed(args.seed)
    subset_indices = sorted(random.sample(range(n_total), n))
    print(f"Subset: {n} indices sampled from {n_total} (seed={args.seed})")

    # if resuming, load existing indices to stay consistent
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        if existing.get("subset_indices"):
            subset_indices = existing["subset_indices"]
            print(f"Resumed: using existing subset indices from {output_path}")

# load model and dataset
model = MODELS[args.model]()
model.load()
dataset = DATASETS[args.dataset]()

print(f"Running {args.model} on {args.dataset} → {output_path}")

results = run_benchmark(
    model,
    dataset,
    output_path,
    max_samples=args.max_samples,
    start_from=args.start_from,
    subset_indices=subset_indices,
)
print(results)