import argparse
from src.benchmark import run_benchmark
from src.models import Whisper, Wav2Vec2, Parakeet, CanaryQwen
from src.datasets import EnglishDialectsScots, CommonVoiceScots, EdAcc

MODELS = {
    "whisper": Whisper,
    "wav2vec2": Wav2Vec2,
    "parakeet": Parakeet,
    "canary": CanaryQwen,
}

DATASETS = {
    "english_dialects": EnglishDialectsScots,
    "commonvoice": CommonVoiceScots,
    "edacc": EdAcc,
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=MODELS.keys())
parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
parser.add_argument("--max_samples", type=int, default=None)
args = parser.parse_args()

model = MODELS[args.model]()
model.load()

dataset = DATASETS[args.dataset]()
output_path = f"results/{args.model}_{args.dataset}.json"

results = run_benchmark(model, dataset, output_path, max_samples=args.max_samples)
print(results)