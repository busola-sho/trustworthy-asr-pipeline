# import argparse
# from src.benchmark import run_benchmark
# from src.models import Whisper, Wav2Vec2, Parakeet, WavLM, HuBERT, Qwen3ASR
# from src.datasets import EnglishDialectsScots, CommonVoiceScots, EdAcc
# from datetime import datetime


# MODELS = {
#     "whisper": Whisper,
#     "wav2vec2": Wav2Vec2,
#     "parakeet": Parakeet,
#     "wavlm": WavLM,
#     "hubert": HuBERT,
#     "qwen": Qwen3ASR
# }

# DATASETS = {
#     "english_dialects": EnglishDialectsScots,
#     "commonvoice": CommonVoiceScots,
#     "edacc": EdAcc,
# }

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", required=True, choices=MODELS.keys())
# parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
# parser.add_argument("--max_samples", type=int, default=None)
# args = parser.parse_args()

# model = MODELS[args.model]()
# model.load()

# dataset = DATASETS[args.dataset]()
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_path = f"benchmarks/{args.model}_{args.dataset}_{timestamp}.json"
# # output_path = f"results/{args.model}_{args.dataset}.json"

# results = run_benchmark(model, dataset, output_path, max_samples=args.max_samples)
# print(results)

import argparse
from datetime import datetime
from src.benchmark import run_benchmark
from src.models import Whisper, Wav2Vec2, Parakeet, Qwen3ASR
from src.datasets import EnglishDialectsScots, CommonVoiceScots, EdAcc

MODELS = {
    "whisper": Whisper,
    "wav2vec2": Wav2Vec2,
    "parakeet": Parakeet,
    "qwen3asr": Qwen3ASR,
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
parser.add_argument("--start_from", type=int, default=0)
parser.add_argument("--output_path", type=str, default=None)
args = parser.parse_args()

model = MODELS[args.model]()
model.load()

dataset = DATASETS[args.dataset]()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = args.output_path or f"benchmarks/{args.model}_{args.dataset}_{timestamp}.json"

results = run_benchmark(model, dataset, output_path, max_samples=args.max_samples, start_from=args.start_from)
print(results)