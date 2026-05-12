from src.benchmark import run_benchmark
from src.models import Whisper
from src.datasets import EnglishDialectsScots

model = Whisper()
model.load()

dataset = EnglishDialectsScots()

results = run_benchmark(model, dataset, "results/whisper_english_dialects.json", max_samples=5)
print(results)