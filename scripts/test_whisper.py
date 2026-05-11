from src.models import Whisper
from src.datasets import EnglishDialectsScots

model = Whisper()
model.load()

dataset = EnglishDialectsScots()

for sample in dataset.load():
    result = model.transcribe(sample.audio, sample.sample_rate)
    print("Reference:", sample.label)
    print("Transcript:", result.text)
    break