from src.models import Whisper, Wav2Vec2
from src.datasets import EnglishDialectsScots

model = Wav2Vec2()
model.load()

dataset = EnglishDialectsScots()

for sample in dataset.load():
    result = model.transcribe(sample.audio, sample.sample_rate)
    print("Reference:", sample.label)
    print("Transcript:", result.text)
    break