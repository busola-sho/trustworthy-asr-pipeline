from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional, Generator
import librosa
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from itertools import chain
import os

@dataclass
class Sample:
    audio: np.ndarray
    sample_rate: int
    label: str


class Dataset(ABC):
    def __init__(self, name: str, dataset_path: Optional[str] = None):
        self.name = name
        self.data_path = dataset_path

    @abstractmethod
    def load(self) -> Generator[Sample, None, None]:
        pass

class CommonVoiceScots(Dataset):
    def __init__(self, path="data/common-voice-scots"):
        super().__init__(name="common_voice", dataset_path=path)

    def load(self):
        audio_dir = Path(self.data_path) / "audios"
        label_path = Path(self.data_path) / "ss-corpus-sco.tsv"
        df = pd.read_csv(label_path, sep="\t").dropna(subset=["transcription"])

        for _, row in df.iterrows():
            audio_path = audio_dir / row["audio_file"]
            label = row["transcription"]
            audio_array, sample_rate = librosa.load(audio_path, sr=None)
            yield Sample(audio=audio_array, sample_rate=sample_rate, label=label)


class EnglishDialectsScots(Dataset):
    def __init__(self):
        super().__init__(name="english_dialects_scots")

    def load(self):
        # cached on first download, reads from disk after that
        dataset_f = load_dataset("ylacombe/english_dialects", "scottish_female", split="train")
        dataset_m = load_dataset("ylacombe/english_dialects", "scottish_male", split="train")
        combined = chain(dataset_f, dataset_m)

        for row in combined:
            audio_array = row['audio']['array']
            sample_rate = row['audio']['sampling_rate']
            label = row['text']
            yield Sample(audio=audio_array, sample_rate=sample_rate, label=label)


class EdAcc(Dataset):
    def __init__(self):
        super().__init__(name="edinburgh_international_accents")

    def load(self):
        # cached on first download, reads from disk after that
        dataset = load_dataset("edinburghcstr/edacc", split="test")

        for row in dataset:
            if row.get("accent") != "Scottish English":
                continue
            if row.get("text", "").startswith("IGNORE"):
                continue
            audio_array = row['audio']['array']
            sample_rate = row['audio']['sampling_rate']
            label = row['text']
            yield Sample(audio=audio_array, sample_rate=sample_rate, label=label)

class Shetland(Dataset):
    """
    Shetland dialect dataset from data/shetland/.
    Audio files in data/shetland/audios/
    Transcripts in data/shetland/shetland.xlsx
    
    Columns used: audio_file, transcript
    """
    def __init__(self, path="data/shetland"):
        super().__init__(name="shetland")
        self.path       = path
        self.audio_dir  = os.path.join(path, "audios")
        self.excel_path = os.path.join(path, "shetland.xlsx")

    def load(self):
        import pandas as pd
        import librosa

        df = pd.read_excel(self.excel_path)
        df = df.dropna(subset=["transcript"])

        for _, row in df.iterrows():
            audio_file = str(row["audio_file"]).strip()
            transcript = str(row["transcript"]).strip()

            if not transcript or transcript == "nan":
                continue

            audio_path = os.path.join(self.audio_dir, audio_file)
            if not os.path.exists(audio_path):
                continue

            audio_array, sample_rate = librosa.load(audio_path, sr=None, mono=True)

            yield Sample(
                audio=audio_array,
                sample_rate=sample_rate,
                label=transcript,
            )
