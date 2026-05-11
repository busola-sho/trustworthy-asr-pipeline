from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional, Generator
import librosa
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from itertools import chain

@dataclass
class Sample:
    audio: np.ndarray
    sample_rate:int
    label: str

class Dataset(ABC):
    def __init__(self, name:str, dataset_path:Optional[str]=None):
        self.name=name
        self.data_path=dataset_path
    
    @abstractmethod
    def load(self)->Generator[Sample, None, None]:
        pass

class CommonVoiceScots(Dataset): # A specific conversational scots dataset
    def __init__(self, path="data/common-voice-scots"):
        super().__init__(name="common_voice", dataset_path=path)

    def load(self):
        audio_dir = Path(self.data_path) / "audios"
        label_path = Path(self.data_path) / "ss-corpus-sco.tsv"
        df=pd.read_csv(label_path, sep="\t")

        for _, row in df.iterrows():
            audio_path=audio_dir/row["audio_file"]
            label=row["transcription"]
            audio_array, sample_rate = librosa.load(audio_path,sr=None)
            sample=Sample(audio=audio_array,sample_rate=sample_rate,label=label)
            yield sample

class EnglishDialectsScots(Dataset): # A scottish read dataset
    def __init__(self):
        super().__init__(name="english_dialects_scots")
    
    def load(self):
        dataset_f = load_dataset("ylacombe/english_dialects", "scottish_female", split="train", streaming=True)
        dataset_m = load_dataset("ylacombe/english_dialects", "scottish_male", split="train", streaming=True)
        combined = chain(dataset_f, dataset_m)

        for row in combined:
            audio_array = row['audio']['array']
            sample_rate = row['audio']['sampling_rate']
            label=row['text']
            sample=Sample(audio=audio_array,sample_rate=sample_rate,label=label)
            yield sample

class EdAcc(Dataset): # A more general accent-diverse dataset
    def __init__(self):
        super().__init__(name="edinburgh_international_accents")
    
    def load(self):
        edacc = load_dataset("edinburghcstr/edacc", split="test", streaming=True)
        for row in edacc:
            audio_array = row['audio']['array']
            sample_rate = row['audio']['sampling_rate']
            label=row['text']
            sample=Sample(audio=audio_array,sample_rate=sample_rate,label=label)
            yield sample

