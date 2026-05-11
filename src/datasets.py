from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional, Generator
import librosa
from pathlib import Path
import pandas as pd


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

class CommonVoice(Dataset):
    def __init__(self, path="data/common-voice-scots"):
        super().__init__(name="common_voice", dataset_path=path)

    def load(self):
        audio_dir = Path(self.data_path) / "audios"
        label_path = Path(self.data_path) / "ss-corpus-sco.tsv"
        df=pd.read_csv(label_path)

        for _, row in df.iterrows():
            audio_path=audio_dir/row["audio_file"]
            label=row["transcription"]
            audio_array, sample_rate = librosa.load(audio_path,sr=None)
            sample=Sample(audio=audio_array,sample_rate=sample_rate,label=label)
            yield sample
