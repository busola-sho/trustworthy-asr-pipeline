from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Segment:
    word: str
    start: float
    end: float
    confidence: float

@dataclass
class Transcription:
    segments: list[Segment]
    transcript: str
    model_name: str

class ASRModel(ABC):
    def __init__(self, model_name:str , device:Optional[str]=None):
        self.model_name=model_name
        self.device= device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def transcribe(self, audio_path:str)->Transcription:
        pass

