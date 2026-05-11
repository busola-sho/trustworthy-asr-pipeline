from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np
import librosa

@dataclass
class Segment:
    word: str
    start: float
    end: float
    confidence: float

@dataclass
class Transcription:
    segments: list[Segment]
    text: str
    model_name: str

class ASRModel(ABC):
    def __init__(self, model_name:str , device:Optional[str]=None):
        self.model_name=model_name
        self.device= device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int)->Transcription:
        pass

class Whisper(ASRModel):
    def __init__(self, model_name="openai/whisper-large-v3"):
        super().__init__(model_name, None)
        self.model=None
        self.processor=None
    
    def load(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name, torch_dtype=torch.float32)

    def transcribe(self, audio: np.ndarray, sample_rate: int):
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device).to(self.model.dtype)
        tokens=self.model.generate(features, return_dict_in_generate=True, output_scores=True)
        decoded_text = self.processor.batch_decode(tokens.sequences, skip_special_tokens=True)[0]
        return Transcription(segments=[], text=decoded_text, model_name=self.model_name)
    
# class Wav2Vec2(ASRModel):
