from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2ForCTC, AutoModelForCTC
import numpy as np
import librosa
# from nemo.collections.speechlm2.models import SALM
import soundfile as sf
import tempfile
import os

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
        self.model=None
        self.processor=None

    @abstractmethod
    def load(self):
        pass
    
    def _resample(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        return audio

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int)->Transcription:
        pass



class Whisper(ASRModel):
    def __init__(self, model_name="openai/whisper-large-v3"):
        super().__init__(model_name, None)
    
    def load(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, dtype=torch.float32)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int):
        audio = self._resample(audio, sample_rate)
        
        chunk_length = 30 * 16000  # 30 seconds at 16kHz
        chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
        
        full_transcript = ""
        for chunk in chunks:
            features = self.processor(
                chunk, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device).to(self.model.dtype)
            
            tokens = self.model.generate(
                features,
                return_dict_in_generate=True,
                output_scores=True,
                language="en"
            )
            
            decoded = self.processor.batch_decode(
                tokens.sequences, skip_special_tokens=True
            )[0]
            full_transcript += " " + decoded
        
        return Transcription(
            segments=[], 
            text=full_transcript.strip(), 
            model_name=self.model_name
        )
    
class Wav2Vec2(ASRModel):
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60-self"):
        super().__init__(model_name, None)

    def load(self):
        self.model = AutoModelForCTC.from_pretrained(self.model_name, dtype=torch.float32)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def transcribe(self, audio, sample_rate):
        audio = self._resample(audio, sample_rate)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.batch_decode(predicted_ids)[0]
        return Transcription(segments=[], text=transcript, model_name=self.model_name)

class Parakeet(ASRModel):
    def __init__(self, model_name="nvidia/parakeet-ctc-1.1b"):
        super().__init__(model_name, None)

    def load(self):
        self.model = AutoModelForCTC.from_pretrained(self.model_name, dtype=torch.float32)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def transcribe(self, audio, sample_rate):
        audio = self._resample(audio, sample_rate)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        transcript = self.processor.batch_decode(outputs)[0]
        return Transcription(segments=[], text=transcript, model_name=self.model_name) 

# class CanaryQwen(ASRModel):
#     def __init__(self, model_name="nvidia/canary-qwen-2.5b"):\
#         super().__init__(model_name, None)
    
#     def load(self):
#         self.model = SALM.from_pretrained(self.model_name).bfloat16().eval().to(self.device)

#     def transcribe(self, audio, sample_rate):
#         audio = self._resample(audio, sample_rate)
#         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
#             tmp_path = f.name
#         sf.write(tmp_path, audio, 16000)

#         answer_ids = self.model.generate(
#             prompts=[[{
#                 "role": "user", 
#                 "content": f"Transcribe the following: {self.model.audio_locator_tag}",
#                 "audio": [tmp_path]
#             }]],
#             max_new_tokens=128,
#         )

#         transcript = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
#         os.remove(tmp_path)
#         return Transcription(segments=[], text=transcript, model_name=self.model_name)

