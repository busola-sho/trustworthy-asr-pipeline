from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import torch
from transformers import AutoProcessor, AutoModelForCTC, WavLMForCTC, HubertForCTC
import numpy as np
import librosa
import os

@dataclass
class Segment:
    word: str
    start: Optional[float]
    end: Optional[float]
    confidence: float

@dataclass
class Transcription:
    segments: List[Segment]
    text: str
    model_name: str


class ASRModel(ABC):
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self):
        pass

    def _resample(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        return audio

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcription:
        pass


# ── Whisper ────────────────────────────────────────────────────────────────────


class Whisper(ASRModel):
    def __init__(self, model_name="openai/whisper-large-v3"):
        super().__init__(model_name, None)

    def load(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcription:
        audio = self._resample(audio, sample_rate)

        chunk_length = 30 * 16000
        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

        full_transcript = ""
        all_segments: List[Segment] = []

        for chunk in chunks:
            features = self.processor(
                chunk, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device).to(self.model.dtype)

            with torch.no_grad():
                tokens = self.model.generate(
                    features,
                    return_dict_in_generate=True,
                    output_scores=True,
                    language="en",
                )

            decoded = self.processor.batch_decode(
                tokens.sequences, skip_special_tokens=True
            )[0]
            full_transcript += " " + decoded

            # extract word-level confidence from token scores
            generated_ids = tokens.sequences[0][4:]
            if tokens.scores:
                for tok_id, score in zip(generated_ids, tokens.scores):
                    tok_str = self.processor.tokenizer.decode([tok_id])
                    prob = torch.softmax(score.float(), dim=-1).max().item()

                    if tok_id in self.processor.tokenizer.all_special_ids:
                        continue

                    if tok_str.startswith(" ") and all_segments:
                        all_segments.append(Segment(
                            word=tok_str.strip(),
                            start=None, end=None,
                            confidence=round(prob, 6),
                        ))
                    elif tok_str.startswith(" "):
                        all_segments.append(Segment(
                            word=tok_str.strip(),
                            start=None, end=None,
                            confidence=round(prob, 6),
                        ))
                    elif all_segments:
                        # continuation token — update word and take min confidence
                        last = all_segments[-1]
                        all_segments[-1] = Segment(
                            word=last.word + tok_str,
                            start=None, end=None,
                            confidence=round(min(last.confidence, prob), 6),
                        )
                    else:
                        all_segments.append(Segment(
                            word=tok_str.strip(),
                            start=None, end=None,
                            confidence=round(prob, 6),
                        ))

        return Transcription(
            segments=all_segments,
            text=full_transcript.strip(),
            model_name=self.model_name,
        )



# ── CTC confidence extraction ──────────────────────────────────────────────────

def _ctc_logits_to_word_segments(logits: torch.Tensor, processor) -> List[Segment]:
    """
    Extract word-level confidence from CTC logits.
    Uses CTC collapsing + word-piece boundaries to identify words.
    Confidence = min frame probability per word (weakest-link).
    No timestamps.
    """
    probs = torch.softmax(logits[0].float(), dim=-1)
    frame_conf = probs.max(dim=-1).values
    predicted_ids = torch.argmax(logits[0], dim=-1)
    blank_id = 0

    # try char offsets first
    try:
        decoded = processor.decode(predicted_ids, output_char_offsets=True)
        if not hasattr(decoded, "char_offsets"):
            raise AttributeError("no char_offsets")

        segments = []
        current_word_chars = []
        current_word_frames = []

        for char_info in decoded.char_offsets:
            char = char_info["char"]
            start_f = char_info["start_offset"]
            end_f = char_info["end_offset"]
            if char == " ":
                if current_word_chars:
                    word = "".join(current_word_chars)
                    all_frames = [f for s, e in current_word_frames
                                  for f in range(min(s, len(frame_conf)), min(e, len(frame_conf)))]
                    conf = frame_conf[all_frames].min().item() if all_frames else 0.0
                    segments.append(Segment(word=word, start=None, end=None,
                                           confidence=round(conf, 6)))
                    current_word_chars = []
                    current_word_frames = []
            else:
                current_word_chars.append(char)
                current_word_frames.append((start_f, end_f))

        if current_word_chars:
            word = "".join(current_word_chars)
            all_frames = [f for s, e in current_word_frames
                          for f in range(min(s, len(frame_conf)), min(e, len(frame_conf)))]
            conf = frame_conf[all_frames].min().item() if all_frames else 0.0
            segments.append(Segment(word=word, start=None, end=None,
                                   confidence=round(conf, 6)))
        return segments

    except Exception:
        pass

    # fallback: CTC collapsing with word-piece boundary detection
    ids = predicted_ids.tolist()
    collapsed = []
    prev = None
    for i, tok in enumerate(ids):
        if tok != prev and tok != blank_id:
            collapsed.append((tok, i))
        prev = tok

    text = processor.batch_decode(predicted_ids.unsqueeze(0))[0].strip()
    words = text.split()
    if not words:
        return []

    segments = []
    current_frames = []
    word_idx = 0

    for tok_id, frame_i in collapsed:
        ch = processor.tokenizer.convert_ids_to_tokens([tok_id])[0] \
            if hasattr(processor, 'tokenizer') else ""
        is_new_word = ch.startswith("▁") or ch == "|"

        if is_new_word and current_frames and word_idx < len(words):
            conf = frame_conf[current_frames].min().item()
            segments.append(Segment(word=words[word_idx], start=None, end=None,
                                   confidence=round(conf, 6)))
            word_idx += 1
            current_frames = [frame_i]
        else:
            current_frames.append(frame_i)

    if current_frames and word_idx < len(words):
        conf = frame_conf[current_frames].min().item()
        segments.append(Segment(word=words[word_idx], start=None, end=None,
                               confidence=round(conf, 6)))
        word_idx += 1

    if len(segments) != len(words):
        avg_conf = frame_conf.mean().item()
        return [Segment(word=w, start=None, end=None,
                       confidence=round(avg_conf, 6)) for w in words]

    return segments


# ── Wav2Vec2 ───────────────────────────────────────────────────────────────────

class Wav2Vec2(ASRModel):
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60-self"):
        super().__init__(model_name, None)

    def load(self):
        self.model = AutoModelForCTC.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcription:
        audio = self._resample(audio, sample_rate)
        chunk_length = 30 * 16000
        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

        full_transcript = ""
        all_segments: List[Segment] = []

        for chunk in chunks:
            inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(predicted_ids)[0]
            full_transcript += " " + transcript
            all_segments.extend(_ctc_logits_to_word_segments(logits.cpu(), self.processor))

        return Transcription(
            segments=all_segments,
            text=full_transcript.strip(),
            model_name=self.model_name,
        )


# ── Parakeet ───────────────────────────────────────────────────────────────────

class Parakeet(ASRModel):
    def __init__(self, model_name="nvidia/parakeet-ctc-1.1b"):
        super().__init__(model_name, None)

    def load(self):
        self.model = AutoModelForCTC.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcription:
        audio = self._resample(audio, sample_rate)
        chunk_length = 30 * 16000
        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

        full_transcript = ""
        all_segments: List[Segment] = []

        for chunk in chunks:
            inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device).to(self.model.dtype) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            full_transcript += " " + transcript
            all_segments.extend(_ctc_logits_to_word_segments(logits.cpu(), self.processor))

        return Transcription(
            segments=all_segments,
            text=full_transcript.strip(),
            model_name=self.model_name,
        )


# ── WavLM ──────────────────────────────────────────────────────────────────────

class WavLM(ASRModel):
    def __init__(self, model_name="patrickvonplaten/wavlm-libri-clean-100h-large"):
        super().__init__(model_name, None)

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = WavLMForCTC.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcription:
        audio = self._resample(audio, sample_rate)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.batch_decode(predicted_ids)[0]
        return Transcription(segments=[], text=transcript, model_name=self.model_name)


# ── HuBERT ─────────────────────────────────────────────────────────────────────

class HuBERT(ASRModel):
    def __init__(self, model_name="facebook/hubert-large-ls960-ft"):
        super().__init__(model_name, None)

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = HubertForCTC.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcription:
        audio = self._resample(audio, sample_rate)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.batch_decode(predicted_ids)[0]
        return Transcription(segments=[], text=transcript, model_name=self.model_name)


# ── Qwen3ASR ───────────────────────────────────────────────────────────────────

class Qwen3ASR(ASRModel):
    def __init__(self, model_name="Qwen/Qwen3-ASR-1.7B"):
        super().__init__(model_name, None)
        self._qwen_model = None

    def load(self):
        from qwen_asr import Qwen3ASRModel
        self._qwen_model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcription:
        audio = self._resample(audio, sample_rate)
        results = self._qwen_model.transcribe((audio, 16000), language="English")
        transcript = results[0].text
        # confidence not available from qwen_asr — segments left empty
        return Transcription(segments=[], text=transcript, model_name=self.model_name)