from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import torch
from transformers import AutoProcessor, AutoModelForCTC
import numpy as np
import librosa


@dataclass
class WordConfidence:
    """One recognised word, its confidence and optional start/end times."""

    word: str
    start: Optional[float]
    end: Optional[float]
    confidence: float


@dataclass
class ASRTranscript:
    """Text and word confidences returned by one ASR model."""

    segments: List[WordConfidence]
    text: str
    model_name: str


class ASRModel(ABC):
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
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
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRTranscript:
        pass


def _ctc_logits_to_word_segments(
    logits: torch.Tensor, processor
) -> List[WordConfidence]:
    """
    Extract word-level confidence from CTC logits.
    Uses CTC collapsing + word-piece boundaries to identify words.
    Confidence = min frame probability per word (weakest-link).
    No timestamps.
    """
    probs = torch.softmax(logits[0].float(), dim=-1)
    frame_confidences = probs.max(dim=-1).values
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
                    all_frames = [
                        f
                        for s, e in current_word_frames
                        for f in range(
                            min(s, len(frame_confidences)),
                            min(e, len(frame_confidences)),
                        )
                    ]
                    conf = (
                        frame_confidences[all_frames].min().item()
                        if all_frames
                        else 0.0
                    )
                    segments.append(
                        WordConfidence(
                            word=word, start=None, end=None, confidence=round(conf, 6)
                        )
                    )
                    current_word_chars = []
                    current_word_frames = []
            else:
                current_word_chars.append(char)
                current_word_frames.append((start_f, end_f))

        if current_word_chars:
            word = "".join(current_word_chars)
            all_frames = [
                f
                for s, e in current_word_frames
                for f in range(
                    min(s, len(frame_confidences)), min(e, len(frame_confidences))
                )
            ]
            conf = frame_confidences[all_frames].min().item() if all_frames else 0.0
            segments.append(
                WordConfidence(
                    word=word, start=None, end=None, confidence=round(conf, 6)
                )
            )
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
    word_index = 0

    for tok_id, frame_i in collapsed:
        ch = (
            processor.tokenizer.convert_ids_to_tokens([tok_id])[0]
            if hasattr(processor, "tokenizer")
            else ""
        )
        is_new_word = ch.startswith("▁") or ch == "|"

        if is_new_word and current_frames and word_index < len(words):
            conf = frame_confidences[current_frames].min().item()
            segments.append(
                WordConfidence(
                    word=words[word_index],
                    start=None,
                    end=None,
                    confidence=round(conf, 6),
                )
            )
            word_index += 1
            current_frames = [frame_i]
        else:
            current_frames.append(frame_i)

    if current_frames and word_index < len(words):
        conf = frame_confidences[current_frames].min().item()
        segments.append(
            WordConfidence(
                word=words[word_index], start=None, end=None, confidence=round(conf, 6)
            )
        )
        word_index += 1

    if len(segments) != len(words):
        mean_confidence = frame_confidences.mean().item()
        return [
            WordConfidence(
                word=w, start=None, end=None, confidence=round(mean_confidence, 6)
            )
            for w in words
        ]

    return segments


class Wav2Vec2(ASRModel):
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60-self"):
        super().__init__(model_name, None)

    def load(self):
        self.model = AutoModelForCTC.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRTranscript:
        audio = self._resample(audio, sample_rate)
        chunk_length = 30 * 16000
        chunks = [
            audio[i : i + chunk_length] for i in range(0, len(audio), chunk_length)
        ]

        full_transcript = ""
        all_segments: List[WordConfidence] = []

        for chunk in chunks:
            inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(predicted_ids)[0]
            full_transcript += " " + transcript
            all_segments.extend(
                _ctc_logits_to_word_segments(logits.cpu(), self.processor)
            )

        return ASRTranscript(
            segments=all_segments,
            text=full_transcript.strip(),
            model_name=self.model_name,
        )


class Parakeet(ASRModel):
    def __init__(self, model_name="nvidia/parakeet-ctc-1.1b"):
        super().__init__(model_name, None)
        # Force CPU — MPS produces NaN logits for certain audio on Apple Silicon
        self.device = torch.device("cpu")

    def load(self):
        self.model = AutoModelForCTC.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRTranscript:
        audio = self._resample(audio, sample_rate)
        chunk_length = 30 * 16000
        chunks = [
            audio[i : i + chunk_length] for i in range(0, len(audio), chunk_length)
        ]

        full_transcript = ""
        all_segments: List[WordConfidence] = []

        for chunk in chunks:
            inputs = self.processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {
                k: v.to(self.device).to(self.model.dtype) for k, v in inputs.items()
            }

            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            full_transcript += " " + transcript
            all_segments.extend(
                _ctc_logits_to_word_segments(logits.cpu(), self.processor)
            )

        return ASRTranscript(
            segments=all_segments,
            text=full_transcript.strip(),
            model_name=self.model_name,
        )


class Qwen3ASR(ASRModel):
    """
    Qwen3-ASR with word-level confidence extraction via token logprobs.
    Subclasses Qwen3ASRModel internally to override _infer_asr_transformers
    and capture output_scores from generate(), then aggregates to word level.
    """

    def __init__(self, model_name="Qwen/Qwen3-ASR-1.7B"):
        super().__init__(model_name, None)
        self._qwen_model = None

    def load(self):
        from qwen_asr import Qwen3ASRModel
        import torch as _torch

        self._qwen_model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            torch_dtype=_torch.float16,  # was float32
        )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int, context: str = ""
    ) -> ASRTranscript:
        import torch as _torch
        import torch.nn.functional as F

        audio = self._resample(audio, sample_rate)
        qwen = self._qwen_model

        from qwen_asr.inference.utils import (
            normalize_audios,
            split_audio_into_chunks,
            SAMPLE_RATE,
            MAX_ASR_INPUT_SECONDS,
            normalize_language_name,
        )

        wavs = normalize_audios((audio, 16000))
        lang = normalize_language_name("English")
        text_prompt = qwen._build_text_prompt(context=context, force_language=lang)

        all_segments: list = []
        full_text = ""

        parts = split_audio_into_chunks(
            wav=wavs[0], sr=SAMPLE_RATE, max_chunk_sec=MAX_ASR_INPUT_SECONDS
        )

        for chunk_wav, _ in parts:
            inputs = qwen.processor(
                text=[text_prompt],
                audio=[chunk_wav],
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(qwen.model.device).to(qwen.model.dtype)
            prompt_len = inputs["input_ids"].shape[1]

            with _torch.no_grad():
                gen_out = qwen.model.generate(
                    **inputs,
                    max_new_tokens=qwen.max_new_tokens,
                    output_scores=True,
                )

            generated_ids = gen_out.sequences[0, prompt_len:]
            chunk_text = qwen.processor.batch_decode(
                [generated_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            full_text += chunk_text

            if not gen_out.scores:
                continue

            scores_stacked = _torch.stack([s[0] for s in gen_out.scores], dim=0)
            token_probs = F.softmax(scores_stacked.float(), dim=-1)

            tokenizer = qwen.processor.tokenizer
            current_toks, current_confs = [], []

            for t_idx, tok_id in enumerate(generated_ids):
                tok_id_int = tok_id.item()
                if tok_id_int == tokenizer.eos_token_id:
                    break
                tok_str = tokenizer.decode([tok_id_int])
                conf = token_probs[t_idx, tok_id_int].item()

                if tok_str.startswith(" ") and current_toks:
                    word = tokenizer.decode(
                        current_toks, skip_special_tokens=True
                    ).strip()
                    if word:
                        geo_mean = float(
                            np.exp(np.mean(np.log(np.clip(current_confs, 1e-9, 1.0))))
                        )
                        all_segments.append(
                            WordConfidence(
                                word=word,
                                start=None,
                                end=None,
                                confidence=round(geo_mean, 6),
                            )
                        )
                    current_toks, current_confs = [tok_id_int], [conf]
                else:
                    current_toks.append(tok_id_int)
                    current_confs.append(conf)

            if current_toks:
                word = tokenizer.decode(current_toks, skip_special_tokens=True).strip()
                if word:
                    geo_mean = float(
                        np.exp(np.mean(np.log(np.clip(current_confs, 1e-9, 1.0))))
                    )
                    all_segments.append(
                        WordConfidence(
                            word=word,
                            start=None,
                            end=None,
                            confidence=round(geo_mean, 6),
                        )
                    )

        return ASRTranscript(
            segments=all_segments,
            text=full_text.strip(),
            model_name=self.model_name,
        )
