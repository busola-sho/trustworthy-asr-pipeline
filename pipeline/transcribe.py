"""
pipeline/transcribe.py

Efficient transcription pipeline:
- Load all ASR models ONCE upfront
- Load audio ONCE into memory
- Slice audio per speaker turn (fast numpy operation)
- Transcribe each slice with pre-loaded models

This gives the best of both worlds: no repeated model loading,
no repeated disk I/O, and accurate per-turn transcription without
needing word-level timestamps.
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from src.models import Whisper, Parakeet, Qwen3ASR

# ── Model registry ─────────────────────────────────────────────────────────────

_models = {}
ASR_MODELS = ["qwen", "whisper", "parakeet"]


def load_all_models(model_names: Optional[List[str]] = None):
    """Load all models upfront. Call once before processing any turns."""
    if model_names is None:
        model_names = ASR_MODELS
    for name in model_names:
        if name not in _models:
            print(f"  Loading {name}...")
            if name == "qwen":
                m = Qwen3ASR()
            elif name == "whisper":
                m = Whisper()
            elif name == "parakeet":
                m = Parakeet()
            else:
                raise ValueError(f"Unknown model: {name}")
            m.load()
            _models[name] = m
            print(f"  {name} ready.")


def _get_model(name: str):
    if name not in _models:
        load_all_models([name])
    return _models[name]


# ── Audio loading ──────────────────────────────────────────────────────────────

def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file to 16kHz mono numpy array. Call once, reuse many times."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    return audio.astype(np.float32), sr


def slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Slice pre-loaded audio to a time range. Fast in-memory numpy op."""
    return audio[int(start * sr):int(end * sr)]


# ── Segment format conversion ──────────────────────────────────────────────────

def _segments_to_dict(segments) -> list:
    if not segments:
        return []
    if hasattr(segments[0], 'word'):
        return [
            {"word": s.word, "confidence": s.confidence,
             "start": s.start, "end": s.end}
            for s in segments if s.word.strip()
        ]
    return [s for s in segments if isinstance(s, dict)]


# ── Core transcription ─────────────────────────────────────────────────────────

def transcribe_segment(
    audio_slice: np.ndarray,
    sr: int,
    models: Optional[List[str]] = None,
) -> Dict:
    """
    Transcribe a pre-sliced audio numpy array with pre-loaded models.
    Models must be loaded via load_all_models() before calling this.

    Returns:
        {model_name: {"hyp": str, "segments": [...]}}
    """
    if models is None:
        models = ASR_MODELS

    results = {}
    for model_name in models:
        try:
            model         = _get_model(model_name)
            transcription = model.transcribe(audio_slice, sr)
            results[model_name] = {
                "hyp":      transcription.text,
                "segments": _segments_to_dict(transcription.segments),
            }
        except Exception as e:
            print(f"  ERROR ({model_name}): {e}")
            results[model_name] = {"hyp": "", "segments": [], "error": str(e)}

    return results


def transcribe_audio(
    audio_path: str,
    models: Optional[List[str]] = None,
    speaker_segment: Optional[Dict] = None,
) -> Dict:
    """
    Convenience function: load audio and transcribe.
    For efficiency with multiple turns use load_audio() + transcribe_segment().
    """
    if models is None:
        models = ASR_MODELS
    audio, sr = load_audio(audio_path)
    if speaker_segment is not None:
        audio = slice_audio(audio, sr,
                            speaker_segment.get("start", 0.0),
                            speaker_segment.get("end", len(audio) / sr))
    return transcribe_segment(audio, sr, models)