"""
pipeline/diarise.py

Speaker diarisation using pyannote.audio.

Takes a raw audio file and returns speaker-labelled time segments,
identifying who spoke when. This runs BEFORE ASR transcription so
that the final transcript can be speaker-attributed.

Usage:
    from pipeline.diarise import diarise_audio

    segments = diarise_audio("interview.wav")
    # [
    #   {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.2},
    #   {"speaker": "SPEAKER_01", "start": 4.2, "end": 9.8},
    #   ...
    # ]

Requires:
    - HF_TOKEN in .env (or environment variable)
    - Accepted terms at https://huggingface.co/pyannote/speaker-diarization-3.1
    - pip install pyannote.audio
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

DIARISATION_MODEL = "pyannote/speaker-diarization-3.1"

_pipeline = None  # lazy-loaded, cached across calls


def _get_pipeline():
    """Lazy-load the pyannote pipeline, cached after first call."""
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN not found in environment. "
                "Add it to your .env file: HF_TOKEN=hf_..."
            )

        print(f"Loading {DIARISATION_MODEL}...")
        _pipeline = Pipeline.from_pretrained(
            DIARISATION_MODEL,
            token=hf_token,
        )

        # use GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                _pipeline.to(torch.device("cuda"))
                print("  Using CUDA")
            elif torch.backends.mps.is_available():
                _pipeline.to(torch.device("mps"))
                print("  Using MPS (Apple Silicon)")
        except Exception as e:
            print(f"  Running on CPU ({e})")

    return _pipeline


def diarise_audio(
    audio_path: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict]:
    """
    Run speaker diarisation on an audio file.

    Args:
        audio_path: path to audio file (wav, mp3, etc.)
        num_speakers: exact number of speakers if known (improves accuracy)
        min_speakers: minimum expected speakers
        max_speakers: maximum expected speakers

    Returns:
        List of segments: [{"speaker": str, "start": float, "end": float}, ...]
        sorted by start time.
    """
    pipeline = _get_pipeline()

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    diarization = pipeline(audio_path, **kwargs)

    segments = []

    # pyannote 4.0+ (community-1) returns object with .speaker_diarization attribute
    # pyannote 3.x returns annotation directly with .itertracks()
    if hasattr(diarization, 'speaker_diarization'):
        for turn, speaker in diarization.speaker_diarization:
            segments.append({
                "speaker": speaker,
                "start":   round(turn.start, 3),
                "end":     round(turn.end, 3),
            })
    else:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start":   round(turn.start, 3),
                "end":     round(turn.end, 3),
            })

    segments.sort(key=lambda s: s["start"])
    return segments


def merge_close_segments(
    segments: List[Dict],
    max_gap: float = 0.5,
) -> List[Dict]:
    """
    Merge consecutive segments from the SAME speaker if the gap between
    them is small (likely a brief pause, not a real turn change).
    Reduces over-segmentation noise.

    Args:
        segments: diarisation segments from diarise_audio()
        max_gap: maximum gap in seconds to merge across

    Returns:
        Merged segment list.
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]
        if (seg["speaker"] == last["speaker"]
                and seg["start"] - last["end"] <= max_gap):
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    return merged


def assign_speaker_to_word(
    word_start: float,
    word_end: float,
    diarisation_segments: List[Dict],
) -> Optional[str]:
    """
    Given a word's time range and diarisation segments, determine which
    speaker most likely said it (by maximum temporal overlap).

    Args:
        word_start, word_end: word timestamp from ASR output
        diarisation_segments: output of diarise_audio() or merge_close_segments()

    Returns:
        speaker label, or None if no overlapping segment found
    """
    best_speaker = None
    best_overlap = 0.0

    for seg in diarisation_segments:
        overlap_start = max(word_start, seg["start"])
        overlap_end   = min(word_end, seg["end"])
        overlap       = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker


def check_diarisation_available() -> bool:
    """Check if pyannote is installed and HF_TOKEN is configured."""
    try:
        import pyannote.audio  # noqa
    except ImportError:
        return False
    return bool(os.environ.get("HF_TOKEN"))