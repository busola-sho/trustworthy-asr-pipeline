from abc import ABC
from dataclasses import dataclass
from typing import Optional
import torch
from models import Transcription, Segment
import difflib

@dataclass
class InterRep: #Intermediate Representation
    position_id: int
    candidates: dict
    scores: dict
    disagreement: bool
    segment_id: int

@dataclass
class AlignedTranscript:
    positions: list[InterRep]
    audio_path: str


def align(transcripts:list[Transcription]):
    
        
    pass

