"""
src/text_normalise.py

Standalone copy of src/judge.py's normalise() function, with ZERO heavy
dependencies (no ollama, no torch) - safe to import in lightweight
environments like the fine-tuning venv, which deliberately doesn't
install ollama.

WHY THIS EXISTS: src/judge.py has `import ollama` and `import torch` at
module level, so importing ANYTHING from it (even just normalise(),
which itself doesn't touch ollama or torch at all) pulls in those
dependencies transitively - breaking on a minimal venv that doesn't have
them installed. This module contains an exact, byte-for-byte copy of
normalise() only, so validation WER computed during fine-tuning stays
directly comparable to every other WER reported elsewhere in the
dissertation pipeline (they must use identical normalisation logic).

If src/judge.py's normalise() is ever edited, this copy must be updated
to match - they are NOT automatically kept in sync.

Usage:
    from src.text_normalise import normalise
"""

import re


def normalise(text: str) -> str:
    """Lowercase and strip punctuation for WER computation.
    Also strips bracketed annotation tags (e.g. <OVERLAP>, <NOISE>, <LAUGH>,
    <INAUDIBLE>) BEFORE punctuation stripping — otherwise the brackets are
    removed but the word inside (e.g. "overlap") survives as a literal
    reference token the ASR model can never correctly produce, silently
    inflating WER wherever these transcription-convention tags appear."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
    return re.sub(r'\s+', ' ', text).strip()