"""Shared transcript normalization and annotation-tag filtering."""

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
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_tag_only(text):
    return len(re.sub(r"<[^>]+>", "", text).strip()) == 0
