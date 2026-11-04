"""Shared checkpoint I/O for resumable sentence judging and confidence scoring."""

import json
from pathlib import Path


def load_progress(path, rerun=False):
    path = Path(path)
    if rerun or not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return json.load(handle).get("transcripts", [])


def save_progress(path, transcripts):
    data = {"progress": len(transcripts), "transcripts": transcripts}
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
