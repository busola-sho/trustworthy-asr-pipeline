"""Read input files and write new output files without replacing existing files."""

import json
from pathlib import Path

from .config import PROJECT_ROOT, SOURCE_MODELS


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def check_new_output_path(path):
    path = Path(path).resolve()
    # New output files must stay outside saved paper results and the archive.
    for root in (
        PROJECT_ROOT / "paper_results",
        PROJECT_ROOT / "_archive",
        PROJECT_ROOT / ".git",
    ):
        if path == root or path.is_relative_to(root):
            raise ValueError(
                f"Output path is inside a saved-results or archive folder: {path}"
            )
    if path.exists():
        raise FileExistsError(
            f"Choose a new output path. This file already exists: {path}"
        )
    return path


def write_json(path, data):
    path = check_new_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def samples_by_index(data):
    indexed = {}
    for position, sample in enumerate(data["samples"]):
        key = sample.get("dataset_index")
        if key is None:
            key = sample.get("sample_index")
        if key is None:
            key = position
        if key in indexed:
            raise ValueError(f"Duplicate sample index: {key}")
        indexed[key] = {**sample, "dataset_index": key}
    return indexed


def read_transcript_files(transcript_files_path):
    transcript_files_path = Path(transcript_files_path).resolve()
    transcript_files = read_json(transcript_files_path)
    if set(transcript_files["sources"]) != set(SOURCE_MODELS):
        raise ValueError(f"Expected exactly {SOURCE_MODELS}")
    sources = {}
    for model in SOURCE_MODELS:
        path = Path(transcript_files["sources"][model])
        if not path.is_absolute():
            path = transcript_files_path.parent / path
        sources[model] = samples_by_index(read_json(path))
    ids = set(sources[SOURCE_MODELS[0]])
    if any(set(rows) != ids for rows in sources.values()):
        raise ValueError("The four ASR files must contain the same sample indices")
    return transcript_files, sources
