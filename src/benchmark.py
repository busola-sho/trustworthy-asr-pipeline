"""
src/benchmark.py

Runs an ASRModel on a Dataset and saves transcripts + per-word confidence scores.
No judge here — judging is done separately in evaluation/judge/.

Usage (via run_benchmark.py):
    python scripts/benchmarking/inference/run_benchmark.py --model whisper --dataset commonvoice --max_samples 150
"""

from jiwer import wer
from src.models import ASRModel
from src.datasets import Dataset
import json
import re
import os
from typing import Optional


def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
    return text.strip()


def run_benchmark(
    model: ASRModel,
    dataset: Dataset,
    output_path: str,
    max_samples: Optional[int] = None,
    start_from: int = 0,
    subset_indices: Optional[list] = None,
) -> dict:
    """
    Run model on dataset, saving transcripts + per-word confidence scores.

    Args:
        model:           loaded ASRModel instance
        dataset:         Dataset instance
        output_path:     path to save JSON output
        max_samples:     stop after this many samples (ignored if subset_indices given)
        start_from:      resume from this sample index
        subset_indices:  if given, only process these specific indices
    """

    # resume support
    if start_from > 0 and os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("samples", [])
        all_refs = [normalise(s["ref"]) for s in results if isinstance(s.get("ref"), str)]
        all_hyps = [normalise(s["hyp"]) for s in results if isinstance(s.get("hyp"), str)]
    else:
        results = []
        all_refs = []
        all_hyps = []

    subset_set = set(subset_indices) if subset_indices is not None else None

    for i, sample in enumerate(dataset.load()):
        # skip if not in subset
        if subset_set is not None and i not in subset_set:
            continue

        # skip already processed
        if i < start_from:
            continue

        # stop at max_samples (only applies when no subset given)
        if subset_set is None and max_samples and len(results) >= max_samples:
            break

        transcript = model.transcribe(sample.audio, sample.sample_rate)
        sample_wer = wer(normalise(sample.label), normalise(transcript.text))

        # serialise per-word confidence segments
        segments_data = [
            {
                "word":       seg.word,
                "confidence": round(seg.confidence, 6),
                "start":      seg.start,
                "end":        seg.end,
            }
            for seg in transcript.segments
        ] if transcript.segments else []

        results.append({
            "sample_index": i,
            "ref":          sample.label,
            "hyp":          transcript.text,
            "sample_WER":   sample_wer,
            "segments":     segments_data,
        })

        all_refs.append(normalise(sample.label))
        all_hyps.append(normalise(transcript.text))

        # save progress every sample
        with open(output_path, "w") as f:
            json.dump({"progress": len(results), "samples": results}, f, indent=2)

    corpus_wer = wer(all_refs, all_hyps) if all_refs else 0.0
    count = len(results)

    output = {
        "model":      model.model_name,
        "dataset":    dataset.name,
        "corpus_wer": corpus_wer,
        "num_samples": count,
        "subset_indices": subset_indices,
        "samples":    results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return {
        "model":      model.model_name,
        "dataset":    dataset.name,
        "num_samples": count,
        "WER":        corpus_wer,
    }