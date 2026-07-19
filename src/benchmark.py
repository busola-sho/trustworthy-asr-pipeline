from jiwer import wer
from src.models import ASRModel
from src.datasets import Dataset
from src.judge import normalise, is_tag_only
import json
import os
from typing import Optional


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

    Samples are SKIPPED from scoring (no sample_WER, excluded from corpus_wer)
    when either:
      - the reference contains the IGNORE_TIME_SEGMENT_IN_SCORING marker, or
      - the reference is tag-only (e.g. "<OVERLAP>") with no real lexical
        content to score the hypothesis against.
    Both are still recorded in "samples" with skipped=True for transparency.
    """

    # resume support
    if start_from > 0 and os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("samples", [])
        all_refs = [
            normalise(s["ref"]) for s in results
            if isinstance(s.get("ref"), str) and not s.get("skipped")
        ]
        all_hyps = [
            normalise(s["hyp"]) for s in results
            if isinstance(s.get("hyp"), str) and not s.get("skipped")
        ]
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

        ref = sample.label

        # --- Skip samples with no scoreable reference content ---
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({
                "sample_index": i,
                "ref":          ref,
                "hyp":          None,
                "sample_WER":   None,
                "segments":     [],
                "skipped":      True,
                "skip_reason":  "ignore_time_segment",
            })
            continue

        if is_tag_only(ref):
            results.append({
                "sample_index": i,
                "ref":          ref,
                "hyp":          None,
                "sample_WER":   None,
                "segments":     [],
                "skipped":      True,
                "skip_reason":  "tag_only_reference",
            })
            continue

        transcript = model.transcribe(sample.audio, sample.sample_rate)
        sample_wer = wer(normalise(ref), normalise(transcript.text))

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
            "ref":          ref,
            "hyp":          transcript.text,
            "sample_WER":   sample_wer,
            "segments":     segments_data,
        })

        all_refs.append(normalise(ref))
        all_hyps.append(normalise(transcript.text))

        # save progress every sample
        with open(output_path, "w") as f:
            json.dump({"progress": len(results), "samples": results}, f, indent=2)

    corpus_wer = wer(all_refs, all_hyps) if all_refs else 0.0
    count = len(results)
    n_skipped = sum(1 for r in results if r.get("skipped"))

    output = {
        "model":       model.model_name,
        "dataset":     dataset.name,
        "corpus_wer":  corpus_wer,
        "num_samples": count,
        "num_scored":  count - n_skipped,
        "num_skipped": n_skipped,
        "subset_indices": subset_indices,
        "samples":     results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return {
        "model":       model.model_name,
        "dataset":     dataset.name,
        "num_samples": count,
        "num_scored":  count - n_skipped,
        "num_skipped": n_skipped,
        "WER":         corpus_wer,
    }