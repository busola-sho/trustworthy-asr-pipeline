"""Four-source sentence confidence: segment_and_judge_sentences."""

from asr_pipeline.confidence.paths import SentenceConfidenceFiles
from asr_pipeline.confidence.progress import load_progress, save_progress

import json
import os
import re
import argparse

from ollama import Client

from asr_pipeline.sentence_boundaries import segment_hypothesis
from asr_pipeline.confidence.inputs import (
    SOURCE_MODELS,
    fusion_result_path,
    load_fusion_samples,
)
from wtpsplit import SaT
from asr_pipeline.parallel_calls import call_models_in_parallel
from asr_pipeline.prompts import SEVERITY_RUBRIC

OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "phi4:14b"
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))

FLAG_THRESHOLD = 2

# STAGE 1: alignment - all segments of a transcript shown together, so
# the model can partition the reference holistically rather than
# guessing one segment's boundary in isolation.
ALIGNMENT_PROMPT_TEMPLATE = """The following is a reference transcript (ground truth) and a list of
predicted segments that together represent the same reference, split
into sentence-like pieces by an automated segmentation tool.

Reference transcript: "{reference}"

Predicted segments (in order):
{segment_list}

For each predicted segment, identify the exact portion of the reference
transcript that corresponds to it. Together, all identified spans should
cover the entire reference transcript with no gaps or overlaps, in the
same order as the segments given.

Respond in exactly this format, one line per segment, nothing else:
[1] <corresponding reference span>
[2] <corresponding reference span>
(continue for every segment)
"""

# STAGE 2: judging - structurally identical in shape to
# DIRECT_SEVERITY_PROMPT (reference span + hypothesis segment). The
# full reference and other segments are NEVER shown here - there is
# nothing left for the judge to be tempted to compare against.
JUDGE_PROMPT_TEMPLATE = """You are evaluating the severity of a speech-to-text
transcription error for use in a high-stakes policing context, where the
transcript may be relied on as evidence.

{rubric}

Reference span (ground truth): "{{reference_span}}"
Predicted segment (ASR output): "{{segment}}"

Compare the predicted segment against the reference span, and assign a
single severity score from 0 to 4 using the scale above. Base your
judgement only on the pair given - do not assume outside context.

Respond in exactly this format:
Severity: <0-4>
Justification: <one to two sentences explaining your score>
""".format(rubric=SEVERITY_RUBRIC)


_ALIGNMENT_LINE_PATTERN = re.compile(r"\[(\d+)\]\s*(.+)")
_SEVERITY_PATTERN = re.compile(
    r"\*{0,2}severity(?:\s*score)?\*{0,2}:\*{0,2}\s*\n?\s*(\d)", re.IGNORECASE
)
_JUSTIFICATION_PATTERN = re.compile(
    r"\*{0,2}justification\*{0,2}:\*{0,2}\s*\n?\s*(.+)", re.IGNORECASE | re.DOTALL
)


def parse_alignment_response(raw_response, segment_count):
    """Parses the [1] <span> / [2] <span> ... format. Returns a list of
    length n_segments (span text or None per position) - None where
    parsing failed for that specific line, so failures are visible per
    segment rather than silently defaulting."""
    spans = [None] * segment_count
    if raw_response is None:
        return spans
    for line in raw_response.splitlines():
        match = _ALIGNMENT_LINE_PATTERN.search(line.strip())
        if match:
            transcript_index = int(match.group(1)) - 1
            if 0 <= transcript_index < segment_count:
                spans[transcript_index] = match.group(2).strip().strip('"').strip()
    return spans


def parse_judge_response(raw_response):
    severity, justification = None, None
    if raw_response is None:
        return severity, justification
    severity_match = _SEVERITY_PATTERN.search(raw_response)
    if severity_match:
        try:
            severity = int(severity_match.group(1))
        except ValueError:
            severity = None
    justification_match = _JUSTIFICATION_PATTERN.search(raw_response)
    if justification_match:
        justification = justification_match.group(1).strip().strip("*").strip()
    return severity, justification


def load_fusion_transcripts(dataset, split="dev", paths=None):
    paths = paths or SentenceConfidenceFiles()
    return load_fusion_samples(dataset, split, paths=paths)


def align_transcript(client, reference, segments, retries=2):
    segment_list = "\n".join(
        f"[{i + 1}] {segment}" for i, segment in enumerate(segments)
    )
    prompt = ALIGNMENT_PROMPT_TEMPLATE.format(
        reference=reference, segment_list=segment_list
    )
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 8192, "num_predict": 500},
                keep_alive="30m",
                think=False,
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (align): {e}")
                return None
    return None


def judge_segment(client, reference_span, segment, retries=2):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        reference_span=reference_span, segment=segment
    )
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 4096, "num_predict": 300},
                keep_alive="30m",
                think=False,
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (judge): {e}")
                return None
    return None


def align_reference_spans(chunk, chunk_segments, client, max_workers):
    align_work_items = [
        (
            s.get("dataset_index"),
            s.get("ref"),
            chunk_segments[s.get("dataset_index")],
        )
        for s in chunk
    ]

    def _align_worker(item):
        _, reference, segments = item
        if len(segments) == 1:
            return None  # signal: no LLM call needed, handled below
        return align_transcript(client, reference, segments)

    align_responses = call_models_in_parallel(
        align_work_items, _align_worker, max_workers=max_workers, progress_every=10
    )

    aligned_spans_by_idx = {}
    raw_alignment_by_idx = {}
    for (transcript_index, reference, segments), raw in zip(
        align_work_items, align_responses
    ):
        if len(segments) == 1:
            aligned_spans_by_idx[transcript_index] = [reference]
            raw_alignment_by_idx[transcript_index] = (
                "[skipped: single-segment transcript, no alignment needed]"
            )
        else:
            aligned_spans_by_idx[transcript_index] = parse_alignment_response(
                raw, len(segments)
            )
            raw_alignment_by_idx[transcript_index] = raw

    return aligned_spans_by_idx, raw_alignment_by_idx


def judge_reference_spans(
    chunk, chunk_segments, aligned_spans_by_idx, client, max_workers
):
    # Phase C (STAGE 2): judge - one call per segment, using ONLY
    # its aligned span (never the full reference)
    judge_work_items = []  # (dataset_index, seg_position, segment_text, aligned_span)
    for s in chunk:
        transcript_index = s.get("dataset_index")
        segments = chunk_segments[transcript_index]
        spans = aligned_spans_by_idx[transcript_index]
        for pos, segment in enumerate(segments):
            span = spans[pos] if pos < len(spans) else None
            judge_work_items.append((transcript_index, pos, segment, span))

    def _judge_worker(item):
        _, _, segment, span = item
        if span is None:
            return (
                None  # alignment failed for this segment - skip judging, flag as error
            )
        return judge_segment(client, span, segment)

    judge_responses = call_models_in_parallel(
        judge_work_items, _judge_worker, max_workers=max_workers, progress_every=25
    )

    judged_by_idx = {}
    for (transcript_index, pos, segment, span), raw in zip(
        judge_work_items, judge_responses
    ):
        judged_by_idx.setdefault(transcript_index, {})[pos] = (segment, span, raw)

    return judged_by_idx


def assemble_sentence_judgements(
    chunk, chunk_segments, judged_by_idx, raw_alignment_by_idx
):
    results = []
    # Phase D: reassemble
    for s in chunk:
        transcript_index = s.get("dataset_index")
        segments = chunk_segments[transcript_index]
        seg_results = []
        for pos in range(len(segments)):
            segment, span, raw = judged_by_idx.get(transcript_index, {}).get(
                pos, (segments[pos], None, None)
            )
            if span is None or raw is None:
                seg_results.append(
                    {
                        "segment": segment,
                        "corresponding_span": span,
                        "severity": None,
                        "justification": None,
                        "flagged": None,
                        "error": True,
                        "error_reason": "alignment_failed"
                        if span is None
                        else "judge_call_failed",
                    }
                )
                continue
            severity, justification = parse_judge_response(raw)
            flagged = (severity >= FLAG_THRESHOLD) if severity is not None else None
            seg_results.append(
                {
                    "segment": segment,
                    "corresponding_span": span,
                    "severity": severity,
                    "justification": justification,
                    "flagged": flagged,
                    "raw_judge_response": raw,
                }
            )

        results.append(
            {
                "dataset_index": transcript_index,
                "ref": s.get("ref"),
                "hyp": s.get("hyp"),
                "raw_alignment_response": raw_alignment_by_idx.get(transcript_index),
                "segments": seg_results,
            }
        )

    return results


def run_dataset(
    dataset,
    client,
    sat,
    split="dev",
    max_samples=None,
    rerun=False,
    max_workers=None,
    chunk_size=25,
    *,
    samples=None,
    output_dir=None,
    fusion_file=None,
    paths=None,
):
    paths = paths or SentenceConfidenceFiles()
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    print(f"\n-- {dataset} | segment_and_judge_sentences (two-stage) split={split} --")

    if samples is None:
        samples = load_fusion_transcripts(dataset, split, paths=paths)
    if max_samples:
        samples = samples[:max_samples]
    print(f"  {len(samples)} valid transcripts")

    output_dir = output_dir or paths.segments_dir
    os.makedirs(output_dir, exist_ok=True)
    filename = f"sentence_segments_{dataset}_{split}.json"
    output_path = os.path.join(output_dir, filename)

    results = load_progress(output_path, rerun)
    start_from = len(results)
    if start_from:
        print(f"  Resuming from transcript {start_from}/{len(samples)}")

    remaining = samples[start_from:]

    for chunk_start in range(0, len(remaining), chunk_size):
        chunk = remaining[chunk_start : chunk_start + chunk_size]

        chunk_segments = {
            sample["dataset_index"]: segment_hypothesis(sample["hyp"], sat)
            for sample in chunk
        }
        # One sentence maps to the whole reference; no alignment call is needed.
        aligned_spans_by_idx, raw_alignment_by_idx = align_reference_spans(
            chunk,
            chunk_segments,
            client,
            max_workers,
        )

        judged_by_idx = judge_reference_spans(
            chunk,
            chunk_segments,
            aligned_spans_by_idx,
            client,
            max_workers,
        )

        results.extend(
            assemble_sentence_judgements(
                chunk,
                chunk_segments,
                judged_by_idx,
                raw_alignment_by_idx,
            )
        )

        save_progress(output_path, results)
        print(f"  chunk done: {len(results)}/{len(remaining) + start_from} transcripts")

    n_segments_total = sum(len(r["segments"]) for r in results)
    flagged_count = sum(
        1
        for r in results
        for segment in r["segments"]
        if segment.get("flagged") is True
    )
    parsed_count = sum(
        1
        for r in results
        for segment in r["segments"]
        if segment.get("severity") is not None
    )
    n_alignment_failed = sum(
        1
        for r in results
        for segment in r["segments"]
        if segment.get("error_reason") == "alignment_failed"
    )

    output = {
        "dataset": dataset,
        "split": split,
        "judge": JUDGE_MODEL,
        "fusion_input": str(
            fusion_file or fusion_result_path(dataset, split, paths=paths)
        ),
        "asr_models": list(SOURCE_MODELS),
        "architecture": "two-stage: alignment (per-transcript) then judging (per-segment, span-only)",
        "flag_threshold": FLAG_THRESHOLD,
        "flag_convention": "y=1 for meaning-altering/flagged (severity>=2), y=0 for meaning-preserved (severity<2)",
        "num_transcripts": len(results),
        "num_segments": n_segments_total,
        "num_segments_parsed": parsed_count,
        "num_segments_flagged": flagged_count,
        "num_segments_alignment_failed": n_alignment_failed,
        "transcripts": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(
        f"\n  {len(results)} transcripts, {n_segments_total} segments "
        f"({parsed_count} parsed, {flagged_count} flagged, {n_alignment_failed} alignment failures)"
    )
    print(f"  Saved: {output_path}")

    return output


def main(argv=None, paths=None):
    paths = paths or SentenceConfidenceFiles()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--parallel-calls",
        "--max-workers",
        dest="max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
    )
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args(argv)

    print("Loading SaT model...")
    sat = SaT("sat-3l")

    client = Client(host=paths.ollama_url)

    run_dataset(
        args.dataset,
        client,
        sat,
        split=args.split,
        max_samples=args.max_samples,
        rerun=args.rerun,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        paths=paths,
    )


if __name__ == "__main__":
    main()
