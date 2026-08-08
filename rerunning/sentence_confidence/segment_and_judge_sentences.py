"""
rerunning/sentence_confidence/segment_and_judge_sentences.py

Step 2 of the sentence-confidence rebuild: applies the FROZEN SaT +
Rule A + Rule B segmentation pipeline to every 5model transcript, then
gets per-SEGMENT severity from Phi-4.

TWO-STAGE JUDGING ARCHITECTURE (revised after a real methodological
bug was caught by inspection - see below):

  Stage 1 - ALIGNMENT (one call per TRANSCRIPT, all segments together):
    Given the full reference and ALL of a transcript's segments at
    once, Phi-4 partitions the reference into one corresponding span
    PER segment. This solves "which part of the reference does each
    segment own" holistically, with full context of every segment at
    once - not one segment at a time in isolation.

  Stage 2 - JUDGING (one call per SEGMENT, using ONLY its aligned span):
    Each segment is then judged against ONLY its own aligned span -
    the full reference and the other segments are NEVER shown at this
    stage. This is structurally identical in shape to the existing
    validated DIRECT_SEVERITY_PROMPT (reference span + hypothesis
    segment), just applied to a smaller unit than a whole transcript.

WHY THIS CHANGED FROM A SINGLE COMBINED CALL: the original design
asked Phi-4 to identify the corresponding span AND judge severity in
one call, with the FULL reference always visible. Real inspection
caught this producing a systematic bug: for a reference like "IT'S SO
GOOD THE KID CAN" that got fused+segmented into two pieces ("It's so
good." / "The kid can."), each segment was correctly matched to its
own span, but then penalized in the SEVERITY score for "omitting" the
OTHER segment's content - content it was never responsible for. No
information was actually lost at the transcript level; SaT had simply
divided one reference utterance into two review units. A stronger
"do not penalize outside the span" instruction was tried first, but
this is a structural bias (the full reference stays visible and
creates a persistent pull toward whole-reference comparison) that
prompting alone cannot reliably override - the two-stage split removes
the possibility entirely by never showing the judge anything to be
tempted by at judging time.

Binary target convention (LOCKED): y=1 for MEANING-ALTERING/FLAGGED
(severity >= 2), y=0 for MEANING-PRESERVED (severity < 2).

Segmentation pipeline (SaT + Rule A + Rule B) is FROZEN as of this run.

Chunked execution - each chunk of transcripts is fully segmented,
aligned, judged, and saved before the next chunk starts.

Usage:
    python segment_and_judge_sentences.py --dataset commonvoice --split dev
"""

import json
import os
import re
import argparse

from ollama import Client

from src.sat_postprocess import load_5model_transcripts, apply_postprocessing
from wtpsplit import SaT
from src.concurrent_ollama import run_concurrent
from rerunning.severity_judge_prompts import SEVERITY_RUBRIC

OUTPUT_DIR = "writeup_results/sentence_confidence/segments"
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


_ALIGNMENT_LINE_PATTERN = re.compile(r'\[(\d+)\]\s*(.+)')
_SEVERITY_PATTERN = re.compile(
    r'\*{0,2}severity(?:\s*score)?\*{0,2}:\*{0,2}\s*\n?\s*(\d)', re.IGNORECASE
)
_JUSTIFICATION_PATTERN = re.compile(
    r'\*{0,2}justification\*{0,2}:\*{0,2}\s*\n?\s*(.+)', re.IGNORECASE | re.DOTALL
)


def parse_alignment_response(raw_response, n_segments):
    """Parses the [1] <span> / [2] <span> ... format. Returns a list of
    length n_segments (span text or None per position) - None where
    parsing failed for that specific line, so failures are visible per
    segment rather than silently defaulting."""
    spans = [None] * n_segments
    if raw_response is None:
        return spans
    for line in raw_response.splitlines():
        match = _ALIGNMENT_LINE_PATTERN.search(line.strip())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < n_segments:
                spans[idx] = match.group(2).strip().strip('"').strip()
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


def load_5model_transcripts_wrapper(dataset, split="dev"):
    return load_5model_transcripts(dataset, split)


def align_transcript(client, reference, segments, retries=2):
    segment_list = "\n".join(f"[{i+1}] {seg}" for i, seg in enumerate(segments))
    prompt = ALIGNMENT_PROMPT_TEMPLATE.format(reference=reference, segment_list=segment_list)
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
    prompt = JUDGE_PROMPT_TEMPLATE.format(reference_span=reference_span, segment=segment)
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


def run_dataset(dataset, client, sat, split="dev", max_samples=None, rerun=False,
                 max_workers=None, chunk_size=25):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    print(f"\n-- {dataset} | segment_and_judge_sentences (two-stage) split={split} --")

    samples = load_5model_transcripts_wrapper(dataset, split)
    if max_samples:
        samples = samples[:max_samples]
    print(f"  {len(samples)} valid transcripts")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"sentence_segments_{dataset}_{split}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("transcripts", [])
        start_from = len(results)
        print(f"  Resuming from transcript {start_from}/{len(samples)}")
    else:
        results = []
        start_from = 0

    def save_progress():
        payload = {"progress": len(results), "transcripts": results}
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    remaining = samples[start_from:]

    for chunk_start in range(0, len(remaining), chunk_size):
        chunk = remaining[chunk_start:chunk_start + chunk_size]

        # Phase A: segment (frozen SaT + Rule A + Rule B)
        chunk_segments = {}  # dataset_index -> list of segment texts
        for s in chunk:
            idx = s.get("dataset_index")
            raw_segments = sat.split(s["hyp"])
            segments, _, _ = apply_postprocessing(raw_segments)
            segments = [seg.strip() for seg in segments if seg.strip()]
            chunk_segments[idx] = segments

        # Phase B (STAGE 1): align - one call per transcript, all
        # segments of that transcript shown together. EXCEPT: when a
        # transcript has only ONE segment, there is no alignment
        # problem to solve - that single segment necessarily
        # corresponds to the ENTIRE reference by definition. Asking
        # Phi-4 to "identify" this trivial 1:1 case turned out to be a
        # real bug source: it would frequently drop the requested
        # [1] <span> bracket format for single-segment inputs (the
        # task feels redundant to it), which the strict parser then
        # failed to match, wrongly marking a perfectly clean
        # near-identical hyp/ref pair as alignment_failed. Skipping the
        # LLM call for this case removes the failure mode entirely AND
        # is methodologically more correct - zero ambiguity when there
        # is nothing to partition.
        align_work_items = [(s.get("dataset_index"), s.get("ref"), chunk_segments[s.get("dataset_index")])
                             for s in chunk]

        def _align_worker(item):
            _, ref, segments = item
            if len(segments) == 1:
                return None  # signal: no LLM call needed, handled below
            return align_transcript(client, ref, segments)

        align_responses = run_concurrent(align_work_items, _align_worker,
                                         max_workers=max_workers, progress_every=10)

        aligned_spans_by_idx = {}
        raw_alignment_by_idx = {}
        for (idx, ref, segments), raw in zip(align_work_items, align_responses):
            if len(segments) == 1:
                aligned_spans_by_idx[idx] = [ref]  # deterministic - whole reference, no LLM needed
                raw_alignment_by_idx[idx] = "[skipped: single-segment transcript, no alignment needed]"
            else:
                aligned_spans_by_idx[idx] = parse_alignment_response(raw, len(segments))
                raw_alignment_by_idx[idx] = raw  # saved for EVERY transcript, success or failure -
                                                  # so alignment behavior is always inspectable
                                                  # after the fact without a live Ollama call

        # Phase C (STAGE 2): judge - one call per segment, using ONLY
        # its aligned span (never the full reference)
        judge_work_items = []  # (dataset_index, seg_position, segment_text, aligned_span)
        for s in chunk:
            idx = s.get("dataset_index")
            segments = chunk_segments[idx]
            spans = aligned_spans_by_idx[idx]
            for pos, seg in enumerate(segments):
                span = spans[pos] if pos < len(spans) else None
                judge_work_items.append((idx, pos, seg, span))

        def _judge_worker(item):
            _, _, seg, span = item
            if span is None:
                return None  # alignment failed for this segment - skip judging, flag as error
            return judge_segment(client, span, seg)

        judge_responses = run_concurrent(judge_work_items, _judge_worker,
                                         max_workers=max_workers, progress_every=25)

        judged_by_idx = {}
        for (idx, pos, seg, span), raw in zip(judge_work_items, judge_responses):
            judged_by_idx.setdefault(idx, {})[pos] = (seg, span, raw)

        # Phase D: reassemble
        for s in chunk:
            idx = s.get("dataset_index")
            segments = chunk_segments[idx]
            seg_results = []
            for pos in range(len(segments)):
                seg, span, raw = judged_by_idx.get(idx, {}).get(pos, (segments[pos], None, None))
                if span is None or raw is None:
                    seg_results.append({"segment": seg, "corresponding_span": span,
                                        "severity": None, "justification": None,
                                        "flagged": None, "error": True,
                                        "error_reason": "alignment_failed" if span is None else "judge_call_failed"})
                    continue
                severity, justification = parse_judge_response(raw)
                flagged = (severity >= FLAG_THRESHOLD) if severity is not None else None
                seg_results.append({
                    "segment": seg,
                    "corresponding_span": span,
                    "severity": severity,
                    "justification": justification,
                    "flagged": flagged,
                    "raw_judge_response": raw,
                })

            results.append({
                "dataset_index": idx,
                "ref": s.get("ref"),
                "hyp": s.get("hyp"),
                "raw_alignment_response": raw_alignment_by_idx.get(idx),
                "segments": seg_results,
            })

        save_progress()
        print(f"  chunk done: {len(results)}/{len(remaining) + start_from} transcripts")

    n_segments_total = sum(len(r["segments"]) for r in results)
    n_flagged = sum(1 for r in results for seg in r["segments"] if seg.get("flagged") is True)
    n_parsed = sum(1 for r in results for seg in r["segments"] if seg.get("severity") is not None)
    n_alignment_failed = sum(1 for r in results for seg in r["segments"]
                              if seg.get("error_reason") == "alignment_failed")

    output = {
        "dataset": dataset,
        "split": split,
        "judge": JUDGE_MODEL,
        "architecture": "two-stage: alignment (per-transcript) then judging (per-segment, span-only)",
        "flag_threshold": FLAG_THRESHOLD,
        "flag_convention": "y=1 for meaning-altering/flagged (severity>=2), y=0 for meaning-preserved (severity<2)",
        "num_transcripts": len(results),
        "num_segments": n_segments_total,
        "num_segments_parsed": n_parsed,
        "num_segments_flagged": n_flagged,
        "num_segments_alignment_failed": n_alignment_failed,
        "transcripts": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  {len(results)} transcripts, {n_segments_total} segments "
          f"({n_parsed} parsed, {n_flagged} flagged, {n_alignment_failed} alignment failures)")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()

    print("Loading SaT model...")
    sat = SaT("sat-3l")

    client = Client(host=OLLAMA_HOST)

    run_dataset(args.dataset, client, sat, split=args.split,
                max_samples=args.max_samples, rerun=args.rerun, max_workers=args.max_workers,
                chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
