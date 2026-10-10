"""
rerunning/sentence_confidence/segment_and_judge_sentences.py

Step 2 of the sentence-confidence rebuild: applies the FROZEN SaT +
Rule A + Rule B segmentation pipeline to every 5model transcript, then
gets per-SEGMENT severity from Phi-4 (not per-transcript, as the old
pipeline did).

Judge prompt design (per locked feedback): full reference transcript +
current predicted segment. Explicit two-step instruction - first
identify the corresponding reference span, THEN judge severity - to
reduce the risk of Phi-4 comparing the segment against the wrong part
of a long reference.

Binary target convention (LOCKED, stated once so it's never ambiguous
downstream): y=1 for MEANING-ALTERING/FLAGGED (severity >= 2), y=0 for
MEANING-PRESERVED (severity < 2) - matching the "flagged" terminology
already used throughout the ensemble chapter. Never switch direction
between methods/metrics.

Segmentation pipeline is FROZEN as of this run - per the locked
protocol, no further adaptation after dev validation. Applied
UNCHANGED here and later to test/Shetland.

Two-phase execution (segment+prep, then judge) - segments are
concurrent-judged via Ollama/phi4, same pattern as the rest of the
pipeline tonight.

Usage:
    python segment_and_judge_sentences.py --dataset commonvoice --split dev
"""

import json
import os
import argparse

from ollama import Client

from src.sat_postprocess import load_5model_transcripts, apply_postprocessing
from wtpsplit import SaT
from src.concurrent_ollama import run_concurrent

OUTPUT_DIR = "writeup_results/sentence_confidence/segments"
OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "phi4:14b"
DATASETS = ["commonvoice", "edacc", "english_dialects"]
DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))

# LOCKED convention: severity >= 2 -> y=1 (flagged/meaning-altering)
FLAG_THRESHOLD = 2

JUDGE_PROMPT_TEMPLATE = """Reference transcript:
{reference}

Predicted segment:
{segment}

First identify the part of the reference transcript that corresponds to the predicted segment.
Then assign a severity score:

0 = fully meaning-preserving
1 = minor wording difference, meaning preserved
2 = some meaning changed or omitted
3 = substantial meaning alteration
4 = meaning largely incorrect or contradictory

Return only:
corresponding_reference_span: ...
severity: ...
reason: ..."""


def parse_judge_response(raw_response):
    """Parses the three labeled fields from Phi-4's response. Returns
    (corresponding_span, severity, reason) - severity is None if
    parsing fails, so failures are visible rather than silently
    defaulting to a value."""
    span, severity, reason = None, None, None

    for line in raw_response.splitlines():
        line = line.strip()
        if line.lower().startswith("corresponding_reference_span:"):
            span = line.split(":", 1)[1].strip()
        elif line.lower().startswith("severity:"):
            sev_str = line.split(":", 1)[1].strip()
            try:
                severity = int(sev_str[0])  # first char handles "2" or "2 (some..." variants
            except (ValueError, IndexError):
                severity = None
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    return span, severity, reason


def judge_segment(client, reference, segment, retries=2):
    prompt = JUDGE_PROMPT_TEMPLATE.format(reference=reference, segment=segment)
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 8192, "num_predict": 300},
                keep_alive="30m",
                think=False,
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (judge): {e}")
                return None
    return None


def run_dataset(dataset, client, sat, split="dev", max_samples=None, rerun=False, max_workers=None):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    print(f"\n-- {dataset} | segment_and_judge_sentences split={split} --")

    samples = load_5model_transcripts(dataset, split)
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

    # Phase A: segment every remaining transcript (frozen SaT + Rule A + Rule B)
    remaining = samples[start_from:]
    work_items = []  # (dataset_index, ref, segment_text) - one per SEGMENT, not per transcript
    transcript_segment_map = {}  # dataset_index -> list of segment texts, to reassemble after

    for s in remaining:
        idx = s.get("dataset_index")
        ref = s.get("ref")
        raw_segments = sat.split(s["hyp"])
        segments, split_events, merge_events = apply_postprocessing(raw_segments)
        segments = [seg.strip() for seg in segments if seg.strip()]
        transcript_segment_map[idx] = segments
        for seg in segments:
            work_items.append((idx, ref, seg))

    print(f"  {len(remaining)} transcripts -> {len(work_items)} segments queued for judging")

    # Phase B: judge every segment concurrently
    def _worker(item):
        _, ref, seg = item
        return judge_segment(client, ref, seg)

    raw_responses = run_concurrent(work_items, _worker, max_workers=max_workers, progress_every=25)

    # Phase C: reassemble per-transcript, parse responses
    responses_by_idx = {}
    for (idx, ref, seg), raw in zip(work_items, raw_responses):
        responses_by_idx.setdefault(idx, []).append((seg, raw))

    for s in remaining:
        idx = s.get("dataset_index")
        segments = transcript_segment_map[idx]
        seg_results = []
        for seg, raw in responses_by_idx.get(idx, []):
            if raw is None:
                seg_results.append({"segment": seg, "corresponding_span": None,
                                    "severity": None, "reason": None,
                                    "flagged": None, "error": True})
                continue
            span, severity, reason = parse_judge_response(raw)
            flagged = (severity >= FLAG_THRESHOLD) if severity is not None else None
            seg_results.append({
                "segment": seg,
                "corresponding_span": span,
                "severity": severity,
                "reason": reason,
                "flagged": flagged,   # y=1 if flagged (meaning-altering), y=0 if preserved
                "raw_judge_response": raw,
            })

        results.append({
            "dataset_index": idx,
            "ref": s.get("ref"),
            "hyp": s.get("hyp"),
            "segments": seg_results,
        })

        if len(results) % 10 == 0:
            save_progress()

    save_progress()

    n_segments_total = sum(len(r["segments"]) for r in results)
    n_flagged = sum(1 for r in results for seg in r["segments"] if seg.get("flagged") is True)
    n_parsed = sum(1 for r in results for seg in r["segments"] if seg.get("severity") is not None)

    output = {
        "dataset": dataset,
        "split": split,
        "judge": JUDGE_MODEL,
        "flag_threshold": FLAG_THRESHOLD,
        "flag_convention": "y=1 for meaning-altering/flagged (severity>=2), y=0 for meaning-preserved (severity<2)",
        "num_transcripts": len(results),
        "num_segments": n_segments_total,
        "num_segments_parsed": n_parsed,
        "num_segments_flagged": n_flagged,
        "transcripts": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  {len(results)} transcripts, {n_segments_total} segments "
          f"({n_parsed} parsed, {n_flagged} flagged)")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()

    print("Loading SaT model...")
    sat = SaT("sat-3l")

    client = Client(host=OLLAMA_HOST)

    run_dataset(args.dataset, client, sat, split=args.split,
                max_samples=args.max_samples, rerun=args.rerun, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
