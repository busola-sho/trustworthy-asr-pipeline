"""
rerunning/sentence_confidence/verbalized_confidence.py

Method 1: Verbalized confidence, per Yang, Tsai & Yamada - "On
Verbalized Confidence Scores for LLMs" - three prompt formulations:
  - confscore:      confidence score, 0-100, targeting MEANING
                     PRESERVATION specifically (not generic correctness)
  - confprobscore:  confidence-as-probability, 0-1, same target
  - probscore:      probability, 0-1, same target

NOTE on scale: confscore uses 0-100, confprobscore/probscore use 0-1 -
these are DIFFERENT scales by design (matching each variant's natural
framing). Normalize confscore/100 for any cross-variant comparison.

Asks gemma4 (the SAME model that produced the fused transcript via
Unanchored Fusion selection - not Phi-4, which is the judge, a
separate evaluative role) to retrospectively rate its own confidence
in a specific fixed sentence segment, given the same 5 source ASR
transcripts as evidence. This is a SECOND gemma4 call, after fusion -
gemma4 does NOT choose sentence boundaries here; SaT already fixed
them, so all confidence methods are evaluated on identical units.

severity/flagged are NEVER exposed to gemma4 in the prompt - only the
segment text and the 5 source transcripts as evidence.

Pipeline:
  5 ASR hypotheses -> gemma4 fusion -> fixed fused transcript ->
  SaT segmentation -> fixed segment + same 5 hypotheses ->
  gemma4 verbalized-confidence call -> confidence score

Reads segment_and_judge_sentences.py's output (fixed segments + ground
truth) and the original 5model source_hyps for evidence context.

Usage:
    python verbalized_confidence.py --dataset commonvoice --split test --variant confscore
"""

import json
import os
import re
import argparse

from ollama import Client

from src.concurrent_ollama import run_concurrent
from src.selector import OLLAMA_MODELS

SEGMENTS_DIR = "writeup_results/sentence_confidence/segments"
FIVEMODEL_DIR = "writeup_results/grid/unanchored_fusion_naive_5model"
OUTPUT_DIR = "writeup_results/sentence_confidence/verbalized_confidence"
OLLAMA_HOST = "http://localhost:11434"
SELECTOR_MODEL_KEY = "gemma4"  # same model that produced the 5model fusion
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))

# NOTE: different scales by variant, matching each framing's natural
# range - confscore is 0-100, confprobscore/probscore are 0-1.
VARIANT_PHRASINGS = {
    "confscore": "a confidence score from 0 to 100 quantifying how confident you are "
                 "that this segment preserves the spoken meaning",
    "confprobscore": "a confidence score between 0 and 1 corresponding to the probability "
                      "that this segment preserves the spoken meaning",
    "probscore": "the probability between 0 and 1 that this segment preserves the spoken meaning",
}

VARIANT_SCALE = {
    "confscore": "0-100",
    "confprobscore": "0-1",
    "probscore": "0-1",
}

# Neutral, evidence-based framing - NOT "you previously combined...",
# which implied gemma4 should recall its own prior act of fusion
# rather than reason freshly from the evidence given.
PROMPT_TEMPLATE = """The following transcripts were produced by different ASR systems
for the same spoken audio:

{source_block}

The fused transcript contains this fixed segment:

"{segment}"

Based only on the evidence in the ASR transcripts, give {variant_phrasing}.

Respond only:
Score: <value>
"""

_SCORE_PATTERN = re.compile(
    r'\*{0,2}score\*{0,2}:\*{0,2}\s*([0-9]*\.?[0-9]+)',
    re.IGNORECASE
)


def parse_score(raw_response, variant):
    """Robust to markdown bolding. Validates the parsed value against
    the variant's own valid range (0-100 for confscore, 0-1 for the
    other two) - a value outside range is treated as a parse failure
    (None), not silently accepted."""
    if raw_response is None:
        return None

    match = _SCORE_PATTERN.search(raw_response)
    if not match:
        return None

    try:
        score = float(match.group(1))
    except ValueError:
        return None

    if variant == "confscore":
        return score if 0 <= score <= 100 else None
    return score if 0 <= score <= 1 else None


def load_segments(dataset, split):
    path = os.path.join(SEGMENTS_DIR, f"sentence_segments_{dataset}_{split}.json")
    return json.load(open(path))


def load_5model_source_hyps(dataset, split):
    path = os.path.join(FIVEMODEL_DIR, f"unanchored_fusion_naive_5model_{dataset}_gemma4_{split}.json")
    data = json.load(open(path))
    return {s["dataset_index"]: s.get("source_hyps", {}) for s in data.get("samples", [])}


def format_source_block(source_hyps):
    return "\n".join(f"Transcript ({model}): {hyp}" for model, hyp in source_hyps.items())


def get_confidence(client, model_name, source_hyps, segment, variant, retries=2):
    source_block = format_source_block(source_hyps)
    prompt = PROMPT_TEMPLATE.format(
        source_block=source_block,
        segment=segment,
        variant_phrasing=VARIANT_PHRASINGS[variant],
    )
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 4096, "num_predict": 20},
                keep_alive="30m",
                think=False,
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR: {e}")
                return None
    return None


def run_dataset(dataset, variant, client, split="test", max_workers=None, rerun=False, chunk_size=25):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    model_name = OLLAMA_MODELS[SELECTOR_MODEL_KEY]
    print(f"\n-- {dataset} | verbalized_confidence ({variant}, scale {VARIANT_SCALE[variant]}) split={split} --")

    segments_data = load_segments(dataset, split)
    source_hyps_by_idx = load_5model_source_hyps(dataset, split)
    transcripts = segments_data.get("transcripts", [])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"verbalized_{variant}_{dataset}_{split}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("transcripts", [])
        start_from = len(results)
        print(f"  Resuming from transcript {start_from}/{len(transcripts)}")
    else:
        results = []
        start_from = 0

    def save_progress():
        payload = {"progress": len(results), "transcripts": results}
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    remaining = transcripts[start_from:]

    for chunk_start in range(0, len(remaining), chunk_size):
        chunk = remaining[chunk_start:chunk_start + chunk_size]

        # NOTE: severity/flagged are read here ONLY to be carried
        # through into the output record for later evaluation - they
        # are NEVER included in the prompt sent to gemma4 (see
        # get_confidence() / PROMPT_TEMPLATE, which only receives
        # source_hyps and segment text).
        work_items = []  # (dataset_index, seg_position, segment_text, source_hyps)
        for t in chunk:
            idx = t["dataset_index"]
            source_hyps = source_hyps_by_idx.get(idx, {})
            for pos, seg in enumerate(t.get("segments", [])):
                if seg.get("severity") is None:
                    continue  # skip segments the judge itself failed to score
                work_items.append((idx, pos, seg["segment"], source_hyps))

        def _worker(item):
            _, _, seg_text, source_hyps = item
            return get_confidence(client, model_name, source_hyps, seg_text, variant)

        raw_responses = run_concurrent(work_items, _worker, max_workers=max_workers, progress_every=25)

        scores_by_idx = {}
        for (idx, pos, seg_text, _), raw in zip(work_items, raw_responses):
            scores_by_idx.setdefault(idx, {})[pos] = (raw, parse_score(raw, variant))

        for t in chunk:
            idx = t["dataset_index"]
            out_segments = []
            for pos, seg in enumerate(t.get("segments", [])):
                if seg.get("severity") is None:
                    continue  # skip entirely - matches model_internal_confidence.py /
                              # cross_model_agreement.py's convention. Keeping a
                              # placeholder entry here (as before) left this file's
                              # segment COUNT different from the other two methods'
                              # per transcript, which silently misaligns any downstream
                              # POSITIONAL join (confirmed: a transcript with a null
                              # segment followed by real ones would pair the wrong
                              # confidence scores against the wrong severity labels).
                raw, score = scores_by_idx.get(idx, {}).get(pos, (None, None))
                out_segments.append({
                    "segment": seg["segment"],
                    "severity": seg.get("severity"),
                    "flagged": seg.get("flagged"),
                    "verbalized_score": score,
                    "raw_response": raw,
                })
            results.append({"dataset_index": idx, "segments": out_segments})

        save_progress()
        print(f"  chunk done: {len(results)}/{len(remaining) + start_from} transcripts")

    n_segments = sum(len(t["segments"]) for t in results)
    n_scored = sum(1 for t in results for s in t["segments"] if s.get("verbalized_score") is not None)
    n_failed = n_segments - n_scored

    output = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "selector_model": SELECTOR_MODEL_KEY,
        "confidence_type": "retrospective_verbalized_confidence",
        "target": "meaning_preservation",
        "prompt_variant": variant,
        "score_scale": VARIANT_SCALE[variant],
        "temperature": 0,
        "prompt_template": PROMPT_TEMPLATE,
        "num_transcripts": len(results),
        "num_segments": n_segments,
        "num_segments_scored": n_scored,
        "num_segments_failed": n_failed,
        "transcripts": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  {n_scored}/{n_segments} segments scored ({n_failed} failed to parse). Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--split", default="test", choices=["dev", "test", "full"])
    parser.add_argument("--variant", required=True, choices=list(VARIANT_PHRASINGS.keys()))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()

    client = Client(host=OLLAMA_HOST)
    run_dataset(args.dataset, args.variant, client, split=args.split,
                max_workers=args.max_workers, rerun=args.rerun, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
