"""
rerunning/sentence_confidence/verbalized_confidence.py

Improved verbalized sentence-confidence implementation.

Key change from the dissertation version:
- The old implementation sent each model's ENTIRE utterance/transcript to Gemma
  for every sentence-like segment, which could overflow the 4096-token context
  window and expose large amounts of irrelevant evidence.
- This version first aligns each fixed fused segment to a LOCAL span in every
  source ASR hypothesis, then gives Gemma only those local excerpts.
- Missing local matches are shown explicitly as [NO MATCH].

The confidence target itself is unchanged:
- confscore:     confidence score, 0-100, targeting meaning preservation
- confprobscore: confidence-as-probability, 0-1
- probscore:     probability, 0-1

Sentence boundaries remain fixed by segment_and_judge_sentences.py, so this file
changes only how source-ASR evidence is presented to the confidence model.

Usage:
    python rerunning/sentence_confidence/verbalized_confidence.py \
        --dataset commonvoice --split test --variant confscore --rerun
"""

import argparse
import json
import os
import re

from ollama import Client

from src.concurrent_ollama import run_concurrent
from src.selector import OLLAMA_MODELS

# Put align_segments_hybrid.py in rerunning/sentence_confidence/.
try:
    from rerunning.sentence_confidence.align_segments_hybrid import align_segments_hybrid
except ImportError:
    # Allows direct execution from inside rerunning/sentence_confidence/.
    from align_segments_hybrid import align_segments_hybrid


SEGMENTS_DIR = "writeup_results/sentence_confidence/segments"
FIVEMODEL_DIR = "writeup_results/grid/unanchored_fusion_naive_5model"
OUTPUT_DIR = "writeup_results/sentence_confidence/verbalized_confidence"

OLLAMA_HOST = "http://localhost:11434"
SELECTOR_MODEL_KEY = "gemma4"
JUDGE_MODEL = "phi4:14b"  # used only for alignment fallback
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]

DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))

VARIANT_PHRASINGS = {
    "confscore": (
        "a confidence score from 0 to 100 quantifying how confident you are "
        "that this segment preserves the spoken meaning"
    ),
    "confprobscore": (
        "a confidence score between 0 and 1 corresponding to the probability "
        "that this segment preserves the spoken meaning"
    ),
    "probscore": (
        "the probability between 0 and 1 that this segment preserves the spoken meaning"
    ),
}

VARIANT_SCALE = {
    "confscore": "0-100",
    "confprobscore": "0-1",
    "probscore": "0-1",
}

NO_MATCH_MARKER = "[NO MATCH]"

PROMPT_TEMPLATE = """The following are the available corresponding local excerpts from ASR
transcripts of the same spoken audio, around the point in question. Some
models may have no comparable content, marked as {no_match_marker}.

{source_block}

The fused transcript contains this fixed segment:

"{segment}"

Based only on the evidence in the excerpts above, give {variant_phrasing}.

Respond only:
Score: <value>
"""

_SCORE_PATTERN = re.compile(
    r"\*{0,2}score\*{0,2}:\*{0,2}\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def parse_score(raw_response, variant):
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
    path = os.path.join(
        SEGMENTS_DIR,
        f"sentence_segments_{dataset}_{split}.json",
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_5model_source_hyps(dataset, split):
    path = os.path.join(
        FIVEMODEL_DIR,
        f"unanchored_fusion_naive_5model_{dataset}_gemma4_{split}.json",
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return {
        s["dataset_index"]: s.get("source_hyps", {})
        for s in data.get("samples", [])
    }


def format_source_block(local_spans, all_models):
    lines = []
    for model in all_models:
        text = local_spans.get(model, NO_MATCH_MARKER)
        lines.append(f"Transcript ({model}): {text}")
    return "\n".join(lines)


def get_confidence(
    client,
    model_name,
    local_spans,
    all_models,
    segment,
    variant,
    retries=2,
):
    prompt = PROMPT_TEMPLATE.format(
        no_match_marker=NO_MATCH_MARKER,
        source_block=format_source_block(local_spans, all_models),
        segment=segment,
        variant_phrasing=VARIANT_PHRASINGS[variant],
    )

    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": 20,
                },
                keep_alive="30m",
                think=False,
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (verbalized confidence): {e}")
                return None

    return None


def build_local_evidence(
    client,
    source_hyps,
    fixed_segments,
    max_workers,
):
    """
    Align every fixed fused segment to a local span in every source ASR
    transcript.

    Returns:
        all_models: list[str]
        spans_by_model: dict[model][segment_index] -> local span or None
        methods_by_model: dict[model][segment_index] -> alignment method
    """
    all_models = list(source_hyps.keys())
    spans_by_model = {}
    methods_by_model = {}

    for model, raw_text in source_hyps.items():
        spans, methods, _stats = align_segments_hybrid(
            client,
            JUDGE_MODEL,
            raw_text or "",
            fixed_segments,
            max_workers=max_workers,
        )
        spans_by_model[model] = spans
        methods_by_model[model] = methods

    return all_models, spans_by_model, methods_by_model


def run_dataset(
    dataset,
    variant,
    client,
    split="test",
    max_workers=None,
    rerun=False,
    chunk_size=25,
):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    model_name = OLLAMA_MODELS[SELECTOR_MODEL_KEY]

    print(
        f"\n-- {dataset} | verbalized_confidence "
        f"({variant}, scale {VARIANT_SCALE[variant]}) split={split} --"
    )

    segments_data = load_segments(dataset, split)
    source_hyps_by_idx = load_5model_source_hyps(dataset, split)
    transcripts = segments_data.get("transcripts", [])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"verbalized_{variant}_{dataset}_{split}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and not rerun:
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        results = existing.get("transcripts", [])
        start_from = len(results)
        print(f"  Resuming from transcript {start_from}/{len(transcripts)}")
    else:
        results = []
        start_from = 0

    def save_progress():
        payload = {
            "progress": len(results),
            "transcripts": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    remaining = transcripts[start_from:]

    for chunk_start in range(0, len(remaining), chunk_size):
        chunk = remaining[chunk_start:chunk_start + chunk_size]

        # Build all local evidence first, one transcript at a time.
        prepared = []

        for t in chunk:
            idx = t["dataset_index"]
            source_hyps = source_hyps_by_idx.get(idx, {})

            valid_segments = [
                (pos, seg)
                for pos, seg in enumerate(t.get("segments", []))
                if seg.get("severity") is not None
            ]
            fixed_segments = [seg["segment"] for _, seg in valid_segments]

            if not fixed_segments:
                prepared.append((t, [], [], {}, {}))
                continue

            if not source_hyps:
                print(f"  WARNING: no source_hyps for {idx}")
                prepared.append((t, valid_segments, [], {}, {}))
                continue

            (
                all_models,
                spans_by_model,
                methods_by_model,
            ) = build_local_evidence(
                client,
                source_hyps,
                fixed_segments,
                max_workers=max_workers,
            )

            prepared.append(
                (
                    t,
                    valid_segments,
                    all_models,
                    spans_by_model,
                    methods_by_model,
                )
            )

        # Score all sentence segments in this chunk concurrently.
        work_items = []

        for (
            t,
            valid_segments,
            all_models,
            spans_by_model,
            methods_by_model,
        ) in prepared:
            idx = t["dataset_index"]

            for local_pos, (original_pos, seg) in enumerate(valid_segments):
                local_spans = {}
                alignment_methods = {}

                for model in all_models:
                    span_text = spans_by_model.get(model, {}).get(local_pos)
                    alignment_methods[model] = (
                        methods_by_model.get(model, {}).get(local_pos)
                    )
                    if span_text is not None:
                        local_spans[model] = span_text

                work_items.append(
                    (
                        idx,
                        original_pos,
                        seg,
                        local_spans,
                        all_models,
                        alignment_methods,
                    )
                )

        def _worker(item):
            _, _, seg, local_spans, all_models, _ = item

            if not local_spans:
                return None

            return get_confidence(
                client,
                model_name,
                local_spans,
                all_models,
                seg["segment"],
                variant,
            )

        raw_responses = run_concurrent(
            work_items,
            _worker,
            max_workers=max_workers,
            progress_every=25,
        )

        scores_by_idx = {}

        for item, raw in zip(work_items, raw_responses):
            (
                idx,
                original_pos,
                seg,
                _local_spans,
                _all_models,
                alignment_methods,
            ) = item

            scores_by_idx.setdefault(idx, {})[original_pos] = {
                "raw": raw,
                "score": parse_score(raw, variant),
                "alignment_methods": alignment_methods,
            }

        # Reassemble in the same output shape as before, with one additive
        # alignment_methods field for auditability.
        for t in chunk:
            idx = t["dataset_index"]
            out_segments = []

            for pos, seg in enumerate(t.get("segments", [])):
                if seg.get("severity") is None:
                    continue

                scored = scores_by_idx.get(idx, {}).get(
                    pos,
                    {
                        "raw": None,
                        "score": None,
                        "alignment_methods": {},
                    },
                )

                out_segments.append(
                    {
                        "segment": seg["segment"],
                        "severity": seg.get("severity"),
                        "flagged": seg.get("flagged"),
                        "verbalized_score": scored["score"],
                        "raw_response": scored["raw"],
                        "alignment_methods": scored["alignment_methods"],
                    }
                )

            results.append(
                {
                    "dataset_index": idx,
                    "segments": out_segments,
                }
            )

        save_progress()
        print(
            f"  chunk done: {len(results)}/"
            f"{len(remaining) + start_from} transcripts"
        )

    n_segments = sum(len(t["segments"]) for t in results)
    n_scored = sum(
        1
        for t in results
        for s in t["segments"]
        if s.get("verbalized_score") is not None
    )
    n_failed = n_segments - n_scored

    output = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "selector_model": SELECTOR_MODEL_KEY,
        "alignment_model": JUDGE_MODEL,
        "confidence_type": "retrospective_verbalized_confidence",
        "evidence_scope": "local_aligned_asr_spans",
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

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(
        f"\n  {n_scored}/{n_segments} segments scored "
        f"({n_failed} failed). Saved: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="commonvoice",
        choices=DATASETS,
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "test", "full"],
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(VARIANT_PHRASINGS.keys()),
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
    )
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()

    client = Client(host=OLLAMA_HOST)

    run_dataset(
        args.dataset,
        args.variant,
        client,
        split=args.split,
        max_workers=args.max_workers,
        rerun=args.rerun,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
