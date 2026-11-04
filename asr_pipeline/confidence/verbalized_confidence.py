"""Four-source sentence confidence: verbalized_confidence."""

from asr_pipeline.confidence.paths import SentenceConfidenceFiles
from asr_pipeline.confidence.progress import load_progress, save_progress

import argparse
import json
import os
import re

from ollama import Client

from asr_pipeline.parallel_calls import call_models_in_parallel
from asr_pipeline.config import OLLAMA_MODELS
from asr_pipeline.confidence.inputs import (
    SOURCE_MODELS,
    fusion_result_path,
    load_source_hypotheses,
    load_confidence_segments,
)

from asr_pipeline.confidence.align_sentence_spans import align_sentence_spans


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


def load_segments(dataset, split, paths=None):
    paths = paths or SentenceConfidenceFiles()
    return load_confidence_segments(dataset, split, paths=paths)


def load_four_model_source_hyps(dataset, split, paths=None):
    paths = paths or SentenceConfidenceFiles()
    return load_source_hypotheses(dataset, split, paths=paths)


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
    if set(source_hyps) != set(SOURCE_MODELS):
        raise ValueError("Expected exactly the four main ASR hypotheses")
    all_models = list(SOURCE_MODELS)
    spans_by_model = {}
    methods_by_model = {}

    for model, raw_text in source_hyps.items():
        spans, methods, _stats = align_sentence_spans(
            client,
            JUDGE_MODEL,
            raw_text or "",
            fixed_segments,
            max_workers=max_workers,
        )
        spans_by_model[model] = spans
        methods_by_model[model] = methods

    return all_models, spans_by_model, methods_by_model


def prepare_confidence_requests(
    chunk, source_hypotheses, client, max_workers, include_unlabelled
):
    # Build all local evidence first, one transcript at a time.
    prepared = []

    for transcript in chunk:
        transcript_index = transcript["dataset_index"]
        source_hyps = source_hypotheses.get(transcript_index, {})

        valid_segments = [
            (segment_position, segment)
            for segment_position, segment in enumerate(transcript.get("segments", []))
            if include_unlabelled or segment.get("severity") is not None
        ]
        fixed_segments = [segment["segment"] for _, segment in valid_segments]

        if not fixed_segments:
            prepared.append((transcript, [], [], {}, {}))
            continue

        if not source_hyps:
            print(f"  WARNING: no source_hyps for {transcript_index}")
            prepared.append((transcript, valid_segments, [], {}, {}))
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
                transcript,
                valid_segments,
                all_models,
                spans_by_model,
                methods_by_model,
            )
        )

    items = []

    for (
        transcript,
        valid_segments,
        all_models,
        spans_by_model,
        methods_by_model,
    ) in prepared:
        transcript_index = transcript["dataset_index"]

        for local_position, (original_position, segment) in enumerate(valid_segments):
            local_spans = {}
            alignment_methods = {}

            for model in all_models:
                span_text = spans_by_model.get(model, {}).get(local_position)
                alignment_methods[model] = methods_by_model.get(model, {}).get(
                    local_position
                )
                if span_text is not None:
                    local_spans[model] = span_text

            items.append(
                (
                    transcript_index,
                    original_position,
                    segment,
                    local_spans,
                    all_models,
                    alignment_methods,
                )
            )

    return items


def assemble_confidence_results(
    chunk, items, raw_responses, variant, include_unlabelled
):
    results = []
    scores_by_transcript = {}

    for item, raw in zip(items, raw_responses):
        (
            transcript_index,
            original_position,
            segment,
            _local_spans,
            _all_models,
            alignment_methods,
        ) = item

        scores_by_transcript.setdefault(transcript_index, {})[original_position] = {
            "raw": raw,
            "score": parse_score(raw, variant),
            "alignment_methods": alignment_methods,
        }

    # Reassemble in the same output shape as before, with one additive
    # alignment_methods field for auditability.
    for transcript in chunk:
        transcript_index = transcript["dataset_index"]
        out_segments = []

        for segment_position, segment in enumerate(transcript.get("segments", [])):
            if segment.get("severity") is None and not include_unlabelled:
                continue

            scored = scores_by_transcript.get(transcript_index, {}).get(
                segment_position,
                {
                    "raw": None,
                    "score": None,
                    "alignment_methods": {},
                },
            )

            out_segments.append(
                {
                    "segment": segment["segment"],
                    "severity": segment.get("severity"),
                    "flagged": segment.get("flagged"),
                    "verbalized_score": scored["score"],
                    "raw_response": scored["raw"],
                    "alignment_methods": scored["alignment_methods"],
                }
            )

        results.append(
            {
                "dataset_index": transcript_index,
                "segments": out_segments,
            }
        )

    return results


def run_dataset(
    dataset,
    variant,
    client,
    split="test",
    max_workers=None,
    rerun=False,
    chunk_size=25,
    *,
    segments_data=None,
    source_hypotheses=None,
    fusion_file=None,
    output_dir=None,
    include_unlabelled=False,
    paths=None,
):
    paths = paths or SentenceConfidenceFiles()
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    model_name = OLLAMA_MODELS[SELECTOR_MODEL_KEY]

    print(
        f"\n-- {dataset} | verbalized_confidence "
        f"({variant}, scale {VARIANT_SCALE[variant]}) split={split} --"
    )

    if segments_data is None:
        segments_data = load_segments(dataset, split, paths=paths)
    source_hypotheses = source_hypotheses
    if source_hypotheses is None:
        source_hypotheses = load_four_model_source_hyps(dataset, split, paths=paths)
    transcripts = segments_data.get("transcripts", [])

    output_dir = output_dir or paths.verbalized_dir
    os.makedirs(output_dir, exist_ok=True)
    filename = f"verbalized_{variant}_{dataset}_{split}.json"
    output_path = os.path.join(output_dir, filename)

    results = load_progress(output_path, rerun)
    start_from = len(results)
    if start_from:
        print(f"  Resuming from transcript {start_from}/{len(transcripts)}")

    remaining = transcripts[start_from:]

    for chunk_start in range(0, len(remaining), chunk_size):
        chunk = remaining[chunk_start : chunk_start + chunk_size]

        items = prepare_confidence_requests(
            chunk,
            source_hypotheses,
            client,
            max_workers,
            include_unlabelled,
        )

        def score_request(item):
            _, _, segment, local_spans, all_models, _ = item

            if not local_spans:
                return None

            return get_confidence(
                client,
                model_name,
                local_spans,
                all_models,
                segment["segment"],
                variant,
            )

        raw_responses = call_models_in_parallel(
            items,
            score_request,
            max_workers=max_workers,
            progress_every=25,
        )

        results.extend(
            assemble_confidence_results(
                chunk,
                items,
                raw_responses,
                variant,
                include_unlabelled,
            )
        )

        save_progress(output_path, results)
        print(f"  chunk done: {len(results)}/{len(remaining) + start_from} transcripts")

    segment_count = sum(len(transcript["segments"]) for transcript in results)
    scored_count = sum(
        1
        for transcript in results
        for s in transcript["segments"]
        if s.get("verbalized_score") is not None
    )
    failed_count = segment_count - scored_count

    output = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "selector_model": SELECTOR_MODEL_KEY,
        "fusion_input": str(
            fusion_file or fusion_result_path(dataset, split, paths=paths)
        ),
        "asr_models": list(SOURCE_MODELS),
        "alignment_model": JUDGE_MODEL,
        "confidence_type": "retrospective_verbalized_confidence",
        "evidence_scope": "local_aligned_asr_spans",
        "target": "meaning_preservation",
        "prompt_variant": variant,
        "score_scale": VARIANT_SCALE[variant],
        "temperature": 0,
        "prompt_template": PROMPT_TEMPLATE,
        "num_transcripts": len(results),
        "num_segments": segment_count,
        "num_segments_scored": scored_count,
        "num_segments_failed": failed_count,
        "transcripts": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(
        f"\n  {scored_count}/{segment_count} segments scored "
        f"({failed_count} failed). Saved: {output_path}"
    )

    return output


def main(argv=None, paths=None):
    paths = paths or SentenceConfidenceFiles()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="commonvoice",
        choices=[*DATASETS, "clip"],
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
        "--parallel-calls",
        "--max-workers",
        dest="max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
    )
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument(
        "--include-unlabelled",
        action="store_true",
        help="also score sentences that have no evaluation labels",
    )
    args = parser.parse_args(argv)

    client = Client(host=paths.ollama_url)

    run_dataset(
        args.dataset,
        args.variant,
        client,
        split=args.split,
        max_workers=args.max_workers,
        rerun=args.rerun,
        chunk_size=args.chunk_size,
        paths=paths,
        include_unlabelled=args.include_unlabelled,
    )


if __name__ == "__main__":
    main()
