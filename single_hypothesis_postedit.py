"""Single-hypothesis Gemma post-editing baseline.

Tests how much improvement comes from LLM post-editing alone, without
multi-model evidence. The development-selected individual ASR system (Qwen
by default) supplies one hypothesis to Gemma 4. Results can be compared with:

  1. the unchanged source ASR hypothesis; and
  2. four-model Unanchored Fusion.

This script performs Gemma inference only. Add Phi-4 severity judgements in a
separate pass using rerunning/add_severity_to_existing_concurrent.py.

Examples:
    python single_hypothesis_postedit.py --dataset commonvoice --split dev
    python single_hypothesis_postedit.py --dataset commonvoice --split test
    python single_hypothesis_postedit.py --dataset shetland --split full
"""

import argparse
import json
import os
import time
from pathlib import Path

from jiwer import wer
from ollama import Client

from src.concurrent_ollama import run_concurrent
from src.judge import is_tag_only, normalise
from src.selector import (
    OLLAMA_MODELS,
    check_selector_available,
    find_canonical_file,
    load_samples,
)
from src.splits import get_indices_for_split


OUTPUT_DIR = Path("writeup_results/single_hypothesis_postedit")
OLLAMA_HOST = "http://localhost:11434"
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
SOURCE_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))
DEFAULT_BATCH_SIZE = 50


POSTEDIT_PROMPT = """You are given one transcript produced by an automatic
speech-recognition system.

Produce the most accurate transcript you can from this hypothesis. Preserve
the speaker's wording, meaning, sentence order, dialect and informal speech.
Make only minimal corrections where the transcript contains a clear speech-
recognition error that can be resolved from linguistic context.

You MUST NOT:
- paraphrase or stylistically rewrite the transcript
- summarise or omit meaningful content
- invent names, numbers, events or other details not supported by the input
- standardise dialect or informal expressions unnecessarily

If a correction is uncertain, retain the original wording. Return only the
final transcript, with no explanation."""


def get_indexed_samples(model, dataset):
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return (
        path,
        {
            sample["sample_index"]: sample
            for sample in samples
            if sample.get("sample_index") is not None
        },
    )


def compute_num_predict(hypothesis):
    estimated = int(len((hypothesis or "").split()) * 1.3) + 50
    return max(300, min(estimated, 2048))


def postedit(client, model_name, hypothesis, retries=2):
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": POSTEDIT_PROMPT},
                    {
                        "role": "user",
                        "content": f"ASR transcript:\n{hypothesis}",
                    },
                ],
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": compute_num_predict(hypothesis),
                },
                keep_alive="30m",
                think=False,
            )
            output = response.message.content.strip()
            return output or None
        except Exception as error:
            if attempt == retries:
                print(f"  ERROR (post-edit): {error}")
                return None
            time.sleep(1.0)
    return None


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def run_dataset(
    dataset,
    split,
    source_model,
    selector_key,
    client,
    max_workers,
    batch_size,
    max_samples=None,
    rerun=False,
):
    selector_model = OLLAMA_MODELS[selector_key]
    source_path, source_samples = get_indexed_samples(source_model, dataset)

    if dataset == "shetland" and split != "full":
        raise ValueError("Shetland should be run with --split full")

    indices = list(get_indices_for_split(dataset, split))
    if max_samples is not None:
        indices = indices[:max_samples]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"single_hypothesis_postedit_{source_model}_{dataset}_"
        f"{selector_key}_{split}.json"
    )
    output_path = OUTPUT_DIR / filename

    if output_path.exists() and not rerun:
        with output_path.open(encoding="utf-8") as file:
            existing = json.load(file)
        results_by_index = {
            sample["dataset_index"]: sample
            for sample in existing.get("samples", [])
        }
        print(
            f"\n-- {dataset} | {source_model} -> {selector_key} | {split} --"
            f"\n  Resuming with {len(results_by_index)}/{len(indices)} saved"
        )
    else:
        results_by_index = {}
        print(f"\n-- {dataset} | {source_model} -> {selector_key} | {split} --")

    def ordered_results():
        return [results_by_index[index] for index in indices if index in results_by_index]

    def save_progress(final=False):
        samples = ordered_results()
        valid = [
            sample
            for sample in samples
            if not sample.get("skipped")
            and not sample.get("error")
            and sample.get("hyp") is not None
        ]

        source_corpus_wer = None
        postedit_corpus_wer = None
        if valid:
            references = [normalise(sample["ref"]) for sample in valid]
            source_hypotheses = [normalise(sample["source_hyp"]) for sample in valid]
            edited_hypotheses = [normalise(sample["hyp"]) for sample in valid]
            source_corpus_wer = wer(references, source_hypotheses)
            postedit_corpus_wer = wer(references, edited_hypotheses)

        payload = {
            "approach": "single_hypothesis_postedit",
            "source_model": source_model,
            "selector": selector_key,
            "selector_model": selector_model,
            "dataset": dataset,
            "split": split,
            "source_file": str(source_path),
            "prompt": POSTEDIT_PROMPT,
            "subset_indices": indices,
            "progress": len(samples),
            "num_samples": len(valid),
            "source_corpus_wer": source_corpus_wer,
            "corpus_wer": postedit_corpus_wer,
            "phase": (
                "postedit_complete - severity not yet judged"
                if final
                else "postedit_in_progress - severity not yet judged"
            ),
            "samples": samples,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return source_corpus_wer, postedit_corpus_wer, len(valid)

    work_items = []
    for index in indices:
        if index in results_by_index:
            continue

        sample = source_samples.get(index)
        if sample is None:
            results_by_index[index] = {
                "ref": None,
                "source_hyp": None,
                "hyp": None,
                "severity": None,
                "sample_WER": None,
                "source_sample_WER": None,
                "error": True,
                "error_reason": "sample missing from source model",
                "dataset_index": index,
            }
            continue

        reference = sample.get("ref")
        hypothesis = sample.get("hyp")

        if reference and "IGNORE_TIME_SEGMENT_IN_SCORING" in reference:
            results_by_index[index] = {
                "ref": reference,
                "source_hyp": hypothesis,
                "hyp": None,
                "severity": None,
                "sample_WER": None,
                "source_sample_WER": None,
                "skipped": True,
                "skip_reason": "ignore_time_segment",
                "dataset_index": index,
            }
        elif reference and is_tag_only(reference):
            results_by_index[index] = {
                "ref": reference,
                "source_hyp": hypothesis,
                "hyp": None,
                "severity": None,
                "sample_WER": None,
                "source_sample_WER": None,
                "skipped": True,
                "skip_reason": "tag_only_reference",
                "dataset_index": index,
            }
        elif not reference or hypothesis is None:
            results_by_index[index] = {
                "ref": reference,
                "source_hyp": hypothesis,
                "hyp": None,
                "severity": None,
                "sample_WER": None,
                "source_sample_WER": None,
                "error": True,
                "error_reason": "missing reference or source hypothesis",
                "dataset_index": index,
            }
        else:
            work_items.append((index, reference, hypothesis))

    print(
        f"  {len(indices)} requested; {len(work_items)} queued; "
        f"{len(results_by_index)} already resolved"
    )

    def worker(item):
        _, _, hypothesis = item
        return postedit(client, selector_model, hypothesis)

    completed = 0
    for batch_number, batch in enumerate(chunked(work_items, batch_size), start=1):
        outputs = run_concurrent(
            batch,
            worker,
            max_workers=max_workers,
            progress_every=0,
        )

        for (index, reference, source_hypothesis), edited in zip(batch, outputs):
            if edited is None:
                results_by_index[index] = {
                    "ref": reference,
                    "source_hyp": source_hypothesis,
                    "hyp": None,
                    "severity": None,
                    "sample_WER": None,
                    "source_sample_WER": wer(
                        normalise(reference), normalise(source_hypothesis)
                    ),
                    "error": True,
                    "error_reason": "Gemma post-edit failed",
                    "dataset_index": index,
                }
                continue

            results_by_index[index] = {
                "ref": reference,
                "source_hyp": source_hypothesis,
                "hyp": edited,
                "source_model": source_model,
                "source_sample_WER": wer(
                    normalise(reference), normalise(source_hypothesis)
                ),
                "sample_WER": wer(normalise(reference), normalise(edited)),
                "severity": None,
                "dataset_index": index,
            }

        completed += len(batch)
        save_progress(final=False)
        print(
            f"  batch {batch_number}: {completed}/{len(work_items)} "
            "new samples processed"
        )

    source_wer, edited_wer, valid_count = save_progress(final=True)
    source_text = "--" if source_wer is None else f"{source_wer * 100:.2f}%"
    edited_text = "--" if edited_wer is None else f"{edited_wer * 100:.2f}%"
    print(f"  Source WER: {source_text}")
    print(f"  Post-edit WER: {edited_text} (N={valid_count})")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="commonvoice")
    parser.add_argument("--split", choices=["dev", "test", "full"], default="dev")
    parser.add_argument(
        "--source-model",
        choices=SOURCE_MODELS,
        default="qwen",
        help="Development-selected individual ASR source (default: qwen)",
    )
    parser.add_argument(
        "--selector",
        choices=list(OLLAMA_MODELS.keys()),
        default="gemma4",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(
            f"dataset={args.dataset} split={args.split} "
            f"source={args.source_model} selector={args.selector}"
        )
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        raise SystemExit(f"Selector model not available: {OLLAMA_MODELS[args.selector]}")

    run_dataset(
        dataset=args.dataset,
        split=args.split,
        source_model=args.source_model,
        selector_key=args.selector,
        client=client,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        rerun=args.rerun,
    )


if __name__ == "__main__":
    main()
