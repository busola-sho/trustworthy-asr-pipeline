"""
rerunning/grid/unanchored_fusion_naive_confidence_inline_full_v2.py

Same compliance fix as unanchored_fusion_naive_confidence_list_full_v2.py,
applied to the INLINE variant (word[conf] annotations directly in the
transcript text, no threshold). Uses the same non-destructive insertion
as the original inline scripts.

Writes to a NEW folder (writeup_results/grid/unanchored_fusion_naive_confidence_inline_full_v2/)
- the original (buggy) run is left untouched for comparison.

Usage:
    python rerunning/grid/unanchored_fusion_naive_confidence_inline_full_v2.py --dataset commonvoice --split dev --selector gemma4
"""

import json
import os
import re
import random
import argparse
from jiwer import wer
from ollama import Client

from src.judge import normalise, is_tag_only
from src.selector import (
    find_canonical_file, OLLAMA_MODELS, check_selector_available, load_samples,
)
from src.splits import get_indices_for_split
from src.concurrent_ollama import run_concurrent

OUTPUT_DIR = "writeup_results/grid/unanchored_fusion_naive_confidence_inline_full_v2"
OLLAMA_HOST = "http://localhost:11434"
ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]
DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))

SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Each transcript has every word annotated inline with its confidence score in square brackets (e.g. word[0.87]), on a 0.00-1.00 scale where higher means more confident. Use these scores as evidence when deciding which words to trust - lower scores indicate the model itself was less certain about that word.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Include the confidence annotations (e.g. "[0.87]") in your final output
- Explain your reasoning, analyze the confidence scores in prose, or add any commentary, headers, or bullet points

Your ENTIRE response must be the final transcript text and nothing else - no "Based on...", no "Analysis:", no markdown formatting of any kind.

Return only the final transcript, with confidence annotations removed, nothing else."""

RETRY_PROMPT_SUFFIX = """

REMINDER: your previous response included explanation, analysis, or commentary instead of just the transcript. Respond with ONLY the corrected transcript text - no headers, no bullet points, no "Based on...", no discussion of confidence scores. Just the plain transcript with no [conf] annotations."""

NONCOMPLIANCE_MARKERS = [
    "**", "##", "\n- ", "\n* ", "Based on", "Analysis:", "Conclusion:",
    "Verdict:", "confidence score", "Confidence Score", "Transcript 1",
    "Transcript 2", "Transcript 3", "Transcript 4", "the two transcripts",
]


def is_compliant(response: str, input_hyps: list) -> bool:
    if any(marker in response for marker in NONCOMPLIANCE_MARKERS):
        return False
    max_input_words = max(len(h.split()) for h in input_hyps)
    response_words = len(response.split())
    if response_words > max_input_words * 2 + 20:
        return False
    return True


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def build_inline_confidence_transcript(hyp: str, segments: list) -> str:
    if not segments or not any(seg.get("confidence") is not None for seg in segments):
        return hyp

    parts = []
    cursor = 0
    matched_any = False

    for seg in segments:
        word = str(seg.get("word", "")).strip()
        conf = seg.get("confidence")
        if not word:
            continue

        match = re.search(re.escape(word), hyp[cursor:], flags=re.IGNORECASE)
        if match is None:
            continue

        start = cursor + match.start()
        end = cursor + match.end()
        surface_word = hyp[start:end]

        parts.append(hyp[cursor:start])
        if conf is not None:
            parts.append(f"{surface_word}[{conf:.2f}]")
        else:
            parts.append(surface_word)

        cursor = end
        matched_any = True

    if not matched_any:
        return hyp

    parts.append(hyp[cursor:])
    return "".join(parts)


def strip_conf_annotations(text: str) -> str:
    """Safety net: strip any remaining word[0.87]-style annotations
    from the final output, in case the model echoed them back despite
    instructions not to."""
    return re.sub(r'\[\d+\.\d+\]', '', text)


def get_rotated_order(models: list, seed_key: int) -> list:
    order = list(models)
    rng = random.Random(seed_key)
    rng.shuffle(order)
    return order


def compute_num_predict(hyps: list) -> int:
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_call(client, model_name, messages, num_predict):
    try:
        response = client.chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
            keep_alive="30m",
            think=False,
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"  ERROR (select): {e}")
        return None


def ollama_select_with_compliance_check(client, model_name, model_order, text_by_model, num_predict):
    hyp_block = "\n".join(
        f"Transcript {i+1} (model: {m}): {text_by_model[m]}" for i, m in enumerate(model_order)
    )
    messages = [
        {"role": "system", "content": SELECTOR_PROMPT},
        {"role": "user", "content": hyp_block},
    ]

    response = ollama_call(client, model_name, messages, num_predict)
    if response is None:
        return None, False, False, True

    input_hyps = list(text_by_model.values())
    if is_compliant(response, input_hyps):
        return response, True, False, False

    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": RETRY_PROMPT_SUFFIX})
    retry_response = ollama_call(client, model_name, messages, num_predict)

    if retry_response is None:
        return response, False, True, True

    if is_compliant(retry_response, input_hyps):
        return retry_response, False, True, False

    return retry_response, False, True, True


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False, split="dev", max_workers=None):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    selector_model = OLLAMA_MODELS[selector_key]
    print(f"\n-- {dataset} | unanchored_fusion_naive_confidence_inline_full_v2 (CONCURRENT) "
          f"selector={selector_key} split={split} --")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"unanchored_fusion_naive_confidence_inline_full_v2_{dataset}_{selector_key}_{split}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results = []
        start_from = 0

    def save_progress():
        payload = {"progress": len(results), "samples": results}
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    remaining_indices = indices[start_from:]
    work_items = []
    skip_slots = {}

    for idx in remaining_indices:
        samples_by_model = {m: model_samples[m].get(idx) for m in ASR_MODELS}

        if not all(samples_by_model.values()):
            skip_slots[idx] = {"ref": None, "hyp": None, "severity": None,
                               "sample_WER": None, "error": True,
                               "error_reason": "missing sample from one or more models",
                               "dataset_index": idx}
            continue

        ref = samples_by_model["qwen"]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            skip_slots[idx] = {"ref": ref, "hyp": None, "severity": None,
                               "sample_WER": None, "skipped": True,
                               "skip_reason": "ignore_time_segment", "dataset_index": idx}
            continue

        if is_tag_only(ref):
            skip_slots[idx] = {"ref": ref, "hyp": None, "severity": None,
                               "sample_WER": None, "skipped": True,
                               "skip_reason": "tag_only_reference", "dataset_index": idx}
            continue

        text_by_model = {
            m: build_inline_confidence_transcript(samples_by_model[m]["hyp"], samples_by_model[m].get("segments"))
            for m in ASR_MODELS
        }
        model_order = get_rotated_order(ASR_MODELS, seed_key=idx)
        num_predict = compute_num_predict(list(text_by_model.values()))

        work_items.append((idx, ref, text_by_model, model_order, num_predict))

    print(f"  {len(work_items)} samples queued for concurrent Ollama calls "
          f"({len(skip_slots)} skipped without needing a call)")

    def _worker(item):
        _, _, text_by_model, model_order, num_predict = item
        return ollama_select_with_compliance_check(client, selector_model, model_order, text_by_model, num_predict)

    call_outputs = run_concurrent(work_items, _worker, max_workers=max_workers, progress_every=10)
    call_results = {item[0]: (item, out) for item, out in zip(work_items, call_outputs)}

    n_compliant_first_try = 0
    n_needed_retry = 0
    n_retry_failed = 0

    for idx in remaining_indices:
        if idx in skip_slots:
            results.append(skip_slots[idx])
            continue

        item, call_out = call_results[idx]
        _, ref, text_by_model, model_order, _ = item

        if call_out is None or call_out[0] is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        raw_hyp, was_compliant_first_try, needed_retry, retry_failed = call_out
        if was_compliant_first_try:
            n_compliant_first_try += 1
        if needed_retry:
            n_needed_retry += 1
        if retry_failed:
            n_retry_failed += 1

        best_hyp = strip_conf_annotations(raw_hyp)
        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref": ref,
            "hyp": best_hyp,
            "source_texts_annotated": text_by_model,
            "model_order": model_order,
            "was_compliant_first_try": was_compliant_first_try,
            "needed_retry": needed_retry,
            "retry_failed": retry_failed,
            "sample_WER": sample_wer_val,
            "severity": None,
            "dataset_index": idx,
        })

        if len(results) % 10 == 0:
            save_progress()

    save_progress()

    valid = [r for r in results if not r.get("skipped") and not r.get("error") and r.get("sample_WER") is not None]
    corpus_wer = wer([normalise(r["ref"]) for r in valid], [normalise(r["hyp"]) for r in valid]) if valid else None

    n_total = n_compliant_first_try + n_needed_retry
    compliance_first_try_rate = n_compliant_first_try / n_total if n_total else None
    compliance_final_rate = (n_total - n_retry_failed) / n_total if n_total else None

    output = {
        "selector": selector_key,
        "approach": "unanchored_fusion_naive_confidence_inline_full_v2",
        "strategy": "unanchored_fusion",
        "context_condition": "confidence_inline_no_threshold_v2_compliance_checked",
        "asr_models": ASR_MODELS,
        "phase": "selector_only - severity not yet judged",
        "dataset": dataset,
        "split": split,
        "compliance_first_try_rate": compliance_first_try_rate,
        "compliance_final_rate": compliance_final_rate,
        "n_needed_retry": n_needed_retry,
        "n_retry_failed": n_retry_failed,
        "subset_indices": indices,
        "corpus_wer": corpus_wer,
        "num_samples": len(valid),
        "samples": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "-"
    comp_str = f"{compliance_first_try_rate*100:.1f}%" if compliance_first_try_rate is not None else "-"
    final_str = f"{compliance_final_rate*100:.1f}%" if compliance_final_rate is not None else "-"
    print(f"\n  WER: {wer_str}  Compliance (1st try): {comp_str}  Compliance (after retry): {final_str}  (N={len(valid)})")
    print(f"  Retries needed: {n_needed_retry}  Retry still failed: {n_retry_failed}")
    print(f"  Saved: {output_path}")
    print(f"\n  Now run severity: python rerunning/add_severity_to_existing_concurrent.py --files {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector", default="gemma4", choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split,
                max_workers=args.max_workers)


if __name__ == "__main__":
    main()
