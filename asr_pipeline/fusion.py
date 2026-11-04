"""Combine four transcripts using selection, fusion, ROVER or MBR."""

import random
import re

from .config import SOURCE_MODELS
from .files import read_transcript_files, read_json, check_new_output_path, write_json
from .prompts import ANCHORED_PROMPT, SELECTION_PROMPT, UNANCHORED_PROMPT
from .rover import rover_combine
from .text_normalization import normalise

METHODS = ("selection", "anchored", "unanchored", "rover", "mbr")


def mbr_select(hypotheses):
    from jiwer import wer

    normalized = {model: normalise(hypotheses[model]) for model in SOURCE_MODELS}
    risks = {
        model: sum(
            wer(normalized[other], normalized[model])
            for other in SOURCE_MODELS
            if other != model
        )
        / 3
        for model in SOURCE_MODELS
    }
    chosen = min(SOURCE_MODELS, key=risks.get)
    return chosen, hypotheses[chosen], risks


def combine(
    hypotheses, method, *, client=None, model="gemma4:12b", index=0, segments=None
):
    if set(hypotheses) != set(SOURCE_MODELS):
        raise ValueError("Supply one transcript from each of the four ASR models")
    if method not in METHODS:
        raise ValueError(f"Unknown fusion method: {method}")
    if method == "rover":
        return rover_combine(hypotheses, list(SOURCE_MODELS), segments or {}), {}
    if method == "mbr":
        chosen, text, risks = mbr_select(hypotheses)
        return text, {"selected_model": chosen, "risk_by_model": risks}
    if client is None:
        raise ValueError("Supply an Ollama client for selection or fusion")
    order = list(SOURCE_MODELS)
    random.Random(index).shuffle(order)
    if method == "selection":
        prompt = SELECTION_PROMPT.format(
            **dict(zip("abcd", [hypotheses[m] for m in order]))
        )
        messages = [{"role": "user", "content": prompt}]
    elif method == "anchored":
        order = list(SOURCE_MODELS)
        messages = [{"role": "user", "content": ANCHORED_PROMPT.format(**hypotheses)}]
    else:
        block = "\n".join(
            f"Transcript {i + 1}: {hypotheses[m]}" for i, m in enumerate(order)
        )
        messages = [
            {"role": "system", "content": UNANCHORED_PROMPT},
            {"role": "user", "content": block},
        ]
    predict = (
        10
        if method == "selection"
        else max(
            300,
            min(int(max(len(h.split()) for h in hypotheses.values()) * 1.3) + 50, 2048),
        )
    )
    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": 0, "num_ctx": 4096, "num_predict": predict},
        keep_alive="30m",
        think=False,
    )
    raw = response.message.content.strip()
    metadata = {"model_order": order}
    if method == "selection":
        label = re.search(r"\b([ABCD])\b", raw.upper())
        if label is None:
            raise ValueError(f"Selection did not return an A/B/C/D label: {raw!r}")
        chosen = order["ABCD".index(label.group(1))]
        metadata.update(selected_model=chosen, selection_raw_response=raw)
        return hypotheses[chosen], metadata
    return raw, metadata


def combine_transcript_files(
    transcript_files_path,
    output,
    method,
    *,
    client=None,
    model="gemma4:12b",
    splits_file=None,
):
    check_new_output_path(output)
    transcript_files, sources = read_transcript_files(transcript_files_path)
    dataset, split = transcript_files["dataset"], transcript_files["split"]
    indices = sorted(sources[SOURCE_MODELS[0]])
    if splits_file is not None:
        assignment = read_json(splits_file)["current_assignments"][dataset][split]
        if not set(assignment).issubset(indices):
            raise ValueError("ASR inputs lack assigned split IDs")
        indices = sorted(assignment)
    results = []
    for index in indices:
        samples = {m: sources[m][index] for m in SOURCE_MODELS}
        references = {
            s.get("ref") for s in samples.values() if s.get("ref") is not None
        }
        if len(references) > 1:
            raise ValueError(f"Source references disagree at sample {index}")
        reference = next(iter(references), "")
        hypotheses = {m: samples[m].get("hyp") or "" for m in SOURCE_MODELS}
        base = {
            "dataset_index": index,
            "ref": reference,
            "source_hyps": hypotheses,
            "source_segments": {
                m: samples[m].get("segments", []) for m in SOURCE_MODELS
            },
        }
        if any(s.get("skipped") or s.get("error") for s in samples.values()):
            results.append({**base, "hyp": None, "skipped": True})
            continue
        text, metadata = combine(
            hypotheses,
            method,
            client=client,
            model=model,
            index=index,
            segments={m: samples[m].get("segments", []) for m in SOURCE_MODELS},
        )
        results.append({**base, "hyp": text, **metadata})
    write_json(
        output,
        {
            "dataset": dataset,
            "split": split,
            "approach": method,
            "selector": model if method in METHODS[:3] else None,
            "asr_models": list(SOURCE_MODELS),
            "input_manifest": str(transcript_files_path),
            "samples": results,
        },
    )
