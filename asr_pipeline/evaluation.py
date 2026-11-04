"""Compare transcripts with references and save WER and severity."""

import re

from .files import read_json, check_new_output_path, write_json
from .prompts import DIRECT_SEVERITY_PROMPT
from .text_normalization import normalise, is_tag_only


def parse_severity(text):
    match = re.search(r"severity\s*:\s*([0-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"(?<!\d)[0-4](?!\d)", text)
    return int(fallback[-1]) if fallback else None


def evaluate_sample(sample, client, model):
    if sample.get("skipped") or sample.get("error"):
        return {**sample, "severity": None}
    reference, hypothesis = sample.get("ref") or "", sample.get("hyp") or ""
    if not reference.strip() or is_tag_only(reference):
        return {**sample, "severity": None}
    if sample.get("severity") is not None:
        return dict(sample)
    if not hypothesis.strip():
        return {**sample, "severity": 4}
    raw = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": DIRECT_SEVERITY_PROMPT.format(
                    reference=reference, hypothesis=hypothesis
                ),
            }
        ],
        options={"temperature": 0},
        think=False,
    ).message.content
    severity = parse_severity(raw)
    if severity is None:
        raise ValueError("Judge did not return a severity from 0 to 4")
    return {**sample, "severity": severity, "judge_raw_response": raw}


def evaluate_transcript_file(input_path, output, client, model="phi4:14b"):
    from jiwer import wer

    check_new_output_path(output)
    data = read_json(input_path)
    samples = [evaluate_sample(s, client, model) for s in data["samples"]]
    scored = [s for s in samples if s.get("severity") is not None]
    wer_samples = [
        s
        for s in samples
        if not s.get("skipped")
        and not s.get("error")
        and (s.get("ref") or "").strip()
        and not is_tag_only(s["ref"])
    ]
    data = {
        **data,
        "samples": samples,
        "judge": model,
        "mean_severity": sum(s["severity"] for s in scored) / len(scored)
        if scored
        else None,
        "mar": sum(s["severity"] >= 2 for s in scored) / len(scored)
        if scored
        else None,
        "corpus_wer": wer(
            [normalise(s["ref"]) for s in wer_samples],
            [normalise(s.get("hyp") or "") for s in wer_samples],
        )
        if wer_samples
        else None,
    }
    write_json(output, data)
