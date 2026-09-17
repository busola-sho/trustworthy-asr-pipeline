"""
fusion_vs_baseline_confidence.py

Controlled experiment: does ensemble fusion make sentence-level
confidence more informative (better correlated with severity) than a
single baseline model? Shetland only, fusion vs qwen and fusion vs
whisperx.

DESIGN (per review): reuses the EXISTING fixed segments and their
corresponding_span from sentence_segments_shetland_full.json - the
same unit of comparison for both conditions, removing segmentation
itself as a variable. For each fixed segment:
  1. Reconstruct the baseline model's own text corresponding to that
     segment (same cursor+SequenceMatcher search as cross_model_
     agreement.py, but mapped back onto the ORIGINAL cased/punctuated
     text, not the lowercased/stripped tokens used for matching -
     judging or scoring confidence on a degraded reconstruction would
     bias both measurements for reasons unrelated to the actual
     question).
  2. Judge the baseline's reconstructed text against the SAME
     corresponding_span already used to judge fusion - no new
     alignment step, matching what fusion's severity was judged
     against.
  3. Run the SAME blinded, transcript-only confidence prompt on BOTH
     conditions (fusion segment + full fused transcript; baseline
     segment + full baseline transcript) - no model identity revealed,
     no cross-model evidence in either condition, only the transcript
     source differs.
  4. Compare confidence-severity Spearman per condition.

Usage:
    python fusion_vs_baseline_confidence.py --baseline qwen
    python fusion_vs_baseline_confidence.py --baseline whisperx
"""

import json
import os
import re
import difflib
import argparse
from ollama import Client
from scipy.stats import spearmanr

from src.selector import find_canonical_file, load_samples
from severity_judge_prompts import SEVERITY_RUBRIC

SEGMENTS_DIR = "writeup_results/sentence_confidence/segments"
OUTPUT_DIR = "writeup_results/sentence_confidence/fusion_vs_baseline"
OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL_KEY = "gemma4"
MAX_EXTRA = 10

from src.selector import OLLAMA_MODELS

SINGLE_TRANSCRIPT_PROMPT_TEMPLATE = """The following transcript was produced by an ASR system
for spoken audio:

Full transcript:
{full_transcript}

Evaluate the following fixed segment from that transcript:

"{segment}"

Based only on the transcript and its surrounding context, give a
confidence score from 0 to 100 quantifying how confident you are that
this segment preserves the spoken meaning for the fixed segment.

Respond only:
Score: <value>
"""

JUDGE_PROMPT = f"""You are judging whether a machine transcription segment preserves
the meaning of its corresponding reference span.

{SEVERITY_RUBRIC}

Reference span:
{{reference_span}}

Transcribed segment:
{{segment}}

Respond only:
Severity: <0-4>
Justification: <one sentence>"""

_SCORE_PATTERN = re.compile(r'\*{0,2}score\*{0,2}:\*{0,2}\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)
_SEVERITY_PATTERN = re.compile(r"severity\s*:\s*([0-4])", re.IGNORECASE)


def parse_confidence(raw):
    if raw is None:
        return None
    m = _SCORE_PATTERN.search(raw)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if 0 <= v <= 100 else None


def parse_severity(raw):
    if raw is None:
        return None
    m = _SEVERITY_PATTERN.search(raw)
    if m:
        return int(m.group(1))
    fb = re.findall(r"(?<!\d)[0-4](?!\d)", raw)
    return int(fb[-1]) if fb else None


def load_baseline_raw_text(dataset, model):
    """Returns {dataset_index: raw_hyp_text} - ORIGINAL cased/
    punctuated text, not tokenized."""
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s.get("hyp", "") for s in samples if s.get("sample_index") is not None}


def tokenize_with_offsets(text):
    """Returns (lowercased_tokens, original_spans) - original_spans[i]
    is (start, end) character offset of token i in the ORIGINAL text,
    so a matched token range can be mapped back to properly-cased,
    punctuated original text instead of the lowercased/stripped form
    used for matching."""
    tokens, spans = [], []
    for m in re.finditer(r"[\w']+", text):
        tokens.append(m.group(0).lower())
        spans.append((m.start(), m.end()))
    return tokens, spans


def find_local_reconstruction_original_text(segment_text, raw_text, tokens, spans, cursor):
    """Same cursor+SequenceMatcher search as cross_model_agreement.py,
    but returns a slice of the ORIGINAL raw_text (via spans), not the
    lowercased tokens - preserves casing/punctuation for fair judging
    and confidence scoring. Returns (original_text_span, score, new_cursor)."""
    anchor_tokens = re.findall(r"[\w']+", segment_text.lower())
    if not anchor_tokens:
        return None, 0.0, cursor

    sent_len = len(anchor_tokens)
    min_len, max_len = max(1, sent_len - MAX_EXTRA), sent_len + MAX_EXTRA
    search_end = min(len(tokens), cursor + sent_len * 3 + MAX_EXTRA)

    best_start, best_end, best_score = None, None, 0.0
    for start in range(cursor, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > len(tokens):
                continue
            score = difflib.SequenceMatcher(None, anchor_tokens, tokens[start:end]).ratio()
            if score > best_score:
                best_score, best_start, best_end = score, start, end
        if best_start is not None and best_score >= 0.88 and start > best_start + 5:
            break

    if best_start is None:
        return None, 0.0, cursor

    char_start = spans[best_start][0]
    char_end = spans[best_end - 1][1]
    original_span_text = raw_text[char_start:char_end]
    return original_span_text, best_score, best_end


def judge_severity(client, reference_span, segment):
    prompt = JUDGE_PROMPT.format(reference_span=reference_span, segment=segment)
    try:
        r = client.chat(model=OLLAMA_MODELS[JUDGE_MODEL_KEY],
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0}, think=False)
        return parse_severity(r.message.content)
    except Exception as e:
        print(f"  ERROR (judge): {e}")
        return None


def get_confidence(client, full_transcript, segment):
    prompt = SINGLE_TRANSCRIPT_PROMPT_TEMPLATE.format(full_transcript=full_transcript, segment=segment)
    try:
        r = client.chat(model=OLLAMA_MODELS[JUDGE_MODEL_KEY],
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0, "num_predict": 20}, think=False)
        return parse_confidence(r.message.content)
    except Exception as e:
        print(f"  ERROR (confidence): {e}")
        return None


def run_experiment(baseline_model, client):
    print(f"\n{'='*70}\n  Fusion vs {baseline_model} - Shetland\n{'='*70}")

    segments_path = os.path.join(SEGMENTS_DIR, "sentence_segments_shetland_full.json")
    segments_data = json.load(open(segments_path))
    transcripts = segments_data.get("transcripts", [])

    baseline_texts = load_baseline_raw_text("shetland", baseline_model)

    fusion_rows, baseline_rows = [], []

    for t in transcripts:
        idx = t["dataset_index"]
        raw_baseline_text = baseline_texts.get(idx, "")
        if not raw_baseline_text:
            continue
        tokens, spans = tokenize_with_offsets(raw_baseline_text)
        cursor = 0

        full_fused_transcript = " ".join(s["segment"] for s in t.get("segments", []) if s.get("severity") is not None)

        for seg in t.get("segments", []):
            if seg.get("severity") is None:
                continue
            fused_segment = seg["segment"]
            reference_span = seg.get("corresponding_span")

            # FUSION condition: reuse existing severity, run blinded confidence
            fusion_conf = get_confidence(client, full_fused_transcript, fused_segment)
            fusion_rows.append({"dataset_index": idx, "severity": seg["severity"], "confidence": fusion_conf})

            # BASELINE condition: reconstruct, judge, run blinded confidence
            baseline_span, match_score, cursor = find_local_reconstruction_original_text(
                fused_segment, raw_baseline_text, tokens, spans, cursor)
            if baseline_span is None:
                continue
            baseline_severity = judge_severity(client, reference_span, baseline_span)
            baseline_conf = get_confidence(client, raw_baseline_text, baseline_span)
            baseline_rows.append({"dataset_index": idx, "severity": baseline_severity, "confidence": baseline_conf})

        print(f"  {idx}: done")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"fusion_vs_{baseline_model}_shetland.json")
    with open(output_path, "w") as f:
        json.dump({"baseline_model": baseline_model, "fusion_rows": fusion_rows,
                  "baseline_rows": baseline_rows}, f, indent=2, ensure_ascii=False)

    for label, rows in [("fusion", fusion_rows), (baseline_model, baseline_rows)]:
        valid = [r for r in rows if r["confidence"] is not None and r["severity"] is not None]
        if len(valid) < 3:
            print(f"  {label}: too few valid rows ({len(valid)}) for Spearman")
            continue
        confs = [r["confidence"] for r in valid]
        sevs = [r["severity"] for r in valid]
        rho, p = spearmanr(confs, sevs)
        print(f"  {label}: N={len(valid)}  Spearman={rho:.3f}  p={p:.4f}")

    print(f"\n  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["qwen", "whisperx"])
    args = parser.parse_args()

    client = Client(host=OLLAMA_HOST)
    run_experiment(args.baseline, client)


if __name__ == "__main__":
    main()
