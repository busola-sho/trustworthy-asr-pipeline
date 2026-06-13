"""
run_calibration_normalised.py

Reruns the MAR judge on the 100 calibration samples with normalised inputs
(lowercase, punctuation stripped) to test whether judge formatting sensitivity
explains the Parakeet/Qwen verdict inconsistencies found in qualitative analysis.

Adds new fields to each sample:
  qwen_verdict_p2_norm  — verdict on normalised input

Outputs a comparison table: original vs normalised verdicts.

Usage:
    python scripts/run_calibration_normalised.py
    python scripts/run_calibration_normalised.py --dry-run
"""

import json
import os
import re
import time
import argparse
from ollama import Client

INPUT_PATH  = "human_eval_samples.json"
OUTPUT_PATH = "human_eval_samples.json"
OLLAMA_HOST = "http://localhost:11434"
QWEN_MODEL  = "qwen2.5:7b"

MAR_PROMPT = """You are evaluating ASR transcripts in a high-stakes context where accuracy matters. Given a reference and hypothesis transcript of the same audio, determine if the hypothesis contains meaning-altering errors that would cause someone to misunderstand what was said.

Ignore: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"→"didn't", "oot"→"out"), filler words.

Flag as meaning-altering if:
- Factual content changes
- Negation is added or removed
- A name, place, or number is wrong
- A dialect word is misrecognised as a different real word (e.g. "bairn"→"barn")
- Content is hallucinated over inaudible segments

Examples:
Reference: she said she wisnae near the pub on saturday night
Hypothesis: she said she was near the pub on saturday night
Reasoning: Negation "wisnae" dropped, reversing the speaker's alibi.
Answer: true

Reference: he works the back shift at the factory on keppoch road
Hypothesis: he works the back shift at the factory on keppoch rd
Reasoning: "rd" is a standard abbreviation for road; same location, no factual content lost.
Answer: false

IMPORTANT: Reply with ONLY the single word true or false. No explanation, no reasoning, no other text."""

def normalise_for_judge(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', "'").replace('\u201d', "'")
    text = text.replace('\u2013', ' ').replace('\u2014', ' ')
    text = re.sub(r"[^\w\s']", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def parse_verdict(result: str):
    result = result.strip().lower()
    if result.startswith("true"):
        return True
    elif result.startswith("false"):
        return False
    elif "true" in result and "false" not in result:
        return True
    elif "false" in result and "true" not in result:
        return False
    return None

def run_judge(client, ref_norm, hyp_norm, sample_wer):
    if sample_wer == 0:
        return False
    try:
        response = client.chat(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": MAR_PROMPT},
                {"role": "user",   "content": f"Reference: {ref_norm}\nHypothesis: {hyp_norm}"},
            ],
            options={"temperature": 0},
        )
        return parse_verdict(response.message.content)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun",   action="store_true",
                        help="Clear existing normalised verdicts and rerun")
    args = parser.parse_args()

    with open(INPUT_PATH) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples")

    if args.dry_run:
        s = samples[0]
        ref_norm = normalise_for_judge(s["ref"])
        hyp_norm = normalise_for_judge(s["hyp"])
        print(f"\n[DRY RUN] Sample 0:")
        print(f"  REF (raw):  {s['ref'][:100]}")
        print(f"  REF (norm): {ref_norm[:100]}")
        print(f"  HYP (raw):  {s['hyp'][:100]}")
        print(f"  HYP (norm): {hyp_norm[:100]}")
        return

    if args.rerun:
        for s in samples:
            s["qwen_verdict_p2_norm"] = None
        print("Cleared existing normalised verdicts")

    client = Client(host=OLLAMA_HOST)
    try:
        available = [m.model for m in client.list().models]
        if not any(QWEN_MODEL in m for m in available):
            print(f"ERROR: {QWEN_MODEL} not pulled.")
            return
        print(f"Ollama connected. Using {QWEN_MODEL}\n")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    done = already = errors = 0

    for i, s in enumerate(samples):
        if s.get("qwen_verdict_p2_norm") is not None:
            already += 1
            continue

        ref_norm = normalise_for_judge(s["ref"])
        hyp_norm = normalise_for_judge(s["hyp"])

        verdict = run_judge(client, ref_norm, hyp_norm, s.get("sample_wer", 1.0))
        s["qwen_verdict_p2_norm"] = verdict

        if verdict is None:
            errors += 1
        done += 1

        with open(OUTPUT_PATH, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        time.sleep(0.05)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} done")

    print(f"\nFinished: done={done} already={already} errors={errors}")

    # ── Comparison table ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("COMPARISON: Original vs Normalised verdicts")
    print(f"{'='*65}")

    valid = [s for s in samples
             if s.get("human_verdict") is not None
             and s.get("qwen_verdict_p2") is not None
             and s.get("qwen_verdict_p2_norm") is not None]

    n = len(valid)
    n_pos = sum(1 for s in valid if s["human_verdict"])
    n_neg = sum(1 for s in valid if not s["human_verdict"])

    def stats(field):
        agree = sum(1 for s in valid if s[field] == s["human_verdict"])
        fn    = sum(1 for s in valid if s["human_verdict"] and not s[field])
        fp    = sum(1 for s in valid if not s["human_verdict"] and s[field])
        prec  = sum(1 for s in valid if s["human_verdict"] and s[field]) / max(sum(1 for s in valid if s[field]), 1)
        rec   = 1 - fn / n_pos if n_pos else 0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return agree/n, prec, rec, f1, fn/n_pos if n_pos else 0, fp/n_neg if n_neg else 0

    orig_stats = stats("qwen_verdict_p2")
    norm_stats = stats("qwen_verdict_p2_norm")

    labels = ["Agree", "Prec", "Recall", "F1", "FN rate", "FP rate"]
    print(f"{'Metric':<12} {'Original':>10} {'Normalised':>12} {'Change':>10}")
    print("-" * 48)
    for label, o, n_ in zip(labels, orig_stats, norm_stats):
        change = n_ - o
        arrow  = "▲" if change > 0 else "▼" if change < 0 else "—"
        print(f"  {label:<10} {o*100:>9.1f}% {n_*100:>11.1f}% {arrow}{abs(change)*100:>7.1f}pp")

    # how many verdicts changed?
    changed = sum(
        1 for s in valid
        if s["qwen_verdict_p2"] != s["qwen_verdict_p2_norm"]
    )
    print(f"\n  Verdicts that changed: {changed}/{n} ({changed/n*100:.1f}%)")

    # breakdown: who changed
    flipped_to_true  = sum(1 for s in valid
                           if not s["qwen_verdict_p2"] and s["qwen_verdict_p2_norm"])
    flipped_to_false = sum(1 for s in valid
                           if s["qwen_verdict_p2"] and not s["qwen_verdict_p2_norm"])
    print(f"  False→True (more flags):  {flipped_to_true}")
    print(f"  True→False (fewer flags): {flipped_to_false}")

    print(f"\n{'='*65}")
    print("SAMPLES WHERE VERDICT CHANGED")
    print(f"{'='*65}")

    changed_samples = [
        s for s in valid
        if s["qwen_verdict_p2"] != s["qwen_verdict_p2_norm"]
    ]

    for i, s in enumerate(changed_samples, 1):
        orig  = s["qwen_verdict_p2"]
        norm  = s["qwen_verdict_p2_norm"]
        human = s.get("human_verdict")
        direction = "False→True" if not orig and norm else "True→False"
        correct   = "norm correct" if norm == human else ("orig correct" if orig == human else "both wrong")
        print(f"\n── {i}. {direction} | human={human} | {correct}")
        print(f"   REF: {s['ref'][:150]}")
        print(f"   HYP: {s['hyp'][:150]}")

if __name__ == "__main__":
    main()