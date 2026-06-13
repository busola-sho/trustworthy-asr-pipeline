"""
run_llama4_judge.py

Runs Llama 4 Scout via Groq API as MAR judge on the 100-sample calibration set.
Adds llama4_verdict_p1, llama4_verdict_p2, llama4_verdict_p3 fields.
Resume-friendly.

Usage:
    python scripts/run_llama4_judge.py
    python scripts/run_llama4_judge.py --prompt p2
    python scripts/run_llama4_judge.py --prompt p2 --rerun
    python scripts/run_llama4_judge.py --dry-run
"""

import json
import os
import time
import argparse
from groq import Groq

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH  = "human_eval_samples.json"
OUTPUT_PATH = "human_eval_samples.json"
GROQ_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Prompts ────────────────────────────────────────────────────────────────────

PROMPT_1 = """You are evaluating automatic speech recognition transcripts for a policing context.

You will be given a Reference transcript and a Candidate transcript of the same spoken audio.

Your task is to determine if the Candidate contains any meaning-altering errors — that is, errors that would cause a police officer or legal professional to misunderstand what was said.

Ignore differences in:
- Capitalisation
- Punctuation
- Contractions (e.g. "I've" vs "I have")
- Dialect variations (e.g. "aboot" vs "about", "didnae" vs "didn't")
- Filler words

Flag as meaning-altering only if:
- A word is substituted with a different word that changes the factual content
- A word is missing or added that changes who did what
- A name, place, or number is transcribed incorrectly

IMPORTANT: Reply with ONLY the single word true or false. No explanation, no reasoning, no other text."""

PROMPT_2 = """You are evaluating ASR transcripts for a policing context. Given a Reference and Candidate transcript of the same audio, determine if the Candidate contains meaning-altering errors that would cause a police officer to misunderstand what was said.

Ignore: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"→"didn't", "oot"→"out"), filler words.

Flag as meaning-altering if:
- Factual content changes
- Negation is added or removed
- A name, place, or number is wrong
- A dialect word is misrecognised as a different real word (e.g. "bairn"→"barn")
- Content is hallucinated over inaudible segments

Examples:
Reference: She said she wisnae near the pub on Saturday night.
Candidate: She said she was near the pub on Saturday night.
Reasoning: Negation "wisnae" dropped, reversing the speaker's alibi.
Answer: true

Reference: He works the back shift at the factory on Keppoch Road.
Candidate: He works the back shift at the factory on Keppoch Rd.
Reasoning: "Rd" is a standard abbreviation for Road; same location, no factual content lost.
Answer: false

IMPORTANT: Reply with ONLY the single word true or false. No explanation, no reasoning, no other text."""

PROMPT_3 = """You are evaluating ASR transcripts for a policing context. Given a Reference and Candidate transcript of the same audio, work through the following checks in order. If any check is true, the Candidate is meaning-altering. If all checks are false, it is not.

Ignore throughout: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"→"didn't", "oot"→"out"), filler words.

Check 1 — Negation: Is a negation added or removed in the Candidate? (e.g. "I did" vs "I didn't", "wisnae"→"was")

Check 2 — Named entity: Is a name, place, or number transcribed incorrectly in a way that could not be resolved by a police officer? (e.g. a station name that does not exist, a person's name spelled differently)

Check 3 — Dialect misrecognition: Is a dialect or colloquial word misrecognised as a different real word that changes the meaning? (e.g. "weans"→"veins", "minted"→"mad")

Check 4 — Hallucination: Is content present in the Candidate that does not exist in the Reference and was not plausibly said? (e.g. fabricated speech over an inaudible segment)

Check 5 — Factual substitution: Is any other word substituted with a different word that changes who did what, or what happened?

If any check is true, reply: true
If all checks are false, reply: false

IMPORTANT: Reply with ONLY the single word true or false. No explanation, no reasoning, no other text."""

PROMPTS = {"p1": PROMPT_1, "p2": PROMPT_2, "p3": PROMPT_3}
FIELD_NAMES = {"p1": "llama4_verdict_p1", "p2": "llama4_verdict_p2", "p3": "llama4_verdict_p3"}

# ── Judge ──────────────────────────────────────────────────────────────────────

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
    print(f"  WARNING: could not parse: '{result[:80]}'")
    return None

def run_judge(client: Groq, prompt: str, ref: str, hyp: str, sample_wer: float):
    if sample_wer == 0:
        return False
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": f"Reference: {ref}\nCandidate: {hyp}"},
            ],
            temperature=0,
            max_tokens=10,
        )
        return parse_verdict(response.choices[0].message.content)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",  default="all", choices=["p1", "p2", "p3", "all"])
    parser.add_argument("--rerun",   action="store_true", help="Clear existing verdicts and rerun")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY"))
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        print("ERROR: set GROQ_API_KEY env var or pass --api-key")
        return

    prompts_to_run = list(PROMPTS.keys()) if args.prompt == "all" else [args.prompt]

    with open(INPUT_PATH) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from {INPUT_PATH}")
    print(f"Running prompts: {prompts_to_run}")
    print(f"Model: {GROQ_MODEL}")
    if args.rerun:
        print("Mode: RERUN (clearing existing verdicts)")

    if args.dry_run:
        print(f"\n[DRY RUN] Sample 0:")
        print(f"  ref: {samples[0]['ref'][:80]}")
        print(f"  hyp: {samples[0]['hyp'][:80]}")
        print(f"  Fields that would be added: {[FIELD_NAMES[p] for p in prompts_to_run]}")
        return

    client = Groq(api_key=args.api_key)

    for prompt_key in prompts_to_run:
        field_name = FIELD_NAMES[prompt_key]
        prompt     = PROMPTS[prompt_key]

        print(f"\n── Running {prompt_key} → {field_name} ──────────────────────")

        # clear existing verdicts if rerunning
        if args.rerun:
            for s in samples:
                s[field_name] = None

        done = already = errors = 0

        for i, s in enumerate(samples):
            # skip if already judged (unless rerun cleared them)
            if s.get(field_name) is not None:
                already += 1
                continue

            verdict = run_judge(
                client,
                prompt,
                s["ref"],
                s["hyp"],
                s.get("sample_wer", 1.0),
            )

            s[field_name] = verdict
            if verdict is None:
                errors += 1
            done += 1

            # save after every sample
            with open(OUTPUT_PATH, "w") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)

            time.sleep(0.3)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(samples)} done")

        print(f"  Finished: done={done} already={already} errors={errors}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — Llama 4 Scout vs human verdicts")
    print(f"{'='*60}")

    valid = [s for s in samples if s.get("human_verdict") is not None]
    print(f"{'Field':<25} {'N':>6} {'Agreement':>10} {'FN rate':>9} {'FP rate':>9}")
    print("-" * 62)

    for prompt_key in prompts_to_run:
        field = FIELD_NAMES[prompt_key]
        judged = [s for s in valid if s.get(field) is not None]
        if not judged:
            continue

        n     = len(judged)
        agree = sum(1 for s in judged if s[field] == s["human_verdict"])
        fn    = sum(1 for s in judged if s["human_verdict"] is True  and s[field] is False)
        fp    = sum(1 for s in judged if s["human_verdict"] is False and s[field] is True)
        n_pos = sum(1 for s in judged if s["human_verdict"] is True)
        n_neg = sum(1 for s in judged if s["human_verdict"] is False)

        print(f"{field:<25} {n:>6} {agree/n*100:>9.1f}% "
              f"{fn/n_pos*100 if n_pos else 0:>8.1f}% "
              f"{fp/n_neg*100 if n_neg else 0:>8.1f}%")

    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()