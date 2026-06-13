"""
run_all_judges.py

Runs all judges (GPT-4o, Selene, Qwen, Llama) with either prompt version.
Writes results to human_eval_samples.json.

Usage:
    python scripts/run_all_judges.py --prompt 2
    python scripts/run_all_judges.py --prompt 2 --judges gpt4o,selene
    python scripts/run_all_judges.py --prompt 1 --judges all
    python scripts/run_all_judges.py --dry-run --prompt 2
"""

import json
import os
import argparse
import time
from openai import OpenAI
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

DEFAULT_INPUT = "human_eval_samples.json"
OLLAMA_HOST   = "http://localhost:11434"
GPT4O_MODEL   = "gpt-4o"

OLLAMA_MODELS = {
    "selene": "atla/selene-mini",
    "qwen":   "qwen2.5:7b",
    "llama":  "llama3.1:8b",
}

ALL_JUDGES = ["gpt4o", "selene", "qwen", "llama"]

# ── Prompts ────────────────────────────────────────────────────────────────────

PROMPT_1 = """You are evaluating automatic speech recognition transcripts for a policing context.

You will be given a reference transcript and a hypothesis transcript of the same spoken audio.

Your task is to determine if the hypothesis contains any meaning-altering errors — that is, errors that would cause a police officer or legal professional to misunderstand what was said.

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

Only flag as meaning-altering if a factual error would mislead a police officer or legal professional.
Minor rewording, paraphrasing, or omission of filler words should not be flagged.

Reply with only: true or false"""

PROMPT_2 = """You are evaluating ASR transcripts for a policing context. Given a reference and hypothesis transcript of the same audio, determine if the hypothesis contains meaning-altering errors that would cause a police officer to misunderstand what was said.

Ignore: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"→"didn't", "oot"→"out"), filler words.

Flag as meaning-altering if:
- Factual content changes
- Negation is added or removed
- A name, place, or number is wrong
- A dialect word is misrecognised as a different real word (e.g. "bairn"→"barn")
- Content is hallucinated over inaudible segments

Examples:
Reference: She said she wisnae near the pub on Saturday night.
Hypothesis: She said she was near the pub on Saturday night.
Reasoning: Negation "wisnae" dropped, reversing the speaker's alibi.
Answer: true

Reference: He works the back shift at the factory on Keppoch Road.
Hypothesis: He works the back shift at the factory on Keppoch Rd.
Reasoning: "Rd" is a standard abbreviation for Road; same location, no factual content lost.
Answer: false

Reply with only: true or false"""

PROMPT_3 = """You are evaluating ASR transcripts for a policing context. Given a reference and hypothesis transcript of the same audio, work through the following checks in order. If any check is true, the hypothesis is meaning-altering. If all checks are false, it is not.

Ignore throughout: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"→"didn't", "oot"→"out"), filler words.

Check 1 — Negation: Is a negation added or removed in the hypothesis? (e.g. "I did" vs "I didn't", "wisnae"→"was")

Check 2 — Named entity: Is a name, place, or number transcribed incorrectly in a way that could not be resolved by a police officer? (e.g. a station name that does not exist, a person's name spelled differently)

Check 3 — Dialect misrecognition: Is a dialect or colloquial word misrecognised as a different real word that changes the meaning? (e.g. "weans"→"veins", "minted"→"mad")

Check 4 — Hallucination: Is content present in the hypothesis that does not exist in the reference and was not plausibly said? (e.g. fabricated speech over an inaudible segment)

Check 5 — Factual substitution: Is any other word substituted with a different word that changes who did what, or what happened?

If any check is true, reply: true
If all checks are false, reply: false

Reply with only: true or false"""


PROMPTS = {"1": PROMPT_1, "2": PROMPT_2, "3": PROMPT_3}

# ── Field naming ───────────────────────────────────────────────────────────────

def field_name(judge: str, prompt: str) -> str:
    """p1 keeps existing field names, p2/p3 add suffix."""
    base = f"{judge}_verdict"
    if prompt == "1":
        return base
    return f"{base}_p{prompt}"

# ── Runners ────────────────────────────────────────────────────────────────────

def run_gpt4o(openai_client, system_prompt: str, ref: str, hyp: str, sample_wer: float) -> bool | None:
    if sample_wer == 0:
        return False
    try:
        completion = openai_client.chat.completions.create(
            model=GPT4O_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Reference: {ref}\nHypothesis: {hyp}"},
            ],
        )
        result = completion.choices[0].message.content.strip().lower()
        return parse_verdict(result)
    except Exception as e:
        print(f"  ERROR (gpt4o): {e}")
        return None

def run_ollama(ollama_client, model: str, system_prompt: str, ref: str, hyp: str, sample_wer: float) -> bool | None:
    if sample_wer == 0:
        return False
    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Reference: {ref}\nHypothesis: {hyp}"},
            ],
            options={"temperature": 0},
        )
        result = response.message.content.strip().lower()
        return parse_verdict(result)
    except Exception as e:
        print(f"  ERROR ({model}): {e}")
        return None

def parse_verdict(result: str) -> bool | None:
    if result.startswith("true"):
        return True
    elif result.startswith("false"):
        return False
    elif "true" in result and "false" not in result:
        return True
    elif "false" in result and "true" not in result:
        return False
    else:
        print(f"  WARNING: could not parse: '{result[:80]}'")
        return None

# ── Agreement stats ────────────────────────────────────────────────────────────

def print_agreement(rows: list, field: str, label: str):
    compared = [
        r for r in rows
        if r.get("human_verdict") is not None
        and r.get(field) is not None
    ]
    if not compared:
        print(f"  {label}: no data")
        return
    n     = len(compared)
    agree = sum(1 for r in compared if bool(r["human_verdict"]) == bool(r[field]))
    fn    = sum(1 for r in compared if r["human_verdict"] and not r[field])
    fp    = sum(1 for r in compared if not r["human_verdict"] and r[field])
    print(f"  {label} ({n} rows):")
    print(f"    Agreement:                     {agree/n*100:.1f}%")
    print(f"    False negatives (missed MA):   {fn/n*100:.1f}%")
    print(f"    False positives (false alarm): {fp/n*100:.1f}%")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=DEFAULT_INPUT)
    parser.add_argument("--prompt",  default="1", choices=["1", "2", "3"])
    parser.add_argument("--judges",  default="all", help="Comma-separated: gpt4o,selene,qwen,llama or all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun",   action="store_true", help="Clear existing verdicts and rerun from scratch")
    args = parser.parse_args()

    judges = ALL_JUDGES if args.judges == "all" else [j.strip() for j in args.judges.split(",")]
    system_prompt = PROMPTS[args.prompt]

    print(f"Prompt:  {args.prompt}")
    print(f"Judges:  {judges}")
    print(f"Input:   {args.input}\n")

    with open(args.input) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} rows\n")

    if args.rerun:
        for row in rows:
            for judge in judges:
                fname = field_name(judge, args.prompt)
                if fname in row:
                    row[fname] = None
        print(f"Cleared existing verdicts for: {judges} (Prompt {args.prompt})\n")

    if args.dry_run:
        for r in rows[:2]:
            print(f"[DRY RUN] clip={r['clip_id']} model={r['model']} wer={r['sample_wer']}")
            for j in judges:
                print(f"  would write to: {field_name(j, args.prompt)}")
            print(f"  ref: {r['ref'][:80]}")
            print(f"  hyp: {r['hyp'][:80]}\n")
        return

    # initialise clients
    openai_client = None
    ollama_client = None

    if "gpt4o" in judges:
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    ollama_judges = [j for j in judges if j in OLLAMA_MODELS]
    if ollama_judges:
        ollama_client = Client(host=OLLAMA_HOST)
        try:
            available = [m.model for m in ollama_client.list().models]
            print(f"Ollama models available: {available}\n")
        except Exception as e:
            print(f"ERROR: could not connect to Ollama — run: ollama serve\n{e}")
            return

    # ensure all fields exist
    for row in rows:
        for judge in judges:
            f = field_name(judge, args.prompt)
            if f not in row:
                row[f] = None

    # run each judge
    for judge in judges:
        f = field_name(judge, args.prompt)

        if judge in OLLAMA_MODELS:
            model_name = OLLAMA_MODELS[judge]
            if not any(model_name in m for m in available):
                print(f"WARNING: {model_name} not pulled. Run: ollama pull {model_name}")
                print(f"Skipping {judge}\n")
                continue

        print(f"Running {judge} (Prompt {args.prompt})...")
        skipped = errors = 0

        for i, row in enumerate(rows):
            if row.get(f) is not None:
                skipped += 1
                continue

            if judge == "gpt4o":
                verdict = run_gpt4o(openai_client, system_prompt, row["ref"], row["hyp"], row["sample_wer"])
                time.sleep(0.2)
            else:
                verdict = run_ollama(ollama_client, OLLAMA_MODELS[judge], system_prompt, row["ref"], row["hyp"], row["sample_wer"])
                time.sleep(0.1)

            row[f] = verdict
            if verdict is None:
                errors += 1

            with open(args.input, "w") as out:
                json.dump(rows, out, indent=2, ensure_ascii=False)

            if (i + 1) % 10 == 0:
                done = sum(1 for r in rows if r.get(f) is not None)
                print(f"  {judge}: {done}/{len(rows)} done")

        done = sum(1 for r in rows if r.get(f) is not None)
        print(f"  {judge} complete: {done}/{len(rows)} judged, {skipped} skipped, {errors} errors\n")

    with open(args.input, "w") as out:
        json.dump(rows, out, indent=2, ensure_ascii=False)
    print(f"Saved to {args.input}")

    # agreement summary
    print(f"\n── Agreement with human verdicts (Prompt {args.prompt}) ──────────────────")
    for judge in judges:
        f = field_name(judge, args.prompt)
        print_agreement(rows, f, f"{judge.upper()} Prompt {args.prompt}")

if __name__ == "__main__":
    main()