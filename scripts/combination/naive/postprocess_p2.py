"""
postprocess_p2.py

Cleans up P2 selector outputs where models returned labels/reasoning
instead of pure transcript text. Strips preamble and reasoning,
recomputes sample_WER and reruns qwen_verdict_p2 on cleaned hyp.
Saves as new _pp files.

Usage:
    python scripts/postprocess_p2.py
    python scripts/postprocess_p2.py --dry-run
"""

import json
import os
import re
import argparse
import time
from jiwer import wer
from ollama import Client

COMBINATION_DIR = "results/combinations"
OLLAMA_HOST     = "http://localhost:11434"
QWEN_MODEL      = "qwen2.5:7b"

MAR_PROMPT = """You are evaluating ASR transcripts in a high-stakes context where accuracy matters. Given a reference and hypothesis transcript of the same audio, determine if the hypothesis contains meaning-altering errors that would cause someone to misunderstand what was said.

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

IMPORTANT: Reply with ONLY the single word true or false. No explanation, no reasoning, no other text."""

# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_hyp(text: str) -> str:
    if not text:
        return text

    # Step 1: remove reasoning at the end
    text = re.sub(r'\n\nThis transcript.*$', '', text, flags=re.DOTALL).strip()

    # Step 2: remove Type 1 preamble ("The most accurate transcript is Transcript N:\n\n")
    text = re.sub(r'^.*?Transcript\s+\d+[:\s]*\n+', '', text, flags=re.DOTALL).strip()

    # Step 3: remove Type 2 label ("Transcript N: ")
    text = re.sub(r'^Transcript\s+\d+[:\s]+', '', text).strip()

    # Step 4: remove surrounding quotes if present
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    return text

def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
    return text.strip()

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

def run_mar_judge(client: Client, ref: str, hyp: str, sample_wer_val: float):
    if sample_wer_val == 0:
        return False
    try:
        response = client.chat(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": MAR_PROMPT},
                {"role": "user",   "content": f"Reference: {ref}\nHypothesis: {hyp}"},
            ],
            options={"temperature": 0},
        )
        return parse_verdict(response.message.content)
    except Exception as e:
        print(f"  ERROR (MAR judge): {e}")
        return None

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # find all p2 files that haven't been post-processed yet
    p2_files = sorted([
        f for f in os.listdir(COMBINATION_DIR)
        if "_p2_" in f and f.endswith(".json") and "_pp" not in f
    ])

    if not p2_files:
        print("No P2 files found.")
        return

    print(f"Found {len(p2_files)} P2 files:\n")
    for f in p2_files:
        print(f"  {f}")

    if args.dry_run:
        print("\n[DRY RUN] Sample cleaning preview:")
        for fname in p2_files:
            path = os.path.join(COMBINATION_DIR, fname)
            with open(path) as f:
                data = json.load(f)
            for s in data.get("samples", []):
                hyp = s.get("hyp", "")
                if hyp and ("Transcript" in hyp or "This transcript" in hyp):
                    cleaned = clean_hyp(hyp)
                    print(f"\n  {fname}:")
                    print(f"  BEFORE: {hyp[:150]}")
                    print(f"  AFTER:  {cleaned[:150]}")
                    break
        return

    # connect to Ollama
    client = Client(host=OLLAMA_HOST)
    try:
        available = [m.model for m in client.list().models]
        if not any(QWEN_MODEL in m for m in available):
            print(f"ERROR: {QWEN_MODEL} not pulled.")
            return
        print(f"Ollama connected. Using {QWEN_MODEL}\n")
    except Exception as e:
        print(f"ERROR: could not connect to Ollama\n{e}")
        return

    for fname in p2_files:
        path     = os.path.join(COMBINATION_DIR, fname)
        out_name = fname.replace(".json", "_pp.json")
        out_path = os.path.join(COMBINATION_DIR, out_name)

        print(f"\nProcessing {fname}...")

        with open(path) as f:
            data = json.load(f)

        samples  = data.get("samples", [])
        cleaned  = 0
        rejudged = 0

        for i, s in enumerate(samples):
            if s.get("skipped") or s.get("error"):
                continue

            hyp = s.get("hyp", "")
            if not hyp:
                continue

            needs_clean = "Transcript" in hyp or "This transcript" in hyp

            if needs_clean:
                new_hyp = clean_hyp(hyp)
                s["hyp"] = new_hyp
                cleaned += 1

                # recompute WER
                ref = s.get("ref", "")
                new_wer = wer(normalise(ref), normalise(new_hyp)) if ref and new_hyp else s.get("sample_WER")
                s["sample_WER"] = new_wer

                # rejudge MAR on cleaned hyp
                verdict = run_mar_judge(client, ref, new_hyp, new_wer)
                s["qwen_verdict_p2"] = verdict
                rejudged += 1
                time.sleep(0.05)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(samples)} done ({cleaned} cleaned, {rejudged} rejudged)")

        # recompute corpus metrics
        valid = [
            s for s in samples
            if not s.get("skipped") and not s.get("error")
            and s.get("sample_WER") is not None
            and s.get("hyp")
        ]

        corpus_wer_val = wer(
            [normalise(s["ref"]) for s in valid],
            [normalise(s["hyp"]) for s in valid]
        ) if valid else None

        mar = (
            sum(1 for s in valid if s.get("qwen_verdict_p2")) / len(valid)
            if valid else None
        )

        out_data = dict(data)
        out_data["samples"]                 = samples
        out_data["corpus_wer"]              = corpus_wer_val
        out_data["meaning_alteration_rate"] = mar
        out_data["num_samples"]             = len(valid)
        out_data["post_processed"]          = True
        out_data["cleaned_count"]           = cleaned
        out_data["rejudged_count"]          = rejudged

        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

        wer_str = f"{corpus_wer_val*100:.2f}%" if corpus_wer_val is not None else "—"
        mar_str = f"{mar*100:.2f}%"            if mar            is not None else "—"

        print(f"  Cleaned: {cleaned}/{len(samples)} | Rejudged: {rejudged}")
        print(f"  WER: {wer_str}  MAR: {mar_str}  (N={len(valid)})")
        print(f"  Saved: {out_name}")

    print("\nDone.")

if __name__ == "__main__":
    main()