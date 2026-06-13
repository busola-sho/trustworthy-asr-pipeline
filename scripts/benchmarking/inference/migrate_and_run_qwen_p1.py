"""
migrate_and_run_qwen_p1.py

Does two things:
1. Migrates all 12 benchmark JSONs:
   - meaning_altering  → gpt4o_verdict_p1
   - qwen_verdict      → qwen_verdict_p2
   - adds qwen_verdict_p1: null (to be filled)

2. Runs Qwen 2.5 7B with Prompt 1 on all 12 files,
   writing results to qwen_verdict_p1.

Usage:
    python scripts/migrate_and_run_qwen_p1.py --dry-run   # preview only
    python scripts/migrate_and_run_qwen_p1.py --migrate-only  # step 1 only
    python scripts/migrate_and_run_qwen_p1.py              # full run

REMINDER: After this script, manually update:
    - src/benchmark.py     → change 'meaning_altering' to 'gpt4o_verdict_p1'
    - scripts/run_reeval_mar.py → change 'qwen_verdict' to 'qwen_verdict_p2'
"""

import json
import os
import argparse
import time
from ollama import Client

# ── Configuration ──────────────────────────────────────────────────────────────

BENCHMARKS_DIR = "benchmarks"
OLLAMA_HOST    = "http://localhost:11434"
QWEN_MODEL     = "qwen2.5:7b"

CANONICAL_FILES = {
    ("qwen",     "commonvoice"):      "qwen_commonvoice_20260524_153426.json",
    ("qwen",     "edacc"):            "qwen_edacc_20260525_204314.json",
    ("qwen",     "english_dialects"): "qwen_english_dialects_20260525_000627.json",
    ("whisper",  "commonvoice"):      "whisper_commonvoice_20260524_083930.json",
    ("whisper",  "edacc"):            "whisper_edacc_20260525_184504.json",
    ("whisper",  "english_dialects"): "whisper_english_dialects_20260525_110315.json",
    ("parakeet", "commonvoice"):      "parakeet_commonvoice_20260524_150129.json",
    ("parakeet", "edacc"):            "parakeet_edacc_20260525_184006.json",
    ("parakeet", "english_dialects"): "parakeet_english_dialects_20260524_234807.json",
    ("wav2vec2", "commonvoice"):      "wav2vec2_commonvoice_20260526_053757.json",
    ("wav2vec2", "edacc"):            "wav2vec2_edacc_20260525_213534.json",
    ("wav2vec2", "english_dialects"): "wav2vec2_english_dialects_20260526_073439.json",
}

# Prompt 1 — zero-shot rubric (original)
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

# ── Step 1: Migration ──────────────────────────────────────────────────────────

def migrate_file(path: str, dry_run: bool = False) -> dict:
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    migrated = renamed_ma = renamed_qv = already_done = 0

    for s in samples:
        changed = False

        # rename meaning_altering → gpt4o_verdict_p1
        if "meaning_altering" in s and "gpt4o_verdict_p1" not in s:
            s["gpt4o_verdict_p1"] = s.pop("meaning_altering")
            renamed_ma += 1
            changed = True
        elif "gpt4o_verdict_p1" in s:
            already_done += 1

        # rename qwen_verdict → qwen_verdict_p2
        if "qwen_verdict" in s and "qwen_verdict_p2" not in s:
            s["qwen_verdict_p2"] = s.pop("qwen_verdict")
            renamed_qv += 1
            changed = True

        # add qwen_verdict_p1 if missing
        if "qwen_verdict_p1" not in s:
            s["qwen_verdict_p1"] = None
            changed = True

        if changed:
            migrated += 1

    if not dry_run and migrated > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return {
        "migrated":    migrated,
        "renamed_ma":  renamed_ma,
        "renamed_qv":  renamed_qv,
        "already_done": already_done,
    }

# ── Step 2: Qwen P1 judge ─────────────────────────────────────────────────────

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

def run_qwen_p1_file(path: str, client: Client) -> dict:
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    skipped = errors = already_done = 0

    for i, s in enumerate(samples):
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in s.get("ref", ""):
            skipped += 1
            continue

        if s.get("qwen_verdict_p1") is not None:
            already_done += 1
            continue

        if s.get("sample_WER", 1.0) == 0:
            s["qwen_verdict_p1"] = False
            continue

        try:
            response = client.chat(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": PROMPT_1},
                    {"role": "user",   "content": f"Reference: {s['ref']}\nHypothesis: {s['hyp']}"},
                ],
                options={"temperature": 0},
            )
            verdict = parse_verdict(response.message.content)
        except Exception as e:
            print(f"  ERROR: {e}")
            verdict = None
            errors += 1

        s["qwen_verdict_p1"] = verdict

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if (i + 1) % 50 == 0:
            done = sum(1 for s in samples if s.get("qwen_verdict_p1") is not None)
            print(f"    {done}/{len(samples)} done")

        time.sleep(0.1)

    done = sum(1 for s in samples if s.get("qwen_verdict_p1") is not None)
    return {"done": done, "skipped": skipped, "errors": errors, "already_done": already_done}

# ── Summary stats ──────────────────────────────────────────────────────────────

def print_mar_comparison(path: str):
    with open(path) as f:
        data = json.load(f)

    valid = [
        s for s in data.get("samples", [])
        if "IGNORE_TIME_SEGMENT_IN_SCORING" not in s.get("ref", "")
        and s.get("sample_WER") is not None
    ]
    if not valid:
        return

    n = len(valid)
    gpt4o_p1 = sum(1 for s in valid if s.get("gpt4o_verdict_p1")) / n
    qwen_p1  = sum(1 for s in valid if s.get("qwen_verdict_p1"))  / n if any(s.get("qwen_verdict_p1") is not None for s in valid) else None
    qwen_p2  = sum(1 for s in valid if s.get("qwen_verdict_p2"))  / n if any(s.get("qwen_verdict_p2") is not None for s in valid) else None

    print(f"    GPT-4o P1: {gpt4o_p1*100:.1f}%  |  Qwen P1: {qwen_p1*100:.1f}% " if qwen_p1 else f"    GPT-4o P1: {gpt4o_p1*100:.1f}%  |  Qwen P1: pending", end="")
    print(f"  |  Qwen P2: {qwen_p2*100:.1f}%" if qwen_p2 else "  |  Qwen P2: pending")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",       action="store_true", help="Preview migration without writing")
    parser.add_argument("--migrate-only",  action="store_true", help="Only run migration, skip Qwen P1")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Migrating field names in all 12 benchmark JSONs")
    print("=" * 60)

    for (model, dataset), filename in CANONICAL_FILES.items():
        path = os.path.join(BENCHMARKS_DIR, filename)
        if not os.path.exists(path):
            print(f"  MISSING: {filename}")
            continue

        result = migrate_file(path, dry_run=args.dry_run)
        status = "[DRY RUN]" if args.dry_run else "✓"
        print(f"  {status} {model}/{dataset}: "
              f"renamed {result['renamed_ma']} meaning_altering, "
              f"{result['renamed_qv']} qwen_verdict, "
              f"already done: {result['already_done']}")

    if args.dry_run:
        print("\nDry run complete. Run without --dry-run to apply changes.")
        return

    if args.migrate_only:
        print("\nMigration complete. Skipping Qwen P1 run.")
        print("\n⚠️  REMINDER: Update these files manually:")
        print("   - src/benchmark.py       → 'meaning_altering' → 'gpt4o_verdict_p1'")
        print("   - scripts/run_reeval_mar.py → 'qwen_verdict' → 'qwen_verdict_p2'")
        return

    print("\n" + "=" * 60)
    print("Step 2: Running Qwen P1 on all 12 benchmark files")
    print("=" * 60)

    client = Client(host=OLLAMA_HOST)
    try:
        available = [m.model for m in client.list().models]
        if not any(QWEN_MODEL in m for m in available):
            print(f"ERROR: {QWEN_MODEL} not pulled. Run: ollama pull {QWEN_MODEL}")
            return
        print(f"Ollama connected. Using {QWEN_MODEL}\n")
    except Exception as e:
        print(f"ERROR: could not connect to Ollama — run: ollama serve\n{e}")
        return

    for (model, dataset), filename in CANONICAL_FILES.items():
        path = os.path.join(BENCHMARKS_DIR, filename)
        if not os.path.exists(path):
            continue

        print(f"\n  {model}/{dataset}...")
        result = run_qwen_p1_file(path, client)
        print(f"    done={result['done']} skipped={result['skipped']} errors={result['errors']} already_done={result['already_done']}")
        print_mar_comparison(path)

    print("\n" + "=" * 60)
    print("All done.")
    print("\n⚠️  REMINDER: Update these files manually:")
    print("   - src/benchmark.py       → 'meaning_altering' → 'gpt4o_verdict_p1'")
    print("   - scripts/run_reeval_mar.py → 'qwen_verdict' → 'qwen_verdict_p2'")
    print("=" * 60)

if __name__ == "__main__":
    main()