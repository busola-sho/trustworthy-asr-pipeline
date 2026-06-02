"""
run_reeval_mar.py

Re-evaluates MAR on all existing benchmark JSONs using Qwen 2.5 7B Prompt 2.
Adds a `qwen_verdict` field to each sample without touching the existing
`meaning_altering` field (which holds GPT-4o verdicts).

Also computes and prints corpus-level MAR under both judges for comparison.

Usage:
    python scripts/run_reeval_mar.py
    python scripts/run_reeval_mar.py --model whisper --dataset commonvoice
    python scripts/run_reeval_mar.py --dry-run
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

MODELS   = ["qwen", "whisper", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc"]

# Prompt 2 — best performing for Qwen
MAR_PROMPT = """You are evaluating ASR transcripts for a policing context. Given a reference and hypothesis transcript of the same audio, determine if the hypothesis contains meaning-altering errors that would cause a police officer to misunderstand what was said.

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

# ── Judge function ─────────────────────────────────────────────────────────────

def run_qwen_judge(client: Client, ref: str, hyp: str, sample_wer: float):
    if sample_wer == 0:
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
        result = response.message.content.strip().lower()
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
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

# ── Per-file runner ────────────────────────────────────────────────────────────

def reeval_file(model_key, dataset, client, dry_run=False):
    path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model_key, dataset)])
    print(f"\n── {model_key} / {dataset} ──────────────────────────────────────")

    with open(path) as f:
        data = json.load(f)

    samples = data["samples"]
    skipped = errors = already_done = 0

    for i, s in enumerate(samples):
        # skip IGNORE segments
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in s.get("ref", ""):
            skipped += 1
            continue

        # skip if already judged
        if s.get("qwen_verdict") is not None:
            already_done += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] sample {i}: ref={s['ref'][:60]}")
            s["qwen_verdict"] = None
            continue

        verdict = run_qwen_judge(
            client,
            s["ref"],
            s["hyp"],
            s.get("sample_WER", 1.0),
        )

        s["qwen_verdict"] = verdict
        if verdict is None:
            errors += 1

        # save after every sample
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if (i + 1) % 50 == 0:
            done = sum(1 for x in samples if x.get("qwen_verdict") is not None)
            print(f"  {done}/{len(samples)} done")

        time.sleep(0.1)

    # compute MAR under both judges
    valid = [
        s for s in samples
        if "IGNORE_TIME_SEGMENT_IN_SCORING" not in s.get("ref", "")
        and s.get("sample_WER") is not None
    ]

    gpt4o_mar = sum(1 for s in valid if s.get("meaning_altering")) / len(valid) if valid else None
    qwen_mar  = sum(1 for s in valid if s.get("qwen_verdict")) / len(valid) if valid else None

    print(f"  Samples: {len(valid)} | Skipped: {skipped} | Errors: {errors} | Already done: {already_done}")
    print(f"  GPT-4o MAR: {gpt4o_mar*100:.2f}%" if gpt4o_mar is not None else "  GPT-4o MAR: N/A")
    print(f"  Qwen P2 MAR: {qwen_mar*100:.2f}%" if qwen_mar is not None else "  Qwen P2 MAR: N/A")

    return {
        "model":      model_key,
        "dataset":    dataset,
        "gpt4o_mar":  gpt4o_mar,
        "qwen_mar":   qwen_mar,
        "n":          len(valid),
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="all", help="Model key or all")
    parser.add_argument("--dataset", default="all", help="Dataset key or all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models   = MODELS   if args.model   == "all" else [args.model]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    client = Client(host=OLLAMA_HOST)

    if not args.dry_run:
        try:
            available = [m.model for m in client.list().models]
            if not any(QWEN_MODEL in m for m in available):
                print(f"ERROR: {QWEN_MODEL} not pulled. Run: ollama pull {QWEN_MODEL}")
                return
            print(f"Ollama connected. Using {QWEN_MODEL}\n")
        except Exception as e:
            print(f"ERROR: could not connect to Ollama — run: ollama serve\n{e}")
            return

    summary = []
    for model_key in models:
        for dataset in datasets:
            result = reeval_file(model_key, dataset, client, dry_run=args.dry_run)
            summary.append(result)

    print(f"\n{'='*70}")
    print("SUMMARY — MAR comparison: GPT-4o vs Qwen P2")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Dataset':<20} {'GPT-4o MAR':>12} {'Qwen P2 MAR':>12} {'N':>6}")
    for r in summary:
        g = f"{r['gpt4o_mar']*100:.2f}%" if r['gpt4o_mar'] is not None else "N/A"
        q = f"{r['qwen_mar']*100:.2f}%"  if r['qwen_mar']  is not None else "pending"
        print(f"{r['model']:<12} {r['dataset']:<20} {g:>12} {q:>12} {r['n']:>6}")

if __name__ == "__main__":
    main()