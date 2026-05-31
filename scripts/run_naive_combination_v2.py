"""
run_naive_combination_v2.py

Naive combination baseline with separate selector and judge models.
Supports any combination of gpt4o, qwen, llama for each role.

Usage:
    python scripts/run_naive_combination_v2.py --dataset commonvoice --selector llama --judge qwen
    python scripts/run_naive_combination_v2.py --dataset edacc --selector qwen --judge llama
    python scripts/run_naive_combination_v2.py --dataset all --selector llama --judge qwen
    python scripts/run_naive_combination_v2.py --dry-run --selector llama --judge qwen
"""

import json
import os
import re
import argparse
import time
from jiwer import wer
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

BENCHMARKS_DIR = "benchmarks"
OUTPUT_DIR     = "benchmarks"
OLLAMA_HOST    = "http://localhost:11434"

OLLAMA_MODELS = {
    "qwen":  "qwen2.5:7b",
    "llama": "llama3.1:8b",
}

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

ASR_MODELS = ["qwen", "whisper", "parakeet", "wav2vec2"]
DATASETS   = ["commonvoice", "english_dialects", "edacc"]

# ── Prompts ────────────────────────────────────────────────────────────────────

SELECTION_PROMPT = """You are given four ASR hypotheses of the same spoken audio.

Your task is to select or construct the single most accurate transcript — the one that best preserves the meaning of what was said, with particular care for:
- Named entities (names, places, numbers)
- Negations
- Dialect words that carry meaning

You may select one hypothesis verbatim or make minimal edits to combine the best parts.

Return only the final transcript, nothing else."""

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

# ── Normalisation ──────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
    return text

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
    print(f"  WARNING: could not parse verdict: '{result[:80]}'")
    return None

# ── Ollama calls ───────────────────────────────────────────────────────────────

def ollama_select(client, model_name, hyps):
    hyp_block = "\n".join([f"Hypothesis {i+1}: {h}" for i, h in enumerate(hyps)])
    try:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SELECTION_PROMPT},
                {"role": "user",   "content": hyp_block},
            ],
            options={"temperature": 0},
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"  ERROR (select/{model_name}): {e}")
        return None

def ollama_mar(client, model_name, ref, hyp, sample_wer):
    if sample_wer == 0:
        return False
    try:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": MAR_PROMPT},
                {"role": "user",   "content": f"Reference: {ref}\nHypothesis: {hyp}"},
            ],
            options={"temperature": 0},
        )
        return parse_verdict(response.message.content)
    except Exception as e:
        print(f"  ERROR (mar/{model_name}): {e}")
        return None

# ── Main runner ────────────────────────────────────────────────────────────────

def run_dataset(dataset, selector_key, judge_key, client, max_samples=None):
    selector_model = OLLAMA_MODELS[selector_key]
    judge_model    = OLLAMA_MODELS[judge_key]

    print(f"\n── {dataset} | selector={selector_key} judge={judge_key} ──────────")

    # load all 4 ASR model files
    model_samples = {}
    for m in ASR_MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(m, dataset)])
        with open(path) as f:
            model_samples[m] = json.load(f)["samples"]

    n = len(model_samples["qwen"])
    if max_samples:
        n = min(n, max_samples)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"naive_{dataset}_{selector_key}sel_{judge_key}jud.json"
    )

    # resume support
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"Resuming from sample {start_from}")
    else:
        results    = []
        start_from = 0

    for i in range(start_from, n):
        ref  = model_samples["qwen"][i]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "meaning_altering": None,
                            "sample_WER": None, "skipped": True})
            continue

        hyps = [model_samples[m][i]["hyp"] for m in ASR_MODELS]

        # step 1: select
        best_hyp = ollama_select(client, selector_model, hyps)
        time.sleep(0.1)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "meaning_altering": None,
                            "sample_WER": None, "error": True})
            continue

        # step 2: WER
        sample_wer = wer(normalise(ref), normalise(best_hyp))

        # step 3: MAR
        ma = ollama_mar(client, judge_model, ref, best_hyp, sample_wer)
        time.sleep(0.1)

        results.append({
            "ref":              ref,
            "hyp":              best_hyp,
            "source_hyps":      {m: model_samples[m][i]["hyp"] for m in ASR_MODELS},
            "meaning_altering": ma,
            "sample_WER":       sample_wer,
        })

        with open(output_path, "w") as f:
            json.dump({"progress": len(results), "samples": results}, f,
                      indent=2, ensure_ascii=False)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n} done")

    # final metrics
    valid      = [r for r in results if not r.get("skipped") and not r.get("error")
                  and r.get("sample_WER") is not None]
    all_refs   = [normalise(r["ref"]) for r in valid]
    all_hyps   = [normalise(r["hyp"]) for r in valid]
    corpus_wer = wer(all_refs, all_hyps) if all_refs else None
    mar        = sum(1 for r in valid if r.get("meaning_altering")) / len(valid) if valid else None

    output = {
        "selector":               selector_key,
        "judge":                  judge_key,
        "dataset":                dataset,
        "models_combined":        ASR_MODELS,
        "corpus_wer":             corpus_wer,
        "meaning_alteration_rate": mar,
        "num_samples":            len(valid),
        "samples":                results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  WER: {corpus_wer:.4f}  MAR: {mar:.4f}  ({len(valid)} samples)")
    print(f"  Saved to {output_path}")
    return {"dataset": dataset, "selector": selector_key, "judge": judge_key,
            "wer": corpus_wer, "mar": mar, "n": len(valid)}

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="all",
                        help="commonvoice, edacc, english_dialects, or all")
    parser.add_argument("--selector",    default="llama", choices=["qwen", "llama"],
                        help="Model used to select/construct best transcript")
    parser.add_argument("--judge",       default="qwen",  choices=["qwen", "llama"],
                        help="Model used to evaluate meaning alteration")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    if args.dry_run:
        print(f"[DRY RUN] selector={args.selector} judge={args.judge} datasets={datasets}")
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[("qwen", datasets[0])])
        with open(path) as f:
            data = json.load(f)
        s = data["samples"][0]
        print(f"Sample 0 ref: {s['ref'][:80]}")
        for m in ASR_MODELS:
            p = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(m, datasets[0])])
            with open(p) as f:
                d = json.load(f)
            print(f"  {m}: {d['samples'][0]['hyp'][:80]}")
        return

    client = Client(host=OLLAMA_HOST)
    try:
        available = [m.model for m in client.list().models]
        print(f"Ollama models available: {available}\n")
        for key in [args.selector, args.judge]:
            model_name = OLLAMA_MODELS[key]
            if not any(model_name in m for m in available):
                print(f"ERROR: {model_name} not pulled. Run: ollama pull {model_name}")
                return
    except Exception as e:
        print(f"ERROR: could not connect to Ollama — run: ollama serve\n{e}")
        return

    summary = []
    for dataset in datasets:
        result = run_dataset(dataset, args.selector, args.judge, client, args.max_samples)
        summary.append(result)

    print("\n── Summary ────────────────────────────────────────────────")
    print(f"{'Dataset':<20} {'Selector':<10} {'Judge':<10} {'WER':>8} {'MAR':>8} {'N':>6}")
    for r in summary:
        print(f"{r['dataset']:<20} {r['selector']:<10} {r['judge']:<10} "
              f"{r['wer']:>8.4f} {r['mar']:>8.4f} {r['n']:>6}")

if __name__ == "__main__":
    main()