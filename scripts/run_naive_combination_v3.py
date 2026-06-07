"""
run_naive_combination_v3.py

Naive combination baseline with:
- Separate selector and judge models
- 3 selector prompt variants (p1=liberal, p2=pure chooser, p3=word-level combiner)
- Output to benchmarks/combination_benchmarks/
- Resume-friendly

Usage:
    python scripts/run_naive_combination_v3.py --dataset commonvoice --selector qwen --judge qwen --selector-prompt p2
    python scripts/run_naive_combination_v3.py --dataset all --selector qwen14b --judge qwen --selector-prompt p3
    python scripts/run_naive_combination_v3.py --dry-run --selector gemma2 --judge qwen --selector-prompt p1
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
OUTPUT_DIR     = "benchmarks/combination_benchmarks"
OLLAMA_HOST    = "http://localhost:11434"

OLLAMA_MODELS = {
    "qwen":    "qwen2.5:7b",
    "llama":   "llama3.1:8b",
    "qwen14b": "qwen2.5:14b",
    "mistral": "mistral:7b",
    "gemma2":  "gemma2:9b",
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

# ── Selector Prompts ───────────────────────────────────────────────────────────

SELECTOR_PROMPT_P1 = """You are given four ASR transcripts of the same spoken audio.

Your task is to produce the single most accurate transcript — the one that best preserves the meaning of what was said, with particular care for:
- Named entities (names, places, numbers)
- Negations
- Dialect words that carry meaning

You may select one transcript verbatim, edit it, or combine the best parts across transcripts. You may also lightly paraphrase where it improves clarity.

Return only the final transcript, nothing else."""

SELECTOR_PROMPT_P2 = """You are given four ASR transcripts of the same spoken audio.

Your task is to select the single most accurate transcript — the one that best preserves the meaning of what was said, with particular care for:
- Named entities (names, places, numbers)
- Negations
- Dialect words that carry meaning

You MUST return one of the four transcripts exactly as written. Do not edit, paraphrase, reorder, or combine them.

Return only the selected transcript, nothing else."""

SELECTOR_PROMPT_P3 = """You are given four ASR transcripts of the same spoken audio.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps

Return only the final transcript, nothing else."""

SELECTOR_PROMPTS = {
    "p1": SELECTOR_PROMPT_P1,
    "p2": SELECTOR_PROMPT_P2,
    "p3": SELECTOR_PROMPT_P3,
}

# ── MAR Prompt (Qwen P2) ───────────────────────────────────────────────────────

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

def ollama_select(client, model_name, hyps, selector_prompt):
    hyp_block = "\n".join([f"Transcript {i+1}: {h}" for i, h in enumerate(hyps)])
    try:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": selector_prompt},
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

def run_dataset(dataset, selector_key, judge_key, selector_prompt_key, client, max_samples=None):
    selector_model  = OLLAMA_MODELS[selector_key]
    judge_model     = OLLAMA_MODELS[judge_key]
    selector_prompt = SELECTOR_PROMPTS[selector_prompt_key]

    print(f"\n── {dataset} | selector={selector_key} ({selector_prompt_key}) judge={judge_key} ──")

    model_samples = {}
    for m in ASR_MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(m, dataset)])
        with open(path) as f:
            model_samples[m] = json.load(f)["samples"]

    n = len(model_samples["qwen"])
    if max_samples:
        n = min(n, max_samples)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"naive_{dataset}_{selector_key}sel_{selector_prompt_key}_{judge_key}jud.json"
    )

    # resume support
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}")
    else:
        results    = []
        start_from = 0

    for i in range(start_from, n):
        ref = model_samples["qwen"][i]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "skipped": True})
            continue

        hyps = [model_samples[m][i]["hyp"] for m in ASR_MODELS]

        # step 1: select
        best_hyp = ollama_select(client, selector_model, hyps, selector_prompt)
        time.sleep(0.1)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "error": True})
            continue

        # step 2: WER
        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        # step 3: MAR
        ma = ollama_mar(client, judge_model, ref, best_hyp, sample_wer_val)
        time.sleep(0.1)

        results.append({
            "ref":             ref,
            "hyp":             best_hyp,
            "source_hyps":     {m: model_samples[m][i]["hyp"] for m in ASR_MODELS},
            "qwen_verdict_p2": ma,
            "sample_WER":      sample_wer_val,
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
    mar        = sum(1 for r in valid if r.get("qwen_verdict_p2")) / len(valid) if valid else None

    output = {
        "selector":                selector_key,
        "selector_prompt":         selector_prompt_key,
        "judge":                   judge_key,
        "dataset":                 dataset,
        "models_combined":         ASR_MODELS,
        "corpus_wer":              corpus_wer,
        "meaning_alteration_rate": mar,
        "num_samples":             len(valid),
        "samples":                 results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  WER: {corpus_wer*100:.2f}%  MAR: {mar*100:.2f}%  ({len(valid)} samples)")
    print(f"  Saved to {output_path}")
    return {
        "dataset": dataset, "selector": selector_key,
        "selector_prompt": selector_prompt_key,
        "judge": judge_key, "wer": corpus_wer, "mar": mar, "n": len(valid)
    }

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",         default="all",
                        help="commonvoice, edacc, english_dialects, or all")
    parser.add_argument("--selector",        default="qwen",
                        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--judge",           default="qwen",
                        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--selector-prompt", default="p2",
                        choices=["p1", "p2", "p3"],
                        help="p1=liberal, p2=pure chooser, p3=word-level combiner")
    parser.add_argument("--max-samples",     type=int, default=None)
    parser.add_argument("--dry-run",         action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    if args.dry_run:
        print(f"[DRY RUN] selector={args.selector} ({args.selector_prompt}) "
              f"judge={args.judge} datasets={datasets}")
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[("qwen", datasets[0])])
        with open(path) as f:
            data = json.load(f)
        s = data["samples"][0]
        print(f"Sample 0 ref: {s['ref'][:80]}")
        print(f"Selector prompt: {args.selector_prompt}")
        print(f"Output dir: {OUTPUT_DIR}")
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
        result = run_dataset(
            dataset, args.selector, args.judge,
            args.selector_prompt, client, args.max_samples
        )
        summary.append(result)

    print("\n── Summary ────────────────────────────────────────────────────────")
    print(f"{'Dataset':<20} {'Selector':<10} {'Prompt':<8} {'Judge':<8} {'WER':>8} {'MAR':>8} {'N':>6}")
    for r in summary:
        print(f"{r['dataset']:<20} {r['selector']:<10} {r['selector_prompt']:<8} "
              f"{r['judge']:<8} {r['wer']*100:>7.2f}% {r['mar']*100:>7.2f}% {r['n']:>6}")

if __name__ == "__main__":
    main()