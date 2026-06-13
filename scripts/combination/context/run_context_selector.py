"""
run_context_selector.py

Context-aware selector using Qwen as anchor.
Passes all 3 transcripts to the selector with explicit priors from error analysis.

Output: benchmarks/combination_benchmarks/context_{dataset}_{selector}.json

Usage:
    python scripts/run_context_selector.py --dataset commonvoice --selector qwen
    python scripts/run_context_selector.py --dataset edacc --selector qwen14b
    python scripts/run_context_selector.py --dataset commonvoice --max-samples 20 --dry-run
    python scripts/run_context_selector.py --dataset commonvoice --selector qwen --rerun
"""

import json
import os
import re
import argparse
import time
from jiwer import wer
from ollama import Client

BENCHMARKS_DIR  = "benchmarks"
OUTPUT_DIR      = "results/combinations"
OLLAMA_HOST     = "http://localhost:11434"

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
}

OLLAMA_MODELS = {
    "qwen":    "qwen2.5:7b",
    "qwen14b": "qwen2.5:14b",
    "mistral": "mistral:7b",
    "gemma2":  "gemma2:9b",
}

DATASETS = ["commonvoice", "english_dialects", "edacc"]

# ── Selector prompt ────────────────────────────────────────────────────────────

SELECTOR_PROMPT = """You are correcting an ASR transcript. You are given three transcripts of the same audio from different models.

TRANSCRIPT A (base — use this as your starting point):
{qwen}

TRANSCRIPT B (Whisper):
{whisper}

TRANSCRIPT C (Parakeet):
{parakeet}

Your task: return Transcript A with targeted corrections where needed.

GENERAL RULE — applies to all words except named entities:
Only change a word if BOTH B and C disagree with A and agree with each other on the same alternative. If only one of B or C disagrees with A, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

NAMED ENTITY RULE — applies to people's names, place names, organisations:
Whisper (B) is more reliable on named entities. If B has a different named entity than A, consider switching — BUT only if C does not agree with A (case-insensitive). If C agrees with A on the named entity, keep A.

ADDITIONAL KNOWN ERROR PATTERNS:
- PROFANITY AND INFORMAL EXPRESSIONS: A sometimes self-censors mild profanity (e.g. "shit-scared"→"scared", "bloody"→"body", "sweet F all"→"sweetie fall") — if B and C have the original expression, restore it.
- NEGATIONS: If A drops or changes a negation and B and C preserve it — this is critical, switch.
- NUMBERS: If A has a different number than B and C agree on — switch.
- SCOTTISH DIALECT WORDS: If B or C preserve a Scottish dialect word that A has normalised (e.g. "wee", "wisnae", "dinnae", "cannae", "braw", "aboot", "carry-out", "noo") and both agree — preserve the dialect word.
- PRONOUNS: If A changes a pronoun (I/you/we/she/they) and B and C agree on the original — switch.

Do NOT paraphrase, reorder, or restructure sentences.
Do NOT add words that appear in none of the three transcripts.

Return ONLY the corrected transcript. No explanation, no labels, no preamble."""

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

# ── Utils ──────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
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

def ollama_select(client, model_name, qwen_hyp, whisper_hyp, parakeet_hyp):
    prompt = SELECTOR_PROMPT.format(
        qwen=qwen_hyp,
        whisper=whisper_hyp,
        parakeet=parakeet_hyp,
    )
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 4096},
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"  ERROR (selector): {e}")
        return None

def ollama_mar(client, model_name, ref, hyp, sample_wer_val):
    if sample_wer_val == 0:
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
        print(f"  ERROR (MAR judge): {e}")
        return None

# ── Main ───────────────────────────────────────────────────────────────────────

def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False):
    selector_model = OLLAMA_MODELS[selector_key]
    judge_model    = OLLAMA_MODELS["qwen"]

    print(f"\n── {dataset} | context selector={selector_key} ──────────────────")

    model_samples = {}
    for model in ["qwen", "whisper", "parakeet"]:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, dataset)])
        with open(path) as f:
            model_samples[model] = json.load(f)["samples"]

    n = len(model_samples["qwen"])
    if max_samples:
        n = min(n, max_samples)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # fixed filename — no timestamp so resume works
    output_path = os.path.join(OUTPUT_DIR, f"context_{dataset}_{selector_key}.json")

    # resume support
    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{n}")
    else:
        results    = []
        start_from = 0

    for i in range(start_from, n):
        ref          = model_samples["qwen"][i]["ref"]
        qwen_hyp     = model_samples["qwen"][i]["hyp"]
        whisper_hyp  = model_samples["whisper"][i]["hyp"]
        parakeet_hyp = model_samples["parakeet"][i]["hyp"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "skipped": True})
            continue

        best_hyp = ollama_select(client, selector_model,
                                  qwen_hyp, whisper_hyp, parakeet_hyp)
        time.sleep(0.05)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "error": True})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))
        verdict        = ollama_mar(client, judge_model, ref, best_hyp, sample_wer_val)
        time.sleep(0.05)

        results.append({
            "ref":             ref,
            "hyp":             best_hyp,
            "qwen_base":       qwen_hyp,
            "whisper_hyp":     whisper_hyp,
            "parakeet_hyp":    parakeet_hyp,
            "sample_WER":      sample_wer_val,
            "qwen_verdict_p2": verdict,
        })

        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump({"progress": len(results), "samples": results}, f,
                          indent=2, ensure_ascii=False)
            print(f"  {i+1}/{n} done")

    valid      = [r for r in results if not r.get("skipped") and not r.get("error")
                  and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None
    mar = sum(1 for r in valid if r.get("qwen_verdict_p2")) / len(valid) if valid else None

    output = {
        "selector":                selector_key,
        "approach":                "context_aware",
        "dataset":                 dataset,
        "corpus_wer":              corpus_wer,
        "meaning_alteration_rate": mar,
        "num_samples":             len(valid),
        "samples":                 results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    mar_str = f"{mar*100:.2f}%"        if mar        is not None else "—"
    print(f"  WER: {wer_str}  MAR: {mar_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")

    return {"dataset": dataset, "selector": selector_key,
            "wer": corpus_wer, "mar": mar, "n": len(valid)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="all",
                        choices=["all"] + DATASETS)
    parser.add_argument("--selector",    default="qwen",
                        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true",
                        help="Rerun from scratch ignoring existing output")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    if args.dry_run:
        print(f"[DRY RUN] selector={args.selector} datasets={datasets}")
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[("qwen", datasets[0])])
        with open(path) as f:
            data = json.load(f)
        s = data["samples"][0]
        print(f"\nSample prompt for clip 0:")
        print(SELECTOR_PROMPT.format(
            qwen=s["hyp"][:200],
            whisper="<whisper transcript>",
            parakeet="<parakeet transcript>",
        )[:600])
        return

    client = Client(host=OLLAMA_HOST)
    try:
        available = [m.model for m in client.list().models]
        if not any(OLLAMA_MODELS[args.selector] in m for m in available):
            print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
            return
        print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}\n")
    except Exception as e:
        print(f"ERROR: could not connect to Ollama\n{e}")
        return

    summary = []
    for dataset in datasets:
        result = run_dataset(dataset, args.selector, client, args.max_samples, args.rerun)
        summary.append(result)

    if len(summary) > 1:
        print(f"\n{'='*65}")
        print("SUMMARY — Context-aware selector")
        print(f"{'='*65}")
        print(f"{'Dataset':<22} {'Selector':<10} {'WER':>8} {'MAR':>8} {'N':>6}")
        for r in summary:
            wer_str = f"{r['wer']*100:.2f}%" if r['wer'] else "—"
            mar_str = f"{r['mar']*100:.2f}%" if r['mar'] else "—"
            print(f"  {r['dataset']:<20} {r['selector']:<10} {wer_str:>8} {mar_str:>8} {r['n']:>6}")

if __name__ == "__main__":
    main()