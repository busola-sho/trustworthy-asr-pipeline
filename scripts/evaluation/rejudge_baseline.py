"""
rejudge_baseline.py

Re-runs the MAR judge (named-entity-excluding v2 prompt) over existing
single-model baseline transcripts, restricted to the 150-sample subset
(seed=42). Supports all four ASR models: qwen, whisper, parakeet, wav2vec2.

For shetland (100 samples total), the full set is used directly.

Output: results/combinations_v2judge/baseline/{model}_{dataset}_sub150.json

Usage:
    python scripts/evaluation/rejudge_baseline.py --model qwen --dataset commonvoice
    python scripts/evaluation/rejudge_baseline.py --model whisper --dataset edacc
    python scripts/evaluation/rejudge_baseline.py --model parakeet --dataset english_dialects
    python scripts/evaluation/rejudge_baseline.py --model wav2vec2 --dataset commonvoice
    python scripts/evaluation/rejudge_baseline.py --model qwen --dataset shetland
"""

import json
import os
import re
import argparse
import time
import random
from jiwer import wer
from ollama import Client

OUTPUT_DIR  = "results/combinations_v2judge/baseline"
OLLAMA_HOST = "http://localhost:11434"

CANONICAL_FILES = {
    ("qwen",     "commonvoice"):      "results/benchmarks/main/qwen_commonvoice_20260524_153426.json",
    ("qwen",     "edacc"):            "results/benchmarks/main/qwen_edacc_20260525_204314.json",
    ("qwen",     "english_dialects"): "results/benchmarks/main/qwen_english_dialects_20260525_000627.json",
    ("qwen",     "shetland"):         "results/benchmarks/shetland/shetland_qwen3asr_20260603_150124.json",

    ("whisper",  "commonvoice"):      "results/benchmarks/main/whisper_commonvoice_20260524_083930.json",
    ("whisper",  "edacc"):            "results/benchmarks/main/whisper_edacc_20260525_184504.json",
    ("whisper",  "english_dialects"): "results/benchmarks/main/whisper_english_dialects_20260525_110315.json",
    ("whisper",  "shetland"):         "results/benchmarks/shetland/shetland_whisper_20260603_123115.json",

    ("parakeet", "commonvoice"):      "results/benchmarks/main/parakeet_commonvoice_20260524_150129.json",
    ("parakeet", "edacc"):            "results/benchmarks/main/parakeet_edacc_20260525_184006.json",
    ("parakeet", "english_dialects"): "results/benchmarks/main/parakeet_english_dialects_20260524_234807.json",
    ("parakeet", "shetland"):         "results/benchmarks/shetland/shetland_parakeet_20260606_134131.json",

    ("wav2vec2", "commonvoice"):      "results/benchmarks/main/wav2vec2_commonvoice_20260526_053757.json",
    ("wav2vec2", "edacc"):            "results/benchmarks/main/wav2vec2_edacc_20260525_213534.json",
    ("wav2vec2", "english_dialects"): "results/benchmarks/main/wav2vec2_english_dialects_20260526_073439.json",
    ("wav2vec2", "shetland"):         "results/benchmarks/shetland/shetland_wav2vec2_20260606_134507.json",
}

OLLAMA_MODELS = {"qwen": "qwen2.5:7b"}

SEED = 42
N_SUBSET = 150
DATASET_SIZES = {
    "commonvoice":      680,
    "edacc":            198,
    "english_dialects": 2543,
    "shetland":         100,
}


def get_subset_indices(dataset: str) -> list:
    n_total = DATASET_SIZES[dataset]
    if n_total <= N_SUBSET:
        return list(range(n_total))
    random.seed(SEED)
    return sorted(random.sample(range(n_total), min(N_SUBSET, n_total)))


MAR_PROMPT = """You are evaluating ASR transcripts in a high-stakes context where accuracy matters. Given a reference and hypothesis transcript of the same audio, determine if the hypothesis contains meaning-altering errors that would cause someone to misunderstand what was said.

Ignore: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"→"didn't", "oot"→"out"), filler words.

NAMED ENTITIES: Ignore spelling or form variation in people's names, place names, and organisation names (e.g. "Forfar" vs "Forfa", "Mhairi" vs "Maria"). Even human transcribers cannot reliably spell unfamiliar names from audio alone. Only flag a named entity as an error if it is unambiguously a DIFFERENT entity altogether (wrong city, wrong person, a number that changes which date/amount is referenced) — not merely a different spelling of the same intended entity.

Flag as meaning-altering if:
- Factual content changes
- Negation is added or removed
- A name, place, or number refers to a genuinely different entity (not just a spelling variant)
- A dialect word is misrecognised as a different real word (e.g. "bairn"→"barn")
- Content is hallucinated over inaudible segments

Example:
Reference: My neighbour Mhairi McTaggart said she heard the noise around midnight.
Hypothesis: My neighbour Maria MacTaggart said she heard the noise around midnight.
This refers to the same person and the same claim — the spelling differs but no meaning changed.
Answer: false

IMPORTANT: After any reasoning, your FINAL line must be ONLY the single word true or false."""


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


def run_dataset(model_key, dataset, client, max_samples=None, rerun=False):
    judge_model = OLLAMA_MODELS["qwen"]
    is_full = DATASET_SIZES[dataset] <= N_SUBSET
    label = f"full set (N={DATASET_SIZES[dataset]})" if is_full else "subset=150"
    print(f"\n── {model_key} / {dataset} | baseline rejudge | {label} ──")

    path = CANONICAL_FILES[(model_key, dataset)]
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return

    with open(path) as f:
        samples = json.load(f)["samples"]

    indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]
    n = len(indices)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = "" if is_full else "_sub150"
    output_path = os.path.join(OUTPUT_DIR, f"baseline_{model_key}_{dataset}{suffix}.json")

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{n}")
    else:
        results    = []
        start_from = 0

    for pos in range(start_from, n):
        idx = indices[pos]
        ref = samples[idx]["ref"]
        hyp = samples[idx]["hyp"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": hyp, "qwen_verdict_p2": None,
                            "sample_WER": None, "skipped": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(hyp))
        verdict        = ollama_mar(client, judge_model, ref, hyp, sample_wer_val)
        time.sleep(0.05)

        results.append({
            "ref":             ref,
            "hyp":             hyp,
            "sample_WER":      sample_wer_val,
            "qwen_verdict_p2": verdict,
            "dataset_index":   idx,
        })

        if (pos + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump({"progress": len(results), "samples": results}, f,
                          indent=2, ensure_ascii=False)
            print(f"  {pos+1}/{n} done")

    valid = [r for r in results
             if not r.get("skipped") and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None
    mar = (sum(1 for r in valid if r.get("qwen_verdict_p2")) / len(valid)
           if valid else None)

    output = {
        "model":                   model_key,
        "approach":                "baseline",
        "dataset":                 dataset,
        "subset_indices":          indices,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True,
                        choices=["qwen", "whisper", "parakeet", "wav2vec2", "all"])
    parser.add_argument("--dataset", required=True,
                        choices=["commonvoice", "edacc", "english_dialects", "shetland", "all"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    models   = ["qwen", "whisper", "parakeet", "wav2vec2"] if args.model == "all" else [args.model]
    datasets = ["commonvoice", "edacc", "english_dialects"] if args.dataset == "all" else [args.dataset]

    client = Client(host=OLLAMA_HOST)
    try:
        available = [m.model for m in client.list().models]
        if not any(OLLAMA_MODELS["qwen"] in m for m in available):
            print(f"ERROR: {OLLAMA_MODELS['qwen']} not pulled.")
            return
        print(f"Ollama connected. Judge: {OLLAMA_MODELS['qwen']}\n")
    except Exception as e:
        print(f"ERROR: could not connect to Ollama\n{e}")
        return

    for model_key in models:
        for dataset in datasets:
            if (model_key, dataset) not in CANONICAL_FILES:
                print(f"  SKIP: no file for {model_key}/{dataset}")
                continue
            run_dataset(model_key, dataset, client,
                        max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()