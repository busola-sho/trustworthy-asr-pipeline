"""
rerunning/selector_ablation.py

Rigorously justifies the choice of selector model (the LLM that combines 4
ASR transcripts into one) instead of inheriting whichever model happened to
be the judge at the time.

Reuses the existing judge shortlist as selector candidates (same
recency/license/tier justification already established), and scores each
candidate's resulting combined transcript on your existing 100-sentence
human-annotated pool - reusing infrastructure rather than building new
annotation from scratch.

For each of the 100 sentences:
  1. Look up the same underlying sample (via dataset + sample_index) in
     each of the 4 models' full-dataset benchmark files, to get all 4 real
     ASR hypotheses for that audio (whisperx, qwen, parakeet, wav2vec2)
  2. Run the naive-combination selector prompt with each candidate model
  3. Score the resulting combined transcript: WER, and severity via the
     locked Phi-4 + direct judge
  4. Also check constraint compliance: did the selector only swap words
     from the 4 inputs, or did it introduce content absent from all of them?

NOTE: candidate_pool.json's "dataset" field carries older/alternate
dataset names from whenever the pool was generated - these don't match
find_canonical_file's keys (or the actual on-disk benchmark filenames).
DATASET_NAME_MAP below normalises them before any lookup, so this script
self-corrects regardless of what naming convention the pool file uses.

NOTE 2: on resume, existing results are deduped by uid (keeping the LAST
entry seen per uid) before determining what's already done, and new
results REPLACE any existing entry for the same uid rather than being
appended alongside it. Without this, rerunning after a fix (e.g. an
earlier run where every sample failed with missing_source_data, followed
by a rerun that succeeds) leaves stale failed entries sitting in the
output file next to the correct ones, with uid no longer unique. Summary
stats were never affected by this (they filter on severity is not None),
but the raw samples list was polluted.

Usage:
    python rerunning/selector_ablation.py --candidates qwen3.5 gemma4 phi4 ministral3 qwen
    python rerunning/selector_ablation.py --candidates all
"""

import json
import os
import re
import argparse
import time
from pathlib import Path
from jiwer import wer
from ollama import Client

from src.judge import normalise, is_tag_only
from src.selector import find_canonical_file, OLLAMA_MODELS, load_samples
from severity_judge_prompts import DIRECT_SEVERITY_PROMPT

CANDIDATE_POOL_PATH = "writeup_results/candidate_pool.json"
OUTPUT_DIR = "writeup_results/selector_ablation"

OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "phi4:14b"   # locked severity judge (Phi-4 + direct, QWK=0.783)
ASR_MODELS  = ["qwen", "whisperx", "parakeet", "wav2vec2"]

ALL_CANDIDATES = ["qwen", "qwen3.5", "gemma4", "phi4", "ministral3"]

# candidate_pool.json still carries older/alternate dataset names from
# whenever it was generated - normalise them to match the actual
# find_canonical_file keys (and your on-disk filenames) before any lookup.
DATASET_NAME_MAP = {
    "edinburgh_international_accents": "edacc",
    "english_dialects_scots": "english_dialects",
    "common_voice": "commonvoice",
}

SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps

Return only the final transcript, nothing else."""


def normalise_dataset_name(name: str) -> str:
    return DATASET_NAME_MAP.get(name, name)


def parse_severity(text: str):
    match = re.search(r"severity\s*:\s*([0-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"(?<!\d)[0-4](?!\d)", text)
    return int(fallback[-1]) if fallback else None


def ollama_severity(client: Client, ref: str, hyp: str, retries: int = 2):
    prompt = DIRECT_SEVERITY_PROMPT.format(reference=ref, hypothesis=hyp)
    text = None
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
                think=False,
            )
            text = response.message.content
            severity = parse_severity(text)
            if severity is not None:
                return severity, text
            if attempt < retries:
                time.sleep(0.5)
        except Exception as e:
            if attempt == retries:
                return None, None
            time.sleep(1.0)
    return None, text


def ollama_select(client: Client, model_name: str, hyps: dict, retries: int = 2):
    hyp_block = "\n".join([f"Transcript {i+1} ({m}): {h}" for i, (m, h) in enumerate(hyps.items())])
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SELECTOR_PROMPT},
                    {"role": "user",   "content": hyp_block},
                ],
                options={"temperature": 0, "num_ctx": 4096},
                think=False,
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"    ERROR (select): {e}")
                return None
            time.sleep(1.0)
    return None


def check_compliance(selected: str, source_hyps: dict) -> bool:
    """
    Rough constraint-compliance check: does every word in the selected
    transcript appear somewhere in the union of the 4 source transcripts?
    A high rate of "invented" words suggests the selector paraphrased or
    hallucinated rather than just swapping, violating the prompt's rules.
    Returns True if compliant (>=90% of words traceable to a source).
    """
    source_words = set()
    for hyp in source_hyps.values():
        source_words.update(normalise(hyp).split())

    selected_words = normalise(selected).split()
    if not selected_words:
        return False

    traceable = sum(1 for w in selected_words if w in source_words)
    return (traceable / len(selected_words)) >= 0.9


def load_all_model_samples():
    """Load full-dataset samples for all 4 ASR models, per dataset, indexed
    by sample_index for fast lookup."""
    cache = {}
    datasets_needed = set()

    with open(CANDIDATE_POOL_PATH) as f:
        pool = json.load(f)
    for entry in pool:
        datasets_needed.add(normalise_dataset_name(entry["dataset"]))

    for dataset in datasets_needed:
        cache[dataset] = {}
        for model in ASR_MODELS:
            try:
                path = find_canonical_file(model, dataset)
                samples = load_samples(path)
                cache[dataset][model] = {
                    s["sample_index"]: s for s in samples if s.get("sample_index") is not None
                }
                print(f"  Loaded {model}/{dataset}: {len(cache[dataset][model])} samples from {path}")
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                cache[dataset][model] = {}

    return cache


def load_existing_results(output_path: str) -> dict:
    """
    Load an existing output file's samples, deduped by uid. If the same
    uid appears more than once (e.g. a stale failed entry from a run
    before a bugfix, sitting alongside a newer successful entry), the
    LAST occurrence in the file wins - later entries in the list reflect
    more recent runs. Returns {uid: entry}.
    """
    if not os.path.exists(output_path):
        return {}
    with open(output_path) as f:
        existing_samples = json.load(f).get("samples", [])
    results_by_uid = {}
    for r in existing_samples:
        results_by_uid[r["uid"]] = r
    return results_by_uid


def run_ablation(candidates: list):
    client = Client(host=OLLAMA_HOST)

    print("Loading candidate pool...")
    with open(CANDIDATE_POOL_PATH) as f:
        pool = json.load(f)
    print(f"  {len(pool)} sentences in pool")

    print("\nLoading all 4 ASR models' full-dataset outputs...")
    model_cache = load_all_model_samples()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for candidate_key in candidates:
        model_name = OLLAMA_MODELS[candidate_key]
        output_path = os.path.join(OUTPUT_DIR, f"selector_{candidate_key}.json")

        # keyed by uid throughout the run, so a new result for a uid
        # REPLACES any existing entry (e.g. a stale error from a prior
        # broken run) instead of piling up alongside it
        results_by_uid = load_existing_results(output_path)
        done_uids = {uid for uid, r in results_by_uid.items() if r.get("severity") is not None}

        print(f"\n── Candidate selector: {candidate_key} ({model_name}) ──")
        print(f"  {len(pool)} sentences, {len(done_uids)} already done, resuming...")

        processed_since_save = 0

        for i, entry in enumerate(pool):
            uid = entry["uid"]
            if uid in done_uids:
                continue

            dataset = normalise_dataset_name(entry["dataset"])
            sample_index = entry.get("sample_index")

            source_hyps = {}
            ref = None
            for model in ASR_MODELS:
                sample = model_cache.get(dataset, {}).get(model, {}).get(sample_index)
                if sample is None:
                    continue
                source_hyps[model] = sample.get("hyp", "") or ""
                ref = sample.get("ref", ref)

            if not ref or len(source_hyps) < 4:
                results_by_uid[uid] = {
                    "uid": uid, "dataset": dataset, "sample_index": sample_index,
                    "severity": None, "error": "missing_source_data",
                }
                continue

            if is_tag_only(ref):
                results_by_uid[uid] = {
                    "uid": uid, "dataset": dataset, "sample_index": sample_index,
                    "severity": None, "skipped": True, "skip_reason": "tag_only_reference",
                }
                continue

            selected = ollama_select(client, model_name, source_hyps)
            if selected is None:
                results_by_uid[uid] = {
                    "uid": uid, "dataset": dataset, "sample_index": sample_index,
                    "severity": None, "error": "selector_call_failed",
                }
                continue

            sample_wer_val = wer(normalise(ref), normalise(selected))
            severity, raw_response = ollama_severity(client, ref, selected)
            compliant = check_compliance(selected, source_hyps)

            results_by_uid[uid] = {
                "uid":                uid,
                "dataset":            dataset,
                "sample_index":       sample_index,
                "ref":                ref,
                "selected_hyp":       selected,
                "source_hyps":        source_hyps,
                "sample_WER":         sample_wer_val,
                "severity":           severity,
                "judge_raw_response": raw_response,
                "constraint_compliant": compliant,
            }

            print(f"  [{i+1}/{len(pool)}] {uid}: WER={sample_wer_val*100:.1f}%  "
                  f"severity={severity}  compliant={compliant}")

            processed_since_save += 1
            if processed_since_save % 10 == 0:
                with open(output_path, "w") as f:
                    json.dump({"candidate": candidate_key, "model": model_name,
                               "samples": list(results_by_uid.values())}, f, indent=2, ensure_ascii=False)

        # final save + summary stats
        valid = [r for r in results_by_uid.values() if r.get("severity") is not None]
        severities = [r["severity"] for r in valid]
        wers = [r["sample_WER"] for r in valid if r.get("sample_WER") is not None]
        compliance_rate = (
            sum(1 for r in valid if r.get("constraint_compliant")) / len(valid)
            if valid else None
        )

        summary = {
            "candidate":         candidate_key,
            "model":             model_name,
            "n_scored":          len(valid),
            "n_total":           len(pool),
            "mean_severity":     round(sum(severities) / len(severities), 3) if severities else None,
            "severity_distribution": {str(i): severities.count(i) for i in range(5)},
            "mean_wer":          round(sum(wers) / len(wers), 4) if wers else None,
            "constraint_compliance_rate": round(compliance_rate, 3) if compliance_rate is not None else None,
        }

        output = {"candidate": candidate_key, "model": model_name,
                  "summary": summary, "samples": list(results_by_uid.values())}
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n  {candidate_key}: mean_severity={summary['mean_severity']}  "
              f"mean_wer={summary['mean_wer']}  compliance={summary['constraint_compliance_rate']}")
        print(f"  Saved: {output_path}")


def print_comparison():
    """Print a summary table comparing all candidates that have been run."""
    rows = []
    for f in sorted(Path(OUTPUT_DIR).glob("selector_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        s = data.get("summary", {})
        if s.get("mean_severity") is not None:
            rows.append(s)

    if not rows:
        print("No completed candidates to compare yet.")
        return

    rows.sort(key=lambda r: r["mean_severity"])
    print(f"\n{'Candidate':<12} {'Model':<16} {'Mean Sev':>9} {'Mean WER':>9} {'Compliance':>11}")
    print("-" * 60)
    for r in rows:
        star = " ★ BEST" if r == rows[0] else ""
        print(f"{r['candidate']:<12} {r['model']:<16} {r['mean_severity']:>9.3f} "
              f"{r['mean_wer']*100:>8.2f}% {r['constraint_compliance_rate']:>10.1%}{star}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=ALL_CANDIDATES,
                        help="Candidate selector keys (from OLLAMA_MODELS), or 'all'")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip running, just print comparison of existing results")
    args = parser.parse_args()

    if args.compare_only:
        print_comparison()
        return

    candidates = ALL_CANDIDATES if "all" in args.candidates else args.candidates
    run_ablation(candidates)
    print_comparison()


if __name__ == "__main__":
    main()