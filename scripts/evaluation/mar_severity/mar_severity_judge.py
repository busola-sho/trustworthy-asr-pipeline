"""
mar_severity_judge.py

Severity-scaled Meaning Alteration Rate (MAR) judge — extends the binary
true/false MAR judge to a 0-4 ordinal scale, giving a more granular and
(hopefully) more stable signal than the binary version, which showed
verdict instability on borderline cases at temperature=0.

Scale:
    0 - No meaning change (perfect match, or only surface-level differences:
        capitalisation, contractions, dialect normalisation, filler words)
    1 - Trivial/cosmetic error (doesn't affect practical interpretation,
        e.g. "Rd" vs "Road", digit vs spelled-out number, minor synonym)
    2 - Ambiguous or softened meaning shift (hedging changed, vague pronoun
        reference altered, certainty/tone shift — could matter depending
        on context but isn't a clean factual reversal)
    3 - Clear factual error, non-critical detail (wrong name/place/number/
        date that's incorrect but doesn't flip the substance of the account)
    4 - Critical meaning reversal or fabrication (negation flipped, alibi
        reversed, hallucinated content, or a name/place/number error that
        changes who/where/when in a way that materially affects the account)

Usage:
    python scripts/evaluation/mar_severity/mar_severity_judge.py --dataset commonvoice --source context_v1
    python scripts/evaluation/mar_severity/mar_severity_judge.py --dataset commonvoice --source context_v2
    python scripts/evaluation/mar_severity/mar_severity_judge.py --dataset commonvoice --source context_v1_confidence --threshold 0.5
    python scripts/evaluation/mar_severity/mar_severity_judge.py --dataset commonvoice --source context_v2_confidence --threshold 0.8
"""

import json
import os
import re
import random
import argparse
import time
from ollama import Client

OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5:7b"

SEVERITY_OUTPUT_DIR = "results/mar_severity_v2judge"

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
    random.seed(SEED)
    return sorted(random.sample(range(n_total), min(N_SUBSET, n_total)))


SEVERITY_PROMPT = """You are evaluating ASR transcripts in a high-stakes context where accuracy matters. Given a reference and hypothesis transcript of the same audio, rate the SEVERITY of any meaning-altering error on a scale of 0-4.

IMPORTANT — NAMED ENTITIES: Ignore spelling or form variation in people's names, place names, and organisation names (e.g. "Forfar" vs "Forfa", "Mhairi" vs "Maria", "McTaggart" vs "MacTaggart"). Even human transcribers cannot reliably spell unfamiliar names from audio alone, so this is not a meaningful error in the same sense as a factual or semantic change. Only score a named entity as an error if it is unambiguously a DIFFERENT entity altogether (e.g. wrong city, wrong person mentioned, a number that changes which date/amount is referenced) — not merely a different spelling or transcription of the same intended entity.

SCALE:
0 = No meaning change. Either a perfect match, or differences are purely surface-level: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"->"didn't", "oot"->"out"), filler words, or named entity spelling variation (see above).
1 = Trivial/cosmetic error. A word changed but does not affect practical interpretation (e.g. "Rd" vs "Road", a digit vs spelled-out number, a minor synonym swap with no factual consequence).
2 = Ambiguous or softened meaning shift. Something changed that could matter depending on context but is not a clean factual reversal (e.g. hedging language altered such as "might have" becoming "did", a vague pronoun reference changed, a shift in certainty or tone).
3 = Clear factual error on a non-critical detail. A wrong name, place, number, or date that is incorrect but does not flip the substance of the account (e.g. wrong street name in an otherwise correctly identified area, a time off by a small margin).
4 = Critical meaning reversal or fabrication. Negation added or removed, an alibi reversed, content hallucinated over an inaudible segment, or a name/place/number error that changes who/where/when in a way that materially affects the account.

Examples:
Reference: She said she wisnae near the pub on Saturday night.
Hypothesis: She said she was near the pub on Saturday night.
Reasoning: Negation "wisnae" dropped, reversing the speaker's alibi.
Answer: 4

Reference: He works the back shift at the factory on Keppoch Road.
Hypothesis: He works the back shift at the factory on Keppoch Rd.
Reasoning: "Rd" is a standard abbreviation for Road; same location, no factual content lost.
Answer: 1

Reference: I think it was around three or four people at the meeting.
Hypothesis: There were three or four people at the meeting.
Reasoning: Hedging ("I think... around") dropped, making an uncertain estimate sound definite — could matter for witness reliability but is not a hard factual reversal.
Answer: 2

Reference: He said he saw her near Gorbals Street that evening.
Hypothesis: He said he saw her near Garscube Street that evening.
Reasoning: Wrong street name — a real factual error, but both are plausible Glasgow streets and the core claim (he saw her, that evening) is unchanged.
Answer: 3

Reference: My neighbour Mhairi McTaggart said she heard the noise around midnight.
Hypothesis: My neighbour Maria MacTaggart said she heard the noise around midnight.
Reasoning: The name is spelled/transcribed differently, but it clearly refers to the same person and the same claim. Named entity spelling variation is excluded from scoring per the rule above.
Answer: 0

IMPORTANT: Reply with ONLY the single digit 0, 1, 2, 3, or 4. No explanation, no reasoning, no other text."""


def parse_severity(result: str):
    result = result.strip()
    match = re.search(r'[0-4]', result)
    if match:
        return int(match.group())
    return None


def ollama_severity(client, ref, hyp, retries=2):
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SEVERITY_PROMPT},
                    {"role": "user",   "content": f"Reference: {ref}\nHypothesis: {hyp}"},
                ],
                options={"temperature": 0},
            )
            severity = parse_severity(response.message.content)
            if severity is not None:
                return severity
            print(f"  WARNING: could not parse severity from: '{response.message.content[:80]}'")
        except Exception as e:
            print(f"  ERROR (severity judge, attempt {attempt+1}): {e}")
        time.sleep(0.2)
    return None


def run_severity_scoring(input_path, output_path, client, dataset=None, subset_only=False,
                          max_samples=None, rerun=False):
    with open(input_path) as f:
        data = json.load(f)

    all_samples = data.get("samples", [])

    if subset_only:
        if dataset is None:
            raise ValueError("dataset is required when subset_only=True")
        subset_indices = set(get_subset_indices(dataset))
        # V2/confidence files already store dataset_index per sample; V1/naive/
        # baseline files don't, so fall back to positional indexing (position
        # == dataset index for these files, since they're saved in original
        # dataset order)
        samples = []
        for pos, s in enumerate(all_samples):
            idx = s.get("dataset_index", pos)
            if idx in subset_indices:
                samples.append(s)
        print(f"  Restricting to 150-subset: {len(samples)}/{len(all_samples)} samples kept")
    else:
        samples = all_samples

    if max_samples:
        samples = samples[:max_samples]

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(samples)}")
    else:
        results = []
        start_from = 0

    for i in range(start_from, len(samples)):
        s = samples[i]
        ref = s.get("ref")
        hyp = s.get("hyp")

        if not ref or not hyp or s.get("skipped") or s.get("error"):
            results.append({**s, "mar_severity": None})
            continue

        severity = ollama_severity(client, ref, hyp)
        results.append({**s, "mar_severity": severity})
        time.sleep(0.05)

        if (i + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump({"progress": len(results), "samples": results}, f,
                          indent=2, ensure_ascii=False)
            print(f"  {i+1}/{len(samples)} done")

    valid = [r for r in results if r.get("mar_severity") is not None]
    severities = [r["mar_severity"] for r in valid]

    severity_counts = {lvl: severities.count(lvl) for lvl in range(5)}
    mean_severity = sum(severities) / len(severities) if severities else None
    binary_mar_equivalent = sum(1 for s in severities if s >= 2) / len(severities) if severities else None

    output = {
        "source_file": input_path,
        "num_samples": len(valid),
        "severity_counts": severity_counts,
        "mean_severity": mean_severity,
        "binary_mar_equivalent_threshold2": binary_mar_equivalent,
        "samples": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Severity distribution: {severity_counts}")
    print(f"  Mean severity: {mean_severity:.3f}" if mean_severity is not None else "  Mean severity: —")
    print(f"  Binary-equivalent MAR (severity>=2): {binary_mar_equivalent*100:.2f}%" if binary_mar_equivalent is not None else "")
    print(f"  Saved: {output_path}")


# non-confidence sources use a fixed lookup table
SOURCE_FILES = {
    ("commonvoice", "baseline"):   "results/combinations_v2judge/baseline/baseline_commonvoice_qwen_sub150.json",
    ("commonvoice", "naive"):      "results/combinations_v2judge/naive_commonvoice_qwensel_p3_qwenjud_sub150.json",
    ("commonvoice", "context_v1"): "results/combinations_v2judge/context/context_commonvoice_qwen_sub150.json",
    ("commonvoice", "context_v2"): "results/combinations_v2judge/context_v2/context_v2_commonvoice_qwen_sub150.json",

    ("edacc",       "baseline"):   "results/combinations_v2judge/baseline/baseline_edacc_qwen_sub150.json",
    ("edacc",       "naive"):      "results/combinations_v2judge/naive_edacc_qwensel_p3_qwenjud_sub150.json",
    ("edacc",       "context_v1"): "results/combinations_v2judge/context/context_edacc_qwen_sub150.json",
    ("edacc",       "context_v2"): "results/combinations_v2judge/context_v2/context_v2_edacc_qwen_sub150.json",

    ("english_dialects", "baseline"):   "results/combinations_v2judge/baseline/baseline_english_dialects_qwen_sub150.json",
    ("english_dialects", "naive"):      "results/combinations_v2judge/naive_english_dialects_qwensel_p3_qwenjud_sub150.json",
    ("english_dialects", "context_v1"): "results/combinations_v2judge/context/context_english_dialects_qwen_sub150.json",
    ("english_dialects", "context_v2"): "results/combinations_v2judge/context_v2/context_v2_english_dialects_qwen_sub150.json",

    ("shetland", "baseline"):   "results/combinations_v2judge/baseline/baseline_shetland_qwen.json",
    ("shetland", "naive"):      "results/combinations_v2judge/naive_shetland_qwensel_p3_qwenjud_sub150.json",
    ("shetland", "context_v1"): "results/combinations_v2judge/context/context_shetland_qwen_sub150.json",
    ("shetland", "context_v2"): "results/combinations_v2judge/context_v2/context_v2_shetland_qwen_sub150.json",

    ("commonvoice", "context_v2_whisperx"): "results/combinations_v2judge/context_v2_whisperx/context_v2_commonvoice_qwen_sub150.json",
    ("edacc",       "context_v2_whisperx"): "results/combinations_v2judge/context_v2_whisperx/context_v2_edacc_qwen_sub150.json",
    ("english_dialects", "context_v2_whisperx"): "results/combinations_v2judge/context_v2_whisperx/context_v2_english_dialects_qwen_sub150.json",

    ("commonvoice", "rover_auditor"): "results/combinations_v2judge/rover/rover_commonvoice_sub150.json",
    ("edacc",       "rover_auditor"): "results/combinations_v2judge/rover/rover_edacc_sub150.json",
    ("english_dialects", "rover_auditor"): "results/combinations_v2judge/rover/rover_english_dialects_sub150.json",

    ("commonvoice", "rover_no_auditor"): "results/combinations_v2judge/rover/rover_commonvoice_sub150_no_resolver.json",
    ("edacc",       "rover_no_auditor"): "results/combinations_v2judge/rover/rover_edacc_sub150_no_resolver.json",
    ("english_dialects", "rover_no_auditor"): "results/combinations_v2judge/rover/rover_english_dialects_sub150_no_resolver.json",

    ("commonvoice", "whisperx_baseline"): "results/benchmarks/subsets/whisperx_commonvoice_sub150.json",
    ("edacc",       "whisperx_baseline"): "results/benchmarks/subsets/whisperx_edacc_sub150.json",
    ("english_dialects", "whisperx_baseline"): "results/benchmarks/subsets/whisperx_english_dialects_sub150.json"
}

# confidence-variant sources need a threshold value to build the path —
# these patterns match the patched selector scripts' output filenames
CONFIDENCE_SOURCE_PATTERNS = {
    "context_v1_confidence": "results/combinations/context_v1_confidence/context_v1conf_{dataset}_{selector}_{thresh_str}_sub150.json",
    "context_v2_confidence": "results/combinations/context_v2_confidence/context_v2conf_{dataset}_{selector}_{thresh_str}_sub150.json",
    "naive_confidence":      "results/combinations/naive_confidence/naive_conf_{dataset}_{selector}_{thresh_str}_sub150.json",
}


def resolve_input_path(dataset, source, selector="qwen", threshold=None):
    """
    Resolve the input file path for a given dataset/source combination.
    Confidence-variant sources require a threshold value to build the
    correct filename (e.g. t070 for 0.7, matching the patched selector
    scripts' naming convention).
    """
    if source in CONFIDENCE_SOURCE_PATTERNS:
        if threshold is None:
            raise ValueError(f"--threshold is required for source='{source}'")
        thresh_str = f"t{threshold:.2f}".replace(".", "")
        return CONFIDENCE_SOURCE_PATTERNS[source].format(
            dataset=dataset, selector=selector, thresh_str=thresh_str)

    return SOURCE_FILES.get((dataset, source))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["commonvoice", "edacc", "english_dialects", "shetland"])
    parser.add_argument("--source",  required=True,
                    choices=["baseline", "naive", "context_v1", "context_v2",
                             "context_v1_confidence", "context_v2_confidence", "naive_confidence",
                             "context_v2_whisperx", "rover_auditor", "whisperx_baseline", "rover_no_auditor" ])
    parser.add_argument("--selector", default="qwen",
                        help="Selector model used (for resolving confidence-variant filenames)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold used in the run (required for *_confidence sources, "
                             "e.g. 0.5, 0.7, 0.8)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--subset-only", action="store_true",
                        help="Restrict scoring to the same 150-sample subset used for V2, "
                             "for fair comparison against context_v2 results")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    try:
        input_path = resolve_input_path(args.dataset, args.source,
                                        selector=args.selector, threshold=args.threshold)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    if not input_path or not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    os.makedirs(SEVERITY_OUTPUT_DIR, exist_ok=True)
    suffix = "_sub150" if args.subset_only else ""
    thresh_tag = f"_t{args.threshold:.2f}".replace(".", "") if args.threshold is not None else ""
    output_path = f"{SEVERITY_OUTPUT_DIR}/severity_{args.dataset}_{args.source}{thresh_tag}{suffix}.json"

    client = Client(host=OLLAMA_HOST)
    try:
        available = [m.model for m in client.list().models]
        if not any(JUDGE_MODEL in m for m in available):
            print(f"ERROR: {JUDGE_MODEL} not pulled.")
            return
    except Exception as e:
        print(f"ERROR: could not connect to Ollama\n{e}")
        return

    print(f"Scoring severity for {args.dataset} / {args.source}"
          f"{f' (threshold={args.threshold})' if args.threshold is not None else ''}"
          f"{' (150-subset)' if args.subset_only else ''}...")
    run_severity_scoring(input_path, output_path, client, dataset=args.dataset,
                         subset_only=args.subset_only, max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()