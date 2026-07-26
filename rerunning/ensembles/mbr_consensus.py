"""
rerunning/ensembles/mbr_consensus.py

MBR (Minimum Bayes-Risk) consensus selection - a classical, well-established
ensembling family distinct from LLM-based fusion (Goel & Byrne, 2000,
"Minimum Bayes-Risk Automatic Speech Recognition", Computer Speech &
Language). No LLM involved at all - this is a purely mechanical baseline.

THE IDEA: without a true reference, MBR approximates "the most likely
correct transcript" as whichever candidate has the LOWEST average
distance (here: WER) to the OTHER candidates - the intuition being that
if multiple independent ASR systems roughly agree, the one closest to
the "consensus" is more likely correct than an outlier, even without
knowing which one is actually right. This is a standard MBR
approximation when you have a small fixed candidate set (here: 4 ASR
outputs) rather than a full n-best list with model probabilities.

For each sample, and for each of the 4 candidate hypotheses h_i:
    risk(h_i) = mean over j != i of WER(reference=h_j, hypothesis=h_i)
Pick whichever h_i has the LOWEST risk - i.e. is "closest" on average to
the other three. No editing, no combining - like whole_transcript_selection,
this picks ONE candidate verbatim, but via a purely mechanical
risk-minimization rule instead of an LLM's judgement.

WHY THIS IS A DIFFERENT FAMILY FROM YOUR OTHER TECHNIQUES:
  - ROVER: aligns all 4 word-by-word, votes at each position (can produce
    a novel sequence not equal to any single input)
  - Naive/context_v1/context_v2: LLM reads all 4, edits/combines
  - whole_transcript_selection: LLM picks one of the 4 unchanged
  - MBR consensus (this script): picks one of the 4 unchanged, via a
    purely mechanical pairwise-distance rule, no LLM at all

No Ollama/GPU needed for this script - it's pure CPU, WER computation
only, so it also runs very fast (no server startup, no concurrency
needed at all).

Usage:
    python rerunning/ensembles/mbr_consensus.py --dataset commonvoice --split dev
    python rerunning/ensembles/mbr_consensus.py --dataset shetland --split full
"""

import json
import os
import argparse
from jiwer import wer

from src.judge import normalise, is_tag_only
from src.selector import find_canonical_file, load_samples
from src.splits import get_indices_for_split

NEW_OUTPUT_DIR = "writeup_results/ensembles/mbr_consensus"
OLD_OUTPUT_DIR = "results/combinations_v2judge/mbr_consensus"
ASR_MODELS     = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def mbr_select(hyp_by_model: dict) -> tuple:
    """
    Returns (chosen_model, chosen_hyp, risk_by_model) - the model whose
    hypothesis has the lowest average WER distance to the other three
    candidates' hypotheses, treating each of the OTHERS in turn as a
    pseudo-reference (WER is asymmetric, normalised by reference length,
    so this direction convention is applied consistently for every
    sample and every candidate).
    """
    models = list(hyp_by_model.keys())
    normalised = {m: normalise(hyp_by_model[m]) for m in models}

    risk_by_model = {}
    for m in models:
        others = [normalised[o] for o in models if o != m]
        distances = [wer(other_norm, normalised[m]) for other_norm in others]
        risk_by_model[m] = sum(distances) / len(distances)

    chosen_model = min(risk_by_model, key=risk_by_model.get)
    return chosen_model, hyp_by_model[chosen_model], risk_by_model


def run_dataset(dataset, max_samples=None, rerun=False, split="dev"):
    print(f"\n── {dataset} | mbr_consensus (PHASE 1: mechanical selection, no LLM) split={split} ──")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"mbr_{dataset}_{split}.json"
    new_output_path = os.path.join(NEW_OUTPUT_DIR, filename)
    old_output_path = os.path.join(OLD_OUTPUT_DIR, filename)

    if os.path.exists(old_output_path) and not rerun:
        with open(old_output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results    = []
        start_from = 0

    def save_progress():
        payload = {"progress": len(results), "samples": results}
        with open(new_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(old_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    pick_distribution = {m: 0 for m in ASR_MODELS}

    for pos in range(start_from, len(indices)):
        idx = indices[pos]
        samples_by_model = {m: model_samples[m].get(idx) for m in ASR_MODELS}

        if not all(samples_by_model.values()):
            results.append({"ref": None, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "missing sample from one or more models",
                            "dataset_index": idx})
            continue

        ref = samples_by_model["qwen"]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "ignore_time_segment", "dataset_index": idx})
            continue

        if is_tag_only(ref):
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "tag_only_reference", "dataset_index": idx})
            continue

        hyp_by_model = {m: samples_by_model[m]["hyp"] for m in ASR_MODELS}
        chosen_model, chosen_hyp, risk_by_model = mbr_select(hyp_by_model)
        pick_distribution[chosen_model] += 1

        sample_wer_val = wer(normalise(ref), normalise(chosen_hyp))

        results.append({
            "ref":            ref,
            "hyp":            chosen_hyp,
            "source_hyps":    hyp_by_model,
            "chosen_source":  chosen_model,
            "risk_by_model":  risk_by_model,
            "sample_WER":     sample_wer_val,
            "severity":       None,
            "dataset_index":  idx,
        })

        if (pos + 1) % 50 == 0:
            save_progress()
            print(f"  {pos+1}/{len(indices)} done")

    save_progress()

    valid = [r for r in results if not r.get("skipped") and not r.get("error")
             and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None

    output = {
        "approach":          "mbr_consensus",
        "asr_models":        ASR_MODELS,
        "phase":             "selection_only - severity not yet judged",
        "dataset":           dataset,
        "split":             split,
        "subset_indices":    indices,
        "corpus_wer":        corpus_wer,
        "num_samples":       len(valid),
        "pick_distribution": pick_distribution,
        "samples":           results,
    }

    with open(new_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(old_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Pick distribution: {pick_distribution}")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2 to add severity scores:")
    print(f"  python rerunning/add_severity_to_existing_concurrent.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} split={args.split}")
        return

    run_dataset(args.dataset, max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()