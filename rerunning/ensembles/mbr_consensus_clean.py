"""
Rerun the MBR-consensus baseline on exactly the samples and ASR hypotheses
used by the clean strategy grid.

The clean-grid Unanchored Fusion (naive) result is used only as a manifest:
for each sample, this script reads its reference, dataset_index, and four
source_hyps, then performs purely mechanical MBR selection. It never uses the
fusion output itself.

Outputs are written to a new directory so older MBR results are preserved.
Severity is intentionally left unset; run the existing severity-scoring script
after this selection stage finishes.

Examples:
    python rerun_clean_mbr_consensus.py --dataset commonvoice --split dev
    python rerun_clean_mbr_consensus.py --dataset edacc --split test
    python rerun_clean_mbr_consensus.py --all
"""

import argparse
import json
import os
from pathlib import Path

from jiwer import wer

from src.judge import normalise


ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "edacc", "english_dialects"]
SPLITS = ["dev", "test"]

CLEAN_GRID_DIR = Path("writeup_results/clean_grid/unanchored_fusion_naive")
OUTPUT_DIR = Path("writeup_results/clean_mbr_consensus")


def clean_grid_path(dataset: str, split: str) -> Path:
    return CLEAN_GRID_DIR / (
        f"unanchored_fusion_naive_{dataset}_gemma4_{split}.json"
    )


def output_path(dataset: str, split: str) -> Path:
    return OUTPUT_DIR / f"mbr_{dataset}_{split}.json"


def mbr_select(hyp_by_model: dict[str, str]) -> tuple[str, str, dict[str, float]]:
    """Select the candidate with the lowest mean WER to the other candidates."""
    normalised = {
        model: normalise(hyp_by_model[model])
        for model in ASR_MODELS
    }

    risk_by_model = {}
    for model in ASR_MODELS:
        distances = [
            wer(normalised[other], normalised[model])
            for other in ASR_MODELS
            if other != model
        ]
        risk_by_model[model] = sum(distances) / len(distances)

    # ASR_MODELS order provides a deterministic tie-break.
    chosen_model = min(ASR_MODELS, key=lambda model: risk_by_model[model])
    return chosen_model, hyp_by_model[chosen_model], risk_by_model


def load_clean_manifest(dataset: str, split: str) -> tuple[Path, dict]:
    path = clean_grid_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"Clean-grid input not found: {path}")

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("dataset") != dataset:
        raise ValueError(
            f"Dataset mismatch in {path}: expected {dataset!r}, "
            f"found {data.get('dataset')!r}"
        )
    if data.get("split") != split:
        raise ValueError(
            f"Split mismatch in {path}: expected {split!r}, "
            f"found {data.get('split')!r}"
        )
    if not isinstance(data.get("samples"), list):
        raise ValueError(f"No samples list found in {path}")

    return path, data


def run_dataset(dataset: str, split: str, overwrite: bool = False) -> None:
    manifest_path, manifest = load_clean_manifest(dataset, split)
    destination = output_path(dataset, split)

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}\n"
            "Use --overwrite only if you intentionally want to replace it."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    pick_distribution = {model: 0 for model in ASR_MODELS}

    for position, sample in enumerate(manifest["samples"], start=1):
        dataset_index = sample.get("dataset_index")
        ref = sample.get("ref")

        # Mirror samples the clean-grid run itself could not score.
        if sample.get("skipped") or sample.get("error"):
            mirrored = {
                "ref": ref,
                "hyp": None,
                "severity": None,
                "sample_WER": None,
                "dataset_index": dataset_index,
            }
            if sample.get("skipped"):
                mirrored["skipped"] = True
                mirrored["skip_reason"] = sample.get("skip_reason", "clean_grid_skipped")
            if sample.get("error"):
                mirrored["error"] = True
                mirrored["error_reason"] = sample.get("error_reason", "clean_grid_error")
            results.append(mirrored)
            continue

        source_hyps = sample.get("source_hyps")
        if not isinstance(source_hyps, dict):
            raise ValueError(
                f"Sample {dataset_index!r} in {manifest_path} has no source_hyps"
            )

        missing_models = [
            model
            for model in ASR_MODELS
            if model not in source_hyps or source_hyps[model] is None
        ]
        if missing_models:
            raise ValueError(
                f"Sample {dataset_index!r} in {manifest_path} is missing "
                f"hypotheses for: {', '.join(missing_models)}"
            )
        if ref is None:
            raise ValueError(
                f"Sample {dataset_index!r} in {manifest_path} has no reference"
            )

        hyp_by_model = {model: source_hyps[model] for model in ASR_MODELS}
        chosen_model, chosen_hyp, risk_by_model = mbr_select(hyp_by_model)
        pick_distribution[chosen_model] += 1

        results.append(
            {
                "ref": ref,
                "hyp": chosen_hyp,
                "source_hyps": hyp_by_model,
                "chosen_source": chosen_model,
                "risk_by_model": risk_by_model,
                "sample_WER": wer(normalise(ref), normalise(chosen_hyp)),
                "severity": None,
                "dataset_index": dataset_index,
            }
        )

        if position % 250 == 0:
            print(f"  {position}/{len(manifest['samples'])} processed")

    valid = [
        result
        for result in results
        if not result.get("skipped")
        and not result.get("error")
        and result.get("sample_WER") is not None
    ]

    corpus_wer = (
        wer(
            [normalise(result["ref"]) for result in valid],
            [normalise(result["hyp"]) for result in valid],
        )
        if valid
        else None
    )

    payload = {
        "approach": "mbr_consensus",
        "asr_models": ASR_MODELS,
        "phase": "selection_only - severity not yet judged",
        "input_manifest": str(manifest_path),
        "dataset": dataset,
        "split": split,
        "subset_indices": [sample.get("dataset_index") for sample in manifest["samples"]],
        "corpus_wer": corpus_wer,
        "num_samples": len(valid),
        "pick_distribution": pick_distribution,
        "samples": results,
    }

    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, destination)

    wer_text = f"{corpus_wer * 100:.2f}%" if corpus_wer is not None else "—"
    print(f"\n{dataset} / {split}")
    print(f"  Input:  {manifest_path}")
    print(f"  Output: {destination}")
    print(f"  N:      {len(valid)}")
    print(f"  WER:    {wer_text}")
    print(f"  Picks:  {pick_distribution}")
    print("  Severity scores: pending")
    print(
        "  Next: python rerunning/add_severity_to_existing_concurrent.py "
        f"--files {destination}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three datasets on both dev and test splits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing file in the new clean-MBR output directory.",
    )
    args = parser.parse_args()

    if args.all:
        if args.dataset or args.split:
            parser.error("Use either --all or --dataset/--split, not both.")
        jobs = [(dataset, split) for dataset in DATASETS for split in SPLITS]
    else:
        if not args.dataset or not args.split:
            parser.error("Provide both --dataset and --split, or use --all.")
        jobs = [(args.dataset, args.split)]

    for dataset, split in jobs:
        run_dataset(dataset, split, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

