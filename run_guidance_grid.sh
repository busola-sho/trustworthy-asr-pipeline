#!/usr/bin/env bash
set -euo pipefail

RULES_FILE="results/eval_suite/selector_rules_dev.txt"
DATASETS=(commonvoice edacc english_dialects)
SPLITS=(dev test)

if [[ ! -s "$RULES_FILE" ]]; then
  echo "Missing frozen V2 rules: $RULES_FILE" >&2
  exit 1
fi

for split in "${SPLITS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    python rerunning/grid/selection_context_v1.py \
      --dataset "$dataset" --split "$split" --selector gemma4 --rerun
    python rerunning/grid/selection_context_v2.py \
      --dataset "$dataset" --split "$split" --selector gemma4 \
      --rules-file "$RULES_FILE" --rerun

    python rerunning/grid/unanchored_fusion_context_v1.py \
      --dataset "$dataset" --split "$split" --selector gemma4 --rerun
    python rerunning/grid/unanchored_fusion_context_v2.py \
      --dataset "$dataset" --split "$split" --selector gemma4 \
      --rules-file "$RULES_FILE" --rerun

    python rerunning/grid/anchored_correction_context_v1.py \
      --dataset "$dataset" --split "$split" --selector gemma4 --rerun
    python rerunning/grid/anchored_correction_context_v2.py \
      --dataset "$dataset" --split "$split" --selector gemma4 \
      --rules-file "$RULES_FILE" --rerun
  done
done

echo "Guidance grid complete: writeup_results/clean_grid_guidance_rerun"
