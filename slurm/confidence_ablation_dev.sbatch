#!/bin/bash
#SBATCH -J confidence_ablation_dev
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -p ampere
#SBATCH --output=logs/confidence_ablation_dev_%j.log

# slurm/confidence_ablation_dev.sbatch
#
# Runs all 4 new confidence-condition scripts (list/inline x
# thresholded-p20/unfiltered) against the locked winning strategy
# (Unanchored Fusion + Naive), across the 3 dev datasets. The
# thresholded scripts use --threshold-split dev (frozen calibration,
# same value regardless of which split is evaluated). Baseline
# (naive.py, no confidence) already exists - not rerun here.
#
# Usage:
#   sbatch slurm/confidence_ablation_dev.sbatch

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_ampere/bin/activate
export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline
export HF_HOME=/rds/user/ors36/hpc-work/hf_cache

export PATH=/rds/user/ors36/hpc-work/ollama_install/bin:$PATH
export LD_LIBRARY_PATH=/rds/user/ors36/hpc-work/ollama_install/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=/rds/user/ors36/hpc-work/ollama_models
mkdir -p "$OLLAMA_MODELS"

export OLLAMA_NUM_PARALLEL=8
export ENSEMBLE_MAX_WORKERS=8

ollama serve &
OLLAMA_PID=$!
sleep 5
ollama pull gemma4:12b

DATASETS=("commonvoice" "edacc" "english_dialects")

for d in "${DATASETS[@]}"; do

    echo "── list, thresholded p20 | gemma4 | ${d} ──"
    python rerunning/grid/unanchored_fusion_naive_confidence_list.py \
        --dataset "$d" --split dev --selector gemma4 --percentile 20
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/unanchored_fusion_naive_confidence_list/unanchored_fusion_naive_confidence_list_${d}_gemma4_p20_thr-dev_dev.json"

    echo "── inline, thresholded p20 | gemma4 | ${d} ──"
    python rerunning/grid/unanchored_fusion_naive_confidence_inline.py \
        --dataset "$d" --split dev --selector gemma4 --percentile 20
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/unanchored_fusion_naive_confidence_inline/unanchored_fusion_naive_confidence_inline_${d}_gemma4_p20_thr-dev_dev.json"

    echo "── list, unfiltered | gemma4 | ${d} ──"
    python rerunning/grid/unanchored_fusion_naive_confidence_list_full.py \
        --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/unanchored_fusion_naive_confidence_list_full/unanchored_fusion_naive_confidence_list_full_${d}_gemma4_dev.json"

    echo "── inline, unfiltered | gemma4 | ${d} ──"
    python rerunning/grid/unanchored_fusion_naive_confidence_inline_full.py \
        --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/unanchored_fusion_naive_confidence_inline_full/unanchored_fusion_naive_confidence_inline_full_${d}_gemma4_dev.json"

done

kill $OLLAMA_PID

