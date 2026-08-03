#!/bin/bash
#SBATCH -J grid_remaining_dev
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH -p ampere
#SBATCH --output=logs/grid_remaining_dev_%j.log

# slurm/grid_remaining_dev.sbatch
#
# Runs the remaining 3 new grid cells (unanchored_fusion_context_v1,
# unanchored_fusion_context_v2, anchored_correction_naive) across
# commonvoice, edacc, english_dialects (dev split), gemma4 selector.
# Completes the 9-cell strategy x context grid alongside
# selection_grid_dev.sbatch (Selection strategy) and the pre-existing
# naive.py/context_v1.py/context_v2.py results.
#
# Usage:
#   sbatch slurm/grid_remaining_dev.sbatch

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

    echo "── unanchored_fusion_context_v1 | gemma4 | ${d} ──"
    python rerunning/grid/unanchored_fusion_context_v1.py --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/unanchored_fusion_context_v1/unanchored_fusion_context_v1_${d}_gemma4_dev.json"

    echo "── unanchored_fusion_context_v2 | gemma4 | ${d} ──"
    python rerunning/grid/unanchored_fusion_context_v2.py --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/unanchored_fusion_context_v2/unanchored_fusion_context_v2_${d}_gemma4_dev.json"

    echo "── anchored_correction_naive | gemma4 | ${d} ──"
    python rerunning/grid/anchored_correction_naive.py --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/anchored_correction_naive/anchored_correction_naive_${d}_gemma4_dev.json"

done

kill $OLLAMA_PID

