#!/bin/bash
#SBATCH -J selection_grid_dev
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -p ampere
#SBATCH --output=logs/selection_grid_dev_%j.log

# slurm/selection_grid_dev.sbatch
#
# Runs all 3 Selection-strategy grid cells (naive, context_v1,
# context_v2 context conditions) across commonvoice, edacc, and
# english_dialects (dev split), gemma4 selector. Part of the 9-cell
# strategy x context grid - writes to writeup_results/grid/, fully
# separate from all prior ensemble results.
#
# Usage:
#   sbatch slurm/selection_grid_dev.sbatch

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

    echo "── selection_naive | gemma4 | ${d} ──"
    python rerunning/grid/selection_naive.py --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/selection_naive/selection_naive_${d}_gemma4_dev.json"

    echo "── selection_context_v1 | gemma4 | ${d} ──"
    python rerunning/grid/selection_context_v1.py --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/selection_context_v1/selection_context_v1_${d}_gemma4_dev.json"

    echo "── selection_context_v2 | gemma4 | ${d} ──"
    python rerunning/grid/selection_context_v2.py --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/selection_context_v2/selection_context_v2_${d}_gemma4_dev.json"

done

kill $OLLAMA_PID

