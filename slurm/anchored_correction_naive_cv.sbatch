#!/bin/bash
#SBATCH -J anchored_correction_naive_cv
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --output=logs/anchored_correction_naive_cv_%j.log

# slurm/anchored_correction_naive_cv.sbatch
#
# Finishes the last missing piece of the 9-cell grid: anchored_correction_naive
# on CommonVoice (EdAcc and English Dialects already done). Resumable, so
# safe to rerun if it doesn't finish within the hour.
#
# Usage:
#   sbatch slurm/anchored_correction_naive_cv.sbatch

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

echo "── anchored_correction_naive | gemma4 | commonvoice ──"
python rerunning/grid/anchored_correction_naive.py --dataset commonvoice --split dev --selector gemma4
python rerunning/add_severity_to_existing_concurrent.py \
    --files "writeup_results/grid/anchored_correction_naive/anchored_correction_naive_commonvoice_gemma4_dev.json"

kill $OLLAMA_PID

