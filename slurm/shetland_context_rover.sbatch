#!/bin/bash
#SBATCH -J shetland_context_rover
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --output=logs/shetland_context_rover_%j.log

set -e
mkdir -p logs

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
ollama pull phi4:14b

echo "── context_v1 (shetland, full) ──"
python rerunning/ensembles/context_v1.py --dataset shetland --split full --selector gemma4
python rerunning/add_severity_to_existing_concurrent.py \
    --files writeup_results/ensembles/context_v1/context_shetland_gemma4_full.json

echo "── rover (shetland, full) ──"
python rerunning/voting/rover.py --dataset shetland --split full
python rerunning/add_severity_to_existing_concurrent.py \
    --files writeup_results/voting/rover/rover_shetland_full.json

echo "Both done."
kill $OLLAMA_PID

