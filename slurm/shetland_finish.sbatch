#!/bin/bash
#SBATCH -J shetland_finish
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -p ampere
#SBATCH --output=logs/shetland_finish_%j.log

# slurm/shetland_finish.sbatch
#
# Runs everything remaining for Shetland in one job:
#   1. Severity-scores all 4 baseline model files (qwen3asr, whisperx,
#      parakeet, wav2vec2) - assumes these already exist at
#      results/benchmarks/shetland/ (rsync'd from local, WhisperX's
#      already regenerated and moved into place).
#   2. Runs naive.py on shetland (--split full - Shetland has no
#      dev/test concept, it's used in its entirety, exactly once).
#   3. Severity-scores naive's shetland output.
#
# ASSUMES: results/benchmarks/shetland/ already has all 4 baseline files
# rsync'd over from local (see the rsync command that goes with this) -
# no raw audio needed for this job, since the WhisperX transcription
# step (the only one that needed raw audio) is already done.
#
# Usage:
#   sbatch slurm/shetland_finish.sbatch

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
export OLLAMA_MAX_LOADED_MODELS=1
export ENSEMBLE_MAX_WORKERS=8

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
sleep 5

echo "Pulling required models (gemma4 selector, phi4 judge)..."
ollama pull gemma4:12b
ollama pull phi4:14b

echo ""
echo "── Step 1: severity-score all 4 baseline models on Shetland ──"
python rerunning/add_severity_to_existing_concurrent.py \
  --files results/benchmarks/shetland/shetland_qwen3asr_20260603_150124.json \
          results/benchmarks/shetland/shetland_whisper_20260603_123115.json \
          results/benchmarks/shetland/shetland_parakeet_20260606_134131.json \
          results/benchmarks/shetland/shetland_wav2vec2_20260606_134507.json

echo ""
echo "── Step 2: run naive on Shetland (full) ──"
python rerunning/ensembles/naive.py --dataset shetland --split full --selector gemma4

echo ""
echo "── Step 3: severity-score naive's Shetland output ──"
python rerunning/add_severity_to_existing_concurrent.py \
  --files writeup_results/ensembles/naive/naive_shetland_gemma4sel_full.json

echo ""
echo "All Shetland steps done: 4 baselines + naive, all severity-scored."

kill $OLLAMA_PID
