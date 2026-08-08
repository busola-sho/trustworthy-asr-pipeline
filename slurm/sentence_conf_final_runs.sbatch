#!/bin/bash
#SBATCH -J sentence_conf_final
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH -p ampere
#SBATCH --output=logs/sentence_conf_final_%j.log

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
ollama pull phi4:14b

echo "── english_dialects | dev (resume) ──"
python rerunning/sentence_confidence/segment_and_judge_sentences.py \
    --dataset english_dialects --split dev

echo "── shetland | full (resume) ──"
python rerunning/sentence_confidence/segment_and_judge_sentences.py \
    --dataset shetland --split full

echo "── edacc | test (rerun) ──"
python rerunning/sentence_confidence/segment_and_judge_sentences.py \
    --dataset edacc --split test --rerun

kill $OLLAMA_PID
