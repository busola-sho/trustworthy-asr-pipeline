#!/bin/bash
#SBATCH -J sentence_conf_commonvoice_dev_finish
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -p ampere
#SBATCH --output=logs/sentence_conf_commonvoice_dev_finish_%j.log

# slurm/sentence_conf_commonvoice_dev_finish.sbatch
#
# Finishes CommonVoice dev's remaining sentence-confidence methods:
# confprobscore resumes from 350/456 (no --rerun), probscore/
# model_internal/crossmodel_agreement start fresh. confscore already
# complete (456/456), not re-run here.
#
# Usage:
#   sbatch slurm/sentence_conf_commonvoice_dev_finish.sbatch

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

echo "── commonvoice | dev | verbalized confprobscore (resume) ──"
python rerunning/sentence_confidence/verbalized_confidence.py \
    --dataset commonvoice --split dev --variant confprobscore

echo "── commonvoice | dev | verbalized probscore ──"
python rerunning/sentence_confidence/verbalized_confidence.py \
    --dataset commonvoice --split dev --variant probscore

echo "── commonvoice | dev | model_internal_confidence ──"
python rerunning/sentence_confidence/model_internal_confidence.py \
    --dataset commonvoice --split dev

echo "── commonvoice | dev | cross_model_agreement ──"
python rerunning/sentence_confidence/cross_model_agreement.py \
    --dataset commonvoice --split dev

kill $OLLAMA_PID

