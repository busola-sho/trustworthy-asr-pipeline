#!/bin/bash
#SBATCH -J sentence_conf_edacc_dev_methods
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --output=logs/sentence_conf_edacc_dev_methods_%j.log

# slurm/sentence_conf_edacc_dev_methods.sbatch
#
# Runs all 5 sentence-confidence methods for EdAcc dev (131
# transcripts - your smallest dev split, should comfortably finish
# within this 1h budget, unlike the English Dialects dev job): 3
# verbalized variants (confscore/confprobscore/probscore, each an LLM
# call per segment via gemma4), plus model_internal and
# cross_model_agreement (both pure arithmetic, no LLM, fast).
#
# Usage:
#   sbatch slurm/sentence_conf_edacc_dev_methods.sbatch

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

echo "── edacc | dev | verbalized confscore ──"
python rerunning/sentence_confidence/verbalized_confidence.py \
    --dataset edacc --split dev --variant confscore

echo "── edacc | dev | verbalized confprobscore ──"
python rerunning/sentence_confidence/verbalized_confidence.py \
    --dataset edacc --split dev --variant confprobscore

echo "── edacc | dev | verbalized probscore ──"
python rerunning/sentence_confidence/verbalized_confidence.py \
    --dataset edacc --split dev --variant probscore

echo "── edacc | dev | model_internal_confidence ──"
python rerunning/sentence_confidence/model_internal_confidence.py \
    --dataset edacc --split dev

echo "── edacc | dev | cross_model_agreement ──"
python rerunning/sentence_confidence/cross_model_agreement.py \
    --dataset edacc --split dev

kill $OLLAMA_PID

