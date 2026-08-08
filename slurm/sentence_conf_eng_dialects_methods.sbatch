#!/bin/bash
#SBATCH -J sentence_conf_eng_dialects_methods
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH -p ampere
#SBATCH --output=logs/sentence_conf_eng_dialects_methods_%j.log

# slurm/sentence_conf_eng_dialects_methods.sbatch
#
# Runs all 5 sentence-confidence methods for English Dialects test
# (744 transcripts - your largest dataset): 3 verbalized variants
# (confscore/confprobscore/probscore, each needs an LLM call per
# segment via gemma4), plus model_internal and cross_model_agreement
# (both pure arithmetic, no LLM, fast regardless of dataset size).
#
# HONEST NOTE: given the scale (744 transcripts x 3 LLM-based variants),
# 40 minutes may not be enough to finish all 5 runs. Each script is
# independently resumable (verbalized_confidence.py is chunked/saves
# incrementally) - if this times out partway, just resubmit and it
# picks up from wherever it stopped, no --rerun needed unless a run
# genuinely needs to restart from scratch.
#
# Usage:
#   sbatch slurm/sentence_conf_eng_dialects_methods.sbatch

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

echo "── english_dialects | test | verbalized confscore ──"
python rerunning/sentence_confidence/verbalized_confidence.py \
    --dataset english_dialects --split test --variant confscore

echo "── english_dialects | test | verbalized confprobscore ──"
python rerunning/sentence_confidence/verbalized_confidence.py \
    --dataset english_dialects --split test --variant confprobscore

echo "── english_dialects | test | verbalized probscore ──"
python rerunning/sentence_confidence/verbalized_confidence.py \
    --dataset english_dialects --split test --variant probscore

echo "── english_dialects | test | model_internal_confidence ──"
python rerunning/sentence_confidence/model_internal_confidence.py \
    --dataset english_dialects --split test

echo "── english_dialects | test | cross_model_agreement ──"
python rerunning/sentence_confidence/cross_model_agreement.py \
    --dataset english_dialects --split test

kill $OLLAMA_PID

