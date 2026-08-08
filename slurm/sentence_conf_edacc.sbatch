#!/bin/bash
#SBATCH -J sentence_conf_english_dialects
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -p ampere
#SBATCH --output=logs/sentence_conf_edacc_%j.log

# slurm/sentence_conf_english_dialects.sbatch
#
# Runs sentence-level segmentation + Phi-4 judging (segment_and_judge_
# sentences.py) on English Dialects, both dev and test splits. This is
# your LARGEST dataset (1780 dev + 744 test = 2524 transcripts), and
# since judging happens per-SEGMENT (not per-transcript, ~7-10
# segments/transcript typically), this is a genuinely large job -
# potentially 20,000+ individual Phi-4 calls across both splits, even
# with concurrency. 12h budget accordingly - do not assume this is a
# quick job just because other per-transcript jobs tonight were fast.
#
# Chunked + resumable (chunk_size=25 transcripts per save) - if this
# needs a second submission, it picks up from the last completed chunk.
#
# Usage:
#   sbatch slurm/sentence_conf_english_dialects.sbatch

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

pip install wtpsplit --break-system-packages

echo "── edacc | dev ──"
python rerunning/sentence_confidence/segment_and_judge_sentences.py \
    --dataset edacc --split dev --rerun

echo "── edaccs | test ──"
python rerunning/sentence_confidence/segment_and_judge_sentences.py \
    --dataset edacc --split test --rerun

kill $OLLAMA_PID

