#!/bin/bash
#SBATCH -J sentence_conf_stale_rerun
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH -p ampere
#SBATCH --output=logs/sentence_conf_stale_rerun_%j.log

# slurm/sentence_conf_stale_rerun.sbatch
#
# Reruns the two sentence-confidence segment+judge files confirmed
# stale (predating the two-stage architecture / single-segment fix):
#   - edacc dev: 135/221 segments alignment_failed (61%)
#   - english_dialects dev: 1864/1864 segments alignment_failed (100%)
# Both --rerun (force fresh, ignore existing stale data).
#
# 35 min budget - EdAcc dev is small (131 transcripts) and should
# finish quickly; English Dialects dev is your largest dataset (1734
# transcripts) and may not fully complete in this window. The
# pipeline is chunked/resumable (saves every 25 transcripts), so if
# it times out, just resubmit this same script WITHOUT --rerun on the
# second run (or resubmit as-is - --rerun on an incomplete file still
# safely restarts, just redoes any partial progress from this attempt
# rather than resuming it).
#
# Usage:
#   sbatch slurm/sentence_conf_stale_rerun.sbatch

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

echo "── edacc | dev (rerun, was stale) ──"
python rerunning/sentence_confidence/segment_and_judge_sentences.py \
    --dataset edacc --split dev --rerun

echo "── english_dialects | dev (rerun, was stale) ──"
python rerunning/sentence_confidence/segment_and_judge_sentences.py \
    --dataset english_dialects --split dev --rerun

kill $OLLAMA_PID

