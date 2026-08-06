#!/bin/bash
#SBATCH -J whisper_ft_chunked_transcribe
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -p ampere
#SBATCH --output=logs/whisper_ft_chunked_transcribe_%j.log

# slurm/whisper_ft_chunked_transcribe.sbatch
#
# Runs the chunked fine-tuned Whisper inference (28s chunks, matching
# training conditions - see transcribe_finetuned_whisper_chunked.py's
# docstring for why) on CommonVoice and EdAcc. Shetland + English
# Dialects run in parallel via a separate interactive sintr session,
# to avoid two processes writing to the same output file.
#
# Uses .venv_finetune (transformers/peft), NOT .venv_ampere - this is
# direct HuggingFace/PEFT model inference, no Ollama involved at all.
#
# Usage:
#   sbatch slurm/whisper_ft_chunked_transcribe.sbatch

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_finetune/bin/activate
export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline
export HF_HOME=/rds/user/ors36/hpc-work/hf_cache

python scripts/transcribe_finetuned_whisper_chunked.py --datasets commonvoice edacc

