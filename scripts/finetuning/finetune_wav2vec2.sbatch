#!/bin/bash
#SBATCH -J finetune_wav2vec2_scots
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -p ampere
#SBATCH --output=logs/finetune_wav2vec2_scots_%j.log

# slurm/finetune_wav2vec2.sbatch
#
# Same single-GPU reasoning as finetune_whisper.sbatch - HF Trainer here
# is also single-GPU unless wrapped with accelerate/torchrun. wav2vec2
# CTC training typically needs more epochs than Whisper LoRA to converge
# (30 vs 10), hence the longer --time ceiling - adjust down once you've
# seen real per-epoch timing from your first run.
#
# ASSUMES: data/finetune/ already rsync'd onto this cluster (see
# finetune_whisper.sbatch's note - same applies here).
#
# The vocab-building step only needs to run ONCE (it's fast, CPU-only,
# builds a fixed tokenizer vocab from your train manifest's transcripts)
# - comment it out on subsequent submissions if data/finetune/wav2vec2_vocab.json
# already exists and hasn't changed.
#
# Usage:
#   sbatch slurm/finetune_wav2vec2.sbatch

module purge
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_finetune/bin/activate

export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python scripts/finetuning/build_wav2vec2_vocab.py

python scripts/finetuning/finetune_wav2vec2.py \
    --output_dir checkpoints/wav2vec2_scots \
    --epochs 30 \
    --lr 3e-4 \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 4
