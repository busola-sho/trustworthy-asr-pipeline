#!/bin/bash
#SBATCH -J finetune_whisper_medium_r8
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -p ampere
#SBATCH --output=logs/finetune_whisper_medium_r8_%j.log

# slurm/finetune_whisper_medium_r8.sbatch
#
# Tests a lower LoRA rank (r8, alpha16 - same 2:1 alpha:r ratio as the
# r32/alpha64 winner) against the current best config (whisper-medium,
# lr=5e-4, 4 epochs, r32: val WER 11.87%). Isolates rank as the ONLY
# variable vs that run.
#
# Motivation: r32's training loss was already near-zero well before
# convergence in earlier runs, suggesting more capacity than the ~8.3h
# training set needs - a smaller adapter may generalize just as well
# (or better, if r32 was mildly overfitting) with far fewer trainable
# parameters.
#
# val WER ONLY - test manifest untouched, matching the sweep discipline.
#
# NOTE on --gres=gpu:1: single-GPU HF Trainer run, no multi-GPU data
# parallelism without accelerate/torchrun.
#
# ASSUMES: data/finetune/ (audio + manifests) already present on this
# cluster.
#
# Usage:
#   sbatch slurm/finetune_whisper_medium_r8.sbatch

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_finetune/bin/activate
export HF_HOME=/rds/user/ors36/hpc-work/hf_cache
export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python scripts/finetuning/finetune_whisper_lora.py \
    --output_dir checkpoints/whisper_medium_lora_scots_r8_lr5e4 \
    --base_model openai/whisper-medium \
    --lora_r 8 \
    --lora_alpha 16 \
    --epochs 4 \
    --lr 5e-4 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2

