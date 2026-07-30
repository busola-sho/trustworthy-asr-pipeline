#!/bin/bash
#SBATCH -J finetune_whisper_medium_lora_scots
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH -p ampere
#SBATCH --output=logs/finetune_whisper_medium_lora_scots_%j.log

# slurm/finetune_whisper_medium.sbatch
#
# Tests whether more base-model capacity (whisper-medium instead of
# whisper-small) helps, given whisper-small's r32 run plateaued early
# with train loss already near-zero by epoch 2 - suggesting small's own
# capacity may be the ceiling, not something LoRA rank/epochs can fix.
# Same lora_r/alpha/lr/epochs as the small-model lr5e4 run, so the base
# model is the one isolated variable versus that job.
#
# Time bumped to 12h (vs 10h for whisper-small) - whisper-medium is a
# larger model, expect somewhat slower per-step training and eval.
#
# NOTE on --gres=gpu:1: single-GPU HF Trainer run, no multi-GPU data
# parallelism without accelerate/torchrun - don't request more GPUs
# without wrapping the script accordingly.
#
# ASSUMES: data/finetune/ (audio + manifests) already present on this
# cluster - this job does not regenerate the manifest.
#
# IMPORTANT: does NOT pass --evaluate_test - val WER only. Only add
# --evaluate_test once a final configuration is chosen across all
# sweep runs, as a separate final job.
#
# Usage:
#   sbatch slurm/finetune_whisper_medium.sbatch

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_finetune/bin/activate
export HF_HOME=/rds/user/ors36/hpc-work/hf_cache
export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python scripts/finetuning/finetune_whisper_lora.py \
    --output_dir checkpoints/whisper_medium_lora_scots_r32_lr5e4 \
    --base_model openai/whisper-medium \
    --lora_r 32 \
    --lora_alpha 64 \
    --epochs 4 \
    --lr 5e-4 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2

