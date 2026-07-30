#!/bin/bash
#SBATCH -J finetune_whisper_lora_scots_lr5e4
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:30:00
#SBATCH -p ampere
#SBATCH --output=logs/finetune_whisper_lora_scots_lr5e4_%j.log

# slurm/finetune_whisper_lr5e4.sbatch
#
# Lower LR + fewer epochs variant, isolating whether r32's early
# convergence (val WER flat from epoch 2 onward, train loss already
# near-zero by then) was caused by too-aggressive a learning rate.
# Same lora_r/lora_alpha as the original r32 run - only --epochs and
# --lr change here.
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
#   sbatch slurm/finetune_whisper_lr5e4.sbatch

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_finetune/bin/activate
export HF_HOME=/rds/user/ors36/hpc-work/hf_cache
export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python scripts/finetuning/finetune_whisper_lora.py \
    --output_dir checkpoints/whisper_small_lora_scots_r32_lr5e4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --epochs 4 \
    --lr 5e-4 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2

