#!/bin/bash
#SBATCH -J finetune_whisper_medium_final
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -p ampere
#SBATCH --output=logs/finetune_whisper_medium_final_%j.log

# slurm/finetune_whisper_medium_final.sbatch
#
# FINAL run for the winning config (whisper-medium, r32, lr=5e-4,
# 4 epochs) - val WER 11.87%, the best of the 3 candidates tested
# (whisper-small r32/lr=1e-3/10ep: 14.79%, whisper-small r32/lr=5e-4/4ep:
# 14.85%, this one: 11.87%). This is the ONLY run that touches the test
# manifest - separate output_dir from the sweep run, so the sweep
# checkpoint (val-only) and this final test-evaluated one stay distinct.
#
# NOTE on --gres=gpu:1: single-GPU HF Trainer run, no multi-GPU data
# parallelism without accelerate/torchrun.
#
# ASSUMES: data/finetune/ (audio + manifests) already present on this
# cluster.
#
# Usage:
#   sbatch slurm/finetune_whisper_medium_final.sbatch

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_finetune/bin/activate
export HF_HOME=/rds/user/ors36/hpc-work/hf_cache
export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python scripts/finetuning/finetune_whisper_lora.py \
    --output_dir checkpoints/whisper_medium_lora_scots_r32_lr5e4_final \
    --base_model openai/whisper-medium \
    --lora_r 32 \
    --lora_alpha 64 \
    --epochs 4 \
    --lr 5e-4 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluate_test

