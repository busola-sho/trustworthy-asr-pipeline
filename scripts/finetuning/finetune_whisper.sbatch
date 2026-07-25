#!/bin/bash
#SBATCH -J finetune_whisper_lora_scots
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH -p ampere
#SBATCH --output=logs/finetune_whisper_lora_scots_%j.log

# slurm/finetune_whisper.sbatch
#
# NOTE on --gres=gpu:1 (vs the 4 GPUs your benchmark jobs used): this
# script trains on a single GPU via HF Trainer, which does not do
# multi-GPU data parallelism unless launched with `accelerate launch` or
# `torchrun`. Requesting more GPUs here would just sit idle and burn SL2
# allocation for no speedup - only bump this if you wrap the script with
# accelerate/torchrun accordingly.
#
# ASSUMES: you've already run scripts/finetuning/prepare_manifest.py
# LOCALLY and rsync'd data/finetune/ (audio + manifests) onto this
# cluster - this job does NOT regenerate the manifest, it expects the
# data to already be sitting in data/finetune/.
#
# IMPORTANT: this job does NOT pass --evaluate_test, so it will train
# and select the best checkpoint using validation WER only - the test
# manifest stays untouched. Only add --evaluate_test to the command
# below once you've finished comparing configs/learning rates and have
# a final configuration picked - run that as a separate, final job
# rather than editing this one in place, so you keep a record of what
# was a sweep run vs. the final test-evaluated run.
#
# Uses a DEDICATED venv (.venv_finetune) separate from whatever env your
# main ensemble/benchmark pipeline uses - the fine-tuning scripts only
# need 6 lightweight packages (see scripts/finetuning/requirements.txt),
# so keeping them isolated avoids the dependency conflicts from before.
#
# Usage:
#   sbatch slurm/finetune_whisper.sbatch

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_finetune/bin/activate

export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python scripts/finetuning/finetune_whisper_lora.py \
    --output_dir checkpoints/whisper_small_lora_scots_r32 \
    --lora_r 32 \
    --lora_alpha 64 \
    --epochs 10 \
    --lr 1e-3 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 2
