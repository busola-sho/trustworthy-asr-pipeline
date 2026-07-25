#!/bin/bash
#SBATCH -J finetune_wav2vec2_scots
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
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
# IMPORTANT: this job does NOT pass --evaluate_test - trains and selects
# checkpoints via validation WER only, test manifest untouched. Add
# --evaluate_test only as a separate final job once configs are fixed.
#
# The vocab-building step below is now OPTIONAL and commented out by
# default - finetune_wav2vec2.py defaults to --use_pretrained_vocab
# (reuses the base checkpoint's own English CTC head/vocab, recommended
# for this data volume). Only uncomment build_wav2vec2_vocab.py and add
# --no-use_pretrained_vocab to the training command if your transcripts
# have characters the pretrained tokenizer can't represent.
#
# Usage:
#   sbatch ****/finetune_wav2vec2.sbatch

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_finetune/bin/activate

export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python scripts/finetuning/build_wav2vec2_vocab.py   # only needed if using --no-use_pretrained_vocab below

python scripts/finetuning/finetune_wav2vec2.py \
    --output_dir checkpoints/wav2vec2_scots \
    --epochs 30 \
    --lr 3e-5 \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 4
