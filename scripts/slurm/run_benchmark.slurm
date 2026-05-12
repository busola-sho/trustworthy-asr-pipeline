#!/bin/bash
#SBATCH -J benchmark_whisper_english_dialects
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH -p ampere
#SBATCH --output=logs/benchmark_whisper_english_dialects_%j.log

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv/bin/activate
module load ffmpeg/6.0
module load python/3.11.9/gcc/abrhyqg7

export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python scripts/run_benchmark.py
