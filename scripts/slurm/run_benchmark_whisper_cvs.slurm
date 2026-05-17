#!/bin/bash
#SBATCH -J benchmark_whisper_commonvoice
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:30:00
#SBATCH -p ampere
#SBATCH --output=logs/benchmark_whisper_commonvoice_%j.log

module purge
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_ampere/bin/activate

export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python scripts/run_benchmark.py --model whisper --dataset commonvoice