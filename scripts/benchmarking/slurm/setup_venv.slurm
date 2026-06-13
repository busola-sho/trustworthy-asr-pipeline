#!/bin/bash
#SBATCH -J setup_venv
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH -p ampere
#SBATCH --output=logs/setup_venv_%j.log

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

cd /rds/user/ors36/hpc-work/trustworthy-asr-pipeline

python -m venv .venv_ampere
source .venv_ampere/bin/activate
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets librosa soundfile jiwer pandas numpy