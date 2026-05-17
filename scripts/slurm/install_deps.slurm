#!/bin/bash
#SBATCH -J install_deps
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH -p ampere
#SBATCH --output=logs/install_deps_%j.log

module purge
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_ampere/bin/activate

pip install torchcodec soundfile
