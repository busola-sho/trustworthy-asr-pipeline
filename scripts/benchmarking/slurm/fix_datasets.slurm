#!/bin/bash
#SBATCH -J fix_datasets
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH -p ampere
#SBATCH --output=logs/fix_datasets_%j.log

module purge
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_ampere/bin/activate

pip uninstall torchcodec -y
pip install "datasets==2.19.0" soundfile
pip show datasets | grep Version