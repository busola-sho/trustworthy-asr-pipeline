#!/bin/bash
#SBATCH -J test_venv
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH -p ampere
#SBATCH --output=logs/test_venv_%j.log

module purge
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_ampere/bin/activate

python --version
python -c "import torch; print(torch.__version__)"