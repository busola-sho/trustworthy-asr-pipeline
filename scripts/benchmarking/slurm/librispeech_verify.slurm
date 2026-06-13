#!/bin/bash
#SBATCH -J librispeech_verify
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
#SBATCH -p ampere
#SBATCH --output=logs/librispeech_%a_%j.log
#SBATCH --array=0-3

MODELS=("whisper" "wav2vec2" "parakeet" "qwen3asr")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Model:     $MODEL"
echo "Started:   $(date)"
echo "======================================================"

module purge
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_ampere/bin/activate

export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline

echo "Running LibriSpeech test-clean verification for $MODEL..."

python scripts/run_librispeech_benchmark.py --model $MODEL

echo "Finished: $(date)"