#!/bin/bash
#SBATCH -J sentence_conf_verbalized
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -p ampere
#SBATCH --output=logs/sentence_conf_verbalized_%j.log

set -e

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

cd /rds/user/ors36/hpc-work/trustworthy-asr-pipeline

source .venv_ampere/bin/activate

export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline
export HF_HOME=/rds/user/ors36/hpc-work/hf_cache

export PATH=/rds/user/ors36/hpc-work/ollama_install/bin:$PATH
export LD_LIBRARY_PATH=/rds/user/ors36/hpc-work/ollama_install/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=/rds/user/ors36/hpc-work/ollama_models

mkdir -p "$OLLAMA_MODELS"
mkdir -p logs

export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_LOADED_MODELS=2
export ENSEMBLE_MAX_WORKERS=8

ollama serve &
OLLAMA_PID=$!

cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 8

ollama pull gemma4:12b
ollama pull phi4:14b

VARIANTS=("confscore" "probscore")
DATASETS=("commonvoice" "edacc" "english_dialects")
SPLITS=("dev" "test")

for variant in "${VARIANTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for split in "${SPLITS[@]}"; do

            echo "=================================================="
            echo "RUNNING/RESUMING: ${variant} | ${dataset} | ${split}"
            echo "=================================================="

            python rerunning/sentence_confidence/verbalized_confidence.py \
                --dataset "$dataset" \
                --split "$split" \
                --variant "$variant" \
                --max-workers 8 \
                --chunk-size 5

        done
    done

    echo "=================================================="
    echo "RUNNING/RESUMING: ${variant} | shetland | full"
    echo "=================================================="

    python rerunning/sentence_confidence/verbalized_confidence.py \
        --dataset shetland \
        --split full \
        --variant "$variant" \
        --max-workers 8 \
        --chunk-size 5
done

echo "All verbalized-confidence runs complete."
