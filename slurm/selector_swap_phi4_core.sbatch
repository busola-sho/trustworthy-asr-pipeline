#!/bin/bash
#SBATCH -J selector_swap_phi4_core
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -p ampere
#SBATCH --output=logs/selector_swap_phi4_core_%j.log

# slurm/selector_swap_phi4_core.sbatch
#
# Core techniques (naive, context_v1, context_v2) with phi4 as
# selector instead of the locked gemma4, across all 4 datasets (dev
# split). rover/mbr_consensus excluded - no LLM selector, unchanged
# control. Confidence-weighted variants run in separate sbatch files.
#
# Usage: sbatch slurm/selector_swap_phi4_core.sbatch

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_ampere/bin/activate
export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline
export HF_HOME=/rds/user/ors36/hpc-work/hf_cache

export PATH=/rds/user/ors36/hpc-work/ollama_install/bin:$PATH
export LD_LIBRARY_PATH=/rds/user/ors36/hpc-work/ollama_install/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=/rds/user/ors36/hpc-work/ollama_models
mkdir -p "$OLLAMA_MODELS"

export OLLAMA_NUM_PARALLEL=8
export ENSEMBLE_MAX_WORKERS=8

ollama serve &
OLLAMA_PID=$!
sleep 5
ollama pull phi4:14b

DATASETS=("commonvoice" "english_dialects" "edacc" "shetland")

for d in "${DATASETS[@]}"; do
    split="dev"
    [ "$d" == "shetland" ] && split="full"

    echo "── naive | phi4 | ${d} ──"
    python rerunning/ensembles/naive.py --dataset "$d" --split "$split" --selector phi4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/ensembles/naive/naive_${d}_phi4_${split}.json"

    echo "── context_v1 | phi4 | ${d} ──"
    python rerunning/ensembles/context_v1.py --dataset "$d" --split "$split" --selector phi4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/ensembles/context_v1/context_v1_${d}_phi4_${split}.json"

    echo "── context_v2 | phi4 | ${d} ──"
    python rerunning/ensembles/context_v2.py --dataset "$d" --split "$split" --selector phi4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/ensembles/context_v2/context_v2_${d}_phi4_${split}.json"
done

kill $OLLAMA_PID

