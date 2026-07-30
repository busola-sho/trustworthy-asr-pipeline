#!/bin/bash
#SBATCH -J selector_swap_phi4_cv2_conf
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -p ampere
#SBATCH --output=logs/selector_swap_phi4_cv2_conf_%j.log

# slurm/selector_swap_phi4_context_v2_confidence.sbatch
#
# context_v2_confidence + context_v2_confidence_inline (p20) with
# phi4 as selector, across all 4 datasets (dev split).
#
# Usage: sbatch slurm/selector_swap_phi4_context_v2_confidence.sbatch

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

    echo "── context_v2_confidence | phi4 | ${d} ──"
    python rerunning/ensembles/context_v2_confidence.py --dataset "$d" --split "$split" --selector phi4 --percentile 20
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/ensembles/context_v2_confidence/context_v2_conf_${d}_phi4_p20_${split}.json"

    echo "── context_v2_confidence_inline | phi4 | ${d} ──"
    python rerunning/ensembles/context_v2_confidence_inline.py --dataset "$d" --split "$split" --selector phi4 --percentile 20
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/ensembles/context_v2_confidence_inline/context_v2_conf_inline_${d}_phi4_p20_${split}.json"
done

kill $OLLAMA_PID

