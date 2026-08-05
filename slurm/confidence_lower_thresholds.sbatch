#!/bin/bash
#SBATCH -J confidence_lower_thresholds
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH -p ampere
#SBATCH --output=logs/confidence_lower_thresholds_%j.log

# slurm/confidence_lower_thresholds.sbatch
#
# Tests lower percentile thresholds (p5, p10) for the list/inline
# confidence-informed conditions on Unanchored Fusion + Naive, across
# the 3 dev datasets. Motivation: p20 flags a very large fraction of
# words per transcript (seen directly in a real CommonVoice example -
# dozens of [LOW-CONF: ...] markers in a single utterance), which may
# be part of why confidence hurt performance - too much of the
# transcript gets flagged to be useful signal. p5/p10 flag a much
# smaller, hopefully more genuinely informative fraction.
#
# --threshold-split dev is used throughout (frozen calibration,
# same discipline as the p20 runs).
#
# Usage:
#   sbatch slurm/confidence_lower_thresholds.sbatch

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
ollama pull gemma4:12b

DATASETS=("commonvoice" "edacc" "english_dialects")
PERCENTILES=(5 10)

for p in "${PERCENTILES[@]}"; do
    for d in "${DATASETS[@]}"; do

        echo "── list | p${p} | gemma4 | ${d} ──"
        python rerunning/grid/unanchored_fusion_naive_confidence_list.py \
            --dataset "$d" --split dev --selector gemma4 --percentile "$p"
        python rerunning/add_severity_to_existing_concurrent.py \
            --files "writeup_results/grid/unanchored_fusion_naive_confidence_list/unanchored_fusion_naive_confidence_list_${d}_gemma4_p${p}_thr-dev_dev.json"

        echo "── inline | p${p} | gemma4 | ${d} ──"
        python rerunning/grid/unanchored_fusion_naive_confidence_inline.py \
            --dataset "$d" --split dev --selector gemma4 --percentile "$p"
        python rerunning/add_severity_to_existing_concurrent.py \
            --files "writeup_results/grid/unanchored_fusion_naive_confidence_inline/unanchored_fusion_naive_confidence_inline_${d}_gemma4_p${p}_thr-dev_dev.json"

    done
done

kill $OLLAMA_PID

