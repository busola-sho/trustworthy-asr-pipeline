#!/bin/bash
#SBATCH -J confidence_ensembles_english_dialects
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH -p ampere
#SBATCH --output=logs/confidence_ensembles_english_dialects_%j.log

# slurm/confidence_ensembles_english_dialects.sbatch
#
# Runs all 6 confidence ensemble scripts (naive_confidence,
# naive_confidence_inline, context_v1_confidence,
# context_v1_confidence_inline, context_v2_confidence,
# context_v2_confidence_inline) for english_dialects, --split dev, each
# followed immediately by its severity-scoring pass.
#
# Uses the CONCURRENT versions of all 6 selector scripts (Phase A prep /
# Phase B thread-pooled Ollama calls) AND the concurrent, batched version
# of the severity judge (add_severity_to_existing_concurrent.py) - only
# actually helps if OLLAMA_NUM_PARALLEL is set high enough on the server
# AND your client-side MAX_WORKERS roughly matches it (both set below).
#
# --time=16:00:00 is a generous first guess given english_dialects' dev
# split is your largest (~1700+ samples) and this runs 6 full
# techniques + severity back to back - tighten this once you've seen
# real per-technique timing from this first run.
#
# Usage:
#   sbatch slurm/confidence_ensembles_english_dialects.sbatch

set -e

mkdir -p logs

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

# NOTE: uses .venv_ampere (your original ensemble-pipeline env with
# ollama/jiwer/etc installed) - NOT .venv_finetune, which is deliberately
# lean and doesn't have ollama installed.
source /rds/user/ors36/hpc-work/trustworthy-asr-pipeline/.venv_ampere/bin/activate
export PYTHONPATH=/rds/user/ors36/hpc-work/trustworthy-asr-pipeline
export HF_HOME=/rds/user/ors36/hpc-work/hf_cache

# redirect Ollama's own model storage off the tight $HOME quota
export OLLAMA_MODELS=/rds/user/ors36/hpc-work/ollama_models
mkdir -p "$OLLAMA_MODELS"

# enable real server-side concurrency - MAX_WORKERS in each script
# defaults to 8, so match that here (adjust both together if you change one)
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_LOADED_MODELS=1

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
sleep 5

echo "Pulling required models (gemma4 selector, phi4 judge)..."
ollama pull gemma4:12b
ollama pull phi4:14b

DATASET="english_dialects"
SELECTOR="gemma4"
PERCENTILE=20
SPLIT="dev"

run_technique() {
    local script="$1"
    local output_file="$2"

    echo ""
    echo "── ${script} (${DATASET}) ──"
    python "rerunning/ensembles/${script}.py" \
        --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR" --percentile "$PERCENTILE"

    echo "  → Phase 2 (severity, concurrent): ${output_file}"
    python rerunning/add_severity_to_existing_concurrent.py --files "$output_file"
}

run_technique "naive_confidence" \
    "writeup_results/ensembles/naive_confidence/naive_conf_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_technique "naive_confidence_inline" \
    "writeup_results/ensembles/naive_confidence_inline/naive_confinline_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_technique "context_v1_confidence" \
    "writeup_results/ensembles/context_v1_confidence/context_v1conf_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_technique "context_v1_confidence_inline" \
    "writeup_results/ensembles/context_v1_confidence_inline/context_v1confinline_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_technique "context_v2_confidence" \
    "writeup_results/ensembles/context_v2_confidence/context_v2conf_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_technique "context_v2_confidence_inline" \
    "writeup_results/ensembles/context_v2_confidence_inline/context_v2confinline_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

echo ""
echo "All 6 confidence techniques + severity done for ${DATASET}."

kill $OLLAMA_PID
