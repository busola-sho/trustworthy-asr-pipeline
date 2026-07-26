#!/bin/bash
#SBATCH -J test_ensembles_english_dialects
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH -p ampere
#SBATCH --output=logs/test_ensembles_english_dialects_%j.log

# slurm/test_ensembles_english_dialects.sbatch
#
# Runs ALL 10 ensemble techniques (naive, naive_confidence,
# naive_confidence_inline, context_v1, context_v1_confidence,
# context_v1_confidence_inline, context_v2, context_v2_confidence,
# context_v2_confidence_inline, rover) for english_dialects on --split test,
# each followed immediately by its severity-scoring pass. This is your
# held-out confirmatory check, run once after picking a winning config
# on dev - not a redo of the dev work.
#
# Uses the CONCURRENT selector scripts + the concurrent/batched severity
# script, same setup as your dev runs.
#
# Usage:
#   sbatch slurm/test_ensembles_english_dialects.sbatch

set -e

mkdir -p logs

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
export OLLAMA_MAX_LOADED_MODELS=1
export ENSEMBLE_MAX_WORKERS=8

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
SPLIT="test"

run_selector_technique() {
    local script="$1"
    local output_file="$2"
    echo ""
    echo "── ${script} (${DATASET}, ${SPLIT}) ──"
    python "rerunning/ensembles/${script}.py" \
        --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR"
    echo "  -> Phase 2 (severity, concurrent): ${output_file}"
    python rerunning/add_severity_to_existing_concurrent.py --files "$output_file"
}

run_confidence_technique() {
    local script="$1"
    local output_file="$2"
    echo ""
    echo "── ${script} (${DATASET}, ${SPLIT}) ──"
    python "rerunning/ensembles/${script}.py" \
        --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR" --percentile "$PERCENTILE"
    echo "  -> Phase 2 (severity, concurrent): ${output_file}"
    python rerunning/add_severity_to_existing_concurrent.py --files "$output_file"
}

# non-confidence techniques
run_selector_technique "naive" \
    "writeup_results/ensembles/naive/naive_${DATASET}_${SELECTOR}sel_${SPLIT}.json"

run_selector_technique "context_v1" \
    "writeup_results/ensembles/context_v1/context_${DATASET}_${SELECTOR}_${SPLIT}.json"

run_selector_technique "context_v2" \
    "writeup_results/ensembles/context_v2/context_v2_${DATASET}_${SELECTOR}_${SPLIT}.json"

echo ""
echo "── rover (${DATASET}, ${SPLIT}) ──"
python rerunning/voting/rover.py --dataset "$DATASET" --split "$SPLIT"
echo "  -> Phase 2 (severity, concurrent): writeup_results/voting/rover/rover_${DATASET}_${SPLIT}.json"
python rerunning/add_severity_to_existing_concurrent.py \
    --files "writeup_results/voting/rover/rover_${DATASET}_${SPLIT}.json"

# confidence techniques
run_confidence_technique "naive_confidence" \
    "writeup_results/ensembles/naive_confidence/naive_conf_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_confidence_technique "naive_confidence_inline" \
    "writeup_results/ensembles/naive_confidence_inline/naive_confinline_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_confidence_technique "context_v1_confidence" \
    "writeup_results/ensembles/context_v1_confidence/context_v1conf_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_confidence_technique "context_v1_confidence_inline" \
    "writeup_results/ensembles/context_v1_confidence_inline/context_v1confinline_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_confidence_technique "context_v2_confidence" \
    "writeup_results/ensembles/context_v2_confidence/context_v2conf_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

run_confidence_technique "context_v2_confidence_inline" \
    "writeup_results/ensembles/context_v2_confidence_inline/context_v2confinline_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

echo ""
echo "All 10 techniques + severity done for ${DATASET} (${SPLIT})."

kill $OLLAMA_PID
