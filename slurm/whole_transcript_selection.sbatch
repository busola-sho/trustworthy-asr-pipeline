#!/bin/bash
#SBATCH -J whole_transcript_selection
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -p ampere
#SBATCH --output=logs/whole_transcript_selection_%j.log

# slurm/whole_transcript_selection.sbatch
#
# Runs whole_transcript_selection.py (Baseline 4 - LLM must pick one of
# the four transcripts completely unchanged, no editing) + severity
# scoring, for the two remaining datasets (english_dialects, edacc).
# CommonVoice was already run interactively.
#
# Usage:
#   sbatch slurm/whole_transcript_selection.sbatch

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

SELECTOR="gemma4"
SPLIT="dev"

run_dataset() {
    local dataset="$1"
    local output_file="writeup_results/ensembles/whole_transcript_selection/wholetranscript_${dataset}_${SELECTOR}sel_${SPLIT}.json"

    echo ""
    echo "── whole_transcript_selection (${dataset}, ${SPLIT}) ──"
    python rerunning/ensembles/whole_transcript_selection.py \
        --dataset "$dataset" --split "$SPLIT" --selector "$SELECTOR"

    echo "  -> Phase 2 (severity, concurrent): ${output_file}"
    python rerunning/add_severity_to_existing_concurrent.py --files "$output_file"
}

run_dataset "english_dialects"
run_dataset "edacc"

echo ""
echo "Both remaining datasets done for whole_transcript_selection."

kill $OLLAMA_PID

