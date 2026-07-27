#!/bin/bash
#SBATCH -J naive_sentence_confidence_variants
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -p ampere
#SBATCH --output=logs/naive_sentence_confidence_variants_%j.log

# slurm/naive_sentence_confidence_variants.sbatch
#
# Runs all 4 naive sentence-confidence variants (naive_confscore,
# naive_probscore, naive_confscore_meaning, naive_probscore_meaning) for
# english_dialects only, each followed immediately by severity scoring.
# CommonVoice was already run interactively for all 4 variants; edacc
# queued separately.
#
# Usage:
#   sbatch slurm/naive_sentence_confidence_variants.sbatch

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
DATASETS=("english_dialects")
VARIANTS=("naive_confscore" "naive_probscore" "naive_confscore_meaning" "naive_probscore_meaning")

run_variant() {
    local variant="$1"
    local dataset="$2"
    local output_file="writeup_results/ensembles/${variant}/${variant}_${dataset}_${SELECTOR}sel_dev.json"

    echo ""
    echo "── ${variant} (${dataset}, dev) ──"
    python "rerunning/ensembles/${variant}.py" --dataset "$dataset" --split dev --selector "$SELECTOR"

    echo "  -> Phase 2 (severity, concurrent): ${output_file}"
    python rerunning/add_severity_to_existing_concurrent.py --files "$output_file"
}

for dataset in "${DATASETS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        run_variant "$variant" "$dataset"
    done
done

echo ""
echo "All 4 sentence-confidence variants done for english_dialects."

kill $OLLAMA_PID

