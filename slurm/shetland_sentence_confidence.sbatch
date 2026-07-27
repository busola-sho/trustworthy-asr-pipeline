#!/bin/bash
#SBATCH -J shetland_sentence_confidence
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -p ampere
#SBATCH --output=logs/shetland_sentence_confidence_%j.log

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

ollama serve &
OLLAMA_PID=$!
sleep 5
ollama pull gemma4:12b
ollama pull phi4:14b

SELECTOR="gemma4"
VARIANTS=("naive_confscore" "naive_probscore" "naive_confscore_meaning" "naive_probscore_meaning")

for variant in "${VARIANTS[@]}"; do
    echo ""
    echo "── ${variant} (shetland, full) ──"
    python "rerunning/ensembles/${variant}.py" --dataset shetland --split full --selector "$SELECTOR"

    output_file="writeup_results/ensembles/${variant}/${variant}_shetland_${SELECTOR}sel_full.json"
    echo "  -> Phase 2 (severity): ${output_file}"
    python rerunning/add_severity_to_existing_concurrent.py --files "$output_file"
done

echo ""
echo "── Method 2: crossmodel agreement (shetland) ──"
python rerunning/sentence_confidence/compute_crossmodel_agreement.py --dataset shetland

echo ""
echo "── Method 3: acoustic confidence (shetland) ──"
python rerunning/sentence_confidence/compute_acoustic_confidence.py --dataset shetland

echo ""
echo "All Shetland sentence-confidence work done."

kill $OLLAMA_PID

