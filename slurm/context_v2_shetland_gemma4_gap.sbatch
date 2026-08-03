#!/bin/bash
#SBATCH -J context_v2_shetland_gemma4_gap
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -p ampere
#SBATCH --output=logs/context_v2_shetland_gemma4_gap_%j.log

# slurm/context_v2_shetland_gemma4_gap.sbatch
#
# Fills in a genuine gap: context_v2, context_v2_confidence, and
# context_v2_confidence_inline were never run on Shetland with gemma4
# (only qwen/phi4 versions exist) - confirmed via find, not present
# anywhere. Every other technique (naive, context_v1, mbr_consensus)
# already has a gemma4+Shetland file. Small job, Shetland is only 100
# samples.
#
# Usage:
#   sbatch slurm/context_v2_shetland_gemma4_gap.sbatch

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
ollama pull phi4:14b

echo "── context_v2 | gemma4 | shetland ──"
python rerunning/ensembles/context_v2.py --dataset shetland --split full --selector gemma4
python rerunning/add_severity_to_existing_concurrent.py \
    --files "writeup_results/ensembles/context_v2/context_v2_shetland_gemma4_full.json"

echo "── context_v2_confidence | gemma4 | shetland ──"
python rerunning/ensembles/context_v2_confidence.py --dataset shetland --split full --selector gemma4 --percentile 20
python rerunning/add_severity_to_existing_concurrent.py \
    --files "writeup_results/ensembles/context_v2_confidence/context_v2conf_shetland_gemma4_p20_full.json"

echo "── context_v2_confidence_inline | gemma4 | shetland ──"
python rerunning/ensembles/context_v2_confidence_inline.py --dataset shetland --split full --selector gemma4 --percentile 20
python rerunning/add_severity_to_existing_concurrent.py \
    --files "writeup_results/ensembles/context_v2_confidence_inline/context_v2confinline_shetland_gemma4_p20_full.json"

kill $OLLAMA_PID

