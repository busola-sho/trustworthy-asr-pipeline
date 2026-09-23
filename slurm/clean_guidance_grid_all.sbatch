#!/bin/bash
#SBATCH -J clean_guidance_grid_all
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --output=logs/clean_guidance_grid_all_%j.log

# Runs/resumes the complete clean V1/V2 guidance grid:
#   3 strategies x 2 guidance conditions x 3 datasets x 2 splits = 36 files.
# It does NOT use --rerun, so completed files are retained and partial files resume.
# Once all 36 selector outputs exist, it adds locked Phi-4 severity scores.

set -euo pipefail

REPO="/rds/user/ors36/hpc-work/trustworthy-asr-pipeline"
cd "$REPO"

module purge
module load python/3.11.0-icl
module load ffmpeg/6.0

source "$REPO/.venv_ampere/bin/activate"
export PYTHONPATH="$REPO"
export HF_HOME="/rds/user/ors36/hpc-work/hf_cache"

export PATH="/rds/user/ors36/hpc-work/ollama_install/bin:$PATH"
export LD_LIBRARY_PATH="/rds/user/ors36/hpc-work/ollama_install/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="/rds/user/ors36/hpc-work/ollama_models"
mkdir -p "$OLLAMA_MODELS"

export OLLAMA_NUM_PARALLEL=8
export ENSEMBLE_MAX_WORKERS=8

RULES_FILE="results/eval_suite/selector_rules_dev.txt"
OUTPUT_ROOT="writeup_results/clean_grid_guidance_rerun"
DATASETS=(commonvoice edacc english_dialects)
SPLITS=(dev test)

if [[ ! -s "$RULES_FILE" ]]; then
    echo "ERROR: missing frozen V2 rules: $RULES_FILE" >&2
    exit 1
fi

ollama serve > "logs/ollama_clean_guidance_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

sleep 5
ollama pull gemma4:12b
ollama pull phi4:14b

for split in "${SPLITS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "===== selection V1 | $dataset | $split ====="
        python rerunning/grid/selection_context_v1.py \
            --dataset "$dataset" --split "$split" --selector gemma4

        echo "===== selection V2 | $dataset | $split ====="
        python rerunning/grid/selection_context_v2.py \
            --dataset "$dataset" --split "$split" --selector gemma4 \
            --rules-file "$RULES_FILE"

        echo "===== unanchored fusion V1 | $dataset | $split ====="
        python rerunning/grid/unanchored_fusion_context_v1.py \
            --dataset "$dataset" --split "$split" --selector gemma4

        echo "===== unanchored fusion V2 | $dataset | $split ====="
        python rerunning/grid/unanchored_fusion_context_v2.py \
            --dataset "$dataset" --split "$split" --selector gemma4 \
            --rules-file "$RULES_FILE"

        echo "===== anchored correction V1 | $dataset | $split ====="
        python rerunning/grid/anchored_correction_context_v1.py \
            --dataset "$dataset" --split "$split" --selector gemma4

        echo "===== anchored correction V2 | $dataset | $split ====="
        python rerunning/grid/anchored_correction_context_v2.py \
            --dataset "$dataset" --split "$split" --selector gemma4 \
            --rules-file "$RULES_FILE"
    done
done

mapfile -t OUTPUT_FILES < <(find "$OUTPUT_ROOT" -type f -name '*.json' | sort)

if [[ "${#OUTPUT_FILES[@]}" -ne 36 ]]; then
    echo "ERROR: expected 36 selector files, found ${#OUTPUT_FILES[@]}." >&2
    echo "Severity judging was not started." >&2
    exit 1
fi

echo "===== Severity judging ${#OUTPUT_FILES[@]} files with Phi-4 ====="
python rerunning/add_severity_to_existing_concurrent.py \
    --files "${OUTPUT_FILES[@]}" \
    --max-workers 8 \
    --batch-size 50

echo "===== Clean guidance grid and severity judging complete ====="

