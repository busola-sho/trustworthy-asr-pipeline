#!/bin/bash
#SBATCH -J clean_grid_anchored_correction_context_v2
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH -p ampere
#SBATCH --output=logs/clean_grid_anchored_correction_context_v2_%j.log

# slurm/clean_grid_anchored_correction_context_v2.sbatch
#
# Regenerates ONE grid cell (anchored_correction_context_v2) FRESH, both dev and test
# splits, all 3 datasets (6 runs total) - into writeup_results/clean_grid/.
# get_indices_for_split() now correctly excludes the 100 calibration
# samples before shuffling, so no patching is needed afterward.
#
# SAFETY: does NOT touch writeup_results/grid/ or grid_calib_fixed/ -
# runs a sed-redirected TEMPORARY copy of the script (OUTPUT_DIR ->
# clean_grid/) from /tmp, leaving the original in rerunning/grid/
# completely untouched.
#
# Split into per-technique jobs (rather than one large 24h job)
# specifically for faster HPC queue scheduling - smaller time
# requests typically queue faster than one large block.
#
# Usage:
#   sbatch slurm/clean_grid_anchored_correction_context_v2.sbatch

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

mkdir -p /tmp/clean_grid_scripts_anchored_correction_context_v2
mkdir -p writeup_results/clean_grid/anchored_correction_context_v2

original="rerunning/grid/anchored_correction_context_v2.py"
temp_copy="/tmp/clean_grid_scripts_anchored_correction_context_v2/anchored_correction_context_v2.py"
sed 's#writeup_results/grid/#writeup_results/clean_grid/#g' "$original" > "$temp_copy"

DATASETS=("commonvoice" "edacc" "english_dialects")
SPLITS=("dev" "test")

for d in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do

        echo "── anchored_correction_context_v2 | ${d} | ${split} ──"
        python "$temp_copy" --dataset "$d" --split "$split" --selector gemma4 --rerun

        out_dir="writeup_results/clean_grid/anchored_correction_context_v2"
        out_file=$(ls -t "${out_dir}"/*"${d}"*"${split}"*.json 2>/dev/null | head -1)

        if [ -n "$out_file" ]; then
            python rerunning/add_severity_to_existing_concurrent.py --files "$out_file"
        else
            echo "  WARNING: could not find output file for anchored_correction_context_v2/${d}/${split} - skipping severity"
        fi

    done
done

kill $OLLAMA_PID

