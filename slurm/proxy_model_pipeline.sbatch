#!/bin/bash
#SBATCH -J proxy_model_pipeline
#SBATCH -A MLMI-ors36-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -p ampere
#SBATCH --output=logs/proxy_model_pipeline_%j.log

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
export OLLAMA_NUM_PARALLEL=8
export ENSEMBLE_MAX_WORKERS=8
mkdir -p "$OLLAMA_MODELS"

ollama serve &
OLLAMA_PID=$!
sleep 5
ollama pull phi4:14b

DATASETS=("commonvoice" "english_dialects" "edacc" "shetland")

echo ""
echo "══════════ Step 1: confscore sentence labels ══════════"
for d in "${DATASETS[@]}"; do
    split="dev"
    [ "$d" == "shetland" ] && split="full"
    echo "── labeling ${d} (confscore) ──"
    python rerunning/sentence_confidence/label_sentence_meanings.py \
        --combo "writeup_results/ensembles/naive_confscore/naive_confscore_${d}_gemma4sel_${split}.json" \
        --output "results/sentence_confidence/sentence_labels_${d}_confscore.json"
done

echo ""
echo "══════════ Step 2: train proxy model (both variants) ══════════"
python rerunning/sentence_confidence/train_proxy_model.py --variant probscore
python rerunning/sentence_confidence/train_proxy_model.py --variant confscore

echo ""
echo "══════════ Step 3: evaluate all methods per dataset ══════════"
for d in "${DATASETS[@]}"; do
    split="dev"
    [ "$d" == "shetland" ] && split="full"

    echo "── ${d}: confscore ──"
    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
        --labels "results/sentence_confidence/sentence_labels_${d}_confscore.json" \
        --confidences "writeup_results/ensembles/naive_confscore/naive_confscore_${d}_gemma4sel_${split}.json" \
        --method confscore \
        --output "results/sentence_confidence/method1_confscore_${d}.json"

    echo "── ${d}: probscore ──"
    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
        --labels "results/sentence_confidence/sentence_labels_${d}_probscore.json" \
        --confidences "writeup_results/ensembles/naive_probscore/naive_probscore_${d}_gemma4sel_${split}.json" \
        --method probscore \
        --output "results/sentence_confidence/method1_probscore_${d}.json"

    echo "── ${d}: crossmodel_mean ──"
    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
        --labels "results/sentence_confidence/sentence_labels_${d}_probscore.json" \
        --confidences "results/sentence_confidence/crossmodel_agreement_${d}.json" \
        --method crossmodel_mean \
        --output "results/sentence_confidence/method2a_crossmodel_mean_${d}.json"

    echo "── ${d}: crossmodel_min ──"
    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
        --labels "results/sentence_confidence/sentence_labels_${d}_probscore.json" \
        --confidences "results/sentence_confidence/crossmodel_agreement_${d}.json" \
        --method crossmodel_min \
        --confidence-field min_agreement \
        --output "results/sentence_confidence/method2b_crossmodel_min_${d}.json"

    echo "── ${d}: acoustic_mean ──"
    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
        --labels "results/sentence_confidence/sentence_labels_${d}_probscore.json" \
        --confidences "results/sentence_confidence/acoustic_confidence_${d}.json" \
        --method acoustic_mean \
        --output "results/sentence_confidence/method3_acoustic_mean_${d}.json"

    echo "── ${d}: proxy_model (probscore-trained) ──"
    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
        --labels "results/sentence_confidence/sentence_labels_${d}_probscore.json" \
        --confidences "results/sentence_confidence/proxy_model_${d}_probscore.json" \
        --method proxy_model \
        --output "results/sentence_confidence/method4_proxy_model_probscore_${d}.json"

    echo "── ${d}: proxy_model (confscore-trained) ──"
    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
        --labels "results/sentence_confidence/sentence_labels_${d}_confscore.json" \
        --confidences "results/sentence_confidence/proxy_model_${d}_confscore.json" \
        --method proxy_model_confscore \
        --output "results/sentence_confidence/method4_proxy_model_confscore_${d}.json"
done

echo ""
echo "══════════ Step 4: summary table ══════════"
python scripts/evaluation/sentence_confidence/summarise_sentence_conf_results.py

kill $OLLAMA_PID

