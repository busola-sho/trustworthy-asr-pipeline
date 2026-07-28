#!/bin/bash
set -e

for v in probscore confscore; do
  for d in commonvoice edacc english_dialects shetland; do
    split="dev"; [ "$d" == "shetland" ] && split="full"

    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
      --labels results/sentence_confidence/sentence_labels_${d}_${v}.json \
      --confidences writeup_results/ensembles/naive_${v}/naive_${v}_${d}_gemma4sel_${split}.json \
      --method ${v} \
      --output results/sentence_confidence/method1_${v}_${d}.json

    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
      --labels results/sentence_confidence/sentence_labels_${d}_${v}.json \
      --confidences results/sentence_confidence/crossmodel_agreement_${v}_${d}.json \
      --method crossmodel_mean \
      --output results/sentence_confidence/method2a_${v}_${d}.json

    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
      --labels results/sentence_confidence/sentence_labels_${d}_${v}.json \
      --confidences results/sentence_confidence/crossmodel_agreement_${v}_${d}.json \
      --method crossmodel_min \
      --confidence-field min_agreement \
      --output results/sentence_confidence/method2b_${v}_${d}.json

    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
      --labels results/sentence_confidence/sentence_labels_${d}_${v}.json \
      --confidences results/sentence_confidence/acoustic_confidence_${v}_${d}.json \
      --method acoustic_mean \
      --output results/sentence_confidence/method3_${v}_${d}.json

    python rerunning/sentence_confidence/evaluate_sentence_conf_and_labels.py \
      --labels results/sentence_confidence/sentence_labels_${d}_${v}.json \
      --confidences results/sentence_confidence/proxy_model_${d}_${v}.json \
      --method proxy_model \
      --output results/sentence_confidence/method4_${v}_${d}.json
  done
done