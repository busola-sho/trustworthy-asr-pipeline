#!/bin/bash
set -e

mkdir -p writeup_results/benchmarks/main

merge_one() {
    local old_file="$1"
    local new_file="$2"
    local output_file="$3"

    echo ""
    echo "── merging: $(basename "$new_file") ──"
    echo "  old (severity): $old_file"
    echo "  new (confidence): $new_file"
    echo "  output: $output_file"

    python transfer_severity.py \
        --old "$old_file" \
        --new "$new_file" \
        --output "$output_file"
}

merge_one \
    "results/benchmarks/main/parakeet_commonvoice_20260524_150129.json" \
    "results/benchmarks/main/parakeet_commonvoice_20260724_195002.json" \
    "writeup_results/benchmarks/main/parakeet_commonvoice_merged.json"

merge_one \
    "results/benchmarks/main/parakeet_english_dialects_20260524_234807.json" \
    "results/benchmarks/main/parakeet_english_dialects_20260725_004037.json" \
    "writeup_results/benchmarks/main/parakeet_english_dialects_merged.json"

merge_one \
    "results/benchmarks/main/wav2vec2_commonvoice_20260526_053757.json" \
    "results/benchmarks/main/wav2vec2_commonvoice_20260725_015238.json" \
    "writeup_results/benchmarks/main/wav2vec2_commonvoice_merged.json"

merge_one \
    "results/benchmarks/main/wav2vec2_english_dialects_20260526_073439.json" \
    "results/benchmarks/main/wav2vec2_english_dialects_20260725_020821.json" \
    "writeup_results/benchmarks/main/wav2vec2_english_dialects_merged.json"

echo ""
echo "=================================================="
echo "  All 4 merges done. Check each file's printed"
echo "  'still need severity judging' count above -"
echo "  if any are non-zero, run add_severity_to_existing.py"
echo "  on that specific output file to finish it off."
echo "=================================================="
