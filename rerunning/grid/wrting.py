for d in commonvoice english_dialects edacc; do

    echo "── anchored_correction_context_v1 | gemma4 | ${d} ──"
    python rerunning/grid/anchored_correction_context_v1.py --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/anchored_correction_context_v1/anchored_correction_context_v1_${d}_gemma4_dev.json"

    echo "── anchored_correction_context_v2 | gemma4 | ${d} ──"
    python rerunning/grid/anchored_correction_context_v2.py --dataset "$d" --split dev --selector gemma4
    python rerunning/add_severity_to_existing_concurrent.py \
        --files "writeup_results/grid/anchored_correction_context_v2/anchored_correction_context_v2_${d}_gemma4_dev.json"

done