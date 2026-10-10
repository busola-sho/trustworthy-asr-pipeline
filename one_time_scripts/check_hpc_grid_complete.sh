#!/bin/bash

TECHNIQUES=(
    "selection_naive"
    "selection_context_v1"
    "selection_context_v2"
    "unanchored_fusion_naive"
    "unanchored_fusion_context_v1"
    "unanchored_fusion_context_v2"
    "anchored_correction_naive"
    "anchored_correction_v1"
    "anchored_correction_v2"
)

echo "=== clean_grid/ completeness check ==="
echo ""
all_good=true

for tech in "${TECHNIQUES[@]}"; do
    dir="writeup_results/clean_grid/${tech}"
    if [ ! -d "$dir" ]; then
        echo "[MISSING DIR] $tech"
        all_good=false
        continue
    fi

    n=$(ls "$dir" 2>/dev/null | wc -l | tr -d ' ')
    echo "$tech: $n files"

    for f in "$dir"/*.json; do
        [ -e "$f" ] || continue
        python3 -c "
import json
data = json.load(open('$f'))
samples = data.get('samples', [])
total = len(samples)
n_severity = sum(1 for s in samples if s.get('severity') is not None)
status = 'OK' if n_severity == total else ('PARTIAL' if n_severity > 0 else 'NO SEVERITY AT ALL')
print(f'    {\"$f\".split(\"/\")[-1]}: {n_severity}/{total} have severity [{status}]')
if n_severity < total:
    exit(1)
" || all_good=false
    done
done

echo ""
if $all_good; then
    echo "All 9 techniques complete AND fully severity-scored - safe to sync to local."
else
    echo "Some files missing, incomplete, or missing severity scores - see above."
fi
