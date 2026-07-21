#!/bin/bash
# rerunning/run_all_ensembles.sh
#
# Runs every ensemble technique (phase 1: combination, phase 2: severity
# judging) across all three DEV datasets (CommonVoice, English Dialects,
# EdAcc). NEVER touches Shetland - held-out test set, per the dev/test
# discipline: nothing gets evaluated there until judge+prompt+selector+
# ensemble technique are all fully locked from dev data alone.
#
# SELECTOR is still the placeholder default ("qwen") - update this once
# the selector ablation (rerunning/selector_ablation.py) gives you a
# justified choice.
#
# SPLIT defaults to "dev" - every ensemble script now takes
# --split {dev,test,full} instead of --full, so iteration only scores the
# dev subset rather than the whole dataset every time. Change SPLIT below
# (or override at call time) once you're ready for a confirmatory run on
# "test", and never set it to "full" here - Shetland aside, "full"
# includes the excluded calibration IDs too, which iteration should not
# be scoring against.
#
# Usage:
#   chmod +x rerunning/run_all_ensembles.sh
#   ./rerunning/run_all_ensembles.sh
#
# Or run one technique/dataset at a time by commenting out sections below -
# given the runtime, that's probably the safer way to go rather than
# kicking this whole thing off unattended in one go.

set -e  # stop on first real error (not on individual sample failures - those are handled inside each script)

SELECTOR="qwen"          # TODO: update once selector_ablation.py gives a result
PERCENTILE=20
SPLIT="dev"              # dev | test | full - see note above; keep at "dev" for iteration
DATASETS=("commonvoice" "english_dialects" "edacc")

run_phase2() {
    local output_file="$1"
    echo "  → Phase 2 (severity): $output_file"
    python rerunning/add_severity_to_existing.py --files "$output_file"
}

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=================================================="
    echo "  DATASET: $DATASET  (split=$SPLIT)"
    echo "=================================================="

    # ── Naive combination ──────────────────────────────────────────────
    echo ""
    echo "── naive ──"
    python rerunning/ensembles/naive.py --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR"
    run_phase2 "writeup_results/ensembles/naive/naive_${DATASET}_${SELECTOR}sel_${SPLIT}.json"

    echo ""
    echo "── naive_confidence (separate/structured) ──"
    python rerunning/ensembles/naive_confidence.py --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR" --percentile "$PERCENTILE"
    run_phase2 "writeup_results/ensembles/naive_confidence/naive_conf_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

    echo ""
    echo "── naive_confidence_inline ──"
    python rerunning/ensembles/naive_confidence_inline.py --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR" --percentile "$PERCENTILE"
    run_phase2 "writeup_results/ensembles/naive_confidence_inline/naive_confinline_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

    # ── Context V1 (hand-written rules) ─────────────────────────────────
    echo ""
    echo "── context_v1 ──"
    python rerunning/ensembles/context_v1.py --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR"
    run_phase2 "writeup_results/ensembles/context_v1/context_${DATASET}_${SELECTOR}_${SPLIT}.json"

    echo ""
    echo "── context_v1_confidence (separate/structured) ──"
    python rerunning/ensembles/context_v1_confidence.py --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR" --percentile "$PERCENTILE"
    run_phase2 "writeup_results/ensembles/context_v1_confidence/context_v1conf_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

    echo ""
    echo "── context_v1_confidence_inline ──"
    python rerunning/ensembles/context_v1_confidence_inline.py --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR" --percentile "$PERCENTILE"
    run_phase2 "writeup_results/ensembles/context_v1_confidence_inline/context_v1confinline_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

    # ── Context V2 (auto-generated rules) ───────────────────────────────
    # NOTE: still blocked on regenerating error_profiles.json from WhisperX
    # error data - the auto-rules text may still reference "whisper" as a
    # model. Worth checking that before trusting these results.
    echo ""
    echo "── context_v2 ──"
    python rerunning/ensembles/context_v2.py --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR"
    run_phase2 "writeup_results/ensembles/context_v2/context_v2_${DATASET}_${SELECTOR}_${SPLIT}.json"

    echo ""
    echo "── context_v2_confidence (separate/structured) ──"
    python rerunning/ensembles/context_v2_confidence.py --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR" --percentile "$PERCENTILE"
    run_phase2 "writeup_results/ensembles/context_v2_confidence/context_v2conf_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

    echo ""
    echo "── context_v2_confidence_inline ──"
    python rerunning/ensembles/context_v2_confidence_inline.py --dataset "$DATASET" --split "$SPLIT" --selector "$SELECTOR" --percentile "$PERCENTILE"
    run_phase2 "writeup_results/ensembles/context_v2_confidence_inline/context_v2confinline_${DATASET}_${SELECTOR}_p${PERCENTILE}_${SPLIT}.json"

    # ── ROVER (Fiscus 1997, no LLM/selector - not selector-dependent) ───
    echo ""
    echo "── rover ──"
    python rerunning/voting/rover.py --dataset "$DATASET" --split "$SPLIT"
    run_phase2 "writeup_results/voting/rover/rover_${DATASET}_${SPLIT}.json"

    echo ""
    echo "  Done with $DATASET."
done

echo ""
echo "=================================================="
echo "  ALL TECHNIQUES x ALL DEV DATASETS COMPLETE (split=$SPLIT)"
echo "  (CommonVoice, English Dialects, EdAcc only - Shetland untouched)"
echo "=================================================="