#!/bin/bash

set -o pipefail

EPOCHS=$1
KFOLDS=$2
ROOTDIR=$3
CLASSES=${*:4}
RESULTS_FILE="$(dirname "$0")/resultats.txt"
LOG_DIR="$(dirname "$0")/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_cnn_$(date +%Y%m%d_%H%M%S).log"
TIMING_DIR=$(mktemp -d)
CURRENT_STAGE="initialisation"

exec > >(tee -a "$LOG_FILE") 2>&1

cleanup() {
    local status=$?
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_cnn stopped during: $CURRENT_STAGE (exit code: $status)"
    echo "Log file: $LOG_FILE"
    rm -rf "$TIMING_DIR"
}
trap cleanup EXIT

if [ -z "$CLASSES" ]; then
    CLASSES="0 1 2 3 4"
fi

echo "EPOCHS set to: $EPOCHS"
echo "KFOLDS set to: $KFOLDS"
echo "CLASSES set to: $CLASSES"

format_duration() {
    awk -v seconds="$1" 'BEGIN {
        minutes = int(seconds / 60)
        remaining = seconds - minutes * 60
        printf "%dm%.3fs", minutes, remaining
    }'
}

run_training() {
    local environment="$1"
    CURRENT_STAGE="training $environment"

    echo "========================================================"
    echo "Starting of the pipeline : training on ${environment: -1}"
    echo "========================================================"
    echo ""
    echo "Training phase..."
    echo "--------------------------------------------------------"
    /usr/bin/time -f '%e' -o "$TIMING_DIR/$environment" \
        python3 train.py \
        --train_env "$environment" \
        --epochs "$EPOCHS" \
        --k-folds "$KFOLDS" \
        --root_dir "$ROOTDIR" \
        --classes $CLASSES
    local status=$?
    if [ "$status" -ne 0 ]; then
        echo "Training failed for $environment (exit code: $status)"
        return "$status"
    fi

    echo ""
    echo "Pipeline terminated successfully for $environment !"
    echo "========================================================"
}

run_training doppler_output_a || exit $?
run_training doppler_output_b || exit $?
run_training doppler_output_c || exit $?
run_training doppler_output_d || exit $?

matrix_start=$(date +%s.%N)
CURRENT_STAGE="confusion matrix"
echo "========================================================"
echo "Starting confusion_matrix.py"
echo "========================================================"
/usr/bin/time -f '%e' -o "$TIMING_DIR/matrix" \
    python3 confusion_matrix.py \
    --epochs "$EPOCHS" \
    --k-folds "$KFOLDS" \
    --classes $CLASSES
matrix_status=$?
if [ "$matrix_status" -ne 0 ]; then
    echo "Confusion matrix failed (exit code: $matrix_status)"
    exit "$matrix_status"
fi
matrix_end=$(date +%s.%N)

TOTAL_SECONDS=$(awk \
    -v a="$(cat "$TIMING_DIR/doppler_output_a")" \
    -v b="$(cat "$TIMING_DIR/doppler_output_b")" \
    -v c="$(cat "$TIMING_DIR/doppler_output_c")" \
    -v d="$(cat "$TIMING_DIR/doppler_output_d")" \
    -v matrix="$(cat "$TIMING_DIR/matrix")" \
    'BEGIN { print a + b + c + d + matrix }')

{
    echo "kfold : $KFOLDS"
    echo "$EPOCHS epochs : $(format_duration "$TOTAL_SECONDS")"
    echo "a : $(format_duration "$(cat "$TIMING_DIR/doppler_output_a")")"
    echo "b : $(format_duration "$(cat "$TIMING_DIR/doppler_output_b")")"
    echo "c : $(format_duration "$(cat "$TIMING_DIR/doppler_output_c")")"
    echo "d : $(format_duration "$(cat "$TIMING_DIR/doppler_output_d")")"
    echo "matrix : $(format_duration "$(cat "$TIMING_DIR/matrix")")"
    echo ""
} >> "$RESULTS_FILE"

echo "Timing summary appended to $RESULTS_FILE"
