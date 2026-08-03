#!/bin/bash

TRAIN_ENV=${1:-doppler_output_a}

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/3] Training phase..."
echo "--------------------------------------------------------"
# python3 train.py --train_env "$TRAIN_ENV"

echo ""
echo "[2/3] Global evaluation..."
echo "--------------------------------------------------------"
# python3 eval.py --env "$TRAIN_ENV"

echo ""
echo "[3/3] Detailed analysis..."
echo "--------------------------------------------------------"
python3 confusion_matrix.py --env "$TRAIN_ENV"

echo ""
echo "Pipeline terminated successfully !"
echo "========================================================"