#!/bin/bash

TRAIN_ENV=${1:-doppler_output01}

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/3] Training phase..."
echo "--------------------------------------------------------"
python train.py --train_env "$TRAIN_ENV"

echo ""
echo "[2/3] Global evaluation..."
echo "--------------------------------------------------------"
python eval.py

echo ""
echo "[3/3] Detailed analysis..."
echo "--------------------------------------------------------"
python confusion.py

echo ""
echo "Pipeline terminated successfully !"
echo "========================================================"