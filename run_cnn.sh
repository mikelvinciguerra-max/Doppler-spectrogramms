#!/bin/bash

TRAIN_ENV=doppler_output_a
EPOCHS=$1
ROOTDIR=$2

echo "EPOCHS set to: $EPOCHS"

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/2] Training phase..."
echo "--------------------------------------------------------"
python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS" --root_dir "$ROOTDIR"

echo ""
echo "[2/2] Global evaluation..."
echo "--------------------------------------------------------"
python3 eval.py --env "$TRAIN_ENV"

echo ""
echo "Pipeline terminated successfully for $TRAIN_ENV !"
echo "========================================================"



TRAIN_ENV=doppler_output_b

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/2] Training phase..."
echo "--------------------------------------------------------"
python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS" --root_dir "$ROOTDIR"

echo ""
echo "[2/2] Global evaluation..."
echo "--------------------------------------------------------"
python3 eval.py --env "$TRAIN_ENV"

echo ""
echo "Pipeline terminated successfully for $TRAIN_ENV !"
echo "========================================================"



TRAIN_ENV=doppler_output_c

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/2] Training phase..."
echo "--------------------------------------------------------"
python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS" --root_dir "$ROOTDIR"

echo ""
echo "[2/2] Global evaluation..."
echo "--------------------------------------------------------"
python3 eval.py --env "$TRAIN_ENV"

echo ""
echo "Pipeline terminated successfully for $TRAIN_ENV !"
echo "========================================================"



TRAIN_ENV=doppler_output_d

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/2] Training phase..."
echo "--------------------------------------------------------"
python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS" --root_dir "$ROOTDIR"

echo ""
echo "[2/2] Global evaluation..."
echo "--------------------------------------------------------"
python3 eval.py --env "$TRAIN_ENV"

echo ""
echo "Pipeline terminated successfully for $TRAIN_ENV !"
echo "========================================================"

python3 confusion_matrix.py
