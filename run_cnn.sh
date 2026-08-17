#!/bin/bash

TRAIN_ENV=doppler_output_a
EPOCHS=${2:-40}

echo "EPOCHS set to: $EPOCHS"

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/3] Training phase..."
echo "--------------------------------------------------------"
python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS"

echo ""
echo "[2/3] Global evaluation..."
echo "--------------------------------------------------------"
# python3 eval.py --env "$TRAIN_ENV"

echo ""
echo "[3/3] Detailed analysis..."
echo "--------------------------------------------------------"
# python3 confusion_matrix.py --env "$TRAIN_ENV"

echo ""
echo "Pipeline terminated successfully for $TRAIN_ENV !"
echo "========================================================"



TRAIN_ENV=doppler_output_b

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/3] Training phase..."
echo "--------------------------------------------------------"
python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS"

echo ""
echo "[2/3] Global evaluation..."
echo "--------------------------------------------------------"
# python3 eval.py --env "$TRAIN_ENV"

echo ""
echo "[3/3] Detailed analysis..."
echo "--------------------------------------------------------"
# python3 confusion_matrix.py --env "$TRAIN_ENV"

echo ""
echo "Pipeline terminated successfully for $TRAIN_ENV !"
echo "========================================================"



TRAIN_ENV=doppler_output_c

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/3] Training phase..."
echo "--------------------------------------------------------"
python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS"

echo ""
echo "[2/3] Global evaluation..."
echo "--------------------------------------------------------"
# python3 eval.py --env "$TRAIN_ENV"

echo ""
echo "[3/3] Detailed analysis..."
echo "--------------------------------------------------------"
# python3 confusion_matrix.py --env "$TRAIN_ENV"

echo ""
echo "Pipeline terminated successfully for $TRAIN_ENV !"
echo "========================================================"



TRAIN_ENV=doppler_output_d

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/3] Training phase..."
echo "--------------------------------------------------------"
python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS"

echo ""
echo "[2/3] Global evaluation..."
echo "--------------------------------------------------------"
# python3 eval.py --env "$TRAIN_ENV"

echo ""
echo "[3/3] Detailed analysis..."
echo "--------------------------------------------------------"
# python3 confusion_matrix.py --env "$TRAIN_ENV"

echo ""
echo "Pipeline terminated successfully for $TRAIN_ENV !"
echo "========================================================"

python3 confusion_matrix.py
