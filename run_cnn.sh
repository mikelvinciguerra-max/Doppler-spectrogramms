#!/bin/bash

TRAIN_ENV=doppler_output_a
EPOCHS=$1
ROOTDIR=$2
CLASSES=${*:3}

if [ -z "$CLASSES" ]; then
    CLASSES="0 1 2 3 4"
fi

echo "EPOCHS set to: $EPOCHS"
echo "CLASSES set to: $CLASSES"

echo "========================================================"
echo "Starting of the pipeline : training on $TRAIN_ENV"
echo "========================================================"

echo ""
echo "[1/2] Training phase..."
echo "--------------------------------------------------------"
time python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS" --root_dir "$ROOTDIR" --classes $CLASSES

# echo ""
# echo "[2/2] Global evaluation..."
# echo "--------------------------------------------------------"
# time python3 eval.py --env "$TRAIN_ENV" --epochs "$EPOCHS" --classes $CLASSES

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
time python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS" --root_dir "$ROOTDIR" --classes $CLASSES

# echo ""
# echo "[2/2] Global evaluation..."
# echo "--------------------------------------------------------"
# time python3 eval.py --env "$TRAIN_ENV" --epochs "$EPOCHS" --classes $CLASSES

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
time python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS" --root_dir "$ROOTDIR" --classes $CLASSES

# echo ""
# echo "[2/2] Global evaluation..."
# echo "--------------------------------------------------------"
# time python3 eval.py --env "$TRAIN_ENV" --epochs "$EPOCHS" --classes $CLASSES

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
time python3 train.py --train_env "$TRAIN_ENV" --epochs "$EPOCHS" --root_dir "$ROOTDIR" --classes $CLASSES


# echo ""
# echo "[2/2] Global evaluation..."
# echo "--------------------------------------------------------"
# time python3 eval.py --env "$TRAIN_ENV" --epochs "$EPOCHS" --classes $CLASSES

echo ""
echo "Pipeline terminated successfully for $TRAIN_ENV !"
echo "========================================================"

time python3 confusion_matrix.py --epochs "$EPOCHS" --classes $CLASSES
