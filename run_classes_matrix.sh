#!/bin/bash

set -euo pipefail

EPOCHS="${1:-40}"
K_FOLDS="${2:-1}"
MODEL_DIR="${3:-models}"
shift $(( $# >= 3 ? 3 : $# ))

if (( $# > 0 )); then
	CLASSES=("$@")
else
	CLASSES=(0 1 2 3 4)
fi

ENVIRONMENTS=(a b c d)

echo "EPOCHS set to: $EPOCHS"
echo "K_FOLDS set to: $K_FOLDS"
echo "MODEL_DIR set to: $MODEL_DIR"
echo "CLASSES set to: ${CLASSES[*]}"

for train_env in "${ENVIRONMENTS[@]}"; do
	test_envs=()
	for test_env in "${ENVIRONMENTS[@]}"; do
		if [[ "$test_env" != "$train_env" ]]; then
			test_envs+=("$test_env")
		fi
	done

	echo "========================================================"
	echo "Evaluating model trained on $train_env"
	echo "Test environments: ${test_envs[*]}"
	echo "========================================================"

	python3 confusion_matrix_classes.py \
		--train-env "$train_env" \
		--test-envs "${test_envs[@]}" \
		--model-dir "$MODEL_DIR" \
		--classes "${CLASSES[@]}" \
		--epochs "$EPOCHS" \
		--k-folds "$K_FOLDS"
done
