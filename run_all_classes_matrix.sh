#!/bin/bash

set -euo pipefail

MODEL_DIR="${1:-models}"
EPOCHS_VALUES=(5 10 40 100)
K_FOLDS_VALUES=(1 2 5)
CLASS_SETS=("0 1 4" "0 1 2 3 4")

for k_folds in "${K_FOLDS_VALUES[@]}"; do
	for epochs in "${EPOCHS_VALUES[@]}"; do
		for class_set in "${CLASS_SETS[@]}"; do
			read -r -a classes <<< "$class_set"

			echo "========================================================"
			echo "EPOCHS=$epochs K_FOLDS=$k_folds CLASSES=${classes[*]}"
			echo "========================================================"

			time ./run_classes_matrix.sh \
				"$epochs" \
				"$k_folds" \
				"$MODEL_DIR" \
				"${classes[@]}"
		done
	done
done

