#!/bin/bash

# python3 confusion_matrix_classes.py --train-env c --test-envs a --model-dir models/ --classes 1 4 --epochs 5
# python3 confusion_matrix_classes.py --train-env c --test-envs a --model-dir models/ --classes 1 4 --epochs 10
# python3 confusion_matrix_classes.py --train-env c --test-envs a --model-dir models/ --classes 1 4 --epochs 40
# python3 confusion_matrix_classes.py --train-env c --test-envs a --model-dir models/ --classes 1 4 --epochs 100

# python3 confusion_matrix_classes.py --train-env c --test-envs a --model-dir models/ --classes 1 2 3 4 --epochs 5
# python3 confusion_matrix_classes.py --train-env c --test-envs a --model-dir models/ --classes 1 2 3 4 --epochs 10
# python3 confusion_matrix_classes.py --train-env c --test-envs a --model-dir models/ --classes 1 2 3 4 --epochs 40
# python3 confusion_matrix_classes.py --train-env c --test-envs a --model-dir models/ --classes 1 2 3 4 --epochs 100

python3 confusion_matrix_classes.py --train-env a --test-envs a --model-dir models/ --classes 1 2 3 4 --epochs 5
python3 confusion_matrix_classes.py --train-env a --test-envs a --model-dir models/ --classes 1 2 3 4 --epochs 10
python3 confusion_matrix_classes.py --train-env a --test-envs a --model-dir models/ --classes 1 2 3 4 --epochs 40
python3 confusion_matrix_classes.py --train-env a --test-envs a --model-dir models/ --classes 1 2 3 4 --epochs 100

python3 confusion_matrix_classes.py --train-env b --test-envs b --model-dir models/ --classes 1 4 --epochs 5
python3 confusion_matrix_classes.py --train-env b --test-envs b --model-dir models/ --classes 1 4 --epochs 10
python3 confusion_matrix_classes.py --train-env b --test-envs b --model-dir models/ --classes 1 4 --epochs 40
python3 confusion_matrix_classes.py --train-env b --test-envs b --model-dir models/ --classes 1 4 --epochs 100