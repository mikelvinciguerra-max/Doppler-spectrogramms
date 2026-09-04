import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader, Subset

from model import CNN
from dataset import load_dataset_with_cache

BATCH_SIZE = 64

CLASS_NAME_BY_INDEX = {
    0: 'Class 0',
    1: 'Class 1',
    2: 'Class 2',
    3: 'Class 3',
    4: 'Class 4',
}


def get_display_labels(target_classes):
    labels = []
    for class_id in target_classes:
        class_id = int(class_id)
        labels.append(CLASS_NAME_BY_INDEX.get(class_id, f'Class {class_id}'))
    return labels


def get_filename_class_suffix(target_classes):
    return '-'.join(str(int(class_id)) for class_id in target_classes)


def filter_dataset_by_classes(dataset, target_classes):
    target_set = set(target_classes)
    filtered_indices = [i for i in range(len(dataset)) if int(dataset[i][1].item()) in target_set]
    return Subset(dataset, filtered_indices)


def plot_confusion_matrix_from_checkpoint(model_path, device='cpu', batch_size=BATCH_SIZE, save=True, test_envs=None):
    """Loads a checkpoint saved by `train.py` / `confusion_matrix.py` and
    computes+plots the confusion matrix across all test environments
    (environments != train_env) for the target classes saved in the checkpoint.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    train_env = checkpoint['train_env']
    env_names = checkpoint['env_names']
    num_classes = checkpoint['num_classes']
    root_dir = checkpoint['root_dir']
    epochs = checkpoint.get('epochs', 'unknown')
    k_folds = checkpoint.get('k_folds', 'unknown')

    if 'target_classes' in checkpoint:
        target_classes = sorted(checkpoint['target_classes'])
    else:
        target_classes = list(range(num_classes))

    # Build model and load weights
    model = CNN(input_channels=1, num_classes=num_classes).to(device)
    dummy = torch.zeros(1, 1, 32, 32).to(device)
    model(dummy)  # initialize lazy layers
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    # Determine which test environments to evaluate
    if test_envs is None:
        eval_envs = [e for e in env_names if e != train_env]
    else:
        # Normalize inputs like 'doppler_output_a' or 'a' -> 'a'
        eval_envs = [t.split('_')[-1] if '_' in t else t for t in test_envs]
    # Loop over the chosen test environments and collect predictions
    for test_env in eval_envs:
        env_dir = os.path.join(root_dir, "doppler_output_" + test_env)
        if not os.path.exists(env_dir):
            print(f"[!] Test folder not found, skipping: {env_dir}")
            continue

        dataset = load_dataset_with_cache(env_dir)
        filtered = filter_dataset_by_classes(dataset, target_classes)
        loader = DataLoader(filtered, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                preds = torch.argmax(model(x), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())

    labels = list(range(len(target_classes)))
    display_labels = [CLASS_NAME_BY_INDEX.get(target_classes[i], f'Class {target_classes[i]}') for i in labels]
    if not all_labels:
        raise RuntimeError(f"No test samples found for target classes {target_classes}")

    mapped_labels = [int(label) for label in all_labels]
    mapped_preds = [int(pred) for pred in all_preds]
    label_to_index = {int(cls): idx for idx, cls in enumerate(target_classes)}
    mapped_labels = [label_to_index[int(label)] for label in mapped_labels]

    cm = confusion_matrix(mapped_labels, mapped_preds, labels=labels, normalize='true')
    print(f"Classes displayed: {list(zip(labels, display_labels))}")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')
    try:
        ax.images[0].set_clim(0, 1)
    except Exception:
        pass

    plt.title(f"Confusion Matrix — trained on {train_env} ({epochs} epochs, {k_folds}-fold CV)")
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if save:
        os.makedirs('matrix/classes', exist_ok=True)
        train_label = str(train_env).split('_')[-1]
        test_labels = [str(env).split('_')[-1] for env in eval_envs]
        test_label = '-'.join(test_labels)
        class_suffix = get_filename_class_suffix(target_classes)
        kfold_suffix = f"_kfolds_{k_folds}" if k_folds != 'unknown' else ''
        save_path = os.path.join('matrix/classes/', f"train_{train_label}_test_{test_label}_classes_{class_suffix}_epochs_{epochs}{kfold_suffix}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix -> {save_path}")
    else:
        plt.show()

    return cm


def find_checkpoint_for_train(train_env, model_dir='models', classes=None, epochs=None, k_folds=None):
    """Find a checkpoint file for a given train_env letter (e.g. 'a')."""
    import glob
    train_letter = train_env.split('_')[-1] if '_' in train_env else train_env
    classes_str = None
    if classes:
        classes_str = '-'.join(map(str, sorted(classes)))

    pattern = os.path.join(model_dir, f"model_doppler_{train_letter}_classes_*.pth")
    candidates = glob.glob(pattern)
    if classes_str:
        candidates = [c for c in candidates if f"classes_{classes_str}_" in os.path.basename(c) or f"classes_{classes_str}.pth" in os.path.basename(c)]
    if epochs is not None:
        candidates = [c for c in candidates if f"_epochs_{epochs}" in os.path.basename(c)]
    if k_folds is not None:
        candidates = [c for c in candidates if f"_kfolds_{k_folds}.pth" in os.path.basename(c)]

    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for train env '{train_env}' in {model_dir}. Searched pattern: {pattern}")

    if len(candidates) > 1:
        # pick the most recently modified
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        print(f"Multiple checkpoints found, selecting newest: {os.path.basename(candidates[0])}")

    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrix for specified train/test environments")
    parser.add_argument('--train-env', required=True, help="Training environment identifier (letter like 'a' or 'doppler_output_a')")
    parser.add_argument('--test-envs', nargs='+', help="One or more test environment identifiers (letters like 'b c')")
    parser.add_argument('--model-dir', default='models', help='Directory where checkpoints are stored')
    parser.add_argument('--epochs', type=int, help='Filter model by epoch count if desired')
    parser.add_argument('--k-folds', type=int, help='Filter model by k-fold count if desired')
    parser.add_argument('--classes', nargs='+', type=int, help='Target classes to consider (overrides checkpoint if provided)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Locate checkpoint automatically
    model_path = find_checkpoint_for_train(args.train_env, model_dir=args.model_dir, classes=args.classes, epochs=args.epochs, k_folds=args.k_folds)
    print(f"Using checkpoint: {model_path}")

    # Load checkpoint to check env names and root dir
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    train_env_ckpt = checkpoint.get('train_env', None)
    env_names = checkpoint.get('env_names', None)

    # Decide which test envs to evaluate
    if args.test_envs:
        test_envs = [t.split('_')[-1] if '_' in t else t for t in args.test_envs]
    else:
        # default: all except training env
        if env_names is None:
            raise RuntimeError('Checkpoint does not contain env_names; please provide --test-envs')
        test_envs = [e for e in env_names if e != train_env_ckpt]

    # Call the plotting routine which will load model and process these test envs
    cm = plot_confusion_matrix_from_checkpoint(model_path, device=device, batch_size=args.batch_size, save=True, test_envs=test_envs)
    print('Confusion matrix (percentages):')
    print(np.array2string(cm, formatter={'float_kind':lambda x: f"{x:.1f}"}))


if __name__ == '__main__':
    main()