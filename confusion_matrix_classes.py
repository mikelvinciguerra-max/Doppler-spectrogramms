import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model import CNN
from dataset import load_dataset_with_cache

BATCH_SIZE = 64
DEFAULT_MODEL_DIR = 'models/'
DEFAULT_MATRIX_DIR = 'matrix/'

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


def build_confusion_matrix(labels, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        if 0 <= label < num_classes and 0 <= prediction < num_classes:
            matrix[label, prediction] += 1
    row_totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)


# ADDED: MappedDataset imported from train script to handle the 3-channel gradient extraction
class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, label_to_index):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        x, y = self.base_dataset[sample_idx]
        label = int(y.item())
        
        # Extract spectral and temporal dynamics for the 3-channel model
        grad_freq, grad_time = torch.gradient(x[0], dim=(0, 1))
        x_enhanced = torch.stack([x[0], grad_freq, grad_time], dim=0)
        
        return x_enhanced, torch.tensor(self.label_to_index[label], dtype=torch.long)


def plot_confusion_matrix_from_checkpoint(model_path, device='cpu', batch_size=BATCH_SIZE,
                                          save=True, test_envs=None, matrix_dir=DEFAULT_MATRIX_DIR):
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
        
    label_to_index = {int(cls): idx for idx, cls in enumerate(target_classes)}

    # Build model and load weights
    # MODIFIED: input_channels changed from 1 to 3 to support gradient features
    model = CNN(input_channels=3, num_classes=num_classes).to(device)
    # MODIFIED: dummy tensor shape updated to match the new 3-channel input
    dummy = torch.zeros(1, 3, 32, 32).to(device)
    model(dummy)  # initialize lazy layers
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    # Determine which test environments to evaluate
    if test_envs is None:
        eval_envs = [e for e in env_names if e != train_env[-1]]
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
        
        # MODIFIED: Use MappedDataset for proper 3-channel filtering and dynamic extraction
        target_set = set(target_classes)
        filtered_indices = [i for i in range(len(dataset)) if int(dataset[i][1].item()) in target_set]
        mapped_dataset = MappedDataset(dataset, filtered_indices, label_to_index)
        
        loader = DataLoader(mapped_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                preds = torch.argmax(model(x), dim=1)
                all_preds.extend(preds.cpu().numpy())
                # Labels from MappedDataset are already converted to indices (0 to num_classes-1)
                all_labels.extend(y.numpy())

    labels = list(range(len(target_classes)))
    display_labels = get_display_labels(target_classes)
    if not all_labels:
        raise RuntimeError(f"No test samples found for target classes {target_classes}")

    cm = build_confusion_matrix(all_labels, all_preds, len(labels))
    print(f"Classes displayed: {list(zip(labels, display_labels))}")

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set(xticks=labels, yticks=labels, xticklabels=display_labels, yticklabels=display_labels)
    threshold = cm.max() / 2 if cm.size else 0
    for row in labels:
        for column in labels:
            ax.text(column, row, f'{cm[row, column]:.2f}',
                    ha='center', va='center',
                    color='white' if cm[row, column] > threshold else 'black')

    plt.title(f"Confusion Matrix — trained on {train_env} ({epochs} epochs, {k_folds}-fold CV)")
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if save:
        os.makedirs(matrix_dir, exist_ok=True)
        train_label = str(train_env).split('_')[-1]
        test_labels = [str(env).split('_')[-1] for env in eval_envs]
        test_label = '-'.join(test_labels)
        class_suffix = get_filename_class_suffix(target_classes)
        kfold_suffix = f"_kfolds_{k_folds}" if k_folds != 'unknown' else ''
        save_path = os.path.join(matrix_dir, f"train_{train_label}_test_{test_label}_classes_{class_suffix}_epochs_{epochs}{kfold_suffix}.png")
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

    pattern = os.path.join(model_dir, f"{train_letter}_classes_*.pth")
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
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR, help='Directory where checkpoints are stored')
    parser.add_argument('--matrix-dir', default=DEFAULT_MATRIX_DIR, help='Directory where the matrix image is saved')
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
        # Check against the last letter to handle both formats ('doppler_output_a' and 'a')
        train_suffix = train_env_ckpt[-1] if isinstance(train_env_ckpt, str) else train_env_ckpt
        test_envs = [e for e in env_names if e != train_suffix]

    # Call the plotting routine which will load model and process these test envs
    cm = plot_confusion_matrix_from_checkpoint(model_path, device=device, batch_size=args.batch_size,
                                               save=True, test_envs=test_envs, matrix_dir=args.matrix_dir)
    print('Confusion matrix (percentages):')
    print(np.array2string(cm, formatter={'float_kind':lambda x: f"{x:.1f}"}))


if __name__ == '__main__':
    main()