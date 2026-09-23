import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import CNN
from dataset import load_dataset_with_cache

BATCH_SIZE = 64
MODEL_DIR = "models/"
MATRIX_DIR = "matrix/"


def normalize_target_classes(classes):
    if classes is None:
        return []
    return sorted({int(cls) for cls in classes if 0 <= int(cls) <= 4})


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


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def plot_full_matrix(matrix, env_names, epochs, target_classes, k_folds=None):
    """
    Displays and saves the cross-correlation matrix.
    Rows are test environments, columns are training environments.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=env_names,
                yticklabels=env_names,
                vmin=0, vmax=1, ax=ax)
    
    ax.set_xlabel("Train")
    ax.set_ylabel("Test")
    classes_str = "-".join(map(str, sorted(target_classes)))
    ax.set_title(f"Intra and Inter-Scenario Performance Matrix (Classes {classes_str})")
    
    # Move the X-axis labels to the top to match the image style
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    
    # Create the matrix folder if it doesn't exist
    os.makedirs(MATRIX_DIR, exist_ok=True)
    classes_str = "-".join(map(str, sorted(target_classes)))
    kfold_suffix = f"_kfolds_{k_folds}" if k_folds is not None else ""
    save_path = f"{MATRIX_DIR}/full_cross_env_accuracy_classes_{classes_str}_epochs_{epochs}{kfold_suffix}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFull matrix saved -> {save_path}")


def build_model_filename(train_env, classes_str, epochs, k_folds=None):
    kfold_suffix = f"_kfolds_{k_folds}" if k_folds is not None else ""
    return f"{train_env[-1]}_classes_{classes_str}_epochs_{epochs}{kfold_suffix}.pth"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-environment evaluation matrix")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs used to train the models")
    parser.add_argument("--classes", nargs='+', type=int, default=[1, 2, 3, 4], help="Classes used to train the models (class 0 is not used)")
    parser.add_argument("--k-folds", type=int, default=1, help="Number of CV folds used during training")
    args = parser.parse_args()
    target_classes = normalize_target_classes(args.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_str = "-".join(map(str, target_classes))

    # 1. Load an initial model just to extract global metadata (env_names, root_dir)
    model_prefix = build_model_filename("a", classes_str, args.epochs, args.k_folds)
    INITIAL_MODEL_PATH = os.path.join(MODEL_DIR, model_prefix)
    if not os.path.exists(INITIAL_MODEL_PATH):
        matching_initial_models = [
            m for m in os.listdir(MODEL_DIR)
            if m.startswith("model_doppler_a_classes_")
            and m.endswith(f"_epochs_{args.epochs}.pth")
            and (args.k_folds is None or f"_kfolds_{args.k_folds}.pth" in m or "_kfolds_" not in m)
        ]
        normalized_candidates = [
            m for m in matching_initial_models
            if classes_str and f"classes_{classes_str}_" in m
        ]
        if not normalized_candidates and classes_str:
            legacy_candidates = [
                m for m in matching_initial_models
                if f"classes_{'-'.join(map(str, sorted({int(cls) for cls in args.classes})))}_" in m
            ]
            if legacy_candidates:
                INITIAL_MODEL_PATH = os.path.join("models", legacy_candidates[0])
            else:
                raise FileNotFoundError(f"No model found for environment a at {args.epochs} epochs with classes {target_classes}. Available: {matching_initial_models}")
        elif not matching_initial_models:
            raise FileNotFoundError(f"No model found for environment a with classes {target_classes}")
        else:
            INITIAL_MODEL_PATH = os.path.join(MODEL_DIR, normalized_candidates[0])

    checkpoint = torch.load(INITIAL_MODEL_PATH, map_location=device, weights_only=False)
    env_names = checkpoint['env_names']
    num_classes = checkpoint['num_classes']
    root_dir = checkpoint['root_dir']
    epochs = checkpoint.get('epochs', args.epochs)
    k_folds = checkpoint.get('k_folds', args.k_folds)
    if 'target_classes' in checkpoint:
        target_classes = sorted(checkpoint['target_classes'])
    print(f"Using target classes: {target_classes}")
    
    label_to_index = {label: idx for idx, label in enumerate(target_classes)}
    
    n_envs = len(env_names)
    
    # Initialize the results matrix (Rows=Test, Columns=Train)
    accuracy_matrix = np.zeros((n_envs, n_envs))
    
    print(f"Beginning cross-evaluation on {n_envs} environments: {env_names}")

    # 2. Loop over columns (Training environments)
    for j, train_env in enumerate(env_names):
        expected_model = build_model_filename(train_env[-1], classes_str, args.epochs, k_folds)
        model_path = os.path.join(MODEL_DIR, expected_model)
        if not os.path.exists(model_path):
            matching_models = [
                m for m in os.listdir(MODEL_DIR)
                if m.startswith(f"model_doppler_{train_env[-1]}_classes_{classes_str}_")
                and m.endswith(f"_epochs_{args.epochs}.pth")
            ]
            print(f"Missing model ignored for env {train_env} at {args.epochs} epochs with classes {sorted(args.classes)}. Available: {matching_models}")
            continue
            
        print(f"\n--- Evaluating model trained on: {train_env} ---")
        
        # Load the model specific to this environment.
        model = CNN(input_channels=3, num_classes=num_classes).to(device)
        dummy = torch.zeros(1, 3, 32, 32).to(device)
        model(dummy) 
        
        model_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(model_checkpoint['model_state_dict'])
        model.eval()

        test_indices = model_checkpoint.get('test_indices')

        # 3. Loop over rows (Test environments)
        for i, test_env in enumerate(env_names):
            env_dir = os.path.join(root_dir, "doppler_output_" + test_env)
            
            # Ensure the test folder exists
            if not os.path.exists(env_dir):
                continue

            test_dataset = load_dataset_with_cache(env_dir)
            
            # Same environment as training: restrict to the held-out test split, otherwise
            # this diagonal cell reuses samples seen during training/validation.
            if i == j and test_indices is not None:
                test_mapped_dataset = MappedDataset(test_dataset, test_indices, label_to_index)
            else:
                filtered_indices = [idx for idx in range(len(test_dataset)) if int(test_dataset[idx][1].item()) in target_classes]
                test_mapped_dataset = MappedDataset(test_dataset, filtered_indices, label_to_index)

            test_loader = DataLoader(test_mapped_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Accuracy calculation.
            acc = evaluate_accuracy(model, test_loader, device)
            
            # Store in the matrix: i = Test (row), j = Train (column)
            accuracy_matrix[i, j] = acc
            print(f"  -> Test on {test_env:<15} : {acc:.4f}")

    # 4. Generate the matrix image
    plot_full_matrix(accuracy_matrix, env_names, epochs, target_classes, k_folds)