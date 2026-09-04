import argparse
import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import CNN
from dataset import load_dataset_with_cache

BATCH_SIZE = 64

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


class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, target_classes):
        self.base_dataset = base_dataset
        self.target_classes = set(target_classes)
        self.label_to_index = {label: index for index, label in enumerate(target_classes)}
        self.indices = [
            index for index in range(len(base_dataset))
            if int(base_dataset[index][1].item()) in self.target_classes
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y = self.base_dataset[self.indices[index]]
        return x, torch.tensor(self.label_to_index[int(y.item())], dtype=torch.long)

def plot_accuracy_matrix(accuracy_matrix, env_names, train_env, epochs, k_folds=None):
    print("EPOCHS : ", epochs)
    test_envs = [e for e in env_names if e != train_env]
    data      = np.array([[accuracy_matrix[e] for e in test_envs]])

    fig, ax = plt.subplots(figsize=(max(6, len(test_envs) * 1.5), 3))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=test_envs,
                yticklabels=[f"Train: {train_env}"],
                vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Cross-environment accuracy — trained on {train_env}")
    plt.tight_layout()
    kfold_suffix = f"_kfolds_{k_folds}" if k_folds is not None else ""
    plt.savefig(f"matrix/accuracy_matrix_{train_env}_epochs_{epochs}{kfold_suffix}.png", dpi=150)
    plt.close()
    print(f"Saved -> accuracy_matrix_{train_env}_epochs_{epochs}{kfold_suffix}.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluating CNN on Doppler profiles")
    parser.add_argument("--env", type=str, default="a", help="Environment letter used during training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs used to train the model")
    parser.add_argument("--classes", nargs='+', type=int, default=[0, 1, 2, 3, 4], help="Classes used to train the model (class 0 means no person)")
    parser.add_argument("--k-folds", type=int, default=None, help="Number of CV folds used during training")
    args = parser.parse_args()

    env_name = args.env.split("_")[-1] if "_" in args.env else args.env
    classes = sorted({cls for cls in args.classes if 0 <= cls <= 4})
    classes_str = "-".join(map(str, classes))
    model_filename = f"model_doppler_{env_name}_classes_{classes_str}_epochs_{args.epochs}"
    if args.k_folds is not None:
        model_filename += f"_kfolds_{args.k_folds}"
    model_filename += ".pth"
    MODEL_PATH = os.path.join("models", model_filename)

    if not os.path.exists(MODEL_PATH):
        matching_models = [
            m for m in os.listdir("models")
            if m.startswith(f"model_doppler_{env_name}_classes_{classes_str}_") and m.endswith(f"_epochs_{args.epochs}.pth")
        ]
        if not matching_models:
            raise FileNotFoundError(f"No model found for env {env_name} with classes {sorted(args.classes)}")
        raise FileNotFoundError(f"No model found for env {env_name} at {args.epochs} epochs with classes {sorted(args.classes)}. Available: {matching_models}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    train_env   = checkpoint['train_env']
    env_names   = checkpoint['env_names']
    num_classes = checkpoint['num_classes']
    root_dir    = checkpoint['root_dir']
    k_folds = checkpoint.get('k_folds', args.k_folds)
    target_classes = sorted(checkpoint.get('target_classes', classes))

    model = CNN(input_channels=1, num_classes=num_classes).to(device)
    dummy = torch.zeros(1, 1, 32, 32).to(device)
    model(dummy)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_envs       = [e for e in env_names if e != train_env]
    accuracy_matrix = {}

    print(f"\n{'='*50}")
    print(f"EVALUATION — Trained on: {train_env}")
    print(f"{'='*50}")

    for env_name in test_envs:
        env_dir = os.path.join(root_dir, "doppler_output_" + env_name)
        test_dataset = load_dataset_with_cache(env_dir)
        test_dataset = MappedDataset(test_dataset, target_classes)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        acc = evaluate_accuracy(model, test_loader, device)
        accuracy_matrix[env_name] = acc

    # Summary table
    print(f"\n{'Test environment':<20} {'Accuracy':>10}")
    print("-" * 32)
    for env_name in test_envs:
        print(f"{env_name:<20} {accuracy_matrix[env_name]:>10.4f}")

    plot_accuracy_matrix(accuracy_matrix, env_names, train_env, checkpoint.get('epochs', 'unknown'), k_folds)