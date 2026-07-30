import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from model import CNN
from dataset import DopplerDataset


MODEL_PATH = "model_doppler.pth"
BATCH_SIZE = 64

def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy    = (all_preds == all_targets).mean()
    labels      = list(range(num_classes))
    conf        = confusion_matrix(all_targets, all_preds, labels=labels)

    return accuracy, conf


def plot_confusion_matrices(confusion_matrices, env_names, train_env, num_classes):
    """One confusion matrix per test environment."""
    test_envs = [e for e in env_names if e != train_env]
    n_envs    = len(test_envs)
    n_cols    = min(3, n_envs)
    n_rows    = (n_envs + n_cols - 1) // n_cols
    labels    = [str(i) for i in range(num_classes)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_envs == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    for i, env_name in enumerate(test_envs):
        conf      = confusion_matrices[env_name]
        row_sums = conf.sum(axis=1, keepdims=True)
        conf_norm = np.divide(conf, row_sums,
                              where=row_sums != 0,
                              out=np.zeros_like(conf, dtype=float))

        sns.heatmap(conf_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    vmin=0, vmax=1, ax=axes[i])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].set_title(f"Test env: {env_name}")

    for j in range(len(test_envs), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Confusion matrices — model trained on {train_env}, "
        f"tested on other environments",
        fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved -> confusion_matrices.png")


def plot_accuracy_matrix(accuracy_matrix, env_names, train_env):
    """Cross-environment accuracy heatmap."""
    test_envs = [e for e in env_names if e != train_env]
    data      = np.array([[accuracy_matrix[e] for e in test_envs]])

    fig, ax = plt.subplots(figsize=(max(6, len(test_envs) * 1.5), 3))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=test_envs,
                yticklabels=[f"Train: {train_env}"],
                vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Cross-environment accuracy — trained on {train_env}")
    plt.tight_layout()
    plt.savefig("accuracy_matrix.png", dpi=150)
    plt.show()
    print("Saved -> accuracy_matrix.png")

if __name__ == "__main__":

    assert os.path.exists(MODEL_PATH), \
        f"Model not found: {MODEL_PATH} — run train.py first."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    train_env   = checkpoint['train_env']
    env_names   = checkpoint['env_names']
    num_classes = checkpoint['num_classes']
    root_dir    = checkpoint['root_dir']

    print(f"\nModel trained on  : {train_env}")
    print(f"All environments  : {env_names}")
    print(f"Number of classes : {num_classes}")

    model = CNN(input_channels=1, num_classes=num_classes).to(device)
    
    dummy = torch.zeros(1, 1, 32, 32).to(device)
    model(dummy)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("\nModel loaded successfully.")

    test_envs        = [e for e in env_names] # if e != train_env]
    accuracy_matrix  = {}
    confusion_matrices = {}

    print(f"\n{'='*50}")
    print(f"Testing on: {test_envs}")
    print(f"{'='*50}")

    for env_name in test_envs:
        env_dir = os.path.join(root_dir, env_name)
        assert os.path.isdir(env_dir), f"Folder not found: {env_dir}"

        print(f"\n  Environment: {env_name}")
        test_dataset = DopplerDataset(env_dir)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        acc, conf = evaluate(model, test_loader, device, num_classes)
        accuracy_matrix[env_name]    = acc
        confusion_matrices[env_name] = conf
        print(f"  -> Accuracy: {acc:.4f}")

    print()
    plot_confusion_matrices(confusion_matrices, env_names, train_env, num_classes)
    plot_accuracy_matrix(accuracy_matrix, env_names, train_env)

    print(f"\n{'='*50}")
    print(f"Summary — trained on {train_env}")
    print(f"{'='*50}")
    print(f"{'Test environment':<20} {'Accuracy':>10}")
    print("-" * 32)
    for env_name in test_envs:
        print(f"{env_name:<20} {accuracy_matrix[env_name]:>10.4f}")