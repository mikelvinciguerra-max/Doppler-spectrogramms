import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import CNN
from dataset import DopplerDataset

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

def plot_accuracy_matrix(accuracy_matrix, env_names, train_env):
    test_envs = [e for e in env_names if e != train_env]
    data      = np.array([[accuracy_matrix[e] for e in test_envs]])

    fig, ax = plt.subplots(figsize=(max(6, len(test_envs) * 1.5), 3))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=test_envs,
                yticklabels=[f"Train: {train_env}"],
                vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Cross-environment accuracy — trained on {train_env}")
    plt.tight_layout()
    plt.savefig("matrix/accuracy_matrix.png", dpi=150)
    plt.close()
    print("Saved -> accuracy_matrix.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluating CNN on Doppler profiles")
    parser.add_argument("--env", type=str, default="a", help="Folder name of the training environment")
    args = parser.parse_args()
    
    MODEL_PATH = f"models/model_doppler_{args.env}.pth"
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    train_env   = checkpoint['train_env']
    env_names   = checkpoint['env_names']
    num_classes = checkpoint['num_classes']
    root_dir    = checkpoint['root_dir']

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
        test_dataset = DopplerDataset(env_dir)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        acc = evaluate_accuracy(model, test_loader, device)
        accuracy_matrix[env_name] = acc

    # Summary table
    print(f"\n{'Test environment':<20} {'Accuracy':>10}")
    print("-" * 32)
    for env_name in test_envs:
        print(f"{env_name:<20} {accuracy_matrix[env_name]:>10.4f}")

    plot_accuracy_matrix(accuracy_matrix, env_names, train_env)