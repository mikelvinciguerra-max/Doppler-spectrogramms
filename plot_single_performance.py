import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# --- Imports of your custom modules ---
from model import CNN
from dataset import DopplerDataset

BATCH_SIZE = 64

def get_accuracy(model, loader, device):
    """Calculates the overall accuracy of a model on a dataloader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

def plot_single_performance_matrix(matrix_data, test_env_names, train_env_name):
    """
    Generates a scientific table plot for a single training environment.
    matrix_data : 2D numpy array of shape (N_test_envs, 1)
    """
    # Narrower figure since we only have 1 column
    fig, ax = plt.subplots(figsize=(3, 0.8 * len(test_env_names)))

    # 1. Create the Heatmap using the "Blues" palette
    sns.heatmap(matrix_data, annot=False, cmap="Blues", cbar=False,
                xticklabels=[train_env_name], yticklabels=test_env_names, 
                vmin=0, vmax=1.0, ax=ax)

    # 2. Add custom annotations (Bold if the value is high)
    for i in range(matrix_data.shape[0]):
        val = matrix_data[i, 0]
        weight = 'bold' if val >= 0.7 else 'normal'
        text = f"{val:.2f}".rstrip('0').rstrip('.') if val != 0 else "0"
        ax.text(0.5, i + 0.5, text,
                ha="center", va="center", color="black",
                fontsize=12, weight=weight)

    # 3. Format axes to look like a scientific table
    ax.xaxis.tick_top()
    ax.set_xticklabels([train_env_name], fontsize=12, weight='bold')
    ax.set_yticklabels(test_env_names, fontsize=12, weight='bold', rotation=0)
    
    ax.tick_params(left=False, top=False)

    # Add the "Test / Train" label (adjusted coordinates for single column)
    ax.text(-0.5, -0.2, "Test / Train", fontsize=12, weight='bold', ha='center', va='center')

    # Add horizontal lines to frame the table
    ax.axhline(0, color='black', linewidth=1.5)
    ax.axhline(matrix_data.shape[0], color='black', linewidth=1.5)

    # 4. Global title
    plt.title("PERFORMANCE MATRIX RESTRICTED TO ONE ENVIRONMENT.",
              pad=35, fontsize=10, loc='center')
    
    plt.tight_layout()
    os.makedirs("matrix", exist_ok=True)
    save_path = "matrix/single_performance_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating CNN on Doppler profiles")
    parser.add_argument("--env", type=str, default="a", help="Folder name of the training environment")
    args = parser.parse_args()
    
    MODEL_PATH = f"models/model_doppler_{args.env}.pth"
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    train_env   = checkpoint['train_env']
    env_names   = checkpoint['env_names']
    num_classes = checkpoint['num_classes']
    root_dir    = checkpoint['root_dir']

    print(f"Loading model trained on: {train_env}...")

    # Rebuild model
    model = CNN(input_channels=1, num_classes=num_classes).to(device)
    dummy = torch.zeros(1, 1, 32, 32).to(device)
    model(dummy) 
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create a column matrix: shape (Number of envs, 1)
    accuracy_matrix = np.zeros((len(env_names), 1))

    print("\nEvaluating on all environments...")
    
    for i, test_env in enumerate(env_names):
        env_dir = os.path.join(root_dir, "doppler_output_" + test_env)
        
        if not os.path.isdir(env_dir):
            print(f"[!] Directory not found: {env_dir} - Skipping.")
            continue

        test_dataset = DopplerDataset(env_dir)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        acc = get_accuracy(model, test_loader, device)
        accuracy_matrix[i, 0] = acc
        print(f"  -> Test on {test_env:<20}: {acc:.4f}")

    # Plot the 1-column scientific table
    plot_single_performance_matrix(accuracy_matrix, env_names, train_env)