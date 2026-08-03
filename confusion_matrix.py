import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import CNN
from dataset import DopplerDataset

BATCH_SIZE = 64

# Using the evaluation function from your script[cite: 11]
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

def plot_full_matrix(matrix, env_names, epochs):
    """
    Displays and saves the cross-correlation matrix.
    Rows are test environments, columns are training environments.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Using the 'Blues' palette to match your reference image
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=env_names,
                yticklabels=env_names,
                vmin=0, vmax=1, ax=ax)
    
    # Formatting axes according to the 'Test / Train' format of your image
    ax.set_xlabel("Train")
    ax.set_ylabel("Test")
    ax.set_title("Intra and Inter-Scenario Performance Matrix")
    
    # Move the X-axis labels to the top to match the image style
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    
    # Create the matrix folder if it doesn't exist
    os.makedirs("matrix", exist_ok=True)
    save_path = f"matrix/full_cross_env_accuracy_epochs_{epochs}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFull matrix saved -> {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load an initial model just to extract global metadata (env_names, root_dir)[cite: 11]
    matching_initial_models = [m for m in os.listdir("models") if m.startswith("model_doppler_a_") and m.endswith('.pth')]
    if not matching_initial_models:
        raise FileNotFoundError("No model found for environment a")
    INITIAL_MODEL_PATH = os.path.join("models", matching_initial_models[0])
    
    checkpoint = torch.load(INITIAL_MODEL_PATH, map_location=device, weights_only=False)
    env_names = checkpoint['env_names']
    num_classes = checkpoint['num_classes']
    root_dir = checkpoint['root_dir']
    epochs = checkpoint.get('epochs', 'unknown')
    
    n_envs = len(env_names)
    
    # Initialize the results matrix (Rows=Test, Columns=Train)
    accuracy_matrix = np.zeros((n_envs, n_envs))
    
    print(f"Beginning cross-evaluation on {n_envs} environments: {env_names}")

    # 2. Loop over columns (Training environments)
    for j, train_env in enumerate(env_names):
        matching_models = [m for m in os.listdir("models") if m.startswith(f"model_doppler_{train_env[-1]}_") and m.endswith('.pth')]
        if not matching_models:
            print(f"Missing model ignored for env {train_env}")
            continue
        model_path = os.path.join("models", matching_models[0])
            
        print(f"\n--- Evaluating model trained on: {train_env} ---")
        
        # Load the model specific to this environment
        model = CNN(input_channels=1, num_classes=num_classes).to(device)
        dummy = torch.zeros(1, 1, 32, 32).to(device)
        model(dummy) # Dummy forward pass to initialize LazyLinear[cite: 11]
        
        model_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(model_checkpoint['model_state_dict'])
        model.eval()

        # 3. Loop over rows (Test environments)
        for i, test_env in enumerate(env_names):
            env_dir = os.path.join(root_dir, "doppler_output_" + test_env)
            
            # Ensure the test folder exists
            if not os.path.exists(env_dir):
                continue
                
            test_dataset = DopplerDataset(env_dir)
            test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Accuracy calculation[cite: 11]
            acc = evaluate_accuracy(model, test_loader, device)
            
            # Store in the matrix: i = Test (row), j = Train (column)
            accuracy_matrix[i, j] = acc
            print(f"  -> Test on {test_env:<15} : {acc:.4f}")

    # 4. Generate the matrix image
    plot_full_matrix(accuracy_matrix, env_names, epochs)