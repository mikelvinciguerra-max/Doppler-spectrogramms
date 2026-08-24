import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import CNN
from dataset import DopplerDataset

BATCH_SIZE = 64

def filter_dataset_by_classes(dataset, target_classes):
    target_classes = set(target_classes)
    filtered_indices = [
        idx for idx in range(len(dataset))
        if dataset[idx][1].item() in target_classes
    ]
    return Subset(dataset, filtered_indices)

# Using the evaluation function from your script[cite: 11]
def evaluate_accuracy(model, loader, device, target_classes=None):
    model.eval()
    correct, total = 0, 0
    target_set = set(target_classes) if target_classes is not None else None
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            if target_set is not None:
                mask = torch.tensor([label.item() in target_set for label in y], device=device)
                if not torch.any(mask):
                    continue
                y = y[mask]
                preds = preds[mask]
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

def plot_full_matrix(matrix, env_names, epochs, target_classes):
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
    classes_str = "-".join(map(str, sorted(target_classes)))
    ax.set_title(f"Intra and Inter-Scenario Performance Matrix (Classes {classes_str})")
    
    # Move the X-axis labels to the top to match the image style
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    
    # Create the matrix folder if it doesn't exist
    os.makedirs("matrix", exist_ok=True)
    classes_str = "-".join(map(str, sorted(target_classes)))
    save_path = f"matrix/full_cross_env_accuracy_classes_{classes_str}_epochs_{epochs}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFull matrix saved -> {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-environment evaluation matrix")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs used to train the models")
    parser.add_argument("--classes", nargs='+', type=int, default=[0, 1, 2, 3, 4], help="Classes used to train the models")
    args = parser.parse_args()
    target_classes = sorted(args.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_str = "-".join(map(str, target_classes))

    # 1. Load an initial model just to extract global metadata (env_names, root_dir)
    model_prefix = f"model_doppler_a_classes_{classes_str}_epochs_{args.epochs}.pth"
    INITIAL_MODEL_PATH = os.path.join("models", model_prefix)
    if not os.path.exists(INITIAL_MODEL_PATH):
        matching_initial_models = [
            m for m in os.listdir("models")
            if m.startswith("model_doppler_a_classes_") and m.endswith(f"_epochs_{args.epochs}.pth")
        ]
        if not matching_initial_models:
            raise FileNotFoundError(f"No model found for environment a with classes {sorted(args.classes)}")
        raise FileNotFoundError(f"No model found for environment a at {args.epochs} epochs with classes {sorted(args.classes)}. Available: {matching_initial_models}")

    checkpoint = torch.load(INITIAL_MODEL_PATH, map_location=device, weights_only=False)
    env_names = checkpoint['env_names']
    num_classes = checkpoint['num_classes']
    root_dir = checkpoint['root_dir']
    epochs = checkpoint.get('epochs', args.epochs)
    if 'target_classes' in checkpoint:
        target_classes = sorted(checkpoint['target_classes'])
    print(f"Using target classes: {target_classes}")
    
    n_envs = len(env_names)
    
    # Initialize the results matrix (Rows=Test, Columns=Train)
    accuracy_matrix = np.zeros((n_envs, n_envs))
    
    print(f"Beginning cross-evaluation on {n_envs} environments: {env_names}")

    # 2. Loop over columns (Training environments)
    for j, train_env in enumerate(env_names):
        expected_model = f"model_doppler_{train_env[-1]}_classes_{classes_str}_epochs_{args.epochs}.pth"
        model_path = os.path.join("models", expected_model)
        if not os.path.exists(model_path):
            matching_models = [
                m for m in os.listdir("models")
                if m.startswith(f"model_doppler_{train_env[-1]}_classes_{classes_str}_") and m.endswith(f"_epochs_{args.epochs}.pth")
            ]
            print(f"Missing model ignored for env {train_env} at {args.epochs} epochs with classes {sorted(args.classes)}. Available: {matching_models}")
            continue
            
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
            filtered_test_dataset = filter_dataset_by_classes(test_dataset, target_classes)
            test_loader = DataLoader(filtered_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Accuracy calculation[cite: 11]
            acc = evaluate_accuracy(model, test_loader, device, target_classes)
            
            # Store in the matrix: i = Test (row), j = Train (column)
            accuracy_matrix[i, j] = acc
            print(f"  -> Test on {test_env:<15} : {acc:.4f}")

    # 4. Generate the matrix image
    plot_full_matrix(accuracy_matrix, env_names, epochs, target_classes)