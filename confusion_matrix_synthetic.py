import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import load_dataset_with_cache
from model import CNN


MODEL_PATH = "models/synthetic/model_doppler_c_classes_0-1-2-3-4_epochs_40_kfolds_1.pth"
OUTPUT_PATH = "matrix/synthetic_confusion_matrix_classes_0-1-2-3-4_epochs_40.png"
BATCH_SIZE = 64


class EnhancedDataset(Dataset):
    def __init__(self, base_dataset, indices, target_classes, input_channels):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.label_to_index = {label: index for index, label in enumerate(target_classes)}
        self.input_channels = input_channels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, label = self.base_dataset[self.indices[index]]
        if self.input_channels == 1:
            return x, torch.tensor(self.label_to_index[int(label.item())], dtype=torch.long)
        grad_frequency, grad_time = torch.gradient(x[0], dim=(0, 1))
        enhanced = torch.stack([x[0], grad_frequency, grad_time], dim=0)
        return enhanced, torch.tensor(self.label_to_index[int(label.item())], dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    target_classes = sorted(checkpoint["target_classes"])
    input_channels = checkpoint["model_state_dict"]["nn.0.weight"].shape[1]

    dataset_path = os.path.join(checkpoint["root_dir"], "doppler_output_synthetic")
    dataset = load_dataset_with_cache(dataset_path)
    test_dataset = EnhancedDataset(dataset, checkpoint["test_indices"], target_classes, input_channels)
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN(input_channels=input_channels, num_classes=checkpoint["num_classes"]).to(device)
    model(torch.zeros(1, input_channels, 32, 32, device=device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = np.zeros((len(target_classes), len(target_classes)), dtype=np.int64)
    with torch.no_grad():
        for inputs, labels in loader:
            predictions = torch.argmax(model(inputs.to(device)), dim=1).cpu().numpy()
            for true_label, predicted_label in zip(labels.numpy(), predictions):
                confusion[true_label, predicted_label] += 1

    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=target_classes,
        yticklabels=target_classes,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix - Synthetic Spectrograms")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()

    accuracy = np.trace(confusion) / confusion.sum()
    print("Raw confusion matrix:")
    print(confusion)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Normalized matrix saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()