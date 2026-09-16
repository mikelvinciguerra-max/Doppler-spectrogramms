import torch
import matplotlib.pyplot as plt
from pathlib import Path

checkpoint_path = Path("models/tests/a_classes_0-1-2-3-4_epochs_20_kfolds_1.pth")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

losses = checkpoint['train_loss_history']
train_env = checkpoint.get('train_env', 'unknown')
target_classes = '-'.join(map(str, checkpoint.get('target_classes', 'unknown')))
epochs = checkpoint.get('epochs', len(losses))
k_folds = checkpoint.get('k_folds', 'unknown')
focal_gamma = checkpoint.get('focal_gamma', 'unknown')

plot_stem = (
	f"loss_curve_{train_env[-1]}_classes_{target_classes}"
	f"_epochs_{epochs}_kfolds_{k_folds}_focal_gamma_{focal_gamma}"
)
plot_path = f"plots/loss/{plot_stem}.png"

plt.plot(losses, label='Train Loss', marker='o')
plt.title("Training Loss Curve")
plt.suptitle(
	f"Environment: {train_env[-1]} | Classes: {target_classes} | "
	f"Epochs: {epochs} | K-folds: {k_folds} | Focal gamma: {focal_gamma}",
	fontsize=9,
	y=0.98,
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Loss curve saved to {plot_path}")