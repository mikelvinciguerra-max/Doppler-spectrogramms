import argparse
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from dataset import load_dataset_with_cache
from model import CNN


def metric(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    correct = (pred_classes == target) 
    return correct.sum().item() / len(target)


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
        return x, torch.tensor(self.label_to_index[label], dtype=torch.long)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017): down-weights examples the
    model already gets right so gradient focuses on the hard ones.

    p_t is recovered from an UNWEIGHTED cross-entropy (weighting it first
    would distort exp(-ce_loss) into p_t ** alpha instead of p_t). alpha is
    applied afterwards, as a separate per-sample multiplier. With gamma=0
    this is exactly a weighted cross-entropy.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            loss = alpha_t * loss
            if self.reduction == 'mean':
                return loss.sum() / alpha_t.sum()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def train(train_loader, valid_loader, epochs, model, criterion, metric, optimizer, device):
    len_train = len(train_loader)
    len_valid = len(valid_loader) if valid_loader is not None else 0

    history = {'w': [], 'b': [], 'loss': [], 'val_loss': [], 'score': [], 'val_score': []}
    if valid_loader is not None:
        print(f"Beginning of training with {len_train} batches, validation on {len_valid} batches.")
    else:
        print(f"Beginning of training with {len_train} batches, no validation (final fit on the full pool).")

    for e in range(epochs):
        train_loss, train_score = 0.0, 0.0
        
        # Training
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            train_pred = model(x)
            loss = criterion(train_pred, y)
            score = metric(train_pred, y)
            train_loss += loss.item()
            train_score += score

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        params = list(model.parameters())
        history['w'].append(params[0].mean().item())
        history['b'].append(params[1].mean().item())
        history['loss'].append(train_loss/len_train)
        history['score'].append(train_score/len_train)

        # Validation
        if valid_loader is not None:
            valid_loss, valid_score = 0.0, 0.0
            model.eval()
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(device), y.to(device)
                    valid_pred = model(x)
                    loss = criterion(valid_pred, y)
                    score = metric(valid_pred, y)
                    valid_loss += loss.item()
                    valid_score += score
            history['val_loss'].append(valid_loss/len_valid)
            history['val_score'].append(valid_score/len_valid)
            print(f"Epoch {e+1:>4}/{epochs} - loss : {train_loss/len_train:>9.3f} - score : {train_score/len_train:>9.3f} - val_loss : {valid_loss/len_valid:>9.3f} - val_score : {valid_score/len_valid:>9.3f}")
        else:
            print(f"Epoch {e+1:>4}/{epochs} - loss : {train_loss/len_train:>9.3f} - score : {train_score/len_train:>9.3f}")

    return history


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Training CNN on Doppler profiles")
    parser.add_argument("--train_env", type=str, default="doppler_output_a", help="Folder name of the training environment")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--root_dir", type=str, default="/media/mikel/Elements1/MikelVinciguerra/dataset_PC_ehunam/" , help="Root dir")
    parser.add_argument("--classes", nargs='+', type=int, default=[0, 1, 2, 3, 4], help="Classes to train on (class 0 is not used)") 
    parser.add_argument("--k-folds", type=int, default=1, help="Number of stratified CV folds for the robustness check")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss focusing parameter (0 = plain weighted cross-entropy)")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (default: 0 for auto-calculation based on dataset size)")
    args = parser.parse_args()

    ROOT_DIR     = args.root_dir    
    TRAIN_ENV    = args.train_env                            
    ENV_NAMES    = ["a", "b", "c", "d"] 
    VALID_CLASSES = [0, 1, 2, 3, 4]
    EPOCHS       = args.epochs    
    LR           = 1e-3
    TARGET_CLASSES = sorted({cls for cls in args.classes if cls in VALID_CLASSES})
    K_FOLDS      = args.k_folds
    if K_FOLDS < 1:
        parser.error("--k_folds must be at least 1")
    if not TARGET_CLASSES:
        parser.error("--classes must contain at least one class between 0 and 4")
    NUM_CLASSES = len(TARGET_CLASSES)
    LABEL_TO_INDEX = {label: idx for idx, label in enumerate(TARGET_CLASSES)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_path = os.path.join(ROOT_DIR, TRAIN_ENV)
    dataset = load_dataset_with_cache(dataset_path)

    print(f"Filtering dataset to train only on classes: {TARGET_CLASSES}")

    filtered_indices = []
    filtered_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in TARGET_CLASSES:
            filtered_indices.append(i)
            filtered_labels.append(int(label))

    unique_classes, class_counts = np.unique(filtered_labels, return_counts=True)
    print(f"Class counts: {dict(zip(unique_classes.tolist(), [int(c) for c in class_counts.tolist()]))}")
    total = class_counts.sum()
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls_idx, count in zip(unique_classes, class_counts):
        weights[LABEL_TO_INDEX[int(cls_idx)]] = total / (len(unique_classes) * count)
    class_weights = torch.tensor(weights, device=device)
    print(f"Class weights: {weights}")
    print(f"Focal gamma: {args.focal_gamma}")
    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)

    train_valid_idx, test_idx, y_train_valid, y_test = train_test_split(
        filtered_indices, filtered_labels,
        test_size=0.175, stratify=filtered_labels, random_state=42
    )

    # ADDED: Dynamic Batch Size Calculation
    num_train_samples = len(train_valid_idx)
    if args.batch_size > 0:
        BATCH_SIZE = args.batch_size
        print(f"\n[Auto-Config] Using manually provided BATCH_SIZE: {BATCH_SIZE}")
    else:
        if num_train_samples <= 120:
            BATCH_SIZE = 8
        elif num_train_samples <= 300:
            BATCH_SIZE = 16
        elif num_train_samples <= 600:
            BATCH_SIZE = 32
        else:
            BATCH_SIZE = 64
        print(f"\n[Auto-Config] Training pool size is {num_train_samples}. Automatically setting BATCH_SIZE to {BATCH_SIZE} to ensure sufficient gradient updates.")

    test_set = MappedDataset(dataset, test_idx, LABEL_TO_INDEX)
    # MODIFIED: Applying the dynamically calculated BATCH_SIZE
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    train_valid_idx = np.array(train_valid_idx)
    y_train_valid = np.array(y_train_valid)
    fold_scores = []

    if K_FOLDS == 1:
        print(f"\nSkipping cross-validation; normal training on {len(train_valid_idx)} samples...")
    else:
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        print(f"\nRunning {K_FOLDS}-fold stratified cross-validation on {len(train_valid_idx)} samples...")

        for fold, (fold_train_pos, fold_valid_pos) in enumerate(skf.split(train_valid_idx, y_train_valid)):
            print(f"\n=== Fold {fold + 1}/{K_FOLDS} ===")

            fold_train_dataset = MappedDataset(dataset, train_valid_idx[fold_train_pos], LABEL_TO_INDEX)
            fold_valid_dataset = MappedDataset(dataset, train_valid_idx[fold_valid_pos], LABEL_TO_INDEX)
            
            # MODIFIED: Applying the dynamically calculated BATCH_SIZE
            fold_train_loader = DataLoader(fold_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
            fold_valid_loader = DataLoader(fold_valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

            fold_model = CNN(input_channels=1, num_classes=NUM_CLASSES).to(device)
            dummy = torch.zeros(1, 1, 32, 32).to(device)
            fold_model(dummy)

            fold_optimizer = optim.Adam(fold_model.parameters(), lr=LR)
            fold_history = train(fold_train_loader, fold_valid_loader, EPOCHS, fold_model,
                                  criterion, metric, fold_optimizer, device)
            fold_scores.append(fold_history['val_score'][-1])

            del fold_train_loader, fold_valid_loader, fold_model, fold_history
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fold_scores = np.array(fold_scores)
    if fold_scores.size:
        print(f"\nCV val_score: {fold_scores.mean():.4f} +/- {fold_scores.std():.4f} (per fold: {[f'{s:.4f}' for s in fold_scores]})")

    print(f"\nFitting final model on the full train+valid pool ({len(train_valid_idx)} samples)...")

    final_train_dataset = MappedDataset(dataset, train_valid_idx, LABEL_TO_INDEX)
    # MODIFIED: Applying the dynamically calculated BATCH_SIZE
    final_train_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = CNN(input_channels=1, num_classes=NUM_CLASSES).to(device)
    dummy = torch.zeros(1, 1, 32, 32).to(device)
    model(dummy)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = train(final_train_loader, None, EPOCHS, model, criterion, metric, optimizer, device)

    del final_train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    test_score = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_score += metric(model(x), y)
    print(f"\nFinal test score: {test_score / len(test_loader):.4f}")

    del test_loader, test_set, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    classes_str = "-".join(map(str, TARGET_CLASSES))

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_env': TRAIN_ENV,
        'env_names': ENV_NAMES,
        'num_classes': NUM_CLASSES,
        'target_classes': TARGET_CLASSES,
        'root_dir': ROOT_DIR,
        'epochs': EPOCHS,
        'k_folds': K_FOLDS,
        'focal_gamma': args.focal_gamma,
        'test_indices': test_idx,
        'cv_val_score_mean': float(fold_scores.mean()) if fold_scores.size else None,
        'cv_val_score_std': float(fold_scores.std()) if fold_scores.size else None,
        'train_loss_history': history['loss'],
        'train_score_history': history['score']
    }
    
    model_dir = "models/tests"
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"{model_dir}/{TRAIN_ENV[-1]}_classes_{classes_str}_epochs_{EPOCHS}_kfolds_{K_FOLDS}.pth"
    torch.save(checkpoint, model_filename)
    print(f"Model saved -> {model_filename}")