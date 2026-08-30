import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from dataset import DopplerDataset
from model import CNN

def metric(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    correct = (pred_classes == target) 
    return correct.sum().item() / len(target)

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
    parser.add_argument("--classes", nargs='+', type=int, default=[0, 1, 2, 3, 4], help="Classes to train on") 
    parser.add_argument("--k_folds", type=int, default=5, help="Number of stratified CV folds for the robustness check")
    args = parser.parse_args()

    ROOT_DIR     = args.root_dir    
    TRAIN_ENV    = args.train_env                            
    ENV_NAMES    = ["a", "b", "c", "d"] 
    NUM_CLASSES  = 5
    BATCH_SIZE   = 64
    EPOCHS       = args.epochs
    LR           = 1e-4
    TARGET_CLASSES = sorted(args.classes) 
    K_FOLDS      = args.k_folds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_path = os.path.join(ROOT_DIR, TRAIN_ENV)
    dataset = DopplerDataset(dataset_path)

    print(f"Filtering dataset to train only on classes: {TARGET_CLASSES}")

    filtered_indices = []
    filtered_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in TARGET_CLASSES:
            filtered_indices.append(i)
            filtered_labels.append(label)

    # Class weights (inverse frequency) to compensate for imbalance without discarding data
    unique_classes, class_counts = np.unique(filtered_labels, return_counts=True)
    print(f"Class counts: {dict(zip(unique_classes.tolist(), class_counts.tolist()))}")
    total = class_counts.sum()
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls, count in zip(unique_classes, class_counts):
        weights[cls] = total / (len(unique_classes) * count)
    class_weights = torch.tensor(weights, device=device)
    print(f"Class weights: {weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Held-out test set (17.5%): untouched by the cross-validation below and by the final
    # fit, so confusion_matrix.py can evaluate it as genuinely unseen data.
    train_valid_idx, test_idx, y_train_valid, y_test = train_test_split(
        filtered_indices, filtered_labels, 
        test_size=0.175, stratify=filtered_labels, random_state=42
    )
    test_set = Subset(dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Stratified K-Fold on the remaining 82.5%: gives a fold-averaged validation score
    # instead of relying on a single train/valid split that can be lucky or unlucky.
    train_valid_idx = np.array(train_valid_idx)
    y_train_valid = np.array(y_train_valid)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    print(f"\nRunning {K_FOLDS}-fold stratified cross-validation on {len(train_valid_idx)} samples...")
    fold_scores = []

    for fold, (fold_train_pos, fold_valid_pos) in enumerate(skf.split(train_valid_idx, y_train_valid)):
        print(f"\n=== Fold {fold + 1}/{K_FOLDS} ===")

        fold_train_loader = DataLoader(Subset(dataset, train_valid_idx[fold_train_pos]), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
        fold_valid_loader = DataLoader(Subset(dataset, train_valid_idx[fold_valid_pos]), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        fold_model = CNN(input_channels=1, num_classes=NUM_CLASSES).to(device)
        dummy = torch.zeros(1, 1, 32, 32).to(device)
        fold_model(dummy)

        fold_optimizer = optim.Adam(fold_model.parameters(), lr=LR)
        fold_history = train(fold_train_loader, fold_valid_loader, EPOCHS, fold_model,
                              criterion, metric, fold_optimizer, device)
        fold_scores.append(fold_history['val_score'][-1])

    fold_scores = np.array(fold_scores)
    print(f"\nCV val_score: {fold_scores.mean():.4f} +/- {fold_scores.std():.4f} (per fold: {[f'{s:.4f}' for s in fold_scores]})")

    # Final model: fit on the whole train+valid pool (all folds combined, no internal
    # validation split) using the hyperparameters just validated above. This is the model
    # that gets saved and used downstream by confusion_matrix.py — the CV loop above is a
    # diagnostic step, not the deployed model.
    print(f"\nFitting final model on the full train+valid pool ({len(train_valid_idx)} samples)...")

    final_train_loader = DataLoader(Subset(dataset, train_valid_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = CNN(input_channels=1, num_classes=NUM_CLASSES).to(device)
    dummy = torch.zeros(1, 1, 32, 32).to(device)
    model(dummy)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = train(final_train_loader, None, EPOCHS, model, criterion, metric, optimizer, device)

    model.eval()
    test_score = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_score += metric(model(x), y)
    print(f"\nFinal test score: {test_score / len(test_loader):.4f}")

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
        'test_indices': test_idx,
        'cv_val_score_mean': float(fold_scores.mean()),
        'cv_val_score_std': float(fold_scores.std())
    }
    
    os.makedirs("models", exist_ok=True)
    model_filename = f"models/model_doppler_{TRAIN_ENV[-1]}_classes_{classes_str}_epochs_{EPOCHS}_kfolds_{K_FOLDS}.pth"
    torch.save(checkpoint, model_filename)
    print(f"Model saved -> {model_filename}")