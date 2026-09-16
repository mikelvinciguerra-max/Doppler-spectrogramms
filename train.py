import argparse
import gc
import os
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
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


CHECKPOINT_RE = re.compile(
    r'^(?P<env>[^_]+)_classes_(?P<classes>[\d\-]+)_epochs_(?P<epochs>\d+)_kfolds_(?P<kfolds>\d+)\.pth$'
)


def list_checkpoints(model_dir, env_suffix, classes_str, k_folds):
    """
    Return the compatible checkpoints as a list of (epochs, path), sorted by
    increasing epoch count.

    Every .pth of the folder is parsed and kept only if it matches on the three
    identifying fields encoded in its name: the environment letter (first token
    of the filename), the target classes and the number of CV folds. The epoch
    count is the only field allowed to differ.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(model_dir, "*.pth"))):
        match = CHECKPOINT_RE.match(os.path.basename(path))
        if match is None:
            continue
        if match.group('env') != env_suffix:
            continue
        if match.group('classes') != classes_str:
            continue
        if int(match.group('kfolds')) != k_folds:
            continue
        found.append((int(match.group('epochs')), path))
    return sorted(found)


def train(train_loader, valid_loader, epochs, model, criterion, metric, optimizer, device,
          start_epoch=0, total_epochs=None):
    len_train = len(train_loader)
    len_valid = len(valid_loader) if valid_loader is not None else 0
    total_epochs = total_epochs if total_epochs is not None else start_epoch + epochs

    history = {'w': [], 'b': [], 'loss': [], 'val_loss': [], 'score': [], 'val_score': []}
    if valid_loader is not None:
        print(f"Beginning of training with {len_train} batches, validation on {len_valid} batches.")
    else:
        print(f"Beginning of training with {len_train} batches, no validation (final fit on the full pool).")

    for e in range(epochs):
        train_loss, train_score = 0.0, 0.0
        epoch_number = start_epoch + e + 1

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
        history['loss'].append(train_loss / len_train)
        history['score'].append(train_score / len_train)

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
            history['val_loss'].append(valid_loss / len_valid)
            history['val_score'].append(valid_score / len_valid)
            print(f"Epoch {epoch_number:>4}/{total_epochs} - loss : {train_loss/len_train:>9.3f} - score : {train_score/len_train:>9.3f} - val_loss : {valid_loss/len_valid:>9.3f} - val_score : {valid_score/len_valid:>9.3f}")
        else:
            print(f"Epoch {epoch_number:>4}/{total_epochs} - loss : {train_loss/len_train:>9.3f} - score : {train_score/len_train:>9.3f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training CNN on Doppler profiles")
    parser.add_argument("--train_env", type=str, default="doppler_output_a", help="Folder name of the training environment")
    parser.add_argument("--epochs", type=int, default=20, help="TARGET total number of epochs. If a checkpoint with fewer epochs exists, training resumes from it and only runs the remaining epochs.")
    parser.add_argument("--root_dir", type=str, default="/media/mikel/Elements1/MikelVinciguerra/dataset_PC_ehunam/", help="Root dir")
    parser.add_argument("--classes", nargs='+', type=int, default=[0, 1, 2, 3, 4], help="Classes to train on (class 0 is not used)")
    parser.add_argument("--k-folds", type=int, default=1, help="Number of stratified CV folds for the robustness check")
    parser.add_argument("--focal_gamma", type=float, default=0, help="Focal loss focusing parameter (0 = plain weighted cross-entropy)")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (default: 0 for auto-calculation based on dataset size)")
    parser.add_argument("--resume", type=str, default="auto", help="'auto' to resume from the most trained compatible checkpoint STRICTLY BELOW --epochs (a checkpoint already at --epochs is rebuilt and overwritten), 'none' to always start fresh, or an explicit path to a .pth checkpoint")

    args = parser.parse_args()

    ROOT_DIR      = args.root_dir
    TRAIN_ENV     = args.train_env
    ENV_NAMES     = ["a", "b", "c", "d"]
    VALID_CLASSES = [0, 1, 2, 3, 4]
    EPOCHS        = args.epochs
    LR            = 1e-3
    TARGET_CLASSES = sorted({cls for cls in args.classes if cls in VALID_CLASSES})
    K_FOLDS       = args.k_folds

    if K_FOLDS < 1:
        parser.error("--k_folds must be at least 1")
    if EPOCHS < 1:
        parser.error("--epochs must be at least 1")
    if not TARGET_CLASSES:
        parser.error("--classes must contain at least one class between 0 and 4")

    NUM_CLASSES = len(TARGET_CLASSES)
    LABEL_TO_INDEX = {label: idx for idx, label in enumerate(TARGET_CLASSES)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes_str = "-".join(map(str, TARGET_CLASSES))
    # Environment letter: last char of the dataset folder name, and first token
    # of every checkpoint filename. Defined once so search and save cannot drift.
    ENV_SUFFIX = os.path.basename(TRAIN_ENV.rstrip("/\\"))[-1]
    model_dir = "models/tests"

    # ---------------------------------------------------------------------
    # Checkpoint selection: --epochs is the TARGET total number of epochs.
    # We look for the most trained compatible checkpoint whose epoch count m
    # is strictly below EPOCHS, and resume from it for (EPOCHS - m) epochs.
    # ---------------------------------------------------------------------
    resume_path = None

    if args.resume.lower() in ("none", "no", "false", ""):
        print("\n[Resume] Disabled by --resume none. Training will start fresh.")
    elif args.resume == "auto":
        print(f"\n[Auto-Resume] Searching '{model_dir}' for checkpoints with env='{ENV_SUFFIX}', classes='{classes_str}', kfolds={K_FOLDS}...")
        checkpoints = list_checkpoints(model_dir, ENV_SUFFIX, classes_str, K_FOLDS)
        existing = dict(checkpoints)
        # Only checkpoints strictly below the target are eligible: one sitting
        # exactly at (or above) the target is never resumed from, it is rebuilt.
        resumable = [(ep, path) for ep, path in checkpoints if ep < EPOCHS]

        if checkpoints:
            print(f"[Auto-Resume] Compatible checkpoints found (epochs): {[ep for ep, _ in checkpoints]}")

        if EPOCHS in existing:
            print(f"\n[Auto-Resume] A checkpoint already exists for the target of {EPOCHS} epochs: {existing[EPOCHS]}")
            print(f"[Auto-Resume] It will be rebuilt from the most trained checkpoint below {EPOCHS} and overwritten.")

        if resumable:
            ckpt_epochs, resume_path = resumable[-1]
            print(f"\n[Auto-Resume] Resuming from the most trained checkpoint below the target: {resume_path} ({ckpt_epochs} epochs)")
            print(f"[Auto-Resume] Running {EPOCHS - ckpt_epochs} additional epochs to reach {EPOCHS}.")
        else:
            print(f"\n[Auto-Resume] No compatible checkpoint below {EPOCHS} epochs. Training will start fresh.")
    else:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            parser.error(f"--resume path does not exist: {resume_path}")
        print(f"\n[Resume] Using explicit checkpoint: {resume_path}")

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
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    train_valid_idx = np.array(train_valid_idx)
    y_train_valid = np.array(y_train_valid)
    fold_scores = []

    if resume_path is not None:
        print(f"\n[INFO] Skipping cross-validation: resuming training directly from checkpoint.")
    elif K_FOLDS == 1:
        print(f"\nSkipping cross-validation; normal training on {len(train_valid_idx)} samples...")
    else:
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        print(f"\nRunning {K_FOLDS}-fold stratified cross-validation on {len(train_valid_idx)} samples...")

        for fold, (fold_train_pos, fold_valid_pos) in enumerate(skf.split(train_valid_idx, y_train_valid)):
            print(f"\n=== Fold {fold + 1}/{K_FOLDS} ===")

            fold_train_dataset = MappedDataset(dataset, train_valid_idx[fold_train_pos], LABEL_TO_INDEX)
            fold_valid_dataset = MappedDataset(dataset, train_valid_idx[fold_valid_pos], LABEL_TO_INDEX)

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

    final_train_dataset = MappedDataset(dataset, train_valid_idx, LABEL_TO_INDEX)
    final_train_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = CNN(input_channels=1, num_classes=NUM_CLASSES).to(device)
    dummy = torch.zeros(1, 1, 32, 32).to(device)
    model(dummy)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch_count = 0
    full_history = {'loss': [], 'score': []}

    if resume_path is not None:
        print(f"\n[INFO] Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        ckpt_classes = checkpoint.get('target_classes')
        if ckpt_classes is not None and list(ckpt_classes) != list(TARGET_CLASSES):
            raise ValueError(
                f"Checkpoint was trained on classes {list(ckpt_classes)} but this run targets {TARGET_CLASSES}. "
                f"Use --resume none to start fresh."
            )

        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[INFO] Model AND Optimizer states loaded successfully.")
        else:
            print("[WARNING] Model weights loaded, but no optimizer state found. Starting Adam fresh.")

        if 'train_loss_history' in checkpoint:
            full_history['loss'] = list(checkpoint['train_loss_history'])
            full_history['score'] = list(checkpoint.get('train_score_history', []))

        start_epoch_count = int(checkpoint.get('epochs', 0))
    else:
        print(f"\nFitting final model on the full train+valid pool ({len(train_valid_idx)} samples)...")

    epochs_to_run = max(0, EPOCHS - start_epoch_count)
    total_epochs_completed = start_epoch_count + epochs_to_run

    if epochs_to_run > 0:
        if start_epoch_count > 0:
            print(f"[INFO] Resuming at epoch {start_epoch_count + 1}, running {epochs_to_run} epochs up to the target of {EPOCHS}...")
        new_history = train(final_train_loader, None, epochs_to_run, model, criterion, metric,
                            optimizer, device, start_epoch=start_epoch_count, total_epochs=total_epochs_completed)
        full_history['loss'].extend(new_history['loss'])
        full_history['score'].extend(new_history['score'])
    else:
        print(f"[INFO] Checkpoint already has {start_epoch_count} epochs >= target {EPOCHS}. Skipping training, evaluating only.")

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

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_env': TRAIN_ENV,
        'env_names': ENV_NAMES,
        'num_classes': NUM_CLASSES,
        'target_classes': TARGET_CLASSES,
        'root_dir': ROOT_DIR,
        'epochs': total_epochs_completed,
        'k_folds': K_FOLDS,
        'focal_gamma': args.focal_gamma,
        'test_indices': test_idx,
        'cv_val_score_mean': float(fold_scores.mean()) if fold_scores.size else None,
        'cv_val_score_std': float(fold_scores.std()) if fold_scores.size else None,
        'train_loss_history': full_history['loss'],
        'train_score_history': full_history['score']
    }

    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"{model_dir}/{ENV_SUFFIX}_classes_{classes_str}_epochs_{total_epochs_completed}_kfolds_{K_FOLDS}.pth"
    torch.save(checkpoint, model_filename)
    print(f"Model saved -> {model_filename}")