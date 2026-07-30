import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os
import glob
from pathlib import Path


class DopplerDataset(Dataset):
    """
    Loads Doppler profiles on demand (lazy loading).
    Each .txt file contains a pickled numpy array of shape (N, 1024).
    Each row = one Doppler profile = one sample.
    Label is extracted from the filename: MC1_01A_**1**_E -> label=1
    """

    def __init__(self, root_dir, transform=None):
        """
        root_dir  : folder containing all .txt files
        transform : optional data augmentation
        """
        self.transform = transform
        self.samples   = []

        files = sorted(glob.glob(os.path.join(root_dir, "*.txt")))
        assert files, f"No .txt files found in {root_dir}"

        for path in files:
            label = self._parse_label(path)
            if label is None:
                continue
            with open(path, 'rb') as f:
                arr = pickle.load(f)
            n_rows = arr.shape[0]
            for i in range(n_rows):
                self.samples.append((path, i, label))

        print(f"Dataset: {len(self.samples)} samples "
              f"from {len(files)} files")

    def _parse_label(self, path):
        """
        Extracts the number of people from the filename.
        MC1_01A_1_E_________01.txt -> 1
        Returns None if the filename does not match the expected format.
        """
        name  = Path(path).stem      
        parts = name.split('_')
        try:
            label = int(parts[2])    
            return label
        except (IndexError, ValueError):
            print(f"[!] Unrecognized filename: {path}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, row, label = self.samples[idx]

        with open(path, 'rb') as f:
            arr = pickle.load(f)     

        profile = arr[row].astype(np.float32)  
        profile = profile.reshape(1, 32, 32)

        if self.transform:
            profile = self.transform(profile)

        x = torch.from_numpy(profile)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ------------------------------------------------------------------ #
#  2. CACHED DATASET (optional)                                        #
#     If your RAM allows it, enable caching for faster training        #
# ------------------------------------------------------------------ #

class CachedDopplerDataset(DopplerDataset):
    """
    Cached version: loads all files into RAM once at startup.
    Faster than lazy loading but uses more memory.
    Only use this if you have enough RAM.
    """

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        print("Loading cache into RAM...")
        self._cache = {}
        paths = set(p for p, _, _ in self.samples)
        for path in paths:
            with open(path, 'rb') as f:
                self._cache[path] = pickle.load(f).astype(np.float32)
        print(f"Cache loaded: {len(self._cache)} files")

    def __getitem__(self, idx):
        path, row, label = self.samples[idx]
        profile = self._cache[path][row].reshape(1, 32, 32)
        if self.transform:
            profile = self.transform(profile)
        x = torch.from_numpy(profile.copy())
        y = torch.tensor(label, dtype=torch.long)
        return x, y    


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        super(CNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5),
            nn.Mish(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.Mish(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.Mish(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(p=0.2),

            nn.Flatten(),

            nn.LazyLinear(out_features=256),
            nn.Mish(),

            nn.Dropout(p=0.3),

            nn.Linear(in_features=256, out_features=128),
            nn.Mish(),

            nn.Linear(in_features=128, out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.nn(x)
    

def metric(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    correct = (pred_classes == target) 
    return correct.sum().item() / len(target)

def train(train_dataset, valid_dataset, batch_size, epochs, model, criterion, metric, optimizer):
    len_train = len(train_loader)
    len_valid = len(valid_loader)

    history = {'w': [], 'b': [], 'loss': [], 'val_loss': [], 'score': [], 'val_score': []}
    print(f"Beginning of training with {len_train} examples, validation on {len_valid} examples.")

    for e in range(epochs):
        train_loss = 0.0
        train_score = 0.0
        valid_loss = 0.0
        valid_score = 0.0
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

        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                valid_pred = model(x)
                loss = criterion(valid_pred, y)
                score = metric(valid_pred, y)
                valid_loss += loss.item()
                valid_score += score

        params = list(model.parameters())
        history['w'].append(params[0].mean().item())
        history['b'].append(params[1].mean().item())
        history['loss'].append(train_loss/len_train)
        history['score'].append(train_score/len_train)
        history['val_loss'].append(valid_loss/len_valid)
        history['val_score'].append(valid_score/len_valid)
        print(f"Epoch {e+1:>4}/{epochs} - loss : {train_loss/len_train:>9.3f} - score : {train_score/len_train:>9.3f} - val_loss : {valid_loss/len_valid:>9.3f} - val_score : {valid_score/len_valid:>9.3f}")

    return history


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Entraînement CNN sur les profils Doppler")
    parser.add_argument(
        "--train_env", 
        type=str, 
        default="doppler_output01", 
        help="Folder name of the training environment (default: doppler_output01)"
    )
    args = parser.parse_args()

    ROOT_DIR    = "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/"     
    TRAIN_ENV    = args.train_env                            
    ENV_NAMES    = ["doppler_output01", "doppler_output02", "doppler_output04", "doppler_output05", "doppler_output06"] 
    NUM_CLASSES  = 5
    BATCH_SIZE   = 64
    EPOCHS       = 20
    LR           = 1e-3

    env_names = ENV_NAMES
    env_dirs  = [os.path.join(ROOT_DIR, e) for e in ENV_NAMES]

    for name, d in zip(env_names, env_dirs):
        assert os.path.isdir(d), f"Folder not found: {d}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DopplerDataset(ROOT_DIR+TRAIN_ENV)

    n       = len(dataset)
    n_train = int(n * 0.65)
    n_valid = int(n * 0.175)
    n_test  = n - n_train - n_valid

    train_set, valid_set, test_set = random_split(dataset, [n_train, n_valid, n_test])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    model     = CNN(input_channels=1, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    history = train(train_loader, valid_loader, BATCH_SIZE, EPOCHS, model, criterion, metric, optimizer)

    model.eval()
    test_score = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_score += metric(model(x), y)
    print(f"\nFinal test score: {test_score / len(test_loader):.4f}")

    torch.save(model.state_dict(), "model_doppler.pth")
    print("Model saved -> model_doppler.pth")