import torch
from torch.utils.data import Dataset
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

        print(f"Dataset: {len(self.samples)} samples from {len(files)} files")

    def _parse_label(self, path):
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


class CachedDopplerDataset(DopplerDataset):
    """
    Cached version: loads all files into RAM once at startup.
    Faster than lazy loading but uses more memory.
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