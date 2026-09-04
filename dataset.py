import gc
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

    Label convention:
      - if the 4th field is "E", it means 0 person -> label = 0
      - otherwise the number of people is the number of letters in the next field (PC_xxx)
        example: MC1_01B_1_PC_ab_... -> label = 2
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
        name = Path(path).stem
        parts = name.split('_')
        try:
            if len(parts) >= 5 and parts[3] == 'E':
                return 0
            if len(parts) >= 5 and parts[3] == 'PC':
                return len(parts[4])
            print(f"[!] Unrecognized filename format: {path}")
            return None
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
        self.transform = transform
        self.samples = []
        print("Loading cache into RAM...")
        self._cache = {}

        files = sorted(glob.glob(os.path.join(root_dir, "*.txt")))
        assert files, f"No .txt files found in {root_dir}"

        for path in files:
            label = self._parse_label(path)
            if label is None:
                continue
            with open(path, 'rb') as f:
                arr = pickle.load(f).astype(np.float32)
            self._cache[path] = arr
            for row in range(arr.shape[0]):
                self.samples.append((path, row, label))

        print(f"Dataset: {len(self.samples)} samples from {len(files)} files")
        print(f"Cache loaded: {len(self._cache)} files")

    def clear_cache(self):
        self._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.clear_cache()
        except Exception:
            pass

    def __getitem__(self, idx):
        path, row, label = self.samples[idx]
        profile = self._cache[path][row].reshape(1, 32, 32)
        if self.transform:
            profile = self.transform(profile)
        x = torch.from_numpy(profile.copy())
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def get_available_ram_mb():
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 8192


def load_dataset_with_cache(dataset_path, safety_factor=1.5, ram_fraction=0.35):
    """Use the cached dataset when its estimated RAM use fits safely."""
    files = sorted(glob.glob(os.path.join(dataset_path, '*.txt')))
    if not files:
        return DopplerDataset(dataset_path)

    dataset_size_mb = sum(os.path.getsize(path) for path in files) / (1024 * 1024)
    available_ram_mb = get_available_ram_mb()
    required_ram_mb = dataset_size_mb * safety_factor
    cache_budget_mb = available_ram_mb * ram_fraction

    print(
        f"RAM estimate: ~{required_ram_mb:.1f} MB required for cache, "
        f"~{available_ram_mb} MB available."
    )

    if required_ram_mb <= cache_budget_mb:
        try:
            print(f"Using CachedDopplerDataset for {len(files)} files ({dataset_size_mb:.1f} MB)")
            return CachedDopplerDataset(dataset_path)
        except (MemoryError, OSError, RuntimeError) as error:
            print(f"Cache unavailable ({error}), falling back to DopplerDataset.")

    print(f"Using DopplerDataset for {len(files)} files ({dataset_size_mb:.1f} MB)")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return DopplerDataset(dataset_path)