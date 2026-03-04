"""
UCI HAR (Human Activity Recognition) dataset loader.
Dataset: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

Auto-downloads and preprocesses raw sensor signals (128 timesteps × 9 channels):
  Channels: body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z

Labels (0-indexed):
  0: WALKING
  1: WALKING_UPSTAIRS
  2: WALKING_DOWNSTAIRS
  3: SITTING
  4: STANDING
  5: LAYING
"""

import os
import urllib.request
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)

SIGNAL_NAMES = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z',
]

ACTIVITY_LABELS = [
    'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
    'SITTING', 'STANDING', 'LAYING'
]


def download_uci_har(data_dir: str):
    """Download and extract UCI HAR dataset if not present."""
    extract_dir = os.path.join(data_dir, 'UCI HAR Dataset')
    if os.path.exists(extract_dir):
        return extract_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'uci_har.zip')

    print(f"Downloading UCI HAR Dataset...")
    urllib.request.urlretrieve(UCI_HAR_URL, zip_path)
    print(f"Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)
    print(f"Dataset ready at: {extract_dir}")
    return extract_dir


def load_signals(dataset_dir: str, split: str) -> np.ndarray:
    """Load all 9 signal channels for a split ('train' or 'test')."""
    signals_dir = os.path.join(dataset_dir, split, 'Inertial Signals')
    signals = []
    for name in SIGNAL_NAMES:
        fname = f"{name}_{split}.txt"
        fpath = os.path.join(signals_dir, fname)
        data = np.loadtxt(fpath)  # (n_samples, 128)
        signals.append(data)
    # Stack: (n_samples, 128, 9)
    return np.stack(signals, axis=2).astype(np.float32)


def load_labels(dataset_dir: str, split: str) -> np.ndarray:
    """Load labels (1-indexed in file → 0-indexed)."""
    fpath = os.path.join(dataset_dir, split, f'y_{split}.txt')
    labels = np.loadtxt(fpath, dtype=np.int64) - 1  # 0-indexed
    return labels


class UCIHARDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: bool = True,
                 mean: np.ndarray = None, std: np.ndarray = None):
        if normalize:
            if mean is None:
                mean = X.mean(axis=(0, 1), keepdims=True)
            if std is None:
                std = X.std(axis=(0, 1), keepdims=True) + 1e-8
            X = (X - mean) / std
        self.X = torch.from_numpy(X)   # (N, 128, 9)
        self.y = torch.from_numpy(y)   # (N,)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(data_dir: str, batch_size: int = 64, num_workers: int = 2):
    """
    Returns (train_loader, test_loader, train_dataset).
    train_dataset exposes .mean and .std for inference normalization.
    """
    dataset_dir = download_uci_har(data_dir)

    X_train = load_signals(dataset_dir, 'train')
    y_train = load_labels(dataset_dir, 'train')
    X_test = load_signals(dataset_dir, 'test')
    y_test = load_labels(dataset_dir, 'test')

    train_set = UCIHARDataset(X_train, y_train, normalize=True)
    # Use train stats for test normalization
    test_set = UCIHARDataset(X_test, y_test, normalize=True,
                             mean=train_set.mean, std=train_set.std)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, train_set


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_loader, test_loader, _ = get_dataloaders(data_dir)
    X, y = next(iter(train_loader))
    print(f"Batch X: {X.shape}, y: {y.shape}")
    print(f"Classes: {ACTIVITY_LABELS}")
