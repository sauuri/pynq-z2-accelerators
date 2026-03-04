"""
Google Speech Commands dataset loader.
Uses torchaudio for auto-download + MFCC feature extraction.

Dataset: ~2.4GB (35 keywords, ~105K 1-second WAV clips @ 16kHz)
For KWS: 10 target keywords + silence + unknown (12 classes total)

Target keywords (standard subset):
    yes, no, up, down, left, right, on, off, stop, go
    + silence (class 10)
    + unknown (class 11) — all other keywords
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchaudio
    import torchaudio.transforms as AT
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

TARGET_KEYWORDS = ['yes', 'no', 'up', 'down', 'left', 'right',
                   'on', 'off', 'stop', 'go']
CLASS_NAMES = TARGET_KEYWORDS + ['silence', 'unknown']
NUM_CLASSES = len(CLASS_NAMES)

# MFCC params
SAMPLE_RATE   = 16000
N_MFCC        = 40
N_FRAMES      = 49   # ~1 second with 25ms window, 10ms hop
WIN_LENGTH    = 400  # 25ms @ 16kHz
HOP_LENGTH    = 160  # 10ms @ 16kHz


class SpeechCommandsDataset(Dataset):
    """Wrapper for torchaudio SpeechCommands with MFCC extraction."""

    def __init__(self, root: str, subset: str, augment: bool = False):
        self.augment = augment
        self.mfcc_transform = AT.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={'n_fft': 512, 'win_length': WIN_LENGTH,
                       'hop_length': HOP_LENGTH, 'n_mels': 40},
        )
        self.time_mask  = AT.TimeMasking(time_mask_param=10)
        self.freq_mask  = AT.FrequencyMasking(freq_mask_param=8)

        # Load from torchaudio
        self.data = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, download=True, subset=subset)

        # Build label map
        self.label_map = {kw: i for i, kw in enumerate(TARGET_KEYWORDS)}
        self.label_map['silence'] = 10

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.data[idx]

        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        # Pad/trim to exactly 1 second
        target_len = SAMPLE_RATE
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        else:
            waveform = waveform[:, :target_len]

        # MFCC: (1, N_MFCC, T)
        mfcc = self.mfcc_transform(waveform)

        # Normalize per sample
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        # Pad/trim to fixed N_FRAMES
        if mfcc.shape[-1] < N_FRAMES:
            mfcc = F.pad(mfcc, (0, N_FRAMES - mfcc.shape[-1]))
        else:
            mfcc = mfcc[:, :, :N_FRAMES]

        # Augmentation
        if self.augment:
            mfcc = self.time_mask(mfcc)
            mfcc = self.freq_mask(mfcc)

        # Transpose to (1, N_FRAMES, N_MFCC)
        mfcc = mfcc.permute(0, 2, 1)

        cls = self.label_map.get(label, 11)  # 11 = unknown
        return mfcc, cls


class SyntheticKWSDataset(Dataset):
    """Synthetic dataset for demo when torchaudio/data not available."""
    def __init__(self, n=5000):
        self.X = torch.randn(n, 1, N_FRAMES, N_MFCC)
        self.y = torch.randint(0, NUM_CLASSES, (n,))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def get_dataloaders(data_dir: str, batch_size=64, num_workers=2, augment=True):
    if not TORCHAUDIO_AVAILABLE:
        print("torchaudio not installed — using synthetic data")
        print("Install: pip install torchaudio")
        train_set = SyntheticKWSDataset(5000)
        test_set  = SyntheticKWSDataset(1000)
    else:
        try:
            train_set = SpeechCommandsDataset(data_dir, 'training',   augment=augment)
            test_set  = SpeechCommandsDataset(data_dir, 'validation', augment=False)
        except Exception as e:
            print(f"Dataset load failed ({e}) — using synthetic data")
            train_set = SyntheticKWSDataset(5000)
            test_set  = SyntheticKWSDataset(1000)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_loader, _ = get_dataloaders(data_dir, batch_size=4, num_workers=0)
    X, y = next(iter(train_loader))
    print(f"MFCC: {X.shape}, labels: {y}, classes: {CLASS_NAMES}")
