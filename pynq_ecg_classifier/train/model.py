"""
1D ResNet for ECG arrhythmia classification.
Dataset: MIT-BIH Arrhythmia (via wfdb or preprocessed CSV)

Input: (batch, 1, 187) — 187 timestep ECG segment
Output: (batch, 5) — 5 arrhythmia classes (MITBIH convention):
    0: Normal (N)
    1: Supraventricular (S)
    2: Ventricular (V)
    3: Fusion (F)
    4: Unknown (Q)

Architecture: ResNet1D (lightweight, ~500K params)
Compatible with torch.quantization PTQ.
"""

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub


class ResBlock1D(nn.Module):
    """1D Residual block: Conv-BN-ReLU × 2 + skip connection."""

    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, downsample=None):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet1D(nn.Module):
    """
    Lightweight 1D ResNet for ECG classification.
    Input:  (B, 1, 187)
    Output: (B, num_classes)
    ~500K parameters
    """

    def __init__(self, num_classes=5, base_ch=32):
        super().__init__()
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # ResNet stages: 187 → 47 → 24 → 12 → 6
        self.layer1 = self._make_layer(base_ch,      base_ch,      blocks=2, stride=1)
        self.layer2 = self._make_layer(base_ch,      base_ch * 2,  blocks=2, stride=2)
        self.layer3 = self._make_layer(base_ch * 2,  base_ch * 4,  blocks=2, stride=2)
        self.layer4 = self._make_layer(base_ch * 4,  base_ch * 8,  blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_ch * 8, num_classes),
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [ResBlock1D(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv1d+BN+ReLU for quantization."""
        for m in self.modules():
            if isinstance(m, ResBlock1D):
                torch.quantization.fuse_modules(
                    m, [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']], inplace=True)


def get_model(num_classes=5) -> ResNet1D:
    return ResNet1D(num_classes=num_classes)


if __name__ == '__main__':
    m = get_model()
    x = torch.randn(4, 1, 187)
    out = m(x)
    total = sum(p.numel() for p in m.parameters())
    print(f"Output: {out.shape}, Params: {total:,}")
