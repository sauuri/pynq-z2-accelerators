"""
Keyword Spotting (KWS) model for "Hey device"-style wake word detection.
Dataset: Google Speech Commands (35 keywords)

Two architectures:
  A) DS-CNN (Depthwise Separable CNN on MFCC) — FPGA-optimal, ~100K params
  B) BC-ResNet (Broadcast-Residual CNN) — higher accuracy, ~300K params

Input: MFCC features (1, 49, 40) — 49 frames × 40 mel bins (1sec @ 16kHz)
Output: (num_classes,) — 10+2 classes (10 keywords + silence + unknown)

Reference: "Hello Edge: Keyword Spotting on Microcontrollers" (arXiv:1711.07128)
"""

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub


class DSCNN(nn.Module):
    """
    DS-CNN: Depthwise Separable CNN for KWS (smallest, most FPGA-friendly).
    Architecture from "Hello Edge" paper, adapted for MFCC (49×40) input.
    ~100K parameters
    """

    def __init__(self, num_classes=12, n_mfcc=40, n_frames=49):
        super().__init__()
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

        # Standard Conv + BN + ReLU (input layer)
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), padding=(5, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )

        # Depthwise separable conv blocks
        def ds_block(ch):
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                # Pointwise
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            )

        self.ds_layers = nn.Sequential(
            ds_block(64), ds_block(64), ds_block(64), ds_block(64),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.input_layer(x)
        x = self.ds_layers(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv+BN+ReLU layers."""
        torch.quantization.fuse_modules(
            self.input_layer, ['0', '1', '2'], inplace=True)
        for block in self.ds_layers:
            torch.quantization.fuse_modules(block, ['0', '1', '2'], inplace=True)
            torch.quantization.fuse_modules(block, ['3', '4', '5'], inplace=True)


class BCResNet(nn.Module):
    """
    BC-ResNet: Broadcast-Residual CNN for KWS.
    Higher accuracy than DS-CNN with ~300K params.
    Reference: "Broadcasting Residual Connections for Better KWS" (arXiv:2106.04140)
    """

    def __init__(self, num_classes=12, scale=1):
        super().__init__()
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()

        def ch(c): return max(1, int(c * scale))

        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(1, ch(16), (5, 5), stride=(2, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(ch(16)), nn.ReLU(inplace=True),

            # Stage 1
            self._res_block(ch(16), ch(24), stride=(2, 1)),
            self._res_block(ch(24), ch(24)),
            self._res_block(ch(24), ch(24)),

            # Stage 2
            self._res_block(ch(24), ch(32), stride=(2, 1)),
            self._res_block(ch(32), ch(32)),
            self._res_block(ch(32), ch(32)),

            # Stage 3
            self._res_block(ch(32), ch(48), stride=(2, 1)),
            self._res_block(ch(48), ch(48)),

            # Head conv
            nn.Conv2d(ch(48), ch(64), 1, bias=False),
            nn.BatchNorm2d(ch(64)), nn.ReLU(inplace=True),
        )
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch(64), num_classes)

    def _res_block(self, in_ch, out_ch, stride=1):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if isinstance(stride, tuple):
            needs_proj = stride != (1, 1) or in_ch != out_ch
        else:
            needs_proj = stride != 1 or in_ch != out_ch
        proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if needs_proj else nn.Identity()
        return _ResWrapper(nn.Sequential(*layers), proj)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


class _ResWrapper(nn.Module):
    def __init__(self, main, proj):
        super().__init__(); self.main = main; self.proj = proj
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.proj(x))


def get_model(arch='dscnn', num_classes=12, **kwargs) -> nn.Module:
    """
    Args:
        arch: 'dscnn' | 'bcresnet'
    """
    if arch == 'dscnn':
        return DSCNN(num_classes=num_classes, **kwargs)
    elif arch == 'bcresnet':
        return BCResNet(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown arch: {arch}")


if __name__ == '__main__':
    for arch in ['dscnn', 'bcresnet']:
        m = get_model(arch); x = torch.randn(2, 1, 49, 40)
        out = m(x); total = sum(p.numel() for p in m.parameters())
        print(f"[{arch}] output={out.shape}, params={total:,}")
