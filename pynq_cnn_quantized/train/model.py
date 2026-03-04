"""
Lightweight CNN for CIFAR-10 with PyTorch quantization support.
Architecture: 3x (Conv + BN + ReLU + MaxPool) -> 2x FC
Compatible with torch.quantization static quantization via QuantStub/DeQuantStub.
"""

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub


class LightCNN(nn.Module):
    """
    Lightweight CNN for CIFAR-10 classification.
    Input: (N, 3, 32, 32)
    Output: (N, 10)
    Target: ~90% top-1 validation accuracy
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv+BN+ReLU layers for quantization efficiency."""
        # Feature block indices: (conv, bn, relu) triplets
        fuse_list = [
            ['features.0', 'features.1', 'features.2'],
            ['features.3', 'features.4', 'features.5'],
            ['features.7', 'features.8', 'features.9'],
            ['features.10', 'features.11', 'features.12'],
            ['features.14', 'features.15', 'features.16'],
        ]
        for layers in fuse_list:
            torch.quantization.fuse_modules(self, layers, inplace=True)


def get_model(num_classes: int = 10) -> LightCNN:
    return LightCNN(num_classes=num_classes)


if __name__ == '__main__':
    model = get_model()
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print(f"Model output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
