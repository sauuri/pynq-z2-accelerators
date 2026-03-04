"""
LSTM-based classifier for Human Activity Recognition (UCI HAR).
Input: (batch, seq_len=128, features=9) — raw accelerometer + gyroscope
Output: (batch, 6) — 6 activity classes

Compatible with torch.quantization.quantize_dynamic (INT8 weights).
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Stacked LSTM classifier for time-series HAR.
    Architecture:
        LSTM(9→128, 2 layers) → Dropout → FC(128→64) → ReLU → FC(64→6)
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        out = out[:, -1, :]            # last timestep
        out = self.dropout(out)
        return self.classifier(out)


class CNNLSTMClassifier(nn.Module):
    """
    CNN feature extractor + LSTM for better local pattern capture.
    CNN extracts local features per window, LSTM captures temporal dependencies.
    Often outperforms pure LSTM on HAR tasks.
    """

    def __init__(
        self,
        input_size: int = 9,
        cnn_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)            # (batch, input_size, seq_len)
        x = self.cnn(x)                    # (batch, cnn_channels, seq_len)
        x = x.permute(0, 2, 1)            # (batch, seq_len, cnn_channels)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.classifier(out)


def get_model(arch: str = 'lstm', **kwargs) -> nn.Module:
    """
    Args:
        arch: 'lstm' | 'cnn_lstm'
    """
    if arch == 'lstm':
        return LSTMClassifier(**kwargs)
    elif arch == 'cnn_lstm':
        return CNNLSTMClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


if __name__ == '__main__':
    for arch in ['lstm', 'cnn_lstm']:
        model = get_model(arch)
        x = torch.randn(4, 128, 9)
        out = model(x)
        total = sum(p.numel() for p in model.parameters())
        print(f"[{arch}] output={out.shape}, params={total:,}")
