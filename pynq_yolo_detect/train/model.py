"""
Lightweight custom object detector for PYNQ-Z2.
Two options:
  A) Thin wrapper around YOLOv8n (ultralytics) — easiest path
  B) MobileNet-SSD style custom detector — no ultralytics dependency

This file provides Option B: a MobileNetV2-backbone SSD-lite detector
for VOC/COCO-style data, fully compatible with torch.quantization.

For YOLOv8n (ultralytics), see train.py --use-yolo flag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: DW + PW, BN + ReLU6."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class InvertedResidual(nn.Module):
    """MobileNetV2 bottleneck block."""
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """Truncated MobileNetV2 as feature extractor for SSD."""
    # (expand_ratio, out_channels, num_blocks, stride)
    CFG = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
    ]

    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
        )
        layers, in_ch = [], 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                layers.append(InvertedResidual(in_ch, c, s if i == 0 else 1, t))
                in_ch = c
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        feat4 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 6:   # after 64-ch block (stride 8 total)
                feat4 = x
        return feat4, x  # multi-scale features


class SSDHead(nn.Module):
    """Lightweight SSD detection head."""
    def __init__(self, num_classes, num_anchors=6):
        super().__init__()
        self.cls4 = nn.Conv2d(64,  num_anchors * num_classes, 3, padding=1)
        self.box4 = nn.Conv2d(64,  num_anchors * 4,           3, padding=1)
        self.cls8 = nn.Conv2d(160, num_anchors * num_classes, 3, padding=1)
        self.box8 = nn.Conv2d(160, num_anchors * 4,           3, padding=1)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, feat4, feat8):
        B = feat4.size(0)
        cls_preds = [
            self.cls4(feat4).permute(0,2,3,1).reshape(B, -1, self.num_classes),
            self.cls8(feat8).permute(0,2,3,1).reshape(B, -1, self.num_classes),
        ]
        box_preds = [
            self.box4(feat4).permute(0,2,3,1).reshape(B, -1, 4),
            self.box8(feat8).permute(0,2,3,1).reshape(B, -1, 4),
        ]
        return torch.cat(cls_preds, 1), torch.cat(box_preds, 1)


class LightDetector(nn.Module):
    """
    MobileNetV2 + SSD-Lite detector.
    Input: (B, 3, 300, 300)
    Output: cls_logits (B, N_anchors, num_classes), bbox_preds (B, N_anchors, 4)

    Compatible with torch.quantization via QuantStub/DeQuantStub.
    """
    def __init__(self, num_classes=21):  # 20 VOC + background
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.backbone = MobileNetV2Backbone()
        self.head = SSDHead(num_classes)

    def forward(self, x):
        x = self.quant(x)
        feat4, feat8 = self.backbone(x)
        cls_logits, bbox_preds = self.head(feat4, feat8)
        cls_logits = self.dequant(cls_logits)
        bbox_preds = self.dequant(bbox_preds)
        return cls_logits, bbox_preds


def get_model(num_classes=21) -> LightDetector:
    return LightDetector(num_classes=num_classes)


if __name__ == '__main__':
    m = get_model()
    x = torch.randn(1, 3, 300, 300)
    cls, box = m(x)
    total = sum(p.numel() for p in m.parameters())
    print(f"cls: {cls.shape}, box: {box.shape}, params: {total:,}")
