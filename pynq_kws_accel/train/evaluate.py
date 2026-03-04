"""
Float vs. Quantized KWS model comparison.
Usage:
    python evaluate.py
"""

import os, sys, time, io
import torch
import numpy as np
from model import get_model
from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def sz(m): b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1e6

def latency(model, n=1000):
    model.eval(); dummy = torch.randn(1, 1, 49, 40)
    for _ in range(50):
        with torch.no_grad(): model(dummy)
    t = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): model(dummy)
    return (time.perf_counter() - t) / n * 1000

@torch.no_grad()
def accuracy(model, loader):
    model.eval(); correct, total = 0, 0
    for X, y in loader:
        correct += model(X).argmax(1).eq(y).sum().item(); total += y.size(0)
    return 100.0 * correct / total

def load_quantized(path, arch):
    m = get_model(arch=arch, num_classes=NUM_CLASSES); m.eval()
    if arch == 'dscnn' and hasattr(m, 'fuse_model'): m.fuse_model()
    m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(m, inplace=True)
    torch.quantization.convert(m, inplace=True)
    ckpt = torch.load(path, map_location='cpu')
    m.load_state_dict(ckpt['model_state_dict'])
    return m

def main():
    float_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    quant_path = os.path.join(CHECKPOINT_DIR, 'quantized_model.pth')
    if not os.path.exists(float_path): print("Run train.py first."); sys.exit(1)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    _, val_loader = get_dataloaders(data_dir, batch_size=128, num_workers=0, augment=False)

    ckpt = torch.load(float_path, map_location='cpu')
    arch = ckpt.get('arch', 'dscnn')
    float_m = get_model(arch=arch, num_classes=NUM_CLASSES)
    float_m.load_state_dict(ckpt['model_state_dict']); float_m.eval()

    print(f"Evaluating [{arch}]...")
    results = {'Float (FP32)': {'acc': accuracy(float_m, val_loader),
                                 'size': sz(float_m), 'lat': latency(float_m)}}
    if os.path.exists(quant_path):
        q = load_quantized(quant_path, arch)
        results['Quantized (INT8)'] = {'acc': accuracy(q, val_loader),
                                        'size': sz(q), 'lat': latency(q)}

    print(f"\n{'='*55}")
    print(f"{'Model':<22} {'Acc':>8} {'Size(MB)':>10} {'Lat(ms)':>10}")
    print(f"{'-'*55}")
    for name, m in results.items():
        print(f"{name:<22} {m['acc']:>7.2f}% {m['size']:>10.2f} {m['lat']:>10.3f}")
    print(f"{'='*55}")

    if len(results) == 2:
        f, q = results['Float (FP32)'], results['Quantized (INT8)']
        print(f"\nAcc drop: {f['acc']-q['acc']:.2f}%  |  "
              f"Size: {f['size']/q['size']:.2f}×  |  Speedup: {f['lat']/q['lat']:.2f}×")
        print(f"Real-time: 16kHz audio → MFCC (10ms/frame) → 1sec window → "
              f"inference {q['lat']:.1f}ms → {'✓ real-time' if q['lat'] < 10 else '✗'}")

if __name__ == '__main__':
    main()
