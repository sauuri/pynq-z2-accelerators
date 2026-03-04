"""
Float TinyViT vs. Quantized TinyViT comparison on CIFAR-10.
Usage:
    python evaluate.py
"""

import os, sys, time, io, copy
import torch
import torchvision
import torchvision.transforms as transforms
from model import get_model

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def get_test_loader(batch=256):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)


@torch.no_grad()
def accuracy(model, loader):
    model.eval(); correct = total = 0
    for x, y in loader:
        correct += model(x).argmax(1).eq(y).sum().item(); total += y.size(0)
    return 100.0 * correct / total


def latency(model, n=500):
    model.eval(); dummy = torch.randn(1, 3, 32, 32)
    for _ in range(50):
        with torch.no_grad(): model(dummy)
    t = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): model(dummy)
    return (time.perf_counter() - t) / n * 1000


def model_size(model):
    b = io.BytesIO(); torch.save(model.state_dict(), b); return b.tell() / 1e6


def load_quantized(ckpt_path, model_kwargs):
    m = get_model(**model_kwargs); m.eval()
    m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(m, inplace=True)
    torch.quantization.convert(m, inplace=True)
    m.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return m


def main():
    float_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    quant_path = os.path.join(CHECKPOINT_DIR, 'quantized_model.pth')
    if not os.path.exists(float_path):
        print("Run train.py first."); sys.exit(1)

    ckpt = torch.load(float_path, map_location='cpu')
    model_kwargs = ckpt.get('model_kwargs', {})
    loader = get_test_loader()

    float_m = get_model(**model_kwargs)
    float_m.load_state_dict(ckpt['model_state_dict']); float_m.eval()

    results = {'Float (FP32)': {
        'accuracy': accuracy(float_m, loader),
        'size_mb': model_size(float_m),
        'latency_ms': latency(float_m),
    }}

    if os.path.exists(quant_path):
        quant_m = load_quantized(quant_path, model_kwargs)
        results['Quantized (INT8)'] = {
            'accuracy': accuracy(quant_m, loader),
            'size_mb': model_size(quant_m),
            'latency_ms': latency(quant_m),
        }

    print("\n" + "=" * 62)
    print(f"{'Model':<22} {'Acc':>8} {'Size(MB)':>10} {'Latency(ms)':>13}")
    print("-" * 62)
    for name, m in results.items():
        print(f"{name:<22} {m['accuracy']:>7.2f}% {m['size_mb']:>10.2f} {m['latency_ms']:>13.3f}")
    print("=" * 62)

    if len(results) == 2:
        f, q = results['Float (FP32)'], results['Quantized (INT8)']
        print(f"\nAcc drop: {f['accuracy']-q['accuracy']:.2f}%  |  "
              f"Size: {f['size_mb']/q['size_mb']:.2f}×  |  "
              f"Speedup: {f['latency_ms']/q['latency_ms']:.2f}×")


if __name__ == '__main__':
    main()
