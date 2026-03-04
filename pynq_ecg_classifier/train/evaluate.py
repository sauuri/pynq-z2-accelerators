"""
Float vs. Quantized ECG classifier comparison.
Reports: Accuracy, Macro-F1, model size, latency.
Usage:
    python evaluate.py
"""

import os, sys, time, io
import torch
import numpy as np
from model import get_model
from dataset import get_dataloaders, CLASS_NAMES

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def sz(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1e6


def latency(model, n=1000):
    model.eval(); dummy = torch.randn(1, 1, 187)
    for _ in range(50):
        with torch.no_grad(): model(dummy)
    t = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): model(dummy)
    return (time.perf_counter() - t) / n * 1000


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct, total = 0, 0
    preds_all, targets_all = [], []
    for X, y in loader:
        p = model(X).argmax(1)
        correct += p.eq(y).sum().item(); total += y.size(0)
        preds_all.extend(p.tolist()); targets_all.extend(y.tolist())
    acc = 100.0 * correct / total
    n_cls = len(CLASS_NAMES)
    tp = np.zeros(n_cls); fp = np.zeros(n_cls); fn = np.zeros(n_cls)
    for p, t in zip(preds_all, targets_all):
        if p == t: tp[t] += 1
        else: fp[p] += 1; fn[t] += 1
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return acc, f1


def load_quantized(path):
    m = get_model(); m.eval()
    m.fuse_model()
    m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(m, inplace=True)
    torch.quantization.convert(m, inplace=True)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    return m


def main():
    float_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    quant_path = os.path.join(CHECKPOINT_DIR, 'quantized_model.pth')
    if not os.path.exists(float_path):
        print("Run train.py first."); sys.exit(1)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    _, test_loader = get_dataloaders(data_dir, batch_size=256)

    ckpt = torch.load(float_path, map_location='cpu')
    float_m = get_model(); float_m.load_state_dict(ckpt['model_state_dict']); float_m.eval()

    print("Evaluating float model...")
    f_acc, f_f1 = evaluate(float_m, test_loader)
    results = {'Float (FP32)': {'acc': f_acc, 'f1': f_f1, 'size': sz(float_m), 'lat': latency(float_m)}}

    if os.path.exists(quant_path):
        print("Evaluating quantized model...")
        q_m = load_quantized(quant_path)
        q_acc, q_f1 = evaluate(q_m, test_loader)
        results['Quantized (INT8)'] = {'acc': q_acc, 'f1': q_f1, 'size': sz(q_m), 'lat': latency(q_m)}

    print(f"\n{'='*68}")
    print(f"{'Model':<22} {'Acc':>8} {'Macro-F1':>10} {'Size(MB)':>10} {'Lat(ms)':>10}")
    print(f"{'-'*68}")
    for name, m in results.items():
        print(f"{name:<22} {m['acc']:>7.2f}% {m['f1'].mean():>10.4f} "
              f"{m['size']:>10.2f} {m['lat']:>10.3f}")
    print(f"{'='*68}")

    if len(results) == 2:
        f, q = results['Float (FP32)'], results['Quantized (INT8)']
        print(f"\nAcc drop: {f['acc']-q['acc']:.2f}%  |  F1 drop: {f['f1'].mean()-q['f1'].mean():.4f}  |  "
              f"Size: {f['size']/q['size']:.2f}×  |  Speedup: {f['lat']/q['lat']:.2f}×")

        print(f"\nPer-class F1 (Quantized):")
        for name, score in zip(CLASS_NAMES, q['f1']):
            bar = '█' * int(score * 20)
            print(f"  {name:<25} {score:.4f} {bar}")


if __name__ == '__main__':
    main()
