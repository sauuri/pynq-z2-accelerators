"""
Comparison: Float LSTM vs. Dynamically Quantized LSTM.
Reports Top-1 Accuracy, model size (MB), and average inference speed (ms/sample).

Usage:
    python evaluate.py
"""

import os
import sys
import time
import io

import torch
import numpy as np

from model import get_model
from dataset import get_dataloaders, ACTIVITY_LABELS

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def get_model_size_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def load_float_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    arch = checkpoint.get('arch', 'lstm')
    model_kwargs = checkpoint.get('model_kwargs', {})
    model = get_model(arch=arch, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def load_quantized_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    arch = checkpoint.get('arch', 'lstm')
    model_kwargs = checkpoint.get('model_kwargs', {})
    # Rebuild float model, then quantize
    model = get_model(arch=arch, **model_kwargs)
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.LSTM, torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.load_state_dict(checkpoint['model_state_dict'])
    quantized_model.eval()
    return quantized_model, checkpoint


@torch.no_grad()
def evaluate_accuracy(model, loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        preds = out.argmax(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
    acc = 100.0 * correct / total
    return acc, all_preds, all_targets


def measure_latency(model, device='cpu', n_runs: int = 500):
    """Measure average latency (ms/sample) with batch_size=1."""
    model.eval()
    dummy = torch.randn(1, 128, 9).to(device)
    # Warmup
    for _ in range(50):
        with torch.no_grad():
            model(dummy)
    # Timed
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000


def print_confusion_summary(preds, targets, labels):
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    for p, t in zip(preds, targets):
        matrix[t][p] += 1
    print("\nPer-class accuracy (quantized model):")
    for i, label in enumerate(labels):
        row_total = matrix[i].sum()
        acc = 100.0 * matrix[i][i] / max(row_total, 1)
        print(f"  {label:<25} {acc:.1f}%")


def main():
    float_ckpt = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    quant_ckpt = os.path.join(CHECKPOINT_DIR, 'quantized_model.pth')

    if not os.path.exists(float_ckpt):
        print(f"ERROR: Float checkpoint not found at {float_ckpt}")
        print("Run train.py first.")
        sys.exit(1)

    device = 'cpu'
    float_model, float_ckpt_data = load_float_model(float_ckpt)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    _, test_loader, _ = get_dataloaders(data_dir, batch_size=256)

    results = {}

    print("Evaluating float model...")
    float_acc, _, _ = evaluate_accuracy(float_model, test_loader, device)
    float_size = get_model_size_mb(float_model)
    float_lat = measure_latency(float_model, device)
    results['Float (FP32)'] = {
        'accuracy': float_acc, 'size_mb': float_size, 'latency_ms': float_lat}

    if os.path.exists(quant_ckpt):
        print("Evaluating quantized model...")
        quant_model, _ = load_quantized_model(quant_ckpt)
        quant_acc, quant_preds, quant_targets = evaluate_accuracy(
            quant_model, test_loader, device)
        quant_size = get_model_size_mb(quant_model)
        quant_lat = measure_latency(quant_model, device)
        results['Quantized (INT8)'] = {
            'accuracy': quant_acc, 'size_mb': quant_size, 'latency_ms': quant_lat}
    else:
        print(f"WARNING: Quantized checkpoint not found at {quant_ckpt}")
        print("Run quantize.py first.")
        quant_preds, quant_targets = [], []

    # Print comparison table
    print("\n" + "=" * 62)
    print(f"{'Model':<22} {'Top-1 Acc':>10} {'Size (MB)':>12} {'Latency (ms)':>14}")
    print("-" * 62)
    for name, m in results.items():
        print(f"{name:<22} {m['accuracy']:>9.2f}% {m['size_mb']:>11.2f} {m['latency_ms']:>13.3f}")
    print("=" * 62)

    if len(results) == 2:
        acc_drop = results['Float (FP32)']['accuracy'] - results['Quantized (INT8)']['accuracy']
        size_ratio = results['Float (FP32)']['size_mb'] / results['Quantized (INT8)']['size_mb']
        speedup = results['Float (FP32)']['latency_ms'] / results['Quantized (INT8)']['latency_ms']
        print(f"\nAccuracy drop:   {acc_drop:.2f}%")
        print(f"Size reduction:  {size_ratio:.2f}x")
        print(f"Speedup:         {speedup:.2f}x")

        if quant_preds:
            print_confusion_summary(quant_preds, quant_targets, ACTIVITY_LABELS)


if __name__ == '__main__':
    main()
