"""
Comparison: Float model vs. Quantized model on CIFAR-10 test set.
Reports Top-1 Accuracy, model size (MB), and average inference speed (ms/sample).

Usage:
    python evaluate.py
"""

import os
import sys
import time
import copy

import torch
import torchvision
import torchvision.transforms as transforms

from model import get_model

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def get_test_loader(batch_size: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def evaluate_accuracy(model, loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
    return 100.0 * correct / total


def measure_latency(model, device='cpu', n_samples: int = 1000, batch_size: int = 1):
    """Measure average inference latency per sample (ms)."""
    model.eval()
    dummy = torch.randn(batch_size, 3, 32, 32).to(device)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            model(dummy)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_samples):
        with torch.no_grad():
            model(dummy)
    elapsed = time.perf_counter() - start

    ms_per_sample = (elapsed / n_samples) * 1000
    return ms_per_sample


def get_model_size_mb(model) -> float:
    """Estimate model size in MB via state_dict buffer."""
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def load_quantized_model(checkpoint_path: str):
    """Rebuild and load a quantized model from its saved state_dict."""
    model = get_model()
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def main():
    device = 'cpu'  # quantized models run on CPU
    test_loader = get_test_loader()

    results = {}

    # --- Float model ---
    float_ckpt = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(float_ckpt):
        print(f"ERROR: Float checkpoint not found at {float_ckpt}")
        print("Run train.py first.")
        sys.exit(1)

    print("Loading float model...")
    checkpoint = torch.load(float_ckpt, map_location=device)
    float_model = get_model()
    float_model.load_state_dict(checkpoint['model_state_dict'])
    float_model.eval()

    print("Evaluating float model...")
    float_acc = evaluate_accuracy(float_model, test_loader, device)
    float_size = get_model_size_mb(float_model)
    float_lat = measure_latency(float_model, device)
    results['Float'] = {'accuracy': float_acc, 'size_mb': float_size, 'latency_ms': float_lat}

    # --- Quantized model ---
    quant_ckpt = os.path.join(CHECKPOINT_DIR, 'quantized_model.pth')
    if not os.path.exists(quant_ckpt):
        print(f"WARNING: Quantized checkpoint not found at {quant_ckpt}")
        print("Run quantize.py first.")
    else:
        print("Loading quantized model...")
        quant_model = load_quantized_model(quant_ckpt)

        print("Evaluating quantized model...")
        quant_acc = evaluate_accuracy(quant_model, test_loader, device)
        quant_size = get_model_size_mb(quant_model)
        quant_lat = measure_latency(quant_model, device)
        results['Quantized (INT8)'] = {
            'accuracy': quant_acc, 'size_mb': quant_size, 'latency_ms': quant_lat}

    # --- Print comparison table ---
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Top-1 Acc':>10} {'Size (MB)':>12} {'Latency (ms)':>14}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<20} {m['accuracy']:>9.2f}% {m['size_mb']:>11.2f} {m['latency_ms']:>13.3f}")
    print("=" * 60)

    if len(results) == 2:
        acc_drop = results['Float']['accuracy'] - results['Quantized (INT8)']['accuracy']
        size_ratio = results['Float']['size_mb'] / results['Quantized (INT8)']['size_mb']
        speedup = results['Float']['latency_ms'] / results['Quantized (INT8)']['latency_ms']
        print(f"\nAccuracy drop:   {acc_drop:.2f}%")
        print(f"Size reduction:  {size_ratio:.2f}x")
        print(f"Speedup:         {speedup:.2f}x")


if __name__ == '__main__':
    main()
