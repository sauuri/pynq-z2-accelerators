"""
Post-Training Static Quantization (PTQ) for LightCNN.
Loads best_model.pth, calibrates with CIFAR-10 train data,
and saves quantized_model.pth.

Usage:
    python quantize.py [--checkpoint ../checkpoints/best_model.pth]
"""

import argparse
import os
import copy

import torch
import torchvision
import torchvision.transforms as transforms

from model import get_model

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def get_calibration_loader(num_samples: int = 1000, batch_size: int = 64):
    """Return a small subset of CIFAR-10 train set for calibration."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    subset = torch.utils.data.Subset(dataset, range(num_samples))
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)


def calibrate(model, loader, device='cpu'):
    """Run forward passes on calibration data to collect activation statistics."""
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            model(inputs)


def main():
    parser = argparse.ArgumentParser(description='PTQ for LightCNN')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    parser.add_argument('--output', type=str,
                        default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    parser.add_argument('--calib-samples', type=int, default=1000)
    args = parser.parse_args()

    # Quantization must run on CPU for static quant
    device = 'cpu'

    # Load float model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = get_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Float model val_acc from training: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    # Step 1: Fuse Conv+BN+ReLU
    model.fuse_model()
    print("Fused Conv+BN+ReLU layers.")

    # Step 2: Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(f"QConfig: {model.qconfig}")

    # Step 3: Prepare for calibration (insert observers)
    torch.quantization.prepare(model, inplace=True)
    print("Inserted observers for calibration.")

    # Step 4: Calibrate
    calib_loader = get_calibration_loader(num_samples=args.calib_samples)
    print(f"Calibrating with {args.calib_samples} samples...")
    calibrate(model, calib_loader, device=device)

    # Step 5: Convert to quantized model
    quantized_model = copy.deepcopy(model)
    torch.quantization.convert(quantized_model, inplace=True)
    print("Converted to quantized model.")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(quantized_model.state_dict(), args.output)
    print(f"Quantized model saved to: {args.output}")

    # Quick size comparison
    float_size = os.path.getsize(args.checkpoint) / 1e6
    quant_size = os.path.getsize(args.output) / 1e6
    print(f"\nCheckpoint sizes:")
    print(f"  Float:     {float_size:.2f} MB")
    print(f"  Quantized: {quant_size:.2f} MB")
    print(f"  Reduction: {(1 - quant_size / float_size) * 100:.1f}%")


if __name__ == '__main__':
    main()
