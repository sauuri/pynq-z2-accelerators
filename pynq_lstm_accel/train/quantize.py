"""
Dynamic Quantization for LSTM classifier.
PyTorch dynamic quantization quantizes weights to INT8 at runtime
(activations remain FP32), which is the recommended approach for RNNs.

Why dynamic (not static) quantization for LSTM?
- LSTM hidden states are irregular → hard to calibrate activation ranges
- Dynamic quant achieves ~4× weight compression with minimal accuracy loss
- Well-supported by PyTorch's quantized LSTM kernel

Usage:
    python quantize.py [--checkpoint ../checkpoints/best_model.pth]
"""

import argparse
import os
import io

import torch

from model import get_model

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def get_model_size_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def main():
    parser = argparse.ArgumentParser(description='Dynamic quantization for LSTM')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    parser.add_argument('--output', type=str,
                        default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    arch = checkpoint.get('arch', 'lstm')
    model_kwargs = checkpoint.get('model_kwargs', {})
    model = get_model(arch=arch, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Architecture: {arch}")
    print(f"Float model test_acc from training: {checkpoint.get('test_acc', 'N/A'):.2f}%")
    float_size = get_model_size_mb(model)
    print(f"Float model size: {float_size:.2f} MB")

    # Dynamic quantization: INT8 weights for Linear + LSTM layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn_type for nn_type in [torch.nn.LSTM, torch.nn.Linear]},
        dtype=torch.qint8,
    )
    print("Applied dynamic quantization (INT8 weights) to LSTM + Linear layers.")

    quant_size = get_model_size_mb(quantized_model)
    print(f"Quantized model size: {quant_size:.2f} MB")
    print(f"Size reduction: {(1 - quant_size / float_size) * 100:.1f}%")

    # Save quantized model state
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'arch': arch,
        'model_kwargs': model_kwargs,
        'model_state_dict': quantized_model.state_dict(),
        'norm_mean': checkpoint.get('norm_mean'),
        'norm_std': checkpoint.get('norm_std'),
        'test_acc': checkpoint.get('test_acc'),
    }, args.output)
    print(f"Saved quantized model to: {args.output}")


if __name__ == '__main__':
    import torch.nn as nn
    main()
