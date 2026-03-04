"""
TinyViT inference + ONNX export for PYNQ-Z2 deployment.
Usage:
    python inference.py --demo
    python inference.py --export-onnx
"""

import argparse, os, sys, time, copy
import torch
import numpy as np

TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'train')
sys.path.insert(0, TRAIN_DIR)
from model import get_model  # noqa

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
CIFAR10 = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def load_quantized(path, model_kwargs):
    m = get_model(**model_kwargs); m.eval()
    m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(m, inplace=True)
    torch.quantization.convert(m, inplace=True)
    m.load_state_dict(torch.load(path, map_location='cpu')); m.eval()
    return m


def export_onnx(output_path):
    """
    Export float TinyViT to ONNX.

    Transformer FPGA deployment notes:
    - Attention layers are NOT directly supported by hls4ml
    - Recommended strategy for PYNQ-Z2:
        1. Use hls4ml for patch embedding (Conv2d) + MLP blocks only
        2. Attention on ARM PS side (software fallback)
        3. Or: replace softmax attention with linear attention for HLS compatibility

    Linear attention alternative (HLS-friendly):
        attn = q * k.mean(dim=-2, keepdim=True)  # no softmax
        # -> fully parallelizable, no sequential dependency

    hls4ml export (when attention is excluded/linearized):
        import hls4ml
        config = hls4ml.utils.config_from_onnx_model(model_onnx)
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model_onnx, hls_config=config,
            output_dir='hls/hls4ml_output', part='xc7z020clg400-1')
    """
    float_ckpt = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    ckpt = torch.load(float_ckpt, map_location='cpu')
    model = get_model(**ckpt.get('model_kwargs', {}))
    model.load_state_dict(ckpt['model_state_dict']); model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(model, dummy, output_path, opset_version=13,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    print(f"ONNX saved: {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--export-onnx', action='store_true')
    parser.add_argument('--checkpoint', default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    args = parser.parse_args()

    if args.export_onnx:
        export_onnx(os.path.join(CHECKPOINT_DIR, 'tinyvit.onnx')); return

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Not found: {ckpt_path}\nRun train.py then quantize.py."); sys.exit(1)

    float_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'), map_location='cpu')
    model = load_quantized(ckpt_path, float_ckpt.get('model_kwargs', {}))

    dummy = torch.randn(1, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(dummy)
        ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(out, 1)[0]
    pred = probs.argmax().item()
    print(f"Predicted: {CIFAR10[pred]} (conf={probs[pred]:.3f}, latency={ms:.2f}ms)")

    n = 500
    with torch.no_grad():
        t = time.perf_counter()
        for _ in range(n): model(torch.randn(1,3,32,32))
    print(f"Avg: {(time.perf_counter()-t)/n*1000:.3f} ms/sample over {n} runs")


if __name__ == '__main__':
    main()
