"""
PYNQ-Z2 deployment inference script for LSTM HAR classifier.
Supports:
  1. CPU inference with dynamically quantized PyTorch model
  2. ONNX export (float model) for hls4ml / FINN downstream toolchain
  3. (Future) PYNQ overlay integration

Note on ONNX + LSTM:
  PyTorch LSTM exports cleanly to ONNX opset 11+ with loop operators.
  hls4ml currently has limited RNN support — see hls/README.md for alternatives.

Usage:
    python inference.py --demo            # Run with random input
    python inference.py --export-onnx     # Export ONNX
    python inference.py --pynq            # PYNQ hardware (on board)
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'train')
sys.path.insert(0, TRAIN_DIR)

from model import get_model  # noqa: E402
from dataset import ACTIVITY_LABELS  # noqa: E402

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
ONNX_PATH = os.path.join(CHECKPOINT_DIR, 'lstm_model.onnx')


def load_quantized_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    arch = checkpoint.get('arch', 'lstm')
    model_kwargs = checkpoint.get('model_kwargs', {})
    model = get_model(arch=arch, **model_kwargs)
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)
    quantized.load_state_dict(checkpoint['model_state_dict'])
    quantized.eval()
    return quantized, checkpoint


def export_onnx(output_path: str):
    """
    Export float model to ONNX (opset 13).

    hls4ml RNN support:
        hls4ml has experimental LSTM support via the 'nnet_recurrent.h' backend.
        For full support, unroll the LSTM manually into dense layers using
        the 'SimpleRNN' approach, or use the following workaround:

        import hls4ml
        config = hls4ml.utils.config_from_onnx_model(model_onnx, granularity='name')
        # Set reuse_factor for dense layers inside LSTM
        config['LayerName']['lstm']['ReuseFactor'] = 1
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model_onnx, hls_config=config,
            output_dir='hls/hls4ml_output', part='xc7z020clg400-1')

    Alternative — unrolled LSTM (better FPGA mapping):
        If LSTM HLS support is insufficient, convert LSTM weights to
        equivalent Dense layers with explicit state management.
        See hls/README.md for the unrolled implementation guide.
    """
    float_ckpt = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(float_ckpt):
        print(f"ERROR: Float checkpoint not found at {float_ckpt}")
        return

    checkpoint = torch.load(float_ckpt, map_location='cpu')
    arch = checkpoint.get('arch', 'lstm')
    model_kwargs = checkpoint.get('model_kwargs', {})
    model = get_model(arch=arch, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy = torch.randn(1, 128, 9)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    print(f"ONNX model exported to: {output_path}")
    print(f"ONNX file size: {os.path.getsize(output_path) / 1e6:.2f} MB")


def preprocess(raw_signal: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """
    Normalize raw sensor input.
    raw_signal: (128, 9) float32 — 128 timesteps, 9 sensor channels
    """
    normed = (raw_signal - mean.squeeze()) / (std.squeeze() + 1e-8)
    return torch.from_numpy(normed).unsqueeze(0)  # (1, 128, 9)


def run_pynq_inference(input_array: np.ndarray):
    """
    PYNQ-Z2 hardware inference stub.

    Prerequisites:
        1. Synthesize bitstream via hls4ml (see hls/README.md)
        2. Copy .bit + .hwh to PYNQ board

    # ---- PYNQ DEPLOYMENT CODE (uncomment on board) ----
    from pynq import Overlay
    import pynq

    overlay = Overlay('/home/xilinx/pynq_lstm.bit')
    dma = overlay.axi_dma_0

    flat = input_array.flatten().astype(np.float32)
    in_buf = pynq.allocate(shape=flat.shape, dtype=np.float32)
    out_buf = pynq.allocate(shape=(6,), dtype=np.float32)

    np.copyto(in_buf, flat)
    dma.sendchannel.transfer(in_buf)
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    return int(np.argmax(out_buf))
    # ---- END PYNQ CODE ----
    """
    raise NotImplementedError(
        "PYNQ hardware inference not implemented yet. "
        "See hls/README.md for HLS synthesis instructions."
    )


def main():
    parser = argparse.ArgumentParser(description='LSTM HAR inference for PYNQ-Z2')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with random input')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export float model to ONNX')
    parser.add_argument('--pynq', action='store_true',
                        help='Use PYNQ hardware overlay')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    args = parser.parse_args()

    if args.export_onnx:
        export_onnx(ONNX_PATH)
        return

    if args.pynq:
        dummy = np.random.randn(128, 9).astype(np.float32)
        cls = run_pynq_inference(dummy)
        print(f"Predicted activity: {ACTIVITY_LABELS[cls]}")
        return

    # Default demo
    quant_ckpt = args.checkpoint
    if not os.path.exists(quant_ckpt):
        print(f"ERROR: Checkpoint not found at {quant_ckpt}")
        print("Run train.py then quantize.py first.")
        sys.exit(1)

    model, checkpoint = load_quantized_model(quant_ckpt)
    norm_mean = checkpoint.get('norm_mean')
    norm_std = checkpoint.get('norm_std')
    print(f"Loaded quantized model (arch={checkpoint.get('arch')})")

    # Demo with random input
    raw = np.random.randn(128, 9).astype(np.float32)
    if norm_mean is not None:
        x = preprocess(raw, norm_mean, norm_std)
    else:
        x = torch.from_numpy(raw).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(x)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(out, dim=1)[0]
    pred = probs.argmax().item()
    print(f"Predicted: {ACTIVITY_LABELS[pred]} (conf={probs[pred]:.3f}, "
          f"latency={elapsed_ms:.3f}ms)")
    print("\nAll class probabilities:")
    for label, p in zip(ACTIVITY_LABELS, probs):
        bar = '█' * int(p * 30)
        print(f"  {label:<25} {p:.3f} {bar}")

    # Throughput benchmark
    n_runs = 1000
    total_t = 0.0
    with torch.no_grad():
        for _ in range(n_runs):
            xi = torch.randn(1, 128, 9)
            t0 = time.perf_counter()
            model(xi)
            total_t += time.perf_counter() - t0
    print(f"\nAvg inference: {(total_t / n_runs) * 1000:.3f} ms/sample "
          f"over {n_runs} runs")


if __name__ == '__main__':
    main()
