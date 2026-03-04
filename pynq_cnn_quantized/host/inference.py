"""
PYNQ-Z2 deployment inference script.
Supports:
  1. CPU inference with the quantized PyTorch model
  2. ONNX export for downstream toolchain (hls4ml / FINN)
  3. (Future) PYNQ overlay integration via hls4ml-generated bitstream

Usage on host PC:
    python inference.py --export-onnx

Usage on PYNQ-Z2 (after bitstream deployment):
    python inference.py --pynq
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

# Adjust path so we can import from train/
TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'train')
sys.path.insert(0, TRAIN_DIR)

from model import get_model  # noqa: E402

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
ONNX_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'quantized_model.onnx')

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_quantized_model(checkpoint_path: str):
    """Rebuild and load quantized INT8 model."""
    model = get_model()
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(model, output_path: str):
    """
    Export to ONNX (float32 model required for ONNX export).
    Note: PyTorch's quantized ops are not fully ONNX-exportable.
    We export the float (fused) model instead, which is the standard
    input for hls4ml / FINN toolchains.

    hls4ml workflow (after ONNX export):
        import hls4ml
        config = hls4ml.utils.config_from_onnx_model(model_onnx, granularity='name')
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model_onnx, hls_config=config,
            output_dir='hls4ml_output', part='xc7z020clg400-1')
        hls_model.compile()   # requires Vivado HLS
        hls_model.build()     # generates bitstream

    FINN workflow:
        Use Brevitas for QAT-trained ONNX, then:
        finn-run --model quantized_model.onnx --platform pynq-z2
    """
    float_ckpt = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(float_ckpt):
        print(f"ERROR: Float checkpoint not found at {float_ckpt}")
        return

    checkpoint = torch.load(float_ckpt, map_location='cpu')
    float_model = get_model()
    float_model.load_state_dict(checkpoint['model_state_dict'])
    float_model.eval()
    float_model.fuse_model()

    dummy_input = torch.randn(1, 3, 32, 32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        float_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    print(f"ONNX model exported to: {output_path}")
    print(f"ONNX file size: {os.path.getsize(output_path) / 1e6:.2f} MB")


def run_pytorch_inference(model, input_tensor: torch.Tensor):
    """Single-sample inference with timing."""
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        output = model(input_tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(output, dim=1)
    top1_prob, top1_class = probs.max(1)
    return top1_class.item(), top1_prob.item(), elapsed_ms


def run_pynq_inference(input_array: np.ndarray):
    """
    PYNQ-Z2 hardware inference via hls4ml-generated overlay.

    Prerequisites:
        1. Run hls4ml to generate HLS from ONNX (see hls/README.md)
        2. Synthesize bitstream with Vivado HLS + Vivado
        3. Copy .bit + .hwh files to PYNQ board

    # ---- PYNQ DEPLOYMENT CODE (uncomment on board) ----
    from pynq import Overlay
    import pynq.lib.dma as dma

    overlay = Overlay('/home/xilinx/pynq_cnn.bit')
    dma_engine = overlay.axi_dma_0

    input_buffer = pynq.allocate(shape=(3, 32, 32), dtype=np.float32)
    output_buffer = pynq.allocate(shape=(10,), dtype=np.float32)

    np.copyto(input_buffer, input_array)
    dma_engine.sendchannel.transfer(input_buffer)
    dma_engine.recvchannel.transfer(output_buffer)
    dma_engine.sendchannel.wait()
    dma_engine.recvchannel.wait()

    return np.argmax(output_buffer)
    # ---- END PYNQ CODE ----
    """
    raise NotImplementedError(
        "PYNQ hardware inference not yet implemented. "
        "See hls/README.md for HLS synthesis instructions."
    )


def main():
    parser = argparse.ArgumentParser(description='LightCNN inference for PYNQ-Z2')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export float model to ONNX')
    parser.add_argument('--pynq', action='store_true',
                        help='Use PYNQ hardware overlay for inference')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    args = parser.parse_args()

    if args.export_onnx:
        # Need to load a dummy model just for the export function
        model = None
        export_onnx(model, ONNX_PATH)
        return

    if args.pynq:
        dummy_input = np.random.randn(3, 32, 32).astype(np.float32)
        class_idx = run_pynq_inference(dummy_input)
        print(f"Predicted class: {CIFAR10_CLASSES[class_idx]}")
        return

    # Default: PyTorch CPU inference demo
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Quantized checkpoint not found at {args.checkpoint}")
        print("Run train.py then quantize.py first.")
        sys.exit(1)

    model = load_quantized_model(args.checkpoint)
    print("Loaded quantized INT8 model.")

    # Demo with random input
    dummy_input = torch.randn(1, 3, 32, 32)
    class_idx, confidence, latency_ms = run_pytorch_inference(model, dummy_input)
    print(f"Predicted: {CIFAR10_CLASSES[class_idx]} (conf={confidence:.3f}, "
          f"latency={latency_ms:.3f}ms)")

    # Batch throughput benchmark
    n_runs = 500
    total_time = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            x = torch.randn(1, 3, 32, 32)
            t0 = time.perf_counter()
            model(x)
            total_time += time.perf_counter() - t0
    avg_ms = (total_time / n_runs) * 1000
    print(f"Average inference: {avg_ms:.3f} ms/sample over {n_runs} runs")


if __name__ == '__main__':
    main()
