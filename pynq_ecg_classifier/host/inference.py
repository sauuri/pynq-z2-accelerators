"""
ECG arrhythmia inference for PYNQ-Z2.
Real-time classification of ECG segments (187 samples @ 125Hz = 1.5 sec window).

Usage:
    python inference.py --demo
    python inference.py --export-onnx
    python inference.py --signal ecg_segment.npy
"""

import argparse, os, sys, time, io
import torch
import numpy as np

TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'train')
sys.path.insert(0, TRAIN_DIR)
from model import get_model  # noqa
from dataset import CLASS_NAMES  # noqa

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def load_quantized(path):
    m = get_model(); m.eval(); m.fuse_model()
    m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(m, inplace=True)
    torch.quantization.convert(m, inplace=True)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    return m


def preprocess_ecg(signal: np.ndarray) -> torch.Tensor:
    """
    Normalize ECG segment to [0, 1] and convert to tensor.
    signal: (187,) numpy array
    """
    if signal.shape != (187,):
        # Resample/pad to 187 samples
        from scipy.signal import resample
        signal = resample(signal, 187)
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    return torch.from_numpy(signal.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,187)


def export_onnx():
    """
    Export float ResNet1D to ONNX.

    hls4ml path for ECG ResNet1D:
        1D Conv ResNet maps well to HLS — each Conv1D → matrix multiplication
        Recommended precision: ap_fixed<16,6> for activations, ap_fixed<8,4> for weights

        import hls4ml, onnx
        model_onnx = onnx.load('checkpoints/ecg_resnet1d.onnx')
        config = hls4ml.utils.config_from_onnx_model(
            model_onnx, default_precision='ap_fixed<16,6>',
            default_reuse_factor=2)
        # Set aggressive precision for small ResBlock layers
        for layer in config['LayerName']:
            if 'conv' in layer.lower():
                config['LayerName'][layer]['Precision'] = {'weight': 'ap_fixed<8,4>',
                                                            'bias':   'ap_fixed<16,6>'}
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model_onnx, hls_config=config,
            output_dir='hls/hls4ml_output',
            part='xc7z020clg400-1',
            io_type='io_stream')
        hls_model.compile()

    PYNQ real-time ECG pipeline:
        ECG sensor (ADS1115 ADC) → GPIO/I2C → PS (preprocessing)
        → PL (ResNet1D HLS, latency ~1ms) → PS (alarm/display)
    """
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'), map_location='cpu')
    model = get_model(); model.load_state_dict(ckpt['model_state_dict']); model.eval()
    dummy = torch.randn(1, 1, 187)
    out_path = os.path.join(CHECKPOINT_DIR, 'ecg_resnet1d.onnx')
    torch.onnx.export(model, dummy, out_path, opset_version=13,
                      input_names=['ecg_segment'], output_names=['class_logits'])
    print(f"ONNX saved: {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")


def run_pynq(signal_array: np.ndarray):
    """
    PYNQ-Z2 real-time ECG inference stub.

    # ---- PYNQ CODE (uncomment on board) ----
    from pynq import Overlay
    import pynq

    overlay = Overlay('/home/xilinx/ecg_resnet1d.bit')
    dma = overlay.axi_dma_0

    in_buf  = pynq.allocate(shape=(187,), dtype=np.float32)
    out_buf = pynq.allocate(shape=(5,),   dtype=np.float32)
    np.copyto(in_buf, signal_array.flatten())

    dma.sendchannel.transfer(in_buf)
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.wait(); dma.recvchannel.wait()

    return int(np.argmax(out_buf))
    # ---- END PYNQ CODE ----
    """
    raise NotImplementedError("See hls/README.md for PYNQ setup.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo',        action='store_true')
    parser.add_argument('--export-onnx', action='store_true')
    parser.add_argument('--pynq',        action='store_true')
    parser.add_argument('--signal',      type=str, help='.npy file with (187,) ECG segment')
    parser.add_argument('--checkpoint',  default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    args = parser.parse_args()

    if args.export_onnx: export_onnx(); return

    model = load_quantized(args.checkpoint)

    if args.signal and os.path.exists(args.signal):
        sig = np.load(args.signal)
    else:
        # Simulate ECG signal (synthetic)
        t = np.linspace(0, 1, 187)
        sig = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(187)
        print("(Using synthetic ECG signal)")

    if args.pynq:
        cls = run_pynq(sig.astype(np.float32)); print(f"PYNQ: {CLASS_NAMES[cls]}"); return

    x = preprocess_ecg(sig)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(x)
        ms = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(out, 1)[0]
    pred = probs.argmax().item()

    print(f"Prediction: {CLASS_NAMES[pred]} (conf={probs[pred]:.3f}, {ms:.3f}ms)")
    print("\nAll classes:")
    for name, p in zip(CLASS_NAMES, probs):
        bar = '█' * int(p * 30)
        print(f"  {name:<25} {p:.4f} {bar}")

    # Throughput (simulates real-time 125Hz stream)
    n = 2000
    with torch.no_grad():
        t = time.perf_counter()
        for _ in range(n): model(torch.randn(1, 1, 187))
    avg = (time.perf_counter() - t) / n * 1000
    print(f"\nAvg: {avg:.3f}ms/sample | Real-time: 125Hz = 8ms/sample → {'✓ OK' if avg < 8 else '✗ too slow'}")


if __name__ == '__main__':
    main()
