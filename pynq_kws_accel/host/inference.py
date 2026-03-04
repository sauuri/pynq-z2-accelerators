"""
KWS inference for PYNQ-Z2.
Real-time keyword detection from microphone input.

Usage:
    python inference.py --demo
    python inference.py --export-onnx
    python inference.py --stream       # Real-time microphone (requires pyaudio)
    python inference.py --audio word.wav
"""

import argparse, os, sys, time, io
import torch
import numpy as np

TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'train')
sys.path.insert(0, TRAIN_DIR)
from model import get_model  # noqa
from dataset import CLASS_NAMES, NUM_CLASSES, N_MFCC, N_FRAMES, SAMPLE_RATE  # noqa

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def load_quantized(path, arch):
    m = get_model(arch=arch, num_classes=NUM_CLASSES); m.eval()
    if arch == 'dscnn' and hasattr(m, 'fuse_model'): m.fuse_model()
    m.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(m, inplace=True)
    torch.quantization.convert(m, inplace=True)
    ckpt = torch.load(path, map_location='cpu')
    m.load_state_dict(ckpt['model_state_dict']); m.eval()
    return m, ckpt.get('arch', arch)


def wav_to_mfcc(waveform: np.ndarray) -> torch.Tensor:
    """
    Convert raw waveform to MFCC tensor.
    waveform: (16000,) float32 numpy array @ 16kHz
    Returns: (1, 1, N_FRAMES, N_MFCC) tensor
    """
    try:
        import torchaudio.transforms as AT
        import torch.nn.functional as F
        wav = torch.from_numpy(waveform).unsqueeze(0)
        mfcc_fn = AT.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
                           melkwargs={'n_fft': 512, 'win_length': 400,
                                      'hop_length': 160, 'n_mels': 40})
        mfcc = mfcc_fn(wav)  # (1, N_MFCC, T)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        if mfcc.shape[-1] < N_FRAMES:
            mfcc = F.pad(mfcc, (0, N_FRAMES - mfcc.shape[-1]))
        mfcc = mfcc[:, :, :N_FRAMES].permute(0, 2, 1)  # (1, N_FRAMES, N_MFCC)
        return mfcc.unsqueeze(0)
    except ImportError:
        # Fallback: random MFCC
        return torch.randn(1, 1, N_FRAMES, N_MFCC)


def export_onnx():
    """
    Export DS-CNN to ONNX for hls4ml conversion.

    hls4ml path for DS-CNN:
        DS-CNN maps very well to FPGA — depthwise conv → row-wise MAC arrays.

        import hls4ml, onnx
        model_onnx = onnx.load('checkpoints/dscnn_kws.onnx')

        config = hls4ml.utils.config_from_onnx_model(
            model_onnx, default_precision='ap_fixed<8,4>',
            default_reuse_factor=1)
        # Depthwise layers: separate channel processing → reuse_factor=1
        for name in config['LayerName']:
            if 'depthwise' in name.lower() or 'dw' in name.lower():
                config['LayerName'][name]['ReuseFactor'] = 1

        hls_model = hls4ml.converters.convert_from_onnx_model(
            model_onnx, hls_config=config,
            output_dir='hls/hls4ml_dscnn',
            part='xc7z020clg400-1',
            io_type='io_stream')
        hls_model.compile()

    MFCC on FPGA:
        MFCC computation (FFT + Mel filterbank) can also be accelerated on PL.
        See pynq_fft1d_medical for FFT HLS reference.
        MFCC pipeline: windowing → FFT → power spectrum → Mel filterbank → log → DCT
    """
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'), map_location='cpu')
    arch = ckpt.get('arch', 'dscnn')
    model = get_model(arch=arch, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    dummy = torch.randn(1, 1, N_FRAMES, N_MFCC)
    out_path = os.path.join(CHECKPOINT_DIR, 'dscnn_kws.onnx')
    torch.onnx.export(model, dummy, out_path, opset_version=13,
                      input_names=['mfcc'], output_names=['keyword_logits'])
    print(f"ONNX saved: {out_path} ({os.path.getsize(out_path)/1e6:.3f} MB)")


def stream_microphone(model, threshold=0.85):
    """
    Real-time keyword detection from microphone.
    Requires: pip install pyaudio
    """
    try:
        import pyaudio
    except ImportError:
        print("Install pyaudio: pip install pyaudio"); sys.exit(1)

    CHUNK = SAMPLE_RATE  # 1-second chunks
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1,
                    rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)
    print(f"Listening for keywords: {CLASS_NAMES[:10]}")
    print("Press Ctrl+C to stop.\n")
    model.eval()
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            waveform = np.frombuffer(data, dtype=np.float32)
            x = wav_to_mfcc(waveform)
            with torch.no_grad():
                t0 = time.perf_counter()
                out = model(x)
                ms = (time.perf_counter() - t0) * 1000
            probs = torch.softmax(out, 1)[0]
            pred = probs.argmax().item(); conf = probs[pred].item()
            if pred < 10 and conf > threshold:  # keyword (not silence/unknown)
                print(f"  [{ms:.1f}ms] KEYWORD: '{CLASS_NAMES[pred]}' (conf={conf:.3f})")
            else:
                print(f"  [{ms:.1f}ms] silence/unknown (conf={conf:.3f})", end='\r')
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream(); stream.close(); p.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo',        action='store_true')
    parser.add_argument('--export-onnx', action='store_true')
    parser.add_argument('--stream',      action='store_true', help='Real-time mic input')
    parser.add_argument('--audio',       type=str, help='.wav file')
    parser.add_argument('--checkpoint',  default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    args = parser.parse_args()

    if args.export_onnx: export_onnx(); return

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Not found: {ckpt_path}\nRun train.py then quantize.py."); sys.exit(1)

    model, arch = load_quantized(ckpt_path, 'dscnn')
    print(f"Loaded quantized {arch} KWS model.")

    if args.stream:
        stream_microphone(model); return

    # Process audio file or demo
    if args.audio and os.path.exists(args.audio):
        try:
            import torchaudio
            wav, sr = torchaudio.load(args.audio)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            waveform = wav[0, :SAMPLE_RATE].numpy()
        except Exception as e:
            print(f"Load failed: {e}"); waveform = np.random.randn(SAMPLE_RATE).astype(np.float32)
    else:
        waveform = np.random.randn(SAMPLE_RATE).astype(np.float32)
        print("(Using random audio — pass --audio word.wav for real inference)")

    x = wav_to_mfcc(waveform)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(x)
        ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(out, 1)[0]
    pred = probs.argmax().item()
    print(f"\nKeyword: '{CLASS_NAMES[pred]}' (conf={probs[pred]:.3f}, {ms:.3f}ms)")
    print("\nTop-5 predictions:")
    top5 = probs.topk(5)
    for p, i in zip(top5.values, top5.indices):
        bar = '█' * int(p * 25)
        print(f"  {CLASS_NAMES[i]:<12} {p:.4f} {bar}")

    # Throughput benchmark
    n = 2000
    with torch.no_grad():
        t = time.perf_counter()
        for _ in range(n): model(torch.randn(1, 1, N_FRAMES, N_MFCC))
    avg = (time.perf_counter() - t) / n * 1000
    print(f"\nAvg: {avg:.3f}ms/inference | Real-time budget: 1000ms → {'✓' if avg < 1000 else '✗'}")


if __name__ == '__main__':
    main()
