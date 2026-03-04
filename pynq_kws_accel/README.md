# pynq_kws_accel

**Keyword Spotting (KWS) + MFCC + DS-CNN Quantization + FPGA Wake Word Detection**
키워드 인식 DS-CNN 양자화 및 PYNQ-Z2 FPGA 실시간 웨이크워드 감지

---

## Overview / 개요

Implements a complete keyword spotting pipeline:
MFCC feature extraction → DS-CNN / BC-ResNet classification → INT8 quantization
→ PYNQ-Z2 real-time deployment for "Hey device"-style wake word detection.

MFCC 특징 추출 → DS-CNN/BC-ResNet 분류 → INT8 양자화 →
PYNQ-Z2에서 실시간 웨이크워드 감지까지의 완전한 KWS 파이프라인.

---

## Architecture / 아키텍처

### DS-CNN (~100K params, FPGA-optimal)
```
Input: MFCC (1×49×40)   — 1sec @ 16kHz, 40 mel bins
    ↓
Conv2d(64, 10×4, stride=2) + BN + ReLU6 + Dropout
    ↓
4× Depthwise Separable Block:
    DW Conv(64, 3×3) + BN + ReLU6
    PW Conv(64, 1×1) + BN + ReLU6
    ↓
Global AvgPool → Linear(64→12)
    ↓
Output: 12 classes (10 keywords + silence + unknown)
```

### BC-ResNet (~300K params, higher accuracy)
```
Input → Stem Conv → 3 Residual Stages → FC → Output
```

**Target keywords**: yes, no, up, down, left, right, on, off, stop, go

---

## Quickstart / 빠른 시작

```bash
pip install -r requirements.txt

# Dataset: auto-downloads Google Speech Commands (~2.4GB)
# If unavailable: synthetic data used automatically

# Train DS-CNN (faster, FPGA-optimal)
python train/train.py --arch dscnn --epochs 50

# Or BC-ResNet (higher accuracy)
python train/train.py --arch bcresnet --epochs 60

# Quantize (PTQ)
python train/quantize.py

# Evaluate
python train/evaluate.py

# Export ONNX
python host/inference.py --export-onnx

# Demo (random input)
python host/inference.py --demo

# Real audio file
python host/inference.py --audio keyword.wav

# Real-time microphone (requires pyaudio)
python host/inference.py --stream
```

---

## Results / 결과 (expected, Google Speech Commands)

| Model | Acc (10-cmd) | Size | Latency |
|-------|-------------|------|---------|
| DS-CNN Float | ~95% | ~0.5 MB | ~0.8 ms |
| DS-CNN INT8 | ~94.5% | ~0.15 MB | ~0.4 ms |
| BC-ResNet Float | ~97% | ~1.5 MB | ~1.2 ms |
| BC-ResNet INT8 | ~96.5% | ~0.4 MB | ~0.6 ms |

Real-time budget: 1 second window → any latency < 100ms is fine

---

## Full FPGA Pipeline / FPGA 파이프라인

```
[MEMS Microphone]
    ↓ I2S/PDM
[PS: audio capture @ 16kHz]
    ↓
[PL (optional): MFCC accelerator]
  FFT (reuse pynq_fft1d_medical) + Mel filterbank
    ↓
[PL: DS-CNN HLS] ← 14k LUTs, 30 BRAMs, 75 DSPs
  inference < 1ms
    ↓
[PS: argmax + debounce]
    ↓
Wake word → trigger action
```

DS-CNN는 PYNQ-Z2에서 가장 구현하기 좋은 딥러닝 모델 중 하나.
전체 모델이 BRAM에 올라가므로 DRAM 접근 없이 초저지연 추론 가능.

---

## Why DS-CNN for FPGA / DS-CNN을 FPGA에 쓰는 이유

| | Standard CNN | DS-CNN |
|---|---|---|
| MACs | N×C_in×C_out×K² | N×C×K² + N×C_in×C_out |
| Parameters | C_in×C_out×K² | C×K² + C_in×C_out |
| FPGA MACs | High | **~8-9× lower** |
| hls4ml support | ✅ | ✅ |

---

## Connection to Other Projects / 다른 프로젝트 연결

```
pynq_fft1d_medical  →  FFT HLS core reuse for MFCC accelerator
pynq_ecg_classifier →  Same ResBlock1D pattern (1D → 2D)
pynq_cnn_quantized  →  PTQ pipeline reference
```

---

## Related Projects / 관련 프로젝트

- [`pynq_fft1d_medical`](../pynq_fft1d_medical/) — FFT HLS (reusable for MFCC)
- [`pynq_cnn_quantized`](../pynq_cnn_quantized/) — CNN PTQ baseline
- [`pynq_ecg_classifier`](../pynq_ecg_classifier/) — ECG 1D signal classification
- [`pynq_transformer_edge`](../pynq_transformer_edge/) — Attention-based models

---

## References / 참고자료

- [Hello Edge: KWS on MCUs](https://arxiv.org/abs/1711.07128) — DS-CNN origin paper
- [BC-ResNet](https://arxiv.org/abs/2106.04140) — broadcast residual KWS
- [Google Speech Commands](https://arxiv.org/abs/1804.03209) — dataset paper
- [DS-CNN on FPGA (FPGA'21)](https://dl.acm.org/doi/10.1145/3431920.3439295)
