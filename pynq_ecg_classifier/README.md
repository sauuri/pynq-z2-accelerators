# pynq_ecg_classifier

**1D ResNet ECG Arrhythmia Classifier + Quantization + FPGA Real-Time Deployment**
ECG 부정맥 분류 1D ResNet 양자화 및 PYNQ-Z2 실시간 배포 파이프라인

---

## Overview / 개요

Trains a lightweight 1D ResNet (~500K params) on the MIT-BIH Arrhythmia dataset
to classify 5 types of heartbeats. Applies PTQ (INT8) and deploys on PYNQ-Z2
for real-time ECG monitoring at 125Hz.

MIT-BIH 부정맥 데이터셋으로 경량 1D ResNet(~500K 파라미터)을 학습하고,
PTQ(INT8)를 적용한 뒤 PYNQ-Z2에서 125Hz 실시간 ECG 모니터링에 배포합니다.

---

## Dataset / 데이터셋

**MIT-BIH Arrhythmia** (Kaggle preprocessed CSV version)
- 109,446 ECG segments (87,554 train / 21,892 test)
- Segment length: 187 samples @ 125Hz = 1.5 seconds
- 5 arrhythmia classes:

| Class | Type | Prevalence |
|-------|------|-----------|
| N | Normal | 82.8% |
| S | Supraventricular | 2.8% |
| V | Ventricular | 6.6% |
| F | Fusion | 0.7% |
| Q | Unknown | 7.1% |

> **Download**: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
> Place `mitbih_train.csv` and `mitbih_test.csv` in `data/`
> If not found: synthetic data is auto-generated for demo

---

## Architecture / 아키텍처

```
Input (1×187) — 1 ECG channel, 187 samples
    ↓
Stem: Conv1d(1→32, k=15, stride=2) + BN + ReLU + MaxPool
    → (32×47)
    ↓
Layer1: 2× ResBlock1D(32→32,  k=7, stride=1)  → (32×47)
Layer2: 2× ResBlock1D(32→64,  k=7, stride=2)  → (64×24)
Layer3: 2× ResBlock1D(64→128, k=7, stride=2)  → (128×12)
Layer4: 2× ResBlock1D(128→256,k=7, stride=2)  → (256×6)
    ↓
AdaptiveAvgPool → Dropout(0.5) → Linear(256→5)
    ↓
Output (5 classes)
```

- Parameters: ~500K
- Target: Accuracy >98%, Macro-F1 >0.90

---

## Quickstart / 빠른 시작

```bash
pip install -r requirements.txt

# (Optional) Download MIT-BIH CSV to data/
# Otherwise synthetic data is auto-generated

# Train
python train/train.py --epochs 50

# Quantize (PTQ)
python train/quantize.py

# Evaluate (accuracy, F1, size, latency)
python train/evaluate.py

# Export ONNX
python host/inference.py --export-onnx

# Demo inference (single ECG segment)
python host/inference.py --demo

# Real ECG segment
python host/inference.py --signal my_ecg.npy
```

---

## Results / 결과 (expected, with real MIT-BIH data)

| Model | Accuracy | Macro-F1 | Size | Latency |
|-------|----------|----------|------|---------|
| ResNet1D Float | ~98.5% | ~0.93 | ~2 MB | ~0.8 ms |
| ResNet1D INT8 | ~98.3% | ~0.92 | ~0.5 MB | ~0.4 ms |

Real-time constraint: 125Hz → 8ms budget → **✓ easily met**

---

## FPGA Deployment / FPGA 배포

1D Conv ResNet is ideal for FPGA:
- All operations are 1D → simple MAC arrays
- Small model → fits in PYNQ-Z2 BRAM

```
ECG Sensor (ADS1115) → I2C → PS (preprocess) → DMA → PL (ResNet1D HLS)
→ DMA → PS (alarm/display)
Target: < 1ms inference on FPGA (vs. 8ms budget at 125Hz)
```

See [`hls/README.md`](hls/README.md) for full hls4ml synthesis guide + PYNQ code.

---

## Story / 포트폴리오 스토리

`pynq_fft1d_medical` → **`pynq_ecg_classifier`**
- FFT: *frequency domain analysis of ECG*
- CNN: *deep learning classification of arrhythmia*
- Together: complete medical signal processing pipeline on FPGA

---

## Disclaimer / 면책사항

본 프로젝트는 교육 및 연구 목적으로만 사용됩니다.
임상 ECG 분석에는 FDA/CE 인증이 필요하며, 실제 의료 진단에 사용해서는 안 됩니다.

---

## Related Projects / 관련 프로젝트

- [`pynq_fft1d_medical`](../pynq_fft1d_medical/) — 1D FFT for medical signals
- [`pynq_lstm_accel`](../pynq_lstm_accel/) — LSTM time-series classification
- [`pynq_kws_accel`](../pynq_kws_accel/) — Keyword spotting (audio)
