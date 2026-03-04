# pynq_lstm_accel

**LSTM Quantization + FPGA Deployment for Human Activity Recognition**
스마트폰 센서 기반 행동 인식(HAR) LSTM 양자화 및 PYNQ-Z2 FPGA 배포 파이프라인

---

## Overview / 개요

This project trains an LSTM (and optional CNN-LSTM hybrid) on the **UCI HAR dataset**
(smartphone accelerometer + gyroscope, 6 activity classes), applies
**Dynamic Quantization** (INT8 weights), and prepares the model for FPGA deployment.

UCI HAR 데이터셋(스마트폰 가속도계+자이로스코프, 6가지 활동 분류)으로 LSTM을 학습하고,
동적 양자화(INT8)를 적용한 뒤 PYNQ-Z2 FPGA에 배포하는 엔드-투-엔드 파이프라인입니다.

---

## Dataset / 데이터셋

**UCI Human Activity Recognition (HAR)**
- 30명의 피험자, 스마트폰 착용
- 샘플링: 50Hz, 윈도우: 2.56초 (128 timestep)
- 센서 채널: body_acc (x,y,z), body_gyro (x,y,z), total_acc (x,y,z) → 9채널
- 학습: 7,352 샘플 / 테스트: 2,947 샘플
- 자동 다운로드 (dataset.py)

**Activity Classes / 활동 분류:**

| Label | Activity |
|-------|----------|
| 0 | WALKING |
| 1 | WALKING_UPSTAIRS |
| 2 | WALKING_DOWNSTAIRS |
| 3 | SITTING |
| 4 | STANDING |
| 5 | LAYING |

---

## Architecture / 아키텍처

### Option A: Pure LSTM (`--arch lstm`)
```
Input (128×9)
    ↓
LSTM(128 units, 2 layers, dropout=0.3)
    ↓  [last timestep]
Dropout(0.3) → FC(128→64) → ReLU → FC(64→6)
    ↓
Output (6 classes)
```

### Option B: CNN-LSTM Hybrid (`--arch cnn_lstm`)
```
Input (128×9)
    ↓
Conv1D(64) + BN + ReLU ×2  [local feature extraction]
    ↓
LSTM(128 units, 1 layer)   [temporal modeling]
    ↓
Dropout → FC(6)
    ↓
Output (6 classes)
```

> CNN-LSTM often outperforms pure LSTM on HAR (+1~2%).
> Also more FPGA-friendly: CNN on PL, LSTM on PS.

---

## Why Dynamic Quantization (not Static)?
RNN에서 동적 양자화를 쓰는 이유:

| | Static PTQ | Dynamic Quantization |
|---|---|---|
| Activation range | Pre-calibrated | Computed at runtime |
| LSTM support | Limited | ✅ Full support |
| Accuracy loss | ~1-2% | < 0.5% |
| Size reduction | ~4× | ~3× |
| Recommended for | CNN | RNN / LSTM |

---

## Project Structure / 프로젝트 구조

```
pynq_lstm_accel/
├── README.md
├── requirements.txt
├── train/
│   ├── model.py        # LSTMClassifier + CNNLSTMClassifier
│   ├── dataset.py      # UCI HAR auto-download + DataLoader
│   ├── train.py        # Training loop + per-class accuracy
│   ├── quantize.py     # Dynamic quantization (INT8 weights)
│   └── evaluate.py     # Float vs. quantized comparison table
├── host/
│   └── inference.py    # ONNX export + PYNQ stub + demo
└── hls/
    └── README.md       # HLS guide (hls4ml / manual LSTM cell / hybrid)
```

---

## Quickstart / 빠른 시작

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (data auto-downloads ~25MB)
python train/train.py --epochs 50 --arch lstm

# Or CNN-LSTM (usually better accuracy):
python train/train.py --epochs 50 --arch cnn_lstm

# 3. Apply dynamic quantization
python train/quantize.py

# 4. Compare float vs. quantized
python train/evaluate.py

# 5. Run demo inference
python host/inference.py --demo

# 6. (Optional) Export ONNX for HLS toolchain
python host/inference.py --export-onnx
```

---

## Results / 결과 (expected)

| Model | Top-1 Acc | Size (MB) | Latency (ms/sample) |
|-------|-----------|-----------|---------------------|
| LSTM Float (FP32) | ~91% | ~2.0 MB | ~1.5 ms |
| LSTM Quantized (INT8) | ~91% | ~0.6 MB | ~0.8 ms |
| CNN-LSTM Float | ~93% | ~2.5 MB | ~1.8 ms |
| CNN-LSTM Quantized | ~92% | ~0.7 MB | ~1.0 ms |

Accuracy drop: < 0.5%
Size reduction: ~3×
Speedup: ~1.5–2×

---

## FPGA Deployment Strategy / FPGA 배포 전략

Three options ranked by difficulty:

```
Easy    → Hard
  ↓
[A] hls4ml ONNX → HLS (experimental LSTM support)
[B] Manual HLS LSTM cell (full control, best performance)
[C] Hybrid: CNN on PL + LSTM on PS (most practical for PYNQ-Z2)
```

See [`hls/README.md`](hls/README.md) for detailed instructions.

**Recommended for PYNQ-Z2**: Option C (Hybrid)
- CNN feature extractor → FPGA (PL side, ~10× speedup)
- Quantized LSTM → ARM Cortex-A9 (PS side, ~2× speedup)

---

## Hardware / 하드웨어

- **Training**: Any machine with PyTorch ≥ 2.0 (CUDA/MPS/CPU)
- **Quantization**: CPU (PyTorch dynamic quant)
- **FPGA target**: PYNQ-Z2 (Zynq-7020, xc7z020clg400-1)

---

## Workflow / 작업 흐름

```
[MacBook]     [GPU 서버 / 다른 PC]                    [PYNQ-Z2]
코드 작성  →  train.py → quantize.py → evaluate.py  →  HLS 합성 + 배포
              (UCI HAR 자동 다운로드)                   host/inference.py (--pynq)
```

---

## References / 참고자료

- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- [PyTorch Dynamic Quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)
- [hls4ml RNN support](https://fastmachinelearning.org/hls4ml/advanced/rnn.html)
- [LSTM on FPGA Survey (arXiv)](https://arxiv.org/abs/1901.02129)
- [PYNQ-Z2](http://www.pynq.io/board.html)

---

## Related Projects / 관련 프로젝트

- [`pynq_saxpy_accel`](../pynq_saxpy_accel/) — SAXPY hardware accelerator
- [`pynq_fft1d_medical`](../pynq_fft1d_medical/) — 1D FFT for medical signals
- [`pynq_cnn_quantized`](../pynq_cnn_quantized/) — CNN PTQ + FPGA deployment
