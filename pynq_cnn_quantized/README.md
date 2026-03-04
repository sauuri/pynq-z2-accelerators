# pynq_cnn_quantized

**CNN Quantization + FPGA Deployment on PYNQ-Z2**
경량 CNN 양자화(PTQ) 및 PYNQ-Z2 FPGA 배포 파이프라인

---

## Overview / 개요

This project implements a lightweight CNN trained on CIFAR-10, applies Post-Training Static Quantization (INT8), and prepares the model for FPGA deployment on the PYNQ-Z2 board via hls4ml or FINN.

CIFAR-10 분류를 위한 경량 CNN을 학습하고, 사후 정적 양자화(INT8)를 적용한 뒤 hls4ml 또는 FINN을 통해 PYNQ-Z2 FPGA에 배포하는 엔드-투-엔드 파이프라인입니다.

---

## Architecture / 아키텍처

```
Input (3×32×32)
    ↓
Conv(64) + BN + ReLU  ×2  → MaxPool  (→ 64×16×16)
    ↓
Conv(128) + BN + ReLU ×2  → MaxPool  (→ 128×8×8)
    ↓
Conv(256) + BN + ReLU ×1  → MaxPool  (→ 256×4×4)
    ↓
Flatten → FC(512) → ReLU → FC(10)
    ↓
Output (10 classes)
```

- Parameters: ~4.5M
- Target accuracy: ~90% (CIFAR-10 validation)
- Quantization: Post-Training Static Quantization (fbgemm backend)

---

## Project Structure / 프로젝트 구조

```
pynq_cnn_quantized/
├── README.md
├── requirements.txt
├── train/
│   ├── model.py        # CNN architecture with QuantStub/DeQuantStub
│   ├── train.py        # CIFAR-10 training loop
│   ├── quantize.py     # PTQ (Post-Training Static Quantization)
│   └── evaluate.py     # Float vs. quantized model comparison
├── host/
│   └── inference.py    # Inference script + ONNX export + PYNQ stub
└── hls/
    └── README.md       # HLS synthesis guide (hls4ml / FINN)
```

---

## Quickstart / 빠른 시작

```bash
# 1. Install dependencies / 의존성 설치
pip install -r requirements.txt

# 2. Train (30 epochs) / 학습
python train/train.py --epochs 30 --lr 0.01 --batch-size 128

# 3. Apply PTQ / 양자화 적용
python train/quantize.py

# 4. Compare float vs. quantized / 비교 평가
python train/evaluate.py

# 5. (Optional) Export ONNX for HLS toolchain / ONNX 내보내기
python host/inference.py --export-onnx
```

---

## Results / 결과 (expected)

| Model | Top-1 Acc | Size (MB) | Latency (ms/sample) |
|-------|-----------|-----------|---------------------|
| Float (FP32) | ~90% | ~17 MB | ~2.5 ms |
| Quantized (INT8) | ~89% | ~4.5 MB | ~0.8 ms |

Accuracy drop: < 1%
Size reduction: ~4×
Speedup: ~3×

---

## FPGA Deployment / FPGA 배포

PYNQ-Z2에 배포하는 두 가지 경로:

| Path | Tool | Input | Notes |
|------|------|-------|-------|
| A | **hls4ml** | ONNX | Easier setup, FP32 or fixed-point |
| B | **FINN** | Brevitas QONNX | Higher throughput, streaming dataflow |

See [`hls/README.md`](hls/README.md) for detailed synthesis instructions.

---

## Hardware / 하드웨어

- **Training**: Any machine with PyTorch ≥ 2.0 (CUDA/MPS/CPU)
- **Quantization**: CPU (PyTorch fbgemm backend)
- **FPGA target**: PYNQ-Z2 (Zynq-7020, xc7z020clg400-1)

---

## Workflow / 작업 흐름

```
[MacBook]          [GPU 서버 / 다른 PC]           [PYNQ-Z2]
코드 작성    →    학습 + 양자화 + ONNX 내보내기  →  hls4ml 합성 + 배포
model.py          train.py                          hls/README.md
                  quantize.py                       host/inference.py (--pynq)
                  evaluate.py
```

---

## References / 참고자료

- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [hls4ml](https://fastmachinelearning.org/hls4ml/)
- [FINN (Xilinx)](https://finn.readthedocs.io/)
- [Brevitas (QAT)](https://github.com/Xilinx/brevitas)
- [PYNQ-Z2](http://www.pynq.io/board.html)

---

## Related Projects in this Repo / 관련 프로젝트

- [`pynq_saxpy_accel`](../pynq_saxpy_accel/) — SAXPY hardware accelerator
- [`pynq_fft1d_medical`](../pynq_fft1d_medical/) — 1D FFT for medical signal processing
