# pynq_transformer_edge

**Tiny Vision Transformer (TinyViT) + Quantization + FPGA Edge Deployment**
경량 Vision Transformer 양자화 및 PYNQ-Z2 엣지 배포 파이프라인

---

## Overview / 개요

Custom TinyViT (3M params) trained on CIFAR-10, quantized with PTQ (INT8),
and deployed on PYNQ-Z2 using a hybrid PL+PS strategy.
Attention on ARM PS, patch embedding accelerated on PL via HLS.

커스텀 TinyViT(3M 파라미터)를 CIFAR-10으로 학습하고 PTQ(INT8)로 양자화.
Attention은 ARM PS, Patch Embedding은 PL(HLS) 가속하는 하이브리드 배포 전략 사용.

---

## Architecture / 아키텍처

```
Input (3×32×32)
    ↓
Patch Embedding (4×4 Conv, stride 4) → 64 patches × 64-dim
    ↓
[CLS] token + Positional Embedding
    ↓
6× Transformer Block:
    LayerNorm → Multi-Head Attention (4 heads) → Residual
    LayerNorm → MLP (64→256→64, GELU) → Residual
    ↓
LayerNorm → CLS token → Linear(64→10)
    ↓
Output (10 classes)
```

| Spec | Value |
|---|---|
| Parameters | ~3M |
| Patch size | 4×4 → 64 patches |
| Embed dim | 64 |
| Depth | 6 blocks |
| Attention heads | 4 |
| Target accuracy | ~85–88% (CIFAR-10) |

---

## Quickstart / 빠른 시작

```bash
pip install -r requirements.txt

# Train (100 epochs, AdamW + warmup + cosine decay)
python train/train.py --epochs 100 --lr 3e-4

# Quantize (PTQ static)
python train/quantize.py

# Evaluate
python train/evaluate.py

# Export ONNX
python host/inference.py --export-onnx

# Demo inference
python host/inference.py --demo
```

---

## Results / 결과 (expected)

| Model | Top-1 Acc | Size | Latency |
|-------|-----------|------|---------|
| TinyViT Float | ~87% | ~12 MB | ~5 ms |
| TinyViT INT8 | ~86% | ~3 MB | ~2 ms |

*ViT needs more epochs than CNN — use 100+ for best results.*

---

## FPGA Strategy / FPGA 전략

**Recommended: Hybrid PL+PS**
```
PL (HLS): Patch Embedding (Conv2d 4×4) → ~10× speedup
PS (ARM): Transformer blocks (quantized INT8)
```

See [`hls/README.md`](hls/README.md) for:
- Hybrid deployment implementation
- Linear attention (HLS-friendly alternative)
- hls4ml MLP-only export

---

## Related Projects / 관련 프로젝트

- [`pynq_cnn_quantized`](../pynq_cnn_quantized/) — CNN PTQ baseline
- [`pynq_lstm_accel`](../pynq_lstm_accel/) — LSTM + dynamic quantization
- [`pynq_yolo_detect`](../pynq_yolo_detect/) — Object detection
