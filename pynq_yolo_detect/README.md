# pynq_yolo_detect

**Object Detection (YOLOv8n / MobileNetV2+SSD) + Quantization + FPGA Deployment**
객체 검출 모델 양자화 및 PYNQ-Z2 FPGA 배포 파이프라인

---

## Overview / 개요

Two-path object detection pipeline for PYNQ-Z2:
- **Path A** (recommended): YOLOv8n fine-tuning + INT8 export via ultralytics
- **Path B**: Custom MobileNetV2+SSD-Lite, fully PyTorch-native, no extra dependencies

PYNQ-Z2용 객체 검출 파이프라인 (두 가지 경로):
- **경로 A**: YOLOv8n fine-tuning + ultralytics INT8 내보내기 (권장)
- **경로 B**: 커스텀 MobileNetV2+SSD-Lite, 순수 PyTorch

---

## Architecture / 아키텍처

### Path A: YOLOv8n
```
Input (640×640) → CSPDarknet53 backbone → C2f neck → Detection head
  Parameters: 3.2M | mAP50: ~37.3 (COCO) | Quantized: ~1.2M params
```

### Path B: LightDetector (MobileNetV2+SSD-Lite)
```
Input (300×300)
    ↓
MobileNetV2 backbone (InvertedResidual blocks, depthwise separable conv)
    ↓ multi-scale features (stride 8, stride 32)
SSD-Lite head (6 anchors per location)
    ↓
cls_logits + bbox_preds → NMS → Detections
  Parameters: ~4M | FPGA-friendly (no complex necks)
```

---

## Quickstart / 빠른 시작

### YOLOv8n (Recommended / 권장)
```bash
pip install -r requirements.txt
pip install ultralytics

# Fine-tune on COCO128 (auto-downloads ~7MB sample)
python train/train.py --use-yolo --data coco128.yaml --epochs 50

# INT8 quantization
python train/quantize.py --use-yolo \
  --weights checkpoints/yolov8n_finetune/weights/best.pt

# Evaluate
python train/evaluate.py --use-yolo \
  --weights checkpoints/yolov8n_finetune/weights/best.pt

# Inference on image
python host/inference.py --use-yolo \
  --weights checkpoints/yolov8n_finetune/weights/best.pt \
  --image your_image.jpg
```

### LightDetector (No extra deps / 추가 의존성 없음)
```bash
pip install -r requirements.txt

# Train skeleton (full VOC training needs detectron2/torchvision reference)
python train/train.py --img-size 300

# Quantize
python train/quantize.py

# Evaluate (size + latency)
python train/evaluate.py

# Export ONNX
python host/inference.py --export-onnx

# Demo inference
python host/inference.py --demo
```

---

## Results / 결과 (expected)

### YOLOv8n
| Model | mAP50 (COCO) | Size | Latency |
|-------|-------------|------|---------|
| YOLOv8n Float | ~37.3% | ~12 MB | ~6 ms |
| YOLOv8n INT8 | ~36.5% | ~3.5 MB | ~3 ms |

### LightDetector
| Model | Size | Latency (300×300) |
|-------|------|-------------------|
| Float | ~16 MB | ~8 ms |
| INT8 | ~4 MB | ~3 ms |

---

## FPGA Deployment Strategy / FPGA 배포 전략

**PYNQ-Z2 리소스 제약으로 전체 모델 PL 배포 불가 → Hybrid 전략 사용**

```
Camera → PL (HLS): Backbone feature extraction
       → PS (ARM): Detection head + NMS → Output
```

| Stage | Where | Tool |
|-------|-------|------|
| Backbone | PL (FPGA) | hls4ml |
| Detection head | PS (ARM) | PyTorch INT8 |
| NMS | PS (ARM) | torchvision.ops |

See [`hls/README.md`](hls/README.md) for detailed HLS synthesis guide.

---

## Related Projects / 관련 프로젝트

- [`pynq_cnn_quantized`](../pynq_cnn_quantized/) — CNN classification baseline
- [`pynq_transformer_edge`](../pynq_transformer_edge/) — ViT on FPGA
- [`pynq_ecg_classifier`](../pynq_ecg_classifier/) — Medical signal detection
