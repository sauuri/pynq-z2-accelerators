"""
Evaluation: Float vs. Quantized detector.
Metrics: model size, inference latency, mAP (if ground truth available).

Usage:
    python evaluate.py
    python evaluate.py --use-yolo --weights checkpoints/yolov8n_finetune/weights/best.pt
"""

import argparse
import os
import sys
import time
import io

import torch

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def model_size(model):
    b = io.BytesIO(); torch.save(model.state_dict(), b); return b.tell() / 1e6


def latency(model, img_size=300, n=200):
    model.eval(); dummy = torch.randn(1, 3, img_size, img_size)
    for _ in range(30):
        with torch.no_grad(): model(dummy)
    t = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): model(dummy)
    return (time.perf_counter() - t) / n * 1000


def evaluate_yolo(weights_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install: pip install ultralytics"); sys.exit(1)

    model = YOLO(weights_path)
    # Val on COCO128 (quick benchmark)
    metrics = model.val(data='coco128.yaml', imgsz=640, batch=16)
    print(f"\nYOLOv8n Metrics:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    # Latency
    results = model.benchmark(imgsz=640, half=False)
    print(results)


def evaluate_light_detector():
    sys.path.insert(0, os.path.dirname(__file__))
    from model import get_model

    float_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    quant_path = os.path.join(CHECKPOINT_DIR, 'quantized_model.pth')

    if not os.path.exists(float_path):
        print("Run train.py first."); sys.exit(1)

    ckpt = torch.load(float_path, map_location='cpu')
    num_classes = ckpt.get('num_classes', 21)
    img_size    = ckpt.get('img_size', 300)

    float_m = get_model(num_classes=num_classes)
    float_m.load_state_dict(ckpt['model_state_dict']); float_m.eval()

    results = {'Float (FP32)': {
        'size_mb': model_size(float_m),
        'latency_ms': latency(float_m, img_size),
    }}

    if os.path.exists(quant_path):
        q_ckpt = torch.load(quant_path, map_location='cpu')
        q_model = get_model(num_classes=num_classes); q_model.eval()
        q_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(q_model, inplace=True)
        torch.quantization.convert(q_model, inplace=True)
        q_model.load_state_dict(q_ckpt['model_state_dict'])
        results['Quantized (INT8)'] = {
            'size_mb': model_size(q_model),
            'latency_ms': latency(q_model, img_size),
        }

    print(f"\n{'='*52}")
    print(f"{'Model':<22} {'Size(MB)':>10} {'Latency(ms)':>14}")
    print(f"{'-'*52}")
    for name, m in results.items():
        print(f"{name:<22} {m['size_mb']:>10.2f} {m['latency_ms']:>13.3f}")
    print(f"{'='*52}")

    if len(results) == 2:
        f, q = results['Float (FP32)'], results['Quantized (INT8)']
        print(f"\nSize: {f['size_mb']/q['size_mb']:.2f}×  |  "
              f"Speedup: {f['latency_ms']/q['latency_ms']:.2f}×")
        print(f"Note: mAP requires a labelled validation set (COCO/VOC)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-yolo', action='store_true')
    parser.add_argument('--weights',  default=os.path.join(
        CHECKPOINT_DIR, 'yolov8n_finetune/weights/best.pt'))
    args = parser.parse_args()

    if args.use_yolo:
        evaluate_yolo(args.weights)
    else:
        evaluate_light_detector()


if __name__ == '__main__':
    main()
