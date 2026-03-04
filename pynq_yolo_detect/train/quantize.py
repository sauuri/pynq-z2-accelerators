"""
Quantization for object detection models.
Supports:
  A) YOLOv8n (ultralytics) → INT8 via built-in export
  B) LightDetector → PTQ static quantization

Usage:
    # YOLOv8n INT8:
    python quantize.py --use-yolo --weights checkpoints/yolov8n_finetune/weights/best.pt

    # LightDetector PTQ:
    python quantize.py
"""

import argparse
import os
import copy
import io
import sys

import torch

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def quantize_yolo(weights_path: str):
    """
    YOLOv8n INT8 quantization via ultralytics export.
    Generates TFLite INT8 or ONNX INT8 model.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install: pip install ultralytics"); sys.exit(1)

    model = YOLO(weights_path)

    # Export INT8 ONNX (requires calibration data)
    export_path = model.export(
        format='onnx',
        int8=True,
        dynamic=True,
        simplify=True,
    )
    print(f"INT8 ONNX exported: {export_path}")

    # Or export to TFLite INT8 (better FPGA toolchain support)
    tflite_path = model.export(format='tflite', int8=True)
    print(f"TFLite INT8 exported: {tflite_path}")


def get_calib_loader(n=500, img_size=300, batch=8):
    """Random calibration data (replace with real images for production)."""
    data = [torch.randn(3, img_size, img_size) for _ in range(n)]
    dataset = torch.utils.data.TensorDataset(torch.stack(data))
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)


def quantize_light_detector(checkpoint_path: str, output_path: str):
    """PTQ static quantization for LightDetector."""
    sys.path.insert(0, os.path.dirname(__file__))
    from model import get_model

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    num_classes = ckpt.get('num_classes', 21)
    img_size = ckpt.get('img_size', 300)

    model = get_model(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    def sz(m):
        b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1e6
    print(f"Float size: {sz(model):.2f} MB")

    # Quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibrate
    print("Calibrating...")
    calib_loader = get_calib_loader(img_size=img_size)
    model.eval()
    with torch.no_grad():
        for (x,) in calib_loader:
            model(x)

    q_model = copy.deepcopy(model)
    torch.quantization.convert(q_model, inplace=True)

    torch.save({
        'model_state_dict': q_model.state_dict(),
        'num_classes': num_classes,
        'img_size': img_size,
    }, output_path)

    print(f"Quantized size: {sz(q_model):.2f} MB")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-yolo',   action='store_true')
    parser.add_argument('--weights',    default=os.path.join(
        CHECKPOINT_DIR, 'yolov8n_finetune/weights/best.pt'))
    parser.add_argument('--checkpoint', default=os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    parser.add_argument('--output',     default=os.path.join(CHECKPOINT_DIR, 'quantized_model.pth'))
    args = parser.parse_args()

    if args.use_yolo:
        quantize_yolo(args.weights)
    else:
        quantize_light_detector(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
