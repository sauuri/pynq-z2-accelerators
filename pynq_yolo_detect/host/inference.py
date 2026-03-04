"""
Object detection inference for PYNQ-Z2.
Supports YOLOv8n (ultralytics) or LightDetector.

Usage:
    python inference.py --demo --image sample.jpg
    python inference.py --export-onnx
    python inference.py --use-yolo --weights best.pt --image img.jpg
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'train')
sys.path.insert(0, TRAIN_DIR)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

VOC_CLASSES = ['background','aeroplane','bicycle','bird','boat','bottle',
               'bus','car','cat','chair','cow','diningtable','dog','horse',
               'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']


def export_onnx():
    """
    Export LightDetector or YOLOv8n to ONNX.

    YOLOv8n ONNX → hls4ml path:
        YOLOv8n has 168 layers; full conversion exceeds PYNQ-Z2 resources.
        Recommended: export backbone only (first 10 layers = CSPDarknet53 stem)

        from ultralytics import YOLO
        model = YOLO('best.pt')
        model.export(format='onnx', simplify=True, opset=13)

    hls4ml conversion (backbone only, INT8):
        import hls4ml, onnx
        # Use YOLOv8n-nano backbone (≈ 1.8M params)
        model_onnx = onnx.load('yolov8n_backbone.onnx')
        config = hls4ml.utils.config_from_onnx_model(
            model_onnx,
            default_precision='ap_fixed<8,4>',
            default_reuse_factor=4,
        )
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model_onnx, hls_config=config,
            output_dir='hls/hls4ml_output',
            part='xc7z020clg400-1',
            io_type='io_stream',
        )
        hls_model.compile()

    PYNQ deployment strategy (practical for Z2):
        PL: Backbone feature extraction (HLS)
        PS: Detection head + NMS (ARM software)
    """
    from model import get_model
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'), map_location='cpu')
    model = get_model(num_classes=ckpt.get('num_classes', 21))
    model.load_state_dict(ckpt['model_state_dict']); model.eval()

    img_size = ckpt.get('img_size', 300)
    dummy = torch.randn(1, 3, img_size, img_size)
    out_path = os.path.join(CHECKPOINT_DIR, 'light_detector.onnx')
    torch.onnx.export(model, dummy, out_path, opset_version=13,
                      input_names=['image'], output_names=['cls_logits', 'bbox_preds'])
    print(f"ONNX saved: {out_path}")


def infer_yolo(weights, image_path):
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print("Install: pip install ultralytics opencv-python-headless"); sys.exit(1)

    model = YOLO(weights)
    t0 = time.perf_counter()
    results = model(image_path)
    ms = (time.perf_counter() - t0) * 1000
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} objects in {ms:.1f}ms:")
        for box in boxes:
            cls = int(box.cls[0]); conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"  {model.names[cls]:<15} conf={conf:.2f} bbox={[round(v,1) for v in xyxy]}")


def infer_light(image_path=None):
    from model import get_model
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'quantized_model.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    num_classes = ckpt.get('num_classes', 21)
    img_size    = ckpt.get('img_size', 300)
    model = get_model(num_classes=num_classes); model.eval()

    if 'quantized' in ckpt_path:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()

    if image_path and os.path.exists(image_path):
        import torchvision.transforms as T
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                        T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        x = tf(img).unsqueeze(0)
    else:
        x = torch.randn(1, 3, img_size, img_size)
        print("(Using random input — pass --image path.jpg for real inference)")

    with torch.no_grad():
        t0 = time.perf_counter()
        cls_logits, bbox_preds = model(x)
        ms = (time.perf_counter() - t0) * 1000

    probs = cls_logits[0].softmax(-1)
    top_scores, top_cls = probs.max(-1)
    top5_idx = top_scores.topk(5).indices
    print(f"Inference: {ms:.2f}ms | Top-5 detections:")
    for i in top5_idx:
        c = int(top_cls[i]); s = float(top_scores[i])
        b = bbox_preds[0][i].tolist()
        print(f"  {VOC_CLASSES[c]:<15} score={s:.3f} box={[round(v,3) for v in b]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo',        action='store_true')
    parser.add_argument('--export-onnx', action='store_true')
    parser.add_argument('--use-yolo',    action='store_true')
    parser.add_argument('--weights',     default=os.path.join(
        CHECKPOINT_DIR, 'yolov8n_finetune/weights/best.pt'))
    parser.add_argument('--image',       type=str, default=None)
    args = parser.parse_args()

    if args.export_onnx:
        export_onnx(); return
    if args.use_yolo:
        infer_yolo(args.weights, args.image); return
    infer_light(args.image)


if __name__ == '__main__':
    main()
