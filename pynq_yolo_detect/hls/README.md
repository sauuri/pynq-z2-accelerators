# HLS Guide — Object Detection on FPGA

Object detection on PYNQ-Z2 is resource-intensive.
The practical strategy is to accelerate the backbone (feature extraction)
on PL and run the detection head on PS.

---

## Recommended Architecture: Backbone on PL + Head on PS

```
Camera/Input
    ↓ PL (HLS — fast)
MobileNetV2 Backbone (feature maps)
    ↓ AXI-Stream DMA
PS (ARM — flexible)
SSD Head + NMS → Bounding boxes + Labels
    ↓
Output
```

---

## Path A: LightDetector (MobileNetV2+SSD) via hls4ml

```bash
# 1. Export ONNX
python host/inference.py --export-onnx

# 2. Convert backbone only (first 7 InvertedResidual blocks)
python - <<'EOF'
import hls4ml, onnx

# Load full model ONNX, extract backbone subgraph
model_onnx = onnx.load('checkpoints/light_detector.onnx')

config = hls4ml.utils.config_from_onnx_model(
    model_onnx,
    default_precision='ap_fixed<8,4>',
    default_reuse_factor=4,
)
# DepthwiseSeparableConv — set reuse_factor=1 for pipeline
for name in config['LayerName']:
    if 'dw' in name.lower():
        config['LayerName'][name]['ReuseFactor'] = 1

hls_model = hls4ml.converters.convert_from_onnx_model(
    model_onnx, hls_config=config,
    output_dir='hls/hls4ml_output',
    part='xc7z020clg400-1',
    io_type='io_stream',
    clock_period=10,
)
hls_model.compile()
EOF
```

---

## Path B: YOLOv8n Backbone (CSPDarknet Stem)

YOLOv8n is too large for full PYNQ-Z2 deployment.
Strategy: accelerate first 5 layers (Conv + C2f blocks) on PL.

```python
# Extract YOLOv8n stem (layers 0-4)
from ultralytics import YOLO
import torch

model = YOLO('best.pt').model
stem = torch.nn.Sequential(*list(model.model.children())[:5])
stem.eval()

torch.onnx.export(stem, torch.randn(1,3,640,640),
                  'hls/yolov8n_stem.onnx', opset_version=13)

# Then convert with hls4ml
import hls4ml, onnx
model_onnx = onnx.load('hls/yolov8n_stem.onnx')
config = hls4ml.utils.config_from_onnx_model(
    model_onnx, default_precision='ap_fixed<8,4>')
hls_model = hls4ml.converters.convert_from_onnx_model(
    model_onnx, hls_config=config,
    output_dir='hls/yolo_stem_hls',
    part='xc7z020clg400-1')
```

---

## Non-Maximum Suppression (NMS) in HLS

NMS is sequential by nature but can be parallelized with a sorted-list approach:

```cpp
// nms.h — simple greedy NMS for FPGA
#include <ap_fixed.h>

typedef ap_fixed<16,8> score_t;
typedef ap_fixed<16,8> coord_t;

const int MAX_BOXES = 100;
const float IOU_THRESH = 0.45;
const float CONF_THRESH = 0.25;

void nms_top(
    coord_t boxes[MAX_BOXES][4],
    score_t scores[MAX_BOXES],
    bool    keep[MAX_BOXES],
    int     num_boxes
);
```

For PYNQ-Z2, NMS is best run on PS (ARM) due to its sequential nature.

---

## Input Pipeline: Camera → FPGA

```
[OV7670 Camera / USB Webcam]
    ↓ AXIS Video DMA
[FPGA PL] → resize to 300×300 → backbone → feature maps
    ↓ AXI-DMA
[ARM PS]  → detection head → NMS → display
```

Camera integration:
```python
# On PYNQ board with camera PMOD
from pynq.lib.video import *
hdmi_in = HDMI('in')
hdmi_in.configure(PIXEL_RGB)
frame = hdmi_in.readframe()  # numpy array (H, W, 3)
```

---

## Resource Estimates

| Component | LUTs | BRAMs | DSPs |
|---|---|---|---|
| MobileNetV2 backbone (8-bit) | ~40,000 | ~80 | ~180 |
| SSD Head (PS side) | — | — | — |
| DMA + AXI interconnect | ~3,000 | ~4 | 0 |
| **Total** | **~43,000** | **~84** | **~180** |

Note: Tight fit for xc7z020. Use `reuse_factor=4` to reduce LUT/DSP usage (increases latency).

---

## References

- [YOLOv8 on FPGA (Xilinx)](https://github.com/Xilinx/Vitis-AI/tree/master/examples/yolov8)
- [MobileNet-SSD hls4ml example](https://github.com/fastmachinelearning/hls4ml-tutorial)
- [PYNQ Video library](https://pynq.readthedocs.io/en/latest/pynq_libraries/video.html)
- [NMS on FPGA (arXiv)](https://arxiv.org/abs/2112.06670)
