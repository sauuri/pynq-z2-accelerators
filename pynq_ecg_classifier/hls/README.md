# HLS Guide — ECG ResNet1D on FPGA

1D ResNet is one of the most FPGA-friendly deep learning architectures:
- No 2D operations — all Conv1D → simple MAC arrays
- Short input (187 samples) → low latency
- Small model (~500K params) → fits in PYNQ-Z2 BRAM/DSPs

Target: < 1ms inference latency (real-time at 125Hz ECG sampling rate)

---

## Path A: hls4ml (Recommended)

1D Conv layers are well-supported by hls4ml.

```bash
# 1. Export ONNX
python host/inference.py --export-onnx
# -> checkpoints/ecg_resnet1d.onnx

# 2. Convert with hls4ml
python - <<'EOF'
import hls4ml, onnx

model_onnx = onnx.load('checkpoints/ecg_resnet1d.onnx')

config = hls4ml.utils.config_from_onnx_model(
    model_onnx, granularity='name',
    default_precision='ap_fixed<16,6>',
    default_reuse_factor=1,
)

# Aggressive quantization for small layers
for layer_name in config['LayerName']:
    config['LayerName'][layer_name]['Precision'] = {
        'weight': 'ap_fixed<8,4>',
        'bias':   'ap_fixed<16,6>',
    }

hls_model = hls4ml.converters.convert_from_onnx_model(
    model_onnx,
    hls_config=config,
    output_dir='hls/hls4ml_ecg',
    part='xc7z020clg400-1',
    clock_period=10,        # 100 MHz
    io_type='io_stream',    # AXI-Stream (for DMA)
)
hls_model.compile()
hls_model.build(reset=False, csim=True, synth=True, cosim=False, export=True)
EOF

# 3. Verify latency
cat hls/hls4ml_ecg/myproject_prj/solution1/syn/report/myproject_csynth.rpt
```

---

## Path B: Manual HLS Conv1D (Maximum Control)

```cpp
// ecg_resnet1d.h
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<16, 6> act_t;
typedef ap_fixed<8,  4> wt_t;

const int SEQ_LEN    = 187;
const int STEM_CH    = 32;
const int NUM_CLASSES = 5;

// Stem: Conv1d(1, 32, k=15, stride=2) + MaxPool(3, stride=2)
// Output length: floor((187 + 2*7 - 15)/2 + 1) = 94 → MaxPool → 47

void ecg_resnet1d(
    hls::stream<act_t> &ecg_in,    // 187 samples
    hls::stream<act_t> &logits_out, // 5 class logits
    wt_t weights[...],
    act_t biases[...]
);

// Key pragmas for Conv1D:
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=shift_reg complete  // sliding window buffer
```

### Sliding Window Buffer for Conv1D
```cpp
// Efficient 1D convolution with shift register
act_t shift_reg[KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=shift_reg complete

for (int i = 0; i < seq_len; i++) {
    #pragma HLS PIPELINE II=1
    // Shift register
    for (int k = KERNEL_SIZE-1; k > 0; k--) shift_reg[k] = shift_reg[k-1];
    shift_reg[0] = input[i];
    // Compute dot product
    act_t acc = 0;
    for (int k = 0; k < KERNEL_SIZE; k++) acc += shift_reg[k] * weight[k];
    output[i] = relu(acc + bias);
}
```

---

## Real-Time ECG Pipeline on PYNQ-Z2

```
[ECG Sensor — ADS1115 ADC]
    ↓ I2C @ 125Hz (8ms/sample)
[PS: Python preprocessing]
    normalize → pad → tensor (1, 1, 187)
    ↓ AXI-DMA transfer (~0.1ms)
[PL: ResNet1D HLS]
    inference (~0.5ms target)
    ↓ AXI-DMA return
[PS: Decision logic]
    argmax → alarm if arrhythmia
    ↓
[Display / Alert]
```

PYNQ code:
```python
from pynq import Overlay
import numpy as np

overlay = Overlay('/home/xilinx/ecg_resnet1d.bit')
dma = overlay.axi_dma_0

def classify_ecg(segment_187: np.ndarray) -> int:
    """Classify 187-sample ECG segment. Returns class 0-4."""
    seg = (segment_187 - segment_187.min()) / (segment_187.ptp() + 1e-8)

    in_buf  = pynq.allocate((187,), dtype=np.float32)
    out_buf = pynq.allocate((5,),   dtype=np.float32)
    np.copyto(in_buf, seg.astype(np.float32))

    dma.sendchannel.transfer(in_buf)
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.wait(); dma.recvchannel.wait()
    return int(np.argmax(out_buf))
```

---

## Resource Estimates (ResNet1D, ap_fixed<8,4>)

| Resource | Available | Estimate (reuse=1) | Estimate (reuse=4) |
|---|---|---|---|
| LUTs | 53,200 | ~25,000 | ~10,000 |
| FFs | 106,400 | ~18,000 | ~8,000 |
| BRAM | 140 | ~30 | ~30 |
| DSPs | 220 | ~120 | ~35 |

**Recommended**: reuse_factor=1 (fits comfortably, ~0.5ms latency)

---

## Medical Device Note / 의료기기 주의사항

This project is for educational and research purposes only.
For clinical ECG analysis, FDA/CE certification is required.
The model is trained on MIT-BIH dataset (synthetic fallback) —
real-world performance requires clinical validation.

---

## References

- [ECG hls4ml paper (TNNLS 2023)](https://arxiv.org/abs/2101.05031)
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [ADS1115 + PYNQ integration](https://pynq.readthedocs.io/)
- [hls4ml Conv1D support](https://fastmachinelearning.org/hls4ml/)
