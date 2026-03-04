# HLS Guide — Keyword Spotting on FPGA

DS-CNN is one of the best-suited KWS architectures for FPGA:
- Depthwise separable conv → low MACs per layer
- Small input (49×40 MFCC) → fast latency
- ~100K parameters → fits entirely in PYNQ-Z2 on-chip memory (BRAM)

Target: < 5ms end-to-end (MFCC extraction + inference)

---

## Full Pipeline on PYNQ-Z2

```
[MEMS Microphone — PDM/I2S]
    ↓
[PS: Audio capture + decimation @ 16kHz]
    ↓
[PL: MFCC accelerator] (optional — see Path B)
  Windowing → FFT → Mel filterbank → log → DCT
    ↓
[PL: DS-CNN HLS] (49×40 MFCC → 12 classes)
    ↓
[PS: Argmax + debounce logic]
    ↓
Wake word detected → action
```

---

## Path A: hls4ml (DS-CNN)

DS-CNN exports cleanly to ONNX and maps well to hls4ml.

```bash
# 1. Export ONNX
python host/inference.py --export-onnx

# 2. Convert with hls4ml
python - <<'EOF'
import hls4ml, onnx

model_onnx = onnx.load('checkpoints/dscnn_kws.onnx')

config = hls4ml.utils.config_from_onnx_model(
    model_onnx,
    granularity='name',
    default_precision='ap_fixed<8,4>',
    default_reuse_factor=1,
)

# Standard conv layers — higher reuse OK
# Depthwise conv — reuse=1 (process each channel independently)
for layer in config['LayerName']:
    if 'ds' in layer.lower() or 'depth' in layer.lower():
        config['LayerName'][layer]['ReuseFactor'] = 1
    else:
        config['LayerName'][layer]['ReuseFactor'] = 2

hls_model = hls4ml.converters.convert_from_onnx_model(
    model_onnx,
    hls_config=config,
    output_dir='hls/hls4ml_dscnn',
    part='xc7z020clg400-1',
    clock_period=10,       # 100 MHz
    io_type='io_stream',
)
hls_model.compile()
hls_model.build(synth=True, export=True)
EOF
```

---

## Path B: MFCC Accelerator on PL (optional)

Computing MFCC in hardware removes PS bottleneck.

```cpp
// mfcc_accel.h — key stages
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<16, 6> audio_t;
typedef ap_fixed<16, 6> mfcc_t;

const int FRAME_LEN  = 400;   // 25ms @ 16kHz
const int HOP_LEN    = 160;   // 10ms
const int N_FFT      = 512;
const int N_MELS     = 40;
const int N_MFCC     = 40;

// Stage 1: Hamming window
void apply_window(audio_t in[FRAME_LEN], audio_t out[FRAME_LEN]);

// Stage 2: FFT (reuse pynq_fft1d_medical HLS module)
void fft_512(audio_t in[N_FFT], audio_t real[N_FFT/2], audio_t imag[N_FFT/2]);

// Stage 3: Power spectrum + Mel filterbank
void mel_filterbank(audio_t power[N_FFT/2], mfcc_t mel_energy[N_MELS]);

// Stage 4: Log + DCT
void log_dct(mfcc_t mel[N_MELS], mfcc_t mfcc_out[N_MFCC]);
```

> Reuse the FFT core from `pynq_fft1d_medical/hls/` for Stage 2!

---

## Path C: Manual DS-CNN HLS

```cpp
// dscnn.h — depthwise separable conv in HLS
#include <ap_fixed.h>

typedef ap_fixed<8, 4> wt_t;
typedef ap_fixed<16, 6> act_t;

const int IN_H  = 25;   // 49/2 after input stride conv
const int IN_W  = 20;   // 40/2
const int CH    = 64;

// Depthwise: process each channel with separate 3×3 kernel
void depthwise_conv(
    act_t in[CH][IN_H][IN_W],
    act_t out[CH][IN_H][IN_W],
    wt_t  weight[CH][3][3]
) {
    #pragma HLS PIPELINE II=1
    #pragma HLS ARRAY_PARTITION variable=weight complete dim=1  // per-channel
    for (int c = 0; c < CH; c++) {
        #pragma HLS UNROLL factor=4
        for (int h = 1; h < IN_H-1; h++)
        for (int w = 1; w < IN_W-1; w++) {
            act_t acc = 0;
            for (int kh = 0; kh < 3; kh++)
            for (int kw = 0; kw < 3; kw++)
                acc += in[c][h+kh-1][w+kw-1] * weight[c][kh][kw];
            out[c][h][w] = relu6(acc + bias[c]);
        }
    }
}
```

---

## Resource Estimates (DS-CNN, ap_fixed<8,4>)

| Component | LUTs | BRAMs | DSPs |
|---|---|---|---|
| Input Conv (64×10×4) | ~5,000 | ~8 | ~30 |
| DS blocks ×4 (64ch) | ~8,000 | ~20 | ~40 |
| Global AvgPool + FC | ~500 | ~2 | ~5 |
| **Total** | **~14,000** | **~30** | **~75** |

**Excellent fit for PYNQ-Z2** — leaves room for MFCC accelerator.

---

## Microphone Integration on PYNQ-Z2

```python
# PYNQ board — MEMS microphone via PMOD
from pynq.lib import MicArray  # or custom I2S driver
import numpy as np

# Capture 1 second of audio @ 16kHz
audio = mic.capture(duration=1.0, sample_rate=16000)  # (16000,)

# Option A: MFCC on PS + DS-CNN on PL
mfcc = compute_mfcc_ps(audio)     # (1, 1, 49, 40)
result = dscnn_overlay(mfcc)       # PL inference

# Option B: Full pipeline on PL
result = full_kws_overlay(audio)   # audio → keyword
print(f"Detected: {CLASS_NAMES[result]}")
```

---

## References

- [Hello Edge: KWS on MCUs (arXiv:1711.07128)](https://arxiv.org/abs/1711.07128)
- [BC-ResNet KWS (arXiv:2106.04140)](https://arxiv.org/abs/2106.04140)
- [DS-CNN hls4ml (FPGA'21)](https://dl.acm.org/doi/10.1145/3431920.3439295)
- [Google Speech Commands Dataset](https://arxiv.org/abs/1804.03209)
- [MFCC on FPGA](https://ieeexplore.ieee.org/document/8977875)
