# HLS Implementation Guide — LSTM on FPGA

LSTM on FPGA is more challenging than CNN due to sequential dependencies.
This guide covers two strategies: direct hls4ml mapping and manual unrolling.

---

## Challenge: Why LSTM is Hard to Accelerate on FPGA

- LSTM has **recurrent dependencies** — each timestep depends on the previous hidden state
- Naive implementation: 128 sequential matrix multiplications (one per timestep)
- Solution strategies:
  1. **Pipelining** — overlap timesteps using double buffering
  2. **Unrolling** — unroll small sequences; trade latency for throughput
  3. **State machine** — implement LSTM cell as FSM with reusable MAC units

---

## Path A: hls4ml (Experimental LSTM Support)

hls4ml ≥ 0.8.0 has experimental support for LSTM via `nnet::lstm`.

```bash
# 1. Export ONNX
python host/inference.py --export-onnx
# -> checkpoints/lstm_model.onnx

# 2. Convert with hls4ml
python - <<'EOF'
import hls4ml
import onnx

model_onnx = onnx.load('checkpoints/lstm_model.onnx')

config = hls4ml.utils.config_from_onnx_model(
    model_onnx,
    granularity='name',
    default_precision='ap_fixed<16,6>',
)
# LSTM-specific: set reuse factor to reduce resource usage
for layer in config['LayerName']:
    config['LayerName'][layer]['ReuseFactor'] = 4

hls_model = hls4ml.converters.convert_from_onnx_model(
    model_onnx,
    hls_config=config,
    output_dir='hls/hls4ml_output',
    part='xc7z020clg400-1',
    clock_period=10,
    io_type='io_stream',   # AXI-Stream for PYNQ DMA
)
hls_model.compile()
EOF
```

**Limitations:**
- Full LSTM unrolling requires significant BRAM (hidden_size=128 → ~64 BRAM36)
- Consider reducing `hidden_size` to 64 for PYNQ-Z2 resource constraints

---

## Path B: Manual HLS LSTM Cell (Recommended for PYNQ-Z2)

Manually implement the LSTM cell in HLS C++ for full control over pipelining.

### LSTM Cell Equations
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   # forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   # input gate
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) # cell gate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   # output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

### HLS C++ Skeleton
```cpp
// lstm_cell.h
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<16, 6> data_t;

const int INPUT_SIZE  = 9;
const int HIDDEN_SIZE = 128;
const int SEQ_LEN     = 128;
const int NUM_CLASSES = 6;

void lstm_inference(
    hls::stream<data_t> &input_stream,   // (SEQ_LEN * INPUT_SIZE,)
    hls::stream<data_t> &output_stream,  // (NUM_CLASSES,)
    data_t W_f[HIDDEN_SIZE][INPUT_SIZE + HIDDEN_SIZE],
    data_t W_i[HIDDEN_SIZE][INPUT_SIZE + HIDDEN_SIZE],
    data_t W_g[HIDDEN_SIZE][INPUT_SIZE + HIDDEN_SIZE],
    data_t W_o[HIDDEN_SIZE][INPUT_SIZE + HIDDEN_SIZE],
    data_t b_f[HIDDEN_SIZE], data_t b_i[HIDDEN_SIZE],
    data_t b_g[HIDDEN_SIZE], data_t b_o[HIDDEN_SIZE],
    data_t W_fc[NUM_CLASSES][HIDDEN_SIZE], data_t b_fc[NUM_CLASSES]
);
```

### Key HLS Pragmas
```cpp
void lstm_cell(...) {
    #pragma HLS PIPELINE II=1        // one output per clock after initiation
    #pragma HLS ARRAY_PARTITION variable=h_prev complete  // fully partition hidden state
    #pragma HLS ARRAY_PARTITION variable=c_prev complete
    #pragma HLS RESOURCE variable=W_f core=RAM_2P_BRAM    // weights in BRAM
}
```

---

## Path C: CNN-LSTM Hybrid → CNN-only on FPGA

Since hls4ml has mature CNN support, consider a pragmatic approach:
1. **Train** the full CNN-LSTM model (arch='cnn_lstm' in train.py)
2. **On FPGA**: implement only the CNN feature extractor in HLS
3. **On ARM (PS side)**: run the LSTM in software using quantized model

This splits the workload between PL (CNN) and PS (LSTM), which is practical for PYNQ-Z2.

```
[PYNQ-Z2]
Input → PL (CNN HLS, ~10× speedup) → PS (quantized LSTM, ~3× vs float) → Output
```

---

## Resource Estimates (hidden_size=128, seq_len=128)

| Resource | Available | Full Unroll | Reuse=4 |
|---|---|---|---|
| LUTs | 53,200 | ~50,000 | ~20,000 |
| FFs | 106,400 | ~35,000 | ~15,000 |
| BRAM | 140 | ~80 | ~40 |
| DSPs | 220 | ~200 | ~60 |

**Recommendation**: Use `hidden_size=64` + `reuse_factor=4` to fit PYNQ-Z2 comfortably.

---

## Weight Export from PyTorch

```python
import torch
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
state = checkpoint['model_state_dict']

# LSTM weight matrices (PyTorch packs all gates together)
# weight_ih_l0: (4*hidden, input)  — i,f,g,o gates for input
# weight_hh_l0: (4*hidden, hidden) — i,f,g,o gates for hidden

W_ih = state['lstm.weight_ih_l0'].numpy()  # (4*128, 9)
W_hh = state['lstm.weight_hh_l0'].numpy()  # (4*128, 128)
b_ih = state['lstm.bias_ih_l0'].numpy()
b_hh = state['lstm.bias_hh_l0'].numpy()

# Split into gates: i=0:128, f=128:256, g=256:384, o=384:512
W_i, W_f, W_g, W_o = W_ih[:128], W_ih[128:256], W_ih[256:384], W_ih[384:]
```

---

## References

- [hls4ml RNN support](https://fastmachinelearning.org/hls4ml/advanced/rnn.html)
- [Vivado HLS LSTM implementation](https://github.com/Xilinx/finn-hlslib)
- [LSTM on FPGA survey](https://arxiv.org/abs/1901.02129)
- [PYNQ DMA tutorial](https://pynq.readthedocs.io/en/latest/pynq_libraries/dma.html)
