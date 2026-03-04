# HLS Guide — Vision Transformer on FPGA

Deploying Transformers on FPGA is an active research area.
The key challenge is the **Softmax attention** operation, which requires
sequential normalization across all query-key products.

---

## Challenge: Why Attention is Hard on FPGA

Standard attention: `Attn = softmax(QKᵀ / √d) · V`
- Softmax requires global max/sum → breaks pipeline
- QKᵀ matrix: N²·d multiplications (N=64 patches, d=16 per head)
- For PYNQ-Z2: 64×64×16 = 65,536 MACs per head × 4 heads = 262,144 MACs/layer

---

## Strategy A: Hybrid PL+PS (Recommended for PYNQ-Z2)

Split the model between programmable logic (PL) and ARM processor (PS):

```
Input Image
    ↓ PL (HLS)
Patch Embedding (Conv2d 4×4) — easy to accelerate, ~10× speedup
    ↓ PS (ARM)
Transformer blocks (attention + MLP) — quantized INT8 on CPU
    ↓ PS (ARM)
Classification head
    ↓
Output
```

Implementation:
```python
# On PYNQ board
from pynq import Overlay
overlay = Overlay('patch_embed.bit')

# PL: patch embedding
patch_features = overlay.patch_embed_accel(image)

# PS: transformer inference
with torch.no_grad():
    tokens = patch_features + pos_embed
    for block in transformer_blocks:
        tokens = block(tokens)
    logits = head(tokens[:, 0])
```

---

## Strategy B: Linear Attention (HLS-Friendly)

Replace softmax attention with linear attention — fully parallelizable:

```python
# Standard (hard for FPGA):
attn = softmax(Q @ K.T / sqrt(d)) @ V

# Linear attention (FPGA-friendly):
# φ(Q)(φ(K)ᵀV) — associativity allows O(N·d²) instead of O(N²·d)
def linear_attention(Q, K, V):
    K = F.elu(K) + 1        # positive kernel
    Q = F.elu(Q) + 1
    KV = K.transpose(-2,-1) @ V   # (d, d) — compute once
    return Q @ KV                  # (N, d)
```

Modify `MultiHeadSelfAttention.forward()` to use linear attention,
then retrain — typically < 1% accuracy loss.

---

## Strategy C: hls4ml (MLP blocks only)

hls4ml currently supports the **MLP (FFN) blocks** of Transformers well.
Export only the feed-forward portion:

```python
import hls4ml, onnx

# Export MLP-only subgraph
mlp_model = extract_mlp_subgraph(tinyvit)  # custom extraction
torch.onnx.export(mlp_model, dummy, 'mlp_only.onnx', opset_version=13)

model_onnx = onnx.load('mlp_only.onnx')
config = hls4ml.utils.config_from_onnx_model(
    model_onnx,
    default_precision='ap_fixed<8,3>',
    default_reuse_factor=2,
)
hls_model = hls4ml.converters.convert_from_onnx_model(
    model_onnx, hls_config=config,
    output_dir='hls/hls4ml_mlp', part='xc7z020clg400-1')
hls_model.compile()
```

---

## Resource Estimate (patch embed + 1 MLP block, 64-dim)

| Resource | Available | Estimate |
|---|---|---|
| LUTs | 53,200 | ~15,000 |
| BRAM | 140 | ~20 |
| DSPs | 220 | ~50 |

Full 6-layer Transformer: exceeds PYNQ-Z2 resources.
**Use Strategy A (hybrid) for practical deployment.**

---

## References

- [FPT: Efficient Transformer on FPGA](https://arxiv.org/abs/2211.08800)
- [ViA: ViT Acceleration on FPGA](https://arxiv.org/abs/2209.05577)
- [hls4ml Transformer](https://github.com/fastmachinelearning/hls4ml/issues/642)
- [Linear Attention](https://arxiv.org/abs/2006.16236)
