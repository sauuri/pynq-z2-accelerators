# pynq_saxpy_accel

SAXPY (`Y = aX + Y`)를 **PYNQ-Z2**에서 **AXI DMA 기반 end-to-end**로 가속합니다.  
Vitis HLS로 커널 생성 → Vivado로 bitstream 생성 → PYNQ(Python)에서 실행/벤치마크.

## What it does
- Operation: `Y = aX + Y`
- Data type: `float32`
- Vector length: configurable (default: `65536`)
- Compare: **CPU (NumPy)** vs **FPGA (DMA + HLS)**

## Requirements
- Board: **PYNQ-Z2**
- PYNQ image: **v3.0+**
- Vivado / Vitis HLS: **2022.1+**
- Python: **3.8+** (on PYNQ)

## Repo layout (example)
- `host/` : Python 실행/벤치마크 스크립트
- `hw/hls/` : HLS 커널 + `script.tcl`
- `hw/vivado/` : Vivado 프로젝트/`build.tcl`
- `bitstreams/` : 결과물 `.bit/.hwh`

## Quickstart (run on board)

### 1) Copy overlay to board
```bash
# on your PC
scp bitstreams/*.bit bitstreams/*.hwh xilinx@<BOARD_IP>:~/saxpy_overlay/
