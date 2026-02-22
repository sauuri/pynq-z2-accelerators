# pynq_saxpy_accel

SAXPY (`Y = aX + Y`)를 PYNQ-Z2에서 **DMA 기반 end-to-end**로 가속합니다.  
Vitis HLS로 커널 생성 → Vivado로 BD/bitstream → PYNQ(Python)에서 실행/벤치마크.

## What it does
- Data type: `float32`
- Vector length: configurable (default: 65536)
- Modes
  - CPU baseline: NumPy
  - FPGA: DMA transfer + HLS kernel

## Requirements
- Board: PYNQ-Z2
- PYNQ image: v3.0+ (권장)
- Tools: Vivado/Vitis HLS 2022.1+
- Python: 3.8+ (PYNQ 포함)

## Quickstart
1) 보드에 bit/hwh 복사
```bash
scp bitstreams/*.bit bitstreams/*.hwh xilinx@<BOARD_IP>:~/saxpy_overlay/
