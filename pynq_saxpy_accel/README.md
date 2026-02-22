# pynq_saxpy_accel

SAXPY (`Y = aX + Y`)를 PYNQ-Z2에서 DMA 기반 end-to-end로 가속합니다.  
Vitis HLS로 커널 생성 → Vivado bitstream → PYNQ(Python)에서 실행/벤치마크.

## What it does
- Data type: `float32`
- Vector length: configurable (default: 65536)
- Compare: CPU(NumPy) vs FPGA(DMA+HLS)

## Requirements
- Board: PYNQ-Z2
- PYNQ image: v3.0+
- Vivado/Vitis HLS: 2022.1+
- Python: 3.8+ (PYNQ)

## Quickstart

1) bit/hwh를 보드에 복사
```bash
scp bitstreams/*.bit bitstreams/*.hwh xilinx@<BOARD_IP>:~/saxpy_overlay/
```

2) 보드에서 실행
```bash
ssh xilinx@<BOARD_IP>
cd ~/pynq_saxpy_accel/host
python3 bench_cpu.py
python3 bench_fpga.py
python3 compare_results.py
```

> 참고: `bench_fpga.py`에서 overlay(.bit/.hwh) 경로를 어디로 잡는지에 따라  
> `~/saxpy_overlay/` 또는 repo 내부 `bitstreams/` 등 경로를 맞춰 주세요.

## Build (reproducible)

### HLS → IP
```bash
cd hw/hls
vitis_hls -f script.tcl
```

### Vivado → bitstream
```bash
cd ../vivado
vivado -mode batch -source build.tcl
```

### Output
- `bitstreams/saxpy.bit`
- `bitstreams/saxpy.hwh`

## Repro info (fill this)
- Board: PYNQ-Z2
- PYNQ image: vX.Y
- Vivado: 2022.1
- Vitis HLS: 2022.1
- PL clock: XXX MHz
- DMA: AXI DMA (MM2S/S2MM), burst: XXX, data width: XXX
- Host: Python X.Y, PYNQ version: X.Y

## Notes: 링크가 `main`으로 보이게 하려면(중요)
- 만약 기본 브랜치가 **master**면: 문서/README에서 `/blob/main/`로 링크를 걸면 나중에 깨질 수 있어요.
- 해결은 둘 중 하나:
  1) **GitHub Repo Settings → Branches → Default branch를 `main`으로 변경** (추천)
  2) 문서/링크를 전부 **`/blob/master/`**로 통일
