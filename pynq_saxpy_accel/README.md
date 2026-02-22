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

보드에서 실행

ssh xilinx@<BOARD_IP>
cd pynq_saxpy_accel/host
python3 bench_cpu.py
python3 bench_fpga.py
python3 compare_results.py
Build (reproducible)

HLS → IP

cd hw/hls
vitis_hls -f script.tcl

Vivado → bitstream

cd ../vivado
vivado -mode batch -source build.tcl

Output

bitstreams/saxpy.bit

bitstreams/saxpy.hwh

Repro info (fill this)

Board: PYNQ-Z2

PYNQ image: vX.Y

Vivado: 2022.1

Vitis HLS: 2022.1

PL clock: XXX MHz

DMA: AXI DMA (MM2S/S2MM), burst: XXX, data width: XXX

Host: Python X.Y, PYNQ version: X.Y


### 2) 링크가 `main`으로 보이게 하려면(중요)
- 만약 기본 브랜치가 **master**면: 지금 네 링크(`main`)는 나중에 깨질 수 있어.
- 해결은 둘 중 하나:
  1) **GitHub Repo Settings → Branches → Default branch를 `main`으로 변경**(추천)
  2) 아니면 문서/링크는 **`/blob/master/`**로 통일

(현재 상태는 raw 기준으로 `master` 쪽이 실제로 열림) :contentReference[oaicite:2]{index=2}

---

원하면 내가 **루트 `README.md`도 지금 레포 상태에 맞춰서(main/master 포함)** “링크 안 깨지게” 버전으로 딱 맞춰서 한 번에 복붙용으로 만들어줄게.
::contentReference[oaicite:3]{index=3}
