# pynq_fft1d_medical

PYNQ-Z2에서 동작하는 **1D FFT FPGA 가속기** — EMG(근전도) 신호 분석용
1D FFT accelerator for **EMG signal analysis** on PYNQ-Z2.

---

## Overview

| 항목 | 내용 |
|------|------|
| 알고리즘 | 1024-point 1D FFT (forward / inverse) |
| FFT IP | Xilinx HLS FFT IP (`hls_fft.h`, `hls::fft<>`) |
| 데이터 타입 | `float` (IEEE 754 single precision) |
| 인터페이스 | M_AXI × 3 + AXI-Lite control |
| 의료 도메인 | EMG (근전도): 근육 피로도 분석 (MDF / MNF 추적) |
| 샘플링 속도 | 4 kHz, 1024-point window = 256 ms |

### 의료 응용

EMG(Electromyography) 주파수 분석은 **근육 피로도**의 비침습적 지표로 사용됩니다.

- **Median Frequency (MDF)**: 총 파워를 반씩 나누는 주파수 → 피로 시 감소
- **Mean Frequency (MNF)**: 파워 가중 평균 주파수 → 피로 시 감소
- 정상 EMG 주파수 대역: 10 – 500 Hz (표면 EMG 기준)

FPGA 가속으로 실시간 다채널 EMG 분석이 가능합니다.

---

## Repository Layout

```
pynq_fft1d_medical/
├── hls/
│   ├── fft1d_emg.h        # HLS FFT config struct + top-function declaration
│   ├── fft1d_emg.cpp      # HLS kernel: M_AXI bridge + hls::fft<> + DATAFLOW
│   └── fft1d_emg_tb.cpp   # C simulation testbench (peak frequency verification)
├── host/
│   ├── emg_signal.py      # EMG signal generation, PSD, MDF/MNF utilities
│   ├── bench_fpga.py      # PYNQ benchmark (requires .bit + .hwh)
│   └── bench_cpu.py       # numpy FFT CPU baseline
├── bitstreams/            # Place synthesised .bit and .hwh files here
└── README.md
```

---

## Hardware Architecture

### AXI Interface

```
                ┌─────────────────────────────────────────┐
   PS DDR ─────►│ M_AXI gmem0  x_re[1024]                 │
                │                                         │
   PS DDR ─────►│ M_AXI gmem1  y_re[1024]  (write-only)  │
                │         fft1d_emg                       │
   PS DDR ─────►│ M_AXI gmem2  y_im[1024]  (write-only)  │
                │                                         │
   PS AXI-Lite ►│ s_axilite control (addresses + scalars) │
                └─────────────────────────────────────────┘
```

**Vivado 연결 권장사항 (PYNQ-Z2):**
- `gmem0` → AXI SmartConnect → HP0 (PS Slave HP0)
- `gmem1`, `gmem2` → AXI SmartConnect → HP1 (PS Slave HP1)

HP0/HP1은 Zynq PS의 High-Performance AXI Slave 포트입니다.
PYNQ-Z2는 HP0, HP1 두 포트를 제공합니다.

### Processing Pipeline (DATAFLOW)

```
DDR (x_re)
    │
    ▼
read_input()        ← M_AXI burst read (burst length=256, 16 transactions)
    │ xn[1024]
    ▼
hls::fft<params>()  ← pipelined_streaming_io, natural_order, N=1024
    │ xk[1024]
    ▼
write_output()      ← M_AXI burst write (burst length=256)
    │
    ▼
DDR (y_re, y_im)
```

`#pragma HLS DATAFLOW` 로 세 단계가 오버랩되어 실행됩니다.

---

## Build & Synthesis

### 1. C Simulation (Vitis HLS)

```tcl
# Vitis HLS Tcl script
open_project fft1d_emg_proj
set_top fft1d_emg
add_files hls/fft1d_emg.h
add_files hls/fft1d_emg.cpp
add_files -tb hls/fft1d_emg_tb.cpp
open_solution solution1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
```

테스트벤치는 50 Hz / 120 Hz / 250 Hz 합성 EMG 신호를 입력하여
FFT 출력의 피크 주파수와 수치 오차를 검증합니다.

### 2. HLS Synthesis & Export IP

```tcl
csynth_design
export_design -format ip_catalog
```

합성 후 생성된 IP를 Vivado Block Design에 추가합니다.

### 3. Vivado Block Design

1. Zynq PS IP 추가 → Run Block Automation
2. `fft1d_emg` IP 추가
3. AXI SmartConnect 추가:
   - `gmem0` → HP0
   - `gmem1`, `gmem2` → HP1
4. AXI-Lite control → GP0
5. Run Connection Automation → Validate → Generate Bitstream
6. `fft1d_emg.bit`, `fft1d_emg.hwh` → `bitstreams/` 폴더로 복사

---

## Running on PYNQ-Z2

### Setup

```bash
# PYNQ-Z2 보드에서 실행
scp bitstreams/fft1d_emg.bit bitstreams/fft1d_emg.hwh  pynq@192.168.2.99:~/
scp host/emg_signal.py host/bench_fpga.py host/bench_cpu.py  pynq@192.168.2.99:~/
```

### CPU Baseline (any machine with numpy)

```bash
python3 bench_cpu.py
```

### FPGA Benchmark (PYNQ-Z2 board only)

```bash
# bitstream과 같은 디렉터리에서 실행
python3 bench_fpga.py
```

출력 예시:
```
Loading overlay: ./fft1d_emg.bit
Validating FFT output...
  Correctness: PASS, max relative error = 3.12e-06
--- FPGA FFT Benchmark Results ---
  FFT size          : N=1024
  Sampling rate     : 4000 Hz  (1 frame = 256.0 ms)
  Latency mean      : 42.3 us
  Throughput        : 23641 FFT frames/s
--- EMG Spectral Analysis (FPGA output) ---
  Median Frequency (MDF): 118.4 Hz
  Mean Frequency   (MNF): 132.7 Hz
```

---

## Reproducibility

| 항목 | 버전 |
|------|------|
| PYNQ image | v2.7 (Ubuntu 20.04) |
| Vivado | 2022.2 |
| Vitis HLS | 2022.2 |
| PYNQ-Z2 board | Rev. C |
| PL clock | 100 MHz (10 ns period) |
| AXI bus width | 32-bit data, 32-bit address |
| DMA burst length | 256 bytes |
| Input data type | float32 (real-valued EMG) |
| Output data type | float32 (complex: y_re + y_im) |

---

## License

MIT
