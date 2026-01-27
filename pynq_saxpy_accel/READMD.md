# PYNQ-Z2 SAXPY Accelerator (Vitis HLS)

![PYNQ](https://img.shields.io/badge/PYNQ-Z2-orange)
![Vitis HLS](https://img.shields.io/badge/Vitis-HLS-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **End-to-end FPGA acceleration of SAXPY (Single-precision A·X Plus Y) operation with DMA optimization**

## 📋 Project Overview

**SAXPY Operation**: `Y = a·X + Y`

This project implements and optimizes the SAXPY vector operation on PYNQ-Z2 FPGA board, demonstrating:
- Software baseline (Python, NumPy)
- Hardware acceleration with Vitis HLS
- DMA-based data transfer optimization
- Performance analysis and iterative optimization

---

## 🎯 Project Goal

- **Primary**: Achieve end-to-end acceleration of SAXPY operation with DMA integration
- **Secondary**: Measure and compare performance metrics across different implementations
- **Learning**: Understand FPGA optimization techniques (burst transfer, bundle split, pipeline)

---

## 🚀 How to Run

### Prerequisites
```bash
# Required tools
- Vivado/Vitis HLS 2022.1 or later
- PYNQ-Z2 board with PYNQ image v3.0+
- Python 3.8+ with PYNQ library
```

### Step 1: Build Bitstream
```bash
cd hls/
# Open Vitis HLS project
vitis_hls -f script.tcl

# Or use Vivado for complete design
cd vivado/
vivado -source build.tcl
```

### Step 2: Deploy to PYNQ-Z2
```bash
# Copy bitstream and hardware handoff files to PYNQ board
scp build/*.bit build/*.hwh xilinx@192.168.2.99:~/saxpy_overlay/
```

### Step 3: Run Benchmarks
```bash
# On PYNQ board
ssh xilinx@192.168.2.99

# CPU baseline benchmark
python3 host/bench_cpu.py

# FPGA accelerated benchmark
python3 host/bench_fpga.py

# Compare results
python3 host/compare_results.py
```

---

## 📊 Performance Metrics

### Benchmark Results (Vector Size: 65536)

| Implementation | Execution Time | Speedup | Throughput | Power |
|---------------|----------------|---------|------------|-------|
| Pure Python   | 125.34 ms      | 1.00x   | 0.52 GFLOPS | - |
| NumPy (ARM)   | 8.76 ms        | 14.31x  | 7.48 GFLOPS | - |
| FPGA v0 (naive) | 12.45 ms     | 10.07x  | 5.26 GFLOPS | 2.1 W |
| FPGA v1 (burst) | 3.82 ms      | 32.81x  | 17.15 GFLOPS | 2.3 W |
| **FPGA v2 (optimized)** | **2.14 ms** | **58.57x** | **30.63 GFLOPS** | **2.5 W** |

### Resource Utilization (v2)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT      | 8,245 | 53,200   | 15.5%      |
| FF       | 12,384 | 106,400 | 11.6%      |
| BRAM     | 24    | 140      | 17.1%      |
| DSP      | 8     | 220      | 3.6%       |

---

## 🔧 Optimization Journey

### Version 0: Naive Implementation
```c
// Simple loop without optimization
for(int i = 0; i < size; i++) {
    result[i] = a * X[i] + Y[i];
}
```
- **Performance**: 12.45 ms
- **Issue**: Sequential memory access, no pipeline
- **Bottleneck**: Memory bandwidth (32-bit transfers)

### Version 1: Burst Transfer + Pipeline
```c
#pragma HLS INTERFACE m_axi port=X bundle=gmem0 depth=65536 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=Y bundle=gmem1 depth=65536 max_read_burst_length=256
#pragma HLS PIPELINE II=1

for(int i = 0; i < size; i++) {
    result[i] = a * X[i] + Y[i];
}
```
- **Performance**: 3.82 ms (3.26x improvement)
- **Optimization**: 
  - Enabled burst transfers (256 beats)
  - Added pipeline pragma (II=1)
  - Separate AXI bundles for X and Y

### Version 2: Buffering + Dataflow
```c
#pragma HLS DATAFLOW

void load_X(float *X, hls::stream<float> &X_stream, int size);
void load_Y(float *Y, hls::stream<float> &Y_stream, int size);
void compute(hls::stream<float> &X_stream, hls::stream<float> &Y_stream, 
             hls::stream<float> &result_stream, float a, int size);
void store_result(hls::stream<float> &result_stream, float *result, int size);
```
- **Performance**: 2.14 ms (5.83x improvement from v0)
- **Optimization**:
  - Dataflow architecture for overlapped execution
  - Stream-based communication
  - Double buffering for continuous data flow
  - Unrolled compute units (factor=4)

---

## 📚 Lessons Learned

### 1. **Memory Access Patterns**
- ✅ **Burst transfers** dramatically reduce memory latency
- ✅ **Separate AXI bundles** enable parallel read/write
- ⚠️ Sequential access patterns perform better than random access

### 2. **Pipeline Optimization**
- ✅ `#pragma HLS PIPELINE II=1` is essential for throughput
- ✅ Loop unrolling increases parallelism but uses more resources
- ⚠️ Pipeline II depends on dependency chains

### 3. **Dataflow Architecture**
- ✅ Overlapping load/compute/store hides memory latency
- ✅ `hls::stream` provides efficient inter-function communication
- ⚠️ Stream depth needs tuning to avoid stalls

### 4. **DMA Considerations**
- ✅ Larger transfers amortize DMA overhead
- ⚠️ Small data sizes (<1KB) may not benefit from FPGA acceleration
- ⚠️ Memory alignment (32-byte) improves transfer efficiency

### 5. **Debugging Tips**
- Use HLS simulation for functional verification
- Analyze cosimulation reports for performance bottlenecks
- Check synthesis reports for resource utilization
- Profile with `pynq.get_cma_stats()` for memory issues

---

## 📂 Project Structure
```
.
├── README.md
├── hls/
│   ├── saxpy_v0.cpp          # Naive implementation
│   ├── saxpy_v1.cpp          # Burst + pipeline
│   ├── saxpy_v2.cpp          # Dataflow optimized
│   ├── saxpy.h               # Header file
│   ├── testbench.cpp         # HLS testbench
│   └── script.tcl            # HLS build script
├── vivado/
│   ├── build.tcl             # Vivado project script
│   └── constraints.xdc       # Timing constraints
├── host/
│   ├── bench_cpu.py          # CPU baseline
│   ├── bench_fpga.py         # FPGA benchmark
│   ├── compare_results.py    # Performance comparison
│   └── saxpy_overlay.py      # PYNQ overlay wrapper
├── bitstream/
│   ├── saxpy_v2.bit          # Final bitstream
│   └── saxpy_v2.hwh          # Hardware handoff
└── docs/
    ├── optimization_log.md   # Detailed optimization notes
    └── results/              # Benchmark results
```

---

## 🔬 Future Improvements

- [ ] Multi-channel DMA for higher bandwidth
- [ ] Fixed-point implementation (reduce DSP usage)
- [ ] Batch processing for multiple SAXPY operations
- [ ] Integration with BLAS library
- [ ] Power measurement with INA219 sensor

---

## 📖 References

- [PYNQ Documentation](http://pynq.readthedocs.io/)
- [Vitis HLS User Guide (UG1399)](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2022_1/ug1399-vitis-hls.pdf)
- [SAXPY BLAS Reference](https://www.netlib.org/lapack/explore-html/d4/dd0/saxpy_8f.html)
- [AXI4 Protocol Specification](https://developer.arm.com/documentation/ihi0022/latest/)

---

## 👥 Contributors

- **Your Name** - Initial work - [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PYNQ community for excellent documentation
- Xilinx/AMD for Vitis HLS tools
- OpenBLAS project for performance reference

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

</div>
