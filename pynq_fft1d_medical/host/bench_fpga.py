"""
bench_fpga.py
FPGA benchmark for the fft1d_emg HLS accelerator on PYNQ-Z2.

Usage (run on PYNQ-Z2 board):
    python3 bench_fpga.py

Requirements:
    - fft1d_emg.bit and fft1d_emg.hwh in the same directory
    - pynq package (pre-installed on PYNQ image v2.7+)
    - emg_signal.py in the same directory
"""

import time
import numpy as np

try:
    from pynq import Overlay, allocate
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("[bench_fpga] WARNING: pynq not available -- cannot run on this machine.")

from emg_signal import (
    generate_emg_synthetic,
    generate_emg_fatigue_sequence,
    compute_power_spectrum,
    compute_median_frequency,
    compute_mean_frequency,
    validate_fft_output,
    plot_emg_spectrum,
    plot_fatigue_trend,
    N_FFT, FS_HZ
)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
BITSTREAM_PATH  = "./fft1d_emg.bit"
N_WARMUP_RUNS   = 5
N_BENCH_RUNS    = 50
DIR_FORWARD     = 1


def run_once(fft_ip, x_re_buf, y_re_buf, y_im_buf):
    """Trigger one FFT execution and wait for completion."""
    fft_ip.register_map.CTRL.AP_START = 1
    while not fft_ip.register_map.CTRL.AP_IDLE:
        pass


def run_benchmark():
    if not PYNQ_AVAILABLE:
        print("ERROR: pynq package not installed. Run on PYNQ-Z2 board.")
        return

    # -------------------------------------------------------
    # 1. Load bitstream overlay
    # -------------------------------------------------------
    print(f"Loading overlay: {BITSTREAM_PATH}")
    ol = Overlay(BITSTREAM_PATH)

    # The IP name matches the HLS top function name.
    # PYNQ auto-generates register_map from the .hwh file.
    fft_ip = ol.fft1d_emg
    print(f"IP loaded: {fft_ip}")

    # -------------------------------------------------------
    # 2. Allocate physically contiguous CMA buffers
    # -------------------------------------------------------
    x_re_buf = allocate(shape=(N_FFT,), dtype=np.float32)
    y_re_buf = allocate(shape=(N_FFT,), dtype=np.float32)
    y_im_buf = allocate(shape=(N_FFT,), dtype=np.float32)

    # -------------------------------------------------------
    # 3. Configure kernel registers (once -- addresses are fixed)
    # M_AXI port addresses: PYNQ maps HLS port names as <port>_1 (lower 32-bit)
    # -------------------------------------------------------
    fft_ip.register_map.x_re_1 = x_re_buf.physical_address
    fft_ip.register_map.y_re_1 = y_re_buf.physical_address
    fft_ip.register_map.y_im_1 = y_im_buf.physical_address
    fft_ip.register_map.dir    = DIR_FORWARD
    fft_ip.register_map.n      = N_FFT

    # -------------------------------------------------------
    # 4. Generate synthetic EMG test signal and load input buffer
    # -------------------------------------------------------
    x_input = generate_emg_synthetic(n_samples=N_FFT, fs=FS_HZ)
    x_re_buf[:] = x_input

    # -------------------------------------------------------
    # 5. Warm-up runs
    # -------------------------------------------------------
    print(f"Warming up ({N_WARMUP_RUNS} runs)...")
    for _ in range(N_WARMUP_RUNS):
        run_once(fft_ip, x_re_buf, y_re_buf, y_im_buf)

    # -------------------------------------------------------
    # 6. Correctness validation
    # -------------------------------------------------------
    print("Validating FFT output...")
    run_once(fft_ip, x_re_buf, y_re_buf, y_im_buf)

    y_re_np = np.array(y_re_buf, dtype=np.float32)
    y_im_np = np.array(y_im_buf, dtype=np.float32)

    passed, max_err = validate_fft_output(x_input, y_re_np, y_im_np)
    status = "PASS" if passed else "FAIL"
    print(f"  Correctness: {status}, max relative error = {max_err:.2e}")
    if not passed:
        print("  WARNING: Output does not match numpy reference within tolerance!")

    # -------------------------------------------------------
    # 7. Latency benchmark
    # -------------------------------------------------------
    print(f"Running latency benchmark ({N_BENCH_RUNS} runs)...")
    latencies_us = []

    for _ in range(N_BENCH_RUNS):
        t0 = time.perf_counter()
        run_once(fft_ip, x_re_buf, y_re_buf, y_im_buf)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    latencies_us = np.array(latencies_us)

    print("\n--- FPGA FFT Benchmark Results ---")
    print(f"  FFT size          : N={N_FFT}")
    print(f"  Sampling rate     : {FS_HZ:.0f} Hz  (1 frame = {N_FFT/FS_HZ*1000:.1f} ms)")
    print(f"  Iterations        : {N_BENCH_RUNS}")
    print(f"  Latency mean      : {latencies_us.mean():.1f} us")
    print(f"  Latency std       : {latencies_us.std():.1f} us")
    print(f"  Latency min/max   : {latencies_us.min():.1f} / {latencies_us.max():.1f} us")
    print(f"  Latency median    : {np.median(latencies_us):.1f} us")
    print(f"  Throughput        : {1e6/latencies_us.mean():.0f} FFT frames/s")

    # -------------------------------------------------------
    # 8. EMG spectral analysis demo using FPGA output
    # -------------------------------------------------------
    print("\n--- EMG Spectral Analysis (FPGA output) ---")
    freqs, psd = compute_power_spectrum(y_re_np, y_im_np, fs=FS_HZ, n_fft=N_FFT)
    mdf = compute_median_frequency(freqs, psd)
    mnf = compute_mean_frequency(freqs, psd)
    print(f"  Median Frequency (MDF): {mdf:.1f} Hz")
    print(f"  Mean Frequency   (MNF): {mnf:.1f} Hz")
    plot_emg_spectrum(freqs, psd, mdf, mnf, title="EMG Spectrum (FPGA FFT)")

    # -------------------------------------------------------
    # 9. Muscle fatigue simulation across multiple windows
    # -------------------------------------------------------
    print("\n--- Muscle Fatigue Simulation ---")
    fatigue_windows = generate_emg_fatigue_sequence(n_windows=20, n_samples=N_FFT, fs=FS_HZ)
    mdf_trend = []
    mnf_trend = []

    for win in fatigue_windows:
        x_re_buf[:] = win
        run_once(fft_ip, x_re_buf, y_re_buf, y_im_buf)
        yr = np.array(y_re_buf, dtype=np.float32)
        yi = np.array(y_im_buf, dtype=np.float32)
        f, p = compute_power_spectrum(yr, yi, fs=FS_HZ, n_fft=N_FFT)
        mdf_trend.append(compute_median_frequency(f, p))
        mnf_trend.append(compute_mean_frequency(f, p))

    print(f"  MDF range: {min(mdf_trend):.1f} - {max(mdf_trend):.1f} Hz")
    print(f"  MNF range: {min(mnf_trend):.1f} - {max(mnf_trend):.1f} Hz")
    plot_fatigue_trend(mdf_trend, mnf_trend)

    # -------------------------------------------------------
    # 10. Cleanup
    # -------------------------------------------------------
    x_re_buf.freebuffer()
    y_re_buf.freebuffer()
    y_im_buf.freebuffer()


if __name__ == "__main__":
    run_benchmark()
