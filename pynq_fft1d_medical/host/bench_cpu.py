"""
bench_cpu.py
CPU baseline benchmark for 1D FFT on EMG signals using numpy.

Provides the reference performance to compare against the FPGA accelerator.

Usage:
    python3 bench_cpu.py
"""

import time
import numpy as np

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
N_WARMUP_RUNS = 10
N_BENCH_RUNS  = 200


def run_once(x_input: np.ndarray):
    """Run one numpy FFT and return (y_re, y_im)."""
    y = np.fft.fft(x_input)
    return y.real.astype(np.float32), y.imag.astype(np.float32)


def run_benchmark():
    # -------------------------------------------------------
    # 1. Generate synthetic EMG signal
    # -------------------------------------------------------
    x_input = generate_emg_synthetic(n_samples=N_FFT, fs=FS_HZ)

    # -------------------------------------------------------
    # 2. Warm-up
    # -------------------------------------------------------
    print(f"Warming up ({N_WARMUP_RUNS} runs)...")
    for _ in range(N_WARMUP_RUNS):
        run_once(x_input)

    # -------------------------------------------------------
    # 3. Latency benchmark
    # -------------------------------------------------------
    print(f"Running latency benchmark ({N_BENCH_RUNS} runs)...")
    latencies_us = []

    for _ in range(N_BENCH_RUNS):
        t0 = time.perf_counter()
        y_re, y_im = run_once(x_input)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    latencies_us = np.array(latencies_us)

    print("\n--- CPU FFT Benchmark Results (numpy) ---")
    print(f"  FFT size          : N={N_FFT}")
    print(f"  Sampling rate     : {FS_HZ:.0f} Hz  (1 frame = {N_FFT/FS_HZ*1000:.1f} ms)")
    print(f"  Iterations        : {N_BENCH_RUNS}")
    print(f"  Latency mean      : {latencies_us.mean():.1f} us")
    print(f"  Latency std       : {latencies_us.std():.1f} us")
    print(f"  Latency min/max   : {latencies_us.min():.1f} / {latencies_us.max():.1f} us")
    print(f"  Latency median    : {np.median(latencies_us):.1f} us")
    print(f"  Throughput        : {1e6/latencies_us.mean():.0f} FFT frames/s")

    # -------------------------------------------------------
    # 4. Correctness self-check
    # -------------------------------------------------------
    passed, max_err = validate_fft_output(x_input, y_re, y_im, rel_tol=1e-5)
    print(f"\n  Self-check vs reference: {'PASS' if passed else 'FAIL'} (max err={max_err:.2e})")

    # -------------------------------------------------------
    # 5. EMG spectral analysis
    # -------------------------------------------------------
    print("\n--- EMG Spectral Analysis (CPU) ---")
    freqs, psd = compute_power_spectrum(y_re, y_im, fs=FS_HZ, n_fft=N_FFT)
    mdf = compute_median_frequency(freqs, psd)
    mnf = compute_mean_frequency(freqs, psd)
    print(f"  Median Frequency (MDF): {mdf:.1f} Hz")
    print(f"  Mean Frequency   (MNF): {mnf:.1f} Hz")
    plot_emg_spectrum(freqs, psd, mdf, mnf, title="EMG Spectrum (CPU / numpy FFT)")

    # -------------------------------------------------------
    # 6. Muscle fatigue simulation
    # -------------------------------------------------------
    print("\n--- Muscle Fatigue Simulation ---")
    fatigue_windows = generate_emg_fatigue_sequence(n_windows=20, n_samples=N_FFT, fs=FS_HZ)
    mdf_trend = []
    mnf_trend = []

    for win in fatigue_windows:
        yr, yi = run_once(win)
        f, p = compute_power_spectrum(yr, yi, fs=FS_HZ, n_fft=N_FFT)
        mdf_trend.append(compute_median_frequency(f, p))
        mnf_trend.append(compute_mean_frequency(f, p))

    print(f"  MDF range: {min(mdf_trend):.1f} - {max(mdf_trend):.1f} Hz")
    print(f"  MNF range: {min(mnf_trend):.1f} - {max(mnf_trend):.1f} Hz")
    plot_fatigue_trend(mdf_trend, mnf_trend)


if __name__ == "__main__":
    run_benchmark()
