"""
emg_signal.py
EMG signal generation and spectral analysis utilities.

Used by bench_fpga.py and bench_cpu.py to:
  - Generate synthetic EMG signals with known frequency content
  - Compute Power Spectral Density from FFT output
  - Calculate Median Frequency (MDF) and Mean Frequency (MNF)
    for muscle fatigue analysis
  - Validate FPGA FFT output against numpy reference
"""

import numpy as np
from typing import Tuple

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
N_FFT   = 1024      # Must match FFT_N in fft1d_emg.h
FS_HZ   = 4000.0    # EMG sampling rate (Hz)
F_LOW   = 10.0      # Physiological EMG band: lower bound (Hz)
F_HIGH  = 500.0     # Physiological EMG band: upper bound (Hz)

# Frequency bins for N_FFT at FS_HZ
FREQS = np.fft.fftfreq(N_FFT, d=1.0 / FS_HZ).astype(np.float32)


# -------------------------------------------------------
# Signal generation
# -------------------------------------------------------

def generate_emg_synthetic(
    n_samples: int = N_FFT,
    fs: float = FS_HZ,
    components: list = None,
    noise_std: float = 0.05,
    seed: int = 42
) -> np.ndarray:
    """
    Generate a synthetic EMG signal as a sum of sinusoids plus Gaussian noise.

    Default components mimic surface EMG during moderate muscle contraction:
      - 50 Hz  (amplitude 0.6): slow motor unit recruitment
      - 120 Hz (amplitude 0.4): intermediate motor units
      - 250 Hz (amplitude 0.3): fast motor units

    Returns:
        x : np.ndarray, shape (n_samples,), dtype float32
    """
    if components is None:
        components = [
            (50.0,  0.6),
            (120.0, 0.4),
            (250.0, 0.3),
        ]

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / fs
    x = np.zeros(n_samples, dtype=np.float64)

    for freq_hz, amp in components:
        x += amp * np.sin(2.0 * np.pi * freq_hz * t)

    x += noise_std * rng.standard_normal(n_samples)
    return x.astype(np.float32)


def generate_emg_fatigue_sequence(
    n_windows: int = 20,
    n_samples: int = N_FFT,
    fs: float = FS_HZ,
    seed: int = 0
) -> list:
    """
    Generate a sequence of EMG windows simulating progressive muscle fatigue.

    Fatigue is modelled by:
      - Decreasing dominant frequency (spectral compression towards lower freqs)
      - Increasing amplitude (motor unit synchronisation)

    Returns:
        List of np.ndarray, each of shape (n_samples,) float32
    """
    windows = []
    for i in range(n_windows):
        fatigue_factor = i / max(n_windows - 1, 1)  # 0.0 (fresh) -> 1.0 (fatigued)

        # Dominant frequency shifts downward (200 Hz -> 80 Hz)
        f_dom = 200.0 - fatigue_factor * 120.0
        amp   = 0.5 + fatigue_factor * 0.5

        components = [
            (f_dom * 0.4,  amp * 0.6),
            (f_dom,        amp * 1.0),
            (f_dom * 1.5,  amp * 0.3),
        ]
        windows.append(generate_emg_synthetic(n_samples, fs, components,
                                               noise_std=0.05, seed=seed + i))
    return windows


# -------------------------------------------------------
# Spectral analysis
# -------------------------------------------------------

def compute_power_spectrum(
    y_re: np.ndarray,
    y_im: np.ndarray,
    fs: float = FS_HZ,
    n_fft: int = N_FFT
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided Power Spectral Density from complex FFT output.

    Args:
        y_re : real part of FFT output (length n_fft), float32
        y_im : imaginary part of FFT output (length n_fft), float32

    Returns:
        freqs : frequency axis (Hz), shape (n_fft//2 + 1,)
        psd   : power spectral density, shape (n_fft//2 + 1,)
    """
    # Magnitude squared
    mag_sq = y_re.astype(np.float64) ** 2 + y_im.astype(np.float64) ** 2

    # One-sided spectrum (DC + positive frequencies only)
    n_one_sided = n_fft // 2 + 1
    psd = mag_sq[:n_one_sided] / (n_fft ** 2)

    # Double non-DC, non-Nyquist bins to account for negative frequencies
    psd[1:-1] *= 2.0

    freqs = np.arange(n_one_sided, dtype=np.float64) * fs / n_fft
    return freqs.astype(np.float32), psd.astype(np.float32)


def compute_median_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_low: float = F_LOW,
    f_high: float = F_HIGH
) -> float:
    """
    Compute Median Frequency (MDF): frequency that splits total power equally.

    MDF is a primary indicator of muscle fatigue -- it decreases as the muscle
    fatigues due to motor unit synchronisation and metabolic changes.

    Returns:
        mdf_hz : float, median frequency in Hz
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    f_band = freqs[mask]
    p_band = psd[mask]

    if p_band.sum() == 0:
        return 0.0

    cumulative = np.cumsum(p_band)
    total = cumulative[-1]
    idx = np.searchsorted(cumulative, total / 2.0)
    idx = min(idx, len(f_band) - 1)
    return float(f_band[idx])


def compute_mean_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_low: float = F_LOW,
    f_high: float = F_HIGH
) -> float:
    """
    Compute Mean Frequency (MNF): power-weighted centroid of the spectrum.

    Returns:
        mnf_hz : float, mean frequency in Hz
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    p_band = psd[mask].astype(np.float64)
    f_band = freqs[mask].astype(np.float64)

    total = p_band.sum()
    if total == 0:
        return 0.0
    return float(np.dot(f_band, p_band) / total)


# -------------------------------------------------------
# Validation
# -------------------------------------------------------

def validate_fft_output(
    x_input: np.ndarray,
    y_re: np.ndarray,
    y_im: np.ndarray,
    rel_tol: float = 1e-3
) -> Tuple[bool, float]:
    """
    Validate FPGA FFT output against numpy.fft.fft() reference.

    Args:
        x_input : real-valued input signal (float32)
        y_re    : FPGA FFT real output (float32)
        y_im    : FPGA FFT imaginary output (float32)
        rel_tol : relative tolerance

    Returns:
        (passed, max_relative_error)
    """
    ref = np.fft.fft(x_input.astype(np.float64))
    ref_re = ref.real.astype(np.float32)
    ref_im = ref.imag.astype(np.float32)

    # Relative error normalised by max magnitude of reference
    ref_mag = np.max(np.abs(ref_re) + np.abs(ref_im))
    if ref_mag == 0:
        return True, 0.0

    err_re = np.abs(y_re - ref_re)
    err_im = np.abs(y_im - ref_im)
    max_err = float(np.max(np.maximum(err_re, err_im))) / ref_mag

    return (max_err < rel_tol), max_err


# -------------------------------------------------------
# Visualisation (optional, requires matplotlib)
# -------------------------------------------------------

def plot_emg_spectrum(
    freqs: np.ndarray,
    psd: np.ndarray,
    mdf_hz: float,
    mnf_hz: float,
    title: str = "EMG Power Spectrum",
    f_low: float = F_LOW,
    f_high: float = F_HIGH
) -> None:
    """Plot EMG power spectrum with MDF and MNF markers."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_emg_spectrum] matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    mask = (freqs >= f_low) & (freqs <= f_high)
    ax.semilogy(freqs[mask], psd[mask], 'b-', linewidth=1.2, label='PSD')
    ax.axvline(mdf_hz, color='r', linestyle='--', linewidth=1.5,
               label=f'MDF = {mdf_hz:.1f} Hz')
    ax.axvline(mnf_hz, color='g', linestyle='-.', linewidth=1.5,
               label=f'MNF = {mnf_hz:.1f} Hz')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fatigue_trend(
    mdf_sequence: list,
    mnf_sequence: list,
    window_duration_s: float = 1.0
) -> None:
    """Plot MDF and MNF trends over time to visualise muscle fatigue."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_fatigue_trend] matplotlib not available, skipping plot.")
        return

    t_axis = np.arange(len(mdf_sequence)) * window_duration_s

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_axis, mdf_sequence, 'r-o', markersize=4, label='MDF (Hz)')
    ax.plot(t_axis, mnf_sequence, 'g-s', markersize=4, label='MNF (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Muscle Fatigue: MDF/MNF Trend Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
