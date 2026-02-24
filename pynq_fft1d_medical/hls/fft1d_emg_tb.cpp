// fft1d_emg_tb.cpp
// C simulation testbench for fft1d_emg kernel
//
// Test signal: synthetic EMG at 4 kHz sampling rate
//   - 50 Hz component  (low-frequency EMG activity)
//   - 120 Hz component (medium-frequency motor units)
//   - 250 Hz component (high-frequency motor units)
//   - Gaussian noise
//
// Verification: compare against O(N^2) DFT reference
//   - Peak bins must match expected frequencies
//   - Max absolute error < PASS_TOL

#include "fft1d_emg.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define FS_HZ      4000.0f
#define M_PI_F     3.14159265358979323846f
#define PASS_TOL   1e-2f    // Tolerance for float FFT vs reference DFT

// -------------------------------------------------------
// Generate synthetic EMG signal
// -------------------------------------------------------
static void generate_emg(float* x, int n, float fs) {
    // Pseudo-random seed for reproducibility
    srand(42);
    for (int i = 0; i < n; i++) {
        float t = (float)i / fs;
        float emg = 0.6f * sinf(2.0f * M_PI_F * 50.0f  * t)
                  + 0.4f * sinf(2.0f * M_PI_F * 120.0f * t)
                  + 0.3f * sinf(2.0f * M_PI_F * 250.0f * t);
        // Add Gaussian noise (Box-Muller approximation via uniform)
        float noise = 0.05f * ((float)rand() / RAND_MAX - 0.5f);
        x[i] = emg + noise;
    }
}

// -------------------------------------------------------
// Reference O(N^2) DFT for verification
// (Not synthesized -- testbench only)
// -------------------------------------------------------
static void reference_dft(const float* xr, float* yr, float* yi, int N) {
    for (int k = 0; k < N; k++) {
        yr[k] = 0.0f;
        yi[k] = 0.0f;
        for (int n = 0; n < N; n++) {
            float angle = -2.0f * M_PI_F * k * n / (float)N;
            yr[k] += xr[n] * cosf(angle);
            yi[k] += xr[n] * sinf(angle);
        }
    }
}

// -------------------------------------------------------
// Find bin index of peak magnitude in range [f_low, f_high]
// -------------------------------------------------------
static int find_peak_bin(const float* yr, const float* yi, int N,
                         float fs, float f_low, float f_high) {
    int bin_low  = (int)(f_low  * N / fs);
    int bin_high = (int)(f_high * N / fs);
    if (bin_high > N / 2) bin_high = N / 2;

    float max_mag = -1.0f;
    int   peak_bin = bin_low;
    for (int k = bin_low; k <= bin_high; k++) {
        float mag = yr[k] * yr[k] + yi[k] * yi[k];
        if (mag > max_mag) {
            max_mag  = mag;
            peak_bin = k;
        }
    }
    return peak_bin;
}

// -------------------------------------------------------
// Main testbench
// -------------------------------------------------------
int main() {
    printf("=== fft1d_emg C Simulation Testbench ===\n");
    printf("FFT size: %d, Sampling rate: %.0f Hz\n\n", FFT_N, FS_HZ);

    // Allocate buffers
    static float x_re[FFT_N];
    static float y_re[FFT_N];
    static float y_im[FFT_N];
    static float ref_re[FFT_N];
    static float ref_im[FFT_N];

    // 1. Generate synthetic EMG signal
    generate_emg(x_re, FFT_N, FS_HZ);

    // 2. Run HLS kernel (C simulation)
    fft1d_emg(x_re, y_re, y_im, /*dir=*/1, /*n=*/FFT_N);

    // 3. Compute reference DFT
    // NOTE: O(N^2) DFT is slow for large N; acceptable in testbench only
    printf("Computing reference DFT (this may take a moment)...\n");
    reference_dft(x_re, ref_re, ref_im, FFT_N);

    // 4. Compare outputs: check max absolute error
    float max_err = 0.0f;
    for (int k = 0; k < FFT_N; k++) {
        float err_re = fabsf(y_re[k] - ref_re[k]);
        float err_im = fabsf(y_im[k] - ref_im[k]);
        float err    = (err_re > err_im) ? err_re : err_im;
        if (err > max_err) max_err = err;
    }
    printf("Max absolute error (HLS vs DFT reference): %.6f\n", max_err);

    // 5. Verify peak frequencies
    // 50 Hz peak expected at bin: 50 * 1024 / 4000 = bin 12 (±1)
    // 120 Hz peak expected at bin: 120 * 1024 / 4000 = bin 31 (±1)
    // 250 Hz peak expected at bin: 250 * 1024 / 4000 = bin 64 (±1)
    int peak_50  = find_peak_bin(y_re, y_im, FFT_N, FS_HZ, 40.0f,  65.0f);
    int peak_120 = find_peak_bin(y_re, y_im, FFT_N, FS_HZ, 100.0f, 145.0f);
    int peak_250 = find_peak_bin(y_re, y_im, FFT_N, FS_HZ, 230.0f, 275.0f);

    float freq_50  = (float)peak_50  * FS_HZ / FFT_N;
    float freq_120 = (float)peak_120 * FS_HZ / FFT_N;
    float freq_250 = (float)peak_250 * FS_HZ / FFT_N;

    printf("\nPeak frequency detection:\n");
    printf("  Expected  50 Hz -> detected %.1f Hz (bin %d)\n", freq_50,  peak_50);
    printf("  Expected 120 Hz -> detected %.1f Hz (bin %d)\n", freq_120, peak_120);
    printf("  Expected 250 Hz -> detected %.1f Hz (bin %d)\n", freq_250, peak_250);

    // 6. Pass/Fail evaluation
    int freq_ok = (fabsf(freq_50  -  50.0f) <  8.0f) &&
                  (fabsf(freq_120 - 120.0f) < 10.0f) &&
                  (fabsf(freq_250 - 250.0f) < 10.0f);

    int err_ok  = (max_err < PASS_TOL);

    printf("\n--- Results ---\n");
    printf("  Frequency peaks: %s\n", freq_ok ? "PASS" : "FAIL");
    printf("  Numerical error: %s (max_err=%.6f, tol=%.6f)\n",
           err_ok ? "PASS" : "FAIL", max_err, PASS_TOL);

    if (freq_ok && err_ok) {
        printf("\n[TESTBENCH] PASS\n");
        return 0;
    } else {
        printf("\n[TESTBENCH] FAIL\n");
        return 1;
    }
}
