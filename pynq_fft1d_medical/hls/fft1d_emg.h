// fft1d_emg.h
// 1D FFT Accelerator for EMG Signal Analysis
// Target: PYNQ-Z2 (Zynq XC7Z020)
// Tool:   Vitis HLS 2022.2

#ifndef FFT1D_EMG_H
#define FFT1D_EMG_H

#include "hls_fft.h"
#include "hls_math.h"
#include <complex>
#include <ap_fixed.h>

// -------------------------------------------------------
// Compile-time FFT parameters
// -------------------------------------------------------
#define FFT_N            1024   // Fixed transform length (2^10)
#define FFT_NFFT_MAX     10     // log2(FFT_N)
#define FFT_CONFIG_WIDTH 16
#define FFT_STATUS_WIDTH 8

// -------------------------------------------------------
// Static configuration struct for hls::ip_fft
// Inherits defaults from hls::ip_fft::params_t and overrides
// -------------------------------------------------------
struct fft_config_params : hls::ip_fft::params_t {
    // Fixed transform length: 2^10 = 1024 samples
    static const unsigned max_nfft     = FFT_NFFT_MAX;

    // No runtime length reconfiguration for max resource efficiency
    static const bool     has_nfft     = false;

    // Pipelined streaming I/O: highest throughput, overlapping frame load/compute/unload
    static const unsigned arch_opt     = hls::ip_fft::pipelined_streaming_io;

    // Natural (linear) output ordering -- no post-reorder needed on host
    static const unsigned ordering_opt = hls::ip_fft::natural_order;

    // Config and status bus widths
    static const unsigned config_width = FFT_CONFIG_WIDTH;
    static const unsigned status_width = FFT_STATUS_WIDTH;

    // Float mode: scaling is not applicable (no fixed-point overflow)
};

// -------------------------------------------------------
// Derived types
// -------------------------------------------------------
typedef hls::ip_fft::config_t<fft_config_params>  fft_config_t;
typedef hls::ip_fft::status_t<fft_config_params>  fft_status_t;

// Synthesizable Xilinx complex type with IEEE 754 float
typedef hls::x_complex<float>  cmpx_t;

// -------------------------------------------------------
// Top-level kernel function declaration
//
// Ports:
//   x_re [in]  : real part of input signal (length FFT_N), M_AXI gmem0
//   y_re [out] : real part of FFT output (length FFT_N), M_AXI gmem1
//   y_im [out] : imaginary part of FFT output (length FFT_N), M_AXI gmem2
//   dir  [in]  : 1 = forward FFT, 0 = inverse FFT
//   n    [in]  : transform length (must equal FFT_N = 1024)
// -------------------------------------------------------
extern "C" void fft1d_emg(
    const float* x_re,
    float*       y_re,
    float*       y_im,
    int          dir,
    int          n
);

#endif // FFT1D_EMG_H
