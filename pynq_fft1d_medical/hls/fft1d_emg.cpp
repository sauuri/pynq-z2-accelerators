// fft1d_emg.cpp
// 1D FFT Accelerator kernel -- HLS implementation
// Bridges M_AXI memory interface with hls::fft<> IP
//
// Architecture:
//   DDR (x_re) --[M_AXI burst]--> xn[1024] --> hls::fft --> xk[1024] --[M_AXI burst]--> DDR (y_re, y_im)
//
// DATAFLOW pragma overlaps the three phases (read / compute / write).

#include "fft1d_emg.h"

// -------------------------------------------------------
// Internal helper: read real-valued EMG input from DDR
// into complex on-chip array (imaginary part = 0)
// -------------------------------------------------------
static void read_input(const float* x_re, cmpx_t xn[FFT_N]) {
    READ_LOOP: for (int i = 0; i < FFT_N; i++) {
#pragma HLS PIPELINE II=1
        xn[i].real(x_re[i]);
        xn[i].imag(0.0f);   // EMG is real-valued
    }
}

// -------------------------------------------------------
// Internal helper: write complex FFT output from on-chip
// array back to DDR as separate real/imag arrays
// -------------------------------------------------------
static void write_output(cmpx_t xk[FFT_N], float* y_re, float* y_im) {
    WRITE_LOOP: for (int i = 0; i < FFT_N; i++) {
#pragma HLS PIPELINE II=1
        y_re[i] = xk[i].real();
        y_im[i] = xk[i].imag();
    }
}

// -------------------------------------------------------
// Top-level kernel
// -------------------------------------------------------
extern "C" void fft1d_emg(
    const float* x_re,
    float*       y_re,
    float*       y_im,
    int          dir,
    int          n
) {
    // --- AXI Interface Declarations ---
    // Three separate M_AXI bundles for simultaneous bus transactions
#pragma HLS INTERFACE m_axi port=x_re offset=slave bundle=gmem0 depth=1024 \
                       max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=y_re offset=slave bundle=gmem1 depth=1024 \
                       max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=y_im offset=slave bundle=gmem2 depth=1024 \
                       max_write_burst_length=256

    // M_AXI port address registers and scalar args via AXI-Lite control
#pragma HLS INTERFACE s_axilite port=x_re  bundle=control
#pragma HLS INTERFACE s_axilite port=y_re  bundle=control
#pragma HLS INTERFACE s_axilite port=y_im  bundle=control
#pragma HLS INTERFACE s_axilite port=dir   bundle=control
#pragma HLS INTERFACE s_axilite port=n     bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // --- On-chip complex buffers ---
    // Cyclic partition by 4 to allow 4 parallel accesses/cycle,
    // matching the pipelined_streaming_io radix-4 FFT internals
    cmpx_t xn[FFT_N];
    cmpx_t xk[FFT_N];
#pragma HLS ARRAY_PARTITION variable=xn cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=xk cyclic factor=4 dim=1

    // --- FFT runtime configuration ---
    fft_config_t fft_config;
    fft_status_t fft_status;

    // dir=1: forward FFT (time domain -> frequency domain)
    // dir=0: inverse FFT
    fft_config.setDir((dir != 0) ? 1 : 0);

    // Scaling schedule: not applicable for float mode
    fft_config.setSch(0x0);

    // --- DATAFLOW: overlap read / compute / write phases ---
    // producer-consumer chain: read_input -> hls::fft -> write_output
#pragma HLS DATAFLOW

    read_input(x_re, xn);
    hls::fft<fft_config_params>(xn, xk, &fft_status, &fft_config);
    write_output(xk, y_re, y_im);
}
