#include "saxpy.h"

extern "C" void saxpy(
    const float* x,
    float* y,
    float a,
    int n
){
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=1024
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1 depth=1024
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=y bundle=control
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // v0 correctness
    // v1: 파이프라인 (II=1 목표)
    for(int i = 0; i < n; i++){
#pragma HLS PIPELINE II=1
        y[i] = a * x[i] + y[i];
    }
}
