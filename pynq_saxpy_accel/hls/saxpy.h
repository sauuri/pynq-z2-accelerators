#pragma once
#include <ap_int.h>

extern "C" void saxpy(
    const float* x,
    float* y,
    float a,
    int n
);
