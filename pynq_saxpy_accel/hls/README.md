# 코드 분석

## SAXPY Header(saxpy.h)

### 개요
ㅇ
'''
#pragma once
#include <ap_int.h>

extern "C" void saxpy(
    const float* x,
    float* y,
    float a, 
    int n
);
'''
### pragma once
헤더 중복 include 방지, 전통적인 #ifndef
