# 코드 분석

## SAXPY Header(saxpy.h)
```
#pragma once
#include <ap_int.h>

extern "C" void saxpy(
    const float* x,    // X 배열이 메모리 어디 있는지 "주소"
    float* y,          // Y 배열이 메모리 어디 있는지 "주소", 값을 갱신하니까 const 붙이면 안됨
    float a,           // 스칼라 값 (예 : 2.5)
    int n              // 몇 개 처리할지 (예: 1,000,000)
);
```

```
# pragma once
```
- 헤더 중복 include 방지. #ifndef SAXPY_H 가드랑 동일.

``` 
include <ap_int.h>
```
- Xilinx Vitis HLS 전용 헤더. ap_int<N>, ap_uint<N> 같은 임의 비트폭 정수 타입 제공.
- HLS에서는 비트 수를 내가 직접 정할 수 있는 정수형을 제공함
- 왜? FPGA는 비트 수가 곧 하드웨어 크기랑 연결되는데 필요한 만큼의 비트를 써서
      자원사용량을 줄이고 성능이나 효율을 맞추려는 경우가 많음.
    - ap = arbitrary precision
    - int = signed 정수, uint = unsigned 정수 
    - '<N>' = 비트 수
    - 즉: N비트짜리 부호 있는(없는) 정수

```
extern "C"
```
- C++ 컴파일러는 함수 이름을 mangling 함
- 예를 들어 saxpy가 _Z5saxpyPKFPffi 같이 변형이 될 수 있는데
- Vitis HLS는 커널 함수를 이름으로 찾는데, mangling 되면 못 찾음.
- extern "C"가 이걸 막아줌.


# saxpy.cpp





