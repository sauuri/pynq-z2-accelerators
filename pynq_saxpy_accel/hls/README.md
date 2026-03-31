# 코드 분석

## SAXPY Header(saxpy.h)
```cpp
#pragma once
#include <ap_int.h>

extern "C" void saxpy(
    const float* x,    // X 배열이 메모리 어디 있는지 "주소"
    float* y,          // Y 배열이 메모리 어디 있는지 "주소", 값을 갱신하니까 const 붙이면 안됨
    float a,           // 스칼라 값 (예 : 2.5)
    int n              // 몇 개 처리할지 (예: 1,000,000)
);
```

### `# pragma once`
- 헤더 중복 include 방지한다.
- `#ifndef SAXPY_H` 형태의 include guard와 같은 목적을 가진다..

### `include <ap_int.h>`
- Xilinx Vitis HLS 전용 헤더. ap_int<N>, ap_uint<N> 같은 임의 비트폭 정수 타입 제공.
- HLS에서는 비트 수를 내가 직접 정할 수 있는 정수형을 제공함
- 왜? FPGA는 비트 수가 곧 하드웨어 크기랑 연결되는데 필요한 만큼의 비트를 써서
      자원사용량을 줄이고 성능이나 효율을 맞추려는 경우가 많음.
    - ap = arbitrary precision
    - int = signed 정수, uint = unsigned 정수 
    - '<N>' = 비트 수
    - 즉: N비트짜리 부호 있는(없는) 정수

### `extern "C"`
- C++ 컴파일러는 함수 이름을 mangling 함
- 예를 들어 saxpy가 _Z5saxpyPKFPffi 같이 변형이 될 수 있는데
- Vitis HLS는 커널 함수를 이름으로 찾는데, name mangling 되면 못 찾음.
- extern "C"가 이걸 막아줌.


# saxpy.cpp
```
#pragma HLS INTERFACE m_axi port=x offest=slave bundle=gmem0 depth=1024
#pragma HLS INTERFACE m_axi port=x offest=slave bundle=gmem1 depth=1024
```
###`m_axi`
- AXI Master 인터페이스. FPGA가 PS(ARM) DRAM에 직접 접근할 수 있게 해줌.
  PS(ARM) DRAM (Processing System, 컴퓨터 두뇌 쪽)
      ↑↓ (AXI Master - FPGA가 주도권 가짐)
    FPGA PL  (Programmable Logic, 회로를 직접 만드는 FPGA 쪽)

  개념 설명
  - PS, PL이 서로 메모리(DRAM)을 주고받을 때 AXI 라는 길을 씀(PS 안에는 ARM CPU가 들어 있음)
  - AXI는 데이터를 주고 받는 인터페이스 규칙(PS, PL, DRAM 사이를 연결하는 도로 규칙)
    즉, 어떻게 주소를 보내고, 데이터를 읽고 쓰고, 완료를 알릴지에 대한 약속이 AXI
  - AXI Master?
    Master는 먼저 요청을 보내는 쪽(이 주소의 데이터를 읽어와, 이 주소에 데이터를 써)
    반대로 Slave는 요청을 받는 쪽임.
    
### `offset=slave`
- 

















