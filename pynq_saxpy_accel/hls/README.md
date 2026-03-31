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
- pragma는 컴파일러나 HLS 도구에게 주는 특별 지시문 같은 것임.
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
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=1024
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1 depth=1024
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
- 포인터 주소값(x, y의 실제 DRAM 주소)을 AXI-Lite 레지스터로 CPU가 써줌.
- 이 포인터의 메모리 주소값을 AXI-Lite 쪽 레지스터(slave 인터페이스)로 받아오겠다라는 뜻.
-  즉, Python에서 ip.write(0x10, x_buf.device_address)하는 부분이 이거임.
- 

### `bundle=gmem0` vs `bundle=gmem1`
- bundle: 같은 인터페이스 묶음 이름
- gmem0, gmem1은 global memory0,1 관례적으로 쓰는 이름(다르게 써도 됨)
- x와 y를 다른 AXI 버스에 묶음. 같은 bundle이면 x 읽고 y 쓰는게 순차적으로 일어나야 하는데,
- 분리하면 동시에 일어날 수 있음 -> 대역폭 2배.
- bundle은 인터페이스 구조를 사람이 직접 힌트 주는 것임.

### `depth=1024`?
- 이 포인터가 카리키는 메모리 깊이(원소 수)를 HLS에게 알려주는 힌트임.
- 즉, x나 y가 대충 몇 개 원소를 다룰지 알려주는 정보임.
- 작은 크기부터 큰 크기까지 여러 구간을 봐야 제대로 평가 할 수 있음.
- 왜 큰 크기를 봐야하나?
  - SAXPY 같은 커널은 실제 계산 자체보다 아래의 영향을 많이 받음.
    - 커널 시작 오버헤드
    - PS ↔ PL 제어 오버헤드
    - 메모리 접근 오버헤드
  - 그래서 입력이 너무 작으면 FPGA의 장점이 잘 안보임.
  - 길이가 아주 작으면?
    준비 시간, 호출 시간, DMA/메모리 접근 시간이 계산 시간보다 더 크게 느껴질 수 있음.
  - 길이가 크면?
    파이프라인 효과, 메모리 구조 분리 효과, 처리량 증가가 더 잘 드러날 수 있음.

- 최대 깊이는 얼마로 할 수 있을까?(최대 깊이는 무엇으로 정해지는가?)
  - (1) 보드의 DRAM 크기
  - (2) x와 y 둘 다 필요함
    - SAXPY 는 보통 x와 y 두 배열이 필요함
    - 총 바이트=2 × N × 4 -> 8N bytes
      - N = 1024 → 8KB, N = 65536 → 512KB, N = 1,000,000 → 약 8MB, N = 10,000,000 → 약 80MB
  - (3) 연속된 버퍼 할당 가능 크기
    - 연속된 물리 메모리 버퍼를 얼마나 크게 잡을 수 있느냐?
    - PYNQ의 allocate 같은 방식은 큰 연속 버퍼가 필요하니까,
    - 이론상 RAM이 남아도 큰 덩어리 할당이 안 될 수 있음

- TODO: performance test
  - 1024
  - 4096
  - 16384
  - 65536
  - 262144
  - 1048576 

### `s_axilite`
```cpp
  #pragma HLS INTERFACE s_axilite port=x bundle=control
  #pragma HLS INTERFACE s_axilite port=y bundle=control
  #pragma HLS INTERFACE s_axilite port=a bundle=control
  #pragma HLS INTERFACE s_axilite port=n bundle=control
```
- 포인터 주소, 스칼라 값들을 CPU가 AXI-Lite 레지스터로 전달하는 채널.
  PS(CPU)가 PL(FPGA 커널)에 실행에 필요한 값을 넣어주는 과정을 적어둔 것임.
  - CPU (Python)               FPGA
  - ip.write(0x10, addr)   →   [AXI-Lite 레지스터] → x 주소 저장
    - x 배열이 DRAM 어디에 있는지 알려주는 과정 
  - ip.write(0x28, n)      →   [AXI-Lite 레지스터] → n 값 저장
    - 몇 개 원소를 처리할지 알려주는 과정
  - ip.write(0x00, 0x1)    →   [control 레지스터]  → 커널 시작
    - 0x00번 쪽은 보통 control register.
    - 0x1을 쓰는 건 보통 start 신호를 1로 올려서 실행 시작이라는 뜻("준비 끝, 이제 실행해") 
- `s_axilite`?
  - AXI-Lite 는 AXI의 가벼운 버전임.
    - 주소 전달, 스칼라 값 전달, 시작 신호, 완료 상태 확인
    - 즉, 배열 전체를 보내는 용도느 아님. 짧은 값 몇 개를 레지스터에 써 넣는 용도.
    - 보통 CPU가 FPGA 커널을 제어하는 창구 로 이해하면 쉬움
  - 왜 `s_axilite`가 필요한가?
    - SAXPY 커널은 아래의 인자를 받음
    - `saxpy(const float* x, float* y, float a, int n)`
    - 이 값들은 커널이 실행되기 전에 알아야 함
      - x는 어디 메모리에 있나
      - y는 어디 메모리에 있나
      - a는 얼마인가
      - n은 몇 개인가
    - 이런 정보는 커널 시작 전에 CPU가 넣어줘야 하는데 그걸 넣는 통로가 `s_axilite`.

- ```cpp
  #pragma HLS INTERFACE s_axilite port=return bundle=control
  ```
  - `port=return` 은 커널의 시작/완료 신호를 AXI-Lite 0x00 레지스터로 연결해줘 라는 뜻.
    - 0x00은 control 레지스터 (start/done/idle 비트), port=return이 여기서 0x00 자리를 차지함.
    - void saxpy(...)는 return 값이 없지만, HLS 입장에선 "이 커널이 끝났다"는 신호가 필요한데
    - 그게 바로 port=return임. 
  - 0x00 레지스터 비트 구조:
    - bit 0 = start (CPU가 1 쓰면 커널 시작)
    - bit 1 = done (커널이 끝나면 1이 됨)
    - 그래서 bench_fpga.py에서 이렇게 폴링하는 거임
    - ```cpp
       ip.write(0x00, 0x01)          # bit0=1 → start                                                  while (ip.read(0x00) & 0x2) == 0:  # bit1이 1 될 때까지 대기
          pass    
      ``` 
 
- ```cpp
      for(int i = 0; i < n; i++){
      #pragma HLS PIPELINE II=1
          y[i] = a * x[i] + y[i];
      }
  ```
  - for 문을 가능한 한 파이프라인 구조로 만들어서,
  - 매 클럭마다 새 반복(iteration)을 시작할 수 있게 해봐 라고 지시하는 것임.

- `II = Initiation Interval`
  - 새 iteration을 시작하는 간격
  - 즉, for 문에서
    - i=0 시작
    - 그 다음 i=1은 몇 클럭 뒤에 시작 가능?
    - 그 다음 i=2은 몇 클럭 뒤에 시작 가능?
  - y[i] = a * x[i] + y[i] 는 아래의 과정과 동일함.
    1. x[i] 읽기 (AXI burst)
    2. y[i] 읽기 (AXI burst)
    3. a * x[i] 곱셈
    4. + y[i] 덧셈
    5. y[i] 쓰기
  - 아래는 파이프라인 있을때와 없을때의 차이의 예시임
    - 파이프라인 없이:
      - [X읽기→Y읽기→곱셈→덧셈→Y쓰기] i=0 끝나야
      - [X읽기→Y읽기→곱셈→덧셈→Y쓰기] i=1 시작

    - II=1 파이프라인:
      - 클럭1: i=0 X읽기
      - 클럭2: i=0 Y읽기,  i=1 X읽기
      - 클럭3: i=0 곱셈,   i=1 Y읽기,  i=2 X읽기
      - 클럭4: i=0 덧셈,   i=1 곱셈,   i=2 Y읽기,  i=3 X읽기
  - 그래서 이게 왜 빠르냐?
    - 한 개를 빨리 끝내는 것보다
    - 계속 새 작업을 밀어 넣는 것이 처리량을 높일 수 있음.
    - 즉, FPGA는 이런 반복 연산에서 동시에 여러 단계가 겹쳐 흐르게 만드는 것이 강점임.
    - **무조건 된다**가 아니라 **가능하면 이렇게 해줘**라는 목표임.
    - 실제로는 메모리 접근이 막히거나, 연산기가 부족하거나, 읽기/쓰기 충돌이 있으면
    - II=2, II=3이 될 수 있음
  - 전체 요약을 하면
    - 반복문을 한 개씩 처리하지 말고, 매 클럭마다 새 반복을 시작할 수 있게 겹쳐서 처리해달라는 뜻   
        






