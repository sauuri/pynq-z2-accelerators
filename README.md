# pynq-z2-accelerators

PYNQ-Z2에서 동작하는 FPGA 가속기 모음입니다.  
A collection of FPGA accelerator projects for PYNQ-Z2.

## Projects
- **[pynq_saxpy_accel](./pynq_saxpy_accel)**  
  SAXPY (`Y = aX + Y`) end-to-end 가속: **Vitis HLS 커널 + AXI DMA + Python 호스트**  
  - Demo: `docs/demo.gif` (추가 예정)
  - Result: `docs/results.md` (SW 대비 속도 향상 요약)

## Repository layout (rule)
- `projects/*` 또는 최상위 폴더별로 “한 프로젝트 = 한 폴더”
- 각 프로젝트 폴더는 아래 산출물 규칙을 따름:
  - `hw/` (HLS/Vivado)
  - `bitstreams/` (.bit/.hwh)
  - `host/` (Python benchmark/overlay)
  - `docs/` (결과/로그/이미지)

## Reproducibility
각 프로젝트 README에 아래 정보를 고정해서 기록합니다.
- PYNQ image / Vivado / Vitis HLS 버전
- PL clock, DMA 설정, 데이터 타입/정렬 조건
- 측정 방법(전송 포함/제외)

## License
MIT
