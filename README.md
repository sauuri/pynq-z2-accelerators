# pynq-z2-accelerators

PYNQ-Z2 FPGA 가속기 프로젝트 모음 — HLS 기초부터 딥러닝 엣지 배포까지
A collection of FPGA accelerator projects for PYNQ-Z2, from HLS basics to deep learning edge deployment.

---

## Projects / 프로젝트 목록

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 1 | [pynq_saxpy_accel](./pynq_saxpy_accel) | SAXPY (Y = aX + Y) 하드웨어 가속 | Vitis HLS, AXI DMA |
| 2 | [pynq_fft1d_medical](./pynq_fft1d_medical) | 1D FFT 가속 (EMG 근전도 신호 분석) | HLS FFT IP, M_AXI |
| 3 | [pynq_cnn_quantized](./pynq_cnn_quantized) | CNN 양자화 + FPGA 배포 (CIFAR-10) | PTQ Static, hls4ml |
| 4 | [pynq_lstm_accel](./pynq_lstm_accel) | LSTM 양자화 + 행동 인식 (UCI HAR) | Dynamic Quant, Hybrid PL+PS |
| 5 | [pynq_transformer_edge](./pynq_transformer_edge) | TinyViT 양자화 + FPGA 엣지 배포 | PTQ, Linear Attention HLS |
| 6 | [pynq_yolo_detect](./pynq_yolo_detect) | 객체 검출 (YOLOv8n / MobileNetV2+SSD) | YOLOv8n INT8, hls4ml |
| 7 | [pynq_ecg_classifier](./pynq_ecg_classifier) | ECG 부정맥 분류 실시간 배포 (MIT-BIH) | ResNet1D, PTQ, 125Hz |
| 8 | [pynq_kws_accel](./pynq_kws_accel) | 키워드 인식 웨이크워드 감지 | DS-CNN, MFCC, Speech Commands |

---

## Learning Path / 학습 경로

```
[HLS 기초]          [신호처리]         [딥러닝 경량화]              [엣지 AI 응용]
saxpy_accel    →   fft1d_medical  →   cnn_quantized    →    ecg_classifier
  (AXI DMA)         (FFT HLS)         (PTQ Static)          (의료 신호)
                                   →   lstm_accel       →    kws_accel
                                        (Dynamic Quant)       (오디오 KWS)
                                   →   transformer_edge →    yolo_detect
                                        (ViT, Attention)      (객체 검출)
```

---

## Hardware / 하드웨어

- **FPGA Board**: PYNQ-Z2 (Zynq-7020, xc7z020clg400-1)
- **Resources**: 53,200 LUTs / 106,400 FFs / 140 BRAMs / 220 DSPs
- **Development**: MacBook (코드 작성) → GPU 서버 (모델 학습) → PYNQ-Z2 (배포)

---

## Project Structure (convention) / 폴더 규칙

```
pynq_<project_name>/
├── README.md          # 프로젝트 설명, 결과, 재현 방법
├── requirements.txt   # Python 의존성
├── train/             # 모델 학습 · 양자화 · 평가 스크립트
│   ├── model.py
│   ├── train.py
│   ├── quantize.py
│   └── evaluate.py
├── host/              # PYNQ 배포용 추론 스크립트
│   └── inference.py
└── hls/               # HLS 구현 · 합성 가이드
    └── README.md
```

---

## HLS/FPGA Toolchain / 도구

| Tool | Usage |
|------|-------|
| Vivado HLS / Vitis HLS | HLS C++ → RTL 합성 |
| Vivado | 비트스트림 생성 |
| hls4ml | 딥러닝 모델 → HLS 자동 변환 |
| FINN | 양자화 모델 → 스트리밍 데이터플로우 |
| PYNQ | Python-based overlay 제어 |

---

## Reproducibility / 재현성

각 프로젝트 README에 기록:
- PYNQ image / Vivado / Vitis HLS 버전
- PL clock, DMA 설정, 데이터 타입/정렬 조건
- 측정 방법 (warmup 횟수, 반복 횟수, 전송 포함 여부)

---

## License

MIT
