# Claude Code 리뷰 - pynq-z2-accelerators

> 작성일: 2026-02-24  
> 작성: Claude Code (claude-sonnet-4-6)  
> 대상: sauuri/pynq-z2-accelerators

---

## 총평

> **"시작은 잘 했지만 절반도 완성 안 된 프로젝트"**

- 포트폴리오로 보여주기엔 아직 부족 (bitstream도 없고 결과도 없음)
- 개인 학습/실험 목적으로는 방향성은 맞음
- 지금 상태로 GitHub에 공개된 건 조금 이른 감이 있음

---

## 긍정적인 점

- **구조 설계는 괜찮음**: `hw/`, `host/`, `docs/` 분리 개념, 재현성(Reproducibility) 명시 의도 등 FPGA 프로젝트 관리 방향은 올바름
- **HLS 기초는 제대로 함**: `m_axi` + `s_axilite` 인터페이스 구분, `PIPELINE II=1` pragma 사용 등 기본기는 있음
- **벤치마크 설계**: warmup + 반복 측정, p50/p90 통계 분리는 좋은 습관

---

## 문제점

### 미완성 상태가 심함

| 항목 | 상태 |
|------|------|
| `bitstreams/` (.bit/.hwh) | **없음** → 실제 실행 불가 |
| `hw/vivado/build.tcl` | **없음** → 빌드 재현 불가 |
| `hw/hls/script.tcl` | **없음** → HLS 재현 불가 |
| `compare_results.py` | README에 언급되지만 **없음** |
| Repro info | "fill this"로 **비어있음** |
| `docs/` 폴더 | **없음** |

### 버그: correctness check 오류 (`host/bench_fpga.py`)

```python
# 현재 코드 (틀림)
y_ref = (a * np.array(x_buf)) + (np.array(y_buf) - a * np.array(x_buf))
# y_buf가 FPGA 실행 후 이미 덮어씌워진 상태라 검증 의미 없음

# 올바른 방법: 실행 전에 y 복사본 저장
y_before = np.array(y_buf).copy()
run_once(ip, x_buf, y_buf, a, n)
y_ref = a * np.array(x_buf) + y_before
assert np.allclose(np.array(y_buf), y_ref, atol=1e-5)
```

### HLS 코드 개선 여지

- `depth=1024` 하드코딩 → n=1,000,000 실행 시 범위 초과 가능
- BURST 힌트, DATAFLOW 최적화 없음
- README는 "DMA 기반"이라고 쓰여 있으나 실제 코드는 IP 레지스터 직접 접근 방식

---

## 보완 우선순위

1. **correctness check 버그 수정** (즉시)
2. **Repro info 채우기** - PL clock, DMA 설정, PYNQ/Vivado 버전 기록
3. **빌드 스크립트 추가** - `hw/hls/script.tcl`, `hw/vivado/build.tcl`
4. **실제 측정 결과 정리** - `bench_log.csv` 내용을 `docs/results.md`에 정리
5. **bitstream 추가 또는 빌드 가이드 보완**

---

## 파일별 상세 평가

### `pynq_saxpy_accel/hls/saxpy.cpp`
- 기본 구조는 정확함
- `PIPELINE II=1` 적용으로 throughput 최적화 의도 명확
- `depth=1024` → 테스트 규모(n=1M)와 불일치, 수정 필요

### `pynq_saxpy_accel/host/bench_fpga.py`
- 레지스터 맵 주소 하드코딩 → 실제 HWH 파일과 맞는지 검증 필요
- correctness check 로직 버그 있음
- warmup(5회) + 측정(30회) 구조는 적절

### `pynq_saxpy_accel/host/bench_cpu.py`
- NumPy 기반 CPU 벤치마크, 구조 깔끔
- warmup(10회) + 측정(50회) 적절

### `pynq_saxpy_accel/host/utils.py`
- `bench_ms`, `stats` 유틸 분리는 좋은 패턴
- 코드 품질 양호
