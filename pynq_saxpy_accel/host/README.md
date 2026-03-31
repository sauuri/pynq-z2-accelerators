# 코드 분석
## utils.py
```python
import time
import numpy as np

def bench_ms(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters

def stats(arr):
    arr = np.array(arr, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
    }
```
- `utils.py`는 벤치마크 스크립트에서 공통으로 사용하는 시간 측정 및 통계 계산 유틸리티를 제공한다.

### `bench_ms(fn, iters=200, warmup=50)`
- 측정할 함수(`fn`)를 인자로 받아 평균 실행 시간을 ms 단위로 반환한다.
- `warmup` 단계에서는 본 측정 전에 함수를 여러 번 실행하여 초기 실행 오버헤드를 줄인다.
  - 예: 캐시 영향, 메모리 초기화, 라이브러리 내부 준비 비용 등
- 이후 `iters`번 반복 실행한 총 시간을 측정하고, 이를 반복 횟수로 나누어 1회 평균 실행 시간(ms)을 계산한다.
- `time.perf_counter()`는 고해상도 타이머로, 짧은 실행 시간 측정에 적합하다.

### `stats(arr)`
- 실행 시간 배열을 입력받아 주요 통계값을 딕셔너리 형태로 반환한다.
- 반환 항목:
  - `mean_ms`: 평균 실행 시간
  - `p50_ms`: 중앙값(median)
  - `p90_ms`: 90번째 백분위수
  - `min_ms`: 최소 실행 시간
- 평균값만 사용할 경우 일부 큰 지연값(outlier)에 의해 결과가 왜곡될 수 있다.
- 따라서 `p50`과 `p90`을 함께 확인하면 전체 성능 분포를 더 안정적으로 해석할 수 있다.

## `bench_cpu.py`
```python


```


