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
