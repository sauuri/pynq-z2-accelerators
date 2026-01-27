import numpy as np
from utils import bench_ms

def saxpy_cpu(x, y, a):
    y[:] = a * x + y

def main():
    n = 1_000_000
    a = np.float32(2.5)
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    def run():
        saxpy_cpu(x, y, a)

    ms = bench_ms(run, iters=50, warmup=10)
    print(f"[CPU numpy] n={n}  {ms:.3f} ms/iter")

if __name__ == "__main__":
    main()
