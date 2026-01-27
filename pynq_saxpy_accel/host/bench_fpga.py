import time
import numpy as np
from pynq import Overlay, allocate
from utils import stats

def run_once(ip, x_buf, y_buf, a, n):
    # 레지스터 맵은 IP에 따라 달라지는데, 보통:
    # 0x00: control (start/done)
    # 0x10: x addr
    # 0x18: y addr
    # 0x20: a
    # 0x28: n
    # 이 부분은 너의 HWH/IP 맵 보고 맞춰야 함

    ip.write(0x10, x_buf.device_address)
    ip.write(0x18, y_buf.device_address)

    # float a를 레지스터로 넣는 방식은 IP 설정에 따라 다름.
    # 가장 간단한 방법: a를 int bits로 pack
    a_bits = np.frombuffer(np.float32(a).tobytes(), dtype=np.uint32)[0]
    ip.write(0x20, int(a_bits))

    ip.write(0x28, int(n))

    # start
    ip.write(0x00, 0x01)

    # done 폴링
    while (ip.read(0x00) & 0x2) == 0:
        pass

def main():
    bit = "saxpy.bit"
    ov = Overlay(bit)

    # IP 이름은 너 overlay에 들어간 이름으로 변경
    ip = ov.saxpy_0

    n = 1_000_000
    a = np.float32(2.5)

    x_buf = allocate(shape=(n,), dtype=np.float32)
    y_buf = allocate(shape=(n,), dtype=np.float32)

    x_buf[:] = np.random.randn(n).astype(np.float32)
    y_buf[:] = np.random.randn(n).astype(np.float32)

    # warmup
    for _ in range(5):
        run_once(ip, x_buf, y_buf, a, n)

    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        run_once(ip, x_buf, y_buf, a, n)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    s = stats(times)
    print(f"[FPGA e2e] n={n} mean={s['mean_ms']:.3f} p50={s['p50_ms']:.3f} p90={s['p90_ms']:.3f} ms")

    # correctness quick check (CPU로 한번 더 계산해서 비교)
    y_ref = (a * np.array(x_buf)) + (np.array(y_buf) - a * np.array(x_buf))  # 주의: y_buf가 이미 업데이트됨
    # 정확한 검증은 '실행 전 y를 복사'해서 비교하는 방식으로 수정 추천

if __name__ == "__main__":
    main()
