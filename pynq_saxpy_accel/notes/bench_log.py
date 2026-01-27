import csv, os
from datetime import datetime

# 너의 측정값 넣기
platform = "PYNQ-Z2"
project  = "saxpy"
n = 1_000_000
a = 2.5
cpu_ms = 68.117
fpga_kernel_ms = 211.247
fpga_e2e_ms = 211.860
speedup_kernel = cpu_ms / fpga_kernel_ms
speedup_e2e = cpu_ms / fpga_e2e_ms
max_error = 0.0
notes = "First end-to-end run; 64-bit addr regs x_1/x_2,y_1/y_2"

now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H:%M:%S")  # ✅ 시:분:초

row = [
    date_str, time_str,
    platform, project, n, a,
    f"{cpu_ms:.3f}", f"{fpga_kernel_ms:.3f}", f"{fpga_e2e_ms:.3f}",
    f"{speedup_kernel:.2f}", f"{speedup_e2e:.2f}",
    f"{max_error:.6g}", notes
]

path = "bench_log.csv"
header = ["date","time","platform","project","n","a","cpu_numpy_ms","fpga_kernel_ms","fpga_e2e_ms",
          "speedup_kernel","speedup_e2e","max_error","notes"]

write_header = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(header)
    w.writerow(row)

print("Wrote:", path)
