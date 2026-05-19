import sys
sys.path.insert(0, 'scripts/rl')
from hecbench import compile_baseline, measure_kernel_time, ARCH
from pathlib import Path

bmark = Path('HeCBench/src/bezier-surface-cuda')
compile_baseline(bmark, arch=ARCH)
t = measure_kernel_time(bmark, arch=ARCH, n_runs=5)
print(f"mean kernel time: {t:.3f} ms")
