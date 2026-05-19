import sys
sys.path.insert(0, 'scripts/rl')
from hecbench import (compile_baseline, compile_single_loop,
                      measure_kernel_time, make_clean, ARCH)
from pathlib import Path

bmark = Path('HeCBench/src/bezier-surface-cuda')

# Baseline
make_clean(bmark, arch=ARCH)
compile_baseline(bmark, arch=ARCH)
baseline_time = measure_kernel_time(bmark, arch=ARCH, n_runs=5)
print(f"baseline kernel time: {baseline_time:.3f} ms")

# Loop 3, unmerge=1, factor=2
ok = compile_single_loop(
    bmark,
    loop_idx=3,
    unmerge=1,
    factor=2,
    filename='main.cu',
    triple='nvptx64-nvidia-cuda',
    arch=ARCH,
)
print(f"compile_single_loop succeeded: {ok}")
modified_time = measure_kernel_time(bmark, arch=ARCH, n_runs=5)
print(f"modified kernel time:  {modified_time:.3f} ms")

reward = (baseline_time - modified_time) / baseline_time
print(f"reward: {reward:.4f}")
