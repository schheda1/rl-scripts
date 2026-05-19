"""
Audit which of the 16 original paper benchmarks pass discovery filters
in HeCBench/src and have eligible loops for the RL pipeline.

Usage:
  python scripts/rl/audit_benchmarks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hecbench import discover_benchmarks, get_loop_features, HECBENCH_SRC

ORIGINALS = [
    "haccmk-cuda",
    "complex-cuda",
    "coordinates-cuda",
    "bezier-surface-cuda",
    "lavaMD-cuda",
    "mandelbrot-cuda",
    "rainflow-cuda",
    "libor-cuda",
    "bspline-vgh-cuda",
    "bn-cuda",
    "quicksort-cuda",
    "clink-cuda",
    "contract-cuda",
    "ccs-cuda",
    "qtclustering-cuda",
    "xsbench-cuda",
]

src = HECBENCH_SRC
discovered = {b.name: b for b in discover_benchmarks(src)}

print(f"{'Benchmark':<30} {'Discovery':<12} {'Eligible loops'}")
print("-" * 60)

for name in ORIGINALS:
    if name not in discovered:
        print(f"{name:<30} {'EXCLUDED':<12} -")
        continue

    # Check eligible loops
    try:
        file_map, _, _ = get_loop_features(discovered[name])
        n_loops = sum(len(df) for df in file_map.values())
        status = str(n_loops) if n_loops > 0 else "0 (filtered out)"
    except Exception as e:
        status = f"ERROR: {e}"

    print(f"{name:<30} {'OK':<12} {status}")
