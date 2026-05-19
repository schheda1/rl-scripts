import sys
sys.path.insert(0, 'scripts/rl')
from hecbench import compile_loopcount, make_clean, ARCH
from pathlib import Path

bmark = Path('HeCBench/src/bezier-surface-cuda')
make_clean(bmark, arch=ARCH)
result = compile_loopcount(bmark, arch=ARCH)

print(f"return code: {result.returncode}")
print("=== STDERR (first 3000 chars) ===")
print(result.stderr[:3000])
