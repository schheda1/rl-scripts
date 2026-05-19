import sys
sys.path.insert(0, "scripts/rl")
from pathlib import Path
from hecbench import compile_baseline, _get_run_command, ARCH

bench = Path("new-HeCBench/src/ccs-cuda")
print("Compiling baseline...")
ok = compile_baseline(bench)
print(f"  compile: {ok}")
print(f"  run command: {_get_run_command(bench, ARCH)}")
