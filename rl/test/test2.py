import subprocess, os, sys
sys.path.insert(0, 'scripts/rl')
from hecbench import _get_run_command, ARCH
from pathlib import Path

bmark = Path('HeCBench/src/bezier-surface-cuda')
run_cmd = _get_run_command(bmark, ARCH)
print(f"run command: {run_cmd}")

# Step 1: run the modified binary directly (no nsys)
result = subprocess.run(run_cmd, cwd=bmark, shell=True,
                        capture_output=True, text=True, timeout=120)
print(f"return code: {result.returncode}")
print("stdout:", result.stdout[:1000])
print("stderr:", result.stderr[:500])

# Step 2: if it ran, check what nsys produces
if result.returncode == 0:
    report = f"/tmp/nsys_diag_{os.getpid()}"
    subprocess.run(f"nsys profile --output={report} --force-overwrite=true {run_cmd}",
                   cwd=bmark, shell=True, timeout=120)
    stats = subprocess.run(
        f"nsys stats --report=cuda_gpu_kern_sum --format=csv {report}.nsys-rep",
        shell=True, capture_output=True, text=True, timeout=30)
    print("=== nsys stats stdout ===")
    print(stats.stdout)
    print("=== nsys stats stderr ===")
    print(stats.stderr)
