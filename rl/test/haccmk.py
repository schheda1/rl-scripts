import sys
sys.path.insert(0, "scripts/rl")
from pathlib import Path
from hecbench import compile_loopcount, parse_loopcount_output, ARCH

bench = Path("HeCBench/src/haccmk-cuda")
result = compile_loopcount(bench)
parsed = parse_loopcount_output(result.stderr)

for triple, file_map in parsed.items():
    if "nvptx" not in triple and "cuda" not in triple.lower():
        continue
    for fname, df in file_map.items():
        print(f"\n--- {fname} ({triple}) ---")
        print(df[["loopIdx","duplicatable","containsBarrier","containsBranch","numPaths","loopSize","function"]].to_string())
