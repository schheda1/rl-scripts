import sys
sys.path.insert(0, 'scripts/rl')
from hecbench import get_loop_features, make_clean, ARCH
from pathlib import Path

bmark = Path('HeCBench/src/bezier-surface-cuda')
make_clean(bmark, arch=ARCH)
file_map, primary_file, triple = get_loop_features(bmark, arch=ARCH)

print(f"triple:       {triple}")
print(f"primary file: {primary_file}")
print(f"arch: {ARCH}")
print()
for filename, df in file_map.items():
    print(f"--- {filename} ({len(df)} eligible loops) ---")
    print(df[['loopIdx','loopDepth','numPaths','loopSize','containsBranch',
               'containsChildLoops','tripCountKnown','tripCount']].to_string())


from hecbench import FEATURE_COLUMNS
for filename, df in file_map.items():
    print(f"--- {filename} ({len(df)} eligible loops) ---")
    print(df[['loopIdx'] + FEATURE_COLUMNS].to_string())
