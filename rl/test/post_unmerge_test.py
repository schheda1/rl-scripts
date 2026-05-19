import sys
sys.path.insert(0, "scripts/rl")
from pathlib import Path
from hecbench import FEATURE_COLUMNS
from environment import GpuLoopEnv

bench = Path("HeCBench/src/bezier-surface-cuda")
env = GpuLoopEnv(n_runs=1)

print("=== reset (baseline + loopcount) ===")
env.reset(bench)
for lr in env.eligible_loops:
    print(f"  loop_idx={lr.loop_idx}  features={lr.pre_features.tolist()}")

# Target loop_idx=3 specifically
lr3 = next(lr for lr in env.eligible_loops if lr.loop_idx == 3)

print(f"\n=== pre-unmerge features (loop_idx=3) ===")
for col, val in zip(FEATURE_COLUMNS, lr3.pre_features.tolist()):
    print(f"  {col:30s} {val}")

print(f"\n=== post-unmerge features (loop_idx=3) ===")
post = env.get_post_unmerge_features(lr3)
for col, val in zip(FEATURE_COLUMNS, post.tolist()):
    print(f"  {col:30s} {val}")

print(f"\n=== diff ===")
changed = False
for col, pre_val, post_val in zip(FEATURE_COLUMNS, lr3.pre_features.tolist(), post.tolist()):
    if pre_val != post_val:
        print(f"  {col:30s} {pre_val} → {post_val}")
        changed = True
if not changed:
    print("  NO CHANGE — post-unmerge features identical to pre")
