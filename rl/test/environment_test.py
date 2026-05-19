import sys
sys.path.insert(0, "scripts/rl")
from pathlib import Path
from environment import GpuLoopEnv
from agent import FACTOR_VALUES

bench = Path("HeCBench/src/bezier-surface-cuda")

env = GpuLoopEnv(n_runs=3)   # 3 runs for speed

print("=== reset() ===")
first_features = env.reset(bench)
print(f"  baseline_time_ms: {env._baseline_time_ms:.3f} ms")
print(f"  eligible loops: {len(env.eligible_loops)}")
for lr in env.eligible_loops:
    print(f"    loop_idx={lr.loop_idx}  features={lr.pre_features.numpy().tolist()}")

if first_features is None:
    print("No eligible loops — aborting")
    sys.exit(1)

print(f"\nFirst features (shape={first_features.shape}): {first_features.numpy().tolist()}")

print("\n=== get_post_unmerge_features() for loop 0 ===")
lr0 = env.eligible_loops[0]
post = env.get_post_unmerge_features(lr0)
print(f"  pre : {lr0.pre_features.numpy().tolist()}")
print(f"  post: {post.numpy().tolist()}")
print(f"  changed: {not lr0.pre_features.equal(post)}")

print("\n=== step() loop 0: unmerge=1, factor_idx=1 (factor=2) ===")
next_feat, reward, done = env.step(lr0, unmerge=1, factor_idx=1)
print(f"  reward={reward:.4f}  done={done}")
if next_feat is not None:
    print(f"  next_features shape: {next_feat.shape}")

print("\n=== step() remaining loops (unmerge=0, factor=1 = no-op) ===")
cursor = 1
while not env.done:
    lr = env.eligible_loops[cursor]
    next_feat, reward, done = env.step(lr, unmerge=0, factor_idx=0)
    print(f"  loop_idx={lr.loop_idx}  reward={reward:.4f}  done={done}")
    cursor += 1

print("\nEnvironment test complete.")
