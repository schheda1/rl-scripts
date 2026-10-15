"""
Box-side check of the trajectory primitive (rl/trajectory.py + hecbench's
compile_trajectory_ex) against the numbers already measured by hand on
multimaterial-cuda/compact.cu (-DFUSED), nest idx 7 > 6 > 5.

Compile-only, no GPU.  Each state_after call is one `make` of the benchmark
(all three programs), so expect a few minutes in total.

Usage (CUDA box, same env as the collectors: HECBENCH_SRC, IR2VEC_VOCAB, TARGET_ARCH):
  python3 rl/test/test_trajectory.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from features import STRUCTURAL_COLUMNS  # noqa: E402
from hecbench import ARCH, HECBENCH_SRC, compile_multi_loop, compile_trajectory_ex  # noqa: E402
from trajectory import (  # noqa: E402
    NOOP,
    load_trajectory_cache,
    parse_post_key,
    parse_trajectory_key,
    post_key,
    prune_noops,
    save_trajectory_cache,
    state_after,
    trajectory_key,
)

NP = STRUCTURAL_COLUMNS.index("numPaths")
SZ = STRUCTURAL_COLUMNS.index("loopSize")
DEPTH = STRUCTURAL_COLUMNS.index("loopDepth")

fails = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global fails
    cond = bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails += 1


def v(t, i: int) -> float:
    """Entry i of a feature tensor as a float (nan when the tensor is missing)."""
    return float("nan") if t is None else float(t[i])


# --- 1. keys: pure logic, no compiler ----------------------------------------
k = trajectory_key("b", {7: (0, 3), 6: (1, 2), 5: NOOP})
check("key sorts by idx and drops no-ops", k == "b|6:1:2,7:0:3", k)
check("empty trajectory key is the baseline", trajectory_key("b", {5: NOOP}) == "b|")
check("key round-trips", parse_trajectory_key(k) == ("b", {6: (1, 2), 7: (0, 3)}))
check("prune_noops normalises types", prune_noops({"6": ("1", "2"), 5: (0, 1)}) == {6: (1, 2)})
check("post_key shape", post_key(k, 7) == "b|6:1:2,7:0:3|7")
check("post_key round-trips", parse_post_key(post_key(k, 7)) == (k, 7))
try:
    parse_trajectory_key("no-separator"); check("malformed key is rejected", False)
except ValueError:
    check("malformed key is rejected", True)

# --- 2. cache: round trip in a temp dir, separate file ------------------------
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "trajectory_cache.json"
    save_trajectory_cache(p, {k: 0.05}, {post_key(k, 7): [1.0, 2.0]},
                          failure_keys={"b|9:1:2"}, timeout_keys={"b|9:1:9"},
                          failure_penalty=-0.16, timeout_penalty=-0.16, deadzone=0.02)
    c = load_trajectory_cache(p)
    check("cache rewards round-trip", c["rewards"] == {k: 0.05})
    check("cache post-features round-trip", c["post_features"] == {post_key(k, 7): [1.0, 2.0]})
    check("cache migration keeps failure and timeout keys separately",
          c["migration"]["failure_keys"] == ["b|9:1:2"] and c["migration"]["timeout_keys"] == ["b|9:1:9"])
    check("cache writes only its own file", sorted(os.listdir(d)) == ["trajectory_cache.json"])

# --- 3. the compiler path on the known nest -----------------------------------
B = Path(os.environ.get("HECBENCH_SRC", str(HECBENCH_SRC))) / "multimaterial-cuda"
if not B.is_dir():
    print(f"SKIP  compiler checks: {B} not found")
    sys.exit(1 if fails else 0)
FILE, T = "compact.cu", "nvptx64-nvidia-cuda"
IN, OUT = 6, 7


def st(actions, target, pre=None):
    return state_after(B, actions, target, FILE, T, arch=ARCH, timeout=300,
                       pre_features_raw=pre)


# baseline rows (plain LoopCount, lookup by loopIdx)
base_out = st({}, OUT)
check("baseline outer read by loopIdx", v(base_out, NP) == 3 and v(base_out, SZ) == 177,
      f"numPaths={v(base_out, NP):.0f} loopSize={v(base_out, SZ):.0f}")
base_in = st({}, IN)
check("baseline inner read by loopIdx", v(base_in, NP) == 6 and v(base_in, SZ) == 153)

# all-no-op shortcut returns the given pre-features without compiling
marker = torch.full_like(base_out, -42.0) if base_out is not None else None
if marker is not None:
    check("all-no-op shortcut returns pre_features_raw", torch.equal(st({IN: NOOP}, OUT, pre=marker), marker))

# the state transition: the outer after the inner's action (numbers from the U_in / S_in runs)
o_after_unmerge = st({IN: (1, 2)}, OUT)
check("outer after inner unmerge x2: numPaths 9, loopSize 1285 (read by stamp while its row moved 7 -> 13)",
      v(o_after_unmerge, NP) == 9 and v(o_after_unmerge, SZ) == 1285,
      f"numPaths={v(o_after_unmerge, NP):.0f} loopSize={v(o_after_unmerge, SZ):.0f}")
o_after_unroll = st({IN: (0, 3)}, OUT)
check("outer after inner unroll x3: numPaths 5, loopSize 481",
      v(o_after_unroll, NP) == 5 and v(o_after_unroll, SZ) == 481,
      f"numPaths={v(o_after_unroll, NP):.0f} loopSize={v(o_after_unroll, SZ):.0f}")

# the cloned case: inner after B (3 identical copies -> mean equals one copy)
i_after_B = st({IN: (1, 2), OUT: (0, 3)}, IN)
check("inner after B is the mean of 3 identical copies: numPaths 36, loopSize 1261, depth 2",
      v(i_after_B, NP) == 36 and v(i_after_B, SZ) == 1261 and v(i_after_B, DEPTH) == 2,
      f"numPaths={v(i_after_B, NP):.0f} loopSize={v(i_after_B, SZ):.0f}")
o_after_B = st({IN: (1, 2), OUT: (0, 3)}, OUT)
check("outer after B: numPaths 729, loopSize 3853",
      v(o_after_B, NP) == 729 and v(o_after_B, SZ) == 3853)

# a stamp that no loop carries -> None, not a wrong row
check("unknown target after a transform returns None", st({IN: (1, 2)}, 9999) is None)

# --- 4. compile_trajectory_ex / compile_multi_loop ----------------------------
ok, sig = compile_trajectory_ex(B, {IN: (1, 1), OUT: (1, 1)}, FILE, T, arch=ARCH, timeout=300)
check("C1 (unmerge above unmerged loop with children) compiles via the primitive", ok, sig)
check("compile_multi_loop wrapper agrees", compile_multi_loop(B, {IN: (1, 1), OUT: (1, 1)}, FILE, T, arch=ARCH))
ok, sig = compile_trajectory_ex(B, {IN: NOOP, OUT: NOOP}, FILE, T, arch=ARCH, timeout=300)
check("all-no-op trajectory compiles as the baseline (no UU flags)", ok, sig)

print(f"\n=== {fails} check(s) failed ===")
sys.exit(1 if fails else 0)
