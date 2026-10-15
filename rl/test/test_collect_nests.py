"""
Self-contained checks for collect_nests.py: nest discovery and the greedy +
coordinate-ascent search, against synthetic data and synthetic reward
landscapes.  No compiler, no GPU, no benchmark tree.  Also runs the driver's
--dry-run on a synthetic checkpoint directory.

  python3 rl/test/test_collect_nests.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collect_nests import find_nests, search_nest  # noqa: E402

fails = 0


def check(name: str, cond, detail: str = "") -> None:
    global fails
    cond = bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails += 1


def rec(idx, depth, file="main.cu", kp=("_Zk",), triple="nvptx64-nvidia-cuda"):
    raw = [0.0] * 20
    raw[0] = float(depth)
    return {"loop_idx": idx, "filename": file, "triple": triple,
            "pre_features_raw": raw, "is_kernel_function": True, "kernel_parents": list(kp)}


# --- 1. nest discovery -----------------------------------------------------------
recs = {
    "pair":      [rec(6, 2), rec(7, 1)],
    "triple":    [rec(0, 3), rec(1, 3), rec(2, 2), rec(3, 1)],      # 0 and 1 are siblings
    "two_files": [rec(6, 2, file="a.cu"), rec(7, 1, file="b.cu")],
    "two_kern":  [rec(6, 2, kp=("_Zk1",)), rec(7, 1, kp=("_Zk2",))],
    "gap":       [rec(1, 2), rec(9, 1)],
    "depth":     [rec(6, 3), rec(7, 1)],
    "flat":      [rec(3, 1), rec(4, 1)],
}
nests = find_nests(recs)
chains = {n["benchmark"]: [l["loop_idx"] for l in n["loops"]] for n in nests}
check("pair found inner->outer", chains.get("pair") == [6, 7], str(chains.get("pair")))
check("triple: one chain through the later sibling, the other sibling stays single",
      chains.get("triple") == [1, 2, 3], str(chains.get("triple")))
check("no chain across files", "two_files" not in chains)
check("no chain across kernels", "two_kern" not in chains)
check("no chain over an index gap > 4", "gap" not in chains)
check("no chain unless depth drops by exactly one", "depth" not in chains)
check("flat kernel yields no nest", "flat" not in chains)
check("nest carries filename/triple/kernel", nests and nests[0]["filename"] == "main.cu"
      and nests[0]["triple"] == "nvptx64-nvidia-cuda" and nests[0]["kernel_parents"] == ["_Zk"])


# --- 2. search on synthetic landscapes ----------------------------------------------
class Oracle:
    """value(traj) from a table; counts how many DISTINCT multi-loop trajectories
    it was asked for (the cells a real run would have to measure)."""
    def __init__(self, table):
        self.table, self.asked = table, set()

    def __call__(self, traj):
        if not traj:
            return 0.0
        key = tuple(sorted(traj.items()))
        if len(traj) >= 2:
            self.asked.add(key)
        return self.table.get(key, 0.0)


IN, OUT = 6, 7
A1, A2, B1, B2 = (1, 2), (0, 3), (0, 2), (1, 2)
legal = {IN: [A1, A2], OUT: [B1, B2]}
DZ = 0.005


def T(**kv):   # table from {"a1": v, "a1b2": v, ...}
    names = {"a1": (IN, A1), "a2": (IN, A2), "b1": (OUT, B1), "b2": (OUT, B2)}
    t = {}
    for k, v in kv.items():
        parts = [names[k[i:i + 2]] for i in range(0, len(k), 2)]
        t[tuple(sorted(parts))] = v
    return t


# (a) additive landscape: greedy is already optimal, ascent changes nothing
o = Oracle(T(a1=0.05, a2=0.01, b1=0.02, b2=0.04, a1b1=0.07, a1b2=0.09, a2b1=0.03, a2b2=0.05))
r = search_nest([IN, OUT], legal, o, DZ, sweeps_max=4)
check("additive: greedy pair is (a1,b2)", r["greedy"] == {IN: A1, OUT: B2} and abs(r["greedy_value"] - 0.09) < 1e-9)
check("additive: ascent keeps it", r["final"] == {IN: A1, OUT: B2})
check("additive: best single recorded", r["best_single"] == {IN: A1} and r["best_single_value"] == 0.05)
check("additive: sweeps = singles + outer|a1 + inner|b2 (outer|a1 not repeated)", r["sweeps"] == 3, str(r["sweeps"]))
check("additive: measured cells = outer sweep (2) + inner re-sweep minus the pair already known (1)",
      len(o.asked) == 3, str(len(o.asked)))
check("additive: the outer was swept at exactly one non-empty context, {a1}",
      {frozenset(c.items()) for c in r["contexts"][OUT]} == {frozenset({(IN, A1)})}, str(r["contexts"][OUT]))
check("additive: inner context is only {} (the outer is not below it)", r["contexts"][IN] == [{}], str(r["contexts"][IN]))

# (b) interaction: the inner's best changes given the outer; ascent must find (a2,b2)
o = Oracle(T(a1=0.05, a2=0.0, b1=0.02, b2=0.03, a1b1=0.07, a1b2=0.10, a2b1=0.01, a2b2=0.20))
r = search_nest([IN, OUT], legal, o, DZ, sweeps_max=4)
check("interaction: greedy stops at (a1,b2)=0.10", r["greedy"] == {IN: A1, OUT: B2} and abs(r["greedy_value"] - 0.10) < 1e-9)
check("interaction: ascent reaches (a2,b2)=0.20", r["final"] == {IN: A2, OUT: B2} and abs(r["final_value"] - 0.20) < 1e-9)
check("interaction: 4 sweeps, within budget", r["sweeps"] == 4, str(r["sweeps"]))
check("interaction: cells measured <= (sweeps-1) * max legal", len(o.asked) <= 3 * 2, str(len(o.asked)))
check("interaction: outer swept at both inner contexts, {a1} and {a2}",
      {frozenset(c.items()) for c in r["contexts"][OUT]} == {frozenset({(IN, A1)}), frozenset({(IN, A2)})},
      str(r["contexts"][OUT]))

# (c) everything hurts: final is the baseline
o = Oracle(T(a1=-0.1, a2=-0.2, b1=-0.05, b2=-0.3, a1b1=-0.2, a1b2=-0.4, a2b1=-0.3, a2b2=-0.5))
r = search_nest([IN, OUT], legal, o, DZ, sweeps_max=4)
check("all-negative: final is no-op with value 0", r["final"] == {} and r["final_value"] == 0.0)
check("all-negative: greedy is also no-op", r["greedy"] == {} and r["greedy_value"] == 0.0)

# (d) budget: sweeps_max=2 means greedy only
o = Oracle(T(a1=0.05, a2=0.0, b1=0.02, b2=0.03, a1b1=0.07, a1b2=0.10, a2b1=0.01, a2b2=0.20))
r = search_nest([IN, OUT], legal, o, DZ, sweeps_max=2)
check("budget: sweeps_max=2 stops after greedy", r["sweeps"] == 2 and r["final"] == {IN: A1, OUT: B2})

# (e) an unmeasurable pair (None) is never chosen; a tie keeps no-op
o = Oracle(T(a1=0.05, b1=0.02, a1b1=0.05, a1b2=0.05))
o.table[tuple(sorted({IN: A1, OUT: B2}.items()))] = None
r = search_nest([IN, OUT], legal, o, DZ, sweeps_max=4)
check("None is never chosen and a tie with no-op keeps no-op", r["final"] == {IN: A1} and r["final_value"] == 0.05, str(r["final"]))

# (f) the outer's single beats the inner's: the incumbent starts there and ascent sweeps the inner under it
o = Oracle(T(a1=0.01, a2=0.0, b1=0.08, b2=0.02, a1b1=0.03, a1b2=0.02, a2b1=0.15, a2b2=0.01))
r = search_nest([IN, OUT], legal, o, DZ, sweeps_max=4)
check("wrong-loop case: best single is the outer's", r["best_single"] == {OUT: B1})
check("wrong-loop case: ascent finds (a2,b1)=0.15 from the outer's single", r["final"] == {IN: A2, OUT: B1}, str(r["final"]))

# (g) triple, additive: greedy chain of three, then one ascent sweep
I0, I1, I2 = 1, 2, 3
leg3 = {I0: [A1], I1: [B1], I2: [A2]}
tbl = {}
def add3(d):
    return sum({(I0, A1): 0.02, (I1, B1): 0.03, (I2, A2): 0.04}[kv] for kv in d)
import itertools
for n in range(1, 4):
    for combo in itertools.combinations([(I0, A1), (I1, B1), (I2, A2)], n):
        tbl[tuple(sorted(combo))] = add3(combo)
o = Oracle(tbl)
r = search_nest([I0, I1, I2], leg3, o, DZ, sweeps_max=4)
check("triple: greedy finds all three", r["greedy"] == {I0: A1, I1: B1, I2: A2} and abs(r["greedy_value"] - 0.09) < 1e-9)
check("triple: contexts of the outermost include the two below it",
      {frozenset(c.items()) for c in r["contexts"][I2]} >= {frozenset({(I0, A1), (I1, B1)})})


# --- 3. the driver's dry run on a synthetic checkpoint ----------------------------
import collect_nests  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    ck = d / "ckpt"; ck.mkdir()
    src = d / "src"; (src / "pair").mkdir(parents=True); (src / "nobl").mkdir()
    (ck / "eligible_benchmarks.json").write_text(json.dumps({
        "eligible": ["pair", "nobl", "flat"],
        "loop_records": {"pair": [rec(6, 2), rec(7, 1)], "nobl": [rec(6, 2), rec(7, 1)],
                         "flat": [rec(3, 1)]},
    }))
    (ck / "reward_cache.json").write_text(json.dumps({"rewards": {"pair|6|1|2": 0.05}}))
    (ck / "baseline_cache.json").write_text(json.dumps({"baselines": {"pair": {"total_ms": 1.0}}}))
    sys.argv = ["collect_nests.py", str(ck), "--dry-run", "--hecbench-src", str(src)]
    try:
        collect_nests.main()
        check("dry run completes on a synthetic checkpoint (one runnable nest, one without baseline, one flat)", True)
    except SystemExit as e:
        check("dry run completes", e.code in (None, 0), f"exit {e.code}")
    except Exception as e:  # noqa: BLE001
        check("dry run completes", False, repr(e))
    check("dry run wrote nothing into the checkpoint",
          sorted(os.listdir(ck)) == ["baseline_cache.json", "eligible_benchmarks.json", "reward_cache.json"],
          str(sorted(os.listdir(ck))))

print(f"\n=== {fails} check(s) failed ===")
sys.exit(1 if fails else 0)
