"""
Self-contained check of train.py's --offline worker path: the worker is
driven IN-PROCESS (plain queue.Queue, no multiprocessing), with every
compile / measure / copy entry point monkeypatched to raise, so any attempt
to build something fails the test loudly.  Needs torch + the rl package;
no GPU (torch.cuda.is_available is forced False), no compiler, no HeCBench.

  python3 rl/test/test_offline_replay.py
"""

from __future__ import annotations

import json
import queue
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

torch.cuda.is_available = lambda: False          # never touch a GPU
torch.manual_seed(0)

import environment  # noqa: E402
import hecbench  # noqa: E402
from agent import (  # noqa: E402
    Agent, FACTOR_VALUES, N_FEATURES, _IDX_TRIP_COUNT, _IDX_TRIP_COUNT_KNOWN,
)
from train import _get_weights, _worker_fn, load_baseline_cache  # noqa: E402

fails = 0


def check(name: str, cond, detail: str = "") -> None:
    global fails
    cond = bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        fails += 1


# --- every way of BUILDING something must be unreachable ----------------------------
def _boom(what):
    def f(*a, **k):
        raise AssertionError(f"{what} was called under --offline")
    return f


hecbench.compile_single_loop_ex = _boom("compile_single_loop_ex")
hecbench.measure_kernel_time = _boom("measure_kernel_time")
environment.GpuLoopEnv.get_post_unmerge_features = _boom("get_post_unmerge_features")
_real_copytree = shutil.copytree
shutil.copytree = _boom("shutil.copytree")

BENCH = "fakebench"
LOOPS = [0, 1, 2]
TIMEOUT_PENALTY = -1.0


def _raw_features() -> list[float]:
    raw = [0.0] * N_FEATURES
    raw[_IDX_TRIP_COUNT_KNOWN] = 1.0
    raw[_IDX_TRIP_COUNT] = 64.0            # every factor legal
    return raw


def _assignments(bench_path: str) -> list[dict]:
    return [{
        "benchmark_name": BENCH, "benchmark_path": bench_path,
        "loop_idx": i, "filename": "main.cu", "triple": "nvptx64-nvidia-cuda",
        "pre_features_raw": _raw_features(), "is_kernel_function": True,
        "kernel_parents": [],
    } for i in LOOPS]


def _hparams(tmp: Path, reward_cache: dict, postf_cache: dict, failure_keys) -> dict:
    # MIRROR: run_parallel_epoch's hparams — every key the worker reads.
    return {
        "offline": True, "failure_keys": sorted(failure_keys),
        "arch": "sm_80", "n_runs": 1, "nsys_timeout": 1, "tmp_dir": str(tmp),
        "compile_timeout_penalty": TIMEOUT_PENALTY, "compile_failure_penalty": -0.25,
        "reward_deadzone": 0.0,
        "clip_eps": 0.2, "K": 1, "batch_size": 4, "lr": 1e-3,
        "value_loss_coef": 0.5, "entropy_coef": 0.01,
        "entropy_coef_unmerge": 0.05, "logit_cap": 4.0,
        "weight_decay": 0.0, "max_grad_norm": 0.5,
        "agent_type": "ppo", "bandit_epsilon": 0.0,
        "normalizer_state": {},
        "baseline_cache": {},                    # offline: absence must not skip
        "reward_cache": reward_cache, "postf_cache": postf_cache,
        "epoch": 1, "total_epochs": 1,
    }


def _run(mode: str, reward_cache: dict, postf_cache: dict, failure_keys=()) -> tuple[list[dict], Path]:
    tmp = Path(tempfile.mkdtemp(prefix="offline_replay_"))
    bench_dir = tmp / "src" / BENCH
    bench_dir.mkdir(parents=True)
    agent = Agent(clip_eps=0.2, K=1, batch_size=4, lr=1e-3, value_loss_coef=0.5,
                  entropy_coef=0.01, weight_decay=0.0, max_grad_norm=0.5,
                  device=torch.device("cpu"), entropy_coef_unmerge=0.05,
                  logit_cap=4.0)
    rq: queue.Queue = queue.Queue()
    wq: queue.Queue = queue.Queue()
    _worker_fn(0, 0, _assignments(str(bench_dir)), _get_weights(agent),
               _hparams(tmp, reward_cache, postf_cache, failure_keys), rq, wq, mode)
    msgs = []
    while not rq.empty():
        msgs.append(rq.get_nowait())
    return msgs, tmp


def _per_loop(msgs: list[dict], tag: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    dup = []
    for m in msgs:
        if m["type"] in ("entry", "eval_result", "step_failed"):
            if m["loop_idx"] in out:
                dup.append(m["loop_idx"])
            out[m["loop_idx"]] = m
    check(f"[{tag}] one message per loop, every loop accounted for",
          not dup and sorted(out) == LOOPS, f"dup={dup} seen={sorted(out)}")
    return out


def _is_noop(m: dict, sample_type: str) -> bool:
    return m.get("type") == sample_type and m.get("status") == "noop"


# --- case 1: loops 0 and 1 fully cached (loop 1 = compile-penalty cells), loop 2 absent
full = {f"{BENCH}|{i}|{u}|{f}": 0.1 * (i + 1)
        for i in (0, 1) for u in (0, 1) for f in FACTOR_VALUES}
# loop 1: every cell is a cached penalty; the unmerge row is a TIMEOUT (-1.0),
# the unroll-only row a compile failure (-0.25) — same key list, told apart by value
for u in (0, 1):
    for f in FACTOR_VALUES:
        full[f"{BENCH}|1|{u}|{f}"] = TIMEOUT_PENALTY if u == 1 else -0.25
fkeys = {k for k in full if k.startswith(f"{BENCH}|1|")}
postf = {f"{BENCH}|0": [0.5] * N_FEATURES}        # loops 1 and 2 lack a vector

for mode, sample_type in (("train", "entry"), ("eval", "eval_result")):
    msgs, tmp = _run(mode, full, postf, fkeys)
    types = [m["type"] for m in msgs]
    check(f"[{mode}] no worker crash", "worker_crashed" not in types,
          str([m.get("error") for m in msgs if m["type"] == "worker_crashed"]))
    check(f"[{mode}] exactly one worker_done, last",
          types.count("worker_done") == 1 and types[-1] == "worker_done")
    by_loop = _per_loop(msgs, mode)

    m0 = by_loop.get(0, {})
    hit0 = m0.get("type") == sample_type and m0.get("cached") is True and m0.get("status") == "ok"
    check(f"[{mode}] loop 0: cache hit (status ok) or free no-op, never a compile",
          _is_noop(m0, sample_type) or hit0, f"{m0.get('type')} status={m0.get('status')} cached={m0.get('cached')}")
    if hit0:
        check(f"[{mode}] loop 0: reward is the cached value", abs(m0["reward"] - 0.1) < 1e-9, str(m0["reward"]))

    m1 = by_loop.get(1, {})
    hit1 = m1.get("type") == sample_type and m1.get("cached") is True
    check(f"[{mode}] loop 1: cache hit or free no-op", _is_noop(m1, sample_type) or hit1,
          f"{m1.get('type')} status={m1.get('status')}")
    if hit1:
        want = "compile_timeout" if m1["unmerge"] == 1 else "compile_failed"
        check(f"[{mode}] loop 1: cached penalty replays with its live status ({want})",
              m1.get("status") == want and m1.get("timeout") == (want == "compile_timeout"),
              f"status={m1.get('status')} timeout={m1.get('timeout')}")
        check(f"[{mode}] loop 1: reward is the cached penalty",
              m1["reward"] == (TIMEOUT_PENALTY if m1["unmerge"] == 1 else -0.25), str(m1["reward"]))

    m2 = by_loop.get(2, {})
    miss2 = (m2.get("type") == "step_failed" and m2.get("reason") == "cache_miss"
             and "unmerge" in m2 and "factor" in m2)
    check(f"[{mode}] loop 2 (uncached): dropped as cache_miss naming the cell, or free no-op",
          _is_noop(m2, sample_type) or miss2, f"{m2.get('type')} {m2.get('reason')}")

    # postf_misses == loops WITHOUT a cached post-unmerge vector on which the
    # policy chose unmerge (the lookup precedes the reward lookup, so a
    # cache-miss drop still counts it).
    expected_pf = sum(1 for i in (1, 2) if by_loop.get(i, {}).get("unmerge") == 1)
    check(f"[{mode}] worker_done reports postf_misses", msgs[-1].get("postf_misses") == expected_pf,
          f"{msgs[-1].get('postf_misses')} vs {expected_pf}")

    ws = tmp / "worker_0" / "working_set"
    check(f"[{mode}] no benchmark copy made", ws.is_dir() and not any(ws.iterdir()))
    shutil.rmtree(tmp, ignore_errors=True)

# --- case 2: empty caches — nothing but no-ops and cache misses ----------------------
msgs, tmp = _run("train", {}, {})
types = [m["type"] for m in msgs]
check("[empty] no crash, one worker_done", "worker_crashed" not in types and types.count("worker_done") == 1)
by_loop = _per_loop(msgs, "empty")
check("[empty] each loop is a free no-op entry (reward 0) or a cache_miss drop",
      all((m["type"] == "entry" and m["status"] == "noop" and m["reward"] == 0.0)
          or (m["type"] == "step_failed" and m.get("reason") == "cache_miss")
          for m in by_loop.values()))
check("[empty] postf_misses == loops where unmerge was chosen",
      msgs[-1].get("postf_misses") == sum(1 for m in by_loop.values() if m.get("unmerge") == 1))
shutil.rmtree(tmp, ignore_errors=True)

# --- case 3: the read-only baseline loader ----------------------------------------------
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "baseline_cache.json"
    check("baseline loader: missing file -> {}", load_baseline_cache(p, "sm_80", 1) == {})
    p.write_text(json.dumps({"arch": "sm_80", "n_runs": 1,
                             "baselines": {BENCH: {"total_ms": 1.0, "per_kernel_ms": {}}}}))
    check("baseline loader: match -> baselines",
          load_baseline_cache(p, "sm_80", 1) == {BENCH: {"total_ms": 1.0, "per_kernel_ms": {}}})
    check("baseline loader: arch mismatch -> {} (never measures)", load_baseline_cache(p, "gfx90a", 1) == {})
    check("baseline loader: n_runs mismatch -> {}", load_baseline_cache(p, "sm_80", 5) == {})
    p.write_text("not json")
    check("baseline loader: unreadable -> {}", load_baseline_cache(p, "sm_80", 1) == {})

shutil.copytree = _real_copytree
print(f"\n=== {fails} check(s) failed ===")
sys.exit(1 if fails else 0)
