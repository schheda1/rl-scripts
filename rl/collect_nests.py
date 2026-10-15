"""
Nested-loop (trajectory) collector: greedy search + coordinate ascent per nest.

A NEST here is an ancestor chain of ELIGIBLE loops inside one kernel (inner ->
outer), found from the eligibility cache the way the nest count was made: same
file and kernel, consecutive loop indices, depth decreasing by one per step.
The population is ~30 chains (29 pairs, 1 triple).

Per nest, the search asks for the reward of TRAJECTORIES — a set of
(loop -> (unmerge, factor)) decisions applied in ONE compile (hecbench.
compile_trajectory_ex; the pass applies them innermost-first):

  singles      R(a on one loop, everything else no-op): read from
               reward_cache.json, NEVER re-measured; the no-op trajectory is 0.
  greedy       inner fixed at its best single, outer swept over its legal cells
               (then the next loop up for a triple).
  ascent       re-sweep each loop holding the others at the incumbent; accept a
               move only if it beats the incumbent by more than the deadzone;
               stop when a full pass changes nothing or at --sweeps-max.

Coordinate ascent is the DATA-COLLECTION search only: it produces the
outer-given-inner slices a sequential policy trains on and the inner-given-outer
slices that give the inner step a value anticipating the outer.  It is not part
of training or zero-shot deployment.

Measurement is BIT-IDENTICAL to collect_cells (blocks marked MIRROR): tree copy,
compile with --compile-timeout, measure_kernel_time with the same per-kernel
filter and run count, clip at -1 then symmetric deadzone, compile failure and
timeout mapped to the same penalties.  Results go to a SEPARATE cache,
trajectory_cache.json (reward_cache.json is read-only here).  For every
context at which a loop was swept, the loop's post-transform RAW feature
vector (trajectory.state_after, by identity stamp) is stored too — the state
the policy will see at that step.

Resumable: trajectories already cached are never re-measured.

Usage:
  python3 collect_nests.py CKPT_DIR --dry-run                      # the plan, no GPU
  python3 collect_nests.py CKPT_DIR --num-workers 2 --n-runs 20 \\
      --compile-failure-penalty -0.16 --reward-deadzone 0.005 \\
      [--heuristic-csv CKPT_DIR/heuristic_decisions.csv] [--sweeps-max 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import shutil
import sys
import time
from pathlib import Path

import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))

from collect_cells import valid_cells                           # noqa: E402
from hecbench import ARCH, HECBENCH_SRC                         # noqa: E402
from trajectory import (                                        # noqa: E402
    TRAJECTORY_CACHE_NAME, load_trajectory_cache, post_key,
    save_trajectory_cache, trajectory_key,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("nests")

_IDX_DEPTH = 0          # loopDepth in pre_features_raw (features.STRUCTURAL_COLUMNS)


# ---------------------------------------------------------------------------
# Nest discovery (pure)
# ---------------------------------------------------------------------------

def find_nests(loop_records_map: dict) -> list[dict]:
    """
    Eligible ancestor chains, inner -> outer, from the eligibility cache.
    Grouping: (benchmark, filename, kernel_parents).  Rule: records sorted by
    loop_idx; consecutive records chain when depth drops by exactly one and
    the index gap is <= 4 (post-order numbering puts a child right before its
    parent, with at most its later siblings' subtrees in between).
    """
    nests: list[dict] = []
    for name, recs in loop_records_map.items():
        groups: dict = {}
        for r in recs:
            k = (r["filename"], tuple(sorted(r.get("kernel_parents", []))))
            groups.setdefault(k, []).append(r)
        for (fname, kp), lst in groups.items():
            lst = sorted(lst, key=lambda r: int(r["loop_idx"]))
            i = 0
            while i < len(lst):
                j = i
                while (j + 1 < len(lst)
                       and int(lst[j + 1]["pre_features_raw"][_IDX_DEPTH])
                       == int(lst[j]["pre_features_raw"][_IDX_DEPTH]) - 1
                       and int(lst[j + 1]["loop_idx"]) - int(lst[j]["loop_idx"]) <= 4):
                    j += 1
                if j > i:
                    nests.append({
                        "benchmark": name, "filename": fname,
                        "triple": lst[i]["triple"],
                        "kernel_parents": list(kp),
                        "loops": [dict(lst[t]) for t in range(i, j + 1)],   # inner -> outer
                    })
                i = j + 1
    return nests


# ---------------------------------------------------------------------------
# Search (pure): greedy + coordinate ascent over a value oracle
# ---------------------------------------------------------------------------

def search_nest(chain: list[int], legal: dict[int, list], value, deadzone: float,
                sweeps_max: int) -> dict:
    """
    chain      loop indices inner -> outer.
    legal[k]   legal non-no-op cells of loop k.
    value(t)   reward of trajectory t ({loop: cell}, no-ops absent), or None if
               it could not be measured (never chosen).  {} is the baseline, 0.
    Returns the incumbent trajectory and value, the greedy pair, the best
    single, the sweeps performed and, per loop, the contexts (actions on loops
    BELOW it) at which it was swept — the post-states the policy needs.
    """
    swept: set = set()
    contexts: dict[int, set] = {k: set() for k in chain}
    sweeps = 0

    def sweep(k, fixed):
        """Best cell of k given the others fixed; None means no-op is best."""
        nonlocal sweeps
        sig = (k, frozenset(fixed.items()))
        if sig in swept:
            return None, None       # already done at this context: nothing new
        base = value(dict(fixed))
        if base is None:
            return None, None       # the context itself is unmeasurable: nothing to compare to
        swept.add(sig)
        sweeps += 1
        below = {j: a for j, a in fixed.items() if chain.index(j) < chain.index(k)}
        contexts[k].add(frozenset(below.items()))
        best_cell, best_v = None, base
        for cell in legal[k]:
            v = value({**fixed, k: cell})
            if v is not None and v > best_v:          # ties keep no-op
                best_cell, best_v = cell, v
        return best_cell, best_v

    # incumbent: the best single over the chain (no-op included) — all cached
    inc, inc_v = {}, 0.0
    singles = {}
    for k in chain:
        for cell in legal[k]:
            v = value({k: cell})
            singles[(k, cell)] = v
            if v is not None and v > inc_v:
                inc, inc_v = {k: cell}, v
    best_single = (dict(inc), inc_v)

    # greedy: inner -> outer, each given the choices below it
    fixed: dict = {}
    g_v = 0.0
    for k in chain:
        cell, v = sweep(k, fixed)
        if v is None:                      # context already swept (cannot happen here)
            continue
        if cell is not None:
            fixed[k] = cell
        g_v = v
    greedy = (dict(fixed), g_v)
    if g_v > inc_v + deadzone:
        inc, inc_v = dict(fixed), g_v

    # coordinate ascent from the incumbent
    while sweeps < sweeps_max:
        improved = False
        for k in chain:
            if sweeps >= sweeps_max:
                break
            ctx = {j: a for j, a in inc.items() if j != k}
            cell, v = sweep(k, ctx)
            if v is None:
                continue
            new = dict(ctx)
            if cell is not None:
                new[k] = cell
            if v > inc_v + deadzone:
                inc, inc_v, improved = new, v, True
        if not improved:
            break

    return {
        "final": inc, "final_value": inc_v,
        "greedy": greedy[0], "greedy_value": greedy[1],
        "best_single": best_single[0], "best_single_value": best_single[1],
        "singles": singles, "sweeps": sweeps,
        "contexts": {k: [dict(c) for c in v] for k, v in contexts.items()},
    }


# ---------------------------------------------------------------------------
# Worker: one nest at a time, cells measured serially on its GPU
# ---------------------------------------------------------------------------

def _worker(rank: int, gpu_id: int, work_q, hp: dict, result_q) -> None:
    import subprocess
    from pathlib import Path as _Path

    _here = _Path(__file__).parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from hecbench import (compile_trajectory_ex, demangle, demangled_to_filter,
                          measure_kernel_time)
    from trajectory import state_after

    _log = logging.getLogger(f"nests.w{rank}")
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s %(levelname)s [W{rank}] %(message)s",
                        datefmt="%H:%M:%S")
    try:
        # MIRROR: collect_cells._collect_worker — pin BOTH runtimes before any GPU call.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        tmp_dir = _Path(hp["tmp_dir"]) / f"nests_{rank}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        rewards: dict = dict(hp["traj_rewards"])      # trajectory cache (shared snapshot)
        single_rewards: dict = hp["single_rewards"]   # reward_cache.json, read-only
        baseline_cache: dict = hp["baseline_cache"]
        dz = hp["reward_deadzone"]

        while True:
            nest = work_q.get()
            if nest is None:
                break
            bench = nest["benchmark"]
            chain = [int(l["loop_idx"]) for l in nest["loops"]]
            filename, triple = nest["filename"], nest["triple"]
            legal = {int(l["loop_idx"]): valid_cells(l["pre_features_raw"]) for l in nest["loops"]}

            # MIRROR: collect_cells — kernel filter and baseline as a COUPLED pair.
            kp = nest["loops"][0].get("kernel_parents", [])
            kernel_filter = None
            baseline_ms = baseline_cache.get(bench, {}).get("total_ms", 0.0)
            if len(kp) == 1:
                _kf = demangled_to_filter(demangle(kp[0]))
                _per = baseline_cache.get(bench, {}).get("per_kernel_ms", {}).get(_kf)
                if _per is not None:
                    kernel_filter, baseline_ms = _kf, _per

            copy_dir = tmp_dir / "working_set" / bench
            try:
                if copy_dir.exists():
                    shutil.rmtree(copy_dir)
                shutil.copytree(_Path(hp["bench_paths"][bench]), copy_dir)
            except Exception as e:
                _log.warning("Failed to copy %s: %s — skipping nest", bench, e)
                result_q.put({"type": "nest_done", "benchmark": bench, "chain": chain,
                              "error": f"copy failed: {e}"})
                continue

            measured = 0

            def value(traj: dict):
                """Trajectory reward: cache -> single-loop cache -> measure."""
                nonlocal measured
                if not traj:
                    return 0.0
                key = trajectory_key(bench, traj)
                if key in rewards:
                    return rewards[key]
                if len(traj) == 1:
                    (k, (u, f)), = traj.items()
                    sk = f"{bench}|{k}|{u}|{f}"
                    if sk in single_rewards:
                        return float(single_rewards[sk])
                # measure — MIRROR: collect_cells cell protocol
                status, err_sig, reward = "ok", "", None
                try:
                    ok, err_sig = compile_trajectory_ex(
                        copy_dir, traj, filename, triple, arch=hp["arch"],
                        timeout=hp["compile_timeout"])
                except subprocess.TimeoutExpired:
                    ok, status, err_sig = False, "compile_timeout", "compile timeout"
                    reward = hp["compile_timeout_penalty"]
                if status == "ok" and not ok:
                    status, reward = "compile_failed", hp["compile_failure_penalty"]
                elif status == "ok":
                    try:
                        modified_ms = measure_kernel_time(
                            copy_dir, arch=hp["arch"], n_runs=hp["n_runs"],
                            nsys_timeout=hp["nsys_timeout"], tmp_dir=tmp_dir,
                            gpu_id=gpu_id, kernel_filter=kernel_filter)
                    except RuntimeError:
                        _log.warning("%s %s — MEASURE FAILED, not cached", bench, key)
                        return None                    # infrastructure, not the action
                    raw = (baseline_ms - modified_ms) / max(baseline_ms, 1e-9)
                    reward = max(raw, -1.0)
                    if dz > 0 and abs(reward) < dz:
                        reward = 0.0
                rewards[key] = float(reward)
                measured += 1
                result_q.put({"type": "cell", "key": key, "reward": float(reward),
                              "status": status, "error": err_sig})
                return float(reward)

            t0 = time.time()
            res = search_nest(chain, legal, value, dz, hp["sweeps_max"])

            # post-states: each loop at every context (actions below it) it was swept at
            n_post = 0
            if hp["collect_post_features"]:
                for k, ctxs in res["contexts"].items():
                    for below in ctxs:
                        if not below:
                            continue                   # no-op prefix: state == pre-features
                        pk = post_key(trajectory_key(bench, below), k)
                        if pk in hp["have_postf"]:
                            continue
                        vec = None
                        try:
                            vec = state_after(copy_dir, below, k, filename, triple,
                                              arch=hp["arch"], timeout=hp["compile_timeout"])
                        except Exception as e:
                            _log.warning("%s post-state %s: %s", bench, pk, e)
                        if vec is not None:
                            result_q.put({"type": "post", "key": pk, "vec": vec.tolist()})
                            n_post += 1

            result_q.put({
                "type": "nest_done", "benchmark": bench, "chain": chain, "error": "",
                "final": {str(k): list(c) for k, c in res["final"].items()},
                "final_value": res["final_value"],
                "greedy": {str(k): list(c) for k, c in res["greedy"].items()},
                "greedy_value": res["greedy_value"],
                "best_single": {str(k): list(c) for k, c in res["best_single"].items()},
                "best_single_value": res["best_single_value"],
                "single_best_per_loop": {str(k): max((v for (kk, _), v in res["singles"].items()
                                                      if kk == k and v is not None), default=0.0)
                                         for k in chain},
                "sweeps": res["sweeps"], "measured": measured, "post_states": n_post,
                "seconds": round(time.time() - t0, 1),
            })
            shutil.rmtree(copy_dir, ignore_errors=True)

        result_q.put({"type": "worker_done", "rank": rank})
    except Exception as e:
        _log.exception("worker crashed")
        result_q.put({"type": "worker_crashed", "rank": rank, "error": str(e)})


# ---------------------------------------------------------------------------
# Heuristic decisions (optional, for the summary)
# ---------------------------------------------------------------------------

def load_heuristic(csv_path, single_rewards: dict) -> dict:
    """{benchmark: {loop_idx: reward of the heuristic's cell (unmerge=1, factor)}}."""
    out: dict = {}
    if not csv_path or not Path(csv_path).exists():
        return out
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if not r.get("loop_idx") or not r.get("factor"):
                continue
            sk = f"{r['benchmark']}|{int(r['loop_idx'])}|1|{int(r['factor'])}"
            out.setdefault(r["benchmark"], {})[int(r["loop_idx"])] = single_rewards.get(sk)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint_dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--sweeps-max", type=int, default=4,
                   help="Sweeps per nest incl. the greedy ones (a pair: inner "
                        "singles, outer given inner, inner given outer, outer "
                        "given the new inner).  Default 4 => <= 54 cells/pair.")
    p.add_argument("--heuristic-csv", type=Path, default=None,
                   help="analysis/heuristic_decisions.py output, for the summary")
    p.add_argument("--no-post-features", action="store_true")
    # --- Must match the single-loop collection that built reward_cache.json ---
    p.add_argument("--compile-failure-penalty", type=float, default=None)
    p.add_argument("--reward-deadzone", type=float, default=None)
    p.add_argument("--compile-timeout-penalty", type=float, default=-1.0)
    p.add_argument("--compile-timeout", type=int, default=300)
    p.add_argument("--arch", default=ARCH)
    p.add_argument("--n-runs", type=int, default=None)
    p.add_argument("--nsys-timeout", type=int, default=300)
    p.add_argument("--tmp-dir", default=None)
    p.add_argument("--hecbench-src", default=None)
    p.add_argument("--save-every", type=int, default=10)
    args = p.parse_args()

    ckpt = args.checkpoint_dir
    elig_file = ckpt / "eligible_benchmarks.json"
    if not elig_file.exists():
        sys.exit(f"missing {elig_file}")
    data = json.loads(elig_file.read_text())
    loop_records_map = data["loop_records"]          # already deduped at save time
    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC

    # read-only inputs
    rc = ckpt / "reward_cache.json"
    single_rewards = {k: float(v) for k, v in
                      (json.loads(rc.read_text()).get("rewards", {}) if rc.exists() else {}).items()}
    bl = ckpt / "baseline_cache.json"
    baseline_cache = json.loads(bl.read_text()).get("baselines", {}) if bl.exists() else {}
    tc_file = ckpt / TRAJECTORY_CACHE_NAME
    tcache = load_trajectory_cache(tc_file)
    heur = load_heuristic(args.heuristic_csv, single_rewards)

    nests = find_nests(loop_records_map)
    runnable, skipped = [], []
    for n in nests:
        if baseline_cache.get(n["benchmark"]) is None:
            skipped.append((n["benchmark"], "no baseline")); continue
        if not (src / n["benchmark"]).is_dir():
            skipped.append((n["benchmark"], "no source dir")); continue
        runnable.append(n)
    runnable.sort(key=lambda n: baseline_cache[n["benchmark"]].get("total_ms", float("inf")))

    log.info("nests: %d found, %d runnable, %d skipped", len(nests), len(runnable), len(skipped))
    for b, why in skipped:
        log.info("  skip %s: %s", b, why)

    # the plan
    bound = 0
    missing_singles = 0
    for n in runnable:
        chain = [int(l["loop_idx"]) for l in n["loops"]]
        legal = {int(l["loop_idx"]): valid_cells(l["pre_features_raw"]) for l in n["loops"]}
        for k in chain:
            for (u, f) in legal[k]:
                if f"{n['benchmark']}|{k}|{u}|{f}" not in single_rewards:
                    missing_singles += 1
        per = max(len(v) for v in legal.values()) * max(args.sweeps_max - 1, 0)
        bound += per
        log.info("  %-32s chain=%s legal=%s  <= %d cells", n["benchmark"], chain,
                 [len(legal[k]) for k in chain], per)
    log.info("upper bound: %d measured cells (+%d single cells missing from reward_cache, "
             "measured on demand); %d already in %s",
             bound, missing_singles, len(tcache["rewards"]), tc_file.name)
    if args.dry_run:
        return

    for req in ("compile_failure_penalty", "reward_deadzone", "n_runs"):
        if getattr(args, req) is None:
            sys.exit(f"--{req.replace('_', '-')} is required for a real run (must match "
                     f"the single-loop collection)")

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else ckpt / "nests_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    hp = {
        "arch": args.arch, "n_runs": args.n_runs, "nsys_timeout": args.nsys_timeout,
        "tmp_dir": str(tmp_dir), "compile_timeout": args.compile_timeout,
        "compile_timeout_penalty": args.compile_timeout_penalty,
        "compile_failure_penalty": args.compile_failure_penalty,
        "reward_deadzone": args.reward_deadzone, "sweeps_max": args.sweeps_max,
        "traj_rewards": tcache["rewards"], "single_rewards": single_rewards,
        "baseline_cache": baseline_cache,
        "bench_paths": {n["benchmark"]: str(src / n["benchmark"]) for n in runnable},
        "collect_post_features": not args.no_post_features,
        "have_postf": set(tcache["post_features"].keys()),
    }

    work_q: "mp.Queue" = mp.Queue()
    for n in runnable:
        work_q.put(n)
    for _ in range(args.num_workers):
        work_q.put(None)
    result_q: "mp.Queue" = mp.Queue()
    procs = []
    for rank in range(args.num_workers):
        pr = mp.Process(target=_worker, args=(rank, rank, work_q, hp, result_q), daemon=True)
        pr.start(); procs.append(pr)
    log.info("launched %d workers over %d nests (fastest baseline first)", len(procs), len(runnable))

    rewards, postf = dict(tcache["rewards"]), dict(tcache["post_features"])
    failure_keys, timeout_keys = set(tcache["migration"].get("failure_keys", [])), \
        set(tcache["migration"].get("timeout_keys", []))
    summaries: list[dict] = []
    done = n_cells = 0
    by_status: dict = {}
    t0 = time.time()
    msg_timeout = args.n_runs * args.nsys_timeout + 600

    def flush():
        save_trajectory_cache(tc_file, rewards, postf, failure_keys, timeout_keys,
                              args.compile_failure_penalty, args.compile_timeout_penalty,
                              args.reward_deadzone)

    while done < len(procs):
        try:
            msg = result_q.get(timeout=msg_timeout)
        except queue.Empty:
            if not any(pr.is_alive() for pr in procs):
                log.error("all workers died"); break
            continue
        t = msg["type"]
        if t == "cell":
            rewards[msg["key"]] = msg["reward"]; n_cells += 1
            by_status[msg["status"]] = by_status.get(msg["status"], 0) + 1
            if msg["status"] == "compile_failed":
                failure_keys.add(msg["key"])
            elif msg["status"] == "compile_timeout":
                timeout_keys.add(msg["key"])
            if n_cells % args.save_every == 0:
                flush()
                rate = n_cells / max(time.time() - t0, 1e-9) * 3600
                log.info("progress: %d cells (%.0f/h), %d nests done", n_cells, rate, len(summaries))
        elif t == "post":
            postf[msg["key"]] = msg["vec"]
        elif t == "nest_done":
            summaries.append(msg)
            if msg.get("error"):
                log.warning("nest %s %s: %s", msg["benchmark"], msg["chain"], msg["error"])
            else:
                log.info("nest %s %s: final=%s %.4f | greedy %.4f | best single %.4f | "
                         "sweeps=%d measured=%d post=%d (%.0fs)",
                         msg["benchmark"], msg["chain"], msg["final"], msg["final_value"],
                         msg["greedy_value"], msg["best_single_value"], msg["sweeps"],
                         msg["measured"], msg["post_states"], msg["seconds"])
        elif t == "worker_crashed":
            log.error("worker %d crashed: %s", msg["rank"], msg.get("error")); done += 1
        elif t == "worker_done":
            done += 1
    for pr in procs:
        pr.join(timeout=30)
    flush()

    # summary: per nest, the three numbers the study reads
    dz = args.reward_deadzone
    rows = []
    for s in summaries:
        if s.get("error"):
            continue
        chain = s["chain"]
        h = heur.get(s["benchmark"], {})
        h_cells = [h[k] for k in chain if k in h and h[k] is not None]
        h_val = h_cells[0] if h_cells else 0.0        # heuristic fired on <=1 loop per chain
        inner_best = s["single_best_per_loop"][str(chain[0])]
        outer_best = max(s["single_best_per_loop"][str(k)] for k in chain[1:])
        rows.append({
            "benchmark": s["benchmark"], "chain": chain,
            "best_single": s["best_single_value"], "heuristic": h_val,
            "greedy": s["greedy_value"], "final": s["final_value"],
            "pair_gain": s["final_value"] - s["best_single_value"],
            "outer_minus_inner_single": outer_best - inner_best,
            "composition": int(s["final_value"] > s["best_single_value"] + dz and len(s["final"]) >= 2),
            "wrong_loop": int(outer_best > inner_best + dz),
            "sweeps": s["sweeps"], "measured": s["measured"],
        })
    out = ckpt / "nests_summary.json"
    out.write_text(json.dumps(rows, indent=1))
    log.info("")
    log.info("=== done in %.1f h: %d cells by status %s; %d nests summarised -> %s ===",
             (time.time() - t0) / 3600, n_cells, dict(sorted(by_status.items())), len(rows), out)
    if rows:
        log.info("composition beats best single (B): %d/%d   outer single beats inner single (A): %d/%d",
                 sum(r["composition"] for r in rows), len(rows),
                 sum(r["wrong_loop"] for r in rows), len(rows))


if __name__ == "__main__":
    main()
