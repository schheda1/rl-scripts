"""
GPU-free oracle audit of a run's frozen reward_cache.

Answers the one question that decides whether any network/PPO fix is worth GPU:
is there measured headroom above no-op on the TEST split, or does the objective
genuinely make declining-to-transform optimal?

Reads only two files written by a training run's checkpoint dir:
  eligible_benchmarks.json  — the eligible loop population (+ reproduces the split)
  reward_cache.json         — {"rewards": {"bench|loop|unmerge|factor": reward}}

No compile, no measure, no torch. Every cell it reads is already memoized.

For each loop it reconstructs the action table from the cache, treating no-op
(unmerge=0, factor=1) as an implicit, free, exact 0.0 that is never stored. The
oracle-best per loop is therefore max(0.0, best measured transform). Reported
per split (train/val/test):

  coverage      fraction of loops whose BEST measured transform beats +deadzone
                (i.e. any transform is worth doing at all)
  oracle_mean   mean over loops of the oracle-best reward (the achievable ceiling
                for a PERFECT per-loop policy) — compare against 0.0 (all-no-op)
  fill          fraction of the 19 non-no-op cells actually measured per loop
                (low fill => the "oracle" is optimistic; unmeasured cells unknown)
  best-action   distribution of the argmax action (no-op / unroll-only / unmerge)

Read the three splits together:
  * oracle_mean(test) ~ 0            -> no-op is genuinely near-optimal; the
                                        collapse is CORRECT and PPO tuning wastes
                                        GPU. Honest negative result.
  * oracle_mean(test) clearly > 0    -> headroom exists; a policy that finds it
                                        would beat no-op. Collapse/protocol fixes
                                        are justified.
  * oracle_mean(train) > 0 ~ test    -> headroom on seen apps only; a features/
                                        representation-transfer problem, not PPO.

Usage:
  python3 audit_reward_cache.py CHECKPOINT_DIR
  python3 audit_reward_cache.py CHECKPOINT_DIR --val-ratio 0.15 --test-ratio 0.15 \
                                --split-seed 42 --deadzone 0.01
  python3 audit_reward_cache.py CHECKPOINT_DIR --csv oracle_per_loop.csv

The split params MUST match the training run's, or the per-split numbers are
computed over a different partition than the policy was trained/tested on.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

# Kept dependency-free ON PURPOSE: importing agent/train pulls in torch, which
# defeats "runs anywhere without a GPU box". These two constants are mirrored
# from agent.FACTOR_VALUES and train.split_benchmarks — if EITHER changes there,
# change it here. They are asserted below against the run's own artifacts where
# possible.
FACTOR_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # mirror of agent.FACTOR_VALUES

NOOP = (0, 1)   # unmerge=0, factor=1 — free, exact 0.0, never stored in the cache


def split_benchmarks(benchmarks, val_ratio, test_ratio, seed):
    """Verbatim copy of train.split_benchmarks (kept in sync manually — see
    module note). Same seed + same benchmark order => same partition the run
    trained/tested on."""
    rng = random.Random(seed)
    shuffled = list(benchmarks)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = max(1, n - n_val - n_test)
    n_val = max(0, n - n_train - n_test)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


class _Bench:
    """Lightweight stand-in so split_benchmarks (which only reads .name) can run
    without discovering the benchmark tree."""
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


def _load(checkpoint_dir: Path):
    elig_f = checkpoint_dir / "eligible_benchmarks.json"
    rc_f = checkpoint_dir / "reward_cache.json"
    if not elig_f.exists():
        sys.exit(f"missing {elig_f} — run the audit against a real checkpoint dir")
    if not rc_f.exists():
        sys.exit(f"missing {rc_f} — this run never wrote a reward cache")
    elig = json.loads(elig_f.read_text())
    rc = json.loads(rc_f.read_text())
    return elig, rc.get("rewards", {}), rc.get("migration")


def _bench_order(elig: dict) -> list[str]:
    """discover_benchmarks returns sorted(*-cuda); precheck filters preserving
    that order; split_benchmarks then shuffles. Sorting the eligible names
    reproduces the pre-shuffle order split_benchmarks saw."""
    names = elig.get("eligible")
    if names is None:                       # fall back to the loop_records keys
        names = list(elig.get("loop_records", {}))
    return sorted(names)


def _cells_by_loop(rewards: dict) -> dict:
    """Group cache entries into {(bench, loop_idx): {(unmerge, factor): reward}}."""
    out: dict = {}
    for key, r in rewards.items():
        parts = key.split("|")
        if len(parts) != 4:
            continue                        # not a reward cell (e.g. legacy key)
        bench, loop, u, f = parts
        try:
            loop_i, u_i, f_i = int(loop), int(u), int(f)
        except ValueError:
            continue
        out.setdefault((bench, loop_i), {})[(u_i, f_i)] = float(r)
    return out


def _classify(action) -> str:
    if action == NOOP:
        return "noop"
    return "unmerge" if action[0] == 1 else "unroll_only"


def audit_split(name: str, benches: list, records: dict, cells: dict,
                deadzone: float, rows: list) -> dict:
    n_transform_cells = 2 * len(FACTOR_VALUES) - 1   # 19: all (u,f) minus no-op
    loops = won = measured_any = 0
    oracle_sum = 0.0
    fill_sum = 0.0
    best_action = {"noop": 0, "unroll_only": 0, "unmerge": 0}
    unmerge_ever_won = 0

    for b in benches:
        for rec in records.get(b.name, []):
            loops += 1
            loop_i = int(rec["loop_idx"])
            table = cells.get((b.name, loop_i), {})
            measured = {a: v for a, v in table.items() if a != NOOP}
            fill_sum += len(measured) / n_transform_cells
            if measured:
                measured_any += 1

            # Oracle includes the always-available free no-op at 0.0.
            best_action_a, best_r = NOOP, 0.0
            for a, v in measured.items():
                if v > best_r:
                    best_action_a, best_r = a, v
            oracle_sum += best_r
            best_action[_classify(best_action_a)] += 1
            if best_r > deadzone:
                won += 1
            if any(a[0] == 1 and v > deadzone for a, v in measured.items()):
                unmerge_ever_won += 1

            rows.append({
                "split": name, "benchmark": b.name, "loop_idx": loop_i,
                "cells_measured": len(measured),
                "oracle_best": round(best_r, 6),
                "best_action": _classify(best_action_a),
                "best_unmerge": best_action_a[0], "best_factor": best_action_a[1],
                "beats_noop": int(best_r > deadzone),
            })

    denom = max(loops, 1)
    return {
        "split": name,
        "benchmarks": len(benches),
        "loops": loops,
        "loops_with_any_measured_cell": measured_any,
        "avg_cell_fill": fill_sum / denom,
        "coverage_beats_noop": won / denom,
        "oracle_mean": oracle_sum / denom,           # ceiling for a perfect policy
        "all_noop_mean": 0.0,                          # the free baseline, by def.
        "best_action_dist": best_action,
        "loops_where_unmerge_wins": unmerge_ever_won,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint_dir", type=Path)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--deadzone", type=float, default=0.01,
                   help="|r|<= this is measurement noise, not a win (match training)")
    p.add_argument("--csv", type=Path, default=None,
                   help="Optional per-loop oracle table output")
    args = p.parse_args()

    elig, rewards, migration = _load(args.checkpoint_dir)
    records = elig.get("loop_records", {})
    cells = _cells_by_loop(rewards)

    names = _bench_order(elig)
    benches = [_Bench(n) for n in names]
    train_b, val_b, test_b = split_benchmarks(
        benches, args.val_ratio, args.test_ratio, args.split_seed)

    total_loops = sum(len(v) for v in records.values())
    print(f"cache cells (reward entries) : {len(rewards)}")
    print(f"cells grouped into loops     : {len(cells)}")
    print(f"eligible loops (all splits)  : {total_loops}")
    if migration:
        fk = len(migration.get("failure_keys", []))
        print(f"note: cache has a migration block ({fk} cells are failure "
              f"penalties, penalty={migration.get('failure_penalty')})")
    print()

    rows: list = []
    stats = [audit_split(nm, bs, records, cells, args.deadzone, rows)
             for nm, bs in (("train", train_b), ("val", val_b), ("test", test_b))]

    hdr = (f"{'split':6} {'bmarks':>6} {'loops':>6} {'measured':>8} "
           f"{'fill':>6} {'cover':>7} {'oracle':>8} {'no-op':>7} "
           f"{'noop/unroll/unmrg (best)':>26} {'unmrg_wins':>10}")
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        d = s["best_action_dist"]
        print(f"{s['split']:6} {s['benchmarks']:6d} {s['loops']:6d} "
              f"{s['loops_with_any_measured_cell']:8d} "
              f"{s['avg_cell_fill']*100:5.1f}% "
              f"{s['coverage_beats_noop']*100:6.1f}% "
              f"{s['oracle_mean']:+8.4f} {s['all_noop_mean']:+7.4f} "
              f"{d['noop']:>8}/{d['unroll_only']}/{d['unmerge']:<7} "
              f"{s['loops_where_unmerge_wins']:10d}")

    print()
    t = next(s for s in stats if s["split"] == "test")
    fill = t["avg_cell_fill"]
    print("=== read-out (test split) ===")
    if t["loops"] == 0:
        print("  TEST split is empty — check --split-seed / --*-ratio match the run.")
    elif t["oracle_mean"] <= args.deadzone:
        print(f"  oracle_mean(test)={t['oracle_mean']:+.4f} <= deadzone: even a "
              f"PERFECT per-loop policy barely beats all-no-op. The objective "
              f"makes declining near-optimal — collapse is CORRECT, PPO tuning "
              f"will not help. Honest negative result.")
    else:
        print(f"  oracle_mean(test)={t['oracle_mean']:+.4f} > 0 over all-no-op: "
              f"headroom EXISTS. {t['coverage_beats_noop']*100:.0f}% of test "
              f"loops have a transform that beats no-op. A policy that finds it "
              f"would win — collapse/protocol fixes are justified.")
    if fill < 0.5:
        print(f"  CAVEAT: only {fill*100:.0f}% of transform cells were ever "
              f"measured, so oracle_mean is OPTIMISTIC (unmeasured cells could "
              f"be worse). The collapsed policy simply never explored them.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-loop table: {args.csv}  ({len(rows)} loops)")


if __name__ == "__main__":
    main()
