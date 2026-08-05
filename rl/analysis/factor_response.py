"""
How much does the UNROLL FACTOR actually matter, given the right category?

The labelled table answers "which action", but its cell_margin only compares the
best factor against the runner-up — which says nothing about the other eight. A
loop whose factors 4/5/6 all reach +0.20 while the rest sit near zero has a tiny
cell margin AND a factor choice that matters enormously. This script measures
the whole response curve instead, per unmerge arm, and reports:

  * how many factors land within T of the best (the width of the good region)
  * the spread between the best and worst factor (what a bad pick costs)
  * which factors win, and how each factor performs on average

Read together those separate two claims that are easy to conflate:
    "unrolling is the right decision here"        <- category, carries the value
    "THIS factor is the right one"                <- only matters if the region is narrow

Emits a tidy per-cell CSV ready for plotting the ablation figure: one row per
(loop, arm, factor) with its distance from that arm's best.

Usage:
  python3 factor_response.py RUN_DIR --deadzone 0.005
  python3 factor_response.py RUN_DIR --deadzone 0.005 --bands 0.01 0.02 0.05
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

FACTOR_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # MIRROR: agent.FACTOR_VALUES
_IDX_TRIP_COUNT_KNOWN = 10
_IDX_TRIP_COUNT = 11
ARM = {0: "unroll_only", 1: "unmerge_unroll"}


def valid_factors(raw: list) -> list:
    """MIRROR: agent.build_factor_mask (see label_loops.valid_factors)."""
    if raw[_IDX_TRIP_COUNT_KNOWN] > 0.5 and int(raw[_IDX_TRIP_COUNT]) > 0:
        tc = int(raw[_IDX_TRIP_COUNT])
        return [f for f in FACTOR_VALUES if f == 1 or f <= tc]
    return list(FACTOR_VALUES)


def pct(x, n):
    return f"{100 * x / n:5.1f}%" if n else "    - "


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deadzone", type=float, required=True,
                   help="REQUIRED: must match the cache. Also the tightest "
                        "'within T of best' band — two factors closer than this "
                        "are indistinguishable by measurement.")
    p.add_argument("--bands", type=float, nargs="+", default=[0.02, 0.05],
                   help="Extra 'within T of best' bands (default: 0.02 0.05)")
    p.add_argument("--min-factors", type=int, default=4,
                   help="Skip arms with fewer than this many measured factors — "
                        "a 2-factor arm cannot say anything about curve shape. "
                        "(default: 4)")
    p.add_argument("--out", type=Path, default=None,
                   help="Per-cell CSV (default: RUN_DIR/factor_curves.csv)")
    args = p.parse_args()

    elig_f = args.run_dir / "eligible_benchmarks.json"
    rc_f = args.run_dir / "reward_cache.json"
    for f in (elig_f, rc_f):
        if not f.exists():
            sys.exit(f"missing {f}")
    elig = json.loads(elig_f.read_text())
    eligible = set(elig.get("eligible", []))
    records = elig.get("loop_records", {})
    rewards = json.loads(rc_f.read_text()).get("rewards", {})

    cells: dict = {}
    for k, v in rewards.items():
        parts = k.split("|")
        if len(parts) != 4 or parts[0] not in eligible:
            continue
        try:
            cells.setdefault((parts[0], int(parts[1]), int(parts[2])),
                             {})[int(parts[3])] = float(v)
        except ValueError:
            continue

    bands = sorted({args.deadzone, *args.bands})
    rows, arms = [], {0: [], 1: []}

    for bench in sorted(eligible):
        for rec in records.get(bench, []):
            li = int(rec["loop_idx"])
            legal = set(valid_factors(rec["pre_features_raw"]))
            for u in (0, 1):
                got = {f: r for f, r in cells.get((bench, li, u), {}).items()
                       if f in legal}
                # factor=1 on the unroll arm IS the no-op, never stored; inject
                # it so the arm's curve is complete and comparable to arm 1.
                if u == 0:
                    got.setdefault(1, 0.0)
                if len(got) < args.min_factors:
                    continue
                best_f = max(got, key=lambda f: got[f])
                best_r = got[best_f]
                worst_r = min(got.values())
                for f, r in sorted(got.items()):
                    rows.append({
                        "benchmark": bench, "loop_idx": li, "arm": ARM[u],
                        "factor": f, "reward": round(r, 6),
                        "delta_from_best": round(best_r - r, 6),
                        "is_best": int(f == best_f),
                    })
                arms[u].append({
                    "n": len(got), "best_f": best_f, "best_r": best_r,
                    "spread": best_r - worst_r,
                    "within": {b: sum(1 for r in got.values() if best_r - r <= b)
                               for b in bands},
                    "helps": best_r > args.deadzone,
                    "per_factor": {f: best_r - r for f, r in got.items()},
                })

    if not rows:
        sys.exit("no arms with enough measured factors")
    out = args.out or (args.run_dir / "factor_curves.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    for u in (0, 1):
        data = [a for a in arms[u] if a["helps"]]
        print("=" * 74)
        print(f"  {ARM[u]}  —  {len(data)} arms where the best factor beats no-op "
              f"(of {len(arms[u])} measured)")
        print("=" * 74)
        if not data:
            print("  none\n"); continue

        print("\n  WIDTH OF THE GOOD REGION — factors within T of the best")
        print(f"  {'T':>8} {'median':>8} {'mean':>7} {'of median':>11}   "
              f"{'arms where ONLY the best is within T':>38}")
        for b in bands:
            w_ = [a["within"][b] for a in data]
            n_ = [a["n"] for a in data]
            only = sum(1 for a in data if a["within"][b] == 1)
            print(f"  {b:>8g} {st.median(w_):>8.1f} {st.mean(w_):>7.2f} "
                  f"{st.median(n_):>11.0f}   {only:>10d}  {pct(only, len(data))}")
        print("\n  Read: 'median 3 of 10' means a policy has 3 equally good choices;\n"
              "  the last column counts arms where the factor genuinely is unique.")

        sp = sorted(a["spread"] for a in data)
        print(f"\n  COST OF A BAD PICK — best minus worst factor")
        print(f"    median {st.median(sp):+.4f}   mean {st.mean(sp):+.4f}   "
              f"p90 {sp[int(0.9 * len(sp)) - 1]:+.4f}   max {sp[-1]:+.4f}")
        big = sum(1 for s in sp if s > 0.05)
        print(f"    arms where the spread exceeds 0.05: {big}  {pct(big, len(data))}"
              f"   <- here the factor DOES matter")

        print(f"\n  PER-FACTOR BEHAVIOUR")
        print(f"  {'factor':>7} {'n arms':>7} {'wins':>7} {'mean gap':>10} "
              f"{'within dz of best':>18}")
        for f in FACTOR_VALUES:
            present = [a for a in data if f in a["per_factor"]]
            if not present:
                continue
            gaps = [a["per_factor"][f] for a in present]
            wins = sum(1 for a in present if a["best_f"] == f)
            near = sum(1 for g in gaps if g <= args.deadzone)
            print(f"  {f:>7} {len(present):>7} {wins:>4} {pct(wins, len(present))}"
                  f" {st.mean(gaps):>10.4f} {near:>8} {pct(near, len(present))}")
        print()

    print(f"Per-cell curves: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
