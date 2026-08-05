"""
Turn the measured reward table into per-loop ground-truth labels.

Produces the artifact every downstream analysis reads: for each eligible loop,
which of the three actions is correct, how much it is worth, and how confident
that label is.

    no-op           declining is best (no transform beats it by more than the deadzone)
    unroll-only     best action is (unmerge=0, factor>1)
    unmerge+unroll  best action is (unmerge=1, factor>=1)

The label is a property of the LOOP, not of any train/test split — nothing here
depends on a seed. Splits are applied downstream when scoring a policy.

LABEL RULES (fixed a priori — see the guardrail in study_plan.md)
----------------------------------------------------------------
* The no-op cell (0,1) is free, exactly 0.0, and never stored in the cache. It
  is injected here, so a loop whose every transform is harmful is correctly
  labelled no-op rather than "least bad transform".
* A transform must beat 0.0 by more than --deadzone to win the label. Below that
  it is measurement noise, not a speedup.
* Ties go to the SIMPLER action (no-op > unroll-only > unmerge+unroll). Fixed
  rule, applied identically everywhere, so no outcome influences the labelling.

CONFIDENCE
----------
Two margins, because two different things can be fragile:
  category_margin  best category minus runner-up category. Small = the LABEL is
                   one re-measurement away from flipping. No-op is always in the
                   running (free, exactly 0.0), so whenever a transform wins this
                   is at most its lead over declining. It is BLANK only when
                   no-op wins and every transform is below the deadzone — i.e.
                   declining is unambiguously correct.
  cell_margin      best cell minus second-best cell inside the winning category.
                   Small = the label is safe but the FACTOR choice is arbitrary.
Loops with category_margin <= --ambiguity are flagged; downstream reports them
as their own bucket rather than counting them as classification errors.

Loops whose cells were never measured are emitted with category=unknown and
labelable=0 — never silently defaulted to no-op, which would inflate the
majority class with missing data.

Usage:
  python3 label_loops.py RUN_DIR --deadzone 0.005 --out loop_labels.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Mirrored from agent.py — kept here so this whole directory stays importable
# without torch. If FACTOR_VALUES or the trip-count indices change there, change
# them here; the two are asserted against the cache's own key range at runtime.
FACTOR_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_IDX_TRIP_COUNT_KNOWN = 10
_IDX_TRIP_COUNT = 11

NOOP = (0, 1)
CATEGORIES = ("noop", "unroll_only", "unmerge_unroll")


def valid_factors(raw: list) -> list:
    """
    Factors legal for this loop under its trip count.
    MIRROR: agent.build_factor_mask — factor 1 is always legal; others must not
    exceed a KNOWN trip count. Raw (un-normalised) values only.
    """
    if raw[_IDX_TRIP_COUNT_KNOWN] > 0.5 and int(raw[_IDX_TRIP_COUNT]) > 0:
        tc = int(raw[_IDX_TRIP_COUNT])
        return [f for f in FACTOR_VALUES if f == 1 or f <= tc]
    return list(FACTOR_VALUES)


def label_loop(raw: list, cells: dict, deadzone: float, ambiguity: float,
               failure_keys: set, key_prefix: str) -> dict:
    """Label one loop from its measured cells. `cells` is {(unmerge,factor): reward}."""
    factors = valid_factors(raw)
    valid = [(u, f) for u in (0, 1) for f in factors if (u, f) != NOOP]
    measured = {a: r for a, r in cells.items() if a in set(valid)}

    n_failed = sum(1 for a in measured
                   if f"{key_prefix}|{a[0]}|{a[1]}" in failure_keys)
    n_wins = sum(1 for r in measured.values() if r > deadzone)

    unroll = {a: r for a, r in measured.items() if a[0] == 0}
    unmerge = {a: r for a, r in measured.items() if a[0] == 1}

    def best(d):
        if not d:
            return None, None
        a = max(d, key=lambda k: d[k])
        return a, d[a]

    a_ur, r_ur = best(unroll)
    a_um, r_um = best(unmerge)

    row = {
        "benchmark": key_prefix.split("|")[0],
        "loop_idx": int(key_prefix.split("|")[1]),
        "n_valid": len(valid),
        "n_measured": len(measured),
        "fill": round(len(measured) / len(valid), 4) if valid else 0.0,
        "n_failed": n_failed,
        "n_wins": n_wins,
        "best_unroll": round(r_ur, 6) if r_ur is not None else "",
        "best_unroll_factor": a_ur[1] if a_ur else "",
        "best_unmerge": round(r_um, 6) if r_um is not None else "",
        "best_unmerge_factor": a_um[1] if a_um else "",
    }

    if not measured:
        # No data at all — must not default to no-op, which would pad the
        # majority class with missing measurements.
        row.update({"oracle_reward": "", "category": "unknown", "runner_up": "",
                    "category_margin": "", "cell_margin": "", "ambiguous": "",
                    "labelable": 0})
        return row

    # Category scores. An unmeasured category scores -inf so it can never win,
    # and the `fill` column above is what flags a loop where that matters.
    scores = {
        "noop": 0.0,
        "unroll_only": r_ur if r_ur is not None else float("-inf"),
        "unmerge_unroll": r_um if r_um is not None else float("-inf"),
    }
    # A transform must clear the deadzone to take the label from no-op.
    for k in ("unroll_only", "unmerge_unroll"):
        if scores[k] <= deadzone:
            scores[k] = float("-inf")

    # Tie-break by simplicity: CATEGORIES is ordered noop, unroll_only,
    # unmerge_unroll, and Python's max keeps the FIRST maximum.
    category = max(CATEGORIES, key=lambda c: scores[c])
    ordered = sorted((scores[c] for c in CATEGORIES), reverse=True)
    runner_up = max((c for c in CATEGORIES if c != category),
                    key=lambda c: scores[c])
    margin = ordered[0] - ordered[1] if ordered[1] != float("-inf") else float("inf")

    # Factor-choice fragility inside the winning category.
    winner_cells = (unroll if category == "unroll_only"
                    else unmerge if category == "unmerge_unroll" else {})
    vals = sorted(winner_cells.values(), reverse=True)
    cell_margin = round(vals[0] - vals[1], 6) if len(vals) >= 2 else ""

    row.update({
        "oracle_reward": round(scores[category], 6),
        "category": category,
        "runner_up": runner_up,
        "category_margin": (round(margin, 6) if margin != float("inf") else ""),
        "cell_margin": cell_margin,
        "ambiguous": int(margin <= ambiguity),
        "labelable": 1,
    })
    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deadzone", type=float, required=True,
                   help="REQUIRED: must match the run that built the cache "
                        "(0.005 for run_sweep_1). A transform must beat 0.0 by "
                        "more than this to take the label from no-op.")
    p.add_argument("--ambiguity", type=float, default=0.02,
                   help="Loops whose winning category leads the runner-up by at "
                        "most this are flagged ambiguous — the label is within "
                        "measurement noise of flipping. (default: 0.02)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV (default: RUN_DIR/loop_labels.csv)")
    args = p.parse_args()

    elig_f = args.run_dir / "eligible_benchmarks.json"
    rc_f = args.run_dir / "reward_cache.json"
    for f in (elig_f, rc_f):
        if not f.exists():
            sys.exit(f"missing {f}")

    elig = json.loads(elig_f.read_text())
    records = elig.get("loop_records", {})
    eligible = set(elig.get("eligible", []))
    cache = json.loads(rc_f.read_text())
    rewards = cache.get("rewards", {})
    failure_keys = set((cache.get("migration") or {}).get("failure_keys", []))

    # Group cells by loop, ignoring anything outside the eligible set — the cache
    # keeps cells for benchmarks pruned from eligibility, and counting them would
    # inflate every denominator.
    by_loop: dict = {}
    skipped = 0
    for k, v in rewards.items():
        parts = k.split("|")
        if len(parts) != 4:
            continue
        bench, li, u, f = parts
        if bench not in eligible:
            skipped += 1
            continue
        try:
            by_loop.setdefault(f"{bench}|{int(li)}", {})[(int(u), int(f))] = float(v)
        except ValueError:
            continue

    rows = []
    for bench in sorted(eligible):
        for rec in records.get(bench, []):
            key = f"{bench}|{int(rec['loop_idx'])}"
            rows.append(label_loop(rec["pre_features_raw"], by_loop.get(key, {}),
                                   args.deadzone, args.ambiguity,
                                   failure_keys, key))

    if not rows:
        sys.exit("no eligible loops found — check eligible_benchmarks.json")
    out = args.out or (args.run_dir / "loop_labels.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lab = [r for r in rows if r["labelable"]]
    counts = {c: sum(1 for r in lab if r["category"] == c) for c in CATEGORIES}
    n = max(len(lab), 1)
    print(f"loops                 : {len(rows)} eligible, {len(lab)} labelable, "
          f"{len(rows) - len(lab)} with no measured cells")
    if skipped:
        print(f"cells ignored         : {skipped} (benchmarks not in the "
              f"eligible set — pruned)")
    print(f"mean fill             : "
          f"{sum(r['fill'] for r in lab) / n:.3f}")
    print()
    print("category distribution (the 'is UU universally useful' answer):")
    for c in CATEGORIES:
        print(f"  {c:<16} {counts[c]:4d}  ({100 * counts[c] / n:5.1f}%)")
    amb = sum(1 for r in lab if r["ambiguous"])
    print(f"\nambiguous (margin <= {args.ambiguity}): {amb} ({100 * amb / n:.1f}%)")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
