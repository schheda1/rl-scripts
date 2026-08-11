"""
Derive the per-cell OUTCOME REGIME from a reward cache, as a read-only sidecar.

WHY THIS EXISTS
---------------
`reward_cache.json` stores one float per cell, and that float conflates three
outcomes that mean different things to a policy:

  ok        the transform built and ran; the number is a measured delta.
  failed    it did not build. The compiler keeps the original, so the program is
            UNCHANGED and the realised effect is exactly 0.0. The stored value
            is --compile-failure-penalty, a number someone chose.
  censored  the value sits at the -1.0 clip. train.py:1443 clips every measured
            reward there AND --compile-timeout-penalty defaults to it, so the
            cell is either a timeout (nothing shipped) or a slowdown of 100% or
            worse. The cache cannot say which; `is_timeout` never reaches it.

The live pipeline KNOWS the regime at measurement time — environment.py:118-119
names all five statuses and sets them at :335/:341/:364/:383/:408, the worker
mirrors that at train.py:1344/:1351/:1393/:1405, and _send_loop_result puts
"status" and "timeout" on the queue. Then train.py:2057 and :2237 write
`reward_cache[key] = msg["reward"]` and the label is discarded. This script
reconstructs what it can from what survived.

GOLD IS NEVER WRITTEN
---------------------
The cache is opened read-only and a separate sidecar is emitted. An md5 of the
`rewards` map goes into the sidecar so a consumer can tell it has gone stale
against a cache that has since grown.

STRICT BY DEFAULT
-----------------
offline_data.py identifies failures by OR-ing two tests: membership in
`migration.failure_keys`, and a value match against FAILURE_VALUES. It notes in
a comment that the two agree exactly on the current cache. That is an invariant,
not a guarantee — a genuine measurement landing on -0.161 would be silently
relabelled a failure, and every reported performance figure depends on it. Here
the agreement is CHECKED and a divergence is a hard error unless
--allow-divergence is passed.

Usage:
  python3 cell_status.py RUN_DIR
  python3 cell_status.py RUN_DIR --out cell_status.json --allow-divergence
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# MIRRORED from offline_data.py, which cannot be imported here: its import chain
# reaches adapt_eval -> torch, and this script is meant to run on any box. If
# either constant changes there, change it here. test_hurdle.py asserts the two
# copies agree, so the drift fails a test rather than a paper.
FAILURE_VALUES = (-0.16, -0.161)
CLIP_FLOOR = -1.0

# Sidecar stores only the non-`ok` cells; `ok` is the lookup default. On the
# current cache that is ~3.4k entries instead of 8.4k.
OK = "ok"
FAILED = "failed"
CENSORED = "censored"


def is_failure_value(v: float) -> bool:
    """True when a stored value is one of the compile-failure penalties.

    Exact-ish equality, not a threshold: these are two specific constants the
    collection wrote, and a range test would swallow genuine measurements that
    happen to be nearby.
    """
    return any(abs(v - x) < 1e-9 for x in FAILURE_VALUES)


def is_clip_floor(v: float) -> bool:
    return v <= CLIP_FLOOR + 1e-9


def parse_key(key: str):
    """('bench', loop, unmerge, factor) or None if the key is not a reward cell."""
    parts = key.split("|")
    if len(parts) != 4:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None


def rewards_md5(rewards: dict) -> str:
    return hashlib.md5(
        json.dumps(rewards, sort_keys=True).encode()).hexdigest()[:12]


def derive(rewards: dict, failure_keys: "set | None") -> tuple:
    """
    (status_map, counts, divergence) over the cells present in `rewards`.

    status_map holds only non-`ok` cells. divergence is
    {"keys_not_value_matched": [...], "value_matched_not_in_keys": [...]} and is
    empty when the two failure tests agree — see the module note on why that is
    checked rather than assumed.
    """
    value_matched, floor_cells = set(), set()
    for key, v in rewards.items():
        if parse_key(key) is None:
            continue
        v = float(v)
        if is_failure_value(v):
            value_matched.add(key)
        elif is_clip_floor(v):
            floor_cells.add(key)

    divergence = {}
    if failure_keys is not None:
        # Restrict to cells that actually exist: a key list can outlive the
        # cells it names (a re-collection under a narrower eligibility filter),
        # and those absences are not disagreements about anything.
        present = {k for k in failure_keys if k in rewards}
        only_keys = sorted(present - value_matched)
        only_values = sorted(value_matched - present)
        if only_keys:
            divergence["keys_not_value_matched"] = only_keys
        if only_values:
            divergence["value_matched_not_in_keys"] = only_values
        failed = present | value_matched
    else:
        failed = value_matched

    # A cell cannot be both: `failed` wins, and by construction it cannot also
    # be at the floor, since the failure penalties are far above -1.0.
    status = {k: FAILED for k in failed}
    for k in floor_cells:
        if k not in status:
            status[k] = CENSORED

    n_cells = sum(1 for k in rewards if parse_key(k) is not None)
    counts = {
        "cells": n_cells,
        FAILED: len(failed),
        CENSORED: sum(1 for v in status.values() if v == CENSORED),
        OK: n_cells - len(status),
    }
    return status, counts, divergence


def load_status(path: Path) -> dict:
    """Read a sidecar back. Consumers should use `status_of` rather than
    indexing, so the `ok` default stays in one place."""
    return json.loads(path.read_text()).get("status", {})


def status_of(status_map: dict, bench: str, loop_idx: int,
              unmerge: int, factor: int) -> str:
    """Regime for one cell. Absent from the map means `ok` — the sidecar stores
    only the exceptions. A cell absent from the CACHE is a different thing and
    is the caller's business (hurdle_run.observe handles it)."""
    return status_map.get(f"{bench}|{loop_idx}|{unmerge}|{factor}", OK)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Sidecar path (default: RUN_DIR/cell_status.json)")
    p.add_argument("--allow-divergence", action="store_true",
                   help="Proceed when failure_keys and the value test disagree, "
                        "taking their UNION — which is offline_data.py's current "
                        "behaviour. Off by default because the disagreement is "
                        "the interesting event.")
    args = p.parse_args()

    rc_file = args.run_dir / "reward_cache.json"
    if not rc_file.exists():
        sys.exit(f"missing {rc_file}")
    rc = json.loads(rc_file.read_text())          # read-only; never written back
    rewards = rc.get("rewards", {})
    if not rewards:
        sys.exit(f"{rc_file} holds no rewards")
    migration = rc.get("migration") or {}
    fkeys = set(migration.get("failure_keys", [])) if migration else None

    status, counts, divergence = derive(rewards, fkeys)

    print(f"cells               : {counts['cells']}")
    print(f"  ok                : {counts[OK]}")
    print(f"  failed            : {counts[FAILED]}")
    print(f"  censored (at clip): {counts[CENSORED]}")
    if fkeys is None:
        print("\nNOTE: no migration block — failures identified by VALUE alone. "
              "A genuine\nmeasurement at -0.16/-0.161 would be misread as a "
              "failure and reported as 0.0.")

    if divergence:
        for name, keys in divergence.items():
            print(f"\nDIVERGENCE {name}: {len(keys)} cell(s)")
            for k in keys[:5]:
                print(f"    {k}  value={rewards[k]}")
        if not args.allow_divergence:
            sys.exit(
                "\nfailure_keys and the value test disagree. offline_data.py "
                "ORs them silently;\nthis refuses. Either the failure penalty "
                "was retuned without a migration entry,\nor a real measurement "
                "collided with one of FAILURE_VALUES. Re-run with\n"
                "--allow-divergence to take the union anyway.")
        print("\n--allow-divergence: taking the union.")

    out = args.out or (args.run_dir / "cell_status.json")
    out.write_text(json.dumps({
        "source_md5": rewards_md5(rewards),
        "failure_values": list(FAILURE_VALUES),
        "clip_floor": CLIP_FLOOR,
        "had_migration_block": fkeys is not None,
        "divergence_allowed": bool(divergence) and args.allow_divergence,
        "counts": counts,
        "status": status,
    }))
    print(f"\nsidecar: {out}  ({len(status)} non-ok cells)")
    print(f"gold cache untouched: {rc_file}")


if __name__ == "__main__":
    main()
