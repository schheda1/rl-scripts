"""
Re-encode a reward_cache.json for the compile-failure penalty / reward deadzone.

WHY the first migration is safe
-------------------------------
Every cached reward of exactly 0.0 is a compile or measurement FAILURE, not a
neutral result:

    reward = (baseline_ms - modified_ms) / baseline_ms

can only be *exactly* 0.0 when the code assigned `modified_ms = baseline_ms`,
which happens solely in the failure fallback paths.  No-ops are never written to
the cache, and a real measurement will not produce bit-identical values.  So
those cells can be relabelled in place — no re-measurement, no GPU time.

WHY IT STAYS REVERSIBLE
-----------------------
Once 0.0 is rewritten to a penalty, the cells are no longer identifiable by
value.  This script therefore records the exact key lists it touched under
`"migration"` in the file, so the penalty can be raised, lowered, or reverted
any number of times without losing information:

    migration = {
      "failure_penalty": -0.15,
      "deadzone": 0.01,
      "failure_keys":  [...],   # cells that were compile/measure failures
      "deadzoned_keys": [...],  # cells zeroed as measurement noise
      "history": [...]
    }

Usage:
  # first migration (detects failures as the exact-0.0 cells)
  python3 migrate_reward_cache.py reward_cache.json --failure-penalty -0.15

  # change your mind later — rewrites the SAME cells, no data lost
  python3 migrate_reward_cache.py reward_cache.json --failure-penalty -0.25

  # undo entirely
  python3 migrate_reward_cache.py reward_cache.json --revert

  python3 migrate_reward_cache.py reward_cache.json --dry-run
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cache", help="Path to reward_cache.json")
    p.add_argument("--failure-penalty", type=float, default=-0.15,
                   help="Value for compile/measure-failure cells (default -0.15); "
                        "must match --compile-failure-penalty in train.py")
    p.add_argument("--deadzone", type=float, default=0.01,
                   help="|r| below this becomes exactly 0.0 (default 0.01); "
                        "must match --reward-deadzone in train.py")
    p.add_argument("--revert", action="store_true",
                   help="Restore failure cells to 0.0 (undo the migration). "
                        "Deadzoned cells cannot be restored — their original "
                        "sub-threshold values were noise and are not retained.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    path = Path(args.cache)
    data = json.loads(path.read_text())
    rewards = {k: float(v) for k, v in data.get("rewards", {}).items()}
    if not rewards:
        print("No 'rewards' in cache — nothing to do.")
        return

    mig = data.get("migration")
    first_time = mig is None

    if first_time:
        # Identify failures by value: exactly 0.0 (see module docstring).
        failure_keys  = [k for k, v in rewards.items() if v == 0.0]
        deadzone_keys = [k for k, v in rewards.items()
                         if 0.0 < abs(v) < args.deadzone] if args.deadzone > 0 else []
        history = []
        print("First migration — failures identified as the exact-0.0 cells.")
    else:
        # Re-migration: reuse the recorded key lists, so the penalty can move
        # freely without the cells being identifiable by value any more.
        failure_keys  = [k for k in mig.get("failure_keys", []) if k in rewards]
        deadzone_keys = [k for k in mig.get("deadzoned_keys", []) if k in rewards]
        history = mig.get("history", [])
        print(f"Re-migration — reusing recorded key lists "
              f"(previous penalty {mig.get('failure_penalty')}).")

    target = 0.0 if args.revert else args.failure_penalty
    total = len(rewards)
    print(f"cells                      : {total}")
    print(f"  failure cells -> {target:<7}: {len(failure_keys)}"
          f"  ({100*len(failure_keys)/total:.1f}%)")
    if first_time and args.deadzone > 0:
        print(f"  |r|<{args.deadzone} -> 0.0 (noise) : {len(deadzone_keys)}"
              f"  ({100*len(deadzone_keys)/total:.1f}%)")
    print(f"  untouched                : "
          f"{total - len(failure_keys) - (len(deadzone_keys) if first_time else 0)}")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return

    for k in failure_keys:
        rewards[k] = target
    if first_time:
        for k in deadzone_keys:
            rewards[k] = 0.0

    backup = path.with_suffix(f".json.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(path, backup)

    history.append({"at": datetime.now().isoformat(),
                    "failure_penalty": target,
                    "deadzone": args.deadzone,
                    "reverted": bool(args.revert)})
    data["rewards"] = rewards
    data["migration"] = {
        "failure_penalty": target,
        "deadzone": args.deadzone,
        "failure_keys": failure_keys,      # kept so future re-migration works
        "deadzoned_keys": deadzone_keys if first_time else mig.get("deadzoned_keys", []),
        "history": history,
    }
    path.write_text(json.dumps(data))
    print(f"\nBackup : {backup}")
    print(f"Written: {path}")
    print("Key lists recorded — re-run with a different --failure-penalty any "
          "time, or --revert to undo.")


if __name__ == "__main__":
    main()
