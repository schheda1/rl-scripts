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

  # repair sign-flip poisoning: cells written as +0.15 (a positive penalty
  # flag somewhere in the cache lineage) are failures mislabelled as WINS.
  # Rewrites every cell equal to VALUE exactly to --failure-penalty and
  # records the keys so later re-tunes / reverts cover them too.
  python3 migrate_reward_cache.py reward_cache.json --relabel 0.15 --failure-penalty -0.15
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
    p.add_argument("--relabel", type=float, default=None, metavar="VALUE",
                   help="Repair mode: every cell whose value equals VALUE "
                        "EXACTLY (bit-equal float, e.g. 0.15 written by a "
                        "sign-flipped penalty flag) is relabelled as a "
                        "compile-failure cell: rewritten to --failure-penalty "
                        "and its key recorded in migration.failure_keys. "
                        "Cells already at exactly --failure-penalty but "
                        "missing from failure_keys are adopted too, so future "
                        "re-tunes move the whole failure population.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    path = Path(args.cache)
    data = json.loads(path.read_text())
    rewards = {k: float(v) for k, v in data.get("rewards", {}).items()}
    if not rewards:
        print("No 'rewards' in cache — nothing to do.")
        return

    mig = data.get("migration")

    if args.relabel is not None:
        # --- Repair mode: adopt mislabelled failure cells by exact value ---
        # Exact float equality is the identifier ON PURPOSE: a written literal
        # (json.loads('0.15') == 0.15) matches, while a genuinely measured
        # reward that happens to be near 0.15 is a computed double and will
        # not be bit-equal.  The probability of a measurement colliding with
        # the exact literal is ~0; 3475 collisions is a writer, not chance.
        if args.revert:
            sys.exit("--relabel and --revert are mutually exclusive")
        mislabelled = [k for k, v in rewards.items() if v == args.relabel]
        strays = [k for k, v in rewards.items()
                  if v == args.failure_penalty
                  and k not in set((mig or {}).get("failure_keys", []))]
        if not mislabelled and not strays:
            print(f"No cells equal {args.relabel} exactly and no unrecorded "
                  f"cells at {args.failure_penalty} — nothing to repair.")
            return
        mig = mig or {"failure_keys": [], "deadzoned_keys": [], "history": []}
        failure_keys = sorted(
            set(mig.get("failure_keys", [])) | set(mislabelled) | set(strays))
        print(f"relabel {args.relabel} -> {args.failure_penalty}:")
        print(f"  mislabelled cells rewritten : {len(mislabelled)}")
        print(f"  unrecorded penalty strays   : {len(strays)} (adopted)")
        print(f"  failure_keys after          : {len(failure_keys)}")
        if args.dry_run:
            print("\nDry run — nothing written.")
            return
        for k in mislabelled:
            rewards[k] = args.failure_penalty
        backup = path.with_suffix(f".json.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(path, backup)
        mig.setdefault("history", []).append({
            "at": datetime.now().isoformat(),
            "relabeled_from": args.relabel,
            "relabeled_count": len(mislabelled),
            "adopted_strays": len(strays),
            "failure_penalty": args.failure_penalty,
        })
        mig["failure_penalty"] = args.failure_penalty
        mig["failure_keys"] = failure_keys
        mig.setdefault("deadzone", args.deadzone)
        mig.setdefault("deadzoned_keys", [])
        data["rewards"] = rewards
        data["migration"] = mig
        path.write_text(json.dumps(data))
        print(f"\nBackup : {backup}")
        print(f"Written: {path}")
        print("Keys recorded — the repaired cells now move with any future "
              "--failure-penalty re-tune or --revert.")
        return

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
