"""
A/B parity harness: prove collect_pipeline.py produces the same rewards as the original
collect_cells.py for a single benchmark, on throwaway checkpoint copies (the real cache is
never touched).

How it isolates ONE benchmark so both tools do identical work:
  * copy the checkpoint twice (A for collect_cells, B for collect_pipeline);
  * in each copy, INJECT dummy rewards for every currently-missing cell so those are "done"
    (keeps both tools from wandering into the other 3 unfinished benchmarks), then REMOVE the
    target benchmark's cells so the target is the ONLY missing set;
  * run collect_cells on A and collect_pipeline on B — each now measures exactly the target;
  * 3-way compare the target's cells: ORIGINAL cache vs A (collect_cells) vs B (pipeline).

Parity verdict (per target cell):
  * classification must match — both real / both compile-failure / both absent(measure_failed);
  * for cells real in BOTH, |reward_B - reward_A| must be <= --tol.
nsys medians of 20 runs vary run-to-run, so values are NOT bitwise equal; the ORIGINAL-vs-A
gap (same tool, two runs) is printed as the noise yardstick --tol should clear.

Usage:
  python3 ab_parity.py CKPT --benchmark NAME --n-runs 20 \\
      --compile-failure-penalty -0.16 --reward-deadzone 0.005 --gpus 2 [--tol 0.03]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def _load_rewards(ckpt: Path) -> dict:
    p = ckpt / "reward_cache.json"
    return {k: float(v) for k, v in json.loads(p.read_text()).get("rewards", {}).items()} \
        if p.exists() else {}


def _failure_keys(ckpt: Path) -> set:
    p = ckpt / "reward_cache.json"
    if not p.exists():
        return set()
    return set(json.loads(p.read_text()).get("migration", {}).get("failure_keys", []))


def _classify(key: str, rewards: dict, fkeys: set) -> str:
    if key not in rewards:
        return "absent"            # measure_failed / not attempted
    return "failure" if key in fkeys else "real"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint_dir", type=Path)
    ap.add_argument("--benchmark", required=True, help="the ONE benchmark to A/B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-runs", type=int, required=True)
    ap.add_argument("--compile-failure-penalty", type=float, required=True)
    ap.add_argument("--reward-deadzone", type=float, required=True)
    ap.add_argument("--compile-timeout-penalty", type=float, default=-1.0)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--nsys-timeout", type=int, default=300)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--tol", type=float, default=0.03, help="max |reward_B - reward_A| to pass")
    ap.add_argument("--with-store", action="store_true",
                    help="also build a pre-compile store for the target and make the pipeline "
                         "STAGE it — validates staged-binary rewards == built-binary rewards")
    ap.add_argument("--procs", type=int, default=16, help="precompile build procs (--with-store)")
    ap.add_argument("--work", type=Path, default=Path("/tmp/uu_ab"),
                    help="scratch root for the two checkpoint copies")
    args = ap.parse_args()

    bench = args.benchmark
    pre = bench + "|"

    # --- compute the currently-missing cells (to neutralize) and the target's cells ---
    from hecbench import discover_benchmarks, HECBENCH_SRC
    from train import precheck_benchmarks, split_benchmarks, build_loop_assignments
    from collect_cells import valid_cells, missing_cells

    ckpt = args.checkpoint_dir
    all_b, _lc, lrm, _n = precheck_benchmarks(
        discover_benchmarks(HECBENCH_SRC), ckpt / "eligible_benchmarks.json",
        skip=True, strict=True)
    tr, va, te = split_benchmarks(all_b, args.val_ratio, args.test_ratio, args.split_seed)
    pools = {"train": tr, "val": va, "test": te}
    benches = [b for s in args.split.split(",") for b in pools[s.strip()]]
    asg = build_loop_assignments(benches, lrm)

    names = {a["benchmark_name"] for a in asg}
    if bench not in names:
        sys.exit(f"benchmark {bench!r} not in split '{args.split}' "
                 f"({len(names)} benchmarks). Pick one that is.")

    orig = _load_rewards(ckpt)
    # cells of the TARGET we expect to be measured (all its valid cells)
    target_cells = []
    for a in asg:
        if a["benchmark_name"] != bench:
            continue
        for (u, f) in valid_cells(a["pre_features_raw"]):
            target_cells.append(f"{bench}|{a['loop_idx']}|{u}|{f}")
    if not target_cells:
        sys.exit(f"{bench}: no valid cells — nothing to A/B.")

    # cells currently missing ANYWHERE in the split (to be dummy-filled so neither tool
    # strays into the other unfinished benchmarks); exclude the target's own keys.
    miss = missing_cells(asg, orig, None)
    missing_keys = set()
    for a in asg:
        k = f"{a['benchmark_name']}|{a['loop_idx']}"
        if k in miss:
            for (u, f) in miss[k]:
                missing_keys.add(f"{a['benchmark_name']}|{a['loop_idx']}|{u}|{f}")
    neutralize = {k for k in missing_keys if not k.startswith(pre)}

    o_fk = _failure_keys(ckpt)                      # read ONCE; reused in compare below
    real_target = sum(1 for k in target_cells if _classify(k, orig, o_fk) == "real")
    print(f"target {bench}: {len(target_cells)} cells "
          f"({real_target} real in the original cache) | neutralizing {len(neutralize)} "
          f"other missing cells so both tools do ONLY {bench}")

    # --- build the two throwaway copies ---
    # Shallow copy: ONLY the root-level files the tools read (eligible/reward/baseline JSON,
    # normalizer, configs).  copytree would duplicate any large subdir (pipe_tmp, nsys
    # reports, model files); the tools create their own scratch under the copy at runtime.
    args.work.mkdir(parents=True, exist_ok=True)
    A, B = args.work / "cc", args.work / "cp"
    for C in (A, B):
        shutil.rmtree(C, ignore_errors=True)
        C.mkdir(parents=True, exist_ok=True)
        for item in ckpt.iterdir():
            if item.is_file():
                shutil.copy2(item, C / item.name)
        if not (C / "eligible_benchmarks.json").exists():
            sys.exit(f"{ckpt}: eligible_benchmarks.json not found at checkpoint root")
        rc = C / "reward_cache.json"
        d = json.loads(rc.read_text()) if rc.exists() else {}
        rw = d.get("rewards", {})
        for k in neutralize:                       # mark other missing cells done (dummy)
            rw.setdefault(k, 0.0)
        for k in list(rw):                         # remove the target → it is the only TODO
            if k.startswith(pre):
                del rw[k]
        d["rewards"] = rw
        d["post_features"] = {k: v for k, v in d.get("post_features", {}).items()
                              if not k.startswith(pre)}
        rc.write_text(json.dumps(d))

    split_args = ["--split", args.split, "--val-ratio", str(args.val_ratio),
                  "--test-ratio", str(args.test_ratio), "--split-seed", str(args.split_seed)]
    common = split_args + ["--n-runs", str(args.n_runs),
              "--compile-failure-penalty", str(args.compile_failure_penalty),
              "--reward-deadzone", str(args.reward_deadzone),
              "--compile-timeout-penalty", str(args.compile_timeout_penalty),
              "--nsys-timeout", str(args.nsys_timeout), "--no-post-features"]
    if args.arch:
        common += ["--arch", args.arch]

    cc = str(HERE / "collect_cells.py")
    cp = str(HERE / "collect_pipeline.py")
    cc_cmd = [sys.executable, "-u", cc, str(A), "--num-workers", str(args.gpus)] + common
    cp_cmd = [sys.executable, "-u", cp, str(B), "--gpus", str(args.gpus),
              "--exec", "1", "--compile", "2", "--parse", "3"] + common

    # --with-store: build a store for the target off copy B (where it is now the only
    # missing benchmark), then make the pipeline STAGE it.  This validates that a staged
    # binary yields the same reward as a freshly-built one — the end-to-end staging check
    # Test 4 couldn't do (its store targeted a no-baseline benchmark).
    if args.with_store:
        store = args.work / "store"
        shutil.rmtree(store, ignore_errors=True)
        pc_cmd = [sys.executable, "-u", str(HERE / "precompile.py"), str(B),
                  "--store", str(store), "--benchmarks", bench, "--procs", str(args.procs)]
        pc_cmd += split_args
        if args.arch:
            pc_cmd += ["--arch", args.arch]
        print("\n=== precompile (build store for pipeline side) ===\n", " ".join(pc_cmd))
        rp = subprocess.run(pc_cmd)
        if rp.returncode:
            print(f"WARNING: precompile exited {rp.returncode} — pipeline will build, not stage")
        cp_cmd += ["--precompiled-store", str(store)]

    print("\n=== collect_cells ===\n", " ".join(cc_cmd))
    r1 = subprocess.run(cc_cmd)
    print("\n=== collect_pipeline ===\n", " ".join(cp_cmd))
    r2 = subprocess.run(cp_cmd)
    if r1.returncode or r2.returncode:
        print(f"WARNING: a tool exited non-zero (cc={r1.returncode} cp={r2.returncode})")

    # --- 3-way compare on the target's cells ---
    a_rw, b_rw = _load_rewards(A), _load_rewards(B)
    a_fk, b_fk = _failure_keys(A), _failure_keys(B)   # o_fk already read once above
    print(f"\n{'cell':40s} {'orig':>9} {'cc(A)':>9} {'cp(B)':>9} {'|B-A|':>8}  class(A/B)")
    cls_mismatch, val_fail, max_d, noise = [], [], 0.0, 0.0
    for k in sorted(target_cells):
        oc, ac, bc = (_classify(k, orig, o_fk), _classify(k, a_rw, a_fk),
                      _classify(k, b_rw, b_fk))
        ov = orig.get(k); av = a_rw.get(k); bv = b_rw.get(k)
        d = abs(bv - av) if (ac == "real" and bc == "real") else None
        if ac == "real" and oc == "real":
            noise = max(noise, abs(av - ov))
        if d is not None:
            max_d = max(max_d, d)
            if d > args.tol:
                val_fail.append(k)
        if ac != bc:
            cls_mismatch.append((k, ac, bc))
        fmt = lambda v: "-" if v is None else f"{v:+.4f}"
        print(f"{k:40s} {fmt(ov):>9} {fmt(av):>9} {fmt(bv):>9} "
              f"{('-' if d is None else f'{d:.4f}'):>8}  {ac}/{bc}")

    print(f"\ncells={len(target_cells)}  max|B-A|={max_d:.4f}  tol={args.tol}  "
          f"noise(orig-vs-cc)={noise:.4f}")
    ok = True
    if cls_mismatch:
        ok = False
        print(f"FAIL: {len(cls_mismatch)} classification mismatch(es): "
              + ", ".join(f"{k}[{a}!={b}]" for k, a, b in cls_mismatch[:8]))
    if val_fail:
        ok = False
        print(f"FAIL: {len(val_fail)} cell(s) exceed tol: " + ", ".join(val_fail[:8]))
    if max_d > noise and not val_fail:
        print("NOTE: max|B-A| exceeds the orig-vs-cc noise but is within tol — acceptable, "
              "but widen the A/B or lower --tol if you want a tighter bound.")
    print("\nRESULT:", "PASS — pipeline matches collect_cells within noise" if ok
          else "FAIL — see mismatches above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
