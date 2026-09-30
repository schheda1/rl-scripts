"""
Pre-compile a fraction of the reward-grid cells UPFRONT, off the GPU node.

Compile is CPU work; doing ~30% of it before the GPU job (on a free CPU node with the
CUDA toolkit) takes that fraction out of the GPU allocation and warm-starts the pipeline
so exec never waits on builds.  The built binaries go into a store keyed by cellkey;
`collect_pipeline.py --precompiled-store DIR` then STAGES them (copy, no build) instead
of compiling, falling back to build for anything not in the store.

SAFE BY CONSTRUCTION here: one clang/LLVM build, one CUDA toolkit → identical codegen →
identical kernel time → identical reward.  The store is still STAMPED (arch + clang
version) and FINGERPRINTED per benchmark (source hash) so a future toolchain change or a
benchmark edit makes the consumer fall back to building rather than trust a stale binary.
Reuses compile_single_loop_ex so the -mllvm UU flags are byte-for-byte the job's.

Requirements: clang++ + the CUDA toolkit (headers + libdevice) installed — NO GPU needed
(CUDA compiles offline).  --arch MUST match the GPU the job runs on (the fatbin encodes
sm_XX).

Usage:
  # pre-compile ~30% of the missing test cells (largest benchmarks first), off-GPU:
  IR2VEC_VOCAB=/path/seed.json python3 precompile.py CKPT --store /scratch/uu_store \\
      --split test --arch sm_90 --frac 0.30 --procs 32
  # sanity: rebuild a sample and confirm identical codegen (needs cuobjdump):
  python3 precompile.py CKPT --store /scratch/uu_store --arch sm_90 --verify-store
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from store_common import (cellkey, sanitize, structural_copy, is_cell_done, write_cell,
                          run_target_rel, benchmark_fingerprint, toolchain_stamp,
                          write_manifest, cell_dir)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("precompile")


def build_one(cell: dict, store: str, arch: str, threshold: int, scratch: str) -> tuple:
    """Build one cell into the store (atomic).  CPU-only.  Returns (status, key)."""
    from hecbench import compile_single_loop_ex, _get_run_command
    key = cellkey(cell["bench"], cell["li"], cell["u"], cell["f"])
    if is_cell_done(store, key):
        return ("skip", key)
    bdir = Path(scratch) / sanitize(key)
    try:
        shutil.rmtree(bdir, ignore_errors=True)
        structural_copy(Path(cell["path"]), bdir, threshold)
        try:
            ok, _err = compile_single_loop_ex(
                bdir, loop_idx=cell["li"], unmerge=cell["u"], factor=cell["f"],
                filename=cell["filename"], triple=cell["triple"], arch=arch)
        except subprocess.TimeoutExpired:
            return ("timeout", key)
        if not ok:
            return ("fail", key)                 # in-job will rebuild → cached penalty
        target = run_target_rel(_get_run_command(bdir, arch))
        if os.path.isabs(target):
            return ("notarget", key)      # absolute run target can't be relocated into store
        binpath = bdir / target
        if not binpath.exists():
            return ("notarget", key)
        write_cell(store, key, binpath, target, cell["bench"], cell["fp"],
                   cell["filename"], cell["triple"])
        return ("ok", key)
    except Exception as e:
        log.warning("build error %s: %s", key, e)
        return ("error", key)
    finally:
        shutil.rmtree(bdir, ignore_errors=True)


def _select_benchmarks(by_bench: dict, frac: float, max_frac: float,
                       explicit: "list | None") -> list:
    """Whole-benchmark selection, largest cell-count first, until >= frac of cells.
    Skips entirely if frac*total < 1 cell.  Overshoot bounded by one benchmark; a
    --max-frac cap stops before exceeding it (once at least one is picked)."""
    total = sum(len(v) for v in by_bench.values())
    if explicit is not None:
        return [b for b in explicit if b in by_bench]
    if total == 0 or frac * total < 1:
        return []
    order = sorted(by_bench, key=lambda b: -len(by_bench[b]))
    target = frac * total
    cap = max_frac * total if max_frac > 0 else float("inf")
    selected, cum = [], 0
    for b in order:
        if selected and cum >= target:
            break
        if selected and cum + len(by_bench[b]) > cap:
            break
        selected.append(b)
        cum += len(by_bench[b])
    return selected


def verify_store(store: str, arch: str, threshold: int, scratch: str, n: int, src) -> int:
    """Rebuild a sample of stored cells and compare GPU codegen (cuobjdump SASS, which is
    path/timestamp-independent).  Confirms the store's binaries match a fresh build."""
    import json
    mpath = Path(store) / "manifest.json"
    if not mpath.exists():
        log.error("verify: no manifest at %s", store)
        return 1
    man = json.loads(mpath.read_text())
    keys = man.get("cells", [])[:max(n, 0)]
    if not keys:
        log.info("verify: no cells in store"); return 0
    if shutil.which("cuobjdump") is None:
        log.warning("verify: cuobjdump not found — cannot compare codegen; stamp + "
                    "fingerprint guards still protect the store. SKIPPED.")
        return 0

    def _sass(binpath):
        try:
            r = subprocess.run(["cuobjdump", "--dump-sass", str(binpath)],
                               capture_output=True, text=True, timeout=180)
            return r.stdout
        except Exception:
            return None

    from hecbench import compile_single_loop_ex, _get_run_command, discover_benchmarks
    benches = discover_benchmarks(src)
    ok = bad = 0
    for key in keys:
        bench, li, u, f = key.split("|")
        meta = json.loads((cell_dir(store, key) / "meta.json").read_text())
        stored_bin = cell_dir(store, key) / meta["target"]
        bpath = next((b for b in benches if b.name == bench), None)
        if bpath is None:
            log.warning("verify %s: benchmark %s not found", key, bench); bad += 1; continue
        bdir = Path(scratch) / ("verify_" + sanitize(key))
        try:
            shutil.rmtree(bdir, ignore_errors=True)
            structural_copy(bpath, bdir, threshold)
            built, _ = compile_single_loop_ex(
                bdir, loop_idx=int(li), unmerge=int(u), factor=int(f),
                filename=meta.get("filename", ""), triple=meta.get("triple", "-"),
                arch=arch)
            if not built:
                log.error("verify %s: rebuild failed", key); bad += 1; continue
            fresh_bin = bdir / run_target_rel(_get_run_command(bdir, arch))
            a, b = _sass(stored_bin), _sass(fresh_bin)
            if a is not None and a == b:
                ok += 1
            else:
                log.error("verify %s: codegen MISMATCH (stored vs fresh)", key); bad += 1
        finally:
            shutil.rmtree(bdir, ignore_errors=True)
    log.info("verify: %d match / %d mismatch of %d sampled", ok, bad, len(keys))
    return 0 if bad == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint_dir", type=Path)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--split", default="test", help="comma list: test,val,train")
    p.add_argument("--frac", type=float, default=0.30, help="fraction of missing cells")
    p.add_argument("--max-frac", type=float, default=0.0, help="hard cap (0 = none)")
    p.add_argument("--benchmarks", default=None, help="comma list; overrides --frac")
    p.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--arch", default=None, help="MUST match the GPU job (fatbin sm_XX)")
    p.add_argument("--copy-threshold", type=int, default=1_000_000)
    p.add_argument("--scratch", default=None, help="build scratch (default STORE/_build)")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--hecbench-src", default=None)
    p.add_argument("--verify-store", action="store_true")
    p.add_argument("--verify-n", type=int, default=5)
    args = p.parse_args()

    from hecbench import ARCH, HECBENCH_SRC, discover_benchmarks, IR2VEC_VOCAB
    from train import precheck_benchmarks, split_benchmarks, build_loop_assignments
    from collect_cells import missing_cells
    if args.arch is None:
        args.arch = ARCH
    scratch = args.scratch or str(args.store / "_build")
    Path(scratch).mkdir(parents=True, exist_ok=True)
    src = Path(args.hecbench_src) if args.hecbench_src else HECBENCH_SRC

    if args.verify_store:
        return verify_store(str(args.store), args.arch, args.copy_threshold, scratch,
                            args.verify_n, src)

    if not IR2VEC_VOCAB or not Path(IR2VEC_VOCAB).exists():
        sys.exit(f"IR2VEC_VOCAB unset/missing ({IR2VEC_VOCAB!r}) — needed for loopcount "
                 f"compiles")
    elig = args.checkpoint_dir / "eligible_benchmarks.json"
    if not elig.exists():
        sys.exit(f"missing {elig}")

    all_b, _lc, loop_records_map, _norm = precheck_benchmarks(
        discover_benchmarks(src), elig, skip=True, strict=True)
    train_b, val_b, test_b = split_benchmarks(all_b, args.val_ratio, args.test_ratio,
                                              args.split_seed)
    pools = {"train": train_b, "val": val_b, "test": test_b}
    wanted = [s.strip() for s in args.split.split(",") if s.strip()]
    if any(s not in pools for s in wanted):
        sys.exit(f"unknown split(s): {wanted}")
    benches = [b for s in wanted for b in pools[s]]
    assignments = build_loop_assignments(benches, loop_records_map)

    # missing cells (from the reward cache) → group by benchmark
    import json as _json
    rc = args.checkpoint_dir / "reward_cache.json"
    rewards = {}
    if rc.exists():
        rewards = {k: float(v) for k, v in
                   _json.loads(rc.read_text()).get("rewards", {}).items()}
    todo = missing_cells(assignments, rewards, None)          # cells only (no postfeat)
    by_bench: dict = {}
    a_by_loop = {(a["benchmark_name"], a["loop_idx"]): a for a in assignments}
    for key, cells in todo.items():
        if not cells:
            continue
        bench, li = key.split("|")[0], int(key.split("|")[1])
        by_bench.setdefault(bench, []).extend((bench, li, u, f) for (u, f) in cells)

    explicit = ([s.strip() for s in args.benchmarks.split(",") if s.strip()]
                if args.benchmarks else None)
    selected = _select_benchmarks(by_bench, args.frac, args.max_frac, explicit)
    total = sum(len(v) for v in by_bench.values())
    sel_cells = sum(len(by_bench[b]) for b in selected)
    log.info("missing: %d cells / %d benchmarks; selecting %d benchmarks / %d cells (%.0f%%)",
             total, len(by_bench), len(selected), sel_cells,
             100 * sel_cells / max(total, 1))
    if not selected:
        log.info("nothing to pre-compile (frac too small or no missing cells).")
        # still write an (empty) stamped manifest so the consumer can verify stamps
        write_manifest(str(args.store), toolchain_stamp(args.arch), {}, [])
        return 0

    # fingerprint each selected benchmark once (its cells share it)
    fps = {b: benchmark_fingerprint(Path(a_by_loop[(b, by_bench[b][0][1])]["benchmark_path"]))
           for b in selected}
    tasks = []
    for b in selected:
        for (bench, li, u, f) in by_bench[b]:
            a = a_by_loop[(bench, li)]
            tasks.append({"bench": bench, "path": a["benchmark_path"], "li": li,
                          "filename": a["filename"], "triple": a["triple"],
                          "u": u, "f": f, "fp": fps[b]})

    log.info("building %d cells with %d procs → %s", len(tasks), args.procs, args.store)
    counts, done_keys = {}, []
    with ThreadPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(build_one, t, str(args.store), args.arch, args.copy_threshold,
                          scratch) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            status, key = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status in ("ok", "skip"):
                done_keys.append(key)
            if i % 100 == 0:
                log.info("  %d/%d  %s", i, len(tasks), counts)

    # manifest = stamps + per-benchmark fp + all cells that are now .done
    from store_common import is_cell_done as _done
    all_done = [cellkey(bench, li, u, f)
                for b in selected for (bench, li, u, f) in by_bench[b]
                if _done(str(args.store), cellkey(bench, li, u, f))]
    write_manifest(str(args.store), toolchain_stamp(args.arch), fps, all_done)
    shutil.rmtree(scratch, ignore_errors=True)
    log.info("DONE  %s  |  %d cells in store  |  manifest written", counts, len(all_done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
