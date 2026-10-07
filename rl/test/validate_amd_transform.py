#!/usr/bin/env python3
"""
Manual AMD/HIP transform validation — run on the box BEFORE the reward grid.
COMPILE-ONLY: no GPU execution, no profiler; it can run while the GPU is busy.

It establishes that everything the reward grid depends on works on amdgcn, mapped
to the grid's ACTION SPACE {no-op, unroll_only, unmerge_unroll}:

  1. HIP benchmarks COMPILE through the pipeline's own path (so the loud-fail HIP
     flag set in hecbench._make trips here, on benchmark #1, not after hours).
  2. LoopCount emits DEVICE loops with isKernelFunction=1  (isGPUKernel / AMDGPU_KERNEL).
  3. The UU transform FIRES on amdgcn for BOTH actions:
       - unroll_only  (unmerge=0, factor>1; forced via ULO.Force so it must change IR)
       - unmerge_unroll (unmerge=1)
     Firing is the load-bearing check: -uu-match-targettriple does an EXACT string
     compare, so a triple mismatch would SILENTLY skip every loop and zero the grid.
     Because UU runs BEFORE LoopCount (pipeline slots 1102<1103), a
     --enable-uu --enable-loopcount compile reports POST-transform IR, so the
     device-loop signature differs from a loopcount-only baseline IFF UU fired.
     Signature = sorted multiset of per-loop structural tuples → loopIdx-independent
     AND sensitive to any per-loop change (no aggregate-sum-collision blind spot).
  4. The published heuristic (NVPTX gate now lifted) also fires — the paper's
     baseline.  Reported as confirmation; does not gate the grid verdict.

VERDICT: AMD support is ESTABLISHED only if, across the benchmarks, HIP compiles,
device loops are recognised, AND both unroll_only and unmerge_unroll fire.  PARTIAL
(only one action fires) or NOT ESTABLISHED (neither → triple mismatch) both exit
non-zero and say exactly what to investigate — so you never start a grid on a
transform that silently no-ops.

Usage (on the box):
  TARGET_ARCH=gfx90a ROCM_PATH=/opt/rocm HECBENCH_SRC=/path/to/hip-tree \\
  IR2VEC_VOCAB=/path/to/seedEmbeddingVocab75D.json \\
  python3 test/validate_amd_transform.py --benchmarks bench1-hip bench2-hip

Prefer benchmarks with a multi-path (numPaths>1) eligible loop so unmerge is exercised.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hecbench import (                                               # noqa: E402
    ARCH, IS_HIP, HECBENCH_SRC, ROCM_PATH, ROCM_DEVICE_LIB,
    discover_benchmarks, get_loop_features,
    compile_loopcount, parse_loopcount_output, _build_extra_cflags, _make,
)

_PASS, _FAIL, _WARN = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"

# IR-derived structural fields; per-loop tuples of these form the firing signature.
_SIG_KEYS = ["loopSize", "numBasicBlocks", "numMemoryInsts", "numComputeInsts",
             "numControlFlowInsts", "numPaths"]
_MAX_PROBE_LOOPS = 3          # try a few loops so one unamenable loop isn't fatal

# Global sufficiency state (set across all benchmarks).
_state = {"compiled": False, "device_loops": False,
          "unroll_fired": False, "unmerge_fired": False, "heuristic_fired": False}
_hard_fail = False            # a loud failure (compile crash, etc.) occurred


def _mark_fail():
    global _hard_fail
    _hard_fail = True


def _is_device_row(r) -> bool:
    is_kernel = str(r.get("isKernelFunction", "0")).strip() in ("1", "1.0")
    kp = str(r.get("kernelParents", "")).strip()
    return is_kernel or (kp != "" and kp.lower() != "nan")


def _device_rows(stderr: str) -> list:
    rows = []
    for fm in parse_loopcount_output(stderr).values():
        for df in fm.values():
            for _, r in df.iterrows():
                if _is_device_row(r):
                    rows.append(r)
    return rows


def _loop_tuples(rows: list):
    """Sorted multiset of per-loop structural tuples: loopIdx-independent and
    sensitive to ANY per-loop structural change (firing signature)."""
    return sorted(
        tuple(round(float(r[k]), 3) if k in r.index else 0.0 for k in _SIG_KEYS)
        for r in rows
    )


def _agg(rows: list) -> dict:
    a = {"n_loops": len(rows)}
    for k in _SIG_KEYS:
        a[k] = round(sum(float(r[k]) for r in rows if k in r.index), 1)
    a["insts"] = a["numMemoryInsts"] + a["numComputeInsts"] + a["numControlFlowInsts"]
    return a


def _compile(bench, filename, triple, idx, unmerge, factor):
    cflags = _build_extra_cflags(
        enable_uu=True, enable_loopcount=True, filename=filename, triple=triple,
        loop_indices=[idx], unmerge_flags=[unmerge], unroll_factors=[factor])
    return _make(bench, extra_cflags=cflags, arch=ARCH)


def _probe(bench, triple, base_tuples, base_agg, loops, unmerge, factor, label, expect):
    """Compile `label` across candidate loops until the IR signature changes.
    Returns True if it fired. Firing is informational (printed), not a hard check."""
    for (fname, idx) in loops[:_MAX_PROBE_LOOPS]:
        try:
            res = _compile(bench, fname, triple, idx, unmerge, factor)
        except Exception as e:
            print(f"    {_WARN} [{label}] loop {idx}: compile raised {type(e).__name__}: {e}")
            _mark_fail()
            continue
        if res.returncode != 0:
            print(f"    {_WARN} [{label}] loop {idx}: compile rc={res.returncode}; "
                  f"{(res.stderr or '')[-200:]}")
            _mark_fail()
            continue
        rows = _device_rows(res.stderr)
        if _loop_tuples(rows) != base_tuples:
            var = _agg(rows)
            # direction hint (semantic colour only — firing is already confirmed
            # by the signature change above, so this never gates the verdict).
            if expect == "grow":
                dirn = "insts %d->%d %s" % (base_agg["insts"], var["insts"],
                                            "OK up" if var["insts"] > base_agg["insts"] else "(no inst growth)")
            else:  # restructure: unmerge+unroll REPLICATES specialized paths, so
                   # numPaths typically goes UP (or n_loops changes) — any change is fine.
                dirn = "numPaths %.0f->%.0f, n_loops %d->%d OK (restructured)" % (
                    base_agg["numPaths"], var["numPaths"],
                    base_agg["n_loops"], var["n_loops"])
            print(f"    {_PASS} [{label}] FIRED on loop {idx}  ({dirn})")
            return True
        print(f"    ·     [{label}] loop {idx}: no IR change (trying next)")
    print(f"    {_WARN} [{label}] did not fire on any of {min(len(loops), _MAX_PROBE_LOOPS)} loops")
    return False


def validate_benchmark(bench: Path) -> None:
    print(f"\n=== {bench.name} ===")

    try:
        base_res = compile_loopcount(bench)
    except Exception as e:
        print(f"  {_FAIL} HIP baseline compile raised {type(e).__name__}: {e}")
        _mark_fail(); return
    if base_res.returncode != 0:
        print(f"  {_FAIL} HIP baseline compile rc={base_res.returncode}; stderr tail:\n"
              f"{(base_res.stderr or '')[-400:]}")
        _mark_fail(); return
    print(f"  {_PASS} HIP baseline compile (loud-fail flags OK)")
    _state["compiled"] = True

    base_rows = _device_rows(base_res.stderr)
    if not base_rows:
        # No eligible device loops here — nothing to validate on this benchmark.
        # This is a benign SKIP (compiled fine), NOT a failure: do not _mark_fail,
        # do not degrade the verdict. (isGPUKernel is confirmed by any benchmark
        # that does report device loops.)
        print(f"  {_WARN} no device loops after filtering — skipping this benchmark "
              "(benign; not an isGPUKernel problem)")
        return
    print(f"  {_PASS} {len(base_rows)} device loops with isKernelFunction=1 (isGPUKernel OK)")
    _state["device_loops"] = True
    base_tuples, base_agg = _loop_tuples(base_rows), _agg(base_rows)

    try:
        file_map, primary_file, triple = get_loop_features(bench)
    except Exception as e:
        print(f"  {_FAIL} get_loop_features raised {type(e).__name__}: {e}")
        _mark_fail(); return
    print(f"  triple fed to -uu-match-targettriple: {triple!r}")

    all_loops, multipath = [], []
    for fname, df in file_map.items():
        for _, r in df.iterrows():
            all_loops.append((fname, int(r["loopIdx"])))
            if float(r.get("numPaths", 1)) > 1:
                multipath.append((fname, int(r["loopIdx"])))
    if not all_loops:
        print(f"  {_WARN} no eligible loops here — pick a benchmark with eligible loops")
        return

    # unroll_only (forced) — expect the body to grow.
    if _probe(bench, triple, base_tuples, base_agg, all_loops, 0, 4, "unroll_only f=4", "grow"):
        _state["unroll_fired"] = True
    # unmerge_unroll — expect path split (numPaths down / n_loops up).
    um_loops = multipath or all_loops
    if not multipath:
        print(f"  {_WARN} no numPaths>1 loop — probing unmerge on single-path loops (may not split)")
    if _probe(bench, triple, base_tuples, base_agg, um_loops, 1, 4, "unmerge_unroll f=4", "split"):
        _state["unmerge_fired"] = True

    # Published heuristic (confirmation only; does not gate the verdict).
    try:
        cflags = _build_extra_cflags(enable_loopcount=True, filename=primary_file,
                                     triple=triple) + " -mllvm --enable-uu-heuristic"
        hres = _make(bench, extra_cflags=cflags, arch=ARCH)
        if hres.returncode == 0 and _loop_tuples(_device_rows(hres.stderr)) != base_tuples:
            print(f"  {_PASS} published heuristic fires on AMD (NVPTX gate lifted)")
            _state["heuristic_fired"] = True
        elif hres.returncode == 0:
            print(f"  {_WARN} heuristic selected no loop here (size/path caps) — try a small-loop benchmark")
        else:
            print(f"  {_WARN} heuristic compile rc={hres.returncode}")
    except Exception as e:
        print(f"  {_WARN} heuristic compile raised {type(e).__name__}: {e}")


def _ir_recipe():
    print("\n--- explicit IR before/after (eyeball one loop) ---")
    print("LoopCount runs AFTER UU, so -loopcount-emit-ir with --enable-uu is post-transform.")
    print("(HIP device IR flag is --cuda-device-only on older clang, --offload-device-only on newer.)")
    print("""
  V="-mllvm --ir2vec-vocab-path=$IR2VEC_VOCAB"
  HIP="-x hip --offload-arch=$TARGET_ARCH --rocm-path=$ROCM_PATH -D__HIP_PLATFORM_AMD__ -I$ROCM_PATH/include --cuda-device-only"
  clang++ $HIP -mllvm --enable-loopcount $V -mllvm -loopcount-emit-ir=base.ll -c <bench>/<src.cu> -o /dev/null
  clang++ $HIP -mllvm --enable-uu -mllvm --enable-loopcount $V \\
      -mllvm -uu-match-targettriple=amdgcn-amd-amdhsa \\
      -mllvm -uu-opt-loop-idx=<idx> -mllvm -uu-opt-loop-unmerge=1 -mllvm -uu-opt-loop-unrollfactors=4 \\
      -mllvm -loopcount-emit-ir=uu.ll -c <bench>/<src.cu> -o /dev/null
  diff base.ll uu.ll | less   # expect unrolled body / split paths in the target loop
""")


def _verdict() -> int:
    s = _state
    print("\n" + "=" * 60)
    print("SUFFICIENCY:")
    for k in ("compiled", "device_loops", "unroll_fired", "unmerge_fired", "heuristic_fired"):
        print(f"  {_PASS if s[k] else _FAIL}  {k}")
    triple_ok = s["unroll_fired"] or s["unmerge_fired"]
    both = s["unroll_fired"] and s["unmerge_fired"]
    print("-" * 60)
    if not s["compiled"]:
        print(f"AMD TRANSFORM SUPPORT: {_FAIL}  NOT ESTABLISHED — HIP compile failed. Fix the flag set first.")
        return 1
    if not triple_ok:
        print(f"AMD TRANSFORM SUPPORT: {_FAIL}  NOT ESTABLISHED — NOTHING fired ⇒ -uu-match-targettriple "
              "exact-string MISMATCH (or all loops ineligible). DO NOT start the grid; fix triple threading.")
        return 1
    if not both:
        miss = "unroll_only" if not s["unroll_fired"] else "unmerge_unroll"
        print(f"AMD TRANSFORM SUPPORT: {_WARN}  PARTIAL — triple OK (one action fires) but {miss} never "
              "fired. Those grid cells would be no-ops; investigate that action before a full grid.")
        return 1
    if _hard_fail:
        print(f"AMD TRANSFORM SUPPORT: {_WARN}  ESTABLISHED, with caveats — both actions fire, but some probe "
              "compiles errored (see WARN lines). Those may be benchmark-specific (the grid penalises "
              "compile failures), but review them before the full run.")
        return 1
    print(f"AMD TRANSFORM SUPPORT: {_PASS}  ESTABLISHED — both grid actions fire; start the grid.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", nargs="*", default=[])
    args = p.parse_args()

    print(f"ARCH={ARCH}  IS_HIP={IS_HIP}  HECBENCH_SRC={HECBENCH_SRC}")
    if IS_HIP:
        print(f"ROCM_PATH={ROCM_PATH}  ROCM_DEVICE_LIB={ROCM_DEVICE_LIB or '(unset)'}")
    if not IS_HIP:
        print(f"{_WARN}  IS_HIP False — NVIDIA path. Set TARGET_ARCH=gfx90a for AMD (running as sanity check).")

    disc = {b.name: b for b in discover_benchmarks(HECBENCH_SRC)}
    if args.benchmarks:
        benches = []
        for name in args.benchmarks:
            if name in disc:
                benches.append(disc[name])
            else:
                cand = Path(HECBENCH_SRC) / name
                if cand.is_dir() and (cand / "Makefile").exists():
                    benches.append(cand)
                else:
                    print(f"{_WARN}  benchmark {name!r} not found under {HECBENCH_SRC}")
    else:
        benches = list(disc.values())[:2]

    if not benches:
        print(f"{_FAIL}  no benchmarks (check HECBENCH_SRC / --benchmarks)")
        sys.exit(1)

    for b in benches:
        validate_benchmark(b)
    _ir_recipe()
    sys.exit(_verdict())


if __name__ == "__main__":
    main()
