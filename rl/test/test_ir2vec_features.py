"""
Server-side functional tests for the IR2Vec loop-embedding features.

Requires: a benchmark tree, the IR2Vec-enabled llvm build on PATH, and
IR2VEC_VOCAB pointing at seedEmbeddingVocab75D.json.  Run on the GPU box, NOT
locally (compilation + LoopCount needed).

  IR2VEC_VOCAB=/path/seedEmbeddingVocab75D.json \
  python3 test/test_ir2vec_features.py [--benchmark mandelbrot-cuda]
                                       [--template-benchmark sortKV-cuda]

Covers plan §Part 3 T-PY 1-5.  T3 (pre vs post-unmerge) is the user-required
test: the full vector MUST change; the embedding-subvector delta is REPORTED,
not asserted (mean composition may legitimately move little under unmerge).
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent
from hecbench import (
    FEATURE_COLUMNS, IR2VEC_DIM, _EMB_COLUMNS, _row_to_tensor,
    compile_loopcount, compile_single_loop, get_loop_features,
    _build_extra_cflags, _make, parse_loopcount_output,
)

_PASS, _FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {_PASS if ok else _FAIL}  {name}" + (f"  — {detail}" if detail else ""))
    _results.append((name, ok))


_KEMB_COLUMNS = [f"kemb{i}" for i in range(IR2VEC_DIM)]


def emb_subvector(row) -> list[float]:
    return [float(row[c]) for c in _EMB_COLUMNS]


def l2(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def test_schema() -> None:
    print("\nT1: schema")
    import features
    n_struct = 18
    # Blocks are variable-width now (emb/femb/kemb are 75, widths is 11), so sum the
    # actual block lengths rather than assuming n_blocks * IR2VEC_DIM.
    expected = n_struct + sum(len(features.BLOCKS[b][0]) for b in features.ENABLED_BLOCKS)
    check(f"FEATURE_COLUMNS length == {expected}",
          len(FEATURE_COLUMNS) == expected, f"{len(FEATURE_COLUMNS)}")
    check("agent.N_FEATURES == len(FEATURE_COLUMNS)",
          agent.N_FEATURES == len(FEATURE_COLUMNS),
          f"{agent.N_FEATURES} vs {len(FEATURE_COLUMNS)}")
    check("trip-count indices unmoved (10,11)",
          FEATURE_COLUMNS[10:12] == ["tripCountKnown", "tripCount"],
          str(FEATURE_COLUMNS[10:12]))
    # Verify EVERY enabled block is contiguous, in canonical order, right after
    # the 18 structural columns — not just the first (so a mis-placed femb/kemb
    # block is caught, not only a wrong leading block).
    off, layout_ok = n_struct, True
    for b in features.ENABLED_BLOCKS:
        cols = features.BLOCKS[b][0]
        if FEATURE_COLUMNS[off:off + len(cols)] != cols:
            layout_ok = False
            break
        off += len(cols)
    check("enabled blocks contiguous & in canonical order after structural",
          layout_ok and off == len(FEATURE_COLUMNS),
          "+".join(f"{b}({len(features.BLOCKS[b][0])})"
                   for b in features.ENABLED_BLOCKS))
    check("IR2VEC_DIM == 75", IR2VEC_DIM == 75)


def test_extraction(bench: Path):
    print(f"\nT2: extraction ({bench.name})")
    file_map, _, _ = get_loop_features(bench)
    n = sum(len(df) for df in file_map.values())
    check("eligible loops found", n > 0, f"{n} loops")
    if n == 0:
        return None
    df = next(iter(file_map.values()))
    have = all(c in df.columns for c in _EMB_COLUMNS)
    check("all 75 emb columns present", have)
    nonzero = df[_EMB_COLUMNS].abs().to_numpy().sum() > 0
    check("embeddings are non-zero", bool(nonzero))
    return file_map


def _is_device_row(row) -> bool:
    """Device loop = in a __global__ kernel OR a __device__ fn with kernel callers."""
    is_kernel = str(row.get("isKernelFunction", "0")).strip() in ("1", "1.0")
    kp = str(row.get("kernelParents", "")).strip()
    return is_kernel or (kp != "" and kp.lower() != "nan")


def test_device_embeddings(bench: Path) -> None:
    """
    THE critical test: embeddings must be generated for DEVICE loops, not just
    host loops.  Parse the RAW LoopCount output (before device filtering) and
    verify that kernel/device rows specifically carry non-zero, distinct
    embeddings — a host-only embedding (device rows all zero) would pass every
    other test but silently feed the policy zero content features.
    """
    print(f"\nT2b: device-loop embeddings ({bench.name}) [the key distinction]")
    res = compile_loopcount(bench)
    parsed = parse_loopcount_output(res.stderr)
    all_rows = [row for fm in parsed.values() for df in fm.values()
                for _, row in df.iterrows()]
    if not all_rows or "emb0" not in all_rows[0].index:
        check("raw output has emb columns", False)
        return

    def nonzero(row) -> bool:
        return any(abs(float(row[c])) > 0 for c in _EMB_COLUMNS)

    dev = [r for r in all_rows if _is_device_row(r)]
    host = [r for r in all_rows if not _is_device_row(r)]
    dev_nz = sum(1 for r in dev if nonzero(r))
    host_nz = sum(1 for r in host if nonzero(r))
    print(f"  rows: {len(all_rows)} total | {len(dev)} device | {len(host)} host")
    print(f"  non-zero embeddings: {dev_nz}/{len(dev)} device | "
          f"{host_nz}/{len(host)} host")

    check("device loops exist in output", len(dev) > 0, f"{len(dev)}")
    check("EVERY device loop has a non-zero embedding",
          len(dev) > 0 and dev_nz == len(dev),
          f"{dev_nz}/{len(dev)}")
    # Distinctness: rule out every device loop getting the same degenerate vector.
    if len(dev) > 1:
        sigs = {tuple(round(float(r[c]), 5) for c in _EMB_COLUMNS) for r in dev}
        check("device embeddings are not all identical",
              len(sigs) > 1, f"{len(sigs)} distinct / {len(dev)} loops")


def test_kernel_embeddings(bench: Path) -> None:
    """
    kemb-specific coverage — runs ONLY when the kemb block is enabled
    (UU_FEATURE_BLOCKS contains kemb); otherwise SKIP, so the default emb run is
    unaffected.  Validates the load-bearing property that kemb is PER-KERNEL: kemb
    is one whole-__global__-kernel pool shared by every loop of that kernel, so
    every row with the same kernelParents MUST carry a byte-identical kemb.  Plus
    two non-degeneracy checks: distinct kernels get distinct kemb (rules out one
    constant vector), and kemb differs from the loop-local emb (rules out kemb
    being accidentally the loop pool).  Uses the RAW parse (all loops, pre-
    eligibility) so kernels expose as many loops as possible to the invariant.
    """
    import features
    if "kemb" not in features.ENABLED_BLOCKS:
        print("\nT6: kernel embeddings — SKIP (kemb not in UU_FEATURE_BLOCKS)")
        return
    print(f"\nT6: kernel embeddings ({bench.name}) [per-kernel invariant]")
    res = compile_loopcount(bench)
    parsed = parse_loopcount_output(res.stderr)
    all_rows = [row for fm in parsed.values() for df in fm.values()
                for _, row in df.iterrows()]
    if not all_rows or "kemb0" not in all_rows[0].index:
        check("raw output has kemb columns", False)
        return
    check("raw output has kemb columns", True)

    def kvec(r):
        return tuple(round(float(r[c]), 6) for c in _KEMB_COLUMNS)

    dev = [r for r in all_rows if _is_device_row(r)]
    dev_nz = sum(1 for r in dev if any(abs(float(r[c])) > 0 for c in _KEMB_COLUMNS))
    check("device loops exist in output", len(dev) > 0, f"{len(dev)}")
    check("EVERY device loop has a non-zero kemb",
          len(dev) > 0 and dev_nz == len(dev), f"{dev_nz}/{len(dev)}")

    # Per-kernel invariant: group device rows by kernelParents; kemb identical
    # within each group.  Only groups with >=2 loops actually exercise it — report
    # how many, so a vacuous pass (all kernels single-loop) is visible.
    groups: dict[str, list] = {}
    for r in dev:
        groups.setdefault(str(r.get("kernelParents", "")).strip(), []).append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    violated = [k for k, v in multi.items() if len({kvec(r) for r in v}) > 1]
    print(f"  kernel groups: {len(groups)} | with >=2 loops (exercise invariant): "
          f"{len(multi)}")
    check("shared-kernel invariant: same kernelParents => identical kemb",
          len(violated) == 0,
          f"{len(multi)} multi-loop kernels checked, {len(violated)} violated")
    if len(multi) == 0:
        print("  NOTE: no kernel has >=2 device loops here — invariant not "
              "exercised; run on a benchmark with a multi-loop kernel to test it.")

    # Non-degeneracy: distinct kernels -> distinct kemb (one per group is enough).
    if len(groups) > 1:
        sigs = {kvec(v[0]) for v in groups.values()}
        check("distinct kernels have distinct kemb (not one constant vector)",
              len(sigs) > 1, f"{len(sigs)} distinct kemb / {len(groups)} kernels")

    # kemb is the KERNEL pool, not a copy of the loop-local emb.
    if "emb0" in all_rows[0].index:
        def evec(r):
            return tuple(round(float(r[c]), 6) for c in _EMB_COLUMNS)
        differ = sum(1 for r in dev if kvec(r) != evec(r))
        check("kemb differs from loop emb (kernel context != loop content)",
              differ > 0, f"{differ}/{len(dev)} rows differ")


def test_pre_post_unmerge(bench: Path, file_map) -> None:
    print(f"\nT3: pre vs post-unmerge ({bench.name})")
    # pick a multi-path loop (unmerge actually restructures it)
    cand = None
    for fname, df in file_map.items():
        for _, row in df.iterrows():
            if int(row.get("numPaths", 1)) > 1:
                cand = (fname, row)
                break
        if cand:
            break
    if not cand:
        # NOT a failure: unmerge only restructures multi-path loops, and many
        # benchmarks (e.g. geglu — straight-line kernel bodies) have only
        # numPaths==1 eligible loops.  Those are eligible via containsBranch==1
        # and benefit from the unroll-only action, not unmerge.  T3 simply does
        # not apply here; run it on a multi-path benchmark (mandelbrot,
        # contract, bezier-surface) to exercise unmerge.
        print("  SKIP: no numPaths>1 eligible loop in this benchmark — unmerge "
              "does not apply (single-path loops use unroll-only). Not a failure.")
        return
    fname, pre_row = cand
    loop_idx = int(pre_row["loopIdx"])
    triple = "-"   # get_loop_features already filtered to device loops
    print(f"  using loop_idx={loop_idx} numPaths={int(pre_row['numPaths'])} in {fname}")

    # Re-extract post-unmerge: compile unmerge=1 factor=1 + loopcount (mirrors
    # GpuLoopEnv.get_post_unmerge_features) and re-parse.
    from hecbench import ARCH as _ARCH
    cflags = _build_extra_cflags(
        enable_uu=True, enable_loopcount=True, filename=fname, triple=triple,
        loop_indices=[loop_idx], unmerge_flags=[1], unroll_factors=[1],
    )
    res = _make(bench, extra_cflags=cflags, arch=_ARCH)
    parsed = parse_loopcount_output(res.stderr)
    post_row = None
    for _t, fm in parsed.items():
        for _f, pdf in fm.items():
            m = pdf[pdf["loopIdx"] == loop_idx]
            if not m.empty:
                post_row = m.iloc[0]
                break
        if post_row is not None:
            break
    if post_row is None:
        check("post-unmerge row recovered", False, "loop_idx not found post-compile")
        return
    check("post-unmerge row recovered", True)

    pre_full = _row_to_tensor(pre_row).tolist()
    post_full = _row_to_tensor(post_row).tolist()
    check(f"FULL {len(pre_full)}-dim vector changes under unmerge",
          pre_full != post_full, f"L2={l2(pre_full, post_full):.4f}")

    emb_delta = l2(emb_subvector(pre_row), emb_subvector(post_row))
    struct_delta = l2(pre_full[:18], post_full[:18])
    # REPORT, not assert — see module docstring.
    print(f"  [report] structural-subvector L2 delta = {struct_delta:.4f}")
    print(f"  [report] embedding-subvector  L2 delta = {emb_delta:.4f}")
    print(f"  [report] {'embedding moved under unmerge' if emb_delta > 1e-4 else 'embedding ~unchanged (mean-composition insensitivity — a finding, not a bug)'}")


def _gather_loops(disc: dict, min_loops: int = 20, max_benches: int = 6,
                  prefer: str = ""):
    """
    Pool eligible-loop rows across benchmarks until >= min_loops (or max_benches
    scanned).  Single-loop benchmarks (e.g. mandelbrot) give a degenerate
    normalizer/dedup view; pooling mirrors what the real precheck fits on.
    Returns (rows, names_scanned).
    """
    rows: list = []
    scanned: list = []
    names = list(disc)
    if prefer in disc:                      # scan the preferred one first
        names = [prefer] + [n for n in names if n != prefer]
    for name in names:
        try:
            fm, _, _ = get_loop_features(disc[name])
        except Exception:
            continue
        for df in fm.values():
            for _, r in df.iterrows():
                rows.append(r)
        scanned.append(name)
        if len(rows) >= min_loops or len(scanned) >= max_benches:
            break
    return rows, scanned


def test_dedup_delta(disc: dict, prefer: str) -> None:
    print(f"\nT4: dedup delta [informational]")
    rows, scanned = _gather_loops(disc, min_loops=30, prefer=prefer)
    print(f"  pooled {len(rows)} loops across {scanned}")
    if not rows:
        check("pooled loops available", False)
        return
    full = [tuple(_row_to_tensor(r).tolist()) for r in rows]
    struct = [tuple(_row_to_tensor(r).tolist()[:18]) for r in rows]
    full_dim = len(FEATURE_COLUMNS)
    print(f"  unique@18-dim: {len(set(struct))}  unique@{full_dim}-dim: {len(set(full))}")
    check(f"{full_dim}-dim disambiguates >= 18-dim (fewer or equal dups)",
          len(set(full)) >= len(set(struct)),
          f"{len(set(full))} >= {len(set(struct))} unique")


def test_normalizer(disc: dict) -> None:
    print(f"\nT5: normalizer")
    from hecbench import FeatureNormalizer
    rows, scanned = _gather_loops(disc, min_loops=20)
    print(f"  pooled {len(rows)} loops across {scanned}")
    if len(rows) < 2:
        check("enough loops to compute std (>=2)", False,
              f"only {len(rows)} — pass benchmarks with more loops")
        return
    tensors = [_row_to_tensor(r) for r in rows]
    n = FeatureNormalizer()
    n.fit(tensors)
    check(f"normalizer mean length == {len(FEATURE_COLUMNS)}",
          len(n.mean) == len(FEATURE_COLUMNS), f"{len(n.mean)}")
    emb_std = n.std[18:].tolist()
    check("some embedding dims have std > 0", any(s > 1e-6 for s in emb_std),
          f"max emb std = {max(emb_std):.4g}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="mandelbrot-cuda")
    p.add_argument("--template-benchmark", default="sortKV-cuda")
    p.add_argument("--hecbench-src", default=None,
                   help="Benchmark tree to search. MUST be the same tree the "
                        "training run used (--hecbench-src there). The default "
                        "og-HeCBench holds only the ~16 original benchmarks — "
                        "e.g. geglu-cuda lives only in the larger upstream tree.")
    args = p.parse_args()

    from hecbench import discover_benchmarks, HECBENCH_SRC as _DEFAULT_SRC
    HECBENCH_SRC = Path(args.hecbench_src) if args.hecbench_src else _DEFAULT_SRC
    print(f"benchmark tree: {HECBENCH_SRC}")
    disc = {b.name: b for b in discover_benchmarks(HECBENCH_SRC)}
    bench = disc.get(args.benchmark)
    if bench is None:
        # This test only compiles + extracts features (never runs the binary),
        # so a benchmark that fails discover_benchmarks' runtime filters
        # (no run: target, external ../data/, DVC) is still usable here.
        cand = HECBENCH_SRC / args.benchmark
        if cand.is_dir() and (cand / "Makefile").exists():
            print(f"note: {args.benchmark} not in discover_benchmarks (likely no "
                  f"run: target / external data) — using on-disk dir (compile-only test)")
            bench = cand
            disc.setdefault(args.benchmark, cand)
        else:
            print(f"benchmark {args.benchmark!r} not found under {HECBENCH_SRC}")
            print(f"  note: the flag is --benchmark (singular), not --benchmarks")
            print(f"  note: pass --hecbench-src <tree> if geglu etc. live in a "
                  f"different tree than the default {_DEFAULT_SRC}")
            print(f"  available (first 20): {sorted(disc)[:20]}")
            sys.exit(1)

    test_schema()
    fm = test_extraction(bench)
    test_device_embeddings(bench)          # the key device-vs-host check
    test_kernel_embeddings(bench)          # kemb per-kernel invariant (skips if off)
    if fm:
        test_pre_post_unmerge(bench, fm)
    # T5/T4 pool loops across benchmarks — a single-loop benchmark gives a
    # degenerate (std=0) normalizer/dedup view.
    test_normalizer(disc)
    test_dedup_delta(disc, prefer=args.template_benchmark)

    passed = sum(1 for _, ok in _results if ok)
    print(f"\n{passed}/{len(_results)} checks passed")
    sys.exit(0 if passed == len(_results) else 1)


if __name__ == "__main__":
    main()
