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


def emb_subvector(row) -> list[float]:
    return [float(row[c]) for c in _EMB_COLUMNS]


def l2(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def test_schema() -> None:
    print("\nT1: schema")
    check("FEATURE_COLUMNS length == 93", len(FEATURE_COLUMNS) == 93,
          f"{len(FEATURE_COLUMNS)}")
    check("agent.N_FEATURES == 93", agent.N_FEATURES == 93, f"{agent.N_FEATURES}")
    check("trip-count indices unmoved (10,11)",
          FEATURE_COLUMNS[10:12] == ["tripCountKnown", "tripCount"],
          str(FEATURE_COLUMNS[10:12]))
    check("emb block appended (18=emb0 .. 92=emb74)",
          FEATURE_COLUMNS[18] == "emb0" and FEATURE_COLUMNS[92] == "emb74",
          f"{FEATURE_COLUMNS[18]}..{FEATURE_COLUMNS[92]}")
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
        check("found a numPaths>1 loop", False, "none — skipping T3")
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
    check("FULL 93-dim vector changes under unmerge",
          pre_full != post_full, f"L2={l2(pre_full, post_full):.4f}")

    emb_delta = l2(emb_subvector(pre_row), emb_subvector(post_row))
    struct_delta = l2(pre_full[:18], post_full[:18])
    # REPORT, not assert — see module docstring.
    print(f"  [report] structural-subvector L2 delta = {struct_delta:.4f}")
    print(f"  [report] embedding-subvector  L2 delta = {emb_delta:.4f}")
    print(f"  [report] {'embedding moved under unmerge' if emb_delta > 1e-4 else 'embedding ~unchanged (mean-composition insensitivity — a finding, not a bug)'}")


def test_dedup_delta(template_bench: Path) -> None:
    print(f"\nT4: dedup delta ({template_bench.name}) [informational]")
    file_map, _, _ = get_loop_features(template_bench)
    rows = [row for df in file_map.values() for _, row in df.iterrows()]
    if not rows:
        check("template benchmark has loops", False)
        return
    full = [tuple(_row_to_tensor(r).tolist()) for r in rows]
    struct = [tuple(_row_to_tensor(r).tolist()[:18]) for r in rows]
    print(f"  loops: {len(rows)}  unique@18-dim: {len(set(struct))}  "
          f"unique@93-dim: {len(set(full))}")
    check("93-dim disambiguates >= 18-dim (fewer or equal dups)",
          len(set(full)) >= len(set(struct)),
          f"{len(set(full))} >= {len(set(struct))} unique")


def test_normalizer(bench: Path) -> None:
    print(f"\nT5: normalizer ({bench.name})")
    from hecbench import FeatureNormalizer
    file_map, _, _ = get_loop_features(bench)
    tensors = [_row_to_tensor(r) for df in file_map.values() for _, r in df.iterrows()]
    n = FeatureNormalizer()
    n.fit(tensors)
    check("normalizer mean length == 93", len(n.mean) == 93, f"{len(n.mean)}")
    emb_std = n.std[18:].tolist()
    check("some embedding dims have std > 0", any(s > 1e-6 for s in emb_std))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="mandelbrot-cuda")
    p.add_argument("--template-benchmark", default="sortKV-cuda")
    args = p.parse_args()

    from hecbench import discover_benchmarks, HECBENCH_SRC
    disc = {b.name: b for b in discover_benchmarks(HECBENCH_SRC)}
    bench = disc.get(args.benchmark)
    if bench is None:
        print(f"benchmark {args.benchmark} not found"); sys.exit(1)

    test_schema()
    fm = test_extraction(bench)
    test_device_embeddings(bench)          # the key device-vs-host check
    if fm:
        test_pre_post_unmerge(bench, fm)
    test_normalizer(bench)
    tb = disc.get(args.template_benchmark)
    if tb:
        test_dedup_delta(tb)
    else:
        print(f"\nT4: skipped — {args.template_benchmark} not found")

    passed = sum(1 for _, ok in _results if ok)
    print(f"\n{passed}/{len(_results)} checks passed")
    sys.exit(0 if passed == len(_results) else 1)


if __name__ == "__main__":
    main()
