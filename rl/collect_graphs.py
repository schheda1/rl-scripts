"""
Stage 4 of the ProGraML track: per-loop graph feature RECORDS for every eligible
loop, pre-collected alongside the IR2Vec feature vectors so training/evaluation
can run offline.

Per benchmark, through the collector's OWN compile path (hecbench._make, same
flags as the eligibility precheck, in a scratch copy of the benchmark):

  stamp+dump   --enable-loopcount -loopcount-stamp-loops  -loopcount-emit-ir=<dir>/
               LoopCount stamps every in-loop instruction with its loopIdx set
               (!loopcount.loops) and the dump pass writes the EARLY, stamped IR
               once per module as <dir>/<source basename>.<triple>.ll — host and
               device modules, every translation unit, every program variant get
               their own file (the last compile of a file+triple wins, exactly as
               the CSV parser keeps the last block), so a full `make` is safe.
  graph        llvm2graph --stdout_fmt=json <device .ll>  ->  graphs/<bench>/<file>.<triple>.json.gz
  vocab        scheme C: one frozen token vocabulary over the WHOLE corpus
               (benchmarks in sorted order => reproducible ids) -> vocab_C.json
  records      pgraph.emit.records_for_loop per eligible loop (loop graph + the
               kernel-context graph(s)), keyed (benchmark, arch, loop_idx) so they
               join reward_cache.json -> records/<bench>.json.gz

The eligible loops, their file, is_kernel and kernel parents come from the
checkpoint's eligible_benchmarks.json — the same records the collectors use —
so a graph record exists for exactly the loops that have reward cells.

Everything here is compile-only and CPU-bound: it runs alongside a GPU collection.
Resumable: a benchmark whose graphs already exist is not recompiled (--force).

Usage:
  python3 collect_graphs.py CKPT_DIR --dry-run
  python3 collect_graphs.py CKPT_DIR --llvm2graph $CGO/ProGraML/build/llvm2graph --workers 8
  python3 collect_graphs.py CKPT_DIR --scheme A --inst2vec-dict <dict.pickle> ...
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# NOTE: hecbench / features (torch) and the pgraph modules are imported lazily
# inside the phases, so this module's helpers are importable and testable without
# torch and without a compiler.

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("graphs")

DEVICE_TRIPLE_MARKERS = ("nvptx", "amdgcn")
STAMP_FLAG = "-mllvm -loopcount-stamp-loops"
DEFAULT_LLVM2GRAPH = Path(__file__).resolve().parents[2] / "ProGraML" / "build" / "llvm2graph"


# ---------------------------------------------------------------------------
# Pure helpers (tested without a compiler)
# ---------------------------------------------------------------------------

def is_device_triple(triple: str) -> bool:
    """MIRROR: hecbench.get_loop_features' device-triple preference."""
    t = triple.lower()
    return any(m in t for m in DEVICE_TRIPLE_MARKERS)


def parse_dump_name(name: str) -> tuple[str, str]:
    """'<source basename>.<triple>.ll' (the dump pass's per-module name) ->
    (source basename, triple).  The basename may contain dots (main.cu); a
    triple never does, so split from the right."""
    if name.endswith(".ll"):
        name = name[:-3]
    base, sep, triple = name.rpartition(".")
    if not sep:
        raise ValueError(f"not a per-module dump name: {name!r}")
    return base, triple


def graph_file(out: Path, bench: str, base: str, triple: str) -> Path:
    return out / "graphs" / bench / f"{base}.{triple}.json.gz"


def device_graphs_for(out: Path, bench: str, filename: str) -> list[Path]:
    """Graph files of *filename*'s DEVICE module(s) (basename match: the CSV's
    filename may carry a directory prefix, the dump name never does)."""
    base = Path(filename).name
    gdir = out / "graphs" / bench
    outp = []
    for p in sorted(gdir.glob(f"{base}.*.json.gz")) if gdir.is_dir() else []:
        _b, triple = parse_dump_name(p.name[:-len(".json.gz")])
        if is_device_triple(triple):
            outp.append(p)
    return outp


def read_gz_text(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return fh.read()


def write_gz_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Phase 1: stamp + dump + graph (parallel over benchmarks, compile-only)
# ---------------------------------------------------------------------------

def compile_diagnosis(stderr: str, maxlen: int = 240) -> str:
    """The one line of compiler output that explains an empty dump: a LOOPCOUNT
    WARNING first (e.g. a clang without the per-module mode cannot open '<dir>/'
    as a file), else the first error line, else the last non-empty line."""
    lines = [l.strip() for l in (stderr or "").splitlines() if l.strip()]
    for l in lines:
        if "LOOPCOUNT WARNING" in l:
            return l[:maxlen]
    for l in lines:
        if "error" in l.lower():
            return l[:maxlen]
    return lines[-1][:maxlen] if lines else "no compiler output"


def dump_benchmark(job: dict):
    """One benchmark: scratch copy -> _make with stamp+dump flags -> llvm2graph on
    each device module -> graphs/<bench>/.  Returns (name, status, [(base, triple)])."""
    name, out = job["name"], Path(job["out"])
    gdir = out / "graphs" / name
    if not job["force"] and gdir.is_dir() and any(gdir.glob("*.json.gz")):
        have = [parse_dump_name(p.name[:-len(".json.gz")]) for p in sorted(gdir.glob("*.json.gz"))]
        return name, "cached", have
    from hecbench import _build_extra_cflags, _make
    from store_common import bundle_paths, structural_copy
    broot, bdir = bundle_paths(job["scratch"], name, name)   # <scratch>/<bench>/<bench>:
    # ABSOLUTE: the flag is consumed by clang running with cwd = the bundle dir,
    # so a relative path (relative --scratch or checkpoint dir) would resolve
    # inside the bundle and vanish with it.
    irdir = (broot / "ir").resolve()                           # -hip sibling symlink works
    try:
        shutil.rmtree(broot, ignore_errors=True)
        structural_copy(Path(job["path"]), bdir, job["copy_threshold"])
        irdir.mkdir(parents=True, exist_ok=True)
        cflags = (_build_extra_cflags(enable_loopcount=True)
                  + f" {STAMP_FLAG} -mllvm -loopcount-emit-ir={irdir}/")
        # Return code deliberately not checked: the stamped dump and the CSV are
        # produced at the compile stage, valid even when the link fails — the
        # eligibility precheck treats them the same way.
        r = _make(bdir, extra_cflags=cflags, arch=job["arch"], timeout=job["compile_timeout"])
        dumps = []
        for ll in sorted(irdir.glob("*.ll")):
            base, triple = parse_dump_name(ll.name)
            if is_device_triple(triple):
                dumps.append((base, triple, ll))
        if not dumps:
            return name, f"no device module dumped: {compile_diagnosis(r.stderr)}", []
        done = []
        for base, triple, ll in dumps:
            r = subprocess.run([job["llvm2graph"], "--stdout_fmt=json", str(ll)],
                               capture_output=True, text=True, timeout=job["graph_timeout"])
            if r.returncode != 0 or not r.stdout.strip():
                log.warning("%s: llvm2graph failed on %s (%s): %s", name, ll.name,
                            r.returncode, (r.stderr or "")[-300:].strip())
                continue
            write_gz_text(graph_file(out, name, base, triple), r.stdout)
            done.append((base, triple))
        return name, ("ok" if done else "llvm2graph failed on every device module"), done
    except subprocess.TimeoutExpired:
        return name, "timeout", []
    except Exception as e:          # noqa: BLE001 — one bad benchmark must not stop the corpus
        return name, f"error: {e}", []
    finally:
        shutil.rmtree(broot, ignore_errors=True)


# ---------------------------------------------------------------------------
# Phase 2: the frozen vocabulary (scheme C), serial, corpus-wide
# ---------------------------------------------------------------------------

def build_corpus_vocab(out: Path, benches: list[str], vocab_path: Path):
    from pgraph.loader import load_graph
    from pgraph.vectorize import build_vocab

    def graphs():
        for b in sorted(benches):                       # fixed order => reproducible ids
            gdir = out / "graphs" / b
            for p in sorted(gdir.glob("*.json.gz")) if gdir.is_dir() else []:
                _base, triple = parse_dump_name(p.name[:-len(".json.gz")])
                if is_device_triple(triple):
                    yield load_graph(read_gz_text(p))

    vocab = build_vocab(graphs())
    vocab.to_json(str(vocab_path))
    return vocab.size


# ---------------------------------------------------------------------------
# Phase 3: records per eligible loop (parallel over benchmarks)
# ---------------------------------------------------------------------------

def records_benchmark(job: dict):
    """One benchmark: for each eligible loop record, slice+vectorize its loop graph
    and kernel-context graph(s).  Returns (name, n_records, [(loop_idx, why)])."""
    name, out, scheme = job["name"], Path(job["out"]), job["scheme"]
    from pgraph.emit import records_for_loop
    from pgraph.loader import load_graph
    from pgraph.slicer import loop_seed
    from pgraph.vectorize import Vocab, compute_tokens
    vocab = Vocab.from_json(job["vocab"]) if scheme == "C" else None
    a_dict = None
    if scheme == "A":
        from pgraph.scheme_a import Inst2vecDict
        a_dict = Inst2vecDict.from_pickle(job["inst2vec_dict"])
    loaded: dict = {}                       # graph file -> (graph, tokens, a_map)
    recs_out: dict = {}
    missing: list = []
    for r in job["records"]:
        idx = int(r.get("loop_idx", -1))
        # Per-LOOP isolation: one loop that fails to slice/vectorize is reported
        # as missing; the benchmark's other records are still written.
        try:
            gfiles = device_graphs_for(out, name, r["filename"])
            if not gfiles:
                missing.append((idx, f"no device graph for {r['filename']}"))
                continue
            gp = gfiles[0]                  # one device module per file (multi-arch dumps never coexist)
            if gp not in loaded:
                g = load_graph(read_gz_text(gp))
                tokens = compute_tokens(g) if scheme == "C" else None
                a_map = None
                if scheme == "A":
                    from pgraph.scheme_a import encode_scheme_a
                    a_map = encode_scheme_a(g, a_dict)["embedding_index"]
                loaded[gp] = (g, tokens, a_map)
            g, tokens, a_map = loaded[gp]
            if not loop_seed(g, idx):
                missing.append((idx, "no instruction stamped with this loopIdx in the graph"))
                continue
            rec = records_for_loop(
                g, idx, scheme=scheme, benchmark=name, arch=job["arch"],
                is_kernel=bool(r.get("is_kernel_function", False)),
                parent_names=list(r.get("kernel_parents", [])),
                vocab=vocab, tokens=tokens, a_map=a_map)
            recs_out[str(idx)] = rec
        except Exception as e:          # noqa: BLE001
            missing.append((idx, f"error: {e}"))
    try:
        if recs_out:
            write_gz_text(out / "records" / f"{name}.json.gz", json.dumps(recs_out))
    except Exception as e:              # noqa: BLE001
        return name, 0, [(-1, f"write error: {e}")]
    return name, len(recs_out), missing


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint_dir", type=Path)
    p.add_argument("--out", type=Path, default=None, help="default CKPT/pgraph")
    p.add_argument("--llvm2graph", type=Path,
                   default=Path(os.environ.get("LLVM2GRAPH", str(DEFAULT_LLVM2GRAPH))))
    p.add_argument("--scheme", choices=["C", "A"], default="C")
    p.add_argument("--inst2vec-dict", type=Path, default=None, help="scheme A only")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--benchmarks", nargs="+", default=None)
    p.add_argument("--force", action="store_true", help="recompile even if graphs exist")
    p.add_argument("--compile-timeout", type=int, default=900)
    p.add_argument("--graph-timeout", type=int, default=600, help="per llvm2graph run")
    p.add_argument("--copy-threshold", type=int, default=1_000_000)
    p.add_argument("--scratch", type=Path, default=None, help="default CKPT/pgraph_tmp")
    p.add_argument("--arch", default=None)
    p.add_argument("--hecbench-src", default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    from features import FEATURES_VERSION
    from hecbench import ARCH, HECBENCH_SRC, IR2VEC_VOCAB, IS_HIP
    arch = a.arch or ARCH
    if str(arch).startswith("gfx") != IS_HIP:
        sys.exit(f"--arch {arch} disagrees with TARGET_ARCH={ARCH!r}; export TARGET_ARCH instead")
    if not IR2VEC_VOCAB or not Path(IR2VEC_VOCAB).exists():
        sys.exit("IR2VEC_VOCAB unset/missing — every --enable-loopcount compile needs it")
    if a.scheme == "A" and (a.inst2vec_dict is None or not a.inst2vec_dict.exists()):
        sys.exit("--scheme A needs --inst2vec-dict <dictionary pickle>")

    ckpt = a.checkpoint_dir
    elig = ckpt / "eligible_benchmarks.json"
    if not elig.exists():
        sys.exit(f"missing {elig}")
    data = json.loads(elig.read_text())
    loop_records_map = data["loop_records"]
    src = Path(a.hecbench_src) if a.hecbench_src else HECBENCH_SRC
    # Absolute paths throughout: the compile runs with cwd = the bundle dir.
    out = Path(a.out or (ckpt / "pgraph")).resolve()
    scratch = Path(a.scratch or (ckpt / "pgraph_tmp")).resolve()

    names = sorted(a.benchmarks or loop_records_map)
    jobs = []
    for n in names:
        if n not in loop_records_map or not loop_records_map[n]:
            log.warning("%s: not in the eligible set — skipped", n); continue
        if not (src / n).is_dir():
            log.warning("%s: no source dir under %s — skipped", n, src); continue
        jobs.append(n)
    n_loops = sum(len(loop_records_map[n]) for n in jobs)
    log.info("benchmarks=%d eligible loops=%d arch=%s scheme=%s llvm2graph=%s out=%s",
             len(jobs), n_loops, arch, a.scheme, a.llvm2graph, out)
    if not a.llvm2graph.exists():
        msg = f"llvm2graph not found at {a.llvm2graph} (build ProGraML or pass --llvm2graph)"
        if a.dry_run:
            log.warning(msg)
        else:
            sys.exit(msg)
    if a.dry_run:
        cached = sum(1 for n in jobs if (out / "graphs" / n).is_dir()
                     and any((out / "graphs" / n).glob("*.json.gz")))
        log.info("dry run: %d benchmarks to compile (%d already have graphs), then vocab, "
                 "then %d loop records.  Nothing written.", len(jobs) - cached, cached, n_loops)
        return 0

    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- phase 1 ---
    p1 = [{"name": n, "path": str(src / n), "out": str(out), "scratch": str(scratch),
           "arch": arch, "llvm2graph": str(a.llvm2graph), "force": a.force,
           "compile_timeout": a.compile_timeout, "graph_timeout": a.graph_timeout,
           "copy_threshold": a.copy_threshold} for n in jobs]
    status: dict = {}
    with Pool(a.workers) as pool:
        for name, st, dumps in pool.imap_unordered(dump_benchmark, p1):
            status[name] = (st, dumps)
            log.info("  graphs %-32s %s %s", name, st, [f"{b}.{t}" for b, t in dumps])
    good = [n for n, (st, d) in status.items() if st in ("ok", "cached") and d]
    log.info("phase 1: %d/%d benchmarks have device graphs (%.1f min)",
             len(good), len(jobs), (time.time() - t0) / 60)

    # --- phase 2 ---
    vocab_path = out / "vocab_C.json"
    vsize = None
    if a.scheme == "C":
        vsize = build_corpus_vocab(out, good, vocab_path)
        log.info("phase 2: vocab of %d tokens -> %s", vsize, vocab_path)

    # --- phase 3 ---
    p3 = [{"name": n, "out": str(out), "scheme": a.scheme, "arch": arch,
           "vocab": str(vocab_path), "inst2vec_dict": str(a.inst2vec_dict) if a.inst2vec_dict else None,
           "records": loop_records_map[n]} for n in good]
    n_rec, missing_all = 0, {}
    with Pool(a.workers) as pool:
        for name, n, missing in pool.imap_unordered(records_benchmark, p3):
            n_rec += n
            if missing:
                missing_all[name] = missing
            log.info("  records %-32s %d loops%s", name, n,
                     f"  ({len(missing)} missing)" if missing else "")

    manifest = {
        "arch": arch, "scheme": a.scheme, "features_version": FEATURES_VERSION,
        "llvm2graph": str(a.llvm2graph), "n_benchmarks": len(good), "n_records": n_rec,
        "n_eligible_loops": n_loops, "vocab_size": vsize,
        "benchmark_status": {n: st for n, (st, _d) in status.items()},
        "missing": {n: [[i, why] for i, why in m] for n, m in missing_all.items()},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("=== done in %.1f min: %d records for %d of %d eligible loops; %d benchmarks "
             "without graphs; manifest -> %s ===", (time.time() - t0) / 60, n_rec, n_rec,
             n_loops, len(jobs) - len(good), out / "manifest.json")
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
