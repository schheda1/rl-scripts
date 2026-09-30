"""
Shared, stdlib-only utilities for the decoupled collector and the pre-compile store.

Kept dependency-free (no torch / hecbench) and as the LEAF module so both
collect_pipeline.py and precompile.py import from here without circular imports or
pulling torch into CPU workers.  Holds: the cell key, the structural bundle copy, and
the pre-compiled-binary store (write/stage/verify) with its fail-safe stamps.
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

BUILD_EXTS = {".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hpp", ".hxx", ".cuh",
              ".inc", ".mk", ".mak", ".make"}


# ---------------------------------------------------------------------------
# Cell key + bundle staging
# ---------------------------------------------------------------------------

def cellkey(bench: str, loop_idx: int, unmerge: int, factor: int) -> str:
    return f"{bench}|{loop_idx}|{unmerge}|{factor}"


def sanitize(key: str) -> str:
    return key.replace("|", "__")


def _is_source(path: Path) -> bool:
    """A compilable build input, by extension only (threshold-independent)."""
    return path.suffix.lower() in BUILD_EXTS or path.name == "Makefile"


def _is_build_input(path: Path, threshold: int) -> bool:
    """Copy (not symlink) build inputs: known source/build extensions OR anything small.
    Large data blobs are symlinked so shm stays tiny and `make` never writes through a
    symlink into the shared template."""
    if _is_source(path):
        return True
    try:
        return path.stat().st_size <= threshold
    except OSError:
        return True


def structural_copy(src: Path, dst: Path, threshold: int) -> None:
    """Copy small/build-input files, symlink large data files (to the resolved original),
    recursing into subdirs.  make writes .o/binary as NEW files in dst; symlinked inputs
    are read-only, so `make clean` in dst is confined to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        d = dst / item.name
        if item.is_symlink():
            os.symlink(os.path.realpath(item), d)
        elif item.is_dir():
            structural_copy(item, d, threshold)
        elif _is_build_input(item, threshold):
            shutil.copy2(item, d)
        else:
            os.symlink(item.resolve(), d)


def _proc_children_map() -> dict:
    """ppid -> [pid] for every process, read from /proc (Linux).  Returns {} where /proc
    is absent (e.g. macOS) — callers then fall back to killpg alone."""
    kids: dict = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return kids
    for e in entries:
        if not e.isdigit():
            continue
        try:
            with open(f"/proc/{e}/stat") as fh:
                data = fh.read()
        except OSError:
            continue
        # comm (in parens) may contain spaces/parens; per proc(5) the fields AFTER the
        # last ')' are: state ppid ...  → ppid is the 2nd token there.
        rp = data.rfind(")")
        rest = data[rp + 1:].split() if rp >= 0 else []
        if len(rest) < 2:
            continue
        try:
            kids.setdefault(int(rest[1]), []).append(int(e))
        except ValueError:
            continue
    return kids


def _descendants(pid: int, kids: dict) -> list:
    """All transitive descendants of pid, via the ppid map (BFS)."""
    out, stack = [], [pid]
    while stack:
        p = stack.pop()
        for c in kids.get(p, []):
            out.append(c)
            stack.append(c)
    return out


def _pids_with_cwd(target) -> list:
    """PIDs whose working directory is `target` (Linux /proc).  Last-resort reaper: kills a
    run's benchmark by the UNIQUE dir it runs in, independent of process group or parent —
    catches a target a profiler reparented to init.  target MUST be per-run unique (a cell
    bundle), else it would match unrelated processes that share a cwd."""
    try:
        tgt = os.path.realpath(target)
    except OSError:
        return []
    out = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for e in entries:
        if not e.isdigit():
            continue
        try:
            if os.path.realpath(f"/proc/{e}/cwd") == tgt:
                out.append(int(e))
        except OSError:
            continue
    return out


def run_hard_timeout(cmd: str, cwd, timeout: int, env: dict,
                     capture: bool = False, scope_dir=None) -> "tuple[bool, str]":
    """Run a shell command with a HARD timeout that kills the WHOLE process tree.

    Why not subprocess.run(timeout=): with shell=True the direct child is /bin/sh, and
    `nsys profile ./main` runs nsys + the benchmark as its GRANDCHILDREN.  subprocess.run's
    timeout kills only the shell, so on timeout nsys and the benchmark are ORPHANED — they
    keep running on the GPU, pile up across the n_runs loop, and (fatally for the metric)
    execute CONCURRENTLY with the next profiled run on the same GPU, corrupting its kernel
    timings.

    Why not killpg alone: nsys launches the benchmark in its OWN process group (observed:
    with killpg deployed, ./main still orphaned on the GPU whose cell kept timing out).
    killpg(shell_group) then reaps nsys but leaves ./main running.  So on timeout we kill,
    in order, three overlapping supersets — all snapshotted BEFORE killing so a
    reparented/regrouped child is still discoverable:
      1. killpg(our group)                 — the shell + anything still in its group,
      2. ppid descendants of our child     — ./main is nsys's child even in nsys's group,
      3. procs whose cwd == scope_dir      — ./main runs in its bundle even if reparented.
    scope_dir MUST be per-run unique (a cell bundle); omit it when cwd is shared (e.g. the
    CPU-side `nsys stats`, whose cwd is the worker dir) or it would over-match.

    Returns (ok, output): ok=True on clean exit, False on timeout/kill.  output is merged
    stdout+stderr when capture=True, else "".  On timeout nsys is killed before it finalizes
    the report, so the caller sees no .nsys-rep and skips that run (unchanged semantics)."""
    kw = dict(cwd=str(cwd), shell=True, env=env, start_new_session=True)
    if capture:
        kw.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    else:
        kw.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc = subprocess.Popen(cmd, **kw)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return True, (out or "")
    except subprocess.TimeoutExpired:
        victims = set(_descendants(proc.pid, _proc_children_map()))   # snapshot tree first
        if scope_dir:
            victims.update(_pids_with_cwd(scope_dir))
        victims.discard(os.getpid())                          # never signal ourselves
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)   # group sweep
        except OSError:
            pass                                              # already gone / no perms
        for v in victims:                                     # tree + cwd sweeps
            try:
                os.kill(v, signal.SIGKILL)
            except OSError:
                pass
        try:
            proc.communicate(timeout=30)                      # reap our direct child
        except Exception:
            pass
        return False, ""


def dir_bytes(root: Path) -> int:
    """Real bytes under root (symlinks counted as 0 — their targets live elsewhere)."""
    total = 0
    for dp, _dns, fns in os.walk(root):
        for f in fns:
            fp = os.path.join(dp, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Toolchain stamp + benchmark fingerprint (fail-safe guards for the store)
# ---------------------------------------------------------------------------

def toolchain_stamp(arch: str) -> dict:
    """Identity the store must match to be trusted: arch (fatbin) + clang build (codegen).
    A store built under a different clang/arch would give plausible-but-wrong rewards, so
    the consumer refuses a mismatched stamp and rebuilds instead."""
    def _first_line(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            for ln in (r.stdout + r.stderr).splitlines():
                if ln.strip():
                    return ln.strip()
        except Exception:
            pass
        return "?"
    return {"arch": arch,
            "clang": _first_line(["clang++", "--version"]),
            "cuda_home": os.environ.get("CUDA_HOME", "/usr/local/cuda")}


def benchmark_fingerprint(bench_dir: Path) -> str:
    """CONTENT hash of a benchmark's SOURCE (relpath + bytes), so a source edit between
    pre-compile and job invalidates its stored binaries (→ rebuild).  Content-based (not
    mtime/size) so it is IDENTICAL across machines — pre-compile runs on a different node
    than the job.  Source is by EXTENSION only, so the fingerprint is independent of
    --copy-threshold (which must otherwise match between the two tools)."""
    h = hashlib.md5()
    files = []
    for dp, _dns, fns in os.walk(bench_dir):
        for f in fns:
            p = Path(dp) / f
            if p.is_symlink() or not _is_source(p):
                continue
            files.append((os.path.relpath(p, bench_dir), p))
    for rel, p in sorted(files):
        h.update(rel.encode())
        h.update(b"\0")
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"?")
        h.update(b"\0")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Pre-compiled binary store
#   store/<sanitized cellkey>/<target>   the built executable
#   store/<sanitized cellkey>/meta.json  {"target","bench","fp"}
#   store/<sanitized cellkey>/.done      marker, written LAST (⇒ binary complete)
#   store/manifest.json                  {"stamps","benchmarks":{bench:fp},"cells":[key]}
# ---------------------------------------------------------------------------

def cell_dir(store: str, key: str) -> Path:
    return Path(store) / sanitize(key)


def is_cell_done(store: str, key: str) -> bool:
    return (cell_dir(store, key) / ".done").exists()


def write_cell(store: str, key: str, binary_src: Path, target_rel: str,
               bench: str, fp: str, filename: str = "", triple: str = "") -> None:
    """Store a built binary atomically, then the .done marker LAST. A reader that sees
    .done is guaranteed a complete binary (no truncated-binary → garbage-reward path).
    filename/triple are recorded so --verify-store can rebuild with the SAME UU flag
    targeting (they select which file's loops get the -mllvm flags)."""
    d = cell_dir(store, key)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / target_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(binary_src, tmp)
    os.chmod(tmp, 0o755)
    os.replace(tmp, dst)                       # atomic
    (d / "meta.json").write_text(json.dumps({"target": target_rel, "bench": bench,
                                             "fp": fp, "filename": filename,
                                             "triple": triple}))
    (d / ".done").write_text("1")              # marker LAST


def stage_from_store(store: str, key: str, bundle: Path) -> bool:
    """Drop the pre-built binary into an already-structural_copied bundle. Unlinks any
    existing target FIRST so we never write THROUGH a symlink into the shared benchmark
    tree.  Returns False (→ caller builds) if the store entry is absent/incomplete."""
    d = cell_dir(store, key)
    if not (d / ".done").exists() or not (d / "meta.json").exists():
        return False
    try:
        meta = json.loads((d / "meta.json").read_text())
    except Exception:
        return False
    target_rel = meta.get("target")
    if not target_rel or os.path.isabs(target_rel):   # absolute can't be relocated safely
        return False
    src_bin = d / target_rel
    if not src_bin.exists():
        return False
    tgt = Path(bundle) / target_rel
    tgt.parent.mkdir(parents=True, exist_ok=True)
    if tgt.exists() or tgt.is_symlink():
        os.remove(tgt)                         # unlink the LINK, never its target
    shutil.copy2(src_bin, tgt)
    os.chmod(tgt, 0o755)
    return True


def run_target_rel(run_cmd: str) -> str:
    """The executable path a run_cmd invokes, relative — e.g. './main x' -> 'main',
    './sub/foo' -> 'sub/foo'.  Same run_cmd the job derives via _get_run_command, so the
    stored binary lands exactly where exec will look."""
    tok = run_cmd.strip().split()[0]
    if tok.startswith("./"):
        tok = tok[2:]
    return tok


def load_manifest(store: str) -> "dict | None":
    p = Path(store) / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def write_manifest(store: str, stamps: dict, bench_fps: dict, cells: list) -> None:
    Path(store).mkdir(parents=True, exist_ok=True)
    p = Path(store) / "manifest.json"
    tmp = p.with_name("manifest.json.tmp")
    tmp.write_text(json.dumps({"stamps": stamps, "benchmarks": bench_fps,
                               "cells": sorted(cells)}, indent=2))
    os.replace(tmp, p)
