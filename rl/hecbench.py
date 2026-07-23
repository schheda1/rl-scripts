"""
HecBench integration for the RL pipeline.

Provides benchmark discovery, UU compilation helpers, LoopCount feature
extraction, and nsys kernel-time measurement.
"""

import getpass
import os
import re
import statistics
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from typing import Optional

import torch

# Default per-user temp directory for all pipeline artifacts (nsys reports, etc.)
DEFAULT_TMP_DIR: Path = Path(f"/tmp/rl_pipeline_{getpass.getuser()}")

import pandas as pd

# ---------------------------------------------------------------------------
# GPU architecture detection
# ---------------------------------------------------------------------------

def detect_arch() -> str:
    """Return sm_XX string for the first GPU found via nvidia-smi."""
    try:
        cap = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
        major, minor = cap.split(".")
        return f"sm_{major}{minor}"
    except Exception as e:
        raise RuntimeError(f"Could not detect GPU arch via nvidia-smi: {e}")


ARCH: str = os.environ.get("TARGET_ARCH") or detect_arch()

# Path to the IR2Vec vocabulary JSON (seedEmbeddingVocab75D.json).  Required for
# every --enable-loopcount compile — without it the LLVM pass cannot produce
# embeddings.  Set via the IR2VEC_VOCAB env var in the slurm/launch script,
# e.g. IR2VEC_VOCAB=$LLVM_SRC/llvm/lib/Analysis/models/seedEmbeddingVocab75D.json
IR2VEC_VOCAB: str = os.environ.get("IR2VEC_VOCAB", "")

# ---------------------------------------------------------------------------
# Benchmark discovery
# ---------------------------------------------------------------------------

# Default benchmark source directory.
# og-HecBench: benchmarks live directly at <repo>/*-cuda  (no src/ subdirectory)
# upstream HeCBench: benchmarks live at <repo>/src/*-cuda
# Override at runtime via --hecbench-src or the HECBENCH_SRC env variable.
HECBENCH_SRC = Path(
    os.environ.get("HECBENCH_SRC", "")
    or str(Path(__file__).parent.parent.parent / "og-HeCBench")
)


def _has_run_target(makefile: Path) -> bool:
    text = makefile.read_text(errors="replace")
    return bool(re.search(r"^run\s*:", text, re.MULTILINE))


def _uses_external_data(makefile: Path) -> bool:
    """Return True if the run: target references ../data/ (external dataset)."""
    text = makefile.read_text(errors="replace")
    in_run = False
    for line in text.splitlines():
        if re.match(r"^run\s*:", line):
            in_run = True
            continue
        if in_run:
            if line.startswith("\t"):
                if "../data/" in line:
                    return True
            else:
                break
    return False


def _has_dvc_files(benchmark_dir: Path) -> bool:
    """Return True if the benchmark directory contains any .dvc files.

    A .dvc file means required data is tracked by DVC and not present in the
    repo — the binary will fail at runtime without it.
    """
    return any(benchmark_dir.glob("*.dvc"))


def discover_benchmarks(hecbench_src: Path = HECBENCH_SRC) -> list[Path]:
    """
    Return sorted list of *-cuda benchmark directories that are
    self-contained and runnable without external data:
      - have a run: target in their Makefile
      - do not reference ../data/ in the run: target
      - do not have DVC-tracked data files (.dvc) that would be absent

    Benchmarks failing any of these checks are excluded.
    """
    import logging
    _log = logging.getLogger("hecbench.discover")

    hecbench_src = Path(hecbench_src)
    if not hecbench_src.exists():
        _log.error("discover_benchmarks: path does not exist: %s", hecbench_src.resolve())
        return []

    candidates = sorted(hecbench_src.glob("*-cuda"))
    if not candidates:
        _log.error(
            "discover_benchmarks: no *-cuda directories found under %s",
            hecbench_src.resolve(),
        )
        return []

    _log.debug("discover_benchmarks: scanning %d candidates under %s",
               len(candidates), hecbench_src.resolve())

    benchmarks = []
    for d in candidates:
        if not d.is_dir():
            continue
        makefile = d / "Makefile"
        if not makefile.exists():
            _log.debug("  SKIP %s — no Makefile", d.name)
            continue
        if not _has_run_target(makefile):
            _log.debug("  SKIP %s — no run: target", d.name)
            continue
        if _uses_external_data(makefile):
            _log.debug("  SKIP %s — references ../data/", d.name)
            continue
        if _has_dvc_files(d):
            _log.debug("  SKIP %s — DVC-tracked data", d.name)
            continue
        benchmarks.append(d)

    if not benchmarks:
        _log.warning(
            "discover_benchmarks: 0 benchmarks passed all filters under %s "
            "(run with DEBUG logging to see per-benchmark skip reasons)",
            hecbench_src.resolve(),
        )
    else:
        _log.debug("discover_benchmarks: %d benchmarks accepted", len(benchmarks))

    return benchmarks


# ---------------------------------------------------------------------------
# EXTRA_CFLAGS builder
# ---------------------------------------------------------------------------

def _build_extra_cflags(
    *,
    enable_uu: bool = False,
    enable_loopcount: bool = False,
    filename: str = "",
    triple: str = "",
    loop_indices: Optional[list[int]] = None,
    unmerge_flags: Optional[list[int]] = None,
    unroll_factors: Optional[list[int]] = None,
    global_unroll_factor: Optional[int] = None,
) -> str:
    parts: list[str] = []

    if enable_loopcount:
        parts.append("-mllvm --enable-loopcount")
        # The IR2Vec vocab flag must accompany EVERY loopcount compile — this
        # includes the post-unmerge feature-extraction compiles, which set both
        # enable_uu and enable_loopcount, so keying off enable_loopcount here
        # covers them.  Fail loudly rather than silently emit zero embeddings.
        if not IR2VEC_VOCAB or not Path(IR2VEC_VOCAB).exists():
            raise RuntimeError(
                "IR2VEC_VOCAB must point to an existing seedEmbeddingVocab75D.json "
                f"for loopcount compiles (got {IR2VEC_VOCAB!r}). Set the "
                "IR2VEC_VOCAB env var in the launch script."
            )
        parts.append(f"-mllvm --ir2vec-vocab-path={IR2VEC_VOCAB}")
    if enable_uu:
        parts.append("-mllvm --enable-uu")
    if filename:
        parts.append(f"-mllvm -uu-match-filename={filename}")
    if triple:
        parts.append(f"-mllvm -uu-match-targettriple={triple}")
    if global_unroll_factor is not None:
        parts.append(f"-mllvm -uu-unrollfactor={global_unroll_factor}")
    if loop_indices:
        idx_str = ",".join(str(i) for i in loop_indices)
        parts.append(f"-mllvm -uu-opt-loop-idx={idx_str}")
    if unroll_factors:
        fac_str = ",".join(str(f) for f in unroll_factors)
        parts.append(f"-mllvm -uu-opt-loop-unrollfactors={fac_str}")
    if unmerge_flags:
        um_str = ",".join(str(u) for u in unmerge_flags)
        parts.append(f"-mllvm -uu-opt-loop-unmerge={um_str}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def make_clean(benchmark_dir: Path, arch: str = ARCH) -> None:
    """Run make clean in the benchmark directory."""
    env = {**os.environ, "ARCH": arch}
    subprocess.run(["make", "clean"], cwd=benchmark_dir, capture_output=True, env=env)


def _make(benchmark_dir: Path, extra_cflags: str, arch: str, timeout: int = 300) -> subprocess.CompletedProcess:
    env = {**os.environ, "ARCH": arch}
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    # Override compiler and flags to use clang++ instead of nvcc.
    # New HecBench Makefiles default to nvcc with nvcc-specific flags;
    # we replicate the og-HeCBench clang++ pattern here.
    cflags = (
        f"-I{cuda_home}/include {extra_cflags} "
        f"-std=c++17 -Wall -O3 --cuda-gpu-arch={arch}"
    )
    ldflags = f"-L{cuda_home}/lib64 -lcudart -lcuda"
    subprocess.run(["make", "clean"], cwd=benchmark_dir, capture_output=True, env=env)
    return subprocess.run(
        f'make CC=clang++ CFLAGS="{cflags}" LDFLAGS="{ldflags}"',
        cwd=benchmark_dir,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def compile_baseline(benchmark_dir: Path, arch: str = ARCH) -> bool:
    """Compile the benchmark with no UU flags (baseline binary)."""
    result = _make(benchmark_dir, extra_cflags="", arch=arch)
    return result.returncode == 0


def compile_loopcount(benchmark_dir: Path, arch: str = ARCH) -> subprocess.CompletedProcess:
    """Compile with --enable-loopcount and return the subprocess result (stderr has CSV)."""
    cflags = _build_extra_cflags(enable_loopcount=True)
    return _make(benchmark_dir, extra_cflags=cflags, arch=arch)


def compile_single_loop(
    benchmark_dir: Path,
    loop_idx: int,
    unmerge: int,
    factor: int,
    filename: str,
    triple: str,
    arch: str = ARCH,
) -> bool:
    """Sequential mode: compile targeting one loop. Used during training."""
    cflags = _build_extra_cflags(
        enable_uu=True,
        filename=filename,
        triple=triple,
        loop_indices=[loop_idx],
        unmerge_flags=[unmerge],
        unroll_factors=[factor],
    )
    result = _make(benchmark_dir, extra_cflags=cflags, arch=arch)
    return result.returncode == 0


def compile_multi_loop(
    benchmark_dir: Path,
    loop_actions: dict[int, tuple[int, int]],
    filename: str,
    triple: str,
    arch: str = ARCH,
) -> bool:
    """Simultaneous mode: apply all loop decisions at once. Used for deployment."""
    indices = sorted(loop_actions)
    cflags = _build_extra_cflags(
        enable_uu=True,
        filename=filename,
        triple=triple,
        loop_indices=indices,
        unmerge_flags=[loop_actions[i][0] for i in indices],
        unroll_factors=[loop_actions[i][1] for i in indices],
    )
    result = _make(benchmark_dir, extra_cflags=cflags, arch=arch)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# LoopCount feature extraction
# ---------------------------------------------------------------------------

# Dimensionality of the IR2Vec loop-content embedding.  Must match IR2VecDim in
# LoopCount.cpp and seedEmbeddingVocab75D.json.
IR2VEC_DIM = 75
_EMB_COLUMNS = [f"emb{i}" for i in range(IR2VEC_DIM)]

# --- Study A (unmerge) eligibility ---------------------------------------
# Study A isolates "when is unmerging appropriate": only loops with divergent
# control flow (numPaths > 1) can be meaningfully unmerged.  The upper bound
# matches the original UU heuristic cap (canUnrollAndUnmerge skips numPaths>16);
# beyond it, path duplication blows up code size and compiles mostly time out.
# NUMPATHS_MIN=1 (exclusive lower bound via > ) disables the Study-A restriction
# and recovers the broad filter.  Overridable via the STUDY_A_NUMPATHS_MAX env.
STUDY_A_NUMPATHS_MIN = 1          # eligible requires numPaths > this
STUDY_A_NUMPATHS_MAX = int(os.environ.get("STUDY_A_NUMPATHS_MAX", "16"))

LOOPCOUNT_COLUMNS = [
    "loopIdx", "loopDepth", "startLine", "startCol", "startIsImplicitCode",
    "endLine", "endCol", "endIsImplicitCode", "function", "numPaths",
    "duplicatable", "loopSize", "sizeIsValid", "containsPHI",
    "exitBlocksContainPHI", "containsUseOutsideLoop", "containsBarrier",
    "containsChildLoops", "containsBranch", "tripCountKnown", "tripCount",
    "numBasicBlocks", "numMemoryInsts", "numComputeInsts", "numControlFlowInsts",
    "containsCall", "numExits",
    # metadata columns — not ML features, used for per-kernel measurement
    "isKernelFunction", "kernelParents",
    # IR2Vec embedding columns — appended at the END (LLVM emits them last).
    *_EMB_COLUMNS,
]

FEATURE_COLUMNS = [
    "loopDepth", "numPaths", "loopSize", "sizeIsValid", "containsPHI",
    "exitBlocksContainPHI", "containsUseOutsideLoop", "containsBarrier",
    "containsChildLoops", "containsBranch", "tripCountKnown", "tripCount",
    "numBasicBlocks", "numMemoryInsts", "numComputeInsts", "numControlFlowInsts",
    "containsCall", "numExits",
    # IR2Vec embedding features — appended at the END so structural feature
    # positions (and the trip-count mask indices 10/11 in agent.py) never move.
    *_EMB_COLUMNS,
]


def parse_loopcount_output(stderr: str) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Parse LoopCount CSV from compilation stderr.

    Returns {triple: {filename: DataFrame}} where each DataFrame has
    LOOPCOUNT_COLUMNS as columns.
    """
    result: dict[str, dict[str, pd.DataFrame]] = {}
    current_filename = ""
    current_triple = ""
    lines: list[str] = []

    def _flush():
        if not current_filename or not lines:
            return
        try:
            # The first LOOPCOUNT:: line is the column header (printed by
            # printColumnHeader when seenLoops == 0); subsequent lines are data.
            df = pd.read_csv(StringIO("\n".join(lines)), sep=";")
            result.setdefault(current_triple, {})[current_filename] = df
        except Exception:
            pass

    for line in stderr.splitlines():
        if line.startswith("LOOPCOUNT METADATA;"):
            _flush()
            parts = line.split(";")
            current_filename = parts[1] if len(parts) > 1 else ""
            current_triple = parts[2] if len(parts) > 2 else ""
            lines = []
        elif line.startswith("LOOPCOUNT::"):
            lines.append(line[len("LOOPCOUNT::"):])

    _flush()
    return result


def get_loop_features(benchmark_dir: Path, arch: str = ARCH) -> tuple[dict, str, str]:
    """
    Compile with LoopCount and return (loop_df_by_file, primary_filename, triple).

    The returned dict maps filename → DataFrame of eligible loops with FEATURE_COLUMNS.

    Filters applied (same as UU eligibility):
      - duplicatable == 1
      - containsBarrier == 0
      - containsBranch == 1
      - non-empty function name
      - device loop: isKernelFunction == 1  OR  kernelParents non-empty

    The last condition is the critical one for excluding CPU host loops.
    LoopCount emits a single METADATA header (on seenLoops == 0) that captures
    the first module's triple — typically the x86 host module in a CUDA clang++
    compilation.  As a result the triple filter ("nvptx" in triple) is unreliable:
    all loops, host and device alike, end up under the host triple.  The
    isKernelFunction / kernelParents columns are the only reliable signal that a
    loop actually runs on the GPU.
    """
    result = compile_loopcount(benchmark_dir, arch=arch)
    parsed = parse_loopcount_output(result.stderr)
    if not parsed:
        raise RuntimeError(f"No LoopCount output from {benchmark_dir.name}")

    # Stale-compiler guard: a pre-IR2Vec llvm emits no emb columns.  Catch it
    # here rather than let the missing FEATURE_COLUMNS surface as NaNs.
    _sample_df = next(iter(parsed[next(iter(parsed))].values()))
    if "emb0" not in _sample_df.columns:
        raise RuntimeError(
            f"{benchmark_dir.name}: LoopCount output has no IR2Vec embedding "
            "columns — rebuild llvm with the IR2Vec LoopCount changes, or check "
            "--ir2vec-vocab-path."
        )

    # Prefer the CUDA device triple when available; fall back to first triple.
    # NOTE: this triple may be the x86 host triple (see docstring).  The device
    # loop filter below is the authoritative guard, not the triple.
    triple = next(
        (t for t in parsed if "nvptx" in t or "cuda" in t.lower()),
        next(iter(parsed)),
    )
    file_map = parsed[triple]

    eligible: dict[str, pd.DataFrame] = {}
    for filename, df in file_map.items():
        # Device-loop guard: isKernelFunction==1 (loop in a __global__ kernel)
        # OR kernelParents non-empty (loop in a __device__ function with kernel callers).
        # CPU host loops have both set to 0 / empty.
        is_kernel   = df["isKernelFunction"].astype(float) == 1.0
        has_parents = (
            df["kernelParents"].notna()
            & (df["kernelParents"].astype(str).str.strip() != "")
        )
        # Study A: only divergent-control-flow loops (numPaths in (1, MAX]) are
        # eligible — unmerge is a no-op on single-path loops, and very high path
        # counts explode under specialisation.  numPaths already parsed as float.
        _np = df["numPaths"].astype(float)
        mask = (
            (df["duplicatable"] == 1.0)
            & (df["containsBarrier"] == 0.0)
            & (df["containsBranch"] == 1.0)
            & df["function"].notna()
            & (df["function"] != "")
            & (is_kernel | has_parents)
            & (_np > STUDY_A_NUMPATHS_MIN)
            & (_np <= STUDY_A_NUMPATHS_MAX)
        )
        df_ok = df[mask].reset_index(drop=True)
        if len(df_ok) > 0:
            eligible[filename] = df_ok

    # All-zero-embedding guard: if every eligible loop's embedding is zero the
    # vocab fell back in-pass (the LLVM warning was emitted).  Training on zero
    # embeddings silently discards the whole point, so make it loud.
    if eligible:
        import logging as _logging
        _emb_cols = [c for c in _EMB_COLUMNS
                     if c in next(iter(eligible.values())).columns]
        all_zero = all(
            (df[_emb_cols].abs().to_numpy().sum() == 0)
            for df in eligible.values()
        )
        if all_zero and _emb_cols:
            _logging.getLogger("hecbench").warning(
                "%s: all IR2Vec embeddings are zero — vocab fell back in-pass "
                "(check --ir2vec-vocab-path / llvm build)", benchmark_dir.name,
            )

    primary_file = next(iter(eligible), "")
    return eligible, primary_file, triple


# ---------------------------------------------------------------------------
# nsys kernel-time measurement
# ---------------------------------------------------------------------------

def _parse_nsys_total_kernel_time_ms(output: str) -> Optional[float]:
    """
    Extract total GPU kernel execution time in ms from nsys --stats=true output.

    Looks for the cuda_gpu_kern_sum / gpukernsum section and sums the
    'Total Time' column (which nsys reports in nanoseconds).
    """
    section_keywords = ("cuda_gpu_kern_sum", "gpukernsum")
    in_section = False
    past_separator = 0  # 0=before header, 1=header seen, 2=separator seen, 3=data
    total_ns = 0.0
    found_any = False

    for line in output.splitlines():
        stripped = line.strip()

        if not in_section:
            if any(kw in line for kw in section_keywords):
                in_section = True
                past_separator = 0
            continue

        # Skip empty lines before header
        if past_separator == 0:
            if stripped.startswith("Time"):
                past_separator = 1
            continue

        if past_separator == 1:
            # separator line (dashes)
            past_separator = 2
            continue

        if past_separator == 2:
            if not stripped:
                break  # end of section
            # Data row: Time(%) Total_Time(ns) Instances Avg Med Min Max StdDev Name
            # Split on whitespace, but name may contain spaces — take first 8 tokens
            tokens = stripped.split(None, 8)
            if len(tokens) < 3:
                break
            try:
                # token[1] is Total Time in ns (may have commas as thousands sep)
                total_ns += float(tokens[1].replace(",", ""))
                found_any = True
            except ValueError:
                break

    if not found_any:
        return None
    return total_ns / 1_000_000.0  # ns → ms


def _get_run_command(benchmark_dir: Path, arch: str) -> str:
    """
    Extract the actual run command from the Makefile via dry-run.
    Returns the shell command string to execute the benchmark binary.
    """
    env = {**os.environ, "ARCH": arch}
    result = subprocess.run(
        ["make", "-n", "run"],
        cwd=benchmark_dir,
        capture_output=True,
        text=True,
        env=env,
    )
    # make -n prints each command it would run; take the last non-empty line
    # that looks like an actual invocation (starts with ./ or a binary name)
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    for line in reversed(lines):
        if line.startswith("./") or line.startswith("/"):
            return line
    # fallback: return last non-empty line
    if lines:
        return lines[-1]
    raise RuntimeError(f"Could not extract run command from {benchmark_dir}/Makefile")


import functools


@functools.lru_cache(maxsize=512)
def demangle(mangled_name: str) -> str:
    """
    Demangle a C++ mangled symbol name using c++filt.

    Results are cached with lru_cache since demangling is deterministic and
    called O(unique kernel parents) times — typically single digits per run.
    Returns the mangled name unchanged if c++filt is unavailable or fails.
    """
    try:
        result = subprocess.run(
            ["c++filt", mangled_name],
            capture_output=True, text=True, timeout=5,
        )
        demangled = result.stdout.strip()
        return demangled if demangled else mangled_name
    except Exception:
        return mangled_name


def demangled_to_filter(demangled: str) -> str:
    """
    Convert a fully-demangled C++ name to a robust nsys substring filter.

    c++filt and nsys format the same symbol differently:
      c++filt: mandel(int*, MandelParameters const*, int, int)   ← East const, no spaces
      nsys:    mandel(int *, const MandelParameters *, int, int) ← West const, spaces

    Using the full signature as a filter fails.  Truncating to "funcname(" —
    everything up to and including the first "(" — is immune to all such
    formatting differences while remaining specific enough to avoid false matches
    in practice (kernels share names only when they are the same function).
    """
    paren = demangled.find("(")
    if paren >= 0:
        return demangled[:paren + 1]   # e.g. "mandel("
    return demangled                   # no parens (plain C / truncated) — use as-is


def _parse_nsys_kernel_times(csv_output: str) -> dict[str, float]:
    """
    Parse `nsys stats --report=cuda_gpu_kern_sum --format=csv` output.

    Returns {kernel_name: total_time_ms} for every kernel row found.
    Skips preamble lines (e.g. 'Generating SQLite...', 'Processing...') and
    locates the CSV block starting at the 'Time (%)' header line.
    Total Time values are converted from nanoseconds to milliseconds.
    Returns an empty dict if parsing fails.
    """
    csv_lines = []
    in_csv = False
    for line in csv_output.splitlines():
        if not in_csv and line.startswith("Time (%)"):
            in_csv = True
        if in_csv and line.strip():
            csv_lines.append(line)

    if not csv_lines:
        return {}

    try:
        df = pd.read_csv(StringIO("\n".join(csv_lines)))
        time_col = next(
            (c for c in df.columns if "time" in c.lower() and "total" in c.lower()),
            None,
        )
        if time_col is None:
            numeric_cols = df.select_dtypes("number").columns
            time_col = numeric_cols[1] if len(numeric_cols) > 1 else None
        name_col = next(
            (c for c in df.columns if c.strip().lower() == "name"),
            None,
        )
        if time_col is None or name_col is None or df.empty:
            return {}
        return {
            str(row[name_col]).strip(): float(row[time_col]) / 1_000_000.0
            for _, row in df.iterrows()
            if pd.notna(row[name_col]) and pd.notna(row[time_col])
        }
    except Exception:
        return {}


def _sum_kernel_times(
    kernel_times: dict[str, float],
    kernel_filter: Optional[str] = None,
) -> Optional[float]:
    """
    Sum kernel times from a {name: ms} dict.

    If *kernel_filter* is given, only sum rows whose name contains the filter
    string (case-sensitive substring match).  The filter is the demangled kernel
    name produced by demangle() so it matches nsys demangled output directly.
    Returns None if the dict is empty or no rows match the filter.
    """
    if not kernel_times:
        return None
    if kernel_filter is None:
        total = sum(kernel_times.values())
        return total if total > 0 else None
    matched = sum(
        ms for name, ms in kernel_times.items() if kernel_filter in name
    )
    return matched if matched > 0 else None


def measure_kernel_time(
    benchmark_dir: Path,
    arch: str = ARCH,
    n_runs: int = 20,
    nsys_timeout: int = 300,
    tmp_dir: Path = DEFAULT_TMP_DIR,
    gpu_id: int = 0,
    kernel_filter: Optional[str] = None,
) -> float:
    """
    Run the benchmark under nsys *n_runs* times and return the median GPU kernel
    time in ms.  The binary must already be compiled.

    *kernel_filter*: if set (demangled kernel name), only the time of matching
    kernel rows is summed per run.  Use this to isolate the specific kernel
    containing the loop being optimised.  If None, all kernels are summed
    (original behaviour — used for total-benchmark baseline and B2 fallback).

    Uses a two-step approach compatible with newer nsys versions:
      1. nsys profile --output=<file> <binary> <args>
      2. nsys stats --report=cuda_gpu_kern_sum --format=csv <file>.nsys-rep

    nsys report files are written under *tmp_dir*.
    *gpu_id* controls which physical GPU is used via CUDA_VISIBLE_DEVICES.
    """
    run_cmd = _get_run_command(benchmark_dir, arch)
    env = {**os.environ, "ARCH": arch, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    tmp_dir.mkdir(parents=True, exist_ok=True)
    report_path = tempfile.mktemp(prefix="nsys_rl_", dir=str(tmp_dir))

    times: list[float] = []
    for _ in range(n_runs):
        # A per-run TimeoutExpired is treated as a failed run rather than
        # propagated: call sites guard this function with `except RuntimeError`
        # (worker: modified_ms = baseline_ms fallback), so an uncaught
        # TimeoutExpired here would kill the whole worker process and silently
        # drop every remaining loop assigned to it.  If all runs time out,
        # the empty `times` list raises RuntimeError below — the contract the
        # callers already handle.
        try:
            # Step 1: profile.  Minimal tracing — we only ever read the
            # cuda_gpu_kern_sum report, so CPU sampling / context-switch
            # tracing is pure overhead.
            subprocess.run(
                f"nsys profile --trace=cuda --sample=none --cpuctxsw=none "
                f"--output={report_path} --force-overwrite=true {run_cmd}",
                cwd=benchmark_dir,
                shell=True,
                capture_output=True,
                text=True,
                timeout=nsys_timeout,
                env=env,
            )

            # Step 2: extract kernel stats as CSV
            stats_result = subprocess.run(
                f"nsys stats --report=cuda_gpu_kern_sum --format=csv {report_path}.nsys-rep",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
        except subprocess.TimeoutExpired:
            continue
        combined = stats_result.stdout + stats_result.stderr
        kernel_times = _parse_nsys_kernel_times(combined)
        t = _sum_kernel_times(kernel_times, kernel_filter)
        if t is not None:
            times.append(t)

    if not times:
        filter_msg = f" (filter={kernel_filter!r})" if kernel_filter else ""
        raise RuntimeError(
            f"nsys produced no parseable kernel times for "
            f"{benchmark_dir.name}{filter_msg}"
        )
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Feature tensor conversion
# ---------------------------------------------------------------------------

def _row_to_tensor(row: "pd.Series") -> torch.Tensor:
    """Convert a LoopCount DataFrame row to a (N_FEATURES,) float32 tensor."""
    values = [float(row.get(col, 0.0)) for col in FEATURE_COLUMNS]
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Feature normalizer
# ---------------------------------------------------------------------------

class FeatureNormalizer:
    """
    Per-feature z-score normalizer for LoopCount feature vectors.

    Fitted once on all eligible loop rows collected during the pre-flight
    precheck, then applied to every feature vector returned by GpuLoopEnv.

    Serialisable to/from a plain dict so it can be:
      - cached in eligible_benchmarks.json alongside loop_counts
      - shipped to worker processes through the hparams dict

    If not fitted (e.g. loaded from an old cache that predates normalization),
    normalize() is a no-op — training still runs, just without normalization.
    """

    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None
        self.std:  Optional[torch.Tensor] = None
        self._fitted: bool = False

    def fit(self, feature_tensors: list) -> None:
        """
        Compute per-feature mean and std from a list of (N_FEATURES,) tensors.
        Clamps std to ≥ 1e-8 so constant features (all-zero flags, etc.) don't
        produce NaN after division.
        """
        if not feature_tensors:
            return
        stacked   = torch.stack(feature_tensors)          # (N, N_FEATURES)
        self.mean = stacked.mean(dim=0).cpu()
        self.std  = stacked.std(dim=0, correction=0).clamp(min=1e-8).cpu()
        self._fitted = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score normalization: (x − mean) / std.
        Moves mean/std to x's device on demand.
        Returns x unchanged if the normalizer has not been fitted.
        """
        if not self._fitted or self.mean is None:
            return x
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def state_dict(self) -> dict:
        """Return a JSON-serialisable snapshot."""
        return {
            "fitted": self._fitted,
            "mean":   self.mean.tolist() if self.mean is not None else None,
            "std":    self.std.tolist()  if self.std  is not None else None,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "FeatureNormalizer":
        """Reconstruct from a snapshot (e.g. loaded from JSON or hparams)."""
        n = cls()
        if d.get("fitted") and d.get("mean") is not None:
            n.mean    = torch.tensor(d["mean"], dtype=torch.float32)
            n.std     = torch.tensor(d["std"],  dtype=torch.float32)
            n._fitted = True
        return n

    def __repr__(self) -> str:
        if not self._fitted:
            return "FeatureNormalizer(not fitted)"
        return (
            f"FeatureNormalizer(fitted, "
            f"mean={self.mean.tolist()}, "
            f"std={self.std.tolist()})"
        )
