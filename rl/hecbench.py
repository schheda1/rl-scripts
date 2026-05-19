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

# ---------------------------------------------------------------------------
# Benchmark discovery
# ---------------------------------------------------------------------------

HECBENCH_SRC = Path(__file__).parent.parent.parent / "HeCBench" / "src"


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
    benchmarks = []
    for d in sorted(hecbench_src.glob("*-cuda")):
        if not d.is_dir():
            continue
        makefile = d / "Makefile"
        if not makefile.exists():
            continue
        if not _has_run_target(makefile):
            continue
        if _uses_external_data(makefile):
            continue
        if _has_dvc_files(d):
            continue
        benchmarks.append(d)
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


def _make(benchmark_dir: Path, extra_cflags: str, arch: str, timeout: int = 600) -> subprocess.CompletedProcess:
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

LOOPCOUNT_COLUMNS = [
    "loopIdx", "loopDepth", "startLine", "startCol", "startIsImplicitCode",
    "endLine", "endCol", "endIsImplicitCode", "function", "numPaths",
    "duplicatable", "loopSize", "sizeIsValid", "containsPHI",
    "exitBlocksContainPHI", "containsUseOutsideLoop", "containsBarrier",
    "containsChildLoops", "containsBranch", "tripCountKnown", "tripCount",
    "numBasicBlocks", "numMemoryInsts", "numComputeInsts", "numControlFlowInsts",
]

FEATURE_COLUMNS = [
    "loopDepth", "numPaths", "loopSize", "sizeIsValid", "containsPHI",
    "exitBlocksContainPHI", "containsUseOutsideLoop", "containsBarrier",
    "containsChildLoops", "containsBranch", "tripCountKnown", "tripCount",
    "numBasicBlocks", "numMemoryInsts", "numComputeInsts", "numControlFlowInsts",
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
    Loops that are non-duplicatable, contain barriers, or have empty function names
    are filtered out.
    """
    result = compile_loopcount(benchmark_dir, arch=arch)
    parsed = parse_loopcount_output(result.stderr)
    if not parsed:
        raise RuntimeError(f"No LoopCount output from {benchmark_dir.name}")

    # Prefer the CUDA device triple; fall back to first if not found.
    triple = next(
        (t for t in parsed if "nvptx" in t or "cuda" in t.lower()),
        next(iter(parsed)),
    )
    file_map = parsed[triple]

    eligible: dict[str, pd.DataFrame] = {}
    for filename, df in file_map.items():
        mask = (
            (df["duplicatable"] == 1.0)
            & (df["containsBarrier"] == 0.0)
            & (df["containsBranch"] == 1.0)
            & df["function"].notna()
            & (df["function"] != "")
        )
        df_ok = df[mask].reset_index(drop=True)
        if len(df_ok) > 0:
            eligible[filename] = df_ok

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


def measure_kernel_time(
    benchmark_dir: Path,
    arch: str = ARCH,
    n_runs: int = 20,
    nsys_timeout: int = 300,
    tmp_dir: Path = DEFAULT_TMP_DIR,
) -> float:
    """
    Run the benchmark under nsys n_runs times and return the mean total GPU
    kernel time in ms.  The binary must already be compiled.

    Uses a two-step approach compatible with newer nsys versions:
      1. nsys profile --output=<file> <binary> <args>
      2. nsys stats --report=cuda_gpu_kern_sum --format=csv <file>.nsys-rep

    nsys report files are written under *tmp_dir*.
    """
    run_cmd = _get_run_command(benchmark_dir, arch)
    env = {**os.environ, "ARCH": arch}
    tmp_dir.mkdir(parents=True, exist_ok=True)
    report_path = tempfile.mktemp(prefix="nsys_rl_", dir=str(tmp_dir))

    times: list[float] = []
    for _ in range(n_runs):
        # Step 1: profile
        subprocess.run(
            f"nsys profile --output={report_path} --force-overwrite=true {run_cmd}",
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
        combined = stats_result.stdout + stats_result.stderr
        t = _parse_nsys_csv_kernel_time_ms(combined)
        if t is not None:
            times.append(t)

    if not times:
        raise RuntimeError(
            f"nsys produced no parseable kernel times for {benchmark_dir.name}"
        )
    return statistics.mean(times)


def _parse_nsys_csv_kernel_time_ms(csv_output: str) -> Optional[float]:
    """
    Parse `nsys stats --format=csv` output for cuda_gpu_kern_sum.

    Skips preamble lines (e.g. 'Generating SQLite...', 'Processing...') and
    finds the CSV block starting at the 'Time (%)' header line.
    Sums 'Total Time (ns)' across all kernel rows and converts to ms.
    """
    # Find the line where the CSV header starts
    csv_lines = []
    in_csv = False
    for line in csv_output.splitlines():
        if not in_csv and line.startswith("Time (%)"):
            in_csv = True
        if in_csv and line.strip():
            csv_lines.append(line)

    if not csv_lines:
        return None

    try:
        df = pd.read_csv(StringIO("\n".join(csv_lines)))
        time_col = next(
            (c for c in df.columns if "time" in c.lower() and "total" in c.lower()),
            None,
        )
        if time_col is None:
            numeric_cols = df.select_dtypes("number").columns
            time_col = numeric_cols[1] if len(numeric_cols) > 1 else None
        if time_col is not None and not df.empty:
            return df[time_col].sum() / 1_000_000.0  # ns → ms
    except Exception:
        pass

    return None
