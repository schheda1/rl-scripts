"""
GpuLoopEnv: per-loop RL environment for the UU optimization pipeline.

Episode structure (per benchmark):
  reset(benchmark_dir) →
    1. compile baseline, measure baseline_kernel_time
    2. compile with LoopCount → parse eligible loops
    3. return first loop's pre-unmerge feature vector

  step() handles the 2-step action per loop:
    call 1: (unmerge_decision, _)
      - if unmerge=1: compile with unmerge-only + LoopCount → return
        post-unmerge features, reward=None, done=False
      - if unmerge=0: return same features, reward=None, done=False
        (caller proceeds directly to factor decision without extra compile)

    call 2: (unmerge_decision, factor_idx)
      - compile loop with final (unmerge, factor)
      - measure kernel time → compute reward
      - advance to next eligible loop
      - return (next_features | None, reward, done)

The training loop (train.py) manages the 2-step sequence explicitly.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from hecbench import (
    ARCH,
    DEFAULT_TMP_DIR,
    FEATURE_COLUMNS,
    FeatureNormalizer,
    _row_to_tensor,
    compile_loopcount,
    compile_single_loop,
    compile_baseline,
    get_loop_features,
    measure_kernel_time,
    parse_loopcount_output,
)
from agent import FACTOR_VALUES


@dataclass
class LoopRecord:
    loop_idx: int
    filename: str
    triple: str
    pre_features: torch.Tensor   # shape (N_FEATURES,), z-score NORMALISED
    kernel_parents: list = None  # mangled parent kernel names from LoopCount BFS
    # RAW trip-count values for factor masking.  pre_features is normalised,
    # so the trip count cannot be recovered from it — the mask must be built
    # from these.  Trip count is invariant under unmerge, so they are valid
    # for the factor decision on both branches.
    trip_count_known: bool = False
    trip_count: int = 0

    def __post_init__(self):
        if self.kernel_parents is None:
            self.kernel_parents = []


class GpuLoopEnv:
    """
    Gym-like environment.  Not a formal gymnasium.Env subclass to keep the
    interface simple; the training loop drives it directly.
    """

    def __init__(
        self,
        arch: str = ARCH,
        n_runs: int = 20,
        nsys_timeout: int = 300,
        tmp_dir: Path = DEFAULT_TMP_DIR,
        compile_timeout_penalty: float = -1.0,
        gpu_id: int = 0,
        normalizer: Optional[FeatureNormalizer] = None,
        baseline_cache: Optional[dict] = None,
    ) -> None:
        self.arch = arch
        self.n_runs = n_runs
        self.nsys_timeout = nsys_timeout
        self.tmp_dir = tmp_dir
        self.compile_timeout_penalty = compile_timeout_penalty
        self.gpu_id = gpu_id
        self.normalizer = normalizer or FeatureNormalizer()  # no-op if not fitted

        # Pre-measured baseline times: {benchmark_dir.name → ms}.
        # Populated by the training script once per run (after train/val/test split)
        # and passed in here so reset() skips re-measurement on every epoch.
        # On a cache miss reset() falls back to on-demand measurement and stores
        # the result, so held-out (test) benchmarks still work correctly.
        self._baseline_cache: dict[str, float] = dict(baseline_cache) if baseline_cache else {}

        # State set by reset()
        self._benchmark_dir: Optional[Path] = None
        self._baseline_time_ms: float = 0.0
        self._kernel_baselines: dict[str, float] = {}  # demangled name → ms
        self._eligible_loops: list[LoopRecord] = []
        self._loop_cursor: int = 0

        # State set mid-step (unmerge=1 path)
        self._awaiting_factor: bool = False
        self._pending_unmerge: int = 0
        self._pending_post_features: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def eligible_loops(self) -> list[LoopRecord]:
        return list(self._eligible_loops)

    @property
    def done(self) -> bool:
        return self._loop_cursor >= len(self._eligible_loops)

    def _to_features(self, row: "pd.Series") -> torch.Tensor:
        """Convert a LoopCount row to a normalized feature tensor."""
        return self.normalizer.normalize(_row_to_tensor(row))

    def reset(self, benchmark_dir: Path) -> Optional[torch.Tensor]:
        """
        Prepare a new episode for *benchmark_dir*.

        Returns the pre-unmerge feature vector for the first eligible loop,
        or None if the benchmark has no eligible loops.

        Baseline time is read from the cache when available (pre-measured once
        per training run after the train/val/test split).  On a cache miss the
        baseline is compiled and measured on-demand and the result is stored so
        subsequent resets for the same benchmark (e.g. test-set eval) are free.
        """
        self._benchmark_dir = benchmark_dir
        self._loop_cursor = 0
        self._awaiting_factor = False
        self._pending_post_features = None

        from hecbench import demangle as _demangle

        bname = benchmark_dir.name
        entry = self._baseline_cache.get(bname)
        if isinstance(entry, dict):
            # New-style cache entry from measure_baselines()
            self._baseline_time_ms = entry["total_ms"]
            self._kernel_baselines  = dict(entry.get("per_kernel_ms", {}))
        elif isinstance(entry, float):
            # Legacy scalar entry (old cache format or on-demand measurement)
            self._baseline_time_ms = entry
            self._kernel_baselines  = {}
        else:
            # Cache miss — compile and measure on demand (test set, etc.)
            if not compile_baseline(benchmark_dir, arch=self.arch):
                raise RuntimeError(f"Baseline compilation failed: {bname}")
            self._baseline_time_ms = measure_kernel_time(
                benchmark_dir, arch=self.arch, n_runs=self.n_runs,
                nsys_timeout=self.nsys_timeout, tmp_dir=self.tmp_dir,
                gpu_id=self.gpu_id,
            )
            self._baseline_cache[bname] = self._baseline_time_ms
            self._kernel_baselines = {}

        file_map, primary_file, triple = get_loop_features(benchmark_dir, arch=self.arch)
        if not file_map:
            return None

        self._eligible_loops = []
        for filename, df in file_map.items():
            for _, row in df.iterrows():
                kp_raw = str(row.get("kernelParents", "")).strip()
                kernel_parents = [p for p in kp_raw.split("|") if p]
                self._eligible_loops.append(
                    LoopRecord(
                        loop_idx=int(row["loopIdx"]),
                        filename=filename,
                        triple=triple,
                        pre_features=self._to_features(row),
                        kernel_parents=kernel_parents,
                        trip_count_known=bool(int(row.get("tripCountKnown", 0))),
                        trip_count=int(row.get("tripCount", 0)),
                    )
                )

        if not self._eligible_loops:
            return None

        return self._eligible_loops[0].pre_features

    def get_post_unmerge_features(self, loop_record: LoopRecord) -> torch.Tensor:
        """
        Compile loop_idx with unmerge-only (factor=1) + LoopCount and return
        the post-unmerge feature vector.  Falls back to pre-unmerge features
        if the compilation fails or the loop is not present in output.
        """
        ok = compile_single_loop(
            self._benchmark_dir,
            loop_idx=loop_record.loop_idx,
            unmerge=1,
            factor=1,
            filename=loop_record.filename,
            triple=loop_record.triple,
            arch=self.arch,
        )
        if not ok:
            return loop_record.pre_features

        # Re-run LoopCount on the unmerged binary to get updated features.
        # We recompile with loopcount enabled on top of the unmerge compilation.
        # Simplest approach: compile with both --enable-uu + unmerge=1 + --enable-loopcount.
        from hecbench import _build_extra_cflags, _make
        cflags = _build_extra_cflags(
            enable_uu=True,
            enable_loopcount=True,
            filename=loop_record.filename,
            triple=loop_record.triple,
            loop_indices=[loop_record.loop_idx],
            unmerge_flags=[1],
            unroll_factors=[1],
        )
        result = _make(self._benchmark_dir, extra_cflags=cflags, arch=self.arch)
        parsed = parse_loopcount_output(result.stderr)

        # Unmerge restructures the loop body but does not create new loops.
        # The total loop count and RPO-based index assignment should be stable,
        # so look up the loop by its original loopIdx.  Fall back to
        # pre-unmerge features if the loop is not found in the output.
        #
        # Host/device disambiguation — DO NOT use the triple.  A CUDA clang++
        # build runs two cc1 subprocesses (device nvptx + host x86) sharing one
        # stderr.  LoopCount has no accelerator gate, so BOTH emit rows; seenLoops
        # is a per-process static that resets to 0 in each cc1, so a host loop and
        # a device loop can share the SAME loopIdx AND the SAME filename (both
        # compile the same .cu), differing only by the unreliable triple.  Keying
        # on (filename, loopIdx) alone could therefore return the host row.
        #
        # Apply the same per-row device guard as get_loop_features instead:
        # isKernelFunction==1 (loop in a __global__/kernel function) OR
        # kernelParents non-empty (loop in a device function with kernel callers).
        # Host rows can never satisfy this — the host module has no kernel-calling-
        # convention function, so isKernelFunction=0 and kernelParents="" there.
        # This is triple-independent and works for any accelerator target whose
        # kernels LoopCount attributes (NVPTX today; AMDGPU once isKernelFunction
        # / kernelParents recognise the AMDGPU_KERNEL calling convention).
        #
        # filename still narrows file-vs-file in multi-.cu benchmarks (each device
        # cc1 restarts the counter); the device guard narrows host-vs-device.
        for _triple, file_map in parsed.items():
            for fname, df in file_map.items():
                if loop_record.filename and fname != loop_record.filename:
                    continue
                is_kernel = df["isKernelFunction"].astype(float) == 1.0
                has_parents = (
                    df["kernelParents"].notna()
                    & (df["kernelParents"].astype(str).str.strip() != "")
                )
                df_dev = df[is_kernel | has_parents]
                row = df_dev[df_dev["loopIdx"] == loop_record.loop_idx]
                if not row.empty:
                    return self._to_features(row.iloc[0])

        return loop_record.pre_features

    def _resolve_measurement(
        self, loop_record: LoopRecord
    ) -> tuple[Optional[str], float]:
        """
        Return (kernel_filter, baseline_ms) as a COUPLED pair whose measurement
        scope is guaranteed symmetric — both per-kernel, or both total.

        Cases A / B1 (exactly one kernel parent): use the per-kernel nsys filter
        ("funcname(" from the demangled symbol — avoids c++filt vs nsys formatting
        differences) AND the matching per-kernel baseline.

        Case B2 (multiple parents) / no parents / per-kernel cache MISS: fall back
        to (None, total_benchmark_ms).

        Why coupled: if the filter and baseline were resolved independently, a
        per-kernel cache miss would give baseline=total while modified still
        measured per-kernel — an asymmetric (baseline=total / modified=per-kernel)
        comparison that corrupts the reward.  Resolving them together makes that
        impossible: a missing per-kernel baseline forces the filter to None too,
        so the modified measurement also falls back to total.
        """
        from hecbench import demangle as _demangle, demangled_to_filter as _to_filter
        parents = loop_record.kernel_parents or []
        if len(parents) == 1:
            kf = _to_filter(_demangle(parents[0]))
            per_kernel = self._kernel_baselines.get(kf)
            if per_kernel is not None:
                return kf, per_kernel
            # Per-kernel baseline missing → force BOTH sides to total.
            return None, self._baseline_time_ms
        return None, self._baseline_time_ms

    def step(
        self,
        loop_record: LoopRecord,
        unmerge: int,
        factor_idx: int,
    ) -> tuple[Optional[torch.Tensor], float, bool]:
        """
        Apply (unmerge, factor_idx) to *loop_record*, measure reward, advance.

        Returns (next_features, reward, done).
          next_features: feature vector for the next loop, or None if done.
          reward:        (kernel_baseline - kernel_modified) / kernel_baseline
                         where kernel_baseline is the per-kernel time for cases
                         A and B1, or total benchmark time for case B2.
          done:          True when all eligible loops in the episode are exhausted.
        """
        factor = FACTOR_VALUES[factor_idx]

        # No-op: unmerge=0 and factor=1 leaves the binary identical to baseline.
        # Skip compilation and measurement entirely — reward is exactly 0 by definition.
        # Any nsys measurement here would only inject noise.
        if unmerge == 0 and factor == 1:
            self._loop_cursor += 1
            if self.done:
                return None, 0.0, True
            return self._eligible_loops[self._loop_cursor].pre_features, 0.0, False

        kernel_filter, baseline_ms = self._resolve_measurement(loop_record)

        try:
            ok = compile_single_loop(
                self._benchmark_dir,
                loop_idx=loop_record.loop_idx,
                unmerge=unmerge,
                factor=factor,
                filename=loop_record.filename,
                triple=loop_record.triple,
                arch=self.arch,
            )
        except subprocess.TimeoutExpired:
            # Compilation timed out — SCEV/unroll complexity exceeded.
            # Return a penalty reward so the agent learns to avoid this
            # (loop features, factor) combination.  Distinguish from a
            # generic compile error which gets reward=0 (no signal).
            self._loop_cursor += 1
            next_features = (
                None if self.done
                else self._eligible_loops[self._loop_cursor].pre_features
            )
            return next_features, self.compile_timeout_penalty, self.done

        if not ok:
            # Compile error (bad flags, source issue) — treat as no-op,
            # no signal to the agent.
            modified_time_ms = baseline_ms
        else:
            try:
                modified_time_ms = measure_kernel_time(
                    self._benchmark_dir, arch=self.arch, n_runs=self.n_runs,
                    nsys_timeout=self.nsys_timeout, tmp_dir=self.tmp_dir,
                    gpu_id=self.gpu_id,
                    kernel_filter=kernel_filter,
                )
            except RuntimeError:
                modified_time_ms = baseline_ms

        # Clip at -1.0 (the timeout-penalty scale).  A pathological slowdown
        # (observed: -52 on wlcpow) would otherwise dominate the normalised
        # advantages of its entire PPO buffer.  The upside is already bounded
        # at 1.0 by construction.  Signal stays monotone: worse is still worse,
        # just capped in magnitude.
        reward = max(
            (baseline_ms - modified_time_ms) / max(baseline_ms, 1e-9),
            -1.0,
        )

        self._loop_cursor += 1
        if self.done:
            return None, reward, True

        next_features = self._eligible_loops[self._loop_cursor].pre_features
        return next_features, reward, False


