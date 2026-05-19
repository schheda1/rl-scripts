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
    pre_features: torch.Tensor   # shape (16,)


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
    ) -> None:
        self.arch = arch
        self.n_runs = n_runs
        self.nsys_timeout = nsys_timeout
        self.tmp_dir = tmp_dir
        self.compile_timeout_penalty = compile_timeout_penalty

        # State set by reset()
        self._benchmark_dir: Optional[Path] = None
        self._baseline_time_ms: float = 0.0
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

    def reset(self, benchmark_dir: Path) -> Optional[torch.Tensor]:
        """
        Prepare a new episode for *benchmark_dir*.

        Returns the pre-unmerge feature vector for the first eligible loop,
        or None if the benchmark has no eligible loops.
        """
        self._benchmark_dir = benchmark_dir
        self._loop_cursor = 0
        self._awaiting_factor = False
        self._pending_post_features = None

        if not compile_baseline(benchmark_dir, arch=self.arch):
            raise RuntimeError(f"Baseline compilation failed: {benchmark_dir.name}")

        self._baseline_time_ms = measure_kernel_time(
            benchmark_dir, arch=self.arch, n_runs=self.n_runs,
            nsys_timeout=self.nsys_timeout, tmp_dir=self.tmp_dir,
        )

        file_map, primary_file, triple = get_loop_features(benchmark_dir, arch=self.arch)
        if not file_map:
            return None

        self._eligible_loops = []
        for filename, df in file_map.items():
            for _, row in df.iterrows():
                features = _row_to_tensor(row)
                self._eligible_loops.append(
                    LoopRecord(
                        loop_idx=int(row["loopIdx"]),
                        filename=filename,
                        triple=triple,
                        pre_features=features,
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
        # pre-unmerge features if the index is not found in the output.
        for _triple, file_map in parsed.items():
            if "nvptx" not in _triple and "cuda" not in _triple.lower():
                continue
            for _fname, df in file_map.items():
                row = df[df["loopIdx"] == loop_record.loop_idx]
                if not row.empty:
                    return _row_to_tensor(row.iloc[0])

        return loop_record.pre_features

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
          reward:        (baseline_time - modified_time) / baseline_time
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
            modified_time_ms = self._baseline_time_ms
        else:
            try:
                modified_time_ms = measure_kernel_time(
                    self._benchmark_dir, arch=self.arch, n_runs=self.n_runs,
                    nsys_timeout=self.nsys_timeout, tmp_dir=self.tmp_dir,
                )
            except RuntimeError:
                modified_time_ms = self._baseline_time_ms

        reward = (self._baseline_time_ms - modified_time_ms) / max(
            self._baseline_time_ms, 1e-9
        )

        self._loop_cursor += 1
        if self.done:
            return None, reward, True

        next_features = self._eligible_loops[self._loop_cursor].pre_features
        return next_features, reward, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_tensor(row: pd.Series) -> torch.Tensor:
    """Convert a LoopCount DataFrame row to a (16,) float32 tensor."""
    values = [float(row.get(col, 0.0)) for col in FEATURE_COLUMNS]
    return torch.tensor(values, dtype=torch.float32)
