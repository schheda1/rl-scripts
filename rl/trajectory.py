"""
Trajectory primitive for multi-loop (nested) actions.

A trajectory is the set of (loop_idx -> (unmerge, factor)) decisions applied to
one benchmark in ONE compile.  The UU pass applies them innermost-first on a
pre-transform snapshot of the loops, so an outer loop's action operates on the
already-transformed inner loop — that is the state transition of the nested
MDP.  This module is shared by the collectors, the training environment, the
deployment evaluator and the offline analysis, so all of them compile, key and
cache a trajectory the same way (MIRROR discipline).

Three things live here:

  * trajectory_key / parse_trajectory_key — the cache key.  No-op entries
    ((0, 1)) are dropped, so the empty trajectory IS the baseline: it is never
    compiled, never measured and never stored (reward exactly 0.0), the same
    convention the single-loop reward cache uses for NOOP.

  * state_after — the RAW post-transform feature vector of ONE loop after a
    trajectory, read back through the pass's identity stamp (the uuIdx column),
    never by row position: any factor >= 2 on a loop with descendants clones
    the subtree and shifts every later loopIdx.  A loop that was cloned has
    several rows with the same stamp; their mean is returned.

  * load_trajectory_cache / save_trajectory_cache — a SEPARATE cache file,
    trajectory_cache.json.  reward_cache.json is never opened for writing here.
    Post-state vectors are stored RAW (un-normalised), so a normalizer refit
    does not invalidate them; a feature-schema change does (features_version).

Compiles go through hecbench (compile_trajectory_ex, _make,
parse_loopcount_output), so numbers are comparable with the single-loop path.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from features import FEATURES_VERSION
from hecbench import (
    ARCH,
    _build_extra_cflags,
    _make,
    _row_to_tensor,
    parse_loopcount_output,
)

NOOP = (0, 1)
TRAJECTORY_CACHE_NAME = "trajectory_cache.json"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def prune_noops(loop_actions: dict) -> dict[int, tuple[int, int]]:
    """Normalise a {loop_idx: (unmerge, factor)} mapping: ints, no-ops dropped."""
    out: dict[int, tuple[int, int]] = {}
    for i, (u, f) in loop_actions.items():
        a = (int(u), int(f))
        if a != NOOP:
            out[int(i)] = a
    return out


def trajectory_key(benchmark_name: str, loop_actions: dict) -> str:
    """
    `bench|idx:u:f,idx:u:f` with entries sorted by loop_idx and no-ops dropped.
    The empty trajectory is `bench|` — the baseline, which callers never store.
    Distinct from the single-loop cache key (`bench|idx|u|f`) on purpose: the
    two caches are different files and must never be confused.
    """
    acts = prune_noops(loop_actions)
    body = ",".join(f"{i}:{u}:{f}" for i, (u, f) in sorted(acts.items()))
    return f"{benchmark_name}|{body}"


def parse_trajectory_key(key: str) -> tuple[str, dict[int, tuple[int, int]]]:
    """Inverse of trajectory_key.  Benchmark names never contain '|'."""
    name, sep, body = key.rpartition("|")
    if not sep:
        raise ValueError(f"not a trajectory key (no '|'): {key!r}")
    acts: dict[int, tuple[int, int]] = {}
    if body:
        for item in body.split(","):
            i, u, f = item.split(":")
            acts[int(i)] = (int(u), int(f))
    return name, acts


def post_key(traj_key: str, target_idx: int) -> str:
    """Key of a post-state vector in the trajectory cache: `<traj_key>|<target>`."""
    return f"{traj_key}|{int(target_idx)}"


def parse_post_key(key: str) -> tuple[str, int]:
    """Inverse of post_key: (trajectory key, target loop_idx)."""
    traj, sep, target = key.rpartition("|")
    if not sep:
        raise ValueError(f"not a post-state key (no '|'): {key!r}")
    return traj, int(target)


# ---------------------------------------------------------------------------
# Post-transform state
# ---------------------------------------------------------------------------

def _device_rows_after(
    benchmark_dir: Path,
    actions: dict[int, tuple[int, int]],
    filename: str,
    triple: str,
    arch: str,
    timeout: int,
) -> Optional[pd.DataFrame]:
    """
    LoopCount rows of *filename*'s DEVICE loops after *actions* (already
    no-op-pruned).  Non-empty actions: --enable-uu + the identity stamp.  Empty
    actions: a plain --enable-loopcount compile (no UU pass, so no stamp; rows
    are then matched by loopIdx, which is stable when nothing was transformed).

    The return code is deliberately not checked: LoopCount prints its rows at
    the compile stage, so they are valid even when a later stage (link) fails,
    exactly as the eligibility precheck treats them.  A compile TIMEOUT or a
    crash before LoopCount yields no rows -> None.

    Host/device disambiguation mirrors GpuLoopEnv.get_post_unmerge_features:
    not by triple (unreliable in a CUDA clang++ build) but by the per-row device
    guard isKernelFunction == 1 or kernelParents non-empty, with *filename*
    narrowing file-vs-file in multi-.cu benchmarks.
    """
    indices = sorted(actions)
    cflags = _build_extra_cflags(
        enable_uu=bool(indices),
        enable_loopcount=True,
        filename=filename if indices else "",
        triple=triple if indices else "",
        loop_indices=indices,
        unmerge_flags=[actions[i][0] for i in indices],
        unroll_factors=[actions[i][1] for i in indices],
        stamp_loops=bool(indices),
    )
    try:
        result = _make(benchmark_dir, extra_cflags=cflags, arch=arch,
                       timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    parsed = parse_loopcount_output(result.stderr)
    frames: list[pd.DataFrame] = []
    for _triple, file_map in parsed.items():
        for fname, df in file_map.items():
            if filename and fname != filename:
                continue
            is_kernel = df["isKernelFunction"].astype(float) == 1.0
            has_parents = (
                df["kernelParents"].notna()
                & (df["kernelParents"].astype(str).str.strip() != "")
            )
            frames.append(df[is_kernel | has_parents])
    if not frames:
        return None
    rows = pd.concat(frames, ignore_index=True)
    if indices and "uuIdx" not in rows.columns:
        raise RuntimeError(
            "LoopCount output has no uuIdx column — the llvm build predates the "
            "identity stamp (UnrollAndUnmerge.cpp uu-stamp-loops); rebuild."
        )
    return rows


def state_after(
    benchmark_dir: Path,
    loop_actions: dict,
    target_idx: int,
    filename: str,
    triple: str,
    arch: str = ARCH,
    timeout: int = 300,
    pre_features_raw: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """
    RAW (un-normalised) feature vector of loop *target_idx* after *loop_actions*
    have been applied, or None when it cannot be read (compile timeout, no
    LoopCount rows, no row with that stamp).  Callers normalise (the
    environment) or store raw (the collector), exactly as for pre-features.

    All-no-op shortcut: with nothing to apply, the IR LoopCount sees is the
    pre-feature IR — both compiles canonicalise with the same simplifyLoop /
    formLCSSA calls before extracting, and the feature columns were verified
    byte-identical — so *pre_features_raw* is returned without compiling when
    given.  Without it, the plain LoopCount compile runs and the row is read by
    loopIdx, which is stable when nothing was transformed.

    Otherwise one compile with --enable-uu, --enable-loopcount and the identity
    stamp (-uu-stamp-loops=1; opt-in, never used for a binary that is measured),
    then the device rows whose uuIdx == target_idx.  Several rows mean the loop
    was cloned by an ancestor's action; their mean is returned.  The target's
    own action, if present in the trajectory, is applied too — so this also
    reads "the loop after its own transform", generalising
    GpuLoopEnv.get_post_unmerge_features (which keeps its loopIdx lookup: an
    unmerge x1 never clones, so that lookup is exact there).
    """
    actions = prune_noops(loop_actions)
    if not actions and pre_features_raw is not None:
        return pre_features_raw
    rows = _device_rows_after(benchmark_dir, actions, filename, triple, arch,
                              timeout)
    if rows is None or rows.empty:
        return None
    col = "uuIdx" if actions else "loopIdx"
    sel = rows[rows[col].astype(float) == float(target_idx)]
    if sel.empty:
        return None
    vecs = [_row_to_tensor(r) for _, r in sel.iterrows()]
    return torch.stack(vecs).mean(dim=0)


# ---------------------------------------------------------------------------
# Cache (separate file; never touches reward_cache.json)
# ---------------------------------------------------------------------------

def load_trajectory_cache(path: Path) -> dict:
    """
    {"rewards": {traj_key: float}, "post_features": {post_key: [raw floats]},
     "features_version": ..., "migration": {...}} — empty dicts when the file
    is absent or unreadable.  Post-state values are plain lists (a tensor is
    stored via .tolist() and rebuilt with torch.tensor by the reader).  Post-state vectors from a different feature
    schema are dropped (they would be the wrong length); rewards survive, they
    are raw measurements.
    """
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    rewards = {k: float(v) for k, v in data.get("rewards", {}).items()}
    postf = data.get("post_features", {}) or {}
    if data.get("features_version") != FEATURES_VERSION:
        postf = {}
    return {
        "rewards": rewards,
        "post_features": postf,
        "features_version": FEATURES_VERSION,
        "migration": data.get("migration") or {},
    }


def save_trajectory_cache(
    path: Path,
    rewards: dict,
    post_features: dict,
    failure_keys: set,
    timeout_keys: set,
    failure_penalty: float,
    timeout_penalty: float,
    deadzone: float,
) -> None:
    """
    Atomic write (tmp + replace, like collect_cells.save_cache), preserving keys
    this module does not own.  *post_features* values must be JSON-serialisable
    (lists of floats, i.e. tensor.tolist()); *rewards* values floats.  The migration block records WHICH trajectories
    hold a penalty rather than a measurement — compile failures AND compile
    timeouts, kept separately — so either penalty can be re-tuned later without
    confusing a penalised cell with a genuine measurement of the same value.
    """
    payload: dict = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except Exception:
            payload = {}
    payload["features_version"] = FEATURES_VERSION
    payload["rewards"] = rewards
    payload["post_features"] = post_features
    mig = payload.get("migration") or {}
    mig["failure_keys"] = sorted(set(mig.get("failure_keys", [])) | set(failure_keys))
    mig["timeout_keys"] = sorted(set(mig.get("timeout_keys", [])) | set(timeout_keys))
    mig.setdefault("failure_penalty", failure_penalty)
    mig.setdefault("timeout_penalty", timeout_penalty)
    mig.setdefault("deadzone", deadzone)
    payload["migration"] = mig
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)
