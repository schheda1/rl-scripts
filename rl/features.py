"""
Single source of truth for the loop feature schema.

Deliberately lightweight — imports only the stdlib, NO torch / pandas / GPU
probing — so both `hecbench` (heavy: pandas, detect_arch) and `agent` (torch)
can depend on it without coupling or circular imports.

The feature vector is always: STRUCTURAL_COLUMNS (18, fixed, front) followed by
the enabled embedding/aux BLOCKS in canonical order. Structural is always first,
so the trip-count indices (10/11) never move regardless of which blocks are on.

Configuration (read once, at import — set these in the launch environment):
  UU_FEATURE_BLOCKS   comma list of enabled blocks (default "emb" == the legacy
                      93-dim schema, byte-identical to before this module).
                      e.g. "emb"  |  "emb,femb"  |  "femb"  |  (later) "emb,femb,kemb"
  LOOPCOUNT_FA_ITERS  flow-aware iteration count N (default 8), a per-compile flag.

Adding a new block (kernel embedding, width histogram, ...) is a one-line entry
in BLOCKS + BLOCK_ORDER; every consumer (columns, dim, emit flags, cache version)
stays in sync automatically.
"""
import os

IR2VEC_DIM = 75

# --- structural block: always present, always first (holds the trip indices) ---
STRUCTURAL_COLUMNS = [
    "loopDepth", "numPaths", "loopSize", "sizeIsValid", "containsPHI",
    "exitBlocksContainPHI", "containsUseOutsideLoop", "containsBarrier",
    "containsChildLoops", "containsBranch", "tripCountKnown", "tripCount",
    "numBasicBlocks", "numMemoryInsts", "numComputeInsts", "numControlFlowInsts",
    "containsCall", "numExits",
]
# Indices of tripCountKnown / tripCount within the RAW (un-normalised) vector.
# Derived, not magic constants — if the structural order ever changes they follow.
IDX_TRIP_COUNT_KNOWN = STRUCTURAL_COLUMNS.index("tripCountKnown")   # 10
IDX_TRIP_COUNT = STRUCTURAL_COLUMNS.index("tripCount")             # 11

_FA_ITERS = os.environ.get("LOOPCOUNT_FA_ITERS", "8")

# Registry of optional embedding/aux blocks:
#   name -> (columns, extra -mllvm flags needed to EMIT it, available_in_toolchain?)
# Each flag string is one "-mllvm <arg>" unit (mirrors how hecbench appends flags).
# 'available' is False for blocks the current LoopCount build cannot emit yet;
# enabling one raises a clear error rather than silently zero-filling.
BLOCKS = {
    "emb":  ([f"emb{i}"  for i in range(IR2VEC_DIM)], [], True),   # SYM IR2Vec (emitted whenever loopcount+vocab)
    "femb": ([f"femb{i}" for i in range(IR2VEC_DIM)],
             ["-mllvm -loopcount-emit-fa", f"-mllvm -loopcount-fa-iters={_FA_ITERS}"], True),  # flow-aware IR2Vec
    # FUTURE (§1e) — declared so the schema accounts for them, gated until the C++ emits them:
    "kemb": ([f"kemb{i}" for i in range(IR2VEC_DIM)],
             ["-mllvm -loopcount-emit-kernel-emb"], False),        # kernel-context embedding (flag name TBD)
    # "widths": (...) add when the IR-level type-width histogram lands (size TBD).
}
# Canonical concatenation order (structural is prepended separately, always first).
BLOCK_ORDER = ["emb", "femb", "kemb"]


def _parse_enabled() -> list:
    req = [b.strip() for b in os.environ.get("UU_FEATURE_BLOCKS", "emb").split(",")
           if b.strip() and b.strip() != "structural"]
    unknown = [b for b in req if b not in BLOCKS]
    if unknown:
        raise ValueError(
            f"UU_FEATURE_BLOCKS has unknown block(s) {unknown}; valid: {list(BLOCKS)}")
    enabled = [b for b in BLOCK_ORDER if b in req]     # canonical order, de-duplicated
    unavailable = [b for b in enabled if not BLOCKS[b][2]]
    if unavailable:
        raise ValueError(
            f"feature block(s) {unavailable} are not emitted by the current LoopCount "
            f"build yet — remove them from UU_FEATURE_BLOCKS.")
    if not enabled:
        raise ValueError(
            "UU_FEATURE_BLOCKS enabled no embedding block; expected at least one of "
            f"{list(BLOCKS)} (structural is always included implicitly).")
    return enabled


ENABLED_BLOCKS = _parse_enabled()

FEATURE_COLUMNS = STRUCTURAL_COLUMNS + [c for b in ENABLED_BLOCKS for c in BLOCKS[b][0]]
N_FEATURES = len(FEATURE_COLUMNS)

# -mllvm flags needed to emit all enabled blocks (order-stable union).
EXTRA_LOOPCOUNT_FLAGS = [f for b in ENABLED_BLOCKS for f in BLOCKS[b][1]]

# First column of each enabled block — used by hecbench's presence guard to
# verify every enabled block's emit flag actually took effect.
ENABLED_BLOCK_FIRST_COLS = [BLOCKS[b][0][0] for b in ENABLED_BLOCKS]
# All columns of each enabled embedding block — used by the all-zero guard.
ENABLED_BLOCK_COLUMNS = [c for b in ENABLED_BLOCKS for c in BLOCKS[b][0]]

# Precheck-cache schema version. The legacy default ("emb" only) keeps the int 2
# so existing eligible_benchmarks.json / reward_cache remain valid; any other
# block set gets a distinct string that invalidates a stale cache on load.
FEATURES_VERSION = 2 if ENABLED_BLOCKS == ["emb"] else "v3:" + "+".join(ENABLED_BLOCKS)


def assert_matches(cache_features_version, where: str = "") -> None:
    """
    Raise if a loaded cache was extracted under a different feature schema than
    the current environment. Use in eval/analysis loaders that read a completed
    run's pre_features_raw WITHOUT going through the training precheck (which
    re-extracts on mismatch). Prevents silently feeding e.g. SYM vectors to an
    FA-trained model — sym and fa are both 93-dim, so no shape error would fire.
    """
    if cache_features_version != FEATURES_VERSION:
        raise ValueError(
            f"feature-schema mismatch{(' in ' + where) if where else ''}: cache was "
            f"extracted under features_version={cache_features_version!r} but this "
            f"environment is features_version={FEATURES_VERSION!r} "
            f"(UU_FEATURE_BLOCKS={','.join(ENABLED_BLOCKS)}). Set UU_FEATURE_BLOCKS to "
            f"match the run you are loading.")
