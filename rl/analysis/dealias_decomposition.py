"""
De-alias decomposition — reward-INDEPENDENT collision-shrinkage counting.

Implements Sequencing step 0 / the Analysis "3-axis de-alias decomposition" in
followup_plan.md: it validates the §1e/§1f representation work (kernel-context
`kemb`, flow-aware `femb`, and — when it lands — IR type `widths`) by measuring how
much of the feature-collision structure each axis removes.

INPUT-EMBEDDINGS ONLY — NO REWARDS.  A "collision" here is two eligible loops with a
byte-identical BASE feature vector (structural + symbolic emb).  Whether an added
axis makes them distinct is a property of the EXTRACTOR, not of any reward label, so
the whole check is reward-independent, safe on a partial grid, and transfers to fresh
data unchanged.  Nothing here reads labels, categories, or the reward table — the
label-dependent questions (which collisions are cross-category, the fitting-ceiling
lift) are deliberately out of scope and wait for the fresh grid.

AXES.  Detected by column presence, so the script works on any extraction:
    flow      femb0..74   (§1f flow-aware)           — live
    boundary  kemb0..74   (§1e kernel-context)        — live
    type      widths      (§1e IR type-width hist)    — activates when that block
                                                        is emitted; reported as
                                                        UNAVAILABLE until then.
Until the type axis exists, the reported "floor" is the floor W.R.T. AVAILABLE AXES
(flow+boundary); the plan's cub-instantiation collisions are expected to survive it
and only fall to the type axis.

USAGE
  # extract fresh across the tree (GPU-free, but compiles each benchmark once):
  UU_FEATURE_BLOCKS=emb,femb,kemb IR2VEC_VOCAB=/path/seed75.json \
    python3 analysis/dealias_decomposition.py --hecbench-src /path/HeCBench/src \
            --save-features feats.csv
  # re-run instantly on the saved table (no recompile):
  python3 analysis/dealias_decomposition.py --features feats.csv
  # verify the counting logic with no compiler/benchmarks needed:
  python3 analysis/dealias_decomposition.py --selftest
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import features as _feat

IR2VEC_DIM = _feat.IR2VEC_DIM
STRUCT = list(_feat.STRUCTURAL_COLUMNS)
EMB = [f"emb{i}" for i in range(IR2VEC_DIM)]
FEMB = [f"femb{i}" for i in range(IR2VEC_DIM)]
KEMB = [f"kemb{i}" for i in range(IR2VEC_DIM)]
BASE_COLS = STRUCT + EMB

# The base is the CURRENT model input minus the extra embedding blocks: structural
# (18) + symbolic emb (75) = the 93-dim vector the aliasing audit was computed on.


def width_cols(df: pd.DataFrame) -> list:
    """Type-axis (widths) columns, once that block exists — else []. Forward-
    compatible: prefer the registry's names, fall back to a 'width' prefix."""
    if "widths" in getattr(_feat, "BLOCKS", {}):
        cols = _feat.BLOCKS["widths"][0]
        if all(c in df.columns for c in cols):
            return list(cols)
    pref = [c for c in df.columns if str(c).startswith("width")]
    return pref


# ---------------------------------------------------------------------------
# Collision counting (pure, testable — no I/O, no compiler)
# ---------------------------------------------------------------------------

def _vecs(df: pd.DataFrame, cols: list, ndp: int) -> list:
    """One hashable rounded tuple per row over `cols`, in row order."""
    return [tuple(x) for x in df[cols].astype(float).round(ndp).to_numpy()]


def _groups(vecs: list) -> dict:
    """Map each distinct vector -> list of ROW POSITIONS holding it. Position-based
    (never id-based) so duplicate loop ids can't collapse distinct rows."""
    g: dict = defaultdict(list)
    for pos, v in enumerate(vecs):
        g[v].append(pos)
    return g


def _stats(vecs: list) -> tuple:
    """(n_distinct, n_colliding_loops, n_collision_groups, collision_map)."""
    g = _groups(vecs)
    coll = {v: m for v, m in g.items() if len(m) > 1}
    return len(g), sum(len(m) for m in coll.values()), len(coll), coll


def decompose(df: pd.DataFrame, avail_axes: list, ndp: int) -> dict:
    """
    avail_axes: ordered list of (name, columns) for the axes present, e.g.
        [("flow", FEMB), ("boundary", KEMB), ("type", WIDTHS)].
    Returns global shrinkage stats + an attribution of the BASE collision groups.

    All counting is by ROW POSITION — never by a loop id — so the result is correct
    even if the id column has duplicates.
    """
    base = _vecs(df, BASE_COLS, ndp)
    per_axis = {name: _vecs(df, BASE_COLS + cols, ndp) for name, cols in avail_axes}
    all_cols = BASE_COLS + [c for _, cols in avail_axes for c in cols]
    allv = _vecs(df, all_cols, ndp)

    stats = {"base": _stats(base)[:3]}
    for name, _ in avail_axes:
        stats[name] = _stats(per_axis[name])[:3]
    d_all, l_all, g_all, _ = _stats(allv)
    stats["all"] = (d_all, l_all, g_all)

    # Attribution over the BASE collision groups (members are ROW POSITIONS).
    _, _, _, base_coll = _stats(base)

    def fully(vecs: list, members: list) -> bool:
        """True iff every member has a DISTINCT vector (group fully split)."""
        return len({vecs[p] for p in members}) == len(members)

    resolved_by = {name: 0 for name, _ in avail_axes}   # this axis ALONE resolves it
    combo_only = 0        # fully resolved by all-axes but by no single axis alone
    fully_resolved = 0    # fully resolved by all-axes (single or combo)
    for members in base_coll.values():
        singles = [name for name, _ in avail_axes if fully(per_axis[name], members)]
        for name in singles:
            resolved_by[name] += 1
        if fully(allv, members):
            fully_resolved += 1
            if not singles:
                combo_only += 1

    # The FLOOR is exactly what still collides under all axes. Because `all` is a
    # refinement of `base`, every all-collision is a subset of a base-collision, so
    # this equals the "+ all available" row of the shrinkage table by construction
    # (single source of truth — the floor line and the table can never disagree).
    return {
        "n_loops": len(df),
        "axes": [n for n, _ in avail_axes],
        "stats": stats,                 # name -> (distinct, colliding_loops, groups)
        "base_groups": len(base_coll),
        "base_colliding": stats["base"][1],
        "resolved_by": resolved_by,     # per-axis (OVERLAPPING) alone-resolvable groups
        "combo_only": combo_only,
        "unresolved_base_groups": len(base_coll) - fully_resolved,
        "floor_loops": l_all,           # loops still colliding after +all
        "floor_subgroups": g_all,       # residual collision groups after +all
    }


def assert_monotonic(stats: dict, axes: list) -> None:
    """Adding columns can only refine a partition — colliding loops must not rise,
    distinct must not fall.  A violation means the counting is broken."""
    d0, l0, _ = stats["base"]
    dall, lall, _ = stats["all"]
    for name in axes:
        d, l, _ = stats[name]
        assert d >= d0, f"{name}: distinct fell below base ({d} < {d0})"
        assert l <= l0, f"{name}: colliding rose above base ({l} > {l0})"
        assert lall <= l, f"all: colliding rose above {name} ({lall} > {l})"
    assert dall >= d0 and lall <= l0, "all-axes not a refinement of base"


# ---------------------------------------------------------------------------
# Feature acquisition
# ---------------------------------------------------------------------------

def extract_features(src: Path, limit: int) -> pd.DataFrame:
    from hecbench import discover_benchmarks, get_loop_features
    benches = discover_benchmarks(src)
    if limit:
        benches = benches[:limit]
    frames = []
    for b in benches:
        try:
            fm, _, _ = get_loop_features(b)
        except Exception as e:
            print(f"  skip {b.name}: {e}", file=sys.stderr)
            continue
        for fn, df in fm.items():
            df = df.copy()
            df["benchmark"] = b.name
            df["__file"] = fn
            frames.append(df)
    if not frames:
        sys.exit("no features extracted — check --hecbench-src, IR2VEC_VOCAB, the build")
    return pd.concat(frames, ignore_index=True)


def add_id(df: pd.DataFrame) -> str:
    """Globally-unique per-loop id; returns the column name."""
    df["__id"] = (df["benchmark"].astype(str) + "|" + df["__file"].astype(str)
                  + "|" + df["loopIdx"].astype(str))
    return "__id"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(res: dict, avail: list, missing: list) -> None:
    st = res["stats"]
    print("=" * 74)
    print("  De-alias decomposition (reward-independent, feature-side counting)")
    print("=" * 74)
    print(f"  loops analysed         : {res['n_loops']}")
    print(f"  axes available         : {', '.join(res['axes']) or '(none)'}")
    if missing:
        print(f"  axes NOT YET available : {', '.join(missing)}  "
              f"(floor below is w.r.t. available axes only)")
    print()
    hdr = f"  {'feature set':<26}{'distinct':>9}{'colliding':>11}{'groups':>8}{'Δ loops':>9}"
    print(hdr)
    base_l = st["base"][1]

    def row(label, key):
        d, l, g = st[key]
        print(f"  {label:<26}{d:>9}{l:>11}{g:>8}{l - base_l:>+9}")

    row("BASE (struct+emb, 93d)", "base")
    for name in res["axes"]:
        row(f"+ {name}", name)
    if len(res["axes"]) > 1:
        row("+ all available", "all")
    print()
    print("  Attribution of BASE collision groups"
          f"  (total {res['base_groups']} groups / {res['base_colliding']} loops):")
    for name in res["axes"]:
        print(f"    resolvable by {name:<10} alone       : {res['resolved_by'][name]}")
    if len(res["axes"]) > 1:
        print(f"    resolvable only by a combination   : {res['combo_only']}")
    print(f"    not resolved by any available axis : {res['unresolved_base_groups']}")
    print(f"    (per-axis 'alone' counts overlap; 'not resolved' is the clean floor)")
    print(f"  Floor — still colliding after +all : {res['floor_loops']} loops "
          f"/ {res['floor_subgroups']} groups  (= the '+ all available' row)")
    if missing:
        print(f"    NOTE: the {', '.join(missing)} axis is not built — cub-instantiation "
              f"collisions are expected to sit in this floor until it lands.")
    print()


# ---------------------------------------------------------------------------
# Self-test (no compiler / benchmarks needed) — verifies the counting logic
# ---------------------------------------------------------------------------

def _synthetic_df() -> pd.DataFrame:
    """Designed base-collision groups spanning every resolvability class.
    (femb0, kemb0) per member; all members of a group share the same base emb0.
        A pair  femb same,   kemb differ         -> boundary-only
        B pair  femb differ, kemb same           -> flow-only
        C pair  femb same,   kemb same           -> floor (fully survives +all)
        D pair  femb differ, kemb differ         -> either single axis
        E triple no single axis splits it, but flow+kemb together do -> combo-only
        F triple splits only PARTIALLY under +all -> residual floor (2 of 3)
        U       unique base                      -> never collides
    """
    def loop(bidx, femb0, kemb0, lidx):
        r = {c: 0.0 for c in BASE_COLS + FEMB + KEMB}
        r["emb0"] = float(bidx)      # base signature (a group shares it)
        r["femb0"] = float(femb0)
        r["kemb0"] = float(kemb0)
        r["benchmark"] = "synth"
        r["__file"] = "s.cu"
        r["loopIdx"] = lidx          # int, as in real extractions
        return r

    rows = [
        loop(10, 0, 1, 0), loop(10, 0, 2, 1),                       # A boundary-only
        loop(20, 1, 0, 2), loop(20, 2, 0, 3),                       # B flow-only
        loop(30, 0, 0, 4), loop(30, 0, 0, 5),                       # C floor (full)
        loop(40, 1, 1, 6), loop(40, 2, 2, 7),                       # D either
        loop(50, 0, 0, 8),                                          # U unique base
        loop(60, 1, 1, 9), loop(60, 1, 2, 10), loop(60, 2, 1, 11),  # E combo-only
        loop(70, 1, 1, 12), loop(70, 1, 1, 13), loop(70, 1, 2, 14),  # F partial floor
    ]
    return pd.DataFrame(rows)


def selftest() -> int:
    df = _synthetic_df()
    add_id(df)                     # exercises the id path; decompose is position-based
    axes = [("flow", FEMB), ("boundary", KEMB)]
    res = decompose(df, axes, ndp=6)
    assert_monotonic(res["stats"], res["axes"])

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))

    st = res["stats"]
    # 15 loops; base groups A,B,C,D,E,F (U unique) = 6 groups / 14 colliding, 7 distinct
    check("base: 6 groups / 14 colliding / 7 distinct",
          st["base"] == (7, 14, 6), f"{st['base']}")
    # +flow: A,C collide(2 each), E->{E1,E2}(2), F->all 3 -> 9 colliding in 4 groups.
    # distinct = 15 - (9-4) = 10 (E collapses to (60,1),(60,2); F to one vector).
    check("+flow: 9 colliding / 4 groups / 10 distinct", st["flow"] == (10, 9, 4),
          f"{st['flow']}")
    # +boundary: B,C(2 each), E->{E1,E3}(2), F->{F1,F2}(2) -> 8 colliding in 4 groups
    check("+boundary: 8 colliding / 4 groups", st["boundary"] == (11, 8, 4),
          f"{st['boundary']}")
    # +all: only C(2) and F1,F2(2) survive -> 4 colliding in 2 groups, 13 distinct
    check("+all: 4 colliding / 2 groups / 13 distinct", st["all"] == (13, 4, 2),
          f"{st['all']}")
    check("flow alone resolves 2 groups (B,D)", res["resolved_by"]["flow"] == 2,
          f"{res['resolved_by']['flow']}")
    check("boundary alone resolves 2 groups (A,D)", res["resolved_by"]["boundary"] == 2,
          f"{res['resolved_by']['boundary']}")
    check("combo-only resolves 1 group (E)", res["combo_only"] == 1, f"{res['combo_only']}")
    check("unresolved base groups = 2 (C,F)", res["unresolved_base_groups"] == 2,
          f"{res['unresolved_base_groups']}")
    # floor: C(2) fully survives + F partial (F1,F2) -> 4 loops in 2 residual groups
    check("floor = 4 loops / 2 groups (C + F-residual)",
          res["floor_loops"] == 4 and res["floor_subgroups"] == 2,
          f"{res['floor_loops']}l/{res['floor_subgroups']}g")
    check("floor loops == '+all' colliding row (single source of truth)",
          res["floor_loops"] == st["all"][1] and res["floor_subgroups"] == st["all"][2])

    # Edge: no axes available (emb-only extraction) — floor is the whole base set,
    # nothing resolves, and monotonicity still holds (all == base).
    res0 = decompose(df, [], ndp=6)
    assert_monotonic(res0["stats"], res0["axes"])
    check("no-axes: floor == base collisions",
          res0["floor_loops"] == 14 and res0["floor_subgroups"] == 6
          and res0["unresolved_base_groups"] == 6 and res0["combo_only"] == 0,
          f"floor={res0['floor_loops']}l/{res0['floor_subgroups']}g "
          f"unresolved={res0['unresolved_base_groups']}")

    # Duplicate loop id must NOT corrupt counts (position-based). Clone row 0's id
    # onto row 1 and confirm the stats are unchanged from the clean run.
    df_dup = df.copy()
    add_id(df_dup)
    df_dup.loc[1, "__id"] = df_dup.loc[0, "__id"]
    res_dup = decompose(df_dup, axes, ndp=6)
    check("duplicate id does not change collision stats",
          res_dup["stats"] == res["stats"]
          and res_dup["floor_loops"] == res["floor_loops"],
          "position-based counting is id-agnostic")

    print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, help="pre-extracted feature table (CSV)")
    p.add_argument("--hecbench-src", type=Path, help="benchmark tree to extract from")
    p.add_argument("--save-features", type=Path, help="write the extracted table here")
    p.add_argument("--limit", type=int, default=0, help="cap benchmarks (0 = all)")
    p.add_argument("--round", type=int, default=6,
                   help="decimals for collision equality (CSV emits %%.6f)")
    p.add_argument("--selftest", action="store_true",
                   help="verify the counting logic with synthetic data; no compiler")
    args = p.parse_args()

    if args.selftest:
        return selftest()

    if args.features:
        df = pd.read_csv(args.features)
        if "benchmark" not in df.columns or "loopIdx" not in df.columns:
            sys.exit("--features table lacks 'benchmark'/'loopIdx' columns")
        if "__file" not in df.columns:
            df["__file"] = ""
    elif args.hecbench_src:
        df = extract_features(args.hecbench_src, args.limit)
        if args.save_features:
            df.to_csv(args.save_features, index=False)
            print(f"features written: {args.save_features}  ({len(df)} loops)\n")
    else:
        sys.exit("give --features <table.csv>, --hecbench-src <tree>, or --selftest")

    missing_base = [c for c in BASE_COLS if c not in df.columns]
    if missing_base:
        sys.exit(f"feature table missing base columns (first: {missing_base[:3]}) — "
                 f"extract with at least UU_FEATURE_BLOCKS=emb")

    id_col = add_id(df)
    if df[id_col].duplicated().any():
        dups = int(df[id_col].duplicated().sum())
        print(f"WARNING: {dups} rows share a loop id (benchmark|file|loopIdx). Counts "
              f"below are position-based and still correct, but a shared id can mean a "
              f"duplicated ROW upstream — check the extraction.\n", file=sys.stderr)

    avail_axes, missing = [], []
    for name, cols in (("flow", FEMB), ("boundary", KEMB), ("type", width_cols(df))):
        if cols and all(c in df.columns for c in cols):
            avail_axes.append((name, cols))
        else:
            missing.append(name)

    analysed = BASE_COLS + [c for _, cols in avail_axes for c in cols]
    n_nan = int(df[analysed].isna().to_numpy().sum())
    if n_nan:
        print(f"WARNING: {n_nan} NaN cells in analysed columns — NaN never equals "
              f"itself, so those rows cannot collide and the floor is UNDER-counted. "
              f"Fix the extraction (strict features should have none).\n",
              file=sys.stderr)

    res = decompose(df, avail_axes, args.round)
    assert_monotonic(res["stats"], res["axes"])
    print_report(res, avail_axes, missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
