"""
Presentation plots for the per-loop UU RL training run.

Reads a metrics.csv (as written by train.py) and produces clean, consistent
figures that pair with initial_results.md:

  fig_performance.png       — "Measured speedup over training" (Result 1)
  fig_decision_policy.png   — "How often the model chooses to act" (Result 2)
  fig_training_overview.png — internal multi-panel tracking view

Design: single y-axis per chart (never dual-axis), a fixed colorblind-safe
palette used consistently across figures, recessive grid/axes, a legend for
every multi-series chart, and one direct value label per line (latest value).

Usage:
  python plot_metrics.py metrics.csv [--outdir figs]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

# --- Palette (validated, colorblind-safe; light surface) -------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"   # primary text
INK_2     = "#52514e"   # secondary text
MUTED     = "#898781"   # axis / ticks
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"   # zero / reference line

BLUE      = "#2a78d6"   # primary series (train / no-op / calm)
ORANGE    = "#eb6834"   # secondary series (held-out / aggressive)
VIOLET    = "#4a3aa7"
AQUA      = "#1baf7a"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   SURFACE,
        "savefig.facecolor":  SURFACE,
        "axes.facecolor":     SURFACE,
        "axes.edgecolor":     BASELINE,
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         GRID,
        "grid.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        MUTED,
        "ytick.color":        MUTED,
        "xtick.labelcolor":   INK_2,
        "ytick.labelcolor":   INK_2,
        "axes.labelcolor":    INK_2,
        "text.color":         INK,
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          11,
        "legend.frameon":     False,
        "lines.solid_capstyle": "round",
    })


def _titles(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                 color=INK, pad=22)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=10.5, color=INK_2, va="bottom")


def _pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}%"))


def _end_label(ax, x, y, text, color) -> None:
    """One direct value label at the last point (never a number on every point)."""
    ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=10, color=INK_2,
                fontweight="bold")


def _grid_only_y(ax) -> None:
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)


def load_metrics(path: Path) -> pd.DataFrame:
    """
    Robust metrics reader.

    Tolerates a metrics.csv that grew columns mid-life (e.g. a checkpoint dir
    reused across schema versions, where newer rows carry more fields than the
    original header).  Rows are padded/truncated to the header width — because
    new columns are always appended, truncation keeps the original columns
    intact and simply omits any that post-date the header.  Any column the
    header doesn't name is unavailable; plots that need it are skipped.
    """
    import csv as _csv
    rows = list(_csv.reader(open(path, newline="")))
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    header = rows[0]
    width = len(header)
    data = []
    for r in rows[1:]:
        if not r or r[0].strip() == "epoch":   # skip blanks / repeated headers
            continue
        data.append((r + [""] * width)[:width])
    df = pd.DataFrame(data, columns=header)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["epoch"].notna()].copy()
    df["epoch"] = df["epoch"].astype(int)
    return df.sort_values("epoch").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 1 — Measured speedup over training
# ---------------------------------------------------------------------------

def plot_performance(df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    e = df["epoch"]
    train = df["train_avg_reward"] * 100
    val   = df["val_avg_reward"] * 100

    # Baseline reference (no transform) and a faint "faster than baseline" band.
    ax.axhline(0, color=BASELINE, lw=1.2, ls="--", zorder=1)
    top = max(val.max(), train.max()) * 1.25 + 1
    ax.axhspan(0, top, color=BLUE, alpha=0.04, zorder=0)

    ax.plot(e, train, color=BLUE, lw=2, marker="o", ms=5,
            label="Training loops", zorder=3)
    ax.plot(e, val, color=ORANGE, lw=2, marker="o", ms=5,
            label="Held-out benchmarks", zorder=3)

    # No end-value labels here: the held-out series is sampled from a still-
    # exploring policy and its final point is a variance dip, so labelling the
    # last value would misrepresent the trend.  Legend + axis carry it.
    ax.text(e.iloc[0], 0, "  baseline (no transform)", va="bottom", ha="left",
            fontsize=9, color=MUTED)

    _titles(ax, "Measured speedup over training",
            "Mean kernel-time improvement vs. the untransformed baseline. Higher is faster.")
    ax.set_xlabel("Training progress (epochs)")
    ax.set_ylabel("Mean speedup")
    ax.set_ylim(top=top)
    ax.margins(x=0.08)
    _pct_axis(ax)
    _grid_only_y(ax)
    ax.legend(loc="lower right", ncol=2, fontsize=10, handlelength=1.4)

    out = outdir / "fig_performance.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — How often the model chooses to act
# ---------------------------------------------------------------------------

def _plot_decision_mix(df: pd.DataFrame, outdir: Path, *, prefix: str, title: str,
                       subtitle: str, filename: str) -> "Path | None":
    """
    Decision-mix chart for either the training-time (sampled) policy or the
    held-out (greedy) policy, depending on *prefix* ('train' or 'val').

    The three actions partition every loop decision (mutually exclusive,
    exhaustive): aggressive path-unmerge, lightweight unroll-only, or no-op.
    unroll-only is derived so the three sum to 100%.  Returns None if the
    required columns are absent (older metrics schema).
    """
    ncol, ncol2 = f"{prefix}_noop_rate", f"{prefix}_unmerge_rate"
    if ncol not in df.columns or ncol2 not in df.columns:
        return None
    if df[ncol].isna().all() or df[ncol2].isna().all():
        return None

    fig, ax = plt.subplots(figsize=(8, 4.8))
    e = df["epoch"]
    unmerge = df[ncol2] * 100
    noop    = df[ncol] * 100
    # Prefer the explicit unroll-only column (newer metrics schema); fall back
    # to the residual (100 - unmerge - noop) for older runs that predate it.
    ncol3 = f"{prefix}_unroll_rate"
    if ncol3 in df.columns and not df[ncol3].isna().all():
        unroll = df[ncol3] * 100
    else:
        unroll = 100 - unmerge - noop

    ax.plot(e, unroll, color=AQUA, lw=2, marker="o", ms=5,
            label="Lightweight unroll-only", zorder=3)
    ax.plot(e, unmerge, color=ORANGE, lw=2, marker="o", ms=5,
            label="Aggressive path-unmerge", zorder=3)
    ax.plot(e, noop, color=BLUE, lw=2, marker="o", ms=5,
            label="No transformation (no-op)", zorder=3)

    _end_label(ax, e.iloc[-1], unroll.iloc[-1], f"{unroll.iloc[-1]:.0f}%", AQUA)
    _end_label(ax, e.iloc[-1], unmerge.iloc[-1], f"{unmerge.iloc[-1]:.0f}%", ORANGE)
    _end_label(ax, e.iloc[-1], noop.iloc[-1], f"{noop.iloc[-1]:.0f}%", BLUE)

    _titles(ax, title, subtitle)
    ax.set_xlabel("Training progress (epochs)")
    ax.set_ylabel("Share of loop decisions")
    ax.margins(x=0.10)
    ax.set_ylim(-3, 100)
    _pct_axis(ax)
    _grid_only_y(ax)
    ax.legend(loc="center right", fontsize=10, handlelength=1.4)

    out = outdir / filename
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_decision_policy(df: pd.DataFrame, outdir: Path) -> "Path | None":
    return _plot_decision_mix(
        df, outdir, prefix="train",
        title="How the model's choices shift as it learns",
        subtitle="A graded response emerges: prefer the cheap transform, escalate only where it pays.",
        filename="fig_decision_policy.png",
    )


def plot_decision_policy_val(df: pd.DataFrame, outdir: Path) -> "Path | None":
    return _plot_decision_mix(
        df, outdir, prefix="val",
        title="What the committed policy does on held-out code",
        subtitle="Greedy (deployment-mode) choices on unseen benchmarks as training proceeds.",
        filename="fig_decision_policy_val.png",
    )


# ---------------------------------------------------------------------------
# Figure 3 — Internal multi-panel overview
# ---------------------------------------------------------------------------

def _has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and not df[col].isna().all()


def plot_training_overview(df: pd.DataFrame, outdir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    e = df["epoch"]

    # (a) Speedup
    ax = axes[0, 0]
    ax.axhline(0, color=BASELINE, lw=1.1, ls="--")
    if _has(df, "train_avg_reward"):
        ax.plot(e, df["train_avg_reward"] * 100, color=BLUE, lw=2, marker="o", ms=4,
                label="Training")
    if _has(df, "val_avg_reward"):
        ax.plot(e, df["val_avg_reward"] * 100, color=ORANGE, lw=2, marker="o", ms=4,
                label="Held-out")
    _titles(ax, "Mean speedup", "vs. untransformed baseline")
    ax.set_ylabel("Speedup")
    _pct_axis(ax); _grid_only_y(ax)
    ax.legend(loc="lower right", fontsize=9)

    # (b) Decisiveness (entropy) — plain-language label
    ax = axes[0, 1]
    if _has(df, "train_entropy"):
        ax.plot(e, df["train_entropy"], color=VIOLET, lw=2, marker="o", ms=4)
        ax.invert_yaxis()
    _titles(ax, "Policy decisiveness",
            "lower = more deliberate, sharper choices")
    ax.set_ylabel("Decision entropy")
    _grid_only_y(ax)

    # (c) Predictor calibration (value loss)
    ax = axes[1, 0]
    if _has(df, "train_value_loss"):
        ax.plot(e, df["train_value_loss"], color=AQUA, lw=2, marker="o", ms=4)
    _titles(ax, "Outcome-predictor error",
            "lower = better calibrated to measured results")
    ax.set_ylabel("Prediction error")
    ax.set_xlabel("Training progress (epochs)")
    _grid_only_y(ax)

    # (d) Cache hit rate (efficiency)
    ax = axes[1, 1]
    if _has(df, "train_cache_hit_rate"):
        ax.plot(e, df["train_cache_hit_rate"] * 100, color=BLUE, lw=2, marker="o",
                ms=4, label="Training")
    if _has(df, "val_cache_hit_rate"):
        ax.plot(e, df["val_cache_hit_rate"] * 100, color=ORANGE, lw=2, marker="o",
                ms=4, label="Held-out")
    _titles(ax, "Measurement reuse",
            "fraction served from cache — rises as coverage grows")
    ax.set_ylabel("Cache hit rate")
    ax.set_xlabel("Training progress (epochs)")
    ax.set_ylim(0, 100)
    _pct_axis(ax); _grid_only_y(ax)
    ax.legend(loc="upper left", fontsize=9)

    out = outdir / "fig_training_overview.png"
    fig.tight_layout(pad=2.0)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", help="Path to metrics.csv")
    p.add_argument("--outdir", default="figs", help="Output directory (default: figs)")
    args = p.parse_args()

    _style()
    df = load_metrics(Path(args.metrics))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figs = [
        plot_performance(df, outdir),
        plot_decision_policy(df, outdir),
        plot_decision_policy_val(df, outdir),   # None if val action-mix absent
        plot_training_overview(df, outdir),
    ]
    for f in figs:
        if f is not None:
            print(f"wrote {f}")
        else:
            print("skipped a figure (columns not present in this metrics.csv)")


if __name__ == "__main__":
    main()
