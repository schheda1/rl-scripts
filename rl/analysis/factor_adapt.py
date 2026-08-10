"""
Few-shot adaptation for the ISOLATED unroller.

The offline adaptation protocol, applied to the factor-only agent: train a base
agent on the training benchmarks, then for each held-out benchmark fine-tune a
FRESH COPY on a couple of that benchmark's measured loops and score it on the
rest. Zero-shot and adapted are scored on IDENTICAL evaluation loops, so the
difference is adaptation and not loop composition.

WHY THE ISOLATED HEAD
---------------------
The known few-shot win was category-side (accuracy 43 -> 64.5%, unmerge recall
49.3 -> 76.4%); whether adaptation ever reached the FACTOR was never measured.
With no category head in the way, everything here is the factor.

WHAT IS UNFROZEN
----------------
`FactorActor` is `_MLP`, so its stack is
    0 Linear(93,128)  1 LayerNorm  2 ReLU
    3 Linear(128,64)  4 LayerNorm  5 ReLU
    6 Linear(64,out)
and "layers" counts LINEARs, each carrying its LayerNorm:

    --adapt-unfreeze 1 -> net[6]              650 parameters
    --adapt-unfreeze 2 -> net[3], net[4], net[6]   9,034

Two adaptation loops supply at most ~38 measured cells. At 1 the fit is
constrained; at 2 the head can memorise those cells outright. If the ADAPTATION
loops improve while the evaluation loops do not, that is what happened — which
is why both settings exist and the choice is empirical.

WHAT IS REUSED, NOT REIMPLEMENTED
---------------------------------
`FactorOnly`, `train`, `evaluate`, `states_for`, `reward_type` handling and the
constant-factor bar all come from factor_only; `split_for_adaptation` mirrors
adapt_eval's own rule (>=3 loops give 2 adaptation loops, 2 give 1, 1 gives 0
and becomes a control). Only freezing, the fine-tune step and the paired
reporting are new.
"""

import argparse
import copy
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from agent import FACTOR_VALUES                                  # noqa: E402
from factor_only import (best_constant_factor, evaluate,        # noqa: E402
                         labelled_loops, load_run, loops_for,
                         random_split, states_for, train)
from offline_train import write_csv                              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("factor_adapt")

# Trailing parameterised layers, by index into _MLP's Sequential. Each Linear
# carries its LayerNorm; ReLU holds nothing.
GROUPS = {1: [6], 2: [3, 4, 6]}


def freeze_for_adaptation(agent, n_layers: int) -> list:
    """Freeze everything but the trailing `n_layers`, return the trainable set."""
    if n_layers not in GROUPS:
        raise ValueError(f"--adapt-unfreeze must be 1 or 2, got {n_layers}")
    net = agent.factor_actor.net
    # The indices are a hardcoded read of _MLP's stack. If it is ever reordered
    # this must fail loudly rather than fine-tune the wrong layers and report
    # adaptation numbers that are quietly wrong.
    expect = {3: nn.Linear, 4: nn.LayerNorm, 6: nn.Linear}
    for i, cls in expect.items():
        if not isinstance(net[i], cls):
            raise AssertionError(
                f"_MLP layout changed: net[{i}] is {type(net[i]).__name__}, "
                f"expected {cls.__name__} — GROUPS is stale")
    for p in agent.factor_actor.parameters():
        p.requires_grad = False
    trainable = []
    for i in GROUPS[n_layers]:
        for p in net[i].parameters():
            p.requires_grad = True
            trainable.append(p)
    return trainable


def adaptation_rows(loops: list, data: dict) -> list:
    """
    [(state, factor_index, reward)] over the MEASURED, LEGAL cells of `loops`.

    Measured only, mirroring offline_adapt._fit_bandit: the premise of few-shot
    is that these loops were actually benchmarked, so an absent cell is one that
    was never paid for. That differs from TRAINING, where an absent cell is
    charged the penalty because collection there was exhaustive — the asymmetry
    is deliberate and is what keeps the adaptation budget honest.
    """
    rows = []
    for l in loops:
        table = data["tables"][(l["benchmark_name"], l["loop_idx"])]
        for u, s2, mask in states_for(l, data):
            for i, f in enumerate(FACTOR_VALUES):
                if bool(mask[i]) and (u, f) in table:
                    rows.append((s2, i, table[(u, f)]))
    return rows


def adapt_in_place(agent, loops: list, data: dict, args):
    """Fine-tune a COPY's trailing layers on the adaptation loops' measured cells."""
    trainable = freeze_for_adaptation(agent, args.adapt_unfreeze)
    rows = adaptation_rows(loops, data)
    if not rows or not trainable:
        return agent, 0, float("nan"), float("nan")
    s = torch.stack([r[0] for r in rows])
    a = torch.tensor([r[1] for r in rows], dtype=torch.long)
    r = torch.tensor([r[2] for r in rows], dtype=torch.float32)
    opt = torch.optim.AdamW(trainable, lr=args.adapt_lr)
    first = last = float("nan")
    for step in range(args.adapt_steps):
        q = agent.factor_actor.forward(s).gather(1, a.unsqueeze(1)).squeeze(1)
        l_ = F.mse_loss(q, r)
        opt.zero_grad()
        l_.backward()
        if agent.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, agent.max_grad_norm)
        opt.step()
        last = float(l_)
        if step == 0:
            first = last
    # first AND last: a fit that never moves and a fit that collapses to zero
    # are different failures, and only the pair distinguishes them.
    return agent, len(rows), first, last


def split_for_adaptation(loops: list, rng: random.Random) -> tuple:
    """MIRROR: adapt_eval.split_adaptation_loops — imported, never re-derived."""
    from adapt_eval import split_adaptation_loops
    return split_adaptation_loops(loops, rng)


def run(data: dict, args) -> None:
    loops = labelled_loops(data)
    rows, curve = [], []
    for sp in range(args.splits):
        sseed = args.split_seed + sp
        tr_b, te_b = random_split(data["benchmarks"], args.test_frac, sseed)
        fit_l = loops_for(loops, tr_b)
        if not fit_l:
            continue
        for si in range(args.seeds):
            seed = args.base_seed + si
            base, cov, hist = train(fit_l, data, args, seed)
            curve.extend(dict(split=sp + 1, seed=seed, **h) for h in hist)
            rng = random.Random(seed * 1000 + sseed)
            here, pooled_ev = [], []
            for b in sorted(te_b):
                bl = [l for l in loops if l["benchmark_name"] == b]
                if not bl:
                    continue
                ad_l, ev_l = split_for_adaptation(bl, rng)
                if not ev_l:
                    continue
                pooled_ev.extend(ev_l)
                zs = evaluate(base, ev_l, data, args)
                if ad_l:
                    # A FRESH copy per benchmark: adapting the shared base would
                    # carry one target's fine-tuning into the next, which is
                    # incremental training over the fold, not few-shot.
                    tuned = copy.deepcopy(base)
                    tuned, n_cells, l0, l1 = adapt_in_place(tuned, ad_l, data, args)
                    ad = evaluate(tuned, ev_l, data, args)
                else:
                    n_cells, l0, l1, ad = 0, float("nan"), float("nan"), zs
                here.append({
                    "split": sp + 1, "seed": seed, "benchmark": b,
                    "n_adapt_loops": len(ad_l), "n_eval_loops": len(ev_l),
                    "adapt_cells": n_cells,
                    "adapt_loss_first": l0, "adapt_loss_final": l1,
                    "coverage": cov,
                    "zs_probe": zs["probe"], "zs_probe_n": zs["probe_n"],
                    "ad_probe": ad["probe"], "bar": zs["probe_bar"],
                    "zs": zs, "ad": ad,
                })
            # The per-benchmark bar is fitted on that benchmark's ONE OR TWO
            # evaluation loops, so it is close to an oracle and almost
            # unbeatable. The honest reference is the best single fixed factor
            # over ALL held-out loops at once — one number a practitioner could
            # actually ship.
            if pooled_ev:
                pf, pbar, pn = best_constant_factor(
                    pooled_ev, data["tables"], data["labels"],
                    args.deadzone, args.missing)
                for r in here:
                    r["pooled_bar"] = pbar
                    r["pooled_bar_f"] = pf
                    r["pooled_bar_n"] = pn
            rows.extend(here)
    report(rows, args)
    if args.csv_out and rows:
        write_csv(Path(args.csv_out), [flatten(r) for r in rows])
        log.info("\n  per-benchmark rows: %s", args.csv_out)
    if args.curve_out and curve:
        write_csv(Path(args.curve_out), curve)
        log.info("  base-training q-loss curve: %s", args.curve_out)


def flatten(row: dict) -> dict:
    """
    One flat CSV record.

    NOT factor_only.flatten: that one prefixes its nested dicts `fit_` and
    `test_`, which here would label the ZERO-SHOT columns "fit" and the ADAPTED
    columns "test". Both names would be wrong and both would be read as the
    thing they are not.
    """
    out = {k: v for k, v in row.items() if k not in ("zs", "ad")}
    for side in ("zs", "ad"):
        for k, v in row[side].items():
            out[f"{side}_{k}"] = round(v, 6) if isinstance(v, float) else v
    return out


def _mean(xs: list) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def report(rows: list, args) -> None:
    if not rows:
        log.error("no benchmark produced an evaluation set")
        return
    ad = [r for r in rows if r["n_adapt_loops"] > 0]
    ctrl = [r for r in rows if r["n_adapt_loops"] == 0]
    scored = [r for r in ad if r["zs_probe"] == r["zs_probe"]
              and r["ad_probe"] == r["ad_probe"]]

    def pooled(side):
        num = sum(r[side]["probe_realized"] for r in scored)
        den = sum(r[side]["probe_oracle"] for r in scored)
        return num / den if den else float("nan")

    def med(xs):
        xs = sorted(x for x in xs if x == x)
        if not xs:
            return float("nan")
        h = len(xs) // 2
        return xs[h] if len(xs) % 2 else (xs[h - 1] + xs[h]) / 2

    nparams = {1: 650, 2: 9034}[args.adapt_unfreeze]
    log.info("\n" + "=" * 74)
    log.info("  FEW-SHOT ADAPTATION — isolated unroller, factor head only")
    log.info("=" * 74)
    log.info("  %d runs with adaptation loops | %d control (single-loop) | "
             "%d scorable", len(ad), len(ctrl), len(scored))
    log.info("  budget: %.1f loops, %.0f measured cells, %d steps, "
             "%d layer(s) = %d params\n",
             _mean([r["n_adapt_loops"] for r in ad]),
             _mean([r["adapt_cells"] for r in ad]),
             args.adapt_steps, args.adapt_unfreeze, nparams)

    log.info("  %-25s %9s %11s %11s", "", "zero-shot", "adapted", "delta")
    log.info("  " + "-" * 60)
    zp, ap = pooled("zs"), pooled("ad")
    log.info("  %-25s %8.1f%% %10.1f%% %9.1fpp", "capture (pooled)",
             100 * zp, 100 * ap, 100 * (ap - zp))
    zm = med([r["zs_probe"] for r in scored])
    am = med([r["ad_probe"] for r in scored])
    log.info("  %-25s %8.1f%% %10.1f%% %9.1fpp", "capture (median)",
             100 * zm, 100 * am, 100 * (am - zm))
    zr = _mean([r["zs"]["probe_mean_ran"] for r in scored])
    ar = _mean([r["ad"]["probe_mean_ran"] for r in scored])
    log.info("  %-25s %+9.4f %+11.4f %+11.4f", "mean realized (ran)",
             zr, ar, ar - zr)
    zs_ = _mean([r["zs"]["probe_slower"] for r in scored])
    as_ = _mean([r["ad"]["probe_slower"] for r in scored])
    log.info("  %-25s %9.1f %11.1f %11.1f", "loops slower (ran)",
             zs_, as_, as_ - zs_)
    zf = _mean([r["zs"]["probe_failed"] for r in scored])
    af = _mean([r["ad"]["probe_failed"] for r in scored])
    log.info("  %-25s %9.1f %11.1f %11.1f", "picks that never built",
             zf, af, af - zf)

    log.info("  " + "-" * 60)
    log.info("  by branch (pooled capture)")
    for u, label in ((0, "pre-unmerge state"), (1, "post-unmerge state")):
        def bp(side, u=u):
            ok = [r for r in scored if r[side][f"u{u}"] == r[side][f"u{u}"]]
            d = sum(r[side][f"u{u}_n"] for r in ok)
            n = sum(r[side][f"u{u}"] * r[side][f"u{u}_n"] for r in ok)
            return n / d if d else float("nan")
        log.info("    %-23s %8.1f%% %10.1f%%", label, 100 * bp("zs"),
                 100 * bp("ad"))

    log.info("  " + "-" * 60)
    log.info("  references on the same evaluation loops")
    # Every line here passes an argument. A log.info with NO args is not
    # %-formatted at all, so an escaped %% would print literally.
    log.info("    %-23s %8.1f%%   capture is 0 by definition", "always-no-op", 0.0)
    pb = _mean([r.get("pooled_bar", float("nan")) for r in scored])
    pbn = _mean([r.get("pooled_bar_n", 0) for r in scored])
    log.info("    %-23s %8.1f%%   one constant over ~%.0f held-out loops",
             "best fixed factor, pooled", 100 * pb, pbn)
    log.info("    %-23s %8.1f%%   refit on %.1f loops each — near-oracle,\n"
             "%s not a policy anyone could ship",
             "best fixed factor, per-bm",
             100 * _mean([r["bar"] for r in scored]),
             _mean([r["zs_probe_n"] for r in scored]), " " * 40)

    log.info("  " + "-" * 62)
    won = sum(1 for r in scored if r["ad_probe"] > r["zs_probe"])
    beat = sum(1 for r in scored
               if r["ad_probe"] > r.get("pooled_bar", float("inf")))
    log.info("  adapted > zero-shot on %d/%d   |   adapted > pooled bar on %d/%d",
             won, len(scored), beat, len(scored))
    log.info("  adaptation q-loss %.4f -> %.4f over %d steps",
             _mean([r["adapt_loss_first"] for r in ad]),
             _mean([r["adapt_loss_final"] for r in ad]), args.adapt_steps)
    log.info("\n  Judge against the POOLED bar. The per-benchmark bar is refit "
             "on the same one\n  or two loops it is scored on, so it is close to "
             "an oracle and losing to it\n  says almost nothing.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--deadzone", type=float, default=0.005)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--base-seed", type=int, default=100)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--buffer-size", type=int, default=128)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--epsilon-final", type=float, default=0.05)
    p.add_argument("--missing", type=float, default=-0.161)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--csv-out", type=str, default=None)
    p.add_argument("--curve-out", type=str, default=None)
    g = p.add_argument_group("few-shot adaptation")
    g.add_argument("--adapt-lr", type=float, default=1e-3)
    g.add_argument("--adapt-steps", type=int, default=50)
    g.add_argument("--adapt-unfreeze", type=int, default=1, choices=(1, 2),
                   help="Trailing parameterised layers to fine-tune. 1 = the "
                        "output projection (~650 params); 2 adds Linear(128,64) "
                        "and its LayerNorm (~8,900). Two adaptation loops give "
                        "~38 cells, so 2 can memorise them outright — pick "
                        "empirically and watch whether the ADAPTATION loops "
                        "improve while the evaluation loops do not.")
    args = p.parse_args()
    torch.set_num_threads(max(1, args.threads))
    data = load_run(args.run_dir, args.deadzone, args.labels)
    log.info("Loaded %d loops / %d benchmarks | %d labelled\n",
             len(data["loops"]), len(data["benchmarks"]),
             data["n_labelled_loops"])
    run(data, args)


if __name__ == "__main__":
    main()
