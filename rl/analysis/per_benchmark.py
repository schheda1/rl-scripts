"""
Per-benchmark RL: train on a couple of an application's loops, decide the rest.

WHAT THIS IS FOR
----------------
The study that normally comes BEFORE a cross-distribution one. This project did
the opposite -- generalization first -- so this fills the gap underneath.

Its job is narrow and positive: applications have per-loop headroom, and an RL
agent finds it. That establishes the RL formulation is sound, so the
generalization work built on it is a real continuation rather than a dead end.

It is NOT the main result. The main result is that cross-application transfer
fails; this is the floor that makes that claim readable, because without it
"generalization failed" cannot be told apart from "the pipeline never worked".

DESIGN
------
Population: benchmarks with >= 3 loops. Split per benchmark -- 2 training loops
if it has <= 5, otherwise 40% -- and evaluate on the rest, which the agent never
trains on. Budget is --epochs draws per training loop. Draws are WITH
replacement (epsilon-greedy re-picks), so 15 draws reach ~10.6 of a loop's 19
legal cells, not 15: the agent cannot enumerate its way to the answer.

Three numbers per benchmark and nothing more elaborate: always-no-op is exactly
+0.0000 by construction, the agent, and the benchmark's own oracle. On this
subset every single fixed action is negative (best is unmerge=1/f=3 at -0.0690)
while the oracle is +0.0871, so any gain has to come from per-loop decisions --
that is the only context the number needs.

Benchmarks with no headroom anywhere are a SEPARATE run (--no-headroom): there is
nothing to find, the only correct behaviour is to decline, and a method that
"recovers headroom" by firing everywhere shows up there as damage.

TRAINING IS THE PIPELINE'S, NOT A RE-IMPLEMENTATION
---------------------------------------------------
`run_epoch`, `rollout_one`, `make_agent` and `_schedules` are imported from
offline_train. The mid-epoch update cadence in particular is load-bearing
(train.py:2088-2090): updating once at the end of an epoch instead of when the
buffer fills gives four times fewer gradient steps on identical samples. Only
the split, the budget loop and the reporting are new here.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402

from offline_data import (labelled_loops, load_run,              # noqa: E402
                          oracle_of_gated, score_decisions)
from offline_train import (_mr, _schedules, build_parser,        # noqa: E402
                           greedy_picks, make_agent, run_epoch, write_csv)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("per_bench")

MIN_LOOPS = 3


def n_train(n_loops: int) -> int:
    """<=5 loops -> 2 for training; >5 -> 40%. Never fewer than 2, and always at
    least one loop left over to evaluate on."""
    n = 2 if n_loops <= 5 else max(2, round(0.4 * n_loops))
    return min(n, n_loops - 1)


def split(loops: list, seed: int) -> tuple:
    """(train, eval) for one benchmark. Sorted before shuffling so the draw is a
    function of the seed alone and not of dict iteration order."""
    ks = sorted(loops, key=lambda l: l["loop_idx"])
    random.Random(seed).shuffle(ks)
    n = n_train(len(ks))
    return ks[:n], ks[n:]


def has_headroom(loop: dict, tables: dict, deadzone: float) -> bool:
    key = (loop["benchmark_name"], loop["loop_idx"])
    return oracle_of_gated(tables[key], deadzone)[1] > deadzone


def train_on(kind: str, train_loops: list, data: dict, args, seed: int):
    """
    Train from scratch on one benchmark's training loops.

    No validation selection: with two training loops there is no split to select
    on, and selecting on the EVALUATION loops would leak them. The final-epoch
    agent is what gets scored.
    """
    from agent import RolloutBuffer

    torch.manual_seed(seed)
    random.seed(seed)
    agent = make_agent(kind, args)
    buf = RolloutBuffer(capacity=args.buffer_size)
    observed: dict = {}
    order = list(train_loops)
    for epoch in range(1, args.epochs + 1):
        _schedules(agent, kind, epoch, args)
        random.shuffle(order)
        run_epoch(agent, order, buf, data, args, observed)
    if len(buf) > 0:                       # MIRROR train.py:2146 end-of-epoch flush
        agent.ppo_update(buf)
        buf.clear()
    for mod in (agent.unmerge_actor, agent.factor_actor, agent.critic):
        mod.eval()
    n_cells = sum(len(v) for v in observed.values())
    return agent, n_cells


def run(data: dict, args) -> None:
    loops = labelled_loops(data)
    by_bench: dict = {}
    for l in loops:
        by_bench.setdefault(l["benchmark_name"], []).append(l)
    elig = {b: ls for b, ls in by_bench.items() if len(ls) >= MIN_LOOPS}

    # Benchmarks with no headroom on ANY loop are a different question and get a
    # separate run: there is nothing to recover, so capture is undefined and the
    # only thing to measure is damage.
    def any_headroom(ls):
        return any(has_headroom(l, data["tables"], args.deadzone) for l in ls)
    with_hd = {b: ls for b, ls in elig.items() if any_headroom(ls)}
    without = {b: ls for b, ls in elig.items() if not any_headroom(ls)}
    chosen = without if args.no_headroom else with_hd

    log.info("%d benchmarks with >=%d loops: %d with headroom, %d without.",
             len(elig), MIN_LOOPS, len(with_hd), len(without))
    log.info("Running the %s group: %d benchmarks, %d loops.\n",
             "NO-HEADROOM" if args.no_headroom else "headroom",
             len(chosen), sum(len(ls) for ls in chosen.values()))
    if not chosen:
        log.error("no benchmarks in this group")
        return

    rows = []
    for b in sorted(chosen):
        ls = chosen[b]
        for sp in range(args.splits):
            tr, ev = split(ls, args.fold_seed + sp)
            if not ev:
                continue
            n_ev_hd = sum(1 for l in ev
                          if has_headroom(l, data["tables"], args.deadzone))
            for si in range(args.seeds):
                seed = args.base_seed + si
                agent, n_cells = train_on(args.agent, tr, data, args, seed)
                m = score_decisions(
                    greedy_picks(agent, ev, data["normalizer"], data["postf"]),
                    data["tables"], data["labels"], args.deadzone, _mr(args))
                orc = score_decisions(
                    [(l["benchmark_name"], l["loop_idx"],
                      oracle_of_gated(data["tables"][(l["benchmark_name"],
                                                      l["loop_idx"])],
                                      args.deadzone)[0]) for l in ev],
                    data["tables"], data["labels"], args.deadzone, _mr(args))
                rows.append({
                    "benchmark": b, "split": sp, "seed": seed,
                    "n_loops": len(ls), "n_train": len(tr), "n_eval": len(ev),
                    "n_eval_headroom": n_ev_hd, "cells_sampled": n_cells,
                    "accuracy": round(m["accuracy"], 6),
                    "mean_realized": round(m["mean_realized"], 6),
                    "capture": round(m["capture"], 6),
                    "oracle_mean_realized": round(orc["mean_realized"], 6),
                    "loops_slower": m["n_regress"],
                })
    report(rows, args)
    if args.csv_out:
        write_csv(Path(args.csv_out), rows)
        log.info("\n  per-benchmark rows: %s", args.csv_out)


def _mean(xs: list) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def report(rows: list, args) -> None:
    if not rows:
        log.error("no rows produced")
        return
    by_b: dict = {}
    for r in rows:
        by_b.setdefault(r["benchmark"], []).append(r)

    log.info("=" * 84)
    log.info("  PER-BENCHMARK RL — trained on a few of the application's loops, "
             "scored on the rest")
    log.info("=" * 84)
    log.info("  %-24s %5s %5s %4s | %6s | %8s %8s %7s %6s",
             "benchmark", "loops", "eval", "hd", "acc", "agent", "oracle",
             "capture", "slower")
    log.info("  " + "-" * 82)
    for b in sorted(by_b):
        rs = by_b[b]
        log.info("  %-24s %5d %5d %4.0f | %5.1f%% | %+8.4f %+8.4f %6.1f%% %6.1f",
                 b[:24], rs[0]["n_loops"], rs[0]["n_eval"],
                 _mean([r["n_eval_headroom"] for r in rs]),
                 100 * _mean([r["accuracy"] for r in rs]),
                 _mean([r["mean_realized"] for r in rs]),
                 _mean([r["oracle_mean_realized"] for r in rs]),
                 100 * _mean([r["capture"] for r in rs]),
                 _mean([r["loops_slower"] for r in rs]))
    log.info("  " + "-" * 82)
    log.info("  %-24s %5s %5s %4s | %5.1f%% | %+8.4f %+8.4f %6.1f%% %6.1f",
             f"MEAN over {len(by_b)} bm", "", "", "",
             100 * _mean([r["accuracy"] for r in rows]),
             _mean([r["mean_realized"] for r in rows]),
             _mean([r["oracle_mean_realized"] for r in rows]),
             100 * _mean([r["capture"] for r in rows]),
             _mean([r["loops_slower"] for r in rows]))
    log.info("\n  always-no-op is exactly +0.0000 on every row by construction, so "
             "the agent column\n  IS its margin over doing nothing. Every fixed "
             "action on this population is\n  negative, so any gain here comes "
             "from per-loop decisions.")
    log.info("  Mean cells sampled per benchmark: %.0f  (%d epochs x %d train "
             "loops, with replacement)",
             _mean([r["cells_sampled"] for r in rows]), args.epochs,
             round(_mean([r["n_train"] for r in rows])))
    if args.no_headroom:
        log.info("\n  NO-HEADROOM GROUP: capture is undefined — there was nothing "
                 "to recover.\n  Read `agent` and `slower`: the correct policy "
                 "declines everywhere and scores\n  exactly +0.0000 with 0 loops "
                 "made slower.")


def main() -> None:
    p = build_parser()
    g = p.add_argument_group("per-benchmark")
    g.add_argument("--splits", type=int, default=3,
                   help="Split draws per benchmark. WHICH loops land in training "
                        "moves the result more than the init seed does, so this "
                        "matters more than --seeds.")
    g.add_argument("--no-headroom", action="store_true",
                   help="Run the benchmarks with NO headroom on any loop instead. "
                        "Nothing to recover; the only correct behaviour is to "
                        "decline, so this measures damage.")
    args = p.parse_args()
    torch.set_num_threads(max(1, args.threads))
    data = load_run(args.run_dir, args.deadzone, args.labels)
    log.info("Loaded %d loops / %d benchmarks | %d labelled\n",
             len(data["loops"]), len(data["benchmarks"]),
             data["n_labelled_loops"])
    run(data, args)


if __name__ == "__main__":
    main()
