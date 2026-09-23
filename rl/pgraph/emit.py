"""Stage 3d: model-ready per-loop feature records (PyG-convertible).

For a given loopIdx this produces the two granularities from the design:
  - the LOOP graph (per-loop, ~ IR2Vec emb/femb)
  - the KERNEL-context graph(s) (whole kernel, shared per kernel, ~ kemb); a
    device-fn loop yields one per calling kernel (pooled at the model)
each vectorized under scheme C or scheme A into a uniform, torch-free RECORD:

  {num_nodes, node_features, edge_index, edge_attr, target_loop, function,
   benchmark, arch, loop_idx, granularity, scheme}

The record is the frozen, serializable feature artifact (keyed to
(benchmark, arch, loop_idx) so it joins the reward cache). `to_pyg` converts one
record to a torch_geometric.data.Data at training time (torch imported lazily, so
feature extraction needs no torch). The composition
GNN(loop) (+) pool(GNN(kernel)) is left to the model.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .loader import Graph
from .slicer import (Subgraph, kernel_context_indices, kernel_graph, loop_slice)
from .vectorize import Vocab, vectorize_a, vectorize_c


def _vectorize(sg: Subgraph, scheme: str, *, vocab: Optional[Vocab] = None,
               tokens: Optional[Dict[int, str]] = None,
               a_map: Optional[Dict[int, int]] = None) -> dict:
    if scheme == "C":
        if vocab is None or tokens is None:
            raise ValueError("scheme C needs vocab and tokens")
        return vectorize_c(sg, vocab, tokens)
    if scheme == "A":
        if a_map is None:
            raise ValueError("scheme A needs a_map")
        return vectorize_a(sg, a_map)
    raise ValueError("unknown scheme %r (use 'C' or 'A')" % scheme)


def records_for_loop(g: Graph, loop_id: int, *, scheme: str, benchmark: str,
                     arch: str, is_kernel: bool, parent_names: List[str],
                     vocab: Optional[Vocab] = None,
                     tokens: Optional[Dict[int, str]] = None,
                     a_map: Optional[Dict[int, int]] = None) -> dict:
    """The loop record + kernel-context record(s) for one loopIdx.

    is_kernel / parent_names come from the LoopCount CSV row for this loop.
    Returns {"key": (benchmark, arch, loop_id), "loop": <record>,
             "kernels": [<record>, ...]}."""
    def make(sg: Subgraph, granularity: str) -> dict:
        rec = _vectorize(sg, scheme, vocab=vocab, tokens=tokens, a_map=a_map)
        rec.update({"benchmark": benchmark, "arch": arch, "loop_idx": loop_id,
                    "granularity": granularity, "scheme": scheme})
        return rec

    loop = make(loop_slice(g, loop_id), "loop")
    kernels = [make(kernel_graph(g, fi), "kernel")
               for fi in kernel_context_indices(g, loop_id, is_kernel, parent_names)]
    return {"key": (benchmark, arch, loop_id), "loop": loop, "kernels": kernels}


def to_pyg(record: dict):
    """Convert one feature record to a torch_geometric.data.Data. torch/PyG are
    imported lazily so the extraction pipeline needs neither."""
    import torch
    from torch_geometric.data import Data

    x = torch.tensor(record["node_features"], dtype=torch.long)
    edge_index = torch.tensor(record["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(record["edge_attr"], dtype=torch.long)
    if edge_index.numel() == 0:                # keep shape [2, 0] for empty edge sets
        edge_index = edge_index.view(2, 0)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    for k in ("benchmark", "arch", "loop_idx", "granularity", "scheme"):
        setattr(data, k, record.get(k))
    return data


# ---------------------------------------------------------------------------
# Self-test (torch-free record checks; to_pyg checked only if torch is present).
# Run: python -m rl.pgraph.emit
# ---------------------------------------------------------------------------
def _selftest() -> None:
    import base64
    from .loader import load_graph
    from .vectorize import (build_vocab, compute_tokens, SCHEME_A_COLUMNS,
                            SCHEME_C_COLUMNS)
    from .scheme_a import Inst2vecDict, encode_scheme_a

    def ft(text: str) -> dict:
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return {"full_text": {"bytes_list": {"value": [b64]}}}

    # fn0 kernel "k": loop 0 = { add (in loop), a variable, its i32 type }.
    g = load_graph({
        "node": [
            {"text": "[external]"},                                         # 0 root
            {"text": "add", "function": 0, "features": {"feature": dict(
                ft("%2 = add nsw i32 %1, 1"),
                **{"loopcount_loops": {"int64_list": {"value": ["0"]}}})}},  # 1 in loop 0
            {"type": "VARIABLE", "text": "var", "function": 0},             # 2
            {"type": "TYPE", "text": "i32"},                                # 3
        ],
        "edge": [
            {"flow": "DATA", "source": 2, "target": 1},
            {"flow": "TYPE", "source": 3, "target": 2},
            {"flow": "CALL", "source": 0, "target": 1},   # root -> entry
        ],
        "function": [{"name": "k"}],
    })

    vocab = build_vocab([g])
    tokens = compute_tokens(g)
    fake = Inst2vecDict({"!UNK": 0, "!IDENTIFIER": 1, "!IMMEDIATE": 2,
                         "<%ID> = add nsw i32 <%ID>, <INT>": 3})
    a_map = encode_scheme_a(g, fake)["embedding_index"]

    # scheme C
    rc = records_for_loop(g, 0, scheme="C", benchmark="accuracy-cuda", arch="sm_80",
                          is_kernel=True, parent_names=[], vocab=vocab, tokens=tokens)
    assert rc["key"] == ("accuracy-cuda", "sm_80", 0)
    lp = rc["loop"]
    assert lp["scheme"] == "C" and lp["granularity"] == "loop"
    assert lp["num_nodes"] == len(lp["node_features"]) > 0
    assert all(len(row) == 5 for row in lp["node_features"]), "scheme C row width 5"
    assert lp["feature_columns"] == SCHEME_C_COLUMNS, lp["feature_columns"]
    assert len(lp["feature_columns"]) == len(lp["node_features"][0]), "C cols match width"
    assert len(rc["kernels"]) == 1, "one kernel-context graph (is_kernel)"
    assert rc["kernels"][0]["granularity"] == "kernel"
    # the loop graph excludes the root; the kernel graph also excludes it.
    assert lp["num_nodes"] < rc["kernels"][0]["num_nodes"] + 5  # sane relative sizes

    # scheme A
    ra = records_for_loop(g, 0, scheme="A", benchmark="accuracy-cuda", arch="sm_80",
                          is_kernel=True, parent_names=[], a_map=a_map)
    la = ra["loop"]
    assert la["scheme"] == "A"
    assert all(len(row) == 4 for row in la["node_features"]), "scheme A row width 4"
    assert la["feature_columns"] == SCHEME_A_COLUMNS, la["feature_columns"]
    assert la["num_nodes"] == lp["num_nodes"], "same slice, same node count both schemes"

    # errors
    try:
        records_for_loop(g, 0, scheme="C", benchmark="b", arch="a",
                         is_kernel=True, parent_names=[])
        raise AssertionError("should have required vocab/tokens")
    except ValueError:
        pass

    # to_pyg only if torch is available
    pyg = "skipped (no torch)"
    try:
        import torch  # noqa: F401
        d = to_pyg(lp)
        assert d.x.shape[0] == lp["num_nodes"] and d.edge_index.shape[0] == 2
        assert d.benchmark == "accuracy-cuda" and d.loop_idx == 0
        pyg = "ok"
    except ImportError:
        pass
    print("pgraph.emit self-test: PASS  (scheme C+A records; to_pyg=%s)" % pyg)


if __name__ == "__main__":
    _selftest()
