"""Scheme C node featurization (categorical) for ProGraML loop/kernel subgraphs.

Each node becomes a small integer/scalar feature vector; the GNN learns the
embeddings (so the "feature engineering" here is the frozen SCHEME, not learned
weights):

    [kind_id, token_id, is_boundary, in_target_loop, loop_depth]

  kind_id   : INSTRUCTION / VARIABLE / CONSTANT / TYPE  (fixed 4-way)
  token_id  : a frozen vocab over node tokens (see node_token), OOV -> <unk>
  is_boundary : 1 for one-level boundary (external live-in producer) leaves
  in_target_loop : 1 if the slice's target loop is in this node's loopcount_loops
  loop_depth : |loopcount_loops| (0 outside loops) — the nesting indicator

Node token policy (faithful ProGraML: types stay their own nodes, which is where
width de-aliasing lives, so only instructions are callee-aware):
  INSTRUCTION -> opcode; for a call, "call:<callee>" (callee from the CALL edge on
                 the FULL graph — slices cut it, so tokens are computed pre-slice)
  VARIABLE/CONSTANT -> its generic text ("var"/"val"); its type rides the TYPE node
  TYPE -> the type token ("i32","float","double","*","struct","[]","vector")

DEFERRED enrichments — the feature scheme is FROZEN as above. The following IR
details are intentionally dropped for now and are to be revisited ONLY IFF
ProGraML falls short in training/evaluation (each lives in full_text and/or the
graph structure, so the GNN may already learn them; enriching is cheap if not):
  - instruction flags: nsw/nuw, inbounds, fast-math, atomic ordering, volatile,
    tail (opcode token is bare, e.g. "add" not "add nsw")
  - compare predicates: "icmp"/"fcmp" (slt/eq/ugt condition dropped)
  - pointer address space: on the early stamped IR every pointer is generic
    ("ptr", addrspace 0) — global(1) is only inferred by a LATER pass, and
    shared(3)/const(4) live on addrspacecast/global nodes, not the pointer type;
    recoverable by a def-use trace or by the GNN from those nodes, if needed
  - vector lane count ("<4 x float>"->"vector") and array length ("[N x T]"->"[]")
  - constant values (stripped by design — same collision floor as IR2Vec)
(Same coverage limits apply to IR2Vec, so the bake-off stays fair.)

Edges carry [flow_id, position]. The vocab is FROZEN once over all benchmarks/archs
(build_vocab + save/load) so feature dims are stable; OOV tokens map to <unk>.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List

from .loader import (CALL, CONSTANT, EXTERNAL_NODE_TEXT, Graph, INSTRUCTION,
                     TYPE, VARIABLE)
from .slicer import Subgraph

# Fixed enumerations (never learned, never OOV).
KINDS = (INSTRUCTION, VARIABLE, CONSTANT, TYPE)
KIND2ID = {k: i for i, k in enumerate(KINDS)}
FLOWS = ("CONTROL", "DATA", "CALL", "TYPE")
FLOW2ID = {f: i for i, f in enumerate(FLOWS)}

# node_features column names per scheme, so a record self-describes its columns
# (kind_id/token_id are categorical → embedded; the rest are scalars).
SCHEME_C_COLUMNS = ["kind_id", "token_id", "is_boundary", "in_target_loop", "loop_depth"]
SCHEME_A_COLUMNS = ["inst2vec_index", "is_boundary", "in_target_loop", "loop_depth"]


def node_token(g: Graph, i: int) -> str:
    """The scheme-C token for node i, computed on the FULL graph (so a call's
    callee — reachable only via the CALL edge that slicing later cuts — is
    captured now)."""
    node = g.nodes[i]
    if node.type != INSTRUCTION:
        return node.text                      # "var"/"val" or a type token
    # Callee-aware ONLY for real call/invoke opcodes. The module root
    # ("[external]") also has outgoing CALL edges (root -> every function entry),
    # so keying off the CALL edge alone would mislabel it as a call.
    if node.text in ("call", "invoke"):
        call_edges = g.out_edges(i, CALL)
        if call_edges:                        # direct call: fold in the callee
            callee_entry = call_edges[0].target
            fn = g.nodes[callee_entry].function
            name = g.functions[fn] if 0 <= fn < len(g.functions) else "?"
            return "call:" + name
    return node.text                          # plain opcode / "[external]"


def compute_tokens(g: Graph) -> Dict[int, str]:
    """Token for every node, keyed by original index (computed once per module)."""
    return {i: node_token(g, i) for i in range(len(g.nodes))}


def scheme_c_wl_label(tokens: Dict[int, str]):
    """Build a WL label_fn (for collide.wl_hash / collision_report) that uses the
    scheme-C token instead of node.text, so the collision floor reflects the ACTUAL
    scheme-C features (esp. callee-aware calls, which the default label collapses to
    "call"). `tokens` = compute_tokens(full_graph); a Subgraph's nodes keep their
    ORIGINAL index, which keys `tokens`. Same shape as collide.default_label
    (kind|token|boundary) — only the token differs."""
    def label(node, is_boundary: bool) -> str:
        tok = tokens.get(node.index, node.text)
        return "%s|%s|%d" % (node.type, tok, 1 if is_boundary else 0)
    return label


class Vocab:
    """Frozen token vocabulary. id 0 is reserved for <unk> (OOV)."""
    UNK = "<unk>"

    def __init__(self):
        self.token2id: Dict[str, int] = {self.UNK: 0}

    def add(self, token: str) -> None:
        if token not in self.token2id:
            self.token2id[token] = len(self.token2id)

    def fit(self, tokens: Iterable[str]) -> "Vocab":
        for t in tokens:
            self.add(t)
        return self

    def token_id(self, token: str) -> int:
        return self.token2id.get(token, 0)    # OOV -> <unk>

    @property
    def size(self) -> int:
        return len(self.token2id)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"token2id": self.token2id}, fh, indent=0, sort_keys=False)

    @classmethod
    def from_json(cls, path: str) -> "Vocab":
        with open(path) as fh:
            d = json.load(fh)
        v = cls()
        v.token2id = {k: int(i) for k, i in d["token2id"].items()}
        return v


def build_vocab(graphs: Iterable[Graph]) -> Vocab:
    """Fit a frozen token vocab over every node of every graph. Call once over the
    full corpus (iterate benchmarks in a fixed order for reproducible ids), save it,
    and reuse it for all vectorization so feature dims stay stable."""
    v = Vocab()
    for g in graphs:
        for i in range(len(g.nodes)):
            v.add(node_token(g, i))
    return v


def _edge_arrays(sg: Subgraph):
    """(edge_index [2 x E], edge_attr [E x [flow_id, position]]) — scheme-agnostic."""
    src: List[int] = []
    dst: List[int] = []
    edge_attr: List[List[int]] = []
    for (s, t, flow, pos) in sg.edges:
        src.append(s)
        dst.append(t)
        edge_attr.append([FLOW2ID.get(flow, 0), pos])
    return [src, dst], edge_attr


def _wrap(sg: Subgraph, node_features: List[List[int]],
          feature_columns: List[str]) -> dict:
    edge_index, edge_attr = _edge_arrays(sg)
    return {
        "num_nodes": sg.num_nodes,
        "node_features": node_features,
        "feature_columns": feature_columns,  # names the node_features columns
        "edge_index": edge_index,          # 2 x E
        "edge_attr": edge_attr,            # E x [flow, position]
        "target_loop": sg.target_loop,
        "function": sg.function,
    }


def vectorize_c(sg: Subgraph, vocab: Vocab, tokens: Dict[int, str]) -> dict:
    """Scheme-C features for one subgraph. `tokens` is compute_tokens(full_graph);
    indexed by original node ids so call tokens survive slicing.
    node_features row = [kind_id, token_id, is_boundary, in_target_loop, loop_depth]."""
    tgt = sg.target_loop
    node_features: List[List[int]] = []
    for k in range(sg.num_nodes):
        node = sg.nodes[k]
        node_features.append([
            KIND2ID.get(node.type, 0),
            vocab.token_id(tokens[sg.orig_index[k]]),
            1 if sg.boundary[k] else 0,
            1 if (tgt is not None and tgt in node.loops) else 0,
            len(node.loops),
        ])
    return _wrap(sg, node_features, SCHEME_C_COLUMNS)


def vectorize_a(sg: Subgraph, a_map: Dict[int, int]) -> dict:
    """Scheme-A features for one subgraph. `a_map` is
    scheme_a.encode_scheme_a(full_graph)["embedding_index"] (node index ->
    inst2vec embedding index). node_features row =
    [emb_index, is_boundary, in_target_loop, loop_depth] — same structural scalars
    as scheme C, but the inst2vec index in place of kind/token."""
    tgt = sg.target_loop
    node_features: List[List[int]] = []
    for k in range(sg.num_nodes):
        node = sg.nodes[k]
        node_features.append([
            a_map.get(node.index, 0),
            1 if sg.boundary[k] else 0,
            1 if (tgt is not None and tgt in node.loops) else 0,
            len(node.loops),
        ])
    return _wrap(sg, node_features, SCHEME_A_COLUMNS)


# ---------------------------------------------------------------------------
# Self-test. Run: python -m rl.pgraph.vectorize
# ---------------------------------------------------------------------------
def _selftest() -> None:
    from .loader import load_graph
    from .slicer import loop_slice
    # fn0 "kern" calls fn1 "llvm.nvvm.barrier" (a declaration). Loop 0 has:
    #   0 instr "add" (loop 0)
    #   1 instr "call" (loop 0) -> CALL edge to callee entry node 5 (fn1)
    #   2 var, 3 const, 4 type i32
    #   5 instr entry of fn1 (the callee; "; undefined function" placeholder)
    g = load_graph({
        "node": [
            {"text": "add", "function": 0, "features": {"feature": {
                "loopcount_loops": {"int64_list": {"value": ["0"]}}}}},           # 0
            {"text": "call", "function": 0, "features": {"feature": {
                "loopcount_loops": {"int64_list": {"value": ["0"]}}}}},           # 1
            {"type": "VARIABLE", "text": "var", "function": 0},                  # 2
            {"type": "CONSTANT", "text": "val", "function": 0},                  # 3
            {"type": "TYPE", "text": "i32"},                                     # 4
            {"text": "; undefined function", "function": 1},                     # 5 callee
        ],
        "edge": [
            {"flow": "DATA", "source": 2, "target": 0},    # var -> add
            {"flow": "DATA", "source": 3, "target": 1},    # const -> call
            {"flow": "TYPE", "source": 4, "target": 2},    # i32 -> var
            {"flow": "CALL", "source": 1, "target": 5},    # call -> callee entry
            {"source": 0, "target": 1},                    # CONTROL add -> call
        ],
        "function": [{"name": "kern"}, {"name": "llvm.nvvm.barrier"}],
    })

    toks = compute_tokens(g)
    assert toks[0] == "add", toks[0]
    assert toks[1] == "call:llvm.nvvm.barrier", toks[1]   # callee-aware
    assert toks[2] == "var" and toks[3] == "val" and toks[4] == "i32", toks

    vocab = build_vocab([g])
    assert vocab.token_id("call:llvm.nvvm.barrier") > 0, "callee token in vocab"
    assert vocab.token_id("never-seen") == 0, "OOV -> unk"

    sl = loop_slice(g, 0)   # seed {0,1}; +var2,const3, type4; callee 5 NOT pulled
    vec = vectorize_c(sl, vocab, toks)
    assert vec["num_nodes"] == sl.num_nodes
    assert len(vec["node_features"]) == sl.num_nodes
    assert len(vec["edge_index"][0]) == len(vec["edge_attr"]) == sl.num_edges
    # the call node's features carry the callee token id and in_target_loop=1.
    call_local = sl.orig_index.index(1)
    kind_id, token_id, boundary, in_target, depth = vec["node_features"][call_local]
    assert kind_id == KIND2ID[INSTRUCTION]
    assert token_id == vocab.token_id("call:llvm.nvvm.barrier"), token_id
    assert in_target == 1 and depth == 1, (in_target, depth)
    # every edge_attr flow id is valid.
    assert all(0 <= fa[0] < len(FLOWS) for fa in vec["edge_attr"])

    # Root node: "[external]" has an outgoing CALL edge (root -> entry) but must
    # NOT be labeled a call — its token is its sentinel text.
    gr = load_graph({
        "node": [{"text": EXTERNAL_NODE_TEXT}, {"text": "add", "function": 0}],
        "edge": [{"flow": "CALL", "source": 0, "target": 1}],
        "function": [{"name": "k"}],
    })
    assert node_token(gr, 0) == EXTERNAL_NODE_TEXT, node_token(gr, 0)

    # collide bridge: two isomorphic single-call loops calling DIFFERENT functions.
    # Default label collapses both to "call" (collision); the scheme-C label uses
    # the callee token and separates them.
    from .collide import wl_hash
    gc = load_graph({
        "node": [
            {"text": "call", "function": 0, "features": {"feature": {
                "loopcount_loops": {"int64_list": {"value": ["0"]}}}}},   # 0 call->A (loop 0)
            {"text": "call", "function": 0, "features": {"feature": {
                "loopcount_loops": {"int64_list": {"value": ["1"]}}}}},   # 1 call->B (loop 1)
            {"text": "; undefined function", "function": 1},              # 2 A entry
            {"text": "; undefined function", "function": 2},              # 3 B entry
        ],
        "edge": [{"flow": "CALL", "source": 0, "target": 2},
                 {"flow": "CALL", "source": 1, "target": 3}],
        "function": [{"name": "k"}, {"name": "A"}, {"name": "B"}],
    })
    s0, s1 = loop_slice(gc, 0), loop_slice(gc, 1)
    assert wl_hash(s0) == wl_hash(s1), "default label should collide the two calls"
    lab = scheme_c_wl_label(compute_tokens(gc))
    assert wl_hash(s0, label_fn=lab) != wl_hash(s1, label_fn=lab), \
        "scheme-C label must distinguish callees"
    print("pgraph.vectorize (scheme C) self-test: PASS")


if __name__ == "__main__":
    _selftest()
