"""Load a ProGraML ProgramGraph (JSON) and index it for per-loop slicing.

Input is the JSON emitted by `llvm2graph --stdout_fmt=json`. That is proto3 JSON,
which OMITS default-valued fields — the single most important thing to get right
here. We default them explicitly so no downstream code has to:

  node.type    absent => "INSTRUCTION"  (enum 0)
  edge.flow    absent => "CONTROL"       (enum 0)
  edge.source  absent => 0               (node 0 is the root)
  edge.target  absent => 0
  edge.position absent=> 0
  node.function absent=> 0

Edge directions (verified against the builder) are source -> target:
  CONTROL: instr      -> next instr
  DATA:    producer   -> variable,   variable/constant/argument -> consumer
  TYPE:    type-node  -> typed node (var/const/composite)
  CALL:    callsite   -> callee entry,  callee exit -> callsite (return),
           root       -> function entry

Scheme C needs only PLAIN fields (node.text = opcode/type-token, node.type,
loopcount_loops int list, edges) — no base64. `full_text` (a string stored as a
bytes feature, base64 in JSON) is left raw here and decoded in vectorize.py only
for scheme A.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Union

# Node kinds (Node.Type enum names as they appear in proto3 JSON).
INSTRUCTION = "INSTRUCTION"
VARIABLE = "VARIABLE"
CONSTANT = "CONSTANT"
TYPE = "TYPE"

# Edge flows (Edge.Flow enum names).
CONTROL = "CONTROL"
DATA = "DATA"
CALL = "CALL"
TYPE_FLOW = "TYPE"

# The module root node: an INSTRUCTION with this sentinel text, created by the
# ProGraML builder with call edges to every function entry (and default
# function==0). It is a module-level connector, not part of any function, so
# slicing/featurization must special-case it.
EXTERNAL_NODE_TEXT = "[external]"


@dataclass
class Node:
    index: int
    type: str                    # INSTRUCTION | VARIABLE | CONSTANT | TYPE
    text: str                    # opcode (instr), "var"/"val", or type token
    function: int                # index into Graph.functions
    loops: FrozenSet[int]        # loopcount_loops set (empty if in no loop)
    features: dict = field(default_factory=dict, repr=False)  # raw feature map


@dataclass
class Edge:
    flow: str
    position: int
    source: int
    target: int


def _int_list_feature(features: dict, key: str) -> List[int]:
    """Extract an int64-list feature; [] if absent."""
    f = features.get(key)
    if not f:
        return []
    return [int(v) for v in f.get("int64_list", {}).get("value", [])]


class Graph:
    """A ProgramGraph plus source/target adjacency indices."""

    def __init__(self, nodes: List[Node], edges: List[Edge], functions: List[str]):
        self.nodes = nodes
        self.edges = edges
        self.functions = functions
        # Adjacency: node index -> list of edges leaving / entering it.
        self._out: Dict[int, List[Edge]] = {}
        self._in: Dict[int, List[Edge]] = {}
        for e in edges:
            self._out.setdefault(e.source, []).append(e)
            self._in.setdefault(e.target, []).append(e)

    # -- adjacency -----------------------------------------------------------
    def out_edges(self, i: int, flow: Optional[str] = None) -> List[Edge]:
        es = self._out.get(i, [])
        return es if flow is None else [e for e in es if e.flow == flow]

    def in_edges(self, i: int, flow: Optional[str] = None) -> List[Edge]:
        es = self._in.get(i, [])
        return es if flow is None else [e for e in es if e.flow == flow]

    def out_neighbors(self, i: int, flow: Optional[str] = None) -> List[int]:
        return [e.target for e in self.out_edges(i, flow)]

    def in_neighbors(self, i: int, flow: Optional[str] = None) -> List[int]:
        return [e.source for e in self.in_edges(i, flow)]

    # -- convenience ---------------------------------------------------------
    def producers_of(self, var_index: int) -> List[int]:
        """Instruction(s) that PRODUCE this variable: DATA edges INTO it whose
        source is an instruction (producer -> variable)."""
        return [e.source for e in self.in_edges(var_index, DATA)
                if self.nodes[e.source].type == INSTRUCTION]

    def type_node_of(self, i: int) -> Optional[int]:
        """The TYPE node of a var/const node: TYPE edge INTO it (type -> node)."""
        ins = self.in_edges(i, TYPE_FLOW)
        return ins[0].source if ins else None

    def instruction_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.type == INSTRUCTION]

    def loop_ids(self) -> FrozenSet[int]:
        """Every distinct loopIdx stamped anywhere in the module."""
        out: set = set()
        for n in self.nodes:
            out |= n.loops
        return frozenset(out)


def load_graph(source: Union[str, bytes, dict]) -> Graph:
    """Load from a dict, a JSON string, or a path to a .json file."""
    if isinstance(source, dict):
        obj = source
    else:
        text: str
        if isinstance(source, bytes):
            text = source.decode("utf-8")
        elif isinstance(source, str) and (source.lstrip().startswith("{") or "\n" in source):
            text = source
        else:  # treat as a filesystem path
            with open(source, "r") as fh:
                text = fh.read()
        obj = json.loads(text)

    nodes: List[Node] = []
    for i, n in enumerate(obj.get("node", [])):
        feats = n.get("features", {}).get("feature", {})
        nodes.append(Node(
            index=i,
            type=n.get("type", INSTRUCTION),          # enum 0 omitted
            text=n.get("text", ""),
            function=int(n.get("function", 0)),        # int 0 omitted
            loops=frozenset(_int_list_feature(feats, "loopcount_loops")),
            features=feats,
        ))

    edges: List[Edge] = []
    for e in obj.get("edge", []):
        edges.append(Edge(
            flow=e.get("flow", CONTROL),               # enum 0 omitted
            position=int(e.get("position", 0)),
            source=int(e.get("source", 0)),            # int 0 omitted
            target=int(e.get("target", 0)),
        ))

    functions: List[str] = [f.get("name", "") for f in obj.get("function", [])]
    return Graph(nodes, edges, functions)


# ---------------------------------------------------------------------------
# Self-test: synthetic graph exercising defaulting, adjacency, and helpers.
# Run: python -m rl.pgraph.loader
# ---------------------------------------------------------------------------
def _selftest() -> None:
    # 4 nodes: 0 root(instr), 1 instr (in loop 0), 2 variable, 3 type.
    # Edges (note omitted defaults): control 0->1 (flow+source omitted),
    # data producer 1->2, type 3->2.
    g_json = {
        "node": [
            {},                                        # 0: instruction (root), no type
            {"text": "add", "features": {"feature": {
                "loopcount_loops": {"int64_list": {"value": ["0", "1"]}}}}},  # 1
            {"type": "VARIABLE", "text": "var"},       # 2
            {"type": "TYPE", "text": "i32"},           # 3
        ],
        "edge": [
            {"target": 1},                             # CONTROL 0->1 (defaults)
            {"flow": "DATA", "source": 1, "target": 2},   # producer 1 -> var 2
            {"flow": "TYPE", "source": 3, "target": 2},   # type 3 -> var 2
        ],
        "function": [{"name": "kern"}],
    }
    g = load_graph(g_json)
    assert len(g.nodes) == 4 and len(g.edges) == 3, "counts"
    assert g.nodes[0].type == INSTRUCTION, "default node.type"
    assert g.edges[0].flow == CONTROL and g.edges[0].source == 0, "default flow/source"
    assert g.nodes[1].loops == frozenset({0, 1}), "loop set parsed"
    assert g.out_neighbors(0, CONTROL) == [1], "control adjacency"
    assert g.producers_of(2) == [1], "producer of variable"
    assert g.type_node_of(2) == 3, "type node of variable"
    assert g.loop_ids() == frozenset({0, 1}), "module loop ids"
    assert [n.index for n in g.instruction_nodes()] == [0, 1], "instruction nodes"
    print("pgraph.loader self-test: PASS")


if __name__ == "__main__":
    _selftest()
