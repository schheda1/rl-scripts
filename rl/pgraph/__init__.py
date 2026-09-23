"""ProGraML per-loop feature engineering.

Turns a module-level ProgramGraph emitted by `llvm2graph` (from the early,
loopcount-stamped IR) into model-ready per-loop inputs for the IR2Vec-vs-ProGraML
bake-off:

  loader.py    - load a ProgramGraph (JSON) + build an adjacency index   [3a]
  slicer.py    - loop-level slice + per-kernel graph + loop->parents map  [3b]
  vectorize.py - node featurization: scheme C (categorical) + A (inst2vec) [3c]
  emit.py      - PyG Data emitter keyed to (benchmark, arch, loopIdx)      [3d]
  collide.py   - WL-hash collision check vs IR2Vec floor                   [3b test]

Design: the loop graph carries per-loop identity (like IR2Vec `emb`); the kernel
graph is shared per-kernel (like `kemb`); `loop+kernel` is the model composition
`GNN(loop) (+) pool(GNN(kernel))`. See the followup plan / uu-final-paper-scope.
"""
