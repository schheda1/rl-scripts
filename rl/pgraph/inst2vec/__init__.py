"""Vendored inst2vec statement preprocessing (from ProGraML, Apache-2.0).

`inst2vec_preprocess.py` and `rgx_utils.py` are copied verbatim from
ProGraML/programl/third_party/inst2vec, with the single intra-package import in
inst2vec_preprocess localized to `from . import rgx_utils`. They are pure-Python
(regex only) and define scheme A's FROZEN 2018 statement normalization, which
must match inst2vec_augmented_dictionary.pickle exactly. Vendored (rather than
imported through the `programl` package) because programl/__init__.py eagerly
imports protobuf, and to keep scheme A self-contained and pinned.
"""
