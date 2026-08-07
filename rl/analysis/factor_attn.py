"""
Factor head with NO dense hidden layers: input projection -> attention -> output
projection.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
Not the discarded scorer. The factor stays an OUTPUT INDEX and no factor
features are added. The hypothesis is about how the 93 LOOP features are
combined: a dense stack mixes them additively and only interacts them through
ReLU, whereas attention mixes them multiplicatively through a learned
query-key product.

`_MLP.net` is
    0 Linear(93,128)  1 LayerNorm  2 ReLU
    3 Linear(128,64)  4 LayerNorm  5 ReLU
    6 Linear(64,out)
i.e. an input projection, a DENSE hidden layer, and an output projection. Here
the dense hidden layer is gone entirely and attention takes its place:

    0 Linear(93,128)   input projection
    1 LayerNorm(128)
    2 Reshape          -> (B, T, d),  T*d = 128, no parameters
    3 AttnStack        1 or 2 self-attention blocks at constant width d
    4 LayerNorm(d)
    5 Flatten          -> (B, 128),   no parameters
    6 Linear(128,out)  output projection

Still seven elements, and the positional semantics survive, which is what keeps
the rest of the system working:

  * `forward` is INHERITED verbatim — no override, so logit_cap cannot be
    silently dropped (a bug the scorer had to guard against by hand);
  * `offline_adapt._freeze_for_adaptation` resolves {1:[6], 2:[3,4,6],
    3:[0,1,3,4,6]}: group 1 is still the output projection, group 2 is now the
    attention plus its norm rather than a dense layer plus its norm, group 3 is
    still everything. Its type assertion reads `unmerge_actor` only
    (offline_adapt.py:139), so this head does not trip it;
  * `net[6].weight`, state_dict plumbing, `sample` and `log_prob` are untouched.

TWO THINGS TO STATE IF THIS EVER GETS WRITTEN UP
------------------------------------------------
1. The tokens are ARBITRARY. Reshaping a 128-d hidden vector into T chunks
   imposes a grouping with no semantic content — chunk 3 is "hidden units
   32..47". Attention over arbitrary chunks is closer to a gated MLP than to
   attention in the useful sense. The grounded alternative is to tokenise the
   INPUT (each of the 93 features gets a learned embedding, attention over 93
   real feature tokens, FT-Transformer style); that is a bigger and different
   experiment.
2. Capacity DROPS, it is not matched. Removing Linear(128,64) takes out 8,256
   parameters and one attention block adds ~1,100. `param_delta()` reports the
   exact figure and it is logged at startup. There is deliberately no FFN inside
   the blocks — an FFN is two dense layers, which is the thing being removed —
   so the only nonlinearity left in the hidden path is the attention softmax.
   A negative result here is therefore confounded with both, and only a positive
   one is clean.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402

from agent import FactorActor                                    # noqa: E402

_HIDDEN = 128            # width of the input projection's output


class _Reshape(nn.Module):
    """(B, T*d) -> (B, T, d). No parameters."""

    def __init__(self, tokens: int, d: int) -> None:
        super().__init__()
        self.tokens, self.d = tokens, d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # .view is safe: x comes from LayerNorm, which returns contiguous.
        return x.view(x.size(0), self.tokens, self.d)

    def extra_repr(self) -> str:
        return f"tokens={self.tokens}, d={self.d}"


class _Flatten(nn.Module):
    """(B, T, d) -> (B, T*d). No parameters."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # .reshape, not .view: attention output is not guaranteed contiguous.
        return x.reshape(x.size(0), -1)


class AttnStack(nn.Module):
    """`blocks` self-attention blocks at constant width d, applied in sequence."""

    def __init__(self, tokens: int, d: int, heads: int, blocks: int) -> None:
        super().__init__()
        if d % heads:
            raise ValueError(
                f"--attn-heads {heads} must divide the token width {d}")
        # Learned positional embedding: the chunks are NOT interchangeable —
        # chunk i is a fixed slice of hidden units — and attention without it is
        # permutation-equivariant over slices that carry no such symmetry.
        self.pos = nn.Parameter(torch.zeros(tokens, d))
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, batch_first=True)
             for _ in range(blocks)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PRE-norm, deliberately: `x = x + attn(norm(x))`.

        Post-norm (`x = norm(x + a)`) would put a LayerNorm at the end of this
        stack immediately before net[4]'s LayerNorm — two normalisations in a
        row, the second of which largely undoes the first's learned scale.
        Pre-norm leaves net[4] as the single, canonical final norm and is the
        more stable arrangement when `blocks` is 2.
        """
        x = x + self.pos
        for attn, norm in zip(self.attn, self.norm):
            h = norm(x)
            a, _ = attn(h, h, h, need_weights=False)
            x = x + a
        return x


class FactorAttnActor(FactorActor):
    """FactorActor with the dense hidden layer replaced by attention."""

    def __init__(self, tokens: int = 8, heads: int = 4, blocks: int = 1,
                 logit_cap: float = 0.0) -> None:
        super().__init__(logit_cap=logit_cap)
        if _HIDDEN % tokens:
            raise ValueError(f"--attn-tokens {tokens} must divide {_HIDDEN}")
        d = _HIDDEN // tokens
        # Assert the layout being replaced rather than trusting the indices. If
        # _MLP's stack is ever reordered this must fail loudly instead of
        # attending over the wrong tensor and reporting plausible numbers.
        want = {0: (nn.Linear, None), 1: (nn.LayerNorm, None),
                3: (nn.Linear, (_HIDDEN, 64)), 6: (nn.Linear, (64, None))}
        for i, (cls, dims) in want.items():
            got = self.net[i]
            if not isinstance(got, cls):
                raise AssertionError(
                    f"_MLP layout changed: net[{i}] is {type(got).__name__}, "
                    f"expected {cls.__name__}")
            if dims and dims[0] is not None and got.in_features != dims[0]:
                raise AssertionError(
                    f"_MLP layout changed: net[{i}].in_features is "
                    f"{got.in_features}, expected {dims[0]}")
        out_dim = self.net[6].out_features

        self.net[2] = _Reshape(tokens, d)
        self.net[3] = AttnStack(tokens, d, heads, blocks)
        self.net[4] = nn.LayerNorm(d)
        self.net[5] = _Flatten()
        # The output projection now reads the flattened token stack (T*d = 128),
        # not the 64-d dense hidden it used to.
        self.net[6] = nn.Linear(_HIDDEN, out_dim)


def param_delta(tokens: int = 8, heads: int = 4, blocks: int = 1,
                logit_cap: float = 0.0) -> tuple:
    """
    (baseline_params, attn_params, pct_change) for the factor head.

    Reported, not asserted. The scorer was built to hold capacity within 3%;
    this cannot, because removing Linear(128,64) is most of the point. An
    unstated capacity change is a confound — a stated one is a caveat.
    """
    base = sum(p.numel() for p in FactorActor(logit_cap=logit_cap).parameters())
    att = sum(p.numel() for p in
              FactorAttnActor(tokens, heads, blocks, logit_cap).parameters())
    return base, att, 100.0 * (att - base) / base
