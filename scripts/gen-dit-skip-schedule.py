#!/usr/bin/env python3
"""Record the DiT's U-Net skip SCHEDULE from upstream, for several depths.

`uvit_skip_connection: true` in the shipped config makes the first half of the
transformer emit its outputs onto a stack and the second half pop them, so layer
7 receives layer 5's output, layer 8 receives layer 4's, and so on. The routing
is pure index logic and every plausible variant (FIFO instead of LIFO, `>=`
instead of `>`, emitting before the layer instead of after) produces a model
that still runs.

This drives upstream's own Transformer and records which layer actually received
which layer's output, rather than restating the formula.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

SRC = Path(sys.argv[1])  # .../indextts/s2mel/modules
sys.path.insert(0, str(SRC.parents[2]))
for name in ("munch",):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.Munch = dict
        sys.modules[name] = stub

spec = importlib.util.spec_from_file_location(
    "indextts.s2mel.modules.gpt_fast.model", SRC / "gpt_fast" / "model.py"
)
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)


def schedule(depth: int):
    """Return [(receiver_layer, emitter_layer), ...] as upstream actually routes."""
    args = gm.ModelArgs(
        block_size=64, n_layer=depth, n_head=2, dim=8, head_dim=4,
        vocab_size=16, uvit_skip_connection=True, time_as_token=False,
    )
    tr = gm.Transformer(args)
    tr.setup_caches(1, 16, use_kv_cache=False)
    tr.eval()

    tag = {}       # id(tensor) -> layer index that produced it
    received = []  # (receiver, emitter)

    for i, layer in enumerate(tr.layers):
        orig = layer.forward

        def wrapped(*a, _i=i, _orig=orig, **kw):
            skip = kw.get("skip_in_x", a[8] if len(a) > 8 else None)
            if skip is not None:
                received.append((_i, tag.get(id(skip), -1)))
            out = _orig(*a, **kw)
            tag[id(out)] = _i
            return out

        layer.forward = wrapped

    x = torch.randn(1, 5, 8)
    c = torch.randn(1, 1, 8)
    input_pos = torch.arange(5)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    with torch.no_grad():
        tr(x, c, input_pos, mask)
    return received, tr.layers_emit_skip, tr.layers_receive_skip


for d in (2, 3, 4, 5, 12, 13):
    rec, emit, recv = schedule(d)
    print(f"depth={d:>3}  emit={emit}  receive={recv}")
    print(f"          pairs(receiver<-emitter)={rec}")
