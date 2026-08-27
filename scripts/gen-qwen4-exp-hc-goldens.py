#!/usr/bin/env python3
"""Dump Qwen4-Exp gated-residual goldens by EXECUTING the pinned oracle source.

    usage: gen-qwen4-exp-hc-goldens.py <out.inc> [path/to/modeling_qwen4_exp.py]

Fetch the oracle source first (it is not vendored -- the sha256 below is the pin):

    curl -sSLO https://raw.githubusercontent.com/huggingface/transformers/\
v5.16.0/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py


Oracle: huggingface/transformers v5.16.0,
  src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
  sha256 77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459

The two classes under test are lifted VERBATIM by line range out of that file and
exec'd; nothing is retyped.  The only harness adaptation is a 4-field stand-in for
`Qwen4ExpTextConfig` (the real dataclass drags in the whole transformers package,
and the installed transformers here is 5.3.0, which predates qwen4_exp entirely).
The write-back is likewise the verbatim two lines of
`Qwen4ExpTextDecoderLayer.forward`.
"""
import hashlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = sys.argv[2] if len(sys.argv) > 2 else "modeling_qwen4_exp.py"
EXPECT_SHA = "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459"

raw = open(SRC, "rb").read()
got = hashlib.sha256(raw).hexdigest()
# NOT an `assert`: `python3 -O` strips those, and this one guard is the entire
# mechanism tying the goldens to the pin.  Under -O the stripped version would
# happily dump goldens from any file handed to it.
if got != EXPECT_SHA:
    raise SystemExit(f"oracle source sha256 {got} != {EXPECT_SHA}")
lines = raw.decode().splitlines(keepends=True)

# 1-based, inclusive, as reported by grep -n on the pinned file.
RMSNORM = (158, 181)      # class Qwen4ExpTextRMSNorm
GATEDRES = (941, 969)     # class Qwen4ExpTextGatedResidual


def lift(span):
    return "".join(lines[span[0] - 1: span[1]])


class Qwen4ExpTextConfig:  # harness stand-in; only the 4 fields the classes read
    def __init__(self, hidden_size, hc_count, hc_lowrank, rms_norm_eps):
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.rms_norm_eps = rms_norm_eps


ns = {"torch": torch, "nn": nn, "F": F, "Qwen4ExpTextConfig": Qwen4ExpTextConfig}
exec(lift(RMSNORM), ns)
exec(lift(GATEDRES), ns)
Qwen4ExpTextRMSNorm = ns["Qwen4ExpTextRMSNorm"]
Qwen4ExpTextGatedResidual = ns["Qwen4ExpTextGatedResidual"]

torch.set_default_dtype(torch.float32)
out = []


def emit(name, t):
    flat = t.detach().reshape(-1).tolist()
    out.append("const float %s[] = {" % name)
    for i in range(0, len(flat), 4):
        out.append("    " + ", ".join("%.9gf" % v for v in flat[i:i + 4]) + ",")
    out.append("};")


def case(tag, hidden, hc, lowrank, eps, tokens, use_combine, seed):
    g = torch.Generator().manual_seed(seed)
    cfg = Qwen4ExpTextConfig(hidden, hc, lowrank, eps)
    mod = Qwen4ExpTextGatedResidual(cfg, use_combine=use_combine)
    hc_h = hc * hidden
    with torch.no_grad():
        # hc_norm.weight is ZERO-init upstream; a zero weight makes (1 + w) == 1
        # and would hide the parameterization entirely, so it is randomized.
        mod.hc_norm.weight.copy_(torch.randn(hc_h, generator=g) * 0.5)
        mod.input_mix_weight_down.weight.copy_(torch.randn(lowrank, hc_h, generator=g) * 0.3)
        mod.input_mix_weight_up.weight.copy_(torch.randn(hc_h, lowrank, generator=g) * 0.3)
        if use_combine:
            mod.block_inject_weight.weight.copy_(torch.randn(hc, hc_h, generator=g) * 0.3)
        hyper = torch.randn(tokens, hc_h, generator=g) * 1.7
        block_out = torch.randn(tokens, hidden, generator=g) * 0.9

        normed = mod.hc_norm(hyper)
        res = mod(hyper)
        if use_combine:
            mixed, hyper_ret, inj = res
            # NOT an `assert`, for the same reason as the sha guard above: this
            # is the only check that upstream still hands back the RAW input for
            # the write-back, and `python3 -O` strips an `assert`.
            if hyper_ret is not hyper:
                raise SystemExit("upstream must return hyper_input RAW")
            injection = block_out.unsqueeze(-2) * inj.unsqueeze(-1)
            written = hyper + injection.flatten(-2)
        else:
            mixed = res

    out.append("")
    out.append("// ---- %s: hidden=%d hc=%d lowrank=%d eps=%g T=%d use_combine=%s seed=%d"
               % (tag, hidden, hc, lowrank, eps, tokens, use_combine, seed))
    emit("k%s_norm_w_hf" % tag, mod.hc_norm.weight)
    emit("k%s_down" % tag, mod.input_mix_weight_down.weight)
    emit("k%s_up" % tag, mod.input_mix_weight_up.weight)
    if use_combine:
        emit("k%s_inject" % tag, mod.block_inject_weight.weight)
    emit("k%s_hyper" % tag, hyper)
    emit("k%s_normed" % tag, normed)
    emit("k%s_mixed" % tag, mixed)
    if use_combine:
        emit("k%s_block_out" % tag, block_out)
        emit("k%s_inj_w" % tag, inj)
        emit("k%s_written" % tag, written)


out.append("// GENERATED by scripts/gen-qwen4-exp-hc-goldens.py -- do not hand-edit.")
out.append("// Oracle: transformers v5.16.0 modeling_qwen4_exp.py")
out.append("//   sha256 %s" % EXPECT_SHA)
out.append("//   Qwen4ExpTextRMSNorm  :158-181   (grouped RMSNorm, (1.0 + weight))")
out.append("//   Qwen4ExpTextGatedResidual :941-969")
out.append("//   write-back: Qwen4ExpTextDecoderLayer.forward, the two lines")
out.append("//     injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)")
out.append("//     hidden_states = hyper_input + injection.flatten(-2)")
out.append("// torch %s" % torch.__version__)

case("A", hidden=6, hc=4, lowrank=5, eps=1e-6, tokens=3, use_combine=True, seed=1234)
case("B", hidden=5, hc=3, lowrank=7, eps=1e-5, tokens=2, use_combine=True, seed=99)
case("C", hidden=6, hc=4, lowrank=5, eps=1e-6, tokens=2, use_combine=False, seed=7)

open(sys.argv[1], "w").write("\n".join(out) + "\n")
print("wrote", sys.argv[1])
