#!/usr/bin/env python3
"""Regenerate tests/vllm/models/qwen4_exp_ple_goldens.inc from the ORACLE.

Issue #1987, spec `.agents/specs/qwen4-exp-flash-next.md`.

Oracle: huggingface/transformers **v5.16.0**, this row's ACCEPTED lane pin. It
is the FIRST release containing `qwen4_exp` (`v5.15.0` returns HTTP 404 for the
same path), and vLLM implements NEITHER of the two components this file covers,
so there is no primary oracle to mirror instead.

NOTHING HERE RE-IMPLEMENTS UPSTREAM. The script downloads the two upstream files
at the `v5.16.0` tag and `exec`s the named line ranges verbatim; only the
scaffolding around them (a config holder, a cache container) is local, and it is
the smallest thing that makes those ranges run. A golden below is therefore an
oracle observation and not a transcription. Where a value upstream computes is
not on a return path -- the n-gram ids -- it is recovered by making the gather
that consumes it invertible, never by re-running the loop that produced it; see
`ngram_ids_for`.

Needs `torch` and network access. Usage:

    python3 scripts/gen-qwen4-exp-ple-goldens.py
"""

import io
import math
import pathlib
import sys
import textwrap
import types
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

TAG = "v5.16.0"
BASE = f"https://raw.githubusercontent.com/huggingface/transformers/{TAG}/src/transformers"
MODELING = f"{BASE}/models/qwen4_exp/modeling_qwen4_exp.py"
CACHE_UTILS = f"{BASE}/cache_utils.py"

OUT_PATH = (pathlib.Path(__file__).resolve().parent.parent
            / "tests" / "vllm" / "models" / "qwen4_exp_ple_goldens.inc")


def fetch(url):
    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise SystemExit(f"{url} returned HTTP {response.status}")
        return response.read().decode("utf-8").splitlines(keepends=True)


def rng(lines, first, last):
    """The upstream lines [first, last], 1-indexed inclusive, verbatim."""
    return "".join(lines[first - 1:last])


SRC = fetch(MODELING)
CACHE = fetch(CACHE_UTILS)

NS = {"torch": torch, "nn": nn, "F": F, "math": math,
      "Cache": object, "Qwen4ExpTextConfig": object}

# modeling_qwen4_exp.py:158-181   Qwen4ExpTextRMSNorm (the group_size arm)
# modeling_qwen4_exp.py:204-213   apply_mask_to_padding_states
# modeling_qwen4_exp.py:971-1015  _MASK64 .. _find_nth_prime_after
# modeling_qwen4_exp.py:1018-1114 Qwen4ExpTextNGramEmbedding
# modeling_qwen4_exp.py:1117-1189 Qwen4ExpTextPLELayer
for first, last in ((158, 181), (204, 213), (971, 1015), (1018, 1114), (1117, 1189)):
    exec(compile(rng(SRC, first, last), f"modeling_qwen4_exp.py@{first}", "exec"), NS)

_splitmix64 = NS["_splitmix64"]
_build_layer_multipliers = NS["_build_layer_multipliers"]
_find_nth_prime_after = NS["_find_nth_prime_after"]
NGram = NS["Qwen4ExpTextNGramEmbedding"]
PLE = NS["Qwen4ExpTextPLELayer"]


class Cfg(types.SimpleNamespace):
    """Only the keys the two modules read."""


class _MixinBase:
    """The attribute surface of LinearAttentionCacheLayerMixin, nothing more."""

    def __init__(self, n=3):
        self.number_of_states = n
        self.conv_states = [None] * n
        self.recurrent_states = [None] * n
        self.conv_kernel_size = [None] * n
        self.is_conv_states_initialized = [False] * n
        self.is_recurrent_states_initialized = [False] * n
        self.has_previous_state = [False] * n
        self.record_past = False
        self.device = None
        self.dtype = None


# cache_utils.py:1003-1075 verbatim: LinearAttentionLayer.lazy_initialization and
# .update_conv_state, which the PLE conv (state 1) and the n-gram token history
# (state 2) both ride on. The zero pad it applies on a short first chunk is
# exactly why the n-gram side EOS-pads explicitly before calling it.
_LAYER_NS = {"torch": torch, "is_torchdynamo_compiling": lambda: True,
             "LinearAttentionCacheLayerMixin": _MixinBase}
exec(compile(rng(CACHE, 1003, 1075), "cache_utils.py@1003", "exec"), _LAYER_NS)
CacheLayer = _LAYER_NS["LinearAttentionLayer"]


class Cache:
    def __init__(self, num_layers=1, n_states=3):
        self.layers = [CacheLayer(n_states) for _ in range(num_layers)]

    def has_previous_state(self, layer_idx, state_idx=0):
        return self.layers[layer_idx].has_previous_state[state_idx]

    def update_conv_state(self, conv_states, layer_idx, state_idx=0, **kwargs):
        return self.layers[layer_idx].update_conv_state(conv_states, state_idx, **kwargs)


torch.manual_seed(20260826)
OUT = io.StringIO()
W = OUT.write

# ----------------------------------------------------------- A. the real config
# `Qwen/Qwen3.8-Flash-Next` text_config.vocab_size, and the dataclass default
# `seed` (config.seed is ABSENT from the published config.json). Together they
# are the UNIQUE preimage below 2e6 of the three multipliers published in #1987,
# which is a fourth confirmation on top of that issue's three.
REAL_VOCAB = 248320
REAL_SEED = 1234
real_mults = _build_layer_multipliers(REAL_VOCAB, 3, 0, REAL_SEED).tolist()
sizes = [_find_nth_prime_after(20_000_000 - 1, i + 1) for i in range(16)]
offs, tot = [], 0
for size in sizes:
    offs.append(tot)
    tot += size
padded = math.ceil(tot / 128) * 128

# The raw chain, so a signed-shift port fails on the value itself and not only
# on a derived multiplier. Half of these have their top bit set.
SPLIT_PROBES = [0, 1, 1234, REAL_SEED + 0x9E3779B97F4A7C15,
                0x9E3779B97F4A7C15, (1 << 63), (1 << 64) - 1, 0xDEADBEEFCAFEF00D]

W("// GENERATED by scripts/gen-qwen4-exp-ple-goldens.py -- do not edit.\n")
W(f"// Oracle: huggingface/transformers {TAG},\n")
W("//   src/transformers/models/qwen4_exp/modeling_qwen4_exp.py\n")
W("//   src/transformers/cache_utils.py\n")
W("// Produced by exec'ing the upstream line ranges VERBATIM, never by\n")
W("// transcribing them; see the generator. transformers 5.16.0 is this row's\n")
W("// accepted lane pin (spec `## Oracles`) and vLLM implements neither of the\n")
W("// two components below, so there is no primary oracle to mirror instead.\n\n")

W("// modeling_qwen4_exp.py:979-983  _splitmix64\n")
W("static const struct { uint64_t in; uint64_t out; } kSplitMix64[] = {\n")
for probe in SPLIT_PROBES:
    W(f"    {{{probe}ULL, {_splitmix64(probe)}ULL}},\n")
W("};\n\n")

W("// modeling_qwen4_exp.py:986-995  _build_layer_multipliers, at the REAL config:\n")
W(f"// vocab_size={REAL_VOCAB}, ngram_size=3, ple_layer_index=0, seed={REAL_SEED}.\n")
W("// Matches the three values published in issue #1987 and range-read from the\n")
W("// released safetensors; vocab_size=248320 is the UNIQUE preimage below 2e6.\n")
W(f"static const int64_t kRealVocabSize = {REAL_VOCAB};\n")
W(f"static const int64_t kRealSeed = {REAL_SEED};\n")
W("static const int64_t kRealLayerMultipliers[3] = {"
  + ", ".join(f"{m}LL" for m in real_mults) + "};\n\n")

W("// modeling_qwen4_exp.py:1009-1015 _find_nth_prime_after, ngram_vocab_size_base\n")
W("// 20000000, ngram_heads 16 (ngram_size 3 x heads_per_ngram 8).\n")
W("static const int64_t kRealHeadVocabSizes[16] = {"
  + ", ".join(f"{s}LL" for s in sizes) + "};\n")
W("static const int64_t kRealHeadOffsets[16] = {"
  + ", ".join(f"{o}LL" for o in offs) + "};\n")
W(f"static const int64_t kRealTotalVocabSize = {tot}LL;\n")
W(f"static const int64_t kRealPaddedVocabSize = {padded}LL;\n\n")

# ------------------------------------------------- B. a tiny runnable config
TINY = Cfg(
    hidden_size=8, hc_count=2, ple_embed_dim=8, ple_conv_kernel_size=4,
    ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=20,
    make_ngram_vocab_size_divisible_by=8, vocab_size=64, eos_token_id=5,
    seed=1234, rms_norm_eps=1e-6,
)
H, HC, E = TINY.hidden_size, TINY.hc_count, TINY.ple_embed_dim
NH = (TINY.ngram_size - 1) * TINY.heads_per_ngram
SCS = (TINY.ple_conv_kernel_size - 1) * TINY.ngram_size
EOS = TINY.eos_token_id

ng = NGram(TINY, E, layer_idx=0, ple_layer_index=0)
W("// ---- tiny config, exercised end to end -------------------------------------\n")
W(f"static const int64_t kTinyHiddenSize = {H};\n")
W(f"static const int64_t kTinyHcCount = {HC};\n")
W(f"static const int64_t kTinyPleEmbedDim = {E};\n")
W(f"static const int64_t kTinyNgramSize = {TINY.ngram_size};\n")
W(f"static const int64_t kTinyHeadsPerNgram = {TINY.heads_per_ngram};\n")
W(f"static const int64_t kTinyNgramVocabBase = {TINY.ngram_vocab_size_base};\n")
W(f"static const int64_t kTinyVocabDivisor = {TINY.make_ngram_vocab_size_divisible_by};\n")
W(f"static const int64_t kTinyVocabSize = {TINY.vocab_size};\n")
W(f"static const int64_t kTinyEosTokenId = {EOS};\n")
W(f"static const int64_t kTinySeed = {TINY.seed};\n")
W(f"static const int64_t kTinyConvKernel = {TINY.ple_conv_kernel_size};\n")
W(f"static const int64_t kTinyShortConvStateLen = {SCS};\n")
W("static const int64_t kTinyHeadVocabSizes[%d] = {%s};\n"
  % (NH, ", ".join(f"{s}LL" for s in ng.head_vocab_sizes)))
W("static const int64_t kTinyHeadOffsets[%d] = {%s};\n"
  % (NH, ", ".join(f"{o}LL" for o in ng.head_offsets)))
W("static const int64_t kTinyLayerMultipliers[%d] = {%s};\n"
  % (TINY.ngram_size, ", ".join(f"{m}LL" for m in ng.layer_multipliers.tolist())))
W(f"static const int64_t kTinyTotalVocabSize = {ng.total_vocab_size}LL;\n")
W(f"static const int64_t kTinyPaddedVocabSize = {ng.ngram_embedding.num_embeddings}LL;\n\n")

# --------------------------------------------- C. _shift_right_ignore_eos
# Deliberately EOS-dense: an EOS at the head, two in the interior, an adjacent
# pair, and a run shorter than the largest shift.
SHIFT_ROW = [EOS, 11, 12, 13, EOS, 21, EOS, EOS, 31, 32, 33, 34]
tok = torch.tensor([SHIFT_ROW], dtype=torch.long)
W("// modeling_qwen4_exp.py:1053-1067  _shift_right_ignore_eos\n")
W(f"static const int64_t kShiftSeqLen = {len(SHIFT_ROW)};\n")
W("static const int64_t kShiftInput[%d] = {%s};\n"
  % (len(SHIFT_ROW), ", ".join(f"{t}LL" for t in SHIFT_ROW)))
W("static const int64_t kShiftExpected[3][%d] = {\n" % len(SHIFT_ROW))
for shift in range(3):
    row = ng._shift_right_ignore_eos(tok, shift)[0].tolist()
    W("    {" + ", ".join(f"{v}LL" for v in row) + "},  // shift=%d\n" % shift)
W("};\n\n")

# ----------------------------------------------- D. n-gram id construction
PREFILL = [7, 8, EOS, 9, 10, 11, EOS, 12, 13, 14]
DECODE = [15, 16]


def ngram_ids_for(chunks):
    """The ids READ OUT of upstream's own forward, never rebuilt from it.

    `forward` returns embeddings rather than ids (:1114), so the obvious way to
    pin the ids is to re-run its block-assembly loop -- and that is exactly the
    transcription this file exists to avoid, because the generator and the port
    would then share one reading of :1097-1112 and a shared misreading would
    pass. Instead the gather is made INVERTIBLE: row i of `ngram_embedding` is
    filled with the scalar i, so `forward` returns the ids themselves, repeated
    `head_dim_per_ngram` times each, and they are read straight off the result.
    The assertion below is what makes the inversion checkable rather than
    assumed. Every line that computes an id is upstream's, executed.
    """
    cache = Cache(num_layers=1, n_states=3)
    module = NGram(TINY, E, layer_idx=0, ple_layer_index=0)
    head_dim = E // module.ngram_heads
    with torch.no_grad():
        for row in range(module.ngram_embedding.num_embeddings):
            module.ngram_embedding.weight[row].fill_(float(row))
        # float32 holds every integer below 2**24 exactly; assert it rather than
        # trust it, because a bigger tiny config would silently round.
        assert module.ngram_embedding.num_embeddings < (1 << 24)
    out = []
    for chunk in chunks:
        ids = torch.tensor([chunk], dtype=torch.long)
        with torch.no_grad():
            embedded = module.forward(ids, cache)[0]
        recovered = embedded[:, ::head_dim].to(torch.long)
        assert torch.equal(
            embedded, recovered.repeat_interleave(head_dim, dim=-1).to(embedded.dtype)), \
            "the embedding is not invertible; the recovered ids would be a guess"
        out.append(recovered.tolist())
    return out


ALL = PREFILL + DECODE
single = ngram_ids_for([ALL])[0]
chunked = ngram_ids_for([PREFILL, [DECODE[0]], [DECODE[1]]])
assert single == chunked[0] + chunked[1] + chunked[2], \
    "cached decode must equal single-shot prefill"

W("// modeling_qwen4_exp.py:1069-1114  Qwen4ExpTextNGramEmbedding.forward, id half.\n")
W("// The same 12 tokens, once as one prefill and once as prefill(10)+decode+decode;\n")
W("// upstream produces IDENTICAL ids, which is what pins the conv-state-2 history.\n")
W(f"static const int64_t kNgramPrefillLen = {len(PREFILL)};\n")
W(f"static const int64_t kNgramTotalLen = {len(ALL)};\n")
W("static const int64_t kNgramTokens[%d] = {%s};\n"
  % (len(ALL), ", ".join(f"{t}LL" for t in ALL)))
W("static const int64_t kNgramExpectedIds[%d][%d] = {\n" % (len(ALL), NH))
for row in single:
    W("    {" + ", ".join(f"{v}LL" for v in row) + "},\n")
W("};\n\n")

# ------------------------------------------------------------- E. the gate
GATE_IN = [0.0, 1e-12, -1e-12, 1e-6, -1e-6, 1e-3, -1e-3, 0.25, -0.25, 4.0, -4.0]
gate = torch.tensor(GATE_IN, dtype=torch.float32)
gate_out = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
W("// modeling_qwen4_exp.py:1181  gate.abs().clamp_min(1e-6).sqrt() * gate.sign()\n")
W("// Clamp BEFORE the sqrt: the magnitude floor is 1e-3, not 1e-6, and exactly\n")
W("// zero maps to zero because sign(0)=0. Discontinuous at the origin on purpose.\n")
W(f"static const int64_t kGateCount = {len(GATE_IN)};\n")
W("static const float kGateInput[%d] = {%s};\n"
  % (len(GATE_IN), ", ".join(f"{v!r}f" for v in GATE_IN)))
W("static const float kGateExpected[%d] = {%s};\n\n"
  % (len(GATE_IN), ", ".join(f"{v!r}f" for v in gate_out.tolist())))

# --------------------------------------------------- F. the dilated conv taps
ple = PLE(TINY, layer_idx=0, ple_layer_index=0)
with torch.no_grad():
    ple.conv1d.weight.zero_()
    # Channel c carries its whole weight at kernel index c % 4, so a wrong lag,
    # a unit stride or a reversed tap order all move the impulse response.
    for channel in range(H * HC):
        ple.conv1d.weight[channel, 0, channel % TINY.ple_conv_kernel_size] = 1.0
TAPLEN = 14
delta = torch.zeros(1, TAPLEN, H * HC)
delta[0, 0, :] = 3.0
taps = ple._short_conv(delta, None)
W("// modeling_qwen4_exp.py:1150-1167  _short_conv. kernel 4, dilation ngram_size=3,\n")
W("// so output t reads input t-9, t-6, t-3, t with weights w0..w3 in that order.\n")
W("// One-hot tap per channel over an impulse at t=0: the response lands at\n")
W("// t = 9, 6, 3, 0 for w0..w3. silu is applied to the conv output.\n")
W(f"static const int64_t kTapSeqLen = {TAPLEN};\n")
W("static const float kTapImpulse = 3.0f;\n")
W("static const float kTapExpected[%d][%d] = {\n" % (TAPLEN, H * HC))
for t in range(TAPLEN):
    W("    {" + ", ".join(f"{v!r}f" for v in taps[0, t].tolist()) + "},\n")
W("};\n\n")

# ----------------------------------------------------------- G. PLE end to end
ple2 = PLE(TINY, layer_idx=0, ple_layer_index=0)
with torch.no_grad():
    for parameter in ple2.parameters():
        parameter.uniform_(-0.6, 0.6)
    ple2.ple_embedding.ngram_embedding.weight.uniform_(-0.5, 0.5)


def dump_tensor(name, tensor):
    flat = tensor.reshape(-1).tolist()
    W("static const float %s[%d] = {\n" % (name, len(flat)))
    for i in range(0, len(flat), 6):
        W("    " + ", ".join(f"{v!r}f" for v in flat[i:i + 6]) + ",\n")
    W("};\n")


hidden = torch.empty(1, len(ALL), H * HC).uniform_(-1.0, 1.0)
with torch.no_grad():
    single_out = ple2(hidden, torch.tensor([ALL], dtype=torch.long),
                      Cache(num_layers=1, n_states=3), conv_mask=None)
    cache = Cache(num_layers=1, n_states=3)
    parts, lo = [], 0
    for count in (len(PREFILL), 1, 1):
        parts.append(ple2(hidden[:, lo:lo + count],
                          torch.tensor([ALL[lo:lo + count]], dtype=torch.long),
                          cache, conv_mask=None))
        lo += count
    incremental = torch.cat(parts, dim=1)
assert torch.allclose(single_out, incremental, atol=1e-5), \
    "incremental PLE forward must equal the single-shot one"

W("// The full Qwen4ExpTextPLELayer forward (modeling_qwen4_exp.py:1169-1189),\n")
W("// tiny config, weights drawn once and frozen here. Prefill(10)+decode+decode\n")
W("// equals the single-shot 12-token prefill upstream, so both are one golden.\n")
dump_tensor("kPleNgramEmbeddingWeight", ple2.ple_embedding.ngram_embedding.weight)
dump_tensor("kPleKeyProjWeight", ple2.key_proj.weight)
dump_tensor("kPleValueProjWeight", ple2.value_proj.weight)
dump_tensor("kPleNormKeyWeight", ple2.norm_key.weight)
dump_tensor("kPleNormQueryWeight", ple2.norm_query.weight)
dump_tensor("kPleNormConvWeight", ple2.norm_conv.weight)
dump_tensor("kPleConv1dWeight", ple2.conv1d.weight)
dump_tensor("kPleHiddenStates", hidden)
dump_tensor("kPleExpectedOutput", single_out)

# ----------------------------------------------------- H. the conv_mask arm
# `conv_mask` is prefill-only (`None` in steady-state decode) and upstream masks
# BOTH tensors at :1185-1187 -- `gated_value`, which is the skip term, AND
# `gated_value_normed`, which is what enters the conv AND what the 9-column
# state keeps. Masking one of the two is a real and easy port defect, so the
# mask has to be gated rather than documented.
#
# The mask is a PAIRED obligation with the caller (see the header): a masked
# position must already carry EOS in `input_ids`, because the hash reads token
# ids and not activations. These tokens honour that, so the golden pins the
# contract rather than an inconsistent state nobody would produce.
#
# Zeros at 3 and 4 are INTERIOR, not trailing: the conv is dilated by 3, so
# output t reads t-9, t-6, t-3 and t, and an interior zero therefore has to move
# t = 3, 6, 9 and 12 as well as its own row. A trailing-pad-only mask would
# leave the conv path almost untouched.
MASK_TOKENS = [7, 8, EOS, EOS, EOS, 11, EOS, 12, 13, 14, 15, EOS]
CONV_MASK = [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
assert len(MASK_TOKENS) == len(CONV_MASK) == len(ALL)
assert all(MASK_TOKENS[i] == EOS for i, m in enumerate(CONV_MASK) if m == 0), \
    "a masked position must carry EOS: the hash reads ids, not activations"

mask_t = torch.tensor([CONV_MASK], dtype=torch.bool)
with torch.no_grad():
    masked_single = ple2(hidden, torch.tensor([MASK_TOKENS], dtype=torch.long),
                         Cache(num_layers=1, n_states=3), conv_mask=mask_t)
    cache = Cache(num_layers=1, n_states=3)
    parts, lo = [], 0
    for count in (len(PREFILL), 1, 1):
        parts.append(ple2(hidden[:, lo:lo + count],
                          torch.tensor([MASK_TOKENS[lo:lo + count]], dtype=torch.long),
                          cache, conv_mask=mask_t[:, lo:lo + count]))
        lo += count
    masked_incremental = torch.cat(parts, dim=1)
assert torch.allclose(masked_single, masked_incremental, atol=1e-5), \
    "the mask must reach the 9-column state, so both arms must agree"
assert not torch.allclose(masked_single, single_out, atol=1e-3), \
    "the mask must actually change the output, or this golden gates nothing"

W("// modeling_qwen4_exp.py:1185-1187 + :204-213 apply_mask_to_padding_states.\n")
W("// BOTH `gated_value` (the skip term) and `gated_value_normed` (the conv\n")
W("// input, and what the 9-column state keeps) are masked. Zeros at 3, 4 and 11;\n")
W("// 3 and 4 are interior, so the dilation carries them to t = 6, 9 and 12 too.\n")
W("// Masked positions carry EOS in the tokens, which is the paired obligation.\n")
W("static const int64_t kPleMaskTokens[%d] = {%s};\n"
  % (len(MASK_TOKENS), ", ".join(f"{t}LL" for t in MASK_TOKENS)))
W("static const unsigned char kPleConvMask[%d] = {%s};\n"
  % (len(CONV_MASK), ", ".join(str(m) for m in CONV_MASK)))
dump_tensor("kPleMaskedExpectedOutput", masked_single)

# ------------- I. the dilated conv ALONE, with the DILATION as the variable
# W5b-3 (#2156). The device op `vt::Qwen4ExpPleConv` is `_short_conv` and
# nothing else, so it needs a golden that is `_short_conv` and nothing else:
# section F's impulse is one-hot per channel, which cannot see an accumulation
# defect, and section G's golden is the whole layer, which cannot localise one.
#
# THE DILATION IS THE VARIABLE, AND THE ORACLE SUPPLIES BOTH SIDES. The same
# upstream method, the same input and the SAME conv weight are run at dilation
# 3 (the model's `ngram_size`), 2 and 1, by swapping only the `nn.Conv1d` and
# the `short_conv_state_len` that upstream itself derives from it. A test
# fixture in which dilation 3 and dilation 1 agree would gate nothing, so the
# separation between the three is asserted here, at generation time, and the
# measured value is written into the file for the reader.
CONV_LEN = 12
CONV_SPLIT = (7, 1, 4)  # prefill(7) + decode(1) + prefill(4)
assert sum(CONV_SPLIT) == CONV_LEN
CONV_DILATIONS = (1, 2, 3)
conv_in = torch.empty(1, CONV_LEN, H * HC).uniform_(-1.0, 1.0)
conv_out = {}
for _dil in CONV_DILATIONS:
    probe = PLE(TINY, layer_idx=0, ple_layer_index=0)
    with torch.no_grad():
        # Only the conv and the state width change. `_short_conv` itself is the
        # upstream method, unmodified and executed, not re-implemented.
        probe.conv1d = nn.Conv1d(H * HC, H * HC, kernel_size=TINY.ple_conv_kernel_size,
                                 groups=H * HC, dilation=_dil, bias=False)
        probe.conv1d.weight.copy_(ple2.conv1d.weight)
        probe.short_conv_state_len = (TINY.ple_conv_kernel_size - 1) * _dil
        single_conv = probe._short_conv(conv_in, None)
        cache = Cache(num_layers=1, n_states=3)
        parts, lo = [], 0
        for count in CONV_SPLIT:
            parts.append(probe._short_conv(conv_in[:, lo:lo + count], cache))
            lo += count
        inc_conv = torch.cat(parts, dim=1)
    assert torch.allclose(single_conv, inc_conv, atol=1e-6), \
        f"the {(TINY.ple_conv_kernel_size - 1) * _dil}-column state must make " \
        f"chunked equal single-shot at dilation {_dil}"
    conv_out[_dil] = single_conv

_seps = {}
for _i, _a in enumerate(CONV_DILATIONS):
    for _b in CONV_DILATIONS[_i + 1:]:
        _seps[(_a, _b)] = (conv_out[_a] - conv_out[_b]).abs().max().item()
assert min(_seps.values()) > 1e-2, \
    f"the dilations must separate or the fixture gates nothing: {_seps}"

W("// modeling_qwen4_exp.py:1150-1167  _short_conv ALONE, dense weight (the same\n")
W("// kPleConv1dWeight above), dense input, at THREE dilations. The upstream\n")
W("// method is executed unmodified; only its `nn.Conv1d` and the\n")
W("// `short_conv_state_len` upstream derives from it are swapped, so the\n")
W("// dilation is the ONLY variable. State width is (kernel - 1) * dilation:\n")
W("// 3, 6 and 9 columns. Each was additionally checked to survive a\n")
W("// prefill(7)+decode(1)+prefill(4) chunking through the cache.\n")
W("// MEASURED pairwise max|difference| between the three answers: "
  + ", ".join(f"d{a} vs d{b} {v:.6g}" for (a, b), v in _seps.items()) + ".\n")
W("// A fixture whose dilations agreed would gate nothing; these do not.\n")
W(f"static const int64_t kConvSeqLen = {CONV_LEN};\n")
W("static const int64_t kConvChunks[%d] = {%s};\n"
  % (len(CONV_SPLIT), ", ".join(f"{c}LL" for c in CONV_SPLIT)))
W("static const int64_t kConvDilations[%d] = {%s};\n"
  % (len(CONV_DILATIONS), ", ".join(f"{d}LL" for d in CONV_DILATIONS)))
dump_tensor("kConvInput", conv_in)
for _dil in CONV_DILATIONS:
    dump_tensor(f"kConvExpectedD{_dil}", conv_out[_dil])
W("\n")

# ---------- J. the PLE GATE ALONE, modeling_qwen4_exp.py:1180-1182 (+ :1184) --
# MODEL-MM-QWEN4-EXP W5e-1 (#2336). `vt::Qwen4ExpPleGate` is those three lines
# and nothing else, so it needs a golden that is those three lines and nothing
# else. Section E already pins :1181 on eleven SCALARS; it cannot see the
# sigmoid, it cannot see the `value.unsqueeze(-2)` broadcast, and a port that
# multiplied the wrong axis would pass it. Section G is the whole layer, which
# sees everything and localises nothing.
#
# ANCHORS. #2336 cites this block as ":1179-1183"; at the pinned v5.16.0 file
# (sha256 77fec77d...c459) :1179 is the `query_normed` unflatten and :1183 is
# the `norm_conv` call, so the gate itself is :1180-1182 and the flatten it
# feeds is :1184. The one-line shift is corrected here and in the row's spec.
#
# HOW THIS IS PRODUCED. The upstream lines are `exec`d VERBATIM by line range
# on inputs chosen here -- :1180 alone first, so its scaled dot is observable,
# then :1181-1182, then the :1184 flatten. No line of the gate is retyped.
#
# THE CLAMP IS THE VARIABLE, AND THE FIXTURE PROBES BOTH SIDES OF IT. The
# `clamp_min(1e-6)` sits BEFORE the sqrt, so the floor on |gate| is 1e-3 and not
# 1e-6, and a fixture on which it never binds would gate nothing at all -- the
# blind spot #2272 recorded for an eps invisible at two of four goldens. Four of
# the twelve (t, j) pairs are therefore built to straddle it:
#
#   (0,0)  key row ZEROED      -> the dot is exactly 0, sign(0) == 0, and the
#                                 gate is 0 rather than the 1e-3 floor. THE
#                                 ORIGIN, where the function is discontinuous.
#   (0,1)  key row * 1e-7      -> |gate| ~ 1e-8, the clamp BINDS, positive
#   (1,0)  key row * -1e-7     -> the clamp BINDS with the sign preserved
#   (1,1)  key row * 1e-5      -> just above the floor; asserted, not assumed
#
# and the remaining eight are dense, where it must be INERT. `kGateClampBinds`
# records which is which, read off upstream's own :1180 output, so the test
# asserts the population rather than trusting this comment.
#
# The value rows of the two probing tokens are scaled up so the clamp's effect
# on the OUTPUT is large against the gate tolerance: the whole dynamic range of
# the clamp is sigmoid(1e-3) - sigmoid(0) = 2.5e-4 per unit of value, so an
# unscaled fixture would separate by 2.5e-4 and a port that dropped the clamp
# would sit 25x above a 1e-5 bound rather than comfortably above it.
GATE_T = 6
_g = torch.Generator().manual_seed(20260830)
gate_key = torch.empty(1, GATE_T, HC, H).uniform_(-1.0, 1.0, generator=_g)
gate_query = torch.empty(1, GATE_T, HC, H).uniform_(-1.0, 1.0, generator=_g)
gate_value = torch.empty(1, GATE_T, H).uniform_(-1.0, 1.0, generator=_g)
gate_key[0, 0, 0].zero_()
gate_key[0, 0, 1] *= 1e-7
gate_key[0, 1, 0] *= -1e-7
gate_key[0, 1, 1] *= 1e-5
gate_value[0, 0] *= 8.0
gate_value[0, 1] *= 8.0


def body_of(first, last):
    """The upstream lines, verbatim, dedented out of their method body."""
    return textwrap.dedent(rng(SRC, first, last))


def run_gate(key, query, value, eps=None):
    """upstream :1180, then :1181-1182, then the :1184 flatten -- verbatim.

    `eps` replaces the clamp floor and exists only for the separation number
    below; it is None for every golden this file emits.
    """
    ns = {"torch": torch, "math": math,
          "self": types.SimpleNamespace(hidden_size=H),
          "key_normed": key, "query_normed": query, "value": value}
    exec(compile(body_of(1180, 1180), "modeling_qwen4_exp.py@1180", "exec"), ns)
    pre = ns["gate"].clone()
    body = body_of(1181, 1182)
    if eps is not None:
        assert "clamp_min(1e-6)" in body
        body = body.replace("clamp_min(1e-6)", f"clamp_min({eps!r})")
    exec(compile(body, "modeling_qwen4_exp.py@1181", "exec"), ns)
    post = ns["gate"].clone()
    exec(compile(body_of(1184, 1184), "modeling_qwen4_exp.py@1184", "exec"), ns)
    return pre, post, ns["gated_value"]


with torch.no_grad():
    gate_pre, gate_post, gate_out = run_gate(gate_key, gate_query, gate_value)
    # The same three lines with the floor taken to zero. torch has no way to
    # DELETE the clamp from a line it is executing, and 0.0 is what deleting it
    # means: `x.abs().clamp_min(0)` is `x.abs()`. This is the fixture's
    # discriminating power, not a golden -- nothing is compared against it.
    _, gate_post_noclamp, gate_out_noclamp = run_gate(
        gate_key, gate_query, gate_value, eps=0.0)

gate_binds = (gate_pre.abs() < 1e-6).reshape(-1)
gate_sep = (gate_out - gate_out_noclamp).abs().max().item()
assert gate_pre[0, 0, 0, 0].item() == 0.0, "the origin probe must be EXACTLY zero"
assert gate_post[0, 0, 0, 0].item() == 0.0, "sign(0) == 0, so the origin maps to 0"
assert int(gate_binds.sum()) == 3, \
    f"expected 3 clamped pairs, got {int(gate_binds.sum())}: {gate_pre.reshape(-1)}"
assert int((~gate_binds).sum()) == GATE_T * HC - 3
assert gate_sep > 1e-3, \
    f"the clamp must move the output or the fixture gates nothing: {gate_sep}"
# |post| is EXACTLY the floor wherever the clamp bound, and strictly above it
# everywhere else. Asserted here so the emitted `kGateClampBinds` cannot drift
# away from the values beside it.
_floor = math.sqrt(1e-6)
for _i, (_b, _p) in enumerate(zip(gate_binds.tolist(),
                                  gate_post.reshape(-1).tolist())):
    if _i == 0:
        assert _p == 0.0, (_i, _p)
    elif _b:
        assert abs(abs(_p) - _floor) < 1e-9, (_i, _p)
    else:
        assert abs(_p) > _floor, (_i, _p)

W("// modeling_qwen4_exp.py:1180-1182 + the :1184 flatten -- the PLE GATE alone,\n")
W("// executed VERBATIM by line range on the inputs below. hc_count = 2,\n")
W("// hidden_size = 8, so `gate` is one scalar per (t, j) and `value` broadcasts\n")
W("// across j: BOTH operands of the :1182 multiply broadcast, which is why no\n")
W("// elementwise op in this tree can express it.\n")
W("// The clamp BINDS on 3 of the 12 (t, j) pairs and is INERT on the other 9;\n")
W("// kGateClampBinds is read off upstream's own :1180 output. (0,0) is the\n")
W("// ORIGIN: the dot is exactly 0, sign(0) = 0, and the gate is 0 rather than\n")
W("// the 1e-3 floor. MEASURED max|difference| between this golden and the same\n")
W("// three lines with the floor taken to zero: %.6g. A fixture on which the\n" % gate_sep)
W("// clamp did not bind would gate nothing.\n")
W(f"static const int64_t kGateT = {GATE_T};\n")
W(f"static const int64_t kGateHc = {HC};\n")
W(f"static const int64_t kGateH = {H};\n")
W("// upstream DIVIDES by math.sqrt(self.hidden_size) at :1180.\n")
W(f"static const float kGateDivisor = {math.sqrt(H)!r}f;\n")
W(f"static const float kGateClampSeparation = {gate_sep!r}f;\n")
W("static const unsigned char kGateClampBinds[%d] = {%s};\n"
  % (GATE_T * HC, ", ".join(str(int(b)) for b in gate_binds.tolist())))
dump_tensor("kGateKeyNormed", gate_key)
dump_tensor("kGateQueryNormed", gate_query)
dump_tensor("kGateValueIn", gate_value)
dump_tensor("kGateScaledDot", gate_pre)
dump_tensor("kGatePostSqrt", gate_post)
dump_tensor("kGateExpectedOut", gate_out)
W("\n")

OUT_PATH.write_text(OUT.getvalue())
print(f"wrote {OUT_PATH} ({len(OUT.getvalue().splitlines())} lines) from transformers {TAG}")
print(f"layer_multipliers at the real config: {real_mults}")
sys.exit(0)
