#!/usr/bin/env python3
"""Gate for `scripts/convert-glm5-next-gguf.py` (W7a of MODEL-MM-GLM53-FLASH).

The converter is the only path to a GLM-5.3-Flash artifact that fits any device
this project reaches, and it has never been run against the real checkpoint --
that is 300-600 GiB and is owed, not done (spec `## Owed` O7). So the gate is
built to need no checkpoint at all:

  1. A SYNTHETIC tiny-shape checkpoint, written here, with the real
     architecture's topology at ~1/16 the width: the 45-layer KDA/DSA hybrid
     becomes 4 layers, 288 experts become 4, and the FP8 e4m3 block form with
     `weight_scale_inv` companions is reproduced exactly, because the dequant
     path is the one part of the plumbing that silently produces a loadable
     wrong file.
  2. An INDEPENDENT GGUF reader, written in this file, that shares no code with
     the converter. A parser that reused the writer's structs would prove the
     two agree, not that either is right.
  3. A FROZEN k-quant golden captured from the pinned llama.cpp reference
     (`tests/scripts/fixtures/glm5_next_kquant_golden_b10451.json`, provenance in
     the fixture). Q2_K, Q6_K and Q8_0 must come back BYTE-IDENTICAL. This is the
     gate that matters most: a k-quant encoder that is close but not exact
     produces a file that loads, generates fluent text, and is quietly worse than
     the arm it claims to be. Bytes or nothing.

     **Three of the golden's six super-blocks are there because the first three
     were BLIND.** Random weight-like data never lands on a rounding tie, so a
     mutation replacing `nearest_int`'s round-half-to-EVEN with round-half-up
     passed, and so did one replacing C `roundf`'s round-half-away-from-zero with
     `np.rint`. Both are real defects and both were invisible. Blocks 3 and 4
     were SEARCHED on an eighth-lattice until the two rules disagreed under Q2_K
     and Q6_K; block 5 is eight Q8_0 sub-blocks with `amax` exactly 127 so that
     `x * id` lands on `.5`. The mutations fail now. A golden is only as strong
     as the inputs it was captured over, and this one had to be measured rather
     than assumed.

The converter's OUTPUT is not yet loadable by this tree -- `glm5next` is not
registered, which W1 owns -- so this suite gates the converter as a command-line
path, entering through `main()` the way a user does. That gap is named in the
commit body, the pull request body, and the spec's `## Owed`.

Exit 0 iff every case passes.
"""

import base64
import json
import re
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONVERTER = os.path.join(ROOT, "scripts", "convert-glm5-next-gguf.py")
FIXTURE = os.path.join(ROOT, "tests", "scripts", "fixtures",
                       "glm5_next_kquant_golden_b10451.json")

try:
    import numpy as np
except ImportError:
    sys.stderr.write(
        "SKIP: numpy is not importable here, so nothing about the converter was "
        "verified. CI installs python3-numpy and runs this suite.\n")
    raise SystemExit(0)

FAILURES = []


def check(cond, msg):
    if cond:
        print("  ok   %s" % msg)
    else:
        print("  FAIL %s" % msg)
        FAILURES.append(msg)


# ---------------------------------------------------------------------------
# The synthetic checkpoint. Shapes are the real config's relations at a small
# width, so a hard-coded dimension in the converter shows up as a shape error
# rather than passing by luck.

HIDDEN = 256
N_LAYERS = 4
LAYER_TYPES = ["linear_attention", "linear_attention",
               "deepseek_sparse_attention", "linear_attention"]
MLP_TYPES = ["dense", "sparse", "sparse", "sparse"]
N_EXPERTS = 4
MOE_INTER = 256
DENSE_INTER = 512
VOCAB = 512
HC_MULT = 4
KDA_HEADS = 2
KDA_HEAD_DIM = 128
KDA_INNER = KDA_HEADS * KDA_HEAD_DIM
CONV_K = 4
Q_LORA = 128
KV_LORA = 64
QK_NOPE = 64
V_HEAD = 64
N_HEADS = 4
IDX_HEADS = 2
IDX_HEAD_DIM = 64
IDX_KPOOL = 4
VIS_DEPTH = 2
VIS_HIDDEN = 128
VIS_INTER = 256


def tiny_config():
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "image_token_id": 154854, "video_token_id": 154855,
        "image_start_token_id": 154830, "image_end_token_id": 154831,
        "video_start_token_id": 154832, "video_end_token_id": 154833,
        "tie_word_embeddings": False,
        "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
                                "quant_method": "fp8",
                                "weight_block_size": [128, 128],
                                "modules_to_not_convert": ["hyper_connection"]},
        "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": HIDDEN, "num_hidden_layers": N_LAYERS,
            "intermediate_size": DENSE_INTER, "moe_intermediate_size": MOE_INTER,
            "vocab_size": VOCAB, "max_position_embeddings": 4096,
            "layer_types": LAYER_TYPES, "mlp_layer_types": MLP_TYPES,
            "indexer_types": ["full"] * N_LAYERS,
            "num_attention_heads": N_HEADS, "num_key_value_heads": N_HEADS,
            "n_routed_experts": N_EXPERTS, "n_shared_experts": 1,
            "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1,
            "norm_topk_prob": True, "scoring_func": "sigmoid",
            "topk_method": "noaux_tc", "routed_scaling_factor": 2.5,
            "first_k_dense_replace": 1,
            "rms_norm_eps": 1e-05, "hc_eps": 1e-06, "hc_mult": HC_MULT,
            "hc_sinkhorn_iters": 20, "mhc": True, "mla_use_nope": True,
            "swiglu_limit": 10.0, "hidden_act": "silu",
            "q_lora_rank": Q_LORA, "kv_lora_rank": KV_LORA,
            "qk_head_dim": QK_NOPE, "qk_nope_head_dim": QK_NOPE,
            "qk_rope_head_dim": 0, "v_head_dim": V_HEAD, "head_dim": 0,
            "index_head_dim": IDX_HEAD_DIM, "index_n_heads": IDX_HEADS,
            "index_topk": 2048, "index_kpool": IDX_KPOOL,
            "index_kpool_compress": True, "index_kpool_always_select_tail": True,
            "index_share_for_mtp_iteration": True, "indexer_rope_interleave": True,
            "num_nextn_predict_layers": 1, "moe_router_dtype": "float32",
            "linear_attn_config": {"num_heads": KDA_HEADS,
                                   "gate_lower_bound": -5.0,
                                   "head_dim": KDA_HEAD_DIM,
                                   "short_conv_kernel_size": CONV_K,
                                   "kda_layers": [0, 1, 3],
                                   "full_attn_layers": [2]},
        },
        "vision_config": {
            "model_type": "glm5_next_vision", "depth": VIS_DEPTH,
            "hidden_size": VIS_HIDDEN, "intermediate_size": VIS_INTER,
            "num_heads": 2, "patch_size": 14, "image_size": 448,
            "spatial_merge_size": 2, "temporal_patch_size": 2,
            "out_hidden_size": HIDDEN, "projection_intermediate_size": 512,
            "in_channels": 3, "rms_norm_eps": 1e-05, "swiglu_limit": 10.0,
            "attention_bias": True,
        },
    }


def det(shape, seed):
    """Deterministic values in a weight-like range, no RNG dependency."""
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    v = np.sin(i * 0.7391 + seed * 1.1) * 0.05 + np.cos(i * 0.113 + seed) * 0.01
    return v.astype(np.float32).reshape(shape)


def f32_to_e4m3(x):
    """Encode f32 -> e4m3 bytes by nearest-value search over the 256 codes.

    Deliberately brute force and written here rather than shared with the
    converter: the converter owns DECODE, this owns ENCODE, and the round trip
    is only evidence when the two sides are independent.
    """
    codes = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            codes[b] = np.nan
        elif e == 0:
            codes[b] = np.float32(s * (m / 8.0) * (2.0 ** -6))
        else:
            codes[b] = np.float32(s * (1.0 + m / 8.0) * (2.0 ** (e - 7)))
    finite = np.where(np.isfinite(codes))[0]
    tbl = codes[finite]
    flat = np.asarray(x, np.float32).reshape(-1, 1)
    idx = np.argmin(np.abs(flat - tbl.reshape(1, -1)), axis=1)
    return finite[idx].astype(np.uint8).reshape(np.shape(x)), tbl[idx].astype(
        np.float32).reshape(np.shape(x))


def st_write(path, tensors):
    """Minimal safetensors writer. `tensors` maps name -> (dtype, ndarray)."""
    header = {}
    blobs = []
    off = 0
    for name in sorted(tensors):
        dt, arr = tensors[name]
        raw = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(header).encode("utf-8")
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for b in blobs:
            fh.write(b)
    return list(header)


FP8 = "F8_E4M3"
BF16 = "BF16"
F32 = "F32"


def bf16(arr):
    u = np.ascontiguousarray(arr, np.float32).view(np.uint32)
    r = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))
         >> np.uint32(16)).astype(np.uint16)
    return r.reshape(arr.shape)


def fp8_pair(shape, seed, block=(128, 128)):
    """An FP8 e4m3 block-quantized tensor plus its `weight_scale_inv` grid."""
    ref = det(shape, seed)
    rows, cols = shape
    sr = (rows + block[0] - 1) // block[0]
    sc = (cols + block[1] - 1) // block[1]
    scale = (det((sr, sc), seed + 3) * 0.5 + 1.0).astype(np.float32)
    ri = np.minimum(np.arange(rows) // block[0], sr - 1)
    ci = np.minimum(np.arange(cols) // block[1], sc - 1)
    grid = scale[np.ix_(ri, ci)]
    q, decoded = f32_to_e4m3(ref / grid)
    return q, scale, (decoded * grid).astype(np.float32)


def build_checkpoint(dirpath, with_mtp=True):
    """Write config.json and one safetensors shard. Returns the exact values the
    converter must reproduce, keyed by HF tensor name."""
    cfg = tiny_config()
    with open(os.path.join(dirpath, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    t = {}
    expect = {}
    seed = [0]

    def add_bf16(name, shape):
        seed[0] += 1
        v = det(shape, seed[0])
        b = bf16(v)
        t[name] = (BF16, b)
        expect[name] = (b.astype(np.uint32) << np.uint32(16)).view(np.float32)

    def add_f32(name, shape):
        seed[0] += 1
        v = det(shape, seed[0])
        t[name] = (F32, v)
        expect[name] = v

    def add_fp8(name, shape):
        seed[0] += 1
        q, scale, ref = fp8_pair(shape, seed[0])
        t[name] = (FP8, q)
        t[name[:-len(".weight")] + ".weight_scale_inv"] = (F32, scale)
        expect[name] = ref

    p = "model.language_model."
    add_bf16(p + "embed_tokens.weight", (VOCAB, HIDDEN))
    add_bf16(p + "norm.weight", (HIDDEN,))
    add_bf16("lm_head.weight", (VOCAB, HIDDEN))

    layers = list(range(N_LAYERS)) + ([N_LAYERS] if with_mtp else [])
    for L in layers:
        b = "%slayers.%d." % (p, L)
        add_bf16(b + "input_layernorm.weight", (HIDDEN,))
        add_bf16(b + "post_attention_layernorm.weight", (HIDDEN,))
        is_mtp = L == N_LAYERS
        if not is_mtp:
            # mHC lives FLAT on the layer and the MTP block carries NONE of it.
            add_bf16(b + "hc_attn_fn", (HC_MULT * (HC_MULT + 2), HC_MULT * HIDDEN))
            add_f32(b + "hc_attn_base", (HC_MULT * (HC_MULT + 2),))
            add_f32(b + "hc_attn_scale", (HC_MULT - 1,))
            add_bf16(b + "hc_ffn_fn", (HC_MULT * (HC_MULT + 2), HC_MULT * HIDDEN))
            add_f32(b + "hc_ffn_base", (HC_MULT * (HC_MULT + 2),))
            add_f32(b + "hc_ffn_scale", (HC_MULT - 1,))
        kind = "deepseek_sparse_attention" if is_mtp else LAYER_TYPES[L]
        if kind == "linear_attention":
            for nm in ("q_proj", "k_proj", "v_proj"):
                add_bf16(b + "self_attn.%s.weight" % nm, (KDA_INNER, HIDDEN))
            for nm in ("q_conv1d", "k_conv1d", "v_conv1d"):
                add_bf16(b + "self_attn.%s.weight" % nm, (KDA_INNER, 1, CONV_K))
            add_bf16(b + "self_attn.f_a_proj.weight", (KDA_HEAD_DIM, HIDDEN))
            add_bf16(b + "self_attn.f_b_proj.weight", (KDA_INNER, KDA_HEAD_DIM))
            add_bf16(b + "self_attn.g_a_proj.weight", (KDA_HEAD_DIM, HIDDEN))
            add_bf16(b + "self_attn.g_b_proj.weight", (KDA_INNER, KDA_HEAD_DIM))
            add_bf16(b + "self_attn.b_proj.weight", (KDA_HEADS, HIDDEN))
            add_f32(b + "self_attn.A_log", (KDA_HEADS,))
            add_f32(b + "self_attn.dt_bias", (KDA_INNER,))
            add_bf16(b + "self_attn.o_norm.weight", (KDA_HEAD_DIM,))
            add_bf16(b + "self_attn.o_proj.weight", (HIDDEN, KDA_INNER))
        else:
            add_fp8(b + "self_attn.q_a_proj.weight", (Q_LORA, HIDDEN))
            add_bf16(b + "self_attn.q_a_layernorm.weight", (Q_LORA,))
            add_fp8(b + "self_attn.q_b_proj.weight", (N_HEADS * QK_NOPE, Q_LORA))
            add_fp8(b + "self_attn.kv_a_proj_with_mqa.weight", (KV_LORA, HIDDEN))
            add_bf16(b + "self_attn.kv_a_layernorm.weight", (KV_LORA,))
            add_bf16(b + "self_attn.kv_b_proj.weight",
                     (N_HEADS * (QK_NOPE + V_HEAD), KV_LORA))
            add_fp8(b + "self_attn.o_proj.weight", (HIDDEN, N_HEADS * V_HEAD))
            add_bf16(b + "self_attn.indexer.wq_b.weight",
                     (IDX_HEADS * IDX_HEAD_DIM, Q_LORA))
            add_bf16(b + "self_attn.indexer.wk.weight", (IDX_HEAD_DIM, HIDDEN))
            add_bf16(b + "self_attn.indexer.k_norm.weight", (IDX_HEAD_DIM,))
            add_bf16(b + "self_attn.indexer.k_norm.bias", (IDX_HEAD_DIM,))
            add_bf16(b + "self_attn.indexer.weights_proj.weight", (IDX_HEADS, HIDDEN))
            add_bf16(b + "self_attn.indexer.index_kpool_compress_ape",
                     (IDX_KPOOL, IDX_HEAD_DIM))
            add_bf16(b + "self_attn.indexer.index_kpool_compress_gate",
                     (IDX_HEAD_DIM, HIDDEN))
        sparse = True if is_mtp else (MLP_TYPES[L] == "sparse")
        if not sparse:
            add_fp8(b + "mlp.gate_proj.weight", (DENSE_INTER, HIDDEN))
            add_fp8(b + "mlp.up_proj.weight", (DENSE_INTER, HIDDEN))
            add_fp8(b + "mlp.down_proj.weight", (HIDDEN, DENSE_INTER))
        else:
            add_bf16(b + "mlp.gate.weight", (N_EXPERTS, HIDDEN))
            add_f32(b + "mlp.gate.e_score_correction_bias", (N_EXPERTS,))
            for e in range(N_EXPERTS):
                add_fp8(b + "mlp.experts.%d.gate_proj.weight" % e, (MOE_INTER, HIDDEN))
                add_fp8(b + "mlp.experts.%d.up_proj.weight" % e, (MOE_INTER, HIDDEN))
                add_fp8(b + "mlp.experts.%d.down_proj.weight" % e, (HIDDEN, MOE_INTER))
            add_fp8(b + "mlp.shared_experts.gate_proj.weight", (MOE_INTER, HIDDEN))
            add_fp8(b + "mlp.shared_experts.up_proj.weight", (MOE_INTER, HIDDEN))
            add_fp8(b + "mlp.shared_experts.down_proj.weight", (HIDDEN, MOE_INTER))
        if is_mtp:
            add_bf16(b + "enorm.weight", (HIDDEN,))
            add_bf16(b + "hnorm.weight", (HIDDEN,))
            add_bf16(b + "eh_proj.weight", (HIDDEN, 2 * HIDDEN))
            add_bf16(b + "shared_head.norm.weight", (HIDDEN,))

    v = "model.visual."
    add_bf16(v + "patch_embed.proj.weight", (VIS_HIDDEN, 3, 2, 14, 14))
    add_bf16(v + "patch_embed.proj.bias", (VIS_HIDDEN,))
    add_bf16(v + "post_layernorm.weight", (VIS_HIDDEN,))
    add_bf16(v + "downsample.weight", (HIDDEN, VIS_HIDDEN, 2, 2))
    add_bf16(v + "downsample.bias", (HIDDEN,))
    add_bf16(v + "merger.proj.weight", (HIDDEN, HIDDEN))
    add_bf16(v + "merger.gate_proj.weight", (512, HIDDEN))
    add_bf16(v + "merger.up_proj.weight", (512, HIDDEN))
    add_bf16(v + "merger.down_proj.weight", (HIDDEN, 512))
    add_bf16(v + "merger.post_projection_norm.weight", (HIDDEN,))
    add_bf16(v + "merger.post_projection_norm.bias", (HIDDEN,))
    for blk in range(VIS_DEPTH):
        vb = "%sblocks.%d." % (v, blk)
        add_bf16(vb + "norm1.weight", (VIS_HIDDEN,))
        add_bf16(vb + "norm2.weight", (VIS_HIDDEN,))
        add_bf16(vb + "attn.qkv.weight", (3 * VIS_HIDDEN, VIS_HIDDEN))
        add_bf16(vb + "attn.qkv.bias", (3 * VIS_HIDDEN,))
        add_bf16(vb + "attn.proj.weight", (VIS_HIDDEN, VIS_HIDDEN))
        add_bf16(vb + "attn.proj.bias", (VIS_HIDDEN,))
        add_bf16(vb + "attn.q_norm.weight", (VIS_HIDDEN // 2,))
        add_bf16(vb + "attn.k_norm.weight", (VIS_HIDDEN // 2,))
        add_bf16(vb + "mlp.gate_proj.weight", (VIS_INTER, VIS_HIDDEN))
        add_bf16(vb + "mlp.gate_proj.bias", (VIS_INTER,))
        add_bf16(vb + "mlp.up_proj.weight", (VIS_INTER, VIS_HIDDEN))
        add_bf16(vb + "mlp.up_proj.bias", (VIS_INTER,))
        add_bf16(vb + "mlp.down_proj.weight", (VIS_HIDDEN, VIS_INTER))
        add_bf16(vb + "mlp.down_proj.bias", (VIS_HIDDEN,))

    names = st_write(os.path.join(dirpath, "model.safetensors"), t)
    with open(os.path.join(dirpath, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {}, "weight_map":
                   {n: "model.safetensors" for n in names}}, fh)
    return expect


# ---------------------------------------------------------------------------
# An independent GGUF reader. No code is shared with the converter.

class Gguf:
    _SC = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
           5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
           12: ("<d", 8)}

    def __init__(self, path):
        self.buf = open(path, "rb").read()
        self.pos = 0
        assert self.raw(4) == b"GGUF", "not a GGUF file"
        self.version = self.u32()
        n_tensors = self.u64()
        n_kv = self.u64()
        self.kv = {}
        for _ in range(n_kv):
            key = self.string()
            self.kv[key] = self.value(self.u32())
        self.tensors = {}
        self.order = []
        for _ in range(n_tensors):
            name = self.string()
            nd = self.u32()
            dims = [self.u64() for _ in range(nd)]
            ttype = self.u32()
            offset = self.u64()
            self.tensors[name] = {"dims": dims, "type": ttype, "offset": offset}
            self.order.append(name)
        align = self.kv.get("general.alignment", 32)
        self.data = self.pos + ((-self.pos) % align)

    def raw(self, n):
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8")

    def value(self, vt):
        if vt == 8:
            return self.string()
        if vt == 9:
            et = self.u32()
            n = self.u64()
            return [self.value(et) for _ in range(n)]
        fmt, sz = self._SC[vt]
        return struct.unpack(fmt, self.raw(sz))[0]

    def blob(self, name, nbytes):
        o = self.data + self.tensors[name]["offset"]
        return self.buf[o:o + nbytes]


BLOCK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 14: (256, 210), 30: (1, 2)}
TNAME = {0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 14: "Q6_K", 30: "BF16"}


def nbytes_of(info):
    n = 1
    for d in info["dims"]:
        n *= d
    ne, bs = BLOCK[info["type"]]
    return (n // ne) * bs


def dequant(blob, ttype, n):
    """Independent decoder, transcribed from the ggml block layouts."""
    if ttype == 0:
        return np.frombuffer(blob, np.float32, n)
    if ttype == 1:
        return np.frombuffer(blob, np.float16, n).astype(np.float32)
    if ttype == 30:
        u = np.frombuffer(blob, np.uint16, n).astype(np.uint32)
        return (u << np.uint32(16)).view(np.float32)
    if ttype == 8:
        b = np.frombuffer(blob, np.uint8).reshape(-1, 34)
        d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
        q = b[:, 2:34].copy().view(np.int8).astype(np.float32)
        return (d * q).reshape(-1)[:n]
    if ttype == 10:
        b = np.frombuffer(blob, np.uint8).reshape(-1, 84)
        sc = b[:, 0:16]
        qs = b[:, 16:80]
        d = b[:, 80:82].copy().view(np.float16).astype(np.float32)
        dmin = b[:, 82:84].copy().view(np.float16).astype(np.float32)
        dl = (d * (sc & 0xF).astype(np.float32))
        ml = (dmin * (sc >> 4).astype(np.float32))
        shift = np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)
        q = (qs.reshape(-1, 2, 1, 32) >> shift) & np.uint8(3)
        q = q.reshape(-1, 16, 16).astype(np.float32)
        out = dl.reshape(-1, 16, 1) * q - ml.reshape(-1, 16, 1)
        return out.reshape(-1)[:n]
    if ttype == 14:
        # Transcribed from `dequantize_row_q6_K` (ggml-quants.c @ b10451): the
        # scale index inside a 128-chunk is `l/16` plus 0/2/4/6 for the four
        # quarters, and `sc` advances by 8 per chunk. Getting that stride wrong
        # decodes a plausible tensor with the wrong scales.
        b = np.frombuffer(blob, np.uint8).reshape(-1, 210)
        nb = b.shape[0]
        ql = b[:, 0:128].reshape(nb, 2, 64).astype(np.int32)
        qh = b[:, 128:192].reshape(nb, 2, 32).astype(np.int32)
        sc = b[:, 192:208].copy().view(np.int8).astype(np.float32).reshape(nb, 2, 8)
        d = b[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
        isx = (np.arange(32) // 16)
        out = np.zeros((nb, 256), np.float32)
        for half in range(2):
            lo = ql[:, half, 0:32]
            hi = ql[:, half, 32:64]
            h = qh[:, half, :]
            quarters = (
                ((lo & 0xF) | (((h >> 0) & 3) << 4), 0, 0),
                ((hi & 0xF) | (((h >> 2) & 3) << 4), 32, 2),
                ((lo >> 4) | (((h >> 4) & 3) << 4), 64, 4),
                ((hi >> 4) | (((h >> 6) & 3) << 4), 96, 6),
            )
            for q, dst, sofs in quarters:
                s_l = sc[:, half, :][:, isx + sofs]
                o = half * 128 + dst
                out[:, o:o + 32] = (q.astype(np.float32) - 32.0) * s_l
        return (out * d).reshape(-1)[:n]

    raise AssertionError("no decoder for ggml type %d" % ttype)


def run(args, cwd=None):
    return subprocess.run([sys.executable, CONVERTER] + args,
                          capture_output=True, text=True, cwd=cwd)


# ---------------------------------------------------------------------------

def case_kquant_golden():
    print("case: k-quant encoders are BYTE-IDENTICAL to llama.cpp b10451")
    if not os.path.exists(FIXTURE):
        check(False, "fixture %s is present" % FIXTURE)
        return
    fx = json.load(open(FIXTURE))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("conv", CONVERTER)
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    x = np.frombuffer(base64.b64decode(fx["input_f32_le_b64"]), np.float32)
    for key, fn, width in (("q2_k", conv.quantize_q2_k, 256),
                           ("q6_k", conv.quantize_q6_k, 256),
                           ("q8_0", conv.quantize_q8_0, 32)):
        want = np.frombuffer(base64.b64decode(fx[key + "_b64"]), np.uint8)
        got = fn(x.reshape(-1, width)).reshape(-1)
        check(want.shape == got.shape and bool((want == got).all()),
              "%s matches the pinned reference byte-for-byte (%d bytes)"
              % (key.upper(), want.size))
    check(fx["provenance"]["pin"] == "b10451",
          "the golden names the pin it was captured at (b10451), not `master`")


def case_convert(tmp):
    print("case: the converter emits a well-formed glm5next GGUF")
    src = os.path.join(tmp, "ckpt")
    os.makedirs(src, exist_ok=True)
    expect = build_checkpoint(src)
    dst = os.path.join(tmp, "out-q2k.gguf")
    r = run(["--src", src, "--dst", dst, "--arm", "q2_k"])
    check(r.returncode == 0, "q2_k conversion exits 0 (rc=%d)\n%s"
          % (r.returncode, r.stderr[-2000:] if r.returncode else ""))
    if r.returncode != 0:
        return None, expect
    g = Gguf(dst)
    check(g.version == 3, "GGUF version is 3")
    check(g.kv.get("general.architecture") == "glm5next",
          "general.architecture is `glm5next`")
    check(g.kv.get("glm5next.block_count") == N_LAYERS,
          "block_count is %d" % N_LAYERS)
    check(g.kv.get("general.file_type") == 10
          and g.kv.get("general.quantization_arm") == "q2_k",
          "general.file_type is the UINT32 `llama_ftype` MOSTLY_Q2_K = 10, not a "
          "string, and the arm name travels beside it")

    print("case: the metadata carries the parameters the port hinges on")
    check(abs(g.kv.get("glm5next.kda.gate_lower_bound", 0.0) - (-5.0)) < 1e-6,
          "`glm5next.kda.gate_lower_bound` is -5.0 -- the value that selects the "
          "sigmoid forget-gate branch over Kimi-Linear's softplus one")
    check(g.kv.get("glm5next.kda.head_dim") == KDA_HEAD_DIM,
          "`glm5next.kda.head_dim` is carried")
    check(g.kv.get("glm5next.ssm.conv_kernel") == CONV_K,
          "the short-conv kernel size is carried")
    check(g.kv.get("glm5next.attention.indexer.kpool") == IDX_KPOOL,
          "`indexer.kpool` is %d, the CHECKPOINT value, not the config class "
          "default of 16" % IDX_KPOOL)
    check(g.kv.get("glm5next.attention.indexer.top_k") == 2048,
          "`indexer.top_k` is 2048")
    check(g.kv.get("glm5next.hyper_connection.count") == HC_MULT,
          "mHC stream count is carried")
    check(abs(g.kv.get("glm5next.hyper_connection.epsilon", 0.0) - 1e-6) < 1e-12,
          "`hc_eps` is 1e-6 and is a DIFFERENT constant from rms_norm_eps 1e-5")
    check(abs(g.kv.get("glm5next.attention.layer_norm_rms_epsilon", 0.0) - 1e-5)
          < 1e-11, "`rms_norm_eps` is 1e-5")
    check(g.kv.get("glm5next.layer_types") == LAYER_TYPES,
          "the AUTHORITATIVE per-layer schedule is carried as `layer_types`")
    check("glm5next.rope.freq_base" not in g.kv
          and g.kv.get("glm5next.rope.dimension_count") == 0,
          "no rope frequency is written: the text stack is NoPE end to end")
    check(g.kv.get("glm5next.expert_count") == N_EXPERTS
          and g.kv.get("glm5next.expert_used_count") == 2,
          "MoE routing width is carried")
    check(abs(g.kv.get("glm5next.swiglu_clamp_exp", 0.0) - 10.0) < 1e-6,
          "the clamped-SwiGLU limit is carried")
    check(g.kv.get("glm5next.image_token_id") == 154854
          and g.kv.get("glm5next.video_token_id") == 154855,
          "both multimodal token ids travel together")
    return g, expect


def case_tensors(g):
    print("case: the tensor set is exactly what the topology generates")
    want = set()
    want |= {"token_embd.weight", "output_norm.weight", "output.weight"}
    for L in range(N_LAYERS):
        want |= {"blk.%d.attn_norm.weight" % L, "blk.%d.ffn_norm.weight" % L}
        for s in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                  "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            want.add("blk.%d.%s.weight" % (L, s))
        if LAYER_TYPES[L] == "linear_attention":
            for s in ("attn_q", "attn_k", "attn_v", "attn_output",
                      "ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v",
                      "ssm_f_a", "ssm_f_b", "ssm_g_a", "ssm_g_b",
                      "ssm_beta", "ssm_norm"):
                want.add("blk.%d.%s.weight" % (L, s))
            want |= {"blk.%d.ssm_a" % L, "blk.%d.ssm_dt" % L}
        else:
            for s in ("attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv_a_mqa",
                      "attn_kv_a_norm", "attn_kv_b", "attn_output",
                      "indexer.attn_q_b", "indexer.attn_k", "indexer.k_norm",
                      "indexer.proj", "indexer_compressor_ape",
                      "indexer_compressor_gate"):
                want.add("blk.%d.%s.weight" % (L, s))
            want.add("blk.%d.indexer.k_norm.bias" % L)
        if MLP_TYPES[L] == "dense":
            for s in ("ffn_gate", "ffn_up", "ffn_down"):
                want.add("blk.%d.%s.weight" % (L, s))
        else:
            for s in ("ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps",
                      "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp",
                      "ffn_down_shexp"):
                want.add("blk.%d.%s.weight" % (L, s))
            want.add("blk.%d.exp_probs_b.bias" % L)
    for s in ("v.patch_embd.weight", "v.patch_embd.bias", "v.post_ln.weight",
              "v.downsample.weight", "v.downsample.bias", "v.merger.proj.weight",
              "v.merger.gate.weight", "v.merger.up.weight", "v.merger.down.weight",
              "v.merger.norm.weight", "v.merger.norm.bias"):
        want.add(s)
    for b in range(VIS_DEPTH):
        for s in ("ln1.weight", "ln2.weight", "attn_qkv.weight", "attn_qkv.bias",
                  "attn_out.weight", "attn_out.bias", "attn_q_norm.weight",
                  "attn_k_norm.weight", "ffn_gate.weight", "ffn_gate.bias",
                  "ffn_up.weight", "ffn_up.bias", "ffn_down.weight",
                  "ffn_down.bias"):
            want.add("v.blk.%d.%s" % (b, s))
    got = set(g.tensors)
    check(got == want, "tensor set matches (%d expected). missing=%s extra=%s"
          % (len(want), sorted(want - got)[:6], sorted(got - want)[:6]))

    print("case: the MTP block is dropped, following the reference")
    mtp = [n for n in got if n.startswith("blk.%d." % N_LAYERS)]
    check(not mtp, "no `blk.%d.*` tensor survives (found %s)" % (N_LAYERS, mtp[:4]))
    check(not any("shared_head" in n or "enorm" in n or "eh_proj" in n for n in got),
          "no MTP-only tensor (`enorm`, `hnorm`, `eh_proj`, `shared_head`) survives")

    print("case: the 288-expert lane is STACKED, not one tensor per expert")
    for L in range(N_LAYERS):
        if MLP_TYPES[L] != "sparse":
            continue
        info = g.tensors["blk.%d.ffn_gate_exps.weight" % L]
        # ggml dims are reversed against the logical shape.
        check(info["dims"] == [HIDDEN, MOE_INTER, N_EXPERTS],
              "layer %d ffn_gate_exps is [ne0=%d, ne1=%d, ne2=%d experts]"
              % (L, HIDDEN, MOE_INTER, N_EXPERTS))
        break
    check(not any(".experts." in n for n in got),
          "no per-expert tensor name leaks into the file")


def case_types(g):
    print("case: the q2_k arm puts the EXPERTS at Q2_K and the rest at Q6_K")
    exps = [n for n in g.tensors if n.endswith("_exps.weight")]
    check(exps and all(g.tensors[n]["type"] == 10 for n in exps),
          "all %d stacked expert tensors are Q2_K" % len(exps))
    check(g.tensors["blk.0.attn_q.weight"]["type"] == 14,
          "a non-expert 2-D weight is Q6_K")
    check(g.tensors["blk.0.attn_norm.weight"]["type"] == 0,
          "a 1-D norm stays F32 rather than being quantized")
    check(g.tensors["blk.0.ssm_a"]["type"] == 0, "`ssm_a` (A_log) stays F32")
    print("case: a row that does not divide the block STEPS DOWN rather than "
          "being reshaped")
    check(g.tensors["blk.2.indexer_compressor_ape.weight"]["type"] == 8,
          "the k-pool positional embedding (ne0=%d, not a multiple of 256) steps "
          "down to Q8_0, whose block is 32" % IDX_HEAD_DIM)
    check(g.tensors["blk.0.ssm_conv1d_q.weight"]["type"] == 0,
          "the depthwise conv kernel (ne0=%d divides neither 256 nor 32) falls "
          "all the way to F32" % CONV_K)
    print("case: a convolution kernel with more axes than ggml carries is "
          "flattened, not written")
    # ggml has GGML_MAX_DIMS = 4. The patch embed is a Conv3d at
    # `[hidden, 3, 2, 14, 14]` -- five axes -- and writing it verbatim would
    # produce a header no reader can index. It is flattened to
    # `[hidden, 3*2*14*14]`, whose row (1176) divides neither 256 nor 32, so it
    # lands at F32.
    check(len(g.tensors["v.patch_embd.weight"]["dims"]) == 2
          and g.tensors["v.patch_embd.weight"]["dims"] == [3 * 2 * 14 * 14,
                                                           VIS_HIDDEN],
          "the 5-D patch-embed kernel is flattened to [%d, %d] rather than "
          "written with five axes" % (3 * 2 * 14 * 14, VIS_HIDDEN))
    check(g.tensors["v.patch_embd.weight"]["type"] == 0,
          "the flattened patch-embed row (1176) divides neither 256 nor 32, so "
          "it lands at F32")
    check(g.tensors["v.downsample.weight"]["dims"] == [VIS_HIDDEN * 2 * 2, HIDDEN]
          and g.tensors["v.downsample.weight"]["type"] == 14,
          "the 4-D downsample kernel is flattened to a row that DOES divide 256 "
          "and is quantized rather than dumped at F32")
    check(all(len(i["dims"]) <= 4 for i in g.tensors.values()),
          "no tensor in the file carries more than ggml's 4 axes")
    check(g.tensors["blk.0.ssm_conv1d_q.weight"]["dims"] == [CONV_K, 1, KDA_INNER]
          and g.tensors["blk.1.ffn_gate_exps.weight"]["dims"]
          == [HIDDEN, MOE_INTER, N_EXPERTS],
          "3-D shapes are NOT flattened: the depthwise conv stays [k, 1, ch] and "
          "the expert lane stays [m, n, experts], because ggml indexes both")


def case_values(g, expect):
    print("case: FP8 e4m3 block dequant and bf16 widening reach the file intact")
    p = "model.language_model."
    name = "blk.0.attn_norm.weight"
    ref = expect[p + "layers.0.input_layernorm.weight"].reshape(-1)
    got = dequant(g.blob(name, nbytes_of(g.tensors[name])), 0, ref.size)
    check(np.array_equal(got, ref), "an F32-carried bf16 norm is bit-exact")

    name = "blk.2.attn_q_a.weight"
    info = g.tensors[name]
    ref = expect[p + "layers.2.self_attn.q_a_proj.weight"].reshape(-1)
    got = dequant(g.blob(name, nbytes_of(info)), info["type"], ref.size)
    rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
    check(rel < 0.02, "an FP8 e4m3 block-quantized MLA projection round-trips "
                      "through Q6_K within 2%% (max rel %.4f)" % rel)

    name = "blk.1.ffn_gate_exps.weight"
    info = g.tensors[name]
    ref = np.concatenate([
        expect[p + "layers.1.mlp.experts.%d.gate_proj.weight" % e].reshape(-1)
        for e in range(N_EXPERTS)])
    got = dequant(g.blob(name, nbytes_of(info)), info["type"], ref.size)
    rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
    check(rel < 0.60, "the stacked expert lane round-trips through Q2_K "
                      "(max rel %.4f)" % rel)
    # Expert ORDER, checked slab by slab rather than by a correlation over the
    # whole tensor: a correlation stays high under a permutation of the slabs,
    # and it cannot see a scale error at all. Every slab must be nearer its own
    # expert than any other, which a swap breaks and a correlation does not.
    per = ref.size // N_EXPERTS
    refs = [expect[p + "layers.1.mlp.experts.%d.gate_proj.weight" % e].reshape(-1)
            for e in range(N_EXPERTS)]
    order_ok = True
    for e in range(N_EXPERTS):
        slab = got[e * per:(e + 1) * per]
        errs = [float(np.abs(slab - r).mean()) for r in refs]
        if int(np.argmin(errs)) != e:
            order_ok = False
    check(order_ok, "every expert slab is nearest its OWN expert: the lane is in "
                    "expert order, not permuted")


def case_refusals(tmp):
    print("case: an unimplemented arm is REFUSED by name")
    src = os.path.join(tmp, "ckpt")
    for arm in ("iq2_xxs", "iq1_s", "iq2_s"):
        r = run(["--src", src, "--dst", os.path.join(tmp, "x.gguf"), "--arm", arm])
        ok = (r.returncode != 0 and arm.upper() in r.stderr
              and "importance matrix" in r.stderr and "circular" in r.stderr)
        check(ok, "--arm %s is refused and names the missing part (imatrix) "
                  "and why it is unreachable here" % arm)
    for arm in ("q4_k", "q3_k"):
        r = run(["--src", src, "--dst", os.path.join(tmp, "x.gguf"), "--arm", arm])
        check(r.returncode != 0 and arm.upper() in r.stderr
              and "encoder is not ported" in r.stderr,
              "--arm %s is refused and names the unported encoder" % arm)
    r = run(["--src", src, "--dst", os.path.join(tmp, "x.gguf"),
             "--arm", "q2_k", "--keep-mtp"])
    check(r.returncode != 0 and "MTP" in r.stderr.upper()
          and "O2" in r.stderr,
          "--keep-mtp is refused and names the owed item that would carry it")

    print("case: a checkpoint of another architecture is REFUSED, not converted")
    other = os.path.join(tmp, "other")
    os.makedirs(other, exist_ok=True)
    cfg = tiny_config()
    cfg["model_type"] = "glm4_moe"
    cfg["architectures"] = ["Glm4MoeForCausalLM"]
    with open(os.path.join(other, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    r = run(["--src", other, "--dst", os.path.join(tmp, "x.gguf"), "--arm", "q2_k"])
    check(r.returncode != 0 and "glm5_next" in r.stderr,
          "a non-glm5_next config is refused by name")

    print("case: an FP8 tensor with no scale companion is REFUSED, not guessed")
    broken = os.path.join(tmp, "broken")
    os.makedirs(broken, exist_ok=True)
    build_checkpoint(broken)
    hdr_path = os.path.join(broken, "model.safetensors")
    with open(hdr_path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
        rest = fh.read()
    victim = ("model.language_model.layers.1.mlp.experts.0."
              "gate_proj.weight_scale_inv")
    if victim in head:
        del head[victim]
        hb = json.dumps(head).encode()
        hb += b" " * ((-len(hb)) % 8)
        # Offsets stay valid; only the entry disappears, which is exactly the
        # shape of a checkpoint published with a missing companion.
        with open(hdr_path, "wb") as fh:
            fh.write(struct.pack("<Q", len(hb)))
            fh.write(hb)
            fh.write(rest)
        idx = os.path.join(broken, "model.safetensors.index.json")
        with open(idx) as fh:
            j = json.load(fh)
        j["weight_map"].pop(victim, None)
        with open(idx, "w") as fh:
            json.dump(j, fh)
        r = run(["--src", broken, "--dst", os.path.join(tmp, "b.gguf"),
                 "--arm", "q8_0"])
        check(r.returncode != 0 and "weight_scale_inv" in r.stderr,
              "a missing `weight_scale_inv` is refused rather than treated as 1.0")
    else:
        check(False, "the synthetic checkpoint carries the scale companion")


def case_dry_run(tmp):
    print("case: --dry-run reports the arm's size without writing a byte")
    src = os.path.join(tmp, "ckpt")
    sizes = {}
    for arm in ("q2_k", "q6_k", "q8_0"):
        r = run(["--src", src, "--arm", arm, "--dry-run"])
        check(r.returncode == 0, "--dry-run --arm %s exits 0" % arm)
        if r.returncode:
            continue
        tot = 0
        for line in r.stdout.strip().split("\n"):
            tot += int(line.split("\t")[3])
        sizes[arm] = tot
    if len(sizes) == 3:
        check(sizes["q2_k"] < sizes["q6_k"] < sizes["q8_0"],
              "the arms order by size q2_k < q6_k < q8_0 (%d < %d < %d)"
              % (sizes["q2_k"], sizes["q6_k"], sizes["q8_0"]))
    r = run(["--src", src, "--arm", "q2_k", "--dry-run"])
    check("out.gguf" not in os.listdir(src),
          "--dry-run writes nothing into the source directory")



# ---------------------------------------------------------------------------
# The name map is a CONTRACT BETWEEN TWO FILES, and both are ours.
#
# No upstream tool writes this container, so nothing external pins the tensor
# spellings: `scripts/convert-glm5-next-gguf.py` writes them and
# `src/vllm/model_executor/models/glm5_next_weights.cpp` reads and enumerates
# them (W1, #2067). Two hand-maintained copies of one mapping is the shape that
# drifts, and the drift would be silent -- a renamed tensor produces a file the
# reader simply does not find, on a model whose only artifact nobody has
# produced yet, so no gate downstream would ever notice.
#
# The C++ side is parsed as TEXT rather than compiled, deliberately: this suite
# needs no C++ build (that is what lets it run wherever the converter runs), and
# the property being checked is that the two literal tables are equal.

CPP_WEIGHTS = os.path.join(
    ROOT, "src", "vllm", "model_executor", "models", "glm5_next_weights.cpp")

CPP_HEADER = os.path.join(
    ROOT, "include", "vllm", "model_executor", "models", "glm5_next_weights.h")


def _cpp_map(text, fn):
    """The `{"hf", "gguf"}` pairs inside one map function's body."""
    start = text.index("std::vector<Glm5NextTensorName> " + fn + "() {")
    end = text.index("\n}\n", start)
    return dict(re.findall(r'\{"([^"]+)",\s*\n?\s*"([^"]+)"\}',
                           text[start:end]))


def case_namemap():
    print("\ncase: the C++ name map equals the converter's, table for table")
    if not os.path.exists(CPP_WEIGHTS):
        check(False, "src/.../glm5_next_weights.cpp is absent")
        return
    text = open(CPP_WEIGHTS).read()
    conv = _load_converter_module()
    if conv is None:
        check(False, "the converter module could not be imported")
        return

    for fn, py in (
            ("Glm5NextCommonTensorMap", conv.COMMON_MAP),
            ("Glm5NextKdaTensorMap", conv.KDA_MAP),
            ("Glm5NextDsaTensorMap", conv.DSA_MAP),
            ("Glm5NextDenseMlpTensorMap", conv.DENSE_MLP_MAP),
            ("Glm5NextSparseMlpTensorMap", conv.SPARSE_MLP_MAP),
            ("Glm5NextVisionTensorMap", conv.VISION_MAP),
            ("Glm5NextVisionBlockTensorMap", conv.VISION_BLOCK_MAP)):
        try:
            cpp = _cpp_map(text, fn)
        except ValueError:
            check(False, "%s: no such map function in the C++ TU" % fn)
            continue
        # Non-empty is asserted separately: a regex that silently matched
        # nothing would make every comparison below trivially true, which is
        # the mute-switch shape this whole case exists to prevent.
        check(len(cpp) > 0, "%s: parsed %d entries from the C++ TU"
              % (fn, len(cpp)))
        if cpp == py:
            check(True, "%s: %d entries identical to the converter's"
                  % (fn, len(py)))
        else:
            only_cpp = sorted(k for k in cpp if py.get(k) != cpp[k])
            only_py = sorted(k for k in py if cpp.get(k) != py[k])
            check(False, "%s: DRIFT -- differs on %s (C++) / %s (converter)"
                  % (fn, only_cpp, only_py))

    # The experts are stacked, so the converter states them per PROJECTION while
    # the C++ states the HF path with an `{e}` placeholder. Same three tensors.
    try:
        cpp = _cpp_map(text, "Glm5NextStackedExpertTensorMap")
    except ValueError:
        check(False, "Glm5NextStackedExpertTensorMap: no such map function")
        return
    want = dict(("mlp.experts.{e}.%s.weight" % k, v)
                for k, v in conv.EXPERT_STACK.items())
    check(cpp == want,
          "Glm5NextStackedExpertTensorMap: %d stacked expert tensors identical"
          % len(want))

    # And the architecture key itself, which is what the dispatch table matches
    # on. One spelling in two files.
    header = open(CPP_HEADER).read() if os.path.exists(CPP_HEADER) else ""
    m = re.search(r'kGlm5NextGgufArch\s*=\s*"([^"]+)"', header)
    check(m is not None and m.group(1) == conv.ARCH,
          "kGlm5NextGgufArch == ARCH == %r" % conv.ARCH)


def _load_converter_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "glm5_next_converter", CONVERTER)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [CONVERTER]
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - reported as a failure
        print("  FAIL importing the converter: %s" % exc)
        FAILURES.append("converter import")
        return None
    finally:
        sys.argv = argv
    return mod


def main():
    if not os.path.exists(CONVERTER):
        print("  FAIL scripts/convert-glm5-next-gguf.py is absent")
        FAILURES.append("converter missing")
        print("\n%d failure(s)" % len(FAILURES))
        return 1
    case_kquant_golden()
    case_namemap()
    with tempfile.TemporaryDirectory() as tmp:
        g, expect = case_convert(tmp)
        if g is not None:
            case_tensors(g)
            case_types(g)
            case_values(g, expect)
        case_refusals(tmp)
        case_dry_run(tmp)
    if FAILURES:
        print("\n%d failure(s):" % len(FAILURES))
        for f in FAILURES:
            print("  - %s" % f)
        return 1
    print("\nAll cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
