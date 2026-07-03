# vllm.cpp parity harness; oracle = PINNED upstream vLLM checkout (e24d1b24).
"""Dump GDN-layer op goldens from the pinned vLLM checkout (M0.7 Task 1).

Runs on the dgx GPU inside the oracle venv, executing the PINNED sources
verbatim (never pip vLLM, never re-implemented math):

    ssh dgx.casa 'cd ~/work/vllm.cpp && ~/venvs/vllm-oracle/bin/python \
        tools/parity/dump_gdn.py --pin ~/work/vllm-pin \
        --out tests/parity/goldens'

Loading technique (oracle_detokenizer.py's exec-stub, extended to packages):
we register package skeletons for `vllm`, `vllm.model_executor...` etc. whose
__path__ points into the pinned tree WITHOUT executing the heavy package
__init__.py files, then import the pinned op modules normally. Only leaf
infra is stubbed (recorded in `STUBBED` below and in every manifest):
  - vllm.triton_utils  -> re-export of the venv's real triton/tl
  - vllm.platforms     -> current_platform facade backed by torch.cuda
  - vllm.v1.attention.backends.utils -> constants PAD_SLOT_ID=-1,
    NULL_BLOCK_ID=0 (pinned v1/attention/backends/utils.py:45-46)
  - vllm.utils.math_utils / platform_utils -> cdiv/next_power_of_2 and
    num_compute_units (infra helpers used only for triton launch configs)
All oracle MATH executes from the pinned files.

Case-size policy (plan M0.7 §Task 1): real-dims cases keep Dk=Dv=128 (dims
drive the math) but slice heads to Hk=1/Hv=2 — GQA ratio 2 preserved, heads
only replicate per-head math. Synthetic-small cases use Hk=2/Hv=4 with small
dims. SSM states are dumped as gathered per-sequence f32 slices. Total budget
<= 2MB.

Oracle callables per op:
  causal_conv1d_fwd    -> mamba/ops/causal_conv1d.py::causal_conv1d_fn
  causal_conv1d_update -> mamba/ops/causal_conv1d.py::causal_conv1d_update
  l2norm               -> fla/ops/l2norm.py::l2norm_fwd
  rmsnorm_gated        -> fla/ops/layernorm_guard.py::rmsnorm_fn (the
                          RMSNormGated.forward_cuda path; cross-checked at
                          dump time against pinned rms_norm_ref for silu)
  gdn_prefill (bf16)   -> fla/ops/chunk.py::chunk_gated_delta_rule
  gdn_prefill (f32)    -> fla/ops/fused_recurrent.py::
                          fused_recurrent_gated_delta_rule  (the pinned
                          sequential kernel; the chunk wrapper hard-rejects
                          f32 at chunk.py:213 — deviation recorded in the
                          manifests; sequential-vs-chunked gap measured on
                          the bf16 case)
  gdn_decode           -> fla/ops/fused_sigmoid_gating.py::
                          fused_sigmoid_gating_delta_rule_update (what
                          qwen_gdn_linear_attn.py:1540-1559 calls for decode)
  (decode g/beta/q_l2/k_l2 derivation for the decomposed C++ chain)
                       -> fla/ops/fused_gdn_prefill_post_conv.py::
                          fused_post_conv_prep
"""

import argparse
import importlib
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import torch

UPSTREAM_PIN = "e24d1b24"
STUBBED = [
    "vllm.triton_utils",
    "vllm.platforms",
    "vllm.v1.attention.backends.utils (PAD_SLOT_ID/NULL_BLOCK_ID consts)",
    "vllm.utils.math_utils",
    "vllm.utils.platform_utils",
]

TIGHT = {"atol": 1e-5, "rtol": 1e-5}
BF16_CHUNK = {"atol": 1.5e-2, "rtol": 1.5e-2}  # sequential C++ vs chunked bf16

DTYPE_NAMES = {torch.float32: "f32", torch.bfloat16: "bf16",
               torch.float16: "f16", torch.int32: "i32", torch.int64: "i64",
               torch.bool: "i32"}


# ---------------------------------------------------------------- loading

def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m
    return m


def load_pinned(pin_root: pathlib.Path):
    """Import the pinned GDN op modules with leaf-infra stubs."""
    v = pin_root / "vllm"
    assert (v / "model_executor/layers/fla/ops/chunk.py").exists(), pin_root

    import triton
    import triton.language as tl
    tldevice = getattr(tl.extra, "libdevice", None) or getattr(
        tl.extra, "cuda", None)  # only touched if FLA_USE_FAST_OPS=1
    _mod("vllm.triton_utils", tl=tl, triton=triton, tldevice=tldevice)

    class _Platform:
        def is_cuda(self):
            return True

        def is_cuda_alike(self):
            return True

        def is_rocm(self):
            return False

        def is_xpu(self):
            return False

        def is_cpu(self):
            return False

        def is_amd(self):
            return False

        def get_device_capability(self, device_id=0):
            cap = torch.cuda.get_device_capability(device_id)
            return types.SimpleNamespace(major=cap[0], minor=cap[1],
                                         to_int=lambda: cap[0] * 10 + cap[1])

        def is_device_capability(self, cap, device_id=0):
            c = torch.cuda.get_device_capability(device_id)
            return c[0] * 10 + c[1] == cap

        def is_device_capability_family(self, cap, device_id=0):
            c = torch.cuda.get_device_capability(device_id)
            return (c[0] * 10 // 10) * 10 == cap // 10 * 10 and \
                c[0] * 10 + c[1] >= cap

        def get_device_name(self, device_id=0):
            return torch.cuda.get_device_name(device_id)

        def current_device(self):
            return torch.cuda.current_device()

    _mod("vllm.platforms", current_platform=_Platform())

    # Constants from pinned v1/attention/backends/utils.py:45-46.
    _mod("vllm.v1.attention.backends.utils", PAD_SLOT_ID=-1, NULL_BLOCK_ID=0)

    def cdiv(a, b):
        return -(a // -b)

    def next_power_of_2(n):
        return 1 if n <= 1 else 2 ** (int(n - 1).bit_length())

    _mod("vllm.utils.math_utils", cdiv=cdiv, next_power_of_2=next_power_of_2)
    _mod("vllm.utils.platform_utils",
         num_compute_units=lambda idx=None: torch.cuda.get_device_properties(
             idx or 0).multi_processor_count)

    # Package skeletons pointing into the pinned tree; parents' __init__.py
    # are NOT executed (they drag the whole engine in).
    _pkg("vllm", v)
    _pkg("vllm.model_executor", v / "model_executor")
    _pkg("vllm.model_executor.layers", v / "model_executor/layers")
    _pkg("vllm.model_executor.layers.fla", v / "model_executor/layers/fla")
    _pkg("vllm.model_executor.layers.fla.ops",
         v / "model_executor/layers/fla/ops")
    _pkg("vllm.model_executor.layers.mamba", v / "model_executor/layers/mamba")
    _pkg("vllm.model_executor.layers.mamba.ops",
         v / "model_executor/layers/mamba/ops")
    # NB: vllm.utils / vllm.v1... already stubbed above as plain modules.

    ops = types.SimpleNamespace()
    ops.l2norm = importlib.import_module(
        "vllm.model_executor.layers.fla.ops.l2norm")
    ops.chunk = importlib.import_module(
        "vllm.model_executor.layers.fla.ops.chunk")
    ops.fused_recurrent = importlib.import_module(
        "vllm.model_executor.layers.fla.ops.fused_recurrent")
    ops.fused_sigmoid_gating = importlib.import_module(
        "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating")
    ops.post_conv = importlib.import_module(
        "vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv")
    ops.layernorm_guard = importlib.import_module(
        "vllm.model_executor.layers.fla.ops.layernorm_guard")
    ops.causal_conv1d = importlib.import_module(
        "vllm.model_executor.layers.mamba.ops.causal_conv1d")

    got = subprocess.run(["git", "-C", str(pin_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if not got:  # rsync'd tree without .git: PIN stamp written at sync time
        got = (pin_root / "PIN").read_text().strip()
    print(f"pinned checkout: {pin_root} @ {got}")
    assert got.startswith(UPSTREAM_PIN), f"pin mismatch: {got}"
    for m in vars(ops).values():
        print(f"  exec'd {m.__file__}")
    return ops


# ---------------------------------------------------------------- saving

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach()
    if t.dtype == torch.bool:
        return t.to(torch.int32).cpu().numpy()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.contiguous().view(torch.uint16).cpu().numpy()
    return t.contiguous().cpu().numpy()


def save_case(root, name, op, callable_name, tensors, args, inputs, outputs,
              tol):
    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    manifest = {
        "op": op, "args": args, "tensors": {}, "inputs": inputs,
        "outputs": outputs, "tol": tol,
        "oracle": {
            "source": f"pinned:{UPSTREAM_PIN}",
            "callable": callable_name,
            "torch": torch.__version__.split("+")[0],
            "device": torch.cuda.get_device_name(0),
            "stubbed_infra": STUBBED,
        },
    }
    total = 0
    for key, t in tensors.items():
        arr = _to_numpy(t)
        np.save(case / f"{key}.npy", arr)
        total += arr.nbytes
        manifest["tensors"][key] = {"file": f"{key}.npy",
                                    "dtype": DTYPE_NAMES[t.dtype],
                                    "shape": list(t.shape)}
    (case / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {case}  ({total / 1024:.1f} KiB)")
    return total


# ------------------------------------------------------------- generators
# Input GENERATION is free-form (only outputs must come from the oracle);
# we mimic the upstream docstring recipe (chunk.py:184-200): q/k l2norm'd,
# beta = sigmoid, g = logsigmoid-ish negative log-decay.

def gen_qkvgb(T, Hk, Hv, Dk, Dv, dtype, dev, l2, seed_off=0):
    torch.manual_seed(seed_off)
    q = torch.randn(T, Hk, Dk, dtype=dtype, device=dev)
    k = torch.randn(T, Hk, Dk, dtype=dtype, device=dev)
    v = torch.randn(T, Hv, Dv, dtype=dtype, device=dev)
    if l2:
        q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(dtype)
        k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(dtype)
    g = torch.nn.functional.logsigmoid(
        torch.rand(T, Hv, dtype=torch.float32, device=dev))
    beta = torch.rand(T, Hv, dtype=torch.float32, device=dev).sigmoid()
    return q, k, v, g, beta


# ------------------------------------------------------------------ dumps

def dump_conv_fwd(ops, root, dev):
    total = 0
    cc = ops.causal_conv1d
    for name, dim, K, lens, tol in (
            ("causal_conv1d_fwd_f32_small", 16, 4, [5, 7], TIGHT),
            ("causal_conv1d_fwd_f32_realdims", 2 * 1 * 128 + 2 * 128, 4,
             [9, 7], TIGHT)):
        torch.manual_seed(0)
        N = len(lens)
        T = sum(lens)
        x = torch.randn(T, dim, device=dev)  # token-major like mixed_qkv
        weight = torch.randn(dim, K, device=dev) * 0.3
        cache_lines = N + 1  # line 0 = NULL block, unused
        conv_state = torch.randn(cache_lines, dim, K - 1, device=dev)
        conv_state[0].zero_()
        conv_state_in = conv_state.clone()
        qsl = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32,
                           device=dev)
        cache_indices = torch.arange(1, N + 1, dtype=torch.int32, device=dev)
        # seq 0 has an initial state, seq 1 does not (upstream zeroes nothing
        # for conv: the kernel just ignores the state when has_init=False).
        has_init = torch.tensor([True, False], device=dev)
        out = cc.causal_conv1d_fn(
            x.transpose(0, 1),  # (dim, cu_seqlen) channel-last
            weight, None, conv_state, qsl,
            cache_indices=cache_indices, has_initial_state=has_init,
            activation="silu", validate_data=True,
        ).transpose(0, 1)
        total += save_case(
            root, name, "causal_conv1d_fwd",
            "mamba/ops/causal_conv1d.py::causal_conv1d_fn",
            {"x": x, "weight": weight,
             "conv_state_in": conv_state_in[1:],  # per-seq slices (skip NULL)
             "query_start_loc": qsl,
             "has_initial_state": has_init,
             "out": out, "conv_state_out": conv_state[1:]},
            {"activation": "silu", "bias": None, "dim": dim, "width": K,
             "seqlens": lens,
             "note": "x dumped token-major [T,dim]; oracle ran on "
                     "x.transpose(0,1) channel-last. conv_state slices are "
                     "cache lines 1..N (line 0 = NULL block)."},
            ["x", "weight", "conv_state_in", "query_start_loc",
             "has_initial_state"],
            ["out", "conv_state_out"], tol)
    return total


def dump_conv_update(ops, root, dev):
    total = 0
    cc = ops.causal_conv1d
    for name, dim, K, B, tol in (
            ("causal_conv1d_update_f32_small", 16, 4, 3, TIGHT),
            ("causal_conv1d_update_f32_realdims", 2 * 1 * 128 + 2 * 128, 4, 3,
             TIGHT)):
        torch.manual_seed(1)
        x = torch.randn(B, dim, device=dev)
        weight = torch.randn(dim, K, device=dev) * 0.3
        conv_state = torch.randn(B + 1, dim, K - 1, device=dev)
        conv_state[0].zero_()
        conv_state_in = conv_state.clone()
        idx = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
        out = cc.causal_conv1d_update(
            x.clone(), conv_state, weight, None, "silu",
            conv_state_indices=idx, validate_data=True)
        total += save_case(
            root, name, "causal_conv1d_update",
            "mamba/ops/causal_conv1d.py::causal_conv1d_update",
            {"x": x, "weight": weight, "conv_state_in": conv_state_in[1:],
             "out": out, "conv_state_out": conv_state[1:]},
            {"activation": "silu", "bias": None, "dim": dim, "width": K,
             "batch": B,
             "note": "conv_state slices are cache lines 1..B "
                     "(line 0 = NULL block)."},
            ["x", "weight", "conv_state_in"],
            ["out", "conv_state_out"], tol)
    return total


def dump_l2norm(ops, root, dev):
    total = 0
    for name, T, D in (("l2norm_f32_small", 12, 6),
                       ("l2norm_f32_realdims", 16, 128)):
        torch.manual_seed(2)
        x = torch.randn(T, D, device=dev)
        x[0].mul_(1e-3)  # exercise the eps
        out = ops.l2norm.l2norm_fwd(x)
        total += save_case(
            root, name, "l2norm", "fla/ops/l2norm.py::l2norm_fwd",
            {"x": x, "out": out}, {"eps": 1e-6},
            ["x"], ["out"], TIGHT)
    return total


def dump_rmsnorm_gated(ops, root, dev):
    total = 0
    lg = ops.layernorm_guard
    for name, T, D, act in (
            ("rmsnorm_gated_f32_small_silu", 8, 32, "silu"),
            ("rmsnorm_gated_f32_realdims_silu", 16, 128, "silu"),
            ("rmsnorm_gated_f32_realdims_sigmoid", 16, 128, "sigmoid")):
        torch.manual_seed(3)
        x = torch.randn(T, D, device=dev)
        z = torch.randn(T, D, device=dev)
        w = torch.randn(D, device=dev) * 0.5 + 1.0
        out = lg.rmsnorm_fn(x, w, None, z=z, eps=1e-6, group_size=None,
                            norm_before_gate=True, activation=act)
        if act == "silu":  # cross-check vs the pinned torch-native reference
            ref = lg.rms_norm_ref(x, w, None, z=z, eps=1e-6, group_size=None,
                                  norm_before_gate=True)
            gap = (out - ref).abs().max().item()
            print(f"  rmsnorm_fn vs rms_norm_ref max|diff| = {gap:.3e}")
            assert gap < 1e-5
        total += save_case(
            root, name, "rmsnorm_gated",
            "fla/ops/layernorm_guard.py::rmsnorm_fn",
            {"x": x, "gate": z, "weight": w, "out": out},
            {"eps": 1e-6, "norm_before_gate": True, "activation": act,
             "group_size": None},
            ["x", "gate", "weight"], ["out"], TIGHT)
    return total


def run_sequential(ops, q, k, v, g, beta, scale, h0, cu):
    """Run the pinned sequential kernel over varlen multi-token sequences.

    fused_recurrent's INPLACE_FINAL_STATE stores the running state at
    ssm_state_indices[i_n*stride_seq + i_t] per TOKEN (fused_recurrent.py:
    153-161 — the spec-decode layout), so we pass a 2D [N, Tmax] index
    tensor whose row i repeats sequence i's cache line: every step
    overwrites the same line and the final store leaves the final state.
    """
    N = cu.numel() - 1
    Tmax = int((cu[1:] - cu[:-1]).max().item())
    cache = torch.zeros(N + 1, *h0.shape[1:], device=q.device,
                        dtype=h0.dtype)
    cache[1:] = h0
    idx2d = (torch.arange(1, N + 1, dtype=torch.int32, device=q.device)
             [:, None].expand(N, Tmax).contiguous())
    o, _ = ops.fused_recurrent.fused_recurrent_gated_delta_rule(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        g=g.unsqueeze(0), beta=beta.unsqueeze(0), scale=scale,
        initial_state=cache, inplace_final_state=True, cu_seqlens=cu,
        ssm_state_indices=idx2d, use_qk_l2norm_in_kernel=False)
    return o.squeeze(0), cache[1:].clone()


def dump_gdn_prefill(ops, root, dev):
    total = 0

    # --- f32 cases: pinned sequential kernel (chunk wrapper rejects f32,
    #     chunk.py:213). This is the tight-tolerance target for the C++
    #     sequential implementation.
    f32_note = ("oracle is the pinned sequential kernel "
                "fused_recurrent_gated_delta_rule: the pinned "
                "chunk_gated_delta_rule wrapper asserts q.dtype != float32 "
                "(chunk.py:213), so f32 goldens cannot come from the chunked "
                "path. bf16 cases below DO use chunk_gated_delta_rule.")
    for name, Hk, Hv, Dk, Dv, lens, with_init in (
            ("gdn_prefill_f32_small", 2, 4, 16, 32, [5, 7], True),
            ("gdn_prefill_f32_small_noinit", 2, 4, 16, 32, [12], False),
            ("gdn_prefill_f32_realdims", 1, 2, 128, 128, [24], True)):
        T = sum(lens)
        N = len(lens)
        q, k, v, g, beta = gen_qkvgb(T, Hk, Hv, Dk, Dv, torch.float32, dev,
                                     l2=True, seed_off=4)
        h0 = torch.randn(N, Hv, Dv, Dk, device=dev) * 0.5 if with_init \
            else torch.zeros(N, Hv, Dv, Dk, device=dev)
        h0_in = h0.clone()
        cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32,
                          device=dev)
        o, ht = run_sequential(ops, q, k, v, g, beta, Dk ** -0.5, h0, cu)
        total += save_case(
            root, name, "gdn_prefill",
            "fla/ops/fused_recurrent.py::fused_recurrent_gated_delta_rule",
            {"q": q, "k": k, "v": v, "g": g, "beta": beta,
             "state_in": h0_in, "query_start_loc": cu,
             "out": o, "state_out": ht},
            {"scale": Dk ** -0.5, "Hk": Hk, "Hv": Hv, "Dk": Dk, "Dv": Dv,
             "seqlens": lens, "use_qk_l2norm_in_kernel": False,
             "q_k_prenormalized": True, "with_initial_state": with_init,
             "note": f32_note},
            ["q", "k", "v", "g", "beta", "state_in", "query_start_loc"],
            ["out", "state_out"], TIGHT)

    # --- bf16 case: the REAL prefill oracle (chunked kernel), two sequences
    #     (one with initial state, one without — zeroed rows like
    #     qwen_gdn_linear_attn.py:1514). Also measure sequential-vs-chunked.
    Hk, Hv, Dk, Dv = 1, 2, 128, 128
    lens = [20, 12]
    T, N = sum(lens), len(lens)
    q, k, v, g, beta = gen_qkvgb(T, Hk, Hv, Dk, Dv, torch.bfloat16, dev,
                                 l2=True, seed_off=5)
    h0 = torch.randn(N, Hv, Dv, Dk, device=dev) * 0.5
    h0[1].zero_()  # seq 1: no initial state
    h0_in = h0.clone()
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32,
                      device=dev)
    o, ht = ops.chunk.chunk_gated_delta_rule(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        g=g.unsqueeze(0), beta=beta.unsqueeze(0),
        initial_state=h0, output_final_state=True, cu_seqlens=cu,
        use_qk_l2norm_in_kernel=False)
    # gap measurement: same inputs through the pinned sequential kernel
    o_seq, ht_seq = run_sequential(ops, q, k, v, g, beta, Dk ** -0.5,
                                   h0_in.clone(), cu)
    gap_o = (o.squeeze(0).float() - o_seq.float()).abs().max().item()
    gap_h = (ht.float() - ht_seq.float()).abs().max().item()
    print(f"  chunk-vs-sequential (bf16 inputs): out {gap_o:.3e} "
          f"state {gap_h:.3e}")
    total += save_case(
        root, "gdn_prefill_bf16_realdims", "gdn_prefill",
        "fla/ops/chunk.py::chunk_gated_delta_rule",
        {"q": q, "k": k, "v": v, "g": g, "beta": beta,
         "state_in": h0_in, "query_start_loc": cu,
         "out": o.squeeze(0).float(), "state_out": ht.float()},
        {"scale": Dk ** -0.5, "Hk": Hk, "Hv": Hv, "Dk": Dk, "Dv": Dv,
         "seqlens": lens, "use_qk_l2norm_in_kernel": False,
         "q_k_prenormalized": True,
         "with_initial_state": [True, False],
         "chunk_vs_sequential_max_abs_out": gap_o,
         "chunk_vs_sequential_max_abs_state": gap_h,
         "note": "chunked (FLA_CHUNK_SIZE=64) oracle output on bf16 inputs; "
                 "out/state dumped as f32 of the oracle's values. Sequential "
                 "C++ compared at the loose bf16 tolerance."},
        ["q", "k", "v", "g", "beta", "state_in", "query_start_loc"],
        ["out", "state_out"], BF16_CHUNK)
    return total


def dump_gdn_decode(ops, root, dev):
    total = 0
    for name, Hk, Hv, Dk, Dv, B in (
            ("gdn_decode_f32_small", 2, 4, 16, 32, 3),
            ("gdn_decode_f32_realdims", 1, 2, 128, 128, 2)):
        torch.manual_seed(6)
        # raw post-conv q/k (NOT normalized: decode l2norms in-kernel)
        q = torch.randn(B, Hk, Dk, device=dev)
        k = torch.randn(B, Hk, Dk, device=dev)
        v = torch.randn(B, Hv, Dv, device=dev)
        a = torch.randn(B, Hv, device=dev)
        b = torch.randn(B, Hv, device=dev)
        A_log = torch.randn(Hv, device=dev).abs().log()  # ~log|N(0,1)|
        dt_bias = torch.randn(Hv, device=dev) * 0.5
        cache = torch.randn(B + 1, Hv, Dv, Dk, device=dev) * 0.5
        cache[0].zero_()  # NULL block
        state_in = cache[1:].clone()
        cu = torch.arange(0, B + 1, dtype=torch.int32, device=dev)
        idx = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
        o, _ = ops.fused_sigmoid_gating.fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, b=b, dt_bias=dt_bias,
            q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
            initial_state=cache, inplace_final_state=True,
            cu_seqlens=cu, ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True)
        state_out = cache[1:].clone()

        # Derived tensors for the decomposed C++ chain (L2Norm -> GdnDecode
        # taking g/beta): computed by the PINNED fused_post_conv_prep.
        mixed = torch.cat([q.reshape(B, -1), k.reshape(B, -1),
                           v.reshape(B, -1)], dim=-1).contiguous()
        q_l2, k_l2, v2, g, beta = ops.post_conv.fused_post_conv_prep(
            conv_output=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            num_k_heads=Hk, head_k_dim=Dk, head_v_dim=Dv,
            apply_l2norm=True, output_g_exp=False)
        assert torch.equal(v2, v)
        total += save_case(
            root, name, "gdn_decode",
            "fla/ops/fused_sigmoid_gating.py::"
            "fused_sigmoid_gating_delta_rule_update",
            {"q": q, "k": k, "v": v, "a": a, "b": b,
             "A_log": A_log, "dt_bias": dt_bias, "state_in": state_in,
             "q_l2": q_l2, "k_l2": k_l2, "g": g, "beta": beta,
             "out": o.squeeze(0), "state_out": state_out},
            {"scale": Dk ** -0.5, "Hk": Hk, "Hv": Hv, "Dk": Dk, "Dv": Dv,
             "batch": B, "use_qk_l2norm_in_kernel": True,
             "softplus_threshold": 20.0,
             "note": "q/k are raw (l2norm applied IN-kernel, eps 1e-6). "
                     "q_l2/k_l2/g/beta are derived via the pinned "
                     "fused_post_conv_prep for the decomposed C++ chain "
                     "L2Norm -> GdnDecode(q_l2,k_l2,v,g,beta). States are "
                     "gathered per-seq slices (cache line 0 = NULL block)."},
            ["q", "k", "v", "a", "b", "A_log", "dt_bias", "state_in"],
            ["out", "state_out"], TIGHT)
    return total


DUMPERS = {
    "causal_conv1d_fwd": dump_conv_fwd,
    "causal_conv1d_update": dump_conv_update,
    "l2norm": dump_l2norm,
    "rmsnorm_gated": dump_rmsnorm_gated,
    "gdn_prefill": dump_gdn_prefill,
    "gdn_decode": dump_gdn_decode,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pin", default="/home/mudler/_git/vllm")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "pinned Triton oracles need the GPU"
    ops = load_pinned(pathlib.Path(args.pin).expanduser())
    dev = "cuda"
    root = pathlib.Path(args.out)
    grand = 0
    for name, fn in DUMPERS.items():
        if args.only is None or name in args.only:
            grand += fn(ops, root, dev)
    print(f"TOTAL golden bytes: {grand / 1024:.1f} KiB "
          f"({grand / (1024 * 1024):.2f} MiB)")
    assert grand <= 2 * 1024 * 1024, "over the 2MB golden budget"
