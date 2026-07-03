# vllm.cpp parity harness; oracle = PINNED upstream vLLM checkout (e24d1b24).
"""Dump Qwen3.6 sparse-MoE goldens from the pinned vLLM checkout (M0.8 Task 1).

Runs on the dgx GPU inside the oracle venv, executing the PINNED sources
verbatim (never pip vLLM, never re-implemented math):

    ssh dgx.casa 'cd ~/work/vllm.cpp && ~/venvs/vllm-oracle/bin/python \
        tools/parity/dump_moe.py --pin ~/work/vllm-pin \
        --out tests/parity/goldens'

Loading technique (dump_gdn.py's exec-stub pattern): package skeletons for
`vllm...`/`tests...` point into the pinned tree WITHOUT executing package
__init__.py files; only leaf infra is stubbed (recorded in STUBBED below and
in every manifest). All oracle MATH executes from the pinned files.

Oracle callables (see .agents/moe-semantics.md §7 for why these):
  moe_router_topk -> tests/kernels/moe/test_fused_topk.py::torch_topk
                     (the pinned torch-native reference the upstream test
                     suite validates the production topk_softmax CUDA kernel
                     against; the kernel itself needs compiled _moe_C which
                     the pinned checkout does not ship). Cross-checked at
                     dump time against the pinned CPU production path
                     cpu_fused_moe.py::select_experts, both renorm modes.
  moe_block       -> composed pinned reference fns, exactly as the pinned
                     module composes them (Qwen3NextSparseMoeBlock is not
                     runnable standalone — needs VllmConfig/distributed/
                     compiled extensions):
                       gate logits  = F.linear(x, gate_w)   bf16
                                      (linear.py:375-386 ->
                                      layers/utils.py:92-98)
                       routing      = torch_topk (renormalize=True)
                       routed       = tests/kernels/utils.py::torch_experts
                                      (per-expert silu-mul MLP loop + f32
                                      weighted combine, lines 855-994)
                       shared       = qwen2_moe.py:112-120 composed with the
                                      pinned SiluAndMul.forward_native
                                      (activation.py:137-141):
                                      sigmoid(x@Wseg.T) * down(silu_mul(
                                      gate_up(x)))
                       out          = shared + routed  (moe_runner.py:717)

Case-size policy (plan M0.8 §Task 1): routing shape drives the math -> small
8-expert/top-2 cases plus one real-RATIO router case (256 experts, top-8,
bf16 logits). Block goldens use synthetic small dims (a 256-expert block
would blow the 2MB budget on w13 alone). Seed 0. Total budget <= 2MB.
"""

import argparse
import importlib
import json
import logging
import pathlib
import subprocess
import sys
import types

import numpy as np
import torch

UPSTREAM_PIN = "e24d1b24"
STUBBED = [
    "vllm.triton_utils",
    "vllm.platforms (is_cpu facade so CustomOp layers pick forward_native)",
    "vllm.logger",
    "vllm.envs",
    "vllm._custom_ops (names only; never called)",
    "vllm._aiter_ops (rocm_aiter_ops facade, everything disabled)",
    "vllm.distributed (divide/tp_rank/tp_world_size for activation.py)",
    "vllm.distributed.eplb.eplb_state (EplbLayerState placeholder)",
    "vllm.v1.worker.ubatching (dbo_current_ubatch_id=0)",
    "vllm.v1.attention.backend (AttentionType consts)",
    "vllm.model_executor.custom_op (CustomOp dispatch shim -> the PINNED "
    "forward_native; op_registry dict)",
    "vllm.model_executor.utils (set_weight_attrs no-op)",
    "vllm.model_executor.layers.fused_moe.config (RoutingMethodType names)",
    "vllm.model_executor.layers.fused_moe.utils (moe_kernel_quantize_input "
    "identity passthrough — only the quant_dtype=None branch is reachable)",
    "vllm.model_executor.layers.quantization.utils.layer_utils "
    "(replace_parameter, unused)",
    "vllm.utils.math_utils / torch_utils (cdiv etc.; "
    "direct_register_custom_op no-op)",
    "tests.kernels.quant_utils (native_w8a8_block_matmul, unreachable for "
    "unquantized cases)",
    "pytest (decorator passthrough, only if not installed)",
]

TIGHT = {"atol": 1e-5, "rtol": 1e-5}
BF16_BLOCK = {"atol": 1.5e-2, "rtol": 1.5e-2}

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
    """Import the pinned MoE reference modules with leaf-infra stubs."""
    v = pin_root / "vllm"
    t = pin_root / "tests"
    assert (t / "kernels/moe/test_fused_topk.py").exists(), pin_root

    import triton
    import triton.language as tl
    _mod("vllm.triton_utils", tl=tl, triton=triton, tldevice=None,
         HAS_TRITON=True)

    class _Platform:
        # is_cpu=True so pinned CustomOp layers (SiluAndMul) select their
        # forward_native branch (activation.py:130-135) instead of the
        # compiled torch.ops._C path the pinned checkout does not ship.
        # forward_native is pure torch and runs on CUDA tensors unchanged.
        def is_cuda(self):
            return False

        def is_cuda_alike(self):
            return False

        def is_rocm(self):
            return False

        def is_xpu(self):
            return False

        def is_cpu(self):
            return True

        def is_out_of_tree(self):
            return False

        def get_cpu_architecture(self):
            return "X86"

        def fp8_dtype(self):
            return torch.float8_e4m3fn

        def seed_everything(self, seed):
            torch.manual_seed(seed)

    class _CpuArchEnum:
        X86 = "X86"
        ARM = "ARM"
        POWERPC = "POWERPC"
        UNKNOWN = "UNKNOWN"

    _mod("vllm.platforms", current_platform=_Platform(),
         CpuArchEnum=_CpuArchEnum)

    _mod("vllm.logger", init_logger=logging.getLogger)

    envs = _mod("vllm.envs")
    envs.__getattr__ = lambda name: (
        "" if name == "VLLM_MOE_ROUTING_SIMULATION_STRATEGY" else None)

    # Names imported by pinned modules at module level; never called on the
    # golden paths (torch_topk / torch_experts are pure torch).
    _mod("vllm._custom_ops", topk_softmax=None, topk_sigmoid=None,
         CPUQuantMethod=None, cpu_fused_moe=None, cpu_prepack_moe_weight=None,
         fused_experts_cpu=None)

    class _AiterOps:
        @staticmethod
        def is_fused_moe_enabled():
            return False

        @staticmethod
        def is_fusion_moe_shared_experts_enabled():
            return False

        topk_softmax = None
        topk_sigmoid = None

    _mod("vllm._aiter_ops", rocm_aiter_ops=_AiterOps())

    _mod("vllm.distributed",
         divide=lambda a, b: a // b,
         get_tensor_model_parallel_rank=lambda: 0,
         get_tensor_model_parallel_world_size=lambda: 1)
    _mod("vllm.distributed.eplb")
    _mod("vllm.distributed.eplb.eplb_state", EplbLayerState=type(
        "EplbLayerState", (), {}))

    _mod("vllm.v1")
    _mod("vllm.v1.worker")
    _mod("vllm.v1.worker.ubatching", dbo_current_ubatch_id=lambda: 0)
    _mod("vllm.v1.attention")
    _mod("vllm.v1.attention.backend", AttentionType=type(
        "AttentionType", (), {"DECODER": "decoder", "ENCODER": "encoder",
                              "ENCODER_ONLY": "encoder_only",
                              "ENCODER_DECODER": "encoder_decoder"}))

    # CustomOp shim: registration bookkeeping only; dispatch is pinned to
    # each op's own forward_native (the pinned math). Real custom_op.py only
    # adds compiled-op dispatch we cannot (and must not) use here.
    op_registry = {}

    class _CustomOp(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._forward_method = self.forward_native

        def forward(self, *args, **kwargs):
            return self._forward_method(*args, **kwargs)

        def forward_native(self, *args, **kwargs):
            raise NotImplementedError

        @classmethod
        def register(cls, name):
            def deco(kls):
                kls.name = name
                op_registry[name] = kls
                return kls

            return deco

        @classmethod
        def register_oot(cls, *a, **k):
            return lambda kls: kls

        @classmethod
        def enabled(cls):
            return False

        @classmethod
        def default_on(cls):
            return False

    _mod("vllm.model_executor.custom_op", CustomOp=_CustomOp,
         op_registry=op_registry)
    _mod("vllm.model_executor.utils",
         set_weight_attrs=lambda *a, **k: None)

    class _RoutingMethodType:
        Default = 0
        Renormalize = 1
        RenormalizeNaive = 2
        DeepSeekV3 = 3
        Llama4 = 4
        TopK = 5
        Unspecified = 6

    _mod("vllm.model_executor.layers.fused_moe.config",
         RoutingMethodType=_RoutingMethodType,
         get_routing_method_type=lambda **k: _RoutingMethodType.Unspecified)

    def _moe_kernel_quantize_input(A, A_scale, quant_dtype,
                                   per_act_token_quant=False,
                                   block_shape=None, is_fp4_scale_swizzled=True):
        # Identity passthrough: golden cases are unquantized, so only the
        # quant_dtype=None branch (return input unchanged) is ever taken.
        assert quant_dtype is None, "quantized golden path not stubbed"
        return A, A_scale

    _mod("vllm.model_executor.layers.fused_moe.utils",
         moe_kernel_quantize_input=_moe_kernel_quantize_input)

    _mod("vllm.model_executor.layers.quantization")
    _mod("vllm.model_executor.layers.quantization.utils")
    _mod("vllm.model_executor.layers.quantization.utils.layer_utils",
         replace_parameter=lambda *a, **k: None)

    def cdiv(a, b):
        return -(a // -b)

    _mod("vllm.utils.math_utils", cdiv=cdiv,
         next_power_of_2=lambda n: 1 if n <= 1 else
         2 ** (int(n - 1).bit_length()))
    _mod("vllm.utils.torch_utils", make_tensor_with_pad=None,
         direct_register_custom_op=lambda *a, **k: None,
         is_torch_equal_or_newer=lambda *a, **k: True)

    try:
        import pytest  # noqa: F401
    except ImportError:
        def _passthrough(*a, **k):
            return lambda fn: fn

        mark = types.SimpleNamespace()
        mark.__getattr__ = lambda name: _passthrough
        mark.parametrize = _passthrough
        mark.skipif = _passthrough
        _mod("pytest", mark=mark)

    # Package skeletons pointing into the pinned tree; __init__.py NOT run.
    _pkg("vllm", v)
    _pkg("vllm.model_executor", v / "model_executor")
    _pkg("vllm.model_executor.layers", v / "model_executor/layers")
    _pkg("vllm.model_executor.layers.fused_moe",
         v / "model_executor/layers/fused_moe")
    _pkg("vllm.model_executor.layers.fused_moe.router",
         v / "model_executor/layers/fused_moe/router")
    _pkg("vllm.utils", v / "utils")  # for the real collection_utils
    _pkg("tests", t)
    _pkg("tests.kernels", t / "kernels")
    _pkg("tests.kernels.moe", t / "kernels/moe")
    _mod("tests.kernels.quant_utils", native_w8a8_block_matmul=None)

    # NB: stubs registered ABOVE package skeletons in sys.modules win over
    # the skeleton path search, so e.g. fused_moe.config stays stubbed while
    # fused_moe.activation imports the pinned file.
    ops = types.SimpleNamespace()
    ops.activation = importlib.import_module(
        "vllm.model_executor.layers.activation")
    ops.moe_activation = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.activation")
    ops.cpu_fused_moe = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.cpu_fused_moe")
    ops.topk_test = importlib.import_module(
        "tests.kernels.moe.test_fused_topk")
    ops.kutils = importlib.import_module("tests.kernels.utils")

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


# ------------------------------------------------------------------ dumps

def _run_router(ops, logits, k, renormalize):
    """torch_topk oracle + pinned-CPU-path cross-check + sanity checks."""
    w, ids = ops.topk_test.torch_topk(logits, topk=k,
                                      renormalize=renormalize)
    # Cross-check vs the pinned CPU production path (cpu_fused_moe.py:
    # 150-159): algebraically the same routing, different pinned statement.
    w2, ids2 = ops.cpu_fused_moe.select_experts(
        hidden_states=logits,  # unused by the softmax branch
        router_logits=logits, top_k=k, use_grouped_topk=False,
        renormalize=renormalize)
    order1 = torch.argsort(ids, dim=-1)
    order2 = torch.argsort(ids2.long(), dim=-1)
    assert torch.equal(ids.gather(-1, order1).long(),
                       ids2.long().gather(-1, order2)), "id mismatch"
    gap = (w.gather(-1, order1).float()
           - w2.float().gather(-1, order2)).abs().max().item()
    print(f"  torch_topk vs cpu select_experts max|dw| = {gap:.3e}")
    # The CPU path softmaxes the top-k LOGITS in the logits dtype, so for
    # bf16 logits it re-rounds through bf16 (~1e-3); the golden oracle
    # (torch_topk) softmaxes in f32 like the production CUDA kernel.
    assert gap < (1e-5 if logits.dtype == torch.float32 else 4e-3)
    # sanity
    T = logits.shape[0]
    assert ids.min() >= 0 and ids.max() < logits.shape[1]
    for row in ids.tolist():
        assert len(set(row)) == k, "duplicate expert in top-k"
    if renormalize:
        assert (w.sum(-1) - 1.0).abs().max().item() < 1e-5
    del T
    return w.float(), ids.to(torch.int32)


def _make_rows_unique(logits):
    """Ensure every row's f32(bf16) values are distinct.

    Nudges each exact duplicate (in ascending index order) to the next
    representable value of ``logits.dtype`` via ``nextafter(., +inf)`` until
    it is unused within its row. Deterministic; only colliding entries move,
    by a few ULPs each, so the top-k routing structure is preserved. Runs on
    CPU (``.item()``/``nextafter`` per element; rows are tiny).
    """
    inf = torch.tensor(float("inf"), device=logits.device, dtype=logits.dtype)
    out = logits.clone()
    for r in range(out.shape[0]):
        seen = set()
        for c in range(out.shape[1]):
            v = out[r, c]
            while v.item() in seen:  # strictly increasing bf16 steps; the
                v = torch.nextafter(v, inf)  # row has < E prior values, so
            seen.add(v.item())               # a free slot is found in < E.
            out[r, c] = v
    return out


def dump_router_topk(ops, root, dev):
    total = 0
    for name, T, E, k, dtype, renorm, tol, tie_free in (
            ("moe_router_topk_f32_small", 6, 8, 2, torch.float32, True,
             TIGHT, False),
            ("moe_router_topk_f32_small_norenorm", 6, 8, 2, torch.float32,
             False, TIGHT, False),
            ("moe_router_topk_bf16_realratio", 16, 256, 8, torch.bfloat16,
             True, TIGHT, True)):
        torch.manual_seed(0)
        logits = (torch.randn(T, E, device=dev) * 2.0).to(dtype)
        note_extra = ""
        if tie_free:
            # bf16 keeps only 8 mantissa bits, so random logits collide
            # within a row and torch.topk breaks those ties HIGHER-index
            # first — the opposite of the production kernel's lowest-index
            # rule (§3). A correct C++ impl would then mismatch topk_ids on
            # the tied rows while the (equal) weights silently agree. Make
            # the ordering unambiguous by nudging exact f32(bf16) duplicates
            # to the next representable bf16 value (deterministic, ascending
            # index order): only colliding entries move, by a few ULPs each,
            # so the top-k routing structure is preserved. (A uniform pre-bf16
            # ramp cannot guarantee this for 256 experts — the step needed to
            # split adjacent indices at this magnitude exceeds the logit
            # range and would destroy the routing.)
            logits = _make_rows_unique(logits)
            note_extra = (" This realratio case is tie-free by construction: "
                          "exact f32(bf16) duplicates are nudged to the next "
                          "representable bf16 value (a few ULPs, ascending "
                          "index order) so no row has duplicate logits; a "
                          "dump-time per-row uniqueness assert enforces it.")
        # Fail loudly on regeneration if any row carries duplicate f32(bf16)
        # logits — a tie would reintroduce the ordering ambiguity.
        lf = logits.float()
        for r in range(T):
            assert torch.unique(lf[r]).numel() == E, \
                f"{name}: duplicate logits in row {r} (tie-break ambiguity)"
        w, ids = _run_router(ops, logits, k, renorm)
        total += save_case(
            root, name, "moe_router_topk",
            "tests/kernels/moe/test_fused_topk.py::torch_topk",
            {"logits": logits, "topk_weights": w, "topk_ids": ids},
            {"num_experts": E, "top_k": k, "renormalize": renorm,
             "scoring_func": "softmax",
             "note": "softmax in f32 over ALL experts (kernel converts "
                     "bf16 logits to f32; topk_softmax_kernels.cu:60-66), "
                     "greedy top-k (weights descending per token), "
                     "renormalize divides the k selected probs by their sum "
                     "(denom guard: sum>0 else 1, .cu:245-253). Tie-break = "
                     "lowest expert index (.cu:530-537) — not exercised "
                     "here (logits unique per row), unit-tested in Task 2. "
                     "Cross-checked at dump time against pinned "
                     "cpu_fused_moe.select_experts." + note_extra},
            ["logits"], ["topk_weights", "topk_ids"], tol)
    return total


def _shared_expert(ops, x, w_gate_up, w_down, w_seg):
    """Pinned Qwen2MoeMLP.forward composition (qwen2_moe.py:112-120)."""
    import torch.nn.functional as F
    silu_and_mul = ops.activation.SiluAndMul()
    gate_up = F.linear(x, w_gate_up)          # MergedColumnParallelLinear
    act = silu_and_mul(gate_up)               # act_fn = SiluAndMul (:109,114)
    out = F.linear(act, w_down)               # RowParallelLinear (:115)
    return torch.sigmoid(F.linear(x, w_seg)) * out  # expert_gate (:117-118)


def dump_moe_block(ops, root, dev):
    import torch.nn.functional as F
    total = 0
    E, k, H, I, Is, T = 8, 2, 64, 32, 48, 6
    for name, dtype, tol in (
            ("moe_block_f32_small", torch.float32, TIGHT),
            ("moe_block_bf16_small", torch.bfloat16, BF16_BLOCK)):
        torch.manual_seed(0)
        x = torch.randn(T, H, device=dev, dtype=dtype)
        gate_w = torch.randn(E, H, device=dev, dtype=dtype) * 0.5
        # w13: gate_proj rows first, up_proj rows second
        # (routed_experts.py:299-302,483)
        w13 = torch.randn(E, 2 * I, H, device=dev, dtype=dtype) * 0.15
        w2 = torch.randn(E, H, I, device=dev, dtype=dtype) * 0.15
        w_shared_gate_up = torch.randn(2 * Is, H, device=dev,
                                       dtype=dtype) * 0.15
        w_shared_down = torch.randn(H, Is, device=dev, dtype=dtype) * 0.15
        w_seg = torch.randn(1, H, device=dev, dtype=dtype) * 0.5

        # 1. internal gate: router_logits = F.linear(x, gate_w) in the model
        #    dtype (moe_runner.py:814-819; linear.py:375-386 ->
        #    layers/utils.py:92-98).
        logits = F.linear(x, gate_w)
        # 2. routing (renormalize=True: norm_topk_prob absent from the
        #    Qwen3.6 config -> getattr default, qwen3_next.py:183).
        topk_w, topk_ids = _run_router(ops, logits, k, True)
        # 3. routed experts: pinned torch_experts (per-expert silu-mul MLP,
        #    router weight applied to the down-proj OUTPUT, f32 weighted
        #    sum over k, cast back — tests/kernels/utils.py:855-994).
        routed = ops.kutils.torch_experts(
            x, w13, w2, topk_w, topk_ids.long(), global_num_experts=E)
        # 4. shared expert (+ sigmoid shared_expert_gate on the block input).
        shared = _shared_expert(ops, x, w_shared_gate_up, w_shared_down,
                                w_seg)
        # 5. combine (moe_runner.py:717).
        out = shared + routed

        total += save_case(
            root, name, "moe_block",
            "composed pinned refs: F.linear gate -> "
            "tests/kernels/moe/test_fused_topk.py::torch_topk -> "
            "tests/kernels/utils.py::torch_experts + "
            "qwen2_moe.py:112-120 shared expert (pinned "
            "SiluAndMul.forward_native) -> moe_runner.py:717 add",
            {"x": x, "gate_w": gate_w, "w13": w13, "w2": w2,
             "w_shared_gate_up": w_shared_gate_up,
             "w_shared_down": w_shared_down, "w_shared_gate": w_seg,
             "router_logits": logits, "topk_weights": topk_w,
             "topk_ids": topk_ids, "routed_out": routed.float(),
             "shared_out": shared.float(), "out": out.float()},
            {"num_experts": E, "top_k": k, "hidden": H,
             "moe_intermediate": I, "shared_intermediate": Is,
             "renormalize": True, "scoring_func": "softmax",
             "activation": "silu",
             "w13_layout": "[E, 2I, H], gate rows 0..I-1, up rows I..2I-1",
             "w2_layout": "[E, H, I]",
             "note": "end-to-end Qwen3NextSparseMoeBlock golden (the "
                     "M0.9 assembly target). out = shared + routed; "
                     "router_logits/topk_*/routed_out/shared_out are "
                     "dumped intermediates for debugging, only `out` is "
                     "the layer contract. Weights/x/out dumped in the case "
                     "dtype except routed_out/shared_out/out (f32 copies "
                     "of the oracle values)."},
            ["x", "gate_w", "w13", "w2", "w_shared_gate_up",
             "w_shared_down", "w_shared_gate"],
            ["out"], tol)
    return total


DUMPERS = {
    "moe_router_topk": dump_router_topk,
    "moe_block": dump_moe_block,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pin", default="/home/mudler/_git/vllm")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    assert torch.cuda.is_available(), \
        "run on dgx for a reproducible oracle device record"
    torch.backends.cuda.matmul.allow_tf32 = False  # exact f32 goldens
    torch.backends.cudnn.allow_tf32 = False
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
