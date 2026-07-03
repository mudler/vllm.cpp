# Qwen3.6 sparse-MoE semantics — pinned oracle notes (M0.8 Task 1)

Recorded from the pinned upstream checkout `/home/mudler/_git/vllm` @
`e24d1b24` (see .agents/upstream-sync.md). Every formula below was read from
source, not memory. Cites are `file:line` relative to the pinned tree
(`vllm/...`, `csrc/...`, `tests/...`). This is the M0.8 op contract reference
AND the M0.9 layer-assembly reference. §8 records the NVFP4 (modelopt)
checkpoint format that Task 4 consumes.

Pinned sources read:

- `vllm/model_executor/models/qwen3_next.py` (`Qwen3NextSparseMoeBlock`;
  reused verbatim by the actual gate model class in
  `models/qwen3_5.py:72-77` — checkpoint architecture
  `Qwen3_5MoeForConditionalGeneration`, registry.py:557-560)
- `vllm/model_executor/models/qwen2_moe.py` (`Qwen2MoeMLP` = the shared
  expert, imported as `Qwen3NextMLP`, qwen3_next.py:50)
- `vllm/model_executor/layers/fused_moe/layer.py` (`FusedMoE` factory →
  `MoERunner`)
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` (forward
  orchestration, internal gate, combine order)
- `vllm/model_executor/layers/fused_moe/router/{router_factory.py,
  fused_topk_router.py,base_router.py}` (router selection + fused_topk)
- `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` (the production
  softmax/top-k kernel: dtype, tie-break, renormalize semantics)
- `tests/kernels/moe/test_fused_topk.py::torch_topk` and
  `tests/kernels/utils.py::torch_experts` (the pinned torch-native
  references — our golden oracles, see §7)
- `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py::select_experts`
  (pinned torch-native production path, used as dump-time cross-check)
- `vllm/model_executor/layers/activation.py` (`SiluAndMul.forward_native`)
- `vllm/model_executor/layers/quantization/modelopt.py` +
  `.../quantization/utils/nvfp4_emulation_utils.py` +
  `.../fused_moe/experts/nvfp4_emulation_moe.py` (NVFP4 format + dequant)
- `vllm/transformers_utils/configs/qwen3_5_moe.py` (config keys)

## 1. Block structure & config (Qwen3.6-35B-A3B actual values)

`Qwen3NextSparseMoeBlock` (qwen3_next.py:101-222):

- `gate = ReplicatedLinear(hidden_size, num_experts, bias=False,
  quant_config=None)` — the router gate is **unquantized** even in quantized
  checkpoints (qwen3_next.py:138-144).
- `shared_expert_gate = ReplicatedLinear(hidden_size, 1, bias=False,
  quant_config=None)` (qwen3_next.py:146-152).
- `shared_expert = Qwen3NextMLP(hidden, shared_expert_intermediate_size,
  hidden_act, reduce_results=False, expert_gate=shared_expert_gate)`
  (qwen3_next.py:165-173; built unless
  `shared_expert_intermediate_size <= 0`, line 163).
- `experts = FusedMoE(num_experts, top_k=num_experts_per_tok, hidden_size,
  intermediate_size=moe_intermediate_size,
  renormalize=getattr(config, "norm_topk_prob", True), gate=self.gate,
  shared_experts=self.shared_expert, ...)` (qwen3_next.py:176-193).

Config (dgx, nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot 491c2f1e, config.json
`text_config`, read 2026-07-03): `hidden_size 2048`, `num_experts 256`,
`num_experts_per_tok 8`, `moe_intermediate_size 512`,
`shared_expert_intermediate_size 512`, `hidden_act "silu"`.
**`norm_topk_prob` is ABSENT from config.json and from the pinned config
class `Qwen3_5MoeTextConfig` (qwen3_5_moe.py:22-121 — not a field)**, so
`getattr(config, "norm_topk_prob", True)` → **renormalize = True**
(qwen3_next.py:183). `scoring_func` defaults to `"softmax"` (layer.py:118),
`routed_scaling_factor = 1.0` (layer.py:119), `e_score_correction_bias =
None`, `use_grouped_topk = False`, `apply_router_weight_on_input = False`
(defaults, layer.py:108-124; qwen3_next.py passes none of these).

Router selection: with no grouped-topk / custom fn / bias / zero-expert /
fused-shared-slots, `create_fused_moe_router` falls through to the default
`FusedTopKRouter` (router_factory.py:67-74 priority list; fused_topk_router.
py:116-163).

## 2. Forward orchestration & router logits dtype

Because `gate` is passed to `FusedMoE`, `experts.is_internal_router` is True
(moe_runner.py:317-319), so the block calls
`self.experts(hidden_states=x, router_logits=x)` (qwen3_next.py:204-208) and
the RUNNER applies the gate: `router_logits, _ = self.gate(hidden_states)`
(moe_runner.py:814-819; `_fse_fuse_gate` is False for our models since
`shared_expert_gate` is only forwarded to the runner when the shared expert
was fused away, qwen3_next.py:189-192).

- The gate is a plain `ReplicatedLinear` (NOT the `GateLinear` fp32 router
  class — that is only used by models that construct it explicitly), so
  **router logits = F.linear(x_bf16, W_bf16) in bf16**
  (linear.py:375-386 → UnquantizedLinearMethod → default_unquantized_gemm =
  `torch.nn.functional.linear`, layers/utils.py:92-98, 332-338).
- The softmax over logits is computed **in f32** regardless: the CUDA kernel
  converts bf16 inputs to float on load (topk_softmax_kernels.cu:60-66,
  345-376) and the torch-native reference does `gating_output.float()`
  (test_fused_topk.py:26). `router_logits_dtype` is not set by Qwen3.6
  (default None, layer.py:132).

## 3. Router: softmax → top-k → renormalize (`kMoeRouterTopK`)

Call chain: `MoERunner._apply_quant_method` → `router.select_experts`
(moe_runner.py:567-573) → `_select_experts` (base_router.py:260-306; EPLB
mapping and dtype conversion are no-ops for us) → `fused_topk`
(fused_topk_router.py:69-113, scoring_func="softmax") → `ops.topk_softmax`
(fused_topk_router.py:17-32; _custom_ops.py:2222-2237) →
`topkGating`/`moeSoftmax`+`moeTopK` (topk_softmax_kernels.cu).

Math, per token row of `logits [E]` (f32 arithmetic):

    p = softmax(logits.float())            # max-subtracted, over ALL E
                                           # (kernels .cu:81-141 & 402-447;
                                           # torch ref test_fused_topk.py:26)
    p[isnan(p) | isinf(p)] = 0             # NaN/Inf clamp (.cu:136, 458-467)
    for j in 0..k-1:                       # greedy argmax, k rounds
        e_j = argmax over experts not yet chosen; ties → LOWEST expert
              index wins (.cu:530-537 "// We want lower indices to win";
              moeTopK path: cub::ArgMax on (idx,val) pairs, .cu:178-243)
        w_j = p[e_j]                       # UNBIASED softmax prob (.cu:233)
    if renormalize:                        # = norm_topk_prob = True for us
        denom = sum_j w_j;  denom = 1 if denom <= 0   (.cu:245-253, 574-586)
        w_j /= denom

Outputs: `topk_weights` **f32** `[T, k]`, `topk_ids` **i32** `[T, k]`
(fused_topk allocates them so, fused_topk_router.py:81-92). Selection order =
descending weight (greedy argmax), so ids/weights are sorted per token.

Torch-native pinned statements of the same math (both verified equivalent):

- `tests/kernels/moe/test_fused_topk.py::torch_topk` (lines 18-44):
  `scores = softmax(logits.float()); w, ids = torch.topk(scores, k);
  if renormalize: w /= w.sum(-1, keepdim=True)`. This is the reference the
  pinned test suite compares the CUDA kernel against (atol 1e-2 on the
  kernel; exact math identical).
- `cpu_fused_moe.py::select_experts` (lines 150-159, the CPU production
  path): topk over LOGITS then `softmax(topk_logits)` when renormalize
  (algebraically identical: softmax is monotonic; renorm cancels logZ) or
  `(topk_logits - logsumexp(logits)).exp()` when not.

**C++ contract**: op takes logits (any float dtype), computes softmax in f32
over all E, greedy top-k with lowest-index tie-break, optional renormalize
with the `denom > 0 else 1` guard, emits f32 weights + i32 ids.

Golden note: the realratio router case is tie-free by construction (dump-time
per-row uniqueness assert + deterministic post-bf16 dedup that nudges exact
f32(bf16) duplicates to the next representable value); the lowest-index
tie-break rule itself is pinned by Task 2 CPU unit tests, not by this golden.

## 4. Routed experts: per-expert silu-mul MLP (composes existing ops)

Weights per expert: `w13 [E, 2*I, H]` — **gate_proj rows first (w1, rows
0..I-1), up_proj rows second (w3, rows I..2I-1)** (routed_experts.py:299-302,
483; ckpt names ("gate_proj","down_proj","up_proj"), layer.py:130) — and
`w2 [E, H, I]` (down_proj) (unquantized_fused_moe_method.py:88-111).

Per token t routed to expert e with weight w (pinned reference
`tests/kernels/utils.py::torch_experts`, lines 855-994, quant_dtype=None
branch 921-928):

    h1  = x_t @ w13[e].T          # [2I], activation dtype (bf16) matmul
    a   = silu(h1[:I]) * h1[I:]   # SiluAndMul.forward_native,
                                  # activation.py:137-141
    y_e = a @ w2[e].T             # [H]

Weighted combine (torch_experts:989-994): the router weight is applied to
the **down-proj OUTPUT** (not the input; `apply_router_weight_on_input` is
False for Qwen3.6), accumulated **in f32**, then cast back:

    routed_out_t = ( sum_j w_j * y_{e_j}.float() ).to(x.dtype)   # bf16

This matches the production Triton path (topk weight multiplied on the w2
output + moe_sum reduction over k). Slots whose expert receives no token
contribute zeros (torch_experts pre-zeroes `out`, line 897).

## 5. Shared expert + sigmoid gate

`Qwen2MoeMLP.forward` (qwen2_moe.py:112-120), on the SAME block input x:

    g_u    = x @ W_gate_up.T                  # MergedColumnParallelLinear
                                              # [2*Is] = [gate; up] halves
    s      = silu(g_u[:Is]) * g_u[Is:]        # SiluAndMul (line 109, 114)
    shared = s @ W_down.T                     # RowParallelLinear (line 115)
    shared = sigmoid(x @ W_seg.T) * shared    # expert_gate = the block's
                                              # shared_expert_gate ([1, H] →
                                              # scalar per token, broadcast);
                                              # F.sigmoid on the gate LOGIT
                                              # computed from x, applied to
                                              # the MLP OUTPUT (lines 117-118)

All in model dtype (bf16); at TP=1 every linear is `F.linear`
(layers/utils.py:92-98). `reduce_results=False` (qwen3_next.py:171) — the
runner handles reduction; single-rank no-op.

## 6. Combine order (`kMoeCombine`) and the block output

`MoERunner.forward` (moe_runner.py:634-723): shared expert runs on
`shared_experts_input = hidden_states` (pre-gate x), routed experts produce
`fused_output` (already includes topk weights, §4), then

    result = shared_output + fused_output        (moe_runner.py:717)

`routed_scaling_factor = 1.0` → the `_maybe_apply_routed_scale_to_output`
step is a no-op (moe_runner.py:389-406); all-reduces are single-rank no-ops;
no zero-expert. Block returns `result.view(orig_shape)`
(qwen3_next.py:214-222).

**C++ `kMoeCombine` contract**: out[t] = sum_j weights[t,j] *
expert_out[t,j,:] (f32 accumulation, §4) + optional shared_term[t,:] added
in activation dtype (§6 order: the f32 weighted sum is rounded to bf16
BEFORE the shared add — torch_experts:989-994 casts, then runner adds bf16
tensors).

## 7. M0.8 golden-dump decisions (tools/parity/dump_moe.py)

- Oracle = pinned checkout executed on dgx (venv torch), manifests record
  `oracle.source = "pinned:e24d1b24"` + exact callable per case.
- The pinned `Qwen3NextSparseMoeBlock.forward` is NOT runnable standalone
  (it needs VllmConfig, distributed groups, the layer registry and the
  compiled `_moe_C`/`_C` extensions), and at this pin `fused_moe` has no
  eager torch path (`fused_topk` calls `torch.ops._moe_C.topk_softmax`;
  experts go through Triton/cutlass modular kernels). Per the plan's
  fallback rule the oracle is the pinned torch-native REFERENCE fns from the
  pinned test suite, composed exactly as the module composes them:
  - `moe_router_topk_*` → `tests/kernels/moe/test_fused_topk.py::torch_topk`
    (cross-checked at dump time against the pinned CPU production path
    `cpu_fused_moe.py::select_experts`, both renormalize modes).
  - `moe_block_*` → gate `F.linear` (§2) → `torch_topk` →
    `tests/kernels/utils.py::torch_experts` (routed loop + f32 weighted
    combine) → shared expert composed per qwen2_moe.py:112-120 using pinned
    `SiluAndMul.forward_native` → `shared + routed` (moe_runner.py:717).
- Tie-break note: goldens use random logits (no ties at f32/bf16); the
  lowest-index tie rule (§3) is exercised by Task 2 unit tests, not goldens.
- Case sizes: routing shape is what matters → small 8e/top-2 cases plus a
  real-RATIO router case (256 experts, top-8, bf16 logits). The full-block
  golden uses synthetic dims (small) — a 256-expert block golden would blow
  the 2MB budget on w13 alone. Block cases dump intermediates
  (router logits/weights/ids, routed_out, shared_out) for C++-side
  debugging; only `out` is the layer-level contract.
- Budget ≤ 2MB total, seed 0 (torch.manual_seed per case).

## 8. NVFP4 (modelopt) checkpoint format — Task 4 input

Verified against the REAL 35B checkpoint on dgx
(`nvidia/Qwen3.6-35B-A3B-NVFP4`, snapshot 491c2f1e, read 2026-07-03 with our
`dump_container` CLI on shard 1) + `hf_quant_config.json` + pinned
modelopt.py.

### hf_quant_config.json

Producer `modelopt 0.44.0`. `quant_algo: "MIXED_PRECISION"` with per-layer
`quantized_layers`: every `*.mlp.experts` and
`*.mlp.shared_expert.{gate,up,down}_proj` is **`W4A16_NVFP4`, group_size
16** (weight-only 4-bit; activations stay bf16); attention/GDN projections
are FP8; `kv_cache_quant_algo: "FP8"`; `exclude_modules: ["mtp.layers.0*",
"mtp*"]` (the mtp.* tensors in shard 3 are unquantized bf16). vLLM handles
this via `ModelOptNvFp4W4A16LinearMethod` (modelopt.py:1041-1048 dispatch;
class at 1243).

### Tensor naming & layout (read from the checkpoint)

Per expert e in layer L (per-projection, NOT stacked — 12 tensors/expert):

    model.language_model.layers.{L}.mlp.experts.{e}.{gate_proj|up_proj|down_proj}.weight          U8       [out, in/2]
    ...{proj}.weight_scale    F8_E4M3  [out, in/16]
    ...{proj}.weight_scale_2  F32      []  (scalar)
    ...{proj}.input_scale     F32      []  (scalar)  ← present but UNUSED for W4A16

Measured: gate/up `weight U8 [512, 1024]` (out=I=512, in=H=2048 packed /2),
`weight_scale F8_E4M3 [512, 128]` (= 2048/16); down `weight U8 [2048, 256]`,
`weight_scale [2048, 32]`. Shared expert: same scheme under
`...mlp.shared_expert.{proj}.*`. Router gate + shared gate are bf16:
`...mlp.gate.weight BF16 [256, 2048]`,
`...mlp.shared_expert_gate.weight BF16 [1, 2048]` (matches §1: gate built
with quant_config=None). gate_proj and up_proj of the same layer share one
`weight_scale_2` value (identical hash) — the fused-partition global scale.

- **Packing**: 2×E2M1 per byte along the INPUT dim; element `2i` = LOW
  nibble, `2i+1` = HIGH nibble (`break_fp4_bytes`,
  nvfp4_emulation_utils.py:316-333: `stack((low, high))`).
- **E2M1 decode**: LUT `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` for magnitude bits
  0-7, sign = bit 3 (nvfp4_emulation_utils.py:20-22, 26-43, 316-333). The
  LUT is 1×-scaled floats.
- **Block scales**: one **IEEE fp8-e4m3fn** byte per 16 input elements,
  stored row-major `[out, in/16]` — LINEAR layout on disk, no swizzle
  (GroupQuantScaleParameter, modelopt.py:1330-1340; swizzling in
  `convert_swizzled_to_linear` is a kernel-side concern for cutlass-prepped
  tensors only, nvfp4_emulation_utils.py:336-344). Decoded by a plain
  `.view(torch.float8_e4m3fn).to(torch.float32)`
  (nvfp4_emulation_utils.py:380-381).
- **Global scale**: `weight_scale_2` f32 scalar = `amax / (6.0 * 448.0)`
  (= FLOAT4_E2M1_MAX * FP8_E4M3_MAX), used **directly, NO reciprocation**
  (modelopt.py:1249-1257 docstring; 1369-1371 "NO reciprocation: ModelOpt
  already stores amax/2688").
- **Dequant math** (Task 4 contract; nvfp4_emulation_moe.py:96-99, 226;
  dequantize_to_dtype nvfp4_emulation_utils.py:346-400):

      w_f32[o, i] = e2m1_decode(nibble[o, i])
                    * ( f32(weight_scale_e4m3[o, i//16]) * weight_scale_2 )
      w_bf16 = w_f32.to(bf16)

- `input_scale`: registered as a placeholder and DISCARDED for W4A16
  (modelopt.py:1258-1264, process_weights_after_loading) — do not use it.

### Contrast with the GGUF fork convention (.agents/gguf-nvfp4-notes.md)

- The fork's **×0.5 LUT trap does NOT apply here** (verified): the fork's
  `ggml_ue4m3_to_fp32` halves the decoded scale only to compensate its
  2×-scaled shared `kvalues_mxfp4` int8 LUT. modelopt checkpoints store
  standard signed fp8-e4m3fn scale bytes and the pinned dequant uses a
  1×-scaled float LUT with a plain e4m3→f32 conversion — no ×0.5, no UE4M3
  special-casing, anywhere (nvfp4_emulation_utils.py:20-22, 380-381;
  nvfp4_emulation_moe.py:98-99).
- Fork GGUF NVFP4 is self-contained per 64-elem block (4× UE4M3 scale bytes
  + 32 packed bytes, no side tensors); modelopt safetensors use separate
  side-channel `weight_scale` (per-16, fp8-e4m3) + `weight_scale_2`
  (per-tensor f32) tensors.
- Fork nibble order interleaves within an 8-byte sub-block (elem j low /
  j+8 high); modelopt is strictly sequential (2i low / 2i+1 high).
