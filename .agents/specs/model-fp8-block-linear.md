# MODEL-FP8-BLOCK-LINEAR — the block-wise FP8 linear method, and the Qwen3.5 dense forward that reaches it

Issue: [#1189](https://github.com/mudler/vllm.cpp/issues/1189), milestone **M4**.
Row: `MODEL-FP8-BLOCK-LINEAR`.
Pinned oracle: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`
(`.agents/upstream-sync.md`), asserted as the HEAD of the local checkout before
any `file:line` below was read.

## Scope

A block-wise FP8 checkpoint LOADS today and refuses to be prepared, because
nothing consumes an `Fp8BlockWeight`. This row makes it RUN on the arm that has
a kernel, and refuses by name on the arm that does not.

1. **`dense_fp8_block::MatmulFp8BlockScaledD`**, the shared compute body:
   dynamic per-token per-group activation quant, then the block-scaled GEMM,
   over lazily-uploaded device-resident packed bytes and scale. Templated on the
   `Dev`/`DBuf` glue exactly as `dense_fp8_gemm.h` is, for the reason recorded
   there: `qwen3_5.cpp` carries its own copies of those two types, so a
   non-template header would have to be COPIED into it rather than called.
2. **`layers::Fp8BlockLinearMethod`** and the
   `MakeLinearMethod(const OwnedTensor&, const Fp8BlockWeight&)` overload — the
   `LinearMethodBase` policy layer over that one body, mirroring
   `Fp8W8A8LinearMethod` beside it.
3. **The Qwen3.5 dense forward**: every one of the ten projections a
   `Qwen3_5ForConditionalGeneration` checkpoint quantizes gains a block rung, so
   no projection can fall through to an empty bf16 tensor.
4. **The M3 refusal is NARROWED, not deleted.** `PrepareQwen3_5Dense` stops
   refusing every loaded block weight and refuses only a block weight this
   DEVICE cannot execute, naming M5.

**Out of scope, each owned by a later milestone of #1189**: the mainloop-scaled
CUTLASS kernel for `sm_121a` and the column-major / TMA-aligned activation-scale
layouts it wants (M5); merged `gate_up` and QKV (M6). No CUDA kernel lands here,
no GPU is leased, and no checkpoint is downloaded.

## What upstream does, read end to end

The whole of upstream's block-wise linear apply is a quantize and a GEMM, and
both halves already exist in this tree as `vt` ops (M1 `ad5f175e7`, M2
`770e49486`). This row is the composition and the wiring, not new arithmetic.

| What | Where |
|---|---|
| `block_quant = self.weight_block_size is not None` is the whole dispatch | `vllm/model_executor/layers/quantization/fp8.py:297-298` |
| the block arm's kernel is selected once at `create_weights`, from the activation quant key's group shape | `fp8.py:387-393`, `kernels/linear/__init__.py:580-600` |
| **the apply, whole**: `q_input, input_scale = self.quant_fp8(input_2d)` then `self.apply_block_scaled_mm(A=q_input, B=weight, As=input_scale, Bs=weight_scale)` | `kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:97-135` |
| the activation quant is DYNAMIC, per token, group `(1, 128)`, and a static one is refused outright | `BlockScaledMMLinearKernel.py:53-58,62-70` |
| the CUTLASS arm's GEMM: `ops.cutlass_scaled_mm(A, B.T, out_dtype, scale_a=As, scale_b=Bs.T)` | `kernels/linear/scaled_mm/cutlass.py:312-326` |
| **the output dtype**: `out_dtype = self.config.out_dtype`, which is `Fp8LinearMethod.out_dtype` | `BlockScaledMMLinearKernel.py:104`, `fp8.py:391-392` |
| and that is `torch.get_default_dtype()` — the model dtype, bf16 for this checkpoint | `fp8.py:284` |
| the f32 accumulator and the cast at the store are inside the kernel, not the caller | `csrc/.../c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh:56-58`, mirrored by `vt::MatmulFp8BlockScaled` |
| there is no bias on this path; upstream refuses one | `csrc/.../c3x/scaled_mm_helper.hpp:54` |
| the divisibility the activation quant demands | `utils/fp8_utils.py:596-599` |
| the M1 op that mirrors the quant | `.agents/specs/vt-quant-fp8-group.md` |
| the M2 op that mirrors the GEMM | `.agents/specs/vt-matmul-fp8-block-ref.md` |
| the M3 weight, loader rung and config reader this consumes | `.agents/specs/model-fp8-block-weight.md` |

## Design

### One body, two instantiations, no copy

```c++
template <class DBufT, class DevT>
DBufT MatmulFp8BlockScaledD(DevT d, const Tensor& x, const Fp8BlockWeight& w,
                            DType out_dtype);
```

`x` is `[M,K]` f32 or bf16. The body is upstream's apply, in order:

```text
a_fp8   [M, K]                  i8   vt::QuantFp8Group(x, block_k)
a_scale [M, K / block_k]        f32  emitted by the same call
w_fp8   [N, K]                  i8   resident, uploaded once
w_scale [cdiv(N,bn), cdiv(K,bk)] f32 resident, uploaded once
out     [M, N]                  out_dtype   vt::MatmulFp8BlockScaled(...)
```

This is the same shape `dense_fp8_gemm.h` gives the per-tensor arm, and it is
templated for the same recorded reason: `qwen3_5.cpp` has its own
anonymous-namespace `Dev`/`DBuf` (`qwen3_5.cpp:669`), distinct TYPES from
`dense_attn::Dev`/`DBuf` even though the layouts match, and unifying the two glue
families is a separate refactor that `dense_nvfp4_gemm.h` already records as
deferred. A non-template header would have to be re-typed into `qwen3_5.cpp`,
which is the hand-rolled parallel path the shared-seam rule forbids.

`layers::Fp8BlockLinearMethod::Apply` is one line over that body, instantiated
with the shared glue. It is the seam a model that binds `LinearMethodBase`
reaches; see `## Owed` for what binds it today and what does not.

### The device guard, and why it is not the per-tensor one

`vt::MatmulFp8BlockScaled` is registered for the CPU only — M5 owns the CUDA
kernel. The body therefore asks the op table, exactly as `MatmulFp8CutlassD`
asks it for `kMatmulFp8CublasLt`:

```c++
VT_CHECK(vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, d.q.device.type) &&
         vt::OpRegistered(vt::OpId::kQuantFp8Group, d.q.device.type), ...);
```

Both ops, not one. `vt::QuantFp8Group` has a CUDA arm and
`vt::MatmulFp8BlockScaled` does not, so a check on the quant alone would pass on
CUDA and fail one call later with a message about the wrong op. The message
names the device, says the CPU reference arm is what exists, and quotes M5.

### The refusal moves from "loaded" to "unrunnable on this device"

M3 put `RefuseUnconsumedQwen3_5DenseFp8Block` at `PrepareQwen3_5Dense` because
nothing could read the weight anywhere. That is no longer true, and deleting the
gate outright would be worse than narrowing it: a CUDA user would then load a
27B checkpoint, prepare it, capture a graph, and only discover at the first GEMM
that no kernel exists.

`RefuseUnrunnableQwen3_5DenseFp8Block(weights, queue)` keeps the same call site
and the same per-projection naming, and refuses only when the PREPARE queue's
device has no block-scaled GEMM. On a CPU queue it is inert and the model runs;
on CUDA it fires before any compute, names the projection and the device, and
quotes M5. `ModelRegistry::Prepare` receives the queue already
(`qwen3_5_dense.cpp:104`), so this costs no new plumbing.

### Output dtype at every call site

Upstream's block linear emits `torch.get_default_dtype()` — bf16 for this
checkpoint — at every projection, so **the block arm emits bf16 wherever the
site's own bf16 arm does**, and nothing here is f32. The two sites where the
per-tensor fp8 arm beside it uses f32 are called out, because a token gate
cannot see a dtype that is too wide (`.agents/porting.md`) and "I copied the arm
next to me" is exactly how that happens.

| Call site | Block arm emits | Why |
|---|---|---|
| `ProjectFullAttnQkv`'s `project` (q/k/v) | `kBF16` | upstream's `out_dtype`; the same dtype the plain-bf16 arm two lines down emits. The per-tensor fp8 arm's `kF32` beside it is that arm's own pre-existing choice, not upstream's, and this row does not adopt it |
| `SigmoidGateOProjD` (o_proj) | `kBF16` | every arm at this site is already bf16 |
| `ProjectGdnQkvz` (`in_proj_qkv`) | `indt` | `GdnInDType()`, bf16 by default, is the site's resolved model dtype and what its bf16 arm emits. The split per-tensor fp8 arm hardcodes `kF32` and says so; the block arm does not inherit that. `GdnProjectedMixedQkvDType` predicts `in_dtype` for a checkpoint with neither a bf16 `qkvz` owner nor an fp8 one, so the packed-decode predictor and the projection agree by construction |
| `ProjectGdnQkvz` (`in_proj_z`) | `outdt` | `GdnOutDType()`, the dtype the gated RMSNorm's gate must carry |
| `GdnOutProjD` (`out_proj`, 3 call sites) | `kBF16` | every arm at this site is already bf16 |
| `DenseMlpBlock` (gate, up) | `kBF16` | upstream's `out_dtype`. The split legacy bf16 arm here emits f32; `vt::MoeSiluMul` is templated on its input dtype and the fp4 arm already feeds it bf16 under `VT_BF16_GEMM_OUT`, so bf16 is both upstream's answer and a supported one |
| `DenseMlpBlock` (down) | `kBF16` | the block's return type |

### Ten projections, one rung each

`Qwen3_5ForConditionalGeneration` quantizes ten projections per layer pair, and
M3's loader fills a block slot for each. Each gets a probe placed FIRST, before
the fp4 and per-tensor fp8 probes, mirroring the loader's own rung order: a
block-wise weight is `F8_E4M3` on disk and the per-tensor probe would otherwise
claim it, which is #1166 seen from the forward's end.

The three GDN `out_proj` tails were three byte-identical copies of the same
three-way conditional in `GdnBlock`, `GdnBlockPagedMixedSpec` and
`GdnBlockPaged`. Adding a fourth arm to each copy is how a parallel path starts,
so they are replaced by one `GdnOutProjD` helper carrying all four arms. The
existing three arms are moved verbatim; only the block probe is new.

### What the shared seam could not represent, and what it could

Nothing. The seam fits: `LinearMethodBase::Apply(Dev, const Tensor&, DType)` is
exactly the signature this scheme needs, because a block weight carries its own
`block_n`/`block_k` and there is no per-call scalar to thread. No exception is
recorded, and none is needed.

### Ragged edges

`vt::MatmulFp8BlockScaled` tiles with `cdiv` on both axes and M2 gates the
ragged grid. `vt::QuantFp8Group` refuses `K % group_size != 0`, and that
asymmetry is upstream's own: `fp8_utils.py:596-599` asserts divisibility on the
ACTIVATION while `fp8_utils.py:930-936` uses `cdiv` on the weight. The body
refuses a `K` the activation quant cannot take, by name, before allocating
anything, rather than letting `vt::QuantFp8Group` refuse it one frame deeper
with a message that does not name the projection.

## Risks

| Risk | Control |
|---|---|
| the block arm is wired but nothing selects it, so the capability is dead | G3 drives `ModelRegistry::Forward` on a loaded block-wise checkpoint and asserts the per-GEMM dispatch counter advanced by exactly the projection count; the reachability mutation deletes the production call site and reds it |
| the linear method is a class the test constructs and no forward reaches | G1 asserts selection through `MakeLinearMethod`; the CAPABILITY's reach is asserted by G3 through the registry. The class's own binding status is stated under `## Owed` rather than implied |
| a silent dequant to bf16, which is numerically BETTER and invisible to a correctness gate | G4 asserts the resident weight is exactly `N*K` fp8 bytes and the scale exactly `cdiv(N,128)*cdiv(K,128)*4`, at the GEMM boundary; a bf16 expansion is `2*N*K` |
| the per-block scale collapses to a per-tensor one | G5 perturbs ONE block's scale by x1.10 and requires the model's logits to move; the fixture's scales already vary across blocks, so a `scale[0][0]` implementation cannot pass G2 either |
| an all-zero output passes every value comparison | G2 and G3 carry `nonzero == numel` vacuity guards |
| the output dtype widens to f32 where upstream is bf16 | G2 and G3 assert the emitted dtype at the method boundary and at each wired site's owner; the table above is the per-site record `.agents/porting.md` requires |
| a CUDA user loads, prepares, captures a graph, and only then finds no kernel | the narrowed `Prepare` refusal, keyed on the prepare queue's device; G6 asserts it by name for a device with no registration |
| the per-tensor fp8 and bf16 arms regress | G7 is the negative control: the same fixtures through the same forward, unchanged, plus the whole existing dense suite |
| `K % block_k != 0` reaches `vt::QuantFp8Group` and is refused without naming the projection | refused in the body first; G8 asserts the message names the projection and the two numbers |

## Tests

`tests/vllm/model_executor/models/test_fp8_block_linear.cpp`, registered in
`tests/CMakeLists.txt`, plus one case appended to
`tests/vllm/model_executor/layers/test_linear_method.cpp` and G7 of
`tests/vllm/model_executor/models/test_fp8_block_weight_load.cpp` rewritten
against the narrowed refusal.

The fixture is M3's: a complete but tiny synthetic
`Qwen3_5ForConditionalGeneration` safetensors checkpoint written to a temp
directory, extended with the geometry a FORWARD needs. No download, no GPU, no
snapshot.

- **G1** selection: `layers::MakeLinearMethod(bf16_empty, block_weight)` returns
  a method whose `Name()` is the block scheme, and the same factory with an
  EMPTY block weight returns the unquantized one. A `dynamic_cast` pins the
  concrete type, so a factory that returned the right name from the wrong class
  fails.
- **G2** numerics: `Fp8BlockLinearMethod::Apply` against an independently
  written `double` reference that spells out upstream's composition — group
  absmax, the divide, the e4m3 round, then the mainloop-scaled accumulate — over
  a weight whose block scales DIFFER across both axes. The vacuity guard and the
  emitted dtype are asserted in the same case.
- **G3** reachability: the block-wise fixture through `ModelRegistry::Load` ->
  `ModelRegistry::Prepare` -> `ModelRegistry::Forward` on a CPU queue. Prepare
  no longer throws; the forward returns finite, non-vacuous logits; and the
  block-GEMM dispatch counter advanced by exactly `projections * layers`. This
  is the case the reachability mutation reds.
- **G4** the memory format at the GEMM boundary: `N*K` fp8 bytes and
  `cdiv(N,128)*cdiv(K,128)*4` scale bytes, read off the loaded weight, with the
  bf16-expansion size named in the message.
- **G5** the per-block scale probe: the same fixture with ONE block scale
  multiplied by 1.10 produces different logits. A per-tensor collapse, an
  epilogue-folded alpha and a transposed scale index each fail it.
- **G6** the narrowed refusal: a prepare queue whose device has no block-scaled
  GEMM refuses by name, quoting the projection, the device and M5; a CPU queue
  does not.
- **G7** the negative controls: the per-tensor fp8 and bf16 fixtures load,
  prepare and forward exactly as before, and the block counter does NOT advance.
- **G8** the refusals the body owns: a `K` not divisible by `block_k`, and a
  weight whose `k` disagrees with the activation's, each named.

## Gates

| Gate | Command |
|---|---|
| focused | `ctest -R test_fp8_block_linear --output-on-failure` |
| the narrowed M3 refusal | `ctest -R test_fp8_block_weight_load --output-on-failure` |
| the seam | `ctest -R test_linear_method --output-on-failure` |
| the M1/M2 ops, unchanged | `ctest -R "test_ops_quant_fp8_group_cpu\|test_ops_matmul_fp8_block_cpu"` |
| the dense forward, unchanged | `ctest -R "test_qwen27_paged_forward\|test_qwen35_paged_forward\|test_qwen27_dense_forward\|test_model_registry"` |
| record | `scripts/agent-preflight.sh --fail-on-skip` |

No GPU lease is taken and none is needed. The CPU reference arm from M2 is what
makes this gateable without hardware, and it is also the reason no throughput
number appears anywhere in this row.

## Owed

- **An f32 KV cache refuses this arm, and every other bf16 arm**
  ([#1249](https://github.com/mudler/vllm.cpp/issues/1249)). This method emits a
  bf16 `V` at `v_proj`, which is upstream's `out_dtype` and the model dtype, and
  `vt::ReshapeAndCache`'s auto path requires `k`/`v` and the cache to share ONE
  dtype — mirroring upstream's own `reshape_and_cache_flash`, which torch types
  identically. So `VT_KV_CACHE_F32=1` refuses this arm. It refuses every bf16
  arm in the tree the same way, so the defect is older and wider than this row
  and closing it is a `vt` semantic change with its own spec and red-first test,
  not a widening of M4. G3 therefore drives the PRODUCTION bf16 cache and says
  in the source that it cannot drive the f32 one, rather than passing quietly on
  a configuration nothing gates.
- **There is no GPU kernel and there is no token gate.** What this row
  establishes is that the composition is correct against upstream's own
  reference and that a production entry point reaches it on a CPU queue. It does
  NOT establish that `Qwen/Qwen3.8-27B-FP8` produces vLLM's tokens, because the
  arm that would run it does not exist: `vt::MatmulFp8BlockScaled` is a CPU
  correctness reference that makes no speed claim, and #1189 **M5** owns the
  mainloop-scaled CUTLASS kernel for `sm_121a`. Until M5 lands, a CUDA device
  refuses this checkpoint by name at `Prepare`. Nothing here is
  production-gated.
- **No model binds `layers::Fp8BlockLinearMethod` yet.** The CAPABILITY is
  reached — `ModelRegistry::Forward` -> the Qwen3.5 dense forward ->
  `dense_fp8_block::MatmulFp8BlockScaledD` — and the linear method is the
  `LinearMethodBase` policy layer over that one body. Its own binding status is
  identical to its per-tensor sibling `Fp8W8A8LinearMethod` (#940,
  `.agents/specs/vt-fp8-shared-seam.md`), which no production model binds
  either: the models that route through `LinearMethodBase` (`qwen3.cpp`,
  `qwen3_vl.cpp`, `voxtral.cpp`) carry no `Fp8BlockWeight` field, and giving
  them one is a loader change those rows own. This is the staged-slice exception
  of `.agents/reachability.md`, named here, in the commit body and in the pull
  request body.
- **Merged `gate_up` and QKV.** The MLP runs three separate GEMMs and the QKV
  runs three more, where upstream runs one `MergedColumnParallelLinear` and one
  `QKVParallelLinear`. Block scales concatenate losslessly along N, so this is
  simpler here than in the per-tensor case, and #1189 **M6** owns it.
- **The column-major and TMA-aligned activation-scale layouts.**
  `CutlassFp8BlockScaledMMKernel` constructs its `QuantFP8` with
  `column_major_scales=True` (`kernels/linear/scaled_mm/cutlass.py:340-345`);
  M1 shipped the row-major layout only and recorded the other two as owed. The
  CPU reference reads row-major, so nothing here needs them; M5 does.
- **`process_fp8_weight_block_strategy`**
  (`BlockScaledMMLinearKernel.py:89-95`) runs at
  `process_weights_after_loading` and may pad or re-lay-out the weight for a
  particular kernel. M3 keeps the checkpoint's `[N,K]` bytes verbatim and the
  CPU reference reads them directly, so no transform is needed on this arm. M5
  owes whichever one its kernel wants.
- **The MoE and GGUF loaders** are untouched, as M3 left them. No block-wise MoE
  or GGUF checkpoint is in play for #1189.

## Stop conditions

Stop and report `NEEDS_DECISION` if any of the following holds.

- The pinned oracle's checkout is not at
  `5559679229bc961848b121ccdeaa8fa5d79bec98`. Every anchor above was read at
  that revision, asserted before the first read.
- The shared `LinearMethodBase` seam cannot represent the scheme. It can, and
  §Design says why; if that turns out to be wrong, the answer is to EXTEND the
  seam or record one exact tracked exception, never to write a second path.
- A wired call site cannot emit upstream's `out_dtype` because a downstream
  consumer refuses bf16. Widening that site to f32 is a divergence a token gate
  cannot see, so it is a decision rather than an implementation detail.

Stop and report `NEEDS_CONTEXT` if the work requires a GPU lease or a checkpoint
download. The row is scoped so that it needs neither.

## Evidence

Filled in when the row lands.

## Now

`ACTIVE` — M4 of #1189. M1 (`ad5f175e7`), M2 (`770e49486`) and M3
(`09597106e`) are `DONE`; M5 and M6 are open.
