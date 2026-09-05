# QUANT-EXL3 — EXL3 as a quantization scheme every architecture can reach

Row: `QUANT-EXL3`
Issues: [#2181](https://github.com/mudler/vllm.cpp/issues/2181) (primary)
Base SHA: `bca11d03d`
Matrix: [`.agents/quantization-matrix.md`](../quantization-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements
no EXL3** (`layers/quantization/` registers no exl3/exllamav3/trellis method at
the pin), so the format is mirrored from the registered secondary oracle
[`exllamav3`](../oracles/exllamav3.md) @ `2398c05635fbbad01a0a51dce63c85c6c8a8450e`
(tag `v1.4.3`, MIT). **The SEAM is vLLM's**, and that half is not a fallback
case: `layers/quantization/base_config.py:87-180` (`QuantizationConfig` +
`get_quant_method(layer, prefix)`) and `layers/linear.py:141-181`
(`LinearMethodBase.create_weights` + `apply`) define where a scheme plugs in,
and this row mirrors them. Where vLLM defines structure, vLLM wins; exllamav3
supplies only the trellis format and its kernels.

## Now

`ACTIVE`. W1a and W1b landed and EXL3 runs a model. **W3 is in flight: the
device arm was instantiated for ONE `(bits, codebook)` pair and it was the wrong
one.**

`cuda_exl3.cu` carried `kInstantiatedBits = 3, kInstantiatedCb = 1`. Codebook 1
is the SparkInfer DeepSeek-V4 artifact -- the EXCEPTION -- so every stock
`turboderp/*-exl3` checkpoint refused on the device one projection at a time and
fell to a single-threaded CPU decode. That is the whole of the 0.040 tok/s the
W1b run measured: 1,235,746,816 weights re-decoded per token at ~50M/s on one
core, because the trellis is decoded inside the GEMM and the GEMM never reached
the GPU.

W3 instantiates three arms -- `(3, 0)` a stock body, `(3, 1)` DeepSeek-V4,
`(6, 0)` the stock 6-bit `lm_head` -- which needed real porting rather than a
wider list: `decode_3inst_2` had `static_assert(cb == 1)` and `dq_dispatch` had
`static_assert(bits == 3)`, and bits 6 needs `dq4` because `dq8` spans
`16 + bits*7` bits across the two words it merges and overflows the 64-bit
funnel at 6 bits (upstream routes 5/6/8 through `dq4` for that reason,
`exl3_dq.cuh:274-293`).

**What this row can and cannot reach.** The speed target named for this work is
`MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark`: 44-47 tok/s decode at 384k context
on one GB10. Read from its README, that number is EXL3 weights **plus** DSpark
K5 speculative decoding with a K64 draft, **plus** an `nvfp4_ds_mla` compressed
KV cache, **plus** the `B12X_MLA_SPARSE` sparse-attention backend, at
`MAX_NUM_SEQS=1` and util 0.94. Only the first of those four is this row's. The
sparse DSA attention is unported and owned by NO row (#1961, #1970, #1976), the
compressed KV topology is `KV-DSV4-MULTICACHE`'s W5, and the residency that
stops the artifact loading at all is #2186. This row makes EXL3 fast; it does
not by itself make that model fast, and no number here should be read as
approaching theirs.

## The gap, measured

`grep -rl Exl3 src/vllm include/vllm` returns three files, all DeepSeek-V4
(`models/deepseek_v4.cpp`, `models/deepseek_v4_weights.cpp`,
`include/vllm/model_executor/models/deepseek_v4.h`).
`.agents/quantization-matrix.md` carries no EXL3 row while registering 20+ other
schemes. `IsExl3Checkpoint` (`deepseek_v4_weights.cpp:229-233`) reads the same
`quantization_config.quant_method == "exl3"` marker every EXL3 checkpoint
carries, and is consulted only from the DeepSeek-V4 loader.

**The kernels are ready and device-proven.** `vt::Exl3HadR128`, `vt::Exl3Gemm`,
`vt::Exl3MoeMlp` and `src/vt/cpu/cpu_exl3_dequant.cpp` exist and passed their
device gates on `dgx:gpu0` (GB10 `sm_121a`, driver 580.173.02, nvcc 13.0.88,
tree `525d2b991`, 2026-08-28, worker `rc-worker-4b8lj`): `had_r_128` CUDA-vs-CPU
`mismatches == 0`; `exl3_gemm` vs f64 `rel_rms 5.538e-4` (bound `1.0e-3`, worst
`0.0334` against 8·ulp `0.0625`); GEMV tier 3c `rel_rms 5.160e-4` (bound
`6.0e-3`). `cuda_exl3.cu.o` carried one `sm_121a` cubin. Everything missing is
ABOVE the kernels.

## The stock layout, measured rather than assumed

Range-read from the safetensors header of `turboderp/Llama-3.2-1B-Instruct-exl3`
revision `3.0bpw` (`f8f438c290680b15622270eff03bef23a458b1cf`) on 2026-08-28 —
373 tensors, one 1.09 GB file, header 40,368 bytes:

- HF-standard keys with EXL3 fields appended:
  `model.layers.N.self_attn.{q,k,v,o}_proj.{trellis,suh,svh}`,
  `model.layers.N.mlp.{gate,up,down}_proj.{trellis,suh,svh}`,
  `lm_head.{trellis,suh,svh}`.
- `trellis` `I16 [k/16, n/16, 16*bits]`, exactly what `Exl3ReconstructInner`
  reads: `mlp.gate_proj.trellis [128, 512, 48]` is k=2048, n=8192, bits=3.
- `suh` `F16 [k]`, `svh` `F16 [n]` — `mlp.down_proj.suh [8192]` /
  `.svh [2048]` confirms `suh` is the INPUT side and `svh` the OUTPUT side on
  every projection, which is what `Exl3DequantLinear` assumes.
- Norms and `model.embed_tokens.weight` stay `F16`, unquantized.
- **No `.rank{r}` segments.** The rank-sliced `rank-sliced-deepseek-v4-v1`
  schema `MODEL-DSV4-EXL3` W1b implements is SparkInfer's variant, not the
  format's ordinary shape; this row's reader is the simpler one.
- `quantization_config = {quant_method: "exl3", version: "0.0.0", bits: 3.0,
  calibration: {rows: 100, cols: 2048}}`.

**`bits` is PER TENSOR, and the config scalar is not it.** `lm_head.trellis` is
`[128, 8016, 96]` — 96 = 16*6, so the head is 6-bit while the body is 3-bit and
`quantization_config.bits` says `3.0`. Every consumer derives `bits` from the
`trellis` last dimension divided by 16, and the config value is used only to
cross-check the modal case. A reader that trusts the config scalar decodes the
head at the wrong width and produces garbage that no shape check catches, since
the tensor's shape is self-consistent at either reading.

**No `mcg` tensor ships in this checkpoint.** The DSV4 artifact carries a per
linear `mcg` int32 codebook marker; this one does not, and upstream's
`Linear.is_exl3_storage` requires only `{key}.trellis` with `suh|su` and
`svh|sv` (`modules/linear.py:385-389`). The codebook therefore defaults to MCG
(`cb == 1`), which is `LinearEXL3`'s own default, and a checkpoint that ships a
marker naming anything else REFUSES BY NAME rather than being decoded as MCG.

## Scope, in waves

**W1 — the scheme, the reader, and one model end to end (this spec's first
dispatch).**
- `layers/quantization/exl3.{h,cpp}`: an `Exl3LinearMethod : LinearMethodBase`
  whose `Apply` is `vt::Exl3HadR128` in, `vt::Exl3Gemm`, `had_r_128` out, and an
  `Exl3Config` recognized from `quantization_config.quant_method == "exl3"`,
  mirroring `get_quant_method`.
- A native-layout reader keyed on the three sibling tensors, beside the
  rank-sliced arm rather than replacing it.
- The shared dense container and forward (`Qwen3DenseWeights`, the Qwen3-dense
  `AttnBlock` path that `LlamaForCausalLM` reuses verbatim) gain the EXL3 arm,
  so Llama and Qwen3-dense both reach it from one change.
- E2E greedy generation from a production entry point on
  `turboderp/Llama-3.2-1B-Instruct-exl3`.

**W2 — residency.** A device-resident destination for the trellis tower.
`MODEL-DSV4-EXL3` `## Owed` already needs this for its host-residency refusal;
a 1.09 GB checkpoint makes it testable without the 100 GB artifact, and it is
the precondition for `vt::Exl3MoeMlp`'s device arm, which skips today because
`CudaBackend::DeviceMemoryIsHostAddressable()` is false by design
(`cuda_backend.cu:330-366`, #1635).

**W3 — width coverage.** The CUDA arm instantiates `bits == 3, codebook == 1`
only. A stock checkpoint's 6-bit head has no device arm. Either widen the
instantiation set the way upstream splits it
(`comp_units/exl3_comp_unit_K_cbX.cu`) or route the head to the generic CPU arm
and say so at the refusal.

**W4 — DeepSeek-V4 routes through this seam.** `MODEL-DSV4-EXL3`'s private
`Exl3Linear` becomes a caller of the shared method. Sequenced last because
#1875's blockers are its DSA composition and its residency, neither of which
this row's seam changes.

## Dependencies

| Depends on | State | Effect if it moves |
|---|---|---|
| `vt::Exl3Gemm` / `Exl3HadR128` / `Exl3MoeMlp` (`MODEL-DSV4-EXL3` W2) | LANDED, device-proven | none — this row consumes them and changes nothing in them |
| `vt::CastF16` | LANDED here (W1a) | its four missing backend arms fall due with W1b |
| the shared dense container + forward (`Qwen3DenseWeights`, the Qwen3-dense `AttnBlock` that `LlamaForCausalLM` reuses) | owned elsewhere | W1b ADDS an arm to it; if that container is restructured, W1b rebases onto the new shape |
| `exllamav3` as a gateable oracle (#1901) | `gateable = no`, does not build on aarch64 | blocks the oracle token gate ONLY; every gate this row states is reachable without it |
| a device-resident trellis tower | OWED (W2, and `MODEL-DSV4-EXL3` `## Owed`) | blocks the `Exl3MoeMlp` device arm and any large EXL3 checkpoint |
| nothing in `MODEL-DSV4-EXL3`'s DSA work (#1961, #1970, #1976) | open | **no dependency either way** — that row's blockers are attention and residency, not the scheme, which is why this row can finish while that one cannot |

## Work breakdown

Dependency order, and what can run in parallel. The content of each wave is in
`## Scope, in waves` above; this is the sequencing.

| Wave | Needs | Can run beside | Gateable on |
|---|---|---|---|
| W1a — the scheme, the method, `CastF16` | nothing | — | CPU, no checkpoint. **LANDED** |
| W1b — the native reader, the dense arm, one model e2e | W1a | W3 | CPU + a 1.09 GB checkpoint |
| W2 — device residency | W1b (for a caller) | W3 | a GPU lease |
| W3 — width coverage (the 6-bit head) | W1a | W1b, W2 | CPU for the refusal, a GPU for the arm |
| W4 — DeepSeek-V4 routed onto this seam | W1b, W2 | — | that row's own gates |

W1b is the critical path and the only wave that turns a class into a
capability. W3 is independent of it and is the one wave a second agent could
take in parallel without touching W1b's files.

## Upstream chain

The two halves come from different places, which is the whole shape of this row.

| Piece | Source | Anchor |
|---|---|---|
| the scheme seam | **vLLM** (primary) | `layers/quantization/base_config.py:87-180` (`QuantizationConfig`, `get_quant_method`); `layers/linear.py:141-181` (`LinearMethodBase`) |
| the local seam it mirrors | this tree | `include/vllm/model_executor/layers/quantization/base_config.h`; `include/vllm/model_executor/layers/linear.h:43` |
| the trellis format | **exllamav3** (secondary, pinned `2398c056`) | `modules/quant/exl3.py:16-40` (the owned tensors), `:183-214` (the runtime form), `:227-237` (the reconstruction form) |
| the codebook | exllamav3 | `exllamav3_ext/quant/codebook.cuh:67-75` (`decode_3inst`, cb == 1) |
| the codeword window | exllamav3 | `exllamav3_ext/quant/exl3_dq.cuh:15-31` |
| the storage predicate | exllamav3 | `modules/linear.py:385-389` (`is_exl3_storage`: `trellis` + `suh\|su` + `svh\|sv`) |

vLLM registers no EXL3 at the parity pin, which is what admits the secondary
oracle for the format. It does NOT admit one for the seam, and the seam is
mirrored from vLLM.

## Our baseline

The kernels already exist and are gated; this row consumes them and adds
nothing to them.

| Piece | Where | State |
|---|---|---|
| CPU reference dequant | `src/vt/cpu/cpu_exl3_dequant.cpp` | gated (`test_exl3_dequant` 3/3, 66 assertions) |
| `Exl3Gemm`, `Exl3HadR128` CPU | `src/vt/cpu/cpu_exl3_kernels.cpp` | gated (`test_exl3_gemm` 13/13, 199) |
| the CUDA arm | `src/vt/cuda/cuda_exl3.cu` | compiles `sm_121a`; `had_r_128` byte-identical, `exl3_gemm` `rel_rms 5.538e-4`, GEMV tier 3c `5.160e-4`, GB10 2026-08-28 |
| `Exl3MoeMlp` device arm | same | UNRUN — needs a device-resident tower (W2) |
| the shape policy | `src/vt/exl3_policy.cpp` | gated, host-side |
| the only consumer | `src/vllm/model_executor/models/deepseek_v4.cpp` | model-private, which is the gap this row closes |

The scheme's own baseline is empty: no `QUANT-EXL3` row existed before #2181 and
no architecture but DeepSeek-V4 could reach the format.

## Port map

| Upstream | Ours | Wave |
|---|---|---|
| `LinearMethodBase.apply` | `layers::Exl3LinearMethod::Apply` (`quantization/exl3.h`) | W1a — LANDED |
| `get_quant_method` | `layers::MakeLinearMethod(const OwnedTensor&, const Exl3Weight&)` | W1a — LANDED |
| the fp16 activation the format assumes | `vt::CastF16` (`ops.h`, CPU + CUDA) | W1a — LANDED |
| `Linear.is_exl3_storage` / `load_exl3` | the native-layout reader, no rank segment | W1b — OWED |
| the dense container's quantized arm | `Qwen3DenseWeights` + the Qwen3-dense forward that `LlamaForCausalLM` reuses verbatim | W1b — OWED |
| `LinearEXL3.tp_import_split` | NOT ported: the stock layout is not TP-sliced, and the rank-sliced arm already exists on `MODEL-DSV4-EXL3` | out of scope |

## Tests to port

exllamav3's own suites are the source, adapted only where the harness forces it.

| Upstream test | Ours | State |
|---|---|---|
| `tests/test_quant_fn.py:83-116` (the tail-biting window and its reference) | `tests/vt/test_exl3_dequant.cpp` | ALREADY PORTED by `MODEL-DSV4-EXL3` W1a |
| the reconstruct-vs-runtime identity (`exl3.py:183-214` vs `:227-237`) | `test_exl3_linear_method.cpp`, `Apply` vs the weight-side dequant at 2.0e-3 | W1a — LANDED |
| no upstream test covers per-tensor `bits` | ours is NEW, and it has to be: upstream reads the width from the tensor everywhere and never had the config-scalar trap to guard | W1a — LANDED |
| an end-to-end generation | `turboderp/Llama-3.2-1B-Instruct-exl3` through a production entry point | W1b — OWED |

Upstream's kernel tests are CUDA-only and unrunnable here while `exllamav3`
records `gateable = no` (#1901); that is recorded as debt rather than adapted
into something weaker wearing the same name.

## Design

**Why a `LinearMethodBase` rather than a fourth field on the dense container.**
`include/vllm/model_executor/layers/linear.h:43` already mirrors
`LinearMethodBase`, and `base_config.h` records why it exists: scheme selection
used to be a per-model tensor-name probe with device gates scattered through
forwards, and this seam is the removal of that tangle. Adding EXL3 as another
inline branch in a model forward would rebuild exactly what that row deleted.

**Why the dense hot path stays byte-identical.** Only Gemma and dots3 consult
the seam today; the dense Qwen3/Llama forward calls `vt::MatmulBT` inline. W1
does NOT migrate that path onto method dispatch. It adds a branch taken only
when the loaded weights carry an EXL3 arm, so a bf16 checkpoint executes the
same instructions it does today. Migrating the dense path onto the seam is a
separate decision belonging to whoever owns that forward, and this spec does not
make it.

**Output dtype.** The dequant reference carries fp16-valued data in `float`
(`MODEL-DSV4-EXL3` risk 5). This row's destination is the model dtype the
checkpoint declares (`torch_dtype: bfloat16` for the Llama artifact), never f32
inherited from a reference signature. `AGENTS.md` §"Inherit vLLM defaults": a
token gate cannot see a dtype that is too wide.

## Risks

1. **The gate is the hard part, not the code.** vLLM has no EXL3 and the
   secondary oracle does not build on aarch64 (#1901), so a token-exact
   comparison against an oracle EXL3 run is unavailable on this fleet. See
   `## Gates`; the bound is chosen BEFORE the gate runs, never widened after a
   red.
2. Per-tensor `bits` (above). A config-scalar reader is silently wrong on the
   head.
3. The 6-bit head has no device arm, so W1's e2e run may be part-host. That is
   recorded as a measurement, never hidden by falling back silently.
4. `tie_word_embeddings` is FALSE in this artifact while the bf16 Llama-3.2-1B
   ties them — the EXL3 repo ships a real quantized `lm_head`. The loader must
   not apply the bf16 path's `skip_prefixes(["lm_head."])` to an EXL3 load.

## Tests

Red first, in this order:

1. `tests/vllm/model_executor/layers/test_exl3_linear_method.cpp` — the method's `Apply`
   against `vt::Exl3DequantLinear` + a dense GEMM on the same fixture, and
   `bits` resolved from the tensor rather than the config (a fixture whose
   config says 3 and whose tensor says 6 must decode at 6).
2. `tests/vllm/models/test_exl3_native_loader.cpp` — a hermetic native-layout
   checkpoint (no rank segments) loads into the dense container; a missing
   `svh` REFUSES BY NAME; a non-MCG marker REFUSES BY NAME.
3. Reachability: the e2e case drives `LoadLlamaForCausalLMWeights` and
   `ModelRegistry::Forward`, never a hand-built struct. Deleting the production
   call site must go RED — the #1923 failure in its exact shape is what that
   guards against.

## Gates

| Gate | Owner |
|---|---|
| W1: method vs the W1a dequant reference on one fixture, within a stated bound | implementer |
| W1: per-tensor `bits` — a 6-bit tensor under a 3-bit config decodes at 6; mutating the reader to trust the config goes RED | implementer |
| W1: the native reader refuses a missing sibling and a foreign codebook BY NAME | implementer |
| W1: a real EXL3 Llama checkpoint GENERATES from a production entry point | operator |
| W1: deleting the production call site goes RED | implementer/reviewer |
| W1: a bf16 Llama load is BYTE-IDENTICAL to its pre-change logits | implementer |
| W2: the trellis tower is device-resident; the MoE device arm stops skipping | implementer |
| W3: the 6-bit head has a device arm, or refuses by name | implementer |

**What the correctness gate CAN bind, since an oracle token match cannot.**
Stated here before code, per risk 1:

- **Reference-model agreement.** The same prompts through the BF16
  `Llama-3.2-1B-Instruct` we already gate token-exact 16/16 vs vLLM, and through
  the EXL3 3.0bpw quant of that same model. Quantization changes tokens, so this
  is NOT a token gate: it is a bounded divergence gate on the logit
  distribution, with the bound stated before the run.
- **Self-consistency.** Our EXL3 dequant-to-dense reconstruction vs our EXL3
  native compute on the same weights — a real gate on the compute path, and the
  one place a token-exact bound IS available.
- **Coherence.** A greedy continuation that is readable English is a weak gate
  and is recorded as weak, never as a pass.
- The oracle gate is OWED and blocked on #1901; when exllamav3 builds on
  aarch64 it becomes the token oracle and this section is replaced, not
  supplemented.

## W5: the shape table's other three kernels ([#2749](https://github.com/mudler/vllm.cpp/issues/2749))

`Exl3SelectGemmShape` (`src/vt/exl3_policy.cpp:65-114`) chooses one of four
kernel shapes, and `GemmKernelForArm` (`src/vt/cuda/cuda_exl3.cu:2022-2039`)
instantiates all four for each of the seven `(bits, codebook)` arms. **One of
the four has ever executed.** The committed device cases run `k 512 n 256` and
`k 256 n 256` at `bits 3`, which on a Blackwell-class `cc` take
`mod_256 && size_n <= 4096` with `size_k > 8192` false and return shape 2.
Shape 4 is not merely unselected at `n = 256`, it is refused: it tiles `n` by
512 and `Exl3GemmShapeCompat` rejects that `n`. Twenty-one instantiations
therefore compile into the fat build for every architecture and ship untested.

The table is this row's surface -- `## Port map` lists `src/vt/exl3_policy.cpp`
here -- so W5 owns the coverage. `quant-exl3-mul1.md` and `quant-exl3-perf.md`
each carried a repeated bullet for it, and each pointed at the other; both go
with this slice.

### Why this is a test loop and not a port

`Exl3GemmArgs::force_shape_idx` already exists and the launcher already honours
it (`cuda_exl3.cu:2270`), mirroring upstream's own `force_shape` parameter. No
product code changes. What changes is that the gate stops measuring one kernel
four times.

### The discrimination problem, and what actually settles it

A forced shape and the selector's shape compute the same product, so **no
comparison against a reference can tell a honoured force from a dropped one.**
The obvious structural discriminator is unavailable too, and it is worth writing
down why so nobody re-derives it: `vt::Exl3Gemm` runs `had_r_128` over `A[m, k]`
and over `C[m, n]`, and that transform refuses a row length that is not a
multiple of 128. So `k % 128 == 0` and `n % 128 == 0` hold on every legal call,
which makes shape 1 (`k % 16`, `n % 128`) and shape 2 (`k % 32`, `n % 128`)
compatible with every legal call, and makes the selector's own answer compatible
in every branch it has. There is no legal `(k, n)` at which a forced shape
succeeds and the selected shape would have refused.

Two things settle it instead, and the gate carries both.

1. **The refusal proves the field is READ.** Forcing shape 4 at `n = 768`
   (`768 % 512 == 256`) must throw `no compatible exl3_gemm kernel shape`,
   because `Exl3GemmShapeCompat(4, k, 768)` is false. A launcher that ignored
   `force_shape_idx` would select shape 3 there -- `768 % 256 == 0` -- and
   return numbers. This case is deterministic and needs no tolerance.

2. **Byte inequality across shapes proves each kernel is a DIFFERENT one.** The
   four shapes decompose the same problem into different `(k, n)` tile grids,
   so `Exl3GemmNumSms` gives them different block counts and the split-K
   reduction over `k` runs a different tree. At `k 256 n 512` the slice counts
   are 64, 32, 16 and 16, and shapes 3 and 4 reach their equal count by
   different routes -- 2 n-tiles of 8 k-tiles against 1 n-tile of 16 -- so the
   per-output reduction chains differ in length. The output is taken in **f32**
   precisely so an fp16 store cannot absorb that difference. If two forced
   shapes returned byte-identical outputs the case would be measuring one
   kernel twice, which is the same failure the `(3,1)`/`(3,2)` discrimination
   check in `test_exl3_gemv.cpp` exists to catch.

### Gates (W5)

| Gate | Owner |
|---|---|
| each of shapes 1, 2, 3, 4 FORCED and agreeing with the CPU arm at `rel_rms <= 1.0e-3` | implementer |
| forcing shape 4 at `n = 768` REFUSES by name | implementer |
| no two forced shapes return byte-identical output | implementer |
| a forced index changed to a wrong one goes RED | implementer/reviewer |

Device, not host: `cuda_exl3.cu` does not compile in a CPU build, so a green CPU
preflight says nothing here and is never reported as a device result.

### Evidence (W5) -- PASSED on `thor:gpu0`, 2026-09-03

Taken inside an `rc` lease, never over `ssh`. Job `1254e1e6-09a3-4749-b1d3-4740c7f52b65`,
harness `/workspace/exl3shape-2749/job.sh`
(`sha256 b721a47c20f34c762eb297acfa0b6df89b38d71263f452eaafd55fa2a08cddc6`),
`JOB_EXIT_STATUS=0` after 1695 s.

| | |
|---|---|
| device | `NVIDIA Thor`, `compute_cap 11.0` (so `Exl3CcFromSm` -> `kBlackwell`), driver `595.78` |
| toolchain | `nvcc 13.0`, `-DVLLM_CPP_CUDA_ARCHITECTURES=110`, `RelWithDebInfo`, `-j 4` |
| tree | `39f29b808`, staged as a `git archive` tarball the job verifies by `sha256 f8089fac...` and aborts on any other |
| tree facts asserted before the build | forced-shape case present = 1; launcher reads `force_shape_idx` = 1; four shape instantiations = 4 |
| suite | doctest reported 244 assertions, 244 passed, 0 failed; 16 test cases, 16 passed, 0 failed, 0 skipped; rc 0 |
| the device arm RAN | `SKIPPED, no CUDA device` count = **0**. A device case that skips reads as a pass in a ctest summary, so the count is read, not the rc alone |
| the `-tc` filter selected the case | 1 case selected, `assertions: 21`. A filter that matches nothing prints `assertions: 0` and `SUCCESS!` at rc 0 |

Per shape, forced, against the CPU arm (bound `1.0e-3`):

| forced shape | tile (m, k, n) | rel_rms |
|---|---|---|
| 1 | 16, 16, 128 | `3.2252e-07` |
| 2 | 16, 32, 128 | `3.19297e-07` |
| 3 | 16, 32, 256 | `3.20648e-07` |
| 4 | 16, 16, 512 | `3.21655e-07` |

The discrimination check, all six pairs, of 2048 f32 outputs:
1-2 **1492**, 1-3 **1515**, 1-4 **1548**, 2-3 **1413**, 2-4 **1505**, 3-4 **1444**.
Every pair differs, so four different kernels ran.

The refusal, verbatim:
`vt cuda exl3: no compatible exl3_gemm kernel shape for k=256 n=768 (selected shape 4)`.

### Mutations (W5)

Each row asserts the mutated file's `sha256` CHANGED, that it COMPILED (a build
failure reads as a passing test), that the test binary's mtime MOVED, and that
the tree was restored byte for byte. `FINAL test sha256` and
`FINAL cuda_exl3.cu sha256` both equal their pristine values, and the restored
tree reruns identical to the first gate.

| # | mutation | sha changed | built | mtime moved | rc | verdict |
|---|---|---|---|---|---|---|
| M1 | the TEST drops the force: `force_shape_idx = 2` for all four legs | `f272ae82` -> `64cd5925` | rc 0, 0 errors | `1788436310` -> `1788436364` | 1 | **DETECTED**, 6 failed of 244 |
| M2 | the LAUNCHER drops the force: `shape_idx` comes from `Exl3SelectGemmShape` unconditionally | `765e9ce4` -> `bf70b835` | rc 0, 0 errors | `1788436364` -> `1788436506` | 1 | **DETECTED**, 9 failed of 244 |

**M1 is the evidence that the tolerances alone gate nothing.** All four legs then
printed the IDENTICAL `rel_rms = 3.19297e-07` -- shape 2's number, four times --
and every `rel <= 1.0e-3` still PASSED. All six pairs differed in **0** of 2048
outputs, and the six failures are exactly the six `differing > 0` checks. A gate
built only on agreement with a reference would have called that green.

**M2 is the production defect.** The launcher ignoring `force_shape_idx` fails
the same six checks plus all three refusal checks, because the selector picks
shape 3 at `n = 768` and returns numbers instead of throwing. Tolerances again
all passed.

### Stop conditions (W5)

- Two forced shapes agree byte for byte. The case is not discriminating. Find
  out which kernel ran before touching the assertion; never delete it.
- A forced shape misses the bound. The instantiation is wrong. Never widen the
  bound to admit it.

## Owed

- **W5 forces four shapes on ONE arm.** The gate runs `(bits 3, codebook 1)`.
  `GemmKernelForArm` is a template over `(BITS, CB)` and all seven arms select
  from the SAME four `exl3_gemm_kernel` instantiations by the same `shape_idx`,
  so the shape SELECTION is now covered for every arm; what is not covered is
  each arm's own four instantiations executing. That is 24 more device legs on a
  box whose lease is contended, and it is named rather than done.
- **No `f16` output leg.** The case takes `C` in f32 so that an fp16 store cannot
  absorb the split-K difference the discrimination check reads. `GemmKernelForShape`
  is also instantiated with `c_fp32 = false`, and that half of the table is
  exercised only by the pre-existing shape-2 cases.
- ~~**W1b: nothing constructs `Exl3LinearMethod` yet.**~~ **RETIRED**: the
  dense forward constructs it, and a real checkpoint generates through it.
- ~~**The device arm refuses codebook 0, which is the COMMON case.**~~ **RETIRED
  by W3**: the arm now instantiates `(3,0)`, `(3,1)` and `(6,0)`, and
  `kInstantiatedCb` no longer exists. What replaces it is narrower and real:
  **a stock codebook-0 checkpoint has no GEMV fast path at `m == 1`**, on this
  tree or upstream's — upstream's envelope refuses `bits != 4 && cb == 0` and
  its instantiation list omits `(3,0)`. It takes the regular shape table
  instead, which is upstream's own behaviour rather than a gap.
- **Bits 6 has NO real-data anchor.** `tests/vt/exl3_real_corner.inc` pins
  codebook 0 at 3 bits, so `test_exl3_real_decode` ties the 3-bit arm to real
  exllamav3 output and the 6-bit `lm_head` to nothing but a device-vs-CPU
  cross-check on RANDOM trellis bytes — where, as that fixture's own header
  says, any codebook and any tile permutation is self-consistent. The two
  readers being independent (`Exl3TileCodeword` against `dq4`) makes it a real
  cross-check and not a tautology, but the only end-to-end evidence for the
  6-bit head is a coherence read this spec already records as WEAK.
- **The CPU threading recovers 4.8x of 20 cores, which is ~25% efficiency.**
  Named as an open gap rather than a result, because AGENTS.md forbids
  declaring a ceiling. One hypothesis worth testing first: `raw` is a plain
  `std::vector<float>` and a 16-float stripe is exactly one 64-byte cache line,
  so adjacent workers' stripes can straddle a line whenever the allocation is
  not 64-byte aligned.
- **q/k/v and gate/up run as separate GEMMs.** The bf16 and NVFP4 arms hold ONE
  merged operand; merging trellis operands joins on the output dim, which
  INTERLEAVES per input tile rather than row-stacking. It is valid for this
  family -- `had_r_128` blocks the output in 128s and Llama-3.2-1B's q (2048),
  k/v (512) and I (8192) are each a multiple of 128, so no block straddles two
  matrices -- and it is the merged-GEMM seam this row does not yet reach. Owed
  with its own gate.
- **`vt::CastF16` is registered on two backends where its siblings have six**
  (CPU and CUDA against CPU/CUDA/ROCm/Vulkan/Metal/Tenstorrent). Now REACHED, so
  this is no longer theoretical for a non-CUDA device build.
- **The two codebook resolutions disagree BY CONSTRUCTION, and W4 owns it.**
  `LoadExl3` reads tensor PRESENCE, which is what `LinearEXL3` does;
  `deepseek_v4_weights.cpp` reads the config string
  `quantization_config.codebook`, which is what the SparkInfer artifact happens
  to declare. Both are correct for their own artifact and neither generalizes:
  a stock checkpoint has no such config key, and a rank-sliced one may ship a
  marker its config does not name. Reconciling them onto presence is part of
  routing DeepSeek-V4 through this seam.
- **No speed number.** The e2e run is 0.040 tok/s on a CPU queue at batch 1.
  That is a functional result and is not offered as a performance one.
- **`vt::CastF16` is registered on TWO backends where its siblings have SIX.**
  `kCastBf16` and `kCastF32` are each registered for CPU, CUDA, ROCm, Vulkan,
  Metal and Tenstorrent; `kCastF16` has CPU and CUDA only. The header calls it
  "the third sibling" and that is true of its semantics, not yet of its reach.
  Nothing is broken today, because no production path calls it and
  `RegisterReferenceTier` installs a CPU fallback on host-addressable targets —
  but the four missing arms fall due with W1b, which is what makes the op
  reachable. Named here because a gap nobody wrote down is the one discovered by
  a red gate on somebody else's row.
- ~~**The CUDA arm of `vt::CastF16` has not been compiled or run.**~~
  **RETIRED 2026-08-28**, before this row's first merge rather than after it.
  Measured in an `rc` lease on `dgx:gpu0` (GB10, worker `rc-worker-4b8lj`,
  boot_id `bc7ae2cb`, nvcc 13.0.88, tree `026d27e99`, Release,
  `-DVLLM_CPP_CUDA_ARCHITECTURES=121a`): `BUILD_RC=0`, and
  `cuda_glue.cu.o` carries one `sm_121a` cubin. All five suites then RAN on that
  device: `test_cast_f16` 3/3 (18 assertions), `test_exl3_linear_method` 7/7
  (275), `test_exl3_gemm` 13/13 (**201**, two more than the CPU run's 199,
  which is the device cases executing rather than skipping),
  `test_exl3_gemv` 6/6 (44), `test_exl3_moe` 8/8 (41).
  **One instrument caveat, recorded because it would otherwise read as
  evidence**: the runner printed `exit=0` after each suite from a `$?` taken
  AFTER a pipe, so it reports the exit status of `grep` and not of the test. The
  doctest `0 failed` counters are the verdict; those `exit=0` strings are not.
- The oracle token gate (#1901).
- W4: `MODEL-DSV4-EXL3`'s private `Exl3Linear` still exists after W1.
- `docs/FEATURES.md` and `docs/USAGE.md` rows, including the checkpoint's file
  name, size, repo and REVISION — a bpw branch name, since this repo publishes
  one revision per bit width and `main` carries no weights at all.

## Stop conditions

- The native reader disagrees with the W1a reference on a fixture → stop and
  re-derive from `exl3.py:227-237`; never tune a constant to green.
- A bf16 load stops being byte-identical → the change is not additive; stop.
