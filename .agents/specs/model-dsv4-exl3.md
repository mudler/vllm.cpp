# MODEL-DSV4-EXL3 — load the SparkInfer EXL3 3.0bpw REAP-K216 DeepSeek-V4-Flash, and match or beat its speed on one GB10

Row: `MODEL-DSV4-EXL3`
Issues: [#1875](https://github.com/mudler/vllm.cpp/issues/1875) (primary)
Base SHA: `3359f4159`
Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements no
EXL3** (`layers/quantization/` has no exl3/exllamav3/trellis at the pin), so
per AGENTS.md the work is gated by a secondary oracle, REGISTERED since
2026-08-25 in [`../oracles/exllamav3.md`](../oracles/exllamav3.md):
`exllamav3` (`turboderp-org/exllamav3` @ `2398c05635fbbad01a0a51dce63c85c6c8a8450e`,
tag `v1.4.3`, MIT), whose HEAD already carries DeepSeek-V4 support
(`exllamav3/exllamav3_ext/{dsv4_compress.cu,dsa_topk.cu,hc_mix.cu}`,
`tests/test_dsv4_*.py`).
Gateability is unmeasured, so that file records `gateable = no` and
[#1901](https://github.com/mudler/vllm.cpp/issues/1901) owes the measurement;
W3a takes it, before any e2e token gate binds. The checkpoint was quantized
at exllamav3 rev `787d1582` from model rev `9e165c30`.
Checkpoint: `0xSero/deepseek-v4-flash-0731-spark` @
`22f28d32b9b29b4352eaa380ff8c2c170b2847ab` (~107 GB), staging to
`/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3` (manifest via
`scripts/rc-stage-checkpoint.sh --make-manifest` once complete). NOTE: the
checkpoint's own README carries `structurally_verified_runtime_pending` — its
publisher has not yet passed end-to-end generation on this exact artifact; the
SparkInfer container run (developer authorization pending) is both our
denominator and the artifact's first independent verification.
Matrix: [`.agents/kernel-matrix.md`](../kernel-matrix.md).
Developer direction (2026-08-24, in session): match or beat the repo's speed on
the same quants. Their README: decode 44–47 tok/s at 384k context, 1 seq, GB10,
WITH "DSpark K5" speculative decoding (acceptance 0.65/0.44/0.31/0.17/0.07); no
bare AR number is published. Our GGUF arm measures 13.0 tok/s AR
(`deepseek-decode-measured-13-launch-not-kernel`).

## Now

`ACTIVE`. **W1a, W1b, W2a+W2b, W2c+W2d and W1c have landed.** W1 gave the CPU
reference dequant (`vt::Exl3*`, `src/vt/cpu/cpu_exl3_dequant.cpp`) and the
rank-sliced loader arm. W2a+W2b gave the two DEVICE ops (`vt::Exl3HadR128`,
`vt::Exl3Gemm`), the portable CPU arm of both, the CUDA port of both, and the
shape-selection policy as pure host code.

**The W2 waves' "the loaded tower is REACHABLE" claim was FALSE and #1923 is the
correction of record.** Those waves wired the expert dispatch and gated it on a
`DeepseekV4Weights` built BY HAND with `has_host_weights = true`; the loader
never wrote `host` and never set that flag, so an actual EXL3 load could not
execute and an end-to-end `vllm-server` probe generated ZERO tokens. W1c
materializes the `carried-*` tower the forward composes with, sets the flag on
the same arm, and rebuilds the forward suite on the PRODUCTION loader entry so
the reachability claim is now the thing the tests measure. See `## W1c design`.
A synthetic rank-sliced checkpoint now loads and emits logits end to end through
`DeepseekV4Model::Forward`; the REAL artifact remains blocked, on its
real-geometry DSA tensors (`## W1c design` W1c-4) and on the tokenizer
([#1924](https://github.com/mudler/vllm.cpp/issues/1924)). W2d replaces that wiring's per-expert loop with `vt::Exl3MoeMlp`
— ONE fused call per layer over every routed expert of every token, where the
loop paid `3 * topk * T` — and keeps the loop as upstream's own tail arm for an
expert over `TEMP_ROWS_FUSED` tokens and as the `VT_DSV4_EXL3_FUSED_MOE=0`
rollback. W2c adds the m<=8 GEMV arm and its selection policy, ported verbatim
and reached from the CUDA GEMM launcher.

**The CUDA arm is UNCOMPILED and UNMEASURED, and nothing in this spec pretends
otherwise.** The implementer host has no `nvcc` and `dgx.casa` is flapping, so
every device number is `PENDING` with its command recorded in `## W2 design` §5,
`## W2cd design` W2cd-9 and `## Owed`. The CPU tier is fully gated and is what
makes the capability reachable today. **No speed figure has been claimed by this
row, at any wave.**
Next: the device compile + parity run the moment the box returns, the shared
dense-MLA policy the real artifact's DSA geometry needs (`## Owed`), and W3.

## The format, as measured (spike, cited at exllamav3 `2398c056`)

One EXL3 quantized linear owns FOUR tensors (`exllamav3/modules/quant/exl3.py:20-91`;
confirmed against the real shard header of
`exl3-layer-000-tp4-rank1.safetensors`):

- `trellis` int16 `(k/16, n/16, 16*K)`; `K = bits` (3.0bpw → K=3). Each 16x16
  tile packs 256 weights as 256 16-bit codewords at K bits/weight
  (`pack.cu:9-57`, `SWAP16` at `:56`); consecutive codewords overlap by 16-K
  bits and the tile is tail-biting
  (`tests/test_quant_fn.py:83-86`; window reference `:98-116`). Weight t's
  codeword sits at bit offset `t*K + K - 16 (mod 256*K)` (`exl3_dq.cuh:18-31`).
- `suh` fp16 `(k,)` and `svh` fp16 `(n,)`: ±1 sign vectors composed with
  blockwise Hadamard-128 on both dims (`had_k = had_n = 128`,
  `exl3_lib/quantize.py:15`; `unpack_bf` `exl3.py:142-158`).
- `mcg` int32 scalar: codebook marker; the value is never read at inference
  (`exl3_lib/quantize.py:1414-1424`).
- **No scales** (`exl3.py:38`: "scale is no longer used").

MCG codebook decode (cb=1), exactly three instructions
(`exllamav3_ext/quant/codebook.cuh:67-75`):
`x *= 0xCBAC1FED; x = (x & 0x8fff8fff) ^ 0x3b603b60;` then reinterpret as two
fp16 and sum. Full-weight reference:
`LinearEXL3.get_weight_tensor` (`exl3.py:227-237`) =
`reconstruct(trellis)` → `preapply_had_l(·,128)` → `* suh[:,None]` →
`preapply_had_r(·,128)` → `* svh[None,:]` (`exl3_lib/quantize.py:340-358`).
Runtime instead transforms activations: `had_r_128(x, suh)` in,
`had_r_128(y, svh)` out (`exl3.py:183-214`; `quant/hadamard.cu:88`).

`rank-sliced-deepseek-v4-v1` (checkpoint `config.json` → `tensor_schema`):
`layers.{L}.ffn.experts.{E}.{w1|w2|w3}.rank{r}.{trellis|suh|svh|mcg}`, tp=4;
177 files = 5 `carried-*.safetensors` + 43 layers x 4 ranks. w1/w3 slice on
OUT (svh + trellis dim1), w2 slices on IN (suh + trellis dim0) — exactly
`tp_import_split` (`exl3.py:296-313`) — so TP1 coalescing is pure
concatenation, lossless (tiles independent per 16 rows/cols; 512 % 128 == 0).
`carried-*.safetensors` hold the un-requantized `deepseek_v4_fp8` tensors
(manifest `preserved_source_scope`: attention/mHC/compressor/indexer, routers
and hash tables, shared experts, embeddings/lm_head/norms, MTP 0-2), which our
existing FP8 arms serve. REAP-K216 is physical compaction
(`REAP_K216_PLAN.json` keep-maps; router rows compacted, hash `tid2eid`
remapped by the publisher); our loader reads `n_routed_experts` from config
(`ParseDeepseekV4Params`, and the per-expert loop in `LoadDeepseekV4Exl3`, both
in `src/vllm/model_executor/models/deepseek_v4_weights.cpp` — cited by SYMBOL
because the W1b commit shifted both line numbers) — no expert-count code
change.

## Scope, in waves

**W1 (this dispatch, CPU tier):**
- `vt` CPU reference dequant of one EXL3 linear: codeword window unpack + the
  3-instruction MCG decode + H128 + sign vectors, mirroring
  `get_weight_tensor` exactly. Small, pure, unit-gated: a fixture tile
  round-trips against independently computed expectations transcribed from
  `tests/test_quant_fn.py:98-128` semantics (encode-side windows generated in
  the test, not shared with the implementation — no
  transcription-gating-its-own-transcription).
- Loader arm: detect `quantization_config.quant_method == "exl3"` +
  `version == "rank-sliced-deepseek-v4-v1"`, coalesce TP4→TP1 at load
  (concatenation per the slicing rule), route `carried-*` tensors into the
  existing DSV4 towers, refuse-by-name anything unimplemented (the standing
  quant-arm rule). Hermetic red-first test through the production loader entry
  with a synthetic 2-expert, 1-layer rank-sliced fixture.
- A dequant-to-bf16 fallback execution arm (dequant at load into the existing
  bf16 expert path) so the model RUNS end-to-end before the trellis GEMM
  exists. Memory cost makes it a testing arm only on the real checkpoint
  (3.0bpw → bf16 x5.3 does not fit GB10); on the synthetic fixture it is the
  W1 reachability gate. Refusal message names the W2 kernel as the production
  path.

**W2 (GPU):** 1:1 ports through the vt OpProvider seam: `exl3_gemm`
(`exl3_gemm_kernel.cuh` + `exl3_gemm_inner.cuh`, shape table
`exl3_kernel_map.cuh:54-62`; plain PTX — cp.async + mma.m16n8k16 + dp4a,
builds for sm_121a), the m<=8 GEMV (`exl3_gemv.cu`, K=3 eligible; upstream
notes Blackwell prefers the regular kernel — measure, never assume), the
fused-MoE mgemm (`exl3_moe.cu` + `comp_units/exl3_moe_inst_k3_cb1.cu`;
first slice may loop the dense GEMM per active expert behind the same op id),
and `had_r_128`. Byte-parity gates against the W1 CPU reference.

**W3 (gates):** exllamav3 gateability measured on GB10 (its DSV4 support may
run this checkpoint; if yes it becomes the token oracle; if not, record
`gateable = no` per the oracle rules and the SparkInfer container run is the
distributional/behavior reference); the SparkInfer denominator run
(authorization pending); speed gates AR-vs-AR and end-config-vs-end-config on
the identical workload (their MAX_NUM_SEQS=1 shape). Bar: same or above their
number on the same quants.

Out of scope: DSpark/our spec-decode integration for this model (own row once
AR parity is measured); the b12x CuTe kernels (Apache-2.0, recorded as a
second kernel reference); any oracle-table edit beyond adding the exllamav3
oracle file.

## Upstream chain

vLLM `555967922`: no EXL3 (`layers/quantization/` swept). Mirror sources:
exllamav3 @ `2398c056` (format + kernels; anchors throughout this spec) and
the checkpoint's own `config.json`/`EXL3_MANIFEST.json` for the rank-sliced
layout (its writer is not public; the schema is documented only there).
SparkInfer container `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489...`
is the behavior/speed denominator; `local-inference-lab/b12x` (Apache-2.0)
the secondary kernel reference.

## Our baseline

DeepSeek-V4-Flash runs here from GGUF (IQ2_XXS mix, 86.33 GiB) at 13.0 tok/s
AR decode on GB10, ~88% GPU-active (memory
`deepseek-decode-measured-13-launch-not-kernel`); MLA/DSA/compressor/mHC/MoE
host+device arms exist (`KERNEL-DSV4-W7-DEVICE`, `KERNEL-MOE-SQRTSOFTPLUS-HASH`),
loader `deepseek_v4_weights.cpp`, ABI client `examples/deepseek_v4_gen`.

## Port map

| upstream | ours |
|---|---|
| `modules/quant/exl3.py:227-237` reference dequant | new `vt` CPU op (W1a) |
| `exl3.py:296-313` tp_import_split | loader TP1 coalescing (W1b) |
| `quant/hadamard.cu:88` `had_r_128` | vt CUDA op (W2a) |
| `exl3_gemm_kernel.cuh` + `exl3_kernel_map.cuh:54-62` | OpProvider trellis GEMM (W2b) |
| `exl3_gemv.cu` | m<=8 GEMV arm (W2c) |
| `exl3_moe.cu` + `comp_units/exl3_moe_inst_k3_cb1.cu` | grouped MoE op (W2d) |

## Tests to port

`tests/test_quant_fn.py:83-128` (tail-biting window + encode/decode
round-trip, rtol/atol 1e-6), `tests/test_qgemm.py` (GEMM vs reconstructed
weights), `tests/test_reconstruct_had.py` parameters; adaptation: fixtures
transcribed into C++ doctest with independently generated windows.

## Dependencies

- The 107 GB checkpoint staged on the NAS with a `SHA256SUMS` manifest — MET
  2026-08-24, 381 lines over all 190 files (see `## Evidence`). W1b's
  real-checkpoint probe and everything in W3 rest on it.
- `dgx:gpu0` lease for W2/W3.
- Developer host-docker authorization for the SparkInfer denominator (asked
  2026-08-24; `PENDING`).
- exllamav3 gateability measurement before any e2e token gate binds (W3a).

## Work breakdown

Non-overlapping; each item names its upstream anchor and lands with its own
red-first test.

- **W1a** CPU reference dequant op (`vt` CPU tier): codeword window unpack
  (`exl3_dq.cuh:15-31` semantics), MCG decode (`codebook.cuh:67-75`), H128 +
  sign composition (`exl3.py:227-237`). Gate: fixture tile byte-parity.
- **W1b** Rank-sliced loader arm in `deepseek_v4_weights.cpp`: detection,
  TP4->TP1 concatenation (`exl3.py:296-313` slicing inverted), carried-tensor
  routing, refusal-by-name for unimplemented arms. Gate: hermetic synthetic
  rank-sliced fixture through the production loader entry.
- **W1c** Dequant-to-bf16 fallback execution arm wired to the existing expert
  path, fixture-gated e2e reachability (`ModelRegistry::Forward`).
- **W2a** `had_r_128` activation transform kernel (`quant/hadamard.cu:88`).
- **W2b** `exl3_gemm` port (`exl3_gemm_kernel.cuh`, `exl3_gemm_inner.cuh`,
  shape table `exl3_kernel_map.cuh:54-62`) via OpProvider; CPU-vs-CUDA parity.
- **W2c** m<=8 GEMV arm (`exl3_gemv.cu`), measured against W2b on GB10 before
  selection.
- **W2d** MoE grouped execution: first slice loops W2b per active expert; the
  fused `exl3_moe.cu` port follows behind the same op id.
- **W3a** `.agents/oracles/exllamav3.md` gateability measurement on GB10.
- **W3b** SparkInfer denominator run (authorization pending) + speed table.
- **W3c** e2e token/distributional gate per the measured oracle verdict.

## W2 design (this wave: W2a + W2b)

`## Work breakdown` names W2a and W2b in one line each. That is a work split, not
a design, and three questions have to be SETTLED HERE rather than discovered when
the gate first runs: what parity means for a path that sums, what dtype the
kernels emit, and which kernel a shape selects. Each is answered below with its
reason and its number.

### 1. The parity contract

`## Scope, in waves` says "byte-parity gates against the W1 CPU reference".
That sentence is TRUE FOR THE DECODE and FALSE FOR EVERYTHING ELSE, and Risk 4
already says why: three different f32 summation orders compute the same weight.
The contract, tier by tier.

**Tier 1 — the trellis decode. BIT-EXACT, no tolerance.** `Exl3TileCodeword`,
`Exl3DecodeMcg`, `Exl3TileRowMajorIndex` and `Exl3DecodeTile` are table lookup,
funnel shifts, one integer multiply, one `lop3`, and ONE fp16 add of two fp16
halves. Nothing accumulates over a length, so `dq_dispatch<3, 1>`
(`exl3_dq.cuh:254-293` -> `dq8<3, 1, 4>` at `:96-161`) and the W1a host decoder
must agree on every one of the 256 codewords of a tile and on the fp16 bit
pattern each decodes to. A tolerance here would hide a wrong constant, which is
exactly the mutation W1 recorded RED four times.

**Tier 2 — `had_r_128`. BIT-EXACT between OUR CPU and OUR CUDA arm; NOT
comparable to W1a.** Upstream's transform is a radix-2 Walsh-Hadamard butterfly
over 128 f32 lanes: levels 1 and 2 done inline on the four values a lane holds
(`hadamard_inner.cuh:118-129`), levels 4..64 done by five `__shfl_xor_sync`
steps (`shuffle_had_f4x32`, `hadamard_inner.cuh:17-44`), then ONE multiply by
`r_scale = scale * 0.088388347648f` (`hadamard.cu:107`; `0.088388347648` is
`1/sqrt(128)`) and one `__floats2half2_rn` store. Every operation is f32 and the
pairing order is the standard FWHT order. `Exl3HadR128`'s CPU arm therefore
performs the SAME operations in the SAME order on the same f32 values, so the two
arms are BYTE-IDENTICAL and the device gate for W2a is a byte gate, not a
tolerance. That is a stronger claim than a bound and it is the reason to mirror
the operation order instead of writing the obvious loop.

It is NOT bit-comparable to `Exl3DequantLinear`'s Hadamard, and that is not a
defect: that one transforms the WEIGHTS with an fp16 round after every 128-block
(`quantize.py:340-356` `.to(x_dtype)`), this one transforms the ACTIVATIONS with
one round at the store. They are the same linear map and different roundings of
it. Nothing in W2 gates one against the other.

**Tier 3 — `exl3_gemm`. BOUNDED, and the bound is stated before the gate runs.**
The kernel accumulates in f32 through `mma.sync.aligned.m16n8k16.row.col.f32`
whose internal accumulation order is UNSPECIFIED by PTX, and adds a split-K
threadblock reduction (`exl3_gemm_inner.cuh:315-423`) on top. No byte claim is
available. The reference is a `double` evaluation of the same chain — decode ->
`had_r_128(x, suh)` -> `x_had @ W_inner` -> `had_r_128(y, svh)` — and the gate is

  RMS relative error <= **1.0e-3**, and elementwise
  |y_kernel - y_f64| <= **8 * ulp_f16(rms(y_f64))**.

WHY THOSE NUMBERS. The output is stored fp16 (`__floats2half2_rn` /
`had_fh_r_128_inner`), whose ulp is `2^-11` = 4.88e-4 relative, so ONE store
round already costs up to 4.9e-4 and an RMS-relative bound below ~5e-4 could not
be met by a correct kernel. f32 accumulation over the largest k this row uses
(4096) contributes at most `sqrt(4096) * 2^-24` = 3.8e-6 relative — two orders
below the store, so summation order is NOT what sets this bound; the fp16
destination is. 1.0e-3 is two fp16 ulps of RMS and 8 ulps elementwise covers an
element whose two orderings straddle a rounding boundary at a higher exponent
than the RMS. A kernel that decodes a wrong codeword misses by ~2^-3 relative,
four orders above the bound, so the bound discriminates the defect it exists for.

The bound is NOT widened if the gate reds. A red means a defect or a wrong
reference; both get fixed. That rule is the whole point of stating a number here.

**Tier 3b — the BASIS cross-check, and why it needs its own number.** The f64
chain above shares its structure with the kernel, so it cannot notice that the
whole basis is wrong. The second comparison is against W1a's `Exl3DequantLinear`
— the WEIGHT-side reference, gated independently by W1 — through
`y = x @ W`. That is the algebraic identity the format rests on
(`exl3.py:183-214` vs `:227-237`): the two Hadamards may ride the activations or
the weights. Its bound is LOOSER and says why: `Exl3DequantLinear` rounds the
weight to fp16 after each of its four stages (`quantize.py:340-356`
`.to(x_dtype)`), four roundings the fused path never performs, each costing up to
half an fp16 ulp = 2.4e-4 relative. Four of those bound the difference at
9.6e-4, so the gate is **2.0e-3 relative RMS** — twice the worst case and still
two orders below a decode defect. Measured 5.53e-4 on the wave's own fixture.

### 2. The output dtype

**`exl3_gemm` emits fp16 (`DType::kF16`), and `A`/`A_had` are fp16.** Not
inherited from W1a and not chosen for convenience:

- Upstream's own memory format. `LinearEXL3.default_out_dtype = out_dtype or
  torch.half` (`exl3.py:72`) and `reconstruct_hgemm` allocates `y` at exactly
  that (`exl3.py:167`). fp16 is the DEFAULT; f32 is the exception a caller asks
  for.
- `A` has no freedom at all. `ldmatrix.sync.aligned.m8n8.x4.shared.b16` +
  `mma.sync...f32.f16.f16.f32` (`ptx.cuh:52-74, 203-212`) read fp16 fragments.
  An f32 activation buffer would have to be narrowed before the load anyway.
- Upstream's `c_fp32` arm is KEPT, because upstream keeps it and because the MoE
  weighted reduction is where it earns its width (`exl3_gemm_kernel.cuh:267-276`
  sums per-expert results). It is selected by the CALLER's `C` dtype, exactly as
  upstream selects it from `C.dtype() == at::kFloat` (`exl3_gemm.cu:134`), and it
  is not the default.

Risk 5 is therefore answered in the negative: `Exl3DequantLinear`'s `float* out`
does NOT propagate. That signature stays what it is — a HOST checkpoint-format
decoder whose f32 is a carrier for fp16-valued data — and no device buffer in
this wave is f32 because a host reference function was.

### 3. Kernel selection policy

The shape table is ported VERBATIM as a pure host function, so it is gateable
with no device: `EXL3_GEMM_SHAPE_1..4` and the three geometry rows
(`exl3_kernel_map.cuh:53-60`), `select_gemm_shape` (`exl3_kernel_map.cu:23-75`),
`exl3_gemm_shape_compat` (`:86-91`) and the empty-block clamp on `num_sms`
(`:153-160`). The compute-capability CLASS is upstream's own five-way bucket
(`exl3_devctx.cu:32-46`): `major >= 10` is `CC_BLACKWELL`, so GB10 (sm_121,
major 12) is `CC_BLACKWELL` — not a new bucket and not an assumption.

Resolved for this checkpoint (K = 3, cb = 1 `mcg`, `multi = false`), whose
TP1-coalesced expert shapes are w1/w3 `k = 4096, n = 2048` and w2 `k = 2048,
n = 4096` (measured from the real shard header, `## The format` above):

| linear | k | n | branch taken | shape |
|---|---|---|---|---|
| w1, w3 | 4096 | 2048 | `mod_256 && size_n <= 4096` -> `size_k > 8192 && K >= 3 ? 3 : 2` | **2** |
| w2 | 2048 | 4096 | same branch, same test | **2** |

Shape 2 is `TILESIZE_M 16, TILESIZE_K 32, TILESIZE_N 128, SH_STAGES 4,
FRAG_STAGES 3`, block dim 512 (`exl3_kernel_map.cuh:54, 58-60`). Both shapes are
compatible (`k % 32 == 0`, `n % 128 == 0`). This is a HOST-side claim and this
wave gates it as one.

**The m <= 8 GEMV is NOT in this wave (W2c), and the reason to measure rather
than assume is now precise.** The line usually quoted for "Blackwell keeps the
regular kernel" — `if (cc != CC_AMPERE) return -1;  // ... Ada/Blackwell are
memory-bound here` — is **COMMENTED OUT** at `exl3_gemv.cu:53`. The LIVE
envelope (`:64-71`) does admit Blackwell: with `K == 3` it takes the narrow
config when `size_n / 32 <= narrow_coresident` or when `size_k <= 2048 &&
size_n <= 8192`, and otherwise returns -1. For our shapes that means **w2
(k=2048, n=4096) IS GEMV-eligible on GB10 and w1/w3 (k=4096, n=2048) fall
through to the regular kernel unless `2048/32 = 64` blocks are co-resident.**
So the prose that would have justified skipping the GEMV is a disabled guard, and
the real answer depends on a device occupancy query. W2c measures it. Nothing in
this wave selects it, and the selection function this wave lands has no GEMV arm
to accidentally take.

### 4. What is reachable, and what is not

`vt::Exl3HadR128` and `vt::Exl3Gemm` are dispatched from `MoeBlock`
(`deepseek_v4.cpp`) whenever `DeepseekV4Weights::has_exl3_weights` is set: one
`Exl3Gemm` per active routed expert per projection (w1 gate, w3 up, w2 down),
which is the "first slice loops the dense GEMM per active expert" shape
`## Work breakdown` W2d allows. The entry point is
`DeepseekV4Model::Forward` -> `ForwardComposeImpl` -> `MoeBlock`, which
`deepseek_v4_registry.cpp` routes `ModelRegistry::Forward` to. Deleting that
dispatch must turn the wave's reachability case RED, and the fresh review
mutates for exactly that.

Consequences recorded rather than hidden:

- The FUSED `exl3_moe.cu` mgemm is **W2d, owed.** The per-expert loop pays one
  launch per (expert, projection) where the fused kernel pays one per layer.
- The **m <= 8 GEMV is W2c, owed**, with the measurement stated in §3.
- The routed-expert arm needs the REST of the tower to run a real checkpoint end
  to end. On the real artifact those are the `carried-*` FP8 tensors and W1c
  still owns materialising them; this wave's reachability case therefore drives
  the production entry point over a TINY host tower plus a real EXL3 expert
  tower, which is the same vehicle W1b used.

### 5. The device measurements this wave CANNOT take, and the command for each

`dgx.casa` hung 2026-08-25 03:24Z with the GB10 unified-memory OOM-reboot
signature and needs a manual power cycle. Every device number below is `PENDING`
on that box returning, and NONE of them is inferred, estimated, or filled in from
a CPU run.

| owed measurement | command, once `rc devices` shows `dgx:gpu0` |
|---|---|
| W2a byte parity, CPU vs CUDA `had_r_128` | `rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemm_device -V` |
| W2b tier-3 bound, CUDA vs the f64 reference | same target; the case is `dsv4 exl3 device: exl3_gemm matches the f64 reference within the stated bound` |
| the CUDA TUs compile for sm_121a | `cmake -S . -B build-cuda -G Ninja -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a && cmake --build build-cuda --target vllm -j 4` |
| shape-2 selection is what the DEVICE takes | `VT_OP_PROVIDER_STATS=1` on the run above; the host claim in §3 is already gated on CPU |
| any speed number at all | not attempted; W3b owns the denominator and no ratio is quoted anywhere in this wave |

The device-parity suite is REGISTERED and SKIPS LOUDLY with a named reason when
no CUDA device is present, so the moment the box returns exactly one command
produces the numbers. A skip that printed a pass would be the
`assertions: 0` trap this row already carries once; the suite therefore reports
its skip through an explicit message and its live cases assert.

## W2cd design (this wave: W2d, the fused MoE mgemm, and W2c, the m<=8 GEMV)

`## Work breakdown` gives W2c and W2d one line each and `## W2 design` §4 records
both as owed. This section settles what they mean BEFORE their gates run, on the
same terms §1-§5 above set: every number is stated here, every divergence from
the oracle names its reason, and every measurement this dispatch cannot take is
`PENDING` with its command rather than inferred.

The order is deliberate. W2d is the SHAPE OUR DECODE HAS — 6 active of 216
experts per token, three projections each — and the per-expert loop W2b landed
pays one kernel launch per (expert, projection) where upstream's fused kernel
pays one per layer. W2c is an ALTERNATIVE ARM for one of those projections whose
value is unmeasured and, as §W2c-3 below shows, is not purely a speed question.

### W2d-1. A new op id, and why not the existing one

`vt::Exl3MoeMlp` is registered as `OpId::kExl3MoeMlp`, appended before `kCount`
so no existing id shifts. It is NOT folded into `kExl3Gemm`, because upstream's
own `exl3_moe` is a separate entry point with a separate argument list
(`exl3_moe.cuh:8-46`): nine per-expert POINTER TABLES, a token-sorted batching
triple, four temp buffers and a per-projection bit width. A single op taking the
union of both signatures would be one op with two disjoint contracts.

The op's arguments mirror `exl3_moe` (`exl3_moe.cu:99-137`) one for one:

| upstream | ours |
|---|---|
| `hidden_state` fp16 `[bsz, hidden_dim]` | `hidden_state`, same |
| `output_state` **f32**, zero-initialized | `output_state`, same |
| `expert_count` i64 `[num_experts + 1]` | `Exl3MoeRouting::expert_count` |
| `token_sorted` i64 `[bsz * top_k]` | `Exl3MoeRouting::token_sorted` |
| `weight_sorted` fp16 `[bsz * top_k]` | `Exl3MoeRouting::weight_sorted` |
| `temp_state_{g,u}` fp16 `[C, R, hidden]` | `Exl3MoeTemps::state_{g,u}` |
| `temp_intermediate_{g,u}` fp16 `[C, R, interm]` | `Exl3MoeTemps::intermediate_{g,u}` |
| `{gate,up,down}_ptrs_{trellis,suh,svh}` | `Exl3MoeExpertTables`, nine i64 tensors |
| `K_gate`, `K_up`, `K_down` | `Exl3MoeArgs::bits_{gate,up,down}` |
| `gate_mcg` … `down_mul1` | `Exl3MoeArgs::codebook` (one value; see W2d-5) |
| `act_function`, `act_limit`, `num_active` | same names on `Exl3MoeArgs` |

**`output_state` is f32 and that is UPSTREAM's width, not a widening we chose.**
`TORCH_CHECK_DTYPE(output_state, kFloat)` (`exl3_moe.cu:151`) and the epilogue
accumulates into it with `atomicAdd` (`hadamard_inner.cuh:469-472`), because the
scatter-add sums one contribution per (token, active expert) and an fp16
accumulator over six weighted contributions loses bits the tokens can see. Risk
5's polarity is preserved: `hidden_state` and every temp buffer stay fp16.

**The merged gate/up seam is not taken, and the reason is the FORMAT.** AGENTS.md
routes mergeable MLP projections through `layers::MlpGateUpMethodBase` and
`vt::MergedGemmGroup`, which merge gate and up into one GEMM over a STACKED
weight. An EXL3 gate and up are two independent trellis tensors with their own
`suh`/`svh` vectors, and stacking them would mean re-quantizing the artifact;
upstream's own fused kernel keeps them as two GEMM calls
(`exl3_moe_kernel.cuh:159-161`). So the seam cannot represent this format, which
is the condition AGENTS.md gives for not taking it. Worth stating plainly: this
is NOT a property W2c or W2d introduces. `grep -n 'MlpGateUpMethodBase\|MergedGemmGroup'
src/vllm/model_executor/models/deepseek_v4*.cpp` is empty, so the whole
DeepSeek-V4 port has always been off that seam and no wave of this row is the
place to move it.

### W2d-2. The activation is vLLM's, not the oracle's, and the difference is 4.5e-4

This is the one place where mirroring the kernel oracle would mirror the WRONG
BEHAVIOR, so it is decided here with a number.

Upstream's fused activation (`hadamard_inner.cuh:284-413`, reached through
`had_hf_r_128_guad_inner`) with `ACT_SILU` and a non-zero `act_limit` computes

    vg = silu(g_had) ; then vg = min(vg, +limit)     (:116-117, AFTER the silu)
    vu = clamp(u_had, -limit, +limit)                (:112-115)
    out = vg * vu

vLLM's DeepSeek-V4 expert activation is `SiluAndMulWithClamp`
(`model_executor/layers/activation.py:197-201`), which this tree already
implements as `deepseek_v4::ClampedSwiGLU`:

    gate = min(g, +limit)                            (BEFORE the silu)
    up   = clamp(u, -limit, +limit)
    out  = gate * sigmoid(alpha * gate) * (up + beta)

The two differ, and by a bounded amount: for `g >= limit` upstream yields
`limit` while vLLM yields `silu(limit) = limit * sigmoid(limit)`, a difference of
`limit * sigmoid(-limit)`. At DeepSeek-V4-Flash's `swiglu_limit = 10.0` that is
`10 * 4.5398e-5 = 4.5398e-4` absolute; for `g < limit` and for every negative `g`
the two are identical. Small — and it is a MODEL behavior, not a kernel detail.
AGENTS.md makes vLLM the authority wherever it implements the behavior and makes
exllamav3 the oracle only where vLLM implements nothing; vLLM implements this
activation. So the port adds a FOURTH `act_function` value,
`Exl3MoeAct::kSiluAndMulClamp = 3`, which clamps BEFORE the silu, and leaves
upstream's 0/1/2 exactly as they are. The DeepSeek-V4 call site passes 3.

A shape gate cannot see this and a token gate can. It is settled here rather
than discovered by the token gate this row cannot yet run.

### W2d-3. The batching is HOST arithmetic, and is gated as such

`expert_count`, `token_sorted`, `weight_sorted` and `num_active` are computed on
the host by upstream too (`modules/block_sparse_mlp.py:1079-1105`), so
`vt::Exl3MoeSortTokensByExpert` is a pure function in `src/vt/exl3_policy.cpp`
next to the shape table, gateable with no device — the same reason §3 gives for
putting `select_gemm_shape` there.

Ported semantics, line for line:

- `flat_expert = topk_ids.reshape(-1)`, `flat_weight = topk_weights.reshape(-1)`,
  `flat_token` the interleaved arange `[0]*topk, [1]*topk, …` (`:1079-1083`).
- `expert_count = bincount(flat_expert, minlength = E + 1)` (`:1100`). The
  `E + 1`-th slot is upstream's sentinel for an assignment outside this shard's
  expert range; a single-shard load never fills it, and it is kept so the tensor
  the kernel reads has upstream's own length.
- `token_sorted`/`weight_sorted` = the assignments grouped by expert (`:1095-1097`).
- `num_active` = the number of experts with `0 < count <= max_tokens_per_expert`
  (`:1105`).

**DEVIATION, recorded: a stable counting sort where upstream calls
`Tensor.argsort`.** torch's default `argsort` is NOT stable, so upstream's
within-expert order is unspecified; ours is the order the tokens appear in. The
two agree on `expert_count` and on the MULTISET of each expert's segment, and
nothing downstream reads the within-segment order except which row of the temp
buffer a token occupies, which the epilogue scatters back by token index. The
stable form is chosen so a fixture is reproducible and a mutation of the sort is
detectable.

### W2d-4. `max_tokens_per_expert`, and why the per-expert loop is NOT only a rollback

`max_tokens_per_expert` is upstream's `TEMP_ROWS_FUSED = 128`
(`block_sparse_mlp.py:19`), the second dimension of the temp buffers. The fused
kernel SKIPS an expert whose token count exceeds it (`exl3_moe_kernel.cuh:66`)
and upstream's caller then runs exactly those experts through its own per-expert
path (`min_rows = TEMP_ROWS_FUSED` at `:1141`, the loop at `:1151-1156`).

Our call site does the same: after `Exl3MoeMlp` returns, `MoeBlock` loops the
experts the fused arm declined and runs each through the W2b `Exl3Linear` path.
So the loop W2b landed is upstream's own tail arm, not merely a switch, and it
stays REACHED at any batch that puts more than 128 assignments on one expert.

`concurrency` is `num_sms / MOE_SMS_PER_EXPERT` (`exl3_moe.cu:14-18`,
`MOE_SMS_PER_EXPERT = 8`) capped at `MOE_MAX_GROUPS = 64` (`exl3_devctx.cuh:13`),
exposed as the pure `vt::Exl3MoeMaxConcurrency(int device_sms)`. The CPU arm
accepts any concurrency `>= 1` and ignores the grouping, because a group is a
device scheduling unit and the CPU reference has nothing to schedule.

### W2d-5. One codebook per call, mirroring upstream's own refusal

`exl3_moe.cu:182-185` refuses a call whose gate/up/down do not share a codebook
and refuses anything that is neither `mcg` nor `mul1`. Our `Exl3MoeArgs` carries
ONE `codebook` value for that reason: three fields that must be equal are three
chances to disagree. As with `Exl3Gemm`, only codebook 1 (`mcg`) is implemented
and every other value refuses BY NAME. The three BIT WIDTHS stay separate,
because upstream keeps them separate and switches `K` at run time when they
differ (`exl3_moe_kernel.cuh:139-149`).

### W2d-6. The flag

`VT_DSV4_EXL3_FUSED_MOE`, default **on**, `0` falls back to the per-expert loop
for EVERY expert. House shape: a pure predicate
`Dsv4Exl3FusedMoeFlagIsOn(const char* env_value)` beside
`Dsv4Exl3HostBudgetFlagIsOn` in `deepseek_v4.h`, unit-tested without touching the
environment, plus a process-cached getter at the one read site. Documented in
`docs/ENVIRONMENT.md` under "Rollback and bisect switches", because it is a
user-facing behaviour knob and `scripts/check-env-doc.py` splits on exactly that.

Default ON is not a performance claim. It is the arm the row exists to build, the
CPU arm is gated against the loop arm at a stated bound below, and the switch is
there so a suspected defect is bisected without a rebuild.

### W2d-7. The parity contract for the fused arm

Tiers 1 and 2 of §1 are unchanged: the fused kernel calls the SAME
`exl3_gemm_kernel_inner` and the SAME 128-point butterfly, so the decode stays
bit-exact and the Hadamard stays byte-exact between our two arms.

**Tier 4 — the fused MoE arm against the per-expert loop arm. BOUNDED at 2.0e-2
relative RMS, and the bound is the ACTIVATION's, not the GEMM's.** The two arms
compute the same algebra by different routes:

- the loop arm converts each projection's output to f32, runs the clamped SwiGLU
  in f32 on the host, converts back to fp16 for the down projection;
- the fused arm keeps the intermediate in fp16 throughout and performs the
  activation in fp16 (`had_hf_r_128_guad_inner` is `half2` arithmetic end to
  end), which is upstream's own choice.

One fp16 round of the intermediate costs up to 4.9e-4 relative; the activation
applies a sigmoid and a multiply to an already-rounded value, and the down
projection then sums `mi = 2048` of them. `sqrt(2048) * 4.9e-4 = 2.2e-2` is the
worst case if every rounding error were independent and aligned, so **2.0e-2** is
the bound, which is the same number `## Evidence`'s reachability pair already
uses for the EXL3-vs-dequantized-dense comparison and for the same reason. A
mis-decoded codeword misses by ~2^-3 relative, two orders above it, so the bound
still discriminates the defect it exists for.

**Tier 4 also covers OUR OWN two arms of the fused op, and that is not a
weakening.** `had_hf_r_128_guad_inner` computes the activation in fp16 through
`h2exp` and `h2rcp` (`hadamard_inner.cuh:323-332`), which are HARDWARE
APPROXIMATIONS with no host equivalent; the CPU arm widens the same fp16 inputs
to f32, evaluates `x * sigmoid(x)` there and rounds once. So unlike tier 2 —
where the CPU and CUDA Hadamards run the same f32 operations in the same order
and a BYTE claim is available — no byte claim is available here, at any effort,
and asserting one would be asserting that two different transcendental
implementations agree bit for bit. The device-vs-CPU gate is therefore the same
2.0e-2, and the reason is written down rather than discovered when it reds.

The bound is NOT widened if the gate reds.

### W2c-1. The line usually cited is COMMENTED OUT, and this dispatch read it

`## W2 design` §3 says the "Ada/Blackwell are memory-bound here" guard is
disabled. VERIFIED at the pin by reading the file: `exl3_gemv.cu:53` is

    //if (cc != CC_AMPERE) return -1;  // measured win on Ampere; Ada/Blackwell are memory-bound here

and the second copy inside the launcher is disabled too (`:119`,
`// if (cc != CC_AMPERE) return false;`). The prose at `:22-26` that describes
the Ampere-only envelope is a COMMENT describing a guard the code no longer
applies. No compute-capability test is live in the eligibility path, so Blackwell
is admitted and the shape envelope alone decides.

### W2c-2. What the envelope resolves to for THIS checkpoint, and what it cannot

Ported verbatim as pure host functions in `src/vt/exl3_policy.cpp`:
`Exl3GemvMaxM` (`exl3_gemv_kernel.cuh:31`, 8), `Exl3GemvHardEligible`
(`exl3_gemv.cu:110-114`), `Exl3GemvSelectConfig` (`:46-72`, returning -1 not
eligible / 0 narrow / 1 wide).

At `K = 3`, `cb = 1` (mcg), `m = 1`, `cc = kBlackwell`, mode 1 (the default), the
live branches are `:64` (`K == 2`, not taken), `:65` (`K == 3 && cc == CC_ADA`,
not taken), `:66` (`size_n / 32 <= narrow_coresident`), `:67`
(`size_k <= 2048 && size_n <= 8192`) and `:68` (`K == 3` -> -1):

| linear | k | n | verdict |
|---|---|---|---|
| w2 (down) | 2048 | 4096 | `:67` fires -> **config 0 (narrow)**, independent of occupancy |
| w1, w3 (gate, up) | 4096 | 2048 | `:67` does not fire; `:68` returns **-1** UNLESS `:66` fires, i.e. unless `2048 / 32 = 64 <= narrow_coresident` |

`narrow_coresident = cudaOccupancyMaxActiveBlocksPerMultiprocessor(narrow_kernel,
512) * num_sms` (`exl3_gemv.cu:127-142`). It is a DEVICE query, it is the only
input that decides w1/w3, and this dispatch has no device. It is therefore a
PARAMETER of the pure function rather than a query made inside it — the same
shape `ReportDeepseekV4Exl3Residency` uses for the host budget, and what makes
the table above gateable on a machine with no GPU.

**Which arm is FASTER on GB10 is not decided here and is not decidable from the
source.** It is `PENDING`, with its command in W2cd-9.

### W2c-3. The GEMV is a different NUMERIC arm, not only a faster one

This is the finding that changes what W2c's gate has to be, and it is recorded
before any of it is measured.

The regular kernel accumulates in f32: `mma.sync.aligned.m16n8k16.row.col
.f32.f16.f16.f32` (`ptx.cuh:52-74`). The GEMV kernel accumulates in **fp16** —
`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
(`exl3_gemv_kernel.cuh:37-52`, `mma_ab_h`) — and folds the fp16 accumulator into
an f32 pair only every `FOLD` iterations (`:317-330`; `FOLD = 4` for config 0,
`2` for config 1).

So an fp16 accumulator absorbs up to `FOLD * 16 = 64` k-elements (config 0)
before it is folded. fp16's unit roundoff is `2^-11 = 4.88e-4`, so the
accumulation error over one window is on the order of `sqrt(64) * 4.88e-4 =
3.9e-3` relative — **four times the tier-3 RMS bound of 1.0e-3 that §1 states for
the regular kernel**. Tier 3 therefore does not cover this arm, and reusing it
would either fail a correct kernel or force the widening that §1 forbids.

**Tier 3c — the GEMV arm, bounded at 6.0e-3 relative RMS and 64 fp16 ulps of the
output RMS elementwise.** `6.0e-3` is `sqrt(2) *` the `3.9e-3` window estimate
above, rounded up: one factor for the f32 sum over the `size_k / 64` windows,
which is negligible, and the rest of the headroom for the case where the window
sum grows monotonically rather than as a random walk. A wrong codeword still
misses by ~`2^-3`, more than an order above the bound.

This is upstream's own accuracy-for-speed trade and it ships enabled by default
there. We mirror the DEFAULT (W2c-4) and we state the BOUND, because a bound
nobody stated is a bound a red gate gets to invent.

### W2c-4. The mode knobs mirror upstream's, defaults included

`VT_EXL3_GEMV` mirrors `EXL3_GEMV` (`exl3_gemv.cu:29-34`): `0` disables the path,
`1` or unset is the heuristic, `2` takes it wherever the hard constraints allow,
`3` forces the narrow config, `4` forces the wide one. `VT_EXL3_GEMV_SMEM`
mirrors `EXL3_GEMV_SMEM` (`:37-42`): `-1` or unset is the per-bits default,
`0` forces shuffle extraction, `1` forces shared-memory staging.

Both are read once per process and both are kernel-internal tuning knobs rather
than user-facing behaviour, so they go on `scripts/env-doc-allowlist.txt` rather
than into `docs/ENVIRONMENT.md` — the split `scripts/check-env-doc.py` itself
draws. `VT_DSV4_EXL3_FUSED_MOE` goes the other way, into the doc, because it
changes which arm a shipped model runs.

**Inheriting upstream's default is mirroring, not deciding.** AGENTS.md requires
every applicable mode and default of the reference to be mirrored; it does not
permit a speed claim, and none is made. The arm is eligible by upstream's own
heuristic and whether it is faster here is W2cd-9's measurement.

### W2c-5. What the CUDA arm instantiates, and what it declines

Narrowed exactly as `exl3_gemm` is: `bits == 3`, `codebook == 1`. Upstream's
`exl3_gemv_select_kernel` (`exl3_gemv.cu:74-90`) covers bits 2/3/4 over three
codebooks; ours covers the shipped arm over `(c_fp32, mmode, cfg, smem)`, which
is 16 instantiations. Every other width DECLINES by returning false from the
try-launch and falling through to the regular kernel, which is upstream's own
failure mode (`exl3_gemv_select_kernel` returns `nullptr`, `try_launch` returns
false) rather than a throw — a decline here is not an unimplemented arm, it is
the heuristic saying no.

The wiring mirrors `exl3_gemm.cu:220-236`: the try-launch runs only when the
caller forced neither a shape nor an SM count, and a false return falls through
to the shape table unchanged.

### W2cd-8. What is reachable

`vt::Exl3MoeMlp` is dispatched from `MoeBlock` (`deepseek_v4.cpp`) whenever
`DeepseekV4Weights::has_exl3_weights` is set and `VT_DSV4_EXL3_FUSED_MOE` is on:
ONE call per layer covering every routed expert of every token in the block,
replacing the `3 * topk * T` `Exl3Gemm` calls the W2b arm makes. The entry point
is unchanged — `DeepseekV4Model::Forward` -> `ForwardComposeImpl` -> `MoeBlock`,
which `deepseek_v4_registry.cpp` routes `ModelRegistry::Forward` to. Deleting
that dispatch must turn the wave's reachability case RED, and the fresh review
mutates for exactly that.

**A VALUE GATE CANNOT SEE THAT DISPATCH, and pretending otherwise would be the
defect.** Delete it and the per-expert loop picks the work up, so the logits stay
right — which is precisely what makes that loop a genuine tail path rather than
dead code behind a flag. The reachability case therefore reads
`vt::OpProviderStats::selections`, the positive signal `include/vt/op_provider.h`
exists for, and asserts the exact launch counts for BOTH arms: the fused arm is
`kExl3MoeMlp` once per MoE layer with `kExl3Gemm` at zero, and the rollback is
the mirror image. `tests/CMakeLists.txt` runs the same suite twice, once under
`VT_DSV4_EXL3_FUSED_MOE=0`, because the flag is read once per process and one
run can only measure one arm.

`Exl3GemvSelectConfig` and its siblings are reached from `Exl3GemmKernelCuda`,
which calls them before the shape table on every EXL3 GEMM on a CUDA queue. The
GEMV kernel itself is reached from that call when the heuristic accepts. Neither
is reachable on a CPU queue and neither should be: the GEMV is a device
occupancy arm, and upstream dispatches it from the CUDA launcher only.

**Which makes W2c's reachability WIRED but UNVERIFIED, and the two words are not
the same.** `src/vt/cuda/cuda_exl3.cu` is one line of the
`target_sources(vllm PRIVATE ...)` block inside `if(VLLM_CPP_CUDA)`, so on a CUDA
build the GEMV policy and kernel sit on the production `Exl3Gemm` path with
nothing optional between them; this is not a staged slice landing unreached. What
has not happened is a COMPILE: no `nvcc` has read that file, here or anywhere,
for W2a, W2b, W2c or W2d. The envelope itself is host code and IS gated on this
build, which is the whole reason `## W2 design` §3 put the tables in a `.cpp`.

### W2cd-9. The device measurements this wave CANNOT take, and the command for each

`dgx.casa` is flapping (boot, brief contact, drop). Every number below is
`PENDING` on that box returning. NONE is inferred, estimated, or filled in from a
CPU run, and no speed figure is quoted anywhere in this wave.

| owed measurement | command, once `rc devices` shows `dgx:gpu0` |
|---|---|
| `narrow_coresident` on GB10, which decides whether w1/w3 are GEMV-eligible at all (W2c-2) | `rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemv_device -V` |
| the tier-3c bound for the GEMV arm, CUDA vs the f64 reference | same target |
| WHICH arm is faster for w2 (k=2048, n=4096), GEMV vs the regular kernel | `rc run --device dgx:gpu0 -- env VT_EXL3_GEMV=0 …` A/B against `VT_EXL3_GEMV=1`, three reps, idle box |
| the tier-4 bound for the fused MoE arm, CUDA vs the loop arm | `rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_moe -V` |
| the launch count actually falls from `3 * topk * T` to 1 per layer | `VT_OP_PROVIDER_STATS=1` on the run above |
| the CUDA TUs compile for sm_121a | `cmake -S . -B build-cuda -G Ninja -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a && cmake --build build-cuda --target vllm -j 4` |

Both device suites are REGISTERED and SKIP LOUDLY with a named reason and the
exact command when no CUDA device is present, and each still asserts the
precondition it skipped on, so neither can report `assertions: 0`.

## W1c design (this wave: the carried tower a forward can actually consume)

`## Work breakdown` gives W1c one line — "dequant-to-bf16 fallback execution arm
wired to the existing expert path, fixture-gated e2e reachability" — and W2
overtook the expert half of that sentence: the routed experts now execute
through `vt::Exl3Gemm` / `vt::Exl3MoeMlp`, so nothing is owed there. What was
NEVER done is the other half, and
[#1923](https://github.com/mudler/vllm.cpp/issues/1923) is the measurement that
says so: an end-to-end probe of `vllm-server` at `cc4ff8681` LOADED a
rank-sliced EXL3 checkpoint, printed its residency line, and then killed the
engine on the first `/v1/completions` with

```
vt: DeepseekV4 forward: host-float weight tower not materialized ...
at src/vllm/model_executor/models/deepseek_v4.cpp:2960
```

Zero tokens. `LoadDeepseekV4Exl3`'s carried-tensor pass was
`require(...)` -> `RequireTensor`, a PRESENCE CHECK that increments a counter.
It never wrote `w.host` and never set `w.has_host_weights`, so
`has_exl3_weights && has_host_weights` was unreachable from any load.

### W1c-1. Why three review rounds and a mutation pass did not see it

`tests/vllm/models/test_deepseek_v4_exl3_forward.cpp` set
`has_host_weights = true` BY HAND at five sites and built `DeepseekV4Weights`
directly instead of going through `LoadDeepseekV4ForCausalLMWeights`. Every
mutation the W2 reviews ran was therefore evaluated on a struct no loader can
produce. This is `.agents/reachability.md`'s documented shape exactly: a unit
test that constructs the type by hand proves the class works and never that
anything reaches it.

So the fix has TWO halves and the second is not optional. The forward suite is
rebuilt on a fixture that goes through the production loader entry, over a
synthetic rank-sliced EXL3 checkpoint written to disk with REAL carried tensors
at their real dtypes. Deleting the loader's materialization, or its flag set,
must redden that suite. Both mutations are recorded under `## Evidence`.

### W1c-2. What the forward consumes, and therefore what the loader writes

`DeepseekV4ForwardExl3` composes with `weights.host` and `be.gguf == nullptr`,
so EVERY non-expert weight is read as f32 out of `DeepseekV4HostWeights`. The
loader therefore dequantizes the carried tensors into that layout. The mapping
is fixed by what `AttentionBlock` / `MoeBlock` / `ForwardComposeImpl` index, and
each destination is shape-checked against the config-derived expectation before
the write, so a layout this arm does not implement REFUSES BY NAME instead of
reading a neighbouring tensor's bytes.

| carried tensor (dtype, measured on the real artifact) | host slot |
|---|---|
| `embed.weight` BF16 `[V,H]` | `host.embed` |
| `head.weight` BF16 `[V,H]` (absent when tied) | `host.lm_head` |
| `norm.weight` BF16 `[H]` | `host.final_norm_weight` |
| `hc_head_{base,fn,scale}` F32 | `host.hc_head_{base,fn,scale}` |
| `layers.L.{attn,ffn}_norm.weight` BF16 `[H]` | `L.{attn,ffn}_norm_weight` |
| `layers.L.hc_{attn,ffn}_{base,fn,scale}` F32 | the six `L.hc_*` |
| `attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}` F8_E4M3 + F8_E8M0 | `L.{wq_a,wq_b,wkv,wo_a,wo_b}` |
| `attn.{q_norm,kv_norm}.weight` BF16 | `L.{q_norm,kv_norm}_weight` |
| `attn.attn_sink` F32 `[nh]` | `L.attn_sink` |
| `attn.compressor.{ape,norm.weight,wgate.weight}` | `L.comp_{ape,norm_weight,wgate}` |
| `attn.indexer.{wq_b.weight+scale, compressor.wkv.weight, weights_proj.weight}` | `L.idx_{wq,wk,wproj}` |
| `ffn.gate.weight` BF16 `[ne,H]` | `L.gate_weight` |
| `ffn.gate.bias` F32 `[ne]` / `ffn.gate.tid2eid` I64 `[V,topk]` | `L.gate_bias` / `L.tid2eid` |
| `ffn.shared_experts.w{1,2,3}.{weight,scale}` FP8-block | `L.shared_w{1,2,3}` |

`L.exp_w{1,2,3}` are deliberately LEFT EMPTY: the routed experts are the EXL3
tower and `MoeBlock` takes them from `Le`, not from `hw`. `attn.compressor.wkv`
is accounted and NOT materialized, because the collapsed-geometry compressor
reuses `kraw` as its KV and no host slot reads it — recorded here rather than
discovered by a reader who cannot find the destination.

### W1c-3. The block-wise FP8 decode, and where it lives

The carried MLA and shared-expert linears are the artifact's own
`exl3_base_quantization_config`: `{fmt: e4m3, scale_fmt: ue8m0,
weight_block_size: [128, 128], activation_scheme: dynamic}`. Measured on
`carried-001.safetensors` (2026-08-25, header + a 16-byte range read, no
download): `layers.2.attn.wq_a.weight` is `F8_E4M3 [1024, 4096]` at 4194304
bytes — ONE byte per element — beside `layers.2.attn.wq_a.scale` `F8_E8M0
[8, 32]` at 256 bytes, i.e. exactly `[ceil(rows/128), ceil(cols/128)]`.

`DequantFp8BlockToF32` is added to the fp8/nvfp4 dequant family
(`model_loader/nvfp4_dequant.{h,cpp}`, which already owns `DequantFp8ToBf16` and
`DequantFp8ChannelToBf16`) rather than being written inline in the DeepSeek
loader, so it is unit-gateable on its own and the next block-wise checkpoint
reuses it. Its semantics are taken from the arm that already executes this
format in this tree — `MatmulFp8BlockScaledKernel` (`src/vt/cpu/cpu_ops.cpp`),
the port of `native_w8a8_block_matmul` (`tests/kernels/quant_utils.py:91-154`):

```
out[o][i] = F8E4M3ToF32(w[o][i]) * E8M0ToF32(scale[o / block_n][i / block_k])
```

The scale MULTIPLIES (it is not a reciprocal), the scale grid is row-major over
`[ceil(N/block_n), ceil(K/block_k)]`, and a ragged final block is legal — all
three are the reference kernel's own conventions, and `col / block_n` indexing
the scale ROW by output column is `offs_bsn = offs_bn // group_n`
(`fp8_utils.py:823`). Nothing new is decided here; the decode is the existing
one, evaluated once at load instead of once per K-block in a GEMM.

### W1c-4. The REAL artifact's DSA geometry still refuses, and why that is right

Measured from the real `carried-001.safetensors`, the compressor and indexer
tensors are TWICE as wide as the host forward's collapsed shapes:
`attn.compressor.wgate.weight` is `[1024, 4096] = [2*head_dim, H]`,
`attn.indexer.compressor.wkv.weight` is `[256, 4096] = [2*index_head_dim, H]`,
and `attn.indexer.wq_b.weight` is `[8192, 1024] = [inh*ihd, q_lora_rank]` — it
projects from the q-LoRA, not from the hidden state. This is the `coff = 2`
compressor width `deepseek_v4.cpp` already documents at the `dsa_dense` comment
(`ds4.c:5016-5021`), and 41 of the real artifact's 43 layers carry a compressor.

So the loader REFUSES BY NAME on those three shapes and names the residual. It
does not improvise, and it does not widen the host slot to a shape the forward
would then mis-index. WITHDRAWN AS WRITTEN by #1970, and recorded here because
this sentence is where the claim started: it said `Gemm`'s host arm is a `MatVec`
with no length check, so a `[2*hd, H]` buffer read as `[hd, H]` is a silently
wrong number. `deepseek_v4.cpp:413` is an unconditional `VT_CHECK` and `Gemm`'s
keep-quant arm checks the shape too, so what that produces is an ANONYMOUS
`vt: MatVec weight size mismatch` naming no tensor and no layer. Refusing by name
buys a DIAGNOSTIC over that, which is still the `.agents/verification.md` concern,
and it is not the difference between wrong tokens and a refusal.

**The obvious fix is wrong and the reason is worth recording.** The GGUF arm
dodges the same geometry by setting `dsa_dense = (be.gguf != nullptr)` and
running dense MLA, which is EXACT for `seq_len <= index_topk`. Extending that
predicate to `|| be.exl3 != nullptr` would break THIS row's own equivalence
gate: the gate compares the EXL3 arm against a dense forward over
`Exl3DequantLinear` of the same trellis, and that reference has
`be.exl3 == nullptr`, so the two arms would take DIFFERENT attention paths and
the 2.0e-2 bound would be measuring a geometry change rather than the format
identity. A shared dense-MLA policy that both arms read is the correct shape and
it is a policy decision this row does not own; it is filed under `## Owed`.

### W1c-5. The row-naming refusal at `deepseek_v4.cpp` is DELETED as dead

#1923's second finding: the EXL3-specific `has_host_weights` refusal inside
`DeepseekV4ForwardExl3` was already unreachable on the default path, because the
runner's default `gather` (`src/vllm/v1/worker/gpu/runner.cpp:1716`) routes to
`ForwardDevice`, whose generic `kHostPending` check fires first.

It is now unreachable on EVERY path, and that is a stronger statement than
"hard to hit": the ONE arm that sets `has_exl3_weights` is `LoadDeepseekV4Exl3`,
and after this wave that arm sets `has_host_weights` in the same function before
returning. `has_exl3_weights && !has_host_weights` cannot be produced by a load.
Making it reachable would mean inventing a production path that reaches a
forward with half a tower, which is the opposite of what the row wants. It is
deleted, its test case with it, and the `has_exl3_weights` check that precedes
it stays. `MoeBlock` gains the check that actually pays: the host routed-expert
arm now refuses BY NAME when `exp_w1` is empty, which is the state a
reachability mutation that deletes the EXL3 dispatch lands in.

### W1c-7. The materialization reads UNALIGNED, and the fixture proves it

The first form of this wave's readers cast the mmap'd safetensors payload
straight to a typed pointer — `reinterpret_cast<const uint16_t*>(t.data)` on the
BF16/F16 arms and `reinterpret_cast<const int64_t*>(t.data)` on the `tid2eid`
arm — and indexed it. A safetensors tensor begins at whatever byte offset the
header's `data_offsets` names, after a header whose length is arbitrary, so
those pointers satisfy no alignment above 1 and forming them is undefined
behaviour. x86 performs the load and returns the right answer, which is why the
whole local suite was green; `sanitize-cpu (address,undefined)` reported it on
the pull request, on the BF16 arm of `Exl3CarriedReader::Float` (`const short
unsigned int`) and the I64 arm of its `HashTable` (`const long int`), failing
all three of this wave's tests. The verbatim reports, line numbers included,
are in the evidence section below; they anchor into the head that carried the
defect and not into this tree.

The tree already had the remedy and the reason recorded: `vt::LoadUnaligned<T>`
(`include/vt/unaligned.h`, issue #627), which Qwen3.5's GDN loader, Nemotron-H,
Voxtral, Olmo2, Phi, Parakeet and the LTX-2 loader all use for exactly this. No
new helper was written. Every arm of the materialization was swept, not only the
two UBSan named: the `Float` F32 arm and the `HashTable` I32 arm are bulk
`std::memcpy`, which has no alignment precondition, and `Fp8Block` hands both
mmap'd buffers to `DequantFp8BlockToF32`, which reads them one `uint8_t` at a
time — `alignof(uint8_t) == 1`, so an arbitrary offset satisfies it. All three
carry a comment saying so, because "this one is safe" is exactly the fact a
later edit needs and cannot re-derive from the code.

The durable half is the fixture. `WriteSafetensors` now pads its JSON header
with spaces — the same in-spec padding HuggingFace's own writer uses, in the
opposite direction — until the payload base is ODD, so every entry at an even
`data_offset` lands at an address that satisfies no alignment above 1.
`kMisalignedPayloadBase` states that guarantee, and
`test_deepseek_v4_exl3_loader.cpp`'s MISALIGNED case asserts it on the three
tensors the widening readers actually consume (`embed.weight` BF16,
`hc_head_fn` F32, `layers.0.ffn.gate.tid2eid` I64) before driving the
production loader over them and recomputing every value. Without that
precondition the pin would be vacuous: an accidentally aligned fixture exercises
nothing and the sanitizer lane goes quiet without the bug being gone. Setting
`kMisalignedPayloadBase = 0` reds the case, which is the mutation that proves
it.

### W1c-6. Residency

`ReportDeepseekV4Exl3Residency` prices the trellis tower against
`MemAvailable`. It now prices the host tower too, because the host tower is no
longer zero: on the real artifact it is ~29 GB of f32 beside ~84 GiB of
trellis, and a refusal that measured only one of them would let the other take
the box down. The reporter keeps its existing shape — injected budget, 0 means
unknown and never refuses, `VT_DSV4_EXL3_HOST_BUDGET=0` for the same-binary
escape — and the projection stays per-layer-exact for the trellis while the host
tower is measured, not projected, because the model-level tensors (`embed`,
`lm_head`) are not per-layer.

## Risks

1. The artifact itself is `runtime_pending` per its publisher — a correctness
   failure may be the checkpoint's, not ours; the SparkInfer run
   disambiguates.
2. The trellis GEMM is a large kernel port; the staged fallback (dequant-bf16
   on fixtures, per-expert dense GEMM loop before the fused mgemm) keeps every
   wave gateable.
3. Their 44-47 includes K5 speculative decoding; the AR-vs-AR comparison is
   the honest kernel-level bar and the end-config comparison the product bar.
4. **W2's parity gate has to decide an ulp bound, not discover one.** Three
   different f32 summation orders are in play for the same weight: W1a mirrors
   `get_weight_tensor`'s blockwise `preapply_had_*` with an fp16 round per
   stage; upstream's inference `had_r_128` (`quant/hadamard.cu:88-110`) is the
   same Sylvester butterfly scaled ONCE at the end by
   `r_scale = scale * 0.088388347648f` (which is 1/sqrt(128)); and W2b's
   `exl3_gemm` dequants inline through tensor-core mma. This spec's "byte-parity
   gates against the W1 CPU reference" holds for anything that does not sum — the
   codeword unpack and the codebook decode are exact — and cannot hold for
   anything that does. W2's own spec states the bound BEFORE its gate runs;
   widening a tolerance after a red is the failure this note exists to prevent.
   The convergence argument is recorded at the source site in
   `src/vt/cpu/cpu_exl3_dequant.cpp`, where W2's implementer will read it.
5. **`Exl3DequantLinear`'s `out` is `float*` carrying fp16-valued data.** That is
   correct for a CPU reference — every stage rounds through fp16 and f32 is just
   the carrier — but a W2 destination that INHERITS the width is the "dtype too
   wide" defect a token gate cannot see: the tokens still match and the goldens
   still pass while the path moves twice the bytes (AGENTS.md, "Inherit vLLM
   defaults"). W2's destination dtype is a decision its spec makes against
   upstream's own memory format, not one inherited from this signature.

## Gates

| Gate | Owner |
|---|---|
| W1: fixture round-trip byte-parity + hermetic loader red→green + full CPU ctest + preflight | implementer |
| W1 review: mutate the MCG constants, the window offset, a slice boundary — each must go red | reviewer |
| W2a: `had_r_128` CPU-vs-CUDA BYTE parity (`## W2 design` §1 tier 2) | implementer |
| W2b: trellis decode BYTE parity (tier 1) + `exl3_gemm` vs the f64 reference within the stated bound (tier 3) | implementer |
| W2: the shape-selection table resolves shape 2 for both expert shapes, HOST-side | implementer |
| W2: the routed-expert dispatch is REACHED from `ModelRegistry::Forward`; deleting the call site goes RED | implementer/reviewer |
| W2d: the token-sorted batching is upstream's, host-side (`## W2cd design` W2d-3) | implementer |
| W2d: the fused MoE arm agrees with the per-expert loop arm within tier 4 (2.0e-2 relative RMS) | implementer |
| W2d: `VT_DSV4_EXL3_FUSED_MOE=0` runs the loop arm and reaches the SAME answer | implementer |
| W2d: the FUSED dispatch is REACHED from `ModelRegistry::Forward`; deleting it goes RED | implementer/reviewer |
| W2c: the GEMV envelope is upstream's, value for value, and resolves w2 to config 0 host-side | implementer |
| W2c: the GEMV arm meets tier 3c (6.0e-3 relative RMS) on the device | operator |
| W1c: the forward suite reaches the forward through `LoadDeepseekV4ForCausalLMWeights`, not a hand-built struct; deleting the loader's materialization OR its `has_host_weights = true` goes RED | implementer/reviewer |
| W1c: the materialized carried VALUES equal the checkpoint's, decoded — BF16 widened, F32 straight through, block-wise FP8 times the UE8M0 scale of ITS block; reading block [0] for every element goes RED | implementer |
| W1c: a carried tensor whose geometry the host forward cannot index REFUSES BY NAME, naming both shapes (the real artifact's `2 * head_dim` DSA tensors) | implementer |
| W1c: the residency refusal prices the carried host tower as well as the trellis; dropping the host term goes RED | implementer |
| W2: e2e greedy token gate vs the W3 oracle | operator |
| W3: oracle gateability file; denominator run; speed table (values + ratios, idle box, 3 reps) | operator |

## Evidence

Spike 2026-08-24 recorded on #1875 (format constants, shard-header
verification, effort estimate).

### W1a + W1b (2026-08-24, CPU-only build, `-DVLLM_CPP_CUDA=OFF`)

Red-first, both waves. W1a's first red was five missing `vt::Exl3*` symbols
(`ninja` rc=1); a zero-filling stub then produced the assertion-level red
(`3 failed | 40 failed assertions`). W1b's red was the absent
`DeepseekV4Weights::exl3` member (`ninja` rc=1). Green:

| suite | cases | assertions |
|---|---|---|
| `test_exl3_dequant` (W1a) | 3 / 3 | 66 / 66 |
| `test_deepseek_v4_exl3_loader` (W1b) | 4 / 4 | 46 / 46 |

The loader figure was recorded here as `44 / 44` and is wrong: the suite emitted
`46 / 46` at the wave's own head. Corrected after the fresh review measured it
twice and the repair measured it a third time, by running the binary rather than
by copying either number.

**The format was re-derived, not assumed.** The 16-bit codeword for weight `t`
is the window ENDING at `t` — `t`'s own K bits lowest, then `t-1`, `t-2` …
wrapping around the 256-weight tile. That falls out of `pack.cu:29-57` writing
16 spans of 16 weights MSB-first and `SWAP16` (`:56`) turning the stored int16
array into a big-endian bit stream under the uint32 view `dq` reads
(`exl3_dq.cuh:15-31`). `lop3.b32 … 0x6a` is `(a & b) ^ c`, so the MCG codebook
is exactly `x *= 0xCBAC1FED; x = (x & 0x8fff8fff) ^ 0x3b603b60;` plus an fp16
sum of the two halves (`codebook.cuh:67-75`). `get_hadamard(128)` finds no
`hadamard_128.txt` in `util/hadamard_data/` and recurses Sylvester down to
`hadamard_1.txt` = `+`, so H128 is the natural-order Sylvester matrix and the
fast Walsh-Hadamard butterfly computes it exactly.

**Independence of the W1a gate.** The synthetic case builds its trellis from the
ENCODE side (window composition transcribed from `tests/test_quant_fn.py:104-112`,
span packing from `pack.cu:29-57`) and its expectations from its own MCG
transcription; the implementation only ever runs the DECODE side. All 256
windows of all 8 K values come back byte-identical. The full ladder is checked
against a dense double-precision blockwise-Hadamard reference at k=n=256 (two
128-blocks per side, so a whole-tensor transform fails).

**Real-checkpoint anchor, and its honest provenance.**
`layers.0.ffn.experts.0.w1.rank1` of
`exl3-layer-000-tp4-rank1.safetensors` (trellis `[256,32,48]`, suh `[4096]`,
svh `[512]`, K=3) dequants to k=4096 n=512, all finite, mean 3.12e-5,
std 0.024504, absmax 0.187378. Ten pre-Hadamard `reconstruct` spot values match
BYTE-EXACTLY (they are pure fp16 codebook values, no summation) and ten
full-weight spot values match to under two fp16 ulps.
**These literals were NOT produced by running upstream's kernel** — `ext.reconstruct`
is a CUDA extension and the implementer host has no GPU. They come from a
throwaway script whose trellis half is a second hand transcription and whose
Hadamard half is upstream's OWN `preapply_had_l`/`preapply_had_r` over the
Sylvester H128, executed by torch 2.11.0. Running upstream's own kernel on this
shard is W3a's job (`.agents/oracles/exllamav3.md`), and until then this anchor
is a transcription cross-check, not an oracle result.

**The NAS manifest exists.**
`/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3/SHA256SUMS`, written
2026-08-24, 381 lines of `sha256  size  path`. It covers all 190 files of the
staged artifact — the 177 `*.safetensors` shards (5 `carried-*` + 43 layers x 4
ranks) plus the 13 config/manifest files — and the 191
`.cache/huggingface/download/*.metadata` sidecars beside them. The one file in
the directory it does not list is `SHA256SUMS` itself. This closes what `## Owed`
carried as the outstanding manifest item; the download it attests completed
2026-08-24.

**Carried-tensor accounting, measured on the COMPLETE 190-file artifact**
(download finished 2026-08-24): 5549 carried tensors; the safetensors name-map
requires 1564 and finds **0 missing**; of the 3985 leftovers **every single one
is `mtp.*`**. So the carried half needs no new mapping at all. (The
`attn_compressor_{ape,gate,kv}` spelling belongs to the GGUF `blk.N.*` map, a
different vehicle; the safetensors arm already uses this checkpoint's own
`layers.{L}.attn.compressor.{ape,norm.weight,wgate.weight,wkv.weight}`.)

**The MTP tail is NVFP4, not EXL3.** `mtp.{0,1,2}.ffn.experts.{0..215}.{w1,w2,w3}.weight`
are I8 `[2048,2048]` with `.scale` F8_E8M0 `[2048,128]` — e2m1 packed
two-per-byte with ue8m0 block scales, the config's
`packed_e2m1_fp4_with_ue8m0_scales`. Only the MAIN model's experts were
requantized to EXL3. The loader SKIPS `mtp.*`, mirroring vLLM's own
`AutoWeightsLoader(skip_substrs=["mtp."])` (nvidia/model.py:1474), but COUNTS
the skip into `DeepseekV4Exl3Weights::skipped_mtp_tensors` so it is visible
rather than silent. Reaching those weights (they are why the upstream repo can
run a K5 speculative draft) is a later row's work.

**IMP-MUTATE, run by the implementer** (each: apply, verify the source sha
CHANGED, rebuild, run, then restore and verify the sha matches the original
byte-for-byte and rebuilds clean). All eight went RED:

| mutation | verdict |
|---|---|
| MCG multiplier `0xCBAC1FED` -> `0xCBAC1FEC` | RED |
| MCG xor constant `0x3b603b60` -> `0x3b603b61` | RED |
| window offset `bits - 16` -> `bits - 15` | RED |
| Hadamard scale `1/sqrt(128)` -> `1.0` | RED |
| tensor-core permutation column `+8` -> `+4` | RED |
| w2 slice axis IN -> OUT | RED |
| trellis concat drops the per-rank dim-1 offset | RED |
| **reachability**: the `IsExl3Checkpoint` call site deleted | RED |

The last one is the "Nothing lands dead" check: with the production branch
removed the loader suite fails, so the arm is genuinely reached from
`LoadDeepseekV4ForCausalLMWeights` — which is what `deepseek_v4_registry.cpp:89`
calls, and the suite also drives `ModelRegistry::Load` end to end.

### W1 fresh-review repair (2026-08-24, same CPU-only build)

The fresh review returned `PASS` with six MINOR findings and two NITs — no
correctness defect. Two of them were code, and both are gated:

- **The detection predicate was gated only in the POSITIVE direction.** The
  reviewer widened `quant_method == "exl3"` to an always-true test and FOUR dsv4
  suites stayed green, because nothing drove the loader with a non-EXL3
  `quantization_config`. That is not a theoretical vehicle: this checkpoint's own
  `quantization_config.base_quantization_config` is the plain `deepseek_v4_fp8`
  block (`{activation_scheme: dynamic, fmt: e4m3, quant_method: fp8, scale_fmt:
  ue8m0, weight_block_size: [128, 128]}`, read from the staged `config.json`), so
  a regression that widened the predicate would carry the FP8 artifact into the
  trellis arm undetected. A dense-expert fixture now loads through the arm the
  predicate must select, in two subcases: `quant_method: "fp8"`, and no
  `quantization_config` block at all. The second does NOT discriminate the
  `== "exl3"` widening — the null guard already returns false for it — and says
  so at the source; it guards a predicate that drops that guard.
- **`DeepseekV4Exl3ResidentBytes` had ZERO production call sites.** Only the
  byte-parity case called it, which measures a function rather than a capability
  ("Nothing lands dead"). It is also exactly the instrument the residency risk
  needs: the reviewer priced the real artifact's trellis alone at ~83.5 GiB (43
  layers x 216 experts x 3 projections x ~3 MiB) on a box whose unified memory
  OOM-reboots. `ReportDeepseekV4Exl3Residency` now runs per layer inside
  `LoadDeepseekV4Exl3`: it prices what is committed, REFUSES a tower the host
  cannot hold, and reports `resident_bytes=` once the last layer is in. The
  budget is a PARAMETER (`host_available_memory_bytes()` in production, 0 ==
  unknown == never refuse), mirroring `check_enough_state_memory`'s shape
  (`kv_cache_utils.h`, issue #371) so the refusal is gateable without a machine
  of a chosen size. The projection is exact rather than extrapolated: every MoE
  layer of this schema carries the same experts at the same two shapes.

Green after the repair, both suites re-measured by running the binaries:

| suite | cases | assertions |
|---|---|---|
| `test_exl3_dequant` | 3 / 3 | 66 / 66 |
| `test_deepseek_v4_exl3_loader` | 6 / 6 | 66 / 66 |

`ctest -R 'exl3|deepseek_v4'` is 13 / 13 on the CPU-only build. One of the 13,
`test_cuda_deepseek_v4`, reports `assertions: 0` — a CUDA-off skip wearing a
pass, pre-existing and unrelated to this repair.

**IMP-MUTATE for the repair.** Each: apply, verify the source sha CHANGED,
rebuild, run, restore, verify the sha matches byte-for-byte, rebuild clean.

| mutation | verdict |
|---|---|
| `IsExl3Checkpoint`: `== "exl3"` -> `!= "__never__"` | RED (`REQUIRE(msg.empty())`, the fp8 fixture refused by the EXL3 arm's `version` check) |
| the `ReportDeepseekV4Exl3Residency` production call site deleted | RED (2 assertions: no `[vt load] dsv4-exl3:` line, no `resident_bytes=`) |

The second mutation was run TWICE, and the first run is the finding. Deleting
the call site alone left `host_available` unused, which `-Werror` rejects: the
build failed, `ninja` stopped, and the STALE binary then printed
`6 passed | 0 failed | SUCCESS`. A mutation that does not compile reads as a
passing test. The compilable form adds `(void)host_available;` beside the
deletion, and only that form is evidence.

The negative-direction subcases drive the load through `ThrowMessage` rather
than bare, for the same class of reason: an uncaught throw is a failed CASE
under a summary line reading `assertions: N | N passed | 0 failed`, so the red
would be invisible in exactly the place a reader looks.

### W1 fresh-review round 2 (2026-08-25, same CPU-only build)

The scoped fresh review of the repair above also returned `PASS`, with three
MINORs and two NITs — again no correctness defect, and again the load-bearing
one was a guarantee nobody could observe.

- **The new refusal was effectively ungated (MINOR-1).** Both log assertions
  (`[vt load] dsv4-exl3:` and `resident_bytes=`) match BOTH branches of the
  report, so the reviewer replaced `host_available` with a literal `0` at the
  production call site — symbol kept referenced, so it compiled — and the suite
  stayed `6 / 6`, `66 / 66`, `SUCCESS`. With that wiring the load printed
  `host MemAvailable unknown (/proc/meminfo unreadable)` on a host where
  `/proc/meminfo` reads fine: a FALSE statement in production output beside a
  silently inert refusal. The case now branches on
  `vllm::v1::host_available_memory_bytes() > 0` and, when the host can read the
  pool, requires the `host MemAvailable=` branch, forbids the unknown branch, and
  PARSES the printed GiB figure to check it against a second reading of the same
  pool. The window is 1/8x .. 8x because MemAvailable moves under other work on
  the box between the load and the check; it is still far tighter than the 1 MiB
  and 1 TiB brackets the refusal cases inject, so a call site rewired to any of
  those constants fails.
- **Line-number citations into this row's own files (MINOR-2).** The repair
  commit cited `deepseek_v4_weights.cpp:609-613` for `expert_suffixes` and the
  SAME commit inserted 71 lines above it, so the anchor was stale before anyone
  read it (`expert_suffixes` sat at 680-684 at that head). Every such citation
  into a file this row edits is now by SYMBOL: that one, `kv_cache_utils.h:518`
  and `kv_cache_utils.cpp:944-963` in the same neighbourhood, and this spec's own
  `deepseek_v4_weights.cpp:137` / `:269` (both already wrong). NOT fixed, because
  they are outside this row's authority and outside its files.

  **The out-of-file citations W1b broke ARE now repaired**, under an operator
  extension of this row's authority, because a record edit rides in the pull
  request whose change made the record stale. W1b inserted 1 line at ~59 and 399
  lines at ~187 of `deepseek_v4_weights.cpp`, and 55 + 13 lines at ~231 / ~299 of
  `deepseek_v4.h`; each citation below was VERIFIED correct at `16f9112a0^` and
  wrong at this head, so this row broke it. Each is now by SYMBOL, the durable
  form, and no code changed:

  | citation site | was | now names |
  |---|---|---|
  | `laguna_weights.cpp:45` | `deepseek_v4_weights.cpp:84-100` | `RawInt`/`RawDouble`/`RawBool`/`RawString` |
  | `laguna_weights.cpp:478` | `deepseek_v4_weights.cpp:410-852` | `struct V4GgufCtx` .. `LoadDeepseekV4FromGguf` |
  | `laguna.h:412` | `deepseek_v4.h:359` | `DeepseekV4KvCache::decode_graph` |
  | `.agents/specs/deepseek-v4-pro.md` §3 | `deepseek_v4_weights.cpp:173` | the `head_dim == 512` `VT_CHECK` in `ParseDeepseekV4Params` |
  | `.agents/specs/deepseek-v4-pro.md` mutation A | `deepseek_v4_weights.cpp:111` | `ParseDeepseekV4Params` |
  | `.agents/specs/deepseek-v4-mtp.md` §2 | `deepseek_v4_weights.cpp:123` | `ParseDeepseekV4Params` |

  A tree-wide sweep for `deepseek_v4_weights.cpp:<n>` / `deepseek_v4.h:<n>` found
  four more that this row did NOT break, and they are left alone so the row that
  broke them owns them: `deepseek-v4-pro.md` mutation C's
  `deepseek_v4_weights.cpp:132` (claims `p.o_groups`, was `p.sliding_window`
  already at `16f9112a0^`), `deepseek-v4-mtp.md`'s `:498` (claims the GGUF
  `nextn_predict_layers` parse, was `d.num_attention_heads`) and `:563-566`
  (claims the `compress_ratios` prefix, was the vocab resolution), and
  `deepseek-v4-pro.md`'s `deepseek_v4.h:127-128` / `:127`, which are still
  CORRECT (`has_compressor` / `has_indexer`) because W1b inserted below them.
  `.agents/benchmark-record.md` also carries line citations; it is an
  append-only historical log and is never rewritten.
- **What the budget is, and is not (MINOR-3).** `/proc/meminfo` MemAvailable is
  an ESTIMATE and is wrong in both directions for this use: it ignores swap and
  under-counts some reclaimables, so it can refuse a tower this host would have
  held; and inside a container it reports the HOST's figure rather than the
  cgroup's limit, so a memory-capped container gets NO protection while the
  logged budget names a pool the process cannot draw on. Both caveats are now
  recorded at the read site in `LoadDeepseekV4Exl3`, and — because a heuristic
  that can be wrong must be overridable — the refusal ships with
  **`VT_DSV4_EXL3_HOST_BUDGET`**: default ON (the refusal enabled), and a
  '0'-leading value hands the reporter an UNKNOWN budget (0), which never
  refuses. That is the house default-ON / '0'-rollback shape, and the parse is a
  pure predicate in the header (`Dsv4Exl3HostBudgetFlagIsOn`) exactly as
  `AsyncRunnerFlagIsOn` is, so it is unit-gated without mutating the environment.
  The refusal MESSAGE names the flag, which is how a blocked developer finds it;
  the name is documented in `docs/ENVIRONMENT.md`, beside the residency budget
  knob `VT_DEVICE_WEIGHT_BUDGET_BYTES` it is analogous to, and it is registered
  exactly once — the allowlist entry W2's first pass added was removed in the
  same change. `scripts/check-env-doc.py:5-8` splits the two surfaces into
  user-facing/behavior-changing knobs (`docs/ENVIRONMENT.md`) and kernel-internal
  tuning (`scripts/env-doc-allowlist.txt`), and a flag that disarms a refusal
  which otherwise BLOCKS a model load is the first kind: a developer who hits the
  refusal must be able to find it in the docs and not only in the refusal string.
  The entry records the default-ON polarity, the '0'-leading disable, that the
  budget is read from `/proc/meminfo` MemAvailable, and both caveats above.
  `scripts/check-env-doc.py` rc 0, 377 vars.
- **The inclusive edge (NIT-1).** The bracketing cases (1 MiB / 1 TiB against a
  ~304 KiB tower) catch a direction flip but not `projected <= budget` narrowing
  to `projected < budget`. A `projected == host_available_bytes` case now pins it.
- **NIT-2 is recorded, not fixed, and says why at the subcase.** The "no
  `quantization_config` at all" subcase discriminates a dropped null guard only
  by SIGSEGV: the deref is inside `IsExl3Checkpoint` itself, so the process dies
  before any `CHECK` runs. MEASURED on this head by dropping the guard (below):
  the binary exits 139 and doctest prints `test cases: 5 | 4 passed | 1 failed |
  2 skipped` with `assertions: 63 | 63 passed | 0 failed` and
  `Status: FAILURE!` — a real ctest red under a clean assertion counter, and two
  later cases never ran at all. That counter must not be read as this case's
  verdict. Making it fail by assertion
  would mean replacing the deref with a checked accessor — i.e. deleting the very
  guard under test — and the sibling `!qc->is_object()` half cannot be
  discriminated at all, because `nlohmann::json::contains` is already safe on a
  non-object, so a string-valued `quantization_config` fixture would take the
  dense arm either way and assert nothing. A memory-safety defect's
  discriminator is a crash or a sanitizer, not a CHECK.

Green after the round-2 repair, both suites re-measured by running the binaries:

| suite | cases | assertions |
|---|---|---|
| `test_exl3_dequant` | 3 / 3 | 66 / 66 |
| `test_deepseek_v4_exl3_loader` | 7 / 7 | 82 / 82 |

`ctest -R 'exl3|deepseek_v4'` is 13 / 13. The twelve sibling dsv4 targets were
RELINKED against the changed `libvllm.a` before the run, because `ctest` never
builds and a stale binary reports green about code it does not contain.

**IMP-MUTATE for round 2.** Each records `ninja`'s rc AND its step count beside
the verdict, because on this row a mutation that fails to build has re-run the
stale binary and printed SUCCESS four times; zero steps is nothing rebuilt and
therefore not evidence. Each: apply, rebuild, run, restore, verify the source
sha256 matches the original byte-for-byte.

| mutation | ninja | verdict |
|---|---|---|
| the MINOR-1 wiring: production call site passes `0` for the budget, `(void)host_available;` keeping the symbol referenced | rc 0, 3 steps | RED — 4 assertions (`host MemAvailable=` absent, `MemAvailable unknown` present, `reported > 0.0`, `reported >= now_gib / 8.0`); cases 6 / 7, assertions 78 / 82 |
| NIT-1's inclusive edge: `projected <= host_available_bytes` -> `projected <` | rc 0, 3 steps | RED — 1 assertion, the `ExpectedTowerBytes() * 43` case; assertions 81 / 82 |
| `Dsv4Exl3HostBudgetFlagIsOn` polarity: `env_value == nullptr` OR-else `env_value[0] != '0'` -> `env_value != nullptr && env_value[0] == '0'` | rc 0, 7 steps | RED — 14 assertions across 2 cases: the 9 parse assertions AND the 4 MINOR-1 budget assertions, because the flipped default disables the refusal in the load itself, which is what proves the parse feeds production |
| NIT-2, to MEASURE the claim rather than transcribe it: the `qc == nullptr` half of the guard dropped, leaving `if (!qc->is_object()) return false;` | rc 0, 3 steps | RED **by crash**: binary rc 139, cases 4 passed / 1 failed / 2 skipped, assertions 63 of 63 passed and 0 failed, `Status: FAILURE!` |

Restoration verified by sha256 after every mutation:
`deepseek_v4_weights.cpp` `483ca7bd…`, `deepseek_v4.h` `e9129233…`, both matching
the pre-mutation bytes, and the restored tree rebuilds clean (rc 0, 7 steps) and
returns 7 / 7, 82 / 82.

### W2a + W2b (2026-08-25, CPU-only build, `-DVLLM_CPP_CUDA=OFF`)

**Which tree these numbers were measured on.** Everything below was first
measured at `8428f0692`, the wave's own commit. `origin/main` then moved five
commits and a merge (`5e690eef9`) landed on this branch — authored from another
session, not by the implementer that wrote the wave. Because main's five commits
touched `AGENTS.md`, `scripts/` and `tests/scripts/`, THE CHECKERS THEMSELVES
moved underneath the recorded results, so the gates were re-run at the merge
rather than assumed to carry:

| gate | at `8428f0692` | re-run at `5e690eef9` |
|---|---|---|
| `libvllm` + every W2 translation unit compiles | 0 errors | 0 errors |
| `test_exl3_gemm` | 13 / 13, 199 / 199 | **13 / 13, 199 / 199** |
| `test_deepseek_v4_exl3_forward` | 2 / 2, 11 / 11 | **2 / 2, 11 / 11** |
| `test_exl3_dequant` (W1a) | 3 / 3, 66 / 66 | **3 / 3, 66 / 66** |
| `test_deepseek_v4_exl3_loader` (W1b) | 7 / 7, 82 / 82 | **7 / 7, 82 / 82** |
| `test_deepseek_v4_forward` / `_moe` (siblings) | — | 6 / 6, 34 / 34 · 12 / 12, 716 / 716 |
| full `ctest` (592 tests) | 592 / 592, 155.38 s | **592 / 592, 330.27 s** (`BUILD_RC=0` over all 1643 targets) |

**A first attempt at the full re-run was abandoned on a FALSE PREMISE, and the
record says so because a wrong cause is worse than an unknown one.** That rebuild
was stopped at 838 of 1643 targets under the belief that a concurrent session had
starved it for about 55 minutes. It had not: the elapsed time was misjudged — the
job had been running roughly 20 minutes and was progressing normally — and the
box was healthy. The one part of that episode that was right is that the `ctest`
was chained behind the build and was never allowed to start, so no `CTEST_RC` was
ever written and nothing was inferred from a partial tree. The gap it left has
since been closed by MEASUREMENT rather than by explanation: the tree was rebuilt
clean at `45589ee5d` (`BUILD_RC=0`, all 1643 targets) and the full suite re-run,
with `ctest` gated on that exit status so a partial tree cannot be tested at all.
The figure now in the table above is that run, not the pre-merge one.

The merge changed 29 files and NONE of this row's source: `git diff 8428f0692
5e690eef9` over every W2 source file is empty. Two of the 29 do overlap what this
wave wrote — `docs/FEATURES.md` and `docs/USAGE.md` — and both were checked row
by row rather than trusted: the merged checkpoint registry holds 32 rows, main's
30 plus this row's 2, and the diff against main is exactly those two lines. Main
also moved `scripts/check-agent-record.py` and `scripts/check-gate-commands.py`,
which is the concrete reason the checker re-run was not optional.


**Red first.** `tests/vt/test_exl3_gemm.cpp` was written and registered before any
implementation: `ninja -C build test_exl3_gemm` rc 1 with EIGHT undefined symbols
— `vt::Exl3CcFromSm`, `Exl3GemmNumShapes`, `Exl3GemmShapeParams`,
`Exl3GemmShapeCompat`, `Exl3GemmNumSms`, `Exl3SelectGemmShape`, `Exl3HadR128`,
`Exl3Gemm`. Before that the build stopped one step earlier still, on
`src/vt/op_provider.cpp:278: enumeration value 'kExl3HadR128' not handled in
switch [-Werror=switch]`, which is the exhaustive `OpName` switch doing exactly
what its header says it is for. `tests/vllm/models/test_deepseek_v4_exl3_forward.cpp`
was likewise written before the `MoeBlock` arm and the `Forward` dispatch existed.

Green, both suites re-measured by RUNNING the binaries:

| suite | cases | assertions |
|---|---|---|
| `test_exl3_gemm` | 13 / 13 | 199 / 199 |
| `test_deepseek_v4_exl3_forward` | 2 / 2 | 11 / 11 |

Two of `test_exl3_gemm`'s thirteen are the DEVICE cases, which skip on this build.
They are not `assertions: 0` skips: each prints a named reason and the exact `rc
run` command, and each still asserts the precondition it skipped on
(`CHECK_FALSE(OpRegistered(..., kCUDA))`), so the counter above is honest about
what ran.

**The numbers the bounds were stated against, measured rather than quoted:**

| comparison | bound (spec `## W2 design`) | measured |
|---|---|---|
| `had_r_128` f32 arm vs an independent f64 Sylvester H128 | 1e-5 relative | worst 3.22e-7 at scale 3.71 |
| `exl3_gemm` CPU vs the f64 chain, RMS | 1.0e-3 | 4.21e-4 |
| `exl3_gemm` CPU vs the f64 chain, elementwise | 8 * ulp_f16(rms) = 0.0625 | 0.0185 |
| `exl3_gemm` CPU vs the W1a WEIGHT basis (tier 3b) | 2.0e-3 | 5.53e-4 |
| the forward's EXL3 arm vs the dequantized-dense arm | 2.0e-2 | 1.01e-3 |
| the forward's EXL3 arm vs an UNRELATED dense arm | > 1.0e-1 | 1.3746 |

The last two are the reachability pair. The fixture attaches the EXL3 tower to a
host tower whose `exp_w*` are UNRELATED random weights, so an arm that missed the
dispatch computes the third number and both checks fail at once — which is what
the R1 mutation below measures.

**The shape policy resolves as `## W2 design` §3 claims**, host-side and with no
device: `Exl3SelectGemmShape(kBlackwell, m, 4096, 2048, 3, false) == 2` and
`(kBlackwell, m, 2048, 4096, 3, false) == 2` for the w1/w3 and w2 shapes, and
both are `Exl3GemmShapeCompat` with shape 2. Nine further branches of upstream's
table (Ampere small/large, Ampere K>=5 mod_512, Ada K<=3 both ways, Blackwell
K==4 shape 1 with and without `multi`, Blackwell K>=7, Blackwell K=3 big-k) are
pinned in the same case, so a later edit to the table cannot move a row this
checkpoint does not take without a red.

**IMP-MUTATE.** Each records `ninja`'s rc AND its step count beside the verdict,
because on this row a mutation that fails to build has re-run the stale binary and
printed SUCCESS four times; zero steps is nothing rebuilt and therefore not
evidence. Each: apply, verify the source sha CHANGED, rebuild, run, restore,
verify the sha matches the original byte for byte, rebuild clean.

| mutation | ninja | verdict |
|---|---|---|
| **R1 reachability**: the `has_exl3_weights` dispatch in `DeepseekV4Model::Forward` disarmed by a `false` guard, the call site KEPT so it still compiles | rc 0, 3 steps | RED — `test_deepseek_v4_exl3_forward` 0 / 2 cases, 6 / 11 assertions |
| M2 policy: the Blackwell `K >= 3` threshold `size_k > 8192` -> `> 4095`, which moves w1/w3 from shape 2 to shape 3 | rc 0, 3 steps | RED — 12 / 13 cases, 197 / 199 assertions |
| M3 `had_r_128`: the xor-shuffle sign flip inverted (`(t & i) != 0` -> `== 0`) | rc 0, 3 steps | RED — 9 / 13 cases, 193 / 199 assertions |
| M4 `had_r_128`: `r_scale` literal `0.088388347648f` -> `1.0f` | rc 0, 3 steps | RED — 9 / 13 cases, 193 / 199 assertions |
| M5 `had_r_128`: the PRE scale applied AFTER the transform instead of before | rc 0, 3 steps | RED — 11 / 13 cases, 196 / 199 assertions |
| M6 `exl3_gemm`: the `svh` output scale dropped, `(void)&svh;` keeping the symbol referenced | rc 0, 3 steps | RED — 11 / 13 cases, 196 / 199 assertions |
| M7 policy: the empty-block clamp loses its `MAX(.., 1)` | rc 0, 3 steps | RED — 12 / 13 cases, 198 / 199 assertions |
| M8 wiring: the gate (`w1`) and up (`w3`) projections swapped in the EXL3 MoE arm | rc 0, 3 steps | RED — 1 / 2 cases, 10 / 11 assertions |
| M9 refusal: the `cols % 128` guard widened to `% 64` | rc 0, 3 steps | RED **and the run ended early** — 7 passed / 1 failed / **5 not run**, assertions 186 / 188 |

M9's shape is worth stating rather than hiding: with the guard widened, a
64-column call reaches a kernel that reads 128, the run does not finish, and the
assertion counter reads 188 rather than 199. It IS a red, and its counter is NOT
the case's verdict — the same trap this row recorded for NIT-2 in round 2.

Restoration verified by sha256 after every mutation, all MATCH:
`exl3_policy.cpp` `e9e3c5d55cd9`, `cpu_exl3_kernels.cpp` `28a2f46ba789`,
`ops.cpp` `95f288c426a5`, `deepseek_v4.cpp` `f66c162afb26`.

**One mutation was VOIDED and is recorded because voiding it is the point.** The
first attempt at R1 wrote a `&&` through a shell layer that mangled it into a
stray backslash; `ninja` returned rc 1 with 3 steps, and the harness refused the
run instead of executing the STALE binary and reporting SUCCESS. That is the
fifth time this row has met the trap, and the first time an instrument caught it
before a verdict was written down.

**The transcription itself was diffed against its source, mechanically.** The
GEMM tile loop is the largest block in the port, so it was normalised (comments
stripped, whitespace collapsed, C-style and named casts folded together, the
renames applied) on both sides and diffed against `exl3_gemm_inner.cuh:22-733`.
Every remaining difference is formatting or one of the four DELIBERATE removals,
each recorded at the top of the file: the sm_86-only `EXL3_GEMM_H_ACC` fp16
accumulator ladder, the `index_m` and `slice1_n` values upstream computes and
never reads, the `shmem_out_had` template parameter fixed to `true` (the non-MoE
kernel is the only caller), and the FRAG_STAGES 2 and 4 ladders no shipped shape
selects. That last one now carries a `static_assert`, because without it a shape
asking for 2 or 4 would compile to a kernel with an EMPTY main loop — a silent
no-op instead of a build error. No semantic slip was found. This is a diff, not
a compile, and it says nothing about whether nvcc accepts the file.

**One piece of the CUDA arm WAS checked without a compiler, and only one.** The
kernel reads eight codewords per lane in a single funnel-shift pair
(`dq8<3, cb, 4>`, `exl3_dq.cuh:96-161`) while W1a's `Exl3TileCodeword` reads one
at a time (`dq`, `:15-31`). Those are different index arithmetic for the same
256 codewords, and getting the 8-wide form wrong is the most likely silent defect
in the port. Both formulas were evaluated over a synthetic K=3 tile for all 32
lanes: `dq8`'s `w0..w7` equal `dq(t_offset + 0..7)` for every lane, 0 mismatches
out of 256 codewords, including the `& 0xffff` masks and the
`% (bits * 256 / 32)` tail-biting wrap.

This gates the FORMULA and NOT the file. It says upstream's 8-at-a-time window
read agrees with upstream's 1-at-a-time window read, which is what
`src/vt/cuda/cuda_exl3.cu` transcribes; it does not say the transcription in that
file is what a compiler will read, and nothing here should be read as if it did.
The tier-1 byte gate against the compiled kernel is still owed.

**The checkpoint reaches `docs/USAGE.md` in this change, not a later one.**
AGENTS.md owes the weights entry to the change that makes a capability
reachable, and this is it. Two rows were added to the checkpoint registry — one
trellis shard and one carried shard — with the sha256 and byte size READ BACK
from the staged artifact rather than copied from a record: `SHA256SUMS` on the
NAS lists `exl3-layer-000-tp4-rank0.safetensors` at
`2ed7ae79…` / 515,850,920 B and `carried-001.safetensors` at `3b67ae29…` /
4,288,630,252 B, and `ls -la` on the same two files reports the same sizes. The
directory holds 190 files (172 trellis + 5 carried + 13 config/manifest) totalling
106,863,039,931 B. Both rows say what does NOT work, because the checkpoint does
not yet run end to end.

**THE CUDA TRANSLATION UNIT HAS NOT BEEN COMPILED BY ANYBODY.** `which nvcc` on
the implementer host returns nothing and no CUDA toolkit is installed; installing
one is a large download this dispatch has no authority for. `dgx.casa`, the box
that would compile and run it, hung 2026-08-25 03:24Z and needs a manual power
cycle. So `src/vt/cuda/cuda_exl3.cu` is a 1:1 transcription that has passed NO
compiler, and the honest state of W2's device half is: written, reviewed against
its anchors line by line, and unverified. It is in the `cuda-fat-build` CI job's
scope automatically: it is one line of the `target_sources(vllm PRIVATE ...)`
block inside `if(VLLM_CPP_CUDA)` in `CMakeLists.txt`, which is the target that
job builds. So the first compile verdict will come from CI or from the box,
whichever answers first. Nothing here claims it builds.

### W2c + W2d (2026-08-25, CPU-only build, `-DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`)

**Which tree these numbers were measured on.** The branch is
`row/MODEL-DSV4-EXL3-W2CD`, cut from `row/MODEL-DSV4-EXL3-W2` at `2edf344fd`.
`origin/main` moved twice underneath it during the wave: first to `d0d4f1f60`
(the oracle registration) and then to `41c5bec53` + `d7d1ee914`, the second of
which is **PR #1899 squash-merged**, i.e. W2a+W2b's own commits arriving on main
in a different shape. Both merges were taken and the spec's four conflicts were
resolved by KEEPING this branch's side; the resolved file is byte-identical to
this branch's pre-merge version, which is the check that main brought nothing to
it beyond what the branch already carried
(`git diff b3413601d -- .agents/specs/model-dsv4-exl3.md` was empty at the merge).

**The build type changed, and it is recorded rather than left to be noticed.**
The first full build died at target 639 of 1211 with
`/usr/bin/ld: final link failed: No space left on device` — the box was at 100%
with 693 MB free and this worktree's `RelWithDebInfo` build alone held 20 GB. The
build was reconfigured `-DCMAKE_BUILD_TYPE=Release`, which differs from
`RelWithDebInfo` only in `-g`; both define `NDEBUG`, so no assert changes state
between them. That ENOSPC is the failure this tree already files under "a
checker starved of disk emits a false policy refusal", and it is named here so a
reader does not read the first log as a code verdict.

**Red first, both suites.** `tests/vt/test_exl3_gemv.cpp` and
`tests/vt/test_exl3_moe.cpp` were written and registered before any
implementation existed.

| suite | red command | rc | the failure |
|---|---|---|---|
| `test_exl3_gemv` | `ninja -C build test_exl3_gemv` | 1 | six missing symbols: `vt::Exl3GemvSelectConfig`, `Exl3GemvHardEligible`, `Exl3GemvParseMode`, `Exl3GemvParseSmemMode`, `vt::kExl3GemvMaxM`, and `Exl3GemmArgs::force_gemv` |
| `test_exl3_moe` | `ninja -C build test_exl3_moe` | 1 | thirteen missing `vt` symbols, `vt::OpId::kExl3MoeMlp` and `vt::Exl3MoeAct` among them |

Green, both re-measured by RUNNING the binaries:

| suite | cases | assertions |
|---|---|---|
| `test_exl3_gemv` | 6 / 6 | 43 / 43 |
| `test_exl3_moe` | 8 / 8 | 41 / 41 |
| `test_deepseek_v4_exl3_forward` (fused arm) | 4 / 4 | 27 / 27 |
| `test_deepseek_v4_exl3_forward` (`VT_DSV4_EXL3_FUSED_MOE=0`) | 4 / 4 | 27 / 27 |

Full `ctest`: **619 / 619, 0 failed, 337.44 s**, `CTEST_RC=0`, chained BEHIND
`BUILD_RC=0` over all 1720 targets so a partial tree cannot be tested at all —
the shape W2a+W2b's evidence adopted after its own abandoned run. All seven EXL3
entries pass inside it, `test_deepseek_v4_exl3_forward_loop_arm` among them.
Five tests report `Skipped` and none is this row's: `test_modelopt_mixed_precision_checkpoint`,
the two `minimax_music3` real-artifact arms, `test_voxtral_e2e` and
`test_qwen35_paged_engine`.

**One trap was met and avoided in the harness rather than in the numbers.** The
`ctest` log was appended to a scratchpad file that already held a COMPLETE run
from the PREVIOUS wave's worktree — `dsv4-w2/build`, 592 tests, "100% tests
passed". Reading the tail of that file for a pass would have recorded another
tree's result as this one's. The run above is the portion after the boundary
line, and its own header names `dsv4-w2cd/build` and 619 tests. This is the
`the state was not the one you believed` shape, and the only defence is that the
verdict names the tree it was measured on.

One case in each of the two new suites is a DEVICE case that skips on this
build. Neither is an `assertions: 0` skip: each prints a named reason and the
exact `rc run` command, and each still asserts the precondition it skipped on.

**The numbers the bounds were stated against, measured rather than quoted:**

| comparison | bound (`## W2cd design`) | measured |
|---|---|---|
| the fused MoE CPU arm vs the per-expert loop arm, relative RMS | tier 4, 2.0e-2 | **6.73e-4** |
| the same comparison against an UNRELATED routing | > 1.0e-1 | **1.26928** |
| the forward's EXL3 arm vs the dequantized-dense arm, FUSED | 2.0e-2 | **1.37068e-3** |
| the forward's EXL3 arm vs the dequantized-dense arm, LOOP | 2.0e-2 | **1.0147e-3** |
| the forward's EXL3 arm vs an UNRELATED dense arm | > 1.0e-1 | **1.37465** |

The last three are worth reading together. `## Evidence`'s W2a+W2b table recorded
**1.0147e-3** for this comparison, and that value is REPRODUCED here by the
`VT_DSV4_EXL3_FUSED_MOE=0` run — which is the check that the rollback is a
rollback. The fused arm's **1.37068e-3** is a different number for the reason
W2d-7 states in advance: it keeps the intermediate in fp16 through the
activation where the loop widens to f32 and back.

**The launch count is not a claim, it is a counter.** A value gate CANNOT see
the fused dispatch, because deleting it leaves the per-expert loop to pick the
work up and the logits stay right — which is exactly what makes that loop a
genuine tail path rather than dead code. So the reachability case reads
`vt::OpProviderStats::selections`, the positive signal `include/vt/op_provider.h`
exists for, over one production `DeepseekV4Model::Forward`:

| arm | `kExl3MoeMlp` selections | `kExl3Gemm` selections |
|---|---|---|
| default (fused) | **2**, one per MoE layer | **0** |
| `VT_DSV4_EXL3_FUSED_MOE=0` | **0** | **36** = 3 tokens x 2 experts x 3 projections x 2 layers |

Both rows are asserted, in the SAME case, which ctest runs twice — once plain
and once under the flag. That is why `tests/CMakeLists.txt` carries
`test_deepseek_v4_exl3_forward_loop_arm`.

**IMP-MUTATE.** Each records `ninja`'s rc AND its step count beside the verdict,
because on this row a mutation that fails to build has re-run the stale binary
and printed SUCCESS five times; zero steps is nothing rebuilt and therefore not
evidence. The harness applies the edit, asserts the source sha CHANGED, rebuilds,
REFUSES to run at rc != 0 or 0 steps, runs, restores, asserts the sha is the
original byte for byte, and rebuilds clean.

| mutation | ninja | verdict |
|---|---|---|
| **M1 reachability**: the `Exl3FusedMoePass` dispatch in `MoeBlock` disarmed by a `false` guard, the call site KEPT so it still compiles | rc 0, 3 steps | RED — `test_deepseek_v4_exl3_forward` 3 / 4 cases, 25 / 27 assertions |
| **M11 rollback reachability**: the loop arm's skip inverted, so it declines the experts the fused arm declined too, under `VT_DSV4_EXL3_FUSED_MOE=0` | rc 0, 3 steps | RED — 2 / 4 cases, 24 / 27 assertions |
| M2 activation: the clamp moved AFTER the silu — upstream's order, not vLLM's | rc 0, 3 steps | RED — `test_exl3_moe` 7 / 8 cases, 40 / 41 assertions |
| M3 batching: the grouping walks the tokens BACKWARDS, so it is no longer stable | rc 0, 3 steps | RED — 7 / 8 cases, 38 / 41 assertions |
| M4 batching: the `max_tokens_per_expert` cut made exclusive (`<=` -> `<`) | rc 0, 3 steps | RED — 7 / 8 cases, 40 / 41 assertions |
| M8 refusal: the f32 accumulator check widened to admit f16 | rc 0, 3 steps | RED — 7 / 8 cases, 39 / 41 assertions |
| M9 wiring: the gate and up OUTPUT scales swapped in the fused stage 3 | rc 0, 3 steps | RED — 6 / 8 cases, 39 / 41 assertions |
| M10 epilogue: the routing weight dropped from the fused scatter-add, `(void)w;` keeping the symbol referenced | rc 0, 3 steps | RED — 7 / 8 cases, 40 / 41 assertions |
| M5 GEMV envelope: the `size_k` threshold at `:67` widened 2048 -> 4096 | rc 0, 3 steps | RED — `test_exl3_gemv` 4 / 6 cases, 41 / 43 assertions |
| **M6 GEMV envelope: the COMMENTED-OUT Ampere-only guard RE-ENABLED** | rc 0, 3 steps | RED — 3 / 6 cases, 27 / 43 assertions |
| M7 GEMV hard check: `EXL3_GEMV_MAX_M` widened to 16 | rc 0, 3 steps | RED — 5 / 6 cases, 42 / 43 assertions |

M6 is the one to read twice. Sixteen assertions move, which is what a guard that
deletes a whole arm looks like, and it is the mutation that proves the claim this
wave rests on: if `exl3_gemv.cu:53` were live rather than commented out, this
checkpoint would have NO GEMV arm at all on GB10, and the row would have been
right to skip W2c. It is not live, and these cases fail the moment anyone makes
it so.

**One mutation was VOIDED, and voiding it is the point.** M10's first attempt
named an anchor that had been reflowed by the formatter and no longer existed in
the file. The harness reported `VOID — the anchor is not in
src/vt/cpu/cpu_exl3_kernels.cpp`, restored nothing because it had changed
nothing, and REFUSED to run. Under the shape this row has met five times it
would instead have re-run the stale binary and printed SUCCESS, and M10 would
have gone into the table above as a green mutation proving nothing. The anchor
was corrected and M10 ran RED.

Restoration verified by sha256 after every mutation, all MATCH:
`exl3_policy.cpp` `db29377241bb`, `cpu_exl3_kernels.cpp` `b1c1d1c0e95a`,
`ops.cpp` `6a61e7812ef7`, `deepseek_v4.cpp` `839c83147118`. The tree was rebuilt
clean afterwards (`ninja -C build` rc 0, 603 steps over every target) and the
three suites re-run green from those binaries.


**Two host-side checks were made WITHOUT a compiler, and only two.**

The first is the GEMV's register-form window read, which is the likeliest silent
defect in that port: `dq8_regs_3bits` plus its per-lane constants
(`exl3_gemv_kernel.cuh:199-209`) resolve two words and a funnel shift per lane,
where `dq8<3, cb, 4>` — already checked against the 1-at-a-time `dq` in W2b —
resolves the same eight codewords through `ptr[i0 % 24]` and `ptr[i2 % 24]`. Both
forms were evaluated over a synthetic 24-word K=3 tile for all 32 lanes:
**0 mismatches out of 256 codewords**, and the three per-lane constants
(`src_a`, `src_b`, `s2`) agree lane for lane.

The second is the transcription itself, read against its anchors line by line.
Both gate FORMULAS and neither gates the FILE: they say upstream's two window
reads agree, not that `src/vt/cuda/cuda_exl3.cu` is what a compiler will accept.

**THE CUDA HALF OF THIS WAVE HAS BEEN COMPILED BY NOBODY.** `which nvcc` on the
implementer host returns nothing, no CUDA toolkit is installed, and `dgx.casa` is
flapping (boot, brief contact, drop). `src/vt/cuda/cuda_exl3.cu` grew the fused
MoE kernel, its two epilogues, the GEMV kernel and the GEMV try-launch, and every
one of them is a transcription that has passed NO compiler. That is the same
state W2a+W2b's device half is in and it is stated the same way. The first
compile verdict comes from `cuda-fat-build` or from the box, whichever answers
first. **Nothing here claims it builds, and no speed figure is quoted anywhere in
this wave.**

### W1c (2026-08-25, CPU-only build, `RelWithDebInfo`, base `a73b26968`)

Host `x86_64` Linux 6.8, no `nvcc`, no GPU. Build tree
`/home/mudler/.cache/sdd/mudler-vllm.cpp/dsv4-w1c/build`, configured
`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo`.

**RED FIRST, and it is the defect #1923 measured.** The rewritten forward suite
was built against the UNCHANGED production sources — the four files this wave
touches restored with `git checkout --`, `ninja` rc=0 — and run:

```
tests/vllm/models/test_deepseek_v4_exl3_forward.cpp:190: FATAL ERROR: REQUIRE( w.has_host_weights ) is NOT correct!
  values: REQUIRE( false )
[doctest] test cases:  4 |  2 passed | 2 failed | 0 skipped
[doctest] assertions: 17 | 15 passed | 2 failed |
```

With that `REQUIRE` and the four tower assertions below it temporarily relaxed
so the forward is actually entered, the refusal itself is verbatim:

```
test case THREW exception: vt: DeepseekV4 forward over an EXL3 load: the routed-expert
TRELLIS tower is loaded and reachable (MODEL-DSV4-EXL3 W2), but the NON-expert tower it
composes with is not materialized. On the real artifact those are the `carried-*` FP8
tensors and MODEL-DSV4-EXL3 W1c owns materializing them; see
.agents/specs/model-dsv4-exl3.md `## Owed`.
at src/vllm/model_executor/models/deepseek_v4.cpp:2869
```

That is the `Forward` path's refusal. The server probe in #1923 hit the
`ForwardDevice` one at `:2960`, because the runner's default `gather` routes
there; both were the same missing tower.

**GREEN.**

| suite | cases | assertions | verdict |
|---|---|---|---|
| `test_deepseek_v4_exl3_forward` (fused arm, default) | 4 | 37 | SUCCESS |
| `test_deepseek_v4_exl3_forward` (`VT_DSV4_EXL3_FUSED_MOE=0`) | 4 | 37 | SUCCESS |
| `test_deepseek_v4_exl3_loader` | 10 | 133 | SUCCESS |
| `test_nvfp4_dequant` (`DequantFp8BlockToF32` added) | 7 | 88 | SUCCESS |

Measured on the loaded synthetic checkpoint, both arms:
`exl3 vs dequantized-dense rel_rms = 1.679e-3` (fused) / `1.321e-3` (loop),
against the stated 2.0e-2 bound; `exl3 vs unrelated-dense rel_rms = 4.32`,
against the stated 1.0e-1 floor. `fused_calls = 2, per_expert_calls = 0` on the
default arm and `0 / 36` on the rollback, which is the 2 MoE layers and the
`3 tokens x 2 experts x 3 projections x 2 layers` the case derives.

**MUTATIONS.** Ninja's exit code AND its step count are recorded beside every
verdict, because a zero-step "rebuild" is a void verdict and this row has met
that trap before. Every mutation was restored byte-for-byte from a copy taken
before it was applied, and the restore was re-verified green.

| # | mutation | ninja | verdict |
|---|---|---|---|
| M1 | `w.host = DeepseekV4HostWeights{};` before the flag set — the materialization deleted, the flag kept | rc=0, 6 steps | **RED**: forward 2/4 cases failed; loader 3 failed (and aborted, so 2 cases never ran) |
| M2 | `w.has_host_weights = true` -> `= false` — the flag set deleted | rc=0, 4 steps | **RED**: forward 2/4 failed on `REQUIRE(w.has_host_weights)`; loader 2/10 failed |
| M3 | `projected = projected_trellis + host_tower` -> `= projected_trellis` — the carried tower dropped from the residency price | rc=0, 3 steps | **RED**: the one-byte-under-budget case stopped refusing |
| M4 | `E8M0ToF32(srow[k / block_k])` -> `srow[0]` — the block scale always read from block 0 | rc=0, 5 steps | **SPLIT, and the split is the finding**: `test_nvfp4_dequant` RED (2 cases, 9 assertions), `test_deepseek_v4_exl3_loader` **GREEN 10/10 131/131**. The fixture's scale generator had a THREE-value alphabet and the two 128-blocks of `layers.0.attn.wq_a` had collided. Widened to five, and the case now asserts that precondition rather than assuming it. |
| M4b | the same mutation, after the fixture repair | rc=0, 4 steps | **RED**: loader 1/10 failed, 133 assertions |
| M5 | `if (false && weights.has_exl3_weights)` in `DeepseekV4Model::Forward` — the reachability mutation | rc=0, 3 steps | **RED**: 2/4 cases threw `DeepseekV4 MoE: no routed-expert weights for this layer ... the EXL3 dispatch was not taken` at `deepseek_v4.cpp`, which is the guard this wave added for exactly that verdict — without it the mutation reads freed memory instead of failing |

M4 is recorded in full rather than tidied away: it is the
`.agents/verification.md` "instrument's own precondition" failure, it made a
real mutation read as a pass, and the fixture comment now carries the
measurement so a future narrowing of that alphabet fails as a broken instrument.

**Real-artifact measurements taken for this wave** (safetensors header plus one
16-byte range read of `carried-001.safetensors`, no download):
`layers.2.attn.wq_a.weight` is `F8_E4M3 [1024, 4096]` at 4,194,304 bytes (one
byte per element) beside `layers.2.attn.wq_a.scale` `F8_E8M0 [8, 32]` at 256
bytes, which is `[ceil(1024/128), ceil(4096/128)]`; the first scale byte is 115
(`2^-12`) and the first weight byte 106 (`+1.25 * 2^6`), so the decoded weight
is ~0.0195. `config.json` carries `quantization_config.base_quantization_config
= {activation_scheme: dynamic, fmt: e4m3, quant_method: fp8, scale_fmt: ue8m0,
weight_block_size: [128, 128]}`. The DSA widths that stop the real artifact:
`compressor.wgate.weight` BF16 `[1024, 4096]`, `compressor.ape` F32
`[4, 1024]`, `indexer.compressor.wkv.weight` BF16 `[256, 4096]`,
`indexer.wq_b.weight` `F8_E4M3 [8192, 1024]`, against `head_dim = 512` and
`index_head_dim = 128`.

**CLEAN BUILD, CHAINED.** `ninja -t clean` then a from-scratch build of the
three targets, chained by `&&` to `ctest` so a partial tree could not be tested:
`CLEAN BUILD rc=0 steps=505`, then `100% tests passed, 0 tests failed out of 4`
(`test_nvfp4_dequant`, `test_deepseek_v4_exl3_loader`,
`test_deepseek_v4_exl3_forward`, `test_deepseek_v4_exl3_forward_loop_arm`),
`CHAIN rc=0`.

**AFTER MERGING `origin/main`.** The branch was cut at `a73b26968` and main
moved to `fc2c5be23` under it, which is why the first staged preflight's trailer
gates SKIPPED. Merged, rebuilt and re-run on the merge result: ninja rc=0 over
507 steps, then `100% tests passed, 0 tests failed out of 4`, ctest rc=0. The
merge touches no file this row changed and `.agents/issue-index.md` union-merged
with both rows present and no duplicate. The two gates that had SKIPPED then
RUN and PASS over `$(git merge-base HEAD origin/main)..HEAD`:
`check-commit-trailers.py` `OK: commit trailer contract` and
`check-commit-style.py` `OK: commit writing style`, both rc=0. The first run of
those found the merge commit itself carrying no trailer block, which is why the
merge commit's message was written rather than left at git's default.

**`scripts/agent-preflight.sh --staged` rc=1 on the pre-merge tree, and neither
cause is this change.**
One gate FAILED and two SKIPPED:

- `test_cpu_x86_llamacpp_floor` failed with its own reason printed:
  `waiting for quiet: 15s busy=111% builders=0 load=39.73`, then
  `ours rep=1 DISCARDED`. That is the harness refusing to measure on a contended
  box, and the box carried four concurrent agent builds at load ~40 for this
  whole dispatch. It is the known load-sensitivity of that suite, not a
  regression: the same run is green in the non-staged preflight taken at the
  start of this dispatch (rc=0).
- `commit-trailers` and `commit-style` SKIPPED, because
  `origin/main fc2c5be23` is not an ancestor of HEAD — the branch was cut at
  `a73b26968` and main moved under it. A skipped gate reports nothing about the
  tree, so this is recorded as UNKNOWN rather than as a pass; the merge and the
  re-run are the operator's step before the push.

**NOT RUN HERE, and why.** A full local `ctest` is not physically possible in
this worktree: 621 registered tests, each statically linked against
`libvllm.a`, measured at 505,844,032 bytes for
`build/tests/test_deepseek_v4_exl3_loader` in this configuration — about 295 GB
against 11 GB free on `/` (`df -h /`: 447G total, 98% used, shared with other
sessions). The 62 registered tests whose names touch
`deepseek|exl3|nvfp4|mxfp4|fp8|dsv4` were built and run ONE AT A TIME with the
binary deleted after each, which is the widest local sweep the disk admits:
**62/62 rc=0**. Ten of them hit `ld: final link failed: No space left on
device` on the first pass and were re-run once the box freed space — an ENOSPC
build failure is not a test result and was not counted as one, which is the
`.agents/environment.md` trap about checkers failing toward a code verdict.
The whole suite is owed to CI on the pull request.

### W1c sanitizer repair (2026-08-25, `-DVLLM_CPP_SANITIZE='address,undefined'`)

The `sanitize-cpu (address,undefined)` lane on the pull request failed all three
of this wave's tests on undefined behaviour in the new materialization. See
`### W1c-7` for the defect and the remedy. Configured exactly as CI's job does
(`.github/workflows/ci.yml`, `sanitize-cpu`): `-DVLLM_CPP_BUILD_TESTS=ON
-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SANITIZE='address,undefined'`, run under
`UBSAN_OPTIONS=print_stacktrace=1
ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1 VT_POOL_BYPASS=1`. Ninja
rather than CI's Make, so a mutation's rebuild reports a step count.

**RED FIRST, on the unmodified branch head `c9771207e`.** ctest rc=8, 0/3
passed, three UBSan reports, verbatim:

```
src/vllm/model_executor/models/deepseek_v4_weights.cpp:431:86: runtime error:
  load of misaligned address 0x78a101098d15 for type 'const short unsigned int',
  which requires 2 byte alignment
src/vllm/model_executor/models/deepseek_v4_weights.cpp:458:93: runtime error:
  load of misaligned address 0x7dd795e63bde for type 'const long int',
  which requires 8 byte alignment
```

(`:458` reported twice, once from each forward arm.) The addresses end `d15`
and `bde` — odd, and `0xbde % 8 == 6` — so the payload really did land at an
offset no element type's alignment divides. The existing fixture reproduced it
without modification; the deliberate odd-base padding was added to stop that
from being luck.

**GREEN AFTER.** Same build, same three tests: ctest rc=0, `100% tests passed,
0 tests failed out of 3`, **zero** `runtime error` lines. Run directly for the
assertion counts, because a doctest binary that executes nothing also exits 0:
`test_deepseek_v4_exl3_loader` 11 cases / 148 assertions / `Status: SUCCESS!`
(10 cases / 136 assertions before this change — the new MISALIGNED case is +1
case and +12 assertions), `test_deepseek_v4_exl3_forward` 4 cases / 37
assertions / `Status: SUCCESS!`.

**MUTATIONS.** Each applied to the green tree, rebuilt (step count recorded, so
a build failure cannot be read as a pass), run, then restored and verified with
`sha256sum -c` against hashes taken before the first mutation — all three files
`OK`.

| # | Mutation | Rebuild | Result |
|---|---|---|---|
| M1 | BF16 arm back to `reinterpret_cast<const uint16_t*>(t.data)` + `p[i]` | ninja 5 steps, rc=0 | ctest **rc=8**, 3/3 FAILED, `deepseek_v4_weights.cpp:440:90: runtime error: load of misaligned address ... 'const short unsigned int'` |
| M2 | `tid2eid` arm back to `reinterpret_cast<const int64_t*>(t.data)` + `p[i]` | ninja 5 steps, rc=0 | ctest **rc=8**, 3/3 FAILED, `deepseek_v4_weights.cpp:473:97: runtime error: load of misaligned address ... 'const long int'` |
| M3 | `kMisalignedPayloadBase = 1` -> `0` (align the fixture payload) | ninja 7 steps, rc=0 | ctest **rc=8**, `test_deepseek_v4_exl3_loader` FAILED at `FATAL ERROR: REQUIRE( misaligned_for(embed, alignof(uint16_t)) ) is NOT correct!` |

M3 is the one that matters most, and it also measures the cost of getting the
fixture wrong: under it the two FORWARD tests PASS. An aligned fixture would
have made the whole hazard invisible to every lane including the sanitizer,
which is precisely how this landed green locally in the first place.

## Owed

- **`exllamav3` is not a REGISTERED secondary oracle.** AGENTS.md says a
  secondary oracle "is valid only when it appears in this table and has a
  recorded pin", and `exllamav3` appears in neither: it is absent from the
  secondary-oracle table and `.agents/oracles/` holds no `exllamav3.md`, while
  the same table gained `ltx-2` on main in this row's own merge window. No gate
  in W1 or W2 is invalidated by that, and the reason is worth stating rather than
  assuming: every gate this row has run so far is against the W1a CPU reference
  or against an INDEPENDENT `double` Sylvester-H128 reference built from the
  definition, never against an exllamav3 run. The registration falls due the
  moment W3 binds a gate to the oracle. W3a owns BOTH halves — the
  `.agents/oracles/exllamav3.md` file with its `gateable` verdict AND the
  AGENTS.md table row — and neither was edited here: this dispatch is W2a+W2b,
  and AGENTS.md is the binding policy file, not a helper's to widen.
- **The CUDA arm compiles nowhere yet.** `src/vt/cuda/cuda_exl3.cu` has never
  been through `nvcc`: the implementer host has no toolkit and `dgx.casa` is
  down. First verdict comes from `cuda-fat-build` or from
  `cmake -S . -B build-cuda -G Ninja -DVLLM_CPP_CUDA=ON
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a && cmake --build build-cuda --target vllm -j 4`.
- **Every W2 device measurement.** The byte gate for `had_r_128`, the tier-3
  bound for `exl3_gemm`, and the shape the device actually takes. All three are
  one command once the box returns:
  `rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemm -V`.
  No speed number was attempted and none is quoted.
- **The CUDA arm instantiates `bits == 3`, `codebook == 1` (mcg) ONLY.** Eight
  template instantiations rather than 64 in a TU the fat build compiles for ten
  architectures. Every other width refuses BY NAME from the launcher and names
  this row; the CPU arm stays generic over all eight widths, so the reference is
  not narrowed with the kernel. Widening the CUDA arm needs upstream's own
  per-K compilation-unit split (`comp_units/exl3_comp_unit_K_cbX.cu`).
- **`narrow_coresident` on GB10.** It is the ONLY input that decides whether the
  w1/w3 shape (k=4096, n=2048) is GEMV-eligible at all (`## W2cd design` W2c-2),
  it is a device occupancy query, and no CPU run can stand in for it. w2
  (k=2048, n=4096) is eligible without it.
- **WHICH arm wins on GB10, for either W2c or W2d.** The GEMV arm's eligibility
  is upstream's heuristic and its default is upstream's default; that it is
  FASTER here has not been measured and is not claimed. The fused MoE arm's
  launch-count reduction is arithmetic (`3 * topk * T` -> 1 per layer); the
  resulting throughput is not. Commands in `## W2cd design` W2cd-9.
- **The tier-3c and tier-4 bounds have been measured on the CPU arms only.** The
  device halves of both are `PENDING` on the same box.
- **The GEMV kernel instantiates `bits == 3`, `codebook == 1` (mcg) ONLY**, the
  same narrowing `exl3_gemm` carries. Upstream covers bits 2/3/4 over three
  codebooks. Every other width DECLINES the arm and falls through to the regular
  kernel, which is upstream's own failure mode rather than a refusal.
- **`Exl3MoeMlp` skips an expert with more than `max_tokens_per_expert` tokens**,
  exactly as upstream does, and the caller's per-expert loop covers it. That tail
  arm has been gated on a fixture; it has never run at a batch large enough to
  reach it on the real checkpoint, because the real checkpoint does not yet run.
- **The EXL3 arm runs on a CPU queue, or on a device that can dereference host
  memory.** `Exl3Linear` (`deepseek_v4.cpp`) refuses any other device BY NAME,
  because W1b's coalesced tower is host-resident and handing a host pointer to a
  CUDA kernel is the #844 / #1435 crash. That refusal is the visible form of the
  "Real-checkpoint residency for the coalesced tower" item already below.
- The SparkInfer denominator run — `PENDING` on the developer's host-docker
  authorization (asked 2026-08-24).
- `.agents/oracles/exllamav3.md` with a measured gateability verdict (W3).
- **A shared dense-MLA policy the EXL3 arm can take, so the REAL artifact's DSA
  tensors load.** W1c materializes every carried tensor the host forward
  consumes and REFUSES BY NAME on three the real artifact stores at twice the
  collapsed width (`compressor.wgate` `[2*hd, H]`, `indexer.compressor.wkv`
  `[2*ihd, H]`, `indexer.wq_b` `[inh*ihd, q_lora_rank]` — `## W1c design`
  W1c-4). 41 of 43 real layers carry a compressor, so the real artifact stops
  there. The GGUF arm already dodges the identical geometry with
  `dsa_dense = (be.gguf != nullptr)`; widening that predicate to the EXL3 source
  would break this row's own equivalence gate, because the gate's dense
  reference has `be.exl3 == nullptr` and the two arms would stop taking the same
  attention path. The fix is a dense-MLA selector BOTH arms read — a policy
  decision about the DSA residual, not about this format — and no row owns it.
  `MODEL-DSV4-EXL3` carries it here until one does, which is what this `## Owed`
  entry is for; it is deliberately NOT attached to
  [#1923](https://github.com/mudler/vllm.cpp/issues/1923), because that issue is
  the loader defect and W1c closes it.
- **The real artifact's DSA geometry now LOADS, and the forward REFUSES on it —
  [#1970](https://github.com/mudler/vllm.cpp/issues/1970), option C of
  [#1961](https://github.com/mudler/vllm.cpp/issues/1961).** This SUPERSEDES the
  dense-MLA-policy entry above: the answer is not a shared dense-MLA selector,
  because dense MLA is not upstream's attention on a `cr > 0` layer at any
  sequence length ([#1964](https://github.com/mudler/vllm.cpp/issues/1964)), so
  routing the EXL3 arm there would have been a wrong-but-plausible path rather
  than a policy. The loader now derives every DSA width as upstream does
  (`coff = 1 + (compress_ratio == 4)`, `vllm/models/deepseek_v4/compressor.py:247-248`)
  and `AttentionBlock` refuses BY NAME when a materialized width is not the one
  its arithmetic indexes. What stays OWED is the DSA composition itself — the
  `coff`-overlapped window with `head_offset` role selection, boundary-only
  emission, a compressed KV cache beside a SWA(128) raw cache, the indexer on
  `qr` over compressed rows, one joint softmax over the union. **No row owns that
  port**; `MODEL-DSV4-EXL3` carries it here until one does, and it needs the
  cache topology [#1960](https://github.com/mudler/vllm.cpp/issues/1960) and
  [#1925](https://github.com/mudler/vllm.cpp/issues/1925) are scoping. Also owed
  and NOT closed by #1970: the GGUF arm's `dsa_dense` still runs the same wrong
  attention on 41 of 43 real layers (#1964, excluded from #1970's scope), the
  `cr == 128` EXL3 layers pass the width check while their `win = 2` pooling is
  still not upstream's 128-wide boundary-emitted compressor, and the
  `indexer.wq_b` input-space defect (`x` where upstream uses `qr`) is real at any
  geometry. Design, anchors and mutations in
  [`specs/dsv4-dsa-loader-accept-forward-refuse.md`](dsv4-dsa-loader-accept-forward-refuse.md).
- **Real-checkpoint residency for the coalesced tower — W2.** W1b copies each
  TP1-coalesced linear into host owner buffers. That is right for the fixture
  and for W2's byte-parity gate, and it is ~100 GB on the real 216-expert
  artifact, so the real load needs borrow / device-resident / per-layer
  streaming. Coalescing cannot borrow the mmap directly: an OUT split
  interleaves ranks along trellis dim 1, so some copy is inherent and the fix is
  to make the destination device-resident rather than host-resident.
- **The MTP NVFP4 draft experts.** Skipped-and-counted today (see `## Evidence`);
  no row owns reaching them yet.
- Upstream's own `ext.reconstruct` run against the W1a anchors, so the
  real-tensor spot values become an ORACLE result rather than a second
  transcription (W3a, needs a GPU).
- **Rank ORDER is unverifiable from inside this tree.** `require_invariant`
  catches a slice taken on the wrong AXIS — that mutation is recorded RED above —
  and the coalescing asserts every reassembled shape. Nothing catches a
  TRANSPOSED `.rank{r}` labelling: four ranks concatenated in the wrong order
  reassemble to the right shape, pass every invariant, and produce a silently
  wrong weight. No fixture can close it, because the fixture writes the labels it
  then reads back. Only W3's oracle run over the real checkpoint decides it.
- **[#1883](https://github.com/mudler/vllm.cpp/issues/1883)**:
  `tests/CMakeLists.txt` registers `test_minimax_music3_e2e_real` unconditionally
  while the `ApiServer` translation unit it links is compiled only under
  `if(VLLM_CPP_SERVER)`, so no `-DVLLM_CPP_SERVER=OFF` build — the configuration
  this row's CPU-tier dispatches use — can build it, and `ctest` reports
  `Not Run`. Found while gating W1; NOT this row's authority to fix, and no row
  owns the test-registration surface, so this row carries it until one does. It
  does not affect any gate recorded here: every suite above is a different
  target.

## Stop conditions

- W1 fixture parity fails against independently transcribed expectations →
  stop and re-derive from `codebook.cuh`/`exl3_dq.cuh`; never tune constants
  to green.
- The real checkpoint's carried tensors need an FP8 layout our arms lack →
  file, name the refusal, keep the synthetic-fixture gates green.
