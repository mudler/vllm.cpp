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

`ACTIVE`. **W1a, W1b and W2a+W2b have landed.** W1 gave the CPU reference dequant
(`vt::Exl3*`, `src/vt/cpu/cpu_exl3_dequant.cpp`) and the rank-sliced loader arm.
W2 gives the two DEVICE ops (`vt::Exl3HadR128`, `vt::Exl3Gemm`), the portable CPU
arm of both, the CUDA port of both, the shape-selection policy as pure host code,
and the wiring that makes the loaded tower REACHABLE: `DeepseekV4Model::Forward`
now routes an EXL3 load through `MoeBlock`'s trellis arm, one `Exl3Gemm` per
active routed expert per projection.

**The CUDA arm is UNCOMPILED and UNMEASURED, and nothing in this spec pretends
otherwise.** The implementer host has no `nvcc` and `dgx.casa` hung 2026-08-25
03:24Z with the GB10 unified-memory OOM-reboot signature, so every device number
is `PENDING` with its command recorded in `## W2 design` §5 and in `## Owed`.
The CPU tier is fully gated and is what makes the capability reachable today.
Next: the device compile + parity run the moment the box returns, then W2c
(the m<=8 GEMV, with the measurement §3 names), W2d (the fused MoE mgemm), W1c
(the `carried-*` tower so a real checkpoint runs end to end) and W3.

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
- **W2c, the m<=8 GEMV** (`exl3_gemv.cu`), with the measurement `## W2 design`
  §3 names: the "Blackwell keeps the regular kernel" line is COMMENTED OUT at
  `exl3_gemv.cu:53` and the live envelope admits w2's shape.
- **W2d, the fused MoE mgemm** (`exl3_moe.cu` + `comp_units/exl3_moe_inst_k3_cb1.cu`).
  This wave loops the dense GEMM per (active expert, projection), which is one
  launch each where the fused kernel pays one per layer.
- **The EXL3 arm runs on a CPU queue, or on a device that can dereference host
  memory.** `Exl3Linear` (`deepseek_v4.cpp`) refuses any other device BY NAME,
  because W1b's coalesced tower is host-resident and handing a host pointer to a
  CUDA kernel is the #844 / #1435 crash. That refusal is the visible form of the
  "Real-checkpoint residency for the coalesced tower" item already below.
- The SparkInfer denominator run — `PENDING` on the developer's host-docker
  authorization (asked 2026-08-24).
- `.agents/oracles/exllamav3.md` with a measured gateability verdict (W3).
- **Execution of the EXL3 tower — W1c, then W2.** W1b loads and coalesces it;
  NOTHING consumes it. A forward over an EXL3 load refuses through the existing
  `has_host_weights` guard, whose message does not name this row: the guard
  lives in `deepseek_v4.cpp`, outside the W1a/W1b dispatch's authority. W1c owns
  both the dequant-to-bf16 fallback arm and making that refusal name
  `MODEL-DSV4-EXL3`.
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
