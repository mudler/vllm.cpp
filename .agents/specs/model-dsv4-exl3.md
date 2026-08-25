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

`ACTIVE`. **W1a and W1b have landed** (this commit): the CPU reference dequant
(`vt::Exl3*`, `src/vt/cpu/cpu_exl3_dequant.cpp`) and the rank-sliced loader arm
(`LoadDeepseekV4ForCausalLMWeights` -> `LoadDeepseekV4Exl3`). Nothing EXECUTES
the tower yet: a forward over an EXL3 load still refuses through the existing
`has_host_weights` guard. Next: **W1c** (dequant-to-bf16 fallback so the model
runs), then W2 (GPU kernels) and W3 (oracle gateability + the speed table).
The spike that grounds the format description ran 2026-08-24 and is recorded on
[#1875](https://github.com/mudler/vllm.cpp/issues/1875). W1's fresh review
returned `PASS` with six MINOR findings and two NITs; the repair for all eight
landed with this commit and is recorded under `## Evidence`.

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
| W2: CPU-vs-CUDA byte parity per kernel; e2e greedy token gate vs the W3 oracle | implementer/operator |
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

## Owed

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
