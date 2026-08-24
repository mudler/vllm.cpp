# MODEL-DSV4-EXL3 — load the SparkInfer EXL3 3.0bpw REAP-K216 DeepSeek-V4-Flash, and match or beat its speed on one GB10

Row: `MODEL-DSV4-EXL3`
Issues: [#1875](https://github.com/mudler/vllm.cpp/issues/1875) (primary)
Base SHA: `3359f4159`
Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements no
EXL3** (`layers/quantization/` has no exl3/exllamav3/trellis at the pin), so
per AGENTS.md the work is gated by a secondary oracle this spec proposes:
`exllamav3` (`turboderp-org/exllamav3` @ `2398c05635fbbad01a0a51dce63c85c6c8a8450e`,
MIT), whose HEAD already carries DeepSeek-V4 support
(`exllamav3_ext/{dsv4_compress.cu,dsa_topk.cu,hc_mix.cu}`, `tests/test_dsv4_*.py`).
Gateability is unmeasured; W3 measures it and writes
`.agents/oracles/exllamav3.md` per the oracle rules before any e2e token gate
binds. The checkpoint was quantized at exllamav3 rev `787d1582` from model rev
`9e165c30`.
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
[#1875](https://github.com/mudler/vllm.cpp/issues/1875).

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
(`src/vllm/model_executor/models/deepseek_v4_weights.cpp:137`, loop `:269`) —
no expert-count code change.

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

- The 107 GB checkpoint staged on the NAS with a `SHA256SUMS` manifest
  (in progress) — W1b's real-checkpoint probe and everything in W3.
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
| `test_deepseek_v4_exl3_loader` (W1b) | 4 / 4 | 44 / 44 |

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

## Owed

- The SparkInfer denominator run — `PENDING` on the developer's host-docker
  authorization (asked 2026-08-24).
- `.agents/oracles/exllamav3.md` with a measured gateability verdict (W3).
- The checkpoint's NAS `SHA256SUMS` manifest (the 100 GB / 190-file download
  completed 2026-08-24; the manifest itself is still owed).
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

## Stop conditions

- W1 fixture parity fails against independently transcribed expectations →
  stop and re-derive from `codebook.cuh`/`exl3_dq.cuh`; never tune constants
  to green.
- The real checkpoint's carried tensors need an FP8 layout our arms lack →
  file, name the refusal, keep the synthetic-fixture gates green.
