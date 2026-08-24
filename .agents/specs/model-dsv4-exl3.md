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

`READY` on this spec's commit; W1 dispatched in the same flow. The spike that
grounds every claim below ran 2026-08-24 and is recorded on
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
verification, effort estimate). W1 evidence lands here with its commit.

## Owed

- The SparkInfer denominator run — `PENDING` on the developer's host-docker
  authorization (asked 2026-08-24).
- `.agents/oracles/exllamav3.md` with a measured gateability verdict (W3).
- The checkpoint's NAS `SHA256SUMS` manifest once the 107 GB download lands.

## Stop conditions

- W1 fixture parity fails against independently transcribed expectations →
  stop and re-derive from `codebook.cuh`/`exl3_dq.cuh`; never tune constants
  to green.
- The real checkpoint's carried tensors need an FP8 layout our arms lack →
  file, name the refusal, keep the synthetic-fixture gates green.
