# BACKEND-TENSTORRENT-KEEPQUANT — keep-quant dense dot on TT-Metal

Row: `BACKEND-TENSTORRENT-KEEPQUANT` · issue
[#2959](https://github.com/mudler/vllm.cpp/issues/2959) · child of
`BACKEND-TENSTORRENT` · branch `row/BACKEND-TENSTORRENT-KEEPQUANT`, base
`origin/main` `bcbfee7bf`.

## Scope

This commit is **W0 staging**: this spec, the backend-matrix row, and the
substrate survey below. No file under `src/`, `include/` or `tests/` changes.
The implementation waves are W1+ under `## Work breakdown`.

In scope for the row overall: the **dense** keep-quant dot on TT-Metal
(Q4_K first, then Q5_K/Q6_K/Q8_0), the `kTENSTORRENT` arm of the admission
predicate, provider registration, capture-safe residency, and the e2e vehicle
gate. Out of scope: MoE grouped keep-quant GEMM (follow-up row),
the block-decoding n-gram gather ([#2394](https://github.com/mudler/vllm.cpp/issues/2394)),
ROCm keep-quant gaps ([#2109](https://github.com/mudler/vllm.cpp/issues/2109),
[#1876](https://github.com/mudler/vllm.cpp/issues/1876)), the IQ-family and
sub-IQ1_S encodings.

## Why this needs a spec before code

The row's central decision — decode-then-bf16-tiles versus a true integer
block dot — depends on what the substrate offers, and the survey below says it
offers nothing. Writing the kernel first would have picked the shape by
accident; the predicate it must feed is exactly-as-wide-as-the-kernel by
policy (the GLM-5.3 W10 lesson), so the kernel's encoding set has to be
settled before `gguf_keep_quant.cpp` changes at all.

## The substrate fact (surveyed 2026-09-05, this worktree)

**tt-metal has no k-quant / packed-weight block-dot primitive.**

- `ttnn`'s quantization op is per-tensor activation dquant —
  `quantize` / `requantize` / `dequantize`
  (`ttnn/cpp/ttnn/operations/eltwise/quantization/quantization.hpp:15-35` at
  the runtime the backend pins, `TT_METAL_HOME=/home/lu_zero/Sources/tt/tt-metal`).
- No int4/packed-weight matmul exists anywhere under `ttnn/cpp/ttnn/` or
  `tt_metal/`; the matmul `gather` hits are MCAST dataflow variants
  (`reader_bmm_tile_layout_in0_ring_all_gather.cpp` and siblings), unrelated to
  quantization.

Consequence: the kernel is ours to write, composed the way `kGdnDecode` was
composed from ttnn matmul+eltwise, or as a lower-level tt-metal device kernel.
There is no vendor quant kernel to call, and `llama.cpp`'s `mul_mat_q` is a
CPU-side floor, not a TT denominator.

## Design

**W1 shape: blocks stay resident; decode runs on-core; the dot reuses the
existing bf16 tile-matmul machinery.**

- Weights keep their file encoding in device DRAM (the 17.1 GB Q4_K_M artifact
  stays 17.1 GB — the residency is the point; a dequantizing expansion that
  doubles the bytes is the defect this row prevents).
- The provider (op `kMatmulBT` with a block-quantized weight, or a sibling op
  the op-provider table already models — decided by the seam the CUDA arm
  uses) streams packed blocks; the compute path decodes each block to bf16
  tiles through an f32 intermediate and feeds the existing `kMatmulBT` tile
  path. Precedent: the GDN decode kernel's ttnn composition.
- **Numerics bar, stated precisely:** block *decode* is bit-exact against the
  CPU decoder (`vt::cpu::BlockToFloat` — the reader is already pinned
  bit-exact vs llama.cpp `b10451` per block). The *dot* is device bf16, NOT
  the CPU int8 `vec_dot`, so its gate is the measured device-bf16 band at the
  residual-golden boundary (`BACKEND-TENSTORRENT-RESIDUAL-GOLDEN` convention:
  device bf16 diverges from host f32 by bf16 rounding, and that is recorded,
  not waved). e2e: the existing 16-prompt battery vs the bf16 arm's committed
  pair — STRICT or inside the ≤500-mnat near-tie band.
- **Predicate:** add the `kTENSTORRENT` arm to `DeviceKeepQuantSupported`
  (`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:136-148`),
  admitting exactly the encodings whose kernels are registered — never wider
  (a widened predicate throws at first forward on a discrete card; the CPU
  fallback the `default:` arm assumes does not exist here).
- **Capture:** the decode path must be device-resident under trace capture
  from day one — no mid-capture host download, CHECK-before-download ordering,
  the #2812/#2907 discipline. A keep-quant weight that forces a host round
  trip inside the captured region is a refused arm, same as any op miss.
- **Named next lever, never a ceiling:** the true integer dot (row-Q8_K
  activation + int8 block dot, the llama.cpp `mul_mat_q` shape) as the perf
  follow-up once the decode shape is correct. The first wave does not claim
  llama.cpp-comparable throughput; it claims a resident, correct arm.

## Vehicles and weights

- **Vehicle:** `Qwen3.5-0.8B` GGUF **Q4_K_M**. The family already adjudicates
  16/16 STRICT on the P150 in bf16 (ambient + captured), so the only delta
  between arms is the keep-quant path. **The checkpoint pin (repo, revision,
  sha256) is OWED at W1 capture time** — a repo id alone is not a pin; record
  it in `docs/USAGE.md` in the same change that first runs the arm.
- **Target artifact:** `unsloth/Qwen3.8-27B-GGUF` Q4_K_M —
  `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes, sha256 `7e78da5d…`
  @ `fe1e2a23d973adb629709749dc4f6756df66ef10`; UD variant 16,464,440,224
  bytes, sha256 `322e194f…` @ `4ca720788d1e01f1bff70c033e0d0028fd02e502`
  (pinned in docs/USAGE.md:914-915). bf16 27B (53.8 GB) cannot fit the P150;
  these can.

## Upstream chain

- ggml k-quant block formats: geometry read from our reader, which is already
  pinned bit-exact against the llama.cpp `b10451` decoders on real file bytes
  (#2240/#2260 lineage). The oracle for decode is our own CPU arm.
- No vLLM mirror exists for a Tenstorrent backend — deviation by design
  (secondary substrate), the standing `BACKEND-TENSTORRENT` convention.

## Our baseline

On `origin/main` `bcbfee7bf`: the TT backend runs the Qwen3.5-0.8B bf16 arm
16/16 STRICT (ambient + captured) and Mistral-7B; `DeviceKeepQuantSupported`
has ROCm and CUDA/CPU-fallback arms and no `kTENSTORRENT` arm, so a GGUF
k-quant checkpoint on TT keeps its blocks per the default arm and the first
device dot refuses — the discrete card has no CPU fallback to hide behind.
No TT quant kernel exists anywhere under `src/vt/tenstorrent/` (surveyed
2026-09-05, `grep -i quant` over the backend: zero hits). The CPU keep-quant
arm is ISA-tiered and bit-exact per block vs llama.cpp `b10451`; the bf16
device-vs-host numerics boundary is characterized by
`BACKEND-TENSTORRENT-RESIDUAL-GOLDEN` (bf16 rounding signature, 0.0459 abs at
the rows>=32 boundary).

## Port map

| Upstream | Local | Note |
|---|---|---|
| llama.cpp `b10451` k-quant block decoders (`ggml-cpu/ggml-cpu.c` `dequantize_row_q4_K` and siblings) | `src/vt/cpu/` `vt::cpu::BlockToFloat` (landed, bit-exact per block) | The decode authority this row's device decode must match bit-for-bit; geometry already read from the pinned llama.cpp oracle by the reader (#2240/#2260 lineage). |
| — (no upstream analogue; tt-metal has no packed-weight matmul — surveyed 2026-09-05) | OWED W1: the Q4_K on-core decode, feeding the existing `kMatmulBT` bf16 tile path | Resident blocks in DRAM; decode through an f32 intermediate rounded to bf16 exactly as the CPU decoder's output rounds; composition precedent is `kGdnDecode`'s ttnn matmul+eltwise shape. |
| — (local; the predicate is the GLM-5.3 W10 lesson made structural) | OWED W2: the `kTENSTORRENT` arm of `DeviceKeepQuantSupported` (`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:136-148`) | Admits exactly the encodings whose device kernels are registered; never wider, because the `default:` arm's CPU fallback does not exist on a discrete P150. |
| — (local; the #2812/#2907 capture discipline applied to a new weight path) | OWED W3: capture-safe residency of the decode path | No mid-capture host download; CHECK-before-download ordering; capture dump ×2 byte-identity as the gate. |

## The vehicle falsified the W3/W4 split (found 2026-09-06)

The Q4_K_M artifact is mixed-quant, not uniform Q4_K. The tensor histogram
of the fetched vehicle (unsloth/Qwen3.5-0.8B-GGUF at `6ab46149`, sha256
`bd258782...`): `token_embd.weight` **Q6_K** (the tied LM head reads the
same tensor), `attn_qkv` and `ssm_out` **Q5_K** (18 each), `ssm_alpha` and
`ssm_beta` **Q8_0** (18 each), `ffn_down` Q6_K x12 / Q4_K x12, `attn_v`
Q6_K x4 / Q4_K x2, everything else Q4_K or F32. A first forward on TT under
the W2 predicate refuses by name — correctly. Consequence: the e2e vehicle
battery needs the Q5_K/Q6_K/Q8_0 decodes first. The decode set moves from
W4 into W3; W4 keeps the int8-dot lever and the 27B arm. The predicate
widens `{Q4_K}` to exactly the registered four in the same change as the
kernels — never before.

The Q6_K token embedding is a design point, not a footnote: kEmbedding on
TT must reach a Q6_K weight either by a device gather that decodes rows
on-core, or by staging the decoded table (248320 x 1024 bf16, ~0.5 GiB —
bounded here, unlike the 27B expansion the design thesis forbids). Decide
at implementation against the CPU arm's behavior.

## W3 capture-safe staging (design, from the W2 survey)

`DecodeQ4KBlocksF32` today EnsureHosts the packed tensor, repacks to i32
words on host, and `from_vector`-uploads per call — a per-call host round
trip that inside a capture is the #2812 class. The fix keeps the residency
thesis (packed stays the resident master; a decoded-bf16 cache would triple
the 0.8B and blow the 27B budget): stage the **i32 word form once per
weight** through the existing persistent-shadow machinery
(`EnsureMatmulWeightDevice`'s keyed-slot pattern, built for the view-staging
fatality — 36 i32 words are exactly 144 packed bytes, zero expansion), and
run the on-core decode chain from the resident words every call. Captured
replay recomputes deterministically from fixed device bytes; a cache miss
during capture CHECKs ("warm the keep-quant arm eagerly first"), the
zero-cache precedent. Red-first leg: with W2 code the captured vehicle run
traces staging writes during capture and must fail the dump x2 byte-identity
leg; after, zero writes during capture and dumps byte-identical across two
reset runs.

The gap golden for the q4km arm teacher-forces `transformers` on the
DEQUANTIZED artifact — `from_pretrained(gguf_file=...)` if the pinned
transformers 5.14.1 parses qwen35 GGUF, otherwise our own bit-exact decoder
(W1-proven) writing a safetensors dir first. Never teacher-force against
the bf16 safetensors checkpoint: those logits are a different model's.

## Tests to port

vLLM has no TT backend and no keep-quant test to port; the oracle chain is the
`BACKEND-TENSTORRENT-GDN` precedent's. The gate family is local: op-level
block-decode bit-exact vs `vt::cpu::BlockToFloat` across a shape sweep (the
red-first core — a decode that is not bit-exact fails here first), the dot
vs the CPU keep-quant arm within the measured device-bf16 band, the predicate
mutation (admit one encoding too many → reproduce the first-forward refusal in
a scratch copy), capture dump ×2 byte-identity with reset between, and the
16-prompt e2e battery vs the committed bf16 pair (STRICT or inside the
≤500-mnat near-tie band — the q35 goldens' existing conventions).

## Dependencies

- GGUF reader + CPU keep-quant decoders (landed, ISA-tiered i8mm on this
  host).
- `kMatmulBT` TT provider (landed, bf16 tiles).
- Trace-capture machinery (#2907 lineage) for the capture leg.

## Work breakdown

- **W0** (this commit): spec, matrix row, substrate survey.
- **W1**: Q4_K block-decode device path, bit-exact vs `BlockToFloat` across a
  shape sweep; red-first tests in `test_tenstorrent_backend.cpp`'s op-level
  pattern.
- **W2**: the dot provider + `kTENSTORRENT` predicate arm + registration;
  predicate mutation red (admit one encoding too many → first-forward throw
  reproduced in a scratch copy).
- **W3**: Q5_K / Q6_K / Q8_0 decodes (bit-exact vs `BlockToFloat` sweeps,
  the W1 template generalizes); the predicate widening to exactly the
  registered four in the same change; capture-safe resident-word staging;
  the capture leg (dump ×2 byte-identity with reset between); the e2e
  vehicle battery on the P150 vs its own captured pair under the committed
  near-tie conventions.
- **W4** (owed): the int8-dot perf lever; the 27B Q4_K_M arm as the first
  qwen3.8 artifact on TT.

## Risks

- **Bandwidth, not just memory:** decoding per dot makes each GEMM read packed
  blocks and write bf16 tiles on-core; if L1 staging forces per-tile DRAM
  round trips the arm may land slower than the host-offload baseline. Measure
  the decode cost standalone (W1) before the e2e claim; a number below the
  CPU-offload floor keeps the row open with the int8 dot as the named lever.
- Tile/rounding conventions: Blackhole bf16 conversion is RNE; the CPU
  decoder's f32 intermediate must round-trip identically — the bit-exact
  decode test is the guard.
- Capture interplay: decoded tiles must not be cached across replay steps in
  a way eager and replay disagree on (the #2812 class).

## Gates

Per wave: `scripts/agent-preflight.sh` clean; op-level suite green; capture
dump ×2 byte-identity; e2e 16-prompt battery vs the bf16 pair (STRICT or
inside the committed near-tie band, `RESULT PASS` form); GPU legs under the
`flock` mutex on the local P150, `tt-smi -r 0` before batches. Token-exactness
at the e2e is adjudicated against the committed pair only — never re-captured
to make a failure pass.

## Stop conditions

- No red-first evidence for a landed kernel.
- Decode not bit-exact vs the CPU decoder on any swept shape.
- The e2e arm refusing by name anywhere on the production path.
- The predicate landing wider than the registered kernel set.

## Owed

- MoE grouped keep-quant GEMM (follow-up row; vehicle pickable after W2).
- Block-decoding n-gram gather ([#2394](https://github.com/mudler/vllm.cpp/issues/2394)).
- IQ-family / sub-IQ1_S encodings (unsloth fork formats).
- The int8-dot perf lever; llama.cpp-comparable throughput numbers.
- `docs/USAGE.md` vehicle pin when the arm first runs end to end (the W3
  capture leg hashes the local bytes); 27B arm entry at W4.

## Now

`ACTIVE`, 2026-09-06. W1 complete (#2989, open). W2 complete on the row
branch: the dot (`MatmulBTQuantKernel` — `DecodeQ4KBlocksF32` factored out of
the W1 kernel, one bf16 RNE, the `kMatmulBT` tile matmul) is reached through
`vt::MatmulBT`'s block-weight dispatch and sits inside the analytic
operand-rounding envelope (bound ratios 0.28-0.53, M=1 GEMV included); the
`kTENSTORRENT` predicate arm admits exactly `{Q4_K}` and the routing test
reds any widening past the registered set. Both red-first: the registration
REQUIRE and the six wrongly-admitted encodings reded before the
implementation. Next: W3, the capture leg (dump ×2 byte-identity, capture-
safe staging for the per-call decode upload) + the e2e vehicle battery on
the P150 (vehicle fetched and hashed). AMENDED 2026-09-06: the vehicle is
mixed-quant, so W3 now carries the Q5_K/Q6_K/Q8_0 decodes and the predicate
widening before the capture leg and the e2e battery (see the falsification
section). W4 owed: the int8 lever, the 27B arm.
