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

## Evidence (W3, 2026-09-06, P150 under the local flock mutex)

- Op-level suites (E1): `test_tenstorrent_backend` keep-quant decode equality,
  registration, staging, capture-safety cases 37/37; `test_gguf_keep_quant`
  predicate cases 7/7; layout/reference cases 260/260; parser cases
  10333/10333. Focused keep-quant battery (E2): 57/57 cases, 6685 assertions;
  the twin-build repair suite re-run after the fix: 6/6 cases, 703 assertions
  (filter `*MatmulBTQuant*,*kKeepQuantDecode*,*stages zero words*,*kMoeSiluMul*`).
- Capture determinism (E3): two bootstrap runs with `tt-smi -r 0` between;
  `our_ids_tenstorrent_capture.i32` byte-identical (md5
  `eb8fb9894c3e4e05504f769b4119f443`, 16x16 i32), 105/105 assertions each,
  keep-quant capture-staging counter 0 across both captured e2e runs.
- Teacher-forced goldens (E4): `transformers` (venv: python 3.12, torch
  2.7.1+cpu, transformers 5.8.1) on the dequantized artifact `/tmp/q4km-dequant`,
  bf16, 16 tokens, prompts from `p{i}_prompt.i32`. Generic anchor pair
  (`our_ids.npy`/`neartie_gap_mnats.npy`): 0 token-divergent, max gap
  0.125 nats. TT capture pair (`our_ids_tenstorrent_capture.npy`/
  `neartie_gap_mnats_tenstorrent_capture.npy`): 51/256 near-tie divergences,
  max gap 0.1875 nats — all inside the 500 mnats band (kNearTieMnats,
  test_qwen35_paged_engine.cpp:83).
- READY adjudication: 147/147 assertions; 16/16 prompts PASS — 11/16 STRICT
  token-exact vs oracle per-prompt greedy, 5/16 near-tie-band only, max gap
  0.188 nats, 0 forward-divergent; backend proof 16/16 ops selections>0 with
  0 declines (kPagedAttention 1536, kGdnDecode 4320, kCausalConv1dUpdate 4320
  among them), staging counter 0. The ladder's `kGdnOps` proof list is
  arm-aware now: the keep-quant vehicle dispatches `kMatmulBTQuant` +
  `kMoeSiluMul`, never `kMatmulBT` + `kSiluAndMul` (bf16 arm unchanged).
- Recipe: `VLLM_CPP_QWEN35_Q4KM_GGUF=<local Q4_K_M>` (+ `VT_DUMP_IDS=1` for
  bootstrap only), `TT_METAL_RUNTIME_ROOT=/home/lu_zero/Sources/tt/tt-metal`,
  build dir first in `LD_LIBRARY_PATH`, `tt-smi -r 0` before each batch,
  `timeout -k 10 3300 ./build/tests/test_qwen35_paged_engine -tc="*GGUF Q4_K_M*"`.
- Vehicle: unsloth/Qwen3.5-0.8B-GGUF @ 6ab46149, `Qwen3.5-0.8B-Q4_K_M.gguf`,
  sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`,
  532517120 bytes (local copy hashed).

## Stop conditions

- No red-first evidence for a landed kernel.
- Decode not bit-exact vs the CPU decoder on any swept shape.
- The e2e arm refusing by name anywhere on the production path.
- The predicate landing wider than the registered kernel set.

## Owed

- MoE grouped keep-quant GEMM — MOVED INTO W4 SCOPE (#3030): the seam
  exists (`ops.cpp:220`, `weight[E*N,K]`) with a production ROCm kernel
  (`MatmulBTQuantGroupedKernelRocm`, native Q8_0/Q4_K/Q6_K) fed by the
  stacked keep-quant tower (`qwen3_5_gguf_weights.cpp:1287`); the TT
  backend registers only `kMoeSiluMul`. "Vehicle pickable after W2" is
  literally true.
- Block-decoding n-gram gather ([#2394](https://github.com/mudler/vllm.cpp/issues/2394)).
- IQ-family / sub-IQ1_S encodings (unsloth fork formats).
- The int8-dot perf lever (#3031); llama.cpp-comparable throughput numbers.
- `docs/USAGE.md` vehicle pin when the arm first runs end to end (the W3
  capture leg hashes the local bytes); the 27B arm entry LANDED with
  wave-3b-2, provenance caveat included — the gate-completion half of that
  pin stays owed to [#3042](https://github.com/mudler/vllm.cpp/issues/3042).
- The 27B e2e gate, left UNREACHED by wave-3b-2:
  [#3042](https://github.com/mudler/vllm.cpp/issues/3042) owns it. The
  committed goldens (`tests/parity/goldens/qwen38_gguf_q4km_27b/`) are
  lane-ready, and the TT-side run against them closes the gate. Evidence and
  the next lever: `## W4`, wave-3b-2.
- Residency reconciliation (RESOLVED BY ARITHMETIC, 2026-09-06, #3030):
  the twin residency is a 0.8B-only shape. Measured on the pinned 27B
  artifact: non-expert keep-quant twins need 18.47 GiB, the expert tower
  32.37 GiB, token_embd's twin 2.37 GiB — against a 32 GB device whose
  packed total is 15.92 GiB. W4 promotes the word-shadow on-core decode to
  the production matmul residency and narrows the twin to the embedding
  gather. The vehicle's OOM history (device-side twin construction,
  4,068,474,880 B `ttnn::where`) judged twin CONSTRUCTION, not shadow use;
  the shadow path's per-call decode is compute, not allocation.
- No eager/ambient TT pair is owed for this arm: the READY gate keys on
  `DecodeCaptureEnabled()` and adjudicates the capture leg; the eager arm is
  covered by the op-level eager decode-equality suites (E1). The ladder's
  `*_tenstorrent.npy` eager names stay available if a later row wants the
  second leg gated e2e.
- No manifest.json: the committed golden convention (`qwen3_greedy_0_6b`)
  carries per-arm `.npy` pairs + `p{i}_prompt.i32` only; the recipe lives in
  this Evidence section and the landing commit body.

## W4

Two changes, two pull requests (developer decision 2026-09-06):

**W4a — the 27B arm ([#3030](https://github.com/mudler/vllm.cpp/issues/3030)).**
(1) RESIDENCY: the flip waves are CANCELLED 2026-09-06 — two falsifications.
Wave-1's blanket flip OOM'd the vehicle at the tied head's per-call decode
(4,068,474,880 B `ttnn::where`). Wave-1b's threshold flip (twin above 128M
elements, shadow decode below) passed focused + suite (59/59, 6704/6704)
but the vehicle capture demanded 425,754,624 B of trace region against the
52,428,800 B allocated (TT_FATAL, mesh_trace.cpp:81) — a weight decoded per
call persists as a bf16 tile inside the captured graph, and replay re-runs
the decode every step, a throughput regression with no compensating win.
The twin-for-all diagnostic re-ran the same gate 16/16 PASS 147/147 (11
strict / 5 near-tie, max 0.188 nats, 0 forward-divergent), isolating the
cause to the flipped weights. Rejected repairs: trace-region enlargement
(~18 GB on 27B), capture-mode twins (words would feed only the eager arm —
dead in production), persistent decode scratch (defeats the arithmetic),
no-capture fallback (breaks the capture model). Production dense keep-quant
matmuls KEEP the memoized twin (the landed W3 behavior). Packed-word
residency pays off only where a kernel consumes the words natively: (2) the TT provider for `vt::MatmulBTQuantGrouped`
(ops.cpp:220, weight[E*N,K], expert_ids[P]) ports the ROCm reference's
native packed-word dequant (rocm_grouped_gemm.hip:1456, Q8_0/Q4_K/Q6_K —
extend to Q5_K with the W3 chain; the 27B pin carries 48 Q5_K tensors).
E=1 covers dense; E=N covers experts fed by the stacked tower
(qwen3_5_gguf_weights.cpp:1282). Twins remain for gather-class operands
(embedding; the vehicle's tied head shares the table). Capture
compatibility of the grouped arm is wave-3's committed obligation. (3) MTP `blk.64.*` skip/refuse by name. (4) 27B e2e greedy
near-tie gate, checkpoint-gated opt-in loud-skip (#2811 precedent), goldens
vs the pinned llama.cpp b10451 oracle, 500-mnat band + 0 forward-divergent.
(5) `docs/USAGE.md` pin in the same change. CORRECTED 2026-09-07
(tensor-name scan of the pinned artifact): the 27B is DENSE — arch
`qwen35`, zero `ffn_*_exps` / `ffn_gate_inp` / `exp_probs` tensors, and
`blk.0` carries singular `ffn_down`/`ffn_gate`/`ffn_up`. The earlier
"experts Q4_K ×294" classification was a dtype-only misread; those 294
Q4_K tensors are dense ffn/attn weights. Memory axis on 27B: the keep-
quant set beyond the gather class (~14.27 GB packed) cannot exist as
twins (~42 GB); it is served PACKED through the E=1 arm with a CHUNKED
slice-decode + accumulate matmul, so the captured graph holds chunk
buffers and never a whole-weight tile — ≈ 15.92 GB packed + 2.37 GiB
embedding twin + 2.37 GiB output twin + chunk tiles + activations ≈
22-23 GB on 32 GB. Replay re-decodes each step: correct, slower;
throughput is W4b's lever. The E=N expert arm serves the family's MoE
models (30B-A3B class) and stays staged-owed: no MoE artifact is on disk
and a ~17-20 GB download needs authority.

**AMENDED 2026-09-07 (fifth): the vehicle capture falsified the chunked
arm's capture claim at model scale — demand is per CHAIN, not per chunk.**
Wave-3b-1's vehicle AFTER leg fataled at capture with 444,424,192 B
demanded of the 52,428,800 B region (mesh_trace.cpp:81) after the two
device-level defects it exposed were fixed red-first (a chained ROW_MAJOR
activation reaching `ttnn::matmul` unconverted — `per_core_M = 0` at
`matmul_program_config.cpp:372`; and the E=1 arm committing ROW_MAJOR into
the output slot, so replay's tile-padded view overran the buffer,
`mesh_tensor_impl.hpp:33`; focused 4/4, suite 66/66, 524,428/524,428 after
both fixes). The demand is ~113 keep-quant weights × ~3.5 MB/chain of
serialized decode commands — the wave-3a per-chain constant — so chunk
sizing cannot pay it down; the cost multiplies by chain count. This
reconciles the third amendment's rejected "trace-region enlargement": that
rejection sized wave-1b's whole-weight-per-step decode planes (~18 GB);
the measured quantity here is the command stream (0.44 GB on 0.8B, to be
measured on 27B). RESOLUTION: the region policy mirrors the pinned
tt-metal's own practice (`models/demos/utils/trace_region_sizes.py`):
DYNAMIC (`trace_region_size=0`, the upstream default for unconfigured
models and deepseek-v3's explicit choice) is tried first on the vehicle;
if the pre-ITEM-5 overlap hazard reproduces, the fallback is a per-model
resolved region sized from measured demand (the upstream YAML shape),
documented in `docs/USAGE.md` beside the demand number, with the fixed
50 MB (the vLLM plugin's generic value, ITEM 5) as that fallback's
unspecified-model default. Capture demand stays a MEASURED, reported axis
(the gate message carries `LastTraceBytesForTest()`) — the axis is the
number, not the carve-out. Perf guard: the vehicle leg records tokens/s on
both sides of the switch ("replay re-decodes each step: correct, slower"
is already the accepted state; W4b owes the lever); a gate that cannot
complete inside the vehicle timeout is a NEEDS_DECISION stop, not an
accepted default. The twin-absence policy is unchanged: no whole-weight
resident decoded shadow on the dense path; decode planes stay per-chain
transients.

**WAVE-3B-2 LANDED AS A STAGED SLICE (2026-09-07, the coordinator resolved
the wave's NEEDS_DECISION as LAND AS A STAGED SLICE).** Landed: the
production MTP drafter skip — the loader's accounting deliberately passes a
declared head because its tensors ARE enumerated as expected, and the
trunk-only load then leaves them unread, so `LogQwen3_5GgufMtpHeadSkip`
prints the skip loud before any weight byte moves: all fifteen `blk.64.*`
tensors, 289,527,808 B, named in full, suppressed only when speculative
method `mtp` is configured (`model_loader.cpp`,
`qwen3_5_gguf_weights.cpp`). The skip message carries the denominator fact:
the pinned llama.cpp `b10451` oracle ignores the same tensors, so a gate
against it is matched work only with this skip loud. Landed with it: the
16-prompt oracle goldens `tests/parity/goldens/qwen38_gguf_q4km_27b/`
(`greedy_ids.npy`), derived from the byte-identical llama.cpp `b10451`
denominator harness — the TT-side run against them closes the gate; the
reachability test `tests/vllm/entrypoints/test_gguf_accounting_reach.cpp`,
which proves the skip through the PRODUCTION loader accounting rather than a
hand-built type; the checkpoint-gated TEST_CASE in
`tests/parity/test_qwen35_paged_engine.cpp`, inert until
`VLLM_CPP_QWEN38_27B_GGUF` names the artifact; and
`VT_TT_KEEPQUANT_CHUNK_BYTES` (env-doc allowlisted), the keep-quant chunk
plane budget — default 256 MiB surveyed on the 0.8B vehicle, empty/unset
keeps the default, a positive integer is a HARD CAP in bytes that trades
command-stream length for live memory (the ceil(N/8) trace term otherwise
forces a 606+ MB head plane whatever the budget says — exactly the alloc
that died at 27B).

UNREACHED: the 27B e2e gate, owed to
[#3042](https://github.com/mudler/vllm.cpp/issues/3042). The OOM evidence,
eight runs: failing allocations 1,073,725,440 B and 134,184,960 B;
free-at-failure 244 MB → 46 MB → 12.7 MB/bank; ~34 GB allocated against the
32 GB device with a 3.7 MB largest free block; batch budget 512 fails
identically, so the demand is not activation-sized; three mitigations tried
and failed. The residency sits ~11 GB above the ~22-23 GB surveyed design
residency above. Suspects, named and unmeasured: the 2.5 GB bf16 embed
twin, f32 plane transients, possible words double-staging. The next lever
is the device-side allocation trace, not another mitigation.

**W4b — the int8-dot lever ([#3031](https://github.com/mudler/vllm.cpp/issues/3031)).**
Quantized-domain integer vec_dot behind the same seam; profile-first
attribution; recorded-only throughput floor. Sequenced after W4a, never
bundled.

## Now

`ACTIVE`, 2026-09-06. W1 complete (#2989, open). W2 complete on the row
branch: the dot (`MatmulBTQuantKernel` — `DecodeQ4KBlocksF32` factored out of
the W1 kernel, one bf16 RNE, the `kMatmulBT` tile matmul) is reached through
`vt::MatmulBT`'s block-weight dispatch and sits inside the analytic
operand-rounding envelope (bound ratios 0.28-0.53, M=1 GEMV included); the
`kTENSTORRENT` predicate arm admits exactly `{Q4_K}` and the routing test
reds any widening past the registered set. Both red-first: the registration
REQUIRE and the six wrongly-admitted encodings reded before the
implementation. AMENDED 2026-09-06: the vehicle is mixed-quant, so W3
carried the Q5_K/Q6_K/Q8_0 decodes and the predicate widening before the
capture leg and the e2e battery (see the falsification section).
W3 EVIDENCE COMPLETE on the row branch (see `## Evidence`): capture dump x2
byte-identity, staging counter 0, READY gate 16/16 PASS (11 strict / 5
near-tie, 0 forward-divergent), backend proof 0 declines. W3 LANDED
2026-09-06 (025c6ed90..f98b63867, #3028). W4 scope committed (see `## W4`,
issues #3030, #3031). AMENDED 2026-09-06 (third): the threshold flip's capture leg
falsified the mechanism itself (trace region 425,754,624 B vs 52,428,800 B,
mesh_trace.cpp:81; per-step decode compute at replay; the twin-for-all
diagnostic ran the same gate 147/147 green). The flip waves are cancelled;
the branch diff reverted; dense twins are the shipped behavior. W4a(1)
re-anchors on the TT `kMatmulBTQuantGrouped` provider (wave-2: packed
tower, native in-kernel dequant, E=1 dense + E=N experts, Q4_K/Q8_0 first,
Q5_K extension owed, staged slice owed to wave-3 wiring and the 27B gate).
The 128M threshold dissolves. Wave-2b landed 14e8fe471 (registered set
{Q4_K,Q5_K,Q6_K,Q8_0}; suite 27,456 green; vehicle 147/147; operator-
passed; pushed). AMENDED 2026-09-07 (fourth): the 27B is dense
(tensor-name scan — the ×294 "experts" were a dtype-only misread), so
the 27B path is the E=1 arm with chunked slice-decode (capture holds
chunks, never whole weights); the E=N expert arm stays staged-owed
behind a MoE artifact. Wave-3a = E=1 chunked capture-compatible packed
dense; wave-3b = 27B wiring + gate + MTP skip + USAGE pin. AMENDED
2026-09-07 (fifth): the vehicle falsified capture-safety at model scale
(444,424,192 B = ~113 chains × ~3.5 MB of decode command stream); the
trace-region policy moves to the pinned tt-metal's dynamic / per-model
practice, demand stays measured and reported, and wave-3b-1c repairs
under it. AMENDED 2026-09-07 (sixth): wave-3b-2 LANDS AS A STAGED SLICE
(the coordinator's NEEDS_DECISION resolution) — production MTP head skip
by name, byte-identical llama.cpp-b10451 denominator goldens committed,
reachability test through the production loader accounting,
`VT_TT_KEEPQUANT_CHUNK_BYTES`, and the `docs/USAGE.md` 27B arm entry. The
27B e2e gate is UNREACHED — eight OOM runs, ~34 GB allocated against 32 GB
— and [#3042](https://github.com/mudler/vllm.cpp/issues/3042) owns it (see
`## Owed` and `## W4`). The row stays `ACTIVE`: W4b and the 27B gate are
the open scope.
