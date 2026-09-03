# Qwen3.5-0.8B CPU and gfx1100 numerical characterization

Row: `BACKEND-ROCM`.

Issue: [#1588](https://github.com/mudler/vllm.cpp/issues/1588).

Base: `4d10c8acc527c34a6a58a309d52ea5f8fbd1d47b`.

Branch: `row/BACKEND-ROCM-NUMERICS-1588`.

Primary oracle: vLLM at `5559679229bc961848b121ccdeaa8fa5d79bec98`.

Target: AMD Radeon RX 7900 XTX, `gfx1100`.

## Now

The issue is open and the implementation has not started. This spec is the
committed prerequisite for the implementation.

Current `main` runs the Qwen3.5-0.8B paged engine on ROCm. The backend matrix
still names its CPU and ROCm numerical characterization as open.

The current gate runs one default cache configuration. It does not characterize
`auto`, `bfloat16`, and `fp8_e4m3` separately.

The current dump records the residual halves and named layer stages. It does not
record full-attention cache writes or persistent Gated Delta Net state writes.

ROCm static graph mode remains false in `src/vllm/platforms/rocm.cpp:91-98`.
This work characterizes the current eager path before issue #332 activates model
graph replay.

## Live gap

The spec implementer checked the pinned base before writing this spec.

- `HEAD`, `origin/main`, and the merge base all resolve to the pinned base.
- `git log --all --grep=1588` finds no landed commit for issue #1588.
- No spec with this scope exists on the pinned base.
- GitHub reports issue #1588 as open.
- No open pull request owns issue #1588.
- The issue comment describes an older Q4_K_M experiment at `e2a9e035d`.
- That experiment predates the current checkpoint gate and the later W1 fix.
- That experiment does not compare persistent state or cache modes.

The current production gate is
`tests/parity/test_qwen35_paged_engine.cpp:159-484`. It loads the model through
`LoadedEngine::FromModelDir` and exercises the paged engine.

The gate checks 16 prompts against a pinned ROCm vLLM oracle. It reports strict
token agreement and applies the ratified 500 milli-nat near-tie rule.

The gate constructs `EngineParams{}` at
`tests/parity/test_qwen35_paged_engine.cpp:231-234`. It therefore covers only
the resolved default cache dtype.

The gate checks native selection and zero declines for 15 required operators at
`tests/parity/test_qwen35_paged_engine.cpp:254-275` and `:422-446`. It does not
run the required three-mode matrix.

The current dump writer lives in
`include/vllm/model_executor/models/act_dump.h:55-227`. Its manifest key is
`(step, layer, stage)` and its schema is:

```text
step  layer  stage  dtype  rows  cols  bytes  file
```

`DenseForwardLayers` records `hidden` and `res` before layer 0 and after every
layer at `src/vllm/model_executor/models/qwen3_5.cpp:9352-9403`.

`RunDenseLayerPaged` records `block_out` and `mlp_out` at
`src/vllm/model_executor/models/qwen3_5.cpp:7700-7772`. On a full-attention
layer, `block_out` is the complete attention output after its output projection.

These existing rows answer the residual-stream, attention-output, and
multilayer perceptron output questions. The comparator must reconstruct the
residual stream as `hidden + res` in FP32.

The full-attention store calls `dense_attn::WriteKvCache` at
`src/vllm/model_executor/models/qwen3_5.cpp:5894-5905`. No dump reads the
destinations after that call.

GDN prefill scatters working state at
`src/vllm/model_executor/models/qwen3_5.cpp:5280-5304` and `:5454-5495`.
GDN decode can update persistent state in place at `:5319-5334` and
`:5431-5445`.

No existing row records the state after all four write paths. Dumping an FP32
working buffer would not answer what the persistent cache stores.

## Scope

### In scope

1. Keep the existing `VT_DUMP_ACT` writer and manifest as the dump surface.
2. Keep `VT_DUMP_ACT_SUB` as the switch for named sub-stage rows.
3. Add no state-specific environment variable.
4. Record active full-attention K and V destinations after the production store.
5. Record active GDN convolution and SSM rows after the production update.
6. Compare CPU and ROCm on identical model inputs.
7. Report `max_abs`, root mean square error, and relative L2 error for each key.
8. Audit every relevant local dtype against the pinned upstream executing chain.
9. Report the byte cost of each local dtype and each upstream difference.
10. Gate the ROCm paged engine in `auto`, `bfloat16`, and `fp8_e4m3` modes.
11. Prove that each ROCm mode has zero reference-tier hits.
12. Run the final acceptance on one RX 7900 XTX with architecture `gfx1100`.

### Out of scope

- Enabling ROCm static graph mode. Issue #332 owns that change.
- Changing a model dtype before the characterization identifies a defect.
- Changing a kernel reduction order to make raw state bytes equal.
- Adding a new activation-dump environment variable.
- Dumping unused KV blocks or unused recurrent-state slots.
- Throughput, latency, and power measurements.
- A CUDA comparison.
- A GGUF or quantized-weight model arm.
- A public benchmark claim.
- A fix for a newly found numerical defect without its required issue flow.

## Artifact and oracle pins

Use this exact model snapshot for every local and oracle run:

```text
repository: Qwen/Qwen3.5-0.8B
revision:   2fc06364715b967f1860aea9cf38778875588b17
path:       /home/vikash/models/Qwen3.5-0.8B
```

The local checkpoint inspection found these hashes:

```text
model.safetensors  04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
config.json        b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
```

The model file is 1,746,942,600 bytes. Its safetensors header declares 452 BF16
tensors and 36 FP32 tensors.

`tests/parity/hf_snapshot.h` pins the same revision and exposes
`parity::Qwen35_08BSnapshot()`. An explicit snapshot override checks existence,
not revision identity.

The characterization must hash the two files before every run. A hash mismatch
is `ARTIFACT_MISMATCH`, not a skipped comparison.

Use the vLLM source checkout at `/home/vikash/oracle/vllm-src`. Its detached
`HEAD` must equal `5559679229bc961848b121ccdeaa8fa5d79bec98`.

Run the pinned vLLM oracle on the identical model, prompt set, cache mode,
sampling configuration, and token count. Use production configuration without
`--enforce-eager` for the end-to-end denominator.

The internal CPU and ROCm comparison remains eager on both local arms. The dump
synchronizes the queue and cannot run inside graph capture.

## Model geometry

The pinned checkpoint declares this text-model geometry:

| Field | Value |
|---|---:|
| Model dtype | `bfloat16` |
| Hidden size | 1,024 |
| Decoder layers | 24 |
| Full-attention interval | 4 |
| Full-attention layers | 6 |
| GDN layers | 18 |
| Query heads | 8 |
| KV heads | 2 |
| Attention head dimension | 256 |
| GDN key heads | 16 |
| GDN value heads | 16 |
| GDN key dimension | 128 |
| GDN value dimension | 128 |
| GDN convolution kernel | 4 |
| GDN SSM dtype | `float32` |

The full-attention layers are 3, 7, 11, 15, 19, and 23. Every other layer is a
GDN layer.

## Upstream executing chain

The implementation must preserve these pinned upstream decisions.

- `vllm/model_executor/models/qwen3_5.py:471-531` routes the text model and
  publishes its recurrent state dtype.
- `vllm/model_executor/models/qwen3_next.py:389-400` projects Q, K, and V, then
  calls attention before the output projection.
- `vllm/model_executor/models/qwen3_next.py:492-550` keeps the residual pair and
  routes each layer to full attention or GDN.
- `vllm/model_executor/models/config.py:744-768` copies checkpoint
  `mamba_ssm_dtype` into the default SSM cache dtype.
- `vllm/model_executor/layers/mamba/mamba_utils.py:96-128` resolves the
  convolution and temporal-state dtypes independently.
- `vllm/model_executor/layers/mamba/mamba_utils.py:180-199` defines the two GDN
  state shapes.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1228-1308`
  identifies the persistent convolution state and its indexed write paths.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1372-1478`
  identifies the persistent SSM updates and the prefill scatter.
- `vllm/v1/attention/ops/paged_attn.py:31-50` calls the paged KV store with the
  slot mapping and cache scales.

The local mirror resolves the cache layout in
`src/vllm/model_executor/models/qwen3_5_common.cpp:37-90`. The runner consumes
that layout at `src/vllm/v1/worker/gpu/runner.cpp:930-999` and `:1747-1793`.

The local full-attention store routes through
`include/vllm/model_executor/models/kv_cache_route.h:48-67`. The float and FP8
arms reach different store operators through this shared seam.

The implementation must inspect the complete executed kernel chain for each
mode. A dispatch wrapper alone does not establish a compute dtype.

## Dtype and byte contract

The checkpoint and pinned upstream source establish these storage types before
any measurement:

| Surface | Shape per active unit | Required storage | Bytes per unit |
|---|---|---|---:|
| Hidden state | `[T, 1024]` | BF16 | `2048 * T` |
| Residual half | `[T, 1024]` | BF16 | `2048 * T` |
| Full-attention K | `[T, 2, 256]` per layer | BF16 or FP8 E4M3 | `1024 * T` or `512 * T` |
| Full-attention V | `[T, 2, 256]` per layer | BF16 or FP8 E4M3 | `1024 * T` or `512 * T` |
| GDN convolution state | `[1, 6144, 3]` per layer | BF16 | 36,864 |
| GDN SSM state | `[1, 16, 128, 128]` per layer | FP32 | 1,048,576 |

`auto` and `bfloat16` both store full-attention K and V in BF16. Across six
full-attention layers, they use 12,288 bytes per cached token.

`fp8_e4m3` stores full-attention K and V in one byte per element. Across six
layers, it uses 6,144 bytes per cached token.

FP8 therefore removes 6,144 bytes per cached token from this model's
full-attention cache. The report must confirm the allocated bytes from the
resolved `FullAttentionSpec` instead of trusting this arithmetic alone.

Across 18 GDN layers, convolution state uses 663,552 bytes per active state
slot. SSM state uses 18,874,368 bytes per active state slot.

The required GDN total is 19,537,920 bytes per active state slot. Storing the
SSM state in BF16 would remove 9,437,184 bytes but would violate this checkpoint.

The dtype audit must include each model-path buffer and each GEMM output that
feeds a compared boundary. For each item, record:

1. The local symbol and runtime dtype.
2. The pinned upstream symbol and runtime dtype.
3. The tensor shape and active element count.
4. The local bytes and upstream bytes.
5. The byte difference at the measured token count.
6. The source annotation for every local FP32 exception.

The audit includes the attention query, K and V before storage, attention
output, GDN projections, GDN working state, MLP output, residual halves, stored
KV, and stored GDN state.

A wider local dtype is not accepted because tokens agree. Classify it as
`DTYPE_WIDTH_MISMATCH` unless a committed source annotation names the upstream
reason for the FP32 exception.

## State dump design

Enable state rows only when `VT_DUMP_ACT` and `VT_DUMP_ACT_SUB` name the same
writable directory. Use the existing writer and the existing manifest.

The existing `hidden`, `res`, `block_out`, and `mlp_out` keys do not change.
Add these four stage keys:

| Stage key | Production source | On-disk dtype | Manifest shape |
|---|---|---|---|
| `state_fa_k` | K destination after `WriteKvCache` | FP32 | `[active_tokens, 512]` |
| `state_fa_v` | V destination after `WriteKvCache` | FP32 | `[active_tokens, 512]` |
| `state_gdn_conv` | Persistent convolution cache after update | FP32 | `[active_requests, 18432]` |
| `state_gdn_ssm` | Persistent SSM cache after update | FP32 | `[active_requests, 262144]` |

FP32 is the canonical comparison format on disk. It does not change the cache
storage type.

For BF16 state, read the persistent destination and widen it to FP32. For FP8
KV, read the stored byte and dequantize it with that layer's recorded scale.

The run log must state the physical storage dtype, K scale, and V scale. The
dtype audit records the physical byte width independently of the FP32 dump.

Gather full-attention rows from nonnegative `slot_mapping` entries only. Preserve
the input token order instead of sorting by physical slot.

Gather GDN rows from the active state indices for the current step. Preserve the
request order used by the scheduler metadata.

Deduplicate no row silently. Refuse duplicate destination indices unless the
executing metadata defines their update order.

Do not download an entire KV block pool. Do not download an entire GDN state
pool. A single-request run must write one GDN row per layer per step.

Place the full-attention probe immediately after the shared production store.
The queue ordering must make the probe read the stored destination.

Place the GDN probe at one shared tail after all prefill and decode update arms.
The probe must gather from `state.conv_state` and `state.ssm_state`.

Do not dump `dcs`, `dss`, `kw`, or `vw` as substitutes. Those tensors precede a
downcast, quantization, scatter, or indexed in-place update.

Extend the per-step narrative with separate stream, stage, and state counts. A
24-layer step must report 48 state blobs, with two blobs from each layer.

An enabled state dump that writes fewer than 48 required blobs must refuse the
run. The manifest row count and active-row shapes remain the stronger capacity
checks.

## Comparison workload

Use one binary and one checkpoint copy for both local arms. Set
`EngineParams::device` to `kCPU` for the CPU arm. Use the detected ROCm platform
for the device arm.

Use one request at concurrency 1. Use the first standard gate prompt:

```text
The capital of France is
```

Use greedy sampling, a fixed seed, MTP disabled, and eight output tokens. Keep
the block size, block count, and scheduler token budget equal on both arms.

Run each cache mode in a separate process and a separate empty dump directory.
This avoids static environment parsing and file-name collisions.

Run each local arm twice with dumps enabled. The two repeats establish the
run-to-run floor.

Run an additional dump-disabled control for each arm and mode. Its token IDs and
logits must equal the corresponding enabled run.

Compare prefill and decode steps only when both arms consume the same token IDs.
The CPU denominator and ROCm treatment must share the complete input prefix.

If greedy outputs diverge, include the first divergent output step. Exclude every
later step because those steps consume different inputs.

Record the excluded step count. Do not compare equal step ordinals after the
input prefixes diverge.

If no decode step has an identical input prefix, classify the end-to-end failure
first. Do not use a re-prefilled prefix as evidence for incremental state.

## Metrics and advance policy

Decode every compared blob to FP32. Let `A` be ROCm and let `B` be CPU.

For `N` elements, report these metrics for every joined key:

```text
max_abs = max_i(abs(A_i - B_i))
rms     = sqrt(sum_i((A_i - B_i)^2) / N)
rel_l2  = sqrt(sum_i((A_i - B_i)^2)) / sqrt(sum_i(B_i^2))
```

If both vectors have zero L2 norm, define `rel_l2` as zero. If only `B` has zero
L2 norm, define `rel_l2` as infinity.

The report must name directory A and directory B in words. It must report both
manifest row counts, joined rows, rejected rows, and excluded post-divergence
steps.

The comparator must refuse missing keys, duplicate keys, dtype mismatches,
shape mismatches, byte-count mismatches, nonfinite values, and partial joins.

Raw byte equality is not the cross-device acceptance rule. CPU and ROCm kernels
can use different valid reduction orders.

Use these unit roundoffs:

```text
u(FP32)       = 2^-24
u(BF16)       = 2^-8
u(FP8 E4M3)   = 2^-4
```

Before reading cross-device results, finish the dtype audit. For each comparison
key `k`, record `n(k,d)`, the executed round-to-nearest terms at dtype `d`.

Count a destination store as one term. Count each reduction term at its actual
accumulator dtype. Cite the executing kernel for every count.

Freeze the audit file hash before the comparator reads either cross-device dump.
Compute this three-root-mean-square envelope:

```text
E(k) = 3 * sqrt((2 / 3) * sum_d(n(k,d) * u(d)^2))
```

The `sqrt(2)` term models two independent rounding paths. The `1 / sqrt(3)` term
is the root mean square error of round-to-nearest.

A key is inside the pre-registered dtype envelope only when both conditions hold:

```text
rel_l2 <= E(k)
rms <= E(k) * sqrt(sum_i(B_i^2) / N)
```

Report `max_abs` and the worst index for diagnosis. Do not invent a fixed
absolute threshold after the values are visible.

For ordered layer boundaries, compute the nonnegative squared-error increment:

```text
delta(k) = max(0, rel_l2(k)^2 - rel_l2(input(k))^2)
```

Classify a discontinuity when `delta(k)` exceeds `E(k)^2` and four times the
median increment for the same stage and layer family. Freeze this factor before
the hardware run.

The envelope classifies a result. It does not replace the end-to-end correctness
gate.

## Mismatch classification

Classify every run as exactly one primary result:

- `ARTIFACT_MISMATCH`: a source revision, model hash, or run configuration differs.
- `INSTRUMENTATION_FAIL`: an enabled dump is incomplete or changes model output.
- `NONDETERMINISTIC`: a same-arm repeat has a nonzero metric.
- `STRUCTURE_MISMATCH`: key sets, active indices, shapes, or physical dtypes differ.
- `DTYPE_WIDTH_MISMATCH`: local storage is wider than upstream without an accepted annotation.
- `NONFINITE`: one arm creates a NaN or infinity that the other arm does not create.
- `WITHIN_DTYPE_ENVELOPE`: every key meets the frozen envelope and has no discontinuity.
- `ORDERING_DRIFT`: tokens pass, but one numerical key exceeds the frozen envelope without a structural error.
- `NUMERICAL_DEFECT_CANDIDATE`: a state or layer discontinuity identifies a bounded production region.
- `CORRECTNESS_FAIL`: the applicable end-to-end token gate fails.

Apply the list in this precedence order:

1. `ARTIFACT_MISMATCH`.
2. `INSTRUMENTATION_FAIL`.
3. `NONDETERMINISTIC`.
4. `STRUCTURE_MISMATCH`.
5. `DTYPE_WIDTH_MISMATCH`.
6. `CORRECTNESS_FAIL`.
7. `NONFINITE`.
8. `NUMERICAL_DEFECT_CANDIDATE`.
9. `ORDERING_DRIFT`.
10. `WITHIN_DTYPE_ENVELOPE`.

`CORRECTNESS_FAIL` therefore outranks every interpretable numerical label.
`INSTRUMENTATION_FAIL` and `NONDETERMINISTIC` stop the cross-device reading.

`WITHIN_DTYPE_ENVELOPE` does not promise equal state bytes. It means the measured
difference fits the dtype-derived rounding model and the token gate passes.

`ORDERING_DRIFT` does not authorize a wider tolerance. Record the exact stage,
kernel chain, and token margin before any change.

## End-to-end correctness gate

Extend the current Qwen3.5 paged-engine gate with named cases for:

1. `auto`.
2. `bfloat16`.
3. `fp8_e4m3`.

Each case must set `EngineParams::kv_cache_dtype` explicitly. The test name and
log must print the requested and resolved cache dtype.

`auto` and `bfloat16` resolve to the same physical BF16 cache. Require their
local token streams to match each other exactly.

Capture a pinned vLLM oracle pair for each distinct physical cache mode. Do not
reuse the BF16 teacher-forced gaps for FP8.

For each mode, run the existing 16-prompt battery at its current token count.
Keep the current strict-token report and ratified 500 milli-nat near-tie rule.

The pinned vLLM oracle must use the same cache mode. Capture at least 10
per-prompt greedy repeats and require determinism before creating a golden.

Every ROCm case must prove all 15 required operators have nonzero native
selections and zero declines. It must also assert `reference_tier_hits=0`.

The state-dump characterization does not need to run the 16-prompt battery with
dumps enabled. The narrow workload supplies internal evidence without dumping
inactive capacity.

## Tests and red-first evidence

The implementation starts with tests that fail because the four state rows are
absent. Capture the red output before editing production code.

Add focused tests for these contracts:

1. The state writer emits all four keys with the declared shape and FP32 dtype.
2. A nonmonotonic slot mapping preserves input token order.
3. Negative slot mappings produce no KV row.
4. Inactive KV blocks and GDN slots produce no bytes.
5. A BF16 state row is widened from the stored destination.
6. An FP8 row is read after quantization and dequantized with its scale.
7. GDN prefill and decode both read persistent post-write state.
8. A missing required state row makes the per-step floor refuse.
9. The comparator refuses duplicate, missing, partial, short, and nonfinite data.
10. The comparator implements the zero-denominator rules exactly.
11. The dtype audit rejects an unannotated wider local dtype.
12. Each cache-mode gate prints its requested and resolved dtype.
13. The ROCm gate rejects one injected reference-tier hit.

The checkpoint-backed focused test must enter through
`LoadedEngine::FromModelDir`. A hand-built `PagedKvCache` test can test the
gather helper but cannot prove production reachability.

Run the focused CPU tests before hardware work. Run the ROCm cases only inside
the required GPU lease and file mutex.

`IMP-TEST-FIRST` and `IMP-MUTATE` do not apply to this spec-only commit. This
commit changes no behavior that a test or mutation can falsify.

They apply to the later implementation. Preserve each red result and each
mutation result in that implementation's evidence.

## Review mutations

A fresh reviewer reviews one immutable implementation head. The reviewer uses a
scratch copy and restores every file byte for byte after each mutation.

Run these mutations separately:

1. Delete the production probe after `dense_attn::WriteKvCache`.
2. Delete the shared GDN post-write probe.
3. Replace the post-write KV read with `kw` and `vw`.
4. Replace the post-write GDN read with `dcs` and `dss`.
5. Remove negative-slot filtering.
6. Sort active rows by physical slot.
7. Dump the full allocated cache capacity.
8. Report the FP8 physical dtype as BF16.
9. Remove one cache-mode test case.
10. Force one ROCm operator to the reference tier.
11. Delete the `LoadedEngine::FromModelDir` production call from the focused test.

Each mutation must make its named test fail for the intended reason. A green
reachability mutation is a review finding.

Record the immutable head, scratch path, command, exit status, and restored tree
hash for each mutation.

## Gates

The later implementation must pass these gates in order:

1. Confirm the checkpoint and oracle hashes.
2. Capture the red focused tests.
3. Pass the dump-writer and comparator tests on CPU.
4. Pass the checkpoint-backed CPU trace case.
5. Run the full controlled preflight.
6. Get a fresh immutable review with all mutations detected.
7. Run two CPU and two ROCm trace repeats for each cache mode.
8. Pass the dump-disabled identity control for every arm and mode.
9. Pass all three ROCm paged-engine end-to-end cases.
10. Prove zero ROCm reference-tier hits in every case.
11. Have the operator rerun the focused, full, and hardware gates.

Use this controlled environment for local repository gates:

```sh
PATH=/usr/local/bin:/usr/bin:/bin \
PYTHONPATH=$PWD \
GIT_CONFIG_GLOBAL=/dev/null \
scripts/agent-preflight.sh
```

An exit status of zero with skipped checks is not green. List each skip and its
reason.

The hardware run must use the repository's lease procedure. Use the file mutex
inside the lease because the target is a non-fleet local GPU unless `rc devices`
reports it as a fleet device.

## Evidence required

Record this evidence for the implementation and hardware run:

- The implementation commit and tree hashes.
- The clean source status before and after each gate.
- The exact checkpoint hashes and upstream revision.
- The compiler, build type, ROCm version, kernel driver, board name, and `gfx1100` architecture.
- The GPU lease identity or the non-fleet mutex evidence.
- The exact commands and exit statuses.
- Every preflight skip.
- The requested and resolved cache dtype for each run.
- The physical cache dtype and allocated bytes for each state surface.
- The operator-provider selection, decline, and reference-tier counts.
- Both enabled repeats and the disabled identity control.
- Both manifests and all raw dump paths.
- The frozen dtype-audit file and its hash.
- The full metric table for every joined key.
- The first token divergence and excluded step count.
- The selected mismatch classification.
- The red-first output and every review mutation result.
- The fresh reviewer report on the immutable head.
- The operator's independent gate rerun.

Keep raw evidence under an issue-specific evidence path. Do not publish a result
from temporary files that the reviewer cannot inspect.

The spec-only preflight ran at the pinned base with the controlled environment.
It returned zero and skipped 10 checks.

Five LTX-2.5 checks skipped because NumPy was unavailable. Three ISA checks
required compile commands. The pull-request size check required base and head
arguments. The Triton AOT check required a vendored source root.

This result is `PENDING`, not green. No source build, checkpoint inference, or
GPU gate applies to this spec-only commit.

## Risks

- The dump can synchronize the queue and perturb scheduling. The disabled
  identity controls detect output changes.
- FP8 K or V near a quantization boundary can change one stored code. Numeric
  comparison occurs after dequantization and never assumes byte equality.
- GDN update kernels use different reduction orders on CPU and ROCm. The frozen
  dtype envelope classifies that difference without widening after measurement.
- A greedy divergence makes later steps incomparable. The comparator excludes
  those steps and reports the exclusion.
- The FP32 canonical dump can hide a physical-width error. The independent byte
  audit and run narrative cover the storage width.
- A probe before the write can look plausible. The pre-write substitutions are
  required review mutations.
- Static environment parsing can mix output directories. Separate processes and
  empty directories prevent that collision.
- The current gate goldens describe the BF16 default. FP8 needs its own pinned
  oracle capture and teacher-forced gaps.
- Issue #1588 currently lacks the required first-line `Row:` metadata. An
  authorized operator must reconcile it before the pull request lands.
- Graph activation changes execution order and invalidates this characterization.
  Issue #332 requires a new graph-on measurement.

## Stop conditions

- Stop with `NEEDS_CONTEXT` if any required revision, hash, model file, or board
  identity is unavailable.
- Stop if the issue, spec, and pull request do not name `BACKEND-ROCM` together.
- Stop if the dump changes tokens or logits on either arm.
- Stop if a same-arm repeat is not deterministic.
- Stop if a manifest omits an active row or includes inactive capacity.
- Stop if the dtype audit is not frozen before cross-device results are read.
- Stop if the local chain cannot name an upstream dtype for a compared surface.
- Stop if any ROCm operator declines or reaches the reference tier.
- Stop if the running device is not an RX 7900 XTX with architecture `gfx1100`.
- Stop if ROCm static graph mode becomes true before the measurement.
- Stop before changing a tolerance after result values are visible.
- Stop before fixing an unexpected numerical defect without the required issue
  and fresh implementation flow.
- Do not report a hardware pass from a skipped test.

## Git integration

Use one pull request for the spec and implementation. This is the repository
policy default for `BACKEND-ROCM` and the selected shape for this row.

Commit this spec before any implementation. A fresh implementer starts from the
committed spec.

Keep the work on `row/BACKEND-ROCM-NUMERICS-1588`. The spec implementer does not
push or open the pull request.

The pull request body must link issue #1588 and close it when the complete work
lands. The issue's `Row:` line, this spec, and the pull request must agree.

## Owed

- Issue #1588 owes its first-line `Row: BACKEND-ROCM` metadata before landing.
- A fresh implementer owes the state probes, tests, comparator changes, and
  three-mode end-to-end gate.
- A fresh reviewer owes static review and every mutation in this spec.
- The operator owes the controlled build and the RX 7900 XTX hardware run.
- The operator owes the pinned vLLM production runs for all cache modes.
- The final implementation change owes an `## Outcome` in this spec.
- Issue #332 owns the graph-enabled repetition after ROCm graph activation.
