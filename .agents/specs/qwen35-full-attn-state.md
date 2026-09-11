# Qwen3.5 full-attention state validation

Row: `ENG-QWEN35-FULL-ATTN-STATE`.

Issue: [#3098](https://github.com/mudler/vllm.cpp/issues/3098).

Branch: `row/ENG-QWEN35-FULL-ATTN-STATE`.

Spec base: `6db4bef906859e864c82523c01107473f7dcca29`.

## Now

`ACTIVE`. The consumer-based implementation passes public completion and the
focused CPU suites. The operator independently reproduced the red result and
reran the public and CPU gates. The row uses one pull request under the
repository default. Specification commit `f62e4d7db0a2e1fb7d0e0a3cb5dad7072f182690`
precedes implementation.

Pinned oracle execution, the final gate, and fresh implementation review remain
`PENDING`. The current task does not authorize merging. This row makes no
performance claim and establishes no performance floor.

## Scope

Allow Qwen3.5 models containing only full-attention layers to prefill and decode
without recurrent state. Select Gated DeltaNet (GDN) validation and preparation
from the actual layer consumers. Preserve strict validation for every model
that contains a GDN layer, including missing caches and malformed metadata.

Cover the dense and mixture-of-experts (MoE) siblings in
`src/vllm/model_executor/models/qwen3_5.cpp`. Cover eager forward, graph entry,
and persistent device inputs because those paths share the same assumption.
Reuse the existing full-attention input builder.

The planner still publishes its legacy empty `gdn` group. This is an existing
local adaptation, not a cache-topology parity claim. Changing group topology,
registry group indices, or runner allocation belongs outside this fix.

Other exclusions are quantized providers, F16 matrix multiplication, fused MoE
kernels, new graph backends, new speculative modes, checkpoint downloads,
package installation, continuous integration changes, and global oracle pins.
The fix has no dependency on the separate gfx1100 provider changes.

### Runner regression amendment

On 9 September 2026 UTC, integration testing found an obsolete expectation in
`tests/vllm/v1/worker/test_runner.cpp`. Its full-attention-only step case still
expects the forward's former GDN refusal. The reviewed consumer checks now
allow that same unchanged fixture to execute successfully.

The operator authorizes a test-only correction under #3098. Require successful
execution, a valid sampled token with request feedback, and populated attention
state while recurrent state remains absent. Preserve the fixture's geometry,
weights, prompt, empty layer types, and single full-attention cache group.
This amendment changes no runner allocation, group topology, or product code.

Capture the current assertion failure before editing. Run the corrected focused
case and the complete runner suite. In a scratch build, restore the dense
forward's unconditional GDN metadata requirement. The corrected case must fail
at that original refusal. Restore all source and archive bytes after the
mutation. Fresh review and operator verification remain required.

## Inventory

This per-row inventory owns the child identity through the canonical spec scan.
No parent matrix or roadmap lifecycle changes. Update this table and `Now`
together when this child changes state.

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `ENG-QWEN35-FULL-ATTN-STATE` | Pinned `Qwen3_5DecoderLayer` and `GPUModelRunner.get_kv_cache_spec` | `CheckDensePagedForward`, `CheckPagedForward`, `BuildFullAttnStepDevInputs`, both graph `Step` methods | G0 to G5 | This file | `ACTIVE` | Row helper, fresh reviewer, operator verification | #3098 |

## Diagnosis and source anchors

At the spec base, the complete failing chain is:

1. `src/vllm/model_executor/models/qwen3_5_common.cpp::MakeQwen3_5KVCacheSpec`,
   lines 75 to 99, always publishes a `gdn` group.
2. `src/vllm/v1/worker/gpu/runner.cpp::GPUModelRunner::initialize_kv_cache`
   allocates recurrent buffers only for `linear_attention` layers at lines
   1511 to 1513. This model has none.
3. `src/vllm/v1/worker/gpu/runner.cpp::GPUModelRunner::execute_model`, lines
   2779 to 2805, builds and remaps GDN metadata when that group exists.
4. `src/vllm/model_executor/models/qwen3_5.cpp::CheckDensePagedForward`, lines
   9240 to 9249, accepts zero GDN layers and zero caches. It then validates
   live slot 0 against zero slots and throws at `ValidateGdnStateIndices`.
5. `BuildStepDevInputs`, lines 4659 to 4682, independently validates and
   prepares the same unused metadata. Repairing the first check alone fails.

The sibling `CheckPagedForward` repeats this assumption at lines 8526 to 8536.
`ForwardLayers` and `DenseForwardLayers` call the GDN builder at lines 8380 and
9323. Both graph `Step` methods validate GDN state before their eager fallback,
at lines 10763 and 11391. Persistent inputs repeat the builder at lines 11036
and 11774. The graph staging functions already test the GDN presence flags.

`git log -S'ValidateGdnAttentionMetadata'` identifies `f344decf4` and
`3ae5cfe06`. The original [packed-decode specification](gdn-packed-decode.md)
records why shared validation protects real consumers. Preserve that guarantee.

The existing `BuildFullAttnStepDevInputs`, line 7845, uploads full-attention
metadata and leaves GDN fields inert. Its current consumer is the multi-token prediction (MTP) draft
head. Reusing it avoids a second full-attention preparation implementation.

## Pinned reference and adaptations

The [primary pin](../upstream-sync.md) is
`e126687a9a828d513c01a07cd69f025f27d63280`, runtime
`0.28.1rc1.dev132+ge126687a9`. Read these executing upstream anchors:

- `vllm/model_executor/models/qwen3_5.py::Qwen3_5DecoderLayer.__init__`,
  lines 144 to 160, constructs GDN only for `linear_attention`.
- `vllm/model_executor/models/qwen3_5.py::Qwen3_5Model.__init__`, lines 249
  to 259, selects every decoder from `config.layer_types`.
- `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.get_kv_cache_spec`,
  lines 530 to 531, delegates to
  `vllm/v1/worker/gpu/attn_utils.py::get_kv_cache_spec`, lines 52 to 65.
  This V2 runner chain enumerates modules that actually require caches.
- `vllm/model_executor/layers/mamba/abstract.py::MambaBase.get_kv_cache_spec`,
  lines 67 to 87, obtains recurrent shapes from a recurrent consumer.

The exact no-GDN completion case is a local regression. The inspected upstream
`tests/models/language/generation/test_common.py`, `test_hybrid.py`,
`tests/models/test_qwen3_5_mtp_config.py`, and
`tests/v1/worker/test_gpu_model_runner.py` do not contain that case.
Preserve the existing ported GDN cases and their upstream anchors. Do not turn
unrelated KV-sharing allocation tests into a new topology obligation.

The operator's pinned runtime reports `Using V2 Model Runner`. The V2 chain
is the primary execution anchor for qualification.

Run the primary oracle on the identical generated dense GGUF before accepting
the fix. The [GGUF plugin pin](../oracles/vllm-gguf-plugin.md) is
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`. Its
`vllm_gguf_plugin/config_parser.py::GGUFConfigParser.parse` reads a local
configuration beside the artifact. Supplying that configuration and explicit
token IDs is an allowed harness adaptation. Preserve all model tensors and
geometry. A successful import or config construction is not an oracle result.

Use the existing pinned runtime, plugin artifact, and dependencies supplied by
the operator. Record their source revisions and binary hashes. Missing runtime
registration or unsupported full-attention execution leaves the oracle gate
`PENDING`. Do not install a package, substitute safetensors, change the model,
or use another oracle without an explicit scope decision. Do not force eager
execution on the oracle. Record its resolved production configuration.

The operator observed a missing `mm_proj` refusal on the sibling F16 oracle
workload. The pinned plugin's
`vllm_gguf_plugin/weights_adapter/qwen3_5.py::QWEN35_ARCHITECTURES`, line 40,
maps `qwen3_5_text` to conditional generation. Its `Qwen35GGUFAdapter.patch_hf_config`
replaces the architecture at line 225. Upstream
`vllm/model_executor/models/qwen3_5.py::Qwen3_5ForConditionalGeneration.__init__`,
lines 503 to 517, dereferences the vision configuration and constructs a tower.
A text configuration alone does not establish a runnable text-only GGUF path.
This known risk leaves G4 pending on an oracle refusal.

## Design

1. Derive the presence of GDN consumers from `weights.layers` and
   `is_linear_attention`. Reuse one internal predicate where practical.
2. Keep layer-count, attention-cache-count, GDN-cache-count, and cache-layout
   checks unconditional. A hybrid model with no caches must still fail.
3. Run GDN token-count, metadata, and graph-state validation only when a GDN
   consumer exists. Keep the validators themselves strict, including direct
   `BuildGdnStepInputs` consumers outside this model.
4. Select `BuildFullAttnStepDevInputs` for a model with no GDN consumers.
   Apply the selection to both eager builders and both persistent graph builders.
   Keep GDN upload and staging flags false on that path.
5. Pass inert empty GDN metadata into graph padding when no consumer exists.
   `BuildPaddedDecode` must not copy unvalidated, arbitrary-length GDN indices
   into its S-entry storage. Retain the bounds enforced for actual consumers.
6. Keep generic attention and graph shape checks effective for both model
   topologies. Preserve real-GDN decode, prefill, mixed, speculative, and padded
   state validation. Do not use GDN metadata as the only generic shape check.

Never infer the absence of consumers from `state_slots == 0` or an empty cache
vector. Never add dummy recurrent state, a fake layer, a validation bypass flag,
or a fixture geometry change. Keep model activations and key-value (KV) storage at their
resolved bf16 dtype. Retain existing annotated f32 exceptions and integer
metadata layouts. This change adds no wider model buffers or recurrent tensors.

## Regression fixture

Use only the dense control from the frozen gather experiment. Extract its
minimal deterministic generator into this row's own fixture file. Preserve the
bytes of `Q4_0-dense.gguf`, even though its embedding and projections are dense.
Do not copy the 19-format provider campaign into this regression.

The artifact has 991296 bytes and SHA256
`0e6554ba521edfde00d4d025a3058eabaaacdce6b24344f959e83d8dde35df7f`.
Its geometry is hidden width 256, intermediate width 256, vocabulary 128,
four query heads, one KV head, head width 64, one full-attention layer, and
`full_attention_interval=1`. Rotary width is 64, base is 1000000, and MRoPE
sections are `[16,8,8,0]`. Norm epsilon is `1e-6`, seed is `0x524f434d`.
Inactive state-space model (SSM) loader keys retain convolution width 4, inner width 64, state width
64, time-step rank 1, and group count 1.

Prompts are `[1,0,63,127,63]` and `[1,127,0,127]`. Request exactly four greedy
tokens with `ignore_eos=1` and seed `0x524f434d`. Use block size 16, four blocks,
maximum length 64, and one sequence. Public device 0 selects AUTO in the HIP-only
build. Assert the actual ROCm dense embedding provider executes.

For both local and oracle runs, execute each prompt three times with a fresh
engine and fresh model state. Keep batch size and concurrency at one. Submit
prompts sequentially and record the actual scheduling and token budgets.

## Gates and evidence

| Gate | Required result | Current result |
|---|---|---|
| G0: independent red | New public regression on pristine product base fails at GDN validation after load | Satisfied: operator reproduced the row-owned failure after public load |
| G1: public completion | Both prompts produce four tokens in three fresh-engine repeats, native ROCm provider executes | Satisfied: 12 fresh engines, default and callback sampling arms |
| G2: consumer distinction | Dense and MoE production routes accept no-GDN state and reject damaged real-GDN state | Satisfied: focused suites and independent operator rerun |
| G3: graph preparation | Both drivers cover cold, capture, persistent staging, and replay routing with no GDN consumers | Satisfied for CPU routing and staging, no GPU replay numerical claim |
| G4: pinned oracle | Same dense artifact and all six fresh-engine runs, exact token IDs, finite logits, preserved dtypes | `PENDING`: pinned plugin rejects the actual text-only engine configuration |
| G5: review and full gate | Focused gates, mutations, full preflight, fresh review, and operator rerun | `PENDING`: fresh review and final operator gate |

G0 starts at `vllm_engine_load` and `vllm_complete_tokens` from `include/vllm.h`.
The test must assert successful load before its expected pre-fix completion
failure. Reproduce with the row's fresh build from unchanged base product code.
The preserved gather red establishes the diagnosis, not this independent gate.

For G2, enter `ModelRegistry::Forward` for both dense and MoE variants. Cover
prefill and decode without GDN consumers, with default-empty and unused
runner-style GDN metadata. For real GDN consumers, reject missing all caches,
wrong cache count, invalid ranks, inconsistent slots, missing metadata,
duplicate indices, and out-of-range indices. Keep valid hybrid completion green.
A validator-only unit test does not prove these production call sites execute.

For G3, extend the existing graph harness for both drivers. Exercise their
production `Step` methods, graph fallback, persistent inputs, and staging flags.
Include no-GDN metadata with an oversized unused index vector. The padding path
must ignore that vector safely. Mutate the inert-metadata selection and require
this regression to fail. Use a memory-sanitized run if needed to expose an
out-of-bounds copy reliably.
Trace the registry dispatch to each driver and mutate each changed call site.
A CPU fake replay proves routing only. Record that limitation and obtain an
operator GPU replay result for any claimed replay numerics. Do not enable a
new backend or claim a GPU graph executed from the CPU harness.

Register a dedicated public test such as `test_capi_qwen35_full_attn_state`.
Run it with `test_qwen27_paged_forward`, `test_qwen35_paged_forward`,
`test_qwen3_5_decode_graph_seam`, and `test_model_registry`. Retain existing
GDN state, speculative metadata, and graph-padding cases in those suites.
Build in a row-owned ignored `build-*` directory with at most `-j 4`.
The implementer records exact configure, build, and focused test commands
before handing GPU commands to the operator. No shared build directory is valid.

For every result, record the immutable source SHA, binary SHA256, fixture hashes,
command, environment, exit status, and evidence path. G4 additionally records
source anchors, runtime and plugin identity, output tokens, resolved dtype,
attention backend, and graph mode. No config-only or local self-comparison
substitutes for G4. A token mismatch remains failing until explained and fixed.

Fresh review uses scratch copies and restores every modified file byte-for-byte.
Restore unconditional GDN validation at each newly conditional entry. Restore
the GDN builder at each full-attention selection. Delete the production call
sites into the changed preparation path. Each relevant positive gate must fail.
Then force the no-GDN branch for a hybrid model, or remove cache-count
validation, and prove the corresponding malformed-hybrid gate fails. Mutate duplicate, range, missing
metadata, and layout guarantees individually. Keep per-mutation failure logs.

Run `scripts/agent-preflight.sh` before edits and the staged form before commit.
Run exact-range record, commit-style, trailer, and PR-size checks after commit.
Classify omitted hardware and build gates explicitly. The operator repeats the
applicable gates on the immutable implementation before publication.

## Existing evidence

On 8 September 2026 PDT, the operator ran the frozen gather public binary on local
gfx1100 under `/home/vikash/gpu.lock`. Completion threw
`qwen3_5: GDN state index out of range`. The process returned 1 with five
passing assertions and one failed completion assertion.

The binary SHA256 is
`f6a109ef33a133b381a112313591534f560adedb0d4919f0d67567cd597552e1`.
Its source was spec commit `670e6d78ddf55231394748e0032939fd53dc56a5` plus
uncommitted test-only changes. The fixture header SHA256 is
`339414f93e59e6be9a4ca545d5e762bb2816714343f466e218dbc2b69483bdfd`.
The public test source SHA256 is
`add58ac23eef26588fb358d7679feeac961baee3a1da3d5ddd3adedaf9a1a825`.
The log is `/home/vikash/.cache/rdna3-gather-impl/evidence/public-red.log`, SHA256
`fbd5d684566961d71f31c8b9e58daba8d5303636adb12ceccb164dbe3b220e1d`.
These are supplied local evidence paths, not environment defaults.

## Implementation evidence

Measurements on 9 September 2026 UTC use the row-owned worktree
`/home/vikash/vllm.cpp-qwen35-full-attn-state-impl`. Evidence is under
`/home/vikash/.cache/qwen35-full-attn-state-impl/evidence`. These paths identify
measured artifacts. They are not environment defaults.

The product source is `qwen3_5.cpp`, SHA256
`899913b8aeff65a08eb5289f6f08777ee78638e37722f9d8138daff723f8f498`.
The red source is the unchanged product at specification commit
`f62e4d7db0a2e1fb7d0e0a3cb5dad7072f182690`. Green snapshots carry that base,
the exact implementation patch, and every test source hash. The implementation
commit containing this record makes those bytes reachable from Git.

### Public red and completion

The operator generated the row's fixture and confirmed its 991296-byte size
and the frozen SHA256. The row-owned red binary has SHA256
`76c4e04bfb1b2715d4010e11a1d85cb076c7ada8e146094a1419fc0566d6efff`.
It returned 1 after `public load succeeded`, with 11 passing assertions and
one failed completion assertion. The error was `GDN state index out of range`.
`red-snapshot/public-red-operator.log` has SHA256
`dbf8bb09eccc4d626d0c6cc4a9d97d93738563954dae0aeefa32b9ed5bfd8981`.

The green binary has SHA256
`14d2fee7f63df4c8ab7721923d429822488e7ac55a058268b9145bd4d5f371a7`.
The operator ran all 12 fresh engines and passed all 3364 assertions.
`green-v1-snapshot/public-green-operator.log` has SHA256
`6318b014c368835a71fe2501444adaf61b3312df178c564d0a0d3fba820f9c4e`.
The two token sequences were `[47,19,4,20]` and `[11,28,104,78]`.
Each sequence was identical in three repetitions of both sampling arms.

The callback-free arm retains default device sampling. The callback arm
observes 512 finite logits per completion. A custom processor stages logits
through the host at `src/vllm/v1/sample/logits_processor/builtin.cpp::apply_logits_processors`,
lines 75 to 159. Both arms reach `vt::GreedyArgmax` through the sampler.
Each run executes one native ROCm embedding prefill, three embedding decodes,
four cache writes, and four greedy operations. Verified named providers call
the original native functions. They check bf16 embedding activations and KV
storage, f32 sampler logits, and i64 sampled IDs. Reference-tier hits do not
increase. The f32 logits belong to the existing sampler ABI.

Both public commands use the operator's local GPU 0 and mutex:

```sh
env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  TMPDIR=/home/vikash/.cache/qwen35-full-attn-state-impl/tmp \
  GIT_CEILING_DIRECTORIES=/home/vikash/.cache/qwen35-full-attn-state-impl/tmp \
  GIT_CONFIG_GLOBAL=/dev/null \
  flock -n -F /home/vikash/gpu.lock \
  /home/vikash/.cache/qwen35-full-attn-state-impl/evidence/green-v1-snapshot/test_capi_qwen35_full_attn_state
```

Use `red-snapshot` in the executable path for the frozen red command.
Each snapshot's `manifest.json` contains exact argument arrays and source
hashes. The operator receipts record execution and restoration checks.

### CPU consumers and graph routing

The implementer and operator passed these complete suites:

| Suite | Test cases passed | Assertions passed |
|---|---|---|
| `test_model_registry` | 24 | 993 |
| `test_qwen27_paged_forward` | 36 | 1406 |
| `test_qwen35_paged_forward` | 10 | 676 |
| `test_qwen3_5_decode_graph_seam` | 16 | 1550 |

The registry suite retains one pre-existing skipped case. All new cases run.
`cpu-green-v1-snapshot/manifest.json` has SHA256
`575e9695acc9600515050a6ba0ed509795df45f7a128b98c6c303eb87c005183`.
It freezes all four executables and their source files. The operator's
per-binary logs, receipts, and `operator-results.json` are beside the manifest.

The no-GDN tests enter both registered forwards for prefill and decode.
Empty metadata, unused runner metadata, and stale configuration labels all
reach the loaded layer consumers. Separate negative cases preserve each
hybrid cache-count, rank, slot-layout, metadata, duplicate, and range guard.
Existing speculative, prefill, mixed, and padded-state tests remain green.

Both graph drivers cover capture enabled and disabled, asynchronous staging
enabled and disabled, batches 1 and 3, and three unused metadata forms.
Six steps cover cold execution, capture, persistent input binding, and replay
routing. Batch 3 pads to 4. Oversized unused indices contain 4096 entries.
Each persistent slot binds exactly five generic inputs, with no GDN input.

The dense registry reaches its actual graph driver under the existing CPU
harness. The MoE registry needs its FP4 capability predicate. Its test uses
valid small NVFP4 expert tensors and a scoped CPU platform answer. A control
without that answer stays eager. Both use the existing CPU arithmetic and fake
graph replay. These cases prove routing and staging, not GPU replay numerics.
No backend or quantized provider changes.

The shared builder's existing consumers also pass:
`test_qwen3_5_gguf_mtp`, `test_qwen35_exl3`, and
`test_qwen3_dflash2_draft`. Their complete CPU run returned 0.

### Build and focused commands

Run from the row worktree. Set `TMPDIR` and `GIT_CEILING_DIRECTORIES` to
`/home/vikash/.cache/qwen35-full-attn-state-impl/tmp`, and set
`GIT_CONFIG_GLOBAL=/dev/null` for gate subprocesses. Configure commands were:

```sh
cmake -S . -B build-full-attn-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF -DVLLM_CPP_VULKAN=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_MLX=OFF -DVLLM_CPP_TENSTORRENT=OFF \
  -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_SERVER=OFF
cmake -S . -B build-full-attn-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1100 \
  -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF -DVLLM_CPP_MLX=OFF \
  -DVLLM_CPP_TENSTORRENT=OFF -DVLLM_CPP_TRITON=OFF \
  -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_SERVER=OFF
cmake --build build-full-attn-cpu -j 4 --target test_model_registry \
  test_qwen27_paged_forward test_qwen35_paged_forward \
  test_qwen3_5_decode_graph_seam test_capi_qwen35_full_attn_state
cmake --build build-full-attn-hip -j 4 --target test_capi_qwen35_full_attn_state
ctest --test-dir build-full-attn-cpu --output-on-failure \
  -R '^(test_qwen27_paged_forward|test_qwen35_paged_forward|test_qwen3_5_decode_graph_seam|test_model_registry)$'
cmake --build build-full-attn-cpu -j 4 --target test_qwen3_5_gguf_mtp \
  test_qwen35_exl3 test_qwen3_dflash2_draft
ctest --test-dir build-full-attn-cpu --output-on-failure \
  -R '^(test_qwen3_5_gguf_mtp|test_qwen35_exl3|test_qwen3_dflash2_draft)$'
```

All configure, final build, and green focused commands returned 0.
The original CPU red returned 8 from CTest. It detected both intended GDN
errors. Two valid-hybrid test assertions initially assumed host logits. The
registered forward returns device logits, so the tests now download them for
finite-value checks. No product behavior changed for that test correction.
`cpu-red-snapshot` preserves the original tests, product source, and binaries.

### Implementer mutations

The three address-sanitized baselines pass. All 30 mutations fail their focused
gates. `mutations/manifest.json` has SHA256
`9f2cb023ea64af396a2fccbbe83b0dd8fc67a591aed9403c399cfa13f7a515ff`.
The external recipe `run-mutations.py` has SHA256
`a0855b588f8d2e7a7286ee9a56b2a0551f8111059537e264127a2f6c9787b3b2`.

Each mutant compiles a separate source copy at `-O0 -g -fsanitize=address`
and links the row's own CPU archive. The archive starts fresh for each mutant.
The corresponding baseline uses the same compiler and linker flags.
`ASAN_OPTIONS=detect_leaks=0:abort_on_error=1` disables unrelated leak reporting.
No tracked source or normal build output changes. Each manifest entry records
its command, mutated source hash, binary hash, failed exit, and restoration.

For each sibling, mutations restore unconditional entry validation, force the
entry's no-consumer branch, remove cache-count validation, and restore the GDN
eager builder. They independently restore graph validation, GDN padding-size
restrictions, unused spec classification, and persistent GDN preparation.
Removing each pre-padding shape check fails. Restoring the unused GDN index
copy causes an address-sanitizer heap-buffer-overflow in both drivers.

Six shared mutations separately remove rank, paired-slot, cross-layer-slot,
duplicate, range, and missing-index guarantees. Both registry suites fail
for each removal. Four further mutations delete each registry's graph and
eager production call. Each relevant positive gate fails. Fresh independent
review must repeat the required mutations on the immutable implementation.

### Full gate and build classification

`preflight-start.log` records the full gate before edits.
`preflight-staged.log` records the staged run before the implementation commit.
The staged run uses the same temporary-directory and Git isolation as the
focused tests, plus the existing NumPy dependency path supplied by the operator.
An external argument wrapper adds `--jobs 4` only to the compile checker.
It preserves every gate and changes no tracked script.

The generic preflight sweep does not supply every build checker's arguments.
The x86 CPU ISA audit runs explicitly against
`build-full-attn-cpu/compile_commands.json` and passes in `cpu-isa-build.log`.
The exact-range checks supply the recorded product base and implementation
head, with draft PR 3101 for path classification. Their results belong in
`range-gates.json` beside the other evidence.

ARM ISA, CUDA fat-binary, and Triton AOT build audits are outside this row's
configured CPU and HIP builds. The change edits no architecture-specific
instruction unit, CUDA gencode setting, or Triton artifact. The normal public
test returns skip code 77 in a CPU-only build because public AUTO requires the
HIP-only configuration here. The operator's HIP run executes every assertion.
GPU graph replay numerics remain unclaimed. G4 and fresh review remain pending.

### Pinned oracle refusal

The operator ran the actual first engine against the unchanged dense GGUF.
The engine returned 1 before becoming usable and emitted no tokens. The
exception reports `Qwen3_5Config` required and `Qwen3_5TextConfig` received.
The executing rejection is in upstream
`vllm/multimodal/processing/context.py::InputProcessingContext.get_hf_config`,
line 140, called by `vllm/model_executor/models/qwen3_5.py::Qwen3_5ProcessingInfo.get_hf_config`,
line 108.
The complete traceback is preserved in the report and operator log.

The runtime reports vLLM `0.28.1rc1.dev132+ge126687a9`, PyTorch
`2.12.0+git6bbd260`, HIP `7.2.53211`, and plugin
`0.0.5+d4c1f0d.gfx1100`. The script supplies only local HF text metadata and
explicit token IDs. Model geometry, tensor bytes, inactive SSM keys, and MRoPE
sections match the frozen fixture. Production graph mode remains requested
with `enforce_eager=False`. No registry, model, or plugin correction is applied.

`oracle-exact-gguf-v1-command.json` records the exact Docker argument arrays,
image digest, read-only runtime and plugin mounts, nonroot user, private
2 GiB IPC allocation, offline settings, and GPU 0 mutex. Each of its six
commands starts a new process and engine. The operator stopped after the first
refusal, because the remaining five commands cannot supply missing runtime
capability. No token, finite-logit, resolved-dtype, or GPU graph result is claimed
for the oracle.

The attempt report `oracle-exact-gguf-v1-p0-r0.json` has SHA256
`46d714e5fe6577663b0cb266c8d88c9a4a629eb2727314775a5b5dbb605ac2c0`.
Its operator log has SHA256
`5244fbca9f48db1bfc6d9817d823fcc4e1fe1a48a89f1c04c216ad1a5db4cdc6`.
The script SHA256 is
`504162e44ba828249a1a825bf2053f53e7902f168bad3ba515334ff8440a7f3e`.
The local HF configuration SHA256 is
`6c4b2c6f3d71b90818730bbfa900a5cbb0a96bd64ab656fd9db8e65111c05638`.

G4 remains `PENDING` on the operator's pinned GGUF runtime capability.
The implementation does not substitute a model or oracle. Issue #3098 retains
ownership of this gate, independent review, and the final operator gate.

### First-offset and ordering coverage repair

Fresh review of `25bea3e67597f6700fc1f2e9cfb5269948338c6d` found two missing
negative cases in `FullAttnGraphShapeGuards`. Deleting the first-offset check
or the ordering check independently left the existing focused suites green.
The reviewer found no product defect for this finding.

The repair adds distinct subcases for both dense and MoE graph `Step` methods.
The first-offset case uses `{1,1}` with one request. The ordering case uses
`{0,2,1,3}` with three requests and matching token, slot, sequence, and block
arrays. Both cases retain empty GDN metadata and the existing exception checks.
All production bytes remain unchanged from the reviewed implementation.

Repair evidence on 9 September 2026 UTC is under
`/home/vikash/.cache/qwen35-full-attn-state-repair1/evidence`.
The linked worktree is `/home/vikash/vllm.cpp-qwen35-full-attn-state-repair1`.
The repaired test source has SHA256
`b8f5d330fd2d6436e19e9573406adda85cb6fd1ce1736332536b1da5df403ba5`.
The normal graph executable has SHA256
`0af438e83c0c15b66860e95dd6ea47660555387a5fc9d6716ee34a5b1d0c5d80`.

Each deleted-guard mutant returns 1 for each sibling's corresponding new
subcase. Each failure reports that the expected exception did not occur.
An unchanged address-sanitized control passes both complete shape cases.
Every mutation uses a scratch source and a fresh copy of this worktree's
CPU archive. Production and test restoration hashes match after every run.
The mutation manifest has SHA256
`5248b0efd6fc9e0cf465aed89a36c87933f8dca0e9cbec71e07bbb157d0d6607`.
It records exact compile, link, and run arguments, binary hashes, and logs.
The external recipe is
`/home/vikash/.cache/qwen35-full-attn-state-repair1/run-mutations.py`, SHA256
`9bb055dbdcbc070ecd9ce06e21968d44328e14552ecd6759f44da4b61c2804cf`.

The four complete CPU suites listed under G2 and G3 pass in `cpu-green.log`.
The complete graph suite passes 16 cases and 1558 assertions in `graph-green.log`.
Configure and build arguments match the earlier CPU recipe, with at most
`-j 4` and this repair worktree's paths. Temporary files stay outside the
worktree, and `GIT_CONFIG_GLOBAL=/dev/null` isolates Git fixtures.

The full preflight before edits returns 0. The staged preflight uses the same
NumPy path and external compile-scheduler wrapper as the implementation gate.
The repair handoff records the staged result and exact-range checks.
The explicit CPU instruction-set audit passes against this worktree's
`build-full-attn-cpu/compile_commands.json`.
Fresh scoped review and the final operator gate remain pending.
G4 retains the pinned engine refusal. This test repair adds no oracle waiver,
GPU replay numerical claim, or performance claim.

### Runner completion expectation repair

The committed amendment `78c7441bd26db62904c18f6c105dd64dac265e77` precedes
this test edit. The unchanged pre-repair case returns one because it expects
an exception from a successful forward. The focused red log SHA256 is
`0750d5cb6e1cedda4e690e8340d3331790193604c04fb99ceffbfeca4c865641`.

The repaired case passes 18 assertions. It checks the actual prompt, produced
logit row and request counts, changed attention-cache bytes in both layers,
an in-range sampled token, and token feedback into the admitted request.
The GDN group, caches, and metadata remain absent. The fixture bytes and
all production source remain unchanged by this repair.
The complete CPU runner suite passes 41 cases and 1914 assertions.

Evidence is under `/home/vikash/.cache/rdna3-f16-repair1/runner-repair`.
`obsolete-expectation-red.json`, `build-receipt.json`, `focused-green.json`,
and `full-green.json` pin the source, private CPU and HIP binaries, exact
commands, and logs. The corrected test source SHA256 is
`8b70d78678245776737965cd461123b62716ed1d2cf130d604234e67519f09b2`.

The separate scratch mutation restores only the dense entry's unconditional
GDN validation. The repaired runner case returns one at the original
`gdn_meta.num_actual_tokens must equal T` refusal. Its two assertions contain
one setup pass and one intended failure. The mutation log SHA256 is
`dd4585b2a527b262ba3447444a48566144f56c30ffc99419719eb69ea4e68d83`.
`../mutations/runner-gdn-consumer/recipe.json` records the fresh archive copy,
compile and link arguments, mutant binary, and byte-exact scratch restoration.
The original production source and archive hashes remain unchanged.
Fresh scoped review and the final operator gate remain required. The pinned
GGUF primary refusal retains its existing G4 disposition.

## Risks and stop conditions

- `NEEDS_CONTEXT`: the frozen fixture bytes, pinned runtime, or source cannot
  be verified. The operator owns the missing artifact or runtime decision.
- `NEEDS_DECISION`: the oracle requires a different model artifact, new
  dependency, or behavior outside the stated scope.
- `BLOCKED`: required GPU execution is unavailable. Only the operator runs
  the GPU work under the recorded mutex or applicable fleet lease.
- `FAILING`: a no-GDN route still prepares device recurrent metadata, a real-GDN guard
  weakens, or a mutation remains green. Repair through a fresh implementer.
- `FAILING`: graph coverage relies on fake replay numerics or a different
  untested sibling. Keep the affected gate pending until valid evidence exists.
- `NEEDS_DECISION`: the minimal fix requires planner topology, unrelated
  quantization, broader speculative behavior, or any excluded file change.

## Owed

Issue #3098 owns the pending pinned oracle execution, fresh review mutations,
and the final operator gate. Public red, local completion, consumer validation,
and CPU graph routing have measured evidence above. Record the measured
outcome and defaults here before changing the row to `DONE`. Keep the issue
open until the work lands.
