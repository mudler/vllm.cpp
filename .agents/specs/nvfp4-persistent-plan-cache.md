# NVFP4 persistent FlashInfer-compatible plan cache (W3-C)

Status: **ACTIVE; W3-C2 and six-process stability pass; W3-C3R proves the
cross-run long-output predicate invalid and the corrected same-plan component
is ready**

Owning row: `KERNEL-GEMM-NVFP4-W4A4`

Claim: `CLAIM-NVFP4-SMALL-M-3`

W3-C mirrors the persistent FP4 autotune-cache behavior that actually executed
in the binding vLLM v0.25.0 oracle. It is first a benchmark-validity and
reproducibility repair: a persistent cache can remove cross-process tactic
selection as an uncontrolled variable, but it is not credited as a
steady-state speedup merely for avoiding startup tuning. Performance credit
requires the same-plan component and exact oracle gates below.

This spike supersedes the v0.24-era classification in
`nvfp4-small-m-dispatch.md` that called persistence optional because that older
oracle disabled its file cache. Historical ledger/state entries remain
historical; current decisions use the v0.25 execution evidence here.

## Why this row is selected

The immutable binding v0.25 trace log proves the production denominator used a
persistent cache:

- `~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27/vllm-profile.log`
  (SHA-256 `367af18d...2da`) names the cache path, loads 64 configs, reports a
  `fp4_gemm` config-file hit, then reloads the same cache after warmup;
- the exact file is
  `~/.cache/vllm/flashinfer_autotune_cache/0.6.13/121a/cbd38fe31b19a593fd4ac474a8a138a545227805f23c425119de8429f384d163/autotune_configs.json`
  (11,947 bytes, SHA-256 `b41a8ecc...677`);
- its metadata binds FlashInfer 0.6.13, CUDA 13.0, cuBLAS 13.1.0, cuDNN
  91900, cuDNN frontend 1.26.0 and `NVIDIA GB10`; its 64 FP4 entries cover
  16 hybrid M buckets for the four post-W3-D Qwen3.6-27B N/K shapes;
- every value selects `CutlassFp4GemmRunner` plus a tactic ID in 0--31. Those
  IDs have the same descriptor order as our checked-in tactic ABI.

Our W3-E component instead retuned all 64 entries in every fresh process.
Although all 12 legs completed with zero lazy misses, paired direct/fallback
runs shared only 18--33 of 64 selected IDs; only 9--17 keys per c16 arm and
12--15 per c2 arm retained one ID across all three repetitions. Five of six
paired 128-token output hashes differed. The fixed-tactic operator gates prove
the direct and composed scale paths are byte-equivalent; variable CUTLASS
reduction order therefore remains a real confounder. It must be removed before
another sub-percent component result can bind.

The dynamic scan ranked the next speed work separately: device-resident FP4
alpha, vectorized activation producers, packed GDN decode fusion, host-weight
release, and an isolated FA2 decode comparison. None is stacked into W3-C.

## Complete upstream and dependency chain

vLLM paths are pinned to v0.25.0 tag
`702f4814fe54fabff350d43cb753ae3e47c0c276`. FlashInfer paths refer to the
installed 0.6.13 source at
`~/venvs/vllm-oracle-v0.25.0-stage/lib/python3.12/site-packages/flashinfer/`
(`autotuner.py` SHA-256 `adaaabe4...692`).

| Concern | Exact source | Contract to mirror |
|---|---|---|
| vLLM enable/dispatch | `vllm/config/vllm.py:193-275`, `vllm/config/kernel.py:165-179`, `vllm/model_executor/warmup/kernel_warmup.py:88-95` | O1/O2/O3 enable FlashInfer autotune; supported Hopper/Blackwell runners enter the warmup path |
| persistent leader lifecycle | `vllm/model_executor/warmup/kernel_warmup.py:133-213` | single-rank uses persistence, resolves the cache, loads existing configs before the maximum-token dummy run, writes/broadcasts results, then installs the final map |
| path/config identity | `vllm/model_executor/warmup/flashinfer_autotune_cache.py:19-41` | place caches under a version/architecture/config-hash namespace; allow an explicit root override |
| atomic publication | `vllm/model_executor/warmup/flashinfer_autotune_cache.py:44-55` | write a same-directory temporary file and replace the destination atomically |
| lookup priority | installed `flashinfer/autotuner.py:978-1013` | in-memory live result first, then user-loaded file result before any bundled/default or profiling path; a hit never re-profiles |
| profiling method | installed `flashinfer/autotuner.py:799-818,1407-1493` | three warmups, synchronize, 5,000-us eager stream delay, ten event-timed repeats; the current v0.25 delay is not the v0.24 1,000-us value |
| selected-plan publication | installed `flashinfer/autotuner.py:1343-1396` | publish the minimum valid runner/tactic for a collision-complete profile key and use it on the same invocation |
| merge/save | installed `flashinfer/autotuner.py:1745-1811` | merge prior valid entries, keep current-process results authoritative, sort output, preserve metadata and atomically replace |
| stale rejection/load | installed `flashinfer/autotuner.py:150-196,1813-1900` | reject an environment-mismatched cache as a unit; malformed JSON fails closed; install only known runner/tactic values |
| tests as executable spec | no direct cache unit module exists under vLLM v0.25.0 `tests/`; quant tests are `tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py` | port the source contracts above as local CPU/cache tests and retain the existing upstream-derived FP4 operator/model tests |

Exact tag-source SHA-256 values for the four vLLM control files are:
`kernel_warmup.py=e5918ce5...2c17`,
`flashinfer_autotune_cache.py=fcdaf113...b7b7`,
`vllm.py=caf6db4d...b05`, and `kernel.py=421cec43...d32`.

## Pre-C2 local baseline and gaps

| Local concern | Current anchor | Gap W3-C closes |
|---|---|---|
| complete in-memory key | `src/vt/cuda/nvfp4_plan_cache.h:96-132` | key exists for M bucket, N/K, device/SM, output dtype and tactic-set version, but has no serialization or explicit scale-layout/runner ABI metadata |
| single-flight cache | `src/vt/cuda/nvfp4_plan_cache.h:134-260` | ready plans are process-local only and cannot be enumerated/imported atomically |
| tactic ABI | `src/vt/cuda/nvfp4_tactic_ids.h:15-70` | stable 0--31 descriptors exist, but no content hash/version metadata is written to disk |
| tuning/selection | `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:293-469` | every fresh process profiles the raw minimum; delay is still 1,000 us; no loaded-file priority |
| pre-serve coverage | `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:472-509`, `src/vllm/entrypoints/model_loader.cpp:211-260` | all 64 current profiles prewarm, but every process independently chooses them |
| diagnostics | `include/vt/cuda/nvfp4_autotune.h:14-42` | requested/tuned/lazy counters exist; loaded/rejected/saved/cache-path/mode evidence is absent |
| tests | `tests/vt/test_ops_nvfp4_fp4.cpp:92-170,283-314,882-922,1457-1513` | bucket/key/tactic/capture tests exist; no round-trip, corruption, stale-metadata, atomicity or fresh-process stability tests |

No Python or PyTorch enters the runtime. `third_party/nlohmann/json.hpp` is
already a project dependency and is the implementation surface for the cache
document.

## W3-C1 implementation checkpoint (2026-07-13)

The CUDA-free document layer is now implemented at
`src/vt/cuda/nvfp4_persistent_cache.{h,cpp}`. It provides the complete native
metadata/plan schema, deterministic tactic-descriptor digest, environment/path
resolution, strict native parse/round-trip, current-wins compatible merge,
same-directory `mkstemp` + fsync + atomic rename, and a bounded parser for the
exact FlashInfer 0.6.13 Python-tuple key. It validates M/N/K relationships,
hybrid buckets, runner/tactic IDs, layout/dtype/device/tactic ABI and every
FlashInfer metadata field; wildcard metadata follows upstream import semantics.
`VT_FP4_PERSISTENT_CACHE`, `VT_FP4_AUTOTUNE_CACHE_PATH`,
`VT_FP4_FLASHINFER_CACHE_PATH`, `VT_FP4_AUTOTUNE_CACHE_READONLY` and
`VT_FP4_AUTOTUNE_DELAY_US` are parsed now, but no engine/CUDA caller consumes
the document until C2.

The immutable 64-plan oracle document and provenance manifest live under
`tests/fixtures/nvfp4_flashinfer_v025_gb10/`. The DGX source file is 11,947
bytes/SHA `b41a8ecc...677`; the repository text copy adds only the required
final LF and is 11,948 bytes/SHA `e81e9181...7edd`. The test asserts all 64
recorded `(M,N,K)->tactic` values, not only count/shape samples.

Focused Release, ASan+UBSan and TSan each pass **6/6 cases, 174/174
assertions**. The first raw TSan invocation failed before tests with its known
`unexpected memory mapping`; the exact rerun under
`setarch $(uname -m) -R` passes, so no race finding exists. A first full CPU
suite attempt reported **101/103** because the cache fixture was incorrectly
placed under `tests/parity/goldens`, whose generic scanner treated its manifest
as an op manifest; after moving it to `tests/fixtures`, reconfiguring and
rebuilding, both failed tests pass **2/2** and a fresh full suite passes
**103/103**. This was a fixture-discovery failure, not a cache-parser failure,
and is retained as failed-attempt evidence.

C1 has no runtime behavior, plan-cache publication, model load, GPU command or
performance result. C2 must still import ready plans into the single-flight
map, switch miss timing to the resolved 5,000-us default, add lifecycle/stats,
load before warmup and save only after `Complete()`.

## W3-C2 runtime integration checkpoint (2026-07-13)

C2 now closes the runtime half of the accepted port map. The generic
single-flight cache has checked ready insertion and a deterministic ready-only
snapshot at `src/vt/cuda/nvfp4_plan_cache.h:147-208`. The CUDA dispatcher builds
real runtime/driver/GPU/CUTLASS/tactic metadata, imports FlashInfer before
native plans, translates tactic IDs into executable candidate/workspace plans,
publishes them before the dummy request, applies the resolved 5,000-us default
to misses, rejects a frozen miss before any tuning, snapshots/saves only after
a successful warmup and exposes mode/path/fingerprint/loaded/rejected/saved plus
the complete selected-plan map. CMake hashes the exact tactic/dispatcher ABI
sources so an implementation edit invalidates the default namespace even in a
source archive or dirty build. `LoadedEngine` passes its real device ordinal at
`src/vllm/entrypoints/model_loader.cpp:243`.

CPU source-contract coverage is now **7/7 cases, 189/189 assertions** in each
of Release, ASan+UBSan and TSan (TSan under the already-recorded ASLR-disabled
launcher). CUDA 13.0.88/sm_121a/CUTLASS 4.5 compiles cleanly. The existing
focused CUDA binary passes **20/20, 26,874/26,874**; the new frozen-import case
passes **20/20** with 64/64 loaded, zero tuning, capture/replay and a fail-closed
uncovered shape; the cancelled/completed lifecycle case passes **14/14** and
proves no file before `Complete()` versus five atomically saved plans after it.
The loaded-map compute-sanitizer rerun reports **0 errors**. Its first command
was command-invalid because non-interactive SSH did not put
`compute-sanitizer` on `PATH`; the exact absolute-path rerun is the result.

The real 27B default and `VT_FP4_DIRECT_SF=0` gates each pass **235/235
assertions + 16/16 vLLM tokens** under one immutable read-only fixture. Both
logs report the same metadata fingerprint `330e0b811014741f`, the same 64/64
selected IDs and **0 tuned / 0 rejected / 0 saved**. Staging evidence is
`~/work/vllm.cpp-nvfp4-persistent-c2-staging/evidence`; default/fallback/runtime/
save log SHA-256 values are `a29ebbee...cf2`, `dcdc0bcd...342`,
`a5f65a6e...9c68` and `35982886...f02`.

Two complete current CPU attempts are **FAILED at 102/103** only because the
unchanged `test_capi` early-stop case intermittently observes one callback
delta rather than two. The old baseline binary reproduces the same failure
after five isolated passes; the C2-relevant implementation diff does not touch
C API/engine/executor sources. The remaining suite passes **102/102**, and the
current C-API test passes an isolated rerun. This is recorded as a pre-existing
flake, not rewritten into 103/103 and not attributed to W3-C2.

C2 changes no accepted performance ratio: immutable `3f256ab` remains 55/124.
W3-C3 ran from pushed `d211b8f`; its result is recorded below. No 35B
performance run is authorized.

## W3-C3 immutable stability and invalid-predicate checkpoint (2026-07-13)

Clean detached `d211b8f80fff831a712f0bfafa4f65f1abe1892d` was configured on
GB10 with CUDA 13.0.88, sm_121a, CUTLASS 4.5 from the vLLM 0.25 FlashInfer
environment, vendored Triton AOT and FA2. The focused persistent-runtime smoke
passes. One read/write 27B seed imports the checked-in oracle fixture, passes
235/235 plus 16/16, and publishes native cache SHA
`2590fc94e0d7f1dc4a59c968b1944c1b8249178cf10a56428b2ab7602653199d`.

Under one `/tmp/gpu` lock, six fresh native-only read-only processes run in
alternating direct/`VT_FP4_DIRECT_SF=0` order. All six logs are byte-identical
(SHA `523c2478...95f2` each); each loads 64/64 from native and zero from
FlashInfer, tunes/rejects/saves 0/0/0, records no lazy miss, selects map SHA
`f2d9be7f...1fa4`, and passes **235/235 assertions + 16/16 oracle tokens**.
This closes the fresh-process plan-stability part of G2.

The strict c2/c16 driver then runs both frozen model gates successfully and
completes its first c2 direct/fallback pair. Both arms use the exact same 64
selected lines, complete 6/6 requests with 6,144 input and 768 output tokens,
and report zero tuning/misses. Their long generated texts nevertheless match
for only **2/6** requests; zero-based request indices 0--3 differ. Direct and
fallback raw SHA are `42047046...842f` / `5bb0e70a...7296`, while compact
generated-text SHA are `900f6134...ec7` / `b34a17e8...437`.

The then-current G2/G4 predicate treated that as a correctness failure, so the
driver correctly stopped, cleaned its active process group, and verified GPU,
lock and port 8001 idle/free. All partial or unpaired timing/memory values
remain **VOID**; c16, G3 tracing, G5 exact vLLM grid and every 35B performance
command did not run. Evidence is
`~/work/vllm.cpp-nvfp4-persistent/d211b8f80fff831a712f0bfafa4f65f1abe1892d/evidence`;
failure-summary/driver/driver-log SHA are `01c9faf6...bd74`,
`fd76dd8b...b886` and `67307adc...b5b`.

The cache implementation is stable. W3-C3R below resolves whether the stopped
pair exposed a direct/composed numerical defect or an invalid production-mode
equivalence boundary. The failed predicate remains part of the historical
record; none of its partial performance evidence is rehabilitated.

## W3-C3R batch-shape localization and gate correction (2026-07-13)

Two lock-held diagnostics from the same immutable `d211b8f` build hold the
input corpus, frozen 64-plan map, sampling, token counts and model constant
while varying only scheduler batch shape:

1. Six requests sent strictly sequentially through direct and
   `VT_FP4_DIRECT_SF=0` servers are **6/6 equal for all 128 output tokens**.
   Each server loads the same 64 native plans, tunes/misses 0/0 and retains the
   fixed model-gate preconditions. Comparison SHA is `42d74898...cf41`.
2. Each arm's earlier c2 output is **0/6 equal** to its own sequential output.
   Therefore the independent A/B text mismatch is batch-shape dependent, not
   a direct-scale producer-byte boundary.
3. The exact vLLM v0.25.0 production server, with prefix caching off and
   `VLLM_BATCH_INVARIANT` explicitly unset, is likewise **0/6 equal** for the
   same requests run sequentially versus at c2; the first divergence occurs at
   output token 0--7. Comparison SHA is `cb717bb2...c597`.

This executed result matches upstream's explicit contract. `vllm/envs.py:89,
576-578` defaults `VLLM_BATCH_INVARIANT` off. The determinism fixture enables
it only for that suite (`tests/v1/determinism/conftest.py:9-12`); the NVFP4
operator test requires a fresh opt-in process and compares full-M rows with
M=1 (`test_nvfp4_batch_invariant_scaled_mm.py`), while the e2e test compares a
prompt alone with the same prompt inside a batch only under that fixture
(`test_nvfp4_batch_invariant.py`). On SM12 the opt-in path also selects a
dedicated persistent-scheduler GEMM configuration
(`nvfp4_scaled_mm_sm120_kernels.cu:212-220`). It is a distinct execution mode,
not a property of the production performance denominator.

Accordingly the old **6/6 exact texts across two independently scheduled
production runs** predicate is reclassified as invalid and removed. This does
not relax model correctness: G2/G4 retain byte-exact direct/composed producer
and fixed-tactic GEMM tests; both 235/235 + fixed vLLM 16/16 model gates; the
new controlled **6/6 x 128-token same-shape sequential** direct/fallback proof;
exact request/input/output counts; frozen 64/64 identical plans; zero tuning /
misses; lifecycle; and every timing/memory axis. Per-leg online output hashes
remain diagnostic, because changing implementation latency can legitimately
change production scheduler batch composition on both vLLM and vllm.cpp.

The combined evidence summary is
`~/work/vllm.cpp-nvfp4-persistent/d211b8f80fff831a712f0bfafa4f65f1abe1892d/evidence/c3r-reclassification-summary.json`
(SHA `a1c500b3...41de`). This checkpoint is correctness-only: all timings and
memory observations are **NOT APPLICABLE**, immutable `3f256ab` remains
55/124, and no 35B performance command ran. vLLM's opt-in batch-invariant mode
is inventoried separately as unsupported parity breadth; it is not enabled for
the production speed gate.

## Cache schema, identity and lifecycle

### Native schema

The default file is JSON with a top-level `_metadata` object and sorted `plans`
array. Metadata is mandatory and includes:

- format/version (`vllm.cpp_nvfp4_autotune_v1`);
- CUDA runtime/driver versions, CUTLASS version, device name, device ordinal
  and encoded SM;
- output dtype, FP4/scale layout ABI, tactic-set version and a deterministic
  digest of every 0--31 tactic descriptor;
- timing ABI (warmups=3, repeats=10, delay=5,000 us) and hybrid-bucket version;
- build/source compatibility identity needed to reject reordered or changed
  kernels.

Each plan serializes the complete executable identity: hybrid M bucket, N, K,
device/SM, output dtype, FP4/scale layout, tactic-set version, runner name and
tactic ID. Unknown fields may be ignored for forward compatibility; missing or
wrong mandatory metadata, duplicate/conflicting keys, out-of-range tactic IDs,
unknown runners, malformed JSON and non-regular paths reject the whole file.
There is no best-effort partial import after a validation error.

The local schema may aggregate collision-complete shapes across model configs,
unlike vLLM's outer `aot_compile_hash_factors` directory. This is a recorded
pure-C++ deviation: the contents, not a Python `VllmConfig` hash, own collision
safety. The environment/tactic directory namespace still prevents unrelated
ABIs from sharing a default file.

### FlashInfer v0.25 import seam

An explicit import path accepts the exact FlashInfer 0.6.13 JSON shape used by
the binding oracle. Only `_metadata` plus `fp4_gemm` /
`CutlassFp4GemmRunner` entries are accepted. The bounded tuple-key parser must
validate every input/output shape, derive M/N/K without heuristics, require the
empty cache-key extras used by this runner, and map only tactic IDs whose local
descriptor equals the upstream order. Foreign/custom-op entries are ignored;
malformed FP4 entries reject the import. Import never modifies the source file.

The binding 64-entry cache becomes a checked-in test fixture with its original
SHA and metadata. A native conversion must yield the exact oracle tactic map,
not a hand-selected subset. This gives the DGX parity gate the same plans that
vLLM actually executed while the normal standalone runtime can populate its
own native cache without installing vLLM.

### Path and modes

1. Normal default: read/write persistence under
   `${XDG_CACHE_HOME:-$HOME/.cache}/vllm.cpp/nvfp4_autotune/<format>/<environment>/autotune_configs.json`.
   If neither cache root is usable, warn once and retain correct process-local
   tuning; do not make library construction depend on a writable home.
2. `VT_FP4_AUTOTUNE_CACHE_PATH=<file>` overrides the native cache file.
3. `VT_FP4_FLASHINFER_CACHE_PATH=<file>` imports a read-only FlashInfer file
   before the native map. An exact key hit has priority over native/live tuning.
4. `VT_FP4_AUTOTUNE_CACHE_READONLY=1` is the benchmark freeze mode: every
   pre-serve profile must be present and valid; a miss fails before server
   readiness, performs no tuning and writes nothing.
5. `VT_FP4_PERSISTENT_CACHE=0` restores W3-B process-local behavior in the same
   binary. `VT_FP4_PLAN_CACHE=0` remains the stronger existing bypass.
6. `VT_FP4_AUTOTUNE_DELAY_US` defaults to 5,000 for a cache miss; `0` retains
   the diagnostic no-delay arm. Invalid/overflow values fail configuration.

### Load, tune and publish order

1. Resolve mode/path and validate/import caches once before the maximum-token
   dummy request.
2. Install every valid ready plan into the same single-flight map used by graph
   replay. A loaded hit never calls CUDA timing.
3. Enumerate the declared hybrid profile set. In read/write mode, tune only
   missing keys with the v0.25 3/5,000-us/10 method; in read-only mode, fail on
   the first missing key.
4. After a successful full warmup, snapshot ready entries, merge them with the
   prior valid native document and write one sorted same-directory temporary
   file followed by atomic rename. A failed/cancelled warmup publishes no file.
5. Report path, mode, loaded/tuned/rejected/saved counts, metadata fingerprint,
   and all selected stable IDs. Do not log per-request cache paths or secrets.

TP>1 is explicitly not claimed by W3-C. vLLM v0.25 disables persistence when
world size exceeds one (`kernel_warmup.py:151-163`); the local TP1 gate mirrors
the executed behavior and leaves future rank broadcast to the distributed row.

## Port map and exact files

| Upstream/dependency concern | Local file | Adaptation |
|---|---|---|
| persistent cache schema/load/save | new `src/vt/cuda/nvfp4_persistent_cache.{h,cpp}` | JSON validation, native round-trip, FlashInfer import, metadata, atomic temp+rename, path/mode resolution; CUDA-free for CPU tests |
| ready-map import/snapshot | `src/vt/cuda/nvfp4_plan_cache.h` | checked ready insertion and deterministic snapshot without exposing tuning entries |
| timing and selection | `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu` | loaded-plan priority, exact 5,000-us default, import/save hooks, fail-closed frozen misses |
| warmup lifecycle/stats | `include/vt/cuda/nvfp4_autotune.h`, `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu` | begin/load/complete/save boundary plus loaded/rejected/saved counters and selected-plan dump |
| engine startup | `src/vllm/entrypoints/model_loader.cpp` | resolve/load before the dummy request and publish only after `Complete()` |
| build | `CMakeLists.txt`, `tests/CMakeLists.txt` | compile the CUDA-free cache helper into CUDA builds and focused CPU/CUDA tests |
| operator/cache tests | `tests/vt/test_ops_nvfp4_fp4.cpp` and, if size requires, new `tests/vt/test_nvfp4_persistent_cache.cpp` | port every source contract listed below |
| oracle fixture | new `tests/fixtures/nvfp4_flashinfer_v025_gb10/` | immutable metadata/64-plan fixture plus manifest with source/cache SHA; kept outside parity-golden discovery because it is a cache document, not an op manifest |
| DGX harness | `scripts/dgx-online-serving.sh`, `tools/bench/online_gate_summary.py` only if needed | require frozen 64/64 map and compare selected IDs/hashes before accepting an A/B |

No scheduler, KV cache, attention, GDN, HTTP, model-loader format, 35B W4A16
compute path, or non-CUDA backend behavior changes.

## Tests to port with the code

vLLM v0.25 has no direct unit module for this new helper, so its source and the
installed dependency behavior are the executable contract. The local port must
check:

| Source contract | Required local test |
|---|---|
| `flashinfer_autotune_cache.py:19-41` | deterministic default/override path; environment/tactic namespace changes when any compatibility input changes |
| `flashinfer_autotune_cache.py:44-55` | same-directory temp file, atomic replacement, old complete file or new complete file only, no retained temp after success/failure |
| `autotuner.py:978-1013` | loaded hit wins and executes zero tune callbacks; missing key tunes once through the existing single-flight contract |
| `autotuner.py:150-196,1813-1900` | accept exact and wildcard-compatible fixture metadata where declared; reject wrong GPU/SM/CUDA/CUTLASS/tactic digest/dtype/layout, malformed JSON, duplicate key and invalid tactic/runner |
| `autotuner.py:1745-1811` | round-trip and merge preserve prior compatible keys, current entries win, output is deterministic and sorted |
| `kernel_warmup.py:133-213` | load-before-warmup, full-profile completeness, save only after successful completion, read-only miss fails before readiness |
| binding FlashInfer file | exact 64-entry import produces the recorded `(M,N,K)->tactic` map and rejects a one-byte/malformed FP4 mutation |
| existing NVFP4 tests | every forced tactic remains reference/capture-correct; direct/composed bytes and GEMM output remain equal under a fixed loaded tactic |

No upstream case is silently dropped. Unsupported distributed broadcast is
checked in as an explicit skipped/future distributed-row case.

## Gates

### G1 — CPU/schema/build

- warning-as-error CPU and CUDA 13.0.88/sm_121a builds pass without Python;
- native round-trip, deterministic serialization, merge, stale/corrupt
  rejection, read-only miss, atomic replacement and all imported 64 oracle
  entries pass under normal and sanitizer builds;
- existing key/single-flight/bucket/tactic tests stay green; document/checker
  mutation suites pass.

### G2 — CUDA/operator/model correctness

- cache miss uses exactly three warmups, 5,000-us eager delay and ten timed
  repeats; cache hit records zero profiles tuned and launches no delay/timing
  sequence;
- seed one native cache, then run at least six fresh CUDA processes: every
  process loads the same 64/64 IDs, reports zero tuning/lazy misses, and
  captures/replays without a miss;
- default and `VT_FP4_DIRECT_SF=0` 27B gates each pass 235/235 plus the fixed
  16/16 oracle token stream with one identical frozen map;
- direct and fallback must be 6/6 equal for all 128 output tokens in the
  controlled same-shape sequential diagnostic. Exact text equality across
  independently scheduled production runs is diagnostic only, matching
  vLLM's default-off batch-invariance contract. The 35B correctness-only
  inertness gate may run; no 35B performance run is authorized.

### G3 — exact executed-oracle plan parity

- native import of the binding vLLM cache matches all 64 IDs and its manifest;
- a pre-serve diagnostic from ours shows the same four N/K shapes, 16 M
  buckets and selected IDs as vLLM, with no extra/missing key;
- node tracing retains 208 FP4 GEMMs/forward, zero eager/graph tactic misses and
  unchanged FP4 kernel family. Cache parsing/writing occurs only at startup and
  never enters graph replay.

### G4 — same-plan component A/B

Under one uncontended `/tmp/gpu` lock, rerun direct versus
`VT_FP4_DIRECT_SF=0` at c2/c16 in AB/BA/AB order with the same immutable
read-only oracle-derived map. Before the timed series, retain the controlled
same-shape 6/6 x 128-token sequential direct/fallback proof plus both fixed
235/235 + vLLM 16/16 model gates. Timed legs require exact request/input/output
counts, all lifecycle/memory returns, 64/64 identical selected IDs, zero
tuning/misses, and no regression on all 40 timing + 8 memory axes. Cross-leg
generated-text hashes are retained as diagnostics, not compared as an exact
predicate, because neither production-default vLLM nor vllm.cpp is
batch-invariant. Report W3-C separately from W3-E: cache correctness can pass
even if direct scales still fail the component.

### G5 — binding vLLM v0.25 grid

Only after G1--G4 pass, run the exact 27B c1/2/4/8/16/32 three-repetition
cache-off grid and paired node trace. vLLM must load the immutable cache named
above; ours must import the same 64-plan map. All 124 throughput/latency/memory
axes remain the acceptance rule. Do not run 35B performance until every 27B
axis closes.

## Reproduction checkpoint

Read-only evidence inspection (no model/GPU execution):

```sh
ssh dgx.casa 'LOG=~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27/vllm-profile.log; CACHE=~/.cache/vllm/flashinfer_autotune_cache/0.6.13/121a/cbd38fe31b19a593fd4ac474a8a138a545227805f23c425119de8429f384d163/autotune_configs.json; grep -nE "FlashInfer autotune cache|Config cache hit|Loaded [0-9]+ configs" "$LOG"; sha256sum "$LOG" "$CACHE"; stat -c "%s bytes" "$CACHE"'
```

Expected log lines are 51--57; expected cache SHA/size are
`b41a8ecc18d69174b20263ee85be097d2dd5cf717efd65a03eb466c0c99cf677`
and 11,947 bytes.

## Dependencies, risks and decisions

- Depends on the already-implemented W2 32-tactic ABI, W3-B all-profile
  prewarm and W3-D packed QKV topology. W3-E remains implemented/`GATING`.
- Hardware: GB10/sm_121 for CUDA/model/trace/performance gates; schema and
  mutation tests are CPU-only.
- Data: local cache contains only shapes, environment metadata and tactic IDs;
  no prompts, weights or user data.
- Risk: a persisted bad local timing winner becomes stable. Mitigation: stale
  rejection, explicit reset by path replacement, frozen oracle import for the
  parity gate, and no claim that persistence proves optimality.
- Risk: tactic IDs change after a CUTLASS/source edit. Mitigation: descriptor
  digest plus tactic-set/build metadata rejects the old file as a unit.
- Risk: crash/parallel startup corrupts the file. Mitigation: complete
  validation, same-directory atomic replace and deterministic merge. A future
  distributed row owns rank broadcast; TP1 owns this implementation.
- Decision: W3-C is mandatory under v0.25 actual execution. It is not itself a
  speed credit, and no unrelated speed lever is stacked into its component.

## Work breakdown

| Work | Deliverable | State |
|---|---|---|
| W3-C0 | whole-chain v0.25/dependency/runtime audit, exact cache fixture contract, files/tests/gates | **complete in this spike** |
| W3-C1 | CUDA-free native JSON schema, metadata/path/modes, atomic load/save/merge, FlashInfer importer and CPU/sanitizer tests | **complete: Release/ASan+UBSan/TSan 6/6 + 174/174; full CPU 103/103** |
| W3-C2 | ready-map import/snapshot, 5,000-us timing parity, warmup lifecycle/stats and model-loader integration | **complete: CPU/sanitizers + CUDA runtime/save/memcheck + both frozen 27B arms pass** |
| W3-C3 | fresh-process 64/64 stability, direct/fallback correctness, read-only same-plan c2/c16 component | **PARTIAL:** six-process stability passes; first component was stopped by the now-invalid cross-run exact-text predicate and all partial performance remains void |
| W3-C3R | fixed-plan direct/fallback localization at request/token and batch-shape boundaries; repair or evidence-backed reclassification, then complete G4 rerun | **complete:** sequential arms are 6/6 x 128 equal; each batched arm is 0/6 equal to itself sequentially; production-default vLLM is also 0/6, so cross-run equality is reclassified and the corrected G4 is ready |
| W3-C4 | conditional exact v0.25 27B grid/trace and lifecycle classification | blocked on C3 acceptance |

Each implementation/gate result updates README, BENCHMARKS, the roadmap and
owning matrices in the same checkpoint. A failed component stops before the
exact grid and cannot authorize 35B performance.
