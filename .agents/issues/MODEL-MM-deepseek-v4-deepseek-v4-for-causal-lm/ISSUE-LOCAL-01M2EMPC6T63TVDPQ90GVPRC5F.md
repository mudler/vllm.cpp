ID: ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F
Title: DeepSeek-V4 cannot serve on any backend here: its KV factory publishes 4/8-token groups and every attention backend declares {16}, so a CUDA build dies at engine construction
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-18
Closed: 2026-09-18

## Problem

MEASURED 2026-09-14 (rc job fc593c9f-f550-4f1a-9adf-a8c734f48822, dgx:gpu0, worker rc-worker-m6z8s, NVIDIA GB10, driver 580.173.02, compute_cap 12.1). This is the FIRST CUDA build ever run against the real checkpoint. Base f1dd76c8b680ee8a20f89f4070adf61db3671f32, CUDA arch 121a, fa2 ENABLED for [121a], CUTLASS 4.5.0, Triton AOT sm_121a, nvcc 13.0.88, BUILD_RC=0, 63 .cu.o, CLI_MD5=57339dd3148052705974bda7ad11af83, CMAKE_BUILD_TYPE=Release with CMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG', so NDEBUG IS DEFINED. Artefact verified byte-exact on the worker at /workspace/dsv4-vision/UD-IQ1_S/: 5,305,248 + 49,991,832,128 + 32,441,484,736 = 82,438,622,112 B, plus mmproj-BF16.gguf 934,462,656 B.

THE DEVICE GATE IS OPEN, WHICH IS WHY THIS IS NEWS. PROBE_RC=0: V4DeviceKernelsAvailable() is TRUE on this build, 31 cases / 90193 assertions / 0 failed / 0 skipped. This is the first run in which the four kDeepseekV4{Mhc,Dsa,Compressor,Moe} ops were registered for kCUDA. Every earlier leg was built -DVLLM_CPP_CUDA=OFF and was refused at src/vllm/model_executor/models/deepseek_v4.cpp:4658 (VT_CHECK(deepseek_v4::V4DeviceKernelsAvailable(), kDevicePending), the W7-device refusal), which told us nothing about the model.

THE NEW WALL. RUN_RC=1, WALL_S=886, OUT_BYTES=0, VmHWM_kB=83460776 (79.6 GiB, so the tower really was materialized). Verbatim: 'vllm-cli: loading model from /workspace/dsv4-vision/UD-IQ1_S/DeepSeek-V4-Flash-Vision-Exp-UD-IQ1_S-00001-of-00003.gguf' / 'engine: device placement: --fit places nothing; the model already fits the budget' / 'INFO auto-fit max_model_len: reduced from 1048576 to 65536 to fit the KV cache (256 blocks x 256 tokens). Raise --num-blocks / --kv-cache-memory for a longer context.' / 'vllm-cli: model load failed (status 2): vllm_engine_load: Block size must be a multiple of 16.' Stage reached: file open OK, header parse OK, tensor mapping OK, KV-cache config OK, weight load OK, ENGINE CONSTRUCTION FAILED, first forward NOT REACHED, sampling NOT REACHED.

MECHANISM, every anchor verified at e1e92400c. MakeDeepseekV4KVCache publishes SEVEN KV-cache groups at block sizes {256,256,256,64,4,4,8}: three MLAAttentionSpec groups take the engine block_size (src/vllm/model_executor/models/deepseek_v4_registry.cpp:546, :547, :548), the SWA group takes kSwaBlockSize = 64 (:550, constant defined at :399 citing upstream sparse_swa.py:82), and the three compressor/indexer state groups take 4, 4 and 8 (:552, :554, :556). deepseek_v4_registry.cpp:501 documents that geometry: 'sliding_window = coff * compress_ratio, and block_size 4 for ratio 4 / 8 for ratio 128'. 256 and 64 pass % 16; the 4, 4 and 8 DSA/compressor groups FAIL.

The refusal is reached from GPUModelRunner::initialize_kv_cache (src/vllm/v1/worker/gpu/runner.cpp:701, called from the two constructor BODIES at :528 and :633), which calls vllm::v1::CheckKvCacheShape at runner.cpp:1719 -- inside that same function, since no other GPUModelRunner:: definition appears between :701 and :1800. NOTE THE EXACT SITE, because it is easy to misattribute: CheckKvCacheShape (src/vllm/v1/attention/registry.cpp:150) does NOT itself emit this string; its own throw at registry.cpp:173 is the 'declares a KV cache shape ... that does not match' message. It calls MakeAttentionBackend(device, name)->get_kv_cache_shape(...) at registry.cpp:160, and the string is thrown INSIDE that backend call.

'Block size must be a multiple of 16.' is emitted BYTE-IDENTICALLY at THREE sites: src/vllm/v1/attention/backend.cpp:253 (FlashAttentionBackend::get_kv_cache_shape), :268 (RocmAttentionBackend, not applicable on a CUDA box) and :279 (TritonMLABackend). WHICH ONE FIRED CANNOT BE DETERMINED FROM THE MESSAGE and the log does not name the resolved backend, so this record does not guess. The predicate behind :279 is AttentionBackend::supports_block_size (backend.cpp:153): a block size is supported when it is a multiple of ANY declared size. The declared set is {16} at include/vllm/v1/attention/backend.h:444 (FlashAttentionBackend, class at :427), :546 (RocmAttentionBackend, class at :532) and :604 (TritonMLABackend, class at :575), plus cpu_attn.h:112; the base default is the permissive {1} at backend.h:360. It is a throw std::invalid_argument and NOT an assert, so NDEBUG cannot delete it -- unlike kv_cache_coordinator.cpp:386.

WHY kv_cache_coordinator.cpp:386 IS NOW UNREACHABLE ON CUDA, which retires a standing question in this row's spec. initialize_kv_cache runs in the BODY of the GPUModelRunner constructor, and runner_ is constructed at src/vllm/entrypoints/model_loader.cpp:2255 BEFORE scheduler_ at :2307 in the same initializer list; scheduler_ is what builds KVCacheManager -> HybridKVCacheCoordinator. So this throw fires strictly before the coordinator exists, and the NDEBUG-deleted equality assert at :386 is not what stops a CUDA build. Corroborated empirically: this run never printed 'Asynchronous scheduling is enabled' (emitted at model_loader.cpp:2443), whereas the earlier CPU leg did print it and then died later at deepseek_v4.cpp:4658.

UPSTREAM, AND IT DOES NOT HAVE THIS PROBLEM. Read at the pinned oracle vLLM e126687a9a. Upstream PUBLISHES the same sub-16 block sizes: vllm/models/deepseek_v4/compressor.py:152-167 hard-codes self.block_size = 4 for compress_ratio 4 and 8 for 128, with a comment deriving both from the shared page, and publishes it into a real SlidingWindowMLASpec at compressor.py:173-185; sparse_swa.py:82-87 sets 64; the MLA and indexer caches take cache_config.block_size (deepseek_v4/attention.py:742, :787), which is lifted from DEFAULT_BLOCK_SIZE = 16 (config/cache.py:79) to 256 by get_preferred_block_size (sparse_swa.py:130-132) via platforms/interface.py:628-640. So {256,256,256,64,4,4,8} is upstream's own geometry, not an artefact of our factory.

UPSTREAM HAS NO GLOBAL 16-MULTIPLE RULE TO RECONCILE AGAINST -- that is the actual finding. AttentionBackend.supports_block_size (vllm/v1/attention/backend.py:116-133) is a multiple-of test whose default is [MultipleOf(1)] (backend.py:72-74), i.e. permissive; the 16-multiple checks are PER-BACKEND OVERRIDES in exactly three backends (triton_attn.py:314, triton_mla.py:152, rocm_aiter_unified_attn.py:53). The reconciliation is PER-GROUP BACKEND DISPATCH: the compressor group is served by CompressorBackend (compressor.py:189-190), whose get_supported_kernel_block_sizes() returns [MultipleOf(1)] (compressor.py:66-68), so 4 and 8 pass; DeepseekSparseSWABackend returns [MultipleOf(64)] (sparse_swa.py:126-128); DeepseekV4IndexerBackend returns [256] (vllm/v1/attention/backends/mla/indexer.py:194-196). Upstream NEVER asks a dense 16-multiple backend about the compressor's block size.

THREE ALTERNATIVE MECHANISMS WERE CHECKED AND RULED OUT UPSTREAM, so none of them is available to copy. (a) 'the DSA cache is not a KV-cache group': false, CompressorStateCache is an AttentionLayerBase with get_kv_cache_spec (compressor.py:133, :173). (b) 'a unifying pass forces one block size': there is none; unify_kv_cache_spec_page_size (kv_cache_utils.py:1113-1175) equalizes PAGE SIZE and, when it must move a block size, only RAISES it (:1158-1162), and it is a no-op for DSv4 because the pages already match by construction. The only cross-group requirement is divisibility (kv_cache_coordinator.py:92, gcd at kv_cache_utils.py:1955). (c) 'kernel_block_size adapts it': the mechanism exists (worker/utils.py:442-483, :310-376) but only ever selects a DIVISOR of the manager block size (:371-375), so it can never lift a 4-token group to 16.

CONSEQUENCE: the production path cannot serve this architecture at all on any backend in this tree, because the model's own KV factory publishes block sizes no attention backend here accepts. Closing it means porting per-group backend dispatch so each group's block size is validated against THAT group's own backend, with a CompressorBackend equivalent declaring the permissive MultipleOf(1) and a sparse-SWA equivalent declaring MultipleOf(64). NOT FIXED HERE: this is a records-only change, and the dispatch port is a production change needing its own spec and fresh review.

EVIDENCE CAVEAT ON THE UPSTREAM SIDE: the local vLLM checkout at /home/mudler/_git/vllm has HEAD 5559679229 (the PRIOR pin) and a dirty tree. Every upstream line quoted above was read as a blob AT e126687a9a via git show/git grep, not from that working tree. No network was used.

## Resolution

FIXED 2026-09-18 by porting upstream's PER-GROUP ATTENTION-BACKEND DISPATCH, on
branch `row/MODEL-MM-deepseek-v4-blocksize-dispatch` from base
`bca4ab09fcaa6e718fbc25740816f60be417bddb`. Spec:
`.agents/specs/dsv4-per-group-attn-backend.md`.

WHICH OF THE THREE SITES FIRED, which this record deliberately did not guess and
which is now determined. Every group `MakeDeepseekV4KVCache` publishes is FUSED
-- `MLAAttentionSpec` or `SlidingWindowMLASpec` -- so `runner.cpp:1412` marks all
nine caches MLA and the view loop takes the `is_mla` arm for every one. That arm
resolved the MLA backend ONCE and cached it in `mla_backend_resolved`, so the
first group (256 tokens) resolved `TRITON_MLA` and the 4/4/8 compressor and
indexer groups inherited it. The site is `backend.cpp:279`,
`TritonMLABackend::get_kv_cache_shape`. `backend.cpp:253` (FlashAttention) and
`:268` (ROCm) did not fire, because no group ever reached the dense arm.

WHY NO TEST SAW IT. On a CPU box the same `is_mla` arm resolves NOTHING -- no MLA
backend is registered for `kCPU` -- the name is empty, and `CheckKvCacheShape` is
skipped entirely. `tests/vllm/entrypoints/test_deepseek_v4_multigroup_kv.cpp` was
therefore green on a defect that stops every CUDA load. Measured on this tree at
base: the child engine reported
`BACKENDS=-:256;-:256;-:256;-:64;-:64;-:64;-:4;-:4;-:8`, i.e. nine caches and not
one backend consulted.

WHAT CHANGED. `KVCacheGroupSpec` gained `attn_backend`, upstream's
`AttentionGroupKey.attn_backend` (`gpu_model_runner.py:7170`), whose value comes
from the LAYER at `:7150`. `MakeDeepseekV4KVCache` now names it per group exactly
as each upstream layer's `get_attn_backend()` does: `DEEPSEEK_V4_INDEXER` for the
indexer key cache (`indexer.py:183-196`, `[256]`), `DEEPSEEK_SPARSE_SWA` for the
sliding-window cache (`sparse_swa.py:116-118`, `:126-128`, `[MultipleOf(64)]`)
and `CompressorBackend` for the three compressor states
(`compressor.py:189-190`, `:66-68`, `[MultipleOf(1)]`). The three backends are
new and are host metadata only. `GPUModelRunner::initialize_kv_cache` uses a
named backend directly and keeps its existing lazy dense/MLA resolution for every
unnamed group, so every other topology resolves byte-identically.

THE `% 16` REFUSAL IS UNTOUCHED, and that is asserted rather than described.
`FlashAttentionBackend`, `RocmAttentionBackend` and `TritonMLABackend` still
declare `{16}` and still throw the verbatim `Block size must be a multiple of
16.`; `CheckKvCacheShape(kCUDA, "TRITON_MLA", 256, 4, 1, 2048, true)` reproduces
that exact string in the test. The defect was never that check. It was that the
wrong backend was consulted.

EVIDENCE. `tests/vllm/entrypoints/test_deepseek_v4_multigroup_kv.cpp`, which
enters at `LoadedEngine::FromModelDir` on DEFAULT `EngineParams`, in a Release
(`-O3 -DNDEBUG`) build on this host. RED before the change: `test cases: 5 | 3
passed | 2 failed`, `assertions: 51 | 44 passed | 7 failed`, binary md5
`34bb7ba872dbe6631cb264ce0b2de745`. GREEN after: `test cases: 5 | 5 passed | 0
failed | 1 skipped`, `assertions: 96 | 96 passed | 0 failed`, binary md5
`8976153e512bcc9e22d5576a91b1385b`. The child engine now reports
`BACKENDS=-:256;-:256;DEEPSEEK_V4_INDEXER:256;DEEPSEEK_SPARSE_SWA:64;DEEPSEEK_SPARSE_SWA:64;DEEPSEEK_SPARSE_SWA:64;CompressorBackend:4;CompressorBackend:4;CompressorBackend:8`
and `CONSTRUCT=ok`. The suite still answers identically without `NDEBUG`
(default configure: `5 | 5 passed`, `64 | 64 passed`, md5
`cdd63175f76328d7b593943698ae6dc1`). 67 attention / KV / DeepSeek-V4 / runner
suites run green beside it, 3 skipped for absent devices.

WHAT IS NOT CLOSED BY THIS. The fixture is SYNTHETIC. Only a leased run against
the real 82,438,622,112-byte artifact can show that the load gets past engine
construction there, and no such run was taken for this change. The next wall on
the production path is unchanged and is NOT this issue: the W7-device forward
refusal at `deepseek_v4.cpp:4658`, owned by
`.agents/specs/deepseek-v4-device-decode.md`. The resolved backend name is
RECORDED and not DISPATCHED on (owed to #1332 M4), so a correct name here is not
a working kernel.

CONFIRMED ON THE REAL ARTIFACT 2026-09-18, which closes the "WHAT IS NOT CLOSED
BY THIS" caveat directly above. This issue was ALREADY `CLOSED` when that run was
taken; nothing about its state changes here, and the paragraph is appended rather
than rewritten so the caveat and its answer both stay readable.

rc job `1ec05714-8cbd-41d6-8c45-9396f4d92223`, `thor:gpu0` (NVIDIA Thor,
compute_cap 11.0, driver 595.78, aarch64), head
`7722c8c716f91be2188408d688e0cc5f3268c6f1` re-verified ON THE WORKER by
re-hashing the extracted tree (`git write-tree` ->
`852f827f5bab1ac85b96a436f20e4e39c9ddf464`), `Release` with NDEBUG defined,
`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110`, nvcc 13.0, 41 `.cu.o`,
`fa2: DISABLED`, cli md5 `964ef06d43231e8125673dae0fdfbc3c`. Artifact byte-exact
on the worker at 82,438,622,112 B plus the 934,462,656 B projector.

`vllm-cli --device cuda` exits `0`. The `vllm_engine_load: Block size must be a
multiple of 16.` line is ABSENT, and the run reaches `vllm.cpp: Asynchronous
scheduling is enabled (max_concurrent_batches=2)`, which only prints once the
engine is BUILT. The first three log lines are byte-identical to the 2026-09-13
run, so this is the same load reaching further. The SYNTHETIC-fixture caveat is
therefore answered on the released checkpoint.

ONE PREDICTION IN THE PARAGRAPH ABOVE WAS WRONG, and it is corrected here rather
than edited away: "the next wall on the production path is unchanged and is NOT
this issue: the W7-device forward refusal at `deepseek_v4.cpp:4658`". That wall
did NOT fire. `V4DeviceKernelsAvailable()` was TRUE on this build, so
`kDevicePending` was never reached, and the model generated (` Paris`, first
token only, NO ORACLE, not a parity result). The wall that did fire is on the
`--device cpu` arm at `deepseek_v4.cpp:631`, and it has its own row-owned issue
`ISSUE-LOCAL-01M2TYBE9TWX09XG62QVA5TXHZ`.
