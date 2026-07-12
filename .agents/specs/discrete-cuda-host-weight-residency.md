# Discrete-CUDA host-weight residency

**Row:** `ENG-HOST-WEIGHT-RESIDENCY`  
**State:** `ACTIVE`
**Claim:** `CLAIM-HOST-WEIGHT-RESIDENCY-1`

## Scope

Close the ordinary model-loading host-RAM gap between vllm.cpp and vLLM on a
discrete CUDA GPU without changing model math, quantization, or CPU/UMA
semantics. The first implementation target is the plain-BF16
`Qwen3_5ForCausalLM` path used by the local 4B diagnostic.

In scope:

- represent CUDA-resident `OwnedTensor` values as logically present after their
  host staging bytes are released;
- release host staging only after all weights needed by a successful forward
  are resident and the upload queue is synchronized;
- preserve an opt-out for same-binary A/B and retain host bytes on CPU, UMA,
  borrowed-test-weight, and quantized-model paths until separately gated;
- remove the tied-embedding copy and packed-source copies where the fallback can
  consume the same canonical storage;
- measure steady-state process PSS separately from launch-to-exit peak PSS;
- follow with a streaming/direct-device loader leaf if peak parity remains open.

Out of scope: CPU weight offload (`ENG-WEIGHT-OFFLOAD`), KV sizing and profiling
(`KV-SIZING`, `KV-WARMUP-PROFILE`), changing CUDA allocator policy, changing the
system/NixOS configuration, or claiming GB10/UMA results from the local RTX.

Dispatch policy: reclamation is default-on only for an engine-owned,
plain-BF16 dense Qwen3.5 model after a successful non-decode CUDA forward on a
backend reporting `UnifiedMemory()==false`. `VT_RELEASE_HOST_WEIGHTS=0` retains
the baseline. Unsupported combinations retain host storage and behavior.

## Upstream chain

| Upstream/dependency anchor | Contract to mirror |
|---|---|
| `vllm/model_executor/model_loader/base_loader.py:43-64` | The model and parameters are constructed on `target_device`, then checkpoint tensors are loaded into that device-owned module. There is no permanent second C++-style host owner. |
| `vllm/model_executor/model_loader/base_loader.py:75-82` | Post-load transformations complete before the model is returned for inference. |
| `vllm/model_executor/model_loader/utils.py:100-126,134-165` | Post-load methods temporarily move CPU-offloaded parameters to the target device and restore only intentionally offloaded values; ordinary device parameters stay on device. |
| `vllm/model_executor/models/qwen3_5.py:279-294,330-344` | Gate/up and GDN B/A checkpoint tensors load as shards of canonical stacked parameters rather than as retained source tensors plus a packed copy. |
| `vllm/model_executor/models/qwen3_5.py:483-492` | `tie_word_embeddings` makes `lm_head` the same object as `embed_tokens`, not a byte copy or second device allocation. |
| `vllm/model_executor/model_loader/weight_utils.py:820-855` | Safetensors iteration yields tensors one at a time, providing the upstream basis for the later peak-memory streaming leaf. |

The local source mirror available on this machine is the production vLLM 0.24
oracle under `.venv-vllm/lib/python3.12/site-packages/vllm`; final oracle runs
record that exact environment. The project parity pin remains `e24d1b24`.

Runtime trace plan: this row changes ownership, not kernel dispatch. Use process
tree PSS/RSS + per-process VRAM sampling and compare kernel-name summaries only
to prove no execution-path drift. Throughput runs are separate and unmonitored
because 10 Hz `smaps_rollup` sampling was measured to perturb eager C++.

## Our baseline

- `include/vllm/model_executor/models/qwen3_5_weights.h:34-61` owns every host
  value in `OwnedTensor::bytes` and separately retains lazy `d_dev`/`d_dev_f32`.
  `Empty()` is currently equivalent to missing host bytes, preventing release.
- `src/vllm/model_executor/models/qwen3_5.cpp:421-472` lazily uploads an
  `OwnedTensor` and then retains both owners for the model lifetime.
- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:217-224,283-299`
  loads separate B/A and gate/up tensors before constructing packed copies.
- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:411-419` copies
  `embed_tokens` into `lm_head` for tied checkpoints.
- Local Qwen3.5-4B baseline: loaded-GPU plateau peak PSS 12.59 GiB vs vLLM
  4.15 GiB (3.04x); launch-to-exit peak PSS 19.77 vs 7.39 GiB (2.68x), while
  process VRAM is effectively equal. Evidence:
  `parity-ledger.md` 2026-07-11 local process-memory row and
  `/tmp/qwen35-memory-20260711-1220`.

## Port map

| Upstream behavior | Local destination | Deviation |
|---|---|---|
| Device-owned parameter lifetime | `OwnedTensor` state in `qwen3_5_weights.h`; release traversal in `qwen3_5_dense_weights.cpp`; first-forward policy in `model_registry.cpp` | The current lazy C++ loader uploads on first use, so release occurs after one synchronized full forward rather than during load. |
| Tied module object | dense weight metadata/lookup in `qwen3_5_weights.h`, `qwen3_5_dense_weights.cpp`, `qwen3_5.cpp` | C++ uses an explicit tied flag/reference selection rather than Python object identity. |
| Stacked parameter loading | dense loader plus packed/split fallback in `qwen3_5_dense_weights.cpp`, `qwen3_5.cpp` | Preserve the same-binary split fallback by slicing canonical packed rows where needed. |
| Streaming checkpoint iterator | follow-on loader work in `safetensors_reader.*` and dense loader | Required for peak parity if mmap + full host materialization remains above vLLM; not hidden by the steady-state leaf. |

## Tests to port

| Upstream test/contract | Local tier |
|---|---|
| `tests/models/test_initialization.py` model initialization and tied-weight lifetime | Extend `tests/vllm/models/test_qwen27_dense_forward.cpp`: tied storage is single-copy and logits remain unchanged. |
| Qwen3.5 stacked mapping at `qwen3_5.py:279-294` | Existing dense-loader order fixtures plus new assertions that canonical packed storage is sufficient for packed and split modes. |
| Device-loading lifetime implied by `base_loader.py:43-82` | Unit-test logical presence and invalid host `View()` after release; CUDA real-model test checks repeat-forward output after reclamation. |
| No upstream process-memory unit | Re-run local 4B process-tree memory series with 1 s sampling for steady state and a separate 100 ms peak-only series; retain raw evidence. |

Quantized 27B/35B release tests are initially **SKIPPED** under this leaf because
the dispatch excludes them; extending release to those representations requires
their own exact traversal and GB10 token/performance/memory gates.

## Gates

1. CPU: full CTest and focused loader/forward tests pass with no changed output.
2. CUDA correctness: local natural greedy remains at least the established
   15/16, byte-identical within each same-binary release ON/OFF A/B; repeated
   forward after release succeeds without upload or host access.
3. Memory: three uncontended repetitions per arm. Steady-state process PSS must
   be no worse than vLLM; peak PSS must improve and remains open unless it is no
   worse than vLLM. VRAM must not regress.
4. Performance: separate unmonitored release ON/OFF runs and a fresh vLLM
   denominator on the identical workload; no material throughput regression.
5. Safety: every process exits, process VRAM returns to zero, and no new
   Xid/UVM/AER/reset/lockup appears. One `flock /tmp/gpu` covers each A/B series.
6. Records: `scripts/check-agent-record.py`, mutation suite, README,
   `docs/BENCHMARKS.md`, matrix, roadmap, ledger, and state agree.

Reproduction starts from the exact command set recorded for
`/tmp/qwen35-memory-20260711-1220`; only sampling interval and the release toggle
change. The implementation checkpoint records full commands and commit IDs.

## Dependencies

- Rows: `MODEL-TEXT-qwen3_5-dense` behavior, existing packed BF16 projection
  work, and the discrete-CUDA backend classification. It does not depend on
  `ENG-WEIGHT-OFFLOAD` or change its contract.
- Toolchain: current Nix CUDA dev shell, native sm_120 Triton-AOT build,
  production vLLM 0.24 oracle, `sample_process_memory.py`, `nvidia-smi`.
- Hardware/data: local RTX 5070 Ti 16 GiB, `Qwen/Qwen3.5-4B`, exact local
  ShareGPT corpus. GB10 gates remain pending and are not inferred.
- License: no copied kernel or third-party implementation; behavior is mirrored
  from the pinned vLLM source under the repository's existing license policy.

## Work breakdown

1. `W1` state/lifetime: distinguish logical presence from host staging,
   traverse plain-BF16 dense weights, synchronize and release after first full
   discrete-CUDA forward, with an opt-out and ownership guards.
2. `W2` duplicates: encode tied embedding identity without a copy; retain only
   canonical packed B/A and gate/up storage while preserving fallback output.
3. `W3` gates: CPU/CUDA correctness, repeat-forward, memory and throughput A/B,
   fresh vLLM denominator, trace-name and kernel-log safety checks.
4. `W4` peak follow-on: the accepted
   [bounded source-page residency spike](safetensors-bounded-page-residency.md)
   first drops consumed file-backed pages at global/layer boundaries, measures
   the remaining owned-model floor, and requires a separately recorded
   direct-device phase if peak PSS still exceeds vLLM.

Checkpoint `6c30657`: W1-W3 are implemented and measured. Stable GPU-resident
PSS is 0.752 GiB versus 4.097 GiB vLLM; peak VRAM is 11,689 versus 12,924 MiB;
prepare-time ON is +1.74% throughput versus OFF. W4 is required because launch
peak remains 15.55 versus 6.69 GiB. The row therefore remains `ACTIVE`.

W4 spike checkpoint: source-chain inspection confirms that vLLM's ordinary
iterator scopes one `safe_open` per shard while this project retains every
mapping through full owned materialization. W4.1-W4.2 are implemented and
CPU/native-sm_120 correctness-gated, but whole-mapping advice is rejected after
one >173.75-s pre-GPU leg. W4.2b narrows advice to consumed tensor ranges before
W4.3; no improved number is accepted.

## Risks and decisions

- An asynchronous H2D transfer followed by early host free is use-after-free.
  The first release synchronizes the model queue before changing ownership.
- `Empty()` is also dispatch metadata. Reclamation must preserve logical
  presence or branches can silently switch implementations.
- Borrowed synthetic weights may be reused by tests/callers; only model-owned
  weights can be reclaimed automatically.
- CPU and GB10 UMA benefit from host-accessible storage and retain it in W1.
- Steady-state and launch peak are independent acceptance axes. Closing the
  former does not permit wording that peak parity is done.
