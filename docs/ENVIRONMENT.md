# Environment variables

vllm.cpp reads a number of environment variables. Most are kernel-internal
tuning and bisect knobs that belong to the engineering record, not to a
deployment; this page documents the subset that changes user-visible behavior or
is a supported operational lever. Everything else is enumerated on the
kernel-internal allowlist (`scripts/env-doc-allowlist.txt`) and kept honest by a
CI check (see [Keeping this reference honest](#keeping-this-reference-honest)).

Unless stated otherwise, a flag-style knob is read as on when its value is a
non-empty, non-`0`/`false`/`off` string, and the listed default applies when the
variable is unset. None of these are required to run the engine.

## Deployment knobs

These change how the engine runs and have no CLI flag (or complement one).

| Variable | Default | What it does |
|---|---|---|
| `VLLM_CPP_CPU_THREADS` | hardware concurrency | Overrides the CPU threadpool width. The single most useful CPU-deployment knob; there is no CLI flag for it |
| `VLLM_PREFIX_CACHING_HASH_SEED` | `0` (fixed) | Seed for the prefix-cache block hash, mirroring vLLM's `PYTHONHASHSEED`. `random` makes block hashes non-deterministic across processes, which takes any persisted or shared KV cache to a 0% hit rate. Keep it fixed if you rely on cross-process prefix reuse |
| `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` | `1` (on) | Whether published KV-cache events carry block hashes as an int (the low 64 bits of the sha256 digest) rather than the raw 32 bytes, mirroring vLLM's env of the same name and its default. Set `0` to publish the raw bytes. Only affects the KV-cache event payload (`--kv-events-config`); it does not change the internal block hashing or the cache itself |
| `VLLM_PLUGINS` | unset (load all registered) | Comma-separated allowlist of general plugins to load in `LoadGeneralPlugins()`, mirroring vLLM's `VLLM_PLUGINS`. Unset loads every registered plugin; an empty string loads none; a list loads only the named plugins. A plugin that throws is logged and skipped (the load never aborts the engine). See [.agents/specs/plugin-system.md](../.agents/specs/plugin-system.md) |
| `VT_LMCACHE_HOST` | `127.0.0.1` | Default LMCache server host for the `lm://` connector. The `kv_connector_extra_config.host` key overrides it. See [KV-OFFLOAD.md](KV-OFFLOAD.md) |
| `VT_LMCACHE_PORT` | `65432` | Default LMCache server port. The `kv_connector_extra_config.port` key overrides it |
| `VT_LMCACHE_HASH_ALGO` | `blake3` | Default LMCache key-derivation algorithm. Set `vllm` (alias `sha256_cbor`) for byte-for-byte interop with a real vLLM + LMCache peer. The `kv_connector_extra_config.hash_algo` key overrides it |
| `VT_VULKAN_DEVICE` | first suitable device | Forces the Vulkan physical device index. Required on a multi-GPU host to pin the intended device |
| `VT_KV_CACHE_F32` | off (native KV dtype) | Forces the KV cache to fp32. A precision/diagnostic lever, at the cost of double the KV memory |
| `VT_ENABLE_JUMP_FORWARD` | off | Opt-in to jump-forward constrained decoding (SGLang parity SW3): when a grammar/structured-output request reaches a state with exactly one valid next token, that token is emitted without a model step. Currently drives only the standalone driver (`DrainForcedTokens`); output-identical by construction (it fires only where the constrained sampler already has a single valid token), so it changes speed, never tokens. Off by default until the production scheduler splice (jumped-token KV recompute) lands. Set `1`/`true`/`on` to enable |

## GGUF loading

Behavior of the GGUF weight loader (CPU path). See also `VT_GGUF_KEEP_QUANT` in
the [quantization format table](BUILD.md#quantization-formats).

| Variable | Default | What it does |
|---|---|---|
| `VT_GGUF_KEEP_QUANT` | on when compute-in-quant is available | Keep GGUF weights compressed from file to matmul on CPU (no BF16 expansion), byte-identical to the reference path. `0` disables it and expands to BF16 |
| `VT_GGUF_NVFP4_FP4` | on where the device can run the NVFP4 GEMM (CUDA; a CPU build expands) | The NVFP4 analog of `VT_GGUF_KEEP_QUANT`: keep an NVFP4 GGUF's weights in native fp4 residency and run `kMatmulNvfp4`, instead of expanding to BF16. `0` is the same-binary opt-out (expand to BF16); forced off under `VT_CPU_REF` so the oracle load stays byte-identical. See [.agents/specs/gguf-nvfp4-native-compute.md](../.agents/specs/gguf-nvfp4-native-compute.md) |
| `VT_GGUF_NVFP4_W4A4` | on (only meaningful when `VT_GGUF_NVFP4_FP4` is on) | Selects which of vLLM's two NVFP4 modes the fp4-resident weights compute in: on = true W4A4 (fp4 activations, using the GGUF's `<stem>.input_scale` sidecars, mirroring the sibling compressed-tensors container); `0` = W4A16 (BF16 activations over the fp4 weights). No effect when the fp4 residency is off |
| `VT_GGUF_KEEP_F16` | on (when weights expand) | Keep F16 GGUF weights in F16 rather than promoting them, an RSS/perf tradeoff |
| `VT_GGUF_MMAP` | on when weights stay quantized | Keep the GGUF file mmap-resident instead of copying weight bytes into owned buffers, trading RSS for page-cache residency |
| `VT_GGUF_PREFAULT` | off | Pre-fault the mmap-resident weight pages at load, trading a slower load for steadier first-token latency |

## Rollback and bisect switches

Default-on fast paths, each with an off switch. They exist so a suspected kernel
bug can be bisected without a rebuild: set the variable to `0` to fall back to the
portable/reference path. In normal operation leave them unset.

| Variable | Default | Off switch falls back to |
|---|---|---|
| `VT_ASYNC_RUNNER` | on | Synchronous model runner (no async/overlap execution) |
| `VT_ASYNC_SCHED` | on | Synchronous scheduling (no scheduler/execution overlap). The documented first-line workaround for a suspected scheduling bug |
| `VT_ASYNC_DEVICE_MIRROR` | off | `=1` engages the discrete-CUDA device-resident sampled-token mirror (ENG-ASYNC-SCHED W4), so the async serving loop's sampled ids stay on the device instead of round-tripping the host. DEFAULT OFF pending a serving A/B, exactly as the W3 device kernels landed: the synchronous `LLMEngine::step()` loop has no overlap for it to unlock, so it can only cost there. Token-identical to the default path either way. No effect on an integrated GPU (which keeps its in-place path) or on CPU |
| `VLLM_CPP_CUDAGRAPH` | on (CUDA) | Eager launches instead of a captured CUDA graph |
| `VLLM_CPP_DENSE_DECODE_GRAPH` | on (CUDA dense) | Non-graphed dense decode |
| `VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH` | off (opt-in) | `=1` routes pure-decode steps for the SHARED pure-dense forward (`Qwen3DenseModel`, i.e. Qwen3 / Llama / InternLM3 / Mistral / InternLM2 `ForCausalLM`) through the captured decode CUDA graph; default OFF keeps the byte-identical eager decode. Token-exact with eager (dgx SACRED near-tie gate, Qwen3-0.6B/4B). Honors `VLLM_CPP_CUDAGRAPH=0` |
| `VT_MM_DECODE_EAGER` | off (graph on) | Set to `1` to force the eager per-step multimodal (Qwen3.6-27B image/video) decode instead of routing it through the captured dense decode graph. Rollback / A-B knob; the graphed path is token-exact with the eager path |
| `VT_KIMI_DEVICE_COMPUTE` | off (opt-in) | `=1` routes the Kimi-Linear-48B-A3B runner path (`KimiLinearModel::ForwardDevice`) through the W7 DBuf-resident device COMPUTE (`ForwardDeviceCompute`, the whole KDA/NoPE-MLA + MoE hybrid over pooled DBufs via the shared vt:: ops) instead of the default W6 host-reference compose. Default OFF keeps the CPU-verified host-ref-compose seam as production until the device compute is GPU-verified against the SACRED oracle; the device compute is CPU-gated (`test_kimi_linear_forward`, device==W2 reference within f32-accumulation tolerance, greedy-token-identical) but its GPU numerics are a NAMED pending. The flag exists so the device path CAN be exercised as the runner path for that verification |
| `VT_WHISPER_ENC_EAGER` | off (flash-tiled attention on) | Set to `1` to force the naive per-key block-reduction attention in the Voxtral/Whisper audio encoder instead of the default flash-tiled kernel. Rollback / A-B knob; token-identical to the default path |
| `VT_WHISPER_ENC_WARP` | off (flash-tiled attention on) | Set to `1` to force the warp-scoped online-softmax attention (`vt::AttentionDenseFast`, the pre-flash default) in the Voxtral/Whisper audio encoder instead of the default flash-tiled kernel (`vt::AttentionDenseFlash`). Rollback / A-B knob; the flash-tiled path is bit-identical to the warp path (encoder self-attention ~1.82x faster) |
| `VT_WHISPER_ENC_REMARSHAL` | off (encoder weights resident) | Set to `1` to disable device-resident encoder weights and re-marshal (host f32->bf16 convert + H2D upload) all Whisper/Voxtral encoder weights on EVERY forward, restoring the pre-residency behavior. Rollback / A-B knob; byte-identical output (moves data only). Default residency uploads each encoder weight once and reuses it, removing ~648 ms of per-call host marshalling from the encoder forward |
| `VT_QWEN3VL_ATTN_WARP` | off (flash-tiled attention on) | Set to `1` to force the warp-scoped online-softmax attention (`vt::AttentionDenseFast`, the pre-flash default) in the Qwen3-VL / Qwen3.6-27B vision tower per-frame self-attention instead of the default flash-tiled kernel (`vt::AttentionDenseFlash`). Rollback / A-B knob; the flash-tiled path is bit-identical to the warp path (tower output token-identical, goldens unchanged) |
| `VT_QWEN3VL_ATTN_EAGER` | off (flash-tiled attention on) | Set to `1` to force the naive per-key block-reduction attention (`vt::Attention`) in the Qwen3-VL / Qwen3.6-27B vision tower instead of the default flash-tiled kernel. Rollback / A-B knob; token-identical to the default path |
| `VT_DEVICE_KV_CACHE` | on (CUDA) | Host-side KV cache instead of the on-device one |
| `VT_GPU_SAMPLE` | on (CUDA) | Host-side sampling instead of on-GPU sampling |
| `VT_GDN_PACKED_DECODE` | on (CUDA GDN) | Unpacked GDN decode path |
| `VT_CONV_REG` | on (CUDA GDN) | The non-register-tiled short causal convolution |
| `VT_FA2_PREFILL` | on (CUDA) | The portable prefill attention instead of the vendored FA2 |
| `VT_FA2_DECODE` | on (CUDA) | The portable decode attention instead of the vendored FA2 |
| `VT_FA2_DECODE_4B` | on (CUDA, Qwen3.5-4B) | The portable paged decode attention instead of the ratio-4 vendored FA2 path; the 27B and 35B selectors are unchanged |
| `VT_CPU_REF` | off | Set on to force the portable reference path (dequantize-everything oracle), the standard "is this a kernel bug?" bisect switch |
| `VT_DFLASH_PAGED` | on (CUDA, DFlash spec-decode) | The materialized `[context;block]` draft forward instead of the fixed-capacity paged draft-KV store read through `vt::DFlashPagedBlockAttention` (bit-identical; only the DFlash single-request propose path) |
| `VT_DFLASH_GRAPH` | on (CUDA, DFlash spec-decode) | The eager paged draft step instead of the captured/replayed draft-step CUDA graph (replayed==eager bit-identical; only the DFlash single-request propose path) |
| `VT_DFLASH_ATTN_BLOCK` | off (CUDA, DFlash spec-decode) | `=1` selects the D12/D13 block-per-(query,head) draft paged-attention kernel instead of the D14 default warp-scoped online-softmax kernel (same f32-softmax math within the bf16 envelope; the D14 warp kernel is ~3x faster and closed the ~2% speed residual; only the DFlash single-request propose path) |

## Diagnostic

Read-only observability; none change output.

| Variable | Default | What it does |
|---|---|---|
| `VT_DFLASH_GRAPH_STATS` | unset | Print DFlash draft-step CUDA-graph capture/replay counts to stderr |
| `VT_OP_PROVIDER_STATS` | off | Print per-op provider (which backend served each op) statistics |
| `VT_OP_PROVIDER_DISABLE` | (none) | Comma-separated provider names to disable, forcing fallback (diagnostic) |
| `VT_GDN_VALIDATE` | off | Run the GDN validation/cross-check path (slower; for kernel debugging) |
| `VT_FP4_AUTOTUNE_VERBOSE` | off | Log the NVFP4 GEMM autotuner's tactic selection |
| `VT_POOL_BYPASS` | off | `=1` makes every device-scratch pool allocation an exact-size driver `Alloc` and every release a real `Free`, so `compute-sanitizer` can see tensor boundaries and use-after-free that the caching, size-class-rounding pool hides. DEBUGGING ONLY: it reinstates the per-op `cudaMalloc`/`cudaFree` device-sync storm the pool exists to remove, so it is never a timing configuration |

## Kernel-internal knobs (deferred)

The remaining variables (roughly 120 `VT_GDN_*`, `VT_FP4_*`, `VT_FP8_*`,
`VT_MOE_*`, `VT_ATTN_*`, `VT_NVFP4_*`, `VT_FUSE*`, and similar) are kernel
implementation and micro-tuning switches. They are not part of the deployment
surface, their meaning is tied to a specific kernel, and they are recorded in the
engineering ledger (`.agents/parity-ledger.md`) and `docs/BENCHMARKS.md` where the
A/B that introduced each one lives. They are enumerated on
`scripts/env-doc-allowlist.txt` so the CI check below can tell a known
kernel-internal knob from a new, undocumented one. If you need to understand one,
grep its name under `src/` for the read site and the ledger for the measurement.

## Keeping this reference honest

`scripts/check-env-doc.py` scans every `VT_*` / `VLLM_*` environment name read
from `src/` and `include/` and fails if any name is neither documented on this
page nor listed on the kernel-internal allowlist
(`scripts/env-doc-allowlist.txt`). A newly-introduced production env var therefore
cannot land silently: the author must either document it here or classify it
kernel-internal on the allowlist. The check runs in CI (the `agent-record` job)
and has a unit/mutation test at `tests/scripts/test_check_env_doc.py`.
