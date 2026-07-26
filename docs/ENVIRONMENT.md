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
| `VT_LMCACHE_HOST` | `127.0.0.1` | Default LMCache server host for the `lm://` connector. The `kv_connector_extra_config.host` key overrides it. See [KV-OFFLOAD.md](KV-OFFLOAD.md) |
| `VT_LMCACHE_PORT` | `65432` | Default LMCache server port. The `kv_connector_extra_config.port` key overrides it |
| `VT_LMCACHE_HASH_ALGO` | `blake3` | Default LMCache key-derivation algorithm. Set `vllm` (alias `sha256_cbor`) for byte-for-byte interop with a real vLLM + LMCache peer. The `kv_connector_extra_config.hash_algo` key overrides it |
| `VT_VULKAN_DEVICE` | first suitable device | Forces the Vulkan physical device index. Required on a multi-GPU host to pin the intended device |
| `VT_KV_CACHE_F32` | off (native KV dtype) | Forces the KV cache to fp32. A precision/diagnostic lever, at the cost of double the KV memory |

## GGUF loading

Behavior of the GGUF weight loader (CPU path). See also `VT_GGUF_KEEP_QUANT` in
the [README Quantization table](../README.md#quantization).

| Variable | Default | What it does |
|---|---|---|
| `VT_GGUF_KEEP_QUANT` | on when compute-in-quant is available | Keep GGUF weights compressed from file to matmul on CPU (no BF16 expansion), byte-identical to the reference path. `0` disables it and expands to BF16 |
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
| `VLLM_CPP_CUDAGRAPH` | on (CUDA) | Eager launches instead of a captured CUDA graph |
| `VLLM_CPP_DENSE_DECODE_GRAPH` | on (CUDA dense) | Non-graphed dense decode |
| `VT_DEVICE_KV_CACHE` | on (CUDA) | Host-side KV cache instead of the on-device one |
| `VT_GPU_SAMPLE` | on (CUDA) | Host-side sampling instead of on-GPU sampling |
| `VT_GDN_PACKED_DECODE` | on (CUDA GDN) | Unpacked GDN decode path |
| `VT_CONV_REG` | on (CUDA GDN) | The non-register-tiled short causal convolution |
| `VT_FA2_PREFILL` | on (CUDA) | The portable prefill attention instead of the vendored FA2 |
| `VT_FA2_DECODE` | on (CUDA) | The portable decode attention instead of the vendored FA2 |
| `VT_FA2_DECODE_4B` | on (CUDA, Qwen3.5-4B) | The portable paged decode attention instead of the ratio-4 vendored FA2 path; the 27B and 35B selectors are unchanged |
| `VT_CPU_REF` | off | Set on to force the portable reference path (dequantize-everything oracle), the standard "is this a kernel bug?" bisect switch |

## Diagnostic

Read-only observability; none change output.

| Variable | Default | What it does |
|---|---|---|
| `VT_OP_PROVIDER_STATS` | off | Print per-op provider (which backend served each op) statistics |
| `VT_OP_PROVIDER_DISABLE` | (none) | Comma-separated provider names to disable, forcing fallback (diagnostic) |
| `VT_GDN_VALIDATE` | off | Run the GDN validation/cross-check path (slower; for kernel debugging) |
| `VT_FP4_AUTOTUNE_VERBOSE` | off | Log the NVFP4 GEMM autotuner's tactic selection |

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
