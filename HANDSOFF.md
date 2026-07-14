# Current handoff

Last refreshed: **2026-07-14 21:48 UTC**. This is a replace-in-place session
handoff, not benchmark evidence or an attempt log. Read [AGENTS.md](AGENTS.md)
first, then resume from this file and the linked canonical rows.

## Resume target

- Priority remains roadmap order 0: restore every-axis performance parity
  before broader roadmap work.
- Active claim: `CLAIM-GDN-BA-ROUNDING-1`, rows
  `KERNEL-GDN-PACKED-DECODE` W1D3/G3 and its `KERNEL-GEMM-BF16` consumer.
- Local worktree: `/home/mudler/_git/vllm.cpp-gdn-ba-rounding`, branch
  `codex/gdn-ba-rounding-w1c`.
- Failed execution source: `d82d282f9efd1a5b97e7c6f1ac7a55b949849d09`.
- Binding result remains `3f256ab`: **55/124** axes pass. At c2, TPOT remains
  **114.841 vs 108.274 ms (6.1% slower)**. `benchmark_binding=false`; there is
  no new speed credit.

Do not start qkvz, rerun the exact grid, run 35B performance, or promote any
partial component data. The next task is to expose and root-cause the hidden
c16 server exception, repair test-first, and rerun the complete component from
a new pushed SHA and new SHA-owned root.

## Failed/incomplete DGX series

The repaired production component ran detached on `dgx.casa` and stopped
without a marker-last terminal status:

| Item | Value |
|---|---|
| Root | `/home/mudler/work/vllm.cpp-gdn-packed-component/d82d282f9efd1a5b97e7c6f1ac7a55b949849d09` |
| Evidence | `$ROOT/evidence` |
| Driver | PID `3866199`, now exited |
| Source | clean detached worktree at `d82d282f9efd1a5b97e7c6f1ac7a55b949849d09` |
| Build | production `RelWithDebInfo`, CUDA 13.0.88, sm_121a, CUTLASS, FA2, vendored Triton AOT, profile control off; **154/154** targets built |
| Model | Qwen3.6-27B NVFP4 snapshot `890bdef7a42feba6d83b6e17a03315c694112f2a` |
| Reference | vLLM v0.25.0 / FlashInfer 0.6.13 via `~/venvs/vllm-oracle/bin/vllm` |
| Corpus | byte-identical copy of binding `3f256ab` corpus/27 |
| Completed | packed and rollback **235/235 + 16/16** model gates; all six c2 AB/BA/AB legs; six c2 memory returns |
| Failure boundary | c16 packed repetition 1: streaming preflight, initial request and 16 warmups passed; the 96-request timed batch completed **0/96** and returned **96/96 HTTP 500** in 0.062 s |
| Terminal files | `component-status.json`, `component-summary.json`, and `component-manifest.json` are absent |
| Cleanup | source clean; no compute process; GPU 0%/P8; `/tmp/gpu` free; port 8001 free |
| CI for source | GitHub Actions run `29368963048` remains queued as of 2026-07-14 21:48 UTC; recheck it |

This is **FAILED / INCOMPLETE**, not `complete-failed`, `complete-void`, or a
performance result. Preserve the root unchanged and never append a replacement
run to it. All c2 and c16 timing/memory files are forensic only.

## What is known about the HTTP 500

The retained evidence identifies the boundary but not the thrown C++ message:

- `evidence/preflight/27/c16/packed/r1-stream.json` proves one 128-token
  streaming request completed before the timed client.
- The pinned vLLM client then passed its initial request and 16 warmups.
- Its main batch wrote 96 identical `Internal Server Error` entries, with the
  server still alive until driver cleanup.
- `ApiServer::handle_completions` catches the engine exception and places
  `e.what()` in a JSON error body (`src/vllm/entrypoints/openai/api_server.cpp`),
  but the pinned vLLM OpenAI benchmark stores only `response.reason`
  (`vllm/benchmarks/lib/endpoint_request_func.py:259`). The server log also has
  no exception line. Therefore the exact engine exception is not recoverable
  from this root; do not guess it.

Core SHA-256 evidence:

| Artifact | SHA-256 |
|---|---|
| `component-order.log` | `a2de5b0754ba13d6afd2303201ed8e25a547b17e53dc013eb1e83bf593fa6de0` |
| `component-run.log` | `297e3c6229321cd84754211cb5b17719e376b9a5a8116b542ef8e8d16d56a6fb` |
| `execution/27-component.json` | `ff71c9f0c10b7f97b9723369ca61391a47e314aba7ac77c7d3482246b1a60684` |
| c16 failed raw JSON | `f8571d4827eb67c808dbaa6a6b7790be72487429351759f55fe5c3063b7f945c` |
| c16 client log | `8b9526f4e5a9ba81af4705279f9b5496b55723e7e7295cb8e3caf9aff91563a9` |
| c16 server log | `7b05e06654fbdd383b07d059897afbacf4fc39154f0b291f9b08e17888827dfe` |
| c16 stream preflight | `a406d859e3984da87abbd642321903ed6a8e5f14608fcf1ebe594d00a69defed` |
| configure log | `c30f850dbd5248a6f1453b2abba43de7741f9d89104510214c32a52a5d09260d` |

## First actions in the next session

1. Re-read `AGENTS.md`, this file, the order-0 roadmap row, the packed-decode
   kernel/engine rows, coordination, and the newest state/ledger entries.
2. Confirm the preserved root and released resources without modifying it:

```sh
ssh dgx.casa 'bash -s' <<'REMOTE'
set -euo pipefail
ROOT=/home/mudler/work/vllm.cpp-gdn-packed-component/d82d282f9efd1a5b97e7c6f1ac7a55b949849d09
test ! -e "$ROOT/evidence/component-status.json"
sha256sum \
  "$ROOT/evidence/component-order.log" \
  "$ROOT/evidence/component-run.log" \
  "$ROOT/evidence/raw/27/ours/c16-r1-gdn-packed.json" \
  "$ROOT/evidence/logs/27/ours/c16-r1-gdn-packed.log" \
  "$ROOT/evidence/logs/27/c16/packed/r1-server.log"
test -z "$(git -C "$ROOT/source" status --porcelain)"
test -z "$(fuser /tmp/gpu 2>/dev/null || true)"
test -z "$(ss -ltn 'sport = :8001' | tail -n +2)"
/usr/bin/nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
REMOTE
```

3. Add a bounded, test-first diagnostic that preserves the non-2xx JSON error
   body and/or logs the caught server exception. Reproduce only the exact c16
   packed boundary under one lock from a new pushed SHA/root. This is diagnostic
   evidence, not a benchmark.
4. Trace the exposed exception to its source before proposing a runtime fix.
   After the fix and CPU/CUDA correctness gates, create another fresh SHA/root
   and rerun the entire twelve-leg one-lock component. Accept only verified
   marker-last `complete-pass`, `complete-failed`, or `complete-void`.

In the same next checkpoint, synchronize `README.md`, `docs/BENCHMARKS.md`,
`.agents/roadmap_v1.md`, the engine/feature/kernel/backend matrices,
coordination, porting inventory, this handoff, the live packed-decode spec, and
append-only state/ledger. Run the repository record and documentation checks,
use the mandatory commit trailers, and push directly to `main`.
