# Qwen3.5-4B H32 packed-decode AOT repair (2026-07-21)

Disposition: the H32 AOT specialization is a reproducible performance repair,
not an end-to-end parity closure. Immutable implementation commit
`176d4926d5c5c4579b1f598604f512c7320dbd51` passes the operator, model and
sanitizer gates. The isolated same-binary A/B gains 4.41% total/output
throughput and the matched node-mode profile attributes the gain to a 43.25%
reduction in packed-recurrence GPU time. The fresh vLLM 0.25.0 comparison is
still only 0.9517x on total throughput, and neither local arm is token-exact to
the deterministic eager oracle over this long free-running corpus. The owning
row therefore remains `GATING` with `benchmark_binding=false`.

## Workload and immutable roots

All performance arms use the same Qwen3.5-4B BF16 snapshot, ShareGPT corpus
`/tmp/qwen35-4b-sharegpt-1024.json` (SHA-256
`9ea13603767c62c267e3f381fbccf42d0c9ca0c393655c37533eadca7aefca0c`),
128 requests / 131,784 input / 16,384 output tokens, concurrency 32, greedy
ignore-EOS, `max-num-batched-tokens=2048`, 1,280 project KV blocks and vLLM
`max-model-len=4096`. One `flock /tmp/gpu` covers each complete series, with
cache eviction, GPU-idle and thermal/throttle guards between legs.

| Purpose | Evidence root |
|---|---|
| current vLLM 0.25.0 ON/reference/OFF, 3 memory + 3 timing repetitions per arm | `/tmp/qwen35-h32-aot-vllm25-176d4926` |
| alternating AOT/fallback same-binary A/B, 3 repetitions per arm | `/tmp/qwen35-h32-aot-ab-176d4926` |
| AOT/fallback nsys, `--cuda-graph-trace=node` | `/tmp/qwen35-h32-aot-profile-176d4926` |
| focused model and compute-sanitizer gates | `/tmp/qwen35-h32-aot-gates-176d4926` |
| deterministic vLLM 0.25.0 eager oracle, 3 repetitions | `/tmp/qwen35-h32-vllm25-eager-176d4926` |

The abandoned `/tmp/qwen35-h32-aot-176d4926` attempt is **VOID pre-leg**:
the outer environment lacked the complete NVML dependency closure, so no model
arm ran and no number was produced.

The exact production comparison is reproducible with:

```sh
flock /tmp/gpu nix develop .#cuda --command bash -c '
  export PATH="/tmp/vllm25-site/bin:$PATH"
  export PYTHONPATH="/tmp/vllm25-site${PYTHONPATH:+:$PYTHONPATH}"
  REQUIRE_TRITON_AOT=1 \
  CPP_BENCH="$PWD/build-nix-cuda-transplant-triton/examples/vllm-bench" \
  CMAKE_CACHE="$PWD/build-nix-cuda-transplant-triton/CMakeCache.txt" \
  tools/bench/run_qwen35_4b_compare.sh \
    /tmp/qwen35-h32-aot-vllm25-176d4926
'
```

Every fully expanded per-arm command is retained as `*.command` in that root.
The vLLM wheel was installed in isolated `/tmp/vllm25-site` and reports
`vllm==0.25.0`; no repository environment was mutated.

## Correctness and safety gates

| Gate | Result |
|---|---|
| packed flag test | 1/1 case, 10/10 assertions |
| exact Hv=48 and Hv=32 AOT vs hand fallback vs CPU reference | 1/1 case, 56/56 assertions |
| full `test_ops_gdn` | 53/53 cases, 3,229/3,229 assertions |
| local sm_120 generation and checked-in sm_121a manifest drift | PASS |
| real 4B default and `VT_GDN_PACKED_DECODE_TRITON=0` model test | each 3/3 cases, 1,664/1,664 assertions; short-test logs byte-identical |
| focused compute-sanitizer memcheck | 56/56 assertions, 0 errors, 0 leaks |

The long A/B stream is deterministic within each arm but AOT and fallback
differ on 9/128 requests (119/128 equal; indices
`13,23,27,28,58,61,88,110,127`). This is why the short model gate alone is not
used to claim free-running equivalence.

Production-graphed vLLM is also unsuitable as a correctness golden on this
batch: same-arm agreement with its first repetition is 128/95/95 requests.
The vLLM 0.25.0 eager, async-off oracle is deterministic across all three runs
(token SHA-256
`3ac9cc31cd9719b11b89aeb6adc6fc238ba69ce2af5793d97662a645d65bb22a`),
but AOT and fallback each match only 97/128 requests. AOT and fallback differ
on which one of two oracle requests they match, so neither is preferred by
this free-running exact-token discriminator. Operator-level AOT-vs-CPU
correctness is green; end-to-end current-oracle closure remains open.

## Isolated repair A/B

The order was AOT1/fallback1/fallback2/AOT2/AOT3/fallback3. Every direction is
positive and the spread is much smaller than the effect.

| Axis | H32 AOT (3 reps; mean) | hand fallback (3 reps; mean) | AOT delta |
|---|---:|---:|---:|
| total throughput tok/s | 6436.76 / 6412.70 / 6431.76; **6427.07** | 6159.83 / 6155.38 / 6151.51; **6155.57** | **+4.41%** |
| output throughput tok/s | 711.76 / 709.10 / 711.21; **710.69** | 681.14 / 680.65 / 680.22; **680.67** | **+4.41%** |
| request throughput req/s | **5.553** | **5.317** | **+4.45%** |
| mean TTFT ms | **722.61** | **723.32** | -0.10% |
| mean TPOT / ITL ms | **39.457** | **41.437** | **-4.78%** |
| mean E2EL ms | **5733.27** | **5985.83** | **-4.22%** |

## Matched profile attribution

Both project arms use the same binary/workload and node-mode nsys command.
Total GPU kernel time falls from **23.595950 s** to **22.527458 s**
(-1.068492 s, -4.53%). The sole large structural change is the recurrence:

| Executed recurrence | Total GPU time | Calls | Mean call |
|---|---:|---:|---:|
| H32 AOT `fused_recurrent_gated_delta_rule_packed_decode_kernel` | **1.467331 s** | 11,016 | **133.200 us** |
| hand `GdnPackedDecodeKernel<bf16,float,8>` | **2.585435 s** | 11,016 | **234.698 us** |

The AOT body is 1.762x faster and removes **1.118103 s / 43.25%** from the
recurrence. That accounts for the whole observed end-to-end gain within the
small changes elsewhere. The old branch's decomposed/BF16-state cubin remains
faster at 47.652 us/call but has the wrong ABI and is not a valid repair target;
the new kernel intentionally mirrors current vLLM's raw-packed/FP32-state FLA
body.

## Fresh vLLM 0.25.0 comparison

This 18/18-leg guarded series is the current performance checkpoint. Direct ON
and OFF remain 128/128 token-identical in every repetition.

| Axis | direct ON | direct OFF | vLLM 0.25.0 | ON / vLLM disposition |
|---|---:|---:|---:|---|
| total throughput tok/s | **6418.26** | 6316.05 | **6744.29** | **0.9517x, FAIL** |
| output throughput tok/s | **709.71** | 698.41 | **745.76** | 0.9517x, FAIL |
| request throughput req/s | **5.547** | 5.460 | **5.826** | 0.9520x, FAIL |
| mean TTFT ms | **722.68** | 819.49 | **900.04** | 0.8030x, PASS |
| mean TPOT / ITL ms | **39.513** | 39.487 | **33.492** | 1.1798x, FAIL |
| mean E2EL ms | **5741.03** | 5834.21 | **5153.57** | 1.1140x, FAIL |
| peak PSS GiB | **2.573** | 8.571 | **7.636** | PASS |
| stable PSS GiB | **0.739** | 8.571 | **4.334** | PASS |
| peak VRAM MiB | **12890** | 12870 | **12956** | vs vLLM PASS; direct ON/OFF +20 MiB mean exceeds the +8-MiB strict loader allowance |

The repair raises the earlier current-main 6155.10 tok/s checkpoint by 4.28%
and reaches **0.9714x** the previous matching-AOT 6607.04 tok/s result. It does
not erase the remaining 2.86% historical gap or the fresh 4.83% vLLM gap.
No 4B result is extrapolated to 27B/35B support.

## Artifact integrity

| Artifact | SHA-256 |
|---|---|
| vLLM-0.25 aggregate JSON | `9d09c8ee984407d76e5bed4c367619618448e75d6f75f3a11af9897c977a1ab7` |
| AOT profile kernel CSV | `7943943f2e3fa9faaadd09fcb3644e6a706526d364cdbf28d3973763851b3dbb` |
| fallback profile kernel CSV | `193c466c7522010edb56df05be2c2bc61c9f81070b4c05d275583c9ee094ab22` |
| AOT nsys stats log | `eff4c7af41d3ac4d0ffa7d124a676ebace26fd4abfe012200113207de93ae58f` |
| fallback nsys stats log | `f17e708a4596f06c20398bfa294f061a0bfd5b627abe2236a1102dbb255d3ca9` |
| memcheck log | `c778b3652b1cce29b697e23deceee2fde3e292cd945b56c771af6099e3bed938` |
| deterministic eager-oracle tokens | `3ac9cc31cd9719b11b89aeb6adc6fc238ba69ce2af5793d97662a645d65bb22a` |

Next gate: resolve the end-to-end oracle divergence, then profile the remaining
vLLM decode-path gap and repeat the exact locked comparison until every axis
passes; separately run the unavailable 27B/35B hardware regressions.
