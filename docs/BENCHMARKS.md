# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-12** (immutable `9cc7191` vLLM v0.25.0 27B
cache-off every-axis result; node-level trace recapture `PENDING`).

The official v0.25.0 tag is `702f4814fe54fabff350d43cb753ae3e47c0c276`.
Of its advertised 558 commits, 413 were already ancestors of our
`e24d1b24` source pin and 145 are the live sync delta: 94 inventory, 51 ignore,
and no immediate runtime port in the currently implemented Qwen T0 slice.
MRV2-by-default,
legacy `paged_attention_v1/v2` removal, DSpark, the Streaming Parser Engine
and Blackwell NVFP4 swizzled-scale zero-init were already in our pin. The audit
found no copied legacy PagedAttention component to retire; our paged-KV
operation remains live. v0.25.0 also retains direct swizzled scales, zeroed
unwritten padding and a device-resident CUTLASS alpha pointer. The first fresh
trace omitted our CUDA-graph child nodes; those candidates receive work only
after the node-level recapture ranks them.

The current binding result is the immutable clean
`9cc71918dbdc10f014c02feb9bab1d00963a16fe` 27B campaign against executable
vLLM v0.25.0 + FlashInfer 0.6.13. The disposition is **FAILED — parity open**:
all **12/12** performance groups, **2/2** memory groups and **124/124** axes are
binding-eligible, but only **54/124** axes pass. The denominator discrepancy is
reproduced rather than noise: all total-throughput CVs are below **0.51%**.
No 35B performance run occurred or is authorized until all 27B axes pass.

All ratios below are direction-normalized, so **≥1.0 passes**. Values are the
median of three interleaved repetitions on the identical cache-off,
input-1,024→output-128, greedy, closed-loop corpus.

| Concurrency | Axes passing | Total tok/s ours / vLLM (ratio) | Output tok/s ours / vLLM (ratio) | Mean TTFT | Mean TPOT / ITL | Mean E2EL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4/20 | 81.738938 / 82.553146 (**0.990137×**) | 9.082104 / 9.172572 (**0.990137×**) | 1.046437× | 0.988446× | 0.990143× |
| 2 | 4/20 | 150.692402 / 158.767169 (**0.949141×**) | 16.743600 / 17.640797 (**0.949141×**) | 1.125257× | 0.938395× | 0.949130× |
| 4 | 5/20 | 280.752912 / 291.434179 (**0.963349×**) | 31.194768 / 32.381575 (**0.963349×**) | 1.156047× | 0.946065× | 0.961786× |
| 8 | 4/20 | 495.694546 / 507.345858 (**0.977035×**) | 55.077172 / 56.371762 (**0.977035×**) | 1.332583× | 0.939562× | 0.976167× |
| 16 | 17/20 | 814.724651 / 791.931512 (**1.028782×**) | 90.524961 / 87.992390 (**1.028782×**) | 1.433179× | 0.988851× | 1.027968× |
| 32 | 18/20 | 1131.453402 / 1081.007187 (**1.046666×**) | 125.717045 / 120.111910 (**1.046666×**) | 1.451241× | 1.009128× | 1.046506× |

| Memory axis | Ours | vLLM | Normalized ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 48,077,626 KiB | 28,151,008 KiB | 0.585532× | **FAIL** |
| Peak RSS | 48,079,860 KiB | 28,531,696 KiB | 0.593423× | **FAIL** |
| Peak GPU memory | 38,924 MiB | 70,531 MiB | 1.812018× | PASS |
| Peak `MemAvailable` drop | 66,123,144 KiB | 80,735,264 KiB | 1.220983× | PASS |

The prior complete `b5c6e4f` grid remains historical only because it used vLLM
0.24.0 with FlashInfer 0.6.12. The follow-up `3cc490c` attempt remains **VOID**
at 28/36 groups, 1,602/2,016 requests, four returns and no trace; no value from
either campaign is mixed into the current denominator.

The replacement vLLM 0.25.0 environment is now validated and active through
`~/venvs/vllm-oracle`; v0.24.0 remains preserved at
`~/venvs/vllm-oracle-v0.24.0-retired`. Exact vLLM/FlashInfer/Torch/
CUTLASS-DSL/Humming/Transformers/Ninja/pandas versions, imports, hashes and CLI
checks pass. One initial smoke command is **VOID / COMMAND-INVALID**: v0.25
preferred its random-dataset 1024/128 defaults over the legacy
`--input-len/--output-len` aliases; the owned process group was stopped and the
lock released before the corrected run, and no value from it is retained. A
lock-held real 27B production-graph offline smoke then loaded the
24.57-GiB checkpoint, compiled/autotuned/captured and completed exact 16+1 token
counts. A second lock-held text-only server smoke returned `/health` 200 and a
1+1-token completion with `finish_reason=length`; log/response hashes are
`f56be69a…3787` / `82307db4…8e1`. These are compatibility checks, not benchmark
rates. `pip check` has one disclosed NVIDIA metadata exception: the installed
cuSPARSELt library is a loadable AArch64 ELF from the aarch64 wheel, whose
internal `manylinux2014_sbsa` tag is rejected by packaging.

The replacement campaign completed from immutable clean
`9cc71918dbdc10f014c02feb9bab1d00963a16fe`. Its evidence root is
`~/work/vllm.cpp-online-gate/evidence/9cc71918dbdc10f014c02feb9bab1d00963a16fe`.
The fail-closed plan, v0.25 oracle fingerprint, sm_121a build log, exact source
corpus and 1,008-request vLLM corpus-view manifests have SHA-256
`5a04cdcf…b2`, `6d39cb90…10c`, `10786029…6a`, `41bd634a…7a` and
`b048d789…5dc`; server/model-gate binaries are `ffddab5f…bd` /
`a24fc776…37`. One first metadata-only `record-oracle` invocation is
**FAILED / PREFLIGHT-COMMAND-INVALID** because the script form omitted the
repository module path and raised `ModuleNotFoundError: tools`; it wrote no
oracle artifact and performed no GPU work. The corrected module invocation
passed, as did plan validation, exact-SHA/source-clean checks, corpus conversion
and compilation. One uncontended whole-series lock then covered the passing
27B model gate, all **36/36** timed groups (**2,016/2,016** requests), all six
memory/cache returns and the first paired trace. The 49-file cache inventory retained
digest `da4c229c…344` throughout. The trace status is `passed:true` under its
original contract; ours retains three c16/48 Nsight windows (144 measured
requests plus client warmups) and vLLM retains warmup plus three c16/48
torch-profiler windows (192 prompts). Summary `all-runs` / ratios /
report SHA-256 are `c46595b8…a894`, `231ec9fd…7591` and `445e2d9b…a692`.

Post-run SQLite inspection found a trace-attribution gap. Ours used Nsight's
CUDA-13 default `--cuda-graph-trace=graph`: the report contains **246,786**
ordinary kernel events (**101.832 s**) and **1,226** whole-graph activities
(**154.978 s**), but **zero** child-node kernel events. vLLM's torch profiler
expands graph nodes. Therefore the 54/124 timing/memory result remains accepted
and stable, but the paired trace is **FAILED / INCOMPLETE FOR KERNEL
ATTRIBUTION** and cannot select a lever. Ours command/report SHA-256 are
`f1d4cde3…2f49` / `35fc9c4e…ad5`. The harness now requires
`--cuda-graph-trace=node`, records that granularity, and provides `--trace-only`
to recapture both profilers plus the model gate under one lock without rerunning
the grid. No node-level GPU result exists at this checkpoint.

Generated texts differ across most FP4 engine pairs and ours' three trace
digests are not all equal; this remains a diagnostic, not an ignored
correctness failure. The commit-bound real-model gate passed 16/16 and every
timed request retained the exact native 128-token count, which are the declared
correctness preconditions. vLLM also logged a missing optional
`triton_kernels.matmul_ogs` import used by GPT-OSS/MXFP4; the executed dense-27B
path resolved FlashInfer NVFP4, FLA/Triton GDN and FA2, so the warning is
recorded as non-path evidence and the frozen oracle was not mutated.

Next: run the node-level paired `--trace-only` checkpoint, then diff the
steady-state kernel lists, rank concrete executed differences, and run a
same-binary 27B A/B for the highest-gain lever. Every
27B throughput, latency and memory axis must pass before 35B; broader roadmap
work, including DSpark, remains queued behind speed parity. Detailed release classification:
[2026-07-12-702f481.md](../.agents/sync/2026-07-12-702f481.md).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **FAILED / GATING — `9cc7191` v0.25.0 27B: 54/124 AXES PASS; NODE TRACE PENDING** | Immutable `9cc7191` completed the correctness gate, all 36 cache-off groups, 2,016 requests and six returns under one uncontended lock. All 124 axes bind; 70 fail. Total-throughput ratios c1→c32 are **0.9901/0.9491/0.9633/0.9770/1.0288/1.0467×**. c16/c32 aggregate throughput wins do not close mean TPOT/ITL through c16 or host PSS/RSS (**0.5855/0.5934×**). Its first paired trace is valid only at whole-graph granularity and cannot attribute child kernels. Evidence and hashes are above; 35B did not run. | Run the new model-gate + node-level paired `--trace-only` checkpoint under one lock. Then diff actual kernel lists and run the highest-ranked same-binary 27B repair A/B; repeat the exact grid until every axis passes, then 35B. |
| `SERVE-ASYNC-LLM` HTTP capacity | **GPU-CLASSIFIED — HEALTHY / STEADY-STATE NEUTRAL; ROW GATING** | Production replaces cpp-httplib's racy 19→76 dynamic pool with a fixed **`max_num_seqs + 4`** floor (36 workers at c32); `VLLM_CPP_HTTP_FIXED_POOL=0` selects the legacy arm in the same binary. The c32 fixed/legacy AB/BA/AB means are **1097.031/1097.290 tok/s = 0.999764×**, with **0.541%/0.311% CV** and 8/20 fixed axes. All **1,152/1,152** requests and six memory returns pass; neither arm reproduces the rare historical stall. The fresh exact fixed ladder completes all three c32 legs without a queued/unread socket and narrows the current c32 oracle ratio to 0.9910×. Fixed/legacy mean GPU peaks are **39,198/38,993 MiB**; fixed PSS/RSS are slightly lower. CPU evidence remains Release/help, API **100/100**, ASan+UBSan **1/1**, and TSan **1/1**. Summary/artifact hashes are `3ce27a16…18ee9` / `27bc7f7d…53df6d`. | The bounded A/B proves no steady-state speed win and did not sample the legacy rare tail, so the broader row remains `GATING`. No more HTTP tuning is inferred: repair the confirmed FP4 path and use the exact full-grid gate to classify the remaining performance gap. |
| `BACKEND-GATE-CUDA-SGLANG-PREFIX` | **PENDING — SOURCE/CONFIG AUDIT COMPLETE; NO NUMBER ACCEPTED** | The cited recipe at `03253ef` withdraws its original 10--40x claim because it compared identical-prefix SGLang cache-on with vLLM cache-off. Its residual 35B-only cache-on cells report SGLang/vLLM 0.23.1 output throughput of **324.4/261.6** at 64k/c32, **85.3/63.8** at 256k/c2 and **133.8/92.6** at 256k/c8, but only 1--2 runs. They do not bind: vLLM 0.25 cache-on is absent; the checked-in arms mismatch BF16/FP8 KV, capacity and MTP frontend; and token-ID correctness, full axes, hit/no-eviction proof, memory and paired traces are missing. Cache-off data slightly favors vLLM and corroborates that the huge gap was configuration. | Distinct row/spike now pins SGLang v0.5.15 `f63458b` and specifies exact BF16/no-spec 64k and 256k reset→seed→timed-branch workloads, vLLM explicit `mamba_cache_mode=align`, native hit/eviction counters, equal byte capacity, three reps, full latency/throughput/memory axes and paired traces. Implement PX1/PX2 plus `KV-MAMBA-ALIGN` after the priority 27B cache-off closure; the faster equivalent reference binds per axis, 27B before 35B. |
| `KV-EXTERNAL-CACHE` / LMCache | **ROADMAP INVENTORY — NOT BENCHMARKED** | Pinned vLLM's config roles, scheduler/worker connector lifecycle, dynamic module override, load-failure policy and built-in LMCache MP/in-process connectors are now explicit source inventory, along with the official LMCache shared-prefix quickstart. vllm.cpp has no connector ABI or LMCache execution path yet, so no hit rate, TTFT, transfer-throughput, memory or reliability result exists. | Write the full spike, port a deterministic fake-provider conformance seam, then gate LMCache MP two-engine store/retrieve and Qwen3.6 hybrid behavior before the in-process leaf. Required axes: token correctness, hit/recompute behavior, TTFT, transfer GB/s, host/GPU memory, failures and metrics. |
| `KERNEL-GEMM-NVFP4-W4A4` small-M dispatch | **ACTIVE — NODE-LEVEL v0.25.0 TRACE PENDING** | W1/W2/W3 correctness and safety remain valid. The exact v0.25 grid binds and fails 70/124 axes. The paired trace token/workload contract passed, but ours captured whole graph activities and omitted child nodes, so direct swizzled scales, scalar alpha, blocking events and other candidates remain unranked. | Recapture both profilers with required Nsight node granularity. Diff executed kernels/call counts and rank gain÷effort; run a same-binary checkpoint only for a trace-proven hot-path divergence, and keep 35B held. |
| `KERNEL-GDN-AOT-BF16` 27B output dtype | **27B DEFAULT / CORRECTNESS-GREEN; STRICT GATE OPEN** | The BF16 `chunk_o` path carries the 27B recurrence output, z projection and gated-norm weight by default, matching vLLM and restoring the native 16/16 stream; `VT_GDN_OUT_BF16=0` restores f32 and every 35B path retains f32. Its BF16/f32 component remains **1.007989×**, **16/20** timing and **2/4** memory. Binding `9cc7191` has c16 total throughput **1.028782×** but normalized mean TPOT/ITL **0.988851×**. | Keep correctness-faithful BF16 for 27B and retain the row `ACTIVE`; use the node-level v0.25.0 paired recapture to re-rank GDN against remaining executed gaps. Do not infer any 35B result. |
| `KV-DEVICE-RESIDENCY` | **ACTIVE — W0+W1 A/B/TRACE/CORRECTNESS PASS; ZERO-LEAK FAIL** | W0/W1 same-binary gains, copy reduction and correctness/safety evidence remain valid. Inherited pools still fail strict teardown (27B **47,290,056 B/101**, 35B **36,822,413,188 B/1,236**); old-oracle host-memory ratios are historical. | Keep W2 scoped until the v0.25.0 grid re-ranks the residual; separately repair model/pool/queue teardown. |

The stream-usage path changes host-side JSON/SSE serialization, not model
kernels. Its performance disposition is nevertheless `PENDING` because the
final frame participates in whole-request throughput and the feature directly
unblocks the binding online latency campaign. Retained ITLs are inter-choice
timings: pinned vLLM's single-slot collector deliberately merges DELTA outputs
when the producer gets ahead, while native usage remains the exact token-count
oracle. The gate accepts fewer than 127 ITLs for that case but still rejects
extra intervals, partial requests, count drift, errors, or unsaturated load.

The audit found that the historical direct-library comparisons were not exact
same-workload runs. Pinned `vllm bench throughput` hard-codes
`temperature=1.0`, while the vllm.cpp arms used greedy `temperature=0`; the 27B
vLLM run also resolved `max_num_batched_tokens=8192`, versus 2048 for vllm.cpp.
They remain useful historical diagnostics, but the strict performance gate is
reopened for both models. Correctness remains established separately by the
commit-bound real-model greedy gates and exact native request counts.

## Historical CUDA engine checkpoints — reopened, non-binding

These values are preserved so the optimization history remains reproducible.
They cannot establish production-vLLM parity under the current protocol because
their oracle sampling differed; the 27B scheduler budget differed as well.

| Model / point | Historical build and workload | vllm.cpp | vLLM | Diagnostic ratio | Disposition |
|---|---|---:|---:|---:|---|
| Qwen3.6-35B-A3B NVFP4, c64 / 200 prompts | Triton-AOT GDN; input 1024, output 128 | 3345.9 tok/s | 3282.0 tok/s | 1.0195× | Non-binding: ours temperature 0, vLLM temperature 1. Correctness remains 16/16 token-exact in its separate greedy gate. [Record](../.agents/state.md#L1740). |
| Qwen3.6-27B NVFP4, c16 / 96 prompts | Triton-AOT + default FA-2 prefill; input 1024, output 128, seed 0 | 764.28 tok/s total; 84.89 output | 758.84 tok/s total; 84.32 output | 1.0072× total; 1.0068× output | Non-binding: temperature 0/1 and token budget 2048/8192. Historical repetition spread retained in the [ledger](../.agents/parity-ledger.md#L284). |
| Qwen3.6-27B NVFP4, c32 / 192 prompts | Same historical build | 1051.24 tok/s total; 116.77 output | 1043.86 tok/s total; 115.98 output | 1.0071× total; 1.0068× output | Non-binding for the same two configuration mismatches; historical peak memory was 61.8 vs 76.2 GB. |

The historical 35B result requires `-DVLLM_CPP_TRITON=ON`; its default pure-C++
build measured 0.99× in the same old campaign. The 27B values require the
default-on vendored FA-2 prefill route. Same-binary component A/Bs still support
the individual kernel choices, but fresh exact oracle runs are required for an
end-to-end acceptance claim, and binding server-to-server latency remains open.

### Reproduce the W3-A correctness and component checkpoints

Use immutable `71f1e894d0c5e496607d08cfe9089a9944128271` in one clean detached
checkout on `dgx.casa`. The two test arms use the same binary and one uncontended GPU lock;
the default delayed arm must report `delay=1000us`, while the fallback must
report `delay=off`. The test command is the correctness/selection preflight.
The archived component driver below then runs the exact c16/96 AB/BA/AB; copy
it to a fresh path and change only its `ev=` destination because it refuses to
overwrite evidence.

```sh
BUILD="$HOME/work/vllm.cpp-nvfp4-small-m/w3-a-build"
cmake -S . -B "$BUILD" -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD" --target test_ops_nvfp4_fp4 \
  test_qwen27_paged_engine server --parallel "$(nproc)"
flock /tmp/gpu sh -c '
  VT_FP4_AUTOTUNE_VERBOSE=1 ctest --test-dir "'"$BUILD"'" \
    -R "^(test_ops_nvfp4_fp4|test_qwen27_paged_engine)$" --output-on-failure &&
  VT_FP4_AUTOTUNE_VERBOSE=1 VT_FP4_AUTOTUNE_DELAY=0 \
    ctest --test-dir "'"$BUILD"'" \
    -R "^(test_ops_nvfp4_fp4|test_qwen27_paged_engine)$" --output-on-failure
'

# Exact measured component driver (SHA-256 425f8521...e9ae).
W3="$HOME/work/vllm.cpp-nvfp4-small-m/71f1e894d0c5e496607d08cfe9089a9944128271/w3"
cp "$W3/component-ab/driver.sh" /tmp/w3a-component-repro.sh
sed -i "s|^ev=.*|ev=$W3/component-ab-repro|" /tmp/w3a-component-repro.sh
cd "$W3/source"
/tmp/w3a-component-repro.sh
```

### Reproduce the W3-B immutable and component checkpoints

Run this from clean pushed `d7cdf66` (or a later code-identical checkpoint) on
`dgx.casa`. The model log must contain exactly one
`profiles_requested=80 profiles_tuned=80 cached_plans=80` completion, no
`lazy-miss after pre-serve warmup`, and the 16/16 production stream.

```sh
SHA=$(git rev-parse HEAD)
BUILD="$HOME/work/vllm.cpp-nvfp4-small-m/$SHA/w3b/build"
cmake -S . -B "$BUILD" -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD" --target test_ops_nvfp4_fp4 \
  test_qwen27_paged_engine server --parallel "$(nproc)"
flock /tmp/gpu sh -c '
  "'"$BUILD"'"/tests/test_ops_nvfp4_fp4 &&
  VT_FP4_EXACT_BUCKETS=0 "'"$BUILD"'"/tests/test_ops_nvfp4_fp4 &&
  VT_FP4_AUTOTUNE_VERBOSE=1 \
    "'"$BUILD"'"/tests/test_qwen27_paged_engine
'

# Exact measured W3-B component driver (SHA-256 e996e6dd...662).
W3="$HOME/work/vllm.cpp-nvfp4-small-m/d7cdf66db0cfcc53d68d49613623ec6cd3807641/w3b"
cp "$W3/component-ab/driver.sh" /tmp/w3b-component-repro.sh
sed -i "s|^ev=.*|ev=$W3/component-ab-repro|" /tmp/w3b-component-repro.sh
cd "$W3/source"
/tmp/w3b-component-repro.sh

# Corrected paired trace (SHA-256 af29681e...22fbf). This is intentionally
# tied to immutable d7cdf66 and writes a new evidence root.
cp "$W3/trace-ab-oracle-r2/driver.sh" /tmp/w3b-trace-repro.sh
sed -i "s|^ev=.*|ev=$W3/trace-ab-oracle-r2-repro|" /tmp/w3b-trace-repro.sh
cd "$W3/source"
/tmp/w3b-trace-repro.sh
```

## Reproduce the current online checkpoint

The existing immutable evidence can be re-aggregated without GPU work by using
a disposable copy. Exit **1** is the expected result: complete evidence whose
every-axis gate failed. Exit 2 is a harness/evidence error.

```sh
SOURCE="$HOME/work/vllm.cpp-online-gate/evidence/9cc71918dbdc10f014c02feb9bab1d00963a16fe"
CHECK="/tmp/vllm-cpp-9cc7191-summary-$USER"
cp -a --reflink=auto "$SOURCE" "$CHECK"
rm -rf "$CHECK/summary-27"
set +e
PYTHONPATH="$PWD" python3 tools/bench/online_gate_summary.py \
  --evidence "$CHECK" --model 27
rc=$?
set -e
test "$rc" -eq 1
sha256sum "$CHECK"/summary-27/{all-runs.json,ratios.json,report.md}
```

The immediate next checkpoint uses a new clean SHA/evidence root for the
node-level paired trace only:

```sh
SHA=$(git rev-parse HEAD)
TRACE_ROOT="$HOME/work/vllm.cpp-online-trace-node"
TRACE_EVIDENCE="$TRACE_ROOT/evidence/$SHA"
TRACE_BUILD="$TRACE_ROOT/build/$SHA"
CLIENT="$HOME/venvs/vllm-oracle/bin/vllm"
M27=$(dirname "$(find "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots" -name config.json -print -quit)")

scripts/dgx-online-serving.sh --dry-run \
  --claim-root "$TRACE_ROOT" --client "$CLIENT" --vllm-cpp-sha "$SHA"
# Execute the exact 27B corpus command recorded in $TRACE_EVIDENCE/manifest.json.
cmake -S . -B "$TRACE_BUILD" -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$TRACE_BUILD" --target server test_qwen27_paged_engine \
  --parallel "$(nproc)"
scripts/dgx-online-serving.sh --trace-only --model 27 --snapshot "$M27" \
  --source-corpus "$TRACE_EVIDENCE/corpus/27" --evidence "$TRACE_EVIDENCE" \
  --build-dir "$TRACE_BUILD" --client "$CLIENT" --vllm-cpp-sha "$SHA"
```

After the trace selects and an A/B accepts a repair, the next full checkpoint
uses another new clean SHA/evidence root and repeats the whole model series:

Run from a clean, merged checkout on `dgx.casa`; the driver owns one
uncontended `/tmp/gpu` lock for each whole-model series and refuses stale or
partial evidence. Snapshot paths below are the pinned gate checkpoints. Before
execution, `CLIENT --version` and the evidence manifest must resolve vLLM
0.25.0, FlashInfer 0.6.13 and the accepted dependency hashes; the retired
0.24.0 environment makes the run void.

```sh
SHA=$(git rev-parse HEAD)
CLAIM_ROOT="$HOME/work/vllm.cpp-online-gate"
EVIDENCE="$CLAIM_ROOT/evidence/$SHA"
CLIENT="$HOME/venvs/vllm-oracle/bin/vllm"
BUILD="$HOME/work/vllm.cpp-online-build"
M27=$(dirname "$(find "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots" -name config.json -print -quit)")
M35=$(dirname "$(find "$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots" -name config.json -print -quit)")

scripts/dgx-online-serving.sh --dry-run \
  --claim-root "$CLAIM_ROOT" --client "$CLIENT" --vllm-cpp-sha "$SHA"
# Execute the exact corpus commands recorded in $EVIDENCE/manifest.json,
# then build the clean SHA in $BUILD before either measured arm.
scripts/dgx-online-serving.sh --execute --model 27 --snapshot "$M27" \
  --source-corpus "$EVIDENCE/corpus/27" --evidence "$EVIDENCE" \
  --build-dir "$BUILD" --client "$CLIENT"
# Do not execute this 35B command until the validator reports every 27B axis
# passing; it is retained here as the eventual second-model gate recipe.
scripts/dgx-online-serving.sh --execute --model 35 --snapshot "$M35" \
  --source-corpus "$EVIDENCE/corpus/35" --evidence "$EVIDENCE" \
  --build-dir "$BUILD" --client "$CLIENT"
```

The full acceptance contract, including corpus generation, zero-residency
cache proof, warmup, repetitions, every-axis validation, memory return, and
paired traces, is in the
[online serving gate spec](../.agents/specs/cuda-online-serving-gate.md) and
[benchmark protocol](../.agents/benchmark-protocol.md).
