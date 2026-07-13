# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-13** (immutable `3f256ab` vLLM v0.25.0 27B
cache-off every-axis result and paired trace; strict parity remains failed at
55/124; generated-code inspection refutes the apparent norm+FP4 fusion lever,
then W3-E direct swizzled activation scales reaches implementation,
correctness, sanitizer and structural-trace gates; immutable `53ab149` c2/c16
A/B strict-fails at 32/40 timing + 6/8 memory; the post-failure scan promotes
v0.25 persistent/frozen FP4 tactic caching to W3-C; C1 document/import tests
pass while C2 runtime integration and same-plan performance remain `PENDING`).

The official v0.25.0 tag is `702f4814fe54fabff350d43cb753ae3e47c0c276`.
Of its advertised 558 commits, 413 were already ancestors of our
`e24d1b24` source pin and 145 are the live sync delta: 94 inventory, 51 ignore,
and no immediate runtime port in the currently implemented Qwen T0 slice.
MRV2-by-default,
legacy `paged_attention_v1/v2` removal, DSpark, the Streaming Parser Engine
and Blackwell NVFP4 swizzled-scale zero-init were already in our pin. The audit
found no copied legacy PagedAttention component to retire; our paged-KV
operation remains live. v0.25.0 also retains direct swizzled scales, zeroed
unwritten padding and a device-resident CUTLASS alpha pointer. The accepted
node trace first ranked packed QKV. After that topology closed, body-level
inspection selected direct scale emission as the next bounded sub-spike. It is
now implemented and structurally verified. Its mean throughput improves, but
the completed every-axis component fails, so no performance number is accepted.
The subsequent execution audit corrects a v0.24-era assumption: the binding
v0.25 log loads 64 persistent FlashInfer FP4 configs and reports a config-file
hit. Ours retuned each W3-E process and shared only 18--33/64 paired tactic IDs.
W3-C is therefore spiked as mandatory benchmark-reproduction control. It has
no accepted number. C1's CUDA-free cache layer now passes focused Release,
ASan+UBSan and TSan at **6/6 cases + 174/174 assertions** each and a fresh full
CPU suite at **103/103**; all 64 binding oracle plans import exactly. C2 runtime
publication, stable 64/64 fresh-process plans, same-plan output hashes,
component axes and any conditional exact grid are all **PENDING**.

The current binding result is the immutable clean
`3f256abdbb558e162bf8a2196284deb119648560` 27B campaign against executable
vLLM v0.25.0 + FlashInfer 0.6.13. The disposition is **FAILED — parity open**:
all **12/12** performance groups, **2/2** memory groups and **124/124** axes are
binding-eligible, but only **55/124** axes pass. The denominator discrepancy is
reproduced rather than noise: the maximum total-throughput CV is **0.189%**.
No 35B performance run occurred or is authorized until all 27B axes pass.

All ratios below are direction-normalized, so **≥1.0 passes**. Values are the
median of three interleaved repetitions on the identical cache-off,
input-1,024→output-128, greedy, closed-loop corpus.

| Concurrency | Axes passing | Total tok/s ours / vLLM (ratio) | Output tok/s ours / vLLM (ratio) | Mean TTFT | Mean TPOT / ITL | Mean E2EL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5/20 | 81.645106 / 82.178953 (**0.993504×**) | 9.071678 / 9.130995 (**0.993504×**) | 1.038340× | 0.992092× | 0.993513× |
| 2 | 4/20 | 150.561023 / 157.744007 (**0.954464×**) | 16.729003 / 17.527112 (**0.954464×**) | 1.196031× | 0.942815× | 0.954468× |
| 4 | 5/20 | 280.291354 / 290.025183 (**0.966438×**) | 31.143484 / 32.225020 (**0.966438×**) | 1.066496× | 0.954755× | 0.964313× |
| 8 | 4/20 | 495.699906 / 505.466352 (**0.980678×**) | 55.077767 / 56.162928 (**0.980678×**) | 1.382124× | 0.941853× | 0.980621× |
| 16 | 17/20 | 812.302839 / 790.263558 (**1.027889×**) | 90.255871 / 87.807062 (**1.027889×**) | 1.432626× | 0.987450× | 1.027464× |
| 32 | 18/20 | 1121.954512 / 1079.407095 (**1.039417×**) | 124.661612 / 119.934122 (**1.039417×**) | 1.446098× | 1.002666× | 1.039521× |

| Memory axis | Ours | vLLM | Normalized ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 48,175,537 KiB | 28,167,719 KiB | 0.584689× | **FAIL** |
| Peak RSS | 48,177,860 KiB | 28,534,276 KiB | 0.592269× | **FAIL** |
| Peak GPU memory | 38,561 MiB | 70,531 MiB | 1.829076× | PASS |
| Peak `MemAvailable` drop | 65,901,992 KiB | 80,911,844 KiB | 1.227760× | PASS |

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
expands graph nodes. Therefore that 54/124 timing/memory result remained valid
until the completed `3f256ab` grid superseded it, but the paired trace is
**FAILED / INCOMPLETE FOR KERNEL
ATTRIBUTION** and cannot select a lever. Ours command/report SHA-256 are
`f1d4cde3…2f49` / `35fc9c4e…ad5`. The harness now requires
`--cuda-graph-trace=node`, records that granularity, and provides `--trace-only`
to recapture both profilers plus the model gate under one lock without rerunning
the grid.

That replacement checkpoint is complete from immutable clean
`def5f752896036d9b35841a278578fd812f75a0d`, with evidence at
`~/work/vllm.cpp-online-trace-node/evidence/def5f752896036d9b35841a278578fd812f75a0d`.
The commit-bound 27B gate passes in **44.79 s**; both profilers execute the same
closed-loop c16 workload with 48 warmup prompts plus three 48-prompt windows,
input 1,024→output 128, cache off. `status.json` is `passed:true` and records
`cuda_graph_trace: node`; all three 49-file cache-drop/lifecycle proofs pass,
and GPU, port and lock exit clean. Status / ours Nsight / ours kernel summary /
vLLM kernel summary / vLLM trace SHA-256 are `c5a07125…11f4`,
`71af83c5…1a36`, `42916a72…36e3`, `e4b2d8fe…6a90` and
`8c4a267e…4291`.

The local SQLite export (`7c8aadd2…eae5`) contains **2,587,766** CUDA-kernel
rows: **2,315,412** graph-child rows (**181.922 s**) and **272,354** eager rows
(**110.387 s**), over 7,711 distinct graph-node IDs. This closes the old
attribution gap. Cross-profiler durations remain diagnostic, but identical-work
launch counts expose the first structural repair. vLLM executes exactly
**330,304 / 1,588 = 208 FP4 GEMMs per forward step**. Our graph executes
**343,461 / 1,430 = 240.18** (240 expected plus 261 capture/warmup launches).
The model has 64 layers—48 linear-attention and 16 full-attention—and source
inspection explains the exact 32-launch delta: vLLM's `QKVParallelLinear`
packs q/k/v into one `qkv_proj`, while ours runs three FP4 GEMMs in each of the
16 full-attention layers. The local decode tactic mix is consequently dominated
by 128×32×256 and 128×32×128, while vLLM also uses 128×64×256 materially.
Packed QKV with one max-derived CT scale/alpha and a split fallback is the
selected W3-D contract. No profiler rate or speedup is accepted from this trace.

W3-D is now implemented and measured from clean pushed `3f256ab`. The 27B CUDA
path concatenates resident Q/K/V FP4 weights
and linear scales at N=`12,288+1,024+1,024=14,336`, applies one maximum-derived
CT input/weight divisor and alpha, quantizes once, launches one GEMM, and feeds
row-strided Q/K/V views directly to the fused preamble, dense value cast and
KV-cache writer. `VT_FP4_MERGED_QKV=0` restores the prior three-GEMM path.
Default pre-serve tuning now materializes **64/64** profiles instead of the
split arm's **80/80**, because packed N=14,336 replaces N=12,288 and N=1,024.
The low-level packed result equals all three logical BF16 outputs exactly
(max absolute difference **0**); row-strided preamble/cache suites pass; both
default and fallback real-model runs pass **235/235 assertions and 16/16
tokens**. Focused compute-sanitizer runs report **zero errors**. Immutable
evidence is
`~/work/vllm.cpp-packed-qkv/3f256abdbb558e162bf8a2196284deb119648560/w3d/component-ab`.
The c16/96 input-1,024/output-128 AB/BA/AB packed runs are
**815.886/810.759/810.047**, split runs **811.294/805.779/807.377 tok/s**;
means are **812.231/808.150 = 1.005049×**, with CV **0.320%/0.287%**. Strict
component acceptance is **FAILED: 14/20 timing + 2/4 memory axes**. Packed
improves sampled GPU peak **38,059→37,765 MiB** and available-memory drop, but
PSS/RSS are about 60 MiB higher and six tail/TTFT axes miss. All **576/576**
requests and six memory returns pass; both immutable correctness arms pass
**235/235 + 16/16**. Summary/selection/tree SHA are `c13ee24e…6976` /
`7eebec5b…bece` / `ff8e7fea…3041`.

The fresh post-pack node trace at
`~/work/vllm.cpp-online-trace-node/evidence/3f256abdbb558e162bf8a2196284deb119648560`
passes its model, node-granularity, cache and lifecycle contracts. Ours contains
**2,170,753 graph-child / 248,529 eager** kernel rows and 7,231 graph-node IDs.
The FP4 topology target closes: **296,674 graph FP4 GEMMs / 1,425 graph
lm-head markers = 208.192 per forward**, exactly the expected 208 plus 274
capture/warmup launches. The same oracle remains **330,304 / 1,588 = 208**.
Profiled rates/durations remain diagnostic. Status / ours Nsight / ours SQLite /
ours kernel summary / vLLM kernel summary SHA are `90350b03…9908` /
`6e7e3c6c…b5f9` / `607877d2…65cd` / `43ae3507…44ac` /
`7988b5ea…08ee`. GPU, port and lock exit idle.

The fresh exact rerun is complete and binding at
`~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560`.
Its vLLM 0.25 plan and exact copied source/vLLM corpus validate; plan/source /
vLLM-corpus SHA are `0e309d8b…9999` / `41bd634a…fd7a` /
`b048d789…e5dc`. One uninterrupted `/tmp/gpu` lock covered the passing model
gate, all **36/36** timed groups (**2,016/2,016** requests), six memory/cache
returns and the paired trace. All **12/12** performance groups, **2/2** memory
groups and **124/124** axes are binding-eligible. Strict parity **fails at
55/124**, one net axis better than `9cc7191`; only c1 p90 ITL crosses the floor.
Total-throughput deltas versus that prior binding grid are +0.337/+0.532/
+0.309/+0.364/-0.089/-0.725 percentage points at c1→c32. Summary all-runs /
ratios / report SHA are `83b3f500…9f8` / `66d7f50e…b4bd` /
`df3d0539…e4d7`; model-gate log SHA is `36191579…6e69`.

The exact paired trace also passes its node/cache/lifecycle contract; status /
ours Nsight / ours SQLite / ours kernel summary / vLLM kernel summary SHA are
`9762c1e6…1d0c6` / `e397289d…8476` / `99cbd04d…93f8` /
`55a1631a…d2be` / `e4e916d1…565`. It reconfirms the repaired packed-QKV
topology. Its three names containing
`fused_add_rms_norm_scaled_fp4_quant` total **127,040 launches**, exactly
**80 per 1,588 forwards**, but the generated bodies do not quantize: they
perform residual-add + RMSNorm, store BF16, and the wrapper then invokes
`torch.ops._C.scaled_fp4_quant.out` separately (the traced
`cvt_fp16_to_fp4`). The oracle log/config also has `fuse_norm_quant: False`.
Thus the long name describes a topologically sorted source group, not a fused
norm+FP4 kernel, and our residual-add RMSNorm followed by separate
`ScaledFp4QuantKernel` is structurally equivalent. `KERNEL-EW-NORM-QUANT`
stays `PARTIAL` and unclaimed; historical neutral `76e9047` remains shelved.
Generated computation graph / compiled subgraph SHA are `d58f81b8…9401` /
`466e359a…9dd8`. The continued local Nsight slice identifies
`SwizzleBlockscaleKernel`: **320,099 launches / 1.238881054 s**, split as
23,524 eager / 506.032544 ms and 296,575 graph-child / 732.848510 ms. Over
1,425 forward markers that is **224.631 launches and 0.869390 ms/forward**.
vLLM's exact summary has zero standalone kernels with that name because its
executed normal and fused FP4 quant producers write the swizzled scale address
directly. This is structural attribution, not accepted speed. The
[W3-E spike](../.agents/specs/nvfp4-direct-swizzled-scales.md) is implemented
and `GATING`. `Fp4ScaleLayout` makes linear versus padded CUTLASS output
explicit; normal and both fused CUDA producers zero/write the swizzled bytes
in their existing launch; true-W4A4 model sites default direct; and
`VT_FP4_DIRECT_SF=0` restores the exact linear producer + standalone swizzle.
CPU/CUDA direct bytes and packed FP4 match the composed path over M=1/127/128/
256 and K=64/1,024/4,096/5,120/14,336/16,384/17,408; direct/composed CUTLASS
BF16 output is byte-identical. Focused CUDA tests pass **4/4 / 24,647/24,647**
assertions, and producer plus M=1,N=14,336,K=5,120 no-pool memchecks report
zero errors and zero leaks. The first packed-QKV memcheck with production pools
enabled is **FAILED for leak checking only**: it reports the intentionally live
4-byte alpha plus 8,389,120-byte workspace caches at process exit, with no
access errors. Reproduction adds `VT_CUTLASS_NOPOOL=1`, which frees both and
passes zero-error/zero-leak; production pool teardown remains the separate
`KV-DEVICE-RESIDENCY` debt. Real 27B direct/fallback each pass **235/235 +
16/16**; the 35B correctness-only run passes **315/315**, confirming W4A16 is
inert. Paired real-model Nsight reports direct/fallback standalone swizzles
**208/832**: all **624** activation-scale swizzles disappear, while the 208
one-time weight swizzles remain. Direct/fallback normal producer counts are
432/432 and one-input fused counts 192/192. Trace report SHA are
`ad87631e…c022` / `c3063f90…e1f8`; kernel CSV SHA are `aee5220e…0779` /
`eb4d5713…1369`. These profiled durations are non-binding. Immutable pushed
`53ab149` completes the c2/c16 same-binary AB/BA/AB: all 12 legs, **612/612**
requests, 12/12 memory returns and both 235/235 model gates pass under one lock.
c2 direct/fallback mean total throughput is
**150.116922/149.801191 = 1.002107665x**, with **16/20 timing + 4/4 memory**;
c16 is **796.834440/791.907102 = 1.006222116x**, with **16/20 timing + 2/4
memory**. Combined strict acceptance is **FAILED: 32/40 timing + 6/8 memory**.
c2 fails median/p90 TPOT and p90/p99 TTFT; c16 fails mean TTFT, p90 TPOT,
p99 E2EL, p99 TTFT and PSS/RSS. Summary/selection SHA are
`cfff5711…50e9` / `ceaa5296…47b4`. The binding `3f256ab` parity result and all
published oracle ratios remain unchanged. GPU and lock exit idle.

Read-only reproduction of the component disposition (no model/GPU execution):

```sh
ssh dgx.casa 'SUMMARY_ONLY=1 ~/work/vllm.cpp-direct-sf/53ab1492983282a9858cc301d4f7e9aad4784c48/summary-driver-corrected.sh'
```

Generated texts differ across most FP4 engine pairs and ours' three trace
digests are not all equal; W3-E's component likewise has five of six unequal
paired 128-token hashes. Only 18--33/64 paired tactic IDs match and just 9--17
keys per arm stay stable across all c16 repetitions, so tactic selection is a
recorded confounder rather than an assigned cause. This remains a diagnostic,
not an ignored correctness failure. The commit-bound real-model gate passed
16/16 and every
timed request retained the exact native 128-token count, which are the declared
correctness preconditions. vLLM also logged a missing optional
`triton_kernels.matmul_ogs` import used by GPT-OSS/MXFP4; the executed dense-27B
path resolved FlashInfer NVFP4, FLA/Triton GDN and FA2, so the warning is
recorded as non-path evidence and the frozen oracle was not mutated.

Next: do not run the conditional W3-E exact grid; its component failed. The
completed scan selects W3-C persistent/frozen plan parity as the next bounded
control. Implement and gate it, require identical 64/64 plans and a same-plan
c2/c16 component, then run an exact grid only after every component axis passes.
Every 27B throughput, latency and memory axis must pass before 35B performance;
broader roadmap work, including DSpark, remains queued behind speed parity.
Detailed release classification:
[2026-07-12-702f481.md](../.agents/sync/2026-07-12-702f481.md).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **FAILED / GATING — `3f256ab` BINDS 55/124** | Immutable `3f256ab` remains **55/124**. W3-E removes 624 swizzles but strict-fails **32/40 timing + 6/8 memory**. Binding v0.25 loads 64 persistent plans while ours is tactic-unstable. W3-C C1 implements native/import JSON, metadata/tactic validation, merge, atomic replace and path/frozen options; Release/ASan+UBSan/TSan **6/6 + 174/174**, full CPU **103/103**, exact oracle import **64/64**. This code is not wired to serving, so no ratio changes. The 35B correctness-only inertness gate ran; no 35B performance run occurred. | Implement C2 ready-map/startup/load-save integration and 5,000-us miss timing; then require 64/64 identical fresh-process plans and same-plan c2/c16 every-axis acceptance before any exact grid. 35B performance remains held. |
| `SERVE-ASYNC-LLM` HTTP capacity | **GPU-CLASSIFIED — HEALTHY / STEADY-STATE NEUTRAL; ROW GATING** | Production replaces cpp-httplib's racy 19→76 dynamic pool with a fixed **`max_num_seqs + 4`** floor (36 workers at c32); `VLLM_CPP_HTTP_FIXED_POOL=0` selects the legacy arm in the same binary. The c32 fixed/legacy AB/BA/AB means are **1097.031/1097.290 tok/s = 0.999764×**, with **0.541%/0.311% CV** and 8/20 fixed axes. All **1,152/1,152** requests and six memory returns pass; neither arm reproduces the rare historical stall. The fresh exact fixed ladder completes all three c32 legs without a queued/unread socket and narrows the current c32 oracle ratio to 0.9910×. Fixed/legacy mean GPU peaks are **39,198/38,993 MiB**; fixed PSS/RSS are slightly lower. CPU evidence remains Release/help, API **100/100**, ASan+UBSan **1/1**, and TSan **1/1**. Summary/artifact hashes are `3ce27a16…18ee9` / `27bc7f7d…53df6d`. | The bounded A/B proves no steady-state speed win and did not sample the legacy rare tail, so the broader row remains `GATING`. No more HTTP tuning is inferred: repair the confirmed FP4 path and use the exact full-grid gate to classify the remaining performance gap. |
| `BACKEND-GATE-CUDA-SGLANG-PREFIX` | **PENDING — SOURCE/CONFIG AUDIT COMPLETE; NO NUMBER ACCEPTED** | The cited recipe at `03253ef` withdraws its original 10--40x claim because it compared identical-prefix SGLang cache-on with vLLM cache-off. Its residual 35B-only cache-on cells report SGLang/vLLM 0.23.1 output throughput of **324.4/261.6** at 64k/c32, **85.3/63.8** at 256k/c2 and **133.8/92.6** at 256k/c8, but only 1--2 runs. They do not bind: vLLM 0.25 cache-on is absent; the checked-in arms mismatch BF16/FP8 KV, capacity and MTP frontend; and token-ID correctness, full axes, hit/no-eviction proof, memory and paired traces are missing. Cache-off data slightly favors vLLM and corroborates that the huge gap was configuration. | Distinct row/spike now pins SGLang v0.5.15 `f63458b` and specifies exact BF16/no-spec 64k and 256k reset→seed→timed-branch workloads, vLLM explicit `mamba_cache_mode=align`, native hit/eviction counters, equal byte capacity, three reps, full latency/throughput/memory axes and paired traces. Implement PX1/PX2 plus `KV-MAMBA-ALIGN` after the priority 27B cache-off closure; the faster equivalent reference binds per axis, 27B before 35B. |
| `KV-EXTERNAL-CACHE` / LMCache | **ROADMAP INVENTORY — NOT BENCHMARKED** | Pinned vLLM's config roles, scheduler/worker connector lifecycle, dynamic module override, load-failure policy and built-in LMCache MP/in-process connectors are now explicit source inventory, along with the official LMCache shared-prefix quickstart. vllm.cpp has no connector ABI or LMCache execution path yet, so no hit rate, TTFT, transfer-throughput, memory or reliability result exists. | Write the full spike, port a deterministic fake-provider conformance seam, then gate LMCache MP two-engine store/retrieve and Qwen3.6 hybrid behavior before the in-process leaf. Required axes: token correctness, hit/recompute behavior, TTFT, transfer GB/s, host/GPU memory, failures and metrics. |
| `KERNEL-GEMM-NVFP4-W4A4` small-M dispatch | **ACTIVE — W3-C1 IMPLEMENTED/GATED; C2 + BENCHMARK PENDING** | W3-E remains strict-failed. W3-C1 adds the CUDA-free pure-C++ cache document/import layer at `nvfp4_persistent_cache.{h,cpp}` plus the immutable 64-plan fixture. It covers native round-trip/merge, exact FlashInfer tuple keys, metadata/layout/dtype/device/tactic rejection, environment/frozen paths and same-directory atomic publication. Release, ASan+UBSan and TSan each pass **6/6 + 174/174**; corrected full CPU is **103/103**. The initial full run was **101/103 FAILED** because the fixture sat under op-golden discovery; relocation to `tests/fixtures` and a fresh rerun close it. No CUDA/model/performance command ran; `3f256ab` remains 55/124. | Implement C2 ready-map snapshot/import, resolved 5,000-us timing, lifecycle/stats and load-before-warmup/save-after-complete. Then run six-process 64/64 and same-plan gates; exact grid/35B performance remain held. |
| `KERNEL-EW-NORM-QUANT` | **PARTIAL — FALSE TRACE-NAME LEVER REFUTED** | vLLM's 127,040 long-named kernels stop after residual-add + RMSNorm to BF16; a separate `scaled_fp4_quant.out`/`cvt_fp16_to_fp4` follows, matching our two-kernel topology. `fuse_norm_quant` is false. Existing FP8 fusion remains gated and historical byte-exact/neutral `76e9047` stays shelved. | No spike/implementation is promoted from this trace. Revisit only if a future body/dispatch difference or surpass-track measurement justifies it independently. |
| `KERNEL-GDN-AOT-BF16` 27B output dtype | **27B DEFAULT / CORRECTNESS-GREEN; STRICT GATE OPEN** | The BF16 `chunk_o` path carries the 27B recurrence output, z projection and gated-norm weight by default, matching vLLM and restoring the native 16/16 stream; `VT_GDN_OUT_BF16=0` restores f32 and every 35B path retains f32. Its BF16/f32 component remains **1.007989×**, **16/20** timing and **2/4** memory. Binding `3f256ab` has c16 total throughput **1.027889×** but normalized mean TPOT/ITL **0.987450×**. Cross-profiler GDN totals remain diagnostic only; no new GDN lever is selected yet. | Keep correctness-faithful BF16 for 27B and retain the row `ACTIVE`; revisit only after body-level residual ranking. Do not infer any 35B result. |
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
SOURCE="$HOME/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560"
CHECK="/tmp/vllm-cpp-3f256ab-summary-$USER"
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

The accepted node-level paired trace is reproducible from immutable
`def5f752896036d9b35841a278578fd812f75a0d` with a fresh evidence root:

```sh
SHA=def5f752896036d9b35841a278578fd812f75a0d
TRACE_ROOT="$HOME/work/vllm.cpp-online-trace-node"
REPRO_ROOT="$HOME/work/vllm.cpp-online-trace-node-repro"
TRACE_EVIDENCE="$REPRO_ROOT/evidence/$SHA"
TRACE_SOURCE="$TRACE_ROOT/checkpoints/$SHA/source"
TRACE_BUILD="$REPRO_ROOT/checkpoints/$SHA/build"
CORPUS="$TRACE_ROOT/evidence/$SHA/corpus/27"
CLIENT="$HOME/venvs/vllm-oracle/bin/vllm"
M27=$(dirname "$(find "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots" -name config.json -print -quit)")

cd "$TRACE_SOURCE"
scripts/dgx-online-serving.sh --dry-run \
  --claim-root "$REPRO_ROOT" --client "$CLIENT" --vllm-cpp-sha "$SHA"
# Execute the exact immutable 27B source corpus from the accepted checkpoint.
cmake -S . -B "$TRACE_BUILD" -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$TRACE_BUILD" --target server test_qwen27_paged_engine \
  --parallel "$(nproc)"
scripts/dgx-online-serving.sh --trace-only --model 27 --snapshot "$M27" \
  --source-corpus "$CORPUS" --evidence "$TRACE_EVIDENCE" \
  --build-dir "$TRACE_BUILD" --client "$CLIENT" --vllm-cpp-sha "$SHA"
sha256sum "$TRACE_EVIDENCE"/trace/status.json \
  "$TRACE_EVIDENCE"/trace/ours/report.nsys-rep \
  "$TRACE_EVIDENCE"/trace/ours/kernel-summary.json
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
