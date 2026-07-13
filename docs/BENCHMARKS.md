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
v0.25 persistent/frozen FP4 tactic caching to W3-C; C2 runtime integration and
frozen-map correctness pass; immutable `d211b8f` passes six-process 64/64
stability; C3's first component remains void, while C3R proves 6/6 x 128-token
direct/fallback equality under identical sequential batch shape and reproduces
default-mode batch-shape dependence in both ours and vLLM, correcting the old
cross-run equality predicate; the corrected 12-leg frozen-plan component then
strict-fails at 39/40 timing + 1/8 memory despite 1.004483×/1.005044× c2/c16
mean throughput; the resumed trace/source scan selects W3-F model-owned device
alpha; its tensor ABI, model-owned individual/merged residency, direct CUDA
pointer path, exact host fallback and upstream-shaped tests are now implemented
and local CPU/sanitizer-green; immutable `7517af4` passes the clean sm_121a
build, focused CUDA/operator, strict no-pool memcheck, both 27B arms, 35B
correctness-only inertness and paired node-trace gates; its completed
frozen-plan c2/c16 component reaches 1.001967×/1.000144× total throughput but
strict-fails at 27/40 timing + 3/8 memory, so no speed credit/exact grid follows
and the completed multi-lens scan selects W3-G FA2 ratio-6 split-KV decode;
the full cast-free BF16/capture/fallback/test/every-axis spike is accepted, but
implementation and every new runtime number are explicitly `PENDING`).

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
W3-C is therefore mandatory benchmark-reproduction control, not speed credit.
C2 now publishes loaded plans into the live ready map before warmup, freezes
misses, uses the 5,000-us miss ABI and saves only successful warmups. The exact
fixture loads **64/64 with 0 tuned / 0 rejected / 0 saved**, cache-hit capture
and save/cancel lifecycle tests pass, compute-sanitizer reports **0 errors**,
and frozen default/fallback 27B each pass **235/235 + 16/16**. Focused Release,
ASan+UBSan and TSan pass **7/7 + 189/189**. Two current full CPU attempts are
**102/103 FAILED** on the unchanged C-API early-stop timing flake; an older
baseline binary reproduces it, while the remaining **102/102** and an isolated
C-API rerun pass. Immutable `d211b8f` then seeds native cache SHA
`2590fc94…199d` and runs six fresh processes: all six logs are byte-identical,
load the same **64/64** map (`f2d9be7f…1fa4`) with zero tuning/rejections/saves
or lazy misses, and pass **235/235 + 16/16**. The first frozen c2 direct/fallback
pair uses that identical map and completes exact 6/6 requests plus 768/768
output tokens, but the then-required independent-run equality check is only
**2/6**. The driver correctly stops; every partial timing/memory value remains
**VOID**, c16 and the conditional exact grid do not run, and no 35B performance
command runs.

W3-C3R resolves that stopped predicate without promoting any performance
number. Direct and fallback servers receiving the same six prompts strictly
sequentially are **6/6 equal for all 128 output tokens**, with the same frozen
64-plan map and zero tuning/misses. Each arm is **0/6** equal to its own earlier
c2 output. The exact vLLM v0.25.0 production server, prefix caching off and
`VLLM_BATCH_INVARIANT` explicitly unset, is also **0/6** equal for the same
sequential-versus-c2 comparison; first divergences occur at output token 0--7.
Upstream defaults the mode off and enables it only in its determinism fixture,
where NVFP4, attention, norm and other dispatch are changed. Cross-run online
text equality is therefore not a production-performance correctness boundary.
The corrected gate retains producer/fixed-tactic byte parity, both 235/235 +
vLLM 16/16 gates, controlled 6/6 x 128-token same-shape equality, exact counts,
frozen plans, lifecycle, and every timing/memory axis. Reclassification summary
SHA is `a1c500b3…41de`; all timings/memory from the stopped first component are
**NOT APPLICABLE**.

The corrected component is now complete from immutable runtime `d211b8f` and
gate definition `69a5c45`. Under one uninterrupted lock, both fixed model arms
pass **235/235 + 16/16**, and all **12/12** c2/c16 AB/BA/AB legs complete
**612/612** requests plus **12/12** memory returns. All 12 processes load the
same frozen **64/64** plan map; every paired repetition has 64/64 equal tactic
IDs, with zero tuning or lazy misses. At c2, direct/fallback mean total
throughput is **150.992608/150.318657 = 1.004483480×**, with **20/20 timing +
1/4 memory** axes. At c16 it is **812.541436/808.463407 = 1.005044173×**, with
**19/20 timing + 0/4 memory** axes; only p99 TPOT misses at **0.997683064×**.
The combined strict result is therefore **FAILED: 39/40 timing + 1/8 memory**.
All 24 before/after cache-drop reports succeed with zero resident bytes after
drop, and GPU/lock/port exit idle, so the memory-axis failures are measured
peak regressions rather than teardown failures. Summary/selection/driver-log/
provenance SHA are `3a3707cb…1249` / `6761a3a7…ef9` / `bc83594d…c4` /
`f5b55d30…2ef`; driver SHA is `3c9c5771…e21`. W3-C cache reproduction control
is complete, but W3-E remains strict-failed and earns no speed credit. The
binding vLLM result remains `3f256ab` at 55/124; no exact grid or 35B
performance command ran.

The completed bounded checkpoint is **W3-F / GATING — CUDA/TRACE PASS; STRICT
COMPONENT FAILED**.
Immutable local node trace `99cbd04d…93f8` contains 296,575 graph-child
`SetScalar` launches over 1,425 forward markers (**208.123/forward**) and
207.124383 ms total. The later direct-scale trace `bb6c6fe…966` independently
contains exactly 624 launches over three forwards (**208/forward**). Binding
vLLM kernel summary `e4e916d1…565` and Torch-profiler trace `0c6f859f…0e2`
contain zero. Whole-chain inspection shows vLLM v0.25 creates `layer.alpha` as
a device F32 parameter, FlashInfer passes its tensor unchanged, and the SM121
runner forwards `globalScale.data_ptr()` to CUTLASS. The accepted
[W3-F spike](../.agents/specs/nvfp4-device-alpha.md) is now implemented through
the local checkpoint: `MatmulNvfp4Cutlass` has a validated rank-0/rank-1 F32
tensor overload while retaining the float overload; the CUDA adapter passes
the tensor pointer directly and stages only the fallback; and true-W4A4
individual, merged gate/up and packed-QKV projections own persistent scalars.
`VT_FP4_DEVICE_ALPHA=0` selects the exact old path. The port covers malformed
dtype/rank/numel/contiguity/device contracts, non-unit host/device equality,
rank-0/rank-1 tensors, capture/replay, all 32 forced tactics and packed QKV.
Full CPU Release passes **103/103**; focused Release, ASan+UBSan and TSan each
pass **17/17 cases + 911/911 assertions**.

Immutable pushed `7517af4f983fe322ac88ce2d9869e1441b7be3fd` is now clean-built
on GB10 with GCC 13.3, CUDA 13.0.88, sm_121a, FlashInfer's CUTLASS 4.5 tree,
vendored Triton AOT, Marlin and FA2. The focused CUDA suite passes **33/33**
CTest registrations, including all 32 forced tactics. Strict
`VT_CUTLASS_NOPOOL=1` compute-sanitizer covers initialization, eager execution
and capture/replay and passes **22/22 cases + 26,884/26,884 assertions**, with
**0 errors and 0 bytes leaked**. The commit-owned native cache imports the same
checked-in vLLM v0.25/FlashInfer 64-plan fixture with **64 loaded, 0 tuned, 0
rejected and 64 saved**, then both read-only 27B arms load 64 native plans with
zero tuning/misses and independently pass **235/235 assertions + 16/16 oracle
tokens**. The correctness-only 35B W4A16 gate passes **2/2 cases + 315/315
assertions**, proving the W3-F dispatch is inert there; this is not a 35B
performance result.

One-lock `nsys --cuda-graph-trace=node` profiles of those exact 27B arms close
the structural contract. Device alpha contains **zero** `SetScalar` kernels.
The host fallback contains **624 eager + 2,912 graph = 3,536** calls totaling
3.082880 ms: exactly **208 per each of 3 eager and 14 graphed forwards**. Both
arms execute **3,536 FP4 GEMMs**, **3,536 FP4 quant producers**, eight identical
CUTLASS kernel identities and the same selected-plan SHA `f2d9be7f…1fa4`;
canonical GEMM/producer hashes are `0b29bd49…37a` / `910bd8df…56f7`.
Relative to the host arm, device alpha records exactly **208** extra
`cudaMalloc`, **208** extra `cudaMemcpyAsync` and **208** extra `cudaFree`
calls, matching one `sizeof(float)` model-owned scalar per logical FP4 linear:
**832 bytes total**, returned at teardown. Retained vLLM kernel evidence
`e4e916d1…565` remains free of `SetScalar`.

Three setup attempts are retained without laundering them into results. The
first configure is **FAILED / ENVIRONMENT-INVALID** because non-interactive SSH
omitted `/usr/local/cuda-13.0/bin`; the clean compiler-pinned retry passes. The
first CTest wrapper is **VOID / COMMAND-INVALID** because nested quoting made
CTest inspect `$HOME` and find zero tests; the corrected 33-test invocation
passes. The first 27B read-only model attempt is **FAILED-CLOSED / STALE-CACHE**:
it rejects the prior commit's native document on `build_id` before inference;
reseeding from the immutable upstream fixture produces the passing
commit-owned cache. No rate or correctness observation is taken from any of
those attempts.

Evidence is under
`~/work/vllm.cpp-nvfp4-device-alpha/7517af4f983fe322ac88ce2d9869e1441b7be3fd/evidence`.
Build/focused-CUDA/memcheck/default-27B/host-27B/35B/trace-summary SHA-256 values
are `56203868…13db` / `8318b668…6900` / `9296bd10…824` /
`77f20b30…05d9` / `77f20b30…05d9` / `575a7faf…b1f` /
`8e6436ef…1455`.

The unprofiled cache-off input-1,024/output-128 c2/c16 AB/BA/AB component then
completes under one uninterrupted `/tmp/gpu` lock. Both fixed model gates pass
**235/235 + 16/16**. All **12/12** legs, **612/612** requests, **626,688** input
tokens, **78,336** output tokens, **12/12** memory returns and all **24/24**
before/after cache drops pass; temperature is 51--65 C and GPU/lock/port exit
idle. Every process loads the identical **64/64 native plan map** with zero
tuning, rejection, save or lazy miss, and every paired repetition has 64/64
equal tactic IDs.

At c2, device/host mean total throughput is
**152.871951898/152.571813445 = 1.0019671946×**, with **16/20 timing + 3/4
memory** axes. Mean ITL/TPOT and median/p90 TPOT fail; peak available-memory
drop also fails. At c16 it is **817.229771058/817.111876605 =
1.0001442819×**, with **11/20 timing + 0/4 memory** axes. Nine E2E/TTFT/TPOT
axes and all four memory axes fail. Combined acceptance is therefore
**FAILED: 27/40 timing + 3/8 memory**. Total-throughput CVs remain low
(device/host: c2 **0.0347%/0.0751%**, c16 **0.0873%/0.0536%**), so the result
is a stable near-neutral classification, not a speed win. Independent online
text hashes are 0/6 equal only as the already-corrected batch-shape diagnostic;
the fixed model gates and immutable W3-C3R 6/6 x 128-token same-shape proof are
the correctness boundary.

Component summary/selection/driver/driver-log/provenance SHA-256 values are
`676612c3…bf08` / `6006d999…cbd2` / `49d486ad…44db` /
`b927ac4f…6871` / `f76174c5…776e`. Therefore **W3-F receives no speed credit;
the binding denominator remains `3f256ab` at 55/124, the exact grid and all
35B performance stay blocked.** The required multi-lens scan is now complete
and selects the separate W3-G FA2 decode checkpoint below.

Read-only baseline and current immutable structural gate:

```sh
ssh dgx.casa 'DB=~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27/ours.sqlite; sqlite3 "$DB" "SELECT COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.graphNodeId IS NOT NULL AND instr(s.value,'\''SetScalar'\'')>0;"'
cmake -S . -B build-cpu -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SERVER=OFF
cmake --build build-cpu --target test_ops_nvfp4_fp4 -j"$(nproc)"
ctest --test-dir build-cpu -R test_ops_nvfp4_fp4 --output-on-failure
ssh dgx.casa 'cat ~/work/vllm.cpp-nvfp4-device-alpha/7517af4f983fe322ac88ce2d9869e1441b7be3fd/evidence/trace/structural-summary.txt'
ssh dgx.casa 'ROOT=~/work/vllm.cpp-nvfp4-device-alpha/7517af4f983fe322ac88ce2d9869e1441b7be3fd/evidence/component-ab-c2-c16-device-vs-host-alpha; jq . "$ROOT/summary.json"; jq "{gate_pass,all_process_plan_maps_equal,all_process_metadata_equal,paired}" "$ROOT/selection-summary.json"; sha256sum "$ROOT/summary.json" "$ROOT/selection-summary.json" "$ROOT/driver.sh" "$ROOT/driver.log" "$ROOT/provenance.txt"'
```

The current bounded checkpoint is **W3-G / READY — TRACE-GROUNDED SPIKE
ACCEPTED; IMPLEMENTATION AND BENCHMARK PENDING**. Qwen3.6-27B uses Hq/Hkv
24/4, ratio 6. Our optimized fused GQA decode is hard-coded to ratio 8, so the
binding trace falls through to
`PagedAttentionDecodeOptKernel<float,bfloat16,float>`: **22,893 calls / 8,793.238
ms**, **384.102 us/call**, or **3.083%** of local GPU kernel time. vLLM's
retained kernel export contains the FA2 BF16 split main **23,616 calls /
7,061.921 ms** plus combine **23,488 / 123.245 ms**. The trace windows and call
counts differ; these values identify an executed structural gap and are **not
an accepted engine ratio**. Per observed layer call the diagnostic difference
is about 80 us, roughly 1.28 ms across 16 attention layers.

Whole-chain inspection resolves the prior apparent contradiction. Ordinary
paged varlen FA2 rejects explicit `num_splits>1`, but the dependency's separate
pure-decode `seqlenq_ngroups_swapped` branch transposes the logical GQA layout,
sets query length to the group count, runs the exact split heuristic and emits
main plus combine. vLLM v0.25.0 pins the same FA2 commit `2c839c33` already
vendored here. W3-G therefore ports the missing thin adapter/model dispatch,
not a new kernel or dependency version.

The accepted [W3-G spike](../.agents/specs/fa2-gqa-split-kv-decode.md) bounds
the first route to CUDA BF16 paged pure decode, D256, ratio 6, no
window/ALiBi/softcap/dropout. It requires cast-free model-side BF16 Q/output,
the upstream 80%/85% split heuristic, capture-stable scratch keyed by graph
shape, queue teardown, and exact `VT_FA2_DECODE=0` fallback. Existing prefill is
unchanged. The 35B ratio-8 path is explicitly ineligible and receives only a
later correctness-inertness check, never a performance run here. Vectorized
normal BF16→FP4 production ranks second and is not stacked into W3-G; host
weight release remains a separate memory repair.

All W3-G implementation, operator tests, sanitizer, 27B ON/OFF model gates,
35B inertness, node traces and c2/c16 performance are **PENDING**. The component
must pass all **40/40 timing + 8/8 memory** axes before an exact vLLM grid may
run. Read-only trace reproduction and the first post-implementation handoff:

```sh
ssh dgx.casa 'ROOT=~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27; sqlite3 -header -column "$ROOT/ours.sqlite" "SELECT s.value,COUNT(*),ROUND(SUM(k.end-k.start)/1e6,3),ROUND(AVG(k.end-k.start)/1e3,3) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE instr(s.value,'\''PagedAttentionDecode'\'')>0 GROUP BY s.value;"; jq ".kernels[] | select((.name // \"\")|test(\"flash_fwd_splitkv\"))" "$ROOT/vllm-kernels.json"'

SHA=$(git rev-parse HEAD)
BUILD="$HOME/work/vllm.cpp-fa2-decode/$SHA/build"
cmake -S . -B "$BUILD" -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD" --target test_ops_paged_attn \
  test_qwen27_paged_engine test_qwen36_paged_engine server --parallel "$(nproc)"
flock /tmp/gpu sh -c '
  ctest --test-dir '"'"'"$BUILD"'"'"' -R "^(test_ops_paged_attn|test_qwen27_paged_engine)$" --output-on-failure &&
  VT_FA2_DECODE=0 ctest --test-dir '"'"'"$BUILD"'"'"' -R "^(test_ops_paged_attn|test_qwen27_paged_engine)$" --output-on-failure
'
```

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
correctness preconditions.

The later corrected component removes that plan-selection confounder. Runtime
`d211b8f` plus gate `69a5c45` holds the same frozen 64/64 map across all 12
processes and every paired repetition. Direct/fallback throughput becomes
**150.992608/150.318657 = 1.004483480×** at c2 and
**812.541436/808.463407 = 1.005044173×** at c16. Timing improves to **20/20 +
19/20**, with only c16 p99 TPOT at 0.997683064×, but memory is **1/4 + 0/4**.
Thus the corrected and better-controlled W3-E disposition remains **FAILED:
39/40 timing + 1/8 memory**. The older 32/40 + 6/8 result remains historical,
not binding. vLLM also logged a missing optional
`triton_kernels.matmul_ogs` import used by GPT-OSS/MXFP4; the executed dense-27B
path resolved FlashInfer NVFP4, FLA/Triton GDN and FA2, so the warning is
recorded as non-path evidence and the frozen oracle was not mutated.

Immutable `d211b8f80fff831a712f0bfafa4f65f1abe1892d` closes the W3-C3
stability question. One
read/write 27B seed imports the checked-in v0.25 fixture and publishes native
cache SHA `2590fc94…199d`; six subsequent read-only native-only processes
(three direct, three `VT_FP4_DIRECT_SF=0`) each load 64/64, tune/reject/save
0/0/0, report no lazy miss, pass 235/235 assertions and reproduce 16/16 oracle
tokens. Their full logs are byte-identical (`523c2478…95f2` each) and their
selected-map SHA is `f2d9be7f…1fa4`.

The lock-held c2/c16 driver then passes both model gates and completes the first
c2 direct/fallback pair with exact 6/6 requests, 6,144/6,144 input tokens and
768/768 output tokens. Both server logs contain the exact same 64 selected
lines and zero tuning/misses, but only zero-based request indices 4 and 5
(equality count **2/6**) have equal generated text; indices 0--3 differ.
Direct/fallback raw SHA are `42047046…842f` / `5bb0e70a…7296`; compact
generated-text SHA are
`900f6134…ec7` / `b34a17e8…437`. The then-current exact-text predicate therefore
stops the run before timing can bind. The remaining series was intentionally
terminated and cleaned up; its partial/unpaired artifacts and every observed
rate/memory value remain **VOID**.
Failure-summary/driver/driver-log SHA are `01c9faf6…bd74` /
`fd76dd8b…b886` / `67307adc…b5b`; GPU, lock and port 8001 exit idle/free.

W3-C3R then runs two correctness-only diagnostics under whole-series locks.
The same six prompts sent sequentially through frozen direct and fallback
servers match **6/6 x 128 tokens** (comparison SHA `42d74898…cf41`); each arm's
earlier c2 output matches its own sequential output **0/6**. The exact
production-default vLLM v0.25.0 server, with prefix cache off and
`VLLM_BATCH_INVARIANT` unset, also matches sequential versus c2 **0/6**
(comparison SHA `cb717bb2…c597`). This mirrors upstream's explicit opt-in-only
determinism tests. Combined summary SHA is `a1c500b3…41de`.

Read-only reproduction of the corrected disposition (no model/GPU execution):

```sh
ssh dgx.casa 'ROOT=~/work/vllm.cpp-nvfp4-persistent/d211b8f80fff831a712f0bfafa4f65f1abe1892d; jq . "$ROOT/evidence/c3r-reclassification-summary.json"; sha256sum "$ROOT/evidence/c3r-reclassification-summary.json" "$ROOT/evidence/c3r-sequential-c2-r1/comparison.json" "$ROOT/evidence/c3r-vllm-default-batch-c2-r1/comparison.json"'
```

The replacement corrected gate uses
`~/work/vllm.cpp-nvfp4-persistent/d211b8f80fff831a712f0bfafa4f65f1abe1892d/evidence/component-ab-c2-c16-corrected-gate-69a5c45`.
Read-only reproduction:

```sh
ssh dgx.casa 'ROOT=~/work/vllm.cpp-nvfp4-persistent/d211b8f80fff831a712f0bfafa4f65f1abe1892d/evidence/component-ab-c2-c16-corrected-gate-69a5c45; jq . "$ROOT/summary.json"; jq "{gate_pass,all_process_plan_maps_equal,all_process_metadata_equal,paired}" "$ROOT/selection-summary.json"; sha256sum "$ROOT/summary.json" "$ROOT/selection-summary.json" "$ROOT/driver.sh" "$ROOT/driver.log" "$ROOT/provenance.txt"'
```

Next: preserve W3-F's upstream-shaped implementation and completed structural
evidence without speed credit, then implement the accepted W3-G FA2 ratio-6
decode spike. Its operator/sanitizer/model/inertness/trace gates precede a
strict c2/c16 same-binary component; the conditional exact grid remains blocked
until every component axis passes. Every 27B throughput, latency and memory axis
must pass before 35B performance; broader roadmap work, including DSpark,
remains queued behind speed parity.
Detailed release classification:
[2026-07-12-702f481.md](../.agents/sync/2026-07-12-702f481.md).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **FAILED / GATING — `3f256ab` BINDS 55/124** | Immutable `3f256ab` remains **55/124**. W3-C passes frozen-map control; W3-E and W3-F strict-fail and earn no speed credit. The completed scan selects W3-G ratio-6 FA2 decode from the actual local **22,893 / 8,793.238-ms** kernel slice versus vLLM split main+combine. Unequal trace windows make it diagnostic only. Its complete BF16/capture/fallback/test spike is accepted; implementation and all runtime evidence are `PENDING`. No exact grid or 35B performance command ran. | Implement W3-G, then require its immutable operator/sanitizer/27B-ON+OFF/35B-inertness/trace gates and all **40 timing + 8 memory** component axes. Only then run the exact vLLM grid; 35B performance remains held. |
| `ENG-BATCH-INVARIANT` | **ROADMAP INVENTORY — NOT IMPLEMENTED / NOT APPLICABLE TO PRODUCTION SPEED FLOOR** | vLLM v0.25.0 defaults `VLLM_BATCH_INVARIANT` off; its opt-in determinism suite changes NVFP4, matmul, norm, attention and collective dispatch. C3R executes only the default-off contrast and records 0/6 sequential-vs-c2 equality for both engines. vllm.cpp exposes no matching opt-in mode, so no support or performance result is claimed. | After production parity, write `specs/batch-invariant-execution.md`, port the upstream operator/e2e determinism cases and gate correctness separately from the default production performance path. |
| `SERVE-ASYNC-LLM` HTTP capacity | **GPU-CLASSIFIED — HEALTHY / STEADY-STATE NEUTRAL; ROW GATING** | Production replaces cpp-httplib's racy 19→76 dynamic pool with a fixed **`max_num_seqs + 4`** floor (36 workers at c32); `VLLM_CPP_HTTP_FIXED_POOL=0` selects the legacy arm in the same binary. The c32 fixed/legacy AB/BA/AB means are **1097.031/1097.290 tok/s = 0.999764×**, with **0.541%/0.311% CV** and 8/20 fixed axes. All **1,152/1,152** requests and six memory returns pass; neither arm reproduces the rare historical stall. The fresh exact fixed ladder completes all three c32 legs without a queued/unread socket and narrows the current c32 oracle ratio to 0.9910×. Fixed/legacy mean GPU peaks are **39,198/38,993 MiB**; fixed PSS/RSS are slightly lower. CPU evidence remains Release/help, API **100/100**, ASan+UBSan **1/1**, and TSan **1/1**. Summary/artifact hashes are `3ce27a16…18ee9` / `27bc7f7d…53df6d`. | The bounded A/B proves no steady-state speed win and did not sample the legacy rare tail, so the broader row remains `GATING`. No more HTTP tuning is inferred: repair the confirmed FP4 path and use the exact full-grid gate to classify the remaining performance gap. |
| `BACKEND-GATE-CUDA-SGLANG-PREFIX` | **PENDING — SOURCE/CONFIG AUDIT COMPLETE; NO NUMBER ACCEPTED** | The cited recipe at `03253ef` withdraws its original 10--40x claim because it compared identical-prefix SGLang cache-on with vLLM cache-off. Its residual 35B-only cache-on cells report SGLang/vLLM 0.23.1 output throughput of **324.4/261.6** at 64k/c32, **85.3/63.8** at 256k/c2 and **133.8/92.6** at 256k/c8, but only 1--2 runs. They do not bind: vLLM 0.25 cache-on is absent; the checked-in arms mismatch BF16/FP8 KV, capacity and MTP frontend; and token-ID correctness, full axes, hit/no-eviction proof, memory and paired traces are missing. Cache-off data slightly favors vLLM and corroborates that the huge gap was configuration. | Distinct row/spike now pins SGLang v0.5.15 `f63458b` and specifies exact BF16/no-spec 64k and 256k reset→seed→timed-branch workloads, vLLM explicit `mamba_cache_mode=align`, native hit/eviction counters, equal byte capacity, three reps, full latency/throughput/memory axes and paired traces. Implement PX1/PX2 plus `KV-MAMBA-ALIGN` after the priority 27B cache-off closure; the faster equivalent reference binds per axis, 27B before 35B. |
| `KV-EXTERNAL-CACHE` / LMCache | **ROADMAP INVENTORY — NOT BENCHMARKED** | Pinned vLLM's config roles, scheduler/worker connector lifecycle, dynamic module override, load-failure policy and built-in LMCache MP/in-process connectors are now explicit source inventory, along with the official LMCache shared-prefix quickstart. vllm.cpp has no connector ABI or LMCache execution path yet, so no hit rate, TTFT, transfer-throughput, memory or reliability result exists. | Write the full spike, port a deterministic fake-provider conformance seam, then gate LMCache MP two-engine store/retrieve and Qwen3.6 hybrid behavior before the in-process leaf. Required axes: token correctness, hit/recompute behavior, TTFT, transfer GB/s, host/GPU memory, failures and metrics. |
| `KERNEL-GEMM-NVFP4-W4A4` small-M dispatch | **ACTIVE / GATING — W3-F STRUCTURAL PASS; PERFORMANCE FAILED** | W3-C reproduction control is complete and W3-E remains failed at **39/40 timing + 1/8 memory**. W3-F supplies resident alpha and passes local/CUDA/memcheck/model/trace gates, but its component strict-fails **27/40 timing + 3/8 memory**; `3f256ab` remains 55/124. The closed scan ranks vectorized normal BF16→FP4 production second, behind W3-G, and no quant change is added now. | Keep the FP4 producer unstacked until W3-G is classified. Exact grid/35B performance remain held. |
| `KERNEL-ATTN-FA2` ratio-6 split-KV decode | **PENDING — W3-G SPIKE ACCEPTED / ROW ACTIVE** | Existing vendored FA2 prefill remains supported. Binding local decode is **22,893 calls / 8,793.238 ms**; vLLM's unequal-window export contains split main **23,616 / 7,061.921 ms** plus combine **23,488 / 123.245 ms**. The same dependency commit is already vendored. No adapter/model/test/runtime change or accepted ratio exists at this checkpoint. | Port exact pure-decode swap/heuristic with graph-stable scratch and `VT_FA2_DECODE=0`; port upstream paged vectors; then pass safety, both 27B arms, 35B inertness, paired node trace and all 40+8 component axes. |
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
