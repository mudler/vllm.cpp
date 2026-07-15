# Benchmarks

This is the public current-state scoreboard for vllm.cpp. It contains the
binding result, the active performance diagnosis, pending gates, and current
reproduction entry points. Attempt chronology and failure forensics live in the
[parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), linked specs, and Git. Those raw records are
append-only within the current era and are frozen under `.agents/completed/`
when the era is rolled up; this page never accumulates their run-by-run history.

Last updated: **2026-07-14**. Qwen3.6-27B parity against vLLM v0.25.0 is
**FAILED / open at 55/124 axes**. The active
[packed-decode checkpoint](../.agents/specs/gdn-packed-decode.md) retains
immutable correctness at `f344dec` and accepted structure from `7ff713e`,
finalized by `24cea4f`: packed has **915** nodes versus rollback's **963**, with
the exact 48-for-96 GDN substitution and invariant remaining topology. The
production-build-only c2/c16 AB/BA/AB runner and marker-last every-axis
finalizer are hardened. Clean source `d82d282`
completed both direct model gates and all six c2 legs, then **FAILED /
INCOMPLETE** at c16 packed repetition 1. The streaming preflight, initial
request and 16 warmups passed, but all **0/96** timed requests returned HTTP 500
in 0.062 s. The driver exited without a marker-last status. The root cause was
unrecoverable because our port dropped two upstream fatal log lines
(`vllm/v1/engine/core.py:1233`, `async_llm.py:703-705`). A bounded **test-first
diagnostic checkpoint** restores them as four unconditional `std::cerr`
error-path channels (`engine-fatal:`, `async-llm:`, `api-server:`, `sse:`),
adds an opt-in `VT_GDN_DIAG_STEP_LOG` geometry trace, and adds a packed-only
`--diagnostic-c16` driver mode isolated from the gating component
(`component-diagnostic.json`; the finalizer refuses diagnostic evidence). Six
RED→GREEN tests land it; all tools pass **132/132**, focused
`test_gdn_packed_component` **49/49**, and CPU `test_async_llm`/`test_openai_api_server`
are green. The `--diagnostic-c16` reproduction at `4a450f9` **captured the
root cause deterministically 3/3**: `vt: qwen3_5: duplicate live GDN state
index` (`qwen3_5.cpp:73`) during a mixed-batch step of the c16 burst — the
runner's GDN state-slot lifecycle assigned one slot to two live requests (last
healthy step: `num_reqs=6, gdn_free_slots=27, gdn_live_slots=5` of the 32-slot
pool).

**Root cause + test-first repair (landed).** The runner keyed its compact GDN
state-slot pool (`remap_gdn_state_slots`, `runner.cpp`) on the mamba **block-id**
(block-table column 0). The 27B GDN group is configured with a sub-sequence
`block_size` (`MakeQwen3_5KVCache`), so once a sequence exceeds one mamba block
`MambaManager::remove_skipped_blocks` nulls every block but the last and column
0 collapses to the shared null block-id **0**. Two long concurrent c16 sequences
therefore both presented block-id 0 and were remapped onto **one** recurrent
state slot → the duplicate the validator (added at `f344dec`) rejects. vLLM
reaches the same per-sequence state index via `mamba_get_block_table_tensor`
(gathering the current state block); because our compact per-sequence state
cache makes the physical block-id irrelevant, the fix keys the pool on the
**request identity** so each live sequence owns exactly one slot for its whole
lifetime, released only when it leaves the batch. A RED `test_runner` case
threw the exact fatal; it is GREEN after the fix (`test_runner` **8/8**).

**Blast-radius caveat (correctness).** The defect predates the validator. (a) It
is independent of `VT_GDN_PACKED_DECODE`: the remap and the validator both run
on the common decode path, so the rollback arm (`=0`) hits the same duplicate.
(b) The compact slot pool was introduced at `66715e1` (2026-07-05); the
uniqueness validator only at `f344dec` (2026-07-14). Binaries in between —
including the `3f256ab` binding grid and earlier c16/c32 campaigns — would have
**silently** run two or more long concurrent sequences on ONE GDN recurrent
state slot (cross-request state corruption) rather than crashing. Low-concurrency
16/16 correctness gates (few concurrent sequences and/or short prompts that stay
within one mamba block) would not surface it; high-concurrency runs measure
throughput, not token correctness, so any such prior c16/c32 numbers are
throughput-valid but their per-token output correctness at concurrency is
**suspect**. No binding throughput number changes; recorded here for honesty.

The c16 component stays **FAILED / INCOMPLETE** until the DGX correctness gates
and a fresh 12-leg component rerun pass; partial c2/c16 files are nonbinding and
`benchmark_binding=false`.

## Binding 27B online gate

| Item | Binding value |
|---|---|
| Model | Qwen3.6-27B NVFP4 |
| Hardware | NVIDIA GB10 / DGX Spark, sm_121a |
| vllm.cpp source | `3f256abdbb558e162bf8a2196284deb119648560` |
| Reference | vLLM v0.25.0, tag `702f4814fe54fabff350d43cb753ae3e47c0c276`, FlashInfer 0.6.13 |
| Workload | Cache off, input 1,024 → output 128, greedy, closed loop, c1/c2/c4/c8/c16/c32 |
| Repetitions | Three interleaved repetitions per point |
| Evidence completeness | 12/12 performance groups, 2/2 memory groups, 124/124 axes eligible |
| Stability | Maximum total-throughput CV **0.189%** |
| Disposition | **FAILED: 55/124 pass, 69/124 fail** |

Ratios are direction-normalized: throughput is ours/vLLM, latency is
vLLM/ours, and **1.0 or higher passes**. Values are medians of three
interleaved repetitions.

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

Total throughput passes at c16/c32, but every throughput, request-rate,
TTFT, TPOT/ITL, E2EL, and memory axis must pass. At c2, ours has better TTFT
but decode TPOT is **114.841 vs 108.274 ms** (**6.1% slower**). No 35B
performance command is authorized until the 27B result reaches 124/124.

## Current checkpoint

| Track | Disposition | Current evidence | Next binding gate |
|---|---|---|---|
| `SERVE-GATE-ONLINE` | **FAILED / GATING** | Immutable `3f256ab` remains **55/124**; packed correctness/structure is accepted. The `4a450f9` reproduction captured the c16 `duplicate live GDN state index` **3/3**. **Test-first repair landed**: the runner's compact GDN state-slot pool now keys on the request identity (was the mamba block-id, which collapsed long concurrent sequences onto one slot); RED `test_runner` threw the exact fatal → GREEN, tools **132/132** | DGX correctness gates + a fresh full 12-leg component rerun from the pushed SHA/root; the exact grid remains unauthorized |
| `KERNEL-GEMM-BF16` | **GATING W1C** | `0091cd1` closes BA structure and `f925294` closes projection/inertness. The isolated BF16-BA + decomposed control remains **233/235**; clean `f344dec` closes W1D2/G2 for the exact coupled BF16-BA + packed path at **235/235**, `benchmark_binding=false` | Depends on the packed W1D3 component gate; qkvz stays blocked |
| `KERNEL-GDN-PACKED-DECODE` | **ACTIVE / W1D3 repair LANDED** | Root cause captured 3/3 at `4a450f9` (6 requests, 5 live + 27 free of 32 slots; `packed_decode=0` lead-up). **Fixed test-first**: `remap_gdn_state_slots` keyed the compact slot pool on the mamba block-id, which collapses to the shared null block-id 0 once a sequence exceeds one mamba block (sub-sequence `block_size`), so two long c16 sequences shared one slot; the pool now keys on the request identity. RED runner test threw `duplicate live GDN state index`, GREEN after fix (`test_runner` 8/8; tools 132/132) | Rerun all twelve legs from the pushed SHA/root after the DGX correctness gates |
| RMSNorm/generated partitions | **CLOSED / DISPROVEN as a parity gap** | The 2026-07-14 [parity rescan](../.agents/specs/parity-rescan-2026-07-14.md) verified vLLM's `RmsNormQuantFusionPass` is FP8-only (no nvfp4 keys); the nvfp4 path runs standalone `scaled_fp4_quant` exactly like ours, and the +1.81 ms residual was a cross-profiler artifact | None — removed from the lever queue |
| Serving transport (TCP_NODELAY) | **DONE / MEASURED NEUTRAL on the gate workload** | Mirror landed (`SERVE-HTTP-TRANSPORT`): `set_tcp_nodelay(true)` matches vLLM's uvicorn/asyncio default; behavioral accepted-socket test RED **0** → GREEN **1**, 22/22 cases. The non-binding one-lock localhost A/B (`~/work/vllm.cpp-tcpnodelay-sizing/ff915e8…`, 4a450f9 Nagle-ON vs ff915e8 Nagle-OFF, c1/c2 ×2 reps, identical pinned-client workload; raw-set SHA `f5b52900…2128`) is **neutral within noise** on every ITL/TPOT/throughput metric (c1 mean ITL ~102.7 both arms; c2 ~108–109; first cold-start leg excluded). Mechanism: ~100 ms per-token write cadence vs µs loopback ACKs means Nagle never coalesces — the rescan's rank-1 gain hypothesis is REFUTED for the loopback gate; the mirror stays for real-network parity | None for the gate — decode-gap attribution moves to the nsys c2 full-step diff (transport is ruled out) |
| Host-weight ownership | **FAILED / WINDOWED-LOAD LANDED — VmHWM A/B PENDING** | The 2026-07-15 precheck (idle loaded server, `~/work/vllm.cpp-memory-precheck-20260715`) measured steady RSS **24.75 GB** (anonymous 24.58 GB ≈ exactly the 22.920 GiB mirror; file pages 129 MB; glibc arena retention ≤0.5 GB, allocator tweaks ruled out) while **VmHWM 48.29 GB ≈ the binding 48.17 GB peak** with zero requests served — the failing peak is the load-time overlap of the mirror build with full resident source mmap; steady RSS is already below vLLM's 28.5 GB peak. The `LOAD-SAFETENSORS` **windowed release** now drops each copied-then-dead source range during the copy loop (`ReleaseSourcePages` interior-page `madvise(MADV_DONTNEED)` on the clean `MAP_PRIVATE` source; every 27B/35B copy helper instrumented; page-lifetime proven copied-then-dead in [the spike](../.agents/specs/safetensors-windowed-load.md)). Default on; `VT_LOAD_WINDOWED_RELEASE=0` rolls back. CPU RED→GREEN tests (per-VMA smaps-Rss drop, byte-identity ON vs OFF, neighbor-page safety) green; loader-exercising suites green | Run the bounded DGX VmHWM A/B from the pushed SHA (`~/work/vllm.cpp-windowed-load/<sha>`, one flock): 27B load-to-ready with `VT_LOAD_WINDOWED_RELEASE=0` vs default, capture `/proc/<pid>/status` VmHWM + `smaps_rollup`, plus one c1 6-request smoke. Expect ≈48.3 GB off vs ≈25 GB on. **The binding Peak PSS/RSS axes remain FAILED until the authorized exact-grid rerun** — the A/B is a mechanism/effect check only, no axis credit. Direct-to-device streaming remains the deeper fix (removes the steady mirror; wanted for 35B) |
| Qwen3.6-35B-A3B performance | **BLOCKED / NOT RUN** | Correctness passes; no current v0.25.0 performance denominator exists | Run only after 27B reaches 124/124 |
| SGLang shared-prefix floor | **PENDING / NO ACCEPTED NUMBER** | No equivalent cache-on vllm.cpp/vLLM/SGLang campaign exists | After cache-off parity, gate equivalent vLLM v0.25.0 and SGLang v0.5.15; the faster reference binds each axis |
| External KV / LMCache | **NOT IMPLEMENTED / NOT BENCHMARKED** | Connector ABI and two-engine store/retrieve remain roadmap inventory | Spike fake-provider semantics, then gate LMCache MP before in-process mode |

### Active packed-decode implementation checkpoint

The semantic and operator gates are retained at clean `f18ca23` and
`9ad8fb7`; their complete hashes and attempt chronology live in the ledger and
state log. The current checkpoint is W1D3 component gating:

| Surface | Current evidence | Disposition |
|---|---|---|
| Exact vLLM boundary | Direct packed output/state BF16 differences **0/1** | PASS; upstream explicit reference has the same one-element state delta |
| Cache ABI | `MambaSpec` conv then temporal; gate allocation BF16 conv + FP32 SSM | PASS on registry/runner production-like tests |
| 27B default | **235/235 + 16/16**; prefill 0 packed calls, first decode exactly 48 | Immutable G2 PASS at `f344dec` |
| 27B rollback | `VT_GDN_PACKED_DECODE=0`, **235/235 + 16/16**, 0 packed calls | Immutable G2 PASS at `f344dec` |
| 35B native/batched | **315/315**, 0 packed calls | Immutable inertness PASS |
| 35B GGUF | Compact **14/14**, Balanced **14/14**, loader **98/98** | Immutable isolated-process PASS |
| Safety | CUDA GDN **43/43, 1,707/1,707**; packed/corner/FP16-SSM memcheck zero errors/leaks | Immutable PASS |
| W1D3 trace harness | Packed **915** nodes with 48 packed recurrence calls; rollback **963** with 48 decomposed + 48 post-conv calls; both retain 145 BF16 GEMMs, isolate exactly 48 mode-coupled BA projections, and require every remaining signature to match | Raw capture PASS at `7ff713e`: **12/12 + 12/12** exact ranges; both oracle traces structurally exact |
| Component runner | Production profile-control-off build; exact source/vLLM corpus manifests and partition hashes; full oracle/toolchain/artifact inventory; exact raw-sample throughput/TTFT/ITL recomputation plus bounded pinned-clock E2E/TPOT validation and duration-span consistency; frozen 64-plan fixture; fresh isolated server per leg; c2=6 requests and c16=96 requests; packed/rollback, rollback/packed, packed/rollback; one lock across exact-snapshot correctness and all 12 legs | Clean `d82d282` root `~/work/vllm.cpp-gdn-packed-component/d82d282f9efd1a5b97e7c6f1ac7a55b949849d09` is **FAILED / INCOMPLETE**: model gates and c2 6/6 complete; c16 packed r1 main batch 0/96 complete, 96/96 HTTP 500; no terminal marker |
| Performance | Marker-last finalization requires all **40 timing + 8 memory** median axes and all **144** paired run axes, ≤4% per-run deviation, fixed 1-GiB memory-return tolerance, parsed throttle counters, successful pinned GPU-idle probes, exact server/client/preflight commands and lifecycle markers. A stable regression is `complete-failed`; a sealable unstable/malformed run is `complete-void` | **FAILED / INCOMPLETE; NO SPEED CREDIT; every partial axis is nonbinding** |

Failure evidence is immutable. Order/run/execution/c16 raw/client/server SHA-256
values are `a2de5b07…6de0` / `297e3c62…a6fb` / `ff71c9f0…0684` /
`f8571d48…945c` / `8b9526f4…63a9` / `7b05e066…7dfe`. The C++ API wraps the
engine exception text into its JSON error body, but pinned vLLM's OpenAI
benchmark stores only `response.reason`; therefore the retained client evidence
says only `Internal Server Error`. Exact exception capture is the next
diagnostic, not a guessed runtime fix.

The current raw root is
`~/work/vllm.cpp-gdn-packed-trace/7ff713e377457130db4ed15929133d1b463aff96`.
Execution / configure / model-gate / run-log SHA-256 values are
`253ff089…a4bb` / `903d10a1…5c12` / `3ebdd9f9…bfca` /
`6c0c2cae…2235`; packed/rollback oracle trace hashes are
`4448c549…c293` / `9d918979…a420`. The first finalizer log
(`a71ac6e4…7e67`) records the expected fail-closed unknown-signature error.
Accepted summary / manifest / marker / pushed-finalizer-log hashes are
`bf5c04b7…702f` / `2e92b3a2…55c0` / `e1314019…8c1` /
`626c6844…8211`; artifact-set SHA is `ea286db4…c3dc`. Source, GPU and lock
returned clean/idle/free; the complete artifacts are retained.

Immutable evidence root:
`~/work/vllm.cpp-gdn-packed-decode/f344decf457a4d50c3bcae78a2903d7fe176a511/evidence-g2`.
Status is `complete-g2`; the complete one-lock order is frozen in
`run-plan.txt`, and its core entry points are:

```sh
flock /tmp/gpu build-cuda/tests/test_ops_gdn
flock /tmp/gpu build-cuda/tests/test_op_parity \
  -tc='qwen27 GDN packed decode boundary*'
flock /tmp/gpu build-cuda/tests/test_qwen27_paged_engine
flock /tmp/gpu env VT_GDN_PACKED_DECODE=0 \
  build-cuda/tests/test_qwen27_paged_engine
flock /tmp/gpu build-cuda/tests/test_qwen36_paged_engine
```

G2 and W1D3 structural evidence are closed. The repaired source has been pushed
and launched exactly once at `d82d282`; that incomplete root must never be
reused or appended. The failure inspection and next diagnostic entry point are
in the newest [state record](../.agents/state.md) entry and the
[spike](../.agents/specs/gdn-packed-decode.md). The failed launch provenance is:

```sh
set -euo pipefail
SOURCE_REPO="$HOME/work/vllm.cpp"
git -C "$SOURCE_REPO" fetch origin main
SHA=d82d282f9efd1a5b97e7c6f1ac7a55b949849d09
ROOT="$HOME/work/vllm.cpp-gdn-packed-component/$SHA"
BINDING="$HOME/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560"
SNAPSHOT="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a"
test ! -e "$ROOT"
mkdir -p "$(dirname "$ROOT")"
git -C "$SOURCE_REPO" worktree add --detach "$ROOT/source" "$SHA"
mkdir -p "$ROOT/evidence/corpus"
cp -a "$BINDING/corpus/27" "$ROOT/evidence/corpus/27"
cmake -S "$ROOT/source" -B "$ROOT/build-production" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$HOME/venvs/vllm-oracle/bin/ninja" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_FLASH_ATTN=ON \
  -DVLLM_CPP_TRITON=ON -DVLLM_CPP_TRITON_REGEN=OFF \
  -DVLLM_CPP_BENCH_PROFILE_CONTROL=OFF \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tee "$ROOT/configure.log"
"$ROOT/source/scripts/dgx-gdn-packed-component.sh" --execute \
  --snapshot "$SNAPSHOT" \
  --source-corpus "$ROOT/evidence/corpus/27" \
  --evidence "$ROOT/evidence" \
  --build-dir "$ROOT/build-production" \
  --configure-log "$ROOT/configure.log" \
  --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA"
```

`$ROOT/source` is the clean detached worktree at `$SHA`. The request corpus is
copied byte-for-byte from the binding `3f256ab` evidence; the separate 64-plan
fixture is committed under `tests/fixtures/` and the runner requires it in
read-only mode. Until the component gate passes, `benchmark_binding=false`.

The **next entry point** (repair now landed) is the DGX correctness gates on the
pushed fix SHA — the default+rollback 27B **235/235 + 16/16** direct model gates
plus the `--diagnostic-c16` reproduction (which must now pass all three c16 reps
instead of throwing) — followed by a full fresh 12-leg `--execute` component
rerun from a NEW root (the `d82d282` component root is immutable and must never
be reused). The `--diagnostic-c16` recipe below is retained as the focused
regression check: it reruns the packed c16 boundary three times under one GPU
lock, captures any `engine-fatal:` root cause on stderr (server log), and
replays the corpus row into a persisted HTTP body — writing only
`component-diagnostic.json` and a `diagnostic/` subtree, never a sealable
component. Provision `SNAPSHOT`/corpus/build exactly as above, then, with a new
root whose basename contains `diagnostic-c16`:

```sh
SHA=<pushed diagnostic SHA>
ROOT="$HOME/work/vllm.cpp-gdn-packed-diagnostic-c16/$SHA"
# ... provision $ROOT/source, $ROOT/evidence-diagnostic-c16/corpus/27,
#     $ROOT/build-production, $ROOT/configure.log exactly as the --execute recipe
"$ROOT/source/scripts/dgx-gdn-packed-component.sh" --diagnostic-c16 \
  --snapshot "$SNAPSHOT" \
  --source-corpus "$ROOT/evidence-diagnostic-c16/corpus/27" \
  --evidence "$ROOT/evidence-diagnostic-c16" \
  --build-dir "$ROOT/build-production" \
  --configure-log "$ROOT/configure.log" \
  --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA"
```

The mode asserts the evidence basename contains `diagnostic-c16` and holds no
`component-*.json`; it runs no model gates, no 2/16 sweep, and never calls
`finalize`. `benchmark_binding` stays `false` throughout.

This reproduction has been executed once at `4a450f9` (root
`~/work/vllm.cpp-gdn-packed-diagnostic-c16/4a450f9…`, build **154/154**,
marker-last `diagnostic_complete`, GPU/lock returned idle/free). All three
fresh-server reps failed identically with
`engine-fatal: … duplicate live GDN state index at qwen3_5.cpp:73`.
Core SHA-256: `component-diagnostic.json` `42de1323…13ea`; r1/r2/r3 server
logs `f26f0030…8bc0` / `8ecba873…02a3` / `e68411cf…6f3f`; r1 error body
`c5aa0933…7fc3`. The root is diagnostic evidence only and is preserved
unchanged; the repair checkpoint and a fresh SHA/root full component rerun
are next.

Closed controls do not remain active leaves: async scheduling measured neutral
for speed, and the prior fused-producer candidate remains default-off after its
strict component failure. Their exact results and reproduction history remain
in the append-only record rather than this live scoreboard.

The completed core correctness/safety root is
`~/work/vllm.cpp-gdn-ba/immutable-581d335fec2e5a96d9ccbb38c1ec001c39ac1789`.
Status / artifact-list SHA-256 values are `3895e658…4cf6` / `ed2bf8d8…895b`.
Focused CTest, merged/split 27B, 35B, and memcheck log hashes are
`4cf699ad…759b`, `c2a6f93f…cf96` (both arms), `b926716e…9875`, and
`a3d61cb9…fb87`; fixture `e81e9181…7edd` loaded 64 plans and the forbidden
native cache stayed absent. The rejected BF16-output preflight log remains
`09078b76…b050`. This closes immutable core correctness/safety only and earns no
speed ratio.

The completed structural root is
`~/work/vllm.cpp-gdn-ba-trace/0091cd192d9a6baa2197a4f3bdb0561bd859baf5`.
All 24 local ranges pass their exact contracts. The merged/split oracle traces
have SHA-256 `b8d26d4c…fc59` / `cef841ce…ede5`; they contain 1,522 / 1,521
internally invariant steady B=2 windows at 1,160 kernels and ordered-name SHA
`858915dd…fad0`. Their full launch signatures are `17e1037e…14ed` /
`f7a3ca1f…cadf`. Pushed finalizer commit `8a1f923` wrote the marker last with
status `complete-structural`. Summary / manifest / marker / artifact-set /
finalizer SHA-256 values are `03601168…54d5` / `b203f0d2…5412` /
`72328c48…63e` / `b93fd633…70a2` / `57395e99…b146`. The summary proves merged
963/145 versus split 1,011/193, exact 48/48 deltas, unchanged non-BF16 family
counts and `benchmark_binding=false`; it grants no speed ratio.

The current W1C hardware root is
`~/work/vllm.cpp-gdn-ba-rounding/f9252943d1e96dbfa43e3b8f2d06dec1aa5f20d3/evidence/w1c-correctness-inertness`.
The source is clean at exact `f925294`, and the binary SHA-256 values are frozen
in `provenance.txt`. Projection / loader / native-35B / real-GGUF / default-27B /
BF16-27B log SHA-256 values are
`a791c567…37d1` / `d455b8fc…05f6` / `72caeca9…06c` /
`87833f22…af8d` / `da5dd836…091e` / `148d743f…86a`.
The final BF16 assertion failure intentionally prevents `status.txt`,
`sha256sum.txt`, and the terminal event from being written. GPU and lock were
verified idle afterward. This is a valid **FAILED** correctness checkpoint,
not a partial performance number.

## Evidence selecting merged GDN projections

The immutable completed root is
`~/work/vllm.cpp-executed-path-c2/179a0fc2afc1c33b63d14de8e50d3fde976c7356`.
Its status is `complete-diagnostic`.

| Artifact | SHA-256 |
|---|---|
| c2 summary | `0ef6a1240d33c16410cd4e43b30ca8667a6d92e6eee8506d7bd03388fe010273` |
| c2 manifest | `2556cfd032fae2201d9f8deb818343731b7dc99d9f8e6329da9b793262712f21` |
| status | `9e0143fa1b9c74e218e486fedd0606850708619a0e859dafe94957e24a507b57` |
| artifact set | `cc248ad2b5bf08f85b0d6b178de70682a104917e16c59c9adf34d661217f823a` |
| fresh oracle trace | `2b3bf41269fd19ef65c5c3e06f067af73d7d997de3b6be17a2af785b6a86785c` |

All **12/12** local B=2 graph ranges are invariant at 1,011 kernels + 7
memcpy + 1 memset. The oracle contains **1,522** invariant steady B=2 windows
at 1,160 kernels plus two bounded B=1 drains. Both engines execute the same
**128 Stream-K + 80 static-persistent** FP4 tactic split.

| BF16 projection structure per B=2 window | vllm.cpp | vLLM v0.25.0 |
|---|---:|---:|
| qkv / packed qkvz | 48 | 48 |
| z | 48 | included in qkvz |
| b+a / packed ba | 96 | 48 |
| lm_head | 1 | 1 |
| Total | **193** | **97** |

The source arithmetic independently gives `(4-2) × 48 = 96`; this is not a
profiler-name classification artifact. Diagnostic family medians are
**51.662672 vs 48.798042 ms** (+2.864630 ms), with a shape-level decomposition
ranking BA at about 1.882 ms and qkvz at about 0.476 ms. These durations cross
Nsight/Torch-profiler domains and are not a speed ratio. The accepted spike
therefore gates BA and qkvz separately and forbids duplicate weight owners or
split-copy kernels.

## Verify or reproduce the current checkpoint

Verify the durable diagnostic without GPU work:

```sh
RAW_SHA=179a0fc2afc1c33b63d14de8e50d3fde976c7356
ROOT="$HOME/work/vllm.cpp-executed-path-c2/$RAW_SHA/evidence/$RAW_SHA/trace/27"
sha256sum "$ROOT"/{c2-summary.json,c2-manifest.json,status-c2.json}
# Expected prefixes: 0ef6a124…0273 / 2556cfd0…2f21 / 9e0143fa…7b57
```

Verify the pushed-SHA core correctness/safety checkpoint without rerunning the GPU:

```sh
SHA=581d335fec2e5a96d9ccbb38c1ec001c39ac1789
ROOT="$HOME/work/vllm.cpp-gdn-ba/immutable-$SHA"
test "$(git -C "$ROOT/src" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$ROOT/src" status --porcelain)"
sha256sum "$ROOT/evidence"/{status.txt,sha256sums.txt}
# Expected: 3895e658…4cf6 / ed2bf8d8…895b
```

Verify the durable structural checkpoint without GPU work:

```sh
RAW_SHA=0091cd192d9a6baa2197a4f3bdb0561bd859baf5
ROOT="$HOME/work/vllm.cpp-gdn-ba-trace/$RAW_SHA"
TRACE="$ROOT/evidence/$RAW_SHA/trace/27"
sha256sum "$TRACE"/{gdn-ba-summary.json,gdn-ba-manifest.json,status-gdn-ba.json}
# Expected: 03601168…54d5 / b203f0d2…5412 / 72328c48…63e
```

Verify the current fail-closed W1C checkpoint without rerunning the GPU:

```sh
SHA=f9252943d1e96dbfa43e3b8f2d06dec1aa5f20d3
ROOT="$HOME/work/vllm.cpp-gdn-ba-rounding/$SHA"
E="$ROOT/evidence/w1c-correctness-inertness"
test "$(git -C "$ROOT/source" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$ROOT/source" status --porcelain)"
test ! -e "$E/status.txt"
sha256sum "$E"/{01-projection.log,02-qwen36-weights.log,03-qwen36-native.log,04-qwen36-gguf.log,05-qwen27-default.log,06-qwen27-bf16.log}
```

The exact single-lock command below reproduces the accepted subgates and final
233/235 failure. Any partial or unlocked execution is void:

```sh
SHA=$(git rev-parse HEAD)
ROOT="$HOME/work/vllm.cpp-gdn-ba-rounding/$SHA"
SOURCE="$ROOT/source"
BUILD="$ROOT/build-cuda"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$SOURCE" status --porcelain)"
flock /tmp/gpu bash -lc "
  set -euo pipefail
  '$BUILD/tests/test_op_parity' \
    -tc='qwen27 GDN BA BF16 projection matches vLLM 0.25 oracle*'
  '$BUILD/tests/test_qwen36_weights'
  '$BUILD/tests/test_qwen36_paged_engine'
  '$BUILD/tests/test_qwen36_gguf_engine'
  '$BUILD/tests/test_qwen27_paged_engine'
  VT_GDN_BA_OUT_BF16=1 '$BUILD/tests/test_qwen27_paged_engine'
"
```

The prior end-to-end failure remains reproducible and is the model-level RED
precondition for the packed production dispatch:

```sh
SHA=f9252943d1e96dbfa43e3b8f2d06dec1aa5f20d3
BUILD="$HOME/work/vllm.cpp-gdn-ba-rounding/$SHA/build-cuda"
flock /tmp/gpu env VT_GDN_BA_OUT_BF16=1 \
  "$BUILD/tests/test_qwen27_paged_engine"
# Current disposition: 233/235; got == greedy_ids_emulation.npy
```

The accepted W1D3 finalization is reproducible without recapturing the GPU
series. Its marker records both the finalizer and imported launch-validator
hashes:

```sh
set -o pipefail
RAW_SHA=7ff713e377457130db4ed15929133d1b463aff96
FINALIZER_SHA=24cea4f1fe28c89968cad1ed845fbfbd64514b0c
ROOT="$HOME/work/vllm.cpp-gdn-packed-trace/$RAW_SHA"
EVIDENCE="$ROOT/evidence/$RAW_SHA"
SOURCE="$HOME/work/vllm.cpp-gdn-packed-finalizer/$FINALIZER_SHA/source"
VERIFY="$ROOT/finalizer-replay-$FINALIZER_SHA"
test -z "$(git status --porcelain)"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$FINALIZER_SHA"
test ! -e "$VERIFY"
cp -a --reflink=auto "$EVIDENCE" "$VERIFY"
rm "$VERIFY/trace/27/gdn-packed-summary.json" \
  "$VERIFY/trace/27/gdn-packed-manifest.json" \
  "$VERIFY/trace/27/status-gdn-packed.json"
PYTHONPATH="$SOURCE" python3 \
  "$SOURCE/tools/bench/finalize_gdn_packed_trace.py" \
  --evidence "$VERIFY" --source-commit "$RAW_SHA" \
  --run-log "$ROOT/gdn-packed-run.log" \
  2>&1 | tee "$ROOT/gdn-packed-finalizer-replay.log"
```

Acceptance requires packed **915** versus rollback **963** total nodes, 145
BF16 GEMMs in both arms, 48 packed recurrence calls replacing 48 decomposed +
48 post-conv calls, exactly 48 BA projection nodes at the accepted `(8,1,1)`
geometry in each arm (hashed separately because BF16-vs-F32 output may change
the cuBLASLt tactic), and an identical normalized signature multiset for every
remaining kernel/memcpy/memset node.
Exact requirements are in the
[packed-decode spike](../.agents/specs/gdn-packed-decode.md).

Re-aggregate the binding result without GPU work; exit **1** is expected for a
complete every-axis failure, while exit 2 means malformed evidence:

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
```

## Benchmark policy

- Correctness is a precondition and cannot be traded for speed.
- Ours and every reference use the same model, requests, sampling, token
  budget, cache policy, concurrency, and hardware.
- A benchmark series runs on an idle GPU under one ownership window and is
  repeated enough to distinguish signal from run noise.
- Partial, contended, stale-denominator, or diagnostically incomplete results
  are `VOID`; they never contribute to an accepted ratio.
- Every 27B throughput, latency, and memory axis must pass before 35B
  performance or broader roadmap execution.

The complete contract is in the
[benchmark protocol](../.agents/benchmark-protocol.md) and
[online serving gate spec](../.agents/specs/cuda-online-serving-gate.md).
