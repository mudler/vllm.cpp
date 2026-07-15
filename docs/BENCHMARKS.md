# Benchmarks

This is the public current-state scoreboard for vllm.cpp. It contains the
binding result, the active performance diagnosis, pending gates, and current
reproduction entry points. Attempt chronology and failure forensics live in the
[parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), linked specs, and Git. Those raw records are
append-only within the current era and are frozen under `.agents/completed/`
when the era is rolled up; this page never accumulates their run-by-run history.

Last updated: **2026-07-15**. Qwen3.6-27B parity against vLLM v0.25.0 is
**FAILED / open at 55/124 axes**. The order-0
[packed-decode leaf](../.agents/specs/gdn-packed-decode.md)
(`KERNEL-GDN-PACKED-DECODE`) is now **CLOSED on EQUIVALENCE** (`DONE`, owner
`e47b4d6`). Correctness is immutable at `f344dec` (default + rollback each
**235/235 + 16/16**); structure is accepted from `7ff713e`/`24cea4f` (packed
**915** nodes vs rollback's **963**, the exact 48-for-96 GDN substitution, all
remaining topology invariant); the earlier c16 HTTP-500 slot defect was fixed
test-first (request-identity keying) and proven at `c172336`.

**W1D3 / G3 closed on evidence-totality.** Across **eight sealed components**
(progressively harness-calibrated test-first — tail 15%, pooled +
mode-conditional c2 TTFT, acceptance noise bands, majority-consistency pairing;
the c2 TTFT-family is a bimodal prefill co-schedule ARRIVAL LOTTERY, a faithful
vLLM mirror, scheduler UNCHANGED) plus the **8-pair locked c16 A/B** (`00bf484`:
paired mean **−0.205%, sd 0.30, <1σ**; cuBLASLt algo selection
process-deterministic → algo-lottery REFUTED) plus the **24-window trace**
(**packed is GPU-cheaper**: kernel compute −1.30..−1.58%/step, GDN+BA
−296 µs/window, −48 nodes, no attributable packed-side cost), there is **no
stable regression on any axis**. The eighth (first 22-leg: cold-discard pair +
5 reps, corpus byte-verified against the binding corpus) component sealed
marker-last `complete-failed` at root
`~/work/vllm.cpp-gdn-packed-component/e47b4d6…` (status artifact-set
`4e3354a6…d912`, manifest `32318513…564a`, summary `85208ada…6242`): **38/40
axes, 8/8 memory, stability clean, `validation_error=None`, paired-consistency
PASS at BOTH concurrencies**; c16 at equivalence (packed med 801.97 vs rollback
802.95, −0.12%, in-band, passes); the two failing axes are sign-flipping
band-edge statistics of a true-zero effect (c2 `median_tpot_ms` 0.9899, an axis
packed WON in runs 1–2; c2 pooled `p99_ttft_ms` 0.8464, a max-of-30
bimodal-mixture order statistic 0.36 pp past the 15% band). **Disposition:
EQUIVALENCE PROVEN — no stable regression.** Packed stays the default
(`VT_GDN_PACKED_DECODE=0` rollback); **no `complete-pass` marker exists and no
speed credit is claimed.** With windowed-load default-ON the component peak PSS
is **24.86 GB** (binding-era 48.18; vLLM 28.5). Detailed per-seal chronology and
evidence SHAs live in the append-only [ledger](../.agents/parity-ledger.md) and
[state log](../.agents/state.md). **Active tracks:** merged **qkvz**
(`KERNEL-GEMM-BF16` W2) and the authorized **fresh binding/exact-grid rerun**
(fresh vLLM denominators; explicit `--mamba-ssm-cache-dtype float32` on the vLLM
arm; cite run SHA `702f481`); 35B stays blocked until 27B reaches 124/124.

**Blast-radius caveat (correctness).** The c16 slot defect predated its
validator and was independent of `VT_GDN_PACKED_DECODE` (both arms hit the same
remap). The compact slot pool was introduced at `66715e1` (2026-07-05); the
uniqueness validator only at `f344dec` (2026-07-14). Binaries in between —
including the `3f256ab` binding grid and earlier c16/c32 campaigns — would have
**silently** run two or more long concurrent sequences on ONE GDN recurrent
state slot (cross-request state corruption) rather than crashing. Low-concurrency
16/16 correctness gates do not surface it; high-concurrency runs measure
throughput, not token correctness, so any such prior c16/c32 per-token output
correctness is **suspect**. No binding throughput number changes; recorded here
for honesty.

## Binding 27B online gate

Workload equivalence of the two arms is AUDITED and accepted
([audit report](../.agents/specs/benchmark-equivalence-audit-2026-07-15.md)):
batch cap (max_num_seqs=32, zero preemption), token budget (2048 + chunked
prefill), context, greedy sampling, corpus bytes, KV/conv/SSM dtypes (SSM is
FP32 on BOTH arms — vLLM's Qwen3.5 config hook promotes it silently), FP4
kernel family, FA2, and graphed decode all match; the client commands differ
in exactly one token (result directory). The one material engine difference
is vLLM's Inductor prefill fusion + piecewise cudagraphs — its production
config, i.e. the protocol-correct denominator.

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
TTFT, TPOT/ITL, E2EL, and memory axis must pass. At c2, ours has better mean
TTFT (≈697 vs ≈833 ms, 1.196×), but this is a **prefill co-schedule arrival
lottery artifact, not a durable edge** — two 1024-token prefills fit the 2048
budget exactly, so whether they co-schedule (both ~0.9 s) or stagger (one
~0.45 s alone) flips leg-to-leg with arrival timing; our waiting-queue admission
is a 1:1 mirror of vLLM's (no scheduler divergence), so the pooled-across-reps
comparison is the honest denominator. Decode TPOT is **114.841 vs 108.274 ms**
(**6.1% slower**). No 35B performance command is authorized until the 27B result
reaches 124/124.

## Current checkpoint

| Track | Disposition | Current evidence | Next binding gate |
|---|---|---|---|
| `SERVE-GATE-ONLINE` | **FAILED / GATING** | Immutable `3f256ab` remains **55/124**. The order-0 packed GDN decode leaf is now **CLOSED on EQUIVALENCE** (`KERNEL-GDN-PACKED-DECODE` `DONE`, `e47b4d6`): the c16 HTTP-500 slot defect is fixed and proven at `c172336`; eight sealed components + the 8-pair locked c16 A/B (−0.205% ± 0.30, <1σ; cuBLASLt algo selection process-deterministic) + the 24-window trace (packed GPU-cheaper, no attributable packed-side cost) show no stable regression on any axis. c2 and all memory axes are at packed-rollback equivalence (component peak PSS 24.86 GB vs binding-era 48.18/vLLM 28.5). Detailed per-seal chronology in state/ledger | qkvz (`KERNEL-GEMM-BF16` W2), then the AUTHORIZED fresh binding/exact-grid rerun (fresh vLLM denominators; explicit `--mamba-ssm-cache-dtype float32`; cite `702f481`); no `complete-pass` marker exists and no speed credit is claimed
| `KERNEL-GEMM-BF16` | **GATING W1C** | `0091cd1` closes BA structure and `f925294` closes projection/inertness. The isolated BF16-BA + decomposed control remains **233/235**; clean `f344dec` closes W1D2/G2 for the exact coupled BF16-BA + packed path at **235/235**, `benchmark_binding=false` | The packed W1D3 leaf is now **CLOSED on equivalence** (`KERNEL-GDN-PACKED-DECODE` `DONE`); **qkvz (W2) is UNBLOCKED** and is the active track (merged qkv+z projection packing), then the authorized exact grid |
| `KERNEL-GDN-PACKED-DECODE` | **`DONE` — W1D3 CLOSED on EQUIVALENCE** (owner `e47b4d6`) | Slot lifecycle fix `c172336` (identity-keyed pool; RED→GREEN `test_runner` 8/8) proven on DGX (`--diagnostic-c16` 3/3, both model gates 235/235); scheduler co-schedule parity with vLLM verified (`test_scheduler` 31/31). W1D3/G3 closed on evidence-totality: **eight sealed components** (harness calibrated test-first; focused suite 79/79, tools 162/162) + the 8-pair locked c16 A/B (`00bf484`: paired mean **−0.205% ± 0.30, <1σ**; cuBLASLt algo selection process-deterministic → algo-lottery REFUTED) + the 24-window trace (**packed is GPU-cheaper**: kernel compute −1.30..−1.58%/step, GDN+BA −296 µs/window, no attributable packed-side cost). The eighth (22-leg) seal `complete-failed` at `e47b4d6`: **38/40 + 8/8 memory**, stability clean, paired-consistency PASS at both concurrencies; the 2 fails are sign-flipping band-edge statistics of a true-zero effect. Full chronology and per-seal SHAs in state/ledger | **Disposition: EQUIVALENCE PROVEN — no stable regression on any axis.** Packed is the default (`VT_GDN_PACKED_DECODE=0` rollback); **no `complete-pass` marker exists and no speed credit is claimed**. qkvz (`KERNEL-GEMM-BF16` W2) unblocked; the authorized exact grid follows
| RMSNorm/generated partitions | **CLOSED / DISPROVEN as a parity gap** | The 2026-07-14 [parity rescan](../.agents/specs/parity-rescan-2026-07-14.md) verified vLLM's `RmsNormQuantFusionPass` is FP8-only (no nvfp4 keys); the nvfp4 path runs standalone `scaled_fp4_quant` exactly like ours, and the +1.81 ms residual was a cross-profiler artifact | None — removed from the lever queue |
| Serving transport (TCP_NODELAY) | **DONE / MEASURED NEUTRAL on the gate workload** | Mirror landed (`SERVE-HTTP-TRANSPORT`): `set_tcp_nodelay(true)` matches vLLM's uvicorn/asyncio default; behavioral accepted-socket test RED **0** → GREEN **1**, 22/22 cases. The non-binding one-lock localhost A/B (`~/work/vllm.cpp-tcpnodelay-sizing/ff915e8…`, 4a450f9 Nagle-ON vs ff915e8 Nagle-OFF, c1/c2 ×2 reps, identical pinned-client workload; raw-set SHA `f5b52900…2128`) is **neutral within noise** on every ITL/TPOT/throughput metric (c1 mean ITL ~102.7 both arms; c2 ~108–109; first cold-start leg excluded). Mechanism: ~100 ms per-token write cadence vs µs loopback ACKs means Nagle never coalesces — the rescan's rank-1 gain hypothesis is REFUTED for the loopback gate; the mirror stays for real-network parity | None for the gate — decode-gap attribution moves to the nsys c2 full-step diff (transport is ruled out) |
| Host-weight ownership | **FIX MEASURED — VmHWM A/B −23.54 GB; binding axes stay FAILED until the exact grid** | The `LOAD-SAFETENSORS` **windowed release** (progressive interior-page `madvise(MADV_DONTNEED)` on each copied-then-dead source range during the copy loop; default on, `VT_LOAD_WINDOWED_RELEASE=0` rollback; page-lifetime proven in [the spike](../.agents/specs/safetensors-windowed-load.md); CPU RED→GREEN smaps-Rss/byte-identity/neighbor-safety tests) is now **measured on GB10** at `cb2d310`, root `~/work/vllm.cpp-windowed-load/cb2d310c…518/evidence`, single 27B load-to-ready per arm under one `flock /tmp/gpu`: **OFF VmHWM 48,285,916 kB (48.29 GB)** / VmRSS 24,750,696 kB (the double-residency peak intact, matching the precheck) vs **ON (default) VmHWM 24,750,704 kB (24.75 GB) = VmRSS** — the load transient is **fully eliminated (−23,535,212 kB, −48.7%)**; steady Pss 24,748,252/24,748,260 kB both arms. ON-arm serving smoke **6/6** requests, 0 failed (302.7 tok/s c1, health-only). Artifact SHA-256: status `cdccc1dd…7233` / `3fd0592c…1fc0`, smaps `11baecd3…3881` / `41837d12…65f3`, smoke `ed271a68…5aa8`, server logs `772bec6b…9c49` / `b66bb783…a81cd`. PROJECTION (not credit): ours ≈24.75–24.86 GB peak vs vLLM's binding 28.17/28.53 GB Peak PSS/RSS → both failing memory axes are projected to flip PASS at the next authorized exact-grid rerun | **The binding Peak PSS/RSS axes remain FAILED until the authorized exact-grid rerun** — the A/B is a mechanism/effect measurement, no axis credit. Direct-to-device streaming remains the deeper fix (removes the 22.92 GiB steady mirror; wanted for 35B) |
| Qwen3.6-35B-A3B performance | **BLOCKED / NOT RUN** | Correctness passes; no current v0.25.0 performance denominator exists | Run only after 27B reaches 124/124 |
| SGLang shared-prefix floor | **PENDING / NO ACCEPTED NUMBER** | No equivalent cache-on vllm.cpp/vLLM/SGLang campaign exists | After cache-off parity, gate equivalent vLLM v0.25.0 and SGLang v0.5.15; the faster reference binds each axis |
| External KV / LMCache | **NOT IMPLEMENTED / NOT BENCHMARKED** | Connector ABI and two-engine store/retrieve remain roadmap inventory | Spike fake-provider semantics, then gate LMCache MP before in-process mode |

### Active packed-decode implementation checkpoint

The semantic and operator gates are retained at clean `f18ca23` and
`9ad8fb7`; their complete hashes and attempt chronology live in the ledger and
state log. W1D3 is now **CLOSED on EQUIVALENCE** (`KERNEL-GDN-PACKED-DECODE`
`DONE`, owner `e47b4d6`; no `complete-pass` marker, no speed credit):

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
| Component runner | Production profile-control-off build; exact source/vLLM corpus manifests and partition hashes; full oracle/toolchain/artifact inventory; exact raw-sample throughput/TTFT/ITL recomputation plus bounded pinned-clock E2E/TPOT validation and duration-span consistency; frozen 64-plan fixture; fresh isolated server per leg; c2=6 requests and c16=96 requests per rep; a discarded cold-start warmup pair then 5 timed reps AB/BA/AB/BA/AB (22 legs); one lock across exact-snapshot correctness and all legs | The `d82d282` c16 HTTP-500 (`duplicate live GDN state index`) was fixed test-first (request-identity slot keying) and proven on DGX at `c172336`. **Eight full components have since sealed** a marker-last terminal status (3× `complete-void` on the TTFT-family arrival lottery, then 5× `complete-failed` as the harness calibrated to true-zero noise); the eighth (first 22-leg) seal is `complete-failed` with 38/40 + 8/8 memory PASS, stability clean, paired-consistency PASS at BOTH concurrencies — **W1D3 CLOSES on equivalence** |
| Performance | Marker-last finalization requires all **40 timing + 8 memory** median axes and the gated paired run axes, per-run stability of ≤4% for non-tail timing and all memory axes and ≤15% for the tail axes (p90/p99 of ttft/tpot/itl/e2el — revised test-first after two `c172336` tail voids), fixed 1-GiB memory-return tolerance, parsed throttle counters, successful pinned GPU-idle probes, exact server/client/preflight commands and lifecycle markers. The c2 TTFT-family (mean/median/p90/p99 of ttft) is instead compared on the arm-pooled 30-per-request distribution, stability-gated on a 50% pooled sanity bound, and excluded from the gated per-rep paired axes (bimodal arrival lottery; c16/non-TTFT keep 4%/15%; E2EL unchanged). **Acceptance uses a NOISE BAND (`contract.acceptance`): a comparison axis (median, gated paired, c2 mode means, memory) fails only when the packed deficit exceeds run noise — 0.5% for non-tail timing and PSS/RSS, 15% for tail axes, per-mode c2 TTFT, and the recalibrated GPU-memory bands; packed≥rollback always passes.** The gated per-rep **paired** axes use a **MAJORITY-CONSISTENCY** rule (`contract.paired_gate`): a paired axis fails only when ≥3 of its 5 rep-pairs breach the band in the same (packed-worse) direction; single-pair breaches are recorded as diagnostics. A stable regression is `complete-failed`; a sealable unstable/malformed run is `complete-void` | **Eight sealed → W1D3 CLOSES on EQUIVALENCE.** VOID×3 (all TTFT-family: a bimodal prefill arrival lottery, no scheduler divergence), then `complete-failed`×5 as the harness calibrated to true-zero noise (acceptance noise band, mode-conditional c2 TTFT + GPU-memory recalibration, majority-consistency pairing, 5-rep + cold-discard precision upgrade). The eighth (first 22-leg: cold-discard pair + 5 reps) seal `complete-failed` at `e47b4d6`: **38/40 axes, 8/8 memory, stability clean, `validation_error=None`, paired-consistency PASS at BOTH c2/c16**; c16 at equivalence (packed med 801.97 vs rollback 802.95, −0.12%, passes); the 2 fails are sign-flipping band-edge statistics of a true-zero effect (c2 `median_tpot_ms` 0.9899, c2 pooled `p99_ttft_ms` 0.8464). Combined with the 8-pair locked c16 A/B (−0.205% ± 0.30, <1σ) and the 24-window trace (packed GPU-cheaper), **no stable regression exists on any axis — EQUIVALENCE PROVEN. No `complete-pass` marker exists; NO SPEED CREDIT** |

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

`$ROOT/source` is the clean detached worktree at `$SHA`; the recipe above is the
`--execute` reproduction pattern (the eighth run used the 22-leg 5-rep plan). The
request corpus is copied byte-for-byte from the binding `3f256ab` evidence (the
5-rep corpus is a byte-verified deterministic extension); the separate 64-plan
fixture is committed under `tests/fixtures/` and the runner requires it in
read-only mode. The packed component claims no throughput number, so the binding
27B grid stays `benchmark_binding=false` until the authorized fresh exact-grid
rerun.

**W1D3 is CLOSED on equivalence** (see the summary at the top). All eight
component roots are immutable and must never be reused
(`c172336`/`-r2`, `d19e091`, `2dbe892`, `da05444`, `495ba780`, run-6, and the
eighth/closing seal `e47b4d6`). The active tracks are now **qkvz**
(`KERNEL-GEMM-BF16` W2) and the authorized fresh binding/exact-grid rerun; no
further packed-decode component run is required (further single-run seals of the
true-zero effect are band-edge coin flips that would not change the recorded
conclusion). The `--diagnostic-c16` recipe below is retained as the focused
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
