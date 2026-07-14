# Qwen GDN packed pure-decode recurrence

**Row:** `KERNEL-GDN-PACKED-DECODE` · **consumers:**
`KERNEL-GEMM-BF16`, `SERVE-GATE-ONLINE` · **status:** spike complete,
implementation `ACTIVE` under `CLAIM-GDN-BA-ROUNDING-1`; W1D1/G1 closed at
clean `9ad8fb7`; clean pushed `f344dec` closes W1D2 model dispatch and
immutable G2; W1D3 finalizer/component is gating · **priority:** roadmap order 0.
Fresh clean `7ff713e`, finalized by pushed `24cea4f`, closes exact marker-last
structural evidence. The production-build-only c2/c16 AB/BA/AB driver and
marker-last every-axis finalizer are implemented and focused CPU tests pass
**49/49**, with all tool tests **132/132**. Clean pushed `d82d282` completed both
direct model gates and all six c2 legs, then failed incomplete at c16 packed r1:
preflight and 16 warmups passed, all **0/96** timed requests returned HTTP 500,
and no marker-last status was sealed. The root cause was unrecoverable because
our port dropped two upstream fatal log lines; a bounded test-first diagnostic
checkpoint now restores them (four `std::cerr` error-path channels +
`VT_GDN_DIAG_STEP_LOG` geometry + a packed-only `--diagnostic-c16` driver mode;
see the W1D3 component-harness row and G3 below). Partial legs are nonbinding.

The W1C projection oracle proved that the 27B BF16 `in_proj_ba` output is
bit-identical to vLLM, but the existing decomposed consumer still produces a
different decode stream. The first-boundary oracle now identifies the exact
cause: vLLM's default pure-decode path consumes raw packed q/k/v, normalizes q
and k in F32 inside the recurrent kernel, and rounds `sigmoid(b)` through the
input dtype before recurrence. Our path materializes normalized q/k in BF16
and retains beta in F32. The selected fix is the complete upstream packed
pure-decode operation, not a beta-only approximation.

## Scope

### In scope

- A public `vt::` packed GDN decode operation with a portable CPU reference
  and a CUDA implementation grounded in vLLM's Triton kernel.
- Raw row-strided `mixed_qkv [B, 2*Hk*Dk + Hv*Dv]`, `a/b [B,Hv]`,
  `A_log/dt_bias [Hv]`, output `[B,Hv,Dv]`, persistent state
  `[slots,Hv,Dv,Dk]`, and one state index per decode token. Activation/output,
  state, `A_log`, and `dt_bias` storage dtypes are independently validated;
  the real 27B boundary is BF16 activation/output + FP32 SSM/`A_log` + BF16
  `dt_bias`.
- The upstream FP16, BF16 and F32 activation/input modes. Gate-model execution
  is BF16; F32 remains the compatibility/reference mode. All arithmetic is
  F32, with output and state rounded to their declared storage dtype.
- Exact packed semantics: q/k loads are normalized in F32 and are never
  materialized through BF16 before recurrence; beta is
  `sigmoid(float(b)) -> b.dtype -> float`; g uses the upstream threshold-20
  softplus expression; q alone receives `Dk^-0.5`.
- Pure non-spec decode selection only: CUDA, no spec masks, zero prefills and
  at least one decode. Mixed decode+prefill, prefill-only and speculative
  branches retain their separately defined upstream paths.
- `VT_GDN_PACKED_DECODE=0` as the process-cached rollback. The packed path is
  default-on only after its operator, real-model, capture, trace and
  same-binary gates pass.
- The 27B BF16 BA-output default transition coupled to the packed path. The
  legacy F32 BA/decomposed arm remains the correctness-preserving rollback;
  35B selection stays byte-inert.

### State-index adapter

vLLM reserves cache slot 0 as `NULL_BLOCK_ID` and skips indices `<=0`. The
local cache manager allows slot 0 and uses negative indices as padding. The
`vt::` contract therefore preserves local `<0 means skip, 0 is valid`
semantics; the model adapter, tests and CPU reference must agree. This is a
cache-ABI translation, not a recurrence deviation.

### Out of scope

- qkv+z projection packing (`KERNEL-GEMM-BF16` W2), which remains blocked
  until this BA correctness/component checkpoint closes.
- Mixed prefill/decode and speculative packed fusion. vLLM deliberately does
  not select the packed kernel for those branches.
- GDN chunked-prefill AOT changes, FP4 tactics, attention, scheduling,
  tensor-parallel packing, LoRA, 35B performance and the full binding grid.
- Deleting the portable decomposed GDN path. It remains the CPU/cross-backend
  reference and rollback even after the CUDA packed path becomes default.

## Upstream chain

The port pin is `e24d1b24fe96a56ba8b0d653efa076d03eb95d6c`; the audited
vLLM v0.25.0 target is
`702f4814fe54fabff350d43cb753ae3e47c0c276`. The sources below are identical
for this path at both commits.

| Layer | Exact upstream source | Required behavior |
|---|---|---|
| Default | `vllm/envs.py:117,1123-1125` | `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` defaults to `1`. |
| Model dispatch | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1286-1298` | Select packed execution only for non-spec pure decode. |
| Conv + packed call | same file `:1644-1695` | Run causal-conv update, then pass raw packed qkv and a/b directly to the recurrent operation. |
| Fused body | `vllm/model_executor/layers/fla/ops/fused_recurrent.py:255-336` | Infer the GQA head, load q/k/v/state, normalize q/k in F32, derive g, round beta through `b.dtype`, update state and store output/state. |
| Validation/launch | same file `:339-478` | Last-dimension contiguity, shape/device checks, `BK=next_power_of_2(Dk)`, `BV=min(next_power_of_2(Dv),32)`, grid `(ceil(Dv/BV), B*Hv)`, one warp and three stages. |
| Mixed/spec fallback | `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py:100-170`; Qwen dispatch `:1392-1559` | Fused sigmoid gating keeps beta in F32 and normalizes q/k in-kernel; this distinct branch must not inherit packed-only beta rounding. |
| Executable spec | `tests/kernels/test_fused_recurrent_packed_decode.py:13-98` | FP16/BF16/F32, contiguous and padded-row mixed-qkv, grouped value heads, negative pad slots, output and state parity. |

Runtime proof is two-part. The binding v0.25 trace identifies the recurrent
decode family on every 27B decode layer; the focused oracle executes the exact
packed callable under the official v0.25 environment. After implementation,
an nsys node trace must name the local packed kernel 48 times per B=2 graph
window and show the old post-conv/decode pair absent on pure decode.

## Our baseline

| Surface | Current anchor | Gap/evidence |
|---|---|---|
| Model assembly | `src/vllm/model_executor/models/qwen3_5.cpp`; `src/vllm/model_executor/models/qwen3_5_internal.h` | Exact CUDA dense pure-non-spec decode selects packed recurrence before decomposed q/k/v/g/beta allocation. Prefill, mixed, spec, CPU and 35B retain their existing branches. Complete host metadata validation precedes upload; padded graphs fall back when row-copy state I/O is selected. |
| State ABI | `src/vllm/model_executor/models/model_registry.cpp`; `src/vllm/v1/worker/gpu/runner.cpp` | Upstream `MambaSpec` order is conv then temporal. Gate allocation is independent BF16 conv + FP32 SSM from nested `mamba_ssm_dtype=float32`; F16/BF16/F32 temporal aliases remain supported. |
| Rollback/default | `src/vllm/model_executor/models/qwen3_5.cpp` | Default 27B path couples BF16 BA to packed recurrence. Process-cached `VT_GDN_PACKED_DECODE=0` restores F32 BA plus decomposed recurrence; prefill remains packed-call zero. |
| Projection proof | `tools/bench/gdn_ba_projection_oracle.py`; `tests/parity/goldens/gdn_ba_projection_bf16_sm121/` | Immutable `f925294` passes all five real BA shapes exactly; the divergence is not the GEMM. |
| Packed oracle | `tools/bench/gdn_packed_decode_oracle.py`; `tests/parity/goldens/gdn-packed-decode-oracle/` | Official v0.25 packed output is bit-stable. Its rounded-beta explicit reference is output-exact and differs by one state element (`1.9073486328125e-06`); full-F32 beta differs at 46 output and 5,834 state BF16 elements. |
| Local boundary replay | `tests/parity/test_op_parity.cpp` focused packed-decode case | Clean pushed `f18ca23`: regenerated official fixture is byte-identical and CUDA **10/10**; current local output/state differ at `306/7552`, beta-only is `308/6558`, and beta rounding plus F32 q/k normalization is `0/1`. Immutable G0 is closed. |
| W1D2 immutable G2 | model/runner/registry tests plus real 27B/35B/GGUF gates | Clean pushed `f344dec` passes default and rollback 27B **235/235 + 16/16**; default selects exactly 48 packed calls on the first decode and zero on prefill, rollback selects zero; native/batched 35B selects zero at **315/315**; Compact/Balanced GGUF each pass **14/14**; full CUDA GDN passes **43/43, 1,707/1,707**; three strict memcheck slices report zero errors/leaks. G2 is closed; no speed credit exists. |
| W1D3 trace harness | `tools/bench/online_gate.py`; `scripts/dgx-online-serving.sh`; `tools/bench/finalize_gdn_packed_trace.py` | Explicit packed/rollback modes preserve historical contracts, require exact `VT_GDN_PACKED_DECODE=1/0`, launch record/model/ours/vLLM commands from `/usr/bin/env -i` plus the fixed host/H1d inventories, and validate those recorded prefixes before accepting evidence. Both complete ours/vLLM arms run under one lock and finalize marker-last only at 915/963 nodes with the exact 48-for-96 GDN replacement. Each arm must contain exactly 48 BA projection nodes at `(8,1,1)`; those mode-coupled signatures are hashed separately because BF16-vs-F32 output may change cuBLASLt selection, while every remaining kernel/memcpy/memset signature must match cross-arm. Clean `7ff713e`, finalized by pushed `24cea4f`, closes marker-last `complete-structural` evidence. |
| W1D3 component harness | `scripts/dgx-gdn-packed-component.sh`; `tools/bench/gdn_packed_component.py`; `tests/tools/test_gdn_packed_component.py` | Production profile-control-off build only; exact source/vLLM corpus manifest and partition binding; full oracle/dependency/toolchain/artifact inventory; exact detailed-sample recomputation for throughput/TTFT/ITL, bounded validation for pinned vLLM's unexported-latency E2E/TPOT skew, and duration-span consistency; exact frozen 64-plan lifecycle and `/usr/bin/env -i` commands; direct packed/rollback **235/235 + 16/16** gates bound to the recorded snapshot and binary before timing; one lock across both gates and 12 fresh-server legs; c2=6 requests and c16=96; AB/BA/AB; all 40 timing + 8 memory median axes plus all 144 paired axes; ≤4% maximum per-run deviation; fixed 1-GiB recomputed memory return; pinned GPU/thermal probes; closed run log plus marker-last summary/manifest/status; symlinked evidence rejected. A stable regression is `complete-failed`; a sealable unstable/malformed run is `complete-void`; post-seal mutation fails verification. The real production invocation has a test-first exact `--profile-control off` contract. Focused CPU contracts pass **49/49**, all tools **132/132**, and shell/dry-run gates pass. Clean `d82d282` failed incomplete at c16 packed r1 after both model gates/all c2 legs: 0/96 timed requests returned HTTP 500, no marker was sealed, and partial legs are nonbinding. A bounded `--diagnostic-c16` mode (mutually exclusive with `--dry-run`/`--execute`) reproduces ONLY the packed c16 boundary — reps 1-3, three fresh servers under ONE `/tmp/gpu` lock, each carrying `VT_GDN_DIAG_STEP_LOG=1`; it runs no model gates, no 2/16 sweep, never calls `finalize`, asserts the evidence basename contains `diagnostic-c16` with no pre-existing `component-*.json`, wraps the failure-tolerant c16 bench (`\|\| bench_failed=1`) and on failure replays corpus row 0 into `diagnostic/c16/packed/r{rep}-error-body.json`, and writes status ONLY to `component-diagnostic.json`. `summarize_evidence`/`finalize_evidence` fail closed on a `component-diagnostic.json` marker or a `diagnostic/` subtree ("refusing to finalize component from diagnostic evidence"). Four unconditional `std::cerr` error-path channels restore the dropped root cause: `engine-fatal:` at the busy-loop guard (`core_client.cpp`, restoring vLLM `core.py:1233`), `async-llm:` at the output handler (`async_llm.cpp`, restoring `async_llm.py:703-705`), `api-server:` at both 500 sites (`api_server.cpp`), and `sse:` mid-flight. |

The beta-only hypothesis is disproven: it improves state agreement but does not
restore output agreement. Both upstream semantics are required, and fusing
them also removes the pure-decode post-conv intermediate launch and buffers.

## Port map

| Upstream behavior | Local destination | Port/deviation |
|---|---|---|
| Packed op ABI and validation | `include/vt/ops.h`, `src/vt/ops.cpp` | W1D1 adds typed shape/stride/device validation and one operation preserving the local negative-pad state-index ABI. CPU validates live values; CUDA consumes engine-validated device metadata and bounds-checks slots without synchronizing. |
| Portable recurrence | `src/vt/cpu/cpu_ops.cpp` plus CPU registration | W1D1 transcribes the exact F32 arithmetic and dtype-rounding points for FP16/BF16/F32 as the cross-backend reference. |
| CUDA packed kernel | `src/vt/cuda/cuda_gdn.cu` plus CUDA registration | W1D1 hand-ports the Triton body, grid, GQA mapping and storage rounding; no Triton runtime or generated cubin is introduced. CUDA maps each value row across an 8-lane group at Dv≥32 instead of exposing Triton's compiler-private one-warp tensor mapping. |
| Pure-decode selection | `src/vllm/model_executor/models/qwen3_5.cpp` | Implemented before post-conv intermediate allocation when host-validated metadata proves exact CUDA dense pure non-spec decode; every other branch keeps the existing path. |
| Rollback/default | same model file | Implemented process-cached `VT_GDN_PACKED_DECODE`; default couples BF16 BA+packed for 27B and `=0` restores F32 BA/decomposed execution from the same binary. |
| Oracle generation | `tools/bench/gdn_packed_decode_oracle.py` | Maintainer-only official-v0.25 generator with version/commit guard, repeated bit-stability and reference-tolerance checks. |
| Test port | `tests/parity/test_op_parity.cpp`, `tests/vt/test_ops_gdn.cpp`, model tests | Port the upstream dtype/stride/state cases and retain the small exact boundary fixture. |
| Trace provenance/finalization | `tools/bench/online_gate.py`, `scripts/dgx-online-serving.sh`, `tools/bench/finalize_gdn_packed_trace.py` | W1D3 adds mode-keyed graph contracts without reinterpreting historical evidence, disjoint arm artifacts, one-lock ordering, environment validation and completion-marker-last structural finalization. |
| Component execution/finalization | `scripts/dgx-gdn-packed-component.sh`, `tools/bench/gdn_packed_component.py`, `tools/bench/run_serve_low.py` | W1D3 adds a separate production-build execution path. It validates the exact source/build/oracle, command environments, frozen plan, correctness, raw metrics, memory return and one-lock order; derived summary/manifest/status artifacts are written marker-last and never change the binding denominator. A separate bounded `--diagnostic-c16` mode (with `run_serve_low.py diagnostic-error-body`) reproduces only the packed c16 boundary, writes only `component-diagnostic.json`, and is refused by the finalizer. |

Every implementation file cites the exact upstream source and commit. The
Triton kernel is a porting reference only; the runtime stays pure C++/CUDA.

## Tests to port

| Upstream case | Local tier and required assertions |
|---|---|
| `test_fused_recurrent_packed_decode_matches_reference[dtype,strided]` | T-unit/T-parity matrix for FP16, BF16 and F32, contiguous and padded-row mixed-qkv, B=32, Hk=4/Hv=8, Dk=Dv=128, negative pad indices, output/state tolerance identical to upstream. |
| Same case's explicit reference | CPU reference versus CUDA packed operation, including q/k F32 norm and dtype-rounded beta. No parameter may be silently omitted; an unavailable backend case is checked in `SKIPPED` with its stable dependency. |
| Qwen pure-decode dispatch `qwen_gdn_linear_attn.py:1286-1298` | Model unit tests prove packed selection only for pure non-spec decode and prove mixed/prefill/spec/35B inertness. |
| State-index behavior `fused_recurrent.py:292-303` | Local adapter test proves negative indices zero/skip, slot 0 remains valid, duplicated/out-of-range indices fail validation, and canaries remain untouched. |

Additional mandatory local gates:

- deterministic official-v0.25 BF16 fixture replay with exact bit-difference
  counters (`0/1` accepted only because upstream's own explicit reference has
  the same one-element state delta);
- B=1/2/4/16/32 and real 27B Hk16/Hv48/Dk128/Dv128 coverage;
- eager, capture and two replays, padded outer strides, canaries, pool on/off,
  ASan/UBSan where applicable and strict compute-sanitizer;
- 27B default and rollback real-model runs, each 235/235 plus 16/16; native,
  legacy and GGUF 35B zero-selection regressions;
- node-level trace with `--cuda-graph-trace=node`, no capture allocation,
  synchronization, D2H or new copy/cast nodes.
- fail-closed tool tests cover exact zero-count legacy families, wrong-mode
  topology rejection, mutual exclusion with BA-mode traces, one-lock arm order,
  fresh-oracle agreement, overwrite refusal and completion-marker-last;
- `tests/tools/test_gdn_packed_component.py` covers the exact 12-leg plan,
  **40+8** median plus **144 paired** every-axis acceptance, valid-regression
  `FAILED` and unstable `VOID` dispositions, missing/malformed/post-seal
  evidence, full source/oracle/toolchain/artifact provenance, exact
  server/client/preflight commands and clean environments, skipped/wrong-model
  gates, repetition stability, fail-closed GPU/thermal probes, fixed-tolerance
  recomputed memory return, signal exit, direct executable reproduction and
  marker-last finalization.

## Gates

### G0 — immutable semantic oracle

- Commit this spike, generator, fixture and focused test before production
  implementation.
- On the pushed SHA, regenerate the fixture in the official v0.25 environment
  and replay the focused CUDA test under one `/tmp/gpu` lock.
- Record source/binary/fixture/log hashes. Clean `f18ca23` closes this gate;
  the earlier mutable preflight remains diagnostic only.

### G1 — operator parity and safety

- Port the upstream 3-dtype x 2-stride matrix first and observe it fail before
  implementation.
- CPU and CUDA outputs/states meet upstream tolerances; BF16 exact fixture
  produces output/state bit differences `0/1` or better.
- Validation, canary, capture/two-replay and compute-sanitizer gates pass with
  zero errors/leaks and no hidden fallback.

### G2 — real-model correctness and dispatch

- Default 27B BF16 BA+packed execution passes 235/235 and 16/16. Its stream
  must differ from the rejected 233/235 emulation path at the previously
  divergent token.
- `VT_GDN_PACKED_DECODE=0` selects the legacy F32 BA/decomposed arm and remains
  235/235 plus 16/16.
- Pure-decode selects exactly 48 packed calls; prefill/mixed/spec and every 35B
  path select zero. Loader ownership and host memory do not grow.

### G3 — structure and component performance

- An uncontended B=2 node trace shows 48 packed kernels replacing 48
  `GdnPostConv` + 48 `GdnDecode` pairs. With qkv/z still split, BF16 GEMMs stay
  at 145 and total local graph nodes fall from 963 to 915; all unrelated
  families remain invariant.
- Under one lock, run packed-default versus rollback c2/c16 in AB/BA/AB order,
  three repetitions, one frozen plan map. All 40 timing and 8 memory axes are
  recorded as per-run values, medians, spread and paired normalized ratios.
  Every run must remain within 4% of its arm median; instability is void.
  Correctness is a precondition and no stable regression is accepted.
- Only after this checkpoint closes may qkvz begin. A passing component
  authorizes the fresh exact grid; a failure resumes the trace-driven scan.
- The `d82d282` c16 packed failure (0/96 HTTP 500, no marker) is diagnosed
  through a bounded `--diagnostic-c16` driver mode, mutually exclusive with
  `--dry-run`/`--execute` and never a component: it shares execute's provenance
  guards, additionally asserts the evidence basename contains `diagnostic-c16`
  with no pre-existing `component-*.json`, then under ONE `/tmp/gpu` lock runs
  the packed arm ONLY at concurrency 16 for reps 1-3, each on a fresh server
  spliced with `VT_GDN_DIAG_STEP_LOG=1` before the binary token (rebuilt array,
  not string substitution, because `env -i` stops consuming assignments at the
  first non-assignment token). It runs no model gates and no 2/16 sweep. The
  failure-tolerant c16 bench (`|| bench_failed=1`, required because the harness
  is `set -euo pipefail` and `online_gate.py` exits non-zero on a partial set)
  is followed on failure by a `run_serve_low.py diagnostic-error-body` replay of
  corpus row 0 into `diagnostic/c16/packed/r{rep}-error-body.json`, then
  `cleanup_server`. Status is written ONLY to `component-diagnostic.json`; it
  never calls `finalize`, and `summarize_evidence`/`finalize_evidence` fail
  closed on that marker or a `diagnostic/` subtree. Four unconditional
  `std::cerr` error-path channels surface the previously-dropped root cause —
  `engine-fatal:` (busy-loop guard, vLLM `core.py:1233`), `async-llm:` (output
  handler, `async_llm.py:703-705`), `api-server:` (both 500 sites), and `sse:`
  (mid-flight) — none containing the `#ifdef VT_BENCH_PROFILE_CONTROL`
  marker bytes.

The initial reproduction entry points are:

```sh
# Official oracle generation (DGX).
/home/mudler/venvs/vllm-oracle/bin/python \
  tools/bench/gdn_packed_decode_oracle.py \
  --out tests/parity/goldens/gdn-packed-decode-oracle

# Focused local replay; exact build path is recorded with the pushed SHA.
flock /tmp/gpu build-cuda/tests/test_op_parity \
  -tc='qwen27 GDN packed decode boundary*'

# Production component plan; `--execute` takes the exact paths recorded in
# docs/BENCHMARKS.md after the source SHA is pushed.
scripts/dgx-gdn-packed-component.sh --dry-run \
  --vllm-cpp-sha "$(git rev-parse HEAD)"

# Bounded c16 packed diagnostic reproduction (next step); evidence basename must
# contain 'diagnostic-c16' and hold no component-*.json. Full recipe in
# docs/BENCHMARKS.md.
scripts/dgx-gdn-packed-component.sh --diagnostic-c16 \
  --snapshot "$SNAPSHOT" --source-corpus "$ROOT/evidence-diagnostic-c16/corpus/27" \
  --evidence "$ROOT/evidence-diagnostic-c16" --build-dir "$ROOT/build-production" \
  --configure-log "$ROOT/configure.log" --vllm-cpp-sha "$SHA"
```

## Dependencies

- `KERNEL-GEMM-BF16` W1C projection/inertness proof (`f925294`) and exact BA
  graph structure (`0091cd1`).
- `KERNEL-GDN-AOT-BF16` and `KERNEL-GDN-SCRATCH` remain unchanged prefill and
  stream-lifecycle preconditions; this row owns only the pure-decode kernel.
- `SERVE-GATE-ONLINE`, its binding `3f256ab`, frozen 64-plan FP4 fixture and
  current vLLM v0.25.0/FlashInfer 0.6.13 denominator.
- CUDA 13.0.88, sm_121a GB10, Nsight Systems 2025.3.2 and one uninterrupted
  `/tmp/gpu` lock for every GPU test, trace or A/B series.
- Local FP16 storage/conversion support is required by the full upstream test
  matrix even though neither gate model uses FP16.

## Work breakdown

| Leaf | Owned work | Entry/exit |
|---|---|---|
| W1D0 | Generator, official packed fixture, focused boundary differential and this spike. | **CLOSED at clean `f18ca23`:** byte-identical regeneration, CUDA **10/10**, `306/7552 -> 0/1`; evidence root `~/work/vllm.cpp-gdn-packed-decode/f18ca23691bc7e38adbf04912da92f819154379e`. |
| W1D1 | Add public op, CPU reference, CUDA packed kernel, registrations and the full upstream dtype/stride/state-index test matrix. | **CLOSED / G1 PASSED at clean `9ad8fb7`:** local full GDN **39/39**, focused ASan+UBSan **5/5**, immutable CUDA full GDN **41/41**, focused packed **5/5**, direct fixture `0/1`, and strict memcheck **2/2 with 0 errors/leaks**. Evidence root `~/work/vllm.cpp-gdn-packed-decode/9ad8fb76940e68737d2a13ad8ddd97d649bb577c`. |
| W1D2 | Add exact pure-decode dispatch, process-cached rollback and BF16 BA default coupling; retain other branches. | **CLOSED / immutable G2 PASS at clean `f344dec`:** local **103/103**; DGX default+rollback 27B **235/235**, 35B **315/315**, isolated GGUF **14/14 + 14/14**, full CUDA GDN **43/43**, boundary `0/1`, and three strict memcheck cases have zero errors/leaks. Evidence root `~/work/vllm.cpp-gdn-packed-decode/f344decf457a4d50c3bcae78a2903d7fe176a511/evidence-g2`. |
| W1D3 | Add the fail-closed packed/rollback trace harness; run node trace and c2/c16 component; update every status surface. | **Structural PASS / component FAILED INCOMPLETE; diagnostic instrumentation landed.** Clean `7ff713e`, finalized by `24cea4f`, closes structure. Clean `d82d282` passed model gates/all c2 legs, then c16 packed r1 returned 0/96 HTTP 500 with no marker. A test-first diagnostic checkpoint now exposes the dropped root cause (four `std::cerr` error-path channels + `VT_GDN_DIAG_STEP_LOG` + a packed-only `--diagnostic-c16` mode → `component-diagnostic.json`; six RED→GREEN tests, tools 132/132). Next: run the DGX `--diagnostic-c16` reproduction from the pushed SHA in a fresh `diagnostic-c16` root, then repair/rerun; qkvz stays blocked. |

No later leaf starts before the previous performance-sensitive checkpoint is
recorded. qkvz and the exact grid remain unauthorized during W1D0-W1D2.

## Risks and decisions

- **Packed and mixed decode have different beta rounding.** Reusing packed
  rounding in the mixed/spec kernel would diverge from upstream. Dispatch is
  therefore part of the numeric contract, not merely an optimization toggle.
- **Beta-only is false.** The fixture proves beta-only changes output agreement
  from 306 to 308 differing elements. Both in-kernel F32 q/k normalization and
  dtype-rounded beta are mandatory.
- **The rollback changes both fusion and BA dtype.** This is intentional: it is
  the existing token-correct shipping arm. Component results attribute the
  complete upstream transition; the semantic pieces cannot be separated while
  preserving both token streams.
- **Slot zero differs from vLLM.** Preserve the local cache ABI at the adapter
  and test it explicitly; do not copy vLLM's null-block convention into the
  engine silently.
- **Device index values cannot be synchronously rejected inside a captured
  CUDA op.** CPU calls reject duplicates/out-of-range values directly. The
  W1D2 host adapter now validates the complete non-spec vector, live range,
  uniqueness and exact prefill suffix/rebase/mask contract before upload; the
  CUDA kernel separately bounds-checks every slot without adding a D2H sync.
  Graph padding uses `-1` only with indexed state I/O; row-copy mode falls back
  before padding.
- **CUDA thread decomposition is a recorded mechanical adaptation.** The grid,
  value tiling, F32 arithmetic and storage boundaries mirror Triton. The hand
  kernel uses one 8-lane group per value row at Dv≥32 so all 32 rows of a tile
  execute in eight warps; Triton's `num_warps=1` tensor-to-lane mapping is a
  compiler detail rather than a callable CUDA launch ABI.
- **A launch reduction is not a speed claim.** The expected 48-node reduction
  is structural evidence only. G3 must measure every axis and memory.
- **Portable fallback stays.** The CPU/decomposed recurrence is required by the
  multi-backend roadmap. Only the CUDA pure-decode default changes after gates.
