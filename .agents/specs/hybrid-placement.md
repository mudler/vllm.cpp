# Hybrid device placement (`ENG-HYBRID-PLACEMENT`)

Assign a model's tensor groups to *different devices in one engine* — routed-MoE
experts on the CPU while attention, dense projections, router and norms stay on
the GPU. Issue [#149](https://github.com/mudler/vllm.cpp/issues/149), the
highest-upvoted request from the r/LocalLLaMA launch thread. Design grounded in
a source scan of llama.cpp at the pinned oracle `237ad9b96`
([`../oracles/llama-cpp.md`](../oracles/llama-cpp.md), `gateable = yes`) plus
the pinned vLLM `555967922`.

**Verdict up front.** The capability is real, the oracle is admissible and
gateable, and the mechanism is smaller than it looks — the CPU expert kernels
already exist on both sides. What is missing everywhere is *per-tensor-group
device selection*. Three findings decide the shape:

- **vLLM has the kernels but not the placement.** `experts/cpu_moe.py`,
  `cpu_fused_moe.py` and `_custom_ops.py:3790` are real CPU MoE experts at the
  pin, but every selection site gates on `current_platform.is_cpu()` — so they
  are reachable only when the *entire* model runs on the CPU platform. There is
  no hybrid mode to mirror. This row is therefore surpass-track, exactly like
  its neighbour `ENG-EXPERT-STREAM`.
- **CPU-MoE is a decode win and a prefill loss.** At batch 1 the expert MLP is
  memory-bound, so computing where the weights already live beats shipping them
  across PCIe by the DDR:PCIe bandwidth ratio (see the bandwidth math). At prefill's arithmetic
  intensity the GPU wins decisively. A placement that ignores the phase gets one
  of the two regimes badly wrong.
- **This is not the same problem as offload.** `ENG-WEIGHT-OFFLOAD` and
  `ENG-EXPERT-STREAM` both move *weights toward the compute*. This row moves
  *compute toward the weights*. They compose but never substitute.

## Scope

| Field | Content |
|---|---|
| Row ID | `ENG-HYBRID-PLACEMENT` (engine-matrix). Issue [#149](https://github.com/mudler/vllm.cpp/issues/149) |
| In | A per-tensor-group device-placement seam resolved at model build, plus its first and only in-scope client: routed-MoE expert compute on the CPU backend with the rest of the model on GPU. Pattern→device override surface, an auto-fit resolver that places by measured free device memory, honest reporting of the resolved placement, and the activation round-trip at each MoE layer boundary |
| Out | Dense-layer CPU offload (that is `ENG-WEIGHT-OFFLOAD`, the vLLM `cpu_offload_gb` mirror — INVENTORIED); disk-tier expert paging (`ENG-EXPERT-STREAM`, READY); multi-GPU placement across two devices ([#147](https://github.com/mudler/vllm.cpp/issues/147) / `BACKEND-DISTRIBUTED-TP`, ACTIVE); EPLB; phase-aware placement that splits prefill and decode across devices (recorded as a surpass hypothesis under Surpass hypothesis, not built here) |
| Supported modes | `off` (default — unchanged single-device engine, byte-identical); `cpu-moe` (all routed experts on CPU); `cpu-moe:N` (first N layers' experts on CPU); `auto` (resolver picks N from measured free device memory) |
| Dispatch behavior | Placement is resolved ONCE at model build into a per-tensor-group device assignment. The forward reads that assignment; it never re-decides per step. A dense model rejects `cpu-moe` with a message naming why. When every group resolves to one device the engine takes the existing single-device path unchanged — the seam must be provably inert when placement is trivial |
| Regimes served | Decode at low-to-moderate concurrency on discrete-GPU hosts where the model does not fit in VRAM. Explicitly NOT unified-memory hosts (GB10: one physical pool, placement saves nothing) and NOT prefill-dominated workloads |

## Upstream chain

vLLM at the pin `555967922`:

- CPU MoE experts exist — `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py`
  (`CPUExpertsMxfp4` and the fp8/int8/int_wna16 siblings),
  `vllm/model_executor/layers/fused_moe/cpu_fused_moe.py:398,430`,
  `vllm/_custom_ops.py:3790,3803`.
- Every selection site is platform-wide, not per-layer:
  `oracle/fp8.py:129`, `oracle/int8.py:53`, `oracle/int_wna16.py:111`,
  `oracle/mxfp4.py:533`, `oracle/unquantized.py:97,202`,
  `oracle/w4a8_int8.py:40` — all `if current_platform.is_cpu():`.
- The offloader moves weights only, never compute:
  `vllm/config/offload.py:23,34-44,47-76`,
  `vllm/model_executor/offloader/uva.py:64,80-108`,
  `vllm/model_executor/offloader/base.py:126-162`,
  `vllm/model_executor/offloader/prefetch.py:557-560` (cpu-only tier).

So vLLM answers "run the whole model on CPU" and "fetch weights to the GPU",
never "run these layers there and those layers here". Per AGENTS.md this is a
path vLLM cannot produce at all, which is what admits a secondary oracle.

## llama.cpp anatomy (what we port from)

Pin `237ad9b96` (`b9892`), verified present in the local checkout.

**The mechanism is tensor-name-pattern → buffer type, applied at load.** Compute
follows the tensor's buffer; there is no separate compute-dispatch decision.

- `common/arg.cpp:2451-2455` — `-ot` / `--override-tensor`, taking
  `<tensor name pattern>=<buffer type>,...` into `parse_tensor_buffer_overrides`.
  This is the general seam; everything else is sugar over it.
- `common/arg.cpp:2457-2462` — `-cmoe` / `--cpu-moe`, "keep all Mixture of
  Experts (MoE) weights in the CPU", one blanket override.
- `common/arg.cpp:2464-2478` — `-ncmoe` / `--n-cpu-moe N`, "keep the MoE weights
  of the first N layers in the CPU", pushing one per-layer override each.
- `common/common.h:1046` — the pattern itself:
  `LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"`.
- `common/common.h:1048-1054` — `llm_ffn_exps_block_regex(idx)` and
  `llm_ffn_exps_cpu_override()`.
- `src/llama-model-loader.cpp:1158-1160` — where an override pattern is matched
  against a tensor name at load and the buffer type is substituted.
- `src/llama-model.cpp:1032` — `has_tensor_overrides`, the flag that changes the
  load path at all.
- `include/llama.h:530` — `llama_max_tensor_buft_overrides()`, the bound.

**The auto-fit resolver** — `common/fit.h:24`, `common/fit.cpp:457,485`, driven
from `common/arg.cpp:743-750`, with `tools/fit-params/`. It projects memory use
against measured free device memory and reduces device residency until it fits,
reporting each step (`tools/fit-params/README.md:17-20`):

```
llama_params_fit_impl: projected to use 61807 MiB of device memory vs. 24077 MiB of free device memory
llama_params_fit_impl: cannot fulfill margin of 1024 MiB, need to reduce device memory by 42444 MiB
llama_params_fit_impl: with only dense weights in device memory there is a total surplus of 16164 MiB
```

That reporting shape is the model for our `auto` mode: resolve, then say what
was resolved and why. A user who cannot see the resolved placement cannot
attribute a slow run to it.

**A limitation worth recording:** `common/fit.cpp:181` refuses —
`"llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort"`. llama.cpp's
auto-fit does not compose with its tensor-parallel split. That is precisely the
intersection this row shares with `BACKEND-DISTRIBUTED-TP`, and it is a known
unsolved problem upstream rather than an oversight we can port around.

## Our baseline

- The CPU backend is the correctness reference and competitive on GGUF — 1.18x
  llama.cpp prefill on aarch64, decode a tie, same GGUF file (#149 body). Quant
  GEMM/dot kernels exist: `src/vt/cpu/cpu_quant_gemm.cpp`,
  `cpu_quant_dot.cpp`, `cpu_quant_dot_arm.cpp`, `cpu_quant_dot_sdot.cpp`,
  `cpu_quant_dot_a76.S`.
- The expert-major compute seam exists on the GPU side:
  `MarlinMoeAlignBlockSize` / `sorted_token_ids` / `expert_ids` /
  `num_tokens_past_padded` (`src/vt/cuda/cuda_moe_marlin.cu:105`, driven from
  `src/vllm/model_executor/models/laguna.cpp:686,790`), plus grouped expert
  GEMMs (`src/vllm/model_executor/models/deepseek_v4.cpp:548`,
  `MoeGateUpSwiGLUGroupedCuda`).
- **What is absent is the seam between them.** Execution is single-device: a
  model builds against one backend and every layer uses it. There is no
  per-tensor-group device assignment and no activation round-trip.

So this row is a *dispatch and placement* problem, not a kernel problem. Both
sets of expert kernels already exist and are already gated.

## The honest bandwidth math (decides viability)

The question CPU-MoE answers: is it cheaper to move the weights to the compute,
or the activations to the weights?

Per decoded token, with `A` bytes of active expert weight:

- **Offload to GPU** — `A` crosses the bus: `t = A / B_pcie`, then GPU compute
  (negligible, memory-bound at batch 1).
- **CPU-MoE** — activations cross the bus (kilobytes, negligible), and the CPU
  reads `A` from system RAM: `t = A / B_dram`.

CPU-MoE wins by exactly `B_dram / B_pcie`. For #149's stated rig — 2x RTX 5060
Ti, 64 GB system RAM:

| Path | Achievable bandwidth | Assumption |
|---|---|---|
| DDR5-5600 dual channel | ~60-70 GB/s | 89.6 GB/s theoretical, ~75% achieved |
| PCIe 5.0 x8 (5060 Ti link width) | ~25 GB/s | 31.5 GB/s theoretical |
| PCIe 4.0 x8 (if the board negotiates Gen4) | ~12.5 GB/s | |

**So CPU-MoE is 2.5-5x better than weight offload on this hardware, at batch 1.**
Both numbers are unmeasured assumptions from published link rates and MUST be
replaced by measured values on a real rig before any gate cites them (W0).

Three regime boundaries fall out and all three must be enforced, not discovered:

1. **Prefill inverts it.** At prefill's arithmetic intensity the expert GEMM is
   compute-bound, not memory-bound, and GPU FLOPs dominate CPU FLOPs by more
   than the bandwidth ratio recovers. CPU-MoE must be expected to *lose* on
   prefill-heavy workloads and must say so rather than silently degrade TTFT.
2. **Concurrency degrades it differently from streaming.** `ENG-EXPERT-STREAM`
   degrades because the touched-expert union grows as `1-(1-k/E)^B`. CPU-MoE has
   no union problem — the weights are already resident — but becomes CPU-FLOP-bound
   as `B` rises. Different wall, so the two rows' regime warnings are not
   interchangeable.
3. **Unified memory makes it pointless.** On GB10 there is one physical pool;
   "CPU" and "GPU" memory are the same LPDDR5X, so placement moves nothing. The
   engine must report `cpu-moe` as inert on unified-memory platforms rather than
   pretending to honour it. (Same finding as `expert-streaming.md` reached for
   tmpfs.)

## Port map

**The seam.** A `DevicePlacement` resolved once at model build, mapping a tensor
group to a device. Deliberately mirroring llama.cpp's shape — pattern → device,
not a bespoke MoE flag — because the general form is the same size as the
specific one and it is what `ENG-WEIGHT-OFFLOAD` and `#147` will need to
compose with later.

**Composition with `BACKEND-DISTRIBUTED-TP` (ACTIVE).** That row already owns
`TensorParallel` / `TpShard` / `TpAllReduceSum`
(`include/vllm/model_executor/models/tensor_parallel.h`) and its scope includes
MoE expert-parallel and the EP combine. Placement and sharding are orthogonal
axes: a shard says *which slice of a tensor*, a placement says *on which device
a group runs*. The seam must therefore be expressed so a group can carry both,
and this row must not introduce a second sharding concept. Where the two
genuinely conflict — llama.cpp itself refuses this combination at
`common/fit.cpp:181` — this row refuses with a message naming `BACKEND-DISTRIBUTED-TP`,
and the combination stays owed rather than half-working.

**The forward.** At a MoE layer whose experts are CPU-placed: router runs on
GPU (it is a trivial GEMV and the hidden state is already there), routing
results and the hidden state cross to host, the CPU backend runs its existing
grouped expert path, the combined output crosses back. One round trip per MoE
layer. The round trip is the cost that decides whether the bandwidth ratio survives
contact, which is why W1 measures it before anything else is built.

**Inertness.** When placement resolves every group to one device the engine must
take the existing path with no added indirection — proven by a byte-identical
golden on an unplaced model, not by inspection.

## Tests to port

From llama.cpp at the pin, adapted (llama.cpp has no unit test for `-ncmoe`; its
coverage is end-to-end), plus our own:

- Pattern-match semantics against `LLM_FFN_EXPS_REGEX` (`common/common.h:1046`):
  a table of tensor names that must and must not match, including the
  `gate_up` and `ch` variants the regex admits, and the per-layer
  `blk\.N` anchoring from `llm_ffn_exps_block_regex`.
- `cpu-moe:N` places exactly the first N layers and no others — asserted by
  *counting* the resolved assignments, never by spot-checking one layer.
- Dense model + `cpu-moe` refuses, naming the missing piece.
- Unified-memory platform + `cpu-moe` reports inert.
- `off` is byte-identical to the current engine on an existing golden.
- Placement resolution is deterministic across runs for a fixed free-memory
  input (the `auto` resolver must not be wall-clock or allocation-order
  dependent).

## Gates

- **Correctness, token-exact.** Same GGUF, same prompts, greedy: `cpu-moe`
  output must be token-identical to the same model run fully on CPU. That is the
  strong form — placement is a scheduling decision and must not change values.
  It is gateable today because the CPU backend is already the correctness
  reference.
- **Speed floor vs llama.cpp**, quant-matched, same GGUF, at the pinned oracle
  `237ad9b96`: `-ncmoe N` against our `cpu-moe:N` on the same rig, same N,
  reporting decode tok/s and TTFT separately. Prefill is expected to be the
  weaker axis (see the bandwidth math) and is recorded as a value, not waived.
- **Inertness.** An existing SACRED golden with placement `off` is byte-identical.
- **Regime honesty.** The prefill-loss and unified-memory-inert cases are
  asserted, not documented.

Every number above needs a discrete-GPU host. The dgx box is GB10 and unified —
it can gate correctness and inertness but **cannot gate the speed floor at all**.
That is a hard blocker on W4 and is named as such under Dependencies rather than discovered.

## Surpass hypothesis (recorded, not scope)

llama.cpp places weights once, so prefill runs on the CPU too and is slow — the
regime inversion in the bandwidth math.1 is unmitigated upstream. A phase-aware placement would
stream CPU-resident experts to the GPU for prefill (high arithmetic intensity
amortizes the transfer) while computing them on the CPU for decode, taking the
better side of both. This is where `ENG-HYBRID-PLACEMENT` and
`ENG-EXPERT-STREAM` would compose into something neither upstream has. It is
explicitly not built here: it needs W1's measured round-trip cost first, and a
row that tries both loses the ability to gate either.

## Dependencies and blockers

- **Discrete-GPU host** for the speed gate. Not available on dgx (unified). W4
  is BLOCKED until a rig exists; #149 has community test rigs offered
  ([#147](https://github.com/mudler/vllm.cpp/issues/147) records the same offer)
  and that is the path to unblocking.
- `BACKEND-DISTRIBUTED-TP` (ACTIVE) — no hard ordering, but the seam shape must
  be reviewed against `tensor_parallel.h` before W2 lands.
- No dependency on `ENG-EXPERT-STREAM` or `ENG-WEIGHT-OFFLOAD`; all three are
  independent and compose later.

## Work breakdown

| ID | Work | Gate |
|---|---|---|
| W0 | Measure the real DDR and PCIe bandwidths on a discrete-GPU rig; replace the assumed bandwidth table with measured values. Kills or confirms the row before code | measured numbers committed |
| W1 | Measure the per-MoE-layer activation round-trip cost (H2D+D2H at decode shapes) standalone. If it exceeds the bandwidth margin the design changes before it is built | measured |
| W2 | The `DevicePlacement` seam + pattern→device resolution at model build, reviewed against `tensor_parallel.h`. No behavior change: `off` only | inertness golden byte-identical |
| W3 | `cpu-moe` / `cpu-moe:N` routing routed-expert compute to the CPU backend; the activation round trip | token-exact vs full-CPU run |
| W4 | Speed floor vs llama.cpp `-ncmoe` at the pin, quant-matched, decode and TTFT reported separately | **BLOCKED on a discrete-GPU rig** |
| W5 | `auto` resolver mirroring `common/fit.cpp`'s project-and-reduce shape, with its reporting | deterministic resolution test |
| W6 | ABI + CLI surface, refusals for dense / unified-memory / TP-conflict | refusal tests |

W0 and W1 are deliberately first and are both measurements. The row's entire
justification is a bandwidth ratio, and a ratio nobody measured on the target
hardware is not a justification.

## Risks and decisions

- **The bandwidth ratio is assumed, not measured.** Every number in that table comes
  from published link rates. W0 exists because the row is worthless if the real
  ratio is below ~1.5x, and that is a plausible outcome on a rig with slow
  single-rank memory.
- **The round trip may eat the win.** One H2D+D2H per MoE layer, times layer
  count, times every decoded token. W1 measures it standalone before W2 builds
  anything.
- **Placement is a second axis next to sharding.** The seam could easily grow
  into a parallel sharding concept and duplicate `BACKEND-DISTRIBUTED-TP`. The
  mitigation is the explicit refusal in Port map and a review against
  `tensor_parallel.h` as W2's gate, not good intentions.
- **The speed gate cannot run on our hardware.** This is the row's real
  scheduling risk: W0-W3 and W5-W6 can all land on dgx while W4 sits blocked, and
  a row that is "done except the number" is exactly the shape that gets
  mistakenly called done. W4 stays open and visible.

## Now

`READY`. Spec committed; no implementation. Issue
[#149](https://github.com/mudler/vllm.cpp/issues/149) is the owning issue and
stays OPEN — this row covers only its CPU-MoE half. The dense layer-offload half
belongs to `ENG-WEIGHT-OFFLOAD` (INVENTORIED) and the multi-GPU half to
[#147](https://github.com/mudler/vllm.cpp/issues/147) / `BACKEND-DISTRIBUTED-TP`
(ACTIVE); #149 closes when all three have landed, not when this one does.

Next action is W0 — a measurement on a discrete-GPU rig, which we do not have.
Until that rig exists the honest state of this row is READY-and-waiting, not
ACTIVE.
