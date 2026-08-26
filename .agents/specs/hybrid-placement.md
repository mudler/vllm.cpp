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
| Supported modes | Absent (default — unchanged single-device engine, byte-identical); `overrides` (the general pattern→device form, our `-ot`); `cpu_moe` (all routed experts on CPU, our `-cmoe`); `n_cpu_moe: N` (first N layers' experts on CPU, our `-ncmoe`); `fit` (resolver places by measured free device memory, our `--fit`). All four live under `--offload-config`'s `vllm_cpp` key — see `## Configuration surface` |
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

Pin `b10451` / `10bf611e5`, the revision
[`../oracles/llama-cpp.md`](../oracles/llama-cpp.md) records with `gateable = yes`,
read with `git show b10451:<path>` rather than from a working tree. **Every anchor
below was re-derived at this pin on 2026-08-26 ([#2015](https://github.com/mudler/vllm.cpp/issues/2015)).**
The first version of this spec cited `237ad9b96` (`b9892`), which the oracle
superseded on 2026-08-22, and none of its line numbers resolve here:
`common/arg.cpp:2451-2478` is now `:2715`, `:2721` and `:2728`, and
`common/common.h:1046-1054` is now `:1113-1120`.

**The mechanism is tensor-name-pattern → buffer type, applied at load.** Compute
follows the tensor's buffer; there is no separate compute-dispatch decision. That
one sentence is the whole design, and it is why the four user-facing surfaces are
one mechanism rather than four.

- `common/arg.cpp:2715-2719` — `-ot` / `--override-tensor`, taking
  `<tensor name pattern>=<buffer type>,...` into `parse_tensor_buffer_overrides`
  (`:253-284`). This is the general seam; the next two are sugar over it.
- `common/arg.cpp:2721-2726` — `-cmoe` / `--cpu-moe`, "keep all Mixture of
  Experts (MoE) weights in the CPU". Its body is one line:
  `params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override())`.
- `common/arg.cpp:2728-2741` — `-ncmoe` / `--n-cpu-moe N`, "keep the MoE weights
  of the first N layers in the CPU". Its body loops `i` over `[0, N)` and pushes
  `{llm_ffn_exps_block_regex(i), ggml_backend_cpu_buffer_type()}`. `N < 0` throws
  `invalid_argument`; `N == 0` is legal and pushes nothing.
- `common/common.h:1113` — the pattern itself:
  `LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"`.
- `common/common.h:1115-1121` — `llm_ffn_exps_block_regex(idx)`, which is
  `string_format("blk\\.%d%s", idx, LLM_FFN_EXPS_REGEX)`, and
  `llm_ffn_exps_cpu_override()`.
- `include/llama.h:302-305` — `llama_model_tensor_buft_override`, a
  `{const char * pattern; ggml_backend_buffer_type_t buft;}` pair, and `:312` the
  **NULL-terminated** list on `llama_model_params`. `:548`
  `llama_max_tensor_buft_overrides()` is the bound.

**Three load-time semantics that a re-implementation gets wrong by default**, all
at `src/llama-model-loader.cpp:1178-1195`:

1. The match is `std::regex_search`, not a full match. `\.ffn_up_exps` matches
   anywhere inside the tensor name, which is what makes an unanchored pattern
   apply to every layer and an anchored `blk\.7` one apply to exactly one.
2. Overrides are scanned **in order and the first match wins**, terminating on the
   NULL sentinel. Order is therefore part of the user's input, not an
   implementation detail, and a later broad pattern cannot override an earlier
   narrow one.
3. A CPU override does **not** simply mean "the CPU buffer". It re-runs
   `select_weight_buft(..., buft_list_cpu)` so the extra CPU buffer types are
   still considered. Ours has no equivalent list today, and this is recorded so
   the difference is a decision rather than a silent divergence.

**`mmap` and a CPU override interact badly, and llama.cpp says so at runtime.**
The same block warns once: `"tensor overrides to CPU are used with mmap enabled -
consider using --load-mode none for better performance"`. This tree ships an
`mmap` tier under the same `vllm_cpp` key (`ENG-RESIDENCY-CONFIG`), default on
wherever weights stay quantized, so the interaction is ours too and is listed
under Risks.

**The auto-fit resolver** — `common/fit.h:24`, `common/fit.cpp:395-399` and
`:459-497`, driven from `common/arg.cpp:2823` (`-fit` / `--fit [on|off]`), with
`tools/fit-params/`. It projects memory use against measured free device memory
and reduces device residency until it fits, reporting each step
(`tools/fit-params/README.md:17-20`):

```
llama_params_fit_impl: projected to use 61807 MiB of device memory vs. 24077 MiB of free device memory
llama_params_fit_impl: cannot fulfill margin of 1024 MiB, need to reduce device memory by 42444 MiB
llama_params_fit_impl: with only dense weights in device memory there is a total surplus of 16164 MiB
```

That reporting shape is the model for our `auto` mode: resolve, then say what
was resolved and why. A user who cannot see the resolved placement cannot
attribute a slow run to it.

**The resolver spills at a finer granularity than "experts or not"**, which the
first version of this spec did not record. `common/fit.h:18-22` defines
`common_layer_fraction_t` with five values — `NONE`, `ATTN`, `UP`, `GATE` and
`MOE` ("everything but sparse MoE weights") — and `fit.cpp:450-451` uses them for
exactly one partial layer at the boundary, every further layer taking
`LAYER_FRACTION_MOE`. So the auto arm can place *part* of one layer. Our `auto`
may legitimately start coarser, but the difference has to be stated in its
reporting rather than left for a user to discover from a memory figure.

**Two refusals upstream, and both are ours to mirror.** `common/fit.cpp:398-399`
refuses when the caller already set an override
(`"model_params::tensor_buft_overrides already set by user, abort"`), so auto and
manual are mutually exclusive rather than merged. `common/fit.cpp:182-183`
refuses `LLAMA_SPLIT_MODE_TENSOR` outright:
`"llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort"`. That second
one is precisely the intersection this row shares with
`BACKEND-DISTRIBUTED-TP`, and it is a known unsolved problem upstream rather than
an oversight we can port around.

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

## Configuration surface

**Developer direction, 2026-08-26: this maps onto the configuration this tree
already has, not onto new top-level flags.** The knobs live under the `vllm_cpp`
extension key of `--offload-config`, beside `mmap`, `expert_stream` and
`device_fit`. `ENG-RESIDENCY-CONFIG` established that key, its parser and its
precedence rule, and the reasoning in
[`include/vllm/config/weight_residency.h`](../../include/vllm/config/weight_residency.h)
applies here unchanged: upstream `OffloadConfig` has no placement concept, so
adding a field to the mirrored struct would break a byte-faithful transcription
to describe behavior vLLM does not have.

```json
{"vllm_cpp": {"placement": {
    "overrides":  [{"pattern": "\\.ffn_(up|down|gate|gate_up)_(ch|)exps",
                    "device":  "cpu"}],
    "cpu_moe":    true,
    "n_cpu_moe":  40,
    "fit":        false
}}}
```

That block shows every key, not a document anyone should write. A real one sets
one of `overrides`, `cpu_moe`, `n_cpu_moe` or `fit`. Every field is optional and
an absent field means UNCHANGED.

| Key | llama.cpp equivalent | Environment override |
|---|---|---|
| `placement.overrides` | `-ot <pattern>=<buffer type>,...` | `VT_PLACEMENT_OVERRIDES` |
| `placement.cpu_moe` | `-cmoe` / `--cpu-moe` | `VT_CPU_MOE` |
| `placement.n_cpu_moe` | `-ncmoe N` / `--n-cpu-moe N` | `VT_N_CPU_MOE` |
| `placement.fit` | `-fit` / `--fit [on\|off]` | `VT_PLACEMENT_FIT` |

Five rules bind this surface, and four of them are inherited rather than invented.

1. **`overrides` is the mechanism; `cpu_moe` and `n_cpu_moe` are sugar that
   desugars into it.** They desugar exactly as `arg.cpp:2721-2741` does:
   `cpu_moe` appends one entry carrying `LLM_FFN_EXPS_REGEX`, and `n_cpu_moe: N`
   appends N entries carrying `blk\.<i>` + that regex for `i` in `[0, N)`. The
   resolved override list is what the engine reports, so a user who wrote the
   sugar can see the general form it produced. This is testable without a GPU
   and is the first thing W2 gates.
2. **Order is input.** The resolved list is scanned first-match-wins, mirroring
   `llama-model-loader.cpp:1180-1184`. Sugar appends, so a hand-written
   `overrides` entry placed before `cpu_moe`'s wins over it. Never sort the list.
3. **Absent means UNCHANGED, never "default".** The install merges field by
   field and the refusal scores a field only when the document sets it. This is
   `weight_residency.h`'s contract and #1133's H1/H2 measured both directions of
   getting it wrong.
4. **Precedence is environment variable > JSON config > built-in default**, in
   both directions, so `VT_CPU_MOE=0` beats a config `true`. Same reason as the
   neighbouring knobs: a benchmark arm has to be switchable without restarting
   the server.
5. **`fit` and a manual placement are mutually exclusive**, mirroring
   `fit.cpp:398-399` rather than merging them. A document carrying both is
   refused at startup by name, not silently resolved in some order.

**Device spelling.** `overrides[].device` takes a `vt::Backend` device name, so
`"cpu"` and `"cuda"` rather than a ggml buffer-type string. llama.cpp looks its
buffer type up among the loaded backends and, on an unknown name, prints the
available list and throws (`arg.cpp:273-278`). Mirror that behavior: refuse an
unknown device by name and name the ones that exist. Do not accept a ggml
buffer-type spelling as an alias, because our devices are not its buffer types
and a silent near-match is worse than a refusal.

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

llama.cpp has no unit test for any of the four surfaces; its coverage is
end-to-end. What is portable is its *semantics*, and each item below names the
upstream line the expectation comes from so a reviewer can check the
expectation rather than trust it.

- **Desugaring is exact.** `cpu_moe: true` produces one override carrying
  `LLM_FFN_EXPS_REGEX` (`common.h:1113`); `n_cpu_moe: N` produces N carrying
  `llm_ffn_exps_block_regex(i)` for `i` in `[0, N)` (`arg.cpp:2728-2741`),
  asserted by *counting* the resolved list, never by spot-checking one entry.
  `n_cpu_moe: 0` produces none, and a negative value is refused.
- **Pattern-match semantics** against `LLM_FFN_EXPS_REGEX`: a table of tensor
  names that must and must not match, including the `gate_up` and `ch` variants
  the regex admits, and the `blk\.N` anchoring. The match is `regex_search`, so
  the table must contain a name where a *substring* match is the only reason it
  hits (`llama-model-loader.cpp:1181-1182`).
- **First-match-wins ordering** (`llama-model-loader.cpp:1180`): a narrow entry
  before a broad one resolves to the narrow device, and reversing the two
  entries changes the result. A test that passes under both orderings is not
  testing this.
- **`fit` beside a manual placement is refused** by name, mirroring
  `fit.cpp:398-399`, and refused at *startup* rather than at first forward.
- **Unknown device is refused** and the refusal names the devices that exist
  (`arg.cpp:273-278`).
- **Config/environment precedence in both directions**, including
  `VT_CPU_MOE=0` beating a config `true`, which is the direction an override
  that could not turn a knob off would fail.
- **Absent means unchanged**: a document that omits `placement` leaves an
  installed placement alone, and one that omits a field inside `placement`
  leaves that field alone (`weight_residency.h`, #1133 H1/H2).
- `cpu-moe` on a dense model refuses, naming the missing piece.
- `cpu-moe` on a unified-memory platform reports inert.
- **`off` is byte-identical to the current engine** on an existing golden.
- Placement resolution is deterministic across runs for a fixed free-memory
  input, so the `auto` resolver is not wall-clock or allocation-order dependent.

## Gates

- **Correctness, token-exact.** Same GGUF, same prompts, greedy: `cpu-moe`
  output must be token-identical to the same model run fully on CPU. That is the
  strong form — placement is a scheduling decision and must not change values.
  It is gateable today because the CPU backend is already the correctness
  reference, and it needs no discrete GPU.
- **Reachability.** Every wave enters through a production entry point:
  `--offload-config` on the real server argv, the C ABI's
  `vllm_model_params.offload_config`, and `ModelRegistry::Forward`. The fresh
  reviewer deletes the install call site in a scratch copy and reruns the
  focused gate, per [`../reachability.md`](../reachability.md). A resolver test
  that constructs the placement by hand proves the class works, never that
  anything reaches it.
- **Speed floor vs llama.cpp**, quant-matched, same GGUF, at the recorded pin
  `b10451`: `-ncmoe N` against our `n_cpu_moe: N` on the same rig, same N,
  reporting decode tok/s and TTFT separately. Prefill is expected to be the
  weaker axis (see the bandwidth math) and is recorded as a value, not waived.
- **Inertness.** An existing SACRED golden with placement absent is
  byte-identical.
- **Regime honesty.** The prefill-loss and unified-memory-inert cases are
  asserted, not documented.

**The speed floor cannot run on this fleet, and that is a hardware fact rather
than a scheduling one.** `rc devices` on 2026-08-26 lists `dgx:gpu0`,
`orin:gpu0` and `thor:gpu0`, and all three are integrated parts where host and
device share one physical pool. This row's own §"Unified memory makes it
pointless" says placement moves nothing there. So the fleet can gate
correctness, reachability, refusals, determinism and inertness, and it **cannot
gate W0 or W5 at all**. Those two stay open and visible rather than waived.

## Surpass hypothesis (recorded, not scope)

llama.cpp places weights once, so prefill runs on the CPU too and is slow — the
regime inversion in the bandwidth math.1 is unmitigated upstream. A phase-aware placement would
stream CPU-resident experts to the GPU for prefill (high arithmetic intensity
amortizes the transfer) while computing them on the CPU for decode, taking the
better side of both. This is where `ENG-HYBRID-PLACEMENT` and
`ENG-EXPERT-STREAM` would compose into something neither upstream has. It is
explicitly not built here: it needs W0's measured round-trip cost first, and a
row that tries both loses the ability to gate either.

## Dependencies and blockers

- **Discrete-GPU host** for the speed gate and the bandwidth measurement. The
  whole fleet is integrated: `rc devices` on 2026-08-26 lists `dgx:gpu0`,
  `orin:gpu0` and `thor:gpu0`, and this capability is inert by construction on
  every one of them. **W5 and W0 are BLOCKED until a rig exists**; #149 has
  community test rigs offered ([#147](https://github.com/mudler/vllm.cpp/issues/147)
  records the same offer) and that is the path to unblocking. W1-W4 do not wait
  on it.
- `BACKEND-DISTRIBUTED-TP` (ACTIVE) — no hard ordering, but the seam shape must
  be reviewed against `tensor_parallel.h` before W2 lands.
- No dependency on `ENG-EXPERT-STREAM` or `ENG-WEIGHT-OFFLOAD`; all three are
  independent and compose later.

## Work breakdown

Renumbered on 2026-08-26 ([#2015](https://github.com/mudler/vllm.cpp/issues/2015)).
The first version put two measurements first, because the row's justification is a
bandwidth ratio. That ordering is now **unbuildable on this fleet**: both
measurements need a discrete-GPU host and all three fleet devices are integrated.
Waiting for a rig we do not have would land nothing, so the hardware-independent
waves come first and the measurements keep their own IDs and stay open.

| ID | Work | Gate | Fleet |
|---|---|---|---|
| W1 | The `placement` config surface under `vllm_cpp`: parse, validate, refuse, merge, precedence, and the desugaring of `cpu_moe` / `n_cpu_moe` into `overrides`. No weight moves. Reaches the real server argv and the C ABI | desugaring counted, refusals by name, precedence both directions, absent-means-unchanged | ✅ any box |
| W2 | The `DevicePlacement` seam and pattern→device resolution at model build, reviewed against `tensor_parallel.h`. Resolution only: no behavior change, placement resolves and is reported | first-match-wins order asserted; inertness golden byte-identical | ✅ any box |
| W3 | `cpu_moe` / `n_cpu_moe` routing routed-expert compute to the CPU backend, and the activation round trip | token-exact vs a full-CPU run | ✅ any box |
| W4 | `fit` auto-resolver mirroring `common/fit.cpp`'s project-and-reduce shape, with its reporting, and the refusal beside a manual placement | deterministic resolution test | ✅ any box |
| W5 | Speed floor vs llama.cpp `-ncmoe` at `b10451`, quant-matched, decode and TTFT reported separately | **BLOCKED: needs a discrete-GPU rig** | ❌ none |
| W0 | Measure the real DDR and PCIe bandwidths, and the per-MoE-layer activation round trip, on a discrete-GPU rig; replace the assumed bandwidth table with measured values | measured numbers committed | ❌ none |

**W0 keeps its number and its meaning even though it now runs late.** The
bandwidth table in §"The honest bandwidth math" is still assumed, every figure in
it still comes from published link rates, and **no gate may cite it until W0
replaces it**. Reordering the waves does not promote an assumption to a
measurement. What changed is only that W1–W4 no longer wait on it, because they
build a placement mechanism whose correctness does not depend on whether the
placement is fast.

**The risk this ordering creates, named rather than discovered.** W1–W4 can all go
green while the row delivers no user-visible win, because the win is a ratio nobody
here can measure. A row that is "done except the number" is exactly the shape that
gets mistakenly called done. W5 and W0 therefore stay `OPEN` in this table, the
`## Now` section states them, and `docs/FEATURES.md` may not carry a ✅ for
this capability until one of them lands.

## Risks and decisions

- **The bandwidth ratio is assumed, not measured.** Every number in that table comes
  from published link rates. W0 exists because the row is worthless if the real
  ratio is below ~1.5x, and that is a plausible outcome on a rig with slow
  single-rank memory.
- **The round trip may eat the win.** One H2D+D2H per MoE layer, times layer
  count, times every decoded token. W0 measures it standalone. Under the
  renumbering that measurement now runs AFTER W3 builds the round trip, because
  no box here can take it — so W3 builds against an assumed cost, and the spec
  says so rather than implying the number was in hand.
- **Placement is a second axis next to sharding.** The seam could easily grow
  into a parallel sharding concept and duplicate `BACKEND-DISTRIBUTED-TP`. The
  mitigation is the explicit refusal in Port map and a review against
  `tensor_parallel.h` as W2's gate, not good intentions.
- **The speed gate cannot run on our hardware.** This is the row's real
  scheduling risk: W1-W4 can all land on the fleet while W5 and W0 sit blocked,
  and a row that is "done except the number" is exactly the shape that gets
  mistakenly called done. Both stay open and visible, in `## Work breakdown`,
  in `## Owed` and in `## Now`.
- **`mmap` residency and a CPU placement fight each other, and upstream says so
  out loud.** `llama-model-loader.cpp:1187-1192` warns once that "tensor
  overrides to CPU are used with mmap enabled" and recommends `--load-mode none`.
  This tree's `vllm_cpp.mmap` tier is default-on wherever weights stay quantized
  (`ENG-RESIDENCY-CONFIG`), so the same collision is reachable from a single
  `--offload-config` document that sets both. W1 must decide whether that pairing
  warns, refuses, or silently wins, and assert the choice — not discover it in a
  benchmark.

## Now

`ACTIVE` as of 2026-08-26, claimed by `CLAIM-ENG-HYBRID-PLACEMENT`
([#2015](https://github.com/mudler/vllm.cpp/issues/2015)). The developer directed
the campaign in session: land support for the ways llama.cpp does hybrid offload
and full CPU-MoE offload, mapped onto the configuration this tree already has.

This spec pull request carries no product code. It re-anchors every llama.cpp
citation at the recorded pin `b10451`, adds `## Configuration surface`, and
renumbers `## Work breakdown` so the hardware-independent waves come first. The
pull request shape for this row is **separate spec and implementation**
(developer, 2026-08-26, recorded at row claim).

Next action is W1, the `placement` config surface under `vllm_cpp`. It needs no
GPU and no checkpoint.

**Two waves are hard-blocked and neither is waived.** W5 (the speed floor against
`-ncmoe`) and W0 (the bandwidth and round-trip measurements) both need a
discrete-GPU host. `rc devices` lists `dgx:gpu0`, `orin:gpu0` and `thor:gpu0`,
all integrated parts where this capability is inert by construction. #149 has
community test rigs offered, and [#147](https://github.com/mudler/vllm.cpp/issues/147)
records the same offer; that is the path to unblocking, and until it opens the
bandwidth table stays an assumption that no gate may cite.

Issue [#149](https://github.com/mudler/vllm.cpp/issues/149) is the campaign issue
and stays OPEN — this row covers only its CPU-MoE half. The dense layer-offload
half belongs to `ENG-WEIGHT-OFFLOAD` (`ACTIVE`, config-only, refuses at startup)
and the multi-GPU half to #147 / `BACKEND-DISTRIBUTED-TP` (`ACTIVE`); #149 closes
when all three have landed, not when this one does.

## Owed

- W5, the speed floor against llama.cpp `-ncmoe` at `b10451`. Blocked on a
  discrete-GPU rig, tracked by [#149](https://github.com/mudler/vllm.cpp/issues/149).
- W0, the measured DDR:PCIe ratio and per-MoE-layer round-trip cost that the
  bandwidth table currently assumes. Same blocker, same issue.
- **W3a's `MoePlacementPlan` lands UNREACHED**, declared here as
  `## Nothing lands dead` requires. It resolves a placement to a per-layer
  decision and nothing calls it: W3b routes on it, and
  [#2026](https://github.com/mudler/vllm.cpp/issues/2026) tracks that.
- **W3b cannot be gated on this fleet, and that is a harder blocker than W5's.**
  The placed path only executes when the engine device and the placement device
  DIFFER. Every device here is integrated, and a CPU-only build has no second
  device type at all, so a CPU test can never enter the placed branch — the
  branch would land untested rather than merely unmeasured. W5 and W0 lack a
  NUMBER; W3b lacks the ability to run the code once. Gating it needs a box with
  a discrete accelerator and a CPU backend in one process, which is the same rig
  #149's community offer would supply.
- **W1's two resolvers land UNREACHED**, which `## Nothing lands dead` permits
  only when it is declared, so it is declared here.
  `ResolvePlacementOverrides()` and `ResolvePlacementFit()`
  (`include/vllm/config/weight_residency.h`) have no production caller: W1 parses,
  validates, desugars and reports a placement, and nothing yet acts on one. **W2
  owns the wiring**, and [#2018](https://github.com/mudler/vllm.cpp/issues/2018)
  tracks it. The rest of W1 is reached and mutation-proven — deleting the loader's
  `DescribePlacementResidencyCollision()` call site turns the server suite red
  while the in-process suite stays green.
- The extra CPU buffer-type list that `llama-model-loader.cpp:1186` consults on a
  CPU override. This tree has no equivalent and the difference is recorded as a
  decision under `## llama.cpp anatomy` rather than closed.
