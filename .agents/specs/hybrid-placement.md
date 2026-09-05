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

### W4 design — the `fit` auto-resolver ([#2384](https://github.com/mudler/vllm.cpp/issues/2384))

W1 landed `ResolvePlacementFit()`. It has **no production caller**: declaration,
definition, two test assertions. `--fit` parses, refuses a manual placement
beside it, prints `placement_fit=on` and is announced by `DescribeEnvOverrides`,
and nothing ever asks it for the answer. That is #2382 one layer over, with the
same reassuring-log-line property.

#### What it computes

One question: **how many of the model's MoE layers must run their routed experts
on the CPU for the rest to fit the device budget.** The answer is expressed in
the surface that already exists — a list of `PlacementOverride`, identical in
kind to what `n_cpu_moe` produces — so `fit` adds a resolver and NOT a second
placement mechanism. Everything downstream (`DevicePlacement`,
`MoePlacementPlan::Resolve`, the seam) is unchanged.

#### Two deliberate divergences from upstream, and why each is right HERE

**1. Static total, not live free.** `common/fit.cpp` measures free memory per
device with `ggml_backend_dev_memory` and re-measures on every trial. This tree
takes the opposite position, and it is already ratified in
`include/vt/backend.h:100-103`, directly above `DeviceMemoryInfo`:

> The load-time GGUF fit refusal deliberately does not read this seam: it is a
> live free/total probe, and **a load-time budget must not be a function of
> contention.** It carries its own `total` on
> `vllm::platforms::ResidencyPolicy::device_memory_total_bytes` instead.

So `fit` reads `GetPlatform(dev).residency_policy().device_memory_total_bytes`,
via the existing `DeviceWeightBudgetBytes()` which already applies the operator
overrides. **This is not a capability gap.** CUDA and ROCm both override
`DeviceMemoryInfo` (`src/vt/cuda/cuda_backend.cu:93` calls `cudaMemGetInfo`);
only the base class returns `false`. An earlier draft of #2384 claimed otherwise
and was wrong. Reading live free here would contradict a landed seam decision,
and it would make the same model with the same flags resolve differently
depending on what else was running.

The cost is real and is the honest half: a box with 20 GiB already in use will
be told a model fits when it does not, and fail later at allocation. That is the
same trade `CheckDeviceWeightFit` already makes, so `fit` inherits a known
behaviour rather than inventing one.

**2. Whole layers, not fractional ones.** `common/fit.h:18-22` spills at
`NONE/ATTN/UP/GATE/MOE` granularity and places *part* of one boundary layer.
This spec already sanctions starting coarser — "Our `auto` may legitimately
start coarser, but the difference has to be stated in its reporting rather than
left for a user to discover from a memory figure" — so W4 places whole layers
and SAYS so in its install line. A user who asked for a fit and got 31 of 48
layers placed must be able to see that the resolver could not have given them
30.5.

#### ON BY DEFAULT, mirroring upstream, and what that costs

`common/common.h:468` @ `b10451` sets `fit_params = true`. Upstream's `--fit` is
ON unless turned off, and that is the whole user-visible point of it: a model too
large for the device just runs. Shipping ours off by default would give the
machinery without the experience, and would silently diverge from a flag we
claim to mirror — a ported llama.cpp command line would behave differently here.

So `ResolvePlacementFit()` defaults to `true`. **This changes residency for every
GGUF load that does not fit**: such a load previously failed at allocation and
now runs with experts on the CPU, slower. That is llama.cpp's behaviour and it is
the intended change, but it is a behavioural change across every model and is
recorded as one rather than slipped in.

**Three conditions bound it, each tested and each mutation-proven:**

1. An UNKNOWN budget places NOTHING. `device_memory_total_bytes` is `0` on any
   platform that does not probe one, and a naive comparison would place every
   layer on a box that was never measured — a wrong placement wearing the face
   of a working resolver.
2. **An arm the resolver cannot answer is INERT when DEFAULTED and FATAL when
   ASKED.** Same condition, opposite correct behaviour. Refusing every
   safetensors load over a feature nobody requested would make this default a
   breaking change; telling an operator who explicitly asked that nothing
   happened is the #2382 failure. `PlacementFitWasRequested()` exists precisely
   to tell the two apart, and the distinction is mutation-proven: making the
   refusal unconditional reds 2 cases and 3 assertions.
3. A placement that happens is ANNOUNCED with its arithmetic — budget,
   footprint, layers, bytes — and states the whole-layer granularity, because an
   operator who cannot see both sides cannot tell a wrong placement from a wrong
   budget.

#### Where it resolves, and the asymmetry that shapes the scope

`InstallMoePlacementPlan` (`model_loader.cpp`) is the one place a plan reaches
the seam, so `fit` resolves there. It needs three inputs, and only two are
available on both paths:

| Input | GGUF branch | safetensors branch |
|---|---|---|
| engine device | yes | yes |
| `num_hidden_layers` | yes | yes |
| model weight bytes | **yes** — `GgufStagedWeightFootprint(gguf, ...)`, and the `GgufFile` is already open at the install site | **no** — the shards are not opened until the loader runs, well after the install |

The safetensors size is computable (`qwen3_5_dense_weights.cpp:900` sums
`shard.Get(name).nbytes`) but only once the shards are open, which is downstream
of where the plan must be installed — `ResidentWeight` aliases host bytes on a
CPU `Dev` and uploads otherwise, so installing after the upload pays the round
trip the placement exists to avoid.

**So W4 ships GGUF-only and REFUSES safetensors by name**, naming the missing
part, as `AGENTS.md` requires of an unimplemented arm. A silent "fit resolved
nothing" on safetensors would be exactly the #2382 failure again: a user asks
for a placement, sees no error, and gets none. Reading the index JSON at the
install site to lift the restriction is W4b, not scope creep dressed as W4.

#### The observable the gate needs, which does not exist yet

`InstallMoePlacementPlan` is unconditional, so a fit-installed plan, a
`cpu_moe`-installed plan and a bare one all report the same
`resolved_layer_count()`. A reachability test copied from
`test_placement_reach.cpp` would therefore **pass against a `fit` that did
nothing** — the precise trap W3e was about.

`MoePlacementPlan` gains a provenance field: how the placement was arrived at
(stated / resolved-by-fit / none), plus the budget and footprint the resolver
compared. That is what the reachability test asserts, and it is also what the
install line prints, so the operator sees the arithmetic rather than a verdict.

#### Tests, and what each is worth

1. **Resolver unit, deterministic** — budget and footprint in, layer count out.
   Table-driven over: fits entirely (places nothing), fits with N placed, cannot
   fit even with all placed, UNKNOWN budget (`0`), and zero layers. No device, so
   CI runs it.
2. **UNKNOWN must not read as "nothing fits."** `device_memory_total_bytes` is
   `0` when the probe failed, and `0` compared naively places every layer. This
   gets its own case because the failure is silent and plausible-looking.
3. **Reachability through `FromModelDir`**, asserting the provenance field —
   red without the install call, per the W3e mutation.
4. **The safetensors refusal fires by name**, so the unimplemented arm cannot
   pass as an inert success.
5. **Mutation**: delete the fit branch inside `InstallMoePlacementPlan` and
   require (1)+(3) red on a mutant that COMPILES. A mutant that fails to build
   reads as a passing test (W3g).

#### Two holes in the exclusivity refusal, CLOSED at resolve time

The parse-time refusal inspects ONE document, so two routes reached a state it
would have refused: the multi-document merge inside `SetWeightResidencyConfig`
copies `fit` and the override fields field-by-field and never re-runs the check,
and the environment was never checked at all — `VT_PLACEMENT_FIT=1` beside
`VT_CPU_MOE=1` was refused nowhere.

`DescribePlacementFitCollision()` closes both by asking the RESOLVED values
instead of a parsed document, so it is source-agnostic by construction rather
than by enumerating routes. One check, both holes, and a third route would be
covered too.

**Only for an EXPLICIT fit, and the default flip is what makes that essential.**
`fit` is now on for every load, so a defaulted fit stands beside every manual
placement anyone configures; refusing that would make `cpu_moe` unusable. A
default yields to an explicit instruction — that is not a collision, the manual
placement simply wins. This is the same explicit-vs-defaulted split the
safetensors arm needs, and the two are deliberately the same rule.

Mutation-proven in BOTH directions on mutants that compile at rc=0: treating a
defaulted fit as a collision reds the case that would break every `cpu_moe`
user, and never detecting a collision reds 5 assertions.

#### Stop conditions

- The resolver cannot be made deterministic from static inputs — stop and report.
- Closing the exclusivity holes would change the meaning of an existing
  configuration — stop; that is a config-surface decision, not a W4 call.
- A GPU is needed to prove correctness beyond the unit level — W4 lands the
  resolver and its reachability, and the NMSE-style device gate is owed
  separately, exactly as W3h is for the install.

### W3d — the fp4 refusal, restored AT THE SEAM ([#2309](https://github.com/mudler/vllm.cpp/issues/2309))

`RunMoeBlockPlaced` refused to place a layer whose routed experts are
fp4-resident. W3c moved every architecture onto the shared seam and did not
carry that refusal across; the old helper became dead code and the live path
accepted the arm.

Placing an fp4-resident arm uploads every expert at load and then computes on
the host across the bus, which is slower than not placing. **A token gate cannot
see it** — the tokens stay correct and only the placement is wrong — so nothing
in this tree would have reported it.

The refusal now lives on the seam itself as a `placeable` / `unplaceable_reason`
contract, not in each caller, so a newly wired architecture inherits it instead
of having to remember it. It fires only when a placement is actually in force
(`placed_on != engine_device`), so an ordinary unplaced load is untouched — a
guard that fired there would break every load, which is the opposite failure and
just as bad.

Proved by mutation, not by reading: with the guard rewritten to never fire and
still compiling clean (rc=0, zero errors), `test_device_placement` goes red at 1
case and 2 assertions. The first attempt at this mutation FAILED TO COMPILE
under `-Werror` on the now-unused parameters, and the stale binary reported
19/19 SUCCESS — a passing mutant that proved nothing. The compiling mutant is
the evidence; the build's exit code is part of it.


### W3e — the INSTALL, which is what made any of it move a weight ([#2314](https://github.com/mudler/vllm.cpp/issues/2314))

W1 landed the config, W2 the resolution and the report, W3 the seam and the
architectures. `SetActiveMoePlacementPlan` was called by **nothing in `src/`**.
The seam read a process-global no production path ever wrote, so
`ActiveMoePlacementPlan()` returned the default on every load,
`PlacesAnything()` was always false, and **no expert was ever placed on the
CPU** on any architecture.

W2's own comment names the seam of it: "W2 RESOLVES AND REPORTS; IT MOVES
NOTHING ... W3 owns the routing that reads it." W3 built the routing and never
added the call between them.

**Two things hid it for three work items.** The loader PRINTS
`engine: device placement: N layers on cpu` from the RESOLVED plan, so the one
signal an operator would check confirmed a feature that was not running. And a
token gate cannot see it: with nothing placed, the placed arm is byte-identical
to the unplaced arm, so the end-to-end comparison this row owed would have
PASSED for the wrong reason and been recorded as the feature working.

`LoadedEngine::FromModelDir` now installs the plan on both the GGUF and
safetensors paths, at each branch's config parse — the first point where the
engine device and `num_hidden_layers` are both known, and still ahead of all
weight I/O, which is required rather than tidy: `ResidentWeight` aliases host
bytes on a CPU `Dev` and uploads otherwise, so installing after the upload would
pay exactly the round trip the placement exists to avoid.

The install is UNCONDITIONAL, including when nothing is placed. The plan is a
process-global, so a second load in the same process must overwrite the first
model's plan; an early return on "no overrides" would leave a stale placement
pointed at the wrong model. `test_placement_reach`'s second case asserts that
directly, by installing a 64-layer plan and requiring the next load to re-resolve
against its own depth.

`MoePlacementPlan::resolved_layer_count()` is new and exists for the gate. In a
CPU-only build the engine device IS the placement target, so an installed plan
and a never-installed one agree on every other accessor — both inert, correctly.
The layer count the plan was resolved against is the one observable that
separates them.

**Proved by the reachability mutation, not by reading.** Deleting both call sites
— with the definition kept and `[[maybe_unused]]` so the mutant COMPILES, rc=0
and zero errors — turns `test_placement_reach` red at 2 cases and 3 assertions.

### W3f — the e2e gate RAN, and its invariant was wrong ([#2314](https://github.com/mudler/vllm.cpp/issues/2314))

**Hybrid offload works end to end.** Qwen3.6-35B-A3B bf16 on GB10, engine
`cuda`, `--offload-config '{"vllm_cpp":{"placement":{"cpu_moe":true}}}'`,
announced by the loader as `40 layers run their routed experts on cpu, the rest
on cuda (resolved against 40 layers)`. Both arms exit 0 and both answer the
prompt correctly. This is the first execution of the placed branch on a real
model, and it was only possible after W3e installed the plan.

**The token gate this row owed asserts something unachievable, and the first run
proved it.** The completions share a prefix and diverge mid-sentence:

- unplaced: `Paris, a city renowned for its rich history, culture, and iconic landmarks`
- placed: `Paris, a city renowned for its iconic landmarks such as the Eiffel Tower`

Both correct, both fluent. The cause is not a defect. `docs/FEATURES.md` says
"the round trip is byte-identical to computing in place, mutation-proven", and
that is true of the DATA MOVEMENT -- which is what the round-trip test proves.
It says nothing about the ARITHMETIC: a placed layer runs the CPU MoE kernels
instead of the CUDA ones, and this project's own cross-device bar for reducing
ops is NMSE <= 5e-4, not bitwise equality
(`tests/vt/test_backend_cross_device.cpp:11`, "CPU is the oracle"). Greedy
decode amplifies any perturbation inside that bar into a different token
eventually. llama.cpp's `-ncmoe` diverges the same way.

I designed the gate on the first sentence and did not check it against the
second. The gate was red for a defect in the gate.

**The gate now measures the gateable claim.**
`VT_PLACEMENT_DUMP_MOE=<path>` makes the seam write layer 0's MoE block output
once, on whichever path ran, and the gate computes NMSE between the two arms
against the 5e-4 bar. Inert unless set: one latched `getenv`, first matching
call only, no allocation and no copy on the placed path because `staging`
already holds host bytes.

The instrument precondition is checked BEFORE the comparison, because an absent
or empty dump would make the arms agree trivially -- the vacuous pass this gate
was written to refuse. A missing dump is `GATE_RC=2`, never a pass. The
completions are recorded but no longer asserted.

### W3h — the NMSE gate PASSES, and hybrid offload is correct ([#2382](https://github.com/mudler/vllm.cpp/issues/2382))

Measured on dgx (GB10, `sm_121`, 31 cubins), Qwen3.6-35B-A3B bf16, engine
`cuda`, `--offload-config '{"vllm_cpp":{"placement":{"cpu_moe":true}}}'`, with
the loader announcing `40 layers run their routed experts on cpu, the rest on
cuda`:

```
PLACEMENT CONFIRMED
values=10240  bitwise_identical=9028/10240
NMSE=1.091e-06   bar=5.000e-04
GATE_RC=0        PLACEMENT_E2E=PASS
```

**NMSE is 458x under the bar.** 9028 of 10240 values are bitwise identical and
the remainder differ inside the cross-device tolerance — the signature of the
CPU and CUDA MoE kernels agreeing to within their reduction order, not of a
defect.

Two things this result depends on, both of which had to be fixed first. Without
W3e's install nothing was placed, so the comparison would have been the unplaced
arm against itself. And under the ORIGINAL token invariant this same passing run
reads RED: the completions still diverge mid-sentence. The gate had to measure
the right quantity before a correct implementation could pass it.

The gated tree is `d8cefafd7`; the merged head `5d3462f4a` differs from it only
in files the rebase brought from `main`. All five placement sources
(`moe_placement_seam.h`, `device_placement.{h,cpp}`, `model_loader.cpp`,
`qwen3_moe.cpp`) are byte-identical between the two, so the verdict covers what
lands.

**Still owed, unchanged by this pass:** the SPEED axis. GB10 is unified memory,
so CPU and GPU memory are the same silicon and a placement's throughput benefit
cannot be measured there at all. That needs a discrete CPU/GPU rig
([#149](https://github.com/mudler/vllm.cpp/issues/149)). This gate establishes
CORRECTNESS only.

### W3g — the seam assumed bf16, and the placed branch is untestable on CPU ([#2383](https://github.com/mudler/vllm.cpp/issues/2383))

The seam hardcoded `vt::DType::kBF16` in six places and never compared it with
the block it was given. `kimi_linear_device.cpp:930` hands it an **f32** `[T,H]`
buffer, so the placed branch copied HALF the bytes and reinterpreted f32 as
bf16 — silently, producing plausible floats rather than a crash.

**It could not fire until W3e.** With no plan installed, `placed_on` always
equalled the engine device and the whole placed branch was dead. Installing the
plan opened the door, so the trap behind it is fixed in the same campaign rather
than left for whoever first placed a Kimi-Linear layer.

The seam now carries `dh.dtype`, sizes the copy-back by the dtype the body
actually produced — which need not equal the input's — and hardcodes bf16
nowhere.

**Why nothing caught it, which is the finding worth keeping.**
`RunMoePlaced` takes its engine device from `engine.q.device.type`, and the CPU
is the only legal placement target, so on a CPU-only build `placed_on` always
equals the engine device and **the placed branch is unreachable**. Every unit
test and all of CI exercise only the inert path; the placed branch's sole
execution is on a GPU box. An entire branch of a shared seam that six
architecture families route through has no coverage any merge gate can see.

`tests/vllm/model_executor/test_placement_dump_dtype.cpp` covers the dump's
half, in its OWN binary because `VT_PLACEMENT_DUMP_MOE` latches on first use and
`test_device_placement` drives the seam first — sharing a binary would latch it
to "unset" and the suite would pass while asserting nothing. Mutation-proven at
1 case / 4 assertions on a mutant that compiles at rc=0.

**Owed — CLOSED by W3i.** This said the placed branch was untested on CPU and
that closing it needed "a loopback placement target or a GPU-gated test". The
first half was true; the second was a false dichotomy. The loopback belongs on
the ENGINE, not on the placement target, and `test_placed_moe_roundtrip.cpp` now
executes the branch on a CPU-only build. The paragraph above still states the
reason nothing caught the truncation at the time, which remains accurate history.

### W3i — the placed branch IS reachable on a CPU-only build ([#2714](https://github.com/mudler/vllm.cpp/issues/2714))

Three places in this spec, and issue #2714 itself, record the same conclusion:
the placed branch needs a GPU or a Vulkan/lavapipe build, because the CPU is the
only legal placement TARGET and a CPU engine therefore always short-circuits.
**The conclusion is wrong, and it was wrong when it was written.**

`PlacementQueue` constrains the DESTINATION. It says nothing about the engine,
and `RunMoePlaced` reads the engine device from `engine.q.device.type` — a field
on a `vt::Queue` the caller supplies. So the branch fires whenever the engine
identifies as anything other than the CPU. It does not need the engine to BE a
GPU; it needs the engine to be a different registered device.

That is a thing this repository already builds, in five files, with a comment in
`test_device_pool.cpp:476` naming `kXPU` as "the type with no `RegisterPlatform`
call anywhere in the tree":

    tests/vllm/entrypoints/test_device_selection.cpp:72
    tests/vllm/entrypoints/test_gguf_device_fit_reach.cpp:150
    tests/vllm/model_executor/test_expert_stream_device_slot.cpp:141
    tests/vllm/model_executor/test_resident_weight_f32_copy_retires.cpp:152
    tests/vllm/model_executor/test_resident_weight_host_addressable.cpp:142

So the engine is a **loopback backend** registered on `kXPU`: a distinct device
identity whose `Alloc`, `Copy`, `Memset` and `Synchronize` delegate to the real
CPU backend. Placement stays `kCPU`. `placed_on != engine_device`, and every line
of the round trip executes — the down copy sized by `dh.dtype`, the block on the
placement queue, the up copy sized by the OUTPUT dtype, the dump hook, and the
hand-back into an engine-pool `DBuf`.

**What this proves and what it does not.** Because both sides are host memory,
the two arms are comparable BYTE-FOR-BYTE rather than within a tolerance, which
is the right bar: placement is a scheduling decision and must never change a
value. It does not prove that a real PCIe or NVLink transfer works — the W3h GB10
run at NMSE 5.239e-06 is that evidence, and the two are complements. What it adds
is the half W3h cannot give: a gate that runs on every merge, on every CI tier,
with no GPU and no checkpoint.

**Why it matters more than a restored test.** W3g's f32 truncation shipped, was
merged, and was reviewed, because no reviewer could execute the branch. The
mutation `reachability.md` asks for is now runnable here: hardcode `kBF16` back
into the seam's six sites and this suite reds.

**Mutation evidence.** 5 cases / 28 assertions green on `d023e3357`. Every
claimed guarantee was mutated in the seam header, and **each mutant compiled at
`rc=0` before its result was believed** — a mutant that fails to build reads as a
passing test, and this tree has been fooled by that before.

| Mutation | Result | The assertion that caught it |
|---|---|---|
| `dt = kBF16` instead of `dh.dtype` (the #2383 defect verbatim) | 2 cases / 3 assertions RED | `copies[0]`: `32 == 64` — half the bytes |
| `placed_on = engine_device` (never cross; the pre-#2382 state) | 4 cases / 4 assertions RED | `copies.size()`: `1 == 2`, and the round trip throws |
| `out_dt = dt` instead of the body's output dtype | 1 case / 2 assertions RED | `copies[1]`: `64 == 32` — twice the bytes out |
| `if (false && ... !placeable)` (drop the fp4 refusal) | 1 case / 2 assertions RED | `CHECK_THROWS_AS ... did NOT throw at all` |

The header was restored byte-for-byte after each (`git status` clean, sha256
`bee99638a9c6`), rebuilt, and re-run green.

**The second mutation exposed a structural property worth keeping.** With the
crossing removed, the round-trip case does not merely disagree — it THROWS, with
`no kernel for op Matmul on device xpu`. The loopback backend deliberately
carries transfers and allocation but no kernels, so a version of this test that
silently fell back into the short circuit cannot quietly pass. The failure mode
this file exists to prevent is structurally unavailable to it.

`tests/vllm/model_executor/test_placed_moe_roundtrip.cpp` restores the name
`6416aab85` deleted, on the mechanism the old one lacked. The old file forced the
round trip by passing `kCPU` as an explicit placement argument to
`RunMoeBlockPlaced`, a parameter the shared seam does not have; this one crosses
a real device boundary instead of asking for one.

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

Next action was W1 when this was written; W1 through W4 have since landed, and
W3i closed the last gate that did not need hardware. What remains needs a
discrete GPU, as the paragraph below says.

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

## W4/W3 outcome — the device gate RAN, and both arms are within the bar

Measured on `dgx:gpu0` (GB10, `sm_121`), `Qwen3.8-Flash-Next-UD-IQ1_S` (67.564
GiB, three shards, the artifact `docs/USAGE.md` pins), engine `cuda`, tree
`b66b2b060`. Four arms on ONE binary, each placed arm compared against a
reference dumping the SAME layer:

| arm | placement | dumped layer | bitwise identical | NMSE | bar |
|---|---|---|---|---|---|
| `cpu_moe` | 48/48, `origin stated` | 0 | 9286/12800 | **5.239e-06** | 5e-04 |
| `fit` | 23/48, `origin fit` | 25 | 3729/12800 | **3.569e-05** | 5e-04 |

Both WITHIN, and reproduced identically across two consecutive runs including
the identical-value counts. `--fit` priced the model at 72,378,689,280 B, took 23
trailing layers (19,333,120,000 B) to fit a 53,687,091,200 B budget, and said so.

**THE LAYER CHOICE IS THE MEASUREMENT, and two wrong ones came first.** Layer 0
is VACUOUS for `fit` — the resolver places TRAILING layers, so both arms computed
it identically and the gate reported a perfect NMSE of 0 over 12800
bitwise-identical values. Layer 47 is CONFOUNDED for both — by then the input has
crossed 46 other placed layers, so the comparison measures ACCUMULATED
divergence, and it read 8.204e-04 against a bar that bounds ONE op. Only each
arm's FIRST placed layer has an identical input in both arms, which is what makes
the per-op bar meaningful. The 23x gap between `fit` at layer 25 (3.569e-05) and
at layer 47 (8.204e-04), from identical code, is that accumulation measured.

A perfect score was the signal that something was wrong, not right: a
cross-device round trip does not reproduce bit-for-bit, so all-identical means
nothing moved. The gate now refuses that case by name.

## Owed

- **`qwen4_exp` is WIRED to the seam and now REACHED.** The entry below recorded
  it as unreached, which was accurate until this gate ran: the wiring is
  `qwen4_exp_forward.cpp` routing its MoE step through `RunMoePlacedPair` with
  the `qwen3_moe.cpp` adapter verbatim, and the `cpu_moe` arm above executes it
  on a real checkpoint at 5.239e-06. What made that possible was three CUDA fixes
  landing meanwhile — `#2449` (inject-weight residency), `a4ced1b13` (rmsnorm
  widened gamma) and `a578705e9` (device-aware `quant_repack`) — after which the
  `cudaMemcpyAsync: illegal memory access` seen earlier did not recur, so it was
  a symptom of those rather than a fourth defect.

  What is NOT proven is that anything reaches it. The forward needs a
  `qwen4_exp` GGUF, and the only one available is `Qwen3.8-2.4T-A95B UD-Q1_0` at
  369 GiB, which fits no host this project reaches. So the deletion mutation
  `reachability.md` asks for cannot be run, and no gate here holds the call site
  in either direction.

  This is the same position `scripts/runner-routing-allowlist.txt` records for
  Nemotron-H, and it is recorded the same way rather than being dressed up: a
  seam checker never proved reachability and was never meant to. The wiring is
  correct BY CONSTRUCTION (shape match, compile), which is weaker than reached
  and is not claimed as more.

  It also unblocks something: `--fit` has no device gate because every GGUF on
  the NAS is either off the seam (`laguna`), has no reachable MoE forward
  (`glm5next`), or does not fit. A runnable `qwen4_exp` GGUF would be the first
  checkpoint that is both GGUF and seam-wired, which is what a `--fit` NMSE gate
  needs.


- W5, the speed floor against llama.cpp `-ncmoe` at `b10451`. Blocked on a
  discrete-GPU rig, tracked by [#149](https://github.com/mudler/vllm.cpp/issues/149).
- W0, the measured DDR:PCIe ratio and per-MoE-layer round-trip cost that the
  bandwidth table currently assumes. Same blocker, same issue.
- **Placement reaches FIVE architectures, and the earlier entry here undercounted
  by reading one file per model.** Wired through `RunMoePlaced`:
  `Qwen3MoeForCausalLM`, Qwen3.5/3.6 (`RunLayer` and `RunLayerPaged`),
  Nemotron-H, DeepSeek-V2, and **Kimi-Linear** — whose `MoeBlockDevice` and
  `MoeBlockDeviceBf16` (`kimi_linear_device.cpp:804,1094`) are seam-shaped and
  were missed because a previous sweep read `kimi_linear_forward.cpp`, saw a host
  `std::vector<float>` path, and generalised it to the architecture. That is the
  same mistake this row made about Laguna, made twice.

  Still out, and the reasons are NOT interchangeable: **Laguna** computes its
  experts on the device but presents a per-token host-float FFN boundary, so it
  has no `[T,H]` block to hand the seam ([#2050](https://github.com/mudler/vllm.cpp/issues/2050));
  **Gemma4**'s `ExpertGeGLUHost` / `ExpertGeGLUDeviceAccum` accumulate into a
  caller's buffer and return `void`, which is a different contract rather than a
  different spelling; **DeepSeek-V4** runs its experts on the host from host
  weights, where a placement has nothing to move; and GLM-5-Next, dots3-note,
  Kimi-K3 and qwen4_exp have no reachable MoE forward at all yet.
- **W3b's forward branch is not test-driven, and since 2026-08-30 NEITHER IS THE
  TRANSFER IT CALLS** (#2345). This bullet used to say `RunMoeBlockPlaced`
  executed under `test_placed_moe_roundtrip`, byte-identical to the direct call
  and mutation-proven. Both halves are now false: `866075b2f` (#2309) deleted the
  helper once W3c had made it dead, and `6416aab85` (#2331) deleted the test,
  which was calling a symbol that no longer existed and had stopped `main`
  building.

  **CLOSED by W3i (#2714).** Both halves above are history and stay as written.
  What followed them was an error worth keeping visible: this bullet concluded
  the gate "cannot simply be rewritten" for a "structural" reason, and the
  structure it named is real but does not have that consequence. The seam does
  short-circuit when the placement device equals the engine device, and the
  transfer IS reachable only cross-device -- so the fix is to move the ENGINE,
  which needs no GPU. `test_placed_moe_roundtrip.cpp` does that with a loopback
  backend on `kXPU`, and the round trip is byte-for-byte gated again on every CI
  tier.

  The selection branch is entered by the same test, for the same reason: it
  needed the engine device and the placement device to differ, and a loopback
  engine makes them differ. The Vulkan gate this row used to owe was never the
  only way to get there.
- ~~**W3a's `MoePlacementPlan` lands UNREACHED**~~ — CLOSED by W3b, which reads
  the plan in `RunMoeLayer`.
- **W3a's `MoePlacementPlan` lands UNREACHED**, declared here as
  `## Nothing lands dead` requires. It resolves a placement to a per-layer
  decision and nothing calls it: W3b routes on it, and
  [#2026](https://github.com/mudler/vllm.cpp/issues/2026) tracks that.
- **W3b IS gateable here, and the entry that said otherwise was wrong.** It
  claimed the placed path could not be run once on this hardware, reasoning that
  every fleet device is integrated and a CPU-only build has no second device
  type. The second half is true and the conclusion does not follow: the placed
  branch needs the engine device and the placement device to DIFFER, not a
  discrete accelerator. **Vulkan is a distinct `vt::DeviceType` and lavapipe is
  installed on this box** (`/usr/share/vulkan/icd.d/lvp_icd.json`), so a Vulkan
  engine with `cpu_moe` enters the placed branch on the machine this was written
  on. `BACKEND-VULKAN` already runs models token-exactly, which is what makes it
  an admissible correctness engine rather than merely a second enum value.
  So W3b owes a token-exactness gate — Vulkan engine with the routed experts
  placed on the CPU, against the same model run wholly on the CPU — and that gate
  is runnable today. What Vulkan-on-lavapipe CANNOT supply is a speed number, and
  it must never be used for one: it is a software rasteriser, so a placement
  measured against it would compare CPU against CPU. The SPEED axis stays with
  W5 and stays blocked on a discrete rig, which is the same rig #149's community
  offer would supply.
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
