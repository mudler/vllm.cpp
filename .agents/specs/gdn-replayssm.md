# ReplaySSM buffered decode for the GDN path — `KERNEL-GDN-REPLAYSSM`

Issue [#1171](https://github.com/mudler/vllm.cpp/issues/1171). Row
`KERNEL-GDN-REPLAYSSM` ([kernel matrix](../kernel-matrix.md)). vLLM parity pin
`5559679229bc961848b121ccdeaa8fa5d79bec98`
([upstream-sync.md](../upstream-sync.md)); SGLang oracle pin
`f63458b5beaceabbd9d749b9fc956370e1b649e6` (v0.5.15,
[oracles/sglang.md](../oracles/sglang.md)). Every anchor below was read at those
two revisions.

## Scope

GDN decode reads the temporal state and writes it back on every token. Per
request, per GDN layer, that state is `[HV, V, K]` in the SSM cache dtype. Our
kernel loads the tile at `src/vt/cuda/cuda_gdn.cu:2393` and stores it at
`:2425`. At the 27B GDN shape (`HV=32, V=128, K=128`, config fixture
`tests/vllm/test_hf_config.cpp:80-82`) one layer moves 2,097,152 bytes in and
2,097,152 bytes out, per request, per token.

ReplaySSM removes the write. It keeps a per-slot ring of the last `L` steps'
rank-1 factors, reconstructs the state in registers on every step, and writes the
full state back only once per `L` steps.

**In scope.** The buffered decode kernel for the non-speculative GDN decode path,
its ring in the `MambaSpec`, the per-slot cursor and flush flag, an opt-in
default-OFF selector, the equivalence tests, and the A/B that would establish a
payoff.

**Out of scope, and each is named again under `## Owed`.** The Mamba2 arm (that
is a straight vLLM mirror and belongs to `KERNEL-SSM-MAMBA` / Nemotron-H, not
here); the KDA arm (SGLang measured it SLOWER, `server_args.py:1976-1981`);
speculative decode (vLLM refuses it outright and we mirror the refusal); prefill;
any non-CUDA backend.

**What this spec does NOT do: claim a speedup.** No number in this document is a
measurement taken on this hardware. *The A/B* states the A/B that would produce one, and
*Traffic model* states three separate reasons the win is smaller than the headline arithmetic
suggests.

## Upstream chain (vLLM at `555967922`) — the algorithm, on a path that cannot reach GDN

### The kernel and its config

| What | Anchor @ `555967922` |
|---|---|
| Kernel file (709 lines) | `vllm/model_executor/layers/mamba/ops/selective_state_update_replayssm_output_only.py` |
| Precompute pass (per-step `k_j . q` products) | `:22-129` (`_replayssm_output_only_precompute_kernel`) |
| Main kernel | `:131-472` (`_replayssm_output_only_kernel`) |
| Non-flush route — "read `y` without materializing the state" | `:275-276` and the comment at `:277-279` |
| Flush route — "reconstruct the state, persist it, then read `y`" | `:358-362` |
| Python entry | `:474-523` (`selective_state_update_replayssm_output_only`) |
| Launch-config heuristic + `override_replayssm_config` | `ops/replayssm_config.py` (`_mamba2_output_only`, Blackwell-vs-not) |
| Introduced by | `866fea2b`, "[Kernel] ReplaySSM: cache SSM inputs for faster Mamba2 standard decode", vllm#48018 |

The kernel's own comments state the two routes exactly, and they are the
mathematics this row ports:

- non-flush: `y = total_decay * (S_0 q) + sum_j s_j (k_j^T q) v_j` (`:277-279`)
- flush: `S_t = total_decay * S_0 + sum_j s_j (v_j k_j^T)`, persist, then
  `y = S_t q` (`:359-361`)

### The ring, in the state spec

vLLM does not put the ring in a side allocation. It appends it to the mamba
state tuple, which is the same surface our `MambaSpec` fills:

| Ring tensor | Shape | Dtype | Anchor |
|---|---|---|---|
| `x_cache` | `(nheads/tp, L, head_dim)` | activation | `mamba_utils.py:202-221` |
| `dt_cache` | `(nheads/tp, L)` | **fp32** | same |
| `B_cache` | `(ngroups/tp, L, state_size)` | activation | same |

`MambaStateDtypeCalculator.append_replayssm_ring` (`mamba_utils.py:84-93`)
returns `(*base, activation, torch.float32, activation)`. The layer calls both
appenders only under `use_replayssm` (`mamba_mixer2.py:1105-1140`).

The decisive structural fact: `MambaStateDtypeCalculator.gated_delta_net_state_dtype`
(`mamba_utils.py:120-128`) does **not** call `append_replayssm_ring`. GDN's state
tuple has no ring at the pin.

### The cursor, the flush flag, and the metadata

| What | Anchor |
|---|---|
| `write_pos_d`, `is_flush_d`, `bc_pre_scratch` on the mamba metadata | `v1/attention/backends/mamba_attn.py:77-82` |
| Ring origin carried on the shared metadata, re-anchored past a preemption | `v1/attention/backend.py:483-486` (`replayssm_decode_base_cpu`) |
| Cursor derivation | `mamba_attn.py:575-620`; `write_pos = decode_steps mod L` (`:617`), `is_flush` at `:618` |
| First step after a prompt is a flush over an empty history | `mamba_attn.py:607` |
| Buffer length, default **16** | `config/cache.py:148` (`replayssm_buffer_len`, `gt=0`) |
| Master switch, default **`False`** | `config/cache.py:152` (`use_replayssm`) |

### Why it cannot reach GDN upstream — four independent walls

1. **Config refusal.** `config/vllm.py:2318-2322` raises unless
   `model_config.supports_replayssm`. The message names Nemotron-H explicitly.
2. **One opted-in model.** `SupportsReplaySSM` is a base of
   `NemotronHForCausalLM` alone (`models/nemotron_h.py:711`), which appends the
   ring at `:755` and `:792`.
3. **Different builder tree.** `GDNAttentionMetadataBuilder` derives from
   `AttentionMetadataBuilder` (`v1/attention/backends/gdn_attn.py:82`), not from
   the `BaseMambaAttentionMetadataBuilder` that computes the cursor
   (`mamba_attn.py:575-620`). Nothing in the GDN metadata carries `write_pos`.
4. **Kernel shape contract.** The kernel asserts a scalar-per-head `A`
   ("Cached kernel requires TIE_HDIM", `:540-542`) and the Mamba2 `(B, C)` group
   structure (`nheads % ngroups == 0`, `:528-530`). GDN's decay is a per-head
   scalar and its key is grouped over `H_k`, which is analogous but not the same
   tensor layout.

**This is still true past the pin.** In the pinned clone, fetched 2026-08-15 and
877 commits ahead of `555967922`,
`git ls-tree -r --name-only origin/main | grep -i replayssm` returns only the
Mamba2 files, `vllm/model_executor/layers/mamba/gdn/` contains no match, and
`vllm/model_executor/layers/fla/` does not exist on either revision. So a pin
advance does not hand us this.

### Upstream refusals to mirror verbatim

`config/vllm.py` refuses, in this order: a non-opted-in architecture (`:2318`),
`mamba_cache_mode == "all"` (`:2324`), **speculative decoding** (`:2329`), a
non-Triton mamba backend (`:2331`), and KV connectors (`:2337`). The speculative
refusal is the one that matters here: our GDN path has a shipped speculative arm
(`vt::GdnSpecDecode`, `include/vt/ops.h:2380-2384`) whose whole correctness
argument is a per-timestep state snapshot. A ring that defers the state write is
incompatible with that argument, and upstream reached the same conclusion.

## Upstream chain (SGLang at `f63458b5`) — the GDN application

At `f63458b5`:

| What | Anchor |
|---|---|
| Kernel (628 lines) | `python/sglang/srt/layers/attention/fla/fused_recurrent_linear_replayssm.py` |
| Credit line | `:50` — "Ported from `vllm/model_executor/layers/fla/ops/fused_recurrent_replayssm.py` (ReplaySSM, commit 3c85112)" |
| `L=1` reduces algebraically to packed decode | `:45-47` |
| `--enable-linear-replayssm`, default `False` | `python/sglang/srt/server_args.py:1972-1983` |
| `--linear-replayssm-cache-len`, default **16** | `server_args.py:1984-1987` |
| Ring allocation | `python/sglang/srt/mem_cache/memory_pool.py:465-487` |
| Per-slot cursor `replayssm_write_pos`, `(size+1,)` int32, shared across layers | `memory_pool.py:641-645` |
| Introduced by | `a10a24e9a7fdc0ddcc7dc3e90c0d4983707db028`, sglang#28451 |

SGLang's ring, which is the layout this row ports:

| Ring tensor | Shape | Dtype |
|---|---|---|
| `replayssm_d` | `(layers, slots, HV, L, V)` | SSM dtype |
| `replayssm_k` | `(layers, slots, H_k, L, K)` | SSM dtype |
| `replayssm_g` | `(layers, slots, HV, L)` | **fp32** always |

The mapping onto vLLM's ring is one-to-one and is why this is a mirror rather
than an invention: `d` is `x_cache`, `g` is `dt_cache` (fp32 in both), and `k` is
`B_cache` — grouped over `H_k` exactly as `B_cache` is grouped over `n_groups`.
The KDA variant widens `g` to `(…, HV, L, K)` (`memory_pool.py:477-482`).

**SGLang's own recorded caveats**, from the flag help at `server_args.py:1976-1981`:
the claim is "~1.2-1.5x at batch >= 64"; KDA is "SLOWER than the packed baseline"
because its per-K `g_cache` is `K` times larger; and the flag "Requires the
Triton linear-attn decode backend and `--mamba-scheduler-strategy no_buffer`".

**The credit line points at a file vLLM does not have.** No
`layers/fla/ops/fused_recurrent_replayssm.py` exists at our pin or 877 commits
past it, and `3c85112` does not resolve in the pinned clone. Treat SGLang as the
secondary oracle for the GDN *application* under `AGENTS.md` "When vLLM has no
implementation", and vLLM's Mamba2 kernel as the mirror source for the
*algorithm*. Do not cite SGLang for anything vLLM defines.

## Our baseline

| What | Anchor |
|---|---|
| Legacy shared-memory packed decode | `src/vt/cuda/cuda_gdn.cu:2312-2427` (state read `:2393`, write `:2425`) |
| Register-tiled packed decode, default **OFF** after a failed proof | `:2505-2630` |
| Fused decode | `:2793` |
| Sequential scan decode | `:2209` |
| CUDA dispatch, incl. the Triton bridge | `:2668-2760` |
| Vendored Triton AOT bridge | `:5189-5245` (`TryTritonPackedDecode`) |
| Speculative arm | `include/vt/ops.h:2380-2384` (`vt::GdnSpecDecode`) |
| State spec (two shape/dtype pairs today) | `src/vllm/model_executor/models/qwen3_5_common.cpp:37-106` |
| SSM cache dtype resolution, mirrors `MambaStateDtypeCalculator` | `src/vllm/model_executor/models/qwen3_5.cpp:460-470` |
| Diagnostic dtype override | `qwen3_5_common.cpp:55-63` (`VT_GDN_STATE_BF16`) |

`grep -rniE 'replayssm' src include tests` returns nothing.

### The constraint that shapes the whole design

The 27B GDN decode default is a **vendored vLLM FLA cubin**, not our own kernel.
`TryTritonPackedDecode` fires first (`cuda_gdn.cu:2696-2706`), and it hard-guards
`state.dtype == kF32` (`:5212`) plus every stride and the exact
`Dk=Dv=128, H_k=16, HV in {32,48}` geometry (`:5206-5216`). A ReplaySSM kernel
cannot be that cubin — no such cubin exists upstream, because upstream's
ReplaySSM is Mamba2-only Triton and SGLang's lives in a third repository.

This is not a packaging inconvenience. The header at
`src/vt/cuda/gdn_packed_decode_triton.h:9-14` records a **measured** codegen
result: the identical register-resident `[BV=32, BK=128]` fp32 state tile is
`REG:205` with zero spill under Triton/ptxas, and `REG:255 + STACK:48` (spilling)
as hand-written CUDA. The register-tiled hand kernel that this measurement came
from is default-OFF today precisely because it lost, `c16` 700 against
794 tok/s (`cuda_gdn.cu:2710`). A hand ReplaySSM kernel carries strictly
*more* register state than that one, because it must also hold the reconstruction
accumulator. *Traffic model* R1 makes this the top risk.

## Design

### The recurrence and why the readout distributes

Our decode step, read off `cuda_gdn.cu:2398-2420`:

```
S_t = g_t * S_{t-1} + d_t k_t^T        d_t = beta_t * (v_t - (g_t S_{t-1}) k_t)
y_t = S_t q_t
```

`S` is `[V, K]` per `(request, v-head)`, `g_t` a per-head scalar, `k_t` a
`K`-vector shared across the v-heads of a k-head group, `d_t` a `V`-vector.

Unroll from the checkpoint `S_0` over the `m` buffered records, with
`G = exp(sum_j g_j)` and `W_j = G / exp(cumsum_j)`:

```
S_t = G * S_0 + sum_j W_j d_j k_j^T
y_t = G * (S_0 q_t) + sum_j W_j (k_j . q_t) d_j
```

The readout distributes over the outer products. `S_0` is therefore contracted
against `q_t` — and, for the `d_t` term, against `k_t` — tile by tile in
registers, and the updated state never reaches HBM on a non-flush step. This is
vLLM's non-flush route (`:277-279`) with GDN's names substituted, and the
`(k_j . q_t)` products are what vLLM precomputes in its first pass (`:22-129`).

### Ring layout

Mirror SGLang's GDN layout, which is vLLM's ring renamed. Appended to the
existing `MambaSpec` shape and dtype vectors at
`qwen3_5_common.cpp:84-87`, so the ring is part of the paged state and the block
manager sizes it. Never a side allocation, for the reason vLLM's own comment at
`config/vllm.py:2315-2317` gives: a ring outside the spec lets the page size
desync.

| Tensor | Shape per slot | Dtype |
|---|---|---|
| `gdn_replay_d` | `(HV, L, V)` | SSM cache dtype |
| `gdn_replay_k` | `(H_k, L, K)` | SSM cache dtype |
| `gdn_replay_g` | `(HV, L)` | fp32, always |

`g` stays fp32 in both upstreams and must here too: it is a log-decay that is
summed over the window, so it is the one term where the reconstruction
accumulates error linearly in `L`.

### Cursor, flush, and the boundaries

Per-slot `write_pos`, `int32`, one vector shared across GDN layers
(SGLang `memory_pool.py:641-645`), advanced once per decode forward, with
`write_pos = decode_steps mod L` and `is_flush` derived from it exactly as
vLLM does at `mamba_attn.py:617-618`. Boundaries, each mirroring a named
upstream decision:

| Boundary | Behavior | Mirrors |
|---|---|---|
| First decode step after prefill | flush over an empty history; prefill writes the state normally and the ring starts empty | `mamba_attn.py:607` |
| Ring origin after preemption or a resumed request | re-anchor to the current `num_computed` rather than counting from zero | `backend.py:483-486` |
| Chunked-prefill boundary | unchanged; the ring is a decode-only structure | vLLM keeps it on the decode metadata (`mamba_attn.py:575`) |
| Eviction / slot reuse | zero the cursor on slot (re)allocation | `memory_pool.py:643-645` |
| Speculative decode | **REFUSE** by name | `config/vllm.py:2329` |
| CUDA-graph capture | the cursor is a device tensor the graph reads; `write_pos` must be advanced *outside* capture, and a captured graph must not bake a flush decision | see R3 |

### Selector

`VT_GDN_REPLAYSSM`, **default OFF**, same-binary rollback, matching both
upstream defaults (`config/cache.py:152`, `server_args.py:1983`) and the house
convention for an unproven lever. `VT_GDN_REPLAYSSM_LEN`, default 16, matching
both. Documented in `docs/ENVIRONMENT.md` in the change that lands the kernel,
not here.

Default-OFF is not timidity, it is the only defensible state: this lever is
unmeasured on our hardware (*Traffic model*, *The A/B*) and it drops the 27B off its vendored-cubin
parity path (*The constraint that shapes the whole design*).

### Correctness — NOT bit-exact, and neither upstream claims it is

The reconstruction re-associates the recurrence. Instead of `L` sequential
fused updates it forms one weighted sum over `L` records, so the floating-point
accumulation order differs by construction, and the decay is refolded as
`exp(sum g)` rather than an `L`-fold product of `exp(g)`. **A token gate cannot
see this** on its own; it is a numeric change that argmax will usually absorb.

Both upstreams gate it as an approximate equivalence:

- vLLM: fp32 `rtol=1e-4, atol=1e-3`, and bf16 a very loose `6e-2, 2e-1`
  (`tests/kernels/mamba/test_replayssm_standard_decode_mamba2.py:47-57`). Its
  comment says why the fp32 pair is what it is: a correct fp32 reconstruction
  matches to ~1e-5 while a TF32 one drifts to ~1e-2, so `atol=1e-3` sits between
  and **flags a TF32 reconstruction** rather than merely passing a good one.
- SGLang: fp32 `atol=1e-4, rtol=1e-3` and bf16 `atol=1e-3, rtol=1e-2`
  (`test_linear_replayssm_decode.py:83-88`) — note the atol/rtol roles are
  swapped relative to vLLM, so do not copy one pair onto the other's argument
  order. It reserves `atol=2e-6, rtol=1e-5` for `L=1` **alone**
  (`:226-231`), where the kernel reduces algebraically to the packed decode
  (`fused_recurrent_linear_replayssm.py:45-47`)

So `L=1` is the one configuration that must be *near-exact*, and it is the
red-first anchor of *Tests to port* for exactly that reason: it is the case where the buffered
kernel and the shipped kernel compute the same expression, so any difference at
`L=1` is a porting bug and not a reassociation.

### The two levers: compose on bytes, conflict on the path, unknown on numerics

The brief asks whether ReplaySSM and a bf16 SSM state compose or compete. Three
different answers, and they must not be collapsed:

1. **On bytes they compose.** They attack orthogonal factors: the dtype halves
   bytes-per-element, ReplaySSM removes one of the two state touches per step.
   The ring overhead ratio is near-invariant to the dtype — 18.85% of the state
   at fp32 and 18.94% at bf16 for the 27B shape, because only `g` stays fp32.
2. **On the executed path they already conflict, today, without ReplaySSM.**
   `TryTritonPackedDecode` requires `state.dtype == kF32` (`cuda_gdn.cu:5212`),
   so `VT_GDN_STATE_BF16=1` silently drops the 27B off the vendored cubin and
   onto the hand kernel. Any measurement that varies the state dtype and reads
   the result as a bandwidth effect is confounded by a kernel swap. This is a
   pre-existing trap that this row must not walk into, and it is the reason
   `.agents/issue-index.md` already carries [#491](https://github.com/mudler/vllm.cpp/issues/491)
   ("the two GDN packed-decode toggles are NOT independent: no 2x2 exists").
3. **On numerics the composition is untested on both sides.** bf16 state plus
   ReplaySSM means the checkpoint is rounded to bf16 every `L` steps instead of
   every step — *fewer* rounding events, but each reconstruction runs over
   bf16 records. Neither upstream test matrix covers bf16 state with `L > 1`.
   Whether that is better or worse than the shipped path is an open question, not
   a detail.

**On the "153.9 MB fp32 -> 78.4 MB bf16" figures.** This row was briefed with
that pair for the whole-model state slot. It is recorded here as *reported, not
reproduced*, because it does not re-derive from anything this tree holds. What
does reconcile: the pair implies a halving part of 151.0 MB and a non-halving
remainder of 2.9 MB, and 151.0 MB divided by the per-layer fp32 SSM state of
2,097,152 B is **72.0 exactly**, so the figures are consistent with 72 GDN layers
at `HV=32, V=128, K=128`. What does not: the 2.9 MB remainder is not the conv
state those 72 layers would carry (72 x 8192 x 3 x 2 = 3.54 MB), and no file in
this tree records the Qwen3.8-27B GDN layer count, so 72 could not be confirmed.
Do not quote the pair as measured until someone reads the layer count off the
checkpoint. A number that gets quoted often enough starts being treated as
measured, which is how an unchecked figure becomes a premise.

Our `mamba_ssm_dtype` handling already mirrors upstream properly
(`qwen3_5.cpp:460-470` against `MambaStateDtypeCalculator._mamba_state_dtype`,
`mamba_utils.py:96-108`), so the bf16 arm is a config key and not a hack. That
is a separate row's business.

## Traffic model, and why the headline number is wrong

Symbols: `S = HV*V*K*s` bytes of state (`s` = SSM dtype width),
`w = (L-1)/2` the mean live window.

**Lever OFF**, per step, per layer, per request: `2S`.

**Lever ON**, averaged over the `L`-cycle:

| Term | Bytes |
|---|---|
| checkpoint read, every step | `S` |
| flush re-read + write, once per `L` | `2S/L` |
| ring write, every step | `(HV*V + H_k*K + HV)*s` |
| ring read, every step | `w*(HV*V + a*H_k*K)*s` |

`a` is the `k`-record re-read amplification: the kernel re-reads a `k` record per
v-head group and per v-tile. Take `a = 2` at the 27B shape (`HV/H_k = 2`).

At `HV=32, V=128, K=128, H_k=16, s=4, L=16`:

| Quantity | Bytes |
|---|---|
| `S` | 2,097,152 |
| OFF total | 4,194,304 |
| ON: checkpoint + flush | 2,359,296 |
| ON: ring write | 24,704 |
| ON: ring read (`w=7.5`) | 245,760 |
| **ON total** | **2,629,760** |
| **ratio** | **0.627x** |

**SGLang publishes 0.53x** (`fla/bench_gdn_replayssm_decode.py:58-63`), which is
`(1 + 1/L)/2`. That model omits the flush **re-read** and every ring access. The
honest figure for our shape is **0.61-0.63x**, and it is a bound on the GDN state
term alone, not on a decode step.

### `L = 16` is the optimum for our shape, and it is not a coincidence

The flush saving decays as `2S/L` while the ring read grows as `(L-1)/2 * C`, so
there is an interior optimum:

```
L* = sqrt(4*HV*V*K / (HV*V + a*H_k*K))
```

At the 27B shape: `4*524288 / 8192 = 256`, so **`L* = 16` exactly** — the same
value both upstreams ship as their default. Neighbours are worse and symmetric:
`L=8` and `L=32` both land at 0.658x against 0.627x at `L=16`. Do not sweep `L`
blindly; predict, then confirm the predicted flatness.

### The ring costs capacity

Ring bytes per slot per layer at `L=16`: `d` 262,144 + `k` 131,072 + `g` 2,048 =
**395,264**, against a 2,097,152-byte state: **+18.9% mamba page**, so roughly
19% fewer GDN slots at a fixed budget. vLLM measured about 7% on Nemotron shapes;
GDN is worse because its state is `V*K` while the ring is `L*(V+K)`. On a device
where the KV budget is already the binding constraint, a 19% page growth can cost
more concurrency than the bandwidth saves.

### Three reasons the end-to-end win is smaller again

1. The 0.627x applies to the GDN **state** term. The share of a 27B decode step
   that term occupies is **not established here**. The closest recorded
   attribution is [#362](https://github.com/mudler/vllm.cpp/issues/362) on the
   NVFP4 arm (Marlin MLP 55%, fp8 tower 40%), which is a different arm.
2. SGLang's own end-to-end figure is **~2.3% TPOT** at 128 concurrency
   (sglang#28451), on an MoE model where GDN is diluted. Its kernel-level claim
   is 1.2-1.5x and its own flag help scopes that to **batch >= 64**
   (`server_args.py:1976-1977`).
3. Our motivating gap is at **low** concurrency, where the ring window `w` is
   full but the batch is too small to amortize anything: Qwen3.8-27B c1 median
   TPOT is 220.6 ms and c4 is 239.0 ms against vLLM's 234.3 ms
   (`docs/BENCHMARKS.md:192-205`). A lever whose published win starts at batch 64
   may be inert exactly where our gap is.

## The A/B that would establish the payoff — UNMEASURED

No performance claim may be recorded from anything weaker than this.

- **Same binary, both arms.** One build carrying the lever, selected by
  `VT_GDN_REPLAYSSM` at run time. Never two builds.
- **Default OFF is the denominator**, and it must be the *shipped default path* —
  which at the 27B means the vendored Triton cubin, not the hand kernel. If the
  ON arm runs the hand kernel and the OFF arm runs the cubin, the measurement is
  a kernel swap wearing a bandwidth result (*The two levers*, [#491](https://github.com/mudler/vllm.cpp/issues/491)).
  Report the OFF-hand-kernel leg as a third arm so the confound is visible rather
  than assumed away.
- **Interleaved arms**, alternating ON/OFF, minimum 3 pairs, cold rep discarded.
  Never all-ON-then-all-OFF.
- **Clocks pinned under the GPU lock** — `flock ${GPU_LOCK:-$HOME/gpu.lock}`,
  then `sudo nvidia-smi -lgc 2100`, release after the last leg
  ([benchmarking.md](../benchmarking.md) §"Pin the clocks before measuring").
  Same boot for both arms; SM-clock spread <= 5%, arm medians within 1%, >= 30
  retained busy samples.
- **`failed == 0` asserted on every leg.** Our SSE keepalive has twice deleted
  the slowest requests and flattered us ([#577](https://github.com/mudler/vllm.cpp/issues/577),
  [#931](https://github.com/mudler/vllm.cpp/issues/931)). A leg with any failed
  request is VOID, not a data point.
- **Axes:** output and total token throughput, median and p99 TTFT, TPOT, ITL,
  and peak memory — as values *and* ratios, at c1, c4 and c8. c1 and c4 are the
  cells where our recorded gap lives.
- **Attribution before acceptance.** A decode-only nsys window on both arms, with
  `--cuda-graph-trace=node`, showing the GDN decode kernel time and the state
  traffic actually moving. A throughput delta without a matching kernel-level
  delta is a measurement, not an explanation.
- **Capacity is an axis, not a footnote.** Record the achievable slot count on
  both arms (*The ring costs capacity*). A throughput win bought with 19% fewer slots is not a win at
  fixed memory.

Establish the *Tests to port* correctness gates first. Correctness before throughput, always.

## Tests to port — red first

Every test below must fail for the intended reason before the kernel exists, and
the failure must be captured.

**T1 — the `L=1` identity (the red-first anchor).** With `L=1` the ring is always
empty, `write_pos == L-1` every step, the total decay is 1, and the buffered
kernel reduces *algebraically* to the shipped packed decode
(`fused_recurrent_linear_replayssm.py:45-47`). Drive `N` decode steps through
both paths on identical inputs and require agreement at `atol=2e-6, rtol=1e-5`
on **both** `out` and the final state. RED before: the op does not exist. This is
the anchor because it is the one case where a difference is unambiguously a bug
rather than reassociation.

**T2 — `L`-step equivalence across a flush boundary.** `L in {2, 4, 16}`, run
`2L + 1` decode steps so at least two flushes and one partial window are
exercised, comparing against the shipped path at vLLM's tolerance
(`rtol=1e-4, atol=1e-3`, `test_replayssm_standard_decode_mamba2.py`). Assert the
**state** as well as the output: an output-only assertion cannot see a
checkpoint that was never written back.

**T3 — the flush actually happens.** Count full-state writes over `N` steps and
require exactly `floor(N/L)`. Without this, a kernel that quietly writes every
step passes T1 and T2 while delivering nothing. This is the test that measures
the *capability* rather than the arithmetic.

**T4 — boundary cases**, one `SUBCASE` each, mirroring *Cursor, flush, and the boundaries*: first step after
prefill is a flush over an empty ring; a slot reused after eviction starts at
cursor 0 and does not read a previous tenant's records; a resumed request
re-anchors its origin; a `< 0` slot index is skipped exactly as
`GdnPackedDecode` already models it.

**T5 — the speculative refusal.** `VT_GDN_REPLAYSSM=1` together with a
speculative configuration must throw, with a message naming ReplaySSM and
speculative decode. Mirrors `config/vllm.py:2329`. RED before: nothing refuses.

**T6 — reachability, per `AGENTS.md` "Nothing lands dead".** The `L`-step
equivalence must be driven through a production entry point — the paged engine
gate on the 27B with the lever ON — not by constructing the op by hand. The
fresh reviewer deletes the production call site in a scratch copy and reruns the
focused gate; a gate that stays green without the call site measured a class,
not a capability.

**T7 — ported upstream cases.** `test_replayssm_standard_decode_mamba2.py` and
`test_replayssm_prefill_decode_equivalence_mamba2.py` supply the parameter grid,
tolerances and failure cases; port the GDN-applicable ones with their revision
anchor recorded. SGLang's `test_linear_replayssm_decode.py:83-88,226-231`
supplies the `L=1` tolerance pair. Document any unavoidable harness adaptation.

## Port map

Upstream construct to local construct. vLLM is the mirror source for every
algorithmic row; SGLang supplies only the GDN naming and the ring layout, and it
is marked where it does.

| Upstream | Anchor | Local target | Source |
|---|---|---|---|
| Ring dtypes appended to the state tuple | `mamba_utils.py:84-93` | third/fourth/fifth `vt::DType` entries in the `MambaSpec` at `qwen3_5_common.cpp:84-87` | vLLM |
| Ring shapes appended to the state tuple | `mamba_utils.py:202-221` | third/fourth/fifth shape vectors in the same `MambaSpec` | vLLM |
| `x_cache (H,L,P)` -> `gdn_replay_d (HV,L,V)` | `mamba_utils.py:214` / `memory_pool.py:466-470` | new cache tensor | vLLM shape, SGLang GDN naming |
| `dt_cache (H,L)` fp32 -> `gdn_replay_g (HV,L)` fp32 | `mamba_utils.py:215` / `memory_pool.py:477-486` | new cache tensor | vLLM (fp32 is vLLM's choice and SGLang keeps it) |
| `B_cache (G,L,N)` -> `gdn_replay_k (H_k,L,K)` | `mamba_utils.py:216` / `memory_pool.py:471-475` | new cache tensor | vLLM shape, SGLang group mapping |
| `write_pos_d` / `is_flush_d` on the decode metadata | `mamba_attn.py:80-81` | new fields on `v1::GDNAttentionMetadata` | vLLM |
| `write_pos = decode_steps mod L`, `is_flush` from it | `mamba_attn.py:617-618` | host-side cursor derivation in the GDN metadata build | vLLM |
| Ring origin re-anchored past a preemption | `backend.py:483-486` | per-slot origin carried beside the cursor | vLLM |
| Per-slot cursor shared across layers, `(slots,)` i32 | `memory_pool.py:641-645` | one device vector, not one per layer | SGLang |
| Non-flush route (`y` without materializing the state) | `selective_state_update_replayssm_output_only.py:275-279` | non-flush branch of the new CUDA kernel | vLLM |
| Flush route (reconstruct, persist, then read `y`) | `:358-362` | flush branch of the same kernel | vLLM |
| Precomputed `k_j . q` products | `:22-129` | fold into the single kernel, or a second pass if occupancy demands it | vLLM |
| Launch-config heuristic keyed on hardware | `ops/replayssm_config.py` | our existing per-shape launch selection in `cuda_gdn.cu` | vLLM |
| `use_replayssm` default `False` | `config/cache.py:152` | `VT_GDN_REPLAYSSM`, default OFF | vLLM |
| `replayssm_buffer_len` default 16 | `config/cache.py:148` | `VT_GDN_REPLAYSSM_LEN`, default 16 | vLLM (SGLang agrees) |
| Refusal on speculative decode | `config/vllm.py:2329` | `VT_CHECK` naming ReplaySSM and speculative decode | vLLM |
| `L=1` reduces algebraically to packed decode | `fused_recurrent_linear_replayssm.py:45-47` | test T1 | SGLang |

Deliberately NOT ported: the KDA per-K `g` widening (`memory_pool.py:477-481`),
which SGLang itself records as slower; the `mamba_cache_mode == "align"` prefix
interaction (`config/vllm.py:2324-2326`), because our GDN spec runs
`mamba_cache_mode = "none"` (`qwen3_5_common.cpp:89`).

## Dependencies

| Depends on | State | Why it matters here |
|---|---|---|
| `KERNEL-GDN-PACKED-DECODE` | `DONE` | Supplies the denominator kernel and the slot ABI (`state_idx < 0` skip) this kernel must reproduce |
| Vendored Triton AOT packed-decode cubin | landed, default ON | It is the *shipped* 27B decode path, so it is the A/B denominator, and it is what the new kernel must beat rather than merely match |
| `MambaSpec` carrying more than two tensors | **UNVERIFIED** | The spec type takes parallel shape and dtype vectors (`qwen3_5_common.cpp:84-87`) so it looks general, but no shipped configuration passes more than two. Confirm before designing around it; if it is two-only, widening it is the first work item |
| Paged runner slot management | landed | The ring is per slot and must be invalidated on slot reuse, alongside the existing conv and ssm tensors |
| CUDA-graph decode capture | landed, default ON | R3: the flush decision varies per step, so capture safety must be established or capture refused |
| GPU lease on `dgx.casa` (GB10, sm_121a) | required for any measurement | Nothing in *The A/B* can run without it, under `flock ${GPU_LOCK:-$HOME/gpu.lock}` |

Blocks nothing. No row waits on this one, which is why a negative result here is
cheap and must be recorded rather than retried.

## Work breakdown

Each item is separately reviewable, and the first two can close the row on their
own.

| W | Work | Completion condition |
|---|---|---|
| **W0** | Register-pressure probe (R1). Write the buffered inner loop far enough to compile for the `[BV=32, BK=128]` 27B tile; `cuobjdump` it | A register count against the recorded REG:205 / REG:255+STACK:48 pair. **If it spills, STOP** and record the negative result under `## Outcome` |
| **W1** | Confirm `MambaSpec` accepts five tensors; widen it if not | A test constructing a five-tensor GDN spec and a runner that allocates it |
| **W2** | Ring in the state spec, cursor and flush flag on the metadata, host-side derivation. No kernel yet | The ring is allocated and the cursor advances; the existing gates stay byte-identical with the lever OFF |
| **W3** | The kernel: non-flush and flush routes, `L=1` identity first | T1 green, then T2 |
| **W4** | Wiring, selector, refusals (speculative, and graph capture per R3) | T3, T4, T5 green; T6 reachability through the paged-engine gate |
| **W5** | Inertness gates, then the A/B | Gate 3 first; then *The A/B* with `failed == 0` on every leg |

W0 before everything. It is a few hours and it can end the row, which is the
cheapest possible ordering.

## Risks

**R1 (HIGH) — register pressure kills the kernel.** *The constraint that shapes the whole design*: the same
register-resident fp32 state tile is `REG:205`/no-spill under Triton and
`REG:255 + STACK:48` as hand CUDA, measured in this tree, and the hand kernel
built on it lost `c16` 700 against 794 tok/s (`cuda_gdn.cu:2710`). A ReplaySSM kernel holds
strictly more live state. *Mitigation:* measure the register count with
`cuobjdump` **before** any A/B, and stop if it spills. A spilling kernel cannot
win a bandwidth argument.

**R2 (HIGH) — leaving the vendored-cubin path is itself a regression.** The 27B
default is vLLM's exact token-identical FLA kernel. Any ReplaySSM arm is a
different kernel, so it forfeits both the parity argument and the measured
`+5.48 tok/s` that path already carries. *Mitigation:* treat the cubin arm as the
denominator (*The A/B*) and accept only a win against it.

**R3 (MEDIUM) — CUDA-graph capture bakes the flush decision.** Our decode graph
captures the GDN call. `is_flush` varies per step by construction, so a captured
graph that baked one branch would be silently wrong, and the cursor advance must
sit outside capture. SGLang has a latent hazard here already. *Mitigation:*
refuse graph capture with the lever ON until a capture-safe design is proven, and
pin the refusal with a test. A refusal is honest; a wrong graph is not.

**R4 (MEDIUM) — 19% fewer slots.** See *The ring costs capacity*. *Mitigation:* capacity is a required
axis in *The A/B*, and `L` is tunable.

**R5 (MEDIUM) — the numeric change is invisible to a token gate.** See *Correctness*.
*Mitigation:* T1-T3 assert tensors and write-counts, not tokens.

**R6 (LOW) — SGLang's credit line points at a vLLM file that does not exist.**
See *Upstream chain (SGLang)*. *Mitigation:* the algorithm is mirrored from vLLM's Mamba2 kernel, which is
real and pinned; SGLang is cited only for the GDN application.

## Gates

1. T1-T7 green on CPU and CUDA; the red-before capture for each recorded.
2. `compute-sanitizer` clean on the CUDA arm.
3. Lever OFF is **byte-identical** to today: the 27B and 35B paged-engine gates
   pass at their recorded counts, and the `=0` rollback arm passes identically.
   This is the inertness gate and it comes before any speed question.
4. A `cuobjdump` register report for the new kernel (R1).
5. The *The A/B* A/B, with `failed == 0` on every leg and clock state recorded.
6. Fresh scoped review including the T6 call-site mutation.

Gate 3 is the one that must never be traded. A default-OFF lever that perturbs
the default path has already failed.

## Evidence

**Evidence that exists today: source only.** Every anchor in *Upstream chain (vLLM)*, *Upstream chain (SGLang)* and *Our baseline* was
read at the two pinned revisions named in the header, and each was re-verified
with `sed -n` against the file before this spec was committed. That is the whole
evidence base. It establishes what upstream does and what this tree lacks. It
establishes nothing about speed.

Two structural claims carry their own commands, so a reader can re-run them
rather than trust this document:

| Claim | Command |
|---|---|
| This tree has no ReplaySSM | `grep -rniE 'replayssm' src include tests` (empty) |
| vLLM has no GDN ReplaySSM, at the pin or 877 commits past it | `git -C /home/mudler/_git/vllm ls-tree -r --name-only origin/main \| grep -i replayssm` (Mamba2 files only) |

**Evidence that does not exist and must not be implied:** any timing, any
kernel-level attribution, any register count for a kernel that has not been
written, and any statement about the GDN share of a 27B decode step. The numbers
in *Traffic model* are *derivations from shapes*, explicitly labelled as such; they are
arithmetic over `HV=32, V=128, K=128, H_k=16`, not observations. A derivation is
a prediction to be tested, and *The A/B* is the test.

**Where evidence will land.** The *Tests to port* red-before captures and gate results go in
this spec under `## Outcome` when the row reaches `DONE`, per `AGENTS.md`. An
accepted or explicitly void measurement additionally goes to
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md), and a lifecycle move goes to
[`docs/STATUS.md`](../../docs/STATUS.md) and this spec's `## Now`. Raw A/B output,
clock manifests and nsys windows stay under an evidence directory named in
`## Outcome`, never summarized away.

## Stop conditions

Stop, record the negative result in this spec's `## Outcome`, and do not land the
kernel, if any of these hold:

- The kernel spills registers (R1). A spilling kernel is a closed door, and
  *The constraint that shapes the whole design*'s measurement says which door.
- Lever OFF is not byte-identical (gate 3).
- The A/B shows no win at c1 or c4 on an idle, clock-pinned host — the cells
  where our gap actually is (*Three reasons the end-to-end win is smaller again*).
- The win exists but is smaller than the capacity loss at fixed memory (*The ring costs capacity*).
- Capture safety cannot be established and refusing capture would disable the
  decode graph on the default path (R3).

A negative result here is a result. Record it; do not retry it silently.

## Owed

- **The Mamba2 arm**, which is the straight vLLM mirror and is *not* this row.
  vLLM's ReplaySSM exists for `NemotronHForCausalLM`
  (`models/nemotron_h.py:711,755,792`), and Nemotron-H is an active port here.
  That work belongs to `KERNEL-SSM-MAMBA` / the Nemotron-H rows, whose SSD core
  is still unported ([#496](https://github.com/mudler/vllm.cpp/issues/496)). No
  separate issue is filed, because #496 already owns the Mamba2 kernel surface
  and filing a second would say one thing twice.
- **The KDA arm.** SGLang implements it in the same kernel and records it as
  SLOWER than the packed baseline (`server_args.py:1976-1981`). Deliberately out
  of scope; `KERNEL-KDA-DELTA` owns KDA.
- **`.agents/sglang-matrix.md` carries no row for `--enable-linear-replayssm`.**
  Its 44 rows predate the sweep that found this lever, and `SGLANG-CUSTOM-KERNELS`
  classifies kernel parity as OUT-OF-SCOPE, so nothing there would have caught
  it. The sweep itself is unrecorded. Owned by
  [#1171](https://github.com/mudler/vllm.cpp/issues/1171): the matrix row lands
  with the implementation, not with this spec, because a spec-only branch should
  not take a lock on a second shared record surface.
- **The GDN share of the 27B decode step is unmeasured** (*Three reasons the end-to-end win is smaller again*). Until it is, the
  0.627x state-traffic ratio bounds nothing end to end. Owned by #1171.
- **Whether a pin advance changes the mirror source.** SGLang cites a vLLM
  `layers/fla/ops/fused_recurrent_replayssm.py` at commit `3c85112` that resolves
  nowhere in the pinned clone (*Upstream chain (SGLang)*). Re-check at the next pin advance; if vLLM
  lands a GDN ReplaySSM, this row reconciles onto vLLM and SGLang stops being
  cited, per `AGENTS.md`.

## Now

Row `KERNEL-GDN-REPLAYSSM` is **`READY`**. This change commits the spec and the
issue only. No product code, no kernel, no measurement.

The gap is verified: `grep -rniE 'replayssm' src include tests` is empty, no open
issue or pull request covers it, and the two nearby unmerged branches
`row/PERF-GDN-BF16-CHAIN` and `row/PERF-27B-BF16-FP8-OUT` were checked
individually and carry no ReplaySSM.

Next action, for a fresh implementer: **do not start with the kernel.** Start
with R1, because it can close the row on its own. Write the buffered kernel far
enough to compile for the `[BV=32, BK=128]` 27B tile, run `cuobjdump`, and read
the register count against the recorded `REG:205` (Triton) and `REG:255 +
STACK:48` (hand CUDA) at `src/vt/cuda/gdn_packed_decode_triton.h:9-14`. If it
spills, stop and record the negative result under `## Outcome`. Only if it does
not spill is T1 (*Tests to port*) worth writing.
