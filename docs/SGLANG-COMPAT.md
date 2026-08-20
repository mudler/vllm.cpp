# SGLang-compatible behaviors

vllm.cpp mirrors vLLM by default, and additionally exposes a small set of
**SGLang-inspired runtime knobs** as first-class, documented toggles. Each one is
**opt-in and defaults to today's behavior**, a caller (library, C ABI, or
server) that sets none of them gets an engine byte-identical to one built without
them.

Four behaviors are covered here. For each: what it does, how to enable it three
ways, (a) the C++ library API (`vllm::entrypoints::EngineParams`), (b) the C ABI
(`include/vllm.h`, `vllm_model_params`), (c) the OpenAI server CLI, and the
honest caveat where one applies. The behavior parity itself is tracked in
[`.agents/sglang-matrix.md`](../.agents/sglang-matrix.md); the ABI-field table and
grounding live in [`.agents/specs/sglang-enablement.md`](../.agents/specs/sglang-enablement.md).

| Behavior | Knob | Default | Output effect |
|---|---|---|---|
| RadixAttention / prefix caching | `enable_prefix_caching` (tri-state, ABI v7) | model default | none (cache reuse) |
| LPM cache-aware scheduling | `scheduling_policy = "lpm"` (string, ABI v9) | `fcfs` | none (admission order) |
| Jump-forward decoding | `enable_jump_forward` (tri-state, ABI v10) | off | none (token-unique subset) |
| Custom logits processors | per-request sampling callback (ABI v8) | none | user-defined |

---

## 1. RadixAttention (automatic prefix caching)

**What it does.** SGLang's RadixAttention keeps a radix tree of KV prefixes and
serves the longest cached prefix of each new request instead of recomputing it.
In vllm.cpp this is **fused into our block-hash automatic prefix caching (APC)**,
there is no separate radix code path; `--enable-radix-attention` is a documented
alias for the APC toggle (see
[`.agents/specs/sglang-radixattention.md`](../.agents/specs/sglang-radixattention.md)
§1). It is output-neutral: it only reuses already-computed KV.

**Enable it.**

(a) C++ library API, `EngineParams::enable_prefix_caching` (tri-state
`std::optional<bool>`):

```cpp
vllm::entrypoints::EngineParams ep;
ep.enable_prefix_caching = true;   // force ON; std::nullopt => model default
auto engine = vllm::entrypoints::LoadedEngine::FromModelDir(model_dir, ep);
```

(b) C ABI, `vllm_model_params.enable_prefix_caching` (ABI v7), tri-state
`0`=model default, `1`=on, `2`=off:

```c
vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/models/Qwen3.6-27B";
mp.enable_prefix_caching = 1;      /* 0 => model default (byte-identical) */
vllm_engine *engine = NULL;
vllm_engine_load(&mp, &engine);
```

(c) Server CLI, vLLM's flag, with the SGLang alias:

```bash
server --model /models/Qwen3.6-27B --enable-prefix-caching
# equivalently: server --model /models/Qwen3.6-27B --enable-radix-attention
# disable: --no-enable-prefix-caching  (alias --disable-radix-attention)
```

**Caveat.** SGLang's SW2 throughput lever (avoiding a redundant same-step prefill
by de-prioritizing the second in-batch collider) is **NOT-APPLICABLE** to our APC:
we cache blocks at allocation time, so the second same-step collider already hits
the first's just-cached prefix, the redundant prefill never occurs here. The
admission-order half of that behavior is folded into LPM scheduling below.

---

## 2. LPM cache-aware scheduling

**What it does.** SGLang's `--schedule-policy=lpm` reorders the waiting queue by
each request's **longest matched cached prefix**, so requests that will hit the
cache are admitted first (maximizing cache reuse under load). vllm.cpp implements
this as the `lpm` scheduling policy over our block-hash APC index. It is
**output-neutral**, it changes admission *order* only; every request computes
identical tokens, and is meaningful **only with prefix caching ON**. With APC
off it degrades to `fcfs`.

**Enable it.**

(a) C++ library API, `EngineParams::policy`:

```cpp
vllm::entrypoints::EngineParams ep;
ep.policy = vllm::SchedulerPolicy::kLPM;   // kFCFS (default) / kPriority / kLPM
ep.enable_prefix_caching = true;           // LPM needs a cache to match against
```

(b) C ABI, `vllm_model_params.scheduling_policy` (ABI v9). A string naming the
policy: `"fcfs"` (default), `"priority"`, or `"lpm"`; `NULL`/`""` => `"fcfs"`.
(There is no separate int scheduler-policy field, the string is the one knob.)

```c
vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/models/Qwen3.6-27B";
mp.scheduling_policy = "lpm";       /* NULL/"" => "fcfs" (byte-identical) */
mp.enable_prefix_caching = 1;        /* recommended: lpm degrades to fcfs w/o APC */
vllm_engine *engine = NULL;
vllm_engine_load(&mp, &engine);
```

(c) Server CLI, vLLM's flag name (`--scheduling-policy`) and SGLang's alias
(`--schedule-policy`), both taking `fcfs|priority|lpm`:

```bash
server --model /models/Qwen3.6-27B --enable-prefix-caching --scheduling-policy lpm
```

**Caveat.** LPM has no effect with prefix caching disabled (the server logs a
warning and falls back to `fcfs` admission order). SGLang itself later **removed
its jump-forward scheduler wiring upstream**, but its LPM scheduling remains; ours
is a scoped opt-in behavior flag, not a blind mirror (vLLM has no `lpm` policy).

---

## 3. Jump-forward decoding

**What it does.** When a structured-output grammar has a **deterministic forced
continuation**, jump-forward emits it *without* running the model per token, the
speed win on constrained decoding. vllm.cpp lands the **provably byte-identical
token-unique subset**: it jumps only while the grammar admits exactly one valid
token at a non-accepting state, so the emitted token is the argmax under any
sampling params (the constrained sampler masks every other token to `-inf`). The
general byte-forced-but-multi-tokenizable span (which needs SGLang's re-tokenize +
boundary rollback) is deliberately not jumped.

**Enable it.**

(a) C++ library API, `EngineParams::enable_jump_forward` (tri-state
`std::optional<bool>`):

```cpp
vllm::entrypoints::EngineParams ep;
ep.enable_jump_forward = true;   // nullopt (default) => OFF unless env override
```

(b) C ABI, `vllm_model_params.enable_jump_forward` (ABI v10), tri-state
`0`=default (env-resolved, off), `1`=on, `2`=off:

```c
vllm_model_params mp = vllm_model_params_default();
mp.model_path = "/models/Qwen3.6-27B";
mp.enable_jump_forward = 1;         /* 0 => default OFF (byte-identical) */
vllm_engine *engine = NULL;
vllm_engine_load(&mp, &engine);
```

(c) Server CLI:

```bash
server --model /models/Qwen3.6-27B --enable-jump-forward   # or --disable-jump-forward
```

**Precedence with the env var.** The legacy `VT_ENABLE_JUMP_FORWARD` env var stays
as an override: **when it is set, it wins** over the C-ABI / C++ / server field
(`1`/`true`/`on` => on, anything else => off), mirroring the `VT_ASYNC_SCHED`
convention. When it is unset, the field decides (default off).

**Caveat.** This is the **token-unique subset only**, the general jump-forward
span is a named residual. The behavior stays **off by default** until the
production scheduler splice (which must recompute KV for the jumped tokens) lands;
SGLang itself removed its own jump-forward scheduler wiring upstream. Enabling it
is output-neutral: the jumped tokens are byte-identical to per-token constrained
decode.

---

## 4. Custom logits processors

**What it does.** A per-request host callback the sampler invokes **once per decode
step, before sampling**, to inspect the tokens generated so far and modify the
logits in place (add a bias, mask tokens, force a token). This satisfies SGLang's
`CustomLogitProcessor` capability (e.g. a thinking-budget processor) and mirrors
vLLM's `SamplingParams.logits_processors`. It runs at vLLM's
non-argmax-invariant logits-processor stage.

**Enable it.**

(a) / (b) C ABI, `vllm_sampling_params.logits_processor` +
`logits_processor_user_data` (ABI v8). The C++ library drives the same sampler
stage; the callback is the delivery surface:

```c
static void my_proc(const int32_t *tokens, int32_t n, float *logits,
                    int32_t vocab, void *user_data) {
  logits[42] = 1e30f;  /* force token 42 (greedy will pick it) */
}
vllm_sampling_params sp = vllm_sampling_params_default();
sp.logits_processor = my_proc;               /* NULL => no processor (default) */
sp.logits_processor_user_data = NULL;
vllm_complete(engine, "hi", &sp, &out);
```

(c) Server CLI, not a CLI flag: it is a per-request programmatic hook (a function
pointer), so it is exposed through the C ABI / library, not the server flags.

**Caveat.** This is a single C callback per request (a function pointer +
`user_data`), not a batched plugin graph, and there is no Python-side / dill
registration. It is otherwise the same capability as vLLM's and SGLang's
per-request logit processor.

---

## When to enable (guidance)

These knobs are **correctness-neutral plumbing, not free speed switches**, whether
they help is workload-dependent, and the shared-prefix performance arm that would
put a number on the gain is **not yet measured** (tracked as
`BACKEND-GATE-CUDA-SGLANG-PREFIX`; the first SGLang competitor-floor run was
cache-neutral). Honest per-knob guidance:

- **RadixAttention / prefix caching**, enable when the workload has **shared
  prefixes** (a common system prompt, few-shot exemplars, multi-turn chat). It
  reuses cached prefix KV instead of recomputing, a real win there; neutral to
  slight overhead when there is no sharing. Already on by default for dense models.
- **LPM scheduling**, enable **together with prefix caching** when concurrent
  requests share prefixes: it admits cache-hitting requests first, raising the hit
  rate under load. It has no effect without APC (falls back to `fcfs`), and it
  changes admission order, so leave it off for latency-sensitive single-stream use.
- **Jump-forward decoding**, a niche constrained-decoding speed lever, off by
  default and conservative (token-unique subset only). Enable only for structured
  output that matches that pattern; it is not a general throughput knob.
- **Custom logits processors**, a programmatic per-request hook, not a performance
  knob; use it when you need per-request logit control (a bias, a mask, a forced
  token, a thinking budget).

**Bottom line:** turn on RadixAttention (and LPM) for **prefix-heavy serving**,
where the reuse is real; the rest are situational. Do not enable everything blindly
for speed. Measured numbers for the prefix-cache-on gain are pending the
shared-prefix benchmark.

## Default inertness

Setting **none** of these knobs (all C-ABI fields `0`/`NULL`, no server flags, no
`VT_ENABLE_JUMP_FORWARD`) yields an engine byte-identical to one built before
these knobs existed. This is enforced by the CPU exact-gate suites
(`tests/capi/test_capi.cpp`, `tests/vllm/v1/test_scheduler_lpm.cpp`,
`tests/vllm/v1/structured_output/test_jump_forward.cpp`).
