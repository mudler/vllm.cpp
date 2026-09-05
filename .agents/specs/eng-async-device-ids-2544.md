# ENG-ASYNC-DEVICE-IDS-2544 — the HOST arm of `device_token_ids`

Row: `ENG-ASYNC-DEVICE-IDS-2544`
Issue: [#2544](https://github.com/mudler/vllm.cpp/issues/2544)
Precedents: [#1305](https://github.com/mudler/vllm.cpp/issues/1305),
[#2496](https://github.com/mudler/vllm.cpp/issues/2496),
[#2596](https://github.com/mudler/vllm.cpp/issues/2596)

## Now

ACTIVE. The three host-gather registrations `#2544` named are wired and gated;
`KimiK3ForConditionalGeneration` is closed as NOT AFFECTED with the tree's own
evidence.

## Scope

`ModelForwardInput::device_token_ids` states its own contract
(`include/vllm/model_executor/models/model_registry.h:634-644`): when non-null,
the `[token_ids.size()]` input ids already live in THAT device buffer and
`token_ids` is stale for decode rows. The asynchronous runner's combine splices
each decode row's sampled token into the device buffer on the main queue and
deliberately never writes it back (`src/vllm/v1/worker/gpu/runner.cpp:2748`, the
mirror arm), because materialising it on the host is the synchronise that path
exists to remove. `token_ids_cpu` is zero-initialised, so a forward that ignores
the field embeds **token id 0** at every decode step, at `rc=0`, with
plausible-looking output.

`async_device_mirror()` is DEFAULT-ON on CUDA, integrated parts included, so
this is the default arm and not an opt-in.

IN SCOPE — the four translation units #2544 named that still do not consume the
field:

| translation unit | disposition |
|---|---|
| `glm5_next_registry.cpp` | CONVICTED on hardware; wired |
| `deepseek_v4_registry.cpp` | candidate; wired, gated on the fixture |
| `laguna_registry.cpp` | candidate; wired, NOT gated -- see D5 |
| `kimi_k3_registry.cpp` | NOT AFFECTED; see `## Risks/decisions` D3 |

OUT OF SCOPE — making the contract ENFORCEABLE (the "better" option #2544's own
prose argues for). Listed under `## Owed` O1, because a runtime refusal for a
forward that ignores a live `device_token_ids` touches every registered model in
the tree and is a separate unit of work.

## Upstream anchors

There is no vLLM anchor for this field. `device_token_ids` is this project's own
seam for a scheduling shape vLLM expresses with a torch tensor that is always
device-resident: upstream's `input_ids` never has a stale host twin, so upstream
has no equivalent defect and therefore no equivalent guard. The behaviour being
mirrored is the OUTPUT — the token vLLM would have generated from the sampled
id — and that is what the hardware legs below compare.

## Design

### Why the existing seam could not be reused as-is

`d8683402b` repaired `glm_moe_dsa` with the two-part seam in
`src/vllm/model_executor/models/qwen3_5_internal.h`:

* PUBLISHER `detail::DeviceTokenIdsScope`, RAII, at the registry forward;
* CONSUMER `detail::ApplyDeviceTokenIds(backend, queue, dst, count, what)`,
  which splices the published DEVICE pointer over a DEVICE buffer the embed has
  already filled from the host.

That consumer takes `void* dst` — a device allocation. It is the right shape for
`glm_moe_dsa`, `qwen3_5`, `qwen3`, `qwen3_moe` and `deepseek_v2`, each of which
uploads `DBuf dids(d, DType::kI32, {T}, token_ids.data())` and embeds from it.

**None of the three registrations in scope has such a buffer.** All three read
the identifiers as a host `std::vector<int32_t>` and gather the embedding rows on
the host:

* `glm5_next_registry.cpp:213` → `glm5_next::Glm5NextHostForward(w, input.token_ids, ...)`
* `deepseek_v4_registry.cpp:164,170,193,198` → four arms, each taking `input.token_ids`
* `laguna_registry.cpp:94,100` → two arms, each taking `input.token_ids`

Calling `ApplyDeviceTokenIds` on a host vector's `data()` would be a device→host
copy through an interface documented as device→device, and — because it is
enqueued and not awaited — would race the host gather that reads it. So the seam
CANNOT represent this architecture's behaviour, which is the exact condition
CLAUDE.md §"Shared seams" gives for extending one.

### The extension

`include/vllm/model_executor/models/host_token_ids.h` +
`src/vllm/model_executor/models/host_token_ids.cpp`:

```cpp
const std::vector<int32_t>& ResolveHostTokenIds(const ModelForwardInput& input,
                                                std::vector<int32_t>* storage,
                                                const char* what);
```

* `input.device_token_ids == nullptr` → returns `input.token_ids` unchanged and
  touches nothing. That is every non-async-CUDA path, so those builds are
  byte-identical by construction.
* otherwise → copies `input.token_ids` into `*storage`, overwrites its first
  `token_ids.size()` entries from the device buffer with one
  `Backend::Copy` on `input.queue`, **synchronises that queue**, and returns
  `*storage`.

The copy goes on `input.queue` — the main queue — so it is ordered AFTER the
combine that produced the source rather than racing it. That is the same
ordering argument `ApplyDeviceTokenIds` makes, and it is why a host read of
`ModelForwardInput::token_ids` cannot substitute for it.

**The synchronise is real and it is named.** These three forwards are host
gathers: they already dereference host memory before any device work, so they
have no asynchrony left to protect. Removing the synchronise on a path that later
gains a device embed would be a use-before-write, which is why it is here and not
deferred to the caller.

### Where it lives, and the naming debt

It goes in `include/vllm/model_executor/models/`, beside `step_token_ids.h`
(which is the DECODE-GRAPH SLOT arm of the same contract), and not in
`src/.../qwen3_5_internal.h`.

`qwen3_5_internal.h` is a model-private header that now publishes a
process-level seam to six unrelated architectures (`qwen3`, `qwen3_moe`,
`qwen3_5`, `deepseek_v2`, `glm4_moe_lite`, `glm_moe_dsa`, plus `llama`,
`mistral`, `internlm2` and `qwen3_dense` for the publisher alone). A model that
has nothing to do with Qwen3.5 including `qwen3_5_internal.h` to reach a
runner contract is a home that has outgrown its name. Moving the DEVICE arm to
join the host arm is `## Owed` O2: it is a pure rename-and-move across ten
translation units, it would conflict with every open branch that touches a
registry, and it is not this row's unit of work.

### The divergence from `glm_moe_dsa`, answered on the three points it is owed

`glm_moe_dsa` (`d8683402b`) publishes the device pointer with
`detail::DeviceTokenIdsScope` and splices DEVICE-side with
`detail::ApplyDeviceTokenIds`, enqueued on the main queue so it is ordered after
the combine rather than racing it. This row reads the ids back to the HOST
instead. That is a divergence and it needs justifying, because a device-to-host
read on the decode path is exactly the synchronise the async mirror exists to
remove.

**1. Where each forward gathers, and where the read-back happens.** Every one of
the three gathers embedding rows on the HOST, out of a host weight vector,
indexing a host `std::vector<int32_t>`:

| registration | the gather | what it indexes |
|---|---|---|
| `Glm5NextForConditionalGeneration` | `glm5_next_forward.cpp:366` `const int32_t tok = token_ids[t];` | a host row read out of `weights.embed_tokens` |
| `DeepseekV4ForCausalLM` | `deepseek_v4.cpp:2929` (in `ForwardComposeImpl`, `:2891`) `const int64_t tok = token_ids[t]; ... x[t*H+h] = hw.embed[tok*H+h];` | `hw.embed`, a host `std::vector<float>` |
| `LagunaForCausalLM` | `laguna.cpp:1457` `const int64_t tok = token_ids[t];` then `memcpy(hidden, embed.data() + tok*H, ...)` | `ReadF32(weights.embed)`, a host vector |

`glm_moe_dsa` does not look like this. It uploads
`DBuf dids(d, DType::kI32, {T}, token_ids.data())` and embeds from that DEVICE
buffer, which is why `ApplyDeviceTokenIds` — whose `dst` is a device allocation —
is the right consumer there and cannot be the right consumer here.

The read-back happens ONCE per forward, at the registry, before the first read of
the identifiers and before any cache work.

**2. What it costs per decode step, named rather than waved past.** One
`vt::Backend::Copy` of `4 * token_ids.size()` bytes device-to-host (4 bytes per
row on a pure-decode step), plus one `Backend::Synchronize` on `input.queue`.

**The synchronise is real, and on these three forwards it is not a new
serialisation point.** Each of them already computes its logits into a host
`std::vector<float>` and returns it by value — `HostLogits(...)` for `glm5_next`
and `laguna`, and for `deepseek_v4`'s so-called device arm
`WrapV4DeviceLogits` (`deepseek_v4.cpp:3946`) merely WRAPS a host
`std::vector<float>` in a `vt::Tensor` pointing at `buf->data()`. A host vector of
logits cannot exist before every device operation that produced it has completed,
and each forward's FIRST act on the identifiers is a host dereference. So there is
no device execution left for this wait to overlap with; the wait these forwards
impose is the one they already imposed.

It WOULD be a genuine regression on a forward whose embed and tower are on the
device, and that is stated in the header rather than left to be discovered: a
caller that later grows a device embed must move to `detail::ApplyDeviceTokenIds`
and stop using this. For `deepseek_v4` in particular, the long-term repair is to
move `ForwardComposeImpl`'s embed onto the device, at which point this call is
replaced rather than kept. That is `## Owed` O7.

The comparison that matters is not "slower against faster". It is "correct
against silently wrong": the arm this replaces reads identifiers the runner never
wrote and generates from token id 0.

**3. Why the shared seam could not carry it, and where the seam should live.**
`detail::ApplyDeviceTokenIds(backend, queue, void* dst, dst_count, what)` splices
over a DEVICE destination. None of these three has one. Passing a host vector's
`data()` as `dst` would be a device-to-host copy through an interface documented
as device-to-device, AND — because that copy is enqueued and never awaited — it
would race the host gather three lines later. So the seam as written cannot
express these callers, which is the condition CLAUDE.md §"Shared seams" gives for
extending a seam rather than routing through it.

There are now three arms of ONE contract, not two competing spellings:

| arm | destination | entry point | who uses it |
|---|---|---|---|
| eager DEVICE | a device embed buffer | `detail::ApplyDeviceTokenIds` | `qwen3`, `qwen3_5`, `qwen3_moe`, `deepseek_v2`, `glm_moe_dsa` |
| decode-GRAPH slot | a slot's persistent device destination | `vllm::StepTokenIds` | the graph drivers |
| HOST gather | the caller's own `std::vector<int32_t>` | `vllm::ResolveHostTokenIds` | `glm5_next`, `deepseek_v4`, `laguna` |

The selector is not a preference: it is where that forward's embed reads from.

**And the seam's home is wrong, which is a finding this row reports rather than
fixes.** The device arm lives in `src/vllm/model_executor/models/qwen3_5_internal.h`,
a MODEL-PRIVATE header that now publishes a runner contract to ten unrelated
translation units — `llama`, `mistral`, `internlm2`, `glm4_moe_lite`,
`deepseek_v2`, `glm_moe_dsa` and four Qwen files. A model that has nothing to do
with Qwen3.5 including `qwen3_5_internal.h` to reach a runner contract is a home
that has outgrown its name. The host arm therefore goes to
`include/vllm/model_executor/models/host_token_ids.h`, beside `step_token_ids.h`,
which is the graph arm of the same contract and already lives there. Moving the
DEVICE arm to join them is a pure rename across ten translation units that would
conflict with every open branch touching a registry, so it is `## Owed` O2 and not
this row's unit of work.

**4. The null path, which is every non-async and every CPU run.**
`if (input.device_token_ids == nullptr) return input.token_ids;` is the first
statement after the argument check. It returns the caller's own vector by
`const&` and never touches `storage`, so the forward receives the identical
object it received before this change — byte-identical by construction, not by
comparison.

The lifetime is safe by the struct's own declaration:
`ModelForwardInput::token_ids` is itself a `const std::vector<int32_t>&`
(`model_registry.h:571`) bound to a vector the caller owns for the duration of
the forward, so the returned reference cannot outlive its referent any more than
`input.token_ids` could.

It is EXERCISED rather than assumed: every other case in both suites — 32 in
`test_glm5_next_forward` and 22 in `test_deepseek_v4_exl3_loader` — passes a step
with no mirror, so the null path runs in all of them and all of them stayed green.
Mutation M4, which makes the body return `input.token_ids` unconditionally, reds
BOTH device-id gates while leaving every one of those cases green, which is the
two-sided proof that the null path and the mirror path are distinct and that each
is reached.

### The three call sites

Each is one `std::vector<int32_t>` of storage plus one call, placed immediately
before the first read of the identifiers, and every downstream argument
switched from `input.token_ids` to the returned reference. No other behaviour
changes.

## Risks/decisions

**D1 — a download is a synchronise, and the async path exists to remove one.**
Accepted, and bounded: it happens only when the mirror is live, only once per
forward, and only for forwards that are host gathers and therefore already
synchronous end to end. It buys correct tokens on a path that is currently
silently wrong, and the alternative on offer is not "faster" but "wrong".

**D2 — `storage` is the caller's, not a thread-local.** A returned reference into
a function-local static would alias across nested forwards (the multimodal
generate helper reaches a second embed inside one call). The caller owning the
buffer makes the lifetime exactly the forward's and is what the
`const std::vector<int32_t>&` return can safely refer to.

**D3 — `kimi_k3_registry.cpp` is NOT wired, and that is the finding rather than
an omission.** Both `KimiK3Model::Forward` and `KimiK3Model::ForwardDevice`
(`src/vllm/model_executor/models/kimi_k3.cpp:54-84`) are
`VT_CHECK(false, kPending)` skeletons that `(void)token_ids` and refuse by name.
There is no embed to feed, so a `ResolveHostTokenIds` call there would be a
device download whose only successor is a throw — dead code, which CLAUDE.md
§"Nothing lands dead" forbids landing silently. #2544 called each of its five
entries "a candidate, not a conviction"; this one is FALSIFIED by the tree, and
the honest disposition is to say so. It becomes live the day the forward does,
and it is listed under `## Owed` O3 against the row that owns that forward.

**D4 — `deepseek_v4` and `laguna` are WIRED BUT NOT CONVICTED.** Neither has a
staged checkpoint or GPU time in this flow. The fixture gate below proves the
identifiers reach the embed through `ModelRegistry::Forward`; it does not prove
either model is ever handed a live `device_token_ids` in service. Recorded as
O4 rather than claimed.

**D5 — `laguna` is WIRED BUT NOT GATED, and the reason is a defect this row
found rather than a gap in the harness.** `ForwardLagunaForCausalLM` routes only
to `LagunaModel::ForwardDevice`, which calls `LagunaModel::Forward`
(`laguna.cpp:1423`), the unit-gated **f32 reference**. That reference reads
`lw.moe.experts_gate` / `experts_up` / `experts_down`. **No loader in this tree
fills them.** `LoadLagunaForCausalLMWeights`
(`laguna_weights.cpp:435-475`) always writes `experts_*_fp4` and sets
`has_nvfp4_weights = true`, and the GGUF arm writes the keep-quant tower; the
real forward for either is `LagunaForwardGguf`, which has **no registry caller
at all**.

Driving the tree's only forward-runnable Laguna checkpoint fixture
(`tests/vllm/models/test_laguna_nvfp4_loader.cpp::BuildFiniteTensors`) through
`ModelRegistry::Load` + `ModelRegistry::Forward` therefore SIGSEGVs inside
`LagunaModel::Forward`, on `origin/main` at `3d045ba1b`, before this row's change
and independent of it:

```
#0  __memcpy_avx512_unaligned_erms ()
#1  vllm::LagunaModel::Forward(...)
#2  vllm::LagunaModel::ForwardDevice(...)
#3  vllm::(anonymous namespace)::ForwardLagunaForCausalLM(...)
#4  vllm::ModelRegistry::Forward(...)
```

That is a separate defect with a separate shape — which arm the registration
should select, what it should refuse, and what gates the selection — and it needs
its own spec and red-first change. Filed as
[#2618](https://github.com/mudler/vllm.cpp/issues/2618) and listed under O6. It is NOT repaired
here, and the `ResolveHostTokenIds` call at `laguna_registry.cpp` is therefore
correct by construction and unproved by test. Said plainly rather than smoothed:
this row cannot gate Laguna until that defect is fixed.

## Tests

One case per gateable registration, each appended to that model's own existing
suite and entering through `ModelRegistry::Forward`, with the three-run A/B/C
recipe `tests/vllm/models/test_moe_async_device_ids.cpp` established for #1305:

```
A  right host ids, no mirror              -> the reference
B  WRONG (zero) host ids, no mirror       -> must DIFFER from A
C  WRONG (zero) host ids, mirror carries A's -> must EQUAL A, bit for bit
```

B is the control. Without it a forward that ignored its identifiers entirely
would satisfy C, and so would a gate whose two runs happened to share a buffer.

Zero is the wrong-host value ON PURPOSE: it is the value the real defect feeds,
because `token_ids_cpu` is zero-initialised.

Every identifier in A is NON-ZERO, so C cannot agree with A for the wrong
reason.

Finiteness is asserted before any comparison. Against a NaN both `==` and `!=`
are false, so an equality gate and a difference gate BOTH report success on a
forward that produced no numbers at all — the failure `MODEL-MM-GLM53-FLASH`
read as a perfect match.

WHAT THE CPU HARNESS CANNOT SHOW, stated the right way round: `vt::Backend::Alloc`
returns host-addressable memory on the CPU backend, so the mirror's buffer and
the host vector are the same kind of pointer. The DEVICE half of the contract —
that the copy reads device memory and that it is ordered on the main queue after
the combine — is not provable there. That half is what the hardware legs are for,
and O5 records it as owed for `deepseek_v4` and `laguna`.

## Gates

Focused: `test_glm5_next_forward` and `test_deepseek_v4_exl3_loader`, which is
where the A/B/C cases live — each beside its own model's fixture, rather than in a
new file that would have to duplicate three of them.

Full: `scripts/agent-preflight.sh`.

Hardware, on `dgx:gpu0`, inside an `rc` lease, one binary, no
`VT_ASYNC_DEVICE_MIRROR` set at all on the default leg:

| leg | model | env | expected |
|---|---|---|---|
| G1 | GLM-5.3-Flash `Glm5NextForConditionalGeneration` UD-Q2_K_XL | (none) | ` Paris.` |
| G2 | same | `VT_ASYNC_DEVICE_MIRROR=0` | ` Paris.` |
| D1 | GLM-5.3 `glm_moe_dsa` UD-IQ1_S | (none) | ` Paris, which is` |
| D2 | same | `VT_ASYNC_DEVICE_MIRROR=0` | ` Paris, which is` |

Prompt `The capital of France is`, `--temperature 0`, `--max-tokens 2` (G) and
`4` (D). Compared as BYTES with `hexdump`, never as rendered text: ` Paris.` and
` Paris Paris` differ late, and that is how this defect hid.

**THE BUILD MUST CARRY THE VENDORED FA2, AND THE JOB MUST ASSERT IT.** On
`sm_121` MLA prefill IS FlashAttention and there is no fallback; both models are
MLA. `CMakeLists.txt:2377` declares `option(VLLM_CPP_FLASH_ATTN ... ON)` but the
`VLLM_CPP_FLASH_ATTN` define is only applied at `:2421`, inside the guard at
`:2381` that also requires `VLLM_CPP_CUTLASS_HEADERS` and a non-empty
`VT_FA2_ARCHS`. A build with the option ON and the CUTLASS headers absent
silently produces a non-FA2 binary that throws inside the first MLA forward and
reads exactly like a model defect. The job therefore greps the configured
`compile_commands.json` for `VLLM_CPP_FLASH_ATTN` and ABORTS if it is absent.
`vt::OpRegistered(kMlaPrefillAttention, kCUDA)` is NOT a proxy for this: that op
is registered unconditionally on CUDA (`src/vt/cuda/cuda_mla_prefill.cu:457`)
and the FA2 refusal is thrown at call time from `:180`.

## Evidence

### Red, then green — BOTH counts on each line

`N failed / 0 assertions failed` means the cases THREW, so both lines are read.

RED is this tree with the three registries reverted to `origin/main`
(`git show origin/main:<path> > <path>`) and the tests unchanged, rebuilt:

| suite | test cases | assertions |
|---|---|---|
| `test_glm5_next_forward` | 32 \| 30 passed \| **2 failed** | 5943 \| 5941 passed \| **2 failed** |
| `test_deepseek_v4_exl3_loader` | 22 \| 21 passed \| **1 failed** | 613 \| 612 passed \| **1 failed** |

The failures are the identifiers and not a throw. `glm5_next`: *102 of 128 logits
differ, so this forward embedded the STALE host ids*. `deepseek_v4`: *32 of 32*.
The `glm5_next` second failure is the shape-disagreement refusal, which the base
tree answers with the model's own "the step carries no tokens" instead.

GREEN, after restoring the three registries and rebuilding:

| suite | test cases | assertions | control moved | mirror vs reference |
|---|---|---|---|---|
| `test_glm5_next_forward` | 32 \| **32 passed** \| 0 failed | 5943 \| **5943 passed** \| 0 failed | 102 floats | **0 of 128** |
| `test_deepseek_v4_exl3_loader` | 22 \| **22 passed** \| 0 failed | 613 \| **613 passed** \| 0 failed | 32 floats | **0 of 32** |

The control still moves in both, so the gate is not passing because the fixture
cannot see an identifier at all.

### Mutations

Each is applied to PRODUCT code, **REBUILT**, run, then restored and verified by
`sha256sum` against a pre-mutation copy. A mutation the compiler rejects and one
the binary never contained both read as a pass, so the rebuild result is recorded
beside every row. All five rebuilt at `rc=0` and all three files restored
byte-identical after every one.

| # | mutation | rebuild | `test_glm5_next_forward` | `test_deepseek_v4_exl3_loader` | what it proves |
|---|---|---|---|---|---|
| M1 | delete the `ResolveHostTokenIds` call in `glm5_next_registry.cpp` | rc=0 | **2 failed** / 2 assertions | 22/22, 613/613 | the `glm5_next` call site is REACHED through `ModelRegistry::Forward` and is load-bearing; and it is not what `deepseek_v4` depends on |
| M2 | delete the call in `deepseek_v4_registry.cpp` (all four arms) | rc=0 | 32/32, 5943/5943 | **1 failed** / 1 assertion | the same for `deepseek_v4`, and the two call sites are independent |
| M3 | delete `backend.Synchronize(input.queue)` from the seam | rc=0 | 32/32 | 22/22 | **UNMOVED, and expected to be.** `vt::Backend::Copy` on the CPU backend is a plain `memcpy` with nothing to await, so this harness structurally cannot see the await. Recorded as a measurement, not read as unreachability. The DEVICE half is `## Owed` O5. |
| M4 | make the seam body always return `input.token_ids`, call sites intact | rc=0 | **2 failed** | **1 failed** | the WORK is in the seam, not in the call sites; a call site that reached a no-op would pass M1 and M2 and fail here |
| M5 | replace the shape-disagreement `VT_CHECK` with a silent no-op return | rc=0 | **1 failed** | 22/22 | the refusal is live and named, rather than a comment |

### Reachability

M1 and M2 are the reachability mutations `.agents/reachability.md` asks for:
each deletes the PRODUCTION call site in the registry forward and each reds a
gate that enters through `ModelRegistry::Forward`. No case in either suite
constructs the model or the input by hand.

`laguna_registry.cpp` has NO such mutation, and that is the finding rather than
an omission — see `## Risks/decisions` D5 and
[#2618](https://github.com/mudler/vllm.cpp/issues/2618).

### Hardware

PENDING. The default-arm legs are staged at `/workspace/devids2544/job.sh` and
queued on `dgx:gpu0`. Nothing is claimed for them until they run.

## Owed

* **O1** — the ENFORCEABLE contract #2544's own prose asks for: a forward that
  ignores a live `device_token_ids` refused BY NAME rather than silently
  generating from the previous step's identifiers. Touches every registered
  model. Owned by [#2544](https://github.com/mudler/vllm.cpp/issues/2544).
* **O2** — move the DEVICE arm of this seam out of
  `src/vllm/model_executor/models/qwen3_5_internal.h` to join the host arm under
  `include/vllm/model_executor/models/`. Ten translation units, pure move.
  Owned by [#2544](https://github.com/mudler/vllm.cpp/issues/2544).
* **O3** — `KimiK3ForConditionalGeneration` becomes affected the day its forward
  stops being a `VT_CHECK(false, ...)` skeleton. Owned by
  [#2544](https://github.com/mudler/vllm.cpp/issues/2544).
* **O4** — `DeepseekV4ForCausalLM` and `LagunaForCausalLM` are wired and gated on
  a fixture, NOT convicted on hardware. Neither has a staged checkpoint in this
  flow. Owned by [#2544](https://github.com/mudler/vllm.cpp/issues/2544).
* **O6** — `ForwardLagunaForCausalLM` routes every step to `LagunaModel::Forward`,
  the f32 reference, whose `moe.experts_*` no loader in the tree fills; the real
  forward `LagunaForwardGguf` has no registry caller. A registry step on the only
  runnable checkpoint fixture SIGSEGVs. Tracked by
  [#2618](https://github.com/mudler/vllm.cpp/issues/2618). See `## Risks/decisions` D5.
* **O7** — move `DeepseekV4ForCausalLM`'s embed (`ForwardComposeImpl`,
  `deepseek_v4.cpp:2929`) onto the device, so its arm takes
  `detail::ApplyDeviceTokenIds` and drops the host read-back entirely. Owned by
  [#2544](https://github.com/mudler/vllm.cpp/issues/2544).
* **O5** — the DEVICE half of the contract (the copy reads device memory; it is
  ordered on the main queue after the combine) is untested for these three on any
  device; only GLM-5.3-Flash gets a hardware leg here. Owned by
  [#2544](https://github.com/mudler/vllm.cpp/issues/2544).

## Stop conditions

* The `rc` lease on `dgx:gpu0` cannot be obtained, or the box reboots mid-leg
  (#545). Report the legs that completed and do not infer the rest.
* The CUTLASS headers cannot be obtained in the container, so no FA2 build is
  possible. That is a reportable blocker; do NOT produce a non-FA2 run and then
  interpret its failure.
* Any registration in scope turns out to consume the field already. Reconcile the
  record and do not implement.
