# Gemma-4 vision tower reachability — wire it, and name what blocks the wire

Issue: [#2173](https://github.com/mudler/vllm.cpp/issues/2173).
Row: `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation`
([model-matrix.md](../model-matrix.md)) owns the wiring. The investigation was
run under a `helper` claim on `ENG-MM-INPUT-PIPELINE`
([engine-matrix.md](../engine-matrix.md)), which owns the shared seam §4.1 says
is missing.
Campaign spec: [`gemma4-multimodal.md`](gemma4-multimodal.md) — §G2-impl landed
the tower, §MM-E2E landed the registry fold; this spec owns the reachability
question those two sections left open.
Sibling spec: [`vision-tower-dtype-polarity.md`](vision-tower-dtype-polarity.md)
§Owed filed #2173 and states the same measurement; §7 below is the only dtype
content here, and it is a note, not a task.

**Why this is a separate file and not a section of `gemma4-multimodal.md`.**
That file is 927 lines and is the shared campaign record every Gemma-4 wave
(G1/G1b/G2/G2-impl/G3/MM-E2E, plus the staged audio track) writes. Adding a
wire-or-remove section with its own gates to it makes it a conflict lock for
work that has nothing to do with this question, which AGENTS.md §Records
forbids. This file is one file for one unit of work, read by glob; the campaign
spec gets a two-line cross-reference and nothing else.

This spec is written BEFORE any implementation. Nothing was built or run for it:
see §9 for exactly what is measured, what is read, and what could not be settled
without a leased box.

---

## 0. The verdict, in one line

**WIRE IT.** vLLM at the parity pin runs a Gemma-4 vision tower on its default
production path, and the precomputed-embeddings shape our code has is upstream's
opt-in `enable_mm_embeds` escape (`vllm/config/multimodal.py:98`, default
`False`, documented "Only enable this flag for trusted users!"). Removing a
faithful port of behaviour the oracle runs by default would be a divergence from
the mirror, not a cleanup, and it would discard the measured per-stage parity
G2-impl established. So the tower stays and the wiring is owed.

**But the wire is not a Gemma-4-local fix, and the blocker is already filed.**
`src/vllm/v1/worker/gpu/runner.cpp` never builds `ModelForwardInput.mm` for
*any* model — that is
[#2300](https://github.com/mudler/vllm.cpp/issues/2300), owned by
`ENG-MM-INPUT-PIPELINE` and listed under `## Owed` in
[`multimodal-track.md`](multimodal-track.md), and this investigation re-derived
it independently at its own base SHA (§3.4) before finding the issue. The three
architectures that can consume the field — Qwen3-VL, Gemma-4, Muse Glimmer —
each reach it only through a `…GenerateGreedyViaRegistry` driver whose sole
caller is its own test.

What this spec adds is the placement: #2173 is not a fourth independent bug
beside [#1358](https://github.com/mudler/vllm.cpp/issues/1358),
[#1566](https://github.com/mudler/vllm.cpp/issues/1566),
[#2257](https://github.com/mudler/vllm.cpp/issues/2257) and #2300; it is one
more *symptom* of the seam #2300 names, and Gemma-4 is the worst-affected of the
three towers because it is also missing the two layers below the seam (§3.5,
§4). A reader who fixes #2300 alone will find Gemma-4 still unreached.

---

## 1. Scope

**In scope.**

- The reachability verdict for `Gemma4VisionForward` / `Gemma4VisionWeights`
  (`include/vllm/model_executor/models/gemma4_vision.h:118,92`) and its
  evidence.
- The upstream production chain at the pin that decides the verdict.
- The design of the wire: what must exist, in what order, and which row owns
  each piece.
- What the four named mm tests actually prove.

**Out of scope, deliberately.**

- Implementation. This is a spec-only change; no product code moves here.
- Gemma-4 **audio**. `gemma4_audio.cpp:26-130` is host-f32 scalar loops all the
  way to the kernel — a genuine memory-for-latency trade, unrelated to the
  vision narrowing, and excluded by
  [`vision-tower-dtype-polarity.md`](vision-tower-dtype-polarity.md) §Owed with
  a reason. It stays excluded. Its own reachability is the same shape and is
  §Owed here.
- The dtype question. §7 records the tie-in in three sentences and owns no work.
- Gemma-4 **video**. Upstream decomposes video into timestamped frames through
  the same tower (`gemma4_mm.py:1375` `_process_video_input`); it inherits this
  spec's answer and needs no separate verdict.

---

## 2. Upstream anchors, at the parity pin

Pin `5559679229bc961848b121ccdeaa8fa5d79bec98`
([`upstream-sync.md`](../upstream-sync.md)), read in a checkout whose `HEAD` is
that exact SHA. Every line below was read at that revision.

### 2.1 The tower is constructed unconditionally on the multimodal arch

| What | `file:line` |
|---|---|
| `self.vision_tower = AutoModel.from_config(config.vision_config)` | `vllm/model_executor/models/gemma4_mm.py:1060` |
| `self.embed_vision = Gemma4MultimodalEmbedder(...)` | `gemma4_mm.py:1061` |
| checkpoint prefixes claimed: `model.vision_tower` → `vision_tower`, `model.embed_vision` → `embed_vision` | `gemma4_mm.py:1016-1017` |
| the tower/connector split the weight mapper publishes | `gemma4_mm.py:2118-2119` |

### 2.2 The production forward chain that reaches it

| Step | `file:line` |
|---|---|
| the GPU runner's encoder pass | `vllm/v1/worker/gpu_model_runner.py:2998` `_execute_mm_encoder` |
| the runner calls the model's encoder hook | `gpu_model_runner.py:3196` `batch_outputs = model.embed_multimodal(**mm_kwargs_batch)` (and `:3165` for the per-frame video micro-batch) |
| Gemma-4's hook | `gemma4_mm.py:1516` `embed_multimodal` |
| image dispatch inside it | `gemma4_mm.py:1523-1526` → `_process_image_input` |
| the tower is taken | `gemma4_mm.py:1258` `_process_image_input`, `:1272` `vt = self.vision_tower` |
| patch embedder / encoder / pooler run | `gemma4_mm.py:1315`, `:1320`, `:1342` |
| the projector runs | `gemma4_mm.py:1358` `flat_proj_embs = self.embed_vision(...)` |
| the runner scatters the result | `gpu_model_runner.py:3220` `_gather_mm_embeddings` |
| the scatter enters the model | `gemma4_mm.py:1948` `embed_input_ids(input_ids, multimodal_embeddings=..., is_multimodal=...)` |

This is the default path. No flag selects it and none disables it.

### 2.3 The one upstream shape that resembles ours, and why it is not a licence

`vllm/config/multimodal.py:98`:

```python
enable_mm_embeds: bool = False
"""If `True`, enables passing multimodal embeddings: … WARNING: The vLLM engine
may crash if incorrect shape of embeddings is passed. Only enable this flag for
trusted users!"""
```

Upstream *does* accept caller-supplied embeddings — behind an off-by-default
flag whose own docstring calls it unsafe, and whose stated purpose is to skip
loading encoder modules when the modality limit is 0. Our
`Gemma4GenerateGreedyViaRegistry(…, const std::vector<float>& mm_projected, …)`
is that escape hatch's shape, permanently on, with no tower behind it and no
flag in front of it. Mirroring upstream means the default path runs the tower;
it does not mean the escape hatch is the port.

**Conclusion.** The verdict is settled by the oracle, not by preference. It is
not a `NEEDS_DECISION`.

---

## 3. What is measured in this tree

All measurements at `e3d628e41` (`origin/main` at investigation time), and
re-run at that SHA after the branch was cut. Line numbers below are from that
tree.

### 3.1 The header has no includer in `src/` or `include/` but its own `.cpp`

```
$ grep -rn 'gemma4_vision\.h' src/ include/ tests/
src/vllm/model_executor/models/gemma4_vision.cpp:16    # its own header
tests/vllm/multimodal/test_gemma4_registry_e2e.cpp:42
tests/vllm/multimodal/test_gemma4_vision_tower.cpp:25
```

CONFIRMED as filed.

### 3.2 The tower's two public symbols have no caller outside tests

`Gemma4VisionForward` is defined at `src/…/gemma4_vision.cpp:183` and declared
at `include/…/gemma4_vision.h:118`. Every *call* is
`tests/vllm/multimodal/test_gemma4_vision_tower.cpp:189` and
`tests/vllm/multimodal/test_gemma4_registry_e2e.cpp:171`.
`Gemma4VisionWeights` is constructed only at `test_gemma4_vision_tower.cpp:139`
and `test_gemma4_registry_e2e.cpp:127`. CONFIRMED as filed.

### 3.3 The consumer of tower output never calls the tower, and has one test caller

`Gemma4GenerateGreedyViaRegistry` (`src/…/gemma4_mm.cpp:165`) takes
`const std::vector<float>& mm_projected` as its third parameter, checks its row
count against the image-token count at `:187-188`, bf16-rounds it at `:251` and
masked-scatters it through `multimodal::Qwen3VLMergeMultimodal` at `:252`. It
never names the tower. Its only caller is
`tests/vllm/multimodal/test_gemma4_registry_e2e.cpp:244`. CONFIRMED as filed;
the merge is at `:252`, not `:247` (`:247` is the section comment above it).

This is `.agents/reachability.md`'s **test-only driver** shape, twice over: the
tower is unreached, and so is the only thing that would consume it.

### 3.4 The seam itself is unreached, for every model — this is #2300

```
$ grep -c 'MultiModal\|mm_features' src/vllm/v1/worker/gpu/runner.cpp
0
```

Derived independently here at `e3d628e41`, and it is
[#2300](https://github.com/mudler/vllm.cpp/issues/2300), filed at `e541be98a`
with the sharper reading: the runner's designated initializer names its fields
and not `.mm` before calling `ModelRegistry::Forward`, and
`include/vllm/v1/worker/gpu/input_batch.h:90` records `mm_features` as DEFERRED
on `CachedRequestState`, so the features the field would be built from never
reach the worker. Two independent derivations at two SHAs agree.

**#2300's two `runner.cpp` line anchors have already gone stale**, which any
wave acting on it must know: at `e541be98a` they were `:2234` (the initializer)
and `:2340` (the `Forward` call); at `e3d628e41` those lines are unrelated code
and the sites are `src/vllm/v1/worker/gpu/runner.cpp:2395`
(`ModelForwardInput forward_input{`) and `:2501`
(`ForwardLogits logits = ModelRegistry::Forward(*model_, forward_input);`).
`input_batch.h:90` and the two registry anchors (`qwen3_vl_registry.cpp:127`,
`gemma4_registry.cpp:145`) still hold. Re-derive before citing; the finding is
unchanged.

One #2300 detail matters for Gemma-4 specifically and is good news: Gemma-4's
registered forward guards with `if (input.mm.has_value())`
(`gemma4_registry.cpp:145`), so a runner step with `mm` unset runs the text path
rather than throwing. Qwen3-VL's `VT_CHECK(input.mm.has_value(), ...)`
(`qwen3_vl_registry.cpp:127`) does throw. So B1 is, for Gemma-4, purely
additive — there is no refusing arm to repair first.

The only `src/` constructions of `MultiModalForwardInput` are the three drivers
— `qwen3_vl.cpp:624`, `gemma4_mm.cpp:224`, `muse_glimmer_mm.cpp:343`, each
assigned into `ModelForwardInput.mm` a few lines later at `:638`, `:240` and
`:358`, which are the lines #2300 cites — and each driver's only caller is its
own registry-e2e test. The seam's *consumers* are
real production code (`gemma4_registry.cpp:145-151`,
`qwen3_vl_registry.cpp:131`, `muse_glimmer_registry.cpp:114`), reachable through
`ModelRegistry::Forward`; the branch that selects them is never taken outside a
test.

The serving side stops short of it independently. `MakeQwen3VLImageChatFn`
(`src/vllm/entrypoints/openai/chat_mm.cpp:301`, installed at
`src/vllm/entrypoints/openai/server_main.cpp:1545`) expands placeholders and
builds `mm_features`; those reach `Request::mm_features`
(`include/vllm/v1/request.h:154`, set at `src/vllm/v1/request.cpp:105`) and are
consumed only by `src/vllm/v1/core/kv_cache_utils.cpp:420-450` as prefix-cache
extra keys. No image pixel ever reaches a forward. `include/vllm.h:288` states
the same thing about the C ABI in its own words: "`vllm_chat` / `vllm_generate`
cannot yet feed the tower an image".

So: **no vision tower in this tree is reached from a production entry point
today.** #2300 is the seam. #1358 is Qwen3-VL's tower instance (its tower is
*loaded* on the production load path, so its weights are reached and its forward
is not), #2257 is the Qwen3.5/3.6 VL drivers having no production caller at all,
#1566 is Muse Glimmer's, #2173 is Gemma-4's.

### 3.5 Gemma-4 is the worst of the three, because two more layers are missing

| Layer | Qwen3-VL | Gemma-4 |
|---|---|---|
| tower weights loaded on the production load path | ✅ `LoadQwen3VLWeights` → `LoadQwen3VLVisionWeights`, tower-skip-gated (`ENG-MM-INPUT-PIPELINE` L3) | ❌ `LoadGemma4ForConditionalGeneration` (`gemma4_registry.cpp:79-95`) loads `Gemma4Weights` only; `gemma4_weights.cpp` claims no `model.vision_tower.*` or `model.embed_vision.*` tensor; not one of L3's three gated loads |
| image processor in `src/` | ✅ `src/vllm/multimodal/qwen3vl_processor.cpp` | ❌ nothing; `src/vllm/multimodal/` has no Gemma-4 entry |
| chat seam body | ✅ `MakeQwen3VLImageChatFn` | ❌ none |
| `ModelForwardInput.mm` built in production | ❌ (§3.4) | ❌ (§3.4) |

`.supports_multimodal = true` on the Gemma-4 registration
(`gemma4_registry.cpp:55`) is a claim with nothing behind it for image input. It
is inert rather than user-visible: `dots3_note_registry.cpp:63-69` records the
measurement that "`supports_multimodal` has NO production reader anywhere in
`src/`, `include/`, `examples/` or `scripts/`". Noted so the next reader does
not mistake it for reach, and so the flip decision that file argues is not
re-litigated here.

### 3.6 Where the soft tokens come from today

`Gemma4GenerateGreedyViaRegistry`'s `mm_projected` argument is filled, at
`test_gemma4_registry_e2e.cpp`, from one of two places, and neither is in `src/`:

1. **Default** — the committed fixture
   `tests/parity/goldens/gemma4_e4b_image/vision_refs/ref_projected.npy`
   (2 621 568 B), the *transformers-eager* Gemma-4 tower's output, dumped by
   `scripts/mm/g2_vision_ref_dump.py` on the dgx.
2. **When `VLLM_GEMMA4_VISION_WEIGHTS` is set** — the live C++ tower, run on the
   equally committed processor-output fixtures
   `vision_refs/proc_pixel_values.npy` (7 741 568 B) and
   `proc_image_position_ids.npy`, which are the *HuggingFace* NaFlex processor's
   output, recorded in `gen_manifest.json` under `processor_outputs` with their
   sha256.

So the answer to "does the tower duplicate work done elsewhere, or is a real
path missing?" is: **a real path is missing, twice.** Nothing in this tree turns
an image into `pixel_values` + `pixel_position_ids`, and nothing in this tree
turns those into soft tokens outside a test. The fixtures are a Python dump, not
a component. That is the crux, and it is why the removal option (§5.3) would
discard the only half of the pipeline we have.

### 3.7 The four tests run 0 assertions off the dgx — CONFIRMED, statically

Each of the four files contains exactly one `TEST_CASE`, and each opens with
early returns guarded by `MESSAGE("SKIP: …")`:

| Test | `TEST_CASE` | first skip |
|---|---|---|
| `tests/vllm/multimodal/test_gemma4_vision_tower.cpp` | `:120` | `:122-125` (`VLLM_GEMMA4_VISION_WEIGHTS` unset), then `:127-130` (no CUDA) |
| `tests/vllm/multimodal/test_gemma4_registry_e2e.cpp` | `:176` | `:178-180` (checkpoint absent), then `:183-185` (no CUDA) |
| `tests/vllm/multimodal/test_qwen3_5_vl_e2e.cpp` | `:101` | `:103-105`, `:108-110` |
| `tests/vllm/multimodal/test_qwen3_5_vl_video_e2e.cpp` | `:126` | `:128-130`, `:133-135`, `:138-140` |

doctest's `MESSAGE` is not an assertion, so on a box with no CUDA and no
checkpoint each binary reports one test case, zero assertions, and **SUCCESS**.
That is the "zero assertions is a skip wearing a pass" shape.

**What it means for this row's evidence.** It does not falsify the G2-impl
numbers — those were measured on the dgx with the env set, and the spec records
220/220 and 239/239 assertions there. What it means is that **CI carries no
signal at all for the Gemma-4 vision path**: a change that broke the tower
outright would leave every lane green. Combined with §3.2, the situation is that
the only gate on this code is one that cannot run in CI, guarding code nothing
in production calls. Any wiring wave must add a gate that runs where CI runs
(§8.1), not only a second dgx-only one.

Establishing that the assertion count is literally 0 on a *run* needs a build;
this spec establishes it from the source instead, which is sufficient for the
conclusion and costs no disk (§9).

### 3.8 ★ The golden this row would close on has contradictory provenance

Found while grounding §4.3, and not part of #2173's filing. It is recorded
because §8.4 — the gate #2173 closes on — is that golden.

`tests/parity/goldens/gemma4_e4b_image/gen_manifest.json` records
`versions = {"vllm": "0.25.0", "transformers": "5.13.1"}` alongside
`ok: true`, `ran_image: true`, `gate_form: "STRICT"` and the 18 coherent tokens.
That version pair is not transcribed: `scripts/mm/g2_gemma4_image_oracle_capture.py:48`
reads it live as `{"vllm": vllm.__version__, "transformers": transformers.__version__}`
in the process that ran the capture.

[`gemma4-multimodal.md`](gemma4-multimodal.md) §0.0 records the opposite about
that exact version, also as a measurement, dated 2026-07-25 on dgx:
`~/venvs/vllm-oracle` at **transformers 5.13.1**, `import
transformers.models.gemma4` **fails with `ModuleNotFoundError`**, therefore
`Gemma4ForConditionalGeneration` cannot be constructed at all.
[`pin-advance.md`](pin-advance.md) §(b) repeats it — "transformers **5.13.1 has
no `gemma4` module**" — and is the argument the pin advance to 5.14.1 rests on.

**Both cannot be true.** A capture that ran the Gemma-4 mm arch to 18 coherent
tokens cannot have run under a transformers that cannot import the module the
arch's towers come from. Either the capture's environment was not the one §0.0
measured (a second venv, or one upgraded between the two dates, in which case
the manifest's live-read string is the surprising part), or §0.0's conclusion is
narrower than it reads. `gemma4-multimodal.md:184` records §0.0 as "DISSOLVED on
the advanced pin", which resolves the *forward* question and leaves this one
open: it does not explain how a 5.13.1 capture succeeded.

**Not resolvable here.** It needs the dgx venvs and, to settle it properly, a
recapture at the current pin. It costs a lease, not a build. Until it is
settled, treat the fixture as provenance-uncertain: it is still the best
available target and is not withdrawn, but no wave should call a match against
it *parity with the pinned oracle*, because the record does not currently say
which oracle produced it.

---

## 4. Design — what the wire is, and who owns each part

Four bricks. They are strictly ordered: each is unreached until the one below it
exists, so landing them out of order re-creates exactly the defect #2173 names.

### 4.1 B1 — the shared runner mm path (owner: `ENG-MM-INPUT-PIPELINE`, issue [#2300](https://github.com/mudler/vllm.cpp/issues/2300))

The blocker, and it already has an issue and an owner. The runner must read
`Request::mm_features`, run the staged encoder output, and populate
`ModelForwardInput.mm`. This is the residual
`gemma4-multimodal.md` §MM-E2E names as "the FULL in-runner BATCHED mm path" and
`engine-matrix.md` names as "the FULL in-runner scheduler-fed tower run"; its
recipe already lives in [`mm-serving.md`](mm-serving.md). Upstream's shape is
`gpu_model_runner.py:2998` `_execute_mm_encoder` + `:3220`
`_gather_mm_embeddings`.

**Not Gemma-4's to build, and not this spec's to design.** It is named here
because #2173 cannot close without it, and because a Gemma-4-local workaround —
promoting `Gemma4GenerateGreedyViaRegistry` into an example or a CLI entry — is
exactly the "parallel path by hand" AGENTS.md §Shared seams forbids. If someone
proposes that as a shortcut, this paragraph is the refusal.

### 4.2 B2 — load the tower on the production load path (owner: this row)

Mirror `LoadQwen3VLWeights` → `LoadQwen3VLVisionWeights`: teach
`LoadGemma4ForConditionalGenerationWeights` to claim `model.vision_tower.*` and
`model.embed_vision.*` (upstream's own prefixes, `gemma4_mm.py:1016-1017`) into
a `Gemma4VisionWeights`, and gate the load with `SkipTowerForModalities`
(`include/vllm/model_executor/models/interfaces.h`) so `--language-model-only`
and a zero image limit skip it as they do for the other three sites. The tower
must **refuse by name** when skipped, per the L3 convention
(`utils.py:693-705`).

B2 alone is reachable from the loader and testable today. It is also, on its
own, precisely #1358's state — weights loaded, forward unreached — so it must
land *saying so*, under the §"Nothing lands dead" staged-slice rule, naming B1
and B3 as the wave that wires it.

### 4.3 B3 — the C++ Gemma-4 NaFlex image processor (owner: this row)

PNG/RGB → pre-patchified `pixel_values [P, 3·16²]` + per-patch `(x,y)`
`pixel_position_ids [P, 2]` with the trailing-contiguous `(-1,-1)` padding block
the tower's header documents. Upstream wraps HuggingFace's processor
(`gemma4_mm.py:558` `Gemma4MultiModalProcessor`, `:578` `_call_hf_processor`,
the `max_soft_tokens` validation at `:586-592`), so the reference implementation
is `transformers` — a listed secondary oracle with a pin
([`../oracles/transformers.md`](../oracles/transformers.md)), used for a
processor vLLM itself mirrors, exactly the row that table authorises.

The M0 precedent applies: gate it BIT/BYTE-identical against a captured oracle
fixture, as `tests/vllm/multimodal/test_qwen3vl_processor.cpp` does. The
fixtures already exist and carry sha256s — `gen_manifest.json`
`processor_outputs.pixel_values.sha256 =
02a3c6f747a5743e57e448c41b9f973fd873ebdc98ea5c6a7ccc67db1c5a8c3b`,
`image_position_ids.sha256 =
934c56f857c80d9b0b401aab88a02ba083e0260b922535150e9847071e3c74bc`, over
`g2_fixed_112.png` whose array sha256 is
`306c792dddc0d376d06afb8176e36e4184631ac676b942431640ef75698fdeda` — so B3 has
a red-first target that needs no new capture and no GPU.

**One caveat B3 must carry, and it is not optional.** These bytes are an oracle
capture at a **superseded** revision, and §3.8 shows their recorded provenance
does not add up. B3's first job is to establish whether the *pinned* processor
still emits them. If it does, say so and the fixture stands; if it does not, the
fixture is recaptured under its own issue BEFORE it gates anything, never inside
the change it gates (R5). Gating against a stale-oracle fixture and calling it
parity is the failure this paragraph exists to prevent.

**B3 is the piece that can be built and gated without B1.** It is a pure host
function with a committed byte-exact oracle fixture, it runs on CPU, and it runs
in CI. If the wiring campaign has to be staged, B3 is the slice to take first:
it is the only one of the four that converts a fixture into a component.

### 4.4 B4 — the chat seam body (owner: this row, after B1–B3)

`MakeGemma4ImageChatFn`, mirroring `MakeQwen3VLImageChatFn`
(`chat_mm.cpp:301`), installed beside it at `server_main.cpp:1545`, with its own
`Gemma4ChatSupportedMmLimits()` feeding `ValidateChatMmLimits`. This is what
makes `/v1/chat/completions` with an `image_url` part a Gemma-4 production
entry point on the default configuration.

### 4.5 The resulting production chain

```
POST /v1/chat/completions (image_url part)
  -> serving_chat.cpp mm branch  (gated on the seam being set)
  -> MakeGemma4ImageChatFn                       [B4]
  -> Gemma4NaFlexImageProcessor                  [B3]   -> pixel_values + position_ids
  -> LoadedModel's Gemma4VisionWeights           [B2]
  -> Gemma4VisionForward                                 -> 256 soft tokens
  -> runner builds ModelForwardInput.mm          [B1]
  -> ModelRegistry::Forward
  -> ForwardGemma4ForConditionalGeneration mm branch (gemma4_registry.cpp:145)
  -> Gemma4Model::ForwardMm
```

Everything from `ModelRegistry::Forward` down already exists and is gated. The
four bricks are the four arrows above it.

---

## 5. The options that were weighed, and why they lost

### 5.1 Wire it — CHOSEN

Mirrors the oracle (§2). Preserves the measured per-stage parity. Cost: B1–B4,
of which only B1 is large and B1 is already owed to another row for two other
models.

### 5.2 Record it as a deliberately unreached scaffold and stop

Rejected. The staged-slice rule permits landing a layer before its wiring; it
requires naming the wave that wires it. That naming has now happened three times
for this tower (G2-impl §RESIDUAL 2026-07-28, MM-E2E §Residuals 2026-07-29,
`vision-tower-dtype-polarity.md` §Owed 2026-08) without a wave. A fourth
restatement converts a staging decision into a standing exemption, which is the
failure the rule exists to prevent. If the developer wants the tower parked, the
honest form is B2 landed with the skip refusing by name, not a fourth note.

### 5.3 Remove `gemma4_vision.{h,cpp}` and its two tests

Rejected on the oracle. vLLM runs this tower on its default path (§2.2), so
deleting our port removes required mirrored behaviour, not scaffolding. It would
also delete the *only* completed half of the Gemma-4 image pipeline: §3.6 shows
the processor half does not exist, so after a removal the tree would hold zero
Gemma-4 vision capability and a golden fixture nothing consumes. Removal is the
right answer for a port of something upstream does not do; that is not this.

### 5.4 Promote the test driver into an example or CLI

Rejected by AGENTS.md §Shared seams ("Never write a parallel path by hand") and
§Nothing lands dead ("An example's internals are not [a production entry
point]"). It would also make the reachability mutation pass while measuring
nothing, which is worse than today's honest red.

---

## 6. Risks

| # | Risk | Handling |
|---|---|---|
| R1 | B2 lands alone and reads as "#2173 fixed" because a symbol now has a `src/` caller. It would not be: the *forward* stays unreached, and that is #1358's exact shape. | B2's commit body and PR body name B1/B3/B4 and this spec; the gate for B2 is a load-path mutation (§8.2), never a claim about the forward. |
| R2 | A wiring wave writes a Gemma-4-only mm runner path because B1 is large. | §4.1 refuses it in advance; the fresh reviewer mutates for it by deleting the shared seam and checking the Gemma-4 gate reds. |
| R3 | B3's processor is gated only against our own committed fixture, which is a transcription rather than an independent check. | The fixture carries upstream's sha256s over a committed PNG; the gate asserts byte-identity to those hashes, and the red-first form is a deliberate wrong normalize/patchify, as `test_qwen3vl_processor.cpp` did. |
| R4 | The new gates are dgx-only like the existing four, so CI stays blind (§3.7). | §8.1 requires at least one CPU/CI-runnable gate per brick. B3's is CPU-runnable by construction; B2's is a loader test on a synthetic checkpoint. |
| R5 | A wave regenerates the `ref_projected.npy` fixture inside the change it gates. | Circular, and the same trap #2166 is blocked on. The fixture is frozen; a change that needs a new one files its own issue for the capture. |
| R6 | The tower's numbers move when it stops being fed a committed f32 fixture and starts being fed our own processor's output. | Expected and bounded: MM-E2E already measured that the live bf16 tower and the committed f32 `ref_projected.npy` diverge *identically* at idx16, so the merge input's precision is not what decides the golden. B3 changes the input upstream of that, so its gate is the processor byte-gate, not the token golden. |

---

## 7. The dtype tie-in — a note, not a task

Gemma-4's tower had the same polarity every other tower in
[`vision-tower-dtype-polarity.md`](vision-tower-dtype-polarity.md) had — an f32
host store narrowed to bf16 at `MakeDevBf16` before the first GEMM — and #1359's
storage fix **already landed for it** at `525d2b991`, so
`gemma4_vision.cpp:106-125` is now a straight `Copy` of the checkpoint's own
bytes and the header's `Gemma4VisionWeights` fields are `uint16_t`. Nothing here
changes that, and nothing here should. What it does mean is that the fix's
saving is real bytes only once B2 makes a loader allocate them; today it saves
nothing outside a test, which is #2173's whole point.
`Gemma4VisionWeights::position_embedding_table` keeps its host f32 store for the
`pos_embed_w` reason recorded there (`gemma4_vision.cpp:199-210` sums the x and
y rows on the host and narrows only the sum) and rides that entry's
reconciliation.

**A false claim not to resurrect.** A pull-request body once asserted that
"Gemma-4 ran that pass on every image". It is false and was withdrawn. Nothing
in production runs this tower (§3.1–§3.4), so it costs nothing per image; there
is no per-image cost to measure until B1–B4 exist. Any future cost claim about
this tower must first name the call path that produces it.

---

## 8. Gates

Declared before any implementation exists. A wave that changes a threshold after
reading a result has broken this spec.

### 8.1 Standing requirement for every brick

Each brick lands with at least one gate that runs on a CPU CI lane. §3.7 is the
reason: the Gemma-4 vision path currently has no CI signal whatsoever, and a
wave that adds only a fifth dgx-only test leaves that unchanged.

### 8.2 B2 — reachability mutation on the load path

Red-first: a loader test on a synthetic Gemma-4 checkpoint asserts the vision
tensors are claimed and that a skipped tower refuses by name. Mutation: delete
the `LoadGemma4VisionWeights` call site in a scratch copy; the focused gate must
go red. This is the mutation `test_tower_skip.cpp` already runs for Qwen3-VL's
three sites, extended to a fourth. Restore byte-for-byte.

### 8.3 B3 — processor byte-gate

Red-first: our processor's `pixel_values` and `pixel_position_ids` must be
byte-identical to the committed fixtures and hash to the two sha256s quoted in
§4.3. The red-first form is a deliberately wrong rescale or patch order
producing a hash mismatch. CPU, no CUDA, no checkpoint, runs in CI.

### 8.4 B4 + B1 — the end-to-end reachability mutation

The gate #2173 actually closes on. An image request through the server's default
configuration produces the `gemma4_e4b_image` golden in the ratified near-tie
form MM-E2E already established (content-exact prefix; first divergence must be
a bf16 near-tie with top1−top2 margin < 0.5 past the sentence midpoint).
**Mutation:** delete the production call site where the chain reaches
`Gemma4VisionForward` and rerun the focused gate. Red proves the test enters
through the production path. Green is the finding, and means the wave repeated
#2173.

Today that mutation cannot be run at all, because there is no production call
site to delete — `.agents/reachability.md`: "A change that has no production
call site to delete has already answered the question." That sentence is the
current gate result for this row, and it is a fail.

### 8.5 Inertness, unchanged

Gemma-4 text SACRED `test_gemma4_paged_engine` STRICT 32/32 must stay unchanged
on every binary any brick touches, as G1b/G2-impl/MM-E2E each required.

---

## 9. Evidence produced by this investigation

Read-only. **Nothing was built and nothing was run on a GPU**; free disk was
7.3 G at start and 6.7 G at the branch cut, against a 15–24 G build tree, and
one-minute load average was 49.87 (another session is compiling). Every claim
below is from `grep`, `sed` and `git` over two checkouts.

| Claim | How |
|---|---|
| §3.1–§3.3 unreached, at `e3d628e41` | greps quoted inline; re-run at that SHA after the branch cut |
| §3.4 runner has no mm | `grep -c 'MultiModal\|mm_features' src/vllm/v1/worker/gpu/runner.cpp` → `0`; agrees with #2300's independent derivation at `e541be98a` |
| §3.5 no Gemma-4 loader/processor/seam | `grep -rn 'vision_tower\|embed_vision' gemma4_weights.cpp gemma4_registry.cpp` → empty; `ls src/vllm/multimodal/` has no Gemma-4 entry |
| §3.6 fixture provenance | `tests/parity/goldens/gemma4_e4b_image/vision_refs/` listing + `gen_manifest.json` `processor_outputs` |
| §3.7 zero assertions | source-level: one `TEST_CASE` per file, `MESSAGE`+`return` guards at the lines tabled |
| §2 upstream chain | `~/_git/vllm` at `HEAD == 5559679229bc961848b121ccdeaa8fa5d79bec98`, the pin exactly |
| §3.4 #2300's anchors re-checked | `runner.cpp:2234`/`:2340` are stale at `e3d628e41` (now `:2395`/`:2501`); `input_batch.h:90`, `qwen3_vl_registry.cpp:127`, `gemma4_registry.cpp:145` still hold |

### 9.1 What could NOT be settled without a leased box

- **The literal runtime assertion count** of the four tests. §3.7 establishes it
  from the source, which settles the conclusion; a run would settle the number.
  It needs a build (~15–24 G) and, for the non-skipping arm, a dgx lease with the
  `unsloth/gemma-4-E4B-it` checkpoint.
- **Any memory or latency figure** for a loaded Gemma-4 tower. None exists,
  because no loader allocates it (§3.5). The `tower_skip_rss.sh` harness has no
  `--model-kind gemma4` arm to run.
- **Whether B2's tensor names match the released checkpoint.** Upstream's
  prefixes are cited (§4.2) but the actual `unsloth/gemma-4-E4B-it` tensor index
  was not read; that needs the checkpoint, which is dgx-local.

None of the three changes the verdict.

---

## 10. Stop conditions

- **Stop if** a reader concludes from §3.4 that #2173 should be closed as a
  duplicate of #1358 or #2300. It is not: #2300 is the seam, #1358 is "loaded
  but not read back", and Gemma-4 is not even loaded (§3.5). One shared blocker,
  several issues, different distances from it. Fixing #2300 does not close
  #2173.
- **Stop if** a wave proposes to close #2173 by landing B2 alone. B2 moves this
  row to #1358's state; the issue closes on §8.4.
- **Stop and return `NEEDS_DECISION`** only if the developer wants the Gemma-4
  image capability dropped as a product scope call. The mirror question is
  settled (§2); scope is the developer's.
- **Stop if** a wave is about to describe a match against the
  `gemma4_e4b_image` fixtures as parity with the pinned oracle. §3.8 says the
  record does not currently support that sentence. Settle the provenance, or
  write the weaker claim.
- **Stop if** disk on the dev box falls below the build headroom. This spec
  needed none, and a wiring wave should take a lease rather than build here.

---

## Now

`INVESTIGATED, NOT IMPLEMENTED.` The verdict is **wire it** (§0), the wiring is
blocked on B1 which another row owns (§4.1), and B3 is the slice that can be
built and gated on CPU today (§4.3). No product code moved. #2173 stays open and
is owned by `MODEL-MM-gemma4-mm-gemma4-for-conditional-generation` through its
`Row:` line and through
[`vision-tower-dtype-polarity.md`](vision-tower-dtype-polarity.md) §Owed.

## Owed

- [#2173](https://github.com/mudler/vllm.cpp/issues/2173) — B1–B4 (§4). Closes
  on §8.4 and on nothing weaker.
- **Gemma-4 audio reachability.** `gemma4_audio.cpp`'s USM-Conformer tower has
  the same shape as the vision tower — G3 landed it per-stage-gated, and
  `gemma4-multimodal.md` §MM-E2E residual 2 names the mel frontend and merge
  wiring as unbuilt. It is not measured in this spec and has no issue of its
  own. It needs one before it is touched, and it is listed here so it is not
  lost.
- **The `gemma4_e4b_image` golden's provenance (§3.8) NEEDS AN ISSUE.** Its
  manifest records a live-read transformers `5.13.1` for a capture that two
  other specs say `5.13.1` could not have run. It is the golden §8.4 closes on,
  so it is not cosmetic. No issue was filed from this session because the
  session had no write access to the tracker; the operator owns filing it. Until
  then this bullet is where it lives.
- **A CI signal for the Gemma-4 vision path**, independent of the wiring. §3.7
  shows there is none today. §8.1 makes it a standing requirement of each brick;
  if the bricks stall, it is owed on its own.
- **`.supports_multimodal = true` on `gemma4_registry.cpp:55`** is a support
  claim the port cannot honour for image input today (§3.5). It is inert, so no
  behaviour depends on it, and `dots3_note_registry.cpp:47-69` argues the
  precedent for both directions. Flipping it is not this spec's call and is
  listed so the next reader sees it was measured, not missed.
