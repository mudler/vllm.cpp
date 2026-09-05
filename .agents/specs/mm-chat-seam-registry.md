# The multimodal chat seam dispatches on the architecture (`ENG-MM-INPUT-PIPELINE`)

Make the OpenAI server's multimodal install select its processor and chat
function by the model's **architecture**, through a registry a new model
self-registers into, instead of by the presence of a file name with Qwen3-VL
hardcoded behind it. Issue
[#2475](https://github.com/mudler/vllm.cpp/issues/2475). It also answers item 5
of [#2408](https://github.com/mudler/vllm.cpp/issues/2408), which records that
this same block has no reachability mutation of its own; #2408 tracks four other
items and stays open.

**Verdict up front.** `src/vllm/entrypoints/openai/server_main.cpp:1513-1566`
decides whether the server can serve images by asking whether
`<model_dir>/preprocessor_config.json` exists, and then constructs
`vllm::multimodal::Qwen3VLImageProcessor` and `oai::MakeQwen3VLImageChatFn`
unconditionally. There is no branch on the architecture anywhere in the block,
and the type system cannot supply one: `MakeQwen3VLImageChatFn` takes
`const multimodal::Qwen3VLImageProcessor&` by concrete type
(`include/vllm/entrypoints/openai/chat_mm.h:268-273`), it is the only image chat
function in the tree, and `include/vllm/multimodal/qwen3vl_processor.h` is the
only processor.

`preprocessor_config.json` is not a Qwen3-VL marker. A second multimodal
checkpoint that ships one gets Qwen3-VL's processor read against its own
`vision_config` shape, or throws into the `catch` at `:1562-1565`, which prints
one line and leaves the server on the text-only path. The silent arm is the
worse one: a multimodal model that answers an image request as though it were
text looks like a working server.

## Scope

In:

- a per-architecture multimodal chat registry in the entrypoints layer, with the
  self-registration idiom `ModelFactory` / `REGISTER_VLLM_MODEL` already uses;
- one production install function, `InstallMultiModalChatSeam`, that both
  `server_main.cpp` and the reachability gate call;
- Qwen3-VL moved behind that registry, its behaviour unchanged;
- a refusal, by name, for an architecture that declares multimodal support and
  has no registered chat seam;
- the reachability mutations #2408 item 5 owes, for the install being removed
  **and** for the wrong architecture's processor being installed.

Out:

- **no second model.** This change adds no `dots3-note` code. It makes the seam
  able to take one. The `dots3-note` W6 port supplies its own factory and one
  `REGISTER_VLLM_MM_CHAT` line, and edits nothing here.
- the `ImageCodecFn` raw-RGB-only limit at `:1524-1548`. It is a named MM-SERVE
  residual, it is not architecture-specific, and it moves verbatim into the
  install context. Nothing about it changes.
- the audio and video seams. `chat_mm.h` carries an audio processor and
  `RouteAudio*`; nothing installs them on the chat path today, and giving them a
  registry entry before a model reaches them would be the dead-code failure this
  row is trying to stop.
- #2450, #2430, #2431, #2399's `windows-msvc` C4456 shadowing and #618. All
  pre-existing and red repo-wide; attributed, not chased.

## Upstream anchors

All at `5559679229bc`, the pin in `.agents/upstream-sync.md`.

vLLM already resolves the multimodal processor by architecture, through a
registry a model self-registers into. The mechanism is
`vllm/multimodal/registry.py`:

| Upstream | What it does | Our mirror |
|---|---|---|
| `registry.py:81-95` `_ProcessorFactories` + `build_processor` | the per-model record: an info factory, a processor factory, a dummy-inputs factory; `build_processor` runs them in order and hands back the processor | `MultiModalChatRegistration` + `MultiModalChatFactoryFn`, one function that builds the whole seam |
| `registry.py:142-172` `register_processor` | the decorator that attaches `_processor_factory` **to the model class**, i.e. keys the record on the architecture | `REGISTER_VLLM_MM_CHAT(tag, architecture, factory)`, keyed on the architecture string |
| `qwen3_vl.py:1673-1678` | Qwen3-VL's own registration, sitting on the model, editing no shared table | `src/vllm/entrypoints/openai/mm_chat_qwen3vl.cpp`, one TU, editing no shared table |
| `registry.py:176-186` `_get_model_cls` | resolves `get_model_architecture(model_config)` and **raises by name** — `"Model class {name} has no registered multimodal processor"` — when the class carries no factory | `MultiModalChatRegistry::MakeSeam` → `RaiseForUnregistered`, which names the architecture and the missing part |
| `registry.py:211-230` `create_processor` | refuses when the model is not multimodal at all, then dispatches through the resolved class | `InstallMultiModalChatSeam`'s three-way outcome below |
| `registry.py:103-139` `supports_multimodal_inputs` | the predicate: `model_config.is_multimodal_model` **and** a registered processor. When the model is multimodal but the processor is missing it logs `warning_once` naming the model and runs text-only | the install's `kRefusing` arm: log loudly, keep serving text, refuse an mm request by name |

Two upstream facts decide the shape:

1. **The key is the architecture, not a file.** `_get_model_cls` reaches
   `get_model_architecture(model_config)` (`registry.py:178-180`). Upstream never
   looks at `preprocessor_config.json` to decide *whether* the model is
   multimodal; the processor reads that file once it has been selected. Our gate
   inverts exactly that, which is #2475.
2. **A missing processor is refused by name, never substituted.**
   `registry.py:182-185` raises rather than falling back to another model's
   factory. There is no default processor upstream and there is none here.

## Design

### Where it lives, and why not on `ModelFactory`

The obvious mirror of "the registration carries the hook" is a new field on
`ModelFactory` beside `encode_mm` / `embed_mm`
(`include/vllm/model_executor/models/model_registry.h:761-767`). It is the wrong
place here, and the reason is layering rather than taste. The seam a chat
factory has to produce is `oai::MultiModalChatFn`; building one needs
`ChatPromptRenderFn`, `ImageCodecFn`, `ChatMessage` and `BaseProcessingInfo` —
all in `entrypoints/openai`. Putting that on `ModelFactory` makes
`model_executor` depend on `entrypoints`, which is backwards, and it is not what
upstream does either: upstream's registered artefact is a
`BaseMultiModalProcessor` from `vllm/multimodal/`, and `chat_utils.py` /
`serving_chat.py` carry no per-model registration at all.

So the registry lives beside the thing it produces,
`include/vllm/entrypoints/openai/mm_chat_registry.h`, and is keyed on the same
architecture string `ModelRegistry` resolves. The idiom is copied from
`REGISTER_VLLM_MODEL` (`model_registry.h:939-976`) verbatim in shape — a
TU-static `…Registrar` whose constructor registers, reached through
`--whole-archive` on the `vllm` static library — so a reader who knows one knows
the other, and adding a model is a new TU with one line and **zero** edits to a
shared array. That is also the record rule: one file per writer, never a surface
every pull request has to write.

The engine half stays exactly where it is. `ModelFactory::encode_mm` /
`embed_mm` remain the model's mm-forward hooks and
`ModelRegistry::SupportsMmInputs` remains their derived predicate
(`model_registry.cpp:661-664`). This row adds the **serving** half of the same
split upstream has.

### The three-way outcome

```
enum class MultiModalChatInstall { kTextOnlyModel, kInstalled, kRefusing };
```

- **`kTextOnlyModel`** — the architecture's `ModelInfo::supports_multimodal` is
  false. Nothing is installed, `mm_chat_fn_` stays null, and the chat path is
  byte-identical to the text-only server. This mirrors
  `supports_multimodal_inputs`' first line (`registry.py:110-111`) and it is
  already strictly better than today's gate, which tries and fails on any
  text-only checkpoint that happens to ship a `preprocessor_config.json`.
- **`kInstalled`** — the architecture declares multimodal support and its
  registered factory built a seam. Qwen3-VL, unchanged.
- **`kRefusing`** — the architecture declares multimodal support and either has
  no registered factory or its factory threw. The server logs the failure with
  the architecture named, and installs a chat function that **throws**
  `vllm::v1::InputValidationError` on any request carrying a multimodal part.

`kRefusing` is the argued decision, and it replaces the swallowed `catch`.
Three candidates were on the table:

1. *keep printing and leave the seam unset* — today's behaviour, and the defect.
   An image request is then answered as text. Rejected outright.
2. *let the throw propagate and refuse to start* — the strictest reading of
   "refuse an unimplemented arm". Rejected: a GGUF Qwen3-VL served with
   `--mmproj` has no `preprocessor_config.json` next to it, and today that
   configuration starts and serves text. Making the server unlaunchable is a
   regression on a live arm to fix a silence, and it also makes a staged model
   port (engine half first, chat half after) unable to serve at all.
3. *install a refusing seam* — chosen. It is impossible to answer an mm request
   silently, because the only chat function present throws; the text path keeps
   working, which is the mode option 2 would have taken away; and it is
   expressible in the existing type, because `serving_chat.cpp:699-710` calls
   `mm_chat_fn_` only when `HasMultiModalParts` is true. It is also the closer
   mirror: upstream's `supports_multimodal_inputs` takes precisely this shape —
   `logger.warning_once` naming the model, then text-only — while the raise it
   pairs with lives in `create_processor`, which is where our
   `MultiModalChatRegistry::MakeSeam` raises.

The refusal is `InputValidationError`, which `api_server.cpp:357-360` maps to
HTTP 400 with `BadRequestError`, the same mapping the `--language-model-only`
refusal already lands on. 400 rather than 500 because the condition is "this
server does not accept images for this model", which is the same class of answer
as "At most 0 image(s) may be provided in one prompt.", and because a 500 tells
the client to retry.

### The seam owns its state

Today `mm_image_proc` and `mm_proc_info` are two `unique_ptr` locals in `main`,
borrowed by the closure, with a comment explaining their declaration order
against `loaded`. The factory returns

```
struct MultiModalChatSeam {
  MultiModalChatFn chat_fn;                    // OWNS its processor state
  std::map<std::string, int> allowed_limits;   // what the server logs
  std::string detail;                          // "Qwen3-VL processor from <path>"
};
```

and the Qwen3-VL factory captures the processor and the `BaseProcessingInfo` as
`shared_ptr` inside `chat_fn`. Two locals and their ordering comment leave
`main`, and a future architecture cannot get its own lifetime wrong by
forgetting to add a third. The `BaseProcessingInfo` still holds the engine's
`MultiModalConfig` by reference, exactly as it does today; `chat` is declared
AFTER `loaded` in `server_main.cpp` and destruction runs in reverse of
declaration order, so the closure is destroyed first, and `~BaseProcessingInfo`
never reads the reference in any case.

`Qwen3VLChatSupportedMmLimits()` — this seam's own per-modality ceiling — is
now read by the Qwen3-VL factory instead of by `main`, which is where upstream
reads it too (`info.supported_mm_limits` comes from the model class,
`context.py:378-390`). Every architecture declares its own; the server only
prints whatever `allowed_limits` comes back.

### The context

`MultiModalChatContext` is our `InputProcessingContext`
(`registry.py:188-195`) plus the two serving collaborators upstream's processor
does not need, because upstream renders the chat template and decodes images
elsewhere:

```
architecture, model_dir, config_path, served_model_name,
tokenizer, prompt_fn, codec, mm_config
```

`codec` is in the context and not in the factory deliberately: the raw-RGB-only
limit is the **server's** residual, not an architecture's, and every future
factory should consume the one codec the server supplies rather than grow its
own.

`LoadedEngine` gains two accessors, in the shape of the `is_pooling_model()`
that already reads the registration (`model_loader.h:561-568`):
`architecture()` and `is_multimodal_model()`.

## Risks

- **A registration that does not link.** Static self-registration in a static
  library is dropped unless the archive is whole-linked. It is
  (`CMakeLists.txt:1451-1453`, an INTERFACE `--whole-archive` / `-force_load`),
  and the model registry already depends on it. The gate case "the Qwen3-VL
  architecture has a registered chat seam" fails loudly if it ever stops being
  true.
- **Static-init order.** Registration order across TUs is unspecified. The
  registry sorts by architecture on first query, exactly as `model_registry.cpp`
  does, so lookup is order-independent.
- **`kRefusing` reached by an operator who wanted text.** Someone serving a
  Qwen3-VL checkpoint deliberately as text gets a 400 on an image request
  instead of a text answer. That is the intended change and it is the whole
  point; it is not a new refusal for `--language-model-only`, which already
  answers "At most 0 image(s)".
- **The production call site is ungated, and it is wider than one line.** An
  earlier draft of this bullet said the residual was "the one line no unit test
  reaches", and that understated it in two ways.

  First, #2408 item 5's literal sentence — deleting the production line leaves
  the P2 suite green — is **still true** after this change. Two mutations on
  `server_main.cpp` prove it, each applied to the head, each with the test
  binary's sha256 showing the mutation reached the binary:

  | mutation | `test_openai_api_server_mm_forward` sha256 | full suite |
  |---|---|---|
  | none (the head) | `8df0aedf…d581f32` | GREEN 9/9 cases, 73/73 assertions |
  | `mm_ctx.architecture = "";` | `0b45f1f5…727c0bd8` | GREEN 9/9 cases, 73/73 assertions |
  | `InstallMultiModalChatSeam(chat, /*is_multimodal_model=*/false, …)` | `5570ca07…663dba1f` | GREEN 9/9 cases, 73/73 assertions |

  `server_main.cpp` is compiled into the `vllm` library
  (`CMakeLists.txt:2528`), so the test binary links it and the changed sha256 is
  what proves each mutation reached the binary rather than being skipped.

  The second reintroduces exactly the defect #2475 names: a multimodal model
  answers image requests from the text path. So what no test reaches is not one
  line. It is the whole eight-field `MultiModalChatContext` assembly in `main`
  together with both new `LoadedEngine` accessors, `architecture()` and
  `is_multimodal_model()` — every input the install reads.

  Second, the earlier claim that "no test binary enters `main`" is false as
  stated; `examples/CMakeLists.txt:283-320` is an in-tree counter-example, a set
  of `ctest` cases that drive `$<TARGET_FILE:server>` and assert on its output.
  The accurate statement is narrower: this install runs **after**
  `LoadedEngine::FromModelDir`, so a `--model /nonexistent-model-dir`
  invocation — the whole trick that precedent turns on, and which its own
  comment says is what makes a direct-call unit test prove "NOTHING about
  reach" — exits at the model load and never reaches the install.

  What is closed, and verified: the **harness-duplication** half of item 5.
  `test_api_server_mm_forward.cpp` no longer calls `MakeQwen3VLImageChatFn`
  itself, so M1 — deleting `chat.set_multimodal_chat_fn(seam.chat_fn)` inside
  `InstallMultiModalChatSeam` — now goes red where before this change it did
  not. The **production-call-site** half is not closed.

  What closing it would need. A checkpoint the real loader accepts, whose
  architecture **declares multimodal support**. The declaration is not optional:
  on a text-only checkpoint the mutated call
  `InstallMultiModalChatSeam(chat, false, ...)` and the real one are
  observationally identical — both return `kTextOnlyModel` and print nothing —
  so neither mutation above is detectable. The only committed checkpoint
  carrying weights is `tests/vllm/models/fixtures/llama_embed_e2e`
  (`LlamaModel`, 154 KB, text-only). Every architecture whose
  `ModelInfo::supports_multimodal` is true is large — Qwen3-VL, Qwen3.5,
  Gemma4, GLM5-Next, Qwen4-Exp, Kimi-K3, MuseGlimmer — and every multimodal
  fixture in-tree is config-only with no weights, for the reason
  `test_api_server_mm_forward.cpp`'s own header records: `LoadQwen3VLWeights`
  hard-codes the 4B tower geometry (hidden 1024, depth 24, ~300M parameters)
  and a test cannot synthesise a checkpoint for it.

  This was measured against the built binary rather than argued, and the
  measurement is what makes it a residual rather than a guess:

  ```
  $ ./build/examples/vllm-server --model tests/vllm/models/fixtures/qwen4_exp --port 1
  server: fatal: tokenizer: cannot open .../qwen4_exp/tokenizer.json

  $ ./build/examples/vllm-server --model tests/vllm/models/fixtures/glm5_next --port 1
  server: fatal: tokenizer: cannot open .../glm5_next/tokenizer.json

  $ ./build/examples/vllm-server --model tests/vllm/models/fixtures/llama_embed_e2e --port 1
  server: failed to bind 0.0.0.0:1
  ```

  The third line is the point. The one checkpoint that loads DOES carry the
  server past the install and out through a controlled exit, so neither the
  reach nor the exit path is the obstacle; the install on that model returns
  `kTextOnlyModel` and prints NOTHING, so there is nothing for a
  `PASS_REGULAR_EXPRESSION` to hold and both residual mutations stay invisible.
  The two multimodal fixtures do not even reach the model load. What is missing
  is exactly one thing: a multimodal-declaring checkpoint with a `tokenizer.json`
  and weights the production loader accepts. Committing one is a fixture with
  its own generator and its own review, not a test, so it stays owed here and
  #2408 item 5's production half stays open.

## Tests

`tests/vllm/entrypoints/openai/test_api_server_mm_forward.cpp`, which already
drives `ApiServer::handle_chat_completions` over the whole production serving
stack on a CPU queue with a synthetic Qwen3-VL checkpoint.

The harness stops hand-wiring the seam. `MmServerHarness::wire_multimodal_seam`
called `oai::MakeQwen3VLImageChatFn` itself, which is exactly why #2408 item 5
could delete the production install and watch the suite stay green: the test was
re-implementing the thing it was supposed to be gating. It now calls
`oai::InstallMultiModalChatSeam` — the production function — against a temporary
model directory carrying a real `preprocessor_config.json` and `config.json`
whose values reproduce the fixture geometry.

Cases:

1. *the Qwen3-VL architecture is registered, and an unregistered one is not* —
   `MultiModalChatRegistry::Find` over both.
2. *a served image chat request reaches the model forward* (existing, now
   through the production install).
3. *two image requests on one server both answer* (existing).
4. *two DIFFERENT images give two different completions* (existing).
5. **new** — *an architecture with no registered chat seam REFUSES the image
   request by name*: install with a made-up architecture that declares
   multimodal support, send the same image body, expect HTTP 400 naming that
   architecture, and expect the answer NOT to be a Qwen3-VL answer.
6. **new** — *the same refusing server still answers a text request*, so the
   refusal is scoped to multimodal parts.
7. **new** — *a text-only architecture installs nothing*: the install reports
   `kTextOnlyModel` and an image request is not touched by any seam.
8. **new (fresh-review F2)** — *a registered factory that returns an EMPTY seam
   refuses rather than reporting success*. The null `make_seam` POINTER was
   checked and the empty `chat_fn` it can RETURN was not, and the second hole is
   the quiet one: `set_multimodal_chat_fn(<empty std::function>)` leaves
   `serving_chat.cpp:699`'s `if (mm_chat_fn_)` false, so the image request takes
   the text path while the install reports `kInstalled` and logs "seam wired for
   architecture 'X'". The case registers a factory through the production macro
   from the test translation unit, asserts the install does NOT report success,
   and asserts the image request does NOT reach the text path — the second by
   the absence of the fixture forward's own "requires multimodal inputs
   (ModelForwardInput.mm)" refusal, which is what the dropped image produces.

## Gates

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_SERVER=ON -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
# The target names are `test_openai_api_server*`, not `test_api_server*`; an
# earlier draft of this block wrote the short spelling, which cmake answers with
# "unknown target" and a reader reads as a compile failure.
cmake --build build -j 2 --target test_openai_api_server_mm_forward \
  test_chat_mm test_openai_serving test_serve_mm_limits \
  test_openai_api_server test_tower_skip test_capi
./build/tests/test_openai_api_server_mm_forward
./build/tests/test_chat_mm
./build/tests/test_openai_serving
./build/tests/test_serve_mm_limits
./build/tests/test_openai_api_server
./build/tests/test_tower_skip
./build/tests/test_capi
scripts/agent-preflight.sh --staged
```

Mutations, each applied to the committed head, captured red, restored
byte-for-byte, captured green, with the test binary's sha256 compared mutated
against restored so a mutation that never reached the binary cannot read as a
pass:

- **M1, the install removed.** Delete `chat.set_multimodal_chat_fn(seam.chat_fn)`
  from `InstallMultiModalChatSeam`. Cases 2, 3 and 4 must go red.
- **M2, the wrong architecture's processor installed.** Make
  `MultiModalChatRegistry::Find` ignore its key and return the first
  registration. Case 5 must go red, because the unregistered architecture then
  gets Qwen3-VL's seam and answers 200 instead of refusing.
- **M4, the empty-seam guard removed** (fresh-review F2). Delete the
  `if (!seam.chat_fn)` raise from `MultiModalChatRegistry::MakeSeam`. Case 8
  must go red, on both of its halves: the install reports `kInstalled` and the
  image request comes back carrying the forward's text-path refusal. M3 and M3b
  are the two residual mutations recorded under `## Risks`; they are green by
  construction and are named there rather than listed here, because a mutation
  nothing detects is a gap and not a gate.

## Evidence

Recorded in the pull request body and in `## Outcome` below when the row lands:
the verbatim red and green for M1 and M2, the two sha256 pairs, and the per-suite
case counts.

## Stop conditions

- Return `NEEDS_DECISION` if mirroring vLLM and mirroring the in-tree
  `ModelFactory` idiom point at genuinely different shapes. They do not: both are
  a per-architecture record of function pointers, self-registered from the
  architecture's own translation unit, refusing by name when the hook is null.
- Stop and report if Qwen3-VL's behaviour cannot be kept byte-identical through
  the registry. It is the only current consumer and its suites are the floor.
- Do not widen into audio, video, or a second model to make a gate pass.
