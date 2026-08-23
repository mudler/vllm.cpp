# `SERVE-CHAT-TEMPLATE` — the vendored Jinja engine has no `is undefined`

Row: `SERVE-CHAT-TEMPLATE`. Owning matrix row:
[`SERVE-CHAT-TEMPLATE`](../engine-matrix.md) (Serving surface, CLI, and
library). Issue: [#1681](https://github.com/mudler/vllm.cpp/issues/1681).
Blocked campaign: [#1574](https://github.com/mudler/vllm.cpp/issues/1574).

Upstream pin:

| Reference | Revision |
|---|---|
| vLLM (parity pin) | `5559679229bc961848b121ccdeaa8fa5d79bec98` |
| google/minja (vendored) | `021c229` + the local edits in `third_party/README.md` |
| Jinja2 built-in test list | `jinja2/tests.py` `TESTS` (the name set mirrored below) |

Read from the local checkout `/home/mudler/_git/vllm` at the parity pin;
`git rev-parse HEAD` was run before any anchor below was taken.

---

## 0. Honesty statement — what this row claims

No GPU lease was taken and no model weights were loaded. Every gate here is CPU
and needs neither. The template is the real published Qwen3.8 one, committed as
a fixture; the engine behind the `/v1/chat/completions` dispatch is the synthetic
one `tests/vllm/entrypoints/openai/test_api_server.cpp` already uses.

The claim is exactly: the production `/v1/chat/completions` dispatch renders the
real published Qwen3.8-27B chat template instead of answering HTTP 500, and a
caller's `chat_template_kwargs` reach the renderer. No throughput, latency or
token claim is made, and no `dgx:gpu0` reproduction of the issue's original
`curl` is claimed.

---

## 1. The defect

`third_party/minja/minja.hpp`, `BinaryOpExpr::do_evaluate`, implements the Jinja
`is` operator over a closed list of names and throws on anything else:

```cpp
if (name == "defined") return !l.is_null();
if (name == "true")    return l.to_bool();
if (name == "false")   return !l.to_bool();
throw std::runtime_error("Unknown type for 'is' operator: " + name);
```

`undefined` is absent. It is a Jinja2 built-in and the standard idiom for "was
this variable supplied?", so the Qwen3.8 family template

```jinja
{%- if enable_thinking is undefined or enable_thinking is true %}
```

throws at row 46. `src/vllm/entrypoints/chat_template.cpp` converts that into a
`ChatTemplateError`, and `ApiServer::handle_chat_completions` maps any
`std::exception` from `create_chat_completion` to HTTP 500. Every chat request
against the whole Qwen3.8 family therefore fails.

**The gap is upstream's, not a vendoring accident.** `google/minja` `main`
carries the identical list (fetched 2026-08-22, same twelve names, same throw),
so there is no upstream revision to advance onto. This tree already patches
`minja.hpp` locally — `59674cf1d` for the GCC 15 `-Werror` build and the
`MacroNode` ownership cycle, plus the documented `lstrip_blocks` edit — so the
established pattern is an in-tree edit recorded in
[`third_party/README.md`](../../third_party/README.md), and that is what this
row does.

### 1a. Why no gate saw it

Every benchmark and gate this project runs against a chat-capable checkpoint
drives `vllm-cli`, which renders no chat template. `test_chat_template.cpp`
renders the committed `qwen35_chat_template.jinja`, which *does* contain

```jinja
{%- elif content is none or content is undefined %}
```

at line 36 — but minja's `or` short-circuits, every gated conversation reaches
that `elif` with `content is none` already true, and the second term is never
evaluated. The construct has been in the tree, unrendered, the whole time.

---

## 2. The built-in test inventory

Measured against Jinja2's `TESTS` mapping. "minja" is the vendored engine before
this row.

| Jinja2 test | Arity | minja before | this row |
|---|---|---|---|
| `boolean` | 0 | yes | yes |
| `callable` | 0 | **no** | OWED |
| `defined` | 0 | yes | yes |
| `divisibleby` | 1 | **no** | OWED |
| `eq` / `equalto` / `==` | 1 | **no** | OWED |
| `escaped` | 0 | **no** | ADDED |
| `even` | 0 | **no** | ADDED |
| `false` | 0 | yes | yes |
| `filter` | 0 | **no** | OWED |
| `float` | 0 | yes | yes |
| `ge` / `>=` | 1 | **no** | OWED |
| `gt` / `greaterthan` / `>` | 1 | **no** | OWED |
| `in` | 1 | **no** | OWED |
| `integer` | 0 | yes | yes |
| `iterable` | 0 | yes | yes |
| `le` / `<=` | 1 | **no** | OWED |
| `lower` | 0 | **no** | ADDED |
| `lt` / `lessthan` / `<` | 1 | **no** | OWED |
| `mapping` | 0 | yes | yes |
| `ne` / `!=` | 1 | **no** | OWED |
| `none` | 0 | yes | yes |
| `number` | 0 | yes | yes |
| `odd` | 0 | **no** | ADDED |
| `sameas` | 1 | **no** | OWED |
| `sequence` | 0 | yes | yes |
| `string` | 0 | yes | yes |
| `test` | 0 | **no** | OWED |
| `true` | 0 | yes | yes |
| `undefined` | 0 | **no** | **ADDED — the defect** |
| `upper` | 0 | **no** | ADDED |

**`is true` was already implemented**, so the issue's suspicion that the second
term of the failing expression was missing too is resolved: it was not. The
first term was the only break.

The line the ADDED/OWED split falls on is reachability, not taste, and it was
drawn by measurement rather than by preference.

- **The nine arity-1 tests** need a grammar change. `parseLogicalCompare` reads the
  right side of `is` with `parseIdentifier()` (`minja.hpp`), so a bare name is
  the only shape the parser accepts and `x is divisibleby(3)` never reaches the
  evaluator at all.
- **`filter` and `test`** ask the engine which filter and test NAMES exist.
  minja has no such registry: its filters are ordinary context callables.
- **`callable` is OWED although it is arity-0**, and this is the one that had to
  be measured. `BinaryOpExpr::do_evaluate` DEFERS every binary operation whose
  left operand is callable, returning a new callable that applies the operation
  to the call's RESULT (the `l.is_callable()` branch at the end of the function).
  So `x is callable` can never be handed a callable, and an implementation would
  answer false for every input a template can construct. Half an answer is worse
  than a refusal, so it refuses; the first draft of this row implemented it and
  the test that asserted the true case is what caught it.

Twelve names in all, and the arithmetic closes: 30 canonical tests, 12 minja
already had, 6 added here, 12 owed.

### 2a. What the shipped checkpoints actually use

Every `is <name>` occurrence in the chat template of every chat-capable
checkpoint in the [`docs/USAGE.md`](../../docs/USAGE.md) weights table, read
from the pinned revision that table names:

| Checkpoint | Template source | Tests used |
|---|---|---|
| `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` | `tokenizer_config.json` and `chat_template.jinja`, byte-identical | `defined`, `false`, `iterable`, `mapping`, `none`, `string`, `true`, **`undefined`** |
| `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d` | `chat_template.jinja` | `defined`, `false`, `iterable`, `mapping`, `none`, `string`, `true`, **`undefined`** |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` @ `29f2d174` | `chat_template.jinja` | `defined`, `iterable`, `mapping`, `sequence`, `string` |
| `Qwen/Qwen3-0.6B` @ `c1899de2` | `tokenizer_config.json` | `defined`, `false`, `string` |

**No arity-1 test appears in any of them**, which is what makes the ADDED/OWED
split above safe rather than convenient. `undefined` is the only name in the
union that the engine lacked.

---

## 3. The second half — `chat_template_kwargs` never reached the renderer

Answering the issue's third scope item. It did not, in two separate ways.

**(a) There is no request field.** `ChatCompletionRequest` carries no
`chat_template_kwargs`, so a caller that sends
`{"chat_template_kwargs":{"enable_thinking":false}}` — the exact body both the
vLLM and the SGLang arms of #1574 were measured with — is silently ignored.
Upstream declares it at `vllm/entrypoints/openai/chat_completion/protocol.py:341`
and merges it into the render kwargs at `:545-556`.

**(b) The default is wrong in the direction that matters.**
`apply_chat_template` unconditionally executes

```cpp
top["enable_thinking"] = enable_thinking;          // default false
context->set("enable_thinking", minja::Value(enable_thinking));
```

so `enable_thinking` is *always* a defined Jinja variable. Even with `undefined`
implemented, `enable_thinking is undefined` would evaluate false and the Qwen3.8
default would be thinking-OFF. Upstream passes only the keys the caller (or
`--default-chat-template-kwargs`, itself defaulting to `None`) supplied
(`vllm/renderers/hf.py:633-661`, `:731-734`), so on a bare request the variable
is genuinely undefined and Qwen3.8 renders thinking-ON. Mirroring vLLM means the
variable must be absent when nobody asked for it, and this row makes it absent.

This is a **behaviour change on the default configuration**, recorded here and
in the pull request body rather than smuggled: a bare
`POST /v1/chat/completions` against a Qwen3.8 checkpoint now renders the
reasoning branch, as it does on vLLM and on SGLang. `--enable-thinking` and
`--no-enable-thinking` keep working and now mean "set the key to true/false";
passing neither means "do not set the key", which is upstream's default.

---

## 4. Design

Four changes, smallest each.

1. **`third_party/minja/minja.hpp`** — the six arity-0 tests from the table.
   `undefined` is `l.is_null()`, the exact complement of the `defined` beside
   it. `escaped` is a constant `false` because Jinja's `escaped` is
   `hasattr(v, "__html__")` and minja has no Markup type, so no value can carry
   one. Recorded in `third_party/README.md`.

   **One divergence from CPython jinja2, deliberate.** minja has no distinct
   Undefined type: an unbound name evaluates to null, and its shipped `defined`
   is `!is_null()`. So a value bound to an explicit null reads as *undefined*
   here, where jinja2 would call it defined-and-None. The divergence belongs to
   `defined`, which shipped with the vendoring and is load-bearing; answering
   `undefined` any other way would make the two built-in tests contradict each
   other on the same value. The coupling is pinned by a test rather than left
   implicit. No template in section 2a can observe it: the one place a real
   template asks (`{%- elif content is none or content is undefined %}` in the
   Qwen3.5 fixture) short-circuits on `is none` first.

2. **`include/vllm/entrypoints/chat_template.h` / `.cpp`** — the
   `bool enable_thinking` parameter of `apply_chat_template` and
   `MakeChatTemplatePromptFn` becomes an `nlohmann::ordered_json`
   `chat_template_kwargs` object whose keys are set into the render context and
   whose absence leaves the variable undefined. A new
   `DefaultChatTemplateKwargs(std::optional<bool>)` carries the
   `--enable-thinking` tri-state rule, so the rule is drivable from a CPU gate
   even though `server_main`'s one call to it is not (it runs only after a real
   tokenizer loads).

   **The request keys are FILTERED, not bound as they come.** This is the seam
   the field opens, and the fresh review of the first implementation caught it
   wide open: bound unfiltered, and bound AFTER the renderer had set its own
   names, a request key silently REPLACED `messages`, so the model was fed a
   conversation that the request log line, `usage`, `ToolsEnabled` and every
   policy layer reading `request.messages` never saw. Nothing upstream can do
   that. Re-executing the pinned chain (vLLM `555967922`, transformers 5.3.0)
   over `tests/fixtures/qwen38_chat_template.jinja` measured all four arms:

   | Request key | Pinned vLLM | Here |
   |---|---|---|
   | `chat_template`, `tokenize` | `resolve_chat_template_kwargs` raises `ValueError: Found unexpected chat template kwargs from request: {...}` (`vllm/renderers/hf.py:639-648`; its only call site takes the default `raise_on_unexpected=True`, `hf.py:731-735`) | refused |
   | `messages`, `tools` | kept by the filter (both are in `find_undeclared_variables`, and `tools` is an `apply_chat_template` parameter), then `TypeError: got multiple values for keyword argument ...` at `tokenizer.apply_chat_template(conversation=..., tools=tools, ...)` and at `compiled_template.render(messages=chat, ..., **kwargs)` | refused |
   | `add_generation_prompt`, `continue_final_message` | kept, but `build_chat_params` already put the request's OWN field on the OVERRIDE side of `merge_kwargs` (`chat_completion/protocol.py:530-544`, `renderers/params.py:28-40`), so the kwarg never reaches the render | skipped; for `add_generation_prompt` the function's parameter of the same name IS that field |
   | a name minja's own builtins layer supplies | dropped: jinja2 keeps its globals, filters and tests OUT of the variable namespace, so `find_undeclared_variables` never reports one and `accept_vars` cannot keep it | skipped, except `raise_exception` (below) |
   | `bos_token`, `eos_token` | dropped when the template names neither; kept when it names either, and then they win over `special_tokens_map` (`template_kwargs = {**self.special_tokens_map, **kwargs}`) | bound, which is the same render in both cases |

   The refusal throws `vllm::v1::InputValidationError`, not `ChatTemplateError`,
   because it is a client mistake: `api_server` maps that type to **400** the
   way upstream's `ValueError`/`TypeError` reach `create_error_response`'s
   `BadRequestError` default (`serve/utils/error_response.py:16-21`), and the C
   ABI maps it to `VLLM_ERR_INVALID_ARGUMENT`. `apply_chat_template` rethrows it
   ahead of the generic arm so it is not rewrapped into a 500.

   **The engine's own names, which the first review's filter could not see.**
   The fourth row above is the second review's F1, and it falsified the
   completeness argument this section used to carry. That argument counted the
   collisions as the four names the ADAPTER sets. The collision set is those
   PLUS every name the ENGINE supplies, because minja resolves a global, a
   filter and an is-test through one `Context` chain: `Context::make` parents
   the render context on `Context::builtins()`, `set()` writes into the CHILD,
   and `select`/`reject` even resolve their test by name through the same
   `context->get` (`third_party/minja/minja.hpp`). So a request key shadowed any
   of minja's 31 builtins, and `{"namespace": 1}` broke line 1 of the shipped
   Qwen3.8 template -- `{%- set image_count = namespace(value=0) %}` -- as a
   client-triggerable HTTP 500 on the default chat path.

   Upstream answers 200 for all of them, and the reason is structural rather
   than incidental: CPython jinja2 resolves a filter through `env.filters`, a
   test through `env.tests` and a global through `env.globals`, none of which is
   the variable namespace `find_undeclared_variables` reports on. Re-executing
   `_resolve_chat_template_kwargs`'s own environment (`hf.py:598-606`) on
   jinja2 3.1.2 over this fixture, and `_get_hf_base_chat_template_params` on
   transformers 5.3.0, `template_vars | hf_base_params` keeps exactly one of
   minja's 31 names:

   ```text
   jinja2 env.globals = {cycler, dict, joiner, lipsum, namespace, range}
   minja's 31 builtins, by where jinja2 supplies each one:
     24 are jinja2 FILTERS  (tojson, items, last, trim, lower, upper, ...)
      6 are jinja2 TESTS    (==, equalto, in, lower, string, upper)
      3 are jinja2 GLOBALS  (joiner, namespace, range)
     -- three names are BOTH a filter and a test, so the union is 30, and
        none of the three namespaces is the variable namespace --
      1 is supplied by jinja2 NOWHERE: raise_exception
   accept_vars & minja_builtins = {raise_exception}
   ```

   `raise_exception` is transformers' own global, added to the environment
   AFTER `_resolve_chat_template_kwargs` has already parsed with a fresh env, so
   jinja2 reports it undeclared, `accept_vars` keeps it, and the request value
   shadows the global at render exactly as it does here. It is therefore the one
   builtin name this filter binds rather than skips, and binding it is the
   mirror in both directions: a template that never calls it renders identically
   on both engines, and one that does fails on both.

   **The one thing SKIP still cannot express, and why it is the right side of
   the trade.** The count above is measured over the committed fixture, and the
   three-way split is not a property of every template. jinja2's filter and test
   namespaces are separate from its variable namespace, so a template MAY use
   `{{ items }}` as an ordinary variable while `| items` still resolves as a
   filter; `find_undeclared_variables` reports that `items`, `accept_vars` keeps
   the kwarg, and upstream binds it. minja has one namespace and cannot hold
   both meanings at once, so the adapter has to choose, and it chooses the
   engine: a dropped kwarg renders a template that works, while the other choice
   renders HTTP 500 for a template that uses the filter. The residual is
   one-sided and named under `## Owed`. It is not reachable from any chat
   template of any checkpoint in `docs/USAGE.md` section 2a, none of which binds
   a variable named after a Jinja built-in.

   **What this still does NOT reproduce, and what that costs.** Upstream's
   filter is `accept_vars = fn_kw | template_vars | hf_base_params -
   {chat_template, tokenize}`, where `template_vars` is
   `jinja2.meta.find_undeclared_variables(chat_template)`. minja exposes no AST
   walk, so there is no `find_undeclared_variables` to port without forking the
   engine. What remains after F1 is one direction only: a name that is neither
   renderer-owned nor an engine builtin is bound here and dropped upstream, and
   "bound but never read" and "dropped" render the same bytes, so no template
   can tell them apart. The two residuals that survive that argument are named
   under `## Owed`.

3. **The `ChatPromptFn` seam** gains the same object as a fourth parameter, so a
   per-request value can reach the renderer at all; `MakeChatTemplatePromptFn`
   merges the server default under the request kwargs, mirroring `merge_kwargs`
   (`vllm/renderers/params.py:28-40`) as reached through
   `ChatParams.with_defaults` (`params.py:93-122`) from
   `vllm/entrypoints/openai/chat_completion/serving.py:208`:

   ```python
   defaults | {k: v for k, v in overrides.items() if v not in unset_values}
   #                                     unset_values = (None, "auto")
   ```

   The request's keys win, **except** that an override valued `null` or `"auto"`
   means "the client did not set this" and leaves the server default standing.
   The first implementation overwrote unconditionally and cited
   `multimodal/media/base.py:53-67`, which is `MediaIO.merge_kwargs` -- the
   media-io path, not this one. The consequence was measurable: with
   `--no-enable-thinking`, a request sending
   `{"chat_template_kwargs":{"enable_thinking":null}}` rendered thinking ON here
   and OFF on vLLM, defeating the operator's server-wide default on the very
   field this row adds.

4. **`ChatCompletionRequest::chat_template_kwargs`** is parsed and handed to
   `prompt_fn_` by `OpenAIServingChat::create_chat_completion`;
   `server_main.cpp` turns `--enable-thinking` / `--no-enable-thinking` into a
   tri-state that leaves the key absent when neither flag is given.

---

## 5. Tests

The gate is `tests/vllm/entrypoints/openai/test_api_server.cpp`, driving
`ApiServer::handle_chat_completions` — the production dispatch — over a prompt
function built by the production `MakeChatTemplatePromptFn` on the real
published Qwen3.8 template, committed as
`tests/fixtures/qwen38_chat_template.jinja`.

A unit test on the renderer would have missed this defect exactly the way every
existing gate did, so the entry point is the point. The synthetic engine behind
the dispatch carries a 22-token fixture vocabulary that cannot encode Qwen text,
so the harness's prompt function records the rendered string and hands the
engine an in-vocabulary one; the render itself, its failure mode, and the HTTP
status are all production. That substitution is the one adaptation, and it is
named here rather than left to be discovered.

Three cases:

- the real template renders through the dispatch: **200**, not 500 — red before
  the minja change with the issue's own message;
- no `chat_template_kwargs` leaves `enable_thinking` undefined, so the rendered
  prompt carries the reasoning branch (upstream's default);
- `{"chat_template_kwargs":{"enable_thinking":false}}` reaches the renderer and
  removes it.

A fourth case, added by the fresh review's repair, is the forgery probe: a
request whose `chat_template_kwargs` tries to replace `messages` must not change
the rendered prompt. It renders a benign request first so that "unchanged" is a
comparison and not an empty string, then drives the forged body, `tools`,
`chat_template` and `tokenize` through the same dispatch and requires a non-200
with the benign prompt still standing; `add_generation_prompt` is driven through
the same dispatch and required to render 200 with the assistant header, because
upstream ignores rather than refuses it.

The **C ABI** is gated too, in `tests/capi/test_capi.cpp`, because `vllm_chat`
parses the same request with `ParseChatRequest` and calls the same
`create_chat_completion`, and `vllm_c.cpp` installs
`vllm::capi::ResolveTemplatePromptFn` as its prompt seam -- so this row changed
the ABI's chat default (an unsupplied kwarg is now Jinja-undefined) and gave the
ABI a `chat_template_kwargs` field. One case drives all three through
`vllm_chat` on the same published Qwen3.8 fixture: the bare request renders the
checkpoint's own reasoning branch, `{"enable_thinking":false}` removes it, and a
forged `messages` returns a non-`VLLM_OK` status with the rendered prompt
unchanged. `tests/CMakeLists.txt` hands `test_capi` the fixtures directory for
it.

Plus, in `tests/vllm/entrypoints/test_chat_template.cpp`, one case per added
built-in test, one asserting that an unknown test name still throws so the
closed list stays closed, one for each arm of the kwargs filter table in section
4, and one for the `unset_values` rule (`null` and `"auto"` leave the server
default standing; `false` and `""` do not).

---

## 6. Gates

| Gate | Command | Result |
|---|---|---|
| focused | `ctest --test-dir build -R 'test_chat_template\|test_openai_api_server'` | see `## Now` |
| full | `scripts/agent-preflight.sh --staged` | see `## Now` |

Known pre-existing reds, not this row's:
[`#618`](https://github.com/mudler/vllm.cpp/issues/618)
`test_cpu_x86_llamacpp_floor`,
[`#1602`](https://github.com/mudler/vllm.cpp/issues/1602) `test_runner`, and the
repo-wide `windows-msvc-*` red of
[`#1649`](https://github.com/mudler/vllm.cpp/issues/1649).

---

## 7. Risks

- **The default flip is user-visible.** A deployment that relied on
  `enable_thinking` being implicitly false now gets the template's own default.
  Mitigated by `--no-enable-thinking`, which still forces it off, and named in
  the pull request body.
- **`escaped` returning a constant.** Correct for an engine with no Markup type
  and no autoescape, but it is a value rather than a refusal; a future minja
  that gains autoescape must revisit it.
- **The fixture is a copy of a published file.** It can drift from the
  checkpoint. It is pinned by repo and revision in the test's header comment and
  in section 2a, and by content here: `tests/fixtures/qwen38_chat_template.jinja`
  is 8,952 bytes, sha256
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`, which is
  the `chat_template` value of `tokenizer_config.json` AND the whole of
  `chat_template.jinja` at `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @
  `36f717a22990e82c54c1d48ee77c491b87825680` -- the two are byte-identical, which
  is also what rules out the standalone-file discovery path as the cause.

---

## Owed

- The nine arity-1 Jinja2 built-in tests (`divisibleby`, `eq`/`equalto`, `ne`,
  `lt`/`lessthan`, `le`, `gt`/`greaterthan`, `ge`, `in`, `sameas`), the two
  registry tests (`filter`, `test`), and `callable` -- twelve of the thirty
  canonical names. None is used by any
  checkpoint in `docs/USAGE.md` (section 2a), and section 2 says what each one
  needs first. Tracked by
  [#1681](https://github.com/mudler/vllm.cpp/issues/1681).
- vLLM's `--default-chat-template-kwargs <json>` server flag itself. This row
  mirrors the request side and keeps our `--enable-thinking` spelling; the
  general server-side JSON flag is not added. Tracked by
  [#1681](https://github.com/mudler/vllm.cpp/issues/1681).
- Reporting the missing tests to `google/minja` upstream, whose `main` has the
  same gap.
- **`jinja2.meta.find_undeclared_variables` is not ported**, so the request
  kwargs are filtered by refusing the four names the adapter supplies and
  skipping the ones the engine supplies, rather than by reproducing upstream's
  `accept_vars` set (section 4). Two one-sided residuals are left. A name that
  collides with nothing is bound here and dropped upstream, and no template can
  tell "bound but never read" from "dropped". And a template that uses a Jinja
  built-in's NAME as an ordinary variable gets the request's value upstream --
  jinja2's filter and test namespaces are separate from its variable namespace,
  so `find_undeclared_variables` reports it -- while minja has one namespace and
  the adapter keeps the built-in, because the alternative answers HTTP 500 for
  every template that uses the filter. Neither is reachable from any checkpoint
  in section 2a. Both close the same way, if minja gains an AST walk and
  separate filter and test registries. Tracked by
  [#1681](https://github.com/mudler/vllm.cpp/issues/1681).
- **`strftime_now` is set AFTER the request kwargs, so ours wins where
  upstream's request value would.** It is transformers' second post-parse global
  and therefore in `raise_exception`'s class, not the engine builtins' one: a
  template that names it has it in `find_undeclared_variables`, upstream keeps
  the request value, and the render fails on the shadow. Here the adapter's own
  callable overwrites the request key instead, so the request is silently
  ignored. The divergence is one-sided and strictly the safer side, and moving
  the `set` earlier would change behaviour no gate can observe: no fixture names
  `strftime_now`, so the only test that could pin it would be a template written
  to prove the change. Tracked by
  [#1681](https://github.com/mudler/vllm.cpp/issues/1681).
- **`ChatParams.with_defaults` returns `self` unchanged when every server
  default is falsy** (`vllm/renderers/params.py:93-104`), so on the
  now-default configuration upstream never reaches `merge_kwargs` at all and a
  request `{"enable_thinking": null}` arrives at the template as a bound `None`
  where `MakeChatTemplatePromptFn` drops it. minja has no distinct Undefined
  type, so a bound null and an unbound name are the same value to every is-test
  it can run (section 2), and the difference collapses into the already-recorded
  `defined`/null divergence. Not separately observable, recorded so the next
  reader does not re-derive it. Tracked by
  [#1681](https://github.com/mudler/vllm.cpp/issues/1681).
- **The C ABI's own `ResolveTemplatePromptFn` install is not gated.**
  `src/capi/vllm_c.cpp:312` installs the production prompt function, and
  `tests/capi/test_capi.cpp:303` drives `vllm_chat` through a capturing wrapper
  it builds around its own `ResolveTemplatePromptFn` call, so the resolver is
  gated while the ABI's install of it is reached by the shipped library and by
  nothing in `ctest`. Same residual shape as
  `server_main`'s `DefaultChatTemplateKwargs` call below, and it closes the same
  way: one lease with a real checkpoint. Tracked by
  [#1681](https://github.com/mudler/vllm.cpp/issues/1681).
- **The multimodal chat path drops `chat_template_kwargs`.** The mm seam is
  `(messages) -> MultiModalInputs` (`chat_mm.h` `MultiModalChatFn`), so there is
  nothing to carry them on and `chat_mm.cpp` renders with an empty object. A
  multimodal request that sets `enable_thinking` is therefore ignored where a
  text-only one is honoured. Widening that seam is a change to the mm chat
  contract, not to this one. Tracked by
  [#1681](https://github.com/mudler/vllm.cpp/issues/1681).
- **`server_main`'s one call to `DefaultChatTemplateKwargs` is not gated.** The
  server resolves its chat template only after a real tokenizer loads, and no
  CPU gate has a checkpoint, so `test_serve_recipe_args.cpp`'s re-exec harness
  aborts before that line. The RULE is gated
  (`chat_template: DefaultChatTemplateKwargs keeps unset apart from explicitly
  false`); the one line that calls it is reached by the shipped server and by
  nothing in `ctest`. Same residual shape as
  [`music3-dit-arm-reachability.md`](music3-dit-arm-reachability.md), and it
  closes the same way: one lease with a real checkpoint.

---

## Now

The owning matrix row stays `ANCHOR-BACKFILL`. This row repairs a defect inside
it and moves no lifecycle state, so it owes no `## Now` move and no
`docs/BENCHMARKS.md` edit (`docs/STATUS.md` was retired on `main` in `1db7e59cf`
while this branch was open). `docs/reference/server.md` changes because the
request field, the refused keys and the `--enable-thinking` default are all
user-visible; `docs/USAGE.md` changes because the C ABI's chat default changed
with this row and `vllm_chat` gained the field.
