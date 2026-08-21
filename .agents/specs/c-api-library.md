# Stable C API / `libvllm` contract

**Row:** `SERVE-C-ABI`  
**Claim:** `CLAIM-SERVE-C-ABI-SPIKE`  
**Scope date/base:** 2026-07-31, `a10bd428`  
**Lifecycle:** accepted spike for anchor backfill; implementation is already
present, but closure requires the focused follow-on rows below.

## Scope

This spec defines the stable, flat C ABI exported by `libvllm`: engine and
request handles, model/sampling parameter structs, blocking/streaming/chat
entry points, callback/error semantics, ownership, versioning, symbol
visibility, and runtime loading. It also defines how future ABI growth is made
without silently breaking older FFI consumers.

In scope are `include/vllm.h`, `src/capi/vllm_c.cpp`, the shared/static library
packaging, `dlopen` consumption, the public C documentation, and the C-ABI
translation of already-supported engine/request options. Model math, scheduler
policy, sampling algorithms, protocol semantics, HTTP transport, and new
feature implementation are out of scope: this layer only translates and owns
their C-facing lifetime.

The current surface is ABI v10 with 19 `VLLM_API` symbols. v10 appends the
default-inert `enable_jump_forward` tri-state to `vllm_model_params`; it does not
add a symbol. The two stale public v9 labels found during this spike are a
documentation bug, not an ABI change.

## Upstream chain and recorded deviation

Pinned vLLM `555967922` has no C ABI or stable shared-library export contract.
Its corresponding behavior sources remain the semantic truth beneath this
adapter:

- synchronous/offline generation: `vllm/entrypoints/llm.py:66,422`;
- asynchronous request lifecycle: `vllm/v1/engine/async_llm.py:70,280,524,637,709`;
- chat/completion shaping: `vllm/entrypoints/openai/chat_completion/serving.py`
  and `vllm/entrypoints/openai/completion/serving.py`;
- sampling contract: `vllm/sampling_params.py`;
- structured output, parser, speculative decode, prefix-cache, scheduler, and
  KV-connector behavior are owned by their dedicated engine rows and merely
  translated here.

The C ABI is therefore an intentional original packaging layer, already
recorded in `.agents/porting-inventory.md` section 9. Its handle/default/free
ergonomics follow llama.cpp's public-C pattern, while all execution semantics
remain vLLM-derived. No future C-ABI change may reimplement engine policy.

There is no runtime dispatch or GPU kernel selection in this layer. A runtime
trace is not applicable to its correctness gate; downstream model/performance
rows own traces for the engine paths reached through it.

## Our baseline

### Public contract

- pure-C/C++ guard, version history, export macro, status codes, opaque handles,
  and ownership rules: `include/vllm.h:1-124`;
- model fields through ABI v10: `include/vllm.h:125-232`;
- logits callback and sampling parameters: `include/vllm.h:234-318`;
- 19 exported declarations: `include/vllm.h:329-475`;
- translation, validation, exception mapping, callback/request lifetime, and
  exported definitions: `src/capi/vllm_c.cpp:229-391,424-975`;
- shared/static packaging and export visibility: `CMakeLists.txt` plus
  `cmake/VerifyExports.cmake`.

### Existing evidence

- behavioral and lifetime suite: `tests/capi/test_capi.cpp:341-1227`;
- pure-C C11 compilation: `tests/capi/c_header_compile.c:1` and
  `tests/CMakeLists.txt:580-585`;
- runtime-load/export/error-path proof: `tests/capi/test_dlopen.cpp:69-132`;
- exact dynamic-export allowlist: `tests/CMakeLists.txt:602-608` and
  `cmake/VerifyExports.cmake`.

The current implementation is substantial, but the row remains
`ANCHOR-BACKFILL`: the spec was missing, the public ABI compatibility policy is
not mechanically checked across releases, `test_dlopen` resolves only 17 of the
19 symbols (it omits `vllm_chat` and `vllm_chat_stream`), and the documented
no-throw guarantee still needs an allocation-failure audit at every catch/error
boundary.

## ABI rules and dispatch behavior

1. The header remains valid C11 and C++. No STL, exceptions, references,
   overloaded names, compiler-specific layout types, or owned C++ objects cross
   the boundary.
2. Callers construct parameter structs with the matching `*_default()` helper.
   Additive fields append at the end and default to inert behavior. A layout or
   signature incompatibility requires an ABI-version bump and explicit migration
   notes; symbol removal is forbidden inside a supported major line.
3. Borrowed strings live for the documented call duration. Library allocations
   are released only by their paired `vllm_*_free` function. An engine outlives
   every request created from it.
4. No C++ exception crosses `extern "C"`. Argument errors map to
   `VLLM_ERR_INVALID_ARGUMENT`, construction/load errors to
   `VLLM_ERR_MODEL_LOAD`, execution errors to `VLLM_ERR_RUNTIME`, and unknown
   exceptions to `VLLM_ERR_UNKNOWN`. `vllm_last_error()` is thread-local.
5. Blocking, streaming, nonblocking, and chat calls reuse the same `LoadedEngine`
   / `AsyncLLM` / serving implementations. The C layer may validate and translate
   but must not select a different algorithm.
6. `libvllm.so` exports exactly the declared `vllm_*` ABI. Runtime consumers must
   be able to `dlopen` the library and `dlsym` every public symbol without linking
   C++ internals.
7. Callback ownership and cancellation are explicit: callback data is borrowed
   for the call; `false` requests early stop; callback failure/cancellation must
   leave the engine reusable; request free happens only after delivery stops.

## Port map

| Semantic source | Local adapter | Rule |
|---|---|---|
| vLLM `LLM` / `AsyncLLM` request lifecycle | `src/capi/vllm_c.cpp`, `include/vllm.h` | Thin handle/lifetime adapter only |
| vLLM sampling parameters | `vllm_sampling_params` translation in `vllm_c.cpp` | Validate then reuse engine sampling |
| vLLM chat/completion serving | `vllm_chat`, `vllm_chat_stream` | JSON in/out adapter; no duplicate parser logic |
| vLLM scheduler/cache/spec/structured-output configs | `vllm_model_params` / `vllm_sampling_params` fields | Dedicated row owns behavior; C ABI carries it |
| Original library packaging | CMake shared/static targets, version script, `VerifyExports.cmake` | Deliberate deviation because upstream has no C ABI |

## Tests to port and strengthen

Upstream has no C-ABI tests to port verbatim. The engine behavior behind each
field stays covered by its upstream-derived row; this row needs adapter and
packaging tests:

- retain `test_capi` coverage for defaults, validation/status mapping, blocking
  and streaming equality, seeded determinism, callback stop/failure, concurrent
  requests, cancellation, chat, parsers, constraints, connectors, and version;
- retain the C11 header compilation gate;
- extend `test_dlopen` to resolve all 19 declarations, including chat and chat
  streaming, so declaration/export drift cannot hide behind the prefix-only
  export checker;
- add compile-time/layout fixtures for every published ABI version that downstream
  compatibility policy promises to accept; older fixtures must be standalone C
  declarations, not aliases of the current header;
- add fault-injection around last-error construction and all `catch (...)`
  boundaries to prove the no-throw claim under allocation failure;
- add sanitizer stress for request/engine teardown and callback reentrancy.

No test is skipped in this spike because it changes records/docs only. The
follow-on compatibility and allocation-failure cases are explicitly unimplemented
and remain separate work rows rather than being represented as passing.

## Gates

### Spike/documentation checkpoint

- `python3 scripts/check-agent-record.py`
- `python3 scripts/check-doc-checkpoint.py`
- `python3 scripts/check-readme-structure.py`
- `python3 scripts/check-model-checklist.py`
- `python3 scripts/check-device-leakage.py`
- `python3 scripts/check-env-doc.py`
- CPU build of `vllm_capi_c_check`, `test_capi`, `vllm_shared`, and
  `test_dlopen`, followed by
  `ctest --test-dir build-cpu -R 'test_capi|test_dlopen|capi_shared_exports_only_abi' --output-on-failure`.

### Implementation closure

- correctness: all adapter suites above pass, including historical-layout and
  fault-injection additions;
- packaging: exact declared symbol set resolves through `dlopen` and no internal
  dynamic symbol leaks;
- concurrency/safety: ASan+UBSan and TSan focused suites pass, including engine
  teardown/callback races;
- e2e: at least one real CPU-loadable tiny model completes through a separately
  compiled C consumer and matches the same engine path's token IDs;
- performance/memory: `NOT APPLICABLE` for pure adapter/default-inert changes;
  any field that activates model behavior inherits that owning row's identical
  vLLM workload, every-axis, and memory gates;
- backends: header, validation, ownership, and packaging gates are CPU-complete;
  device-specific functionality is never promoted by this row.

## Dependencies

- `SERVE-ASYNC-LLM`, `TOOLS-STRUCTURED-CORE`, `TOOLS-CALLING-CORE`,
  `SAMPLE-REASONING`, `SPEC-MTP`, `KV-PREFIX-CACHE`, `KV-CONNECTORS`, and
  `ENG-SGLANG-BEHAVIOR-FLAG` own behavior exposed by current fields.
- C11 compiler, C++20 compiler, CMake, `dlopen`/`dlsym`, `nm`, sanitizer-capable
  toolchains; Windows needs an equivalent export/import gate rather than ELF
  version scripts.
- No model, network service, proprietary data, GPU, or new third-party license is
  required for this spike.

## Work breakdown

| Work | Files/ownership | Exit condition |
|---|---|---|
| W0 (this change) | spec, row, current-state docs/records | accepted spec; ABI v10 public labels fixed; CPU gates green |
| W1 export completeness | `tests/capi/test_dlopen.cpp`, export checker/tests | all 19 declarations individually resolved and exact-export set enforced |
| W2 compatibility policy | new C fixtures + ABI policy doc/checker | historical supported layouts compile and current additive-growth rules are mutation-tested |
| W3 no-throw hardening | `src/capi/vllm_c.cpp`, focused fault-injection tests | allocation failures cannot escape any entry point |
| W4 lifetime/concurrency | request implementation + sanitizer tests | teardown, cancellation, callback failure/reentrancy proven race/UAF-free |
| W5 real consumer gate | standalone C fixture + tiny CPU model | separately compiled consumer loads shared lib and matches engine tokens |

W1-W5 require new non-overlapping claims. W3/W4 may not proceed concurrently
because both touch `vllm_c.cpp` and request lifetime.

## Risks and decisions

- **Decision:** keep one small flat ABI and append-only default-inert growth;
  do not expose C++ classes or duplicate engine policy.
- **Risk:** `VLLM_ABI_VERSION` alone does not make a larger current struct safe
  when passed by an older binary. W2 must define the supported compatibility
  direction honestly; do not claim backward binary compatibility before that
  gate exists.
- **Risk:** prefix-based export checking can pass while a declared symbol is
  absent; W1 closes this by resolving every declaration.
- **Risk:** error reporting can allocate while already handling allocation
  failure. W3 must use a non-throwing fallback and test it.
- **Risk:** callback reentrancy and engine/request destruction can deadlock or
  race. W4 must state forbidden operations or support them explicitly, then gate
  the chosen contract.
- **Decision:** GPU throughput is not a C-ABI property. This row takes no speed
  credit and never substitutes CPU adapter tests for a feature row's device gate.

## Owed

| Issue | Stage | State |
|---|---|---|
| [#1535](https://github.com/mudler/vllm.cpp/issues/1535) | W1 (export completeness), Apple leg | **owed, and UNMEASURED rather than unfixed.** See `### The ELF export guarantee has no Apple equivalent` below for what is confirmed, what is not, and what would close it |

### The ELF export guarantee has no Apple equivalent

**Confirmed from the tree, not inferred.** `cmake/vllm_export.map` declares
`VLLM_ABI_1 { global: vllm_*; local: *; }` and `CMakeLists.txt` applies it to
`vllm_shared` as `LINKER:--version-script,...` inside `if(UNIX AND NOT APPLE)`.
On ELF that makes every non-`vllm_*` symbol in the force-linked `vllm` archive
local, which is why `examples/video_studio/main.cpp` is safe: it links
`vllm::shared`, `vllm_shared` links `vllm` PRIVATE so the example inherits
neither `CPPHTTPLIB_OPENSSL_SUPPORT` nor `OpenSSL::SSL`, and it therefore
compiles the vendored `third_party/httplib/httplib.h` in the **no-TLS layout**
while the library holds the **TLS layout**. `CPPHTTPLIB_OPENSSL_SUPPORT` is a
whole-header switch over the layout of `httplib::Result` and
`httplib::ClientConnection`, so the two definitions are ODR-incompatible; the
version script is what keeps them from ever meeting.

**The version script is the ONLY mechanism, and a reader is likely to think
otherwise.** `vllm_shared` is created with `CXX_VISIBILITY_PRESET hidden` and
`VISIBILITY_INLINES_HIDDEN ON`, which looks like a second line of defence and is
not one: those properties add `-fvisibility=hidden` to the target's OWN sources,
and `vllm_shared`'s only source is the generated empty stub
(`vllm_shared_stub.cpp`). The content comes from the force-linked `vllm` archive,
which no `CMakeLists.txt` line compiles with hidden visibility — a repo-wide grep
for `VISIBILITY_PRESET` returns exactly those two lines. So on the Apple leg,
where the script is not applied, nothing localizes anything.

**MEASURED on the ELF leg, which is the closest positive control this row can
take.** One `Release`, `VLLM_CPP_CUDA=OFF` x86_64 build of this tree, one build
directory, `libvllm.so.0.0.3` linked twice from the same objects — once as the
build produces it, once with the `-Wl,--version-script` argument removed from the
recorded ninja link line and nothing else changed:

| link | dynamic defined symbols | non-`vllm_*` | `httplib` |
|---|---:|---:|---:|
| as built (version script) | 48 | **1** (the `VLLM_ABI_1` version node) | **0** |
| script removed | 9376 | **9329** | **734** |

The 734 break down as 505 `W`, 209 `V` and 20 `u` — weak text, weak data and
unique-global, i.e. exactly the coalescable class. `libvllm.a` holds 2008 weak
`httplib` definitions in total. So the version script is not belt-and-braces: it
is the whole of the guarantee, and its absence exports 9329 internals including
the `httplib` family. The Apple leg links without it.

**One narrowing the issue does not make, and it matters for whoever measures
this.** `httplib::Result` itself has NO out-of-line definition in either link —
it is header-inline and gets inlined into its callers, so `grep`ing the export
table for it finds nothing and that is not reassurance. What IS exported weak is
the family whose signatures embed the layout-dependent types:
`httplib::ClientImpl::handle_request`, `::write_request`, `::process_socket`,
`::shutdown_ssl` and their neighbours, all taking `httplib::Request&` /
`httplib::Response&`. A macOS measurement should look there, not for `Result`.

**What is UNMEASURED, stated as one question.** Whether ld64's Mach-O
weak-definition coalescing can bind the example's no-TLS `httplib::Result` /
`httplib::ClientConnection` code against the dylib's TLS-layout definitions, or
the reverse. If it can, the symptom is a silently corrupted response object with
a clean link and no diagnostic — the failure mode `a50c57d69` avoided on the
library side by putting the define on the target rather than per file.
`tests/CMakeLists.txt` gates `capi_shared_exports_only_abi` on the same
`UNIX AND NOT APPLE` condition, so no gate covers it either.

**"We have no macOS host" is the wrong summary, and this is the part the issue
does not say.** `.github/workflows/release.yml` runs `metal_arm64` and
`mlx_arm64` on `macos-15`. What that lane does NOT do is build either artifact
this question is about: `scripts/build-macos-release.sh` configures with
`-DVLLM_CPP_BUILD_EXAMPLES=ON` but then builds `--target server
test_metal_backend` only, and `server` links `vllm::vllm` (the static archive),
not `vllm::shared`. So on macOS `libvllm.dylib` is never linked, `video-studio`
is never compiled, and the two layouts have never been in the same process. The
trigger is `workflow_dispatch` or a `v*` tag, so no pull request reaches it.
The gap is a lane that builds the wrong two targets, not an absent machine.

**What would close it, in either direction.** (1) MEASURE: on `macos-15`, build
`vllm_shared` and `video-studio`, then read the dylib's exported weak definitions
(`nm -gU` filtered on `__ZN7httplib`) and the example's resolved bindings
(`nm -m`, `dyld_info -bind`), and record whether either side's `httplib` symbols
resolve across the image boundary. (2) REMOVE THE QUESTION: give `vllm_shared`
an `-exported_symbols_list` holding `_vllm_*` on the ld64 leg and extend
`capi_shared_exports_only_abi` to macOS, which gives the Apple build the same
by-construction guarantee ELF has. (2) is the smaller change and needs the same
host to prove it, because an export list that silently exports nothing would
also pass a link.

**Severity: low, and the ELF leg is not at risk.** Every CI lane and every
released Linux artifact carries the version script. `video-studio` is an example
rather than a shipped path. Found while fixing
[#1531](https://github.com/mudler/vllm.cpp/issues/1531); recorded rather than
closed because both routes above need a macOS host and this row has not taken
one.
