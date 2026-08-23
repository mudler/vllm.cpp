# Vendored third-party

| Library | Version | Source | License |
|---|---|---|---|
| doctest | v2.5.2 | github.com/doctest/doctest | MIT |
| nlohmann/json | v3.12.0 | github.com/nlohmann/json | MIT |
| cpp-httplib | v0.49.0 | github.com/yhirose/cpp-httplib | MIT |
| minja (`minja/minja.hpp`, `minja/chat-template.hpp`, `minja/LICENSE`) | commit `021c229` (2026) | github.com/google/minja | MIT |
| Vulkan-Headers (`vulkan_core.h`, `vk_platform.h`) | vulkan-sdk-1.4.328.1 (`VK_HEADER_VERSION` 328) | github.com/KhronosGroup/Vulkan-Headers | Apache-2.0 |
| BLAKE3 (`c/`: `blake3.{h,c}`, `blake3_impl.h`, `blake3_dispatch.c`, `blake3_portable.c`) | 1.5.5 (commit `81f772a`) | github.com/BLAKE3-team/BLAKE3 | CC0-1.0 OR Apache-2.0 |

Update procedure: re-download the pinned header(s)/source at a newer tag, update
this table, note it in .agents/parity-ledger.md. **minja carries local edits**
(below), so re-downloading it means re-applying them, not overwriting.

## Local edits to minja

Every one is marked in `minja/minja.hpp` with a comment naming this project, so
`grep -n 'vllm.cpp' third_party/minja/minja.hpp` finds them all. None is a fork:
each is a gap in upstream that a real published chat template walks into.

| Where | What | Why |
|---|---|---|
| `TemplateNode` text assembly, the `lstrip_blocks` branch | strip the line-leading whitespace before a BLOCK tag only, never before an expression tag (upstream stripped before both) | transformers renders with Jinja2's `lstrip_blocks=True`, and byte-exact prompt parity needs the same whitespace. See `include/vllm/entrypoints/chat_template.h`. |
| `MacroNode::do_render`, the callable capture | capture the macro's context WEAKLY | the strong capture was an ownership cycle: the context owns the callable, the callable owned the context, and ASan reported the leak once the sanitizer lane could reach it (`59674cf1d`) |
| `BinaryOpExpr::do_evaluate`, the `is`-test table | add the arity-0 Jinja2 built-in tests upstream lacks and can reach: `undefined`, `even`, `odd`, `lower`, `upper`, `escaped` | upstream minja implements twelve of Jinja2's tests and throws on the rest. `undefined` is the standard idiom for "was this variable supplied?", the Qwen3.8 family's own template uses it, and without it every chat request against that family answered HTTP 500 ([#1681](https://github.com/mudler/vllm.cpp/issues/1681)). Measured 2026-08-22: `google/minja` `main` still has the same gap, so there is no revision to advance onto. The arity-1 tests are deliberately NOT added -- `parseLogicalCompare` reads the right side of `is` with `parseIdentifier()`, so adding them is a grammar change, and no chat template of any checkpoint in `docs/USAGE.md` uses one. Neither are `filter`/`test` (they need a name registry minja does not have) nor `callable` (`do_evaluate` defers every binary op with a callable left operand, so the test can never be handed one). `even`/`odd` answer over Python's numeric tower, not C++'s: a bool counts as an int, a non-integral float is neither, and the remainder is folded to Python's floored `%` before the comparison, because `get<int64_t>()` alone called `4.5 is even` true and threw on `True is even`. |

Everything here is header-only EXCEPT **BLAKE3**, which is the one vendored
COMPILED dependency. It exists because an LMCache C++ client must key cache
chunks on LMCache's OWN blake3 rolling token hash
(`lmcache/v1/multiprocess/token_hasher.py`, `KV-EXTERNAL-CACHE` W1) — a bit-exact
requirement (one wrong bit = zero cache hits) that is the external cache's own
hash, not something mirrorable from a vLLM dependency. The official BLAKE3-team C
reference implementation is the cleanest mirror; the three `.c` files are kept
UNMODIFIED and compiled as a SEPARATE `blake3_vendored` static library (see the
top-level `CMakeLists.txt`) with the portable backend forced
(`BLAKE3_NO_AVX512/AVX2/SSE41/SSE2`, `BLAKE3_USE_NEON=0`), so they stay off the
project's `-Werror` path and the digest is byte-identical on x86-64 and aarch64.
Provenance and rationale are recorded in `.agents/porting-inventory.md §9`.

**Vulkan-Headers is the one entry that is two headers rather than one**
(`vulkan_core.h` includes `vk_platform.h`), and it is worth stating why it is
vendored at all rather than found with `find_package(Vulkan)`: neither of our
boxes has the Vulkan development package (no `/usr/include/vulkan`, no
`libvulkan.so` link name, no sudo — measured 2026-07-22), only the runtime
loader. These are the Khronos-generated API TYPE declarations and nothing else:
they are compiled with `VK_NO_PROTOTYPES`, so no symbol is imported from them and
nothing is linked; the entry points are resolved with `dlopen`/`vkGetInstanceProcAddr`
at run time (`src/vt/vulkan/vulkan_loader.cpp`). Hand-declaring the subset we use
was the alternative and was rejected: a single wrong struct field would be silent
memory corruption rather than a compile error. The pin matches the loader version
on `dgx.casa` (1.4.328). Version bumps are a mechanical re-download; the backend
uses only Vulkan 1.1 core.
