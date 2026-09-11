ID: ISSUE-GH-308
Title: The test tree does not build on macOS: three portability defects, six failing targets
Row: BACKEND-METAL-MLX
State: OPEN
Kind: UNKNOWN
GitHub: 308
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-10
Updated: 2026-08-10
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Summary
>
> A full `cmake --build build` of the default configuration fails on macOS (Apple
> clang) at six targets, in three independent classes. None of them is a code
> defect on Linux; all three are places where the source relies on something the
> Darwin SDK spells differently, or on a warning that only fires off-Linux. A
> contributor on an Apple machine cannot reach a green tree at all — the first
> build stops on `test_kv_offload_fs`, and `-k` reveals five more behind it.
>
> **The build undercounts the first class, and that is the most important thing in
> this report.** `::getpid()` is called at **nine** sites across nine files, not
> the three that fail. Five of the nine have no `<unistd.h>` in their own
> translation unit; three of those five fail today, and the other two compile only
> because some project header happens to pull the declaration in for them. A
> transitive-only include is not a working include — it is a latent break that
> presents the day someone tidies the header that was carrying it, in a file that
> has nothing to do with this. Counting call sites rather than build failures is
> the difference between a fix and a fix that stays fixed.
>
> macOS is a supported target — `release.yml` builds `macos-arm64-metal` and
> `macos-arm64-metal-mlx` on `macos-15` — so this is not an unsupported platform
> drifting. The reason it is not caught is narrower: the macOS release lane runs
> `scripts/build-macos-release.sh`, which configures with
> `-DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_BUILD_EXAMPLES=ON` but then builds only
> `--target server test_metal_backend`. **No job anywhere builds the macOS test
> tree**, and every build lane in `ci.yml` (`build-test-cpu`,
> `build-test-cpu-arm64`, `sanitize-cpu`, ...) is `ubuntu-*`.
>
> ## Reproduce
>
> ```sh
> cmake -S . -B build
> cmake --build build -j -- -k
> ```
>
> On macOS 26.5.2 / AppleClang 21.0.0 (Apple Silicon), that produces exactly six
> failing targets. Reproduced at `97e01cb`; all three defects are unchanged at
> `688eea1`, and none of the affected files has been touched since:
>
> | class | targets | error |
> |---|---|---|
> | missing `<unistd.h>` | `test_kv_offload_fs`, `test_kv_offload_tiering`, `test_kv_offload_connector` | `error: no member named 'getpid' in the global namespace` |
> | `::`-qualified byte-order call | `test_lmcache_client`, `test_lmcache_connector` | `error: expected unqualified-id` |
> | dead code under `-Werror` | `vllm-cpu-kernel-bench` | `error: unused function 'A76Event' [-Werror,-Wunused-function]` |
>
> ## 1. `::getpid()` without `<unistd.h>` — 9 sites, 5 of them unsound
>
> ```
> tests/vllm/v1/test_kv_offload_fs.cpp:34:62: error: no member named 'getpid' in the global namespace
> ```
>
> Each `TempDir` helper calls `::getpid()` to name a temporary directory. On glibc
> the declaration arrives transitively through a C++ standard header; libc++ does
> not provide it, so the name is genuinely absent and the qualified lookup fails.
>
> Full inventory of `::getpid()` at `688eea1`, by whether the TU includes
> `<unistd.h>` itself:
>
> | file | direct `<unistd.h>` | status |
> |---|---|---|
> | `src/vllm/v1/kv_offload/fs_io.cpp` | yes | correct |
> | `tests/vllm/gguf_builder.h` | yes | correct |
> | `tests/vllm/models/test_kimi_linear_paged.cpp` | yes | correct |
> | `tests/vllm/models/test_minimax_h3_video_fold.cpp` | yes | correct |
> | `tests/vllm/v1/test_kv_offload_fs.cpp` | **no** | **fails on macOS** |
> | `tests/vllm/v1/test_kv_offload_tiering.cpp` | **no** | **fails on macOS** |
> | `tests/vllm/v1/test_kv_offload_connector.cpp` | **no** | **fails on macOS** |
> | `tests/capi/test_capi.cpp` | **no** | latent — builds via a transitive include |
> | `tests/vllm/test_gguf.cpp` | **no** | latent — via `gguf_builder.h` |
>
> The last two are the same defect that has not presented yet. They are worth
> repairing in the same change rather than leaving as a tripwire, and doing so is
> this tree's own convention rather than an imported style: across the 369 test
> `.cpp` files, only **4%** of (TU, standard header) pairs rely on a transitive
> include — `<cstring>` and `<thread>` are at 0%, `<vector>` at 1%. These five
> were the exceptions.
>
> (`src/vllm/entrypoints/openai/server_main.cpp` calls `getpid()` unqualified and
> includes `<unistd.h>` itself, so it is already correct and is left alone.)
>
> ## 2. `::htonl` / `::ntohs` against a Darwin macro
>
> ```
> tests/vllm/v1/kv_offload/lmcache/test_lmcache_client.cpp:55:30: error: expected unqualified-id
>    55 |     addr.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
>       |                              ^
> /…/MacOSX.sdk/usr/include/sys/_endian.h:137:25: note: expanded from macro 'htonl'
>   137 | #define htonl(x)        __DARWIN_OSSwapInt32(x)
> /…/MacOSX.sdk/usr/include/libkern/_OSByteOrder.h:70:5: note: expanded from macro '__DARWIN_OSSwapInt32'
>    70 |     (__builtin_constant_p(x) ? __DARWIN_OSSwapConstInt32(x) : _OSSwapInt32(x))
> ```
>
> POSIX permits these to be macros and the Darwin SDK takes that option: `htonl`
> expands to a parenthesised conditional expression, so `::htonl(x)` becomes `::`
> followed by `(`, which is not a qualified-id. Preprocessing happens before the
> `::` is ever considered, so no include can rescue the qualified spelling — the
> qualification itself is the defect, and no `#ifdef` is needed to repair it.
>
> **The portable spelling is already in this tree and already proven by your CI:**
> `tests/vllm/entrypoints/openai/test_api_server.cpp` performs the same
> bind/getsockname sequence with **unqualified** `htons`/`ntohs`
> (lines 404, 412, 1620, 1640), and that target is green. Only the two lmcache
> mock servers qualify, and they are the same copied block — the file comment on
> the second one says as much.
>
> For completeness, since the glibc `#ifdef __OPTIMIZE__` byte-order macros are
> sometimes assumed to cause the same problem: they do not. Measured on gcc 14 /
> glibc 2.41, `::htonl(x)` inside an anonymous namespace compiles at both `-O0`
> and `-O2`. This defect is Darwin-only, and the unqualified spelling is a no-op
> on Linux rather than a second fix.
>
> ## 3. `A76Event` unused off-Linux, under `-Werror`
>
> ```
> examples/cpu_kernel_bench/main.cpp:447:26: error: unused function 'A76Event' [-Werror,-Wunused-function]
> ```
>
> `A76Event` reads `/sys/bus/event_source/devices/armv8_cortex_a76/`, and its only
> call site is inside the `#if defined(__linux__)` half of `MakeCounterGroups`.
> The function definition itself sits *outside* that guard, so off-Linux it is
> compiled and unreferenced. The file already uses `#if defined(__linux__)` for
> exactly this purpose — the `PerfGroup` stub immediately above it, `SetAffinity`,
> `ThrottlingStatus` — and this one function was left outside.
>
> Note for anyone fixing it: `ParsePerfEvent` has exactly one caller and it is
> `A76Event`, so guarding or marking only `A76Event` moves the identical error up
> one function. The two are one Linux-only unit and belong on the same side of the
> guard.
>
> ## Suggested fix
>
> Minimal and platform-neutral, no `#ifdef __APPLE__` anywhere:
>
> 1. `#include <unistd.h>` in the five TUs that call `::getpid()` without it.
> 2. Drop the `::` from the four byte-order calls in the two lmcache tests.
> 3. Move `ParsePerfEvent` + `A76Event` inside the `#if defined(__linux__)` the
>    file already uses. Nothing deleted; the Linux path is unchanged.
>
> A PR implementing exactly this follows. Verified both ways: the **entire**
> default configuration builds on macOS afterwards (384 targets, 0 errors, 0
> warnings) — which also demonstrates there is no seventh failure hiding behind
> the six — and the same targets build and pass on Linux (gcc 14.4, glibc 2.41),
> where the A76 path is live.
>
> ## Relation to #199
>
> #199 ("macOS MLX build fails because warnings in MLX headers are treated as
> errors") is a different failure and is already closed: it is third-party MLX
> headers under the project's `-Werror`, in the opt-in `VLLM_CPP_MLX` preview
> build. These six are first-party sources in the **default** configuration
> (`cmake -S . -B build` with no options), and none involves a third-party header.
>
> ## Follow-up worth considering (not in the PR)
>
> Nothing prevents this class from recurring: no CI lane builds the macOS test
> tree, which is how three independent defects accumulated without anything going
> red. A `macos-15` job that configures the default CPU build and builds the tests
> would catch it at the commit that introduces it. Left out of the fix
> deliberately — it is a CI policy decision with runner-cost implications, not
> part of the repair.
>

## Resolution

-
