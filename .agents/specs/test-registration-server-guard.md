# TEST-REG-SERVER-GUARD — the test registration must agree with the link

Issue: [#1883](https://github.com/mudler/vllm.cpp/issues/1883)
Row: `TEST-REG-SERVER-GUARD`
Base: `1724be38e`

## 0. The established fact this spec starts from

`tests/CMakeLists.txt:269` registers `test_minimax_music3_e2e_real`
unconditionally. The translation unit that defines what it calls,
`src/vllm/entrypoints/openai/api_server.cpp`, is compiled into `libvllm` only
inside `if(VLLM_CPP_SERVER)` at `CMakeLists.txt:2446`. Configure therefore
succeeds on `-DVLLM_CPP_SERVER=OFF -DVLLM_CPP_BUILD_TESTS=ON` and the **link**
fails:

```
undefined reference to `vllm::entrypoints::openai::ApiServer::...'
```

The test source really does call it. `ApiServer` appears at
`tests/parity/test_minimax_music3_e2e_real.cpp:443` and `:781` as a constructed
object, and the suite drives `serve`, `bind_to_any_port`, `is_running` and
`stop` on it. The file `#include`s `vllm/entrypoints/openai/api_server.h` at
line 68.

Every other suite that reaches an `ApiServer` symbol is already registered
inside the guard: `test_openai_api_server` (:1733), `test_openai_conformance`
(:1750), `test_serve_*` (:1767-:1796), and the download suites (:1704-:1716).
The registration of this one target is the sole disagreement.

The observed consequence is on `main`: the `build-test-cpu-arm64-full` job
configures `-DVLLM_CPP_SERVER=OFF` and is red for exactly this, with 16
`undefined reference` lines, all attributed to
`test_minimax_music3_e2e_real.cpp`.

## 1. Scope

In:

1. Move the `test_minimax_music3_e2e_real` registration and the three
   properties that belong to it into the existing `if(VLLM_CPP_SERVER)` block.
2. Add a **structural** gate that fails when any `vllm_cpp_add_test` target
   registered outside `if(VLLM_CPP_SERVER)` reaches a header declared by a
   `VLLM_CPP_SERVER`-gated translation unit.

Out:

- Turning the server ON in the arm64 job. That job sets it OFF deliberately, to
  exercise the configuration; the defect is real for any user who configures
  OFF, and flipping the job hides it rather than fixing it.
- Weakening what the suite asserts, or excluding it from server-ON builds. It
  runs exactly as before wherever `ApiServer` exists.
- Rewriting the suite to avoid `ApiServer`. It is the production HTTP entry
  point the `/v1/audio/speech` cases exist to drive, and replacing it with a
  hand-built seam would make the suite measure a class rather than a
  capability.

## 2. Design

### 2.1 The move

`vllm_cpp_add_test(test_minimax_music3_e2e_real ...)` and its
`target_compile_definitions`, `target_include_directories` and
`set_tests_properties` move verbatim into the `if(VLLM_CPP_SERVER)` block that
already holds every other ApiServer-linking suite. Nothing else changes: the
same definitions, the same include directory, the same `RUN_SERIAL ON`.

### 2.2 The gate

The natural gate is the link itself, and it costs a full `libvllm` build. That
is honest but is not a preflight instrument. `scripts/check-test-registration.py`
already owns the "registration is not vacuous" contract, already configures a
`-DVLLM_CPP_SERVER=OFF` tree, and is already wired into preflight and CI. The
new check goes there, as `server_guard_errors`.

It is static, and it derives both sides rather than transcribing either:

1. Parse the top-level `CMakeLists.txt` with a CMake command scanner and
   collect every `target_sources` argument ending in `.cpp` that sits inside
   `if(VLLM_CPP_SERVER)`. Six today.
2. For each such source, find the header that declares it by basename, under
   `include/` or beside it in `src/`. Six today, `api_server.h` among them.
3. Build the repository's own `#include "..."` graph over `include/`, `src/`
   and `tests/`, and compute which include spellings **transitively** reach one
   of those headers. Transitively, because `downloader.h` includes `hf_hub.h`
   and a test that includes only the former still needs the latter's
   definitions.
4. Parse `tests/CMakeLists.txt` with the same scanner, and for every
   `vllm_cpp_add_test` target **outside** `if(VLLM_CPP_SERVER)`, check whether
   any of its sources includes a spelling in that set.

The scanner strips CMake `#` comments outside quotes, joins a command's
arguments across lines, and tracks `if`/`elseif`/`else`/`endif` nesting; an
`else()` clears the frame's gated flag, because the else branch of
`if(VLLM_CPP_SERVER)` is precisely the un-gated one. C and C++ comments are
stripped from a source before its includes are read, which is what keeps
`tests/parity/test_minimax_music3_depth_arm_real.cpp` — whose only mention of
`ApiServer` is prose at line 37 — out of the violation set.

### 2.3 What this check cannot see

Stated rather than implied, because a gate whose limit is unwritten gets quoted
as though it had none.

- The proxy is the **declaring header**, not the symbol. A test that declares a
  gated symbol by hand, or reaches one through a macro, is invisible to it.
- It reads `#include "..."` only. An angle-bracket include of a repository
  header would be missed; the tree uses quotes for its own headers throughout.
- Conversely it can fire on an include that no longer carries a call. That is
  still a defect worth naming — an unused include of a conditionally compiled
  header — and the fix is to drop the include or move the target, never to
  widen the check.
- It says nothing about whether a suite is *reached*. That is
  `.agents/reachability.md`'s question and this gate does not answer it.

## 3. RED first

Two reds, one honest and one fast, both captured before the fix.

1. **Link.** Configure `-DVLLM_CPP_SERVER=OFF -DVLLM_CPP_BUILD_TESTS=ON` and
   build the single target. Expected: `undefined reference` to `ApiServer`
   members and `ld returned 1 exit status`. Exit code captured directly, never
   from a pipeline tail.
2. **Structural.** `scripts/check-test-registration.py` names
   `test_minimax_music3_e2e_real` and no other target. One violation, not a
   floor: the sweep over all registered targets is the evidence that the failing
   set is exactly one.

Both go green after the move, with the same commands.

## 4. Tests

`tests/scripts/test_check_test_registration.py` gains mutation cases in the
existing fixed-manifest scheme, each of which must fail the new check:

- a gated-linking target registered outside the guard;
- the same target moved inside the guard, which must pass;
- a target reaching the gated header **transitively** through an intermediate
  header, which the direct-include form would miss;
- a target that names the gated symbol only in a comment, which must pass;
- an `else()` branch of `if(VLLM_CPP_SERVER)`, which is not gated;
- the gated `target_sources` line commented out, which must not silently empty
  the gated set.

`tests/scripts/check_test_registration_mutations.txt` and its pinned digest in
`scripts/check-test-registration.py` are updated in the same change, because the
manifest is a fixed inventory and a new case that is not in it is a case the
integrity check would reject.

## 5. Gates

```sh
# structural, fast
python3 scripts/check-test-registration.py
python3 tests/scripts/test_check_test_registration.py

# the link, slow and honest
cmake -S . -B build-off -DVLLM_CPP_SERVER=OFF -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_BUILD_EXAMPLES=OFF -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=OFF \
  -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF -DVLLM_CPP_MLX=OFF \
  -DVLLM_CPP_TRITON=OFF -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build-off --target test_minimax_music3_e2e_real -j 16

# the full tree gate
scripts/agent-preflight.sh --fail-on-skip
```

## 6. Wiring

No new lane. `check-test-registration` is already in the `CHECKERS` array of
`scripts/agent-preflight.sh` (:107) and `test_check_test_registration` in its
`SUITES` array (:157); `.github/workflows/ci.yml` runs both at :375-:376. The
new check therefore runs wherever the old one runs, on the first commit that
carries it. This is the reason it was added to that checker rather than to a new
script: a checker in no lane catches nothing (#680, #408, #1509).

## 7. Risks

- **R1 — the scanner is a second CMake parser.** It is, and it is deliberately
  small: it needs command names, argument lists and `if` nesting, not
  evaluation. It never decides truth of a condition other than the literal
  presence of `VLLM_CPP_SERVER`, so it cannot silently mis-evaluate a generator
  expression. The mutation cases pin the shapes it must get right.
- **R2 — a false positive blocks unrelated work.** Measured before landing: the
  sweep over every `vllm_cpp_add_test` target in the tree returns exactly one
  violation, and zero after the move.
- **R3 — the digest pin makes the manifest a shared file.** It already is, and
  it is a fixed inventory by design; the alternative that this protocol forbids
  is a per-PR counter. This change adds rows and re-pins the digest once.

## 8. Now

`ACTIVE`. The move and the gate land together, because the gate is what keeps
the move from being undone silently.

## 9. Outcome

Recorded on landing.
