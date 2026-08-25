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

`DONE`. The move and the gate landed together, because the gate is what keeps
the move from being undone silently.

## 9. Outcome

### 9.1 The reds, and that they were real

**The link**, at base `1724be38e`, on the exact configure of §5:

```
/usr/bin/ld: tests/CMakeFiles/test_minimax_music3_e2e_real.dir/parity/test_minimax_music3_e2e_real.cpp.o:
  in function `DOCTEST_ANON_FUNC_21()':
test_minimax_music3_e2e_real.cpp:(.text+0x11070): undefined reference to
  `vllm::entrypoints::openai::ApiServer::ApiServer(...)'
...
collect2: error: ld returned 1 exit status
ninja: build stopped: subcommand failed.
BUILD_RC=1
```

**16** `undefined reference` lines, which is the same count the
`build-test-cpu-arm64-full` job reports on `main`, over `ApiServer`'s
constructor, `bind_to_any_port`, `is_running`, `serve`, `stop` and destructor.
The exit code was captured directly and not from a pipeline tail.

**The structural check**, through the production entry point, before the move:

```
ERROR: tests/CMakeLists.txt:269: test_minimax_music3_e2e_real is registered
outside `if(VLLM_CPP_SERVER)`, but tests/parity/test_minimax_music3_e2e_real.cpp
reaches vllm/entrypoints/openai/api_server.h, declared by
src/vllm/entrypoints/openai/api_server.cpp, which is compiled only inside that
guard. A -DVLLM_CPP_SERVER=OFF build configures and then fails to link. Move the
registration inside the guard
CHECKER_RC=1
```

### 9.2 The failing set is exactly one, and it was swept for

Not asserted: measured. The check ran over **every** `vllm_cpp_add_test` target
in `tests/CMakeLists.txt` and returned one violation. The six gated translation
units resolve to six declaring headers, and the tests that reach them are
`test_openai_api_server`, `test_openai_conformance`, the five `test_serve_*`
suites, `test_hf_hub`, `test_downloader`, `test_model_resolver`,
`test_tls_transport` — all already inside the guard — and this one, which was
not.

The sibling `test_minimax_music3_depth_arm_real` names `ApiServer` at line 37
and is correctly clean: the mention is prose, and the check strips C and C++
comments before reading a source's includes.

### 9.3 The greens

- `-DVLLM_CPP_SERVER=OFF`, all 1139 test targets: clean.
- `-DVLLM_CPP_SERVER=ON`, the moved target: links, `ctest -N` still lists it,
  `RUN_SERIAL = True` survives, both `-D` definitions and the
  `tests/parity` include directory are on the compile line, and the binary runs
  and skips loudly for the absent checkpoint exactly as before. **The suite is
  not disabled anywhere it could previously run.**
- `python3 scripts/check-test-registration.py` → 0.
- `python3 tests/scripts/test_check_test_registration.py` → 68 tests, 0
  failures (61 before this change).

### 9.4 Both halves were mutated, and both moved

A green suite is not evidence that its cases can fail.

- Reverting `tests/CMakeLists.txt` to its unfixed content reds
  `test_server_guard_accepts_the_real_tree` and
  `test_shipped_tree_is_registered_and_wired`, suite exit 1. Tree restored
  byte-for-byte and verified with `diff -q`.
- Neutering `server_guard_errors` to `return []` reds M49 through M54, suite
  exit 1. Restored and verified the same way. `__pycache__` was purged on both
  sides and every run set `PYTHONDONTWRITEBYTECODE=1`, because a restored file
  can still execute a mutant's bytecode.

The integrity layer also caught the author: `test_M55` was first written with
`assertIn` and the suite refused it with `test_M55 ... has no semantic outcome
assertion`, which is the check doing its job on a new case rather than on a
fixture.

### 9.5 What was decided against

- **Turning the server ON in the arm64 job.** It sets OFF deliberately. The
  defect is real for any user who configures OFF, and flipping the job would
  have made a green wall out of an unfixed bug.
- **A checker in a new script.** It would have needed its own lane.
  `check-test-registration.py` was already in preflight's `CHECKERS` and in
  `ci.yml`, so the new check runs on the first commit that carries it.
- **Reproducing the link inside the gate.** It is the exact instrument and it
  costs a full `libvllm` build. Kept as the acceptance run in §5, not as a
  preflight check.
- **Matching symbols rather than headers.** It would need a C++ parser to be
  better than the header proxy, and worse than one to be cheap. §2.3 records
  what the proxy cannot see instead of implying it sees everything.

## 10. Found in flow: the evidence harness could not run the module it judges

Issue: [#1892](https://github.com/mudler/vllm.cpp/issues/1892). Filed and fixed
in this flow, because filing without fixing defers the fix.

Changing `scripts/check-test-registration.py` triggers `check-pr-size.py`'s
checker-change evidence contract, which reruns the checker's evidence module in
a sanitized environment: `PATH` is `os.defpath` plus a private tools directory
built from `EVIDENCE_REQUIRED_TOOLS`. That map named one module, and this one
was not it, although the module configures CMake in nearly every case, queries
CTest, and drives the `Ninja Multi-Config` generator in one case.

So the harness could not start the module. CI reported
`FileNotFoundError: [Errno 2] No such file or directory: 'cmake'` and
`FAILED (errors=26)`, every line charged to the checker under change rather
than to the harness. This is the broken-instrument shape: the instrument fails
toward a verdict about the code. It is the same gap [#458] closed for the
windows-portability module, and it stayed invisible because the harness only
runs when a checker **and** its evidence file change in one pull request.

**The first fix was incomplete, and that is the point of the second.** Declaring
`cmake` and `ninja` moved the failure to `ctest`, which is a separate binary
from `cmake` and need not sit on the same path. Adding a third name by hand
would have left the trap armed for a fourth. The test therefore derives its
expectation from the checker's own source: it parses the argument-list literals
for every program the checker executes, adds the generator the suite names, and
asserts each one resolves under `_sanitized_env` — from a mocked stand-in for
the system default path, or from the private tools directory. A checker that
starts calling a new binary now reds locally rather than in CI.

The derivation is non-vacuous by assertion: the parse must find `cmake` and
`ctest`, so a parse that silently matched nothing cannot pass. The system path
is mocked rather than read, so the result does not depend on where a host
installed cmake.

RED with the map entry reverted: three subtest failures naming `cmake`, `ctest`
and `ninja`, each as `executes X, which the sanitized environment cannot
reach`. Green after. `python3 scripts/check-pr-size.py` then exits 0 on this
branch, and the harness was proven live rather than skipped — with
`server_guard_errors` neutered the gate exits 1 on a real assertion failure
from inside the sanitized worktree, not on a missing tool.

## 11. A note on the gate that failed for the disk

One intermediate `scripts/agent-preflight.sh --fail-on-skip` run reported
`test_check_release_binary_contract` rather than a clean pass. The captured
output holds the reason: `FileNotFoundError: [Errno 2] No usable temporary
directory found`. That is
[#1353](https://github.com/mudler/vllm.cpp/issues/1353)'s recorded shape
exactly — a full disk making preflight return a verdict about records. The
suite passes standalone on this branch and on a clean `origin/main` worktree,
and freeing space and rerunning the identical command turned it green with no
tree change. Recorded here rather than filed again, because #1353 already owns
it.
