# Stop `--gpu-memory-utilization` from lying

Row: `FIX-GPU-MEM-UTIL-INERT`. Issue:
[#1165](https://github.com/mudler/vllm.cpp/issues/1165). Parity pin: vLLM
`555967922` (0.26.0.dev0).

## Scope

This row makes an already-recorded gap audible. It does not close the gap.

`--gpu-memory-utilization` is parsed, threaded, carried on the C application
binary interface (ABI), defaulted to 0.92, and then read by nothing. A user who
passes `--gpu-memory-utilization 0.85` believes they sized the key value (KV)
pool. They sized nothing, and the engine reports no difference from a run that
never set the flag.

### The boundary against `ROAD-V1-MEM` M3, and why it is not moved here

[#83](https://github.com/mudler/vllm.cpp/issues/83) owns making the flag work.
That is `ROAD-V1-MEM` M3, designed in [`kv-sizing.md`](kv-sizing.md) and
recorded on the roadmap at `.agents/roadmap_v1.md:71`. M3 needs a device profile
run that measures the non-KV footprint before a free-memory fraction can become
a block count, and its gate is our KV pool matching vLLM's own at a matched
`--gpu-memory-utilization`, which needs a working pinned oracle on `dgx.casa`.

M3 is therefore dgx-gated and cannot land on the CPU tier. **This row does not
close #83, does not implement the profile run, and does not stub one.** A later
reader who finds this spec must not read it as evidence that the utilization
path works. The fallback is still 256 blocks. The only change is that the engine
now says so.

## Anchors

| What | Where |
|---|---|
| Flag parsed | `src/vllm/entrypoints/openai/server_main.cpp:440-441` |
| Threaded to both engines | `src/vllm/entrypoints/openai/server_main.cpp:952,1039` |
| Command-line client | `examples/cli/main.cpp:118-119,209-212` |
| C ABI field | `include/vllm.h:474-486` |
| C ABI mapping | `src/capi/vllm_c.cpp:577-580` |
| Engine field and default | `include/vllm/entrypoints/model_loader.h:87-90` |
| The place it is discarded | `src/vllm/entrypoints/model_loader.cpp:954-959` |
| Upstream knob this mirrors | `vllm/config/cache.py:68` @ `555967922` |
| Upstream override precedence | `vllm/config/cache.py:189` @ `555967922` |

## Design

### Accept the flag. Never refuse it

`.agents/roadmap_v1.md:71` records the design intent that this engine keeps
vLLM's exact flag name and fraction semantics so an existing vLLM launch line
ports unchanged. Refusing the flag would break that intent and contradict the
mirror rule in `AGENTS.md`, `## vLLM is the reference`. The value stays
accepted, and its resolution stays exactly what it is today.

### Warn only when the caller chose the value

A notice on every start for a default nobody set is noise. A missing notice for
a value somebody deliberately chose is the defect this row exists to fix. So the
engine needs to tell an explicit ask apart from the built-in default, and a
plain `double` that defaults to 0.92 cannot.

`EngineParams::gpu_memory_utilization` becomes `std::optional<double>`, where
`nullopt` means unset and resolves to vLLM's 0.92. This mirrors the tri-state
`enable_prefix_caching` already in the same struct
(`include/vllm/entrypoints/model_loader.h:120`, resolved at
`src/vllm/entrypoints/model_loader.cpp:718-728`). Nothing in the tree reads the
field today, so the type change moves no behavior.

Each surface then says what it means:

- The server sets the optional only when `--gpu-memory-utilization` was parsed.
- `examples/cli` sets the ABI field only when the flag was parsed, and otherwise
  passes the ABI's documented unset spelling, `0.0`.
- The C ABI mapping is unchanged. `> 0.0` still means explicit.

### The C ABI asymmetry, stated rather than fixed here

`vllm_model_params_default()` pre-fills `gpu_memory_utilization` with `0.92`
(`src/capi/vllm_c.cpp:524`), so a C caller who never touched the field is
indistinguishable from one who typed `0.92`. Such a caller gets the notice.

That is left alone deliberately. Changing the value `vllm_model_params_default()`
returns is an observable change to the ABI surface, it is not needed for the
notice to be true, and a struct that carries `0.92` into an engine that ignores
it is a caller who benefits from hearing that. `include/vllm.h` gains one
sentence saying the notice fires and that `0.0` is the spelling that suppresses
it. The command-line path, which does have a real typed-or-not distinction, is
silent by default after this row.

### Where the notice fires, and how often

The notice fires inside `LoadedEngine::ResolveNumBlocks`, at step 3, on the line
that discards the value. That is the one seam every entry point reaches: the
server, `examples/cli` through the C ABI, and any `include/vllm.h` client all
build a `LoadedEngine`, whose constructor calls `MakeKVCacheResolved`, which
calls `ResolveNumBlocks`
(`src/vllm/entrypoints/model_loader.cpp:1081-1083,972`).

It fires once per engine load, not once per process. A server that loads a text
engine and an embedding engine reports twice, because there are two pools and
the value was discarded twice. A process-wide latch was rejected: it would make
the second engine silent, and it would make the test order dependent, which is
the shape recorded in `a-test-class-after-the-main-guard-never-runs` and
`the-state-was-not-the-one-you-believed`.

Steps 1 and 2 return before step 3, so the notice cannot fire when
`--num-blocks` or `--kv-cache-memory` sized the pool. That is correct and not
incidental. vLLM's `cache.py:189` ignores `gpu_memory_utilization` under
`kv_cache_memory_bytes`, so a caller who set both gets vLLM's exact semantics
and has nothing to be warned about.

### What the notice says

It names four things: that the value did not size the pool, the count that
resolved instead, the flags that do bind today, and the row and issue that own
the real fix.

```text
vllm.cpp: WARNING --gpu-memory-utilization 0.85 was accepted but did NOT size the KV cache.
vllm.cpp:   The profile run that turns a free-memory fraction into a block count is not
vllm.cpp:   implemented yet (ROAD-V1-MEM M3, https://github.com/mudler/vllm.cpp/issues/83).
vllm.cpp:   The pool fell back to 256 blocks. To size it today, pass
vllm.cpp:   --kv-cache-memory <bytes> for an absolute KV budget, or --num-blocks <n> for an
vllm.cpp:   exact block count.
```

### The GB10 hazard belongs in the document, not in the notice

`gpu_memory_utilization` reserves host random access memory (RAM) on GB10's
unified 119 GiB pool, and a 0.85-class value has hard-rebooted that box
(`.agents/roadmap_v1.md:71`, [`mtp-k-gt-1.md`](mtp-k-gt-1.md) around `:523-542`).
The hazard is real and load-bearing.

It is recorded in `docs/USAGE.md` and kept out of the runtime notice, for two
reasons. The flag is inert today, so it cannot reboot anything today, and a
hazard warning attached to a value nothing reads misdirects the reader from the
sentence that matters. Second, the notice disappears when M3 lands, which is
exactly when the hazard becomes real, so putting the hazard there would delete
it at the moment it starts to bite. No platform detection is added, per the
row's constraint.

## Records

`docs/USAGE.md` is owed, and `scripts/check-doc-checkpoint.py` demands it: this
change touches `src/vllm/entrypoints/`, `include/vllm/`, `include/vllm.h`, and
`examples/cli/`, all of which classify as `user_usage`.

The page does not document `--gpu-memory-utilization` or `--kv-cache-memory` at
all today. Its server-flag table lists `--num-blocks N | 256 | KV blocks`, which
is stale in its own right: `num_blocks` defaults to 0, meaning auto, and 256 is
the resolved fallback rather than the default value. All three entries are
corrected here, and the GB10 hazard lands beside the utilization row.

`.agents/feature-matrix.md:92` and `docs/FEATURES.md:365` were both checked and
are **not** edited. Both already record the exact state: the row is `PARTIAL`,
M1 and M2 landed, and `--gpu-memory-utilization` needs the dgx-gated M3 profile
run. Neither states anything this row makes false. Editing the matrix would pull
`docs/FEATURES.md` in behind it through the `feature_surface` class with nothing
new to say, which is the shape `check-doc-checkpoint.py` records at its
`registration_changes` docstring as the defect that blocked every branch under
[#1055](https://github.com/mudler/vllm.cpp/issues/1055).

## Tests

`tests/vllm/entrypoints/test_loaded_engine_dense.cpp`, already registered at
`tests/CMakeLists.txt:1208`. It builds a real `LoadedEngine` over synthetic
dense weights, which is the loader entry point, so the test enters the change
through production rather than by calling the resolver. `ResolveNumBlocks` is
private, so no test can reach it directly.

Four cases, using the `CerrRedirect` idiom from
`tests/vllm/v1/test_async_llm.cpp:130-144`:

1. An explicit `gpu_memory_utilization` prints the notice, and the notice names
   the value, the 256-block fallback, both binding flags, and issue 83.
2. An unset `gpu_memory_utilization` prints nothing.
3. An explicit `gpu_memory_utilization` beside `kv_cache_memory_bytes` prints
   nothing, because knob 2 sized the pool and vLLM ignores the fraction there.
4. An explicit `gpu_memory_utilization` beside a `num_blocks` override prints
   nothing, for the same reason at knob 1.

Case 2 is the one that fails if the notice is made unconditional, and cases 3
and 4 are the ones that fail if it is moved above the early returns.

## Gates

CPU only. No GPU lease, and no `rc` call.

- Focused: `ctest -R test_loaded_engine_dense`.
- Full: `scripts/agent-preflight.sh`.
- Reachability: delete the `ResolveNumBlocks` call in `MakeKVCacheResolved` in a
  scratch copy, rerun the focused gate, and require red. Restore byte for byte.

## Stop conditions

Stop and report `NEEDS_DECISION` if closing this needs the M3 profile run, if it
needs a GPU, or if the only way to make the notice fire is to refuse the flag.

## Owed

Nothing. `ROAD-V1-MEM` M3 still owns the utilization path itself, and this row
leaves that debt exactly where it was. Its issue is deliberately not linked in
this section: `owed_issues()` in `scripts/check-agent-record.py` reads every
issue number under a `## Owed` heading as a claim of ownership, and this row
does not own it.

## Now

Landed on `row/FIX-GPU-MEM-UTIL-INERT`. The next step for the utilization path
is `ROAD-V1-MEM` M3 on `dgx.casa`, unchanged by this row.
