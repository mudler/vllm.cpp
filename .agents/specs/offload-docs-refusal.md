# The docs describe a weight offload that is accepted and inert; the engine refuses it

Row: `FIX-OFFLOAD-DOCS-957`
Issue: [#957](https://github.com/mudler/vllm.cpp/issues/957)
Introduced by: `4a183b731` (#887, `ENG-WEIGHT-OFFLOAD` W2c)
Baseline: `origin/main` @ `2daa3287f`

## 1. Scope

Correct `docs/WEIGHT-OFFLOAD.md` and `docs/USAGE.md` so they describe what the
engine does today: refuse an enabled weight offload at startup, on every
architecture, naming the architecture.

**Out of scope:** wiring a loader (that is `ENG-WEIGHT-OFFLOAD` W2c, issue
[#797](https://github.com/mudler/vllm.cpp/issues/797)), and changing any
refusal. This row moves no code. It also does not attempt to make
`documentation-checkpoint` green — §5 explains why nothing can.

## 2. What the documents said, and what the engine does

Both documents describe the behaviour `4a183b731` replaced.

`docs/WEIGHT-OFFLOAD.md`, the callout and the summary:

> A budget you set is accepted, reported, and does not free memory.

> So the honest summary is: you can configure weight offload, the engine tells
> you what it resolved, and your device memory does not change. The engine
> prints one line at startup saying so [...]

`docs/USAGE.md:1473`:

> **Accepted and inert today: no weight moves yet**

What actually happens, at `2daa3287f`:

`RefuseUnsupportedWeightOffload`
(`src/vllm/model_executor/weight_offloader.cpp:72-83`) throws
`std::invalid_argument` when offloading is enabled and the resolved
architecture's factory does not set `supports_weight_offload`. It is called
unconditionally from the load path
(`src/vllm/entrypoints/model_loader.cpp:1410-1414`), immediately after
`ModelRegistry::Resolve(config)` and **before any weight I/O** — so nothing is
read from disk before the operator is told.

`ModelFactory::supports_weight_offload` defaults to `false`
(`include/vllm/model_executor/models/model_registry.h:371`), and **no model in
the tree sets it to `true`**. Measured by grepping `src/` and `include/` for the
identifier: every hit is the declaration, the two guard parameters, or the
`model_loader.cpp` read. A deliberately misspelled needle returns 0, so the
search term is not the reason the set is empty. The tree asserts the same fact
independently — `tests/vllm/model_executor/test_weight_offloader.cpp:376-379`
counts the declaring models and carries the message "a model now declares
supports_weight_offload; update this".

So the practical statement is the one neither document made: **enabling weight
offload fails startup, for every model.** A reader following the old text would
add `--offload-config` expecting a no-op and get a server that does not start.

A second refusal from the same commit was also undocumented.
`VerifyWeightOffloadWasConsulted` (`weight_offloader.cpp:86-97`) throws after
load when a model declares support and then asks the offloader about zero
weights, and reports it as a defect in that loader rather than a configuration
error. The document's refusal list covered only config validation — malformed
document, unknown backend, wrong type, negative budget, `offload_num_in_group`
above `offload_group_size`, `offload_prefetch_step` below 1.

## 3. The change

`docs/WEIGHT-OFFLOAD.md`:

- the callout leads with the startup failure instead of "accepted, reported, and
  does not free memory";
- the *What works today* table gains a row for the model gate, so the table
  answers "can I run with this on?" rather than only "what is built?";
- the summary says you can write and validate a configuration and cannot yet run
  with one enabled;
- a new subsection under *What the engine refuses* carries the real message text
  and the post-load consistency check.

`docs/USAGE.md`: the `--offload-config` row says enabling it fails startup on
every model, and that a config which leaves offloading disabled still parses and
reports.

Both keep the existing point that on unified memory such as GB10 offload cannot
help regardless, because host and device share one pool.

`docs/STATUS.md` needed no change: it already lists weight offload under what is
owed and never claimed the inert behaviour.

## 4. Evidence

No code moves, so the gate is the accuracy of the text against the tree.

| Claim in the new text | How it was checked at `2daa3287f` |
|---|---|
| the refusal is on the load path, before weight I/O | read `model_loader.cpp:1403-1415`; the call sits between `ModelRegistry::Resolve` and `LoadShards` |
| it fires only when offloading is enabled | `weight_offloader.cpp:75` — `if (!config.is_offloading_enabled()) return;` |
| the quoted message is the real one | transcribed from the `throw` at `weight_offloader.cpp:77-83` |
| `Qwen3MoeForCausalLM` is a real architecture string | `REGISTER_VLLM_MODEL(qwen3_moe, "Qwen3MoeForCausalLM", ...)` at `qwen3_moe_registry.cpp:181` — an invented name in an error example would be indistinguishable from a real one to a reader |
| no model declares support | grep over `src/` and `include/`; all hits are declaration, parameters, or the loader read. Negative control: a misspelled needle returns 0 |
| the post-load check exists and is worded as a loader defect | `weight_offloader.cpp:86-97` |

## 5. This does not make `documentation-checkpoint` green, and nothing can

`scripts/check-doc-checkpoint.py` walks the range one commit at a time and
evaluates each commit's **own** paths against its **own** parent (`main()`, the
`--base`/`--head` branch, via `commit_paths(commit)`). No later commit can
satisfy an earlier one, and there is no waiver or exemption mechanism — the file
says at `:256` that exempting named paths was deliberately avoided.

Measured rather than reasoned: `docs/FEATURES.md` and `docs/USAGE.md` have
changed in **66 commits** since `73d217db` landed, and
`check-doc-checkpoint.py --commit 73d217db` still reports ERROR.

The 28 flagged commits are immutable on `main`, which is never rewritten. So the
backlog is closed only by advancing the base or by a checker change, both of
which are decisions this row does not take.

## 6. What the other 27 commits owed, which was nothing

The audit that found this defect covered all 28. Recorded so the next person
does not repeat it.

| Group | Count | Verdict |
|---|---|---|
| `MODEL-MM-indextts2` (#634) | 19 | **Already documented.** Intermediate waves — CAMPPlus, w2v-bert, EnhancedCodec, S2Mel DiT primitives. The campaign documented itself at `f374ab8ed` once the pipeline rendered end to end, and every flagged commit is an ancestor of it. `docs/FEATURES.md:200` records the partial state honestly, defects included |
| `ENG-EXPERT-STREAM` (#913, #916, #918) | 3 | **Must NOT be documented.** Deliberately unwired staged slices; #918's own body says "Nothing wired to a loader; the engine is unchanged." A `docs/FEATURES.md` row would claim a capability no user can reach |
| `ENG-WEIGHT-OFFLOAD` (#843, #877, #879, #884) | 4 | **Already documented** by `62406c30e`, which postdates them |
| `ENG-WEIGHT-OFFLOAD` (#887) | 1 | **This row.** The only one that postdates its campaign's docs commit, and the only one where the docs are wrong rather than silent |
| `LTX25-PROMPT-ADALN` (`fba312c67`) | 1 | **Nothing owed.** A record/measurement commit — specs, goldens, a measurement script, tests. Ships no user-facing surface |

The pattern worth keeping: a staged campaign that documents itself once, at the
point the capability becomes reachable, is the *correct* shape. The gate flags
every wave that is not that commit, so a large flagged count is expected and is
not by itself evidence of stale documentation. Only a commit that changes
user-visible behaviour **after** its campaign's docs landed can make a document
wrong, and exactly one of the 28 did.

## 7. Now

`main` is red on `documentation-checkpoint` for 28 commits and stays red after
this row. What changes is that the two documents stop describing a behaviour the
engine has not had since `4a183b731`.

## Owed

Nothing by this row. The wiring that would make an enabled offload succeed is
[#797](https://github.com/mudler/vllm.cpp/issues/797) under
`ENG-WEIGHT-OFFLOAD`; when a loader is wired, the *What works today* table row
and the refusal subsection added here both have to move with it.
