# ORACLE-LLAMACPP-GATEABLE: the llama.cpp pin builds and runs, or it says why not

Issue: [#857](https://github.com/mudler/vllm.cpp/issues/857).
Row: `ORACLE-LLAMACPP-GATEABLE`.
Predecessor: `ORACLE-LLAMACPP-REPIN-STOCK`, in
[`oracle-llamacpp-repin-stock.md`](oracle-llamacpp-repin-stock.md), which moved
the record from the local fork `237ad9b96` to stock `b10451` and left this half
open.

**Secondary oracle:** `llama-cpp`

## Scope

`.agents/oracles/llama-cpp.md` records `gateable = no` and names #857 as the
issue that owes the measurement. `AGENTS.md` §"Measure gateability" admits
`gateable = yes` only after an oracle "demonstrably builds and runs the model.
Constructing a config proves nothing." This row runs that measurement and
records one of two results:

- stock llama.cpp at the pin builds and generates tokens from the recorded
  Qwen3.8-27B Q4_K_M artifact, so `gateable` becomes `yes` and `evidence`
  becomes a path in this tree; or
- it does not, and the record keeps `gateable = no` with the exact reason and
  the condition that would settle it.

In scope:

1. Confirm that the recorded object and the recorded label describe the same
   reproducible thing, on the remote and not only in a developer's clone.
2. Build stock llama.cpp at `10bf611e533d81f739128304991c5e133c6aebd8` on a
   leased fleet device, from a source tree whose identity the job asserts.
3. Run the recorded GGUF on the CPU and capture the generated text.
4. Record the build recipe, the run recipe, the asserted identities, and the
   binary sha256 in an evidence file.
5. Set `gateable` and `evidence` in `.agents/oracles/llama-cpp.md` to whatever
   the run measured.

Out of scope, and deliberately so:

- The quant-matched llama.cpp versus vllm.cpp comparison. That is `#821` W3.
- The thirteen contaminated measurements #1003 owes a re-take of. Proving the
  oracle runs is the precondition for those re-takes, not one of them.
- Any speed or memory floor. A timing this row prints is a by-product of the
  proof that the model ran, and §"Gates" marks it non-binding.

## Upstream chain

| Item | Value |
|---|---|
| Upstream | `https://github.com/ggml-org/llama.cpp` |
| Pinned object | `10bf611e533d81f739128304991c5e133c6aebd8` |
| Pinned label | `b10451` |
| Subject | `llama : check LoRA tensor data is within file bounds (#27056)` |
| Author date | 2026-08-16T02:38:01-04:00 |
| Binary that runs the model | `tools/completion` -> `llama-completion` |
| Architecture the artifact declares | `qwen35`, registered at `src/llama-arch.cpp:41` at the pin |

Two facts about the pin's own tree decide the build recipe, and both were read
at the pin rather than assumed from an older release:

- `llama-cli` moved to `tools/cli` and its `add_subdirectory` sits inside
  `if (LLAMA_BUILD_SERVER)` in `tools/CMakeLists.txt`. It links
  `llama-server-impl`, sets `params.verbosity = LOG_LEVEL_ERROR`, and no longer
  calls `common_perf_print`. It is a chat client now.
- `llama-completion` in `tools/completion` is the one-shot completion binary.
  It links `llama-common` and `llama` only, it honours `-no-cnv`, and it calls
  `common_perf_print`. It is outside the server guard.

The row therefore builds `llama-completion` and `llama-bench` and leaves the
server, the web UI, the unified app, the examples and the tests unbuilt. A
recipe copied from a pre-`b10451` memory would either fail to produce a binary
or produce a chat client whose output a postcondition cannot read.

## Our baseline

`.agents/oracles/llama-cpp.md` at `08c81a892`:

```text
pin = 10bf611e533d81f739128304991c5e133c6aebd8
pin_label = b10451
pinned_on = 2026-08-16
gateable = no
evidence = #857
```

`scripts/check-oracle-pins.py:137-149` enforces the two directions that make
this row's edit non-cosmetic. `gateable = yes` refuses an `#N` evidence value
and requires a path that exists in the tree. `gateable = no` requires an `#N`.
So the flag and the evidence move together or the gate refuses the change.

Nothing in the tree has ever built this pin. Every recorded llama.cpp number
predates it, and the enumeration of what that invalidates lives in the
predecessor spec, not here.

## Port map

Nothing is ported. This row measures an upstream oracle and edits one record.

| Surface | Change |
|---|---|
| `.agents/oracles/llama-cpp.md` | `gateable`, `evidence`, and the prose that states the reason |
| `docs/bench-evidence/oracle-llamacpp-b10451-gateable-2026-08-22.md` | new: the measured identity, recipe, and output |
| `.agents/specs/oracle-llamacpp-gateable.md` | this file |

## Tests to port

None. llama.cpp is the subject of the measurement, not a source of behaviour to
mirror. `AGENTS.md` §"Port the upstream tests in the same change" binds a port,
and this row ports nothing.

## Gates

1. **Object and label agree, on the remote.** `git fetch --depth 1 origin
   <pin>` succeeds from `https://github.com/ggml-org/llama.cpp`,
   `git rev-parse HEAD` equals the pin, `git fetch` of `refs/tags/b10451`
   succeeds, and `git rev-parse b10451^{commit}` equals the pin. This is the
   check the retired `pin_label = b9892` would have failed, and it runs on the
   remote inside the job rather than in a developer's clone.
2. **The source tree is the pin and nothing else.** `git status --porcelain` is
   empty, asserted in the job and written to the evidence directory. The oracle
   file's preamble asks for exactly this.
3. **The artifact is the recorded one.** The GGUF header parses, the computed
   data end equals the file size, and the sha256 of the bytes the run reads
   equals `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
   A remote hash is not admissible here: `lfs.oid` is fabricated for gated
   repositories.
4. **The build produces a binary.** `configure_rc` and `build_rc` are 0, and
   `llama-completion` exists and is executable. The exit code alone does not
   settle it, because a missing wrapper makes a wrapped command silently not
   run, so the file test is the postcondition.
5. **The model generates.** `llama-completion` exits 0, its stdout is longer
   than the prompt, and the combined output contains `common_perf_print`. Exit
   0 with an empty stdout is a load that never decoded, and this gate reads the
   bytes rather than the status.
6. **Not gated, and named so it is not quoted:** any tokens-per-second or
   memory figure this row's logs contain. One repetition, no clock pinning, no
   contention control, a CIFS-staged artifact. It is not a floor, and #1003
   still owes every floor.
7. **The record gate agrees.** `python3 scripts/check-oracle-pins.py` passes
   after the edit, and it is mutation-proved: `gateable = yes` with
   `evidence = #857` must be refused.

## Dependencies

- An `rc` lease on a fleet device. `AGENTS.md` §"Work on a GPU happens inside a
  lease" makes the lease the only path to `dgx:gpu0`, `thor:gpu0` and
  `orin:gpu0`, and the condition is the device, not the shell.
- The recorded artifact reachable from the leased worker. `/workspace` is the
  `rc/` subfolder of the house NAS share, so
  `/mnt/nas_share/checkpoints/qwen3.8-27b-gguf/` is NOT visible to a job. The
  artifact is staged to `/mnt/nas_share/rc/oracle857/ckpt/` first.
- No CUDA toolkit. The build is CPU only, so the worker needs none.

## Work breakdown

1. Verify the pin locally: both objects resolve, and `10bf611e5` is upstream tag
   `b10451` on `origin/master` rather than another fork object.
2. Verify the artifact semantically and stage it where a leased job can read it.
3. Write the job script on the NAS, because an `rc run` submit dies with the
   detaching client and the tool call has a ten-minute ceiling.
4. Lease, build, run, copy the artifacts back.
5. Write the evidence file and edit the record to whatever ran.

## Risks/decisions

**Device: `thor:gpu0` first, `dgx:gpu0` second.** #857 asks for dgx.casa and
`.env` names it `GATE_HOST`, so dgx is where the answer belongs. At claim time
`dgx:gpu0` was busy with another session's clock experiment and carried four
queued jobs behind it, and a second session was told that queueing behind it is
correct rather than contending. `thor:gpu0` was free, is a fleet device reached
by the same lease, shares the same `/workspace` folder, and has 14 cores and
132 GB against a 17 GB artifact. The build and the run need no GPU at all, so
the device is a CPU and a filesystem, and thor supplies both. `orin:gpu0` is
refused for two measured reasons: its `/workspace` is local disk that does not
carry the artifact, and 32 GB of RAM leaves no headroom over a 17 GB model.

**Gateability is not a floor, and one device answers it.** A floor is
host-specific and #1003 owes one per host. "Builds and runs the model" is a
property of the pin. Proving it on one leased fleet device settles the flag;
running it on dgx as well settles #857's own wording.

**`GGML_NATIVE=ON`, so the binary sha256 is per host.** A user gets a native
build, so a floor must be one. The consequence is that the recorded sha256
identifies a build on a named box and never a tree, which is the same
distinction our own binaries carry.

**The 17 GB stage is a real cost and is paid once.** The copy runs at about
12 MB/s over CIFS on the same server, so about 25 minutes. Both jobs read the
one staged copy.

**A CIFS mmap is not a run surface.** llama.cpp mmaps the GGUF. The job copies
it to the worker's local disk first, verifies the sha256 of the local bytes,
and runs from there.

**The shared worker container is long-lived.** The job builds under `/tmp` and
deletes its own tree, and it installs nothing globally.

## Stop conditions

- Stop if the build fails at the pin. Record the compiler, the error, and the
  first failing target. `gateable` stays `no`.
- Stop if the model refuses to load. Record the refusal verbatim. `gateable`
  stays `no`.
- Stop and return `NEEDS_DECISION` if the pin does not fetch from the remote,
  because that is #857 again on a new object and it is a record decision.
- Do not run the quant-matched comparison. Return to the operator instead.

## Outcome

**The pin builds and runs. `gateable` is `yes`.** The measurement ran on
2026-08-22 in one `rc` lease on `thor:gpu0`, job
`d96c2867-4344-4064-84e3-d3a04a1b1925`, and every gate in §"Gates" passed. The
identity chain, both recipes, the generated text and the raw log are in
[`../../docs/bench-evidence/oracle-llamacpp-b10451-gateable-20260822.md`](../../docs/bench-evidence/oracle-llamacpp-b10451-gateable-20260822.md).

| Gate | Result |
|---|---|
| 1. Object and label agree on the remote | PASS. `fetch_commit_rc=0`, `fetch_tag_rc=0`, `src_head` = `tag_b10451_commit` = the pin |
| 2. Source tree is the pin and nothing else | PASS. `git_status_porcelain_bytes=0` |
| 3. Artifact is the recorded one | PASS. Header parses, data end equals file size, and the worker-local sha256 equals `7e78da5d...c6fe169` |
| 4. Build produces a binary | PASS. `configure_rc=0`, `build_rc=0`, `llama-completion` executable |
| 5. Model generates | PASS. `run_rc=0`, 228 stdout bytes against a 29-byte prompt, `common_perf_print` present |
| 6. Timings not gated | Held. Recorded and marked unquotable in the evidence file |
| 7. Record gate agrees | PASS, and mutation-proved four ways |

What was measured, and it changed the plan: at `b10451` `llama-cli` is a
server-backed chat client behind `LLAMA_BUILD_SERVER`, so the one-shot binary is
`llama-completion`. The spec fixed that before the run rather than discovering
it in a failure.

What was rejected: `dgx:gpu0`, because it was held with four jobs queued and
this work uses no GPU; `orin:gpu0`, because its `/workspace` is local disk
without the artifact and 32 GB leaves no headroom over a 17 GB model; running
the artifact from CIFS, because llama.cpp mmaps it; and `llama-bench`, which was
built but deliberately not run, so this row produces no number anyone can quote
as a floor.

Why each default has its value: `-j 4` because unconstrained parallelism has
OOM-rebooted a fleet box. `GGML_NATIVE=ON` because a user gets a native build,
so a floor must be one, at the cost of a per-host binary sha256. Static
libraries so the recorded sha256 covers the whole binary.

One fact the run produced that outlives it: llama.cpp loads 851 of the
artifact's 866 tensors and ignores all 15 of `blk.64`, including the four
`nextn.*` multi-token prediction tensors. A quant-matched comparison is
therefore not automatically a matched-work comparison. Recorded for #821 and
#1003 in the evidence file and in the oracle record.

The checker mutations, all four red and each restored byte-for-byte:

| Mutation | Reported |
|---|---|
| `gateable = yes` with `evidence = #857` | `an issue is a promise of a measurement, not one` |
| `gateable = yes` with a non-existent path | `evidence path ... does not exist in this tree` |
| `gateable = no` with a path | refused: a `no` record must name the owing issue as an issue reference |
| Evidence file deleted, record untouched | `evidence path ... does not exist in this tree` |

The fourth is the one that matters most: it proves the flag is tied to a file
that has to keep existing, not to a string somebody typed once.

## Now

Claimed 2026-08-22 from `08c81a89218906cac08209a63c6301f03fdc8ec7`. The spec
landed before the measurement, the measurement passed, and the record now says
`yes` against a path in this tree. #857 is discharged. Owed elsewhere and not
here: every floor (#1003), the vllm.cpp comparison (#821 W3), and a GB10 build,
which #1003's per-host re-takes carry.
