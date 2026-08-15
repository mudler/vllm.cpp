# `tests/parity/goldens/` — the schema, declared

This directory is a **shared surface**. Its schema used to be implicit, and that
cost someone else a day: `test_op_parity`'s `RunGoldenPass` walks *every*
`manifest.json` under this root and used to index `m["op"]` unconditionally, so
an oracle capture recorded at `goldens/<name>/manifest.json` — a perfectly
reasonable place for it — took down a test belonging to a different row through
an unhandled key lookup that did not even name the offending case. It happened
with `minimax_music3_oracle` (`34dc57876`, #672 / #708) and an unrelated
campaign had to trace a `453/454` `ctest` before it could trust its own numbers
(#755).

So the schema is written down here, and the walker enforces it.

## The rule

A directory under this root is in exactly one of three states.

| Shape | What the walker does |
|---|---|
| **No `manifest.json`** | Ignored entirely. Fixture directories owned by a dedicated test (`tokenizer_qwen36`, `gemma4_e4b_image/vision_refs`, …) live here this way. |
| **`manifest.json` declaring a string `"op"`** | An **op-parity golden**. `RunGoldenPass` dispatches on `"op"` to a runner and executes it on every backend. An `"op"` with no runner is a hard failure — see `PendingRunnerOps()` for the "goldens landed before their runner" case, which is a loud skip. |
| **`manifest.json` with no `"op"`, and the directory named in `NonOpGoldenDirs()`** | A **declared non-op artifact** — an oracle capture, an engine acceptance fixture. Skipped **loudly and counted**: the walker emits a `SKIP <name>` message and reports the total. |

**Anything else is a by-name failure.** A `manifest.json` with no `"op"` whose
directory is not listed fails naming itself and saying what to do; so does a
listed directory whose manifest *does* declare an `"op"`, so the list cannot rot
into a mute exclusion that leaves a real op case ungated.

That is what closes the walker's input set: the set of things it executes is
exactly the set of manifests declaring a known op, and everything else is either
loud-and-listed or loud-and-failing. A future oracle capture dropped into this
tree cannot land unnoticed — it fails, by name, with the fix in the message.

## A malformed golden never aborts the pass

Closing the input set is not the same as closing the *exception* surface, and
#776 is the difference. A manifest can be listed, named, and accept a runner and
still make the walker throw: the file can be invalid JSON, or a field the runner
reads can be absent or `null`. Either exception escaping `RunGoldenPass` aborts
the whole test case, so **every golden the walker had not reached yet goes
unchecked** — including the `no runner for op` check this directory depends on.
That is a gate that has stopped gating while still looking like one red line.

Both throw sites are now guarded. An exception from parsing a manifest, or from
the runner reading it, becomes a `FAIL_CHECK` that names
`goldens/<case>/manifest.json` and quotes the original exception, and the pass
**continues to the next golden**. So a malformed golden costs you exactly its
own case, and the report names the file instead of a line number in
`test_op_parity.cpp`.

The guard catches `std::exception` and nothing wider. doctest's
`TestFailureException` is deliberately not derived from it, so a `REQUIRE` or
`FAIL` inside a runner — the unregistered-op refusal above included — still
aborts the pass exactly as before. The guard cannot mute an assertion; it only
converts a thrown diagnostic that names no file into one that does.

## Adding an op-parity golden

`manifest.json` carries at least:

```json
{
  "op": "<runner name>",
  "args": { },
  "tol": { "atol": 1e-5, "rtol": 1e-5 },
  "tensors": { "x": { "file": "x.npy", "dtype": "f32", "shape": [8, 128] } }
}
```

`dtype` is one of `f32 bf16 f16 i32 i64`, and `shape` must match the `.npy`
header element-wise — the loader cross-checks both and fails naming the file.
Add the runner in `tests/parity/test_op_parity.cpp` in the same change.

## Adding an oracle capture

Prefer a root of your own, outside this tree — the walker never sees it, and
that is the cheapest correct answer. `tests/CMakeLists.txt` already points
individual tests at their own fixture directories by absolute path, so nothing
requires a capture to live under `goldens/`.

If a capture must live here — `minimax_music3_oracle` does, because its
`manifest.json` is the `evidence =` path of `.agents/oracles/diffusers.md` and
`scripts/check-oracle-pins.py` requires that path to exist in this tree — then
name the directory in `NonOpGoldenDirs()` in `tests/parity/test_op_parity.cpp`,
in the same change that commits the capture. The walker will refuse it until you
do, which is the point.
