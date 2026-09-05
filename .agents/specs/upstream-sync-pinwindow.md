# Sync cycle `db92053e97`, wave PINWINDOW

Issue: [#2589](https://github.com/mudler/vllm.cpp/issues/2589).
Predecessors: [#2490](https://github.com/mudler/vllm.cpp/issues/2490) (steps 1
to 4 against `cdefd9d499`) and [#2524](https://github.com/mudler/vllm.cpp/issues/2524)
(steps 5 to 7 against the same target, stalled).
Guide: [`../upstream-sync.md`](../upstream-sync.md) §"The sync cycle".

## Now

The pin has **not** advanced and does not advance in this wave. The active parity
pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

One question decides this wave, and the rest of it is the record that answers
that question and sizes what would follow a yes.

**The question.** #2524 measured that `origin/main` does not install on this
fleet. `RUNDEPS_RC=1`, `IMPORT_RC=1`, `EXT_RC=1`, because `2d7f42b4f3`
vllm#52801 promoted `instanttensor` from `requirements/test/*`, where it carries
`platform_machine == "x86_64"`, into `requirements/cuda.txt` with no platform
marker, and every fleet device is aarch64. A build that cannot import is not an
oracle. So: **is there a revision below the promotion that this fleet CAN
install, and does the pin therefore have anywhere to go at all?**

**The target.** `db92053e97b5630a6a36118693b1dffe9b03be36`, 2026-08-19,
`[Core] Skip broadcasting mm tensor data to workers for prefix-cache-covered items (#52041)`.
It is the immediate parent of `2d7f42b4f3`, the pin is its ancestor, the range is
1026 commits, and `grep -c instanttensor requirements/cuda.txt` returns 0 there.

**Why no revision below it carries `qwen4_exp`.** `instanttensor` landed
2026-08-19, twelve days before `e126687a9a` vllm#53896 registered `qwen4_exp` on
2026-08-31, and is its ancestor. Every revision carrying the model carries the
blocker. This target therefore deliberately does not reach `qwen4_exp`, and
`MODEL-MM-QWEN4-EXP`'s vLLM citations stay ahead-of-pin and must keep saying so.

In scope:

1. Measure gateability at the target on an aarch64 fleet device, reporting the
   same five separate return codes #2524 used so the two are comparable.
2. Partition the 1077-commit classified universe of `555967922..cdefd9d499` onto
   this range, which is a strict prefix of it, and carry each disposition by SHA
   rather than re-deriving it.
3. Record which ordering constraints and which recorded deviations fall inside
   the narrower range.
4. Write [`../sync/2026-09-02-db92053.md`](../sync/2026-09-02-db92053.md) and
   carry over what is left.

Out of scope:

- Advancing the pin. See §5.
- Working the 206-entry PORT-NOW queue. That is more than one wave, and the
  guide's documented behavior for a cycle that stalls is to keep the old pin and
  record the remainder.
- Any product code. This wave touches records and documents only, so it mixes no
  feature work into a sync cycle.
- Tiering the INVENTORY entries. Still owed by #2524.

## 2. Design

The report is derived, not re-derived. `2026-09-01-cdefd9d.md` classified 1077
commits of the 1566-commit range `555967922..cdefd9d499` by twelve readers. This
range ends 540 commits earlier at the same start point, so every one of its
commits is one of those 1077 or is outside the mirrored universe, and a
disposition keyed on an immutable SHA does not change because the range end
moved. The partition is therefore a set intersection against
`git rev-list 5559679229..db92053e97`, and the report states which entries were
carried and which were re-read.

Re-reading a sample is still owed and is still C7's obligation. This wave
re-checks only the entries its own conclusions rest on, and names them.

## 3. Risks

- **A carried disposition is only as good as its original reading.** #2524
  measured that reading: of 63 sampled PORT-NOW entries, 11 were real gaps, 12
  were already satisfied and 40 did not apply. The count is an upper bound on
  the queue, not a count of code gaps, and the report repeats that beside the
  number rather than only in a footnote.
- **A partial dependency series is worse than none.** The KV-cache layout
  refactor's stages 4 and 5 are inside this range and stage 6 is not, so an
  advance to this target lands a half-migrated layout surface unless the wave
  that works the queue treats 4, 5 and 6 as one unit that this target cannot
  complete.
- **`gateable` is a measurement, not an inference.** A green install and import
  is not "builds and runs the model". The report says which half is measured.

## 4. Gates

This wave changes documents and records rather than product code, so its gate is
the record and style gate rather than a token or throughput gate. It claims no
numeric parity result, regenerates no golden dump and re-baselines no benchmark.

```sh
scripts/agent-preflight.sh --staged
python3 scripts/check-commit-style.py
python3 scripts/check-agent-record.py
```

`agent-ready.py` runs a compile step and has timed out at ten minutes on this
box under load, so its commit-scoped checkers are run individually rather than a
timeout being reported as a pass.

## 5. Stop conditions

- **Do not advance the pin while the PORT-NOW queue is unworked.** A pin
  advanced over an unported queue makes every downstream gate measure against an
  oracle the tree does not mirror.
- **Do not advance the pin on an install-and-import result.** An oracle is
  gateable only once it demonstrably builds and runs the model.
- Stop and report rather than hunt further back if this target also fails to
  install. That the pin cannot advance at all on this fleet is the finding, and
  it belongs to the developer, not to a wider search.
- Stop rather than edit a disposition list. A reclassification is a step-3
  defect and belongs to a cycle.

## Owed

- The 206-entry PORT-NOW queue for this range (#2589), in the dependency order
  the report names.
- Re-derivation of every carried disposition against this tree (#2589, C7).
- The run half of gateability: a model served from the built target inside a
  lease on a device whose container exposes the GPU (#2589).
- The 356 INVENTORY entries' tiers and `porting-inventory.md` §9's four
  deviations (#2524).
