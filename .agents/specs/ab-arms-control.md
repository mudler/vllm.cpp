# The two-arm guard has to depend on the change, and the hash does not

Issue: [#1516](https://github.com/mudler/vllm.cpp/issues/1516)
Row: `BENCH-AB-ARMS-CONTROL`

Found on `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation`,
which owns #1516 in `.agents/issue-index.md` and whose spec §16.6a states the
rule this row repairs. The repair is not a Music3 change: it is the shared
control every two-arm A/B in this tree quotes, so it gets its own row and its own
spec, and `.agents/specs/minimax-music3.md` §16.6a and §16.6b point here.

## Why

§16.6a voided a Thor pair because both arms turned out to be one binary, and the
rule it drew is right: **whenever two artifacts are required to differ, assert
that they differ before believing anything downstream.** The tell was not the
times, which were 0.26 % apart and looked like noise; it was the **identical call
count**, 808 on both arms of a change whose whole claim is that the count moves.

The assertion as implemented cannot carry that rule for an end-to-end pair.
`minimax-music3-gen` is a **72 744-byte client** of the shared library
(`examples/CMakeLists.txt:425-426` links `vllm::shared`, `CMakeLists.txt:2633`
makes it `SHARED`), every line of a change under test lives in
`libvllm_shared.so`, and no `CMAKE_SKIP_BUILD_RPATH` is set anywhere in
`CMakeLists.txt`, `cmake/` or `examples/CMakeLists.txt`. CMake therefore writes
the build-tree RPATH into the client, and two build directories produce two
hashes whatever the source says. Both arms of the corrected pair were 72 744
bytes to the byte and hashed differently, and the guard passed. It would have
passed identically on two clones of the same commit.

### Reproduced, with the mechanism attributed

A minimal CMake project of the same SHAPE — one `SHARED` library carrying the
change, one thin client that links it — built from **byte-identical sources**
(`diff -r` rc 0) into two build directories:

```text
16016 bytes  d4ead254e9b3461d91ffc96807de171aa5bfe888f1893981862b48833fe488ac  bld-old/abdemo-client
16016 bytes  7a791ed2bac9790b6b01121234ad48346f798e21e872e74f056639f71426b11f  bld-new/abdemo-client
ARMS_DIFFER=yes  ->  the old guard PASSES

readelf -d bld-old/abdemo-client:  RUNPATH [.../abrepro/bld-old]
readelf -d bld-new/abdemo-client:  RUNPATH [.../abrepro/bld-new]

ar.depth_forward calls=808     (arm A)
ar.depth_forward calls=808     (arm B)
```

Equal sizes, different hashes, one string of difference, and a behavioural
control that does not move. That is §16.6a's own tell, arriving through a guard
that reports `yes`.

**Then the change was actually made** — `frames * 8` became `frames * 4` in the
library, and only there:

```text
7a791ed2bac9790b6b01121234ad48346f798e21e872e74f056639f71426b11f  bld-new/abdemo-client
ar.depth_forward calls=404
```

**The client is byte-for-byte the hash it already had.** So the client's hash
differs between the arms for a reason that has nothing to do with the change,
and does not differ when the change happens. It is a function of the build
directory, not of the code under test. A guard built on it is not weak, it is
inverted.

### Both of the obvious repairs were measured, and both fail

**Hash the shared library instead of the client** is the fix #1516 proposes, and
it holds only under a condition the recipe forbids:

| arms | `libabdemo_shared.so` | `abdemo-client` |
|---|---|---|
| one source dir, two build dirs | `5f5f89f4…` = `5f5f89f4…` **equal** | differ |
| two source dirs, identical bytes | `5f5f89f4…` vs `d2eea281…` **differ** | differ |

The library hash is location-independent across build directories and
location-DEPENDENT across source directories, because `VT_CHECK`
(`include/vt/dtype.h:11-17`) embeds `__FILE__`, CMake compiles with absolute
source paths, and no `-ffile-prefix-map` is set in this tree. 121 files under
`src/` reference `__FILE__`. The demonstration library reproduces it with one
`DEMO_CHECK`, and `strings` finds the absolute source path in each arm's `.so`.
§16.6a's own repair mandates **separate clones**, so the shape that makes the
library hash meaningful is exactly the shape it forbids.

**Set `CMAKE_SKIP_BUILD_RPATH` on the client** makes the client hash
location-independent, and that is worse. The client contains none of the change,
so once the RPATH is gone two arms carrying a real library change hash the SAME,
and the guard fires `FATAL: both arms are the SAME BINARY` on a correct pair. It
also stops the example running from its build tree without `LD_LIBRARY_PATH`.
Rejected.

**Conclusion.** In the two-clone, two-build-directory shape this project's
benchmark recipes are required to use, **no artifact hash in this tree is
falsifiable**. The load-bearing leg has to be something that depends on the
change: a value the arms compute, or the content of the source the change is in.

## Design

`scripts/ab-arms-differ.py` renders one verdict from three legs and states what
each leg is worth.

1. **The hashes.** Equal is still `FATAL: ARMS_IDENTICAL`, because that is
   §16.6a's original defect and it is real. Different is reported and decides
   nothing.
2. **Location dependence.** Each artifact is searched for the byte string of its
   own build or source root, named with `--root-a` / `--root-b`. A hit prints
   the root and the offset and sets `HASH_LOCATION_DEPENDENT=yes`, which is the
   RPATH in the reproduction above. With no root given the line reads
   `UNMEASURED`, never `no`: an absent probe must not look like a passing one.
3. **The controls.** `--control NAME A B`, repeatable, and **at least one must
   have moved** or the verdict is `FATAL`. Passing none at all is
   `FATAL: NO_CONTROL`, so a hash-only verdict is refused by name rather than
   accepted quietly.

Two kinds of control, and they catch different failures. A **behavioural** one is
a value the arms computed — a bucket's call count, an emitted stage — and it is
the only leg that catches a STALE BINARY, which is what §16.6a actually suffered.
A **source** one is the hash of the file the change lives in, and it catches TWO
ARMS THAT ARE THE SAME SOURCE. Neither subsumes the other, and the tool prints
which it was given rather than inferring it.

**The best shape needs none of this.** Where a runtime switch can turn the change
off in one binary, `VT_OP_PROVIDER_DISABLE` style, a same-binary A/B removes the
question: there is one artifact, so no hash can be vacuous, and the control is
the switch itself. `.agents/benchmarking.md` recommends that first and this tool
second.

## Scope

In scope: `scripts/ab-arms-differ.py`, `tests/scripts/test_ab_arms_differ.py`,
the guard block of `scripts/music3-vocoder-conv-ab.sh`, the paragraphs of
`.agents/benchmarking.md` that state the rule, an appended correction in
`.agents/benchmark-record.md`, and the §16.6a and §16.6b sentences in
`.agents/specs/minimax-music3.md` that read the hash leg as load-bearing.

Out of scope and stated as owed:

- **The recorded Thor figures are NOT withdrawn.** `MUSIC3-DEPTH-THOR-PAIR-2`
  carries `ar.depth_forward` at 1414 -> 808 calls, which is a behavioural control
  that moved, so the pair is separated by the leg that counts. Only the sentence
  naming the hashes as "the precondition this section exists to insist on" is
  wrong, and it is corrected in place rather than by re-running anything.
- `-ffile-prefix-map` for the tree. It would make the library hash
  location-independent across clones and is a build-wide change with its own
  debug-info consequences. Not this row; named because it is the one lever that
  would make leg 1 mean something.

## Gates

`python3 tests/scripts/test_ab_arms_differ.py` is the focused gate.

| Case | Old guard | New tool |
|---|---|---|
| two builds of IDENTICAL source, two build dirs | PASS (`ARMS_DIFFER=yes`) | `FATAL: NO_CONTROL_MOVED` |
| the same pair with no control offered at all | PASS | `FATAL: NO_CONTROL` |
| two arms that are one binary | FATAL | `FATAL: ARMS_IDENTICAL` |
| a real change, control moved 808 -> 404 | PASS | PASS |
| an artifact embedding its own build root | silent | `HASH_LOCATION_DEPENDENT=yes` + offset |
| no root named | silent | `UNMEASURED` |

The identical-source case is built for real by the suite, from a CMake project of
the same shape, and is skipped by name when no `cmake` or C++ compiler is present
rather than passing.

## Risks

**A control that is easy to satisfy.** The tool cannot know whether a value is
meaningful; it can only refuse a verdict that has none. The recipe author still
picks a control that the change must move, and the spec says which kind catches
which failure.

**A guard that always fails is not a guard.** The positive control — a real
library change, count 808 -> 404 — is asserted in the same suite as the negative
one, so a tool that refused everything would be red.

**The reproduction depends on a toolchain.** It is skipped, by name, with the
reason printed, when `cmake` or a C++ compiler is absent. The verdict logic is
pinned unconditionally on fabricated inputs, so the tool is never untested.

## Now

Implementing.
