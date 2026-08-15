# Unaligned safetensors loads — the fourth recurrence, and the two UBSan cannot see

Identity: `FIX-UNALIGNED-LOADERS-772`

Issue: [#772](https://github.com/mudler/vllm.cpp/issues/772)

Class predecessors: [#301](https://github.com/mudler/vllm.cpp/issues/301)
(closed; it left the `vt::LoadUnaligned` seam),
[#627](https://github.com/mudler/vllm.cpp/issues/627) (`qwen3_5_weights.cpp`;
rescoped to the grep-able checker this file's finding argues for), and
[#674](https://github.com/mudler/vllm.cpp/issues/674) / PR #688
(`ltx2_loader.cpp`, whose forced-odd-offset case is the pattern mirrored here).

Status: `ACTIVE`. Base `af026e5241e16e1d2a89da001b978ffcb3e0f634`.

## Scope

Remove the four remaining `reinterpret_cast<const uint16_t*>(<mmap'd
StTensor>.data)` formations named in #772, and route
`minimax_h3_vae_loader.cpp:96-101` — which hand-rolls the same `std::memcpy`
instead of calling the shared seam — through `vt::LoadUnaligned`. Add the gate
that forces an odd payload offset for each of them.

Change no loaded value, no shape rule, no dtype rule, no error message and no
public behavior. The only surface addition is two declarations in a new INTERNAL
header under `src/`, without which the gate cannot drive the production code
(see Design). Nothing under `include/` is touched, so the public ABI is
unchanged.

Explicitly excluded: `scripts/check-pr-size.py`,
`scripts/check-commit-trailers.py` (PR #782),
`scripts/check-windows-portability.py` (#774), `scripts/audit-live-rows.py`,
`.github/workflows/ci.yml` (#726); the checker #627 now proposes, which is that
issue's own work and needs its own spec; and the two pre-existing `sanitize-cpu`
reds, [#775](https://github.com/mudler/vllm.cpp/issues/775) and
[#776](https://github.com/mudler/vllm.cpp/issues/776).

## Observed baseline and root cause

A safetensors tensor begins at `8 + <JSON header length> + <sum of the preceding
tensors' sizes>`. None of those three terms is required to be even, so a BF16
tensor that follows an odd-length tensor — or that sits behind a header of odd
length — starts on an odd byte. That is an ordinary file, not a corrupt one.

Four loaders formed a `const uint16_t*` over that address:

| Site | Shape | Sanitizer-visible? |
|---|---|---|
| `voxtral.cpp:51` `StBf16ToF32` | cast, then `src[i]` | YES |
| `voxtral.cpp:344` `PermuteQKBf16` | cast, then row address + `memcpy` | **NO** |
| `qwen3_vl.cpp:78` `LoadVisionF32` | cast, then `p[i]` in `Bf16BitsToF32` | YES |
| `qwen3_5_mtp.cpp:71` `CopyRawNK` | cast, `+ offset`, then `memcpy` | **NO** |

**The two `NO` rows are the finding.** Forming and advancing a misaligned
`uint16_t*` is undefined in its own right, but both of them then launder every
access through `std::memcpy`, which reads bytes — so `-fsanitize=alignment`
never fires. All three earlier recurrences of this class were found by UBSan.
These two would have survived every one of those sweeps, and would survive the
next one. That is the argument for #627's checker, and it is why the gate below
is a value gate rather than another sanitizer run.

All four were latent by luck: the offsets happen to be even in the fixtures we
run. That is a property of the files we load, not of the code.

## Design

Two shapes of fix, because the two shapes of defect are different.

**Scalar loads** (`StBf16ToF32`, `LoadVisionF32`) go through
`vt::LoadUnaligned<uint16_t>` over an `unsigned char` base — the seam #301 left,
already used by `dense_loaders::TransposeBf16` and by
`minimax_h3_vae_loader.cpp`. It is free: verified during the #688 review that at
`-O2` it compiles instruction-for-instruction identically to the raw cast
(`movzwl (%rdi,%rax,2)`), emits no `memcpy` call, and is exhaustively
value-correct over all 65536 bit patterns × byte offsets 0-7.

**Bulk copies** (`PermuteQKBf16`, `CopyRawNK`) drop the typed pointer entirely
and do the arithmetic in `unsigned char`. `vt::LoadUnaligned` would be the wrong
tool: the payload is already moved by `memcpy`, and there is no scalar to load.
The one thing this rewrite must not lose is the `sizeof(uint16_t)` the pointer
type used to supply — `CopyRawNK`'s `offset` counts BF16 **elements**, not bytes
— so that factor is what the gate's mutation arm targets.

`minimax_h3_vae_loader.cpp` is **not** a defect: it already did the byte-wise
load and already carried the reason in prose. It is in scope because it
hand-rolled `std::memcpy` where the shared seam exists, which is the parallel
path AGENTS.md's shared-seam rule is about — and it is the file that carried the
repair while three other loaders shipped the cast anyway.

**Why two declarations exist at all, and why they are INTERNAL.**
`StBf16ToF32` and `PermuteQKBf16` live in an anonymous namespace, and the only
entry point above them is `LoadVoxtralWeights`, which is unreachable at test
scale: `VoxtralEncoderConfig()` is fixed at 32 layers of `d_model` 1280 / `ffn`
5120, so a synthetic checkpoint that satisfies it is ~1.2 GiB. The alternative to
declaring them is a gate that tests a copy of the code rather than the code,
which proves nothing.

They go in `src/vllm/model_executor/models/voxtral_loader_internal.h`, **not**
`include/`. They are loader internals, not a shipped capability, and putting them
in the public tree would overstate what they are — AGENTS.md's ABI rule is about
capabilities reaching `include/vllm.h`, and these reach nothing. The tree already
has this shape: the NemotronH MIXED_PRECISION resolver header is internal until a
loader consumes it, and its two suites reach it through
`-I${CMAKE_SOURCE_DIR}/src` (`tests/CMakeLists.txt:80-96`). The functions move out
of the anonymous namespace unchanged and gain a `Voxtral` prefix, since the bare
names are generic enough to collide.

The other three sites need no such change: `LoadQwen3VLVisionWeights`,
`LoadQwen3_5MTP` and `LoadMiniMaxH3AudioVaeWeights` are already public and are
reachable with a checkpoint of a few hundred bytes.

## Tests and RED evidence

One new suite, `tests/vllm/models/test_loader_unaligned_offsets.cpp`, with one
case per site, mirroring `test_ltx2_video.cpp`'s "ODD safetensors payload offset
(#674)" case:

- The safetensors file is written by the test, so the header length — and with
  it the payload parity — is ours to choose. Every BF16 tensor is an even number
  of bytes, so the parity of the first payload byte is the parity of every
  tensor in the file, and one space of padding inside the counted JSON header
  flips all of them at once. Trailing whitespace is legal JSON, and padding the
  header is exactly how real writers align their payloads.
- The file is written **both** ways and whichever lands odd is kept, so no case
  depends on the exact length of the generated JSON. `REQUIRE` proves the two
  spellings really differ in parity, so padding could not become inert silently.
- The mapped **address** parity is `REQUIRE`d, not inferred. A fixture edit that
  makes the address even fails the REQUIRE instead of passing while covering
  nothing.
- Every case checks the loaded **values** against the fixture's own bit
  patterns, not merely that the load returned — because for two of the four
  sites the values are the only witness there is.

RED, at base + the `voxtral.h` export and nothing else, under
`-DVLLM_CPP_SANITIZE='address,undefined'`:

- `voxtral.cpp` — `runtime error: load of misaligned address … for type 'const
  short unsigned int', which requires 2 byte alignment`, in
  `vllm::VoxtralStBf16ToF32`.
- `qwen3_vl.cpp:58` — the same diagnostic, in `Bf16BitsToF32` under
  `LoadVisionF32`.
- `PermuteQKBf16`, `CopyRawNK` and the minimax loop — **pass**, which is the
  point being pinned: the sanitizer cannot see them.

Because the lane builds with `-fno-sanitize-recover=all`, the first finding
aborts the process, so each case is run in its own process to witness its own
site.

For the two sanitizer-invisible sites the RED is a **mutation**: dropping the
`* sizeof(uint16_t)` from the byte arithmetic — the exact mistake this rewrite
can make — must red the case.

## Gates

1. RED: the two sanitizer-visible cases report the misaligned-load diagnostic at
   base; the two invisible ones pass, and that asymmetry is recorded.
2. GREEN: all five cases pass under `address,undefined` with no new finding.
3. Mutation: `* sizeof(uint16_t)` dropped at `qwen3_5_mtp.cpp` and at
   `voxtral.cpp` must red their cases. Restore verified by checksum.
4. Inertness: the object code of the four changed functions, compiled at the
   project's normal optimization level, before and after.
5. Full `ctest` on a clean CPU build, and `scripts/agent-preflight.sh --staged`,
   and `python3 -m pytest tests/scripts/`.

Baseline to subtract: `windows-msvc-*` is PR-only and red on every PR (#584);
`test_cpu_threadpool` flakes under load (#631); `sanitize-cpu` on `main` is red
on #775 and #776; `agent-record` fails via `audit-live-rows` (#726).

## Risks and stop conditions

The risk is a rewrite that is UB-clean and quietly wrong — an element offset
read as a byte offset loads the wrong expert's rows and still returns a
plausible tensor. Gate arm 3 exists for exactly that, and every expectation in
the suite is derived from the fixture's own stacked layout so a confusion
between neighbouring slices shows up as specific wrong values rather than as
"something differs".

Stop with `NEEDS_DECISION` if any loaded value moves anywhere: this change is
required to be inert on every already-aligned checkpoint, which is all of them
today.

## Outcome

**The two shapes of defect need two different fixes, and conflating them would
have been a bug.** #772 proposes `vt::LoadUnaligned` for all four sites. That is
right for the two scalar loads and wrong for the two bulk copies:
`PermuteQKBf16` copies a whole row per iteration and `CopyRawNK` copies a whole
`[N,K]` slice, so there is no scalar to load and wrapping them would have meant
rewriting a `memcpy` as an element loop. Both were fixed by moving the pointer
arithmetic into `unsigned char` instead, which is what the issue itself
prescribes for `CopyRawNK` alone.

**The asymmetry #772 predicted was reproduced exactly.** At base, the two
sanitizer-visible sites reported `load of misaligned address … requires 2 byte
alignment` and the two laundered ones passed clean — on the identical fixture,
at the identical odd offset, in the same binary. The class is provably not fully
discoverable by the lane that caught its first three recurrences.

**The mutation arm is not vacuous.** Dropping `* sizeof(uint16_t)` from
`CopyRawNK`'s byte advance red 48 of 93 assertions in the MTP case; dropping it
from `PermuteQKBf16`'s row address red 21 of 33. Both reverted to a byte-identical
tree, verified by `md5sum` rather than by `git status`.

**What could not be proved here.** The 27B / 35B / Coder token gates #627 asks
for need the GB10 box and staged checkpoints, and other work was live on that
GPU; they are not claimed. What is claimed instead is object-code inertness
(gate arm 4) plus the full CPU suite, which is the strongest evidence available
without taking the GPU. Voxtral's own e2e gate additionally requires a
checkpoint that is not staged here and exits 77.
