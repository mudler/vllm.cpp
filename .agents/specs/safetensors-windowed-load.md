# Spike — windowed safetensors load (progressive source-page release)

Row: **`LOAD-SAFETENSORS`** (engine-matrix; currently `PARTIAL`). Claim
`CLAIM-LOAD-WINDOWED-1`.

## Scope

In: cut the LOAD-time host-memory **peak** (VmHWM) of the safetensors weight
load from ~48.3 GB (measured 27B) to ≈ the steady mirror size (~25 GB) by
releasing each source byte range's resident pages the instant it has been
copied into the persistent owned host mirror, during the copy loop. The fix is
a progressive `madvise(MADV_DONTNEED)` over already-consumed, copied-then-dead
ranges of the read-only `MAP_PRIVATE` safetensors mmaps, gated by a
process-cached env `VT_LOAD_WINDOWED_RELEASE` (DEFAULT ON, `=0` rollback for a
same-binary A/B).

Supported model paths (both in scope): **27B dense** (`LoadQwen3_5Dense`) and
**35B MoE** (`LoadQwen3_5Moe`). **GGUF is out of scope** (separate reader/loader;
`qwen3_5_gguf_weights.cpp`).

Dispatch behavior: the release is a no-op when `VT_LOAD_WINDOWED_RELEASE=0`;
otherwise every copy helper releases the interior (fully page-covered) portion
of the just-consumed source range. Correctness is dtype/path independent because
the released pages are clean file-backed pages that re-fault from disk on any
later read.

Out: the direct-to-final-device streaming redesign (the deeper fix that also
removes the steady mirror; still wanted for 35B) is NOT in scope here. Not
touched: `src/vllm/v1/worker/gpu/runner.cpp`, the GDN slot-lifecycle region of
`qwen3_5.cpp` (owned by `CLAIM-GDN-BA-ROUNDING-1`), and everything in
`scripts/tools` owned by the c16 component track. **No axis credit** is claimed
here: the binding memory axes stay `FAILED` until an authorized exact-grid
rerun; this checkpoint records only a bounded VmHWM A/B disposition.

## Upstream chain

vLLM streams shards and copies each parameter immediately into the
target-device module, so its source pages never all sit resident at once — the
exact memory profile our progressive release reproduces on the mmap path:

- `vllm/model_executor/model_loader/base_loader.py:43-82` — `load_weights`
  drives `_get_all_weights` streaming into the constructed model, then returns;
  the shard iterator is consumed once.
- `vllm/model_executor/model_loader/weight_utils.py:905-954` —
  `safetensors_weights_iterator` opens each shard with `safe_open(...,
  framework="pt")` and **yields one tensor at a time** (`f.get_tensor(name)`),
  materializing each and dropping the handle scope per shard.
- `vllm/model_executor/models/utils.py:170-180,252-279` — `AutoWeightsLoader`
  copies the yielded tensor into the destination param immediately
  (`weight_loader(param, loaded_weight, ...)` / `default_weight_loader` →
  `param.data.copy_(loaded_weight)`), so the yielded source tensor is dead right
  after the copy.

Our reader is an ORIGINAL container (no 1:1 upstream mirror): a whole-file
`mmap(MAP_PRIVATE, PROT_READ)` per shard
(`src/vllm/model_executor/model_loader/safetensors_reader.cpp:55`), whose spans
are copied out by the model loaders. Because we retain the mapping across the
whole load, the source pages accumulate resident while the mirror grows — the
double-residency the precheck localized.

## Our baseline

- `src/vllm/entrypoints/model_loader.cpp:47-63` (`LoadShards`, all mappings
  opened) and `:303-312` (mappings retained through `ModelRegistry::Load`, then
  `shards.clear()` releases them **after** the whole load).
- `src/vllm/model_executor/model_loader/safetensors_reader.cpp:43-70`
  (`MAP_PRIVATE` mmap; `StTensor::data` points into it).
- Copy sites (persistent owned bytes): `qwen3_5_weights.cpp` (35B) and
  `qwen3_5_dense_weights.cpp` (27B); container `include/.../qwen3_5_weights.h:36-106`.
- Test: `tests/vllm/test_safetensors.cpp` (reader contracts, green).
- Precheck evidence (2026-07-15, `~/work/vllm.cpp-memory-precheck-20260715`):
  steady RSS 24.75 GB (anonymous 24.58 GB ≈ 22.920 GiB mirror; file pages 129
  MB), VmHWM 48.29 GB reached at LOAD before any request. Allocator retention
  ≤0.5 GB (ruled out). Gap: no load-time page-release exists yet.

## Page-lifetime analysis (CRITICAL — copied-then-dead vs referenced-live)

Every model weight is copied out of the mmap into an owned heap buffer during
load; **no** `OwnedTensor`/`Nvfp4Weight`/`Fp8Weight`/`GdnLayerWeights` field
retains a pointer into the mmap after the owning load helper returns. Proof, per
copy helper (all take `const StTensor& t = get(name)` and fully consume
`t.data`/`t.nbytes` before returning an owned buffer):

35B MoE (`qwen3_5_weights.cpp`):

| Helper | Consume of source `t.data` | Post-return residency of source |
|---|---|---|
| `LoadBf16Direct` :76 | `std::memcpy(o.bytes, t.data, t.nbytes)` | dead (copied) |
| `LoadBf16Transposed` :90 | transpose loop reads `t.data` → owned | dead |
| `LoadBf16ToF32` :105 | upcast loop reads `t.data` → owned | dead |
| `LoadFp8Raw` :150 | `memcpy` packed; 2 f32 scalars read | dead |
| `LoadFp8Transposed` :171 (non-default) | `DequantFp8ToBf16(w.data,...)` | dead |
| `LoadNvfp4Raw` :196 | `memcpy` packed + `memcpy` scale; 1 scalar | dead |

27B dense (`qwen3_5_dense_weights.cpp`):

| Helper | Consume of source | Post-return residency |
|---|---|---|
| `LoadBf16Direct` :52 | `memcpy` | dead |
| `LoadBf16Transposed` :65 | transpose read | dead |
| `LoadBf16ToF32` :94 | upcast read | dead |
| `LoadBf16RawNK` :85 | via `LoadBf16Direct` (`memcpy`) | dead |
| `LoadMergedBf16RawNK` :211 | per-shard `memcpy` into merged owner | dead |
| `LoadCtNvfp4Raw` :118 | `memcpy` packed + `memcpy` scale; 2 scalars | dead |
| `MaterializeCtNvfp4Bf16Transposed` :292 (non-default) | dequant reads | dead |

Referenced-live-after-load ranges: **NONE.** Confirmation:
`entrypoints/model_loader.cpp:310` does `shards.clear();  // the mmap'd shards
may be released after the load.` — the whole mapping is `munmap`'d immediately
after `ModelRegistry::Load` returns, which is only sound because every tensor is
already copied. Each logical tensor is resolved and copied **exactly once**
(structured walk of `config.layer_types`; no name loaded twice), so releasing a
range after its copy never races a later read of the same range.

Header pages (`[0, 8+header_len)`, faulted by `nlohmann::json::parse` in
`Open`) are NOT tensor data and are never released by this change; they are
small (hundreds of KB).

Safety of `MADV_DONTNEED` on these ranges: the mappings are `PROT_READ`
`MAP_PRIVATE` and never written, so their pages are **clean** and file-backed;
`MADV_DONTNEED` drops the resident page and a subsequent read re-faults it from
the backing file (NOT zero-filled — that zero-fill caveat applies only to
anonymous mappings). Therefore even an over-aggressive release would be a
performance (re-fault), never a correctness, regression. To avoid even the
re-fault cost and to be provably safe regardless of copy order, the primitive
releases only the **fully covered interior pages** of a consumed range (round
begin UP, end DOWN to page bounds): a page fully inside a tensor's
non-overlapping span holds only that tensor's bytes, so a partially-copied
neighbor sharing an edge page is never dropped. Residual un-released edge pages
are ≤1 page per tensor boundary (≤~5 MB total at 4 KiB pages over ~1155
tensors) — negligible vs the 22.9 GiB peak reduction.

## Port map

| Change | File |
|---|---|
| `ReleaseSourcePages` (unconditional interior-page `madvise`), `LoadWindowedReleaseEnabled` (process-cached env), `MaybeReleaseSourcePages` (gate+primitive), `detail::SetLoadWindowedReleaseOverrideForTesting` (test seam) — declarations | `include/vllm/model_executor/model_loader/safetensors_reader.h` |
| Same — definitions (uses `<sys/mman.h>`/`unistd.h` already included) | `src/vllm/model_executor/model_loader/safetensors_reader.cpp` |
| `MaybeReleaseSourcePages(t.data, t.nbytes)` after each consume | `src/vllm/model_executor/models/qwen3_5_weights.cpp` (35B) |
| Same | `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp` (27B) |

No thin adapter; the loaders already include the reader header transitively via
`qwen3_5_weights.h`.

## Tests to port

No direct upstream test module (our reader is original). New CPU-tier cases in
`tests/vllm/test_safetensors.cpp` (named traceably):

1. `safetensors: windowed release drops consumed source pages (mincore)` —
   build a synthetic multi-page-per-tensor safetensors file, `Open` it (real
   mmap), copy each tensor (faulting its pages resident — assert via `mincore`
   they ARE resident post-copy), call the release primitive, assert the interior
   pages are NO LONGER resident. New-behavior assertion; the pre-implementation
   RED is the missing symbol / the OFF arm proving pages stay resident.
2. `safetensors: windowed release preserves copied bytes` — copy all tensors
   with release ON and with release OFF; assert the owned copies are byte-
   identical to the original file bytes in both arms (correctness invariant).
3. `safetensors: VT_LOAD_WINDOWED_RELEASE gate semantics` — via the test
   override seam, assert `MaybeReleaseSourcePages` releases when ON and is inert
   when OFF; assert `LoadWindowedReleaseEnabled()` default-ON / `=0`-off through
   the override.
4. `safetensors: interior-page release never drops a neighbor's bytes` — two
   adjacent tensors sharing an edge page; release the first; assert the second's
   bytes are still intact (edge page retained).

## Gates

- Correctness (CPU): the four new cases GREEN; full `test_safetensors` green;
  touched suites green; full tools suite green.
- Build: clean rebuild `-Werror` unmasked (CPU); production CUDA build on dgx
  for the A/B tree.
- Memory (bounded VmHWM A/B, dgx, one `flock /tmp/gpu`, ~10 min): build the
  pushed SHA in a fresh `~/work/vllm.cpp-windowed-load/<sha>` (production
  configure recipe, docs/BENCHMARKS.md), load the 27B snapshot to ready with (a)
  `VT_LOAD_WINDOWED_RELEASE=0` and (b) default ON, capture `/proc/<pid>/status`
  VmHWM + `smaps_rollup` for each, no requests, kill. Also one 6-request c1
  smoke on the ON arm (pinned client) for serving health. Expect (a) ≈48.3 GB,
  (b) ≈25 GB. If (b) does not clearly separate, record honestly and DO NOT land
  the effect claim. GPU rule: verify `nvidia-smi` idle; hold one flock for the
  series.
- **Axis credit:** NONE from the A/B. The binding Peak PSS/RSS axes remain
  `FAILED` until an authorized exact-grid rerun; the A/B is a mechanism check.

Exact commands: CPU `cmake --build <b> --target test_safetensors && ctest -R
test_safetensors`; full tools `python3 -m unittest discover -s tests/tools -p
'test_*.py'`; record `scripts/check-agent-record.py`,
`tests/scripts/test_agent_record.py`, `tests/scripts/test_doc_checkpoint.py`.

## Dependencies

None on other rows. Toolchain: Linux `madvise`/`mincore` (present). Hardware:
GB10 for the A/B only (CPU tests need no GPU). Model: 27B NVFP4 snapshot on dgx
for the A/B. No new licenses.

## Work breakdown

- W1: reader primitive + gate + test seam (header + cpp).
- W2: RED tests in `test_safetensors.cpp` (mincore + gate + correctness +
  neighbor-safety).
- W3: instrument the 27B + 35B copy helpers with `MaybeReleaseSourcePages`.
- W4: CPU gates + record surfaces + code checkpoint push (A/B disposition
  PENDING).
- W5: dgx VmHWM A/B from the pushed SHA; second docs checkpoint records the
  measured numbers + disposition.

W1–W4 are one atomic checkpoint (small, non-overlapping within this claim's
file set); W5 is the follow-up evidence checkpoint (two-push pattern).

## Risks/decisions

- No product calls; the behavior mirrors vLLM's stream-then-copy memory profile.
- Rollback: `VT_LOAD_WINDOWED_RELEASE=0` restores retain-all-source-pages.
- Risk: on a filesystem where re-fault would matter, edge-page-only release
  keeps re-faults at zero for the copied ranges (interior pages are never read
  again). Risk: `madvise` failure is non-fatal (page merely stays resident — a
  memory, not correctness, regression); the return is deliberately ignored.
