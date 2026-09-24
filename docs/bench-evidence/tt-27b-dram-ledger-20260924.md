# The 27B decode DRAM ledger — growth attributed (2026-09-24, P150a)

**STATUS: DRAFT for operator review.** Row `BACKEND-TENSTORRENT`, spec
`.agents/specs/tenstorrent-27b-dram-growth.md` (W1: attribute, don't fix),
branch `row/TT-27B-DRAM-GROWTH` at `eb30b1fb9`. Issue
`ISSUE-LOCAL-01M3918K0WTSCZ2X1S2WZA5NHW`. Nothing was fixed on this branch;
instruments and measurement only.

## Verdict up front

The decode DRAM exhaustion is **per-REQUEST accumulation, triggered inside the
keepquant repair web, retained OUTSIDE our slot table**. Free DRAM is flat
across every decode step within a request and drops ~950 MB once per new
request, during that request's FIRST decode step(s), with the retained drops
tagged `kq-decode/repair` (`src/vt/tenstorrent/tenstorrent_keepquant.cpp:446`).
The `DumpSlotCensus` census is FLAT across the same span (dev 10.529 GiB in 430
slots, constant), so the retained tensors are held by tt-metal-side owners
(program/op cache under `VT_TT_PROGRAM_CACHE=1`), not by any `Slots()` entry.
Per the spec's interpretation table this binds the follow-up row to the
"per-request unbalanced alloc in the keepquant path, retained outside our slot
table" class, with the tt-metal program-cache owner named as the retention
suspect. Not fragmentation (total free and largest-free-block fall together);
not per-step (flat within a request); not our keyed slot caches (census flat).

## Host, build, revision (the revision is part of the result)

- Host: personal Tenstorrent Blackhole P150a workstation (aarch64, GCC 16).
  Every device command under `flock -x $HOME/gpu.lock`, `luwen reset` +
  `sleep 15` inside the hold; `source ~/Sources/tt/env-tt-common.sh`.
- Tree: `row/TT-27B-DRAM-GROWTH` at `eb30b1fb9` (spec commit; no product code
  on this branch). Release Ninja build `/tmp/row-dram-growth/build`
  (`-DVLLM_CPP_TENSTORRENT=ON -DVLLM_BUILD_TESTS=ON`, tt-metal build_Release
  cmake prefix); TT linkage verified (`nm vllm-bench | grep -c tenstorrent`
  = 396).
- tt-metal: `81f3bbf3b405fc46bbff59894c3ee42fa48c9d5c`
  (`v0.79.0-dev20260911-82-g81f3bbf3b40`, built 2026-09-18) — the SAME revision
  the 2026-09-23 INT8DOT session ran, so the margins are comparable.
- Model: `/mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf`
  (11,240,605,152 bytes, mtime 2026-09-11).
- Prompts: the committed fixture
  `tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json` (64 prompts).
- Instrument: `VT_TT_ALLOC_TRACE=1` plus the standing 27B recipe
  `VT_TT_AFFINE_F32=1 VT_TT_NORM_PAD=1 VT_TT_PROGRAM_CACHE=1`. Instrument
  anchors: `AllocTraceSnapshot` (`src/vt/tenstorrent/tenstorrent_capture.cpp:299-328`),
  `FreeDeviceDramBytesForTest` (`:390-396`), `DumpSlotCensus` (`:415-464`).
- Contention: device held exclusively under the mutex for the whole leg; no
  other GPU work on the host.

## DEVIATION 1 — batch>1 GDN prefill asserts on this tt-metal revision

Both the 16-request and 4-request concurrency shapes died at LOAD with
`TT_FATAL @ chunk_gdn_phased_program_factory.cpp:137: BH <= ncores — num_heads
480/192 exceeds compute cores 110` (BH = batch × heads inside
`ttnn::chunk_gated_delta_rule`, reached through `GdnPrefillKernel`,
`src/vt/tenstorrent/tenstorrent_gdn.cpp:214`). Batch 1 (BH ≤ 110) is the only
executable prefill lane on this revision, so the leg ran at
`--concurrency 1`. This is a device/tt-metal constraint discovered by this
leg, not a DRAM result; it needs its own row before any concurrency>1 leg can
re-run on 27B.

## The leg

One process: `--num-prompts 16 --output-len 16 --concurrency 1 --seed 0
--temperature 0 --ignore-eos`, traced. The process reproduced the OOM
signature exactly and died during request 5 (61 decode steps, 5 prefills):

```
TT_FATAL: Out of Memory: Not enough space to allocate 268369920 B DRAM buffer
across 8 banks ... (allocated: 4208599168 B, free: 63742208 B, largest free
block: 63742208 B)
TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:495
```

— the same ~268 MB ask and ~63 MB largest-free-block the spec records, in the
same decode path (`Qwen3_5DenseDecodeGraph::Step` → keepquant web).

## The ledger

Settled free DRAM at each request boundary (the settle mark is the LAST
`[TT-ALLOC]` total_free within each decode step; the trace line format is
`tenstorrent_capture.cpp:321-327`):

| Request | Settled total_free after its first steps | Δ vs previous request |
|---|---|---|
| load-complete / prefill 1 | 4.755 GiB (largest-free 596.5 MB) | — |
| request 2 (steps 2-16) | 3.812 GiB (473.4 MB) | **-964.8 MB** |
| request 3 | 2.885 GiB (357.2 MB) | -949.3 MB |
| request 4 | 1.958 GiB (237.2 MB) | -949.3 MB |
| request 5 (dies mid-request) | OOM at bank_manager.cpp:495, free 63.7 MB | -1.83 GiB to death |

Within-request behaviour (every decode step of requests 2-4): total_free and
largest_free are CONSTANT to the MiB across all 15 steps after the first —
e.g. request 2 holds 3.812 GiB / 473.4 MB from step 2 through step 16. The
growth is NOT per-step.

The per-request drop lands inside the first decode step(s) of the new request
as a staircase of retained allocations tagged `kq-decode/repair`
(`tenstorrent_keepquant.cpp:446`): repeated new-low stairs of -283.9 MB
(later -651.8 MB) that never come back. Transient allocations in the same web
(up to -298 MB, reclaimed +683 MB within the step) recover; the staircase
floor does not.

## Attribution per the spec's interpretation table

| Class | Verdict | Evidence |
|---|---|---|
| keyed cache in `Slots()` | **EXCLUDED** | `DumpSlotCensus` flat: dev = 10.529 GiB / 430 slots at request 2, 3, 4 and 5; pers grew by ONE 1.5 KB slot across four requests. The decode-graph parity ring is keyed by shape (S, q, spec), not request (`src/vllm/model_executor/models/qwen3_5.cpp:12020`); state/KV pools are preallocated slot-major (`src/vllm/v1/worker/gpu/runner.cpp:1884-1896`). |
| per-STEP accumulation | **EXCLUDED** | settled free flat across all steps within each request (ledger rows above). |
| fragmentation | **EXCLUDED** | largest_free tracks total_free proportionally (596→473→357→237→63 MB as total falls 4.76 GiB→0); no collapse of largest while total stays high. |
| per-request accumulation retained outside `Slots()` | **ATTRIBUTED** | -950 MB at each request boundary, tagged `kq-decode/repair` (`tenstorrent_keepquant.cpp:446`), invisible to the slot census → the retained owners are tt-metal-side (program/op cache under `VT_TT_PROGRAM_CACHE=1`). No kernel compiles fire after request 1 (the only `riscv-tt-elf-g++` invocations are at load and request 1's first step), so the retained region is cached device TENSORS in tt-metal's op/program caches, not new kernel binaries. |

Anchors for the keepquant path the drops are tagged in:
`DecodeKeepQuantWordsF32` entry and its trace tag
(`src/vt/tenstorrent/tenstorrent_keepquant.cpp:326-334`), the repair lambda and
its tag (`:446`), the reclaim webs (`:374-375`, `:387`, `:424`, `:431`) — the
reclaims cover tensors we own; whatever the tt-metal op cache retains per new
request-variant is not reclaimable from our side. Grouped-chunk callers:
`:1280`, `:1405`; chunk-loop bracket tags `:1259`, `:1338`.

## Growth rate

**~0.95 GiB per request** at concurrency 1 (settled free: 4.755 → 3.812 →
2.885 → 1.958 GiB across requests 1-4; -964.8/-949.3/-949.3 MB). The steady
margin after load is ~4.8 GiB, so exhaustion lands at request ~5.

### DEVIATION 2 — magnitude vs the 2026-09-23 session

The INT8DOT session's concurrency-4 processes died at ~request 14, implying
the much smaller per-request estimate (~4-5 MB) in the spec's problem
statement. This leg measures ~950 MB/request. That estimate was DERIVED from
death counts, not from a ledger; this ledger supersedes it. The class verdict
(growth with requests served, in the keepquant web, slots flat) is consistent
across both sessions. The concurrency dependence of the RATE (1 GiB/req at
concurrency 1 vs ~0.1 GiB/req implied at concurrency 4) is unexplained and
belongs to the follow-up row. Leg 2 (8-request repeat) was skipped: leg 1's
staircase is unambiguous, and the OOM signature reproduced in-leg.

## Reproduction

```sh
cd /tmp/row-dram-growth   # branch row/TT-27B-DRAM-GROWTH @ eb30b1fb9
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_BUILD_TESTS=ON -DVLLM_CPP_TENSTORRENT=ON \
  -DCMAKE_PREFIX_PATH="$HOME/Sources/tt/tt-metal/build_Release/lib64/cmake/tt-metalium;\
$HOME/Sources/tt/tt-metal/build_Release/lib64/cmake/fmt;\
$HOME/Sources/tt/tt-metal/build_Release/lib64/cmake/tt-nn;\
$HOME/Sources/tt/tt-metal/build_Release/lib64/cmake;\
$HOME/Sources/tt/tt-metal/build_Release/share/cmake"
cmake --build build -j 16
flock -x ~/gpu.lock bash -c '
  ~/Sources/tt/luwen/target/release/reset && sleep 15
  source ~/Sources/tt/env-tt-common.sh
  VT_TT_ALLOC_TRACE=1 VT_TT_AFFINE_F32=1 VT_TT_NORM_PAD=1 VT_TT_PROGRAM_CACHE=1 \
  ./build/examples/vllm-bench \
    --model /mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf \
    --dataset-path tests/fixtures/tt-int8dot-sweep-sharegpt-64-20260923.json \
    --num-prompts 16 --output-len 16 --concurrency 1 --seed 0 \
    --temperature 0 --ignore-eos 2>&1 | tee /tmp/dram-leg.log'
# ledger: grep "[TT-ALLOC]" /tmp/dram-leg.log   (settled total_free per request)
# census: grep "[TT-SLOT-CENSUS]" /tmp/dram-leg.log
```

Raw leg log: `/tmp/dram-leg1.log` (this session; ~125k trace lines).

## Stop conditions

Not triggered: the instruments DID see the growing region (the staircase is
fully visible); no escalation to tt-metal's profiler is needed. The row
proceeds to its fix row per the interpretation table.
