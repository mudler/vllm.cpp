# Handover: closing the ROCm Q4_K/Q6_K WMMA prefill gap vs llama.cpp

Written 2026-09-06, end of session. Purpose: let a fresh context (no prior
conversation memory) resume this investigation without re-deriving anything
below. This is a navigation index into the real record
(`.agents/specs/kernel-quant-ciq-gemm-rocm.md`'s `## Now`/`## Owed`
sections, and the PRs/issues cited) — it does not duplicate their content,
it points at it.

## Read this first

- Row: `KERNEL-QUANT-CIQ-GEMM-ROCM` (already `ACTIVE`).
- Full spec, with every measurement below in detail:
  [`kernel-quant-ciq-gemm-rocm.md`](kernel-quant-ciq-gemm-rocm.md) — read its
  `## Now` section top to bottom; it is written in chronological waves and
  each one states what was measured, accepted, or rejected, with numbers.
- This machine is `isravale` (the only host with the relevant GPU, an AMD
  RX 9060 XT / gfx1200 / ROCm 7.2.3 — RDNA4). Everything here is
  gfx1200/gfx1201-only by construction (`#if defined(__gfx1200__) ||
  defined(__gfx1201__)` in `rocm_grouped_gemm.hip`).

## The question this whole thread answers

Issue [#2109](https://github.com/mudler/vllm.cpp/issues/2109)'s original
WMMA port made ROCm Q4_K/Q6_K keep-quant prefill 3-21x faster than the
scalar arm it replaced (PRs #2990/#2991, merged) — but a same-tool
`rocprofv3` trace against llama.cpp's own `mul_mat_q` on the identical
checkpoint showed we were still ~11x slower than llama.cpp on prefill
specifically (issue #3032). Everything since has been chasing that
remaining gap, one measured hypothesis at a time.

## State of the four PRs right now (all still open, none merged)

| PR | What | Verdict |
|---|---|---|
| [#3033](https://github.com/mudler/vllm.cpp/pull/3033) | Widen the WMMA block 4→8 warps, alone | **REJECTED**: geomean -4.3% |
| [#3035](https://github.com/mudler/vllm.cpp/pull/3035) | Spec only: cooperative activation share design | Landed as a spec; its own implementation (below) came back negative |
| [#3036](https://github.com/mudler/vllm.cpp/pull/3036) | `Shared` (8x reuse) + `BigTile` (24x reuse, wider tile) | `Shared` **REJECTED** (-16% geomean); `BigTile` **ACCEPTED** (+16.0% geomean, -12.5% real-model, re-measured after the barrier repair) — **five review passes, fourteen findings; thirteen repaired, one held by the operator** |

**#3036 has been through FIVE full review→fix cycles; check `gh pr view
3036` for anything that has happened since.** The first two reviewers
(protocol: `.agents/prompts/reviewer.md`, static read plus real mutation
testing, not just reading the diff) found real defects that the green
suite was structurally incapable of seeing. The third, fourth and fifth
found no correctness defect, only eight further ways this record, the
code's comments and the pull request body had drifted from the code.
Three of the eight were themselves drift inside a CORRECTION of earlier
drift: pass 4's LOW, and both of pass 5's record findings.
The full account, with every number, is the spec's
`### Review cycle on #3036` section. Read that; this is the index entry.

- **Pass 1, head `bc5d494ce`, verdict FAIL, one HIGH**: an out-of-range
  device read in `BigTile`'s activation staging for the ragged last M-block
  (`m_tiles % ItGroup != 0`, the common case). It never corrupted *output*
  — those rows are never consumed — so every test was green; found by
  tracing address arithmetic. Fixed in `35f9400bd` (staging clamped to
  `min(ItGroup*16, (m_tiles-it_base)*16)`). The STATEMENT of it has since
  been corrected twice, and pass 4's is the one to read. What it broke in
  every case is the LOGICAL bound: the kernel is defined over
  `m_tiles*16` activation rows. Whether it ALSO left the allocation
  depended on the pool. The unclamped read ran up to `(ItGroup-1)*16` = 32
  rows past `m_tiles*16`, and the grow-only scratch keeps its high-water
  mark in BYTES. So the read was past the allocation in the exactly-sized
  case and in every insufficiently-grown one, and in-pool-but-stale only
  where capacity happened to cover that overrun. Pass 3's "not the
  allocation, because the pool had grown" was too narrow. The full
  statement, with the counterexample, is the spec's pass-1 paragraph.
- **Pass 2, head `8b25f30c8`, five findings**: a HIGH cross-warp
  write-after-read race on the block-shared `act_stage` (no barrier at the
  end of the `sb` loop, affecting all FOUR cooperative kernels), two
  record/comment contradictions, no runnable guard for the pass-1 OOB
  class, and a test skip condition that omitted `VT_ROCM_QUANT_WMMA_WIDE`.
  All repaired in one commit, which also RE-MEASURED both performance axes
  because the barrier changed the inner loop.
- **Pass 3, head `b795c70fd`, three MEDIUM findings, no correctness
  finding**: all three were record or comment drift — a comment claiming
  the `OobProbe` instantiation costs no LDS when it costs 256 B and leaves
  that arm 64 B under the limit, a `## Owed` entry still presenting the
  superseded first-cut geomean as current, and a clamp justification that
  named the scratch allocation rather than the logical row bound it
  actually enforces. Repaired without changing kernel behaviour: the only
  executable change is the LDS budget `static_assert`, which now bounds
  the probe instantiation, the tightest one this file emits.
- **Pass 4, head `cff373abe`, one MEDIUM and one LOW, no correctness
  finding**: only the LOW landed on pass 3's own repair. The MEDIUM
  landed on prose authored at `bc5d494ce` that the re-measurement commit
  never came back for: this file still argued the K-chunking case from
  the superseded 6.98x gap rather than the re-measured 7.12x. The LOW is
  that pass 3's replacement for the clamp justification drew a "grown
  pool means in-pool, exactly-sized means overrun" dichotomy that is
  false in both directions, which the pass-1 bullet earlier now states
  correctly. Repaired with no executable change at all.
- **Pass 5, head `6074ba02b`, two MEDIUM and one LOW, no correctness
  finding**: pass 4's repair claimed both of pass 4's findings had landed
  on pass 3's repair, which `git blame` refutes and which that same
  commit contradicted 180 lines lower; and it claimed every claim in the
  spec's review section cites a `file:line` when only three of them do.
  Both corrected here, with no executable change. The third finding is the
  pull request body carrying superseded figures; it is off this branch
  and the operator holds it.

**If you are picking this up fresh:** `bc5d494ce` has a real OOB read and
`8b25f30c8` has a real race. The head after pass 2 is the first one no
review pass has found a correctness defect in, and three further passes
have since read it without finding one. What those three passes DID find,
every time, is drift between this record, the kernel comments and the
kernel. Treat the correctness as reviewed. Re-derive any load-bearing
sentence here from the code before you rely on it.

## What's proven, in the order it was learned

1. **Wider block alone doesn't help** (issue #3032, PR #3033). Each warp in
   `KQuantGemmKWmmaQ4K`/`Q6K` already owns a fully independent 16x16 tile
   with zero cross-warp sharing; packing more of them into one block just
   spreads the same redundant-load pattern wider. Measured, geomean -4.3%.

2. **Read llama.cpp's actual source rather than guessing** (issue #3034).
   `mul_mat_q<Q4_K, I=128, J=128>` cooperatively loads ONE big tile per
   block and shares it across 8 warps x 8 activation sub-tiles each (~64x
   reuse per staged load). Confirmed via `ggml/src/ggml-cuda/mmq.cuh`,
   `mmq-config-rdna4.cuh`, `mmq-load-tiles.cuh` — not inferred.

3. **Activation-sharing alone (8x reuse) is also a net regression**
   (`KQuantGemmKWmmaQ6KShared`/`Q4KShared`, on `COOPTILE-w1`). Three copy
   implementations tried (naive byte copy → per-row loop → flat word loop);
   the result barely moved between them, which is itself evidence the cost
   is the staging mechanism (one more shared-memory round trip + sync per
   superblock), not a coding inefficiency. Geomean -16% vs default, -11.5%
   vs the already-rejected wide-block arm.

   **Correctness caveat, found and fixed the hard way:** the repo's
   *existing* WMMA tests use N=48 (`n_tiles=3`), which never satisfies
   `n_tiles % WarpsPerBlock(8) == 0` — the precondition both `Shared` and
   `BigTile` need. Every earlier "46/46 passing" claim for `Shared` was
   real but hollow: those tests were silently falling back to the plain
   kernel, never exercising `Shared`'s actual code. Fixed by adding
   per-variant dispatch counters (`g_kq_wmma_share_dispatches` etc., see
   `rocm_grouped_gemm.hip`) and two new tests at N=128/M=80 that prove the
   specific kernel launched (`test_backend_cross_device.cpp`, search
   `"cooperative-tile arms"`). **If you write a new WMMA variant, use a
   shape with `n_tiles % 8 == 0`, or you are testing the fallback.**

4. **ISA-level check: read the actual compiled machine code on both
   sides**, not just source. Recipe below. Finding: llama.cpp uses MORE
   registers than us (247 vs our 192 for Q4_K, 202 vs our 142 for Q6_K)
   with zero spilling, ruling out "fewer registers → more occupancy" as
   their advantage. We do have one real spill (Q4_K, 60B) but traced it to
   one-time setup/epilogue code, outside the per-superblock loop — cosmetic,
   not a meaningful cost. This corroborates the structural story rather
   than finding a codegen bug.

5. **Why llama.cpp's 128x128 tile fits 64KiB LDS and a naive copy of our
   design doesn't (86.5KiB): it isn't a smaller encoding, it's a narrower
   K-chunk.** `block_q8_1_mmq` (144B/128 elements) and our `BlockQ8_K`
   (292B/256 elements) have nearly identical bytes/element density. The
   difference: llama.cpp stages only 32 (weight) or 128 (activation)
   K-elements at a time (`MMQ_TILE_NE_K=32`), re-staging up to 8x more
   often than our 256-wide-superblock-at-once approach, in exchange for a
   much smaller peak footprint that affords a wider tile. Exact formula and
   arithmetic in the spec's `## Now`, "BigTile" section.

6. **BigTile: `ItGroup=3` (48 activation rows, ~24x reuse) is the largest
   that fits our EXISTING 256-wide staging inside 64KiB, and it wins.**
   `KQuantGemmKWmmaQ6KBigTile`/`Q4KBigTile<OutT, WarpsPerBlock, ItGroup>`,
   2D grid (`blockIdx.x`→fixed `jt` per warp as before, `blockIdx.y`→shared
   `it_base` spanning `ItGroup`, looped inside each warp). Weight dequant
   (and Q4_K's scale/min unpack) now execute once per superblock and serve
   all `ItGroup` iterations — reuse `Shared` didn't have either.

   Numbers below are the RE-MEASUREMENT taken after the pass-2 barrier
   repair; the first cut's figures are in parentheses, and the difference
   between them is what that barrier cost.

   - Op-level (`quant-gemm-bench`, best-of-4): **geomean +16.0%** (was
     +16.8%) vs shipping default over the six non-tail shapes, +17.2% over
     all eight; the ragged non-16-aligned tail shapes came in slightly
     ahead of the aligned ones at +20-22%.
   - Real model (`Ornith-1.5-9B-Q4_K_M.gguf`, isolated prefill,
     `rocprofv3`): prefill total kernel time **-12.5%** (was -14.0%); the
     isolated quant-GEMM-kernel gap vs llama.cpp's `mul_mat_q` narrowed
     from **8.89x to 7.12x slower** (was 8.92x to 6.98x). The oracle and
     our own default arm were re-run in the same session and reproduce the
     first cut to within noise, so these deltas are the barrier and not
     drift.
   - Correctness: `ctest -R rocm|cross_device` 8/8, and the full
     `test_backend_cross_device` binary 48/48 in each of the four toggle
     configurations, via the real (N=128/M=80) tests, not the stale N=48
     ones.
   - **Two caveats this section did not know when it was first written:
     `bc5d494ce` had a real out-of-range read and `8b25f30c8` a real
     cross-warp race, both invisible to every assertion above. See "State
     of the four PRs" up top.**
   - Still default OFF, behind `VT_ROCM_QUANT_WMMA_WIDE=1
     VT_ROCM_QUANT_WMMA_BIGTILE=1`. Whether to flip the default, and
     landing this PR, is undecided — see "What's not done" below.

## What's NOT done — the next traceable step

**Finer K-chunking on our own side.** BigTile kept this row's existing
256-wide-superblock-at-once staging for both weight and activation, and
only shrank the *reuse width* (`ItGroup`) to fit budget. It never adopted
llama.cpp's actual lever — staging narrower K-slices (item 5 above) — which
would shrink BOTH operands' per-load footprint and could afford `ItGroup`
much closer to llama.cpp's 8, closing more of the remaining 7.12x gap.
7.12x is the re-measured figure recorded earlier in this section. This
line quoted the superseded first cut's 6.98x until pass 4.

This is a **materially bigger change than BigTile was**: it means
restructuring `w_stage`/`DequantQ6KGroup16`/`DequantQ4KTile16` (the
weight-side staging every wave of this row has left untouched so far) to
work in, say, 128-wide half-superblocks instead of 256-wide, with twice as
many stage+sync rounds per superblock. Concretely:
- Weight tile at 128-wide chunking: roughly half of today's ~50KiB (8
  warps) → ~25KiB.
- Activation tile at 128-wide chunking, `ItGroup=8` (matching llama.cpp):
  8*128*292/2 ≈ Recompute exactly before trusting this estimate — do the
  arithmetic the way the spec's BigTile section does, don't assume.
- Whether 2x the sync frequency costs more than the wider reuse buys is
  an open, unmeasured question — exactly the shape of question this row's
  last three waves have each gotten wrong by intuition and right by
  measurement.

Do **not** skip straight to implementing this. Follow the same discipline
the last three waves used: read the exact mechanism, do the arithmetic,
write a spec update (or a fresh spec section) before code, implement behind
a new toggle (`VT_ROCM_QUANT_WMMA_*`, add it to
`scripts/env-doc-allowlist.txt` or `check-env-doc.py` will fail preflight),
prove correctness with a shape that actually satisfies whatever new
precondition it needs, then measure — op-level first, real-model second.

## Exact reproduction recipes

All under the GPU file mutex (`isravale` is not a fleet device):
`flock -n ${GPU_LOCK:-$HOME/gpu.lock} -c '<command>'`.

**Build** (from the worktree root):
```sh
cmake -S . -B build-hip -DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1200 -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j 4 --target test_backend_cross_device quant-gemm-bench vllm-cli
```

**Correctness** (the real, precondition-satisfying test, not the stale
N=48 ones):
```sh
VT_ROCM_QUANT_WMMA_WIDE=1 VT_ROCM_QUANT_WMMA_BIGTILE=1 \
  ./build-hip/tests/test_backend_cross_device --test-case="*cooperative-tile*"
# swap BIGTILE=1 for SHARE_ACT=1 to re-check the rejected arm
```
`VT_ROCM_QUANT_WMMA_WIDE=1` is part of the precondition, not decoration:
neither cooperative arm is dispatched without the 8-warp block. Without it
these cases MESSAGE-and-return rather than run.

**Re-running the race reproduction** (the one defect in this row with no
permanent runnable guard — a race is invisible until the warps drift, and
on this workload they do not drift by themselves). In
`rocm_grouped_gemm.hip`, immediately before the final `act_stage` read in
any of the four cooperative kernels (the `acc_f` accumulation at the end of
the `sb` loop body), insert a timing-only delay for one warp:
```c++
if (wy == 0) { for (int z = 0; z < 64; ++z) __builtin_amdgcn_s_sleep(127); }
__asm__ __volatile__("" ::: "memory");
```
That adds and removes no memory operation. With the `__syncthreads()` at
the end of the `sb` loop deleted, the four kernels return NMSE 0.050182 /
0.0555232 / 0.104243 / 0.113616 against a 0.0005 tolerance; with it
restored and the identical delay still present, all four are green. Remove
the probe afterwards.

**Op-level A/B** (`examples/quant-gemm-bench`, no args, best-of-N by
running multiple times and taking the max per shape):
```sh
VT_ROCM_QUANT_WMMA_WIDE=1 VT_ROCM_QUANT_WMMA_BIGTILE=1 ./build-hip/examples/quant-gemm-bench
```
Toggle combinations: unset both = shipping default (4-warp); `WIDE=1` alone
= rejected 8-warp-no-share; `WIDE=1 SHARE_ACT=1` = rejected Shared;
`WIDE=1 BIGTILE=1` = accepted BigTile.

**Real-model isolated-prefill trace** (checkpoint at
`/home/justin/Downloads/Ornith-1.5-9B-Q4_K_M.gguf`; needs the GPU
exclusively — `llama-server.service` (a real production systemd user
service) also uses this GPU; check `systemctl --user is-active
llama-server` and ask the user before stopping/starting it, and restart it
after):
```sh
rm -rf /tmp/rocprof-<label>
VT_ROCM_QUANT_WMMA_WIDE=1 VT_ROCM_QUANT_WMMA_BIGTILE=1 /opt/rocm/bin/rocprofv3 \
  --kernel-trace --stats -d /tmp/rocprof-<label> -f rocpd -- \
  ./build-hip/examples/vllm-cli --model /home/justin/Downloads/Ornith-1.5-9B-Q4_K_M.gguf \
  --prompt "<a real ~512-token prompt>" --max-tokens 1 --device auto --repeat 3 --temperature 0
```
Then query the per-kernel breakdown:
```sh
DB=$(find /tmp/rocprof-<label> -iname "*_results.db" | head -1)
sqlite3 -header -column "$DB" "SELECT name, total_calls, total_duration, percentage FROM top_kernels LIMIT 15;"
sqlite3 "$DB" "SELECT SUM(K.end-K.start)/1000.0 FROM rocpd_kernel_dispatch K;"   # total kernel time, all reps, microseconds
```
A 1-token prompt instead of the 512-token one isolates decode instead of
prefill (`llama-bench -p 0 -n 128` on the oracle side for the matching
decode-only number).

**Oracle (llama.cpp, pinned)**: binary at
`/tmp/build-llamacpp-b10451/bin/llama-bench` — **this lives in `/tmp` and
may not survive a reboot.** If missing, the source is a linked worktree at
`/tmp/llama-cpp-b10451` off `/home/justin/Projects/llama.cpp`, pinned to
tag `b10451` (commit `10bf611e533d81f739128304991c5e133c6aebd8`) per
`.agents/upstream-sync.md`; rebuild with `-DGGML_HIP=ON
-DAMDGPU_TARGETS=gfx1200`.
```sh
/tmp/build-llamacpp-b10451/bin/llama-bench -m /home/justin/Downloads/Ornith-1.5-9B-Q4_K_M.gguf -p 512 -n 0 -ngl 99 -r 5   # prefill-only
/tmp/build-llamacpp-b10451/bin/llama-bench -m /home/justin/Downloads/Ornith-1.5-9B-Q4_K_M.gguf -p 0 -n 128 -ngl 99 -r 5   # decode-only
```

**ISA comparison** (read the actual generated machine code, don't infer):
- Ours: recompile the exact translation unit with `-save-temps` (grab the
  real compile command from `build-hip/compile_commands.json` first, don't
  guess flags), producing a human-readable `.s` file with
  `.amdhsa_kernel`/`num_vgpr`/`private_seg_size` per instantiation.
- llama.cpp's: their per-`ggml_type` kernels are compiled into SEPARATE
  concatenated clang-offload bundles inside `.hip_fatbin` (134 of them in
  `libggml-hip.so`). Extract the section
  (`objcopy -O binary --only-section=.hip_fatbin`), find offsets of each
  `__CLANG_OFFLOAD_BUNDLE__` magic string, grep each slice's strings for
  `ggml_type(\d+)E` to identify which one is Q4_K (12) / Q6_K (14), unbundle
  the right one with `clang-offload-bundler --unbundle --type=o
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1200`, then
  `llvm-readobj --notes <hsaco>` to dump the AMDGPU metadata (msgpack,
  human-readable) with `.vgpr_count`/`.vgpr_spill_count` per kernel symbol.
  Full working sequence is in this session's transcript if you need the
  exact commands verbatim rather than re-deriving them.

## Toggles (all default OFF)

| Env var | Selects | Verdict |
|---|---|---|
| `VT_ROCM_QUANT_WMMA=0` | scalar arm, bypass WMMA entirely | — (pre-existing) |
| `VT_ROCM_QUANT_WMMA_WIDE=1` | 8 warps/block instead of 4 | rejected alone, required alongside the two below |
| `VT_ROCM_QUANT_WMMA_SHARE_ACT=1` | `Shared` kernel (8x activation reuse) | **rejected** |
| `VT_ROCM_QUANT_WMMA_BIGTILE=1` | `BigTile` kernel (24x reuse, `ItGroup=3`) | **accepted, not yet default** |

All of these need `scripts/env-doc-allowlist.txt` entries or
`check-env-doc.py` fails preflight — this bit twice in this session, watch
for it.

## Known environment gotchas from this session

- `scripts/agent-preflight.sh` (both plain and `--staged`) genuinely takes
  ~3-8 minutes on this box; it is not hung, don't kill it early. Its own
  self-test suite includes flaky, environment-sensitive cases unrelated to
  ROCm work: `test_tower_skip_rss_arm`, the `tools` suite's
  `test_drop_file_cache` (fails because `/tmp` here is tmpfs, and
  `posix_fadvise(DONTNEED)` is a no-op on tmpfs), and occasionally
  `test_cpu_x86_llamacpp_floor` — all pre-existing, unrelated to this row,
  confirmed by re-running in isolation.
- Running multiple `agent-preflight.sh` invocations concurrently corrupts
  results (a shared self-test mutates `.agents/benchmark-record.md` and can
  leave it dirty if killed mid-run — `git checkout --` it if you find stray
  uncommitted changes there that you didn't make).
- `isravale`'s GPU is not a fleet device — use the file mutex
  (`${GPU_LOCK:-$HOME/gpu.lock}`), not `rc`.
- The harness's own permission layer sometimes blocks `systemctl --user
  stop llama-server` even with prior in-conversation authorization for the
  same action — if that happens, ask the user to run it, don't work around
  it.
- **Green NMSE tests do not prove the absence of an out-of-range device
  read, and they do not prove the absence of a race either.** PR #3036's
  `BigTile` kernels passed every test with a real out-of-range read in
  them, and then passed every test again with a real cross-warp
  write-after-read race in them (see "State of the four PRs" above).
  Neither is a value defect on
  this workload: the stale read never feeds a value that reaches output,
  and the racing warps never happen to drift far enough apart. Both were
  found by reading the code — address arithmetic against the row range the
  kernel is defined over, and barrier coverage of every block-shared array
  — and then made visible by deliberate mutation. Device-side ASan for HIP
  is CDNA-only (not available for RDNA/gfx12, confirmed by the reviewer),
  so there is no sanitizer to lean on. Three habits follow, and this row
  paid for all three: any kernel that computes a global-memory address
  from a block/grid index needs its address range checked by hand against
  the LOGICAL extent it is defined over, which a grow-only scratch pool's
  physical capacity is not; any array written by the whole block needs a
  barrier between its last read in one iteration and its first write in
  the next, the loop back-edge included; and any claim about a kernel's
  LDS, register or occupancy cost is measured with
  `-Rpass-analysis=kernel-resource-usage` on the file's own compile
  command, never reasoned about — `__syncthreads_or` looked free and costs
  256 B a block. Async agents that stall
  waiting on their own backgrounded build/preflight/monitor and end their
  turn without actually finishing (rather than reporting real output) have
  happened more than once in this row's history — if you dispatch one and
  it reports "waiting on X" as its final message, resume it explicitly
  rather than assuming it will keep going on its own, and independently
  verify whatever it claims to have done before trusting it (this is also
  just "the operator reruns the gate," AGENTS.md's own rule, applied
  literally).
