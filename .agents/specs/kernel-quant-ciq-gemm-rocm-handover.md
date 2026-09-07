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

## State of the three PRs right now (all still open, none merged)

| PR | What | Verdict |
|---|---|---|
| [#3033](https://github.com/mudler/vllm.cpp/pull/3033) | Widen the WMMA block 4→8 warps, alone | **REJECTED**: geomean -4.3% |
| [#3035](https://github.com/mudler/vllm.cpp/pull/3035) | Spec only: cooperative activation share design | Landed as a spec; its own implementation (below) came back negative |
| *(not yet opened)* | `Shared` (activation-only sharing, 8x reuse) + `BigTile` (24x reuse, wider tile) | `Shared` **REJECTED** (-16% geomean); `BigTile` **ACCEPTED** (+16.8% geomean, -14% real-model) |

**The third PR does not exist yet.** All of `Shared` and `BigTile`'s code,
tests, and spec updates are committed on branch
`row/KERNEL-QUANT-CIQ-GEMM-ROCM-COOPTILE-w1` (worktree:
`/home/justin/Projects/vllm.cpp-KERNEL-QUANT-CIQ-GEMM-ROCM-COOPTILE-w1`) —
check `git log --oneline -3` there and `gh pr list` to see whether a
session between this one and yours already pushed/opened it. If not, that
worktree's HEAD is the thing to push and open a PR from (base:
`row/KERNEL-QUANT-CIQ-GEMM-ROCM-COOPTILE`, which is #3035's already-merged
spec — check whether #3035 itself has merged by the time you read this; if
not, this branch is base-reachable from it but not from `main`, same
pattern as `KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT-RDNA4` used earlier in this
row's history).

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

   - Op-level (`quant-gemm-bench`, best-of-4): **geomean +16.8%** vs
     shipping default; holds on the ragged non-16-aligned tail shape too
     (+19-20%).
   - Real model (`Ornith-1.5-9B-Q4_K_M.gguf`, isolated prefill,
     `rocprofv3`): prefill total kernel time **-14.0%**; the isolated
     quant-GEMM-kernel gap vs llama.cpp's `mul_mat_q` narrowed from **8.92x
     to 6.98x slower**.
   - Correctness: `ctest -R rocm|cross_device`, 48/48, 84084/84084
     assertions, zero regression, via the real (N=128) tests, not the
     stale N=48 ones.
   - Still default OFF, behind `VT_ROCM_QUANT_WMMA_WIDE=1
     VT_ROCM_QUANT_WMMA_BIGTILE=1`. Whether to flip the default, and
     landing this PR, is undecided — see "What's not done" below.

## What's NOT done — the next traceable step

**Finer K-chunking on our own side.** BigTile kept this row's existing
256-wide-superblock-at-once staging for both weight and activation, and
only shrank the *reuse width* (`ItGroup`) to fit budget. It never adopted
llama.cpp's actual lever — staging narrower K-slices (item 5 above) — which
would shrink BOTH operands' per-load footprint and could afford `ItGroup`
much closer to llama.cpp's 8, closing more of the remaining 6.98x gap.

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
