// The two geometry constants of the `vt::Conv1d` CPU provider, and the block
// length derived from them (#672, #1334, #1664).
//
// WHY THEY ARE IN A HEADER RATHER THAN IN THE KERNEL'S ANONYMOUS NAMESPACE.
// `Conv1dTimeBlock` decides how many units of work the kernel hands the pool,
// and a gate that cannot evaluate the function cannot assert either of the two
// properties of its answer that the row rests on -- the tile alignment below,
// and that the shipped shapes produce more than ONE block, which is the
// difference between a case that crosses a boundary and a case that silently
// does not. `tests/vt/test_ops_conv1d_general.cpp` includes this header for
// exactly that.
//
// AND THE ALIGNMENT IS A PERFORMANCE INVARIANT, NOT THE BIT-IDENTITY ARGUMENT.
// An earlier revision of this comment said the bit-identity of the (time block,
// output row) decomposition rested on the multiple-of-`kConv1dPosTile` property.
// IT DOES NOT, and the review of #1678 proved that by mutation: misaligning the
// block leaves every arithmetic assertion in `test_ops_conv1d_general` green --
// 19 615 assertions, including both engineered cancellation cases -- and reds
// only the two property `CHECK`s that read the function's answer directly. The
// reason is in the kernel: a cell's reduction is `seed`, then `ic` ascending,
// then `k` ascending, and that sequence does not mention `t0`, so re-cutting the
// tiles cannot move a bit. What the alignment buys is that a tile which takes
// the `span == kConv1dPosTile` constant-trip-count path today still takes it
// after the change -- .agents/specs/minimax-music3.md §18.4 prices that path at
// up to 5x, so a block length that silently re-cut the tiles would report a
// SPEED result about a different kernel. The property gate stays, for that
// reason rather than for the one this comment used to give.
#pragma once

#include <cstdint>

namespace vt::cpu {

// How many OUTPUT POSITIONS share one sweep of the (ic, k) weights. It is not a
// blocking heuristic and it is not tunable at run time: it is the number of
// INDEPENDENT accumulator chains the kernel offers the machine, and the whole
// of #1334 is that the shipped loop offered exactly one. It measured best of
// {8, 16, 32, 64} at the vocoder's shapes on both -O2 and -O3 — measured while
// these accumulators were f64. The tile was NOT re-swept after #1474 halved
// their width, which doubles how many fit one vector register and may well move
// the optimum. Recorded as owed rather than assumed away:
// .agents/specs/vt-conv1d-f32-accumulator.md §7.
constexpr int64_t kConv1dPosTile = 32;

// THE ACTIVATION SLICE ONE UNIT OF WORK IS ALLOWED TO TOUCH, in bytes, and the
// whole of the blocking decision. It is a CACHE BUDGET rather than a tuned
// constant: the block length is derived from it and from the geometry, so a
// convolution over 96 input channels gets a long block and one over 1024 gets a
// short one, and neither is written down anywhere.
//
// WHY THE DECOMPOSITION HAS A SECOND AXIS AT ALL. The shipped loop partitions
// OUTPUT ROWS and nothing else, so every thread sweeps the WHOLE input tensor
// for its own slice of output channels and the reuse available across output
// channels is never taken. At the MiniMax-Music3 vocoder's b3 residual geometry
// and 344 latent frames that asks the machine for a 67.6 MiB activation 96
// times per convolution, against a `thor:gpu0` core that has 64 KiB of L1d,
// 1 MiB of PRIVATE L2 and NO shared last-level cache at all — so every one of
// those 96 sweeps is a DRAM sweep. Blocking the time axis turns them into ONE
// DRAM sweep plus a weight re-read per block.
//
// THE VALUE IS MEASURED, NOT PICKED. `tools/bench/conv1d_scaling_probe.cpp`
// sweeps the footprint at each of the vocoder's own geometries and reports the
// knee; `.agents/specs/vt-conv1d-time-block.md` §2 carries the table this came
// from. Half of L2 is the budget rather than all of it, because the weight rows
// of the output channels in flight stream THROUGH the same L2 and a slice sized
// to fill it evicts itself.
constexpr int64_t kConv1dSliceBytes = 512 * 1024;

// Output positions per time block, or `length` for "do not block at all".
//
// TWO RULES, AND BOTH ARE MEASURED RATHER THAN PICKED.
//
// (1) BLOCK ONLY WHEN THE WEIGHTS ARE THE SMALLER TENSOR. Blocking trades one
// cost for another: the unblocked loop reads the activation `out_channels`
// times and the weights once, and the blocked loop reads the activation once
// and the weights once per block. That trade is only worth taking when what it
// makes resident is the SMALLER of the two tensors — weights
// `out_channels * in_per_group * kernel` against activation
// `in_per_group * in_len`, so the condition reduces to
// `out_channels * kernel <= in_len` with no constant in it at all.
//
// MEASURED, paired, on `thor:gpu0` at 14 threads over the vocoder's eleven
// geometries (spec §2b). Blocking unconditionally is 1.26x, 1.32x, 1.78x and
// 2.04x on the four residual shapes where the activation dominates, and 10.49x
// on `conv_out` — and **0.82x and 0.89x on the two b0 shapes, where the weights
// are 16.5 MiB against a 2.1 MiB activation.** The condition above is the sign
// of that same comparison, and it selects correctly on all eleven: it declines
// `conv_in` (10 752 > 92), `dec_in_proj` (1 024 > 92) and both b0 shapes
// (5 376 > 694, 768 > 694), and takes every shape that gained.
//
// (2) THE BLOCK IS THE LARGEST MULTIPLE OF THE POSITION TILE THAT FITS
// `kConv1dSliceBytes`. The multiple is load-bearing and is not rounding: every
// block boundary is then also a position-tile boundary, so each tile spans
// exactly the positions it spans today and takes exactly the code path it takes
// today — including the `span == kConv1dPosTile` constant-trip-count fast path,
// which §18.4 prices at up to 5x. A block length that was not a multiple would
// silently re-cut the tiles and report a speed result about a different kernel.
// `tests/vt/test_ops_conv1d_general.cpp` gates both rules.
inline int64_t Conv1dTimeBlock(int64_t in_per_group, int64_t out_channels, int64_t kernel,
                               int64_t stride, int64_t dilation, int64_t in_len, int64_t length) {
  // Rule (1). `length` means one block, which is the pre-decomposition loop.
  if (out_channels * kernel > in_len) return length;
  // One block of `b` output positions reads input positions
  // [t0*stride - pad, t0*stride - pad + (b-1)*stride + (kernel-1)*dilation], so
  // the span grows with `stride` and the constant term with the dilated kernel.
  const int64_t row_bytes = in_per_group * stride * 4;
  const int64_t fixed = in_per_group * ((kernel - 1) * dilation + 1) * 4;
  int64_t budget = kConv1dSliceBytes - fixed;
  if (budget < row_bytes) budget = row_bytes;  // one tile minimum, always
  int64_t block = budget / row_bytes;
  block = block / kConv1dPosTile * kConv1dPosTile;
  if (block < kConv1dPosTile) block = kConv1dPosTile;
  if (block > length) block = length;
  return block;
}

}  // namespace vt::cpu
