#pragma once
// SPEC-DFLASH2 #2202: the DFlash block attention's QUERY-BLOCK GRID.
//
// The MMA kernel used to tile the query axis globally: block `b` owned rows
// `[b*64, b*64+64)` of the whole batch. At c=8 with `1+k = 9` query rows per
// request that is `Tq = 72` rows in two blocks, and the FIRST block spans all
// eight requests. A block stages the UNION of its rows' key ranges, so that
// block walked the entire combined sequence — about 303 32-key tiles at ctx
// 2048 — while each of its rows is live for only its own request's ~38. Roughly
// 87% of the MMA lanes were masked, and the layer's latency was that one
// block's serial chain.
//
// Tiling PER REQUEST instead makes a block's union exactly one request's key
// run. Same bytes and same arithmetic per (row, key) pair; a much shorter
// critical path and more blocks to fill the SMs.
//
// The mapping lives here, in a header, because it is the part that can be wrong
// in a way no CUDA-free machine could otherwise catch: `tests/vt/test_dflash_attn_grid.cpp`
// asserts it covers every query row exactly once across ragged shapes, on the CPU.
#include <cstdint>

#if defined(__CUDACC__)
#define VT_DFLASH_GRID_HD __host__ __device__
#else
#define VT_DFLASH_GRID_HD
#endif

namespace vt {

// Query rows one MMA block owns: `kMmaWarps * kMmaQ` in the kernel.
inline constexpr int64_t kDFlashQueryBlockRows = 64;

struct DFlashQueryBlock {
  int64_t req = 0;    // which request this block serves
  int64_t qblk = 0;   // first query row, ABSOLUTE
  int64_t qend = 0;   // one past the last, ABSOLUTE, clamped to the request
  bool live = false;  // false when the tile lies past this request's rows
};

// Widest request in query rows, which fixes how many tiles each request gets.
// A ragged batch pads to the widest, and the padded blocks return immediately.
VT_DFLASH_GRID_HD inline int64_t DFlashQueryTilesPerReq(const int32_t* qcu, int num_reqs,
                                                        int64_t block_rows) {
  int64_t widest = 0;
  for (int r = 0; r < num_reqs; ++r) {
    const int64_t n = static_cast<int64_t>(qcu[r + 1]) - static_cast<int64_t>(qcu[r]);
    if (n > widest) widest = n;
  }
  if (widest <= 0) return 0;
  return (widest + block_rows - 1) / block_rows;
}

// `blockIdx.x` -> (request, query-row span). `tiles_per_req` must be what
// DFlashQueryTilesPerReq returned for the same `qcu`.
VT_DFLASH_GRID_HD inline DFlashQueryBlock DFlashResolveQueryBlock(const int32_t* qcu,
                                                                 int num_reqs,
                                                                 int64_t tiles_per_req,
                                                                 int64_t block_rows,
                                                                 int64_t bx) {
  DFlashQueryBlock b;
  if (tiles_per_req <= 0) return b;
  const int64_t r = bx / tiles_per_req;
  if (r >= num_reqs) return b;
  const int64_t tile = bx - r * tiles_per_req;
  const int64_t qs = static_cast<int64_t>(qcu[r]);
  const int64_t qe = static_cast<int64_t>(qcu[r + 1]);
  const int64_t start = qs + tile * block_rows;
  b.req = r;
  b.qblk = start;
  b.qend = (start + block_rows < qe) ? (start + block_rows) : qe;
  b.live = start < qe;  // a padded tile of a narrower request
  return b;
}

}  // namespace vt
