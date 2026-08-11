// Tenstorrent op providers — the ttnn adapter layer (BACKEND-TENSTORRENT,
// .agents/specs/tenstorrent-backend.md). vllm.cpp original; no upstream
// mirror (vLLM has no Tenstorrent platform). Op table: OPT-125m's 9 ops plus
// the Qwen3-dense deltas (kRmsNorm first; kSiluAndMul / RoPE / Cast next),
// matching Metal's OPT→Qwen3 sequencing. ttnn for compute where available;
// host-staged pure data-movement / attention for the remainder (see
// HOST-STAGED OPS note below).
//
// SCOPE: F32 for the W0 path unless noted. kAdd allows rank-1 bias
// broadcast; kLayerNorm optional rank-1 weight/bias; kRmsNorm weight +
// optional residual stream; kEmbedding i32/i64 ids. Every other shape/dtype
// is a VT_CHECK failure — no CPU reference tier (UnifiedMemory()==false).
//
// HOST-STAGED OPS (kReshapeAndCache, kPagedAttention): this backend's Alloc is
// host memory (tenstorrent_backend.cpp). ReshapeAndCache is a pure contiguous /
// stride-aware page write. PagedAttention uses the CPU-oracle f32 softmax over
// the host-resident paged cache; mapping vLLM's block-table contract onto
// ttnn::sdpa_decode is deferred. kQkvSplit is hybrid: device-slice when the
// merged qkv already has a resident shadow (post MatmulBT), else host memcpy.
#include "vt/backend.h"
#include "vt/cpu/cpu_threadpool.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tenstorrent/tenstorrent_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/embedding/embedding.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp>
#include <ttnn/operations/transformer/sdpa_config.hpp>
#include <ttnn/operations/trace.hpp>
#include <ttnn/common/queue_id.hpp>
// chunked_scaled_dot_product_attention lives in sdpa.hpp, but the installed
// TT-NN tree ships that header with a missing device/ include. Forward-declare
// the overload we call; the symbol is linked via TTNN::TTNN.
namespace ttnn::transformer {
ttnn::Tensor chunked_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q, const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v, const ttnn::Tensor& page_table_tensor,
    int64_t chunk_start_idx, std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride>
        paged_cache_geometry = std::nullopt);
}  // namespace ttnn::transformer
// experimental/paged_cache pulls op_profiler which expects a 6-arg
// ___tracy_alloc_srcloc (with color); the TracyC.h on this tree only has 5-arg.
// Temporarily disable Tracy for this include chain so the op headers compile.
#ifdef TRACY_ENABLE
#undef TRACY_ENABLE
#define VT_RESTORE_TRACY_ENABLE 1
#endif
#include <ttnn/operations/experimental/paged_cache/paged_cache.hpp>
#include <ttnn/operations/core/to_memory_config/to_memory_config_op.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#ifdef VT_RESTORE_TRACY_ENABLE
#define TRACY_ENABLE 1
#undef VT_RESTORE_TRACY_ENABLE
#endif

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

namespace vt::tenstorrent {
namespace {

// ---- Host/device residency -------------------------------------------------
// vt::Tensor.data is always a host pointer from Backend::Alloc. A shadow map
// (Metal AllocMap shape) holds an optional device-resident ttnn::Tensor for
// that host base so multi-op chains need not download after every matmul.

struct BufferSlot {
  void* host = nullptr;
  size_t bytes = 0;
  std::optional<ttnn::Tensor> device;
  uint32_t dev_rows = 0;
  uint32_t dev_cols = 0;
  bool host_current = true;    // host bytes match the latest value
  bool device_current = false; // device tensor matches the latest value
};

std::mutex& SlotMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, BufferSlot>& Slots() {
  static std::map<uintptr_t, BufferSlot> m;
  return m;
}

// Base slot for `p` or any interior pointer into a registered allocation.
BufferSlot* FindSlot(void* p) {
  if (p == nullptr) return nullptr;
  auto& m = Slots();
  const uintptr_t key = reinterpret_cast<uintptr_t>(p);
  auto it = m.upper_bound(key);
  if (it == m.begin()) return nullptr;
  --it;
  BufferSlot& s = it->second;
  const uintptr_t base = reinterpret_cast<uintptr_t>(s.host);
  if (key < base || key >= base + s.bytes) return nullptr;
  return &s;
}

tt::tt_metal::TensorSpec SpecOf(tt::tt_metal::Shape shape, ttnn::DataType dtype,
                                ttnn::Layout layout) {
  return tt::tt_metal::TensorSpec(
      std::move(shape),
      tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout),
                                 tt::tt_metal::MemoryConfig{}));
}

tt::tt_metal::TensorSpec TileSpecOf(uint32_t rows, uint32_t cols) {
  return SpecOf(tt::tt_metal::Shape({rows, cols}), ttnn::DataType::BFLOAT16,
                ttnn::Layout::TILE);
}

// OPT-125m (and the rest of the dense path) runs BF16 weights/activations with
// F32 logits. Host-stage every float dtype to f32 for from_vector, then round
// back on download — ttnn already computes in BFLOAT16 tiles.
float LoadElemF32(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[i]);
    default: VT_CHECK(false, "tenstorrent: unsupported float dtype"); return 0.0f;
  }
}

void StoreElemF32(Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = F32ToBF16(v); break;
    default: VT_CHECK(false, "tenstorrent: unsupported out dtype (f32/bf16)");
  }
}

bool IsFloatDType(DType d) {
  return d == DType::kF32 || d == DType::kBF16 || d == DType::kF16;
}

void DownloadToHost(ttnn::Tensor& dev, Tensor& out) {
  std::vector<float> result = dev.to_vector<float>();
  VT_CHECK(static_cast<int64_t>(result.size()) == out.Numel(),
           "tenstorrent: unexpected result size");
  for (int64_t i = 0; i < out.Numel(); ++i)
    StoreElemF32(out, i, result[static_cast<size_t>(i)]);
}

// Pull device → host if the host view is stale (required before host kernels).
void EnsureHost(Tensor& t) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s == nullptr || s->host_current) return;
  VT_CHECK(s->device_current && s->device.has_value(),
           "tenstorrent: EnsureHost with no current device or host copy");
  DownloadToHost(*s->device, t);
  s->host_current = true;
}

void EnsureHost(const Tensor& t) {
  // const_cast: host bytes are filled in place; logical tensor is unchanged.
  EnsureHost(const_cast<Tensor&>(t));
}

std::vector<float> ToHostF32(const Tensor& t) {
  EnsureHost(t);
  const int64_t n = t.Numel();
  std::vector<float> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = LoadElemF32(t, i);
  return host;
}

ttnn::Tensor UploadRows(const float* data, uint32_t rows, uint32_t cols, MeshDevice& device) {
  std::vector<float> host(data, data + static_cast<size_t>(rows) * cols);
  return ttnn::Tensor::from_vector<float>(host, TileSpecOf(rows, cols), &device);
}

// Return a TILE BFLOAT16 device tensor for rank-2 `t`, uploading only when the
// device shadow is missing or stale. Same-numel reshape reuses a resident
// shadow without host round-trip — needed so qk-RmsNorm on a [T*H, Dh] view
// can consume QkvSplit's [T, H*Dh] device result.
ttnn::Tensor EnsureDevice2D(const Tensor& t, MeshDevice& device) {
  VT_CHECK(t.rank == 2 && t.IsContiguous(),
           "tenstorrent: EnsureDevice2D expects contiguous rank-2");
  const uint32_t rows = static_cast<uint32_t>(t.shape[0]);
  const uint32_t cols = static_cast<uint32_t>(t.shape[1]);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->device_current && s->device.has_value()) {
      if (s->dev_rows == rows && s->dev_cols == cols) {
        return *s->device;
      }
      const uint64_t have =
          static_cast<uint64_t>(s->dev_rows) * static_cast<uint64_t>(s->dev_cols);
      const uint64_t want = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
      if (have == want) {
        ttnn::Tensor reshaped =
            ttnn::reshape(*s->device, ttnn::Shape({rows, cols}));
        s->device = reshaped;
        s->dev_rows = rows;
        s->dev_cols = cols;
        return *s->device;
      }
    }
  }
  // Need host truth to upload (may download first if only device was current
  // under a different shape — rare).
  EnsureHost(t);
  const auto host = ToHostF32(t);
  ttnn::Tensor dev = UploadRows(host.data(), rows, cols, device);
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s != nullptr) {
    s->device = dev;
    s->dev_rows = rows;
    s->dev_cols = cols;
    s->device_current = true;
    s->host_current = true;
  }
  return dev;
}

// True when `t` already has a device-resident shadow matching [rows, cols]
// (exact shape). Used to pick device vs host kernels without forcing upload.
bool DeviceShadowExact(const Tensor& t, uint32_t rows, uint32_t cols) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  return s != nullptr && s->device_current && s->device.has_value() &&
         s->dev_rows == rows && s->dev_cols == cols;
}

// ---- Paged KV device shadows (ttnn layout) ---------------------------------
// Host keeps vLLM NHD [nb, block, nkv, d] (LMCache/plane). Device PA needs
// ttnn order [nb, nkv, block, d] TILE DRAM.
//
// Incremental path:
//  1) Host-side ttnn-order *mirror* patched on each ReshapeAndCache write.
//  2) When a device shadow already covers the written blocks, push them on-device:
//       - paged_fill_cache for long sequential prefill (interleaved [1,nkv,T,d])
//       - else batched paged_update_cache (height-sharded [1,B,nkv_pad,d], B tokens
//         per call, chunked to the device core count)
//     On failure, leave device dirty → Ensure re-uploads from the mirror only.

struct PagedKvShadow {
  std::optional<ttnn::Tensor> device;  // [nb, nkv, bs, d] TILE BF16 DRAM
  std::vector<float> mirror;           // ttnn-order f32, size nb*nkv*bs*d
  uint32_t nb = 0, nkv = 0, bs = 0, d = 0;
  bool mirror_valid = false;   // mirror matches host NHD for [0, nb)
  bool device_current = false; // device matches mirror
};

std::mutex& PagedKvMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, PagedKvShadow>& PagedKvShadows() {
  static std::map<uintptr_t, PagedKvShadow> m;
  return m;
}

// Drop device + mirror (geometry change or Free).
void DropPagedKvShadow(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(PagedKvMutex());
  PagedKvShadows().erase(reinterpret_cast<uintptr_t>(host));
}

// Convert host NHD blocks [0, used_nb) → ttnn [used_nb, nkv, bs, d] f32.
std::vector<float> NhdToTtnnLayoutPrefix(const Tensor& cache, uint32_t used_nb) {
  EnsureHost(cache);
  VT_CHECK(cache.rank == 4 && cache.IsContiguous(), "NhdToTtnn: rank-4 contiguous");
  const int64_t nb = cache.shape[0], bs = cache.shape[1], nkv = cache.shape[2],
                d = cache.shape[3];
  VT_CHECK(used_nb > 0 && static_cast<int64_t>(used_nb) <= nb, "NhdToTtnn: used_nb");
  std::vector<float> out(static_cast<size_t>(used_nb) * static_cast<size_t>(nkv * bs * d));
  for (int64_t b = 0; b < static_cast<int64_t>(used_nb); ++b) {
    for (int64_t g = 0; g < nkv; ++g) {
      for (int64_t off = 0; off < bs; ++off) {
        for (int64_t e = 0; e < d; ++e) {
          const int64_t src = ((b * bs + off) * nkv + g) * d + e;
          const int64_t dst = ((b * nkv + g) * bs + off) * d + e;
          out[static_cast<size_t>(dst)] = LoadElemF32(cache, src);
        }
      }
    }
  }
  return out;
}

// Grow mirror to cover at least `need_nb` blocks (zero-fill new blocks).
void EnsureMirrorCapacity(PagedKvShadow& s, uint32_t need_nb, uint32_t nkv, uint32_t bs,
                          uint32_t d) {
  if (s.mirror_valid && s.nkv == nkv && s.bs == bs && s.d == d && s.nb >= need_nb) return;
  if (s.mirror_valid && s.nkv == nkv && s.bs == bs && s.d == d && s.nb < need_nb) {
    // Grow: keep existing prefix, zero the new blocks.
    const size_t old_n = s.mirror.size();
    s.mirror.resize(static_cast<size_t>(need_nb) * nkv * bs * d, 0.0f);
    (void)old_n;
    s.nb = need_nb;
    s.device_current = false;
    return;
  }
  // Geometry mismatch or cold: allocate zeros; caller may fill from NHD.
  s.mirror.assign(static_cast<size_t>(need_nb) * nkv * bs * d, 0.0f);
  s.nb = need_nb;
  s.nkv = nkv;
  s.bs = bs;
  s.d = d;
  s.mirror_valid = true;
  s.device_current = false;
  s.device = std::nullopt;
}

// Patch one token into the ttnn-order mirror (and mark device stale).
// `tok` is contiguous [nkv, d] for that cache plane (K or V).
void PatchMirrorToken(PagedKvShadow& s, uint32_t block, uint32_t offset, const float* tok,
                      uint32_t nkv, uint32_t d) {
  VT_CHECK(s.mirror_valid && s.nkv == nkv && s.d == d && block < s.nb && offset < s.bs,
           "PatchMirrorToken: mirror geometry");
  for (uint32_t g = 0; g < nkv; ++g) {
    const size_t dst =
        (static_cast<size_t>(block) * nkv + g) * s.bs * d + static_cast<size_t>(offset) * d;
    std::memcpy(s.mirror.data() + dst, tok + static_cast<size_t>(g) * d,
                static_cast<size_t>(d) * sizeof(float));
  }
  s.device_current = false;
}

// True when tokens form a sequential fill from logical position 0 of a page
// table: offset[i] == i % bs and block is constant per logical page. Enables
// paged_fill_cache (one interleaved write for the whole prefill chunk).
bool IsSequentialFillEligible(const std::vector<uint32_t>& blocks,
                              const std::vector<uint32_t>& offsets, uint32_t bs) {
  const size_t T = blocks.size();
  if (T == 0 || offsets.size() != T || bs == 0) return false;
  if (offsets[0] != 0) return false;
  for (size_t i = 0; i < T; ++i) {
    if (offsets[i] != static_cast<uint32_t>(i % bs)) return false;
    const size_t group0 = (i / bs) * bs;
    if (blocks[i] != blocks[group0]) return false;
  }
  return true;
}

// Prefill fill via paged_fill_cache. Input layout [1, nkv, T_pad, d] INTERLEAVED
// TILE. The kernel walks padded_shape[2] in TILE rows, so T is rounded up to a
// multiple of 32 and the pad region is zero-filled (safe for unused tail slots
// in a fresh prefill block). `toks` is packed [T, nkv, d] token-major.
bool TryDevicePagedFill(ttnn::Tensor& cache_dev, MeshDevice& device,
                        const std::vector<uint32_t>& blocks, const float* toks, uint32_t T,
                        uint32_t nkv, uint32_t d, uint32_t bs) {
  if (T == 0 || blocks.size() < T) return false;
  try {
    const uint32_t T_pad = ((T + 31u) / 32u) * 32u;
    // Pack [1, nkv, T_pad, d] from token-major [T, nkv, d]; pad tail with zeros.
    std::vector<float> x(static_cast<size_t>(nkv) * T_pad * d, 0.0f);
    for (uint32_t t = 0; t < T; ++t) {
      for (uint32_t g = 0; g < nkv; ++g) {
        const float* src = toks + (static_cast<size_t>(t) * nkv + g) * d;
        float* dst = x.data() + (static_cast<size_t>(g) * T_pad + t) * d;
        std::memcpy(dst, src, static_cast<size_t>(d) * sizeof(float));
      }
    }
    ttnn::Tensor xt = ttnn::Tensor::from_vector<float>(
        x, SpecOf(tt::tt_metal::Shape({1u, nkv, T_pad, d}), ttnn::DataType::BFLOAT16,
                  ttnn::Layout::TILE),
        &device);

    const uint32_t n_logical = (T_pad + bs - 1u) / bs;
    std::vector<int32_t> pt(static_cast<size_t>(n_logical));
    for (uint32_t j = 0; j < n_logical; ++j) {
      const uint32_t tok_i = std::min(j * bs, T - 1u);
      pt[static_cast<size_t>(j)] = static_cast<int32_t>(blocks[static_cast<size_t>(tok_i)]);
    }
    ttnn::Tensor page_table = ttnn::Tensor::from_vector<int32_t>(
        pt, SpecOf(tt::tt_metal::Shape({1u, n_logical}), ttnn::DataType::INT32,
                   ttnn::Layout::ROW_MAJOR),
        &device);

    cache_dev = ttnn::experimental::paged_fill_cache(
        cache_dev, xt, page_table, /*batch_idx_tensor=*/std::nullopt, /*batch_idx=*/0,
        /*compute_kernel_config=*/std::nullopt, /*mesh_coords=*/std::nullopt);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// Pack token-major [C, nkv, d] → height-sharded [1, C, nkv_pad, d] L1 TILE.
ttnn::Tensor MakeHeightShardedUpdateInput(MeshDevice& device, const float* toks, uint32_t base,
                                          uint32_t C, uint32_t nkv, uint32_t nkv_pad, uint32_t d,
                                          const tt::tt_metal::CoreCoord& grid) {
  std::vector<float> x(static_cast<size_t>(C) * nkv_pad * d, 0.0f);
  for (uint32_t b = 0; b < C; ++b) {
    const float* src = toks + (static_cast<size_t>(base + b) * nkv * d);
    float* dst = x.data() + static_cast<size_t>(b) * nkv_pad * d;
    for (uint32_t g = 0; g < nkv; ++g) {
      std::memcpy(dst + static_cast<size_t>(g) * d, src + static_cast<size_t>(g) * d,
                  static_cast<size_t>(d) * sizeof(float));
    }
  }
  ttnn::Tensor xt = ttnn::Tensor::from_vector<float>(
      x, SpecOf(tt::tt_metal::Shape({1u, C, nkv_pad, d}), ttnn::DataType::BFLOAT16,
                ttnn::Layout::TILE),
      &device);
  const tt::tt_metal::CoreRangeSet core_set =
      tt::tt_metal::num_cores_to_corerangeset(C, grid, /*row_wise=*/true);
  tt::tt_metal::ShardSpec shard_spec(core_set, {nkv_pad, d},
                                     tt::tt_metal::ShardOrientation::ROW_MAJOR);
  tt::tt_metal::MemoryConfig sharded_mem(tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                                         tt::tt_metal::BufferType::L1, shard_spec);
  return ttnn::to_memory_config(xt, sharded_mem);
}

// Batched in-place device write via paged_update_cache. Treats each token as a
// "batch user" with a 1-entry synthetic page table (phys_block) and
// update_idx=offset. Height-shards onto B cores; chunks to the device grid when
// B exceeds available cores. `toks` is packed [B, nkv, d].
bool TryDevicePagedUpdateBatch(ttnn::Tensor& cache_dev, MeshDevice& device,
                               const std::vector<uint32_t>& phys_blocks,
                               const std::vector<uint32_t>& offsets, const float* toks,
                               uint32_t nkv, uint32_t d, uint32_t /*bs*/) {
  const uint32_t B = static_cast<uint32_t>(phys_blocks.size());
  if (B == 0 || offsets.size() != phys_blocks.size()) return false;
  try {
    const uint32_t nkv_pad = std::max(32u, ((nkv + 31u) / 32u) * 32u);
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t max_cores =
        std::max(1u, static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y));

    for (uint32_t base = 0; base < B; base += max_cores) {
      const uint32_t C = std::min(max_cores, B - base);
      ttnn::Tensor xt =
          MakeHeightShardedUpdateInput(device, toks, base, C, nkv, nkv_pad, d, grid);

      std::vector<int32_t> pt(static_cast<size_t>(C));
      std::vector<uint32_t> update_idxs(static_cast<size_t>(C));
      for (uint32_t b = 0; b < C; ++b) {
        pt[static_cast<size_t>(b)] = static_cast<int32_t>(phys_blocks[static_cast<size_t>(base + b)]);
        update_idxs[static_cast<size_t>(b)] = offsets[static_cast<size_t>(base + b)];
      }
      ttnn::Tensor page_table = ttnn::Tensor::from_vector<int32_t>(
          pt, SpecOf(tt::tt_metal::Shape({C, 1u}), ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);

      cache_dev = ttnn::experimental::paged_update_cache(
          cache_dev, xt, update_idxs, /*update_idxs_tensor=*/std::nullopt,
          /*share_cache=*/false, page_table, /*batch_offset=*/0,
          /*compute_kernel_config=*/std::nullopt, /*mesh_coords=*/std::nullopt);
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// Fused K+V paged_update — one device op for both caches (half the launches).
bool TryDevicePagedFusedUpdateBatch(ttnn::Tensor& k_dev, ttnn::Tensor& v_dev, MeshDevice& device,
                                    const std::vector<uint32_t>& phys_blocks,
                                    const std::vector<uint32_t>& offsets, const float* k_toks,
                                    const float* v_toks, uint32_t nkv, uint32_t d,
                                    uint32_t /*bs*/) {
  const uint32_t B = static_cast<uint32_t>(phys_blocks.size());
  if (B == 0 || offsets.size() != phys_blocks.size()) return false;
  try {
    const uint32_t nkv_pad = std::max(32u, ((nkv + 31u) / 32u) * 32u);
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t max_cores =
        std::max(1u, static_cast<uint32_t>(grid.x) * static_cast<uint32_t>(grid.y));

    for (uint32_t base = 0; base < B; base += max_cores) {
      const uint32_t C = std::min(max_cores, B - base);
      ttnn::Tensor xt_k =
          MakeHeightShardedUpdateInput(device, k_toks, base, C, nkv, nkv_pad, d, grid);
      ttnn::Tensor xt_v =
          MakeHeightShardedUpdateInput(device, v_toks, base, C, nkv, nkv_pad, d, grid);

      std::vector<int32_t> pt(static_cast<size_t>(C));
      std::vector<uint32_t> update_idxs(static_cast<size_t>(C));
      for (uint32_t b = 0; b < C; ++b) {
        pt[static_cast<size_t>(b)] = static_cast<int32_t>(phys_blocks[static_cast<size_t>(base + b)]);
        update_idxs[static_cast<size_t>(b)] = offsets[static_cast<size_t>(base + b)];
      }
      ttnn::Tensor page_table = ttnn::Tensor::from_vector<int32_t>(
          pt, SpecOf(tt::tt_metal::Shape({C, 1u}), ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);

      auto updated = ttnn::experimental::paged_fused_update_cache(
          k_dev, xt_k, v_dev, xt_v, update_idxs, /*update_idxs_tensor=*/std::nullopt,
          /*share_cache=*/false, page_table, /*batch_offset=*/0,
          /*compute_kernel_config=*/std::nullopt, /*mesh_coords=*/std::nullopt);
      k_dev = std::move(std::get<0>(updated));
      v_dev = std::move(std::get<1>(updated));
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// Prefer fill for longer sequential prefills; otherwise batched update.
// Threshold keeps short decode/smoke on the height-sharded update path.
constexpr uint32_t kPagedFillMinTokens = 16;

bool TryDevicePagedPush(ttnn::Tensor& cache_dev, MeshDevice& device,
                        const std::vector<uint32_t>& blocks, const std::vector<uint32_t>& offsets,
                        const float* toks, uint32_t nkv, uint32_t d, uint32_t bs) {
  const uint32_t T = static_cast<uint32_t>(blocks.size());
  if (T >= kPagedFillMinTokens && IsSequentialFillEligible(blocks, offsets, bs)) {
    if (TryDevicePagedFill(cache_dev, device, blocks, toks, T, nkv, d, bs)) return true;
    // Fall through to batched update on fill failure.
  }
  return TryDevicePagedUpdateBatch(cache_dev, device, blocks, offsets, toks, nkv, d, bs);
}

// Push K and V together: fill (2 calls) or fused update (1 call). Falls back to
// independent pushes if the paired path fails.
bool TryDevicePagedPushPair(ttnn::Tensor& k_dev, ttnn::Tensor& v_dev, MeshDevice& device,
                            const std::vector<uint32_t>& blocks,
                            const std::vector<uint32_t>& offsets, const float* k_toks,
                            const float* v_toks, uint32_t nkv, uint32_t d, uint32_t bs) {
  const uint32_t T = static_cast<uint32_t>(blocks.size());
  if (T >= kPagedFillMinTokens && IsSequentialFillEligible(blocks, offsets, bs)) {
    if (TryDevicePagedFill(k_dev, device, blocks, k_toks, T, nkv, d, bs) &&
        TryDevicePagedFill(v_dev, device, blocks, v_toks, T, nkv, d, bs)) {
      return true;
    }
    // Fall through.
  }
  if (TryDevicePagedFusedUpdateBatch(k_dev, v_dev, device, blocks, offsets, k_toks, v_toks, nkv, d,
                                     bs)) {
    return true;
  }
  // Last resort: independent updates.
  return TryDevicePagedUpdateBatch(k_dev, device, blocks, offsets, k_toks, nkv, d, bs) &&
         TryDevicePagedUpdateBatch(v_dev, device, blocks, offsets, v_toks, nkv, d, bs);
}

// After host NHD RAC writes: patch ttnn mirrors for all valid slots, then push
// to device in one (or few) paged_fill / paged_update call(s) when a shadow
// already covers every written block.
// `blocks`/`offsets` length B; `k_toks`/`v_toks` packed [B, nkv, d].
void NotePagedKvRacWrites(Tensor& k_cache, Tensor& v_cache, const std::vector<uint32_t>& blocks,
                          const std::vector<uint32_t>& offsets, const std::vector<float>& k_toks,
                          const std::vector<float>& v_toks) {
  if (k_cache.rank != 4 || v_cache.rank != 4) return;
  const uint32_t B = static_cast<uint32_t>(blocks.size());
  if (B == 0 || offsets.size() != blocks.size()) return;
  const uint32_t bs = static_cast<uint32_t>(k_cache.shape[1]);
  const uint32_t nkv = static_cast<uint32_t>(k_cache.shape[2]);
  const uint32_t d = static_cast<uint32_t>(k_cache.shape[3]);
  if ((d % 32u) != 0 || (bs % 32u) != 0) return;  // device PA won't run
  if (k_toks.size() < static_cast<size_t>(B) * nkv * d ||
      v_toks.size() < static_cast<size_t>(B) * nkv * d) {
    return;
  }

  uint32_t max_block = 0;
  for (uint32_t b : blocks) {
    if (b >= static_cast<uint32_t>(k_cache.shape[0])) return;
    if (b > max_block) max_block = b;
  }
  for (uint32_t o : offsets) {
    if (o >= bs) return;
  }
  const uint32_t need = max_block + 1u;

  // Phase 1: patch host mirrors under lock; snapshot device tensors if present.
  std::optional<ttnn::Tensor> k_dev, v_dev;
  bool k_can_update = false, v_can_update = false;
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    auto prepare = [&](void* host, const std::vector<float>& toks,
                       std::optional<ttnn::Tensor>& dev_out, bool& can_update) {
      PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(host)];
      EnsureMirrorCapacity(s, need, nkv, bs, d);
      for (uint32_t i = 0; i < B; ++i) {
        PatchMirrorToken(s, blocks[static_cast<size_t>(i)], offsets[static_cast<size_t>(i)],
                         toks.data() + static_cast<size_t>(i) * nkv * d, nkv, d);
      }
      if (s.device.has_value() && s.nb > max_block && s.nkv == nkv && s.bs == bs && s.d == d) {
        dev_out = s.device;
        can_update = true;
      }
    };
    prepare(k_cache.data, k_toks, k_dev, k_can_update);
    prepare(v_cache.data, v_toks, v_dev, v_can_update);
  }

  // Phase 2: optional on-device push (outside lock). Prefer fused K+V.
  MeshDevice* device = nullptr;
  try {
    device = &SharedMeshDevice();
  } catch (...) {
    return;
  }
  auto mark_dirty = [&](void* host) {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(host)];
    if (s.device.has_value()) s.device_current = false;
  };
  auto publish = [&](void* host, ttnn::Tensor dev) {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(host)];
    s.device = std::move(dev);
    s.device_current = true;
  };

  if (k_can_update && v_can_update && k_dev.has_value() && v_dev.has_value()) {
    if (TryDevicePagedPushPair(*k_dev, *v_dev, *device, blocks, offsets, k_toks.data(),
                               v_toks.data(), nkv, d, bs)) {
      publish(k_cache.data, std::move(*k_dev));
      publish(v_cache.data, std::move(*v_dev));
    } else {
      mark_dirty(k_cache.data);
      mark_dirty(v_cache.data);
    }
    return;
  }
  auto try_one = [&](void* host, std::optional<ttnn::Tensor>& dev, bool can, const float* toks) {
    if (!can || !dev.has_value()) return;
    if (!TryDevicePagedPush(*dev, *device, blocks, offsets, toks, nkv, d, bs)) {
      mark_dirty(host);
      return;
    }
    publish(host, std::move(*dev));
  };
  try_one(k_cache.data, k_dev, k_can_update, k_toks.data());
  try_one(v_cache.data, v_dev, v_can_update, v_toks.data());
}

// Return a current ttnn-layout device tensor covering physical blocks [0, used_nb).
ttnn::Tensor EnsurePagedKvTtnn(const Tensor& cache_nhd, MeshDevice& device, uint32_t used_nb) {
  VT_CHECK(cache_nhd.rank == 4 && cache_nhd.IsContiguous(),
           "EnsurePagedKvTtnn: contiguous rank-4 NHD cache");
  const uint32_t pool_nb = static_cast<uint32_t>(cache_nhd.shape[0]);
  const uint32_t bs = static_cast<uint32_t>(cache_nhd.shape[1]);
  const uint32_t nkv = static_cast<uint32_t>(cache_nhd.shape[2]);
  const uint32_t d = static_cast<uint32_t>(cache_nhd.shape[3]);
  VT_CHECK(used_nb > 0 && used_nb <= pool_nb, "EnsurePagedKvTtnn: used_nb out of range");

  std::vector<float> upload;
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];

    if (s.device_current && s.device.has_value() && s.nb >= used_nb && s.nkv == nkv &&
        s.bs == bs && s.d == d) {
      return *s.device;
    }

    // Prefer incremental mirror; rebuild from NHD if cold/short/wrong geometry.
    if (!s.mirror_valid || s.nkv != nkv || s.bs != bs || s.d != d || s.nb < used_nb) {
      // Release path: convert without holding the mutex for the whole NHD walk.
    } else {
      const size_t n_elems = static_cast<size_t>(used_nb) * nkv * bs * d;
      upload.assign(s.mirror.begin(),
                    s.mirror.begin() + static_cast<std::ptrdiff_t>(n_elems));
    }
  }

  if (upload.empty()) {
    // Cold / short mirror: full NHD→ttnn for the used prefix (outside the lock).
    upload = NhdToTtnnLayoutPrefix(cache_nhd, used_nb);
  }

  const auto spec = SpecOf(tt::tt_metal::Shape({used_nb, nkv, bs, d}), ttnn::DataType::BFLOAT16,
                           ttnn::Layout::TILE);
  ttnn::Tensor dev = ttnn::Tensor::from_vector<float>(upload, spec, &device);

  std::lock_guard<std::mutex> g(PagedKvMutex());
  PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];
  // If RAC raced and built a larger mirror, keep the larger one; still publish dev.
  if (!s.mirror_valid || s.nkv != nkv || s.bs != bs || s.d != d || s.nb < used_nb) {
    s.mirror = std::move(upload);
    s.nb = used_nb;
    s.nkv = nkv;
    s.bs = bs;
    s.d = d;
    s.mirror_valid = true;
  }
  s.device = dev;
  s.device_current = true;
  return dev;
}

// Publish a device result as the current value of `out` WITHOUT downloading
// to host (the residency win). Host is marked stale until EnsureHost.
// Device tensor is stored as logical [rows, cols] TILE (may differ from out's
// rank-3 view as long as numel matches) so a later Reshape+EnsureDevice2D hits.
void CommitDeviceLogical2D(Tensor& out, ttnn::Tensor dev, uint32_t rows, uint32_t cols) {
  VT_CHECK(out.IsContiguous(), "tenstorrent: CommitDeviceLogical2D expects contiguous out");
  VT_CHECK(out.Numel() == static_cast<int64_t>(rows) * static_cast<int64_t>(cols),
           "tenstorrent: CommitDeviceLogical2D numel mismatch");
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(out.data);
  if (s == nullptr) {
    // Untracked buffer (e.g. stack/test scratch): fall back to host write.
    DownloadToHost(dev, out);
    return;
  }
  s->device = std::move(dev);
  s->dev_rows = rows;
  s->dev_cols = cols;
  s->device_current = true;
  s->host_current = false;
}

void CommitDevice2D(Tensor& out, ttnn::Tensor dev) {
  VT_CHECK(out.rank == 2 && out.IsContiguous(),
           "tenstorrent: CommitDevice2D expects contiguous rank-2 out");
  CommitDeviceLogical2D(out, std::move(dev), static_cast<uint32_t>(out.shape[0]),
                        static_cast<uint32_t>(out.shape[1]));
}

// Host wrote `out` in place — drop any device shadow.
void CommitHost(Tensor& out) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(out.data);
  if (s == nullptr) return;
  s->host_current = true;
  s->device_current = false;
  s->device = std::nullopt;
}

// Device compute: keep result on device (CommitDevice2D). Host round-trip only
// when the consumer is a host-staged op (EnsureHost) or an untracked buffer.
void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmul: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmul: float in, f32/bf16 out");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[1]);
  VT_CHECK(b.shape[0] == K, "tenstorrent kMatmul: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmul: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmul: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b = EnsureDevice2D(b, device);
  ttnn::Tensor dev_c = ttnn::operations::matmul::matmul(dev_a, dev_b);
  CommitDevice2D(out, std::move(dev_c));
}

// kMatmulBT: `b` is a [N,K] row-major torch nn.Linear weight; computes
// `a @ b^T` (cpu_ops.cpp's MatmulBTKernel contract). ttnn's matmul() already
// exposes a transpose_b flag, so this is the same sequence as kMatmul with
// that flag flipped — no separate upload shape needed since `b` is uploaded
// in its native [N,K] layout and ttnn transposes on device.
void MatmulBTKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && b.rank == 2 && out.rank == 2,
           "tenstorrent kMatmulBT: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kMatmulBT: float in, f32/bf16 out");
  const uint32_t M = static_cast<uint32_t>(a.shape[0]);
  const uint32_t K = static_cast<uint32_t>(a.shape[1]);
  const uint32_t N = static_cast<uint32_t>(b.shape[0]);
  VT_CHECK(b.shape[1] == K, "tenstorrent kMatmulBT: a/b inner dimension mismatch");
  VT_CHECK(out.shape[0] == M && out.shape[1] == N, "tenstorrent kMatmulBT: out shape mismatch");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kMatmulBT: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b = EnsureDevice2D(b, device);
  ttnn::Tensor dev_c =
      ttnn::operations::matmul::matmul(dev_a, dev_b, /*transpose_a=*/false, /*transpose_b=*/true);
  CommitDevice2D(out, std::move(dev_c));
}

// kAdd: elementwise add, plus the rank-1 `b` row-broadcast form used for
// nn.Linear bias (cpu_layernorm.cpp's AddKernel contract). ttnn::add needs
// same-rank operands, so the broadcast case uploads `b` replicated into a
// [rows, d] tile rather than relying on ttnn's own broadcast rules — keeps
// this kernel's behavior pinned to the CPU reference rather than to
// whatever ttnn::add happens to support today.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(a.rank == 2 && out.rank == 2, "tenstorrent kAdd: `a`/`out` must be rank-2 in W0");
  VT_CHECK(b.rank == 2 || b.rank == 1, "tenstorrent kAdd: `b` must be rank-1 or rank-2 in W0");
  VT_CHECK(IsFloatDType(a.dtype) && IsFloatDType(b.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kAdd: float in, f32/bf16 out");
  VT_CHECK(a.IsContiguous() && b.IsContiguous() && out.IsContiguous(),
           "tenstorrent kAdd: strided (non-contiguous) tensors are not supported in W0");
  const uint32_t rows = static_cast<uint32_t>(a.shape[0]);
  const uint32_t d = static_cast<uint32_t>(a.shape[1]);
  VT_CHECK(out.shape[0] == rows && out.shape[1] == d, "tenstorrent kAdd: out shape mismatch");
  const bool bcast = b.rank == 1;
  VT_CHECK(bcast ? b.shape[0] == d : (b.shape[0] == rows && b.shape[1] == d),
           "tenstorrent kAdd: `b` shape mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_a = EnsureDevice2D(a, device);
  ttnn::Tensor dev_b;
  if (bcast) {
    EnsureHost(b);
    std::vector<float> replicated(static_cast<size_t>(rows) * d);
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < d; ++c)
        replicated[static_cast<size_t>(r) * d + c] = LoadElemF32(b, c);
    dev_b = ttnn::Tensor::from_vector<float>(replicated, TileSpecOf(rows, d), &device);
  } else {
    dev_b = EnsureDevice2D(b, device);
  }
  ttnn::Tensor dev_c = ttnn::add(dev_a, dev_b);
  CommitDevice2D(out, std::move(dev_c));
}

// kRelu: elementwise max(0, x) (cpu_layernorm.cpp's ReluKernel contract).
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2, "tenstorrent kRelu: only rank-2 tensors are supported in W0");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRelu: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kRelu: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kRelu: strided (non-contiguous) tensors are not supported in W0");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  ttnn::Tensor dev_y = ttnn::relu(dev_x);
  CommitDevice2D(out, std::move(dev_y));
}

// ---- Persistent embedding-table shadows (ROW_MAJOR BF16 on device) --------
// The vocab table is multi-hundred MB for Qwen3; re-uploading every forward
// was a pure tax. Keyed by host table base; invalidated by MarkHostWritten /
// UnregisterHostBuffer.
struct EmbedTableShadow {
  std::optional<ttnn::Tensor> device;
  uint32_t vocab = 0, h = 0;
};
std::mutex& EmbedTableMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, EmbedTableShadow>& EmbedTableShadows() {
  static std::map<uintptr_t, EmbedTableShadow> m;
  return m;
}
void DropEmbedTableShadow(void* host) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(EmbedTableMutex());
  EmbedTableShadows().erase(reinterpret_cast<uintptr_t>(host));
}

ttnn::Tensor EnsureEmbedTableDevice(const Tensor& table, MeshDevice& device) {
  VT_CHECK(table.rank == 2 && table.IsContiguous(), "EnsureEmbedTable: rank-2 contiguous");
  const uint32_t vocab = static_cast<uint32_t>(table.shape[0]);
  const uint32_t h = static_cast<uint32_t>(table.shape[1]);
  {
    std::lock_guard<std::mutex> g(EmbedTableMutex());
    auto it = EmbedTableShadows().find(reinterpret_cast<uintptr_t>(table.data));
    if (it != EmbedTableShadows().end() && it->second.device.has_value() &&
        it->second.vocab == vocab && it->second.h == h) {
      return *it->second.device;
    }
  }
  EnsureHost(table);
  std::vector<float> host_table = ToHostF32(table);
  ttnn::Tensor dev_table = ttnn::Tensor::from_vector<float>(
      host_table,
      SpecOf(tt::tt_metal::Shape({vocab, h}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR),
      &device);
  std::lock_guard<std::mutex> g(EmbedTableMutex());
  EmbedTableShadow& s = EmbedTableShadows()[reinterpret_cast<uintptr_t>(table.data)];
  s.device = dev_table;
  s.vocab = vocab;
  s.h = h;
  return dev_table;
}

// kEmbedding: row gather `out[i,:] = table[ids[i],:]` (cpu_ops.cpp
// EmbeddingKernel contract). Two layout departures from the TILE/BFLOAT16
// linear ops, forced by ttnn::embedding's validate path:
//   1. ids upload as ROW_MAJOR UINT32 (ttnn rejects i32/i64; vt still accepts
//      kI32/kI64 at the seam and converts host-side, matching Metal/Vulkan).
//   2. table is ROW_MAJOR BFLOAT16 (cached on device after first use).
// Parameter order at the ttnn call is (ids, table) — reversed from
// vt::EmbeddingFn's (table, ids). Output is TILE so the next matmul can keep
// the activation device-resident without a host round-trip.
void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  VT_CHECK(table.rank == 2 && ids.rank == 1 && out.rank == 2,
           "tenstorrent kEmbedding: table rank-2, ids rank-1, out rank-2");
  VT_CHECK(IsFloatDType(table.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kEmbedding: float table, f32/bf16 out");
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "tenstorrent kEmbedding: ids must be i32 or i64");
  VT_CHECK(table.IsContiguous() && ids.IsContiguous() && out.IsContiguous(),
           "tenstorrent kEmbedding: strided (non-contiguous) tensors are not supported");
  const uint32_t vocab = static_cast<uint32_t>(table.shape[0]);
  const uint32_t h = static_cast<uint32_t>(table.shape[1]);
  const uint32_t t = static_cast<uint32_t>(ids.shape[0]);
  VT_CHECK(out.shape[0] == t && out.shape[1] == h, "tenstorrent kEmbedding: out shape mismatch");

  EnsureHost(ids);
  std::vector<uint32_t> host_ids(t);
  if (ids.dtype == DType::kI32) {
    const int32_t* p = ids.Ptr<int32_t>();
    for (uint32_t i = 0; i < t; ++i) {
      VT_CHECK(p[i] >= 0 && static_cast<uint32_t>(p[i]) < vocab,
               "tenstorrent kEmbedding: id out of range");
      host_ids[i] = static_cast<uint32_t>(p[i]);
    }
  } else {
    const int64_t* p = ids.Ptr<int64_t>();
    for (uint32_t i = 0; i < t; ++i) {
      VT_CHECK(p[i] >= 0 && static_cast<uint64_t>(p[i]) < vocab,
               "tenstorrent kEmbedding: id out of range");
      host_ids[i] = static_cast<uint32_t>(p[i]);
    }
  }
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_ids = ttnn::Tensor::from_vector<uint32_t>(
      host_ids, SpecOf(tt::tt_metal::Shape({t}), ttnn::DataType::UINT32, ttnn::Layout::ROW_MAJOR),
      &device);
  ttnn::Tensor dev_table = EnsureEmbedTableDevice(table, device);
  // TILE output → CommitDevice2D so the first residual/RMS/matmul reuses it.
  ttnn::Tensor dev_out = ttnn::embedding(dev_ids, dev_table, /*pad_token=*/std::nullopt,
                                         /*layout=*/ttnn::Layout::TILE);
  // embedding may return [t, h] or a higher-rank view; normalize to [t, h].
  if (dev_out.logical_shape().rank() != 2 ||
      dev_out.logical_shape()[0] != t || dev_out.logical_shape()[1] != h) {
    dev_out = ttnn::reshape(dev_out, ttnn::Shape({t, h}));
  }
  CommitDevice2D(out, std::move(dev_out));
}

// Upload a rank-1 affine vector as TILE BFLOAT16 [1, d], caching on the weight's
// host buffer slot so RmsNorm/LayerNorm do not re-upload every layer call.
// ttnn's TILE-gamma path requires padded height == tile_height (32); from_vector
// with TILE pads a [1,d] tensor to that.
ttnn::Tensor EnsureAffine1D(const Tensor& t, uint32_t d, MeshDevice& device) {
  VT_CHECK(t.rank == 1 && t.IsContiguous() && t.shape[0] == static_cast<int64_t>(d),
           "tenstorrent EnsureAffine1D: rank-1 [d] contiguous");
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->device_current && s->device.has_value() && s->dev_rows == 1 &&
        s->dev_cols == d) {
      return *s->device;
    }
  }
  EnsureHost(t);
  std::vector<float> host(d);
  for (uint32_t i = 0; i < d; ++i) host[i] = LoadElemF32(t, i);
  ttnn::Tensor dev = ttnn::Tensor::from_vector<float>(
      host, SpecOf(tt::tt_metal::Shape({1, d}), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE),
      &device);
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s != nullptr) {
    s->device = dev;
    s->dev_rows = 1;
    s->dev_cols = d;
    s->device_current = true;
    s->host_current = true;
  }
  return dev;
}


// kLayerNorm: per-row mean/var over the last dim (cpu_layernorm.cpp
// LayerNormKernel / ATen native_layer_norm). Biased (1/N) variance; optional
// rank-1 weight/bias (elementwise_affine). Uses ttnn::layer_norm with the
// same TILE/BFLOAT16 upload path as the linear ops; eps comes from
// LayerNormArgs (OPT default 1e-5, not ttnn's 1e-12 default).
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kLayerNorm: only rank-2 tensors are supported in this step");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kLayerNorm: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kLayerNorm: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kLayerNorm: strided (non-contiguous) tensors are not supported");
  VT_CHECK(args.eps >= 0.0f, "tenstorrent kLayerNorm: eps must be non-negative");
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);
  for (const Tensor* p : {weight, bias}) {
    if (p == nullptr) continue;
    VT_CHECK(p->rank == 1 && p->shape[0] == d,
             "tenstorrent kLayerNorm: weight/bias must be rank-1 [D]");
    VT_CHECK(IsFloatDType(p->dtype), "tenstorrent kLayerNorm: float weight/bias");
    VT_CHECK(p->IsContiguous(), "tenstorrent kLayerNorm: weight/bias must be contiguous");
  }

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  std::optional<ttnn::Tensor> dev_w;
  std::optional<ttnn::Tensor> dev_b;
  if (weight != nullptr) dev_w = EnsureAffine1D(*weight, d, device);
  if (bias != nullptr) dev_b = EnsureAffine1D(*bias, d, device);
  ttnn::Tensor dev_y = ttnn::layer_norm(dev_x, args.eps, dev_w, dev_b);
  CommitDevice2D(out, std::move(dev_y));
}

// kRmsNorm: per-row RMS over the last dim (cpu_ops.cpp RmsNormKernel). First
// Qwen3-dense (`Qwen3ForCausalLM`) op beyond OPT's LayerNorm set — Qwen3 uses
// RMSNorm for input/post-attn/final norms and per-head q/k norms. Weight is
// always present at the seam; optional residual is the residual stream
// (pre-norm sum written back, then normed), matching CPU residual round-trip
// for bf16 faithfulness. Gemma style (w+1) is host-only for now — Qwen3 does
// not set gemma=true.
//
// Device path: ttnn::rms_norm after residual merge (when any) and weight
// upload via the same TILE [1,D] affine helper as kLayerNorm.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                   const RmsNormArgs& args, Tensor* residual) {
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kRmsNorm: only rank-2 tensors are supported in this step");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRmsNorm: float in, f32/bf16 out");
  VT_CHECK(x.shape[0] == out.shape[0] && x.shape[1] == out.shape[1],
           "tenstorrent kRmsNorm: out shape mismatch");
  VT_CHECK(x.IsContiguous() && out.IsContiguous() && weight.IsContiguous(),
           "tenstorrent kRmsNorm: strided (non-contiguous) tensors are not supported");
  VT_CHECK(args.eps >= 0.0f, "tenstorrent kRmsNorm: eps must be non-negative");
  const uint32_t rows = static_cast<uint32_t>(x.shape[0]);
  const uint32_t d = static_cast<uint32_t>(x.shape[1]);
  VT_CHECK(weight.rank == 1 && weight.shape[0] == d && IsFloatDType(weight.dtype),
           "tenstorrent kRmsNorm: weight must be rank-1 float [D]");
  if (residual != nullptr) {
    VT_CHECK(residual->rank == 2 && residual->shape[0] == rows && residual->shape[1] == d,
             "tenstorrent kRmsNorm: residual shape must match x");
    VT_CHECK(IsFloatDType(residual->dtype) && residual->IsContiguous(),
             "tenstorrent kRmsNorm: residual must be contiguous float");
  }

  // Host path for gemma (w+1) and for tiny residual merges: short decode
  // (rows=1) pays more for device add+rms launches than a host loop, and was
  // a measurable e2e regression vs host residual.
  constexpr uint32_t kDeviceResidualMinRows = 32;
  const bool host_residual =
      args.gemma || (residual != nullptr && rows < kDeviceResidualMinRows);
  if (host_residual) {
    EnsureHost(x);
    EnsureHost(weight);
    if (residual != nullptr) EnsureHost(*residual);
    for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
      float sumsq = 0.0f;
      for (int64_t j = 0; j < static_cast<int64_t>(d); ++j) {
        const int64_t idx = r * static_cast<int64_t>(d) + j;
        float v = LoadElemF32(x, idx);
        if (residual != nullptr) {
          v += LoadElemF32(*residual, idx);
          StoreElemF32(*residual, idx, v);
          v = LoadElemF32(*residual, idx);
        }
        sumsq += v * v;
      }
      const float inv =
          1.0f / std::sqrt(sumsq / static_cast<float>(d) + args.eps);
      for (int64_t j = 0; j < static_cast<int64_t>(d); ++j) {
        const int64_t idx = r * static_cast<int64_t>(d) + j;
        float v =
            residual != nullptr ? LoadElemF32(*residual, idx) : LoadElemF32(x, idx);
        float wj = LoadElemF32(weight, j);
        if (args.gemma) wj += 1.0f;
        StoreElemF32(out, idx, v * inv * wj);
      }
    }
    CommitHost(out);
    if (residual != nullptr) CommitHost(*residual);
    return;
  }

  // Device path: plain rms, or residual stream (x+residual → residual, then
  // rms) when rows are large enough that launches amortize.
  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  ttnn::Tensor dev_w = EnsureAffine1D(weight, d, device);
  ttnn::Tensor to_norm = dev_x;
  if (residual != nullptr) {
    ttnn::Tensor dev_r = EnsureDevice2D(*residual, device);
    to_norm = ttnn::add(dev_x, dev_r);
    CommitDevice2D(*residual, to_norm);
  }
  ttnn::Tensor dev_y = ttnn::rms_norm(to_norm, args.eps, dev_w);
  CommitDevice2D(out, std::move(dev_y));
}

// kSiluAndMul: SwiGLU gate half — out[i,j] = silu(x[i,j]) * x[i,j+d]
// with d = x.shape[1]/2 (cpu_ops.cpp SiluAndMulKernel). Second Qwen3-dense
// op beyond OPT (MLP: gate_up GEMM -> SiluAndMul -> down GEMM). Device path
// keeps the gate_up → SiluAndMul → down GEMM chain on-device: slice the
// last-dim halves, ttnn::silu(gate), ttnn::multiply by up. BF16 tile path
// (same envelope as matmul/norm); not bit-exact vs host f32.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  VT_CHECK(x.rank == 2 && out.rank == 2,
           "tenstorrent kSiluAndMul: only rank-2 tensors are supported");
  VT_CHECK(IsFloatDType(x.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kSiluAndMul: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kSiluAndMul: strided (non-contiguous) tensors are not supported");
  VT_CHECK(x.shape[1] % 2 == 0, "tenstorrent kSiluAndMul: last dim must be even");
  const int64_t t = x.shape[0];
  const int64_t d = x.shape[1] / 2;
  VT_CHECK(out.shape[0] == t && out.shape[1] == d,
           "tenstorrent kSiluAndMul: out shape must be [T, D] with D = x.shape[1]/2");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = EnsureDevice2D(x, device);
  const uint32_t tu = static_cast<uint32_t>(t);
  const uint32_t du = static_cast<uint32_t>(d);
  // x = [gate | up] along last dim.
  ttnn::Tensor gate = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, 0},
                                  ttsl::SmallVector<uint32_t>{tu, du},
                                  ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor up = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, du},
                                ttsl::SmallVector<uint32_t>{tu, 2 * du},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor silu_gate = ttnn::silu(gate);
  ttnn::Tensor dev_y = ttnn::multiply(silu_gate, up);
  CommitDevice2D(out, std::move(dev_y));
}

// kCastBf16 / kCastF32: elementwise dtype convert via Load/Store (cpu_ops
// CastBf16Kernel / CastF32Kernel). Qwen3 uses these for K/V cache dtype and
// the logits / rope-cache paths. Host-staged; bit-exact for supported pairs.
void CastBf16Kernel(Queue&, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kBF16, "tenstorrent kCastBf16: out must be bf16");
  VT_CHECK(IsFloatDType(in.dtype), "tenstorrent kCastBf16: in must be float");
  VT_CHECK(out.Numel() == in.Numel(), "tenstorrent kCastBf16: numel mismatch");
  VT_CHECK(out.IsContiguous() && in.IsContiguous(),
           "tenstorrent kCastBf16: contiguous required");
  EnsureHost(in);
  const int64_t n = out.Numel();
  for (int64_t i = 0; i < n; ++i) StoreElemF32(out, i, LoadElemF32(in, i));
  CommitHost(out);
}

void CastF32Kernel(Queue&, Tensor& out, const Tensor& in) {
  VT_CHECK(out.dtype == DType::kF32, "tenstorrent kCastF32: out must be f32");
  VT_CHECK(IsFloatDType(in.dtype), "tenstorrent kCastF32: in must be float");
  VT_CHECK(out.Numel() == in.Numel(), "tenstorrent kCastF32: numel mismatch");
  VT_CHECK(out.IsContiguous() && in.IsContiguous(),
           "tenstorrent kCastF32: contiguous required");
  EnsureHost(in);
  const int64_t n = out.Numel();
  for (int64_t i = 0; i < n; ++i) StoreElemF32(out, i, LoadElemF32(in, i));
  CommitHost(out);
}

// Llama-3 frequency rescale (cpu_ops Llama3ScaleFreq); no-op when scaling_factor
// is unset. Kept so Qwen3 / Llama rope paths share one host implementation.
inline double Llama3ScaleFreq(double freq, const RopeArgs& a) {
  const double sf = static_cast<double>(a.llama3_scaling_factor);
  if (!(sf > 0.0)) return freq;
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double lo = static_cast<double>(a.llama3_low_freq_factor);
  const double hi = static_cast<double>(a.llama3_high_freq_factor);
  const double omax = static_cast<double>(a.llama3_orig_max_position);
  const double low_freq_wavelen = omax / lo;
  const double high_freq_wavelen = omax / hi;
  const double wave_len = kTwoPi / freq;
  double smooth = 0.0;
  if (lo != hi) smooth = (omax / wave_len - lo) / (hi - lo);
  if (wave_len < high_freq_wavelen) return freq;
  if (wave_len > low_freq_wavelen) return freq / sf;
  return (1.0 - smooth) * freq / sf + smooth * freq;
}

// Expand per-token cos|sin [T, half] to per-(token,head) [T*H, half] for a
// device NeoX apply over the flat [T*H, D] view of qs/ks.
void ExpandCosSinPerHead(const float* cos_t, const float* sin_t, int64_t tokens,
                         int64_t heads, int64_t half, std::vector<float>& cos_exp,
                         std::vector<float>& sin_exp) {
  cos_exp.resize(static_cast<size_t>(tokens * heads * half));
  sin_exp.resize(static_cast<size_t>(tokens * heads * half));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      const size_t dst = static_cast<size_t>((t * heads + h) * half);
      const size_t src = static_cast<size_t>(t * half);
      std::memcpy(cos_exp.data() + dst, cos_t + src, static_cast<size_t>(half) * sizeof(float));
      std::memcpy(sin_exp.data() + dst, sin_t + src, static_cast<size_t>(half) * sizeof(float));
    }
  }
}

// Device NeoX apply: view [T,H,D] as [T*H,D], rotate leading `rot` cols via
// slice + mul/sub/add + concat. Reuses EnsureDevice2D so a prior RmsNorm on the
// [T*H,D] view leaves the shadow resident (no re-upload). BF16 tile path.
void RopeApplyDeviceNeox(Tensor& x3, const float* cos_t, const float* sin_t,
                         int64_t tokens, int64_t heads, int64_t d, int64_t rot,
                         MeshDevice& device) {
  VT_CHECK(x3.rank == 3 && x3.IsContiguous() && x3.shape[0] == tokens &&
               x3.shape[1] == heads && x3.shape[2] == d,
           "tenstorrent device rope: rank-3 contiguous [T,H,D]");
  VT_CHECK(rot > 0 && (rot % 2) == 0 && rot <= d, "tenstorrent device rope: rotary_dim");
  const int64_t half = rot / 2;
  const int64_t th = tokens * heads;
  Tensor x_mat = x3.View({th, d});
  ttnn::Tensor dev_x = EnsureDevice2D(x_mat, device);

  std::vector<float> cos_exp, sin_exp;
  ExpandCosSinPerHead(cos_t, sin_t, tokens, heads, half, cos_exp, sin_exp);
  const uint32_t thu = static_cast<uint32_t>(th);
  const uint32_t halfu = static_cast<uint32_t>(half);
  const uint32_t rotu = static_cast<uint32_t>(rot);
  const uint32_t du = static_cast<uint32_t>(d);
  ttnn::Tensor dev_cos = UploadRows(cos_exp.data(), thu, halfu, device);
  ttnn::Tensor dev_sin = UploadRows(sin_exp.data(), thu, halfu, device);

  // x1 = x[..., :half], x2h = x[..., half:rot]  (NeoX half-split)
  ttnn::Tensor x1 = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, 0},
                                ttsl::SmallVector<uint32_t>{thu, halfu},
                                ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor x2h = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, halfu},
                                 ttsl::SmallVector<uint32_t>{thu, rotu},
                                 ttsl::SmallVector<uint32_t>{1, 1});
  ttnn::Tensor o1 = ttnn::subtract(ttnn::multiply(x1, dev_cos), ttnn::multiply(x2h, dev_sin));
  ttnn::Tensor o2 = ttnn::add(ttnn::multiply(x1, dev_sin), ttnn::multiply(x2h, dev_cos));
  ttnn::Tensor rotated = ttnn::concat(std::vector<ttnn::Tensor>{o1, o2}, /*dim=*/1);
  ttnn::Tensor out_dev;
  if (rot == d) {
    out_dev = std::move(rotated);
  } else {
    ttnn::Tensor tail = ttnn::slice(dev_x, ttsl::SmallVector<uint32_t>{0, rotu},
                                    ttsl::SmallVector<uint32_t>{thu, du},
                                    ttsl::SmallVector<uint32_t>{1, 1});
    out_dev = ttnn::concat(std::vector<ttnn::Tensor>{rotated, tail}, /*dim=*/1);
  }
  CommitDevice2D(x_mat, std::move(out_dev));
}

// Gather per-token cos|sin from a [P, rot] cache via rank-1 positions.
void GatherCosSinRows(const Tensor& cache, const Tensor& positions, int64_t tokens,
                      int rot, std::vector<float>& cos_t, std::vector<float>& sin_t) {
  EnsureHost(cache);
  EnsureHost(positions);
  const int64_t half = rot / 2;
  cos_t.resize(static_cast<size_t>(tokens * half));
  sin_t.resize(static_cast<size_t>(tokens * half));
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t position = positions.dtype == DType::kI32
                                 ? static_cast<int64_t>(positions.Ptr<int32_t>()[t])
                                 : positions.Ptr<int64_t>()[t];
    VT_CHECK(position >= 0 && position < cache.shape[0],
             "tenstorrent rope: position outside cache");
    const int64_t cache_off = position * rot;
    for (int64_t i = 0; i < half; ++i) {
      cos_t[static_cast<size_t>(t * half + i)] = LoadElemF32(cache, cache_off + i);
      sin_t[static_cast<size_t>(t * half + i)] =
          LoadElemF32(cache, cache_off + half + i);
    }
  }
}

// Build per-token cos|sin from RopeNeox frequencies (double angles, f32 store).
void BuildCosSinFromPositions(const Tensor& pos, int64_t tokens, int rot, double base,
                              const RopeArgs& args, std::vector<float>& cos_t,
                              std::vector<float>& sin_t) {
  EnsureHost(pos);
  const int64_t half = rot / 2;
  cos_t.resize(static_cast<size_t>(tokens * half));
  sin_t.resize(static_cast<size_t>(tokens * half));
  for (int64_t t = 0; t < tokens; ++t) {
    const int64_t p =
        pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[t] : pos.Ptr<int64_t>()[t];
    for (int64_t i = 0; i < half; ++i) {
      double freq = std::pow(base, -2.0 * static_cast<double>(i) / static_cast<double>(rot));
      freq = Llama3ScaleFreq(freq, args);
      const double angle = static_cast<double>(p) * freq;
      cos_t[static_cast<size_t>(t * half + i)] = static_cast<float>(std::cos(angle));
      sin_t[static_cast<size_t>(t * half + i)] = static_cast<float>(std::sin(angle));
    }
  }
}

// Host NeoX/GPT-J apply from a precomputed cos|sin table. Fast path for short
// decode: many tiny device launches (slice/mul/concat × q/k) lose to this.
void RopeApplyHost(Tensor& qs, Tensor* ks, const float* cos_t, const float* sin_t,
                   int64_t tokens, int64_t hq, int64_t hk, int64_t d, int rot,
                   bool is_neox) {
  EnsureHost(qs);
  if (ks != nullptr) EnsureHost(*ks);
  const int64_t half = rot / 2;
  auto apply_one = [&](Tensor& x, int64_t heads) {
    for (int64_t token = 0; token < tokens; ++token) {
      for (int64_t pair = 0; pair < half; ++pair) {
        const float c = cos_t[static_cast<size_t>(token * half + pair)];
        const float s = sin_t[static_cast<size_t>(token * half + pair)];
        const int64_t first = is_neox ? pair : pair * 2;
        const int64_t second = is_neox ? pair + half : pair * 2 + 1;
        for (int64_t head = 0; head < heads; ++head) {
          const int64_t off = (token * heads + head) * d;
          const float xv = LoadElemF32(x, off + first);
          const float yv = LoadElemF32(x, off + second);
          StoreElemF32(x, off + first, xv * c - yv * s);
          StoreElemF32(x, off + second, xv * s + yv * c);
        }
      }
    }
  };
  apply_one(qs, hq);
  if (ks != nullptr) apply_one(*ks, hk);
  CommitHost(qs);
  if (ks != nullptr) CommitHost(*ks);
}

// Prefer device apply only when T*H amortizes the slice/mul/concat launches.
// Short Qwen3 decode (T=1,H=16) is host-faster even when Q is already on device
// (measured regression when always-device-for-resident was forced).
inline bool PreferDeviceRope(int64_t tokens, int64_t heads) {
  return tokens * heads >= 64;
}

// kRopeNeox: Qwen3-dense RoPE. Device NeoX for large [T*H]; host for short decode.
void RopeNeoxKernel(Queue&, Tensor& qs, Tensor& ks, const Tensor& pos, const RopeArgs& args) {
  VT_CHECK(qs.rank == 3 && ks.rank == 3, "tenstorrent kRopeNeox: qs/ks rank-3");
  VT_CHECK(IsFloatDType(qs.dtype) && qs.dtype == ks.dtype,
           "tenstorrent kRopeNeox: qs/ks float same dtype");
  VT_CHECK(pos.rank == 1 && (pos.dtype == DType::kI32 || pos.dtype == DType::kI64),
           "tenstorrent kRopeNeox: positions rank-1 i32/i64");
  VT_CHECK(qs.IsContiguous() && ks.IsContiguous() && pos.IsContiguous(),
           "tenstorrent kRopeNeox: contiguous required");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0 &&
               args.rotary_dim <= qs.shape[2],
           "tenstorrent kRopeNeox: rotary_dim must be even and <= head_dim");
  const int64_t t = qs.shape[0], hq = qs.shape[1], hk = ks.shape[1], d = qs.shape[2];
  VT_CHECK(ks.shape[0] == t && ks.shape[2] == d, "tenstorrent kRopeNeox: ks shape");
  VT_CHECK(pos.shape[0] == t, "tenstorrent kRopeNeox: positions length");

  std::vector<float> cos_t, sin_t;
  BuildCosSinFromPositions(pos, t, args.rotary_dim, static_cast<double>(args.base), args, cos_t,
                           sin_t);
  if (PreferDeviceRope(t, hq)) {
    MeshDevice& device = SharedMeshDevice();
    RopeApplyDeviceNeox(qs, cos_t.data(), sin_t.data(), t, hq, d, args.rotary_dim, device);
    RopeApplyDeviceNeox(ks, cos_t.data(), sin_t.data(), t, hk, d, args.rotary_dim, device);
  } else {
    RopeApplyHost(qs, &ks, cos_t.data(), sin_t.data(), t, hq, hk, d, args.rotary_dim,
                  /*is_neox=*/true);
  }
}

// kRopeCosSinCache: per-step cos|sin table [T, rot] (cpu_ops RopeCosSinCacheKernel).
// Stays host — table is small and built once per step; apply is device.
void RopeCosSinCacheKernel(Queue&, Tensor& cos_sin, const Tensor& positions,
                           const RopeArgs& args) {
  VT_CHECK(cos_sin.rank == 2 && cos_sin.dtype == DType::kF32 && cos_sin.IsContiguous(),
           "tenstorrent kRopeCosSinCache: cos_sin contiguous f32 [T,rot]");
  VT_CHECK(positions.rank == 1 &&
               (positions.dtype == DType::kI32 || positions.dtype == DType::kI64) &&
               positions.IsContiguous(),
           "tenstorrent kRopeCosSinCache: positions rank-1 i32/i64");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0,
           "tenstorrent kRopeCosSinCache: rotary_dim even > 0");
  const int64_t t = cos_sin.shape[0];
  const int rot = args.rotary_dim;
  VT_CHECK(cos_sin.shape[1] == rot && positions.shape[0] == t,
           "tenstorrent kRopeCosSinCache: shape mismatch");
  EnsureHost(positions);
  const int64_t half = rot / 2;
  const double base = static_cast<double>(args.base);
  for (int64_t i = 0; i < t; ++i) {
    const int64_t p = positions.dtype == DType::kI32 ? positions.Ptr<int32_t>()[i]
                                                     : positions.Ptr<int64_t>()[i];
    for (int64_t pair = 0; pair < half; ++pair) {
      double freq =
          std::pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(rot));
      freq = Llama3ScaleFreq(freq, args);
      const double angle = static_cast<double>(p) * freq;
      StoreElemF32(cos_sin, i * rot + pair, static_cast<float>(std::cos(angle)));
      StoreElemF32(cos_sin, i * rot + half + pair, static_cast<float>(std::sin(angle)));
    }
  }
  CommitHost(cos_sin);
}

// kRopeFromCache: apply precomputed cos|sin (cpu_ops RopeFromCacheKernel).
// Rank-1 positions only (Qwen3-dense); mrope deferred. DEFAULT Qwen3 path
// (VT_QWEN3_ROPE_CACHE). Device NeoX when T*H is large; host for short decode
// and GPT-J interleave.
void RopeFromCacheKernel(Queue&, Tensor& qs, Tensor* ks, const Tensor& positions,
                         const Tensor& cache, const RopeArgs& args) {
  VT_CHECK(qs.rank == 3 && IsFloatDType(qs.dtype) && qs.IsContiguous(),
           "tenstorrent kRopeFromCache: qs rank-3 contiguous float");
  VT_CHECK(positions.rank == 1 &&
               (positions.dtype == DType::kI32 || positions.dtype == DType::kI64) &&
               positions.IsContiguous(),
           "tenstorrent kRopeFromCache: rank-1 positions only (no mrope yet)");
  VT_CHECK(cache.rank == 2 && IsFloatDType(cache.dtype) && cache.IsContiguous(),
           "tenstorrent kRopeFromCache: cache rank-2 contiguous float");
  VT_CHECK(args.rotary_dim > 0 && (args.rotary_dim % 2) == 0 &&
               args.rotary_dim <= qs.shape[2],
           "tenstorrent kRopeFromCache: rotary_dim");
  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t d = qs.shape[2];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  if (ks != nullptr) {
    VT_CHECK(ks->rank == 3 && ks->dtype == qs.dtype && ks->IsContiguous(),
             "tenstorrent kRopeFromCache: ks must match qs");
    VT_CHECK(ks->shape[0] == tokens && ks->shape[2] == d,
             "tenstorrent kRopeFromCache: ks shape");
  }
  VT_CHECK(positions.shape[0] == tokens, "tenstorrent kRopeFromCache: positions length");

  std::vector<float> cos_t, sin_t;
  GatherCosSinRows(cache, positions, tokens, args.rotary_dim, cos_t, sin_t);

  if (args.is_neox_style && PreferDeviceRope(tokens, hq)) {
    MeshDevice& device = SharedMeshDevice();
    RopeApplyDeviceNeox(qs, cos_t.data(), sin_t.data(), tokens, hq, d, args.rotary_dim, device);
    if (ks != nullptr) {
      RopeApplyDeviceNeox(*ks, cos_t.data(), sin_t.data(), tokens, hk, d, args.rotary_dim,
                          device);
    }
    return;
  }
  RopeApplyHost(qs, ks, cos_t.data(), sin_t.data(), tokens, hq, hk, d, args.rotary_dim,
                args.is_neox_style);
}

// kQkvSplit: column split of merged [T, q+k+v] into q/k/v (cpu_ops QkvSplitKernel).
//
// Device path when qkv already has a resident shadow (post MatmulBT): slice the
// last dim on-device and CommitDevice2D each shard so qk-RmsNorm can reshape-
// reuse without download+reupload. Host path (bit-exact memcpy) when qkv is
// host-only — unit tests and weight-load style callers.
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  VT_CHECK(qkv.rank == 2 && IsFloatDType(qkv.dtype),
           "tenstorrent kQkvSplit: rank-2 float qkv required");
  VT_CHECK(q_out.dtype == qkv.dtype && k_out.dtype == qkv.dtype && v_out.dtype == qkv.dtype,
           "tenstorrent kQkvSplit: q/k/v out must match qkv dtype");
  VT_CHECK(q_out.IsContiguous() && k_out.IsContiguous() && v_out.IsContiguous() &&
               qkv.IsContiguous(),
           "tenstorrent kQkvSplit: contiguous required");
  const int64_t t = qkv.shape[0];
  const int64_t q_dim = q_out.Numel() / t;
  const int64_t k_dim = k_out.Numel() / t;
  const int64_t v_dim = v_out.Numel() / t;
  const int64_t total = q_dim + k_dim + v_dim;
  VT_CHECK(qkv.shape[1] == total, "tenstorrent kQkvSplit: inner dim mismatch");
  VT_CHECK(q_out.rank == 2 && k_out.rank == 2 && v_out.rank == 2 &&
               q_out.shape[0] == t && k_out.shape[0] == t && v_out.shape[0] == t &&
               q_out.shape[1] == q_dim && k_out.shape[1] == k_dim && v_out.shape[1] == v_dim,
           "tenstorrent kQkvSplit: out shapes must be [T, *]");

  const uint32_t tu = static_cast<uint32_t>(t);
  const uint32_t total_u = static_cast<uint32_t>(total);
  const uint32_t qd = static_cast<uint32_t>(q_dim);
  const uint32_t kd = static_cast<uint32_t>(k_dim);
  const uint32_t vd = static_cast<uint32_t>(v_dim);

  if (DeviceShadowExact(qkv, tu, total_u)) {
    MeshDevice& device = SharedMeshDevice();
    ttnn::Tensor dev = EnsureDevice2D(qkv, device);
    ttnn::Tensor dq = ttnn::slice(dev, ttsl::SmallVector<uint32_t>{0, 0},
                                  ttsl::SmallVector<uint32_t>{tu, qd},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor dk = ttnn::slice(dev, ttsl::SmallVector<uint32_t>{0, qd},
                                  ttsl::SmallVector<uint32_t>{tu, qd + kd},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    ttnn::Tensor dv = ttnn::slice(dev, ttsl::SmallVector<uint32_t>{0, qd + kd},
                                  ttsl::SmallVector<uint32_t>{tu, qd + kd + vd},
                                  ttsl::SmallVector<uint32_t>{1, 1});
    CommitDevice2D(q_out, std::move(dq));
    CommitDevice2D(k_out, std::move(dk));
    CommitDevice2D(v_out, std::move(dv));
    return;
  }

  EnsureHost(qkv);
  const size_t esz = SizeOf(qkv.dtype);
  const auto* src = static_cast<const uint8_t*>(qkv.data);
  auto* qdst = static_cast<uint8_t*>(q_out.data);
  auto* kdst = static_cast<uint8_t*>(k_out.data);
  auto* vdst = static_cast<uint8_t*>(v_out.data);
  for (int64_t i = 0; i < t; ++i) {
    const uint8_t* row = src + static_cast<size_t>(i * total) * esz;
    std::memcpy(qdst + static_cast<size_t>(i * q_dim) * esz, row,
                static_cast<size_t>(q_dim) * esz);
    std::memcpy(kdst + static_cast<size_t>(i * k_dim) * esz, row + static_cast<size_t>(q_dim) * esz,
                static_cast<size_t>(k_dim) * esz);
    std::memcpy(vdst + static_cast<size_t>(i * v_dim) * esz,
                row + static_cast<size_t>(q_dim + k_dim) * esz, static_cast<size_t>(v_dim) * esz);
  }
  CommitHost(q_out);
  CommitHost(k_out);
  CommitHost(v_out);
}

// kReshapeAndCache: write per-token K/V into paged NHD cache slots
// (cpu_cache.cpp ReshapeAndCacheKernel). Stride-driven so unbind-style
// [num_blocks,2,bs,H,D] views work; slot < 0 is a padded-token skip.
// Host-staged pure element copy for F32.
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  VT_CHECK(k.rank == 3 && v.rank == 3 && k_cache.rank == 4 && v_cache.rank == 4,
           "tenstorrent kReshapeAndCache: k/v rank-3, caches rank-4");
  VT_CHECK(IsFloatDType(k.dtype) && k.dtype == v.dtype && k_cache.dtype == k.dtype &&
               v_cache.dtype == k.dtype,
           "tenstorrent kReshapeAndCache: k/v/caches must share one float dtype");
  EnsureHost(k);
  EnsureHost(v);
  EnsureHost(k_cache);
  EnsureHost(v_cache);
  EnsureHost(slot_mapping);
  VT_CHECK(slot_mapping.rank == 1 && slot_mapping.dtype == DType::kI64,
           "tenstorrent kReshapeAndCache: slot_mapping rank-1 i64");
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  const int64_t head_size = k_cache.shape[3];
  const int64_t n_elems = num_kv_heads * head_size;
  VT_CHECK(k.shape[1] == num_kv_heads && k.shape[2] == head_size && v.shape[1] == num_kv_heads &&
               v.shape[2] == head_size,
           "tenstorrent kReshapeAndCache: k/v head shape must match cache");
  VT_CHECK(k.shape[0] >= num_slots && v.shape[0] >= num_slots,
           "tenstorrent kReshapeAndCache: token count must cover slots");
  // Contiguous NHD page: head stride == head_size (ops.cpp contract).
  VT_CHECK(k_cache.stride[2] == head_size && v_cache.stride[2] == head_size &&
               k_cache.stride[3] == 1 && v_cache.stride[3] == 1,
           "tenstorrent kReshapeAndCache: cache pages must be dense NHD");
  VT_CHECK(k.stride[2] == 1 && v.stride[2] == 1,
           "tenstorrent kReshapeAndCache: k/v innermost stride must be 1");

  const int64_t k_block_stride = k_cache.stride[0];
  const int64_t k_page_stride = k_cache.stride[1];
  const int64_t v_block_stride = v_cache.stride[0];
  const int64_t v_page_stride = v_cache.stride[1];
  const int64_t k_tok_stride = k.stride[0];
  const int64_t v_tok_stride = v.stride[0];
  const int64_t* slots = slot_mapping.Ptr<int64_t>();
  const size_t esz = SizeOf(k.dtype);
  const auto* ksrc = static_cast<const uint8_t*>(k.data);
  const auto* vsrc = static_cast<const uint8_t*>(v.data);
  auto* kdst = static_cast<uint8_t*>(k_cache.data);
  auto* vdst = static_cast<uint8_t*>(v_cache.data);
  const size_t bytes = static_cast<size_t>(n_elems) * esz;

  // Optional float staging for ttnn-mirror incremental patches (TILE-legal only).
  // Collect all valid slots then one batched device push (fill or multi-token update).
  const bool patch_mirror = (head_size % 32) == 0 && (block_size % 32) == 0;
  std::vector<uint32_t> rac_blocks, rac_offsets;
  std::vector<float> k_toks, v_toks;
  if (patch_mirror) {
    rac_blocks.reserve(static_cast<size_t>(num_slots));
    rac_offsets.reserve(static_cast<size_t>(num_slots));
    k_toks.reserve(static_cast<size_t>(num_slots) * static_cast<size_t>(n_elems));
    v_toks.reserve(static_cast<size_t>(num_slots) * static_cast<size_t>(n_elems));
  }

  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0) continue;
    const int64_t block = slot / block_size;
    const int64_t offset = slot % block_size;
    const int64_t kdst_off = block * k_block_stride + offset * k_page_stride;
    const int64_t vdst_off = block * v_block_stride + offset * v_page_stride;
    std::memcpy(kdst + static_cast<size_t>(kdst_off) * esz,
                ksrc + static_cast<size_t>(t * k_tok_stride) * esz, bytes);
    std::memcpy(vdst + static_cast<size_t>(vdst_off) * esz,
                vsrc + static_cast<size_t>(t * v_tok_stride) * esz, bytes);
    if (patch_mirror) {
      rac_blocks.push_back(static_cast<uint32_t>(block));
      rac_offsets.push_back(static_cast<uint32_t>(offset));
      const size_t base = k_toks.size();
      k_toks.resize(base + static_cast<size_t>(n_elems));
      v_toks.resize(base + static_cast<size_t>(n_elems));
      for (int64_t i = 0; i < n_elems; ++i) {
        k_toks[base + static_cast<size_t>(i)] = LoadElemF32(k, t * k_tok_stride + i);
        v_toks[base + static_cast<size_t>(i)] = LoadElemF32(v, t * v_tok_stride + i);
      }
    }
  }
  CommitHost(k_cache);
  CommitHost(v_cache);
  if (patch_mirror) {
    if (!rac_blocks.empty()) {
      NotePagedKvRacWrites(k_cache, v_cache, rac_blocks, rac_offsets, k_toks, v_toks);
    }
  } else {
    DropPagedKvShadow(k_cache.data);
    DropPagedKvShadow(v_cache.data);
  }
}

// Try pure-decode device PA via ttnn::paged_scaled_dot_product_attention_decode.
// Host keeps NHD; we upload a ttnn-layout [nb,nkv,bs,d] shadow (rebuilt when
// ReshapeAndCache dirties it). Returns true if `out` was written.
bool TryPagedAttentionDeviceDecode(Tensor& out, const Tensor& query, const Tensor& k_cache,
                                   const Tensor& v_cache, const Tensor& block_table,
                                   const Tensor& seq_lens, const Tensor& query_start_loc,
                                   const PagedAttentionArgs& args) {
  if (!args.causal || args.logits_soft_cap > 0.0f) return false;
  if (args.window_size.has_value()) return false;
  if (args.kv_cache_dtype != Fp8KVCacheDataType::kAuto) return false;
  if (query.rank != 3 || out.rank != 3 || k_cache.rank != 4 || v_cache.rank != 4) return false;
  if (!query.IsContiguous() || !out.IsContiguous()) return false;

  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1];
  const int64_t d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t nkv = k_cache.shape[2];
  if (d != k_cache.shape[3] || d != v_cache.shape[3]) return false;
  if (hq % nkv != 0) return false;
  // TILE constraints used by sdpa_decode validation.
  if ((d % 32) != 0 || (block_size % 32) != 0) return false;
  if (block_table.rank != 2 || seq_lens.rank != 1 || query_start_loc.rank != 1) return false;

  // Query may stay device-resident after rope — do not force EnsureHost(query).
  EnsureHost(block_table);
  EnsureHost(seq_lens);
  EnsureHost(query_start_loc);

  const int64_t num_reqs = seq_lens.shape[0];
  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  // Pure decode batch: one query token per request.
  if (total_q != num_reqs) return false;
  for (int64_t r = 0; r < num_reqs; ++r) {
    if (qsl[r + 1] - qsl[r] != 1) return false;
    if (slens[r] <= 0) return false;
  }

  const int64_t max_blocks = block_table.shape[1];
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];

  // page_table [B, max_blocks] + highest physical block id we must cover.
  std::vector<int32_t> pt(static_cast<size_t>(num_reqs * max_blocks));
  int32_t max_phys = -1;
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t c = 0; c < max_blocks; ++c) {
      const int32_t id = btab[r * bt_row + c * bt_col];
      pt[static_cast<size_t>(r * max_blocks + c)] = id;
      if (id > max_phys) max_phys = id;
    }
  }
  if (max_phys < 0) return false;
  const uint32_t used_nb = static_cast<uint32_t>(max_phys) + 1u;
  if (static_cast<int64_t>(used_nb) > k_cache.shape[0]) return false;

  try {
    MeshDevice& device = SharedMeshDevice();
    ttnn::Tensor dev_k = EnsurePagedKvTtnn(k_cache, device, used_nb);
    ttnn::Tensor dev_v = EnsurePagedKvTtnn(v_cache, device, used_nb);

    const uint32_t Bu = static_cast<uint32_t>(num_reqs);
    const uint32_t hu = static_cast<uint32_t>(hq);
    const uint32_t du = static_cast<uint32_t>(d);

    // Q: [1, B, H, D]. Prefer reshape of a resident [B*H, D] / [B, H*D] shadow
    // (post device rope) so we never download then re-upload.
    bool identity_q = true;
    for (int64_t r = 0; r < num_reqs; ++r) {
      if (qsl[r] != r) {
        identity_q = false;
        break;
      }
    }
    ttnn::Tensor dev_q;
    bool q_from_device = false;
    if (identity_q) {
      try {
        Tensor q_flat = query.View({total_q * hq, d});
        ttnn::Tensor dev_q2d = EnsureDevice2D(q_flat, device);
        dev_q = ttnn::reshape(dev_q2d, ttnn::Shape({1u, Bu, hu, du}));
        q_from_device = true;
      } catch (const std::exception&) {
        q_from_device = false;
      }
    }
    if (!q_from_device) {
      EnsureHost(query);
      std::vector<float> q_host(static_cast<size_t>(num_reqs * hq * d));
      for (int64_t r = 0; r < num_reqs; ++r) {
        const int64_t t = qsl[r];
        for (int64_t h = 0; h < hq; ++h) {
          for (int64_t e = 0; e < d; ++e) {
            q_host[static_cast<size_t>((r * hq + h) * d + e)] =
                LoadElemF32(query, (t * hq + h) * d + e);
          }
        }
      }
      dev_q = ttnn::Tensor::from_vector<float>(
          q_host, SpecOf(tt::tt_metal::Shape({1u, Bu, hu, du}), ttnn::DataType::BFLOAT16,
                         ttnn::Layout::TILE),
          &device);
    }
    ttnn::Tensor dev_pt = ttnn::Tensor::from_vector<int32_t>(
        pt, SpecOf(tt::tt_metal::Shape({Bu, static_cast<uint32_t>(max_blocks)}),
                   ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
        &device);

    // cur_pos [B] = seq_len - 1
    std::vector<int32_t> cpos(static_cast<size_t>(num_reqs));
    for (int64_t r = 0; r < num_reqs; ++r) cpos[static_cast<size_t>(r)] = slens[r] - 1;
    ttnn::Tensor dev_pos = ttnn::Tensor::from_vector<int32_t>(
        cpos, SpecOf(tt::tt_metal::Shape({Bu}), ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
        &device);

    const auto grid = device.compute_with_storage_grid_size();
    ttnn::operations::transformer::SDPAProgramConfig prog{
        grid,
        std::nullopt,
        /*q_chunk_size=*/32,
        /*k_chunk_size=*/32,
        /*exp_approx_mode=*/false,
        /*max_cores_per_head_batch=*/16};

    ttnn::Tensor dev_out = ttnn::transformer::paged_scaled_dot_product_attention_decode(
        dev_q, dev_k, dev_v, dev_pt,
        /*is_causal=*/true,
        /*attn_mask=*/std::nullopt,
        /*cur_pos_tensor=*/dev_pos,
        /*attention_sink=*/std::nullopt,
        /*scale=*/args.scale,
        /*sliding_window_size=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        /*program_config=*/prog,
        /*compute_kernel_config=*/std::nullopt,
        /*paged_cache_geometry=*/std::nullopt,
        /*cache_position_modulo=*/std::nullopt);

    // Prefer keeping activations on device for o_proj: flatten to [B, H*D].
    // Pure-decode with identity token order (qsl[r]==r) matches out's storage
    // layout [T,H,D] == [B,H,D] so Reshape→MatmulBT hits EnsureDevice2D.
    bool identity_order = true;
    for (int64_t r = 0; r < num_reqs; ++r) {
      if (qsl[r] != r) {
        identity_order = false;
        break;
      }
    }
    if (identity_order && total_q == num_reqs) {
      try {
        ttnn::Tensor flat =
            ttnn::reshape(dev_out, ttnn::Shape({Bu, static_cast<uint32_t>(hq * d)}));
        CommitDeviceLogical2D(out, std::move(flat), Bu, static_cast<uint32_t>(hq * d));
        return true;
      } catch (const std::exception&) {
        // Fall through to host materialization.
      }
    }

    // Output ~ [1, B, H, D] → host [B, H, D] in request order, then scatter to
    // global query token indices.
    std::vector<float> result = dev_out.to_vector<float>();
    VT_CHECK(static_cast<int64_t>(result.size()) >= num_reqs * hq * d,
             "tenstorrent device PA: unexpected output size");
    for (int64_t r = 0; r < num_reqs; ++r) {
      const int64_t t = qsl[r];
      for (int64_t h = 0; h < hq; ++h) {
        for (int64_t e = 0; e < d; ++e) {
          const float v = result[static_cast<size_t>((r * hq + h) * d + e)];
          StoreElemF32(out, (t * hq + h) * d + e, v);
        }
      }
    }
    CommitHost(out);
    return true;
  } catch (const std::exception&) {
    // Fall back to host oracle (shape/grid/dtype edge cases).
    return false;
  }
}

// Multi-token (prefill) device PA via ttnn::chunked_scaled_dot_product_attention.
// Each request is processed independently (B=1) so seq lengths need not match.
// Query tokens for request r cover absolute positions [seq_len-q_len, seq_len);
// chunk_start must be a multiple of the program q/k chunk size (32).
// Pads the last Q chunk with zeros (causal → pad queries do not affect earlier
// real positions). Returns true if every request was written.
bool TryPagedAttentionDevicePrefill(Tensor& out, const Tensor& query, const Tensor& k_cache,
                                    const Tensor& v_cache, const Tensor& block_table,
                                    const Tensor& seq_lens, const Tensor& query_start_loc,
                                    const PagedAttentionArgs& args) {
  if (!args.causal || args.logits_soft_cap > 0.0f) return false;
  if (args.window_size.has_value()) return false;
  if (args.kv_cache_dtype != Fp8KVCacheDataType::kAuto) return false;
  if (query.rank != 3 || out.rank != 3 || k_cache.rank != 4 || v_cache.rank != 4) return false;
  if (!query.IsContiguous() || !out.IsContiguous()) return false;

  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1];
  const int64_t d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t nkv = k_cache.shape[2];
  if (d != k_cache.shape[3] || d != v_cache.shape[3]) return false;
  if (hq % nkv != 0) return false;
  if ((d % 32) != 0 || (block_size % 32) != 0) return false;
  if (block_table.rank != 2 || seq_lens.rank != 1 || query_start_loc.rank != 1) return false;

  EnsureHost(query);
  EnsureHost(k_cache);
  EnsureHost(v_cache);
  EnsureHost(block_table);
  EnsureHost(seq_lens);
  EnsureHost(query_start_loc);

  const int64_t num_reqs = seq_lens.shape[0];
  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  if (qsl[num_reqs] != total_q) return false;

  // Chunk size must divide TILE and match program_config (decode uses 32 too).
  constexpr int64_t kChunk = 32;
  bool any_multi = false;
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q_len = static_cast<int64_t>(qsl[r + 1] - qsl[r]);
    const int64_t seq = static_cast<int64_t>(slens[r]);
    if (q_len <= 0 || seq < q_len) return false;
    const int64_t chunk_start = seq - q_len;
    if ((chunk_start % kChunk) != 0) return false;
    if (q_len > 1) any_multi = true;
  }
  // Pure-decode batches stay on the decode SDPA path.
  if (!any_multi) return false;

  const int64_t max_blocks = block_table.shape[1];
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];

  int32_t max_phys = -1;
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t c = 0; c < max_blocks; ++c) {
      const int32_t id = btab[r * bt_row + c * bt_col];
      if (id > max_phys) max_phys = id;
    }
  }
  if (max_phys < 0) return false;
  const uint32_t used_nb = static_cast<uint32_t>(max_phys) + 1u;
  if (static_cast<int64_t>(used_nb) > k_cache.shape[0]) return false;

  try {
    MeshDevice& device = SharedMeshDevice();
    ttnn::Tensor dev_k = EnsurePagedKvTtnn(k_cache, device, used_nb);
    ttnn::Tensor dev_v = EnsurePagedKvTtnn(v_cache, device, used_nb);

    const auto grid = device.compute_with_storage_grid_size();
    ttnn::operations::transformer::SDPAProgramConfig prog{
        grid,
        std::nullopt,
        /*q_chunk_size=*/static_cast<uint32_t>(kChunk),
        /*k_chunk_size=*/static_cast<uint32_t>(kChunk),
        /*exp_approx_mode=*/false,
        /*max_cores_per_head_batch=*/16};

    const uint32_t hu = static_cast<uint32_t>(hq);
    const uint32_t du = static_cast<uint32_t>(d);

    // Single-request contiguous prefill (qsl[0]==0, q_len==total_q): keep
    // result on device as [T, H*D] for o_proj. Multi-chunk uses permute+concat
    // so we never host-materialize the full sequence.
    const bool device_prefill_out =
        num_reqs == 1 && qsl[0] == 0 &&
        static_cast<int64_t>(qsl[1] - qsl[0]) == total_q && total_q > 0;

    std::vector<ttnn::Tensor> out_pieces;  // each [n_real, H*D]
    out_pieces.reserve(static_cast<size_t>((total_q + kChunk - 1) / kChunk));

    for (int64_t r = 0; r < num_reqs; ++r) {
      const int64_t q_begin = static_cast<int64_t>(qsl[r]);
      const int64_t q_len = static_cast<int64_t>(qsl[r + 1] - qsl[r]);
      const int64_t seq = static_cast<int64_t>(slens[r]);
      const int64_t chunk_start0 = seq - q_len;

      // Page table for this request: [1, max_blocks].
      std::vector<int32_t> pt(static_cast<size_t>(max_blocks));
      for (int64_t c = 0; c < max_blocks; ++c) {
        pt[static_cast<size_t>(c)] = btab[r * bt_row + c * bt_col];
      }
      // KV address space must cover seq (and any Q pad to kChunk).
      const int64_t q_pad = ((q_len + kChunk - 1) / kChunk) * kChunk;
      const int64_t need_kv = chunk_start0 + q_pad;
      if (max_blocks * block_size < need_kv) return false;

      ttnn::Tensor dev_pt = ttnn::Tensor::from_vector<int32_t>(
          pt, SpecOf(tt::tt_metal::Shape({1u, static_cast<uint32_t>(max_blocks)}),
                     ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);

      for (int64_t local = 0; local < q_len; local += kChunk) {
        const int64_t n_real = std::min(kChunk, q_len - local);
        const int64_t chunk_start = chunk_start0 + local;
        // Pack Q chunk [1, H, kChunk, D] (pad tail with zeros).
        std::vector<float> q_host(static_cast<size_t>(hq) * static_cast<size_t>(kChunk) * d, 0.0f);
        for (int64_t i = 0; i < n_real; ++i) {
          const int64_t t = q_begin + local + i;
          for (int64_t h = 0; h < hq; ++h) {
            for (int64_t e = 0; e < d; ++e) {
              q_host[static_cast<size_t>((h * kChunk + i) * d + e)] =
                  LoadElemF32(query, (t * hq + h) * d + e);
            }
          }
        }
        ttnn::Tensor dev_q = ttnn::Tensor::from_vector<float>(
            q_host,
            SpecOf(tt::tt_metal::Shape({1u, hu, static_cast<uint32_t>(kChunk), du}),
                   ttnn::DataType::BFLOAT16, ttnn::Layout::TILE),
            &device);

        ttnn::Tensor dev_out = ttnn::transformer::chunked_scaled_dot_product_attention(
            dev_q, dev_k, dev_v, dev_pt, chunk_start, /*scale=*/args.scale,
            /*memory_config=*/std::nullopt, /*program_config=*/prog,
            /*compute_kernel_config=*/std::nullopt, /*paged_cache_geometry=*/std::nullopt);

        if (device_prefill_out) {
          // [1, H, S, D] → [1, S, H, D] → slice real tokens → [n_real, H*D].
          ttnn::Tensor perm =
              ttnn::permute(dev_out, ttsl::SmallVector<int64_t>{0, 2, 1, 3});
          if (n_real < kChunk) {
            const uint32_t nr = static_cast<uint32_t>(n_real);
            perm = ttnn::slice(perm, ttsl::SmallVector<uint32_t>{0, 0, 0, 0},
                               ttsl::SmallVector<uint32_t>{1u, nr, hu, du},
                               ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
          }
          const uint32_t nr = static_cast<uint32_t>(n_real);
          out_pieces.push_back(
              ttnn::reshape(perm, ttnn::Shape({nr, static_cast<uint32_t>(hq * d)})));
          continue;
        }

        std::vector<float> result = dev_out.to_vector<float>();
        // Expected dense logical [1, H, kChunk, D].
        VT_CHECK(static_cast<int64_t>(result.size()) >= hq * kChunk * d,
                 "tenstorrent device prefill PA: unexpected output size");
        for (int64_t i = 0; i < n_real; ++i) {
          const int64_t t = q_begin + local + i;
          for (int64_t h = 0; h < hq; ++h) {
            for (int64_t e = 0; e < d; ++e) {
              const float v = result[static_cast<size_t>((h * kChunk + i) * d + e)];
              StoreElemF32(out, (t * hq + h) * d + e, v);
            }
          }
        }
      }
    }

    if (device_prefill_out && !out_pieces.empty()) {
      ttnn::Tensor flat = out_pieces.size() == 1
                              ? std::move(out_pieces[0])
                              : ttnn::concat(out_pieces, /*dim=*/0);
      CommitDeviceLogical2D(out, std::move(flat), static_cast<uint32_t>(total_q),
                            static_cast<uint32_t>(hq * d));
      return true;
    }
    CommitHost(out);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// kPagedAttention: causal/non-causal GQA softmax over the paged NHD cache
// (cpu_paged_attn.cpp PagedAttentionKernel). Host-staged f32 oracle matching
// the CPU reference while Alloc is host memory.
//
// Device paths (ttnn-layout K/V shadows [nb,nkv,bs,d]):
//   * pure decode → paged_scaled_dot_product_attention_decode
//   * multi-token prefill → chunked_scaled_dot_product_attention (per request)
// Host NHD stays the source of truth for ReshapeAndCache / LMCache plane layout.
//
// Host perf levers:
//   * NEON FMA dots/axpy + Q hoist + specialized f32/bf16 loads
//   * Short single-req sequences: gather pages → dense [seq,nkv,d] (capped) for
//     sequential inner loops / GQA reuse
//   * Prefill (T*H ≥ 64): small dedicated 4–16 thread pool (not 128-core global)
//
// Correctness: each (t,h) writes a disjoint out row with the same j-order and
// max-subtracted softmax as the serial path → bit-identical to n_threads==1.
//
// This step supports: F32/BF16/F16 query/cache, f32/bf16 out, kAuto KV (no
// fp8), optional softcap and window_size (same math as CPU). OPT-125m uses
// causal + full window + no softcap.
void PagedAttentionKernel(Queue&, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table,
                          const Tensor& seq_lens, const Tensor& query_start_loc,
                          const PagedAttentionArgs& args) {
  // Device SDPA (ttnn-layout shadows). Softcap / window / odd TILE dims /
  // misaligned chunk starts fall through to the host oracle.
  if (TryPagedAttentionDeviceDecode(out, query, k_cache, v_cache, block_table, seq_lens,
                                    query_start_loc, args)) {
    return;
  }
  if (TryPagedAttentionDevicePrefill(out, query, k_cache, v_cache, block_table, seq_lens,
                                     query_start_loc, args)) {
    return;
  }

  EnsureHost(query);
  EnsureHost(k_cache);
  EnsureHost(v_cache);
  EnsureHost(block_table);
  EnsureHost(seq_lens);
  EnsureHost(query_start_loc);
  VT_CHECK(query.rank == 3 && out.rank == 3 && k_cache.rank == 4 && v_cache.rank == 4,
           "tenstorrent kPagedAttention: query/out rank-3, caches rank-4");
  VT_CHECK(IsFloatDType(query.dtype) && (out.dtype == DType::kF32 || out.dtype == DType::kBF16) &&
               IsFloatDType(k_cache.dtype) && k_cache.dtype == v_cache.dtype,
           "tenstorrent kPagedAttention: float query/cache, f32/bf16 out");
  VT_CHECK(args.kv_cache_dtype == Fp8KVCacheDataType::kAuto,
           "tenstorrent kPagedAttention: fp8 KV not supported in this step");
  VT_CHECK(args.scale > 0.0f, "tenstorrent kPagedAttention: scale must be > 0");
  VT_CHECK(query.IsContiguous() && out.IsContiguous() && seq_lens.IsContiguous() &&
               query_start_loc.IsContiguous(),
           "tenstorrent kPagedAttention: query/out/seq_lens/query_start_loc contiguous");

  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1];
  const int64_t d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];
  VT_CHECK(d == k_cache.shape[3], "tenstorrent kPagedAttention: head_size mismatch");
  VT_CHECK(hq % num_kv_heads == 0, "tenstorrent kPagedAttention: GQA ratio");
  const int64_t qpk = hq / num_kv_heads;
  const float scale = args.scale;
  const float softcap = args.logits_soft_cap;
  const int64_t window_left = args.window_size.has_value() ? args.window_size->left : -1;
  const int64_t window_right = args.window_size.has_value() ? args.window_size->right : -1;

  const int64_t num_reqs = seq_lens.shape[0];
  const int32_t* qsl = query_start_loc.Ptr<int32_t>();
  const int32_t* slens = seq_lens.Ptr<int32_t>();
  const int32_t* btab = block_table.Ptr<int32_t>();
  const int64_t bt_row = block_table.stride[0], bt_col = block_table.stride[1];
  const int64_t kc_blk = k_cache.stride[0], kc_pg = k_cache.stride[1], kc_hd = k_cache.stride[2];
  const int64_t vc_blk = v_cache.stride[0], vc_pg = v_cache.stride[1], vc_hd = v_cache.stride[2];

  std::vector<int32_t> tok_pos(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_slen(static_cast<size_t>(total_q));
  std::vector<int32_t> tok_req(static_cast<size_t>(total_q));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t q0 = qsl[r], q1 = qsl[r + 1];
    const int64_t query_len = q1 - q0;
    if (query_len <= 0) continue;
    const int64_t seqlen = slens[r];
    const int64_t context = seqlen - query_len;
    for (int64_t local = 0; local < query_len; ++local) {
      tok_pos[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(context + local);
      tok_slen[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(seqlen);
      tok_req[static_cast<size_t>(q0 + local)] = static_cast<int32_t>(r);
    }
  }

  // F32 contiguous inner kernels (NEON on aarch64). Bit-identical to scalar
  // for normal finite scores (same j order / max-subtracted softmax).
  auto dot_f32 = [](const float* a, const float* b, int64_t n) -> float {
#if defined(__aarch64__)
    float32x4_t vacc = vdupq_n_f32(0.0f);
    int64_t e = 0;
    for (; e + 4 <= n; e += 4) {
      vacc = vfmaq_f32(vacc, vld1q_f32(a + e), vld1q_f32(b + e));
    }
    float sum = vaddvq_f32(vacc);
    for (; e < n; ++e) sum += a[e] * b[e];
    return sum;
#else
    float sum = 0.0f;
    for (int64_t e = 0; e < n; ++e) sum += a[e] * b[e];
    return sum;
#endif
  };
  auto axpy_f32 = [](float* acc, const float* v, float pw, int64_t n) {
#if defined(__aarch64__)
    const float32x4_t vp = vdupq_n_f32(pw);
    int64_t e = 0;
    for (; e + 4 <= n; e += 4) {
      float32x4_t a = vld1q_f32(acc + e);
      a = vfmaq_f32(a, vp, vld1q_f32(v + e));
      vst1q_f32(acc + e, a);
    }
    for (; e < n; ++e) acc[e] += pw * v[e];
#else
    for (int64_t e = 0; e < n; ++e) acc[e] += pw * v[e];
#endif
  };

  // Contiguous float loaders — dtype is fixed for the whole kernel call.
  // Captured bases avoid per-element Tensor::Ptr + switch in the hot loop.
  const DType q_dt = query.dtype, k_dt = k_cache.dtype, v_dt = v_cache.dtype,
              o_dt = out.dtype;
  const float* q_f = (q_dt == DType::kF32) ? query.Ptr<float>() : nullptr;
  const uint16_t* q_h =
      (q_dt == DType::kBF16 || q_dt == DType::kF16) ? query.Ptr<uint16_t>() : nullptr;
  const float* k_f = (k_dt == DType::kF32) ? k_cache.Ptr<float>() : nullptr;
  const uint16_t* k_h =
      (k_dt == DType::kBF16 || k_dt == DType::kF16) ? k_cache.Ptr<uint16_t>() : nullptr;
  const float* v_f = (v_dt == DType::kF32) ? v_cache.Ptr<float>() : nullptr;
  const uint16_t* v_h =
      (v_dt == DType::kBF16 || v_dt == DType::kF16) ? v_cache.Ptr<uint16_t>() : nullptr;
  float* o_f = (o_dt == DType::kF32) ? out.Ptr<float>() : nullptr;
  uint16_t* o_h = (o_dt == DType::kBF16) ? out.Ptr<uint16_t>() : nullptr;

  auto load_half = [](DType dt, const uint16_t* base, int64_t i) -> float {
    return dt == DType::kBF16 ? BF16ToF32(base[i]) : F16ToF32(base[i]);
  };

  // Work unit = (token, head). Decode has total_q==1 so the head axis is the
// useful one. Small dedicated pool (not the 128-thread global) for prefill
// fan-out; decode (nwork=hq≈16) stays serial — ParallelForRows barriers still
// lose to the NEON body at that size (measured).
  const int64_t nwork = total_q * hq;
  auto& pa_pool = []() -> vt::cpu::Threadpool& {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int n = std::clamp(hw > 0 ? hw / 8 : 8, 4, 16);
    static vt::cpu::Threadpool pool(n);
    return pool;
  }();
  // Prefill (T*H large) parallelizes; pure decode (T=1,H=16) stays serial.
  constexpr int64_t kPaParallelMinWork = 64;

  // Single-request causal, full-window, no softcap: gather paged K/V into
  // dense [seq, nkv, d] once so the inner j-loop is sequential (better for
  // NEON + GQA reuse). Cap size — long-seq gather doubles traffic and loses
  // (measured: seq=512 gather ~20ms vs paged walk ~7ms). Short decode/prefill
  // stays under the cap and benefits.
  constexpr int64_t kGatherMaxElems = 32 * 1024;  // floats per of K/V (~128 KiB)
  std::vector<float> k_dense, v_dense;
  int64_t gather_seq = 0;
  if (num_reqs == 1 && args.causal && softcap <= 0.0f && window_left < 0 &&
      window_right < 0 && total_q > 0) {
    gather_seq = tok_slen[0];
    const int64_t gather_elems = gather_seq * num_kv_heads * d;
    if (gather_seq > 0 && gather_elems <= kGatherMaxElems) {
      k_dense.resize(static_cast<size_t>(gather_elems));
      v_dense.resize(static_cast<size_t>(gather_elems));
      auto load_k_i = [&](int64_t i) -> float {
        return k_f != nullptr ? k_f[i] : load_half(k_dt, k_h, i);
      };
      auto load_v_i = [&](int64_t i) -> float {
        return v_f != nullptr ? v_f[i] : load_half(v_dt, v_h, i);
      };
      for (int64_t j = 0; j < gather_seq; ++j) {
        const int64_t blk = btab[(j / block_size) * bt_col];
        const int64_t off = j % block_size;
        for (int64_t g = 0; g < num_kv_heads; ++g) {
          const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
          const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
          const int64_t dst = (j * num_kv_heads + g) * d;
          for (int64_t e = 0; e < d; ++e) {
            k_dense[static_cast<size_t>(dst + e)] = load_k_i(kbase + e);
            v_dense[static_cast<size_t>(dst + e)] = load_v_i(vbase + e);
          }
        }
      }
    }
  }
  const float* k_dense_p = k_dense.empty() ? nullptr : k_dense.data();
  const float* v_dense_p = v_dense.empty() ? nullptr : v_dense.data();

  auto process_range = [&](int64_t w0, int64_t w1) {
    std::vector<float> probs;
    std::vector<float> acc(static_cast<size_t>(d));
    std::vector<float> qloc(static_cast<size_t>(d));
    std::vector<float> kvloc;  // bf16/f16 K/V page staging (paged path)
    const bool k_is_f32 = k_f != nullptr;
    const bool v_is_f32 = v_f != nullptr;
    if (k_dense_p == nullptr && (!k_is_f32 || !v_is_f32))
      kvloc.resize(static_cast<size_t>(d));

    for (int64_t w = w0; w < w1; ++w) {
      const int64_t t = w / hq;
      const int64_t h = w % hq;
      const int64_t r = tok_req[static_cast<size_t>(t)];
      const int64_t p = tok_pos[static_cast<size_t>(t)];
      const int64_t seqlen = tok_slen[static_cast<size_t>(t)];
      const int64_t jmin = window_left >= 0 ? std::max<int64_t>(0, p - window_left) : 0;
      int64_t jmax = args.causal ? p : seqlen - 1;
      if (window_right >= 0) jmax = std::min(jmax, p + window_right);
      jmax = std::min(jmax, seqlen - 1);
      if (jmax < jmin) continue;

      const int64_t g = h / qpk;
      const int64_t qoff = (t * hq + h) * d;
      // Hoist Q head once (was re-loaded for every key position).
      if (q_f != nullptr) {
        std::memcpy(qloc.data(), q_f + qoff, static_cast<size_t>(d) * sizeof(float));
      } else {
        for (int64_t e = 0; e < d; ++e) qloc[static_cast<size_t>(e)] = load_half(q_dt, q_h, qoff + e);
      }

      probs.assign(static_cast<size_t>(jmax - jmin + 1), 0.0f);
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = jmin; j <= jmax; ++j) {
        float dot;
        if (k_dense_p != nullptr) {
          const int64_t kbase = (j * num_kv_heads + g) * d;
          dot = dot_f32(qloc.data(), k_dense_p + kbase, d);
        } else {
          const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
          const int64_t off = j % block_size;
          const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
          if (k_is_f32) {
            dot = dot_f32(qloc.data(), k_f + kbase, d);
          } else {
            for (int64_t e = 0; e < d; ++e)
              kvloc[static_cast<size_t>(e)] = load_half(k_dt, k_h, kbase + e);
            dot = dot_f32(qloc.data(), kvloc.data(), d);
          }
        }
        dot *= scale;
        if (softcap > 0.0f) dot = softcap * std::tanh(dot / softcap);
        probs[static_cast<size_t>(j - jmin)] = dot;
        if (dot > m) m = dot;
      }
      float denom = 0.0f;
      for (int64_t j = jmin; j <= jmax; ++j) {
        const float e = std::exp(probs[static_cast<size_t>(j - jmin)] - m);
        probs[static_cast<size_t>(j - jmin)] = e;
        denom += e;
      }
      const float inv = 1.0f / denom;
      std::fill(acc.begin(), acc.end(), 0.0f);
      for (int64_t j = jmin; j <= jmax; ++j) {
        const float pw = probs[static_cast<size_t>(j - jmin)] * inv;
        if (v_dense_p != nullptr) {
          const int64_t vbase = (j * num_kv_heads + g) * d;
          axpy_f32(acc.data(), v_dense_p + vbase, pw, d);
        } else {
          const int64_t blk = btab[r * bt_row + (j / block_size) * bt_col];
          const int64_t off = j % block_size;
          const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
          if (v_is_f32) {
            axpy_f32(acc.data(), v_f + vbase, pw, d);
          } else {
            for (int64_t e = 0; e < d; ++e)
              kvloc[static_cast<size_t>(e)] = load_half(v_dt, v_h, vbase + e);
            axpy_f32(acc.data(), kvloc.data(), pw, d);
          }
        }
      }
      if (o_f != nullptr) {
        std::memcpy(o_f + qoff, acc.data(), static_cast<size_t>(d) * sizeof(float));
      } else {
        for (int64_t e = 0; e < d; ++e) o_h[qoff + e] = F32ToBF16(acc[static_cast<size_t>(e)]);
      }
    }
  };

  if (nwork >= kPaParallelMinWork && pa_pool.NThreads() > 1) {
    vt::cpu::ParallelForRows(pa_pool, nwork, process_range);
  } else {
    process_range(0, nwork);
  }
  CommitHost(out);
}

// kGreedyArgmax: per-row lowest-index max of f32 logits (cpu_sample.cpp).
// OPT's lm_head produces F32 logits; host-staged, bit-exact with CPU.
void GreedyArgmaxKernel(Queue&, Tensor& token_ids, const Tensor& logits) {
  VT_CHECK(logits.rank == 2 && logits.dtype == DType::kF32 && logits.IsContiguous(),
           "tenstorrent kGreedyArgmax: logits must be contiguous f32 [N,V]");
  VT_CHECK(token_ids.rank == 1 && token_ids.dtype == DType::kI64 && token_ids.IsContiguous() &&
               token_ids.shape[0] == logits.shape[0],
           "tenstorrent kGreedyArgmax: token_ids must be i64 [N]");
  EnsureHost(logits);
  const int64_t n = logits.shape[0], v = logits.shape[1];
  const float* lp = logits.Ptr<float>();
  int64_t* out = token_ids.Ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const float* row = lp + i * v;
    int64_t best = 0;
    float best_v = row[0];
    for (int64_t j = 1; j < v; ++j) {
      if (row[j] > best_v) {
        best_v = row[j];
        best = j;
      }
    }
    out[i] = best;
  }
  CommitHost(token_ids);
}

struct Registrar {
  Registrar() {
    if (!DeviceAvailable()) return;
    RegisterOp(OpId::kMatmul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernel)));
    RegisterOp(OpId::kAdd, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastBf16Kernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastF32Kernel)));
    RegisterOp(OpId::kRopeNeox, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeFn>(&RopeNeoxKernel)));
    RegisterOp(OpId::kRopeCosSinCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeCosSinCacheFn>(&RopeCosSinCacheKernel)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RopeFromCacheFn>(&RopeFromCacheKernel)));
    RegisterOp(OpId::kQkvSplit, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernel)));
  }
} registrar;

}  // namespace

// ---- ttnn mesh-trace capture (Backend graph-capture mapping) ----------------
// Process-local single-slot capture + multi-graph handles (opaque MeshTraceId*).
// Mirrors the CUDA backend's single-exec_ vs EndCaptureGraph split.

namespace {
struct TraceState {
  bool capturing = false;
  bool has_replay = false;
  ttnn::MeshTraceId capturing_id{0};
  ttnn::MeshTraceId replay_id{0};
};
TraceState& TraceSlot() {
  static TraceState s;
  return s;
}
constexpr auto kTraceCq = ttnn::QueueId(0);
}  // namespace

void TraceBeginCapture() {
  TraceState& s = TraceSlot();
  VT_CHECK(!s.capturing, "tenstorrent: nested TraceBeginCapture");
  MeshDevice& device = SharedMeshDevice();
  s.capturing_id = ttnn::operations::trace::begin_trace_capture(&device, kTraceCq);
  s.capturing = true;
}

void TraceEndCapture() {
  TraceState& s = TraceSlot();
  VT_CHECK(s.capturing, "tenstorrent: TraceEndCapture without Begin");
  MeshDevice& device = SharedMeshDevice();
  ttnn::operations::trace::end_trace_capture(&device, s.capturing_id, kTraceCq);
  // Drop previous single-slot replay if any.
  if (s.has_replay) {
    try {
      ttnn::operations::trace::release_trace(&device, s.replay_id);
    } catch (...) {
    }
  }
  s.replay_id = s.capturing_id;
  s.has_replay = true;
  s.capturing = false;
}

void TraceReplay() {
  TraceState& s = TraceSlot();
  VT_CHECK(!s.capturing, "tenstorrent: TraceReplay during capture");
  VT_CHECK(s.has_replay, "tenstorrent: TraceReplay with no captured trace");
  MeshDevice& device = SharedMeshDevice();
  ttnn::operations::trace::execute_trace(&device, s.replay_id, kTraceCq, /*blocking=*/true);
}

void* TraceEndCaptureGraph() {
  TraceState& s = TraceSlot();
  VT_CHECK(s.capturing, "tenstorrent: TraceEndCaptureGraph without Begin");
  MeshDevice& device = SharedMeshDevice();
  ttnn::operations::trace::end_trace_capture(&device, s.capturing_id, kTraceCq);
  s.capturing = false;
  // Opaque handle: heap-allocated MeshTraceId for the multi-graph API.
  return new ttnn::MeshTraceId(s.capturing_id);
}

void TraceReplayGraph(void* graph) {
  VT_CHECK(graph != nullptr, "tenstorrent: TraceReplayGraph null");
  VT_CHECK(!TraceSlot().capturing, "tenstorrent: TraceReplayGraph during capture");
  MeshDevice& device = SharedMeshDevice();
  const auto id = *static_cast<ttnn::MeshTraceId*>(graph);
  ttnn::operations::trace::execute_trace(&device, id, kTraceCq, /*blocking=*/true);
}

void TraceDestroyGraph(void* graph) {
  if (graph == nullptr) return;
  auto* id = static_cast<ttnn::MeshTraceId*>(graph);
  try {
    MeshDevice& device = SharedMeshDevice();
    ttnn::operations::trace::release_trace(&device, *id);
  } catch (...) {
  }
  delete id;
}

// ---- Called from TenstorrentBackend::Alloc/Free/Copy (no ttnn in that TU). ----
void RegisterHostBuffer(void* host, size_t bytes) {
  if (host == nullptr) return;
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot s;
  s.host = host;
  s.bytes = bytes;
  s.host_current = true;
  s.device_current = false;
  Slots()[reinterpret_cast<uintptr_t>(host)] = std::move(s);
}

void UnregisterHostBuffer(void* host) {
  if (host == nullptr) return;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    Slots().erase(reinterpret_cast<uintptr_t>(host));
  }
  DropPagedKvShadow(host);
  DropEmbedTableShadow(host);
}

void MarkHostWritten(void* host) {
  if (host == nullptr) return;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s != nullptr) {
      s->host_current = true;
      s->device_current = false;
      s->device = std::nullopt;
    }
  }
  // Weight tables may be rewritten in place during load — drop embed cache.
  DropEmbedTableShadow(host);
}

void EnsureHostBytes(void* host) {
  if (host == nullptr) return;
  ttnn::Tensor dev;
  size_t bytes = 0;
  void* base = nullptr;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s == nullptr || s->host_current) return;
    VT_CHECK(s->device_current && s->device.has_value(),
             "tenstorrent: EnsureHostBytes with no current device data");
    dev = *s->device;
    bytes = s->bytes;
    base = s->host;
  }
  std::vector<float> result = dev.to_vector<float>();
  const size_t n = result.size();
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(host);
    if (s == nullptr || s->host_current) return;
    // Device results are f32 via to_vector. Host Alloc is typically
    // numel*sizeof(float) (tests/f32 path) or numel*2 (bf16 activations).
    if (bytes >= n * sizeof(float)) {
      std::memcpy(base, result.data(), n * sizeof(float));
    } else if (bytes >= n * sizeof(uint16_t)) {
      auto* dst = static_cast<uint16_t*>(base);
      for (size_t i = 0; i < n; ++i) dst[i] = F32ToBF16(result[i]);
    } else {
      VT_CHECK(false, "tenstorrent: EnsureHostBytes host buffer too small");
    }
    s->host_current = true;
  }
}

}  // namespace vt::tenstorrent
