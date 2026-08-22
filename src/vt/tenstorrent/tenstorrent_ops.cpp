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
// This OBJECT library is not the `vllm` target, so it does not inherit the
// PUBLIC VLLM_CPP_TENSTORRENT define. Force the real declarations; the
// header's inline no-ops are only for CPU/Vulkan/Windows TUs.
#ifndef VLLM_CPP_TENSTORRENT
#define VLLM_CPP_TENSTORRENT
#endif
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
#include <tuple>
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
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/experimental/reshape/view.hpp>
#include <ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp>
#include <ttnn/operations/transformer/sdpa_config.hpp>
#include <ttnn/operations/trace.hpp>
#include <ttnn/common/queue_id.hpp>
// experimental/paged_cache pulls op_profiler which expects a 6-arg
// ___tracy_alloc_srcloc (with color); the TracyC.h on this tree only has 5-arg.
// Temporarily disable Tracy for this include chain so the op headers compile.
#ifdef TRACY_ENABLE
#undef TRACY_ENABLE
#define VT_RESTORE_TRACY_ENABLE 1
#endif
#include <ttnn/operations/experimental/paged_cache/paged_cache.hpp>
#include <ttnn/operations/experimental/plusone/plusone.hpp>
#include <ttnn/operations/core/to_memory_config/to_memory_config_op.hpp>
#include <ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp>

// Forward declare clone (header not in installed includes)
namespace ttnn { Tensor clone(const Tensor&, const std::optional<DataType>&, const std::optional<MemoryConfig>&, const std::optional<DeviceComputeKernelConfig>&); }
// chunked_scaled_dot_product_attention lives in sdpa.hpp and
// ttnn::transformer::chunk_gated_delta_rule in its own op header, but neither
// is in the installed TT-NN include set at our pin; both symbols are exported
// by TTNN::TTNN's _ttnncpp.so. Forward-declare, link via TTNN (same doctrine
// as clone above).
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
// The GDN chunked scan (BACKEND-TENSTORRENT-GDN W1): the pinned source tree
// carries ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/
// chunk_gated_delta_rule.hpp; the signature below mirrors it 1:1.
std::tuple<ttnn::Tensor, std::optional<ttnn::Tensor>> chunk_gated_delta_rule(
    const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v,
    const ttnn::Tensor& g, const ttnn::Tensor& beta,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool output_final_state = false, uint32_t chunk_size = 64,
    bool use_qk_l2norm = false, bool output_head_major = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<ttnn::Tensor>& eye = std::nullopt,
    const std::optional<ttnn::Tensor>& tril = std::nullopt,
    const std::optional<ttnn::Tensor>& ones = std::nullopt,
    const std::optional<ttnn::Tensor>& masks = std::nullopt);
}  // namespace ttnn::transformer
#include <ttnn/operations/data_movement/copy/copy.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/tensor/tensor_ops.hpp>  // create_device_tensor, copy_to_device
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

// Bisection diagnostic: logs op entry during capture (VT_TT_TRACE_DEBUG).
#define TT_OP_TRACE(name)                                          \
  do {                                                             \
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr &&             \
        tt_capture_active())                                       \
      std::fprintf(stderr, "[TT-OP] %s\n", name);                  \
  } while (0)

// ---- Host/device residency -------------------------------------------------
// vt::Tensor.data is always a host pointer from Backend::Alloc. A shadow map
// (Metal AllocMap shape) holds an optional device-resident ttnn::Tensor for
// that host base so multi-op chains need not download after every matmul.

// File-scope capture flag (flipped by TraceBeginCapture/TraceEndCapture) so the
// residency helpers below can detect readbacks during capture (ttnn prohibits
// them). Defined here, before the helpers that query it.
namespace {
bool& tt_capture_active() {
  static bool b = false;
  return b;
}
}  // namespace

// ITEM 5 (rope): persistent device cos/sin (expanded per head), built OUTSIDE
// capture and ttnn::copy'd in-region — the UploadRows in RopeApplyDeviceNeox
// was the enqueue_write that killed capture at mid-layer-0. The cache is
// keyed by (tokens*heads, half) + the exact host cos/sin CONTENT: if the
// step's positions changed the table, we must NOT silently reuse a stale
// cached tensor — during capture that is a hard error (the driver must warm
// the new table first, the SizeSlot::Refresh pattern).
namespace {
std::mutex& RopeCSMutex() {
  static std::mutex m;
  return m;
}
struct RopeCSEntry {
  ttnn::Tensor cos;
  ttnn::Tensor sin;
  std::vector<float> cos_host;  // content identity for the reuse check
};
std::map<std::string, RopeCSEntry>& RopeCSCache() {
  static std::map<std::string, RopeCSEntry> c;
  return c;
}
std::string RopeCSKey(uint32_t th, uint32_t half) {
  return std::to_string(th) + "x" + std::to_string(half);
}
}  // namespace

namespace {
std::mutex& ZeroCacheMutex() {
  static std::mutex m;
  return m;
}
std::map<std::string, ttnn::Tensor>& ZeroCache() {
  static std::map<std::string, ttnn::Tensor> c;
  return c;
}
std::string ZeroCacheKey(const ttnn::Shape& shape, ttnn::DataType dt,
                         ttnn::Layout lt) {
  std::string k;
  for (auto d : shape.view()) k += std::to_string(d) + "x";
  k += std::to_string(static_cast<int>(dt)) + "x" +
       std::to_string(static_cast<int>(lt));
  return k;
}
}  // namespace

ttnn::Tensor ZeroCacheGet(const ttnn::Tensor& like, MeshDevice& device) {
  const std::string key = ZeroCacheKey(like.logical_shape(), like.dtype(),
                                       like.layout());
  std::lock_guard<std::mutex> g(ZeroCacheMutex());
  auto& c = ZeroCache();
  auto it = c.find(key);
  if (it == c.end()) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent: zero-cache miss during capture — warm the "
             "host-free path eagerly (VT_TT_HOST_FREE_DECODE warmup) first");
    it = c.emplace(key, ttnn::zeros(like.logical_shape(), like.dtype(),
                                     like.layout(), std::ref(device)))
             .first;
  }
  return it->second;
}

void ZeroCachePrime(const ttnn::Shape& shape, ttnn::DataType dt,
                    ttnn::Layout lt, MeshDevice& device) {
  const std::string key = ZeroCacheKey(shape, dt, lt);
  std::lock_guard<std::mutex> g(ZeroCacheMutex());
  auto& c = ZeroCache();
  if (c.find(key) == c.end()) {
    c.emplace(key, ttnn::zeros(shape, dt, lt, std::ref(device)));
  }
}


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
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] to_vector readback DURING CAPTURE\n");
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
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] UploadRows ptr=%p rows=%u cols=%u\n",
                 static_cast<const void*>(data), rows, cols);
  std::vector<float> host(data, data + static_cast<size_t>(rows) * cols);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] UploadRows from_vector WRITE during capture\n");
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
  if (HostFreeDecodeEnabled()) {
    // Prime the persistent-zero cache for this spec during the eager warmup
    // (capture-safe zeroing replays ttnn::copy(zero, dst) — see MemsetDevice).
    ZeroCachePrime(ttnn::Shape({rows, cols}), ttnn::DataType::BFLOAT16,
                   ttnn::Layout::TILE, device);
  }
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
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-UP] TryDevicePagedFill from_vector WRITE during capture\n");
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
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-UP] TryDevicePagedFill from_vector WRITE during capture\n");
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
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] MakeHeightShardedUpdateInput from_vector WRITE during capture\n");
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
      std::vector<int32_t> idxs(static_cast<size_t>(C));
      for (uint32_t b = 0; b < C; ++b) {
        pt[static_cast<size_t>(b)] = static_cast<int32_t>(phys_blocks[static_cast<size_t>(base + b)]);
        idxs[static_cast<size_t>(b)] = static_cast<int32_t>(offsets[static_cast<size_t>(base + b)]);
      }
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
        std::fprintf(stderr, "[TT-UP] TryDevicePagedUpdateBatch from_vector WRITE during capture\n");
      ttnn::Tensor page_table = ttnn::Tensor::from_vector<int32_t>(
          pt, SpecOf(tt::tt_metal::Shape({C, 1u}), ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);
      // Paged mode requires update_idxs as a DEVICE tensor (the vector form
      // alone is rejected: "Paged cache requires update_idxs tensor").
      ttnn::Tensor update_idxs_tensor = ttnn::Tensor::from_vector<int32_t>(
          idxs, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(C)}),
                      ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);

      cache_dev = ttnn::experimental::paged_update_cache(
          cache_dev, xt, /*update_idxs=*/{}, update_idxs_tensor,
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
  // The fused op rejects overlapping K/V sharded input grids
  // ("input_tensor1 and input_tensor2 must not overlap"). Both K and V
  // shards land on the same C cores via MakeHeightShardedUpdateInput.
  // Fall back to two separate TryDevicePagedUpdateBatch calls (the paired
  // path at TryDevicePagedPushPair handles this).
  return false;
  (void)k_dev; (void)v_dev; (void)device; (void)k_toks; (void)v_toks;
  (void)nkv; (void)d;  // suppress unused-param warnings
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
// HOST-FREE-DECODE: the device shadow is allocated ONCE at the FULL pool size
// (all num_blocks blocks, tail zero-filled). Sizing it to the used prefix — as
// this function originally did — reallocates and FREES the previous device
// buffer whenever `used_nb` crosses a block boundary. While a mesh trace is
// alive the captured RAC/PA commands still reference the old address; the
// allocator hands it to a new allocation and tt-metal's warning fires
// ("buffers may be corrupted once a trace is executed"). Observed as every
// replay collapsing to all-zero logits at the first block-table growth
// (slot 63→64 with block_size 32). A pool-sized shadow never moves.
ttnn::Tensor EnsurePagedKvTtnn(const Tensor& cache_nhd, MeshDevice& device, uint32_t used_nb) {
  VT_CHECK(cache_nhd.rank == 4 && cache_nhd.IsContiguous(),
           "EnsurePagedKvTtnn: contiguous rank-4 NHD cache");
  const uint32_t pool_nb = static_cast<uint32_t>(cache_nhd.shape[0]);
  const uint32_t bs = static_cast<uint32_t>(cache_nhd.shape[1]);
  const uint32_t nkv = static_cast<uint32_t>(cache_nhd.shape[2]);
  const uint32_t d = static_cast<uint32_t>(cache_nhd.shape[3]);
  VT_CHECK(used_nb > 0 && used_nb <= pool_nb, "EnsurePagedKvTtnn: used_nb out of range");

  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];
    if (s.device_current && s.device.has_value() && s.nb >= used_nb && s.nkv == nkv &&
        s.bs == bs && s.d == d) {
      return *s.device;
    }
    // Mirror grows to the full pool (zero-filled tail) so both mirror and
    // device shadow stay at one stable size for the cache's lifetime.
    EnsureMirrorCapacity(s, pool_nb, nkv, bs, d);
  }

  // Cold / short / geometry change: rebuild the used prefix from the host NHD
  // cache into the (full-size) mirror; the tail stays zero.
  std::vector<float> used;
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];
    if (!s.mirror_valid || s.nkv != nkv || s.bs != bs || s.d != d || s.nb < used_nb) {
      s.mirror_valid = false;  // content below [0,used) not trustworthy yet
    }
  }
  used = NhdToTtnnLayoutPrefix(cache_nhd, used_nb);
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];
    EnsureMirrorCapacity(s, pool_nb, nkv, bs, d);
    VT_CHECK(s.mirror.size() >= used.size(), "EnsurePagedKvTtnn: mirror shorter than used");
    std::memcpy(s.mirror.data(), used.data(), used.size() * sizeof(float));
    s.mirror_valid = true;
  }

  // Build the pool-sized device shadow WITHOUT a full-pool from_vector: the
  // TILE-layout host transform is per-element and a 256-block vector costs
  // seconds per cache (56 caches stalled cold for minutes). Upload only the
  // used prefix, allocate the zero tail with ttnn::zeros (host std::fill +
  // straight DMA — a constant fill is layout-order-agnostic), and stitch with
  // a device-side concat. paged_update_cache / sdpa_decode never read the
  // zero tail (page-table entries only cover allocated blocks).
  const auto used_spec = SpecOf(tt::tt_metal::Shape({used_nb, nkv, bs, d}),
                                ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EnsurePagedKvTtnn from_vector WRITE during capture\n");
  const auto dbg = std::getenv("VT_TT_TRACE_DEBUG") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  ttnn::Tensor dev = ttnn::Tensor::from_vector<float>(used, used_spec, &device);
  if (dbg) {
    const auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[TT-KVTIM] %p from_vector used_nb=%u %.0fms\n",
                 cache_nhd.data, used_nb,
                 std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  if (used_nb < pool_nb) {
    ttnn::Tensor tail = ttnn::zeros(
        tt::tt_metal::Shape({pool_nb - used_nb, nkv, bs, d}), ttnn::DataType::BFLOAT16,
        ttnn::Layout::TILE, std::ref(device));
    if (dbg) {
      const auto t2 = std::chrono::steady_clock::now();
      std::fprintf(stderr, "[TT-KVTIM] %p zeros tail_nb=%u %.0fms\n",
                   cache_nhd.data, pool_nb - used_nb,
                   std::chrono::duration<double, std::milli>(t2 - t0).count());
    }
    dev = ttnn::concat(std::vector<ttnn::Tensor>{dev, tail}, 0);
    if (dbg) {
      const auto t3 = std::chrono::steady_clock::now();
      std::fprintf(stderr, "[TT-KVTIM] %p concat pool_nb=%u %.0fms\n",
                   cache_nhd.data, pool_nb,
                   std::chrono::duration<double, std::milli>(t3 - t0).count());
    }
  }

  std::lock_guard<std::mutex> g(PagedKvMutex());
  PagedKvShadow& s = PagedKvShadows()[reinterpret_cast<uintptr_t>(cache_nhd.data)];
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
  TT_OP_TRACE("Matmul");
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
  TT_OP_TRACE("MatmulBT");
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
  TT_OP_TRACE("Add");
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
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-UP] AddKernel from_vector WRITE during capture\n");
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
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EnsureEmbedTableDevice from_vector WRITE during capture\n");
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
  TT_OP_TRACE("Embedding");
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
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EmbeddingKernel from_vector WRITE during capture\n");
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
  // HOST-FREE-DECODE: when the caller's buffer already carries a CURRENT
  // device shadow of the same shape (the decode-graph driver's PERSISTENT
  // hidden buffer), refresh that shadow IN PLACE (device->device copy) so
  // its device address never moves. A captured region reads the address
  // recorded at capture time; replacing the shadow here would leave every
  // replay reading the capture-step embedding. First call (no shadow yet)
  // commits normally.
  bool in_place = false;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(out.data);
    in_place = s != nullptr && s->device_current && s->device.has_value() &&
               s->dev_rows == t && s->dev_cols == h;
    if (in_place) {
      ttnn::copy(dev_out, *s->device);
      s->host_current = false;
    }
  }
  if (!in_place) CommitDevice2D(out, std::move(dev_out));
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
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] EnsureAffine1D from_vector WRITE during capture\n");
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
  TT_OP_TRACE("LayerNorm");
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
  TT_OP_TRACE("RmsNorm");
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
  // HOST-FREE-FORWARD R1: force the residual merge + RMS device path at T=1 when
  // capture is desired (ttnn trace prohibits host ops in the captured region).
  // Default ON since the R5 flip; VT_TT_HOST_FREE_DECODE=0 opts out (the
  // pre-flip default). Numerics proven by BACKEND-TENSTORRENT-RESIDUAL-GOLDEN.
  const bool host_free_decode = HostFreeDecodeEnabled();
  const bool host_residual = !host_free_decode &&
      (args.gemma || (residual != nullptr && rows < kDeviceResidualMinRows));
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

// kFusedChain: dispatch kFusedAddRmsNormStd to the same device RmsNorm path
// (residual += x; out = rms_norm(residual, weight)). Other recipes fall
// through to the CPU interpreter (host round-trip). Without this registration,
// FusedChain falls back to the CPU kernel which reads HOST memory — fatal
// when the PA output is device-resident (VT_TT_HOST_FREE_DECODE).
void FusedChainKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  TT_OP_TRACE("FusedChain");
  // kFusedAddRmsNormStd: step0 kAdd(residual = x + residual),
  // step1 kRmsNorm(out = rms_norm(residual, weight)).
  // This is exactly RmsNormKernel with the residual parameter.
  if (r.n == 2 &&
      r.steps[0].op == FOp::kAdd && r.steps[0].out == 2 &&
      r.steps[1].op == FOp::kRmsNorm && r.steps[1].out == 3 &&
      r.steps[1].gemma == false) {
    RmsNormKernel(q, out, x, weight, RmsNormArgs{eps, false}, residual);
    return;
  }
  // kFusedAddRmsNorm (gemma variant): same but gemma=true.
  if (r.n == 2 &&
      r.steps[0].op == FOp::kAdd && r.steps[0].out == 2 &&
      r.steps[1].op == FOp::kRmsNorm && r.steps[1].out == 3 &&
      r.steps[1].gemma == true) {
    RmsNormKernel(q, out, x, weight, RmsNormArgs{eps, true}, residual);
    return;
  }
  // Unknown recipe: fall back to host (safe outside capture).
  VT_CHECK(!tt_capture_active(),
           "tenstorrent: unknown FusedChain recipe during capture");
  EnsureHost(out);
  EnsureHost(x);
  EnsureHost(weight);
  if (residual != nullptr) EnsureHost(*residual);
  // Delegate to the CPU interpreter by calling the registered CPU op.
  auto cpu_fn = reinterpret_cast<FusedChainFn>(
      GetOpFallback(OpId::kFusedChain, DeviceType::kTENSTORRENT, "vt-tenstorrent"));
  if (cpu_fn) {
    cpu_fn(q, out, x, weight, residual, r, eps);
  } else {
    VT_CHECK(false, "tenstorrent: no FusedChain fallback available");
  }
}

// kSiluAndMul: SwiGLU gate half — out[i,j] = silu(x[i,j]) * x[i,j+d]
// with d = x.shape[1]/2 (cpu_ops.cpp SiluAndMulKernel). Second Qwen3-dense
// op beyond OPT (MLP: gate_up GEMM -> SiluAndMul -> down GEMM). Device path
// keeps the gate_up → SiluAndMul → down GEMM chain on-device: slice the
// last-dim halves, ttnn::silu(gate), ttnn::multiply by up. BF16 tile path
// (same envelope as matmul/norm); not bit-exact vs host f32.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  TT_OP_TRACE("SiluAndMul");
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
  TT_OP_TRACE("CastBf16");
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
  TT_OP_TRACE("CastF32");
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
  // ITEM 5: persistent cos/sin — build outside capture, copy in-region.
  const std::string rk = RopeCSKey(thu, halfu);
  ttnn::Tensor dev_cos, dev_sin;
  bool cache_hit = false;
  {
    std::lock_guard<std::mutex> g(RopeCSMutex());
    auto it = RopeCSCache().find(rk);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr,
                   "[TT-TRACE] rope lookup key=%s found=%d content_eq=%d "
                   "(want first=%f n=%zu)\n",
                   rk.c_str(), it != RopeCSCache().end(),
                   it != RopeCSCache().end() && it->second.cos_host == cos_exp,
                   cos_exp.empty() ? -1.0f : cos_exp.front(), cos_exp.size());
    if (it != RopeCSCache().end() && it->second.cos_host == cos_exp) {
      dev_cos = it->second.cos;
      dev_sin = it->second.sin;
      cache_hit = true;
    }
  }
  if (!cache_hit) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent: rope cos/sin cache miss during capture — the "
             "table changed (positions moved); the decode-graph driver must "
             "call WarmRopeCosSin for the step's positions BEFORE BeginCapture "
             "(the SizeSlot::Refresh pattern)");
    dev_cos = UploadRows(cos_exp.data(), thu, halfu, device);
    dev_sin = UploadRows(sin_exp.data(), thu, halfu, device);
    std::lock_guard<std::mutex> g(RopeCSMutex());
    RopeCSEntry e;
    e.cos = dev_cos;
    e.sin = dev_sin;
    e.cos_host = cos_exp;
    RopeCSCache()[rk] = std::move(e);
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] rope cos/sin cache %s during capture "
                 "(key th=%u half=%u first=%f)\n",
                 cache_hit ? "HIT" : "MISS", thu, halfu,
                 cos_exp.empty() ? -1.0f : cos_exp.front());

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
  // HOST-FREE-FORWARD R1: force device RoPE at T=1 for capture (see RmsNorm note).
  if (HostFreeDecodeEnabled()) return true;
  return tokens * heads >= 64;
}

// kRopeNeox: Qwen3-dense RoPE. Device NeoX for large [T*H]; host for short decode.
void RopeNeoxKernel(Queue&, Tensor& qs, Tensor& ks, const Tensor& pos, const RopeArgs& args) {
  TT_OP_TRACE("RopeNeox");
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
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] rope kernel pos0=%d t=%lld hq=%lld cos_first=%f\n",
                 (int)(pos.dtype == DType::kI32 ? pos.Ptr<int32_t>()[0]
                                                : static_cast<int32_t>(pos.Ptr<int64_t>()[0])),
                 (long long)t, (long long)hq,
                 cos_t.empty() ? -1.0f : cos_t.front());
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
  TT_OP_TRACE("QkvSplit");
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
// ---- ITEM 5 (PA): persistent page_table + cur_pos device tensors -------------
namespace {
struct PaMetaEntry {
  ttnn::Tensor page_table;  // int32 [B, max_blocks] device
  ttnn::Tensor cur_pos;     // int32 [B] device
  std::vector<int32_t> pt_host;
  std::vector<int32_t> cp_host;
  bool allocated = false;  // ttnn::Tensor::is_allocated() crashes on default-constructed tensors in this build
};
std::mutex& PaMetaMutex() { static std::mutex m; return m; }
std::map<std::pair<int64_t, int64_t>, PaMetaEntry>& PaMetaCache() {
  static std::map<std::pair<int64_t, int64_t>, PaMetaEntry> c;
  return c;
}
}  // namespace

// ---- R2: persistent cur_pos advanced on-device via plus_one ----------------
// The PaMeta cur_pos tensor (read by sdpa_decode) and the RAC update_idxs
// tensor (read by paged_update_cache) both hold `seq_lens - 1` for decode.
// R2 aliases them: WarmDecodePos seeds the single persistent cur_pos tensor;
// CaptureDecodePosAdvance does plus_one on it inside the trace; WarmRacIdx's
// update_idxs copy_to_device is skipped (it reuses this tensor).
namespace {
struct DecodePosEntry {
  ttnn::Tensor cur_pos;  // int32 [num_reqs] device — advanced in-trace
  bool allocated = false;
};
std::mutex& DecodePosMutex() { static std::mutex m; return m; }
std::map<int64_t, DecodePosEntry>& DecodePosCache() {
  static std::map<int64_t, DecodePosEntry> c;
  return c;
}
}  // namespace

// ---- ITEM 5 (RAC): persistent update-idx / page-table device tensors -------
// Refreshed by WarmRacIdx (driver Refresh slot, outside capture) so the
// captured paged_update_cache replays against stable addresses. Keyed by the
// slot-mapping HOST buffer (the decode-graph slot's persistent buffer), so a
// different graph size gets its own entries.
namespace {
struct RacIdxEntry {
  ttnn::Tensor update_idxs;  // int32 [C] device (persistent, content refreshed)
  ttnn::Tensor page_table;   // int32 [C,pt_width] device (persistent, content refreshed)
  std::vector<int32_t> pt_host;   // last page-table content copied to device
  int64_t pt_width = 0;           // columns of the allocated page_table
  // Retired page-table tensors from width growth, kept ALIVE deliberately:
  // a freed device buffer can hand its address to a new allocation while a
  // (doomed, never-replayed-again) trace still records it. Bounded by the
  // number of block boundaries crossed (~context/block_size).
  std::vector<ttnn::Tensor> retired_pts;
  // Persistent height-sharded RAC input: logical [1,1,nkv,d], padded
  // [1,1,nkv_pad,d] (shard [nkv_pad,d] on one core). The in-region RAC
  // ttnn::copy's the rope output into it; paged_update_cache reads only the
  // first nkv rows (num_heads loop bound), so the padded tail rows are never
  // read and may hold garbage — no zeros tail, no concat, no allocation.
  ttnn::Tensor sharded_in;   // K input (height-sharded)
  ttnn::Tensor sharded_in_v;  // V input (separate — K and V must NOT share the same buffer)
  uint32_t nkv = 0;
  uint32_t d = 0;
  bool allocated = false;        // ttnn::Tensor::is_allocated() crashes on default-constructed tensors in this build
  bool sharded_in_is_alloc = false;
};
std::mutex& RacIdxMutex() { static std::mutex m; return m; }
// Keyed by (num_slots, block_size) shape — idx tensors depend on slot values + block_size.
std::map<std::pair<int64_t, int64_t>, RacIdxEntry>& RacIdxCache() {
  static std::map<std::pair<int64_t, int64_t>, RacIdxEntry> c;
  return c;
}
}  // namespace

// Warm hook: stage persistent idx tensors for THIS slot mapping. Host reads
// here are legal (called outside capture). Idempotent per content change.

// Host-free decode RAC: device shadows in, paged_update_cache out. Returns
// false (host path) unless every precondition holds.
bool TryReshapeAndCacheDeviceDecode(const Tensor& k, const Tensor& v,
                                    Tensor& k_cache, Tensor& v_cache,
                                    const Tensor& slot_mapping) {
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] TryRACDevice called capturing=%d\n",
                 (int)tt_capture_active());
  const int64_t T = k.shape[0];
  const int64_t nkv = k.shape[1];
  const int64_t d = k.shape[2];
  const int64_t bs = k_cache.shape[1];
  const int64_t num_slots = slot_mapping.shape[0];
  if (T < 1 || num_slots < 1) return false;
  if ((d % 32u) != 0u || (bs % 32u) != 0u) return false;
  if (num_slots > 1) return false;  // decode T=1 only for now

  // k/v must carry CURRENT device shadows ([T*nkv, d] TILE bf16 from rope).
  std::optional<ttnn::Tensor> k_dev, v_dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* sk = FindSlot(k.data);
    BufferSlot* sv = FindSlot(v.data);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] RAC kv shadow k_ptr=%p sk=%p dc=%d val=%d | v_ptr=%p sv=%p dc=%d val=%d\n",
                   k.data, (void*)sk, sk?sk->device_current:0, sk?(int)sk->device.has_value():0,
                   v.data, (void*)sv, sv?sv->device_current:0, sv?(int)sv->device.has_value():0);
    if (sk == nullptr || !sk->device_current || !sk->device.has_value()) return false;
    if (sv == nullptr || !sv->device_current || !sv->device.has_value()) return false;
    k_dev = sk->device;
    v_dev = sv->device;
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && !tt_capture_active()) {
      auto dump_src = [](const char* tag, const ttnn::Tensor& t) {
        const auto ls = t.logical_shape();
        const auto ps = t.padded_shape();
        std::fprintf(stderr,
                     "[TT-TRACE] RAC src %s: logical=[", tag);
        for (size_t i = 0; i < ls.rank(); ++i)
          std::fprintf(stderr, "%s%u", i ? "," : "", ls[i]);
        std::fprintf(stderr, "] padded=[");
        for (size_t i = 0; i < ps.rank(); ++i)
          std::fprintf(stderr, "%s%u", i ? "," : "", ps[i]);
        std::fprintf(stderr,
                     "] dtype=%d layout=%d pages=%u strides0123=[%u,%u,%u,%u]\n",
                     (int)t.dtype(), (int)t.layout(),
                     t.buffer() ? t.buffer()->num_pages() : 0u,
                     ls.rank() > 0 ? t.strides()[0] : 0, ls.rank() > 1 ? t.strides()[1] : 0,
                     ls.rank() > 2 ? t.strides()[2] : 0, ls.rank() > 3 ? t.strides()[3] : 0);
      };
      dump_src("k", *k_dev);
      dump_src("v", *v_dev);
    }
  }

  // Paged-KV shadows must exist and cover the target block.
  const int64_t slot = slot_mapping.Ptr<int64_t>()[0];
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] RAC slot=%lld cap=%d\n",
                 (long long)slot, (int)tt_capture_active());
  if (slot < 0) return true;  // nothing to write; treat as handled
  const uint32_t block = static_cast<uint32_t>(slot / bs);
  const uint32_t offset = static_cast<uint32_t>(slot % bs);

  std::optional<ttnn::Tensor> kc_dev, vc_dev;
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadow* skc = &PagedKvShadows()[reinterpret_cast<uintptr_t>(k_cache.data)];
    PagedKvShadow* svc = &PagedKvShadows()[reinterpret_cast<uintptr_t>(v_cache.data)];
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-TRACE] RAC paged-kv shadow k=%d v=%d k_nb=%u\n",
                   skc->device.has_value(), svc->device.has_value(), skc->nb);
    if (!skc->device.has_value() || !svc->device.has_value()) return false;
    if (skc->nb <= block || skc->nkv != static_cast<uint32_t>(nkv) ||
        skc->bs != static_cast<uint32_t>(bs) || skc->d != static_cast<uint32_t>(d)) return false;
    if (svc->nb <= block || svc->nkv != static_cast<uint32_t>(nkv) ||
        svc->bs != static_cast<uint32_t>(bs) || svc->d != static_cast<uint32_t>(d)) return false;
    kc_dev = skc->device;
    vc_dev = svc->device;
  }

  // Persistent idx tensors for THIS slot-mapping buffer (warmed outside
  // capture). Both must exist; content refresh happens at warm time.
  {
    std::lock_guard<std::mutex> g(RacIdxMutex());
    const auto key = std::make_pair(num_slots, static_cast<int64_t>(bs));
    auto it = RacIdxCache().find(key);
    const int64_t slot0 = slot_mapping.Ptr<int64_t>()[0];
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] RAC idx-check slot0=%lld cap=%d key=(%lld,%lld)\n",
                   (long long)slot0, (int)tt_capture_active(),
                   (long long)num_slots, (long long)bs);
    // WarmRacIdx (driver Refresh slot) refreshes update_idxs/page_table content
    // every step via copy_to_device; here we just verify the tensors exist.
    if (it == RacIdxCache().end() || !it->second.allocated) {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent: RAC idx tensors not warmed — call WarmRacIdx "
               "outside capture (driver Refresh slot) first");
      return false;
    }
    // sharded_in must exist (WarmRacIdx needs the paged-KV shadow geometry).
    if (!it->second.sharded_in_is_alloc) {
      VT_CHECK(!tt_capture_active(),
               "tenstorrent: RAC sharded input not warmed — call WarmRacIdx "
               "outside capture after WarmPagedKvShadow");
      return false;
    }
  }

  // Eager (cold) step AND capture: the identical op sequence (see
  // build_input below). Running it eagerly first compiles the programs;
  // capture then hits the program cache and replays against the persistent
  // addresses.
  RacIdxEntry rac_entry = [&] {
    std::lock_guard<std::mutex> g(RacIdxMutex());
    return RacIdxCache().at(std::make_pair(num_slots, static_cast<int64_t>(bs)));
  }();

  // Single code path for the eager (cold) step AND capture. The cold step must
  // compile the exact programs the captured region will replay, so the op
  // sequence and every TensorSpec must be identical in both phases:
  //   1. reshape (metadata-only) to logical [1,1,nkv,d]
  //   2. scalar multiply by 1.0 — eltwise ops allocate a FRESH output with a
  //      native 4D spec. Feeding the bare 2D→4D reshape view straight into
  //      ttnn::copy only writes head0 (the view's 2D-allocated storage
  //      confuses the tilized copy program), and a host-side to_vector →
  //      from_vector round-trip is illegal during capture and hashes
  //      differently (program-cache miss → binary load during capture).
  //   3. ttnn::copy into the persistent sharded_in (preallocated output; the
  //      interleaved-TILE→height-sharded copy program uses only CBs)
  // paged_fused_update_cache (in-place, has override_runtime_arguments) then
  // ingests the sharded input against persistent addresses.
  auto build_input = [&](const ttnn::Tensor& src, ttnn::Tensor& sharded_dst,
                         const RacIdxEntry& entry) -> ttnn::Tensor {
      // Materialize a NATIVE [1,1,nkv,d] TILE tensor on device with a single
      // code path on cold and capture (host round-trips are illegal during
      // capture and compile differently-hashed programs).
      //
      // TILE readers map a logical element of a 4D [1,1,nkv,d] tensor to
      // in-page ROW h of the (d/32)-page grid — so only [nkv,d]-SHAPED
      // storage (heads at in-page rows 0..nkv-1) can be viewed; a [1,N]
      // native (data at in-page row 0 of N/32 pages) mis-maps (head0-only
      // or stale garbage — verified by per-head value dumps). Therefore:
      //   * [nkv,d] source (rope K output): explicit tile-padded 4D view —
      //     its spec is byte-identical to a native 4D's; multiply reads
      //     per-head exact (verified).
      //   * [1,N] source (QkvSplit V slice): materialize [nkv,d] storage
      //     first — per-head [1,d] slices concatenated on dim 0. Only ops
      //     already proven capture-safe in-region (slice/concat/eltwise).
      // The scalar multiply materializes a fresh native 4D allocation;
      // ttnn::copy moves it into the persistent sharded input.
      const uint32_t nkv_pad = ((entry.nkv + 31u) / 32u) * 32u;
      ttnn::Tensor laid_out = src;
      if (src.logical_shape().rank() == 2 && src.logical_shape()[0] == 1) {
        std::vector<ttnn::Tensor> heads;
        heads.reserve(entry.nkv);
        for (uint32_t h = 0; h < entry.nkv; ++h) {
          heads.push_back(ttnn::slice(
              src, ttsl::SmallVector<uint32_t>{0u, h * entry.d},
              ttsl::SmallVector<uint32_t>{1u, (h + 1u) * entry.d},
              ttsl::SmallVector<uint32_t>{1u, 1u}));
        }
        laid_out = ttnn::concat(heads, /*dim=*/0);
      }
      ttnn::Tensor native4 = ttnn::multiply(
          ttnn::experimental::view(
              laid_out, ttnn::Shape({1u, 1u, entry.nkv, entry.d}),
              ttnn::Shape({1u, 1u, nkv_pad, entry.d})),
          1.0f);
      auto head_maxima = [](const ttnn::Tensor& t, uint32_t nkv, uint32_t d) {
        auto v = t.to_vector<float>();
        std::string s;
        for (uint32_t h = 0; h < nkv; ++h) {
          float mx = 0;
          for (uint32_t e = 0; e < d; ++e) {
            const size_t i = static_cast<size_t>(h) * d + e;
            if (i < v.size()) mx = std::max(mx, std::abs(v[i]));
          }
          s += std::to_string(mx) + ",";
        }
        return s;
      };
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && !tt_capture_active()) {
        auto chk = native4.to_vector<float>();
        int nonzero = 0;
        for (uint32_t h = 0; h < entry.nkv; ++h) {
          float mx = 0;
          for (uint32_t e = 0; e < entry.d; ++e) {
            const size_t i = static_cast<size_t>(h) * entry.d + e;
            if (i < chk.size()) mx = std::max(mx, std::abs(chk[i]));
          }
          if (mx > 1e-6f) ++nonzero;
        }
        std::fprintf(stderr, "[TT-TRACE] RAC native4 nonzero_heads=%d/%u "
                     "src_headmax=[%s] laid_headmax=[%s] out_headmax=[%s]\n",
                     nonzero, entry.nkv,
                     head_maxima(src, entry.nkv, entry.d).c_str(),
                     head_maxima(laid_out, entry.nkv, entry.d).c_str(),
                     head_maxima(native4, entry.nkv, entry.d).c_str());
      }
    ttnn::copy(native4, sharded_dst);
    return sharded_dst;
  };
  // V first, then K
  // Debug: dump v_dev properties before sharding
  ttnn::Tensor v_in = build_input(*v_dev, rac_entry.sharded_in_v, rac_entry);
  ttnn::Tensor k_in = build_input(*k_dev, rac_entry.sharded_in, rac_entry);
  // Debug: check v_in for all heads
  // num_kv_heads_override pins the kernel's head loop to nkv rows: the input
  // shard is tile-padded (nkv_pad rows) but only the first nkv rows hold data
  // (upstream decode pattern, test_paged_cache_flexible_geometry.py).
  // Use paged_fused_update_cache (single call for K+V) instead of two separate
  // paged_update_cache calls. The fused op has override_runtime_arguments
  // (the non-fused doesn't), so it works correctly with program cache enabled.
  // The second separate call would reuse the first's cached program with the
  // first's buffer addresses (program cache collision).
  auto [new_kc, new_vc] = ttnn::experimental::paged_fused_update_cache(
      *kc_dev, k_in, *vc_dev, v_in,
      /*update_idxs=*/{}, rac_entry.update_idxs,
      /*share_cache=*/false, rac_entry.page_table,
      /*batch_offset=*/0, /*compute_kernel_config=*/std::nullopt,
      /*mesh_coords=*/std::nullopt);
  {
    std::lock_guard<std::mutex> g(PagedKvMutex());
    PagedKvShadows()[reinterpret_cast<uintptr_t>(k_cache.data)].device = std::move(new_kc);
    PagedKvShadows()[reinterpret_cast<uintptr_t>(k_cache.data)].device_current = true;
    PagedKvShadows()[reinterpret_cast<uintptr_t>(v_cache.data)].device = std::move(new_vc);
    PagedKvShadows()[reinterpret_cast<uintptr_t>(v_cache.data)].device_current = true;
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] RAC device update (copy+paged_update_cache)\n");
  (void)offset; (void)block;
  return true;
}

void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  TT_OP_TRACE("ReshapeAndCache");
  VT_CHECK(k.rank == 3 && v.rank == 3 && k_cache.rank == 4 && v_cache.rank == 4,
           "tenstorrent kReshapeAndCache: k/v rank-3, caches rank-4");
  VT_CHECK(IsFloatDType(k.dtype) && k.dtype == v.dtype && k_cache.dtype == k.dtype &&
               v_cache.dtype == k.dtype,
           "tenstorrent kReshapeAndCache: k/v/caches must share one float dtype");
  VT_CHECK(slot_mapping.rank == 1 && slot_mapping.dtype == DType::kI64,
           "tenstorrent kReshapeAndCache: slot_mapping rank-1 i64");

  // ITEM 5 (RAC): host-free decode branch. The host path below downloads k/v
  // (rope output shadows) and re-uploads via from_vector in the device push —
  // both fatal during capture. This branch instead feeds the DEVICE shadows
  // straight into paged_update_cache with persistent idx/page-table tensors.
  // Conditions: capturing (or host-free flag), all inputs device-shadowed,
  // TILE-legal dims, and the warm hook already staged the idx tensors.
  // Live read, NOT a function-local static: a latch here would cache the
  // now-default-ON value and silently strip VT_TT_HOST_FREE_DECODE=0 of its
  // effect on this path for the rest of the process (#1688).
  const bool host_free_rac = HostFreeDecodeEnabled();
  if (host_free_rac || tt_capture_active()) {
    if (TryReshapeAndCacheDeviceDecode(k, v, k_cache, v_cache, slot_mapping)) {
      return;
    }
  }

  EnsureHost(k);
  EnsureHost(v);
  EnsureHost(k_cache);
  EnsureHost(v_cache);
  EnsureHost(slot_mapping);
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
  TT_OP_TRACE("TryPagedAttentionDeviceDecode");
  if (!args.causal || args.logits_soft_cap > 0.0f) return false;
  if (args.window_size.has_value()) return false;
  if (args.kv_cache_dtype != Fp8KVCacheDataType::kAuto) return false;
  if (query.rank != 3 || out.rank != 3 || k_cache.rank != 4 || v_cache.rank != 4) return false;
  if (!query.IsContiguous() || !out.IsContiguous()) return false;
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] TryPADecode entered cap=%d\n", (int)tt_capture_active());

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
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA reached EnsurePagedKvTtnn cap=%d used_nb=%u\n", (int)tt_capture_active(), used_nb);

    MeshDevice& device = SharedMeshDevice();
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-TRACE] PA EnsurePagedKvTtnn k used_nb=%u\n", used_nb);
    // Use the cached shadow when it exists (primed by WarmPagedKvShadow).
    // This skips EnsurePagedKvTtnn's from_vector upload AND its contiguous
    // check (KvSlice returns a non-contiguous strided view that the VT_CHECK
    // rejects). Needed on BOTH cold and capture steps so sdpa_decode compiles.
    ttnn::Tensor dev_k, dev_v;
    {
  std::lock_guard<std::mutex> g(PagedKvMutex());
      auto& sk = PagedKvShadows()[reinterpret_cast<uintptr_t>(k_cache.data)];
      auto& sv = PagedKvShadows()[reinterpret_cast<uintptr_t>(v_cache.data)];
      if (sk.device_current && sk.device.has_value() && sk.nb >= used_nb &&
          sv.device_current && sv.device.has_value() && sv.nb >= used_nb) {
        dev_k = *sk.device;
        dev_v = *sv.device;
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA using cached KV shadows (k_nb=%u v_nb=%u) cap=%d\n",
                       sk.nb, sv.nb, (int)tt_capture_active());
      } else if (tt_capture_active()) {
        throw std::runtime_error("PA: no KV shadow during capture");
      } else {
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA shadow miss: k_dc=%d k_dev=%d k_nb=%u/%u v_dc=%d v_dev=%d v_nb=%u/%u\n",
                       (int)sk.device_current, (int)sk.device.has_value(), sk.nb, used_nb,
                       (int)sv.device_current, (int)sv.device.has_value(), sv.nb, used_nb);
        // Cold step without shadow: fall through to EnsurePagedKvTtnn
        // (may fail on non-contiguous KvSlice; that's OK — the host path runs).
        g.~lock_guard();  // release before EnsurePagedKvTtnn
        dev_k = EnsurePagedKvTtnn(k_cache, device, used_nb);
        dev_v = EnsurePagedKvTtnn(v_cache, device, used_nb);
      }
    }
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-TRACE] PA KV shadows OK, building page_table\n");

    const uint32_t Bu = static_cast<uint32_t>(num_reqs);
    const uint32_t hu = static_cast<uint32_t>(hq);
    const uint32_t du = static_cast<uint32_t>(d);

    // Q: [1, B, H, D]. Prefer reshape of a resident [B*H, D] / [B, H*D] shadow
    // (post device rope) so we never download then re-upload.
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA before identity_q cap=%d total_q=%lld num_reqs=%lld qsl0=%d qsl1=%d\n",
                   (int)tt_capture_active(), (long long)total_q, (long long)num_reqs,
                   qsl[0], num_reqs > 0 ? qsl[1] : -1);
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
        // SINGLE code path (cold compiles exactly what capture replays):
        // resident [B*H, D] rope shadow → padding-correct 4D view → scalar
        // multiply materializes a native [1, B, H, D] TILE tensor. The old
        // Tensor::reshape view carried an UNPADDED spec (padded=logical) so
        // sdpa_decode mis-mapped the storage (capture read head0-only while
        // the cold host round-trip was correct); the explicit tile-padded
        // view's spec is identical to a native 4D tensor's.
        {
          Tensor q_flat = query.View({total_q * hq, d});
          ttnn::Tensor dev_q_2d = EnsureDevice2D(q_flat, device);
          const uint32_t hu_pad = ((hu + 31u) / 32u) * 32u;
          dev_q = ttnn::multiply(
              ttnn::experimental::view(
                  dev_q_2d, ttnn::Shape({1u, Bu, hu, du}),
                  ttnn::Shape({1u, Bu, hu_pad, du})),
              1.0f);
        }
        // Shard if needed
        if (std::getenv("VT_TT_SHARD_Q") != nullptr) {
          const uint32_t padded_hq = std::max(32u, hu);
          const auto q_grid = device.compute_with_storage_grid_size();
          const tt::tt_metal::CoreRangeSet q_core_set =
              tt::tt_metal::num_cores_to_corerangeset(Bu, q_grid, true);
          tt::tt_metal::ShardSpec q_ss(q_core_set, {padded_hq, du},
                                       tt::tt_metal::ShardOrientation::ROW_MAJOR);
          tt::tt_metal::MemoryConfig q_mc(
              tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
              tt::tt_metal::BufferType::L1, q_ss);
          dev_q = ttnn::to_memory_config(dev_q, q_mc);
        }
        q_from_device = true;
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA q_from_device OK cap=%d\n", (int)tt_capture_active());
      } catch (const std::exception& e) {
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA q_from_device FAILED: %s\n", e.what());
        q_from_device = false;
      }
    }
    if (!q_from_device) {
      if (tt_capture_active()) {
        VT_CHECK(false, "tenstorrent: PA Q host path is not capture-safe "
                        "(from_vector readback); the resident rope shadow "
                        "must be used during capture");
      }
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
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
        std::fprintf(stderr, "[TT-UP] TryPagedAttentionDeviceDecode from_vector WRITE during capture\n");
      dev_q = ttnn::Tensor::from_vector<float>(
          q_host, SpecOf(tt::tt_metal::Shape({1u, Bu, hu, du}), ttnn::DataType::BFLOAT16,
                         ttnn::Layout::TILE),
          &device);
    }
    ttnn::Tensor dev_pt, dev_pos;
    bool use_warm_meta = false;
    {
      std::lock_guard<std::mutex> g(PaMetaMutex());
      const auto pkey = std::make_pair(static_cast<int64_t>(num_reqs),
                                       static_cast<int64_t>(max_blocks));
      auto it = PaMetaCache().find(pkey);
      if (it != PaMetaCache().end() && it->second.allocated) {
        dev_pt = it->second.page_table;
        dev_pos = it->second.cur_pos;
        const int32_t expect_cp = slens[0] - 1;  // WarmPaMeta stores seq_lens - 1
        VT_CHECK(it->second.cp_host.size() >= 1 && it->second.cp_host[0] == expect_cp,
                 "tenstorrent: PA meta not warmed for this step");
        use_warm_meta = true;
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA using cached meta (pt+cp) cap=%d\n",
                       (int)tt_capture_active());
      }
    }
    if (!use_warm_meta) {
      // Trimmed inline page_table ([B, max_blocks_per_seq] like the
      // upstream test): sdpa_decode mis-executes with a wide [B, 256]
      // table when only a few blocks are allocated.
      VT_CHECK(!tt_capture_active(),
               "tenstorrent: PA meta not warmed for this step");
      const int64_t max_vblk = (num_reqs > 0 && slens[0] > 0)
          ? (slens[0] - 1) / block_size + 1 : 1;
      const int64_t pt_cols = std::min(max_blocks, std::max<int64_t>(2, max_vblk));
      std::vector<int32_t> pt_trim(static_cast<size_t>(Bu * pt_cols));
      for (int64_t r = 0; r < num_reqs; ++r) {
        for (int64_t c = 0; c < pt_cols; ++c) {
          pt_trim[static_cast<size_t>(r * pt_cols + c)] =
              pt[static_cast<size_t>(r * max_blocks + c)];
        }
      }
      dev_pt = ttnn::Tensor::from_vector<int32_t>(
          pt_trim, SpecOf(tt::tt_metal::Shape({Bu, static_cast<uint32_t>(pt_cols)}),
                      ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);
      std::vector<int32_t> cpos(static_cast<size_t>(num_reqs));
      for (int64_t r = 0; r < num_reqs; ++r) cpos[static_cast<size_t>(r)] = slens[r] - 1;
      dev_pos = ttnn::Tensor::from_vector<int32_t>(
          cpos, SpecOf(tt::tt_metal::Shape({Bu}), ttnn::DataType::INT32,
                       ttnn::Layout::ROW_MAJOR),
          &device);
    }

    // DON'T pass program_config — let sdpa_decode use its default.
    // Our explicit config may interact badly with the program cache when
    // called after other ops in the model forward.

    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA calling sdpa_decode cap=%d\n", (int)tt_capture_active());
    // Don't pass compute_kernel_config — let sdpa_decode use its default (HiFi2).
    // Our HiFi4 was needed for nkv>1 correctness, but the 2-head issue is separate.
     // Debug: when VT_TT_SDPA_TEST is set, create FRESH Q/KV/pt/pos from
    // scratch (random data, all heads populated) and call sdpa_decode.
    // This tests whether sdpa_decode works inside the model forward context
    // with tensors created the same way as the Python standalone test.
    if (std::getenv("VT_TT_SDPA_TEST") != nullptr && !tt_capture_active()) {
      static bool tested = false;
      if (!tested) {
        tested = true;
        fprintf(stderr, "[TT-SDPA-TEST] Running standalone sdpa_decode test inside model forward...\n");
        // Create fresh Q: [1,1,16,128] with random data in ALL heads
        std::vector<float> test_q(16 * 128);
        for (auto& v : test_q) v = static_cast<float>(rand()) / 2147483647.0f * 2.0f - 1.0f;
        ttnn::Tensor test_q_rm = ttnn::Tensor::from_vector<float>(test_q,
            SpecOf(tt::tt_metal::Shape({1u, 1u, 16u, 128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR),
            nullptr);
        ttnn::Tensor test_q_dev = test_q_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
        ttnn::Tensor test_q_tile = ttnn::to_layout(test_q_dev, ttnn::Layout::TILE);
        // Create fresh KV: [2,8,32,128] with data in block 1, offset 0
        std::vector<float> test_k(2*8*32*128, 0.0f), test_v(2*8*32*128, 0.0f);
        for (uint32_t h = 0; h < 8; h++)
          for (uint32_t e = 0; e < 128; e++) {
            size_t off = (1*8*32 + h*32 + 0) * 128 + e;
            test_k[off] = static_cast<float>(rand()) / 2147483647.0f * 2.0f - 1.0f;
            test_v[off] = static_cast<float>(rand()) / 2147483647.0f * 2.0f - 1.0f;
          }
        ttnn::Tensor test_k_rm = ttnn::Tensor::from_vector<float>(test_k,
            SpecOf(tt::tt_metal::Shape({2u,8u,32u,128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR), nullptr);
        ttnn::Tensor test_k_dev = test_k_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
        ttnn::Tensor test_k_tile = ttnn::to_layout(test_k_dev, ttnn::Layout::TILE);
        ttnn::Tensor test_v_rm = ttnn::Tensor::from_vector<float>(test_v,
            SpecOf(tt::tt_metal::Shape({2u,8u,32u,128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR), nullptr);
        ttnn::Tensor test_v_dev = test_v_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
        ttnn::Tensor test_v_tile = ttnn::to_layout(test_v_dev, ttnn::Layout::TILE);
        // page_table: [1,2] = [1,0]
        std::vector<int32_t> test_pt = {1, 0};
        ttnn::Tensor test_pt_dev = ttnn::Tensor::from_vector<int32_t>(test_pt,
            SpecOf(tt::tt_metal::Shape({1u,2u}), ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
        // cur_pos: [1] = [0]
        std::vector<int32_t> test_pos = {0};
        ttnn::Tensor test_pos_dev = ttnn::Tensor::from_vector<int32_t>(test_pos,
            SpecOf(tt::tt_metal::Shape({1u}), ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
        // Call sdpa_decode with FRESH Q + model KV/pt/pos
        // But first check if dev_k has data at block 1
        {
          auto k_check = dev_k.to_vector<float>();
          const uint32_t k_nkv = dev_k.logical_shape()[1];
          const uint32_t k_bs = dev_k.logical_shape()[2];
          const uint32_t k_d = dev_k.logical_shape()[3];
          size_t b1_off = (1 * k_nkv * k_bs + 0 * k_bs + 0) * k_d;
          fprintf(stderr, "[TT-SDPA-TEST] dev_k block1: [%f,%f,%f,%f] (off=%zu/%zu)\n",
                  k_check.size()>b1_off?k_check[b1_off]:0,
                  k_check.size()>b1_off+1?k_check[b1_off+1]:0,
                  k_check.size()>b1_off+2?k_check[b1_off+2]:0,
                  k_check.size()>b1_off+3?k_check[b1_off+3]:0,
                  b1_off, k_check.size());
          // Check V at block 1 for ALL 8 KV heads
          auto v_check = dev_v.to_vector<float>();
          const uint32_t v_nkv = dev_v.logical_shape()[1];
          const uint32_t v_bs = dev_v.logical_shape()[2];
          const uint32_t v_d = dev_v.logical_shape()[3];
          for (uint32_t h = 0; h < v_nkv; h++) {
            size_t voff = (1 * v_nkv * v_bs + h * v_bs + 0) * v_d;
            fprintf(stderr, "[TT-SDPA-TEST] dev_v block1 head%u: [%f,%f,%f,%f]\n", h,
                    v_check.size()>voff?v_check[voff]:0,
                    v_check.size()>voff+1?v_check[voff+1]:0,
                    v_check.size()>voff+2?v_check[voff+2]:0,
                    v_check.size()>voff+3?v_check[voff+3]:0);
          }
          // Also check if model KV WITHOUT RAC works: create fresh KV via
          // to_layout(TILE) with the SAME data as dev_k
          auto k_vec = dev_k.to_vector<float>();
          auto v_vec = dev_v.to_vector<float>();
          ttnn::Tensor fresh_k_rm = ttnn::Tensor::from_vector<float>(k_vec,
              SpecOf(tt::tt_metal::Shape({2u,8u,32u,128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR), nullptr);
          ttnn::Tensor fresh_k_dev = fresh_k_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
          ttnn::Tensor fresh_k_tile = ttnn::to_layout(fresh_k_dev, ttnn::Layout::TILE);
          ttnn::Tensor fresh_v_rm = ttnn::Tensor::from_vector<float>(v_vec,
              SpecOf(tt::tt_metal::Shape({2u,8u,32u,128u}), ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR), nullptr);
          ttnn::Tensor fresh_v_dev = fresh_v_rm.to_device(&device, tt::tt_metal::MemoryConfig{});
          ttnn::Tensor fresh_v_tile = ttnn::to_layout(fresh_v_dev, ttnn::Layout::TILE);
          // Call sdpa_decode with fresh Q + fresh-KV-from-model-data
          ttnn::Tensor fresh_out = ttnn::transformer::paged_scaled_dot_product_attention_decode(
              test_q_tile, fresh_k_tile, fresh_v_tile, dev_pt,
              true, std::nullopt, dev_pos, std::nullopt,
              1.0f/std::sqrt(128.0f), std::nullopt, std::nullopt,
              std::nullopt, std::nullopt, std::nullopt, std::nullopt);
          auto fresh_out_vec = fresh_out.to_vector<float>();
          int fresh_nonzero = 0;
          for (uint32_t h = 0; h < 16; h++) {
            size_t off = static_cast<size_t>(h) * 128;
            float maxval = 0;
            for (size_t i = off; i < off + 128 && i < fresh_out_vec.size(); i++)
              maxval = std::max(maxval, std::abs(fresh_out_vec[i]));
            if (maxval > 0.001f) fresh_nonzero++;
          }
          fprintf(stderr, "[TT-SDPA-TEST] fresh-KV-from-model-data: %d/16 heads\n", fresh_nonzero);
        }
        ttnn::Tensor test_out = ttnn::transformer::paged_scaled_dot_product_attention_decode(
            test_q_tile, dev_k, dev_v, dev_pt,
            true, std::nullopt, test_pos_dev, std::nullopt,
            1.0f/std::sqrt(128.0f), std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt);
        auto test_out_vec = test_out.to_vector<float>();
        int non_zero = 0;
        for (uint32_t h = 0; h < 16; h++) {
          size_t off = static_cast<size_t>(h) * 128;
          float maxval = 0;
          for (size_t i = off; i < off + 128 && i < test_out_vec.size(); i++)
            maxval = std::max(maxval, std::abs(test_out_vec[i]));
          if (maxval > 0.001f) non_zero++;
          if (h % 2 == 0)
            fprintf(stderr, "[TT-SDPA-TEST] head%u: max=%.4f %s\n", h, maxval, maxval > 0.001f ? "OK" : "ZERO");
        }
        fprintf(stderr, "[TT-SDPA-TEST] Non-zero heads: %d/16\n", non_zero);
      }
    }
    // Dump all shapes right before sdpa_decode
    ttnn::Tensor dev_out = ttnn::transformer::paged_scaled_dot_product_attention_decode(
        dev_q, dev_k, dev_v, dev_pt,
        /*is_causal=*/true,
        /*attn_mask=*/std::nullopt,
        /*cur_pos_tensor=*/dev_pos,
        /*attention_sink=*/std::nullopt,
        /*scale=*/args.scale,
        /*sliding_window_size=*/std::nullopt,
        /*memory_config=*/tt::tt_metal::MemoryConfig{},
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt,
        /*paged_cache_geometry=*/std::nullopt,
        /*cache_position_modulo=*/std::nullopt);

    // Dump PA output for comparison (first layer, cold step)
    {
    }

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
        const uint32_t flat_cols = static_cast<uint32_t>(hq * d);
        ttnn::Tensor flat = ttnn::reshape(dev_out,
            ttnn::Shape({Bu, flat_cols}));
        CommitDeviceLogical2D(out, std::move(flat), Bu, flat_cols);
        // Verify the committed output matches the PA output
        {
        }
        return true;
      } catch (const std::exception&) {
        // Fall through to host materialization.
      }
    }

    // Output ~ [1, B, H, D] → host [B, H, D] in request order, then scatter to
    // global query token indices.
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA decode to_vector\n");
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
  } catch (const std::exception& e) {
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] PA device decode FAILED: %s\n", e.what());
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

      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
        std::fprintf(stderr, "[TT-UP] TryPagedAttentionDevicePrefill from_vector WRITE during capture\n");
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
        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
          std::fprintf(stderr, "[TT-UP] TryPagedAttentionDevicePrefill from_vector WRITE during capture\n");
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

        if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
          std::fprintf(stderr, "[TT-TRACE] PA prefill to_vector\n");
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

// ---- GDN prefill op set (BACKEND-TENSTORRENT-GDN W1) -------------------------
// kL2Norm / kRmsNormGated / kCausalConv1dFwd / kGdnPrefill: the op chain the
// Qwen3.5-family GDN layer issues in prefill. The CPU f32 arm (cpu_ops.cpp
// GdnPrefillKernel / CausalConv1dFwdKernel / L2NormKernel / RmsNormGatedKernel)
// is the correctness oracle; the tt-metal substrate is the implementation.
// Device-composed: kL2Norm (square → row-sum → rsqrt → scale, bf16 tiles),
// kRmsNormGated (ttnn::rms_norm + silu/sigmoid gate eltwise), kGdnPrefill
// (ttnn::transformer::chunk_gated_delta_rule behind a varlen→dense adapter).
// Host-staged in W1: kCausalConv1dFwd (the varlen window build + rolling
// conv_state writeback is pure data movement at these shapes; the conv-state
// device shadow that would make a composed path win is W2's decode work).

// Rank-flexible rows view of a contiguous float tensor as [rows, cols] TILE
// bf16 on device: reuses a resident same-numel shadow without a host
// round-trip (the EnsureDevice2D residency win, reached from rank-3 GDN
// shapes [T,H,D]).
ttnn::Tensor DeviceRows(const Tensor& t, uint32_t rows, uint32_t cols,
                        MeshDevice& device) {
  if (t.rank == 2 && t.IsContiguous()) return EnsureDevice2D(t, device);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && s->device_current && s->device.has_value() &&
        static_cast<uint64_t>(s->dev_rows) * s->dev_cols ==
            static_cast<uint64_t>(rows) * cols) {
      ttnn::Tensor reshaped =
          ttnn::reshape(*s->device, ttnn::Shape({rows, cols}));
      s->device = reshaped;
      s->dev_rows = rows;
      s->dev_cols = cols;
      return reshaped;
    }
  }
  EnsureHost(t);
  const auto host = ToHostF32(t);
  return UploadRows(host.data(), rows, cols, device);
}

// Upload an arbitrary-rank host f32 buffer as a device tensor of `dtype` /
// `layout` (from_vector handles tile padding of logical dims).
ttnn::Tensor UploadTensor(std::vector<float> host, const ttnn::Shape& shape,
                          ttnn::DataType dtype, ttnn::Layout layout,
                          MeshDevice& device) {
  return ttnn::Tensor::from_vector<float>(
      std::move(host),
      tt::tt_metal::TensorSpec(tt::tt_metal::Shape(shape),
                               tt::tt_metal::TensorLayout(
                                   dtype, tt::tt_metal::PageConfig(layout),
                                   tt::tt_metal::MemoryConfig{})),
      &device);
}

// kL2Norm: y = x * rsqrt(sum(x^2) + eps) over the last dim (cpu_ops.cpp
// L2NormKernel; gdn-semantics.md §4). GDN callers run it on q/k [T,H,D] rows.
// Device path: square → row sum → +eps → rsqrt → broadcast multiply, bf16
// tiles (same envelope as kRmsNorm).
void L2NormKernel(Queue&, Tensor& out, const Tensor& x, const L2NormArgs& args) {
  TT_OP_TRACE("L2Norm");
  VT_CHECK(x.rank == 2 || x.rank == 3,
           "tenstorrent kL2Norm: rank 2 or 3 required");
  VT_CHECK(out.rank == x.rank, "tenstorrent kL2Norm: out rank must match x");
  VT_CHECK(IsFloatDType(x.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kL2Norm: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && out.IsContiguous(),
           "tenstorrent kL2Norm: contiguous required");
  const uint32_t d = static_cast<uint32_t>(x.shape[x.rank - 1]);
  const uint32_t rows = static_cast<uint32_t>(x.Numel() / d);

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = DeviceRows(x, rows, d, device);
  ttnn::Tensor sq = ttnn::multiply(dev_x, dev_x);
  ttnn::Tensor s = ttnn::sum(sq, ttsl::SmallVector<int>{1}, true);
  ttnn::Tensor denom = ttnn::add(s, args.eps);
  ttnn::Tensor inv = ttnn::rsqrt(denom);
  ttnn::Tensor dev_y = ttnn::multiply(dev_x, inv);
  CommitDeviceLogical2D(out, std::move(dev_y), rows, d);
}

// kRmsNormGated: out = x * rsqrt(mean(x^2)+eps) * w * act(gate) with
// norm_before_gate=True semantics baked in (cpu_ops.cpp RmsNormGatedKernel;
// gdn-semantics.md §5). Device path reuses the kRmsNorm machinery
// (ttnn::rms_norm with the [1,D] affine upload) + a silu/sigmoid gate pass.
// The gate may be a padded-row rank-3 view of the merged qkvz z-slice: its
// rows are gathered honoring the token stride and uploaded compactly.
void RmsNormGatedKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& gate,
                        const Tensor& weight, const RmsNormGatedArgs& args) {
  TT_OP_TRACE("RmsNormGated");
  VT_CHECK((x.rank == 2 || x.rank == 3) && gate.rank == x.rank &&
               out.rank == x.rank && weight.rank == 1,
           "tenstorrent kRmsNormGated: x/gate/out rank-2 or rank-3, weight rank-1");
  VT_CHECK(IsFloatDType(x.dtype) && IsFloatDType(gate.dtype) &&
               IsFloatDType(weight.dtype) &&
               (out.dtype == DType::kF32 || out.dtype == DType::kBF16),
           "tenstorrent kRmsNormGated: float in, f32/bf16 out");
  VT_CHECK(x.IsContiguous() && weight.IsContiguous() && out.IsContiguous(),
           "tenstorrent kRmsNormGated: x/out/weight contiguous required");
  const uint32_t d = static_cast<uint32_t>(x.shape[x.rank - 1]);
  const uint32_t rows = static_cast<uint32_t>(x.Numel() / d);
  VT_CHECK(weight.shape[0] == d,
           "tenstorrent kRmsNormGated: weight size mismatch");

  MeshDevice& device = SharedMeshDevice();
  ttnn::Tensor dev_x = DeviceRows(x, rows, d, device);
  ttnn::Tensor dev_w = EnsureAffine1D(weight, d, device);
  EnsureHost(gate);
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  std::vector<float> gh(static_cast<size_t>(rows) * d);
  for (uint32_t i = 0; i < rows; ++i) {
    const int64_t gbase =
        (i / gate_group) * gate_outer + (i % gate_group) * d;
    for (uint32_t j = 0; j < d; ++j)
      gh[static_cast<size_t>(i) * d + j] = LoadElemF32(gate, gbase + j);
  }
  ttnn::Tensor dev_g = UploadRows(gh.data(), rows, d, device);
  ttnn::Tensor act =
      args.sigmoid_gate ? ttnn::sigmoid(dev_g) : ttnn::silu(dev_g);
  ttnn::Tensor dev_y = ttnn::multiply(ttnn::rms_norm(dev_x, args.eps, dev_w), act);
  CommitDeviceLogical2D(out, std::move(dev_y), rows, d);
}

// kCausalConv1dFwd: depthwise causal conv over time with the rolling
// conv_state writeback (cpu_ops.cpp CausalConv1dFwdKernel; gdn-semantics.md
// §2). W1 HOST-STAGED: this backend's Alloc is host memory
// (tenstorrent_backend.cpp), so the scalar port below runs the exact oracle
// instruction order on the host bytes — outputs read the OLD state row
// (buffered), the new row carries the last K-1 RAW x tokens (left-shifted
// from the old state when T < K-1). A composed slice/concat+MAC path pays a
// full [T*K,C] window materialization to build what this loop streams; the
// conv-state device shadow that would flip that trade is W2's decode work.
void CausalConv1dFwdKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, Tensor& conv_state,
                           const Tensor& qsl, const Tensor& his,
                           const CausalConv1dArgs& args) {
  TT_OP_TRACE("CausalConv1dFwd");
  const int64_t total = x.shape[0], c_dim = x.shape[1], k = w.shape[1];
  const int64_t width = k - 1;
  const int64_t n = conv_state.shape[0];
  const int64_t x_rs = x.stride[0];
  EnsureHost(x);
  EnsureHost(w);
  if (bias != nullptr) EnsureHost(*bias);
  EnsureHost(conv_state);
  EnsureHost(qsl);
  EnsureHost(his);
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == total,
           "tenstorrent causal_conv1d_fwd: bad query_start_loc bounds");
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s] && qslp[s] >= 0,
             "tenstorrent causal_conv1d_fwd: query_start_loc not monotonic");
  }
  // Per (sequence, channel): mirrors the oracle's row-chunked decomposition —
  // disjoint out columns / conv_state rows, so the loop order is free.
  std::vector<float> old_row(static_cast<size_t>(width));
  for (int64_t s = 0; s < n; ++s) {
    const bool init = his.dtype == DType::kI8 ? his.Ptr<int8_t>()[s] != 0
                                              : his.Ptr<int32_t>()[s] != 0;
    const int64_t begin = qslp[s], t_len = qslp[s + 1] - begin;
    for (int64_t c = 0; c < c_dim; ++c) {
      float* srow = conv_state.Ptr<float>() + (s * c_dim + c) * width;
      for (int64_t j = 0; j < width; ++j)
        old_row[static_cast<size_t>(j)] = srow[j];
      const float b = bias != nullptr ? LoadElemF32(*bias, c) : 0.0f;
      for (int64_t t = 0; t < t_len; ++t) {
        float acc = b;
        for (int64_t j = 0; j < k; ++j) {
          const int64_t ti = t - (k - 1 - j);  // token index of window[j]
          float v = 0.0f;
          if (ti >= 0) {
            v = LoadElemF32(x, (begin + ti) * x_rs + c);
          } else if (init) {
            v = old_row[static_cast<size_t>(width + ti)];  // state col (K-1)+(t-i)
          }
          acc += LoadElemF32(w, c * k + j) * v;
        }
        const float y = args.silu_activation
                            ? acc / (1.0f + std::exp(-acc))
                            : acc;
        StoreElemF32(out, (begin + t) * c_dim + c, y);
      }
      for (int64_t j = 0; j < width; ++j) {
        const int64_t tj = t_len - width + j;  // new state col j holds token tj
        float v = 0.0f;
        if (tj >= 0) {
          v = LoadElemF32(x, (begin + tj) * x_rs + c);
        } else if (init) {
          v = old_row[static_cast<size_t>(width + tj)];  // shifted old state
        }
        srow[j] = v;
      }
    }
  }
  CommitHost(out);
  CommitHost(conv_state);
}

// kGdnPrefill: the chunked gated-delta-rule scan over a varlen batch, behind
// the tt-metal fused kernel (ttnn::transformer::chunk_gated_delta_rule — one
// Tensix core per (B·HV) head, recurrent state on-core, fp32 state / HiFi4).
// Adapter duties (spec "Design"):
//   - varlen [T,...] + query_start_loc -> ONE dense padded [N,L,...] batch:
//     per-sequence initial states ride the batch's [B,HV,K,V] initial_state,
//     so N sequences cost one op call, not N. L is padded to the 64-token
//     chunk multiple so the op's own time-pad path stays idle; padded rows
//     carry zero q/k/v/g/beta, and g=0/beta=0 is an IDENTITY state update
//     (exp(0)=1, v'=0), so empty sequences and short tails leave the state
//     exactly where the oracle leaves it.
//   - q/k arrive PRE-normalized and GdnArgs::scale multiplies q (the op folds
//     scale into q itself), so use_qk_l2norm stays FALSE (the op fatal-errors
//     on true; FLA scale semantics match the CPU GdnHeadTokenStep).
//   - state layout: ours [N,Hv,Dv,Dk], the op's [B,HV,K,V] — one transpose of
//     the trailing dims on upload and download, inside the adapter.
// Host-staged upload/download in W1 (the decode shadow that keeps the state
// resident is W2's); outputs are written to the host bytes and committed.
void GdnPrefillKernel(Queue&, Tensor& out, const Tensor& q_in, const Tensor& k_in,
                      const Tensor& v_in, const Tensor& g, const Tensor& beta,
                      Tensor& state, const Tensor& qsl, const GdnArgs& args) {
  TT_OP_TRACE("GdnPrefill");
  const int64_t n = state.shape[0], hv = state.shape[1], dv = state.shape[2],
                dk = state.shape[3];
  const int64_t hk = q_in.shape[1];
  const int64_t total = q_in.shape[0];
  // chunk_gated_delta_rule device constraints (validated TILE dims): refuse
  // by name rather than silently reshape into a wrong answer (spec Risk #4).
  VT_CHECK(dk % 32 == 0 && dv % 32 == 0,
           "tenstorrent gdn_prefill: tt-metal chunk_gated_delta_rule needs "
           "Dk/Dv multiples of 32 (tile), got Dk=" +
               std::to_string(dk) + " Dv=" + std::to_string(dv));
  EnsureHost(qsl);
  const int32_t* qslp = qsl.Ptr<int32_t>();
  VT_CHECK(qslp[0] == 0 && qslp[n] == total,
           "tenstorrent gdn_prefill: bad query_start_loc bounds");
  int64_t max_len = 0;
  for (int64_t s = 0; s < n; ++s) {
    VT_CHECK(qslp[s + 1] >= qslp[s],
             "tenstorrent gdn_prefill: query_start_loc not monotonic");
    max_len = std::max(max_len, static_cast<int64_t>(qslp[s + 1] - qslp[s]));
  }
  if (total == 0) return;  // all sequences empty: no out rows, state as-is

  constexpr int64_t kChunk = 64;
  const int64_t len = ((max_len + kChunk - 1) / kChunk) * kChunk;

  EnsureHost(q_in);
  EnsureHost(k_in);
  EnsureHost(v_in);
  EnsureHost(g);
  EnsureHost(beta);
  EnsureHost(state);
  EnsureHost(out);

  // Dense padded batch [N, len, ...]: zero-filled tails are identity updates.
  const size_t qk_elems = static_cast<size_t>(n) * len * hk * dk;
  const size_t v_elems = static_cast<size_t>(n) * len * hv * dv;
  std::vector<float> qp(qk_elems, 0.0f), kp(qk_elems, 0.0f), vp(v_elems, 0.0f);
  std::vector<float> gp(static_cast<size_t>(n) * len * hv, 0.0f);
  std::vector<float> bp(static_cast<size_t>(n) * len * hv, 0.0f);
  // initial state [N,Hv,Dk,Dv] — the trailing-dims transpose of ours.
  std::vector<float> s0(static_cast<size_t>(n) * hv * dk * dv, 0.0f);
  for (int64_t s = 0; s < n; ++s) {
    const int64_t t0 = qslp[s], t_len = qslp[s + 1] - t0;
    for (int64_t t = 0; t < t_len; ++t) {
      const int64_t src = t0 + t, dst = s * len + t;
      for (int64_t e = 0; e < hk * dk; ++e) {
        qp[static_cast<size_t>(dst * hk * dk + e)] =
            LoadElemF32(q_in, src * hk * dk + e);
        kp[static_cast<size_t>(dst * hk * dk + e)] =
            LoadElemF32(k_in, src * hk * dk + e);
      }
      for (int64_t e = 0; e < hv * dv; ++e)
        vp[static_cast<size_t>(dst * hv * dv + e)] =
            LoadElemF32(v_in, src * hv * dv + e);
      for (int64_t e = 0; e < hv; ++e) {
        gp[static_cast<size_t>(dst * hv + e)] = LoadElemF32(g, src * hv + e);
        bp[static_cast<size_t>(dst * hv + e)] = LoadElemF32(beta, src * hv + e);
      }
    }
    for (int64_t h = 0; h < hv; ++h) {
      const float* ours =
          state.Ptr<float>() + ((s * hv + h) * dv) * dk;  // [Dv,Dk] rows
      float* theirs = &s0[static_cast<size_t>((s * hv + h) * dk) * dv];  // [Dk,Dv]
      for (int64_t i = 0; i < dv; ++i)
        for (int64_t j = 0; j < dk; ++j)
          theirs[static_cast<size_t>(j * dv + i)] =
              ours[static_cast<size_t>(i * dk + j)];
    }
  }

  MeshDevice& device = SharedMeshDevice();
  const uint32_t un = static_cast<uint32_t>(n), ul = static_cast<uint32_t>(len),
                 uhk = static_cast<uint32_t>(hk), uhv = static_cast<uint32_t>(hv),
                 udk = static_cast<uint32_t>(dk), udv = static_cast<uint32_t>(dv);
  ttnn::Tensor dev_q =
      UploadTensor(std::move(qp), ttnn::Shape({un, ul, uhk, udk}),
                   ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_k =
      UploadTensor(std::move(kp), ttnn::Shape({un, ul, uhk, udk}),
                   ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_v =
      UploadTensor(std::move(vp), ttnn::Shape({un, ul, uhv, udv}),
                   ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_g =
      UploadTensor(std::move(gp), ttnn::Shape({un, ul, uhv}),
                   ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_b =
      UploadTensor(std::move(bp), ttnn::Shape({un, ul, uhv}),
                   ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  ttnn::Tensor dev_s0 =
      UploadTensor(std::move(s0), ttnn::Shape({un, uhv, udk, udv}),
                   ttnn::DataType::FLOAT32, ttnn::Layout::TILE, device);
  auto [o, final_state] = ttnn::transformer::chunk_gated_delta_rule(
      dev_q, dev_k, dev_v, dev_g, dev_b, args.scale, dev_s0,
      /*output_final_state=*/true, static_cast<uint32_t>(kChunk),
      /*use_qk_l2norm=*/false,  // caller pre-normalized q/k
      /*output_head_major=*/false);
  VT_CHECK(final_state.has_value(),
           "tenstorrent gdn_prefill: chunk_gated_delta_rule returned no final state");
  const std::vector<float> ov = o.to_vector<float>();           // [N,len,Hv,Dv]
  const std::vector<float> fv = final_state->to_vector<float>();  // [N,Hv,Dk,Dv]

  // Scatter token-major outputs back into the packed varlen rows.
  for (int64_t s = 0; s < n; ++s) {
    const int64_t t_len = qslp[s + 1] - qslp[s];
    for (int64_t t = 0; t < t_len; ++t)
      for (int64_t e = 0; e < hv * dv; ++e)
        StoreElemF32(out, (qslp[s] + t) * hv * dv + e,
                     ov[static_cast<size_t>(((s * len + t) * hv) * dv + e)]);
  }
  // Final state: transpose the trailing dims back into [N,Hv,Dv,Dk].
  for (int64_t s = 0; s < n; ++s)
    for (int64_t h = 0; h < hv; ++h) {
      const float* theirs =
          &fv[static_cast<size_t>((s * hv + h) * dk) * dv];  // [Dk,Dv]
      float* ours = state.Ptr<float>() + ((s * hv + h) * dv) * dk;  // [Dv,Dk]
      for (int64_t i = 0; i < dv; ++i)
        for (int64_t j = 0; j < dk; ++j)
          ours[static_cast<size_t>(i * dk + j)] =
              theirs[static_cast<size_t>(j * dv + i)];
    }
  CommitHost(out);
  CommitHost(state);
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
    RegisterOp(OpId::kFusedChain, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
    RegisterOp(OpId::kL2Norm, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<L2NormFn>(&L2NormKernel)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernel)));
    RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<CausalConv1dFwdFn>(&CausalConv1dFwdKernel)));
    RegisterOp(OpId::kGdnPrefill, DeviceType::kTENSTORRENT,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernel)));
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

// Stall-bisection helper: counts completed TraceEndCaptureGraph calls. The
// bisection skip flags (VT_TT_NO_*_WARM) must NOT fire on the capture step
// itself — the captured rope cache-HIT guard requires fresh warm content —
// so they skip only once a graph exists (steady replay regime).
std::atomic<int>& GraphCapturesCounter() {
  static std::atomic<int> n{0};
  return n;
}
int GraphCapturesDone() { return GraphCapturesCounter().load(); }
void NoteGraphCaptured() { GraphCapturesCounter().fetch_add(1); }
bool ReplayRegimeBisectSkip(const char* flag) {
  return std::getenv(flag) != nullptr && GraphCapturesDone() > 0;
}

// A replayed trace rewrote the device memory of every tensor the captured
// region produced, but the slot registry cannot know which host buffers those
// shadows belong to. Mark the host cache of EVERY device-current slot stale so
// the next host read re-downloads. Without this, DBuf::Download ->
// Backend::Copy -> EnsureHostBytes short-circuits on host_current and serves
// the bytes captured at trace time on every later replay (frozen logits).
// Replay is non-blocking, so device writes may still be in flight here; the
// invalidation only marks device memory as newer than the host copy, and the
// re-download at the next host read is the blocking sync point.
// Input shadows (weights, embeddings) are only re-read, never re-uploaded:
// replay does not modify them, and the extra download is identical bytes.
void InvalidateHostCachesAfterTrace() {
  std::lock_guard<std::mutex> g(SlotMutex());
  for (auto& [addr, slot] : Slots())
    if (slot.device_current && slot.device.has_value()) slot.host_current = false;
}
}  // namespace

void TraceBeginCapture() {
  TraceState& s = TraceSlot();
  VT_CHECK(!s.capturing, "tenstorrent: nested TraceBeginCapture");
  MeshDevice& device = SharedMeshDevice();
  s.capturing_id = ttnn::operations::trace::begin_trace_capture(&device, kTraceCq);
  s.capturing = true;
  tt_capture_active() = true;
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] BeginCapture (flag set)\n");
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
  tt_capture_active() = false;
}

void TraceReplay() {
  TraceState& s = TraceSlot();
  VT_CHECK(!s.capturing, "tenstorrent: TraceReplay during capture");
  VT_CHECK(s.has_replay, "tenstorrent: TraceReplay with no captured trace");
  MeshDevice& device = SharedMeshDevice();
  // NON-BLOCKING, matching models/common/models/executor.py's long-decode
  // pattern (execute_trace(blocking=False) + a later blocking readback):
  // repeated blocking replays hang the mesh trace completion wait after a
  // few dozen executions on this tt-metal build. The caller's post-replay
  // device readback (logits Download) provides the synchronization; queue
  // order keeps any later input refresh behind the replay.
  ttnn::operations::trace::execute_trace(&device, s.replay_id, kTraceCq, /*blocking=*/false);
  InvalidateHostCachesAfterTrace();
}

void* TraceEndCaptureGraph() {
  TraceState& s = TraceSlot();
  VT_CHECK(s.capturing, "tenstorrent: TraceEndCaptureGraph without Begin");
  MeshDevice& device = SharedMeshDevice();
  ttnn::operations::trace::end_trace_capture(&device, s.capturing_id, kTraceCq);
  NoteGraphCaptured();
  s.capturing = false;
  tt_capture_active() = false;
  // Opaque handle: heap-allocated MeshTraceId for the multi-graph API.
  return new ttnn::MeshTraceId(s.capturing_id);
}

void TraceReplayGraph(void* graph) {
  VT_CHECK(graph != nullptr, "tenstorrent: TraceReplayGraph null");
  VT_CHECK(!TraceSlot().capturing, "tenstorrent: TraceReplayGraph during capture");
  MeshDevice& device = SharedMeshDevice();
  const auto id = *static_cast<ttnn::MeshTraceId*>(graph);
  // NON-BLOCKING — see TraceReplay: the qwen3 graph driver downloads the
  // logits right after this call, and that blocking readback is the sync
  // point (the upstream traced-decode executor's pattern).
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-STEP] execute_trace begin\n");
  ttnn::operations::trace::execute_trace(&device, id, kTraceCq, /*blocking=*/false);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-STEP] execute_trace enqueued\n");
  InvalidateHostCachesAfterTrace();
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-STEP] replay step complete\n");
}

void TraceDestroyGraph(void* graph) {
  if (graph == nullptr) return;
  auto* id = static_cast<ttnn::MeshTraceId*>(graph);
  try {
    MeshDevice& device = SharedMeshDevice();
    ttnn::operations::trace::release_trace(&device, *id);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-STEP] release_trace ok\n");
  } catch (const std::exception& ex) {
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-STEP] release_trace THREW: %s\n", ex.what());
  } catch (...) {
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-STEP] release_trace THREW (unknown)\n");
  }
  delete id;
}

// ---- HOST-FREE-DECODE: persistent decode ids + capture-safe embedding -----
// The replay step must perform ZERO eager device allocations: per-step eager
// alloc/free churn (the from_vector + embedding output of the old EmbedInto
// refresh) eventually hands a live trace's fixed buffer addresses to new
// allocations — tt-metal warns allocations while a trace exists "may be
// corrupted once a trace is executed", observed as a device hang ~60 replays
// in. The embedding therefore moves INSIDE the captured region: ids are
// refreshed into one persistent device tensor (allocation-free
// copy_to_device), the captured ttnn::embedding runs over that stable
// address, and its output tensor is kept alive so the trace's write address
// is never returned to the allocator.
namespace {
struct DecodeIdsEntry {
  ttnn::Tensor ids;  // device ROW_MAJOR UINT32 [n], content refreshed in place
  ttnn::Tensor out;  // embedding output [n, hidden] TILE; held for the trace
  bool allocated = false;
};
std::map<int64_t, DecodeIdsEntry>& DecodeIdsCache() {
  static std::map<int64_t, DecodeIdsEntry> m;
  return m;
}
std::mutex& DecodeIdsMutex() {
  static std::mutex m;
  return m;
}
}  // namespace

void WarmDecodeIds(const int32_t* ids, int64_t n) {
  if (!HostFreeDecodeEnabled()) return;
  if (ids == nullptr || n < 1) return;
  MeshDevice& device = SharedMeshDevice();
  std::vector<uint32_t> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    VT_CHECK(ids[i] >= 0, "tenstorrent WarmDecodeIds: negative id");
    host[static_cast<size_t>(i)] = static_cast<uint32_t>(ids[i]);
  }
  const auto spec = SpecOf(
      tt::tt_metal::Shape({static_cast<uint32_t>(n)}),
      ttnn::DataType::UINT32, ttnn::Layout::ROW_MAJOR);
  std::lock_guard<std::mutex> g(DecodeIdsMutex());
  DecodeIdsEntry& e = DecodeIdsCache()[n];
  const bool dbg_ids = std::getenv("VT_TT_TRACE_DEBUG") != nullptr;
  if (dbg_ids) std::fprintf(stderr, "[TT-STEP] WarmDecodeIds begin n=%lld\n", (long long)n);
  // VT_TT_NO_IDS_WARM: stall bisection only — skip the per-step H2D copy
  // after the first capture (same token embedded every replay, numerically
  // wrong, mechanics test only).
  if (ReplayRegimeBisectSkip("VT_TT_NO_IDS_WARM")) {
    if (dbg_ids) std::fprintf(stderr, "[TT-STEP] WarmDecodeIds skipped\n");
    return;
  }
  if (!e.allocated) {
    e.ids = ttnn::Tensor::from_vector<uint32_t>(host, spec, &device);
    e.allocated = true;
  } else {
    // Allocation-free refresh: host staging tensor + H2D copy into the SAME
    // device buffer (the WarmRacIdx pattern).
    ttnn::Tensor h = ttnn::Tensor::from_vector<uint32_t>(host, spec, nullptr);
    ttnn::copy_to_device(h, e.ids);
  }
  if (dbg_ids) std::fprintf(stderr, "[TT-STEP] WarmDecodeIds done\n");
}

void EmbedDeviceIdsInto(void* out_host, int64_t rows, int64_t cols,
                        const void* table_host, int64_t vocab, int64_t hidden,
                        int64_t n) {
  ttnn::Tensor dev_ids;
  {
    std::lock_guard<std::mutex> g(DecodeIdsMutex());
    auto it = DecodeIdsCache().find(n);
    VT_CHECK(it != DecodeIdsCache().end() && it->second.allocated,
             "tenstorrent: EmbedDeviceIdsInto without WarmDecodeIds(n)");
    dev_ids = it->second.ids;
  }
  ttnn::Tensor dev_table;
  {
    std::lock_guard<std::mutex> g(EmbedTableMutex());
    auto it = EmbedTableShadows().find(reinterpret_cast<uintptr_t>(table_host));
    VT_CHECK(it != EmbedTableShadows().end() && it->second.device.has_value() &&
                 it->second.vocab == static_cast<uint32_t>(vocab) &&
                 it->second.h == static_cast<uint32_t>(hidden),
             "tenstorrent: EmbedDeviceIdsInto without a warmed embed table "
             "(run one eager embedding step before capture)");
    dev_table = *it->second.device;
  }
  ttnn::Tensor dev_out = ttnn::embedding(dev_ids, dev_table, /*pad_token=*/std::nullopt,
                                         /*layout=*/ttnn::Layout::TILE);
  if (dev_out.logical_shape().rank() != 2 ||
      dev_out.logical_shape()[0] != n || dev_out.logical_shape()[1] != hidden) {
    dev_out = ttnn::reshape(
        dev_out, ttnn::Shape({static_cast<uint32_t>(n),
                              static_cast<uint32_t>(hidden)}));
  }
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(out_host);
    VT_CHECK(s != nullptr && s->device_current && s->device.has_value() &&
                 s->dev_rows == static_cast<uint32_t>(rows) &&
                 s->dev_cols == static_cast<uint32_t>(cols),
             "tenstorrent: EmbedDeviceIdsInto hidden shadow not resident");
    ttnn::copy(dev_out, *s->device);
    s->host_current = false;
  }
  // Hold the embedding output for the trace's lifetime (its address is baked
  // into the captured command sequence; freeing it would return the buffer
  // to the allocator).
  {
    std::lock_guard<std::mutex> g(DecodeIdsMutex());
    DecodeIdsCache()[n].out = dev_out;
  }
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
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] EnsureHostBytes DURING CAPTURE\n");
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

// ITEM 5: persistent zero tensors, created OUTSIDE capture (ttnn::zeros
// host-fills + to_device()s = an enqueue_write, illegal during trace capture).
// EnsureDevice2D primes the cache during the eager warmup so the captured
// res.Zero finds its entry and replays a warm device->device ttnn::copy.

// HOST-FREE-FORWARD R2: device->device copy when capturing, so Backend::Copy
// does not to_vector inside the captured region. Both dst and src must carry a
// current device shadow of equal byte size; dst's shadow becomes a copy of src.
bool CopyDeviceDeviceIfCapture(void* dst, const void* src) {
  // Run the device->device copy when EITHER capturing OR in host-free-decode
  // mode (the env opt-in). The latter is essential so the EAGER warmup step
  // (which the decode-graph framework runs BEFORE capture) also exercises
  // ttnn::empty+ttnn::copy, compiling those programs into the cache so the
  // subsequent capture doesn't hit "Cannot load new binaries during trace
  // capture." Read LIVE, not cached in a static: the inertness-guard case in
  // test_tenstorrent_backend unsets the env mid-process and must observe the
  // decline, and a suite run under an ambient flag must not pin the armed
  // behavior for cases that unset it.
  const bool host_free = HostFreeDecodeEnabled();
  if (!tt_capture_active() && !host_free) return false;
  static bool once = [&] {
    // Enable program cache once on the first host-free path use — ttnn trace
    // requires every captured op to be program-cache-warm.
    MeshDevice& device = SharedMeshDevice();
    device.enable_program_cache();
    return true;
  }();
  (void)once;
  ttnn::Tensor src_dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(const_cast<void*>(src));
    BufferSlot* d = FindSlot(dst);
    if (s == nullptr || !s->device_current || !s->device.has_value()) return false;
    if (d == nullptr) return false;
    if (s->bytes != d->bytes) return false;
    src_dev = *s->device;
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] device->device copy (capture-safe)\n");
  MeshDevice& device = SharedMeshDevice();
  // Allocate a destination device tensor matching src's shape/dtype/layout,
  // then copy. No host readback.
  ttnn::Tensor cloned = ttnn::empty(src_dev.logical_shape(), src_dev.dtype(),
                                    src_dev.layout(), &device,
                                    src_dev.memory_config());
  cloned = ttnn::copy(src_dev, cloned);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* d = FindSlot(dst);
    if (d == nullptr) return false;
    d->device = std::move(cloned);
    d->device_current = true;
    d->host_current = false;
  }
  return true;
}

// HOST-FREE-FORWARD R3: on-device fill (for DBuf::Zero -> Backend::Memset)
// when host-free decode is active, so no host write happens inside capture.
// Reinterprets the buffer as a 2D [rows, cols] f32 tensor matching the
// existing device shadow's numel (zeros is the only value the forward uses).
bool MemsetDeviceIfCapture(void* p, int value) {
  // Live read for the same reason as CopyDeviceDeviceIfCapture above.
  const bool host_free = HostFreeDecodeEnabled();
  if (!tt_capture_active() && !host_free) return false;
  if (value != 0) return false;  // only zero-fill is handled on-device
  // Need an existing shadow to know shape/dtype; or allocate from the slot.
  std::optional<ttnn::Tensor> dev;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s != nullptr && s->device_current && s->device.has_value()) {
      dev = *s->device;
    }
  }
  if (!dev.has_value()) {
    // No shadow yet: DBuf::Zero on a brand-new buffer with no device tensor.
    return false;  // fall back to host memset; the buffer is host-only for now
  }
  MeshDevice& device = SharedMeshDevice();
  const ttnn::Tensor& shadow = *dev;
  // ITEM 5: ttnn::zeros/full is NOT capture-safe — full_impl host-fills and
  // to_device()s (creation.cpp:52-71), i.e. an enqueue_write that ttnn trace
  // fatals on. The plugin pattern instead: keep PERSISTENT zero tensors
  // (created outside capture, at warmup) and ttnn::copy one onto the target —
  // a device->device program that is captured/replayed like any other warm op.
  ttnn::Tensor zero_src = ZeroCacheGet(shadow, device);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] device zero-fill (capture-safe)\n");
  // Copy the persistent zero onto the shadow IN PLACE (keeps the shadow's
  // device address stable — the whole point of persistent buffers).
  ttnn::Tensor z = ttnn::copy(zero_src, shadow);
  (void)z;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(p);
    if (s == nullptr) return false;
    s->device_current = true;
    s->host_current = false;
  }
  return true;
}


// ITEM 5 (rope): driver-side warm hook. The decode-graph driver calls this
// for the step's (padded) positions BEFORE BeginCapture — the exact
// SizeSlot::Refresh slot in qwen3.cpp — so the persistent cos/sin tensors
// are populated outside capture and the captured rope cache-HITs on content.
// hq/hk select the expanded layouts to warm; base/args must match RopeNeox.
void WarmRopeCosSin(const int32_t* positions, int64_t tokens, int64_t hq,
                    int64_t hk, int64_t rot, double base) {
  if (!HostFreeDecodeEnabled()) return;
  MeshDevice& device = SharedMeshDevice();
  std::vector<float> cos_t, sin_t;
  Tensor pos = Tensor::Contiguous(const_cast<int32_t*>(positions), DType::kI32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {tokens});
  const RopeArgs no_scale{};  // plain rope only on the warm path
  BuildCosSinFromPositions(pos, tokens, rot, base, no_scale, cos_t, sin_t);
  // Byte-exact with what the captured rope reads: the per-step cos|sin CACHE
  // stores f32-built values into a BF16 tensor (RopeCosSinCacheKernel's
  // StoreElemF32 rounds), and the rope-side gather reads them back. Round the
  // warm content through the same bf16 round-trip so the content-HIT
  // comparison is exact.
  for (auto& v : cos_t) v = BF16ToF32(F32ToBF16(v));
  for (auto& v : sin_t) v = BF16ToF32(F32ToBF16(v));
  auto warm_one = [&](int64_t heads) {
    std::vector<float> ce, se;
    ExpandCosSinPerHead(cos_t.data(), sin_t.data(), tokens, heads, rot / 2, ce, se);
    const uint32_t thu = static_cast<uint32_t>(tokens * heads);
    const uint32_t halfu = static_cast<uint32_t>(rot / 2);
    std::lock_guard<std::mutex> g(RopeCSMutex());
    auto& c = RopeCSCache();
    const std::string k = RopeCSKey(thu, halfu);
    auto it = c.find(k);
    if (it == c.end()) {
      RopeCSEntry e;
      e.cos = UploadRows(ce.data(), thu, halfu, device);
      e.sin = UploadRows(se.data(), thu, halfu, device);
      e.cos_host = ce;
      c[k] = std::move(e);
    } else if (it->second.cos_host != ce) {
      // In-place CONTENT refresh of the SAME device tensors: a captured rope
      // op reads the address recorded at capture time, so replacing the
      // tensors here would leave every replay reading the capture-step
      // cos/sin (stale positions). The host tensors are built with the
      // identical bf16 TILE spec so copy_to_device writes byte-matching
      // data. Legal here: the driver calls this outside capture.
      // VT_TT_NO_ROPE_REFRESH: stall bisection only — skip the per-step H2D
      // copies AFTER the first capture (stale cos/sin on replays, numerically
      // wrong, mechanics test only).
      if (ReplayRegimeBisectSkip("VT_TT_NO_ROPE_REFRESH")) return;
      ttnn::Tensor cos_h = ttnn::Tensor::from_vector<float>(
          ce, TileSpecOf(thu, halfu), nullptr);
      ttnn::Tensor sin_h = ttnn::Tensor::from_vector<float>(
          se, TileSpecOf(thu, halfu), nullptr);
      ttnn::copy_to_device(cos_h, it->second.cos);
      ttnn::copy_to_device(sin_h, it->second.sin);
      it->second.cos_host = ce;
    }
  };
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] WarmRopeCosSin tokens=%lld hq=%lld hk=%lld"
                 " rot=%lld first_pos=%d cos_first=%f\n",
                 (long long)tokens, (long long)hq, (long long)hk,
                 (long long)rot, (int)positions[0],
                 cos_t.empty() ? -1.0f : cos_t.front());
  warm_one(hq);
  warm_one(hk);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
    std::lock_guard<std::mutex> g(RopeCSMutex());
    for (auto& [k, e] : RopeCSCache())
      std::fprintf(stderr, "[TT-TRACE] warm stored key=%s first=%f n=%zu\n",
                   k.c_str(), e.cos_host.empty() ? -1.0f : e.cos_host.front(),
                   e.cos_host.size());
  }
}

void WarmPagedKvShadow(void* k_cache_data, void* v_cache_data,
                      int64_t num_blocks, int64_t block_size,
                      int64_t num_kv_heads, int64_t head_size,
                      int64_t used_blocks) {
  if (!HostFreeDecodeEnabled()) return;
  if (num_blocks < 1 || block_size < 1 || used_blocks < 1) return;
  MeshDevice& device = SharedMeshDevice();
  auto warm_one = [&](void* data) {
    Tensor cache = Tensor::Contiguous(
        data, DType::kBF16, Device{DeviceType::kTENSTORRENT, 0},
        {num_blocks, block_size, num_kv_heads, head_size});
    const uint32_t used = static_cast<uint32_t>(
        std::min(used_blocks, num_blocks));
    EnsurePagedKvTtnn(cache, device, used);
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
      std::lock_guard<std::mutex> pg(PagedKvMutex());
      auto& sh = PagedKvShadows();
      std::fprintf(stderr, "[TT-TRACE] WarmPagedKvShadow ptr=%p nb=%lld used=%u shadows=%zu dev=%d\n",
                   data, (long long)num_blocks, used, sh.size(),
                   sh.count(reinterpret_cast<uintptr_t>(data)) ?
                       (int)sh[reinterpret_cast<uintptr_t>(data)].device.has_value() : -1);
    }
  };
  warm_one(k_cache_data);
  warm_one(v_cache_data);
}

void WarmRacIdx(const void* /*slot_mapping_owner*/, const int64_t* slots,
                int64_t num_slots, int64_t block_size,
                const int32_t* block_table, int64_t block_table_cols,
                const int32_t* seq_lens) {
  if (!HostFreeDecodeEnabled()) return;
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] WarmRacIdx n=%lld bs=%lld slot0=%lld sl0=%d\n",
                 (long long)num_slots, (long long)block_size, (long long)slots[0],
                 seq_lens ? seq_lens[0] : -1);
  // VT_TT_NO_IDX_WARM: stall bisection only — skip the per-step H2D copies
  // after the first capture (stale idx/page-table on device, numerically
  // wrong, mechanics test only).
  if (ReplayRegimeBisectSkip("VT_TT_NO_IDX_WARM")) return;
  const bool r2_steady = HostFreeDecodeEnabled()
                         && GraphCapturesDone() > 0;
  if (num_slots < 1) return;
  MeshDevice& device = SharedMeshDevice();
  // paged_update_cache needs:
  //   update_idxs[t] = the sequence position of the token being written
  //     (= seq_lens[t] - 1, the current decode position for user t)
  //   page_table       = the block table (virtual→physical block mapping)
  // The PA reads KV up to cur_pos = seq_lens - 1, so the RAC must write at
  // exactly that position for the PA to see the current token's KV.
  std::vector<int32_t> ptv;
  std::vector<int32_t> idxv;
  // The kernel maps update_idx -> page_table_ptr[update_idx / block_size]
  // (reader_{,paged_fused_}update_cache: virtual_block_id indexes the
  // page-table STICK), so the device page_table must carry the user's WHOLE
  // block-table row, not just the current virtual block (#1476: the old
  // [C,1] tensor made every write past the first block land in a garbage
  // physical block the moment cur_pos crossed block_size).
  ptv.reserve(static_cast<size_t>(num_slots * block_table_cols));
  idxv.reserve(static_cast<size_t>(num_slots));
  for (int64_t t = 0; t < num_slots; ++t) {
    const int64_t slot = slots[t];
    if (slot < 0 || seq_lens == nullptr || block_table == nullptr) {
      // Padding slot: paged_update_cache skips when update_idx == -1.
      for (int64_t c = 0; c < block_table_cols; ++c) ptv.push_back(0);
      idxv.push_back(-1);
      continue;
    }
    // update_idx = the 0-indexed position of the token being decoded this step
    // (= seq_lens[t] - 1, since seq_lens is the length BEFORE this token).
    // paged_update_cache writes to page_table[vblk] * block_size + update_idx % block_size,
    // which must equal the slot_mapping from the scheduler.
    const int32_t cur_pos = seq_lens[t] - 1;
    idxv.push_back(cur_pos);
    // Full row: every virtual block the kernel may resolve this step and
    // later in this width regime (steady state refreshes on content change).
    for (int64_t c = 0; c < block_table_cols; ++c) {
      ptv.push_back(block_table[t * block_table_cols + c]);
    }
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr) {
      const int32_t vblk = cur_pos / static_cast<int32_t>(block_size);
      const int32_t pblk = block_table[t * block_table_cols + vblk];
      std::fprintf(stderr, "[TT-TRACE] WarmRacIdx user=%lld slot=%lld cur_pos=%d vblk=%d pblk=%d bt_cols=%lld (expect slot=%d)\n",
                   (long long)t, (long long)slot, cur_pos, vblk, pblk, (long long)block_table_cols,
                   pblk * static_cast<int32_t>(block_size) + cur_pos % static_cast<int32_t>(block_size));
    }
  }
  const auto key = std::make_pair(num_slots, block_size);
  std::lock_guard<std::mutex> g(RacIdxMutex());
  RacIdxEntry& e = RacIdxCache()[key];
  // idx/page-table tensors are allocated ONCE per key and their CONTENT is
  // refreshed in place each step (copy_to_device, outside capture). The
  // captured paged_update_cache replays against the stable address and reads
  // the fresh values device-side.
  if (!e.allocated || block_table_cols != e.pt_width) {
    // ANY width change (block boundary growth, or the shrink when the longest
    // request of a multi-request batch finishes and block_table_num_cols drops)
    // reallocates: the else-branch copy_to_device would TT_FATAL on a shape
    // mismatch, and the driver resets + re-captures on any column-count change
    // (`cols_changed` compares with `!=`), so the new address is what the next
    // capture records. The retired tensor stays alive (see the field) — never
    // free a buffer a recorded trace addresses.
    if (e.allocated) e.retired_pts.push_back(std::move(e.page_table));
    e.page_table = ttnn::Tensor::from_vector<int32_t>(
        ptv, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_slots),
                    static_cast<uint32_t>(block_table_cols)}),
                    ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
        &device);
    // R2: alias update_idxs to the on-device-advanced cur_pos (DecodePosCache)
    // when VT_TT_HOST_FREE_DECODE and the shapes match (decode T=1:
    // num_slots == num_reqs). plus_one on cur_pos then advances update_idxs
    // too, eliminating the per-replay update_idxs copy_to_device (toxic class).
    bool aliased = false;
    if (HostFreeDecodeEnabled()) {
      std::lock_guard<std::mutex> dg(DecodePosMutex());
      auto dit = DecodePosCache().find(num_slots);
      if (dit != DecodePosCache().end() && dit->second.allocated) {
        e.update_idxs = dit->second.cur_pos;  // share the same device buffer
        aliased = true;
      }
    }
    // After the first capture, a standalone update_idxs is never plus_one'd.
    // Refuse rather than freeze the write index and emit fluent wrong tokens.
    VT_CHECK(!r2_steady || aliased,
             "tenstorrent: WarmRacIdx allocated a standalone update_idxs after "
             "capture — plus_one will not advance it. Seed DecodePos per "
             "cache entry; recapture does NOT clear this (#1105).");
    if (!aliased) {
      e.update_idxs = ttnn::Tensor::from_vector<int32_t>(
          idxv, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_slots)}),
                       ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR),
          &device);
    }
    e.allocated = true;
    e.pt_width = block_table_cols;
  } else {
    // Steady state within one width: update_idxs is aliased to the
    // on-device-advanced cur_pos (plus_one on replay steps, WarmDecodePos
    // re-seed on cold/capture steps) — never copied here. The RAC page_table
    // refreshes ONLY when its content changed (a new block was mapped):
    // zero copies inside a block, so the toxic every-step-interleaved-write
    // class stays out of the steady state; the copy that does fire rides the
    // same step as the boundary re-capture (#1476 — the "Phase 2 full"
    // refresh the old comment owed but never implemented).
    if (ptv != e.pt_host) {
      ttnn::Tensor pt_host = ttnn::Tensor::from_vector<int32_t>(
          ptv, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_slots),
                      static_cast<uint32_t>(block_table_cols)}),
                      ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR));
      ttnn::copy_to_device(pt_host, e.page_table);
    }
  }
  e.pt_host = ptv;
  // Build the persistent sharded RAC input ONCE from the first available
  // paged-KV shadow's geometry (same nkv/d as the cache): logical
  // [1,1,nkv,d], padded [1,1,nkv_pad,d], HEIGHT_SHARDED L1, shard
  // [nkv_pad,d] on one core. paged_update_cache never reads the padded tail
  // rows (num_heads loop bound), so it is left uninitialized — no zeros, no
  // concat.
  if (!e.sharded_in_is_alloc) {
    std::lock_guard<std::mutex> pg(PagedKvMutex());
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
      std::fprintf(stderr, "[TT-TRACE] WarmRacIdx shadow loop: %zu shadows\n",
                   PagedKvShadows().size());
    for (auto& [ptr, shadow] : PagedKvShadows()) {
      if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
        std::fprintf(stderr, "[TT-TRACE] shadow ptr=%p nkv=%u d=%u dc=%d\n",
                     (void*)ptr, shadow.nkv, shadow.d, shadow.device_current);
      if (shadow.nkv > 0 && shadow.d > 0 && (shadow.d % 32u) == 0u) {
        const uint32_t np = std::max(32u, ((shadow.nkv + 31u) / 32u) * 32u);
        const auto grid = device.compute_with_storage_grid_size();
        const tt::tt_metal::CoreRangeSet core_set =
            tt::tt_metal::num_cores_to_corerangeset(1u, grid, true);
        tt::tt_metal::ShardSpec ss(core_set, {np, shadow.d},
                                   tt::tt_metal::ShardOrientation::ROW_MAJOR);
        tt::tt_metal::MemoryConfig sm(
            tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            tt::tt_metal::BufferType::L1, ss);
        // Logical [1,1,nkv,d]; TILE alignment derives the physical
        // [1,1,nkv_pad,d] and the [nkv_pad,d] shard covers it on one core.
        e.sharded_in = ttnn::create_device_tensor(
            tt::tt_metal::TensorSpec(
                tt::tt_metal::Shape({1u, 1u, shadow.nkv, shadow.d}),
                tt::tt_metal::TensorLayout(
                    ttnn::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(ttnn::Layout::TILE), sm)),
            &device);
        // V needs a SEPARATE sharded buffer on a DIFFERENT core (forces a
        // program cache miss so interleaved_to_sharded compiles a fresh
        // program with V's buffer. Without different cores, the second call
        // reuses K's cached program and writes to K's buffer).
        const tt::tt_metal::CoreRangeSet core_set_v =
            tt::tt_metal::CoreRangeSet({
                tt::tt_metal::CoreRange(
                    tt::tt_metal::CoreCoord(1, 0),
                    tt::tt_metal::CoreCoord(1, 0))
            });
        tt::tt_metal::ShardSpec ss_v(core_set_v, {np, shadow.d},
                                     tt::tt_metal::ShardOrientation::ROW_MAJOR);
        tt::tt_metal::MemoryConfig sm_v(
            tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            tt::tt_metal::BufferType::L1, ss_v);
        e.sharded_in_v = ttnn::create_device_tensor(
            tt::tt_metal::TensorSpec(
                tt::tt_metal::Shape({1u, 1u, shadow.nkv, shadow.d}),
                tt::tt_metal::TensorLayout(
                    ttnn::DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(ttnn::Layout::TILE), sm_v)),
            &device);
        e.nkv = shadow.nkv;
        e.d = shadow.d;
        e.sharded_in_is_alloc = true;
        break;
      }
    }
  }
  // paged_update_cache + the interleaved-TILE→sharded ttnn::copy are warmed
  // naturally: WarmPagedKvShadow (called by the driver BEFORE WarmRacIdx)
  // primes the shadows, and the cold step's eager ForwardLayers runs
  // TryReshapeAndCacheDeviceDecode (host_free is set, capturing is false)
  // which runs the identical copy+update sequence, compiling both programs.
}

void WarmPaMeta(const int32_t* block_table, int64_t num_reqs, int64_t max_blocks,
                int64_t bt_row_stride, int64_t bt_col_stride,
                const int32_t* seq_lens) {
  if (!HostFreeDecodeEnabled()) return;
  if (num_reqs < 1) return;
  MeshDevice& device = SharedMeshDevice();
  std::vector<int32_t> pt(static_cast<size_t>(num_reqs * max_blocks));
  for (int64_t r = 0; r < num_reqs; ++r) {
    for (int64_t c = 0; c < max_blocks; ++c) {
      const int32_t id = block_table[r * bt_row_stride + c * bt_col_stride];
      pt[static_cast<size_t>(r * max_blocks + c)] = id;
    }
  }
  std::vector<int32_t> cpos(static_cast<size_t>(num_reqs));
  for (int64_t r = 0; r < num_reqs; ++r) cpos[static_cast<size_t>(r)] = seq_lens[r] - 1;
  // Allocate ONCE per key; refresh CONTENT in place (copy_to_device, outside
  // capture) so the captured sdpa_decode replays against a stable address
  // while reading the fresh block-table/cur-pos values.
  const auto key = std::make_pair(num_reqs, max_blocks);
  // VT_TT_NO_IDX_WARM: legacy bisection override (skip ALL per-step copies).
  if (ReplayRegimeBisectSkip("VT_TT_NO_IDX_WARM")) return;
  const bool r2_steady = HostFreeDecodeEnabled()
                         && GraphCapturesDone() > 0;
  // R2 steady state: cur_pos/update_idxs advance on-device (plus_one); only
  // page_table needs a host refresh, and only when it actually changed (block
  // boundary crossed). The capture step (r2_steady==false) seeds everything.
  ttnn::Tensor pt_host = ttnn::Tensor::from_vector<int32_t>(
      pt, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs),
                static_cast<uint32_t>(max_blocks)}),
                ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR));
  std::lock_guard<std::mutex> g(PaMetaMutex());
  PaMetaEntry& e = PaMetaCache()[key];
  if (!e.allocated) {
    e.page_table = ttnn::Tensor::from_vector<int32_t>(
        pt, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs),
                  static_cast<uint32_t>(max_blocks)}),
                  ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
    // R2: alias cur_pos to the on-device-advanced DecodePos cur_pos (advanced
    // by plus_one in the trace) when VT_TT_HOST_FREE_DECODE and it exists.
    // sdpa_decode reads this tensor; plus_one advances it → no per-replay
    // copy_to_device (the toxic ~38-replay hang class).
    bool aliased = false;
    if (HostFreeDecodeEnabled()) {
      std::lock_guard<std::mutex> dg(DecodePosMutex());
      auto dit = DecodePosCache().find(num_reqs);
      if (dit != DecodePosCache().end() && dit->second.allocated) {
        e.cur_pos = dit->second.cur_pos;  // share the same device buffer
        aliased = true;
      }
    }
    // After the first capture, a standalone cur_pos is never plus_one'd.
    // Refuse rather than freeze KV length and emit fluent wrong tokens.
    VT_CHECK(!r2_steady || aliased,
             "tenstorrent: WarmPaMeta allocated a standalone cur_pos after "
             "capture — plus_one will not advance it. Seed DecodePos per "
             "cache entry; recapture does NOT clear this (#1105).");
    if (!aliased) {
      e.cur_pos = ttnn::Tensor::from_vector<int32_t>(
          cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                       ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
    }
    e.allocated = true;
  } else {
    // R2 steady state: page_table refreshes ONLY when content changed (block
    // boundary crossed). cur_pos/update_idxs advance on-device via plus_one.
    const bool pt_changed = (e.pt_host != pt);
    if (pt_changed || !r2_steady) {
      ttnn::copy_to_device(pt_host, e.page_table);
    }
    if (!r2_steady) {
      ttnn::Tensor cp_host = ttnn::Tensor::from_vector<int32_t>(
          cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                       ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR));
      ttnn::copy_to_device(cp_host, e.cur_pos);
    }
  }
  e.pt_host = pt;
  e.cp_host = cpos;
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] WarmPaMeta n=%lld mb=%lld cp0=%d r2=%d pt_chg=%d\n",
                 (long long)num_reqs, (long long)max_blocks, (int)cpos[0],
                 (int)r2_steady, (int)(e.pt_host != pt));
}

// R2: seed the persistent cur_pos device tensor (= seq_lens - 1) and warm the
// plus_one program (program cache) so CaptureDecodePosAdvance can run inside
// the trace. Called on the capture/warm step (re-seed), NOT every replay.
void WarmDecodePos(const int32_t* seq_lens, int64_t num_reqs, bool replay_regime) {
  if (!HostFreeDecodeEnabled()) return;
  if (num_reqs < 1 || seq_lens == nullptr) return;
  // Replay regime: cur_pos advances on-device by the captured plus_one —
  // re-seeding here would overwrite the advance and break correctness.
  // A new num_reqs that was never seeded is refused rather than left frozen.
  if (replay_regime) {
    std::lock_guard<std::mutex> g(DecodePosMutex());
    auto it = DecodePosCache().find(num_reqs);
    VT_CHECK(it != DecodePosCache().end() && it->second.allocated,
             "tenstorrent: WarmDecodePos after capture for a num_reqs that "
             "was never seeded — cur_pos would freeze. Seed DecodePos per "
             "cache entry; recapture does NOT clear this (#1105).");
    return;
  }
  // Cold/warm/capture step: (re-)seed cur_pos = seq_lens - 1 for THIS step.
  // The regime flag comes from the driver (graph captured?), NOT from
  // GraphCapturesDone(): Reset() releases the trace without clearing that
  // process-global counter, and the cold eager step that follows a Reset
  // runs no plus_one — so after a re-capture the on-device cur_pos is one
  // position behind unless it is re-seeded here (#1476). This also keeps the
  // FIRST capture correct (its capture step re-seeds, which is why the bug
  // only surfaced at the first block boundary, where Reset+re-capture runs).
  MeshDevice& device = SharedMeshDevice();
  std::vector<int32_t> cpos(static_cast<size_t>(num_reqs));
  for (int64_t r = 0; r < num_reqs; ++r)
    cpos[static_cast<size_t>(r)] = seq_lens[r] - 1;

  std::lock_guard<std::mutex> g(DecodePosMutex());
  DecodePosEntry& e = DecodePosCache()[num_reqs];
  if (!e.allocated) {
    e.cur_pos = ttnn::Tensor::from_vector<int32_t>(
        cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                     ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
    e.allocated = true;
  } else {
    ttnn::Tensor cp_host = ttnn::Tensor::from_vector<int32_t>(
        cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                     ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR));
    ttnn::copy_to_device(cp_host, e.cur_pos);
  }
  // Warm plus_one (program cache) on a SCRATCH tensor so the in-trace call
  // doesn't trigger "Cannot load new binaries during trace capture" — but
  // leave e.cur_pos at its seeded value (the warm must NOT advance it, or the
  // captured body reads cur_pos+1).
  {
    ttnn::Tensor scratch = ttnn::Tensor::from_vector<int32_t>(
        cpos, SpecOf(tt::tt_metal::Shape({static_cast<uint32_t>(num_reqs)}),
                     ttnn::DataType::INT32, ttnn::Layout::ROW_MAJOR), &device);
    ttnn::operations::experimental::plus_one(scratch,
        /*sub_core_grids=*/std::nullopt, /*skip_negative_entries=*/true);
  }
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] WarmDecodePos n=%lld cp0=%d (seeded+warmed plus_one)\n",
                 (long long)num_reqs, (int)cpos[0]);
}

// R2: capture ttnn::plus_one(cur_pos) at the END of the trace body. The NEXT
// replay sees cur_pos+1. Must be called INSIDE BeginCapture/EndCapture, after
// all reads of cur_pos (sdpa_decode / paged_update_cache) in the body.
void CaptureDecodePosAdvance(int64_t num_reqs) {
  if (!HostFreeDecodeEnabled()) return;
  std::lock_guard<std::mutex> g(DecodePosMutex());
  auto it = DecodePosCache().find(num_reqs);
  if (it == DecodePosCache().end() || !it->second.allocated) {
    std::fprintf(stderr, "[TT-TRACE] CaptureDecodePosAdvance: no seeded cur_pos for n=%lld\n",
                 (long long)num_reqs);
    return;
  }
  ttnn::operations::experimental::plus_one(it->second.cur_pos,
      /*sub_core_grids=*/std::nullopt, /*skip_negative_entries=*/true);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr)
    std::fprintf(stderr, "[TT-TRACE] CaptureDecodePosAdvance n=%lld (plus_one captured)\n",
                 (long long)num_reqs);
}

}  // namespace vt::tenstorrent
