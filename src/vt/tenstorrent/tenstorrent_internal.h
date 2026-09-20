// Tenstorrent internal seam shared by the tenstorrent_ops.cpp split TUs
// (BACKEND-TENSTORRENT-SPLIT stage 1). Content moved verbatim from
// tenstorrent_ops.cpp: the include/forward-declaration preamble, the
// file-local capture flag / caches / staging counters (now inline), the
// BufferSlot slot table, and the host/device residency declarations.
#ifndef VT_TENSTORRENT_TENSTORRENT_INTERNAL_H_
#define VT_TENSTORRENT_TENSTORRENT_INTERNAL_H_

// merged qkv already has a resident shadow (post MatmulBT), else host memcpy.
#include "vt/backend.h"
#include "vt/cpu/cpu_threadpool.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
// This OBJECT library is not the `vllm` target, so it does not inherit the
// PUBLIC VLLM_CPP_TENSTORRENT define. Force the real declarations; the
// header's inline no-ops are only for CPU/Vulkan/Windows TUs.
#ifndef VLLM_CPP_TENSTORRENT
#define VLLM_CPP_TENSTORRENT
#endif
#include "vt/tenstorrent/tenstorrent_device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
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
#include <ttnn/core.hpp>
// W4 lever 1 (#2107): the bulk bf16 upload hands the master's own bytes to
// ttnn::Tensor::from_span<bfloat16> — these two headers provide the element
// type and the span view it takes.
#include <tt-metalium/bfloat16.hpp>
#include <tt_stl/span.hpp>
// W5 (#2244): the persistent staging route re-uploads through tt-metal's
// in-place H2D (ttnn::copy_to_device) instead of a fresh from_span creation.
#include <tt-metalium/memory_reporter.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/ternary/ternary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/experimental/reshape/view.hpp>
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
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
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
// W4b int8-dot (#3031): the raw below-ttnn device kernel path. The host API
// (CreateProgram/CreateKernelFromString/CreateCircularBuffer/SetRuntimeArgs),
// the program/workload types, the data-movement kernel config, the CB config,
// and the mesh-aware TensorAccessorArgs — the last so the kernel reads the
// staged PACKED words / ROW_MAJOR activation / f32 out through tt-metal's own
// bank-interleave math instead of hand-rolled DRAM addressing.
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <filesystem>
#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
// Exact row gather/scatter for the GDN caches (BACKEND-TENSTORRENT-GDN W2):
// ttnn::gather (data_movement/gather/gather.hpp) and ttnn::indexed_fill
// (indexed_fill/indexed_fill.hpp) are not in the installed include set at our
// pin; the declarations below mirror the source-tree signatures 1:1 (same
// doctrine as the transformer declarations above — the symbols are exported
// by _ttnncpp.so).
namespace ttnn {
Tensor gather(const Tensor& input_tensor, int8_t dim,
              const Tensor& input_index_tensor, bool sparse_grad,
              const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
              std::optional<Tensor> optional_output_tensor = std::nullopt,
              const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids =
                  std::nullopt);
Tensor indexed_fill(const Tensor& batch_id, const Tensor& input_tensor_a,
                     const Tensor& input_tensor_b,
                     const std::optional<tt::tt_metal::MemoryConfig>& memory_config =
                         std::nullopt,
                     int64_t dim = 0);
// data_movement/transpose/transpose.hpp is also outside the include set.
Tensor transpose(const Tensor& input_tensor, int64_t dim1, int64_t dim2,
                 float pad_value = 0.0f);
}  // namespace ttnn
#ifdef VT_RESTORE_TRACY_ENABLE
#define TRACY_ENABLE 1
#undef VT_RESTORE_TRACY_ENABLE
#endif

#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>

namespace vt::tenstorrent {

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
inline bool& tt_capture_active() {
  static bool b = false;
  return b;
}

// Capture-safe reshape: the free ttnn::reshape (from reshape_view/reshape.hpp)
// launches ReshapeViewTiledProgramFactory::create_program_artifacts which
// calls to_device — forbidden during trace capture, and a cache miss when
// the slot state differs between eager warmup and capture. During capture,
// the member Tensor::reshape(logical, old_padded) is a pure metadata view
// (view_device, same buffer, no program). The old padded shape is reused so
// the buffer size check passes. The data is correct for same-numel reshapes
// because TILE layout stores data in flat row-major within the tile grid —
// element i maps to the same physical byte regardless of the logical shape
// interpretation (the tile grid is the same; only the logical dims change).
// Downstream ops may see a different padded shape than the eager step warmed,
// but element-wise ops (sigmoid, multiply, typecast, add) don't depend on
// the padded shape for correctness; shape-dependent ops (matmul, rms_norm)
// use the logical shape which matches.
inline ttnn::Tensor CaptureSafeReshape(const ttnn::Tensor& t, const ttnn::Shape& shape) {
  if (!tt_capture_active()) {
    return ttnn::reshape(t, shape);
  }
  // During capture, use the member Tensor::reshape (pure metadata view,
  // same buffer, no device program). Try the old padded shape first
  // (always fits the buffer), then tile-aligned padded as fallback.
  const auto old_padded = t.padded_shape();
  try {
    return t.reshape(shape, old_padded);
  } catch (...) {
    const auto rank = shape.rank();
    ttsl::SmallVector<uint32_t> padded_dims;
    for (uint32_t i = 0; i < rank; ++i) {
      if (i + 2 >= rank) {
        padded_dims.push_back(((shape[i] + 31u) / 32u) * 32u);
      } else {
        padded_dims.push_back(shape[i]);
      }
    }
    try {
      return t.reshape(shape, ttnn::Shape(padded_dims));
    } catch (...) {
      return t;  // both failed: return original (may cause downstream issues)
    }
  }
}

// KEEPQUANT W3 capture-safety probe (tenstorrent_device.h): staging writes the
// keep-quant decode performed while a capture was active. The staged arm must
// hold this at zero across a captured run (the W3 red-first test reads it).
// The counter lives at internal linkage; the accessors below the file's
// anonymous namespace give the parity/test TUs the external surface.
inline std::atomic<int64_t>& KeepQuantCaptureStagingWritesCounter() {
  static std::atomic<int64_t>* c = new std::atomic<int64_t>(0);  // never destroyed (#1486)
  return *c;
}

// W4a wave-3a: the E=1 slice-decode chunk-rows override (0 = production
// policy). Internal linkage; the ForTest setter below the anonymous
// namespace is the external surface (the staging-counter pattern).
inline std::atomic<int64_t>& KeepQuantChunkRowsOverride() {
  static std::atomic<int64_t> v{0};
  return v;
}

// ITEM 5 (rope): persistent device cos/sin (expanded per head), built OUTSIDE
// capture and ttnn::copy'd in-region — the UploadRows in RopeApplyDeviceNeox
// was the enqueue_write that killed capture at mid-layer-0. The cache is
// keyed by (tokens*heads, half) + the exact host cos/sin CONTENT: if the
// step's positions changed the table, we must NOT silently reuse a stale
// cached tensor — during capture that is a hard error (the driver must warm
// the new table first, the SizeSlot::Refresh pattern).
inline std::mutex& RopeCSMutex() {
  static std::mutex m;
  return m;
}
struct RopeCSEntry {
  ttnn::Tensor cos;
  ttnn::Tensor sin;
  std::vector<float> cos_host;  // content identity for the reuse check
};
inline std::map<std::string, RopeCSEntry>& RopeCSCache() {
  // Every cache accessor in this file follows this shape: the singleton is
  // HEAP-ALLOCATED and deliberately never destroyed (#1486). A static-storage
  // ttnn::Tensor dies in a __run_exit_handlers destructor that unwinds AFTER
  // tt-metal's own teardown — its deallocate reaches GraphTracker::is_enabled()
  // on a torn-down tracker and the process SIGSEGVs after an all-green
  // summary. The destructor registered at first use is always NEWER in the
  // LIFO exit order than any drain we could register at load, so an exit hook
  // cannot fix this; not destroying the cache does. The OS reclaims the pages;
  // the device teardown frees the allocations it backs.
  static std::map<std::string, RopeCSEntry>* c = new std::map<std::string, RopeCSEntry>();
  return *c;
}
inline std::string RopeCSKey(uint32_t th, uint32_t half) {
  return std::to_string(th) + "x" + std::to_string(half);
}

inline std::mutex& AttnCSMutex() {
  static std::mutex m;
  return m;
}
// kAttnQkNormRopeGate's per-step cos|sin table [T, rot] f32 TILE at a fixed
// device address. Same doctrine as RopeCSCache above: content-keyed, and a
// captured op must never see the tensor replaced under it — the refresh is an
// in-place copy_to_device (WarmRopeCosSin's).
struct AttnCSEntry {
  ttnn::Tensor cs;
  std::vector<float> cs_host;  // content identity for the reuse check
};
inline std::map<std::string, AttnCSEntry>& AttnCSCache() {
  // Heap-allocated, never destroyed (#1486 — every cache accessor here).
  static std::map<std::string, AttnCSEntry>* c =
      new std::map<std::string, AttnCSEntry>();
  return *c;
}
inline std::string AttnCSKey(uint32_t t, uint32_t rot) {
  return std::to_string(t) + "x" + std::to_string(rot);
}

inline std::mutex& ZeroCacheMutex() {
  static std::mutex m;
  return m;
}
inline std::map<std::string, ttnn::Tensor>& ZeroCache() {
  static std::map<std::string, ttnn::Tensor>* c = new std::map<std::string, ttnn::Tensor>(); // never destroyed (#1486)
  return *c;
}
inline std::string ZeroCacheKey(const ttnn::Shape& shape, ttnn::DataType dt,
                         ttnn::Layout lt) {
  std::string k;
  for (auto d : shape.view()) k += std::to_string(d) + "x";
  k += std::to_string(static_cast<int>(dt)) + "x" +
       std::to_string(static_cast<int>(lt));
  return k;
}

struct BufferSlot {
  void* host = nullptr;
  size_t bytes = 0;
  std::optional<ttnn::Tensor> device;
  uint32_t dev_rows = 0;
  uint32_t dev_cols = 0;
  bool host_current = true;    // host bytes match the latest value
  bool device_current = false; // device tensor matches the latest value
  // #2812: the gemma (w+1) F32 baked affine form, staged once and reused.
  // Separate from `device` (the raw BF16 form EnsureAffine1D caches) so a
  // non-gemma consumer of the same buffer never reads the +1 values. The
  // per-call transient upload this replaces is a host write, which a trace
  // capture forbids — the Qwen3.5 captured arm fatalled here on every norm.
  std::optional<ttnn::Tensor> gemma_device;
  // tenant (MarkScratchAcquired) and no producer has established its content
  // yet. The resident allocation and the host bytes both hold the PREVIOUS
  // tenant's bytes — undefined for the new tenant, whose pending device op
  // overwrites the buffer it is served. EnsureDevice2D therefore stages
  // NOTHING for a reserved slot: it serves the resident persistent buffer at
  // the same geometry, or hands out an empty one at a new geometry. Restage
  // semantics: the service ALIASES the slot's persistent buffer in place (W5)
  // — no fresh snapshot is created. The shadow-installing transitions clear
  // the flag (CommitHost, MarkHostWritten, the device commits, the staging
  // commits, the memset/copy arms); paths that install content without
  // spending it (the plain staging arm, the embed-table patch) are still safe
  // because they set device_current, and the reserved arm refuses whenever
  // device_current is set — a leaked flag can cost a restage but never
  // discards a live shadow (the #2282 drift: a stale [5,1024] persistent
  // served over a live [8,256] commit poisoned prompt 1's first token).
  bool device_reserved = false;
  // BACKEND-TENSTORRENT-GDN W2: the conv-state shadow is stored TIME-MAJOR
  // ([sl+1, slots*C], one scratch row) because ttnn slice/concat are exact on
  // dim 0 only at this pin (sub-tile last-dim slice/concat is broken — see
  // the W2 evidence). When set, host materialization transposes back into
  // the caller's [slots, C, sl] byte order. Cleared by every commit/host
  // write that replaces the shadow with a different layout.
  bool conv_transposed = false;
  uint32_t conv_slots = 0, conv_c = 0, conv_sl = 0;
  // BACKEND-TENSTORRENT-QWEN35 W5 (#2244): the slot's PERSISTENT staged-device
  // buffer. Allocated once per (slot, staging geometry) by the bulk bf16 arm
  // and rewritten IN PLACE through the mesh command queue on every later
  // staging, so an identical-geometry upload no longer pays from_span's fresh
  // MeshBuffer allocation / tensor creation path. `device` above remains the
  // consumer-visible shadow (dropped by every host write, replaced by commits
  // and reshapes); `persistent` survives those drops and holds the resident
  // device allocation. Its content is only ever observed through a shadow
  // that a full staging write has just refreshed, so a stale resident buffer
  // is unreachable. The slot lives in the never-destroyed Slots() map
  // (#1486), so the tensor is never destroyed after tt-metal teardown.
  std::optional<ttnn::Tensor> persistent;
  uint32_t persist_rows = 0, persist_cols = 0;
};

// ---- Persistent weight-view staging (TILE BF16 on device) ------------------
ttnn::Tensor EnsureDevice2D(const Tensor& t, MeshDevice& device);
// MatmulBT receives interior weight views: the packed GDN projections are
// Slice'd per call (qwen3_5.cpp qkvz/z splits), and EnsureDevice2D refuses to
// serve an interior view from the base's tracked staging (W2c: those are the
// wrong offset's bytes). The view therefore re-staged anonymously on every
// call — a per-call host write that trace capture forbids (#2812: the
// Qwen3.5-0.8B captured battery fatalled in exactly this arm, fd_mesh_command_queue.cpp:760,
// 2/2 deterministic) and a pure eager tax. Key the shadow by the VIEW's own
// data pointer plus its geometry: a view's address is stable for the model's
// lifetime (the parent host buffer outlives the step loop), and a recycled
// address must also match rows x cols to collide. Weight windows are
// immutable after load; if a mutable weight window ever appears, its parent's
// MarkHostWritten must drop the views' shadows the way DropEmbedTableShadow
// does for the vocab table.
struct WeightViewShadow {
  std::optional<ttnn::Tensor> device;
  uint32_t rows = 0, cols = 0;
};

inline std::string DevShapeStr(const ttnn::Tensor& t) {
  const auto s = t.logical_shape();
  std::string r;
  for (uint32_t i = 0; i < s.rank(); ++i) {
    if (i != 0) r += 'x';
    r += std::to_string(s[i]);
  }
  r += " dt=" + std::to_string(static_cast<int>(t.dtype())) +
       " lay=" + std::to_string(static_cast<int>(t.layout()));
  return r;
}
struct BufferSlot;  // defined above
// ---- moved declarations (definitions in tenstorrent_residency.cpp) ---------
std::mutex& SlotMutex();
std::map<uintptr_t, BufferSlot>& Slots();
BufferSlot* FindSlot(void* p);
tt::tt_metal::TensorSpec SpecOf(tt::tt_metal::Shape shape, ttnn::DataType dtype,
                                ttnn::Layout layout);
tt::tt_metal::TensorSpec TileSpecOf(uint32_t rows, uint32_t cols);
float LoadElemF32(const Tensor& t, int64_t i);
void StoreElemF32(Tensor& t, int64_t i, float v);
bool IsFloatDType(DType d);
void DownloadToHost(ttnn::Tensor& dev, Tensor& out, const char* ctx);
void EnsureHost(Tensor& t);
void EnsureHost(const Tensor& t);
std::vector<float> ToHostF32(const Tensor& t);
ttnn::Tensor UploadRows(const float* data, uint32_t rows, uint32_t cols, MeshDevice& device);
ttnn::Tensor UploadRowsBf16(const Tensor& t, uint32_t rows, uint32_t cols,
                            MeshDevice& device);
ttnn::Tensor ZeroCacheGet(const ttnn::Shape& shape, ttnn::DataType dt,
                          ttnn::Layout lt, MeshDevice& device);
ttnn::Tensor ZeroCacheGet(const ttnn::Tensor& like, MeshDevice& device);
void ZeroCachePrime(const ttnn::Shape& shape, ttnn::DataType dt,
                    ttnn::Layout lt, MeshDevice& device);
std::mutex& WeightViewMutex();
std::map<uintptr_t, WeightViewShadow>& WeightViewShadows();
bool IsTrackedBase2D(const Tensor& t);
ttnn::Tensor EnsureWeightViewDevice(const Tensor& t, MeshDevice& device);
ttnn::Tensor EnsureMatmulWeightDevice(const Tensor& b, MeshDevice& device);
ttnn::Tensor EnsureDevice2D(const Tensor& t, MeshDevice& device);
bool DeviceShadowExact(const Tensor& t, uint32_t rows, uint32_t cols);
// ---- BFP8 weight residency (spec .agents/specs/tenstorrent-bfp-weight-
// residency.md) ----------------------------------------------------------
// VT_TT_WEIGHT_RESIDENCY=bfp8: a bf16 matmul WEIGHT operand stages once as a
// device-resident BFLOAT8_B tensor (bf16 upload, device typecast, bf16 staging
// dropped) and the native ttnn matmul consumes it (bf16 activation x BFP8
// weight). Read fresh on every call so a test can flip the env mid-process.
bool Bfp8WeightsEnabled();
ttnn::Tensor EnsureBfp8WeightDevice(const Tensor& b, MeshDevice& device);
void Bfp8MatmulUse();  // a matmul consumed a BFP8-resident weight operand
// Probes for the focused tests (declared in the test TU, defined here — the
// test TU never includes internal headers).
uint64_t Bfp8ResidentWeights();   // conversions completed
uint64_t Bfp8MatmulUses();        // matmul calls that consumed a BFP8 weight
uint64_t Bfp8Refusals();          // named refusals (fell through to the bf16 arm)
const char* Bfp8LastRefusal();    // the last refusal reason, "" if none
void NoteBfp8Refusal(std::string reason);  // record one named refusal
// ---- KEEPQUANT W3: the resident i32 word shadow -------------------------
struct KeepQuantWordShadow {
  ttnn::Tensor words;  // {B, wpb} INT32 ROW_MAJOR — the packed stream, word-staged
  int64_t rows = 0;
  int64_t nb = 0;
  int wpb = 0;
};

inline std::mutex& KeepQuantWordMutex() {
  static std::mutex m;
  return m;
}
// Keyed by the packed tensor's host base pointer — the EnsureMatmulWeightDevice
// persistent-shadow pattern. An interior view carries its own pointer, so a
// differently-offset view stages its own shadow and never consumes another
// slice's bytes (the Qwen3.5 BA interior-view fatality).
inline std::map<const void*, KeepQuantWordShadow>& KeepQuantWordShadows() {
  static std::map<const void*, KeepQuantWordShadow>* m =
      new std::map<const void*, KeepQuantWordShadow>();  // never destroyed (#1486)
  return *m;
}

// dense keep-quant matmul no longer inserts (it serves packed words through
// the E=1 grouped arm), so a populated entry for a matmul weight is exactly
// the regression the probe names. Same collision discipline as before: the
// free path drops by host pointer (UnregisterHostBuffer).
struct DecodedWeightShadow {
  std::optional<ttnn::Tensor> device;
  uint32_t rows = 0, cols = 0;
  DType enc = DType::kF32;
};
inline std::mutex& DecodedWeightMutex() {
  static std::mutex m;
  return m;
}
inline std::map<uintptr_t, DecodedWeightShadow>& DecodedWeightShadows() {
  static std::map<uintptr_t, DecodedWeightShadow>* m =
      new std::map<uintptr_t, DecodedWeightShadow>(); // never destroyed (#1486)
  return *m;
}

// GroupedActShadow: the grouped-quant activation's device-side bf16 TILE
// staging, keyed by the host activation pointer. EnsureDevice2D's slot
// staging corrupts under trace capture (ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5);
// from_span stages directly from host, and the shadow cache serves the same
// device tensor under capture (the warm-first contract — the eager step stages
// before capture, the capture-time miss refuses). Same collision discipline as
// the other resident shadows: the free path drops by host pointer.
struct GroupedActShadow {
  ttnn::Tensor device;
  uint32_t rows = 0, cols = 0;
  DType dtype = DType::kF32;
};
inline std::mutex& GroupedActMutex() {
  static std::mutex m;
  return m;
}
inline std::map<uintptr_t, GroupedActShadow>& GroupedActShadows() {
  static std::map<uintptr_t, GroupedActShadow>* m =
      new std::map<uintptr_t, GroupedActShadow>(); // never destroyed (#1486)
  return *m;
}

// ---- moved declarations (definitions in tenstorrent_keepquant.cpp) ----
ttnn::Tensor EnsureKeepQuantWords(const Tensor& packed, DType enc, int64_t rows,
                                  int64_t nb, MeshDevice& device);
void TTReclaimPlanes(MeshDevice& device, std::vector<ttnn::Tensor>& planes);
void KeepQuantDecodeKernel(Queue&, Tensor& out, const Tensor& packed);
void MatmulBTQuantKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b);
void MatmulBTQuantGroupedKernel(Queue&, Tensor& out, const Tensor& act,
                                const Tensor& weight, const Tensor& expert_ids);
void MatmulBTQuantGroupedKernel(Queue&, Tensor& out, const Tensor& act,
                                const Tensor& weight);
void MatmulBTQuantInt8DotKernel(Queue& q, Tensor& out, const Tensor& a,
                                const Tensor& b);
void DropKeepQuantWordShadow(void* host);
void DropDecodedWeightShadow(void* host);
void DropGroupedActShadow(void* host);
void CommitDeviceLogical2D(Tensor& out, ttnn::Tensor dev, uint32_t rows,
                           uint32_t cols);

// ---- BACKEND-TENSTORRENT-QWEN35 W4 (#2107): bulk staging counters ----
std::atomic<uint64_t>& StagingBulkUploads();
std::atomic<uint64_t>& StagingBulkBytes();
std::atomic<uint64_t>& StagingF32Elems();
std::atomic<uint64_t>& StagingPersistentWrites();
std::atomic<uint64_t>& StagingPersistentAllocs();
std::atomic<uint64_t>& StagingPersistentBytes();
std::atomic<uint64_t>& StagingAvoidedReservation();
std::atomic<uint64_t>& StagingAvoidedMemset();
std::atomic<uint64_t>& StagingAvoidedDeviceCopy();
}  // namespace vt::tenstorrent

#endif  // VT_TENSTORRENT_TENSTORRENT_INTERNAL_H_
