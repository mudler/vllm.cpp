// Tenstorrent host/device residency (BACKEND-TENSTORRENT-SPLIT stage 1).
// Definitions moved verbatim from tenstorrent_ops.cpp; declarations live
// in tenstorrent_internal.h.
#include "vt/tenstorrent/tenstorrent_internal.h"

#include <cstdlib>

#include "vllm/config/tt_weight_residency.h"

namespace vt::tenstorrent {

ttnn::Tensor ZeroCacheGet(const ttnn::Shape& shape, ttnn::DataType dt,
                          ttnn::Layout lt, MeshDevice& device) {
  const std::string key = ZeroCacheKey(shape, dt, lt);
  std::lock_guard<std::mutex> g(ZeroCacheMutex());
  auto& c = ZeroCache();
  auto it = c.find(key);
  if (it == c.end()) {
    VT_CHECK(!tt_capture_active(),
             "tenstorrent: zero-cache miss during capture — warm the "
             "host-free path eagerly (VT_TT_HOST_FREE_DECODE warmup) first");
    it = c.emplace(key, ttnn::zeros(shape, dt, lt, std::ref(device))).first;
  }
  return it->second;
}

ttnn::Tensor ZeroCacheGet(const ttnn::Tensor& like, MeshDevice& device) {
  return ZeroCacheGet(like.logical_shape(), like.dtype(), like.layout(),
                      device);
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



std::mutex& SlotMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, BufferSlot>& Slots() {
  static std::map<uintptr_t, BufferSlot>* m = new std::map<uintptr_t, BufferSlot>(); // never destroyed (#1486)
  return *m;
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

void DownloadToHost(ttnn::Tensor& dev, Tensor& out, const char* ctx) {
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-TRACE] to_vector readback DURING CAPTURE\n");
  std::vector<float> result = dev.to_vector<float>();
  if (result.size() != static_cast<size_t>(out.Numel())) {
    void* bt[8];
    const int nbt = ::backtrace(bt, 8);
    char** sym = ::backtrace_symbols(bt, nbt);
    std::fprintf(stderr, "[TT-SLOT] MISMATCH self=%p frames=%d\n",
                 reinterpret_cast<const void*>(&DownloadToHost), nbt);
    for (int i = 0; sym != nullptr && i < nbt; ++i)
      std::fprintf(stderr, "[TT-SLOT]   bt[%d]=%p %s\n", i, bt[i], sym[i]);
    std::fflush(stderr);
    std::free(sym);
  }
  VT_CHECK(static_cast<int64_t>(result.size()) == out.Numel(),
           std::string("tenstorrent: unexpected result size: got ") +
               std::to_string(result.size()) + " want " +
               std::to_string(out.Numel()) + " out_shape=" +
               std::to_string(out.shape[0]) + "x" + std::to_string(out.shape[1]) +
               "x" + std::to_string(out.shape[2]) + "x" + std::to_string(out.shape[3]) +
               " rank=" + std::to_string(out.rank) +
               " ctx=" + std::string(ctx) +
               " dev[" + DevShapeStr(dev) + "]");
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
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr)
    std::fprintf(stderr,
                 "[TT-SLOT] ensure t=%p numel=%" PRId64
                 " slot=%p bytes=%zu dev=%ux%u dt=%d ht=%d dc=%d ct=%d\n",
                 static_cast<const void*>(t.data), t.Numel(),
                 static_cast<void*>(s->host), s->bytes, s->dev_rows,
                 s->dev_cols, static_cast<int>(s->device->dtype()),
                 s->host_current, s->device_current, s->conv_transposed);
  if (s->conv_transposed) {
    std::vector<float> v = s->device->to_vector<float>();
    for (int64_t i = 0; i < t.Numel(); ++i) {
      // host[s*C*sl + c*sl + j] == device[j*R + s*C + c]
      const int64_t j = i % s->conv_sl, rc = i / s->conv_sl;
      const int64_t col = rc % s->conv_c, slot = rc / s->conv_c;
      StoreElemF32(t, i, v[static_cast<size_t>(j * (s->conv_slots * s->conv_c) +
                                               slot * s->conv_c + col)]);
    }
    s->host_current = true;
    return;
  }
  // Interior contiguous view of the owner slot: the host tensor is a window
  // (a strided row view, a rank-3 activation) whose bytes live at one flat
  // span of the owner's device plane. Download the owner once and copy the
  // span — the owner plane is what the device holds, the span is what this
  // view owns. Without this branch EnsureHost compared the OWNER's numel
  // against the WINDOW's and refused by name (the vllm-bench chunked-prefill
  // GdnPrefillKernel readback, ISSUE-LOCAL-01M2E5F69CMWDERKXG32YY9P8N). A
  // non-contiguous view is not served — its bytes are not one span.
  const int64_t total_dev =
      static_cast<int64_t>(s->dev_rows) * s->dev_cols;
  // A slot whose registered allocation does not COVER this view is not this
  // view's owner: the slot map is never erased (#1486) and a freed tensor's
  // range can be re-allocated to an unrelated later tensor (the vllm-bench
  // prefill chunk buffers). Treat it as untracked host memory — the bytes
  // are whoever the current producer wrote.
  const int64_t t_elem =
      t.dtype == DType::kF32 ? 4 : 2;  // host-side float storages
  const uintptr_t slot_base = reinterpret_cast<uintptr_t>(s->host);
  if (reinterpret_cast<uintptr_t>(t.data) + t.Numel() * t_elem >
      slot_base + s->bytes)
    return;
  if (total_dev != t.Numel()) {
    // A device plane SMALLER than the view cannot back the view's bytes at
    // all — the slot's shadow is a stale per-step commit (the captured
    // decode web commits one token's plane into a chunk-sized arena slot;
    // the prefill chunk then reuses the arena and its producer writes the
    // host bytes directly). The host bytes are the truth here: read them,
    // touch nothing on the slot (ISSUE-LOCAL-01M2E5F69CMWDERKXG32YY9P8N).
    if (total_dev < t.Numel()) return;
    VT_CHECK(t.IsContiguous(),
             "tenstorrent: EnsureHost on a non-contiguous window of a "
             "device-authoritative owner is not served");
    const int64_t elem_bytes =
        s->device->dtype() == ttnn::DataType::FLOAT32 ? 4 : 2;
    const int64_t delta = static_cast<const char*>(t.data) -
                          static_cast<const char*>(s->host);
    if (!(delta >= 0 && delta % elem_bytes == 0 &&
          delta / elem_bytes + t.Numel() <= total_dev)) {
      std::fprintf(stderr,
                   "[TT-WINDOW] reader t=%p rank=%d shape=%" PRId64 "x%" PRId64
                   "x%" PRId64 "x%" PRId64 " strides=%" PRId64 "x%" PRId64
                   "x%" PRId64 "x%" PRId64 " dt=%d delta=%" PRId64
                   " numel=%" PRId64 " dev=%s total_dev=%" PRId64 "\n",
                   static_cast<const void*>(t.data), t.rank, t.shape[0],
                   t.shape[1], t.shape[2], t.shape[3], t.stride[0], t.stride[1],
                   t.stride[2], t.stride[3], static_cast<int>(t.dtype), delta,
                   t.Numel(), DevShapeStr(*s->device).c_str(), total_dev);
      void* bt[10];
      const int nbt = ::backtrace(bt, 10);
      char** sym = ::backtrace_symbols(bt, nbt);
      for (int i = 0; sym != nullptr && i < nbt; ++i)
        std::fprintf(stderr, "[TT-WINDOW]   bt[%d]=%p %s\n", i, bt[i], sym[i]);
      std::fflush(stderr);
      std::free(sym);
    }
    VT_CHECK(delta >= 0 && delta % elem_bytes == 0 &&
                 delta / elem_bytes + t.Numel() <= total_dev,
             "tenstorrent: EnsureHost window is not an in-bounds flat span "
             "of the owner slot (delta=" + std::to_string(delta) +
                 " elem_bytes=" + std::to_string(elem_bytes) +
                 " off=" + std::to_string(delta / elem_bytes) +
                 " numel=" + std::to_string(t.Numel()) +
                 " total_dev=" + std::to_string(total_dev) +
                 " dev=" + DevShapeStr(*s->device) + ")");
    const int64_t off = delta / elem_bytes;
    std::vector<float> v = s->device->to_vector<float>();
    for (int64_t i = 0; i < t.Numel(); ++i)
      StoreElemF32(t, i, v[static_cast<size_t>(off + i)]);
    // The OWNER slot's host buffer is NOT filled — only this window's bytes
    // are. Leave the slot device-authoritative (host_current stays false) so
    // no consumer reads stale owner bytes outside the span.
    return;
  }
  DownloadToHost(*s->device, t, "EnsureHost");
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

// ---- BACKEND-TENSTORRENT-QWEN35 W4 (#2107): bulk staging counters ----
// Incremented by EnsureDevice2D's two staging routes; read through the
// tenstorrent_device.h API (GetStagingStats below) so the W4 route pin is
// evidence, not assumption. At vt::tenstorrent scope so both the staging
// paths above and the readers below see them.
std::atomic<uint64_t>& StagingBulkUploads() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingBulkBytes() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingF32Elems() {
  static std::atomic<uint64_t> v{0};
  return v;
}
// W5 (#2244): the persistent-route counters — in-place mesh CQ writes into
// the per-slot buffer, the (re)allocations of that buffer, and the bytes
// pushed through it. Read through GetStagingStats below.
std::atomic<uint64_t>& StagingPersistentWrites() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingPersistentAllocs() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingPersistentBytes() {
  static std::atomic<uint64_t> v{0};
  return v;
}
// BACKEND-TENSTORRENT-QWEN35 W7 (#2282): staging writes the precise-residency
// arms eliminate, one counter per class — reservation (EnsureDevice2D serving
// a pool-acquired block from its resident allocation), device_memset
// (MemsetDeviceFill keeping the shadow across an eager full-slot zero-fill),
// device_copy (CopyDeviceDeviceIfResident skipping the download+restage pair).
std::atomic<uint64_t>& StagingAvoidedReservation() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingAvoidedMemset() {
  static std::atomic<uint64_t> v{0};
  return v;
}
std::atomic<uint64_t>& StagingAvoidedDeviceCopy() {
  static std::atomic<uint64_t> v{0};
  return v;
}

// W4 lever 1 (#2107): bulk upload of a contiguous bf16 master. The host
// bytes ARE the payload: one from_span over the tensor's own memory — no f32
// intermediate, no per-element dtype dispatch, no extra vector copy, and no
// bfloat16::from_float round-trip on the ttnn side. Bit-identical to the f32
// path: bf16→f32 widening is exact, and packing a value whose low 16 mantissa
// bits are zero back to bf16 returns the same bits under any rounding rule.
//
// W5 (#2244): the upload no longer pays tt-metal's per-upload creation path
// on every step. A tracked base slot stages through its PERSISTENT device
// buffer: the first staging for a geometry runs the full from_span creation
// (and the buffer stays resident in the slot); every later staging packs the
// host bytes with the SAME function from_span calls (tt-metal
// host_tensor_from_span_with_pad_value, ttnn/core/tensor/tensor.cpp:170) and
// writes them through tt-metal's in-place H2D — ttnn::copy_to_device into the
// resident MeshTensor, which reaches MeshCommandQueue::enqueue_write /
// enqueue_write_shards against the existing buffer. No fresh MeshBuffer, no
// cluster/chip rediscovery, no new tensor attributes. The bytes on the device
// are the same packed bytes from_span writes, into a buffer of the same
// geometry, fully overwritten each time: bit-identical. Untracked pointers
// and interior views keep the anonymous from_span arm (W2c: a view must
// never store against the base slot).
ttnn::Tensor UploadRowsBf16(const Tensor& t, uint32_t rows, uint32_t cols,
                            MeshDevice& device) {
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr, "[TT-UP] UploadRowsBf16 from_span WRITE during capture ptr=%p rows=%u cols=%u\n",
                 (const void*)t.data, rows, cols);
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  // The bytes at t.Ptr are the window's own bf16 bits (bfloat16 is a 2-byte
  // class wrapping the same uint16 pattern).
  const bfloat16* src = reinterpret_cast<const bfloat16*>(t.Ptr<uint16_t>());
  // Short locked probe: resolve the persistent buffer once. The CQ write runs
  // OUTSIDE the lock — the same probe/upload/re-lock discipline EnsureDevice2D
  // uses — and the copied handle (shared TensorAttributes) keeps the resident
  // MeshBuffer alive even if the slot is unregistered mid-upload.
  std::optional<ttnn::Tensor> persistent;
  bool tracked_base = false;
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    tracked_base = (s != nullptr && t.data == s->host);
    if (tracked_base && s->persistent.has_value() &&
        s->persist_rows == rows && s->persist_cols == cols) {
      persistent = s->persistent;
    }
  }
  if (!tracked_base) {
    // Untracked pointer or interior view (W2c): anonymous staging, no
    // persistent buffer, no W5 counters — the caller's bulk counters still see it.
    return ttnn::Tensor::from_span(ttsl::Span<const bfloat16>(src, n),
                                   TileSpecOf(rows, cols), &device);
  }
  if (persistent.has_value()) {
    // In-place arm. Host half: exactly the packing from_span performs
    // (tt-metal ttnn/core/tensor/tensor.cpp:170 — same function, same spec,
    // same pad), with no device argument so no device work happens. Device
    // half: tt-metal's own in-place H2D (ttnn/core/tensor/tensor_ops.cpp:161
    // copy_to_device → enqueue_write_tensor into the EXISTING MeshTensor,
    // tt_metal/impl/tensor/tensor_apis.cpp:149) — the same write path
    // from_span's to_device takes, minus the fresh MeshBuffer allocation and
    // tensor creation. Bytes on the device are the same packed bytes, into a
    // buffer of the same geometry, fully overwritten: bit-identical.
    if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
      std::fprintf(stderr, "[TT-UP] UploadRowsBf16 persistent enqueue_write during capture ptr=%p rows=%u cols=%u\n",
                   (const void*)t.data, rows, cols);
    ttnn::Tensor host = ttnn::Tensor::from_span(ttsl::Span<const bfloat16>(src, n),
                                                TileSpecOf(rows, cols),
                                                /*device=*/nullptr);
    ttnn::copy_to_device(host, *persistent);
    StagingPersistentWrites().fetch_add(1, std::memory_order_relaxed);
    StagingPersistentBytes().fetch_add(static_cast<uint64_t>(n) * 2,
                                       std::memory_order_relaxed);
    return *persistent;
  }
  // Allocating arm (cold slot or staging-geometry change): the full W4
  // creation path, and the returned tensor becomes the slot's persistent
  // buffer. The stale resident buffer, if any, is released here.
  ttnn::Tensor dev = ttnn::Tensor::from_span(ttsl::Span<const bfloat16>(src, n),
                                             TileSpecOf(rows, cols), &device);
  {
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    if (s != nullptr && t.data == s->host) {
      s->persistent = dev;
      s->persist_rows = rows;
      s->persist_cols = cols;
    }
  }
  StagingPersistentWrites().fetch_add(1, std::memory_order_relaxed);
  StagingPersistentAllocs().fetch_add(1, std::memory_order_relaxed);
  StagingPersistentBytes().fetch_add(static_cast<uint64_t>(n) * 2,
                                     std::memory_order_relaxed);
  return dev;
}

std::mutex& WeightViewMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, WeightViewShadow>& WeightViewShadows() {
  static std::map<uintptr_t, WeightViewShadow>* m =
      new std::map<uintptr_t, WeightViewShadow>(); // never destroyed (#1486)
  return *m;
}

// True iff `t` is a rank-2 contiguous tensor whose pointer is a tracked BASE
// (EnsureDevice2D's fast-path condition). Views and untracked pointers are
// the ones EnsureDevice2D stages anonymously.
bool IsTrackedBase2D(const Tensor& t) {
  if (t.rank != 2 || !t.IsContiguous()) return false;
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  return s != nullptr && t.data == s->host;
}

// Persistent device staging for an interior/untracked BF16 weight view.
// Stages the window's own bytes once (the same from_span the anonymous arm
// runs, so the packed device bytes are identical) and reuses the resident
// tensor from then on — capture-safe and eager-cheap.
ttnn::Tensor EnsureWeightViewDevice(const Tensor& t, MeshDevice& device) {
  VT_CHECK(t.rank == 2 && t.IsContiguous() && t.dtype == DType::kBF16,
           "tenstorrent EnsureWeightViewDevice: contiguous rank-2 bf16");
  const uint32_t rows = static_cast<uint32_t>(t.shape[0]);
  const uint32_t cols = static_cast<uint32_t>(t.shape[1]);
  const uintptr_t key = reinterpret_cast<uintptr_t>(t.data);
  {
    std::lock_guard<std::mutex> g(WeightViewMutex());
    auto it = WeightViewShadows().find(key);
    if (it != WeightViewShadows().end() && it->second.device.has_value() &&
        it->second.rows == rows && it->second.cols == cols) {
      return *it->second.device;
    }
  }
  EnsureHost(t);
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (std::getenv("VT_TT_TRACE_DEBUG") != nullptr && tt_capture_active())
    std::fprintf(stderr,
                 "[TT-UP] EnsureWeightViewDevice from_span WRITE during "
                 "capture ptr=%p rows=%u cols=%u\n",
                 t.data, rows, cols);
  ttnn::Tensor dev = ttnn::Tensor::from_span(
      ttsl::Span<const bfloat16>(static_cast<const bfloat16*>(t.data), n),
      TileSpecOf(rows, cols), &device);
  // The view stage IS a bulk bf16 staging: report it through the same
  // counters EnsureDevice2D's bf16 arm bumps, so the W4 contract ("each view
  // stage is its own bulk upload") holds for the shadow arm too. Later calls
  // serve the resident shadow and, like a resident serve there, count nothing.
  StagingBulkUploads().fetch_add(1, std::memory_order_relaxed);
  StagingBulkBytes().fetch_add(static_cast<uint64_t>(n) * 2,
                               std::memory_order_relaxed);
  std::lock_guard<std::mutex> g(WeightViewMutex());
  WeightViewShadow& s = WeightViewShadows()[key];
  s.device = dev;
  s.rows = rows;
  s.cols = cols;
  return dev;
}

// MatmulBT's weight staging: tracked bases keep EnsureDevice2D's fast path;
// interior views get the persistent shadow instead of the per-call anonymous
// staging.
ttnn::Tensor EnsureMatmulWeightDevice(const Tensor& b, MeshDevice& device) {
  // One lever selects the residency; a future variant is a new VALUE of
  // VT_TT_WEIGHT_RESIDENCY, not a new flag.
  switch (vllm::ParseTtWeightResidency(
      std::getenv("VT_TT_WEIGHT_RESIDENCY"))) {
    case vllm::TtWeightResidency::kBfp8:
      return EnsureBfp8WeightDevice(b, device);
    case vllm::TtWeightResidency::kBfp4:
      NoteBfp8Refusal("VT_TT_WEIGHT_RESIDENCY=bfp4: not implemented; staged "
                      "bf16");
      break;
    case vllm::TtWeightResidency::kOff:
      break;
  }
  if (b.dtype == DType::kBF16 && !IsTrackedBase2D(b))
    return EnsureWeightViewDevice(b, device);
  return EnsureDevice2D(b, device);
}

// ---- Weight residency ----------------------------------------------------
// spec .agents/specs/tenstorrent-bfp-weight-residency.md. DEFAULT OFF
// (VT_TT_WEIGHT_RESIDENCY=off / unset leaves the byte-identical bf16 staging
// path —
// the inertness gate pins that). When bfp8, a bf16 matmul weight uploads once in
// bf16, is TYPECAST on device to BFLOAT8_B (BFP8: 1 sign + 7 shared-group
// mantissa bits per element, one 8-bit exponent per 16-element group —
// tt_metal/impl/data_format/blockfloat_common.cpp `convert_bfp_to_u32`,
// Bfp8_b arm), and the bf16 staging is dropped: exactly the expand-bf16
// conversion shape, no second full-precision device copy. The native
// ttnn::matmul consumes it (bf16 activation x bfloat8_b weight is a supported
// operand pair — ttnn/operations/matmul/matmul.cpp:521-532).

std::atomic<uint64_t> g_bfp8_resident{0};
std::atomic<uint64_t> g_bfp8_uses{0};
std::atomic<uint64_t> g_bfp8_refusals{0};
std::mutex& Bfp8RefusalMutex() {
  static std::mutex m;
  return m;
}
std::string& Bfp8RefusalMsg() {
  static std::string* s = new std::string();  // never destroyed (#1486)
  return *s;
}

bool Bfp8WeightsEnabled() {
  return vllm::ParseTtWeightResidency(std::getenv("VT_TT_WEIGHT_RESIDENCY")) ==
         vllm::TtWeightResidency::kBfp8;
}
uint64_t Bfp8ResidentWeights() { return g_bfp8_resident.load(std::memory_order_relaxed); }
uint64_t Bfp8MatmulUses() { return g_bfp8_uses.load(std::memory_order_relaxed); }
uint64_t Bfp8Refusals() { return g_bfp8_refusals.load(std::memory_order_relaxed); }
void Bfp8MatmulUse() { g_bfp8_uses.fetch_add(1, std::memory_order_relaxed); }
const char* Bfp8LastRefusal() {
  std::lock_guard<std::mutex> g(Bfp8RefusalMutex());
  return Bfp8RefusalMsg().c_str();
}
void NoteBfp8Refusal(std::string reason) {
  g_bfp8_refusals.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> g(Bfp8RefusalMutex());
  Bfp8RefusalMsg() = std::move(reason);
}

// Keyed by the weight's host base pointer — the EnsureWeightViewShadow
// persistent-shadow pattern. An interior view carries its own pointer, so a
// differently-offset view stages its own BFP8 tensor and never consumes
// another slice's bytes (the Qwen3.5 BA interior-view fatality).
struct Bfp8WeightShadow {
  std::optional<ttnn::Tensor> device;
  uint32_t rows = 0, cols = 0;
};
std::mutex& Bfp8WeightMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, Bfp8WeightShadow>& Bfp8WeightShadows() {
  static std::map<uintptr_t, Bfp8WeightShadow>* m =
      new std::map<uintptr_t, Bfp8WeightShadow>();  // never destroyed (#1486)
  return *m;
}

ttnn::Tensor EnsureBfp8WeightDevice(const Tensor& b, MeshDevice& device) {
  const uint32_t rows = static_cast<uint32_t>(b.shape[0]);
  const uint32_t cols = static_cast<uint32_t>(b.shape[1]);
  const auto refuse = [&](std::string why) {
    NoteBfp8Refusal(std::move(why));
    if (b.dtype == DType::kBF16 && !IsTrackedBase2D(b))
      return EnsureWeightViewDevice(b, device);
    return EnsureDevice2D(b, device);
  };
  if (b.dtype != DType::kBF16)
    return refuse("tenstorrent BFP8 weight residency: weight is not bf16 (" +
                  std::to_string(static_cast<int>(b.dtype)) + ")");
  if (b.rank != 2 || !b.IsContiguous())
    return refuse("tenstorrent BFP8 weight residency: non-contiguous or "
                  "non-rank-2 weight");
  if (rows % 32 != 0 || cols % 32 != 0)
    return refuse("tenstorrent BFP8 weight residency: shape " +
                  std::to_string(rows) + "x" + std::to_string(cols) +
                  " is not TILE-aligned");
  const uintptr_t key = reinterpret_cast<uintptr_t>(b.data);
  {
    std::lock_guard<std::mutex> g(Bfp8WeightMutex());
    auto it = Bfp8WeightShadows().find(key);
    if (it != Bfp8WeightShadows().end() && it->second.device.has_value() &&
        it->second.rows == rows && it->second.cols == cols) {
      return *it->second.device;
    }
  }
  EnsureHost(b);
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  ttnn::Tensor bf16_dev = ttnn::Tensor::from_span(
      ttsl::Span<const bfloat16>(static_cast<const bfloat16*>(b.data), n),
      TileSpecOf(rows, cols), &device);
  // The device typecast is the SAME single RNE BFP pack the tt-metal host
  // packer performs (typecast_sharded_program_factory.cpp:48 — BFLOAT16 ->
  // BFLOAT8_B is a supported TILE conversion), executed on-device so the
  // packed operand never exists on the host and the bf16 staging is dropped
  // with the temporary.
  ttnn::Tensor bfp8_dev = ttnn::typecast(bf16_dev, ttnn::DataType::BFLOAT8_B);
  g_bfp8_resident.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> g(Bfp8WeightMutex());
  Bfp8WeightShadow& s = Bfp8WeightShadows()[key];
  s.device = bfp8_dev;
  s.rows = rows;
  s.cols = cols;
  return bfp8_dev;
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
  // The per-slot cache describes the BASE allocation. An interior view
  // (e.g. packed_weight.Slice(0, Hv, 2*Hv) fed to a matmul) resolves to the
  // base slot but must NEVER hit nor store against it: the cached tensor is
  // the BASE's staging, and returning it for a differently-offset view makes
  // the consumer read another slice's bytes (Qwen3.5 BA: TT computed the `a`
  // projection with the `b` weight rows — BACKEND-TENSTORRENT-QWEN35 W2c).
  bool tracked_base = false;
  bool need_host_refresh = false;
  bool reserved_base = false;
  {
    // W4 lever 2 (#2107): ONE locked probe resolves everything this call
    // needs — the tracked fast paths (exact-shape hit and same-numel
    // reshape), the interior-view refusal, and whether the base's host
    // master must be pulled back from the device before its bytes are read.
    // The pre-W4 shape re-probed the slot map under the mutex up to four
    // times per call.
    std::lock_guard<std::mutex> g(SlotMutex());
    BufferSlot* s = FindSlot(t.data);
    tracked_base = (s != nullptr && t.data == s->host);
    reserved_base = tracked_base && s->device_reserved;
    if (tracked_base && s->device_current && s->device.has_value()) {
      if (s->dev_rows == rows && s->dev_cols == cols) {
        const auto ls = s->device->logical_shape();
        if (ls.rank() == 2 && ls[0] == rows && ls[1] == cols)
          return *s->device;
        if (tt_capture_active()) {
          s->device = CaptureSafeReshape(*s->device, ttnn::Shape({rows, cols}));
          return *s->device;
        }
        ttnn::Tensor reshaped = ttnn::to_layout(
            ttnn::reshape(ttnn::to_layout(*s->device, ttnn::Layout::ROW_MAJOR),
                          ttnn::Shape({rows, cols})),
            s->device->layout());
        s->device = reshaped;
        return *s->device;
      }
      const uint64_t have =
          static_cast<uint64_t>(s->dev_rows) * static_cast<uint64_t>(s->dev_cols);
      const uint64_t want = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
      if (have == want) {
        if (tt_capture_active()) {
          ttnn::Tensor reshaped =
              CaptureSafeReshape(*s->device, ttnn::Shape({rows, cols}));
          s->device = reshaped;
          s->dev_rows = rows;
          s->dev_cols = cols;
          return *s->device;
        }
        ttnn::Tensor reshaped =
            ttnn::reshape(*s->device, ttnn::Shape({rows, cols}));
        s->device = reshaped;
        s->dev_rows = rows;
        s->dev_cols = cols;
        return *s->device;
      }
    }
    // Interior view over a base whose host master is neither current nor
    // device-refreshable would read stale bytes — refuse loudly, the same
    // state the post-refresh check below the old probe observed.
    need_host_refresh = (s != nullptr && s->device_current && !s->host_current);
    VT_CHECK(tracked_base || s == nullptr || need_host_refresh || s->host_current,
             "tenstorrent: EnsureDevice2D interior view of a device-current "
             "base is unsupported (would read stale bytes); stage via the "
             "base tensor");
  }
  // The reservation arm is bf16-only: it serves the slot's W5 persistent
  // staging buffer (BFLOAT16/TILE) or a fresh BFLOAT16 empty, so a master of
  // any other dtype taking the arm would receive a bf16 device tensor for its
  // declared dtype. A non-bf16 master whose first device use lands on an
  // acquired block falls through to the normal staging path below, whose f32
  // arm keeps the declared dtype (the W7 invariant).
  if (reserved_base && t.dtype == DType::kBF16) {
    // W7 (#2282) reservation: see BufferSlot::device_reserved. The previous
    // tenant's bytes sit on BOTH sides and the new tenant's pending device op
    // overwrites the buffer it is served — stage NOTHING. Serve the resident
    // persistent allocation at the same geometry, or hand out an empty one at
    // a new geometry (no bytes move in either direction). Restage semantics:
    // the service ALIASES the slot's persistent buffer in place (W5), it does
    // not create a fresh snapshot. The re-check under the lock covers a host
    // write that raced the probe: once the bytes are real, the upload
    // contract applies again.
    bool served = false;
    ttnn::Tensor dev;
    {
      std::lock_guard<std::mutex> g(SlotMutex());
      BufferSlot* s = FindSlot(t.data);
      // The reservation describes a slot whose bytes are still the previous
      // tenant's on both sides. A live shadow means some producer already
      // established the content — serving the resident persistent buffer here
      // would discard it and hand the consumer stale bytes (the W7 drift,
      // #2282). Refuse: the normal paths below serve or restage the truth.
      if (s != nullptr && t.data == s->host && s->device_reserved &&
          !s->device_current) {
        if (s->persistent.has_value() && s->persist_rows == rows &&
            s->persist_cols == cols) {
          dev = *s->persistent;
        } else {
          dev = ttnn::empty(ttnn::Shape({rows, cols}), ttnn::DataType::BFLOAT16,
                            ttnn::Layout::TILE, &device, ttnn::MemoryConfig{});
        }
        s->device = dev;
        s->dev_rows = rows;
        s->dev_cols = cols;
        s->device_current = true;
        s->host_current = false;
        s->device_reserved = false;
        StagingAvoidedReservation().fetch_add(1, std::memory_order_relaxed);
        served = true;
      }
    }
    if (served) return dev;
  }
  if (need_host_refresh) EnsureHostBytes(t.data);
  // HOST-FREE-FORWARD defect: this loop consumed HOST bytes without checking
  // slot residency. Under host-free decode a producer commits device-only
  // (host_current=false), so the next consumer staging through here uploaded
  // pool-fresh zeros and then marked the slot host_current=true — corrupting
  // both the value and the record. Refresh the host bytes from the device
  // shadow before reading them.
  EnsureHostBytes(t.data);
  ttnn::Tensor dev;
  if (t.dtype == DType::kBF16) {
    // W4 lever 1 (#2107): contiguous bf16 masters skip the f32 intermediate
    // entirely — the raw window bytes go to the device in one from_span.
    dev = UploadRowsBf16(t, rows, cols, device);
    StagingBulkUploads().fetch_add(1, std::memory_order_relaxed);
    StagingBulkBytes().fetch_add(static_cast<uint64_t>(rows) * cols * 2,
                                 std::memory_order_relaxed);
  } else {
    // f16/f32 masters keep the f32 reference arm (genuine conversion); the
    // element count is hoisted — the old loop evaluated Numel() per element.
    const int64_t n = t.Numel();
    std::vector<float> host(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      host[static_cast<size_t>(i)] = LoadElemF32(t, i);
    dev = UploadRows(host.data(), rows, cols, device);
    StagingF32Elems().fetch_add(static_cast<uint64_t>(n),
                                std::memory_order_relaxed);
  }
  if (HostFreeDecodeEnabled()) {
    // Prime the persistent-zero cache for this spec during the eager warmup
    // (capture-safe zeroing replays ttnn::copy(zero, dst) — see MemsetDevice).
    ZeroCachePrime(ttnn::Shape({rows, cols}), ttnn::DataType::BFLOAT16,
                   ttnn::Layout::TILE, device);
  }
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  if (s != nullptr && t.data == s->host) {
    s->device = dev;
    s->dev_rows = rows;
    s->dev_cols = cols;
    s->device_current = true;
    s->host_current = true;
  }
  return dev;
}

// (StageWeightBf16ForTest is defined at vt::tenstorrent scope below — it
// needs external linkage for the focused test and calls EnsureDevice2D,
// which stays TU-internal here.)

// DEBUG (BACKEND-TENSTORRENT-QWEN35 W2c): ensure the tensor is staged on
// device exactly as a consuming kernel would, then read the DEVICE copy back.
// Comparing this against the host master exposes staging corruption that a
// host-side dump cannot see.
// True when `t` already has a device-resident shadow matching [rows, cols]
// (exact shape). Used to pick device vs host kernels without forcing upload.
bool DeviceShadowExact(const Tensor& t, uint32_t rows, uint32_t cols) {
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(t.data);
  return s != nullptr && s->device_current && s->device.has_value() &&
         s->dev_rows == rows && s->dev_cols == cols;
}

// to host (the residency win). Host is marked stale until EnsureHost.
// Device tensor is stored as logical [rows, cols] TILE (may differ from out's
// rank-3 view as long as numel matches) so a later Reshape+EnsureDevice2D hits.
void CommitDeviceLogical2D(Tensor& out, ttnn::Tensor dev, uint32_t rows, uint32_t cols) {
  VT_CHECK(out.IsContiguous(), "tenstorrent: CommitDeviceLogical2D expects contiguous out");
  VT_CHECK(out.Numel() == static_cast<int64_t>(rows) * static_cast<int64_t>(cols),
           "tenstorrent: CommitDeviceLogical2D numel mismatch");
  {
    int64_t vol = 1;
    const auto ds = dev.logical_shape();
    for (uint32_t i = 0; i < ds.rank(); ++i) vol *= ds[i];
    VT_CHECK(vol == static_cast<int64_t>(rows) * static_cast<int64_t>(cols),
             std::string("tenstorrent: CommitDeviceLogical2D device volume ") +
                 std::to_string(vol) + " != rows*cols " +
                 std::to_string(rows * cols));
  }
  std::lock_guard<std::mutex> g(SlotMutex());
  BufferSlot* s = FindSlot(out.data);
  if (std::getenv("VT_TT_SLOT_TRACE") != nullptr) {
    const auto ds = dev.logical_shape();
    int64_t vol = 1;
    for (uint32_t i = 0; i < ds.rank(); ++i) vol *= ds[i];
    std::fprintf(stderr,
                 "[TT-SLOT] commit out=%p rows=%u cols=%u tracked=%d vol=%" PRId64
                 " dt=%d\n",
                 static_cast<const void*>(out.data), rows, cols,
                 s != nullptr ? 1 : 0, vol, static_cast<int>(dev.dtype()));
  }
  if (s == nullptr) {
    // Untracked buffer (e.g. stack/test scratch): fall back to host write.
    DownloadToHost(dev, out, "CommitDeviceLogical2D(untracked)");
    return;
  }
  s->device = std::move(dev);
  s->dev_rows = rows;
  s->dev_cols = cols;
  s->device_current = true;
  s->host_current = false;
  s->conv_transposed = false;  // logical [rows, cols] — oracle layout
  s->device_reserved = false;  // real bytes committed — the reservation is spent
}

}  // namespace vt::tenstorrent
