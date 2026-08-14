// vllm.cpp original; see qwen3_5_weights.h. Weight naming/quant verified
// against nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot 491c2f1e
// (.agents/specs/qwen36-forward-notes.md §6).
#include "vllm/model_executor/models/qwen3_5_weights.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/unaligned.h"

namespace vllm {

int64_t OwnedTensor::Numel() const {
  if (rank == 0) return 0;
  int64_t n = 1;
  for (int i = 0; i < rank; ++i) n *= shape[i];
  return n;
}

void OwnedTensor::ReleaseHost() const {
  // Free the host mirror once the device-resident copy is authoritative
  // (residency_policy().release_host_weights_after_upload; BACKEND-PLATFORM
  // item 2). Logically const — only the dead host buffer is reclaimed — so a
  // narrow const_cast, consistent with the mutable lazy-device-residency design.
  auto& self = *const_cast<OwnedTensor*>(this);
  if (self.bytes.empty()) return;
  // A BORROWED buffer owns no anonymous pages: the bytes live in the GGUF mmap
  // (clean, file-backed, reclaimable by the kernel without our help) or in an
  // expansion another tensor still references. madvise'ing them away would be
  // wrong on both counts, so releasing one means only dropping the keep-alive.
  if (self.bytes.borrowed()) {
    self.bytes.Reset();
    self.mmap_src = nullptr;  // nothing borrows the mapping through this tensor now
    self.mmap_src_bytes = 0;
    self.host_released = true;
    return;
  }
#if defined(__unix__) || defined(__APPLE__)
  // RETURN THE PHYSICAL PAGES TO THE OS NOW. std::vector's free() alone does not:
  // glibc raises its dynamic mmap threshold as large blocks are freed, so the
  // ~0.5 MB per-expert fp4 allocations end up served from the sbrk arena, where
  // free() only returns them to the free-list — RSS/PSS stay resident (measured:
  // the 35B serving PSS did NOT drop from the logical swap alone). madvise(
  // MADV_DONTNEED) drops the resident anonymous pages immediately (the bytes are
  // dead — the device Marlin resident is authoritative; a stray later read would
  // fault fresh zero pages, but nothing reads a released host weight). Interior
  // whole pages only, so the free()-metadata boundary words are untouched. This
  // mirrors the LOAD-SAFETENSORS windowed release (safetensors_reader.cpp:317).
  if (!self.bytes.empty()) {
    const long ps_l = ::sysconf(_SC_PAGESIZE);
    const auto ps = static_cast<uintptr_t>(ps_l > 0 ? ps_l : 4096);
    const auto begin = reinterpret_cast<uintptr_t>(self.bytes.data());
    const uintptr_t end = begin + self.bytes.size();
    const uintptr_t page_begin = (begin + ps - 1) & ~(ps - 1);
    const uintptr_t page_end = end & ~(ps - 1);
    if (page_end > page_begin) {
      ::madvise(reinterpret_cast<void*>(page_begin),
                static_cast<size_t>(page_end - page_begin), MADV_DONTNEED);
    }
  }
#endif
  // Reset() forces the underlying vector to deallocate its capacity.
  self.bytes.Reset();
  self.host_released = true;
}

// ENG-LOAD-DIRECT-UPLOAD (issue #150). A weight whose bytes BORROW the
// safetensors mmap has just been uploaded to the device, so its source range is
// consumed-and-dead exactly as a copied one is: drop its resident (clean,
// file-backed) pages, mirroring the loaders' windowed release. Correctness-safe
// on any backend -- the mapping is PROT_READ MAP_PRIVATE, so a later read of the
// still-valid borrowed view simply re-faults it from the file.
namespace {

void ReleaseDirectUploadSource(const OwnedTensor& w) {
  if (w.mmap_src == nullptr) return;
  // Deliberately NOT MaybeReleaseSourcePages: that call site is the host-copy
  // BYTE COUNTER (every copy helper reaches it after materializing a range),
  // and these bytes were never materialized on the host -- they went straight to
  // the device. Counting them there would report the very copy this row removes.
  if (LoadWindowedReleaseEnabled())
    ReleaseSourcePages(w.mmap_src, w.mmap_src_bytes);
}

}  // namespace

void AdoptDeviceBytesAsHost(vt::Backend& backend, const OwnedTensor& w) {
  if (w.d_dev == nullptr) return;
  // ENG-LOAD-DIRECT-UPLOAD: a direct-upload borrow is the ONE borrow that may be
  // adopted. Its bytes are the file mapping, not an anonymous copy and not a
  // tied pair's shared expansion, so re-pointing it at the (host-addressable)
  // device allocation loses nothing and stops the page cache from holding a
  // second copy of the model for the rest of the process. Where the device
  // allocation is NOT host-addressable the borrow stays as it is (a valid,
  // re-faultable view) and only the resident pages are dropped.
  if (w.mmap_src != nullptr && w.bytes.borrowed()) {
    // ORDERING IS LOAD-BEARING, AND IT IS THE FIRST STATEMENT FOR A REASON.
    //
    // (1) Release BEFORE re-pointing `bytes`. The assignment below overwrites
    // the OwnedBytes that carries this tensor's keep-alive on
    // `StTensor::mapping`. By adoption time the shard's `SafetensorsFile` is
    // long gone (`LoadedEngine::FromModelDir` drops `shards` at the end of the
    // load; adoption runs lazily at first forward use), so for the LAST
    // adopted weight of a shard that keep-alive is the last reference and
    // `~Mapping` munmaps SYNCHRONOUSLY inside the assignment. Releasing after
    // it would madvise a range that is no longer mapped -- observed under
    // strace as `munmap(...) = 0` followed by `madvise(..., MADV_DONTNEED) =
    // -1 ENOMEM`. That is a silent no-op only for as long as nothing else
    // mmaps into the hole first; on a private anonymous mapping landing there,
    // MADV_DONTNEED DISCARDS live data. `ReleaseHost()`'s borrowed branch
    // already orders it this way.
    //
    // (2) Release BEFORE the two early returns. The page release and the
    // adoption are two independent levers: a backend without host-addressable
    // device memory, and the `VT_ADOPT_DEVICE_BYTES=0` A/B, must both still
    // drop the consumed source pages. Otherwise the documented adoption knob
    // silently turns off the direct-upload release as well.
    ReleaseDirectUploadSource(w);
    // Not host-addressable: the borrow STAYS as it is -- a valid, re-faultable
    // PROT_READ MAP_PRIVATE view -- so `mmap_src` is deliberately left set.
    if (!backend.DeviceMemoryIsHostAddressable()) return;
    if (const char* v = std::getenv("VT_ADOPT_DEVICE_BYTES");
        v != nullptr && v[0] == '0') {
      return;
    }
    auto& self = *const_cast<OwnedTensor*>(&w);
    const size_t nb = self.bytes.size();
    std::shared_ptr<const void> keep(w.d_dev,
                                     static_cast<const void*>(w.d_dev.get()));
    self.bytes = OwnedBytes::Borrow(static_cast<const uint8_t*>(w.d_dev.get()),
                                    nb, std::move(keep));
    self.mmap_src = nullptr;
    self.mmap_src_bytes = 0;
    return;
  }
  if (!backend.DeviceMemoryIsHostAddressable()) return;
  // A BORROWED buffer owns no anonymous pages (a GGUF mmap is clean and
  // file-backed; a shared expansion is a tied pair's single copy), so adopting
  // would reclaim nothing and would break the tie. Same reasoning as
  // ReleaseHost's borrowed branch.
  if (w.bytes.empty() || w.bytes.borrowed()) return;
  if (const char* v = std::getenv("VT_ADOPT_DEVICE_BYTES"); v != nullptr && v[0] == '0') {
    return;
  }
  auto& self = *const_cast<OwnedTensor*>(&w);
  const size_t nb = self.bytes.size();
#if defined(__unix__) || defined(__APPLE__)
  // Drop the resident anonymous pages BEFORE the vector is destroyed, for
  // exactly the reason ReleaseHost above does it: glibc raises its dynamic mmap
  // threshold as large blocks are freed, so free() alone leaves many weight
  // buffers on the sbrk arena free-list with their pages still resident, and
  // the whole point here is the RSS. Interior whole pages only, so free()'s
  // boundary metadata is untouched.
  {
    const long ps_l = ::sysconf(_SC_PAGESIZE);
    const auto ps = static_cast<uintptr_t>(ps_l > 0 ? ps_l : 4096);
    const auto begin = reinterpret_cast<uintptr_t>(self.bytes.data());
    const uintptr_t end = begin + nb;
    const uintptr_t page_begin = (begin + ps - 1) & ~(ps - 1);
    const uintptr_t page_end = end & ~(ps - 1);
    if (page_end > page_begin) {
      ::madvise(reinterpret_cast<void*>(page_begin),
                static_cast<size_t>(page_end - page_begin), MADV_DONTNEED);
    }
  }
#endif
  // The keep-alive is the device allocation's own shared_ptr, so the borrowed
  // view cannot outlive the bytes it points at: the aliasing constructor shares
  // d_dev's control block, and the buffer is freed through the vt Backend only
  // when BOTH the weight's d_dev and this view are gone.
  std::shared_ptr<const void> keep(w.d_dev, static_cast<const void*>(w.d_dev.get()));
  self.bytes = OwnedBytes::Borrow(static_cast<const uint8_t*>(w.d_dev.get()), nb,
                                  std::move(keep));
}

namespace {

// nullopt => env-driven; set => forced (test seam).
std::optional<bool>& DirectUploadOverride() {
  static std::optional<bool> value;
  return value;
}

}  // namespace

void detail::SetLoadDirectUploadOverrideForTesting(std::optional<bool> value) {
  DirectUploadOverride() = value;
}

bool LoadDirectUploadEnabled() {
  const std::optional<bool> forced = DirectUploadOverride();
  if (forced.has_value()) return *forced;
  // Process-cached: read the env once. DEFAULT ON; "=0" rolls back to the
  // copy-then-upload behavior (same-binary A/B, house convention).
  static const bool enabled = [] {
    const char* e = std::getenv("VT_LOAD_DIRECT_UPLOAD");
    return e == nullptr || e[0] != '0';
  }();
  return enabled;
}

bool BorrowStTensorBytes(OwnedTensor& o, const StTensor& t, vt::DType dtype,
                         const std::vector<int64_t>& shape) {
  if (!LoadDirectUploadEnabled()) return false;
  // FAIL CLOSED on anything that is not a whole-range verbatim view of a live
  // mapping: no keep-alive (a synthetic StTensor), no data, or a destination
  // whose element count times dtype width is not EXACTLY the source span. The
  // caller then runs its ordinary copy, so a mismatch can only cost the lever,
  // never correctness.
  if (t.mapping == nullptr || t.data == nullptr || t.nbytes == 0) return false;
  const auto rank = static_cast<int>(shape.size());
  if (rank <= 0 || rank > vt::kMaxRank) return false;
  size_t numel = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0) return false;
    const auto d = static_cast<size_t>(dim);
    if (numel > SIZE_MAX / d) return false;
    numel *= d;
  }
  const size_t elem = vt::SizeOf(dtype);
  if (elem == 0 || numel > SIZE_MAX / elem) return false;
  if (numel * elem != t.nbytes) return false;

  o.dtype = dtype;
  o.rank = rank;
  for (int i = 0; i < rank; ++i) o.shape[i] = shape[static_cast<size_t>(i)];
  o.bytes = OwnedBytes::Borrow(t.data, t.nbytes, t.mapping);
  o.mmap_src = t.data;
  o.mmap_src_bytes = t.nbytes;
  load_stats::AddBorrowed(t.nbytes);
  return true;
}

vt::Tensor OwnedTensor::View() const {
  VT_CHECK(!host_released,
           "OwnedTensor::View: host bytes were released after device upload");
  vt::Tensor t;
  t.data = const_cast<uint8_t*>(bytes.data());
  t.dtype = dtype;
  t.repacked = repacked;  // CIQ G7: carry the i8mm-repack marker to the kernel
  t.q8_0_aligned = q8_0_aligned;  // Brick 4: carry the CUDA coalesced-Q8_0 marker
  t.device = vt::Device{};  // default = CPU host
  t.rank = rank;
  int64_t stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    t.shape[i] = shape[i];
    t.stride[i] = stride;
    stride *= shape[i];
  }
  return t;
}

namespace {

OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "qwen3_5 weights: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

float ReadF32Scalar(const StTensor& t) {
  VT_CHECK(t.data != nullptr && t.nbytes >= sizeof(float),
           "qwen3_5 weights: scalar tensor too small for f32");
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  return v;
}

// src bf16 [rows, cols] -> dst bf16 [cols, rows].
//
// `src` is `const void*`, not `const uint16_t*`, because the caller below hands
// it a pointer INTO the mmap'd safetensors payload. A tensor's offset there is
// the running byte total of everything ahead of it, so a bf16 tensor that
// follows an odd-length one starts on an odd byte and the typed pointer is
// undefined to form or load through (issue #627). `vt::LoadUnaligned` is the
// project's seam for that — the same one `ReadF32Scalar` above open-codes with
// memcpy and `dense_loaders::TransposeBf16` already uses for this exact loop.
void TransposeBf16(const void* src, int64_t rows, int64_t cols, uint16_t* dst) {
  const auto* bytes = static_cast<const uint8_t*>(src);
  for (int64_t r = 0; r < rows; ++r) {
    const uint8_t* src_row = bytes + r * cols * 2;
    for (int64_t c = 0; c < cols; ++c) {
      dst[c * rows + r] = vt::LoadUnaligned<uint16_t>(src_row + c * 2);
    }
  }
}

// BF16 tensor copied verbatim (optionally reshaped): shape override lets the
// [conv_dim,1,K] conv weight collapse to [conv_dim,K] with no data change.
OwnedTensor LoadBf16Direct(const TensorResolver& get, const std::string& name,
                           const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 weights: expected BF16 for " + name);
  std::vector<int64_t> shape =
      shape_override.empty() ? t.shape : shape_override;
  OwnedTensor borrowed;
  // ENG-LOAD-DIRECT-UPLOAD (issue #150): whole-range verbatim copy; qualifies.
  if (BorrowStTensorBytes(borrowed, t, vt::DType::kBF16, shape)) return borrowed;
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  VT_CHECK(t.nbytes == o.bytes.size(),
           "qwen3_5 weights: byte-size mismatch for " + name);
  std::memcpy(o.bytes.data(), t.data, t.nbytes);
  // LOAD-SAFETENSORS: source range now copied-then-dead; drop its resident pages
  // so the owned mirror never double-resides with the mmap (spec §page-lifetime).
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// BF16 [out, in] -> owned bf16 [in, out] (Matmul-B layout).
OwnedTensor LoadBf16Transposed(const TensorResolver& get,
                               const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 weights: expected BF16 for " + name);
  VT_CHECK(t.shape.size() == 2,
           "qwen3_5 weights: expected 2-D weight for " + name);
  const int64_t out_dim = t.shape[0];
  const int64_t in_dim = t.shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(t.data, out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// BF16 [n] -> owned f32 [n] (A_log / dt_bias; upcast is lossless).
OwnedTensor LoadBf16ToF32(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 weights: expected BF16 for " + name);
  VT_CHECK(t.shape.size() == 1,
           "qwen3_5 weights: expected 1-D tensor for " + name);
  const int64_t n = t.shape[0];
  OwnedTensor o = MakeOwned(vt::DType::kF32, {n});
  // Unaligned: `t.data` is an arbitrary byte offset into the mmap (#627).
  const uint8_t* src = t.data;
  auto* dst = reinterpret_cast<float*>(o.bytes.data());
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(src + i * 2));
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// Native-precision dense projections — keep the 35B's per-tensor FP8 W8A8 attn
// (q/k/v/o) + GDN (in_proj_qkv/z, out_proj) weights RESIDENT in fp8 (LoadFp8Raw)
// + load the static input_scale, so the forward runs a native fp8 W8A8 GEMM
// (static per-tensor activation quant + fp8 tensor cores — either the cutlass
// sm120 kernel or, by default, cuBLASLt fp8; see MatmulFp8CutlassD /
// DenseCublasLtFp8Enabled). Mirrors vLLM's actual scheme (the checkpoint IS
// W8A8) and HALVES those projections' weight bytes vs bf16.
//   DEFAULT ON (VT_DENSE_NATIVE) on a CUDA+cutlass build. VT_DENSE_NATIVE=0 (or
// the legacy VT_FP8_CUTLASS=0) restores the dequant-fp8->bf16-AT-LOAD path
// (LoadFp8Transposed into the bf16 fields + cublas bf16 W8A16) — the previous
// default, kept for the parent's A/B (and the only path on a build WITHOUT the
// fp8 cutlass kernel, guarded by VT_CUTLASS_FP8 so we never route to an
// uncompiled GEMM). The dense NVFP4 sinks (shared-expert + lm_head) are already
// native (fp4-resident + Marlin W4A16) and not gated here.
bool DenseNativeEnabled() {
#ifdef VT_CUTLASS_FP8
  const char* dn = std::getenv("VT_DENSE_NATIVE");
  if (dn != nullptr && dn[0] == '0') return false;
  const char* legacy = std::getenv("VT_FP8_CUTLASS");  // back-compat opt-out
  if (legacy != nullptr && legacy[0] == '0') return false;
  return true;
#else
  return false;
#endif
}

// Per-tensor FP8 projection `<proj>.weight` F8_E4M3 [out,in] + `.weight_scale`
// scalar + `.input_scale` scalar -> RAW fp8-resident W8A8 Fp8Weight (no dequant,
// no transpose: the bytes stay in the on-disk [N=out, K=in] orientation the
// cutlass W8A8 GEMM reads directly). `alpha = input_scale * weight_scale` is
// precomputed (per-tensor scalars) — the folded scalar vt::MatmulFp8Cutlass
// multiplies the fp8 accumulator by (mirror vLLM ScaledEpilogue for per-tensor).
Fp8Weight LoadFp8Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "F8_E4M3",
           "qwen3_5 weights: expected F8_E4M3 for " + proj + ".weight");
  VT_CHECK(w.shape.size() == 2,
           "qwen3_5 weights: expected 2-D weight for " + proj);
  Fp8Weight r;
  r.n = w.shape[0];
  r.k = w.shape[1];
  r.weight_scale = ReadF32Scalar(get(proj + ".weight_scale"));
  r.input_scale = ReadF32Scalar(get(proj + ".input_scale"));
  r.alpha = r.input_scale * r.weight_scale;
  r.packed = MakeOwned(vt::DType::kI8, {r.n, r.k});
  VT_CHECK(w.nbytes == r.packed.bytes.size(),
           "qwen3_5 weights: fp8 byte-size mismatch for " + proj);
  std::memcpy(r.packed.bytes.data(), w.data, w.nbytes);
  MaybeReleaseSourcePages(w.data, w.nbytes);
  return r;
}

// Per-tensor FP8 projection `<proj>.weight` [out,in] + `<proj>.weight_scale`
// scalar -> owned bf16 [in, out] (dequant then transpose).
OwnedTensor LoadFp8Transposed(const TensorResolver& get,
                              const std::string& proj) {
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "F8_E4M3",
           "qwen3_5 weights: expected F8_E4M3 for " + proj + ".weight");
  VT_CHECK(w.shape.size() == 2,
           "qwen3_5 weights: expected 2-D weight for " + proj);
  const int64_t out_dim = w.shape[0];
  const int64_t in_dim = w.shape[1];
  const float scale = ReadF32Scalar(get(proj + ".weight_scale"));

  std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
  DequantFp8ToBf16(w.data, scale, out_dim * in_dim, dq.data());
  MaybeReleaseSourcePages(w.data, w.nbytes);

  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(dq.data(), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// NVFP4 W4A16 projection `<proj>.weight` U8 [out,in/2] + `.weight_scale` F8
// [out,in/16] + `.weight_scale_2` f32 scalar -> RAW fp4-resident Nvfp4Weight
// (M2.2b). No dequant, no transpose: the bytes are kept in the on-disk
// [N=out, K=in] orientation vt::MatmulNvfp4 reads directly. This is the
// storage refactor that removes the ~40-min CPU dequant + ~70GB bf16 tensors.
Nvfp4Weight LoadNvfp4Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "U8",
           "qwen3_5 weights: expected U8 for " + proj + ".weight");
  VT_CHECK(w.shape.size() == 2,
           "qwen3_5 weights: expected 2-D packed weight for " + proj);
  const int64_t out_dim = w.shape[0];
  const int64_t in_dim = w.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 weights: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "qwen3_5 weights: expected F8_E4M3 for " + proj + ".weight_scale");
  const float ws2 = ReadF32Scalar(get(proj + ".weight_scale_2"));

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.scale2 = ws2;
  r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  VT_CHECK(w.nbytes == r.packed.bytes.size(),
           "qwen3_5 weights: packed byte-size mismatch for " + proj);
  std::memcpy(r.packed.bytes.data(), w.data, w.nbytes);
  MaybeReleaseSourcePages(w.data, w.nbytes);
  r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  VT_CHECK(ws.nbytes == r.scale.bytes.size(),
           "qwen3_5 weights: scale byte-size mismatch for " + proj);
  std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
  MaybeReleaseSourcePages(ws.data, ws.nbytes);
  return r;
}

GdnLayerWeights LoadGdn(const TensorResolver& get, const std::string& base) {
  const std::string la = base + "linear_attn.";
  GdnLayerWeights g;
  // W8A8 fp8 (35B), DEFAULT: keep raw fp8 + input_scale, run the native fp8 GEMM.
  // VT_DENSE_NATIVE=0 restores dequant-at-load into the bf16 fields (parent A/B).
  if (DenseNativeEnabled()) {
    g.in_proj_qkv_fp8 = LoadFp8Raw(get, la + "in_proj_qkv");
    g.in_proj_z_fp8 = LoadFp8Raw(get, la + "in_proj_z");
    g.out_proj_fp8 = LoadFp8Raw(get, la + "out_proj");
  } else {
    g.in_proj_qkv = LoadFp8Transposed(get, la + "in_proj_qkv");
    g.in_proj_z = LoadFp8Transposed(get, la + "in_proj_z");
    g.out_proj = LoadFp8Transposed(get, la + "out_proj");
  }
  g.in_proj_b = LoadBf16Transposed(get, la + "in_proj_b.weight");
  g.in_proj_a = LoadBf16Transposed(get, la + "in_proj_a.weight");
  // conv1d.weight ships [conv_dim,1,K]; collapse the singleton to [conv_dim,K].
  const StTensor& conv = get(la + "conv1d.weight");
  VT_CHECK(conv.shape.size() == 3 && conv.shape[1] == 1,
           "qwen3_5 weights: unexpected conv1d shape");
  g.conv1d_weight =
      LoadBf16Direct(get, la + "conv1d.weight", {conv.shape[0], conv.shape[2]});
  g.a_log = LoadBf16ToF32(get, la + "A_log");
  g.dt_bias = LoadBf16ToF32(get, la + "dt_bias");
  g.norm_weight = LoadBf16Direct(get, la + "norm.weight");
  return g;
}

FullAttnLayerWeights LoadAttn(const TensorResolver& get,
                              const std::string& base) {
  const std::string sa = base + "self_attn.";
  FullAttnLayerWeights a;
  // W8A8 fp8 (35B), DEFAULT: keep raw fp8 + input_scale, run the native fp8 GEMM.
  // VT_DENSE_NATIVE=0 restores dequant-at-load into the bf16 fields (parent A/B).
  if (DenseNativeEnabled()) {
    a.q_proj_fp8 = LoadFp8Raw(get, sa + "q_proj");
    a.k_proj_fp8 = LoadFp8Raw(get, sa + "k_proj");
    a.v_proj_fp8 = LoadFp8Raw(get, sa + "v_proj");
    a.o_proj_fp8 = LoadFp8Raw(get, sa + "o_proj");
  } else {
    a.q_proj = LoadFp8Transposed(get, sa + "q_proj");
    a.k_proj = LoadFp8Transposed(get, sa + "k_proj");
    a.v_proj = LoadFp8Transposed(get, sa + "v_proj");
    a.o_proj = LoadFp8Transposed(get, sa + "o_proj");
  }
  a.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  a.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");
  return a;
}

// The routed-expert host copies (E × gate/up/down NVFP4, MakeOwned+memcpy). Split
// out so the 35B load-phase peak-PSS interleave can materialize EXACTLY ONE
// layer's experts just before the device Marlin build + host free (the
// `load_layer_experts` closure in LoadQwen3_5Moe), instead of all N layers'
// experts coexisting on the host.
void LoadMoeExpertsInto(const TensorResolver& get, const std::string& mlp,
                        int64_t num_experts, MoeBlockWeights& m) {
  m.expert_gate_fp4.reserve(static_cast<size_t>(num_experts));
  m.expert_up_fp4.reserve(static_cast<size_t>(num_experts));
  m.expert_down_fp4.reserve(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    const std::string ex = mlp + "experts." + std::to_string(e) + ".";
    m.expert_gate_fp4.push_back(LoadNvfp4Raw(get, ex + "gate_proj"));
    m.expert_up_fp4.push_back(LoadNvfp4Raw(get, ex + "up_proj"));
    m.expert_down_fp4.push_back(LoadNvfp4Raw(get, ex + "down_proj"));
  }
}

MoeBlockWeights LoadMoe(const TensorResolver& get, const std::string& base,
                        int64_t num_experts, bool with_experts) {
  const std::string mlp = base + "mlp.";
  MoeBlockWeights m;
  m.router_gate = LoadBf16Transposed(get, mlp + "gate.weight");
  m.shared_gate = LoadBf16Transposed(get, mlp + "shared_expert_gate.weight");
  // M2.2b: the NVFP4 expert + shared projections are kept fp4-resident (raw
  // packed + scales, no dequant/transpose); the bf16 expert_*/shared_*_proj
  // fields stay EMPTY. The forward calls vt::MatmulNvfp4 on the fp4 fields.
  // with_experts=false DEFERS the routed-expert loop (the ~16.9 GiB host
  // consumer) to the per-layer streaming path; the small shared projections are
  // always loaded eagerly (retained through the device build).
  if (with_experts) LoadMoeExpertsInto(get, mlp, num_experts, m);
  const std::string se = mlp + "shared_expert.";
  m.shared_gate_proj_fp4 = LoadNvfp4Raw(get, se + "gate_proj");
  m.shared_up_proj_fp4 = LoadNvfp4Raw(get, se + "up_proj");
  m.shared_down_proj_fp4 = LoadNvfp4Raw(get, se + "down_proj");
  return m;
}

// Full decoder layer minus (optionally) the routed experts. Shared by the public
// LoadQwen3_5MoeLayer (with_experts=true) and the deferred streaming load.
Qwen3_5MoeLayerWeights LoadLayerImpl(const TensorResolver& get,
                                     const std::string& layer_type,
                                     int64_t layer_idx, int64_t num_experts,
                                     bool with_experts,
                                     const std::string& backbone_prefix) {
  const std::string base =
      backbone_prefix + "layers." + std::to_string(layer_idx) + ".";
  Qwen3_5MoeLayerWeights layer;
  layer.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  layer.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  if (layer_type == "linear_attention") {
    layer.is_linear_attention = true;
    layer.gdn = LoadGdn(get, base);
  } else if (layer_type == "full_attention") {
    layer.is_linear_attention = false;
    layer.attn = LoadAttn(get, base);
  } else {
    VT_CHECK(false, "qwen3_5 weights: unknown layer_type " + layer_type);
  }
  layer.moe = LoadMoe(get, base, num_experts, with_experts);
  return layer;
}

// True iff any name in `names` is a backbone tensor under `prefix`. Only the
// three structural backbone spellings vote (see qwen3_5_weights.h): the
// vision tower (`model.visual.*`) and the top-level `lm_head.*` / `mtp.*` are
// deliberately NOT backbone names, so they cannot decide the namespace.
bool HasBackboneUnder(const std::vector<std::string>& names,
                      std::string_view prefix) {
  const std::string embed = std::string(prefix) + "embed_tokens.weight";
  const std::string norm = std::string(prefix) + "norm.weight";
  const std::string layers = std::string(prefix) + "layers.";
  for (const std::string& name : names) {
    if (name == embed || name == norm) return true;
    if (name.compare(0, layers.size(), layers) == 0) return true;
  }
  return false;
}

// --- Unimplemented MoE expert arms, REFUSED BY NAME (issue #490) -------------
//
// `LoadMoeExpertsInto` above reads exactly ONE routed-expert layout: per-expert
// NVFP4 (`...mlp.experts.<e>.{gate,up,down}_proj` through `LoadNvfp4Raw`, which
// hard-requires a `U8` `.weight`, an `F8_E4M3` `.weight_scale` and a
// `.weight_scale_2`). There is no stacked branch and no bf16 branch — unlike
// `gemma4_weights.cpp:326`, which dispatches between layouts.
//
// The PUBLISHED Qwen3.5-family MoE repos do not have that layout. Read off the
// live safetensors indices 2026-08-12: `Qwen/Qwen3.8-2.4T-A95B` has 93x
// `mlp.experts.gate_up_proj` + 93x `.down_proj` (3-D STACKED) and ZERO
// `weight_scale` / `input_scale` tensors; `Qwen/Qwen3.6-35B-A3B` is the same
// under the VL prefix. Our gated 35B row reads the REQUANTIZED
// `nvidia/Qwen3.6-35B-A3B-NVFP4`, so this loader has never read a published
// Qwen bf16 MoE checkpoint. Left alone, such a load dies at
// `LoadNvfp4Raw(get, "lm_head")` with "expected U8 for lm_head.weight" — which
// reads as a corrupt checkpoint, not as an unimplemented arm.
//
// AGENTS.md: an arm that is not implemented "is refused with a message naming
// the missing piece ... never left to be discovered later", and the row's spec
// (.agents/specs/qwen38-text-only.md) says the same in its stop conditions.
// This is that refusal and ONLY that: the stacked/bf16 MoE expert arm is OWED,
// and implementing it needs its own spec, RED-first test and NVFP4 inertness
// proof. Inert on the supported layout — every name it inspects already has to
// exist for the load to succeed at all.
void CheckMoeExpertLayoutSupported(const std::vector<std::string>& names,
                                   const std::string& backbone) {
  static const std::string kRequired =
      " This loader implements only the per-expert NVFP4 layout: "
      "<layer>.mlp.experts.<e>.{gate,up,down}_proj.weight (U8 packed) + "
      ".weight_scale (F8_E4M3) + .weight_scale_2, and lm_head the same way. The "
      "published bf16 repos (Qwen/Qwen3.8-2.4T-A95B, Qwen/Qwen3.6-35B-A3B) ship "
      "the 3-D stacked, unquantized layout; an NVFP4 requant (e.g. "
      "nvidia/Qwen3.6-35B-A3B-NVFP4) ships the supported one. The stacked and "
      "unquantized MoE expert arms are OWED, not silently unsupported: see "
      ".agents/specs/qwen38-text-only.md.";
  const std::string layers = backbone + "layers.";
  const std::string experts = ".mlp.experts.";
  const std::string weight = ".weight";
  const std::unordered_set<std::string> present(names.begin(), names.end());
  for (const std::string& name : names) {
    if (name.compare(0, layers.size(), layers) != 0) continue;
    const size_t at = name.find(experts);
    if (at == std::string::npos) continue;
    const size_t rest = at + experts.size();
    if (rest >= name.size()) continue;
    // `experts.<digit>` is the per-expert spelling; anything else — the
    // published `experts.gate_up_proj` / `experts.down_proj` — is the stacked
    // one, where a single 3-D tensor holds every expert.
    if (std::isdigit(static_cast<unsigned char>(name[rest])) == 0) {
      VT_CHECK(false,
               "qwen3_5 weights: 3-D stacked routed experts are not implemented "
               "for the safetensors MoE arm -- found \"" +
                   name + "\"." + kRequired);
    }
    if (name.size() > weight.size() &&
        name.compare(name.size() - weight.size(), weight.size(), weight) == 0 &&
        present.count(name + "_scale") == 0) {
      VT_CHECK(false,
               "qwen3_5 weights: unquantized routed experts are not implemented "
               "for the safetensors MoE arm -- \"" +
                   name + "\" has no \"" + name + "_scale\" beside it." +
                   kRequired);
    }
  }
  // The MoE head is likewise NVFP4-only here, where the DENSE loader routes a
  // head by dtype (`LoadDenseLmHead` / `LoadLmHeadAnyDtype`). A checkpoint with
  // no `lm_head.weight` at all is the tied-head case and is not this refusal.
  if (present.count("lm_head.weight") != 0 &&
      present.count("lm_head.weight_scale") == 0) {
    VT_CHECK(false,
             "qwen3_5 weights: an unquantized lm_head is not implemented for "
             "the safetensors MoE arm -- \"lm_head.weight\" has no "
             "\"lm_head.weight_scale\" beside it." +
                 kRequired);
  }
}

}  // namespace

std::string ResolveQwen3_5BackbonePrefix(
    const std::vector<std::string>& tensor_names) {
  // `model.language_model.` is tested FIRST because it is also a `model.`
  // name: a plain "starts with model." test would match both spellings.
  const bool vl = HasBackboneUnder(tensor_names, kQwen3_5VlBackbonePrefix);
  // ...so the canonical probe must EXCLUDE the VL-prefixed names, which the
  // backbone spellings above already do (`model.language_model.` is neither
  // `model.embed_tokens.weight`, nor `model.norm.weight`, nor `model.layers.`).
  const bool flat = HasBackboneUnder(tensor_names, kQwen3_5TextBackbonePrefix);
  VT_CHECK(!(vl && flat),
           "qwen3_5 weights: checkpoint carries backbone tensors under BOTH "
           "\"model.language_model.\" and \"model.\"; refusing a mixed weight "
           "namespace rather than binding half the model from each");
  VT_CHECK(vl || flat,
           "qwen3_5 weights: no Qwen3.5 backbone tensors found under either "
           "\"model.language_model.\" or \"model.\"");
  return std::string(vl ? kQwen3_5VlBackbonePrefix
                        : kQwen3_5TextBackbonePrefix);
}

// External-linkage seam so the DENSE loader can keep an FP8 GDN tower native.
Fp8Weight LoadFp8RawShared(const TensorResolver& get, const std::string& proj) {
  return LoadFp8Raw(get, proj);
}

Qwen3_5MoeLayerWeights LoadQwen3_5MoeLayer(const TensorResolver& get,
                                           const std::string& layer_type,
                                           int64_t layer_idx,
                                           int64_t num_experts,
                                           const std::string& backbone_prefix) {
  return LoadLayerImpl(get, layer_type, layer_idx, num_experts,
                       /*with_experts=*/true, backbone_prefix);
}

Qwen3_5MoeWeights LoadQwen3_5Moe(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    std::shared_ptr<const std::vector<SafetensorsFile>> shards_owner) {
  // Build a name -> shard index from each shard's own header. The target loader
  // does not request mtp.*: LoadQwen3_5MTP loads that optional draft head only
  // when speculative decoding is enabled. The resolver throws on missing names.
  // The map is heap-owned (shared_ptr) so the deferred per-layer expert loader
  // below can reuse it without rebuilding — its entries point into `shards`,
  // which `shards_owner` keeps mmap'd.
  auto where =
      std::make_shared<std::unordered_map<std::string, const SafetensorsFile*>>();
  std::vector<std::string> all_names;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      (*where)[name] = &shard;
      all_names.push_back(name);
    }
  }
  // ONE namespace decision for the whole checkpoint (qwen3_5_weights.h): the
  // VL-nested spelling for the wrappers we gate, the flat `model.` spelling for
  // a text-only arm, and a refusal for a mixed index.
  const std::string backbone = ResolveQwen3_5BackbonePrefix(all_names);
  // ...and ONE decision about the routed-expert layout, before any tensor is
  // touched, so an arm we do not implement is refused by name rather than
  // discovered as a dtype complaint about `lm_head` (issue #490).
  CheckMoeExpertLayoutSupported(all_names, backbone);
  const TensorResolver get =
      [where](const std::string& name) -> const StTensor& {
    auto it = where->find(name);
    VT_CHECK(it != where->end(), "qwen3_5 weights: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 weights: layer_types size must equal num_hidden_layers");
  VT_CHECK(config.num_experts > 0,
           "qwen3_5 weights: num_experts must be > 0 for the MoE model");

  // With a shared owner we DEFER the routed-expert host copies: load every layer
  // WITHOUT its experts now (small: norms, attn/GDN, router, shared expert), and
  // install a closure that materializes ONE layer's experts on demand during
  // PrepareMarlinResident — bounding peak host residency to a single layer's
  // ~256 experts instead of all N layers coexisting (the 35B load-phase peak-PSS
  // lever). Without an owner the shards may be released right after this returns,
  // so the experts must be loaded eagerly (GGUF/synthetic/borrowed paths).
  const bool defer_experts = shards_owner != nullptr;

  Qwen3_5MoeWeights w;
  w.embed_tokens = LoadBf16Direct(get, backbone + "embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, backbone + "norm.weight");
  w.lm_head_fp4 = LoadNvfp4Raw(get, "lm_head");  // M2.2b fp4-resident
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    w.layers.push_back(LoadLayerImpl(get,
                                     config.layer_types[static_cast<size_t>(l)],
                                     l, config.num_experts,
                                     /*with_experts=*/!defer_experts,
                                     backbone));
  }

  if (defer_experts) {
    const int64_t num_experts = config.num_experts;
    // Captures the shared name-map AND the shards owner (keepalive): the mmap'd
    // shards stay valid until PrepareMarlinResident resets this closure after the
    // last layer is built. Does NOT capture the (movable) Qwen3_5MoeWeights — the
    // target MoE block is passed in by reference, so the closure survives the
    // model's move into the LoadedModel.
    // `backbone` is captured BY VALUE: the closure outlives this frame, and it
    // must keep using the ONE namespace resolved above rather than re-deciding.
    w.load_layer_experts = [where, shards_owner, num_experts, backbone](
                               int64_t layer, MoeBlockWeights& moe) {
      const TensorResolver g =
          [where](const std::string& name) -> const StTensor& {
        auto it = where->find(name);
        VT_CHECK(it != where->end(),
                 "qwen3_5 weights: tensor not found: " + name);
        return it->second->Get(name);
      };
      const std::string mlp =
          backbone + "layers." + std::to_string(layer) + ".mlp.";
      LoadMoeExpertsInto(g, mlp, num_experts, moe);
    };
  }
  return w;
}

}  // namespace vllm
