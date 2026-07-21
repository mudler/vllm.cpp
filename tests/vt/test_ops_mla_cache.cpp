// vt::ConcatAndCacheMla unit tests (MLA campaign W3).
//
// Upstream test module ported per .agents/test-porting.md:
//   vllm/tests/kernels/attention/test_cache.py::test_concat_and_cache_mla @ pin
//   e24d1b24 — which builds a reference by writing kv_c into
//   `ref_kv_cache[block, offset, :kv_lora_rank]` and k_pe into
//   `[..., kv_lora_rank:]` and comparing against the op. That is exactly the
//   round-trip spec asserted below, re-expressed on our tensors.
// Kernel under port: vllm/csrc/libtorch_stable/cache_kernels.cu:401-442
//   (`concat_and_cache_mla_kernel`) + host wrapper `:842-905`.
//
// Golden strategy: NO external oracle (the op is a pure strided copy). The spec
// is a layout-consistent WRITE→READ round-trip on the 3-D MLA cache
// (num_blocks, block_size, kv_lora_rank + qk_rope_head_dim), then CUDA-vs-CPU
// parity guarded by HasCuda.
//
// THE MLA-SPECIFIC PROPERTY under test, and the one ReshapeAndCache cannot
// express: the two sources are CONCATENATED into ONE cache entry (no K/V axis,
// no second tensor), the latent at columns [0, kv_lora_rank) and the decoupled
// rope part at [kv_lora_rank, kv_lora_rank + pe_dim).
#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

template <typename T>
Tensor StridedView(T* data, DType dt, Device dev, int64_t off_elems,
                   const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
  Tensor t;
  t.data = data + off_elems;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride[static_cast<size_t>(i)];
  }
  return t;
}

// Flat element offset of the ENTRY for `slot` in a contiguous 3-D MLA cache.
int64_t EntryIdx(int64_t slot, int64_t block_size, int64_t entry_width) {
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  return (block * block_size + offset) * entry_width;
}

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = Contig(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

}  // namespace

TEST_CASE("concat_and_cache_mla concatenates latent + rope into one entry") {
  // Small, hand-checkable: block_size=4, kv_lora_rank=3, pe_dim=2 → entry 5.
  const int64_t num_blocks = 2, block_size = 4, lora = 3, pe = 2;
  const int64_t width = lora + pe;
  std::vector<float> kv_c = {1.0f, 2.0f, 3.0f};
  std::vector<float> k_pe = {7.0f, 8.0f};
  std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * width), -99.0f);
  std::vector<int64_t> slots = {5};  // block 1, offset 1

  Tensor tc = Contig(kv_c.data(), DType::kF32, Cpu(), {1, lora});
  Tensor tp = Contig(k_pe.data(), DType::kF32, Cpu(), {1, pe});
  Tensor tcache = Contig(cache.data(), DType::kF32, Cpu(), {num_blocks, block_size, width});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {1});
  Queue qq = Q();
  vt::ConcatAndCacheMla(qq, tc, tp, tcache, ts);

  const size_t base = static_cast<size_t>(EntryIdx(5, block_size, width));
  // Latent occupies [0, lora); rope occupies [lora, lora+pe). THE concatenation.
  CHECK(cache[base + 0] == doctest::Approx(1.0f));
  CHECK(cache[base + 1] == doctest::Approx(2.0f));
  CHECK(cache[base + 2] == doctest::Approx(3.0f));
  CHECK(cache[base + 3] == doctest::Approx(7.0f));
  CHECK(cache[base + 4] == doctest::Approx(8.0f));
  // Neighbouring entries untouched.
  const size_t other = static_cast<size_t>(EntryIdx(4, block_size, width));
  for (int64_t i = 0; i < width; ++i) {
    CHECK(cache[other + static_cast<size_t>(i)] == doctest::Approx(-99.0f));
  }
}

TEST_CASE("concat_and_cache_mla at DeepSeek-V2-Lite's REAL geometry") {
  // W0-confirmed config: kv_lora_rank=512, qk_rope_head_dim=64 → a 576-wide
  // entry, num_kv_heads=1, block_size=16. The page is 16*1*576*2 == 18,432 B
  // per block per layer at bf16 — the number the W1 MLAAttentionSpec asserts.
  const int64_t num_blocks = 3, block_size = 16, lora = 512, pe = 64;
  const int64_t width = lora + pe;
  CHECK(width == 576);
  const int64_t nt = 5;
  auto kv_c = RandF32(static_cast<size_t>(nt * lora), 11);
  auto k_pe = RandF32(static_cast<size_t>(nt * pe), 22);
  std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * width), -1.0f);
  std::vector<int64_t> slots = {0, 17, 47, 5, 32};  // spans all three blocks

  Tensor tc = Contig(kv_c.data(), DType::kF32, Cpu(), {nt, lora});
  Tensor tp = Contig(k_pe.data(), DType::kF32, Cpu(), {nt, pe});
  Tensor tcache = Contig(cache.data(), DType::kF32, Cpu(), {num_blocks, block_size, width});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {nt});
  Queue qq = Q();
  vt::ConcatAndCacheMla(qq, tc, tp, tcache, ts);

  for (int64_t t = 0; t < nt; ++t) {
    const size_t base = static_cast<size_t>(EntryIdx(slots[static_cast<size_t>(t)],
                                                     block_size, width));
    for (int64_t i = 0; i < lora; ++i) {
      CHECK(cache[base + static_cast<size_t>(i)] ==
            kv_c[static_cast<size_t>(t * lora + i)]);
    }
    for (int64_t i = 0; i < pe; ++i) {
      CHECK(cache[base + static_cast<size_t>(lora + i)] ==
            k_pe[static_cast<size_t>(t * pe + i)]);
    }
  }
}

TEST_CASE("concat_and_cache_mla skips slot == -1 and trailing padded rows") {
  // Upstream cache_kernels.cu:419-422 (slot < 0 → return) and :855-863
  // (slot_mapping.size(0) is the token count; kv_c/k_pe may be longer).
  const int64_t num_blocks = 2, block_size = 2, lora = 2, pe = 1;
  const int64_t width = lora + pe;
  std::vector<float> kv_c = {1, 1, 2, 2, 3, 3, 9, 9};  // 4 rows, last is padding
  std::vector<float> k_pe = {10, 20, 30, 90};
  std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * width), 0.0f);
  std::vector<int64_t> slots = {0, -1, 3};  // token 1 padded; token 3 not mapped

  Tensor tc = Contig(kv_c.data(), DType::kF32, Cpu(), {4, lora});
  Tensor tp = Contig(k_pe.data(), DType::kF32, Cpu(), {4, pe});
  Tensor tcache = Contig(cache.data(), DType::kF32, Cpu(), {num_blocks, block_size, width});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {3});
  Queue qq = Q();
  vt::ConcatAndCacheMla(qq, tc, tp, tcache, ts);

  const size_t s0 = static_cast<size_t>(EntryIdx(0, block_size, width));
  const size_t s3 = static_cast<size_t>(EntryIdx(3, block_size, width));
  CHECK(cache[s0 + 0] == doctest::Approx(1.0f));
  CHECK(cache[s0 + 2] == doctest::Approx(10.0f));
  CHECK(cache[s3 + 0] == doctest::Approx(3.0f));
  CHECK(cache[s3 + 2] == doctest::Approx(30.0f));
  // Slots 1 and 2 never written — the padded token left nothing behind, and the
  // trailing (unmapped) source row 3 (value 9 / 90) appears nowhere.
  for (int64_t slot : {1, 2}) {
    const size_t b = static_cast<size_t>(EntryIdx(slot, block_size, width));
    for (int64_t i = 0; i < width; ++i) {
      CHECK(cache[b + static_cast<size_t>(i)] == doctest::Approx(0.0f));
    }
  }
}

TEST_CASE("concat_and_cache_mla bf16 write is bit-exact") {
  auto bf = [](float x) { return vt::F32ToBF16(x); };
  const int64_t num_blocks = 1, block_size = 2, lora = 2, pe = 2;
  const int64_t width = lora + pe;
  std::vector<uint16_t> kv_c = {bf(1.5f), bf(-2.25f)};
  std::vector<uint16_t> k_pe = {bf(3.75f), bf(-4.5f)};
  std::vector<uint16_t> cache(static_cast<size_t>(num_blocks * block_size * width), 0);
  std::vector<int64_t> slots = {1};

  Tensor tc = Contig(kv_c.data(), DType::kBF16, Cpu(), {1, lora});
  Tensor tp = Contig(k_pe.data(), DType::kBF16, Cpu(), {1, pe});
  Tensor tcache = Contig(cache.data(), DType::kBF16, Cpu(), {num_blocks, block_size, width});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {1});
  Queue qq = Q();
  vt::ConcatAndCacheMla(qq, tc, tp, tcache, ts);

  const size_t base = static_cast<size_t>(EntryIdx(1, block_size, width));
  CHECK(cache[base + 0] == kv_c[0]);  // raw copy → identical bit patterns
  CHECK(cache[base + 1] == kv_c[1]);
  CHECK(cache[base + 2] == k_pe[0]);
  CHECK(cache[base + 3] == k_pe[1]);
}

TEST_CASE("concat_and_cache_mla is STRIDE-driven (cache view + split sources)") {
  // Upstream sources block_stride/entry_stride from kv_cache.stride(0)/stride(1)
  // and the token strides from kv_c/k_pe.stride(0) (cache_kernels.cu:882-884).
  // Two real consequences, both exercised here:
  //   * the MLA cache may be a per-LAYER slice of one multi-layer allocation, so
  //     block_stride is NOT block_size*width;
  //   * kv_c and k_pe are the two column halves of the single
  //     `kv_a_proj_with_mqa` output [T, 576] (deepseek_v2.py:511), so each is a
  //     row-strided view, NOT a materialized copy.
  const int64_t num_blocks = 2, block_size = 2, lora = 3, pe = 2;
  const int64_t width = lora + pe;
  const int64_t layers = 2;
  // One buffer for BOTH layers: [num_blocks, layers, block_size, width].
  const int64_t entry_stride = width;
  const int64_t block_stride = layers * block_size * width;  // NOT block_size*width
  const size_t buf_n = static_cast<size_t>(num_blocks * layers * block_size * width);
  std::vector<float> buf(buf_n, -5.0f);
  // Layer 1's view starts one layer in.
  Tensor tcache = StridedView(buf.data(), DType::kF32, Cpu(), block_size * width,
                              {num_blocks, block_size, width},
                              {block_stride, entry_stride, 1});

  // The fused [T, 576] projection output, sliced into its two halves.
  const int64_t nt = 3;
  std::vector<float> proj(static_cast<size_t>(nt * width));
  for (size_t i = 0; i < proj.size(); ++i) proj[i] = static_cast<float>(i) + 0.5f;
  Tensor tc = StridedView(proj.data(), DType::kF32, Cpu(), 0, {nt, lora}, {width, 1});
  Tensor tp = StridedView(proj.data(), DType::kF32, Cpu(), lora, {nt, pe}, {width, 1});

  std::vector<int64_t> slots = {0, 3, 1};
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {nt});
  Queue qq = Q();
  vt::ConcatAndCacheMla(qq, tc, tp, tcache, ts);

  // Each token's whole 576-wide row is reassembled contiguously in layer 1.
  for (int64_t t = 0; t < nt; ++t) {
    const int64_t slot = slots[static_cast<size_t>(t)];
    const int64_t blk = slot / block_size, off = slot % block_size;
    const size_t base =
        static_cast<size_t>(block_size * width + blk * block_stride + off * entry_stride);
    for (int64_t i = 0; i < width; ++i) {
      CHECK(buf[base + static_cast<size_t>(i)] ==
            doctest::Approx(proj[static_cast<size_t>(t * width + i)]));
    }
  }
  // Layer 0's region is completely untouched — a shape-derived block_stride
  // (block_size*width) would have written into it.
  for (int64_t blk = 0; blk < num_blocks; ++blk) {
    for (int64_t i = 0; i < block_size * width; ++i) {
      CHECK(buf[static_cast<size_t>(blk * block_stride + i)] == doctest::Approx(-5.0f));
    }
  }
}

TEST_CASE("concat_and_cache_mla validates shapes/dtypes") {
  std::vector<float> buf(256, 0.0f);
  std::vector<int64_t> slots = {0};
  std::vector<int32_t> slots32 = {0};
  Queue qq = Q();

  Tensor tc = Contig(buf.data(), DType::kF32, Cpu(), {1, 3});
  Tensor tp = Contig(buf.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tcache = Contig(buf.data(), DType::kF32, Cpu(), {2, 2, 5});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {1});
  CHECK_NOTHROW(vt::ConcatAndCacheMla(qq, tc, tp, tcache, ts));

  // Entry width must equal kv_lora_rank + pe_dim (cache_kernels.cu:876).
  {
    Tensor bad = Contig(buf.data(), DType::kF32, Cpu(), {2, 2, 6});
    CHECK_THROWS_AS(vt::ConcatAndCacheMla(qq, tc, tp, bad, ts), std::runtime_error);
  }
  // The cache is rank-3: MLA has NO K/V axis. A rank-4 (flash-shaped) cache is
  // refused rather than silently reinterpreted.
  {
    Tensor bad = Contig(buf.data(), DType::kF32, Cpu(), {2, 2, 1, 5});
    CHECK_THROWS_AS(vt::ConcatAndCacheMla(qq, tc, tp, bad, ts), std::runtime_error);
  }
  // slot_mapping must be i64.
  {
    Tensor bad = Contig(slots32.data(), DType::kI32, Cpu(), {1});
    CHECK_THROWS_AS(vt::ConcatAndCacheMla(qq, tc, tp, tcache, bad), std::runtime_error);
  }
  // kv_c and k_pe must share num_tokens.
  {
    Tensor bad = Contig(buf.data(), DType::kF32, Cpu(), {2, 2});
    CHECK_THROWS_AS(vt::ConcatAndCacheMla(qq, tc, bad, tcache, ts), std::runtime_error);
  }
  // Mixed dtypes: the "auto" path only. fp8_ds_mla is out of scope and must
  // throw loudly rather than mis-size the 656-byte layout.
  {
    std::vector<uint16_t> half(64, 0);
    Tensor bad = Contig(half.data(), DType::kBF16, Cpu(), {2, 2, 5});
    CHECK_THROWS_AS(vt::ConcatAndCacheMla(qq, tc, tp, bad, ts), std::runtime_error);
  }
  // num_tokens smaller than slot_mapping length.
  {
    std::vector<int64_t> two = {0, 1};
    Tensor ts2 = Contig(two.data(), DType::kI64, Cpu(), {2});
    CHECK_THROWS_AS(vt::ConcatAndCacheMla(qq, tc, tp, tcache, ts2), std::runtime_error);
  }
}

// ===========================================================================
// CUDA parity: byte-identical cache to the CPU reference. The op is a pure copy
// with no reduction, so equality is EXACT, not toleranced.
TEST_CASE("concat_and_cache_mla CUDA matches CPU at V2-Lite geometry") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping concat_and_cache_mla parity");
    return;
  }
  const int64_t num_blocks = 4, block_size = 16, lora = 512, pe = 64;
  const int64_t width = lora + pe;
  const int64_t nt = 10;
  auto kv_c = RandF32(static_cast<size_t>(nt * lora), 7);
  auto k_pe = RandF32(static_cast<size_t>(nt * pe), 99);
  std::vector<int64_t> slots = {0, 9, 17, -1, 8, 63, 3, -1, 24, 15};

  const size_t cache_n = static_cast<size_t>(num_blocks * block_size * width);
  std::vector<float> cache_cpu(cache_n, -7.0f);
  Tensor tc = Contig(kv_c.data(), DType::kF32, Cpu(), {nt, lora});
  Tensor tp = Contig(k_pe.data(), DType::kF32, Cpu(), {nt, pe});
  Tensor tcache = Contig(cache_cpu.data(), DType::kF32, Cpu(),
                         {num_blocks, block_size, width});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(),
                     {static_cast<int64_t>(slots.size())});
  Queue cpuq = Q();
  vt::ConcatAndCacheMla(cpuq, tc, tp, tcache, ts);

  std::vector<float> cache_init(cache_n, -7.0f);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dc(gpu, g.q, DType::kF32, {nt, lora}, kv_c.data());
  DeviceTensor dp(gpu, g.q, DType::kF32, {nt, pe}, k_pe.data());
  DeviceTensor dcache(gpu, g.q, DType::kF32, {num_blocks, block_size, width},
                      cache_init.data());
  DeviceTensor ds(gpu, g.q, DType::kI64, {static_cast<int64_t>(slots.size())},
                  slots.data());
  vt::ConcatAndCacheMla(g.q, dc.tensor(), dp.tensor(), dcache.tensor(), ds.tensor());

  std::vector<float> cache_got(cache_n, 0.0f);
  dcache.Download(g.q, cache_got.data());
  CHECK(cache_got == cache_cpu);  // EXACT — a pure copy has no rounding
}

TEST_CASE("concat_and_cache_mla CUDA matches CPU on strided cache + split sources") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping concat_and_cache_mla strided parity");
    return;
  }
  const int64_t num_blocks = 4, block_size = 16, lora = 512, pe = 64, layers = 2;
  const int64_t width = lora + pe;
  const int64_t entry_stride = width;
  const int64_t block_stride = layers * block_size * width;
  const size_t buf_n = static_cast<size_t>(num_blocks * layers * block_size * width);
  const int64_t nt = 10;
  std::vector<float> proj = RandF32(static_cast<size_t>(nt * width), 31);
  std::vector<int64_t> slots = {0, 9, 17, -1, 8, 63, 3, -1, 24, 15};
  const std::vector<int64_t> cshape = {num_blocks, block_size, width};
  const std::vector<int64_t> cstride = {block_stride, entry_stride, 1};
  const int64_t layer_off = block_size * width;

  std::vector<float> buf_cpu(buf_n, -7.0f);
  Tensor tcache = StridedView(buf_cpu.data(), DType::kF32, Cpu(), layer_off, cshape, cstride);
  Tensor tc = StridedView(proj.data(), DType::kF32, Cpu(), 0, {nt, lora}, {width, 1});
  Tensor tp = StridedView(proj.data(), DType::kF32, Cpu(), lora, {nt, pe}, {width, 1});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(),
                     {static_cast<int64_t>(slots.size())});
  Queue cpuq = Q();
  vt::ConcatAndCacheMla(cpuq, tc, tp, tcache, ts);

  std::vector<float> buf_init(buf_n, -7.0f);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  void* dbuf = gpu.Alloc(buf_n * sizeof(float));
  gpu.Copy(g.q, dbuf, buf_init.data(), buf_n * sizeof(float));
  Tensor dcache = StridedView(static_cast<float*>(dbuf), DType::kF32, Gpu(), layer_off,
                              cshape, cstride);
  DeviceTensor dproj(gpu, g.q, DType::kF32, {nt, width}, proj.data());
  auto* proj_dev = static_cast<float*>(dproj.tensor().data);
  Tensor dc = StridedView(proj_dev, DType::kF32, Gpu(), 0, {nt, lora}, {width, 1});
  Tensor dp = StridedView(proj_dev, DType::kF32, Gpu(), lora, {nt, pe}, {width, 1});
  DeviceTensor ds(gpu, g.q, DType::kI64, {static_cast<int64_t>(slots.size())},
                  slots.data());
  vt::ConcatAndCacheMla(g.q, dc, dp, dcache, ds.tensor());

  std::vector<float> buf_got(buf_n, 0.0f);
  gpu.Copy(g.q, buf_got.data(), dbuf, buf_n * sizeof(float));
  gpu.Synchronize(g.q);
  gpu.Free(dbuf);
  CHECK(buf_got == buf_cpu);
}

TEST_CASE("concat_and_cache_mla CUDA bf16 is bit-exact vs CPU") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping concat_and_cache_mla bf16 parity");
    return;
  }
  // bf16 is the production MLA cache dtype (W0: torch_dtype=bfloat16).
  const int64_t num_blocks = 3, block_size = 16, lora = 512, pe = 64;
  const int64_t width = lora + pe;
  const int64_t nt = 7;
  const auto cf = RandF32(static_cast<size_t>(nt * lora), 5);
  const auto pf = RandF32(static_cast<size_t>(nt * pe), 6);
  std::vector<uint16_t> kv_c(cf.size()), k_pe(pf.size());
  for (size_t i = 0; i < cf.size(); ++i) kv_c[i] = vt::F32ToBF16(cf[i]);
  for (size_t i = 0; i < pf.size(); ++i) k_pe[i] = vt::F32ToBF16(pf[i]);
  std::vector<int64_t> slots = {0, 16, 33, -1, 47, 2, 20};

  const size_t cache_n = static_cast<size_t>(num_blocks * block_size * width);
  std::vector<uint16_t> cache_cpu(cache_n, 0x1234);
  Tensor tc = Contig(kv_c.data(), DType::kBF16, Cpu(), {nt, lora});
  Tensor tp = Contig(k_pe.data(), DType::kBF16, Cpu(), {nt, pe});
  Tensor tcache = Contig(cache_cpu.data(), DType::kBF16, Cpu(),
                         {num_blocks, block_size, width});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(),
                     {static_cast<int64_t>(slots.size())});
  Queue cpuq = Q();
  vt::ConcatAndCacheMla(cpuq, tc, tp, tcache, ts);

  std::vector<uint16_t> cache_init(cache_n, 0x1234);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dc(gpu, g.q, DType::kBF16, {nt, lora}, kv_c.data());
  DeviceTensor dp(gpu, g.q, DType::kBF16, {nt, pe}, k_pe.data());
  DeviceTensor dcache(gpu, g.q, DType::kBF16, {num_blocks, block_size, width},
                      cache_init.data());
  DeviceTensor ds(gpu, g.q, DType::kI64, {static_cast<int64_t>(slots.size())},
                  slots.data());
  vt::ConcatAndCacheMla(g.q, dc.tensor(), dp.tensor(), dcache.tensor(), ds.tensor());

  std::vector<uint16_t> cache_got(cache_n, 0);
  dcache.Download(g.q, cache_got.data());
  CHECK(cache_got == cache_cpu);  // bit patterns, not decoded floats
}
