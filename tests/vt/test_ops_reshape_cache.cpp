// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// reshape_and_cache op unit tests. Write semantics ported from
// vllm/csrc/.../cache_kernels.cu::reshape_and_cache_flash @ e24d1b24; the cache
// is the NHD FlashAttention layout (num_blocks, block_size, num_kv_heads,
// head_size) — the two dim-1 slices of get_kv_cache_shape's
// (num_blocks, 2, block_size, num_kv_heads, head_size).
//
// Golden strategy (M1.6 Task-2 review): NO external oracle. The spec is a
// layout-consistent WRITE→READ round-trip: after ReshapeAndCache, reading slot s
// back from the NHD cache at [block, offset, kv_head, :] must equal the input
// k/v for that token. Then CUDA-vs-CPU parity (guarded by HasCuda).
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

// NHD flat element offset of cache[block, offset, head, e].
int64_t CacheIdx(int64_t slot, int64_t block_size, int64_t num_kv_heads, int64_t head_size,
                 int64_t head, int64_t e) {
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  return (((block * block_size + offset) * num_kv_heads) + head) * head_size + e;
}

// Build a rank-4 view with explicit element strides and a base-element offset
// into `data` (element-typed pointer). Mirrors the unbind(1) slice of the
// single (num_blocks, 2, block_size, H, D) allocation get_kv_cache_shape hands
// us: K/V are strided rank-4 views, NOT independently-packed tensors.
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
}  // namespace

TEST_CASE("reshape_and_cache round-trip: single token, single head") {
  // block_size=4, 1 kv-head, head_size=2, 1 block. Write token 0 to slot 3.
  const int64_t num_blocks = 1, block_size = 4, H = 1, D = 2;
  std::vector<float> k = {1.0f, 2.0f};
  std::vector<float> v = {3.0f, 4.0f};
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * H * D), -99.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * H * D), -99.0f);
  std::vector<int64_t> slots = {3};

  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {1, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {1, H, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {1});
  Queue qq = Q();
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  // slot 3 → block 0, offset 3.
  CHECK(kc[static_cast<size_t>(CacheIdx(3, block_size, H, D, 0, 0))] == doctest::Approx(1.0f));
  CHECK(kc[static_cast<size_t>(CacheIdx(3, block_size, H, D, 0, 1))] == doctest::Approx(2.0f));
  CHECK(vc[static_cast<size_t>(CacheIdx(3, block_size, H, D, 0, 0))] == doctest::Approx(3.0f));
  CHECK(vc[static_cast<size_t>(CacheIdx(3, block_size, H, D, 0, 1))] == doctest::Approx(4.0f));
  // slots 0..2 untouched (sentinel preserved).
  CHECK(kc[static_cast<size_t>(CacheIdx(0, block_size, H, D, 0, 0))] == doctest::Approx(-99.0f));
  CHECK(vc[static_cast<size_t>(CacheIdx(2, block_size, H, D, 0, 1))] == doctest::Approx(-99.0f));
}

TEST_CASE("reshape_and_cache multi-token, multi-head, block-spanning") {
  // block_size=2, 2 kv-heads, head_size=3, 3 blocks. 4 tokens landing in
  // different blocks AND different offsets:
  //   t0 → slot 0  (block 0, off 0)
  //   t1 → slot 3  (block 1, off 1)
  //   t2 → slot 4  (block 2, off 0)
  //   t3 → slot 1  (block 0, off 1)
  const int64_t num_blocks = 3, block_size = 2, H = 2, D = 3;
  const int64_t page = H * D;  // 6 elements per token
  std::vector<float> k(static_cast<size_t>(4 * page)), v(static_cast<size_t>(4 * page));
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = static_cast<float>(i) + 0.5f;
    v[i] = static_cast<float>(i) + 100.5f;
  }
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), -1.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), -1.0f);
  std::vector<int64_t> slots = {0, 3, 4, 1};

  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {4, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {4, H, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {4});
  Queue qq = Q();
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  // Every written token round-trips element-for-element across all heads.
  for (int64_t t = 0; t < 4; ++t) {
    const int64_t slot = slots[static_cast<size_t>(t)];
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t e = 0; e < D; ++e) {
        const size_t src = static_cast<size_t>((t * H + h) * D + e);
        const size_t idx = static_cast<size_t>(CacheIdx(slot, block_size, H, D, h, e));
        CHECK(kc[idx] == doctest::Approx(k[src]));
        CHECK(vc[idx] == doctest::Approx(v[src]));
      }
    }
  }
  // slot 5 (block 2, off 1) was never a target → untouched.
  CHECK(kc[static_cast<size_t>(CacheIdx(5, block_size, H, D, 1, 2))] == doctest::Approx(-1.0f));
}

TEST_CASE("reshape_and_cache skips slot == -1 (CUDA-graph padding)") {
  const int64_t num_blocks = 2, block_size = 2, H = 1, D = 2;
  const int64_t page = H * D;
  std::vector<float> k = {1, 1, 2, 2, 3, 3};  // 3 tokens
  std::vector<float> v = {4, 4, 5, 5, 6, 6};
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<int64_t> slots = {0, -1, 3};  // token 1 padded → skipped

  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {3, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {3, H, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {3});
  Queue qq = Q();
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  // token 0 → slot 0 written; token 2 → slot 3 written.
  CHECK(kc[static_cast<size_t>(CacheIdx(0, block_size, H, D, 0, 0))] == doctest::Approx(1.0f));
  CHECK(kc[static_cast<size_t>(CacheIdx(3, block_size, H, D, 0, 1))] == doctest::Approx(3.0f));
  CHECK(vc[static_cast<size_t>(CacheIdx(3, block_size, H, D, 0, 0))] == doctest::Approx(6.0f));
  // No slot was left holding token 1's data anywhere: slots 1 and 2 untouched.
  CHECK(kc[static_cast<size_t>(CacheIdx(1, block_size, H, D, 0, 0))] == doctest::Approx(0.0f));
  CHECK(kc[static_cast<size_t>(CacheIdx(2, block_size, H, D, 0, 0))] == doctest::Approx(0.0f));
}

TEST_CASE("reshape_and_cache ignores trailing padded k/v rows (num_tokens > slots)") {
  // k/v have 3 rows, slot_mapping has 2: the 3rd row is CUDA-graph padding.
  const int64_t num_blocks = 1, block_size = 2, H = 1, D = 2;
  const int64_t page = H * D;
  std::vector<float> k = {1, 1, 2, 2, 9, 9};  // row 2 = padding
  std::vector<float> v = {3, 3, 4, 4, 9, 9};
  std::vector<float> kc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<float> vc(static_cast<size_t>(num_blocks * block_size * page), 0.0f);
  std::vector<int64_t> slots = {0, 1};

  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {3, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {3, H, D});
  Tensor tkc = Contig(kc.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor tvc = Contig(vc.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {2});
  Queue qq = Q();
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  CHECK(kc[static_cast<size_t>(CacheIdx(0, block_size, H, D, 0, 0))] == doctest::Approx(1.0f));
  CHECK(kc[static_cast<size_t>(CacheIdx(1, block_size, H, D, 0, 0))] == doctest::Approx(2.0f));
  CHECK(vc[static_cast<size_t>(CacheIdx(1, block_size, H, D, 0, 1))] == doctest::Approx(4.0f));
}

TEST_CASE("reshape_and_cache bf16 round-trip is bit-exact") {
  auto bf = [](float x) { return vt::F32ToBF16(x); };
  const int64_t num_blocks = 1, block_size = 2, H = 1, D = 2;
  const int64_t page = H * D;
  std::vector<uint16_t> k = {bf(1.5f), bf(-2.25f)};
  std::vector<uint16_t> v = {bf(3.75f), bf(-4.5f)};
  std::vector<uint16_t> kc(static_cast<size_t>(num_blocks * block_size * page), 0);
  std::vector<uint16_t> vc(static_cast<size_t>(num_blocks * block_size * page), 0);
  std::vector<int64_t> slots = {1};

  Tensor tk = Contig(k.data(), DType::kBF16, Cpu(), {1, H, D});
  Tensor tv = Contig(v.data(), DType::kBF16, Cpu(), {1, H, D});
  Tensor tkc = Contig(kc.data(), DType::kBF16, Cpu(), {num_blocks, block_size, H, D});
  Tensor tvc = Contig(vc.data(), DType::kBF16, Cpu(), {num_blocks, block_size, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {1});
  Queue qq = Q();
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  // Raw copy → identical bit patterns (and thus identical decoded floats).
  CHECK(kc[static_cast<size_t>(CacheIdx(1, block_size, H, D, 0, 0))] == k[0]);
  CHECK(kc[static_cast<size_t>(CacheIdx(1, block_size, H, D, 0, 1))] == k[1]);
  CHECK(vc[static_cast<size_t>(CacheIdx(1, block_size, H, D, 0, 0))] == v[0]);
  CHECK(vc[static_cast<size_t>(CacheIdx(1, block_size, H, D, 0, 1))] == v[1]);
}

TEST_CASE("reshape_and_cache validates shapes/dtypes") {
  std::vector<float> buf(64, 0.0f);
  std::vector<int64_t> slots = {0};
  std::vector<int32_t> slots32 = {0};
  Queue qq = Q();

  Tensor tk = Contig(buf.data(), DType::kF32, Cpu(), {1, 1, 2});
  Tensor tv = Contig(buf.data(), DType::kF32, Cpu(), {1, 1, 2});
  Tensor tkc = Contig(buf.data(), DType::kF32, Cpu(), {1, 2, 1, 2});
  Tensor tvc = Contig(buf.data(), DType::kF32, Cpu(), {1, 2, 1, 2});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {1});

  // OK baseline does not throw.
  CHECK_NOTHROW(vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts));

  // slot_mapping must be i64.
  {
    Tensor bad = Contig(slots32.data(), DType::kI32, Cpu(), {1});
    CHECK_THROWS_AS(vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, bad), std::runtime_error);
  }
  // k/v head_size mismatch with cache.
  {
    std::vector<float> k4(4, 0.0f);
    Tensor bk = Contig(k4.data(), DType::kF32, Cpu(), {1, 1, 4});
    Tensor bv = Contig(k4.data(), DType::kF32, Cpu(), {1, 1, 4});
    CHECK_THROWS_AS(vt::ReshapeAndCache(qq, bk, bv, tkc, tvc, ts), std::runtime_error);
  }
  // cache not rank-4.
  {
    Tensor bkc = Contig(buf.data(), DType::kF32, Cpu(), {2, 1, 2});
    CHECK_THROWS_AS(vt::ReshapeAndCache(qq, tk, tv, bkc, tvc, ts), std::runtime_error);
  }
  // num_tokens (k.shape[0]) smaller than slot_mapping length.
  {
    std::vector<int64_t> two = {0, 1};
    Tensor ts2 = Contig(two.data(), DType::kI64, Cpu(), {2});
    CHECK_THROWS_AS(vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts2), std::runtime_error);
  }
}

// ===========================================================================
// STRIDED-SLICE regression (M1.6 Task-2 defect): the real cache is ONE
// allocation of shape (num_blocks, 2, block_size, H, D); K and V are the two
// dim-1 unbind slices, i.e. rank-4 views whose block stride is 2*bs*H*D (NOT
// bs*H*D) and whose data pointers differ by bs*H*D elements. The write must be
// driven by the tensor STRIDES, not by k_cache.shape. A shape-derived
// block_stride (= bs*H*D, half the real stride) makes block b write into the
// interleaved K/V memory and clobbers the other slice. These tests feed the op
// the genuine strided slices — the exact case Contig(...) caches never exercise.
// ===========================================================================
TEST_CASE("reshape_and_cache writes into the real strided K/V unbind slices") {
  // Single contiguous buffer laid out as (nb, 2, bs, H, D), row-major.
  const int64_t nb = 2, bs = 2, H = 2, D = 2;
  const int64_t page = H * D;              // 4 elems/token
  const int64_t within_block = bs * page;  // 8 elems (one K or V block)
  const int64_t blk_stride = 2 * within_block;  // 16 elems: the REAL block stride
  const size_t buf_n = static_cast<size_t>(nb * 2 * bs * page);  // 32
  std::vector<float> buf(buf_n, -99.0f);

  // K = buf[:, 0, ...] (offset 0); V = buf[:, 1, ...] (offset within_block).
  const std::vector<int64_t> cshape = {nb, bs, H, D};
  const std::vector<int64_t> cstride = {blk_stride, page, D, 1};
  Tensor tkc = StridedView(buf.data(), DType::kF32, Cpu(), 0, cshape, cstride);
  Tensor tvc = StridedView(buf.data(), DType::kF32, Cpu(), within_block, cshape, cstride);

  // One token → slot 2 = block 1, offset 0. With the buggy shape-derived stride
  // (bs*page = 8) this would land at buf[8..], i.e. V's block-0 region.
  std::vector<float> k(static_cast<size_t>(page)), v(static_cast<size_t>(page));
  for (int64_t i = 0; i < page; ++i) {
    k[static_cast<size_t>(i)] = 1.0f + static_cast<float>(i);
    v[static_cast<size_t>(i)] = 50.0f + static_cast<float>(i);
  }
  std::vector<int64_t> slots = {2};
  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {1, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {1, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {1});
  Queue qq = Q();
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  // slot 2 → block 1, offset 0. K lands at buf[block*blk_stride + off*page + ..].
  const int64_t kbase = 1 * blk_stride + 0 * page;             // 16
  const int64_t vbase = within_block + 1 * blk_stride + 0 * page;  // 8 + 16 = 24
  for (int64_t h = 0; h < H; ++h) {
    for (int64_t e = 0; e < D; ++e) {
      const int64_t koff = kbase + h * D + e;
      const int64_t voff = vbase + h * D + e;
      CHECK(buf[static_cast<size_t>(koff)] == doctest::Approx(k[static_cast<size_t>(h * D + e)]));
      CHECK(buf[static_cast<size_t>(voff)] == doctest::Approx(v[static_cast<size_t>(h * D + e)]));
    }
  }
  // V's block-0 region [8,16) is where the buggy stride would have dumped K.
  // It must remain the untouched sentinel.
  for (int64_t i = within_block; i < blk_stride; ++i) {
    CHECK(buf[static_cast<size_t>(i)] == doctest::Approx(-99.0f));
  }
  // K's block-0 region [0,8) also untouched (nothing landed in block 0).
  for (int64_t i = 0; i < within_block; ++i) {
    CHECK(buf[static_cast<size_t>(i)] == doctest::Approx(-99.0f));
  }
}

TEST_CASE("reshape_and_cache strided slices: K and V do not clobber each other") {
  // Larger, multi-token: write into BOTH slices at several slots and confirm
  // every K write and every V write round-trips through its own strided view
  // with no cross-contamination.
  const int64_t nb = 3, bs = 2, H = 2, D = 3;
  const int64_t page = H * D;              // 6
  const int64_t within_block = bs * page;  // 12
  const int64_t blk_stride = 2 * within_block;  // 24
  const size_t buf_n = static_cast<size_t>(nb * 2 * bs * page);  // 72
  std::vector<float> buf(buf_n, -1.0f);
  const std::vector<int64_t> cshape = {nb, bs, H, D};
  const std::vector<int64_t> cstride = {blk_stride, page, D, 1};
  Tensor tkc = StridedView(buf.data(), DType::kF32, Cpu(), 0, cshape, cstride);
  Tensor tvc = StridedView(buf.data(), DType::kF32, Cpu(), within_block, cshape, cstride);

  const int64_t nt = 4;
  std::vector<float> k(static_cast<size_t>(nt * page)), v(static_cast<size_t>(nt * page));
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = static_cast<float>(i) + 0.25f;
    v[i] = static_cast<float>(i) + 500.25f;
  }
  std::vector<int64_t> slots = {0, 5, 2, 3};  // blocks 0,2,1,1 / offsets 0,1,0,1
  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {nt, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {nt, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {nt});
  Queue qq = Q();
  vt::ReshapeAndCache(qq, tk, tv, tkc, tvc, ts);

  for (int64_t t = 0; t < nt; ++t) {
    const int64_t slot = slots[static_cast<size_t>(t)];
    const int64_t blk = slot / bs, off = slot % bs;
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t e = 0; e < D; ++e) {
        const int64_t inner = blk * blk_stride + off * page + h * D + e;
        const size_t src = static_cast<size_t>((t * H + h) * D + e);
        CHECK(buf[static_cast<size_t>(inner)] == doctest::Approx(k[src]));          // K slice
        CHECK(buf[static_cast<size_t>(within_block + inner)] == doctest::Approx(v[src]));  // V slice
      }
    }
  }
}

// ===========================================================================
// CUDA parity: the CUDA kernel must produce a byte-identical cache to the CPU
// reference on the same inputs (incl. block-spanning slots + a -1 skip).
// Guarded by HasCuda so CPU-only builds skip cleanly.
namespace {

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

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

TEST_CASE("reshape_and_cache CUDA matches CPU (block-spanning + skip)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping reshape_and_cache parity");
    return;
  }
  const int64_t num_blocks = 4, block_size = 8, H = 2, D = 16;
  const int64_t num_tokens = 10;
  const int64_t page = H * D;
  auto k = RandF32(static_cast<size_t>(num_tokens * page), 7);
  auto v = RandF32(static_cast<size_t>(num_tokens * page), 99);
  // Slots span multiple blocks + offsets, with two padded (-1) tokens.
  std::vector<int64_t> slots = {0, 9, 17, -1, 8, 31, 3, -1, 24, 15};

  const size_t cache_n = static_cast<size_t>(num_blocks * block_size * page);
  std::vector<float> kc_cpu(cache_n, -7.0f), vc_cpu(cache_n, -7.0f);
  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {num_tokens, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {num_tokens, H, D});
  Tensor tkc = Contig(kc_cpu.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor tvc = Contig(vc_cpu.data(), DType::kF32, Cpu(), {num_blocks, block_size, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(),
                     {static_cast<int64_t>(slots.size())});
  Queue cpuq = Q();
  vt::ReshapeAndCache(cpuq, tk, tv, tkc, tvc, ts);

  // CUDA: pre-fill caches with the same sentinel so untouched slots must agree.
  std::vector<float> kc_init(cache_n, -7.0f), vc_init(cache_n, -7.0f);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dk(gpu, g.q, DType::kF32, {num_tokens, H, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kF32, {num_tokens, H, D}, v.data());
  DeviceTensor dkc(gpu, g.q, DType::kF32, {num_blocks, block_size, H, D}, kc_init.data());
  DeviceTensor dvc(gpu, g.q, DType::kF32, {num_blocks, block_size, H, D}, vc_init.data());
  DeviceTensor ds(gpu, g.q, DType::kI64, {static_cast<int64_t>(slots.size())}, slots.data());
  vt::ReshapeAndCache(g.q, dk.tensor(), dv.tensor(), dkc.tensor(), dvc.tensor(), ds.tensor());

  std::vector<float> kc_got(cache_n, 0.0f), vc_got(cache_n, 0.0f);
  dkc.Download(g.q, kc_got.data());
  dvc.Download(g.q, vc_got.data());
  for (size_t i = 0; i < cache_n; ++i) {
    CHECK(kc_got[i] == doctest::Approx(kc_cpu[i]));
    CHECK(vc_got[i] == doctest::Approx(vc_cpu[i]));
  }
}

TEST_CASE("reshape_and_cache CUDA matches CPU on the real strided unbind slices") {
  // Same defect surface as the CPU strided tests, on-device: ONE (nb,2,bs,H,D)
  // buffer, K/V are strided rank-4 views (block stride 2*bs*H*D). CPU and CUDA
  // must agree byte-for-byte over the whole shared buffer. dgx-pending on this
  // CPU-only box; build-guarded by HasCuda.
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping reshape_and_cache strided parity");
    return;
  }
  const int64_t nb = 4, bs = 8, H = 2, D = 16;
  const int64_t num_tokens = 10;
  const int64_t page = H * D;
  const int64_t within_block = bs * page;
  const int64_t blk_stride = 2 * within_block;
  const size_t buf_n = static_cast<size_t>(nb * 2 * bs * page);
  auto k = RandF32(static_cast<size_t>(num_tokens * page), 7);
  auto v = RandF32(static_cast<size_t>(num_tokens * page), 99);
  std::vector<int64_t> slots = {0, 9, 17, -1, 8, 31, 3, -1, 24, 15};

  const std::vector<int64_t> cshape = {nb, bs, H, D};
  const std::vector<int64_t> cstride = {blk_stride, page, D, 1};

  // CPU reference: one host buffer, two strided views.
  std::vector<float> buf_cpu(buf_n, -7.0f);
  Tensor tkc = StridedView(buf_cpu.data(), DType::kF32, Cpu(), 0, cshape, cstride);
  Tensor tvc = StridedView(buf_cpu.data(), DType::kF32, Cpu(), within_block, cshape, cstride);
  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {num_tokens, H, D});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {num_tokens, H, D});
  Tensor ts = Contig(slots.data(), DType::kI64, Cpu(), {static_cast<int64_t>(slots.size())});
  Queue cpuq = Q();
  vt::ReshapeAndCache(cpuq, tk, tv, tkc, tvc, ts);

  // CUDA: one device buffer, two strided device views into it.
  std::vector<float> buf_init(buf_n, -7.0f);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  void* dbuf = gpu.Alloc(buf_n * sizeof(float));
  gpu.Copy(g.q, dbuf, buf_init.data(), buf_n * sizeof(float));
  Tensor dkc = StridedView(static_cast<float*>(dbuf), DType::kF32, Gpu(), 0, cshape, cstride);
  Tensor dvc =
      StridedView(static_cast<float*>(dbuf), DType::kF32, Gpu(), within_block, cshape, cstride);
  DeviceTensor dk(gpu, g.q, DType::kF32, {num_tokens, H, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kF32, {num_tokens, H, D}, v.data());
  DeviceTensor ds(gpu, g.q, DType::kI64, {static_cast<int64_t>(slots.size())}, slots.data());
  vt::ReshapeAndCache(g.q, dk.tensor(), dv.tensor(), dkc, dvc, ds.tensor());

  std::vector<float> buf_got(buf_n, 0.0f);
  gpu.Copy(g.q, buf_got.data(), dbuf, buf_n * sizeof(float));
  gpu.Synchronize(g.q);
  gpu.Free(dbuf);
  for (size_t i = 0; i < buf_n; ++i) {
    CHECK(buf_got[i] == doctest::Approx(buf_cpu[i]));
  }
}
