// vllm.cpp original (vt runtime). The DIRECT unit gate for `vt::AttentionCross`
// on EVERY backend that registers it — which, since LTX-2.5 phase L8, means the
// CPU reference AND the native CUDA kernel (src/vt/cuda/cuda_attention_cross.cu).
//
// WHY THIS FILE EXISTS. When the CUDA kernel landed, every test that reached it
// went through the LTX-2.5 DiT, and the six distinct call geometries that model
// produces are ALL `tiles=1, npl=1, nblk=1, Hq==Hkv`. So the kernel's whole
// flash-tiling machinery — the cross-tile online-softmax rescale, the
// multi-query-block launch, the >32-element-per-lane head split, the GQA
// broadcast, and `ChooseTileCols`' shared-memory halving — was DEAD CODE in every
// gate, including the one comment claiming the halving "prevents a launch failure
// on exactly the real geometry". A real render puts S at prompt length and Tq in
// the thousands, straight into that untested regime. Spec §7.0(c): a fixture that
// cannot reach the regime which discriminates is the same defect as a missing
// test, wearing different clothes.
//
// THE ORACLE IS INDEPENDENT OF BOTH KERNELS. `Reference` below computes the
// attention in f64 with an explicit max-subtracted softmax, from the tensor
// layout contract in include/vt/ops.h, and nothing else. Gating the CUDA kernel
// against the CPU kernel alone would prove the two agree; it would not prove
// either is right, which is exactly the "gate on a shared helper" failure this
// project has already been bitten by. Both backends are held to `Reference`, and
// the CPU-vs-CUDA comparison is kept as a SEPARATE, tighter statement.
//
// THE GEOMETRIES ARE CHOSEN BY WHAT THEY DRIVE, not by what looks plausible:
//
//   S  > kCrossBcMax(=64)      -> tiles > 1, so the online rescale runs
//   Tq > kCrossBr(=16)         -> nblk > 1, so blockIdx.x is load-bearing
//   D  > 32                    -> npl > 1, so the per-lane head split runs
//   Hq > Hkv                   -> the GQA broadcast runs
//   D == 128 with an f32 stream -> 2*64*128*4 = 64 KiB > 48 KiB, so
//                                 ChooseTileCols MUST halve or the launch fails
//
// A `[Tq, S]` dense bias is carried across a tile boundary in the multi-tile
// cases, because a kernel that read bias row 0 for every query, or that indexed
// the bias by the within-tile column instead of the absolute key, is
// indistinguishable from correct at tiles == 1.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "support/max_abs_diff.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::AttentionCrossArgs;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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

bool HasCuda() { return vt::TryGetBackend(DeviceType::kCUDA) != nullptr; }

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
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeT(p_, dt, Gpu(), shape);
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

// Deterministic LCG in [-2, 2). Same stream shape the other vt op tests use, so a
// failure is reproducible without <random>'s cross-library divergence.
std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

// bf16 round-trip, so a bf16 arm compares against the reference computed from the
// values the kernel ACTUALLY reads rather than from their f32 originals.
float RoundToBf16(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  const uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
  float out = 0.0f;
  std::memcpy(&out, &rounded, sizeof(out));
  return out;
}

struct Geometry {
  int64_t tq = 0, s = 0, hq = 0, hk = 0, d = 0;
};

// THE ORACLE. f64, explicit max-subtracted softmax, straight from the layout
// contract: query/out are [Tq, Hq, D], key/value are [S, Hkv, D], kv-head
// g = h / (Hq/Hkv), bias row is the query's own or the single broadcast row, and
// the bias joins the SCALED score before the max-subtraction.
std::vector<float> Reference(const Geometry& g, const std::vector<float>& query,
                             const std::vector<float>& key, const std::vector<float>& value,
                             const std::vector<float>* bias, int64_t bias_rows, float scale) {
  std::vector<float> out(static_cast<size_t>(g.tq * g.hq * g.d), 0.0f);
  const int64_t qpk = g.hq / g.hk;
  std::vector<double> scores(static_cast<size_t>(g.s));
  for (int64_t h = 0; h < g.hq; ++h) {
    const int64_t kv = h / qpk;
    for (int64_t i = 0; i < g.tq; ++i) {
      const int64_t qoff = (i * g.hq + h) * g.d;
      double m = -std::numeric_limits<double>::infinity();
      for (int64_t j = 0; j < g.s; ++j) {
        const int64_t koff = (j * g.hk + kv) * g.d;
        double dot = 0.0;
        for (int64_t e = 0; e < g.d; ++e) {
          dot += static_cast<double>(query[static_cast<size_t>(qoff + e)]) *
                 static_cast<double>(key[static_cast<size_t>(koff + e)]);
        }
        dot *= static_cast<double>(scale);
        if (bias != nullptr) {
          const int64_t row = bias_rows == 1 ? 0 : i;
          dot += static_cast<double>((*bias)[static_cast<size_t>(row * g.s + j)]);
        }
        scores[static_cast<size_t>(j)] = dot;
        if (dot > m) m = dot;
      }
      double denom = 0.0;
      for (int64_t j = 0; j < g.s; ++j) {
        const double e = std::exp(scores[static_cast<size_t>(j)] - m);
        scores[static_cast<size_t>(j)] = e;
        denom += e;
      }
      for (int64_t e = 0; e < g.d; ++e) {
        double acc = 0.0;
        for (int64_t j = 0; j < g.s; ++j) {
          const int64_t voff = (j * g.hk + kv) * g.d;
          acc += scores[static_cast<size_t>(j)] *
                 static_cast<double>(value[static_cast<size_t>(voff + e)]);
        }
        out[static_cast<size_t>(qoff + e)] = static_cast<float>(acc / denom);
      }
    }
  }
  return out;
}

// The tile/block/lane counts a geometry drives, so each case can ASSERT it
// reaches the regime it claims to test rather than believing its own comment.
// Mirrors cuda_attention_cross.cu's kCrossBr / kCrossBcMax / ChooseTileCols.
constexpr int kCrossBr = 16;
constexpr int kCrossBcMax = 64;
constexpr size_t kMaxDynamicSmem = 48u * 1024u;

int TileCols(int64_t d, size_t elem_bytes) {
  int bc = kCrossBcMax;
  while (bc > 1 && static_cast<size_t>(2 * bc) * static_cast<size_t>(d) * elem_bytes >
                       kMaxDynamicSmem) {
    bc /= 2;
  }
  return bc;
}
int64_t Tiles(const Geometry& g, size_t elem_bytes) {
  const int bc = TileCols(g.d, elem_bytes);
  return (g.s + bc - 1) / bc;
}
int64_t Blocks(const Geometry& g) { return (g.tq + kCrossBr - 1) / kCrossBr; }
int64_t PerLane(const Geometry& g) { return (g.d + 31) / 32; }

struct Inputs {
  std::vector<float> query, key, value, bias;
  int64_t bias_rows = 0;
  float scale = 1.0f;
  bool has_bias = false;
};

Inputs MakeInputs(const Geometry& g, uint32_t seed, int64_t bias_rows) {
  Inputs in;
  in.query = RandF32(static_cast<size_t>(g.tq * g.hq * g.d), seed);
  in.key = RandF32(static_cast<size_t>(g.s * g.hk * g.d), seed + 101);
  in.value = RandF32(static_cast<size_t>(g.s * g.hk * g.d), seed + 202);
  in.scale = 1.0f / std::sqrt(static_cast<float>(g.d));
  if (bias_rows > 0) {
    in.has_bias = true;
    in.bias_rows = bias_rows;
    in.bias = RandF32(static_cast<size_t>(bias_rows * g.s), seed + 303);
  }
  return in;
}

// Run one geometry on the CPU op and (when present) the CUDA op, holding BOTH to
// the independent f64 reference and then to each other.
void RunGeometry(const std::string& label, const Geometry& g, uint32_t seed, int64_t bias_rows,
                 DType stream, double tol) {
  INFO(label << " [Tq=" << g.tq << " S=" << g.s << " Hq=" << g.hq << " Hkv=" << g.hk
             << " D=" << g.d << " stream=" << (stream == DType::kBF16 ? "bf16" : "f32") << "]");
  Inputs in = MakeInputs(g, seed, bias_rows);
  if (stream == DType::kBF16) {
    // The kernel reads bf16; the reference must read the SAME numbers, or the
    // comparison measures the rounding rather than the kernel.
    for (float& v : in.query) v = RoundToBf16(v);
    for (float& v : in.key) v = RoundToBf16(v);
    for (float& v : in.value) v = RoundToBf16(v);
  }
  const std::vector<float> want =
      Reference(g, in.query, in.key, in.value, in.has_bias ? &in.bias : nullptr, in.bias_rows,
                in.scale);

  // The regime this geometry claims to reach, asserted rather than assumed.
  const size_t elem = vt::SizeOf(stream);
  MESSAGE(label << ": tiles=" << Tiles(g, elem) << " nblk=" << Blocks(g)
                << " npl=" << PerLane(g) << " bc=" << TileCols(g.d, elem)
                << " hq/hkv=" << (g.hq / g.hk));

  AttentionCrossArgs args;
  args.scale = in.scale;

  // ── the CPU op ────────────────────────────────────────────────────────────
  std::vector<float> cpu_out(static_cast<size_t>(g.tq * g.hq * g.d), 0.0f);
  {
    Queue q{Cpu(), nullptr};
    Tensor tq = MakeT(in.query.data(), DType::kF32, Cpu(), {g.tq, g.hq, g.d});
    Tensor tk = MakeT(in.key.data(), DType::kF32, Cpu(), {g.s, g.hk, g.d});
    Tensor tv = MakeT(in.value.data(), DType::kF32, Cpu(), {g.s, g.hk, g.d});
    Tensor to = MakeT(cpu_out.data(), DType::kF32, Cpu(), {g.tq, g.hq, g.d});
    Tensor tb = MakeT(in.bias.data(), DType::kF32, Cpu(), {in.bias_rows, g.s});
    vt::AttentionCross(q, to, tq, tk, tv, in.has_bias ? &tb : nullptr, args);
  }
  const double cpu_diff = vllm_test::MaxAbsDiff(cpu_out, want.data(), want.size());
  INFO("cpu max|diff| vs the f64 reference = " << cpu_diff);
  CHECK(cpu_diff < tol);

  if (!HasCuda()) {
    MESSAGE(label << ": SKIP the CUDA arm (no CUDA backend registered)");
    return;
  }

  // ── the CUDA op ───────────────────────────────────────────────────────────
  Backend& gpu = *vt::TryGetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  std::vector<float> gpu_out(static_cast<size_t>(g.tq * g.hq * g.d), 0.0f);
  if (stream == DType::kF32) {
    DeviceTensor dq(gpu, guard.q, DType::kF32, {g.tq, g.hq, g.d}, in.query.data());
    DeviceTensor dk(gpu, guard.q, DType::kF32, {g.s, g.hk, g.d}, in.key.data());
    DeviceTensor dv(gpu, guard.q, DType::kF32, {g.s, g.hk, g.d}, in.value.data());
    DeviceTensor dout(gpu, guard.q, DType::kF32, {g.tq, g.hq, g.d});
    DeviceTensor dbias(gpu, guard.q, DType::kF32, {in.has_bias ? in.bias_rows : 1, g.s},
                       in.has_bias ? in.bias.data() : nullptr);
    vt::AttentionCross(guard.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(),
                       in.has_bias ? &dbias.tensor() : nullptr, args);
    dout.Download(guard.q, gpu_out.data());
  } else {
    // bf16 in, f32 out: the OUT dtype is held f32 so the comparison measures the
    // kernel's arithmetic and not a bf16 store quantizing the answer.
    std::vector<uint16_t> qb(in.query.size()), kb(in.key.size()), vb(in.value.size());
    auto pack = [](const std::vector<float>& src, std::vector<uint16_t>& dst) {
      for (size_t i = 0; i < src.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &src[i], sizeof(bits));
        dst[i] = static_cast<uint16_t>(bits >> 16);
      }
    };
    pack(in.query, qb);
    pack(in.key, kb);
    pack(in.value, vb);
    DeviceTensor dq(gpu, guard.q, DType::kBF16, {g.tq, g.hq, g.d}, qb.data());
    DeviceTensor dk(gpu, guard.q, DType::kBF16, {g.s, g.hk, g.d}, kb.data());
    DeviceTensor dv(gpu, guard.q, DType::kBF16, {g.s, g.hk, g.d}, vb.data());
    DeviceTensor dout(gpu, guard.q, DType::kF32, {g.tq, g.hq, g.d});
    DeviceTensor dbias(gpu, guard.q, DType::kF32, {in.has_bias ? in.bias_rows : 1, g.s},
                       in.has_bias ? in.bias.data() : nullptr);
    vt::AttentionCross(guard.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(),
                       in.has_bias ? &dbias.tensor() : nullptr, args);
    dout.Download(guard.q, gpu_out.data());
  }
  const double gpu_diff = vllm_test::MaxAbsDiff(gpu_out, want.data(), want.size());
  INFO("cuda max|diff| vs the f64 reference = " << gpu_diff);
  CHECK(gpu_diff < tol);
  // And the two backends against each other. They use DIFFERENT softmax
  // algorithms — online recurrence on CUDA, explicit three-pass on the CPU — so
  // this is summation-order slack, not an identity; it is stated as the bound it
  // is rather than as "bit-identical", which nothing here measures.
  if (stream == DType::kF32) {
    const double cross = vllm_test::MaxAbsDiff(gpu_out, cpu_out.data(), cpu_out.size());
    MESSAGE(label << ": cuda-vs-cpu max|diff| = " << cross);
    CHECK(cross < tol);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// The regimes the LTX-2.5 forward never reaches
// ---------------------------------------------------------------------------

TEST_CASE("attention-cross: MULTI-TILE keys drive the cross-tile online rescale") {
  // S = 200 > kCrossBcMax = 64, so the key loop runs 4 tiles and the running
  // (m, l, acc) must be rescaled at every tile boundary. Every LTX-2.5 gate
  // geometry is tiles == 1, where the rescale is a no-op.
  const Geometry g{4, 200, 2, 2, 8};
  REQUIRE(Tiles(g, sizeof(float)) > 1);
  RunGeometry("multi-tile", g, 11u, /*bias_rows=*/0, DType::kF32, 2e-5);
}

TEST_CASE("attention-cross: a DENSE bias is indexed by the ABSOLUTE key across tiles") {
  // The same multi-tile shape with a [Tq, S] bias. Two defects are invisible at
  // tiles == 1 and caught here: reading bias row 0 for every query, and indexing
  // the bias by the WITHIN-TILE column `j` instead of the absolute key `c0 + j`.
  const Geometry g{20, 200, 2, 2, 8};
  REQUIRE(Tiles(g, sizeof(float)) > 1);
  REQUIRE(Blocks(g) > 1);
  RunGeometry("multi-tile + dense bias", g, 23u, /*bias_rows=*/20, DType::kF32, 2e-5);
}

TEST_CASE("attention-cross: MULTI-BLOCK queries, including a ragged last block") {
  // Tq = 40 with kCrossBr = 16 gives 3 query blocks whose last one is RAGGED
  // (40 = 2*16 + 8), so the `qi < tq` guard, the inactive-warp `continue` and the
  // tile-load loop all have to be right together.
  const Geometry g{40, 70, 4, 4, 16};
  REQUIRE(Blocks(g) == 3);
  REQUIRE(g.tq % kCrossBr != 0);
  RunGeometry("multi-block ragged", g, 37u, /*bias_rows=*/1, DType::kF32, 2e-5);
}

TEST_CASE("attention-cross: head_dim > 32 splits the head across lane iterations") {
  // D = 64 gives npl = 2, so each lane owns two head elements and the register
  // arrays are indexed rather than trivially single-slot.
  const Geometry g{5, 80, 2, 2, 64};
  REQUIRE(PerLane(g) == 2);
  REQUIRE(Tiles(g, sizeof(float)) > 1);
  RunGeometry("npl=2", g, 53u, /*bias_rows=*/0, DType::kF32, 2e-5);
}

TEST_CASE("attention-cross: Hq > Hkv broadcasts each kv-head to its query group") {
  // The GQA broadcast, which no LTX-2.5 call reaches: every one of its
  // cross-attentions has Hq == Hkv. A kernel that used `h` where it needs
  // `h / (Hq/Hkv)` is exactly correct at Hq == Hkv and wrong everywhere else.
  const Geometry g{20, 100, 8, 2, 32};
  REQUIRE(g.hq > g.hk);
  REQUIRE(Blocks(g) > 1);
  REQUIRE(Tiles(g, sizeof(float)) > 1);
  RunGeometry("gqa 8/2", g, 71u, /*bias_rows=*/0, DType::kF32, 2e-5);
}

TEST_CASE("attention-cross: the REAL head_dim 128 geometry, where the tile MUST halve") {
  // `ChooseTileCols`' halving carries a comment saying a fixed 64-column tile
  // "would fail to launch on exactly the real geometry", and nothing executed it:
  // 2 * 64 * 128 * 4 = 64 KiB against the 48 KiB a launch gets without opting in.
  // This is that geometry — LTX-2.5's video stream head_dim — with Tq and S both
  // past their block/tile thresholds.
  const Geometry g{64, 100, 2, 2, 128};
  REQUIRE(TileCols(g.d, sizeof(float)) == 32);   // the halving actually fired
  REQUIRE(TileCols(g.d, sizeof(uint16_t)) == 64);  // ...and bf16 does not need it
  REQUIRE(Tiles(g, sizeof(float)) > 1);
  REQUIRE(Blocks(g) > 1);
  REQUIRE(PerLane(g) == 4);
  RunGeometry("d=128 f32 (tile halved)", g, 97u, /*bias_rows=*/64, DType::kF32, 2e-5);
}

TEST_CASE("attention-cross: the bf16 stream at the real head_dim, multi-tile") {
  // bf16 is the PRODUCTION dtype of the LTX-2.5 device forward. Held to the same
  // f64 reference over bf16-rounded inputs, at a bf16 band.
  const Geometry g{40, 150, 2, 2, 128};
  REQUIRE(TileCols(g.d, sizeof(uint16_t)) == 64);
  REQUIRE(Tiles(g, sizeof(uint16_t)) > 1);
  REQUIRE(Blocks(g) > 1);
  RunGeometry("bf16 d=128", g, 131u, /*bias_rows=*/1, DType::kBF16, 5e-3);
}

TEST_CASE("attention-cross: a fully masked key in a LATER tile drops out of the softmax") {
  // ops.h records that a fully masked key arrives as a large negative number
  // rather than -inf, so an all-masked row degenerates to a uniform average
  // exactly as torch's does. Put the mask in tile 2 so it is the multi-tile
  // rescale, not the first-tile initialisation, that has to preserve it.
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  const Geometry g{3, 130, 1, 1, 16};
  REQUIRE(Tiles(g, sizeof(float)) > 1);
  Inputs in = MakeInputs(g, 149u, /*bias_rows=*/1);
  // Upstream's own form: `(mask - 1) * finfo.max` for a dropped key, 0 otherwise
  // (transformer_args.py:204). -FLT_MAX, never -inf.
  for (int64_t j = 0; j < g.s; ++j) {
    in.bias[static_cast<size_t>(j)] = (j >= 70 && j < 100) ? -3.40282347e+38F : 0.0f;
  }
  const std::vector<float> want =
      Reference(g, in.query, in.key, in.value, &in.bias, 1, in.scale);

  AttentionCrossArgs args;
  args.scale = in.scale;
  Backend& gpu = *vt::TryGetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  DeviceTensor dq(gpu, guard.q, DType::kF32, {g.tq, g.hq, g.d}, in.query.data());
  DeviceTensor dk(gpu, guard.q, DType::kF32, {g.s, g.hk, g.d}, in.key.data());
  DeviceTensor dv(gpu, guard.q, DType::kF32, {g.s, g.hk, g.d}, in.value.data());
  DeviceTensor dout(gpu, guard.q, DType::kF32, {g.tq, g.hq, g.d});
  DeviceTensor dbias(gpu, guard.q, DType::kF32, {1, g.s}, in.bias.data());
  vt::AttentionCross(guard.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(),
                     &dbias.tensor(), args);
  std::vector<float> got(static_cast<size_t>(g.tq * g.hq * g.d), 0.0f);
  dout.Download(guard.q, got.data());
  // MaxAbsDiff treats a non-finite operand as a FAILURE, so a NaN produced by an
  // -inf substitution cannot read as agreement here (issue #449).
  const double diff = vllm_test::MaxAbsDiff(got, want.data(), want.size());
  MESSAGE("masked-tile max|diff| = " << diff);
  CHECK(diff < 2e-5);
}

TEST_CASE("attention-cross: the CUDA provider is registered and is not the CPU kernel") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // Registration is what stops `RegisterReferenceTier` serving the CPU kernel to
  // a CUDA queue on unified memory, which is how "it ran on the GPU" becomes
  // false while every gate stays green.
  CHECK(vt::OpRegistered(vt::OpId::kAttentionCross, DeviceType::kCUDA));
  CHECK(vt::OpRegistered(vt::OpId::kAttentionCross, DeviceType::kCPU));
}
// ---------------------------------------------------------------------------
// The BLOCKED CUDA provider (#1555)
//
// `vt-cross-blocked` is a SECOND CUDA kernel for this op, selected above
// `vt-native` and declining per call on a shape it has no tiling for. Every case
// below states which arm it means to reach and ASSERTS it through
// `GetOpProviderStats`, because the two kernels compute the same function and no
// numerical check can tell them apart.
//
// `declines` is the discriminator, not `last_selected`: the blocked provider is
// always the SELECTED one, and what separates a served call from a forwarded one
// is whether it declined.
// ---------------------------------------------------------------------------

namespace {

constexpr const char* kBlockedProvider = "vt-cross-blocked";

// MEASURED, not chosen. On `thor:gpu0`, worst of four seeds, the cancellation
// case below reports the SHIPPED kernel at 8.50797e-04 and the BLOCKED one at
// 9.10163e-05 -- a ratio of 0.1398, so the restructure is 7.2x MORE accurate
// there, not less. The absolute bound is set on the binding arm (the shipped
// kernel, 2.35x of margin) rather than on the better one.
//
// The RATIO bound is 2.0 and not the measured 0.14 plus a margin, because what
// it has to catch is a DEGRADATION: the blocked kernel must not become less
// accurate than the kernel it replaces. 2.0 gives that claim 14x of headroom
// against today's reading while still failing on any real inversion.
constexpr double kCancellationTol = 2e-3;
constexpr double kCancellationRatio = 2.0;

// The shape gate, restated from cuda_attention_cross.cu so a case can assert the
// regime it claims rather than believe its own comment. A copy is deliberate:
// if the kernel's gate moves and this does not, the REQUIREs below go red, which
// is the point of writing it twice.
bool BlockedShape(const Geometry& g) {
  if (g.hk == 0 || g.hq % g.hk != 0) return false;
  if (g.d == 64) return g.tq >= 64;
  if (g.d == 128) return g.tq >= 32;
  return false;
}

// EXACTNESS IS NOW A PROPERTY OF THE SEAM, not of this file's case order
// ([#1584](https://github.com/mudler/vllm.cpp/issues/1584), fixed). The
// `declines ==` assertions below used to need a warm-up call placed outside
// every counted window, because `GetOpFallback` counted a decline itself and the
// caching pattern `op_provider.h` prescribes called `NoteOpDecline` beside it --
// so the FIRST decline of the process counted twice and each case passed or
// failed on whether an earlier case in the same binary had already declined.
// `BlockedFallback` now hoists `GetOpFallbackUncounted`, so one decline is one
// count from the first, and these cases are exact standalone and under `-tc=`.
// If a warm-up ever becomes necessary again, the seam has regressed and the
// right repair is there rather than here.
//
// What that paragraph does NOT say, and what #1812 measured, is that these cases
// cannot SEE the seam regress: by the time any of them runs, the static is warm
// and the extra increment has already been reset away. The probe below the
// harness is what sees it, and each exact case asserts the probe's reading
// beside its own so that a filtered run of one case is falsifiable too.

struct ProviderStatsGuard {
  ProviderStatsGuard() {
    vt::EnableOpProviderCallStats(true);
    vt::ResetOpProviderStats(vt::OpId::kAttentionCross, DeviceType::kCUDA);
  }
  ~ProviderStatsGuard() {
    vt::ResetOpProviderStats(vt::OpId::kAttentionCross, DeviceType::kCUDA);
    vt::EnableOpProviderCallStats(false);
  }
};

// Runs ONE CUDA AttentionCross and returns the raw bytes plus the decline count
// it produced. `disable_blocked` takes the same lever the A/B measurement uses,
// so a test and a benchmark exercise the same switch.
struct CudaRun {
  std::vector<float> out;
  unsigned long long declines = 0;
  const char* selected = nullptr;
};

CudaRun RunCuda(const Geometry& g, const Inputs& in, DType stream, bool disable_blocked) {
  Backend& gpu = *vt::TryGetBackend(DeviceType::kCUDA);
  QueueGuard guard(gpu);
  AttentionCrossArgs args;
  args.scale = in.scale;
  CudaRun r;
  r.out.assign(static_cast<size_t>(g.tq * g.hq * g.d), 0.0f);

  vt::DisableOpProvider(kBlockedProvider, disable_blocked);
  ProviderStatsGuard stats;
  {
    DeviceTensor dout(gpu, guard.q, DType::kF32, {g.tq, g.hq, g.d});
    DeviceTensor dbias(gpu, guard.q, DType::kF32, {in.has_bias ? in.bias_rows : 1, g.s},
                       in.has_bias ? in.bias.data() : nullptr);
    if (stream == DType::kF32) {
      DeviceTensor dq(gpu, guard.q, DType::kF32, {g.tq, g.hq, g.d}, in.query.data());
      DeviceTensor dk(gpu, guard.q, DType::kF32, {g.s, g.hk, g.d}, in.key.data());
      DeviceTensor dv(gpu, guard.q, DType::kF32, {g.s, g.hk, g.d}, in.value.data());
      vt::AttentionCross(guard.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(),
                         in.has_bias ? &dbias.tensor() : nullptr, args);
      dout.Download(guard.q, r.out.data());
    } else {
      auto pack = [](const std::vector<float>& src) {
        std::vector<uint16_t> dst(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
          uint32_t bits = 0;
          std::memcpy(&bits, &src[i], sizeof(bits));
          dst[i] = static_cast<uint16_t>(bits >> 16);
        }
        return dst;
      };
      const std::vector<uint16_t> qb = pack(in.query), kb = pack(in.key), vb = pack(in.value);
      DeviceTensor dq(gpu, guard.q, DType::kBF16, {g.tq, g.hq, g.d}, qb.data());
      DeviceTensor dk(gpu, guard.q, DType::kBF16, {g.s, g.hk, g.d}, kb.data());
      DeviceTensor dv(gpu, guard.q, DType::kBF16, {g.s, g.hk, g.d}, vb.data());
      vt::AttentionCross(guard.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(),
                         in.has_bias ? &dbias.tensor() : nullptr, args);
      dout.Download(guard.q, r.out.data());
    }
    const vt::OpProviderStats s =
        vt::GetOpProviderStats(vt::OpId::kAttentionCross, DeviceType::kCUDA);
    r.declines = s.declines;
    r.selected = s.last_selected;
  }
  vt::DisableOpProvider(kBlockedProvider, false);
  return r;
}

// ---------------------------------------------------------------------------
// THE FIRST DECLINE OF THE PROCESS, measured before any case runs (#1812)
//
// `BlockedFallback()` in cuda_attention_cross.cu caches its fallback pointer in
// a function-local static, so the seam guarantee this row exists for -- one
// decline is one count -- is OBSERVABLE only around the call that RESOLVES that
// static. Every later decline reads the same whether the resolver counted or
// not, because the double count happens once per static and not once per call.
//
// That is not a subtlety, it is the whole gate, and it was MEASURED rather than
// reasoned (spec 12.2, `thor:gpu0`): with this row's entire CUDA edit reverted
// -- `GetOpFallbackUncounted` back to the counting `GetOpFallback` -- the FULL
// suite stayed GREEN at 156/156. The plain `attention-cross:` cases above run
// geometries `BlockedShape` rejects, so THEY resolve the static first, outside
// any counted window, and the next `ProviderStatsGuard` erases the extra
// increment before the exact `declines == 1` cases below ever look. The suite
// warmed itself exactly as #1555's deleted `WarmDeclineOnce` warmed it by hand,
// and `tests/CMakeLists.txt` registers ONE ctest entry per suite, so nothing in
// this repository could see that revert.
//
// The repair is to take the measurement while it is still available: ONE
// declining call, inside a counted window, before doctest runs a single case. A
// doctest LISTENER's `test_run_start()` is that point -- doctest.h:5983 fires it
// once before the case loop and skips it only for the `--list-*`/`--count`
// queries -- and it is order-proof by construction rather than by case order,
// which doctest does not guarantee and which `--order-by=rand` can invert.
struct FirstDeclineProbe {
  bool ran = false;   // the listener fired at all
  bool cuda = false;  // ... and there was a device to run on
  unsigned long long declines = 0;
  std::string error;  // an exception, kept rather than swallowed
};

FirstDeclineProbe& FirstDecline() {
  static FirstDeclineProbe p;
  return p;
}

// One declining CUDA AttentionCross, and what `declines` read around it. The
// geometry is the cheapest one `BlockedShape` rejects (head_dim 32 has no
// blocked tiling), asserted below rather than assumed, so the probe costs a
// single tiny launch. It goes through `RunCuda`, so the window is the same
// reset-call-read that every case below uses.
void MeasureFirstDecline() {
  FirstDeclineProbe& p = FirstDecline();
  p.ran = true;
  if (!HasCuda()) return;
  p.cuda = true;
  const Geometry g{1, 8, 1, 1, 32};
  if (BlockedShape(g)) {
    p.error = "the probe geometry is NOT one the blocked provider declines";
    return;
  }
  try {
    const Inputs in = MakeInputs(g, 1584u, /*bias_rows=*/0);
    p.declines = RunCuda(g, in, DType::kF32, /*disable_blocked=*/false).declines;
  } catch (const std::exception& e) {
    p.error = e.what();
  } catch (...) {
    p.error = "unknown exception";
  }
}

// A LISTENER, not a reporter: listeners are always active, whatever `-r=`
// selects, so this cannot be switched off from the command line.
struct FirstDeclineListener : public doctest::IReporter {
  explicit FirstDeclineListener(const doctest::ContextOptions&) {}
  void test_run_start() override { MeasureFirstDecline(); }
  void report_query(const doctest::QueryData&) override {}
  void test_run_end(const doctest::TestRunStats&) override {}
  void test_case_start(const doctest::TestCaseData&) override {}
  void test_case_reenter(const doctest::TestCaseData&) override {}
  void test_case_end(const doctest::CurrentTestCaseStats&) override {}
  void test_case_exception(const doctest::TestCaseException&) override {}
  void subcase_start(const doctest::SubcaseSignature&) override {}
  void subcase_end() override {}
  void log_assert(const doctest::AssertData&) override {}
  void log_message(const doctest::MessageData&) override {}
  void test_case_skipped(const doctest::TestCaseData&) override {}
};

DOCTEST_REGISTER_LISTENER("vt-cross-first-decline", 1, FirstDeclineListener);

}  // namespace

TEST_CASE("attention-cross blocked: the FIRST decline of the process counts exactly ONE") {
  const FirstDeclineProbe& p = FirstDecline();
  // The instrument's own precondition, asserted before its reading. A listener
  // that never fired leaves a zeroed struct, which would otherwise read as a
  // device that was simply absent.
  REQUIRE(p.ran);
  if (!p.cuda) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  INFO("measured in test_run_start(), around the call that RESOLVES "
       "BlockedFallback()'s static: reset -> one declining AttentionCross -> read");
  INFO("probe error: " << p.error);
  REQUIRE(p.error.empty());
  // 2 is #1584's double count -- the resolver counted the decline and
  // `NoteOpDecline` counted it again. 0 is the opposite defect, a dropped
  // `NoteOpDecline`. Both break the same guarantee, so this is exact and not a
  // bound. This is the ONLY assertion in the tree that reds a FULL run when
  // `cuda_attention_cross.cu` goes back to `GetOpFallback`.
  CHECK(p.declines == 1);
}

TEST_CASE("attention-cross blocked: the provider is registered ABOVE vt-native") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // Two providers, and the blocked one first in the deterministic selection
  // order. If this inverts, every case below still passes numerically while
  // measuring the wrong kernel — which is the whole reason it is asserted.
  REQUIRE(vt::OpProviderCount(vt::OpId::kAttentionCross, DeviceType::kCUDA) == 2);
  CHECK(std::string(vt::OpProviderNameAt(vt::OpId::kAttentionCross, DeviceType::kCUDA, 0)) ==
        kBlockedProvider);
  CHECK(std::string(vt::OpProviderNameAt(vt::OpId::kAttentionCross, DeviceType::kCUDA, 1)) ==
        vt::kNativeProviderName);
}

TEST_CASE("attention-cross blocked: the DiT's own head_dim 64, with a ragged query tile") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // MiniMax-Music3's DiT is head_dim 64. Tq = 80 gives TWO query tiles at BR = 64
  // whose second one is RAGGED (80 = 64 + 16), and S = 200 gives 7 key tiles at
  // BC = 32 with a ragged last one (200 = 6*32 + 8) — so the tail guard on BOTH
  // axes and the cross-tile rescale all run together.
  const Geometry g{80, 200, 4, 4, 64};
  REQUIRE(BlockedShape(g));
  REQUIRE(g.tq % 64 != 0);
  REQUIRE(g.s % 32 != 0);
  RunGeometry("blocked d=64 ragged", g, 211u, /*bias_rows=*/0, DType::kF32, 2e-5);

  const Inputs in = MakeInputs(g, 211u, /*bias_rows=*/0);
  const CudaRun blocked = RunCuda(g, in, DType::kF32, /*disable_blocked=*/false);
  // TWO-SIDED. The positive half alone is an addition proof; the zero is what
  // makes it a routing proof.
  CHECK(std::string(blocked.selected) == kBlockedProvider);
  CHECK(blocked.declines == 0);
}

TEST_CASE("attention-cross blocked: a DENSE bias across tiles, on the blocked path") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // The bias arm has NO vLLM equivalent (its Triton prefill kernel carries no
  // additive bias), so it is the arm most likely to be dropped by a restructure.
  // Dense [Tq, S], multi-tile on both axes, so reading row 0 for every query or
  // indexing by the within-tile column is caught.
  const Geometry g{96, 150, 2, 2, 64};
  REQUIRE(BlockedShape(g));
  RunGeometry("blocked d=64 dense bias", g, 223u, /*bias_rows=*/96, DType::kF32, 2e-5);
  const Inputs in = MakeInputs(g, 223u, /*bias_rows=*/96);
  CHECK(RunCuda(g, in, DType::kF32, false).declines == 0);
}

TEST_CASE("attention-cross blocked: Hq > Hkv broadcasts on the blocked path too") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  const Geometry g{128, 100, 8, 2, 64};
  REQUIRE(BlockedShape(g));
  REQUIRE(g.hq > g.hk);
  RunGeometry("blocked gqa 8/2", g, 227u, /*bias_rows=*/0, DType::kF32, 2e-5);
  const Inputs in = MakeInputs(g, 227u, /*bias_rows=*/0);
  CHECK(RunCuda(g, in, DType::kF32, false).declines == 0);
}

TEST_CASE("attention-cross blocked: head_dim 128 -- LTX-2.5's stream, which its OWN suite cannot reach") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // WHY THIS CASE HAS TO BE HERE. LTX-2.5's device forward runs head_dim 128 in
  // production and therefore takes the blocked kernel, but every geometry in
  // `ltx2_goldens.inc` is REDUCED-DIMENSION -- `kLtx2Arch_attention_head_dim` is
  // 8 and `kLtx2VideoTokens` is 8 -- so every LTX-2.5 case DECLINES and its
  // suites are byte-identical under this change. Their green says nothing about
  // the path LTX-2.5 actually runs. That is the same defect this file's header
  // was written for, and this is where the coverage lives instead.
  //
  // The 32x16 tiling is the one the 48 KiB shared budget forces at head_dim 128,
  // and it is a DIFFERENT instantiation from the 64x32 one every case above
  // drives: QT and KT fall to 2 and DT rises to 16, so the register tile, the
  // cross-group reduction width and the output mapping are all different code.
  const Geometry g{64, 100, 2, 2, 128};
  REQUIRE(BlockedShape(g));
  REQUIRE(g.tq % 32 == 0);
  REQUIRE(g.s % 16 != 0);  // a ragged key tile at BC = 16
  RunGeometry("blocked d=128 dense bias", g, 269u, /*bias_rows=*/64, DType::kF32, 2e-5);
  const Inputs in = MakeInputs(g, 269u, /*bias_rows=*/64);
  const CudaRun blocked = RunCuda(g, in, DType::kF32, /*disable_blocked=*/false);
  CHECK(std::string(blocked.selected) == kBlockedProvider);
  CHECK(blocked.declines == 0);
  const CudaRun native = RunCuda(g, in, DType::kF32, /*disable_blocked=*/true);
  CHECK(std::string(native.selected) == vt::kNativeProviderName);
  const double cross = vllm_test::MaxAbsDiff(blocked.out, native.out.data(), native.out.size());
  MESSAGE("d=128 arm-vs-arm max|diff| = " << cross);
  CHECK(cross < 2e-5);
}

TEST_CASE("attention-cross blocked: the bf16 stream takes it at head_dim 64") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // The gate is on head_dim and query count, NOT on dtype: a bf16 caller with a
  // long query tile gets the blocked kernel too, and the staging tiles widen the
  // operand once instead of on every read.
  const Geometry g{64, 130, 2, 2, 64};
  REQUIRE(BlockedShape(g));
  RunGeometry("blocked bf16 d=64", g, 229u, /*bias_rows=*/1, DType::kBF16, 5e-3);
  const Inputs in = MakeInputs(g, 229u, /*bias_rows=*/1);
  CHECK(RunCuda(g, in, DType::kBF16, false).declines == 0);
}

TEST_CASE("attention-cross blocked: a fully masked key still degenerates, not NaNs") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // The same guarantee the warp kernel's case asserts, re-asserted on the
  // blocked path — where the max and the sum are reduced across EIGHT key groups
  // through shared memory, so an all-masked row travels a different route.
  const Geometry g{64, 130, 1, 1, 64};
  REQUIRE(BlockedShape(g));
  Inputs in = MakeInputs(g, 233u, /*bias_rows=*/1);
  for (int64_t j = 0; j < g.s; ++j) {
    in.bias[static_cast<size_t>(j)] = (j >= 70 && j < 100) ? -3.40282347e+38F : 0.0f;
  }
  const std::vector<float> want =
      Reference(g, in.query, in.key, in.value, &in.bias, 1, in.scale);
  const CudaRun got = RunCuda(g, in, DType::kF32, /*disable_blocked=*/false);
  CHECK(got.declines == 0);
  // MaxAbsDiff treats a non-finite operand as a FAILURE (#449), so a NaN cannot
  // read as agreement here.
  const double diff = vllm_test::MaxAbsDiff(got.out, want.data(), want.size());
  MESSAGE("blocked masked-tile max|diff| = " << diff);
  CHECK(diff < 2e-5);
}

TEST_CASE("attention-cross blocked: the DEPTH-DECODER shape declines, byte for byte") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // MiniMax-Music3's RVQ depth decoder calls this op with ONE query row against
  // its cached history (minimax_music3_depth_device.cpp). That is the regime the
  // warp-per-query kernel is the right shape for, and the blocked gate must
  // leave it alone. Not "within tolerance" — BYTE FOR BYTE, which is the only
  // claim worth making about a path that did not change.
  const Geometry g{1, 200, 4, 4, 64};
  REQUIRE(!BlockedShape(g));
  const Inputs in = MakeInputs(g, 239u, /*bias_rows=*/0);
  const CudaRun with = RunCuda(g, in, DType::kBF16, /*disable_blocked=*/false);
  const CudaRun without = RunCuda(g, in, DType::kBF16, /*disable_blocked=*/true);
  // ONE call, ONE decline: the provider was selected and forwarded.
  CHECK(with.declines == 1);
  // ... and this case reads 1 only because the process's FIRST decline counted
  // 1, which happened long before it ran. Assert the measurement that could
  // still see that, so a `-tc=` run of THIS case stays falsifiable (#1812).
  CHECK(FirstDecline().declines == 1);
  // With the provider disabled it is not selected at all, so nothing declines.
  CHECK(without.declines == 0);
  CHECK(std::string(without.selected) == vt::kNativeProviderName);
  REQUIRE(with.out.size() == without.out.size());
  size_t bitdiff = 0;
  for (size_t i = 0; i < with.out.size(); ++i) {
    uint32_t a = 0, b = 0;
    std::memcpy(&a, &with.out[i], sizeof(a));
    std::memcpy(&b, &without.out[i], sizeof(b));
    if (a != b) ++bitdiff;
  }
  MESSAGE("depth-decoder shape bitdiff = " << bitdiff << " / " << with.out.size());
  CHECK(bitdiff == 0);
}

TEST_CASE("attention-cross blocked: an unhandled head_dim declines once per call") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // head_dim 32 has no blocked tiling. The decline count is the behavioural
  // control the A/B rests on: it must move with the shape and with nothing else.
  const Geometry g{128, 100, 2, 2, 32};
  REQUIRE(!BlockedShape(g));
  const Inputs in = MakeInputs(g, 241u, /*bias_rows=*/0);
  const CudaRun r = RunCuda(g, in, DType::kF32, /*disable_blocked=*/false);
  CHECK(r.declines == 1);
  // Same order-proofing as the depth-decoder case above (#1812).
  CHECK(FirstDecline().declines == 1);
  const std::vector<float> want =
      Reference(g, in.query, in.key, in.value, nullptr, 0, in.scale);
  CHECK(vllm_test::MaxAbsDiff(r.out, want.data(), want.size()) < 2e-5);
}

TEST_CASE("attention-cross blocked: CATASTROPHIC CANCELLATION in the head-dim sum") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // THE CASE WITH TEETH, and the reason it has to exist. The restructure moves
  // the head-dim reduction from a 32-lane PAIRWISE butterfly to a SEQUENTIAL f32
  // accumulation over `head_dim` terms. On benign data a wide accumulator makes
  // that reordering unobservable: every other case in this file reports
  // max|diff| near 1e-6 and would report the same if the summation had not moved
  // at all.
  //
  // CONSTRUCTION. K alternates sign along the head dim while Q does not, so
  // consecutive products cancel: the leading term sums to EXACTLY zero over 64
  // terms, the partial sums reach ~2.6e5, and what survives is a random walk of
  // order 3e3. Pairwise and sequential summation genuinely differ there.
  //
  // THE SCALE IS PART OF THE CONSTRUCTION, and a first draft of this case got it
  // wrong in a way worth recording. At the natural 1/sqrt(D) the score SPREAD is
  // ~4e2, the softmax is a one-hot, both kernels return one row of V exactly and
  // BOTH report max|diff| = 0 -- a case that passes while measuring nothing. The
  // scale here puts the spread near unity so an error in the dot product
  // actually reaches the output.
  //
  // WHAT IS ASSERTED, and why it is three things over FOUR seeds. (1) The case is
  // ill-conditioned ENOUGH -- the EXISTING kernel must itself miss the f64
  // reference by more than the benign 2e-5 band, or the bounds below are a mute
  // switch. (2) Both arms stay inside one committed absolute bound. (3) The NEW
  // summation is within a committed RATIO of the old one, which is the actual
  // question a reordering raises and the one an absolute bound cannot answer.
  // Four seeds because a bound fitted to one sample is a bound fitted to one
  // sample.
  const Geometry g{64, 96, 2, 2, 64};
  REQUIRE(BlockedShape(g));
  double worst_blocked = 0.0, worst_native = 0.0, worst_ratio = 0.0;
  for (uint32_t seed : {251u, 263u, 271u, 281u}) {
    Inputs in = MakeInputs(g, seed, /*bias_rows=*/0);
    for (size_t i = 0; i < in.query.size(); ++i) {
      in.query[i] = 512.0f + 0.5f * in.query[i];
    }
    for (size_t i = 0; i < in.key.size(); ++i) {
      const size_t e = i % static_cast<size_t>(g.d);
      in.key[i] = ((e % 2 == 0) ? 1.0f : -1.0f) * (512.0f + 0.5f * in.key[i]);
    }
    in.scale = 1.0f / 1024.0f;
    const std::vector<float> want =
        Reference(g, in.query, in.key, in.value, nullptr, 0, in.scale);
    const CudaRun blocked = RunCuda(g, in, DType::kF32, /*disable_blocked=*/false);
    const CudaRun native = RunCuda(g, in, DType::kF32, /*disable_blocked=*/true);
    REQUIRE(blocked.declines == 0);
    REQUIRE(native.declines == 0);
    const double d_blocked = vllm_test::MaxAbsDiff(blocked.out, want.data(), want.size());
    const double d_native = vllm_test::MaxAbsDiff(native.out, want.data(), want.size());
    MESSAGE("cancellation seed=" << seed << ": blocked max|diff| = " << d_blocked
                                 << "  native max|diff| = " << d_native
                                 << "  ratio = " << (d_blocked / d_native));
    worst_blocked = worst_blocked > d_blocked ? worst_blocked : d_blocked;
    worst_native = worst_native > d_native ? worst_native : d_native;
    const double ratio = d_blocked / d_native;
    worst_ratio = worst_ratio > ratio ? worst_ratio : ratio;
  }
  MESSAGE("cancellation WORST: blocked = " << worst_blocked << "  native = " << worst_native
                                           << "  ratio = " << worst_ratio);
  CHECK(worst_native > 2e-5);  // the case is genuinely hard, not a mute switch
  CHECK(worst_native < kCancellationTol);
  CHECK(worst_blocked < kCancellationTol);
  CHECK(worst_ratio < kCancellationRatio);
}

TEST_CASE("attention-cross blocked: the same-binary A/B lever agrees with itself") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  // `VT_OP_PROVIDER_DISABLE=vt-cross-blocked` is what the speed measurement uses
  // to run both arms from one build, so the two arms have to be the same
  // function. Held at the op's committed f32 band, NOT bit-identical — the
  // reduction order differs by construction and this file says so rather than
  // implying otherwise.
  const Geometry g{128, 200, 2, 2, 64};
  REQUIRE(BlockedShape(g));
  const Inputs in = MakeInputs(g, 257u, /*bias_rows=*/0);
  const CudaRun blocked = RunCuda(g, in, DType::kF32, false);
  const CudaRun native = RunCuda(g, in, DType::kF32, true);
  CHECK(blocked.declines == 0);
  CHECK(native.declines == 0);
  CHECK(std::string(blocked.selected) == kBlockedProvider);
  CHECK(std::string(native.selected) == vt::kNativeProviderName);
  const double cross = vllm_test::MaxAbsDiff(blocked.out, native.out.data(), native.out.size());
  MESSAGE("A/B arm-vs-arm max|diff| = " << cross);
  CHECK(cross < 2e-5);
}
