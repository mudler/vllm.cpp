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
