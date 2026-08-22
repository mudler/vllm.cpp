// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Dense causal attention op unit tests. Formula reference:
// .agents/specs/qwen36-forward-notes.md §5 (Qwen3NextAttention core: causal GQA
// scaled-dot-product). Golden coverage (qk-norm + RoPE + gate composed) lives in
// tests/parity/test_op_parity.cpp (dense_attention_*); these tests pin
// hand-computed values and the corners the random goldens do not isolate:
// causal masking, GQA head→kv mapping, scale, bf16, and the composed core
// (RmsNorm + RoPE + Attention + sigmoid gate) integration.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::AttentionArgs;
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

Tensor F32(std::vector<float>& v, const std::vector<int64_t>& shape) {
  return Contig(v.data(), DType::kF32, Cpu(), shape);
}
}  // namespace

TEST_CASE("attention tiny hand-checkable causal case (Hq=Hk=1, D=2)") {
  // T=2, one head, head_dim=2, scale=1, causal.
  //   q = [[1,0],[0,1]], k = [[1,0],[0,1]], v = [[1,2],[3,4]]
  // query 0 (causal → only key 0): softmax({q0·k0})={1} → out0 = v0 = [1,2].
  // query 1 (keys 0,1): scores = {q1·k0, q1·k1} = {0,1};
  //   softmax({0,1}) = {1/(1+e), e/(1+e)} = {0.26894, 0.73106};
  //   out1 = 0.26894*[1,2] + 0.73106*[3,4] = [2.46211, 3.46211].
  std::vector<float> q = {1, 0, 0, 1};
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {1, 2, 3, 4};
  std::vector<float> out(4, 0.0f);
  Tensor tq = F32(q, {2, 1, 2}), tk = F32(k, {2, 1, 2}), tv = F32(v, {2, 1, 2});
  Tensor to = F32(out, {2, 1, 2});
  Queue qq = Q();
  vt::Attention(qq, to, tq, tk, tv, AttentionArgs{1.0f, /*causal=*/true});
  CHECK(out[0] == doctest::Approx(1.0f));
  CHECK(out[1] == doctest::Approx(2.0f));
  CHECK(out[2] == doctest::Approx(2.46211f).epsilon(1e-4));
  CHECK(out[3] == doctest::Approx(3.46211f).epsilon(1e-4));
}

TEST_CASE("attention causal mask hides future keys from query 0") {
  // query 0 must ignore key 1 entirely — even a huge v[1] leaves out0 == v0.
  std::vector<float> q = {5, 0, 0, 5};
  std::vector<float> k = {1, 0, 0, 1};
  std::vector<float> v = {1, 1, 1000, 1000};
  std::vector<float> out(4, 0.0f);
  Tensor tq = F32(q, {2, 1, 2}), tk = F32(k, {2, 1, 2}), tv = F32(v, {2, 1, 2});
  Tensor to = F32(out, {2, 1, 2});
  Queue qq = Q();
  vt::Attention(qq, to, tq, tk, tv, AttentionArgs{1.0f, true});
  CHECK(out[0] == doctest::Approx(1.0f));  // untouched by the huge future v
  CHECK(out[1] == doctest::Approx(1.0f));
}

TEST_CASE("attention non-causal attends to all keys") {
  // Non-causal, zero queries (D=1) → uniform softmax over both keys → mean of v.
  // With causal masking query 0 would see only key 0 (→2); non-causal it sees
  // both → mean({2,8}) = 5 for BOTH queries.
  std::vector<float> q = {0, 0};     // [T=2, H=1, D=1]
  std::vector<float> k = {1, 5};     // [T=2, H=1, D=1]
  std::vector<float> v = {2, 8};     // [T=2, H=1, D=1]
  std::vector<float> out(2, 0.0f);
  Tensor tq = F32(q, {2, 1, 1}), tk = F32(k, {2, 1, 1}), tv = F32(v, {2, 1, 1});
  Tensor to = F32(out, {2, 1, 1});
  Queue qq = Q();
  vt::Attention(qq, to, tq, tk, tv, AttentionArgs{1.0f, /*causal=*/false});
  CHECK(out[0] == doctest::Approx(5.0f));
  CHECK(out[1] == doctest::Approx(5.0f));
}

TEST_CASE("attention GQA broadcast maps q-head h to kv-head h/(Hq/Hk)") {
  // Hq=4, Hk=2, D=1, T=1, single key → softmax=1 so out == the mapped kv v.
  // q-heads 0,1 → kv0 (v=7); q-heads 2,3 → kv1 (v=9).
  std::vector<float> q = {1, 2, 3, 4};  // [T=1, Hq=4, D=1]; values irrelevant
  std::vector<float> k = {1, 1};        // [T=1, Hk=2, D=1]
  std::vector<float> v = {7, 9};        // [T=1, Hk=2, D=1]
  std::vector<float> out(4, 0.0f);
  Tensor tq = F32(q, {1, 4, 1}), tk = F32(k, {1, 2, 1}), tv = F32(v, {1, 2, 1});
  Tensor to = F32(out, {1, 4, 1});
  Queue qq = Q();
  vt::Attention(qq, to, tq, tk, tv, AttentionArgs{1.0f, true});
  CHECK(out[0] == doctest::Approx(7.0f));  // head 0 → kv 0
  CHECK(out[1] == doctest::Approx(7.0f));  // head 1 → kv 0
  CHECK(out[2] == doctest::Approx(9.0f));  // head 2 → kv 1
  CHECK(out[3] == doctest::Approx(9.0f));  // head 3 → kv 1
}

TEST_CASE("attention scale changes the softmax sharpness") {
  // T=2, 1 head, D=1. query 1 sees keys {k0=0, k1=1}·q1=1 → scores {0, scale}.
  // With scale=0 the two keys tie → out = mean(v) = 1.5.
  std::vector<float> q = {0, 1};
  std::vector<float> k = {0, 1};
  std::vector<float> v = {1, 2};
  std::vector<float> out(2, 0.0f);
  Tensor tq = F32(q, {2, 1, 1}), tk = F32(k, {2, 1, 1}), tv = F32(v, {2, 1, 1});
  Tensor to = F32(out, {2, 1, 1});
  Queue qq = Q();
  // scale must be > 0 (validation); use a tiny scale to approach the uniform tie.
  vt::Attention(qq, to, tq, tk, tv, AttentionArgs{1e-6f, true});
  CHECK(out[1] == doctest::Approx(1.5f).epsilon(1e-4));  // near-uniform over {1,2}
}

TEST_CASE("attention bf16 in/out matches the f32 golden within bf16 eps") {
  // Same numbers as the tiny hand-check, stored bf16 in and out.
  auto bf = [](float x) { return vt::F32ToBF16(x); };
  std::vector<uint16_t> q = {bf(1), bf(0), bf(0), bf(1)};
  std::vector<uint16_t> k = {bf(1), bf(0), bf(0), bf(1)};
  std::vector<uint16_t> v = {bf(1), bf(2), bf(3), bf(4)};
  std::vector<uint16_t> out(4, 0);
  Tensor tq = Tensor::Contiguous(q.data(), DType::kBF16, Cpu(), {2, 1, 2});
  Tensor tk = Tensor::Contiguous(k.data(), DType::kBF16, Cpu(), {2, 1, 2});
  Tensor tv = Tensor::Contiguous(v.data(), DType::kBF16, Cpu(), {2, 1, 2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {2, 1, 2});
  Queue qq = Q();
  vt::Attention(qq, to, tq, tk, tv, AttentionArgs{1.0f, true});
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(1.0f).epsilon(0.02));
  CHECK(vt::BF16ToF32(out[2]) == doctest::Approx(2.46211f).epsilon(0.02));
  CHECK(vt::BF16ToF32(out[3]) == doctest::Approx(3.46211f).epsilon(0.02));
}

TEST_CASE("attention validates shapes/args") {
  std::vector<float> buf(8, 0.0f);
  std::vector<float> out(8, 0.0f);
  Queue qq = Q();
  // Hq not a multiple of Hk (3 q-heads, 2 kv-heads).
  {
    Tensor tq = F32(buf, {1, 3, 1});
    std::vector<float> kv(2, 0.0f);
    Tensor tk = F32(kv, {1, 2, 1}), tv = F32(kv, {1, 2, 1});
    std::vector<float> o(3, 0.0f);
    Tensor to = F32(o, {1, 3, 1});
    CHECK_THROWS_AS(vt::Attention(qq, to, tq, tk, tv, AttentionArgs{1.0f, true}),
                    std::runtime_error);
  }
  // scale must be > 0.
  {
    Tensor tq = F32(buf, {1, 1, 2}), tk = F32(buf, {1, 1, 2}), tv = F32(buf, {1, 1, 2});
    Tensor to = F32(out, {1, 1, 2});
    CHECK_THROWS_AS(vt::Attention(qq, to, tq, tk, tv, AttentionArgs{0.0f, true}),
                    std::runtime_error);
  }
  // head_dim mismatch between q and k.
  {
    Tensor tq = F32(buf, {1, 1, 2});
    std::vector<float> k4(4, 0.0f);
    Tensor tk = F32(k4, {1, 1, 4}), tv = F32(k4, {1, 1, 4});
    Tensor to = F32(out, {1, 1, 2});
    CHECK_THROWS_AS(vt::Attention(qq, to, tq, tk, tv, AttentionArgs{1.0f, true}),
                    std::runtime_error);
  }
}

// Composed core (qk-norm + partial RoPE + attention + sigmoid gate) integration:
// mirrors how Task 4's full-attn layer and RunDenseAttention assemble the op.
// pos=0 makes RoPE the identity; gate=0 makes sigmoid(gate)=0.5, so the gated
// output is exactly half the (normed) attention output — a clean hand-check that
// the gate multiply and the RmsNorm/RoPE composition wire up correctly.
TEST_CASE("dense-attn core: qk-norm + RoPE(pos=0) + attention + sigmoid gate") {
  const int64_t T = 2, H = 1, D = 4;
  // Per-head gemma-RMSNorm over D with weight 0 → normalizes to unit-RMS and
  // multiplies by (1+0)=1. RoPE at pos=0 is identity. So attention runs on the
  // RMS-normalized q/k; gate=0 halves the result.
  std::vector<float> q = {1, 0, 0, 0, 0, 1, 0, 0};  // [T,H,D]
  std::vector<float> k = {1, 0, 0, 0, 0, 1, 0, 0};
  std::vector<float> v = {2, 2, 2, 2, 4, 4, 4, 4};
  std::vector<float> qnw(D, 0.0f), knw(D, 0.0f);
  std::vector<int32_t> pos = {0, 0};
  Queue qq = Q();

  // qk-norm (gemma) over D, viewing [T*H, D].
  std::vector<float> qn(T * H * D, 0.0f), kn(T * H * D, 0.0f);
  Tensor tq2 = F32(q, {T * H, D}), tk2 = F32(k, {T * H, D});
  Tensor tqn = F32(qn, {T * H, D}), tkn = F32(kn, {T * H, D});
  Tensor tqnw = F32(qnw, {D}), tknw = F32(knw, {D});
  vt::RmsNorm(qq, tqn, tq2, tqnw, vt::RmsNormArgs{1e-6f, /*gemma=*/true});
  vt::RmsNorm(qq, tkn, tk2, tknw, vt::RmsNormArgs{1e-6f, true});
  // q row [1,0,0,0] → RMS = sqrt(1/4)=0.5 → normalized [2,0,0,0].
  CHECK(qn[0] == doctest::Approx(2.0f));

  // RoPE pos=0 identity on [T,H,D].
  Tensor tqn3 = F32(qn, {T, H, D}), tkn3 = F32(kn, {T, H, D});
  Tensor tpos = Tensor::Contiguous(pos.data(), DType::kI32, Cpu(), {T});
  vt::RopeNeox(qq, tqn3, tkn3, tpos, vt::RopeArgs{10000.0f, 4});
  CHECK(qn[0] == doctest::Approx(2.0f));  // unchanged at pos 0

  // Attention (scale=1, causal). query 0 → only key 0 → out0 = v0 = [2,2,2,2].
  std::vector<float> attn(T * H * D, 0.0f);
  Tensor tattn = F32(attn, {T, H, D}), tv3 = F32(v, {T, H, D});
  vt::Attention(qq, tattn, tqn3, tkn3, tv3, AttentionArgs{1.0f, true});
  CHECK(attn[0] == doctest::Approx(2.0f));  // out0 == v0

  // Sigmoid gate at gate=0 → 0.5 → gated0 == 1.0.
  const float gate = 1.0f / (1.0f + std::exp(0.0f));  // sigmoid(0) = 0.5
  CHECK(attn[0] * gate == doctest::Approx(1.0f));
}

// ===========================================================================
// CUDA parity: the CUDA attention kernel must match the CPU reference to 1e-4
// (f32) on random inputs. Guarded by HasCuda so CPU-only builds skip cleanly.
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

std::vector<float> RandF32(size_t n, uint32_t seed) {
  // Deterministic LCG in [-2,2); avoids <random> divergence across libstdc++.
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

void RunCudaCase(int64_t T, int64_t Hq, int64_t Hk, int64_t D, float scale, bool causal,
                 uint32_t seed) {
  auto q = RandF32(static_cast<size_t>(T * Hq * D), seed);
  auto k = RandF32(static_cast<size_t>(T * Hk * D), seed + 1);
  auto v = RandF32(static_cast<size_t>(T * Hk * D), seed + 2);

  // CPU reference.
  std::vector<float> cpu(static_cast<size_t>(T * Hq * D), 0.0f);
  Tensor cq = MakeT(q.data(), DType::kF32, Cpu(), {T, Hq, D});
  Tensor ck = MakeT(k.data(), DType::kF32, Cpu(), {T, Hk, D});
  Tensor cv = MakeT(v.data(), DType::kF32, Cpu(), {T, Hk, D});
  Tensor co = MakeT(cpu.data(), DType::kF32, Cpu(), {T, Hq, D});
  Queue cpuq = Q();
  vt::Attention(cpuq, co, cq, ck, cv, AttentionArgs{scale, causal});

  // CUDA.
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {T, Hq, D}, q.data());
  DeviceTensor dk(gpu, g.q, DType::kF32, {T, Hk, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kF32, {T, Hk, D}, v.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {T, Hq, D});
  vt::Attention(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(),
                AttentionArgs{scale, causal});
  std::vector<float> got(static_cast<size_t>(T * Hq * D), 0.0f);
  dout.Download(g.q, got.data());

  for (size_t i = 0; i < cpu.size(); ++i)
    CHECK(got[i] == doctest::Approx(cpu[i]).epsilon(1e-4));
}

}  // namespace

TEST_CASE("attention CUDA matches CPU (causal, GQA, real head_dim)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping CUDA attention parity");
    return;
  }
  RunCudaCase(/*T=*/9, /*Hq=*/16, /*Hk=*/2, /*D=*/256, /*scale=*/std::pow(256.0f, -0.5f),
              /*causal=*/true, /*seed=*/1234);
  RunCudaCase(/*T=*/6, /*Hq=*/4, /*Hk=*/2, /*D=*/8, 0.35f, true, 77);
  RunCudaCase(/*T=*/5, /*Hq=*/2, /*Hk=*/1, /*D=*/16, 0.25f, /*causal=*/false, 999);
}

// --- AttentionDenseFlash's head_dim contract (#1544) -------------------------
//
// The op advertised `head_dim <= 256` while requesting `2*64*d*sizeof(Tin)` bytes
// of dynamic shared memory with no `cudaFuncSetAttribute` anywhere in
// src/vt/cuda/, so the driver's default 48 KiB cap made the real ceiling 192 in
// bf16 and 96 in f32. A caller at head_dim 256 got a bare launch failure naming
// nothing it could do instead. These cases pin the arithmetic that fixes it, and
// they run with no GPU because that arithmetic is a pure host function — which is
// the whole reason it lives in include/vt/ops.h rather than inside the .cu.
TEST_CASE("attention-dense-flash: the advertised head_dim is the launchable one") {
  // The tile the CTA stages: K and V, 64 columns each, head_dim wide.
  CHECK(vt::AttentionDenseFlashSmemBytes(64, 2) == 16384);
  CHECK(vt::AttentionDenseFlashSmemBytes(96, 2) == 24576);   // Muse Glimmer, #1545
  CHECK(vt::AttentionDenseFlashSmemBytes(192, 2) == 49152);  // exactly the cap
  CHECK(vt::AttentionDenseFlashSmemBytes(256, 2) == 65536);  // over it
  CHECK(vt::AttentionDenseFlashSmemBytes(96, 4) == 49152);   // exactly the cap
  CHECK(vt::AttentionDenseFlashSmemBytes(192, 4) == 98304);  // Kimi, over it
  CHECK(vt::AttentionDenseFlashSmemBytes(256, 4) == 131072);

  // The honest bounds. These are the widths the launcher now refuses above.
  CHECK(vt::AttentionDenseFlashMaxHeadDim(/*bf16=*/2) == 192);
  CHECK(vt::AttentionDenseFlashMaxHeadDim(/*f32=*/4) == 96);

  // Neither is the 256 the op used to advertise, and that gap IS the defect.
  CHECK(vt::AttentionDenseFlashMaxHeadDim(2) < vt::kAttentionDenseMaxHeadDim);
  CHECK(vt::AttentionDenseFlashMaxHeadDim(4) < vt::kAttentionDenseMaxHeadDim);

  // The bound is INCLUSIVE at the cap and exclusive one element past it. head_dim
  // 192 in bf16 sits exactly on 49152 and launches today, so a bound that refused
  // it would be a regression rather than a repair.
  CHECK(vt::AttentionDenseFlashSmemBytes(vt::AttentionDenseFlashMaxHeadDim(2), 2) <=
        vt::kCudaDefaultDynamicSmemBytes);
  CHECK(vt::AttentionDenseFlashSmemBytes(vt::AttentionDenseFlashMaxHeadDim(2) + 1, 2) >
        vt::kCudaDefaultDynamicSmemBytes);
  CHECK(vt::AttentionDenseFlashSmemBytes(vt::AttentionDenseFlashMaxHeadDim(4), 4) <=
        vt::kCudaDefaultDynamicSmemBytes);
  CHECK(vt::AttentionDenseFlashSmemBytes(vt::AttentionDenseFlashMaxHeadDim(4) + 1, 4) >
        vt::kCudaDefaultDynamicSmemBytes);

  // The register blocking still bounds it from the other side: a hypothetical
  // 1-byte input would clear the shared-memory cap far past what the kernel can
  // hold in registers, and the bound must report the register limit there.
  CHECK(vt::AttentionDenseFlashMaxHeadDim(1) == vt::kAttentionDenseMaxHeadDim);
}

TEST_CASE("attention-dense-flash: an over-cap head_dim is REFUSED, naming the rung") {
  // This one needs a device: the refusal lives in the CUDA launcher, and nothing
  // on a CPU-only box executes it. Loud on purpose — a quiet skip here would let
  // the launcher lose the bound with every gate still green (spec R1, #1573).
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; the AttentionDenseFlash head_dim refusal is PENDING");
    return;
  }
  const int64_t T = 4, Hq = 2, Hk = 2, D = 256;  // f32 => a 128 KiB K/V tile
  auto q = RandF32(static_cast<size_t>(T * Hq * D), 5150);
  auto k = RandF32(static_cast<size_t>(T * Hk * D), 5151);
  auto v = RandF32(static_cast<size_t>(T * Hk * D), 5152);
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {T, Hq, D}, q.data());
  DeviceTensor dk(gpu, g.q, DType::kF32, {T, Hk, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kF32, {T, Hk, D}, v.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {T, Hq, D});
  bool threw = false;
  std::string what;
  try {
    vt::AttentionDenseFlash(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(),
                            AttentionArgs{std::pow(256.0f, -0.5f), /*causal=*/false});
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }
  CHECK(threw);
  // Refusing is only half of it. The message must name the rung that DOES serve
  // this width, or the caller is left where the opaque launch error left them.
  CHECK(what.find("AttentionDenseFast") != std::string::npos);
  CHECK(what.find("head_dim") != std::string::npos);

  // ...and the honest bound is not a blanket refusal: head_dim 96 in f32 lands
  // exactly on the cap and must still run.
  const int64_t Dok = 96;
  auto q2 = RandF32(static_cast<size_t>(T * Hq * Dok), 5153);
  auto k2 = RandF32(static_cast<size_t>(T * Hk * Dok), 5154);
  auto v2 = RandF32(static_cast<size_t>(T * Hk * Dok), 5155);
  DeviceTensor dq2(gpu, g.q, DType::kF32, {T, Hq, Dok}, q2.data());
  DeviceTensor dk2(gpu, g.q, DType::kF32, {T, Hk, Dok}, k2.data());
  DeviceTensor dv2(gpu, g.q, DType::kF32, {T, Hk, Dok}, v2.data());
  DeviceTensor dout2(gpu, g.q, DType::kF32, {T, Hq, Dok});
  vt::AttentionDenseFlash(g.q, dout2.tensor(), dq2.tensor(), dk2.tensor(),
                          dv2.tensor(),
                          AttentionArgs{std::pow(96.0f, -0.5f), /*causal=*/false});
  std::vector<float> got(static_cast<size_t>(T * Hq * Dok), 0.0f);
  dout2.Download(g.q, got.data());
  CHECK(std::isfinite(got[0]));
}
