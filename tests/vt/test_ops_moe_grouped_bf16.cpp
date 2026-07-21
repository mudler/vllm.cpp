// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// W5 (Qwen3-Coder-30B-A3B, `Qwen3MoeForCausalLM`) — the BF16 grouped-MoE GEMM
// (vt::MoeGroupedGemmBf16), the dtype-native analog of the NVFP4 grouped GEMM
// validated by test_ops_moe_grouped.cpp. Mirrors that file's contract exactly:
// the grouped GEMM's output row p must equal the SINGLE-EXPERT reference (per-row
// f32 dot against expert expert_ids[p]'s bf16 [K,N] weight, act row row_map[p]) —
// i.e. exactly what the per-expert bf16 MoE reference loop (qwen3_5.cpp
// `ExpertMlp`) computes for that (token, expert) pair. Covers all three launch
// regimes the dispatcher selects between (cuda_matmul_nvfp4.cu `LaunchGroupedBf16`):
//   * P < kTileMinRows(32)            -> naive one-thread-per-output kernel;
//   * kTileMinRows <= P <= kMoeDecodeMaxP(512) -> BM=16 decode WMMA tile;
//   * P > kMoeDecodeMaxP              -> BM=64 prefill WMMA tile;
// plus both output dtypes (f32 for gate/up, bf16 for down) and the identity
// (row_map == nullptr) routing the down projection uses. CUDA-only (skips cleanly
// with no GPU) — the CPU/GGUF MoE keeps the per-expert MatmulBf16 reference.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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
    t_ = MakeTensor(p_, dt, q.device, shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void* ptr() { return p_; }
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

void CheckClose(const std::vector<float>& got, const std::vector<float>& want, float atol,
                float rtol) {
  REQUIRE(got.size() == want.size());
  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float tol = atol + rtol * std::fabs(want[i]);
    if (!(std::fabs(got[i] - want[i]) <= tol)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(want[first_bad]);
  }
  CHECK(bad == 0);
}

std::vector<uint16_t> RandomBf16(size_t numel, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<uint16_t> v(numel);
  for (auto& x : v) x = vt::F32ToBF16(dist(rng));
  return v;
}

// Runs vt::MoeGroupedGemmBf16 over E synthetic bf16 [K,N] experts and checks every
// output row against the single-expert f32 reference dot. `use_row_map` false
// exercises the identity routing (row(p) == p) the down projection uses; `bf16_out`
// exercises the bf16 output dtype (down) vs f32 (gate/up).
void RunGroupedBf16Case(int64_t e_count, int64_t t_rows, int64_t top_k, int64_t k_dim,
                        int64_t n_cols, uint32_t seed, bool use_row_map, bool bf16_out) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t P = t_rows * top_k;
  // With identity routing the activation IS the per-pair buffer, so it has P rows.
  const int64_t act_rows = use_row_map ? t_rows : P;

  std::vector<std::vector<uint16_t>> w(static_cast<size_t>(e_count));
  for (int64_t e = 0; e < e_count; ++e)
    w[static_cast<size_t>(e)] =
        RandomBf16(static_cast<size_t>(k_dim * n_cols), seed + 17 * static_cast<uint32_t>(e));
  const auto act = RandomBf16(static_cast<size_t>(act_rows * k_dim), seed + 991);

  std::vector<int32_t> expert_ids(static_cast<size_t>(P));
  std::vector<int32_t> row_map(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p) {
    row_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / top_k);
    // Deterministic pseudo-routing; co-prime stride so experts get ragged counts.
    expert_ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 7 + 3) % e_count);
  }

  // Reference: out[p,n] = sum_k f32(act[row(p),k]) * f32(W_e[k,n]) in f32 — the
  // per-expert bf16 MoE row math (ExpertMlp's MatmulF32/MatmulBf16 contract).
  std::vector<float> ref(static_cast<size_t>(P * n_cols), 0.0f);
  for (int64_t p = 0; p < P; ++p) {
    const int64_t e = expert_ids[static_cast<size_t>(p)];
    const int64_t r = use_row_map ? row_map[static_cast<size_t>(p)] : p;
    const auto& we = w[static_cast<size_t>(e)];
    for (int64_t n = 0; n < n_cols; ++n) {
      float acc = 0.0f;
      for (int64_t kk = 0; kk < k_dim; ++kk)
        acc += vt::BF16ToF32(act[static_cast<size_t>(r * k_dim + kk)]) *
               vt::BF16ToF32(we[static_cast<size_t>(kk * n_cols + n)]);
      ref[static_cast<size_t>(p * n_cols + n)] = acc;
    }
  }

  QueueGuard gq(gpu);
  DeviceTensor dact(gpu, gq.q, DType::kBF16, {act_rows, k_dim}, act.data());
  DeviceTensor deids(gpu, gq.q, DType::kI32, {P}, expert_ids.data());
  DeviceTensor drow(gpu, gq.q, DType::kI32, {P}, row_map.data());

  std::vector<std::unique_ptr<DeviceTensor>> wbufs;
  std::vector<int64_t> wptrs(static_cast<size_t>(e_count));
  for (int64_t e = 0; e < e_count; ++e) {
    wbufs.push_back(std::make_unique<DeviceTensor>(gpu, gq.q, DType::kBF16,
                                                   std::vector<int64_t>{k_dim, n_cols},
                                                   w[static_cast<size_t>(e)].data()));
    wptrs[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(wbufs.back()->ptr());
  }
  DeviceTensor dwp(gpu, gq.q, DType::kI64, {e_count}, wptrs.data());

  DeviceTensor dout(gpu, gq.q, bf16_out ? DType::kBF16 : DType::kF32, {P, n_cols});
  vt::MoeGroupedGemmBf16(gq.q, dout.tensor(), dact.tensor(), deids.tensor(),
                         use_row_map ? &drow.tensor() : nullptr, dwp.tensor());

  std::vector<float> got(static_cast<size_t>(P * n_cols));
  if (bf16_out) {
    std::vector<uint16_t> raw(static_cast<size_t>(P * n_cols));
    dout.Download(gq.q, raw.data());
    for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(raw[i]);
  } else {
    dout.Download(gq.q, got.data());
  }
  // f32 accumulation over K bf16 products: tolerance scales with K (the WMMA tile
  // and the naive kernel accumulate in a different ORDER than the host reference).
  // bf16 output additionally costs one 8-bit-mantissa rounding of the result.
  const float rtol = bf16_out ? 8e-3f : 2e-3f;
  const float atol = rtol * static_cast<float>(k_dim) * 0.05f;
  CheckClose(got, ref, atol, rtol);
}

// Runs the same grouped GEMM `reps` times and returns true iff every run is
// BIT-IDENTICAL. The split-K decode path reduces its per-split partials in a
// fixed ascending order (never atomicAdd), so greedy decode stays reproducible.
bool GroupedBf16Bitwise(int64_t e_count, int64_t t_rows, int64_t top_k, int64_t k_dim,
                        int64_t n_cols, uint32_t seed, int reps) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  const int64_t P = t_rows * top_k;
  std::vector<std::vector<uint16_t>> w(static_cast<size_t>(e_count));
  for (int64_t e = 0; e < e_count; ++e)
    w[static_cast<size_t>(e)] =
        RandomBf16(static_cast<size_t>(k_dim * n_cols), seed + 17 * static_cast<uint32_t>(e));
  const auto act = RandomBf16(static_cast<size_t>(t_rows * k_dim), seed + 991);
  std::vector<int32_t> expert_ids(static_cast<size_t>(P)), row_map(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p) {
    row_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / top_k);
    expert_ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 7 + 3) % e_count);
  }

  QueueGuard gq(gpu);
  DeviceTensor dact(gpu, gq.q, DType::kBF16, {t_rows, k_dim}, act.data());
  DeviceTensor deids(gpu, gq.q, DType::kI32, {P}, expert_ids.data());
  DeviceTensor drow(gpu, gq.q, DType::kI32, {P}, row_map.data());
  std::vector<std::unique_ptr<DeviceTensor>> wbufs;
  std::vector<int64_t> wptrs(static_cast<size_t>(e_count));
  for (int64_t e = 0; e < e_count; ++e) {
    wbufs.push_back(std::make_unique<DeviceTensor>(gpu, gq.q, DType::kBF16,
                                                   std::vector<int64_t>{k_dim, n_cols},
                                                   w[static_cast<size_t>(e)].data()));
    wptrs[static_cast<size_t>(e)] = reinterpret_cast<int64_t>(wbufs.back()->ptr());
  }
  DeviceTensor dwp(gpu, gq.q, DType::kI64, {e_count}, wptrs.data());
  DeviceTensor dout(gpu, gq.q, DType::kF32, {P, n_cols});

  std::vector<float> first, cur(static_cast<size_t>(P * n_cols));
  for (int i = 0; i < reps; ++i) {
    vt::MoeGroupedGemmBf16(gq.q, dout.tensor(), dact.tensor(), deids.tensor(), &drow.tensor(),
                           dwp.tensor());
    dout.Download(gq.q, cur.data());
    if (i == 0)
      first = cur;
    else if (cur != first)
      return false;
  }
  return true;
}

}  // namespace

// P = 6 < kTileMinRows(32) -> naive one-thread-per-output kernel, f32 out (gate/up).
TEST_CASE("CUDA moe_grouped_gemm_bf16 naive path (small P) matches the per-expert reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  RunGroupedBf16Case(/*e_count=*/5, /*t_rows=*/3, /*top_k=*/2, /*k_dim=*/64, /*n_cols=*/8,
                     /*seed=*/5000, /*use_row_map=*/true, /*bf16_out=*/false);
}

// P = 40 (kTileMinRows <= P <= kMoeDecodeMaxP) -> BM=16 decode WMMA tile. K=80 is a
// multiple of 16 but NOT of BK=32 (partial last K-tile); N=130 crosses the BN=64
// tile boundary unevenly. Mirrors the NVFP4 tiled-path case's awkward shapes.
TEST_CASE("CUDA moe_grouped_gemm_bf16 decode WMMA tile (BM=16) matches the per-expert reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  RunGroupedBf16Case(/*e_count=*/7, /*t_rows=*/20, /*top_k=*/2, /*k_dim=*/80, /*n_cols=*/130,
                     /*seed=*/6000, /*use_row_map=*/true, /*bf16_out=*/false);
}

// P = 1024 > kMoeDecodeMaxP(512) -> BM=64 prefill WMMA tile.
TEST_CASE("CUDA moe_grouped_gemm_bf16 prefill WMMA tile (BM=64) matches the per-expert reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  RunGroupedBf16Case(/*e_count=*/9, /*t_rows=*/128, /*top_k=*/8, /*k_dim=*/96, /*n_cols=*/70,
                     /*seed=*/7000, /*use_row_map=*/true, /*bf16_out=*/false);
}

// W6 PIPELINED prefill tile (BM=64, BN=128, BK=32, 3-stage cp.async). Selected
// only when the activation/weight row pitches are 8-bf16 multiples
// (`Bf16PipeShapeOk`) — the cases above deliberately use ragged pitches (n=70,
// n=130) and therefore still cover the W5 fallback tile. Here K=264 is a multiple
// of 8 but NOT of BK=32 (partial last K-tile, exercising the cp.async `zfill`
// tail) and N=200 is a multiple of 8 but NOT of BN=128 (partial last N-tile).
TEST_CASE("CUDA moe_grouped_gemm_bf16 pipelined prefill tile matches the per-expert reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  RunGroupedBf16Case(/*e_count=*/9, /*t_rows=*/128, /*top_k=*/8, /*k_dim=*/264, /*n_cols=*/200,
                     /*seed=*/7100, /*use_row_map=*/true, /*bf16_out=*/false);
}

// W6 PIPELINED decode tile (BM=16, BN=128, BK=32, 3-stage), aligned pitches, plus
// the identity row-map + bf16-out (down-projection) call shape on the same tile.
TEST_CASE("CUDA moe_grouped_gemm_bf16 pipelined decode tile matches the per-expert reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  RunGroupedBf16Case(/*e_count=*/7, /*t_rows=*/20, /*top_k=*/2, /*k_dim=*/264, /*n_cols=*/200,
                     /*seed=*/7200, /*use_row_map=*/true, /*bf16_out=*/false);
  RunGroupedBf16Case(/*e_count=*/6, /*t_rows=*/16, /*top_k=*/8, /*k_dim=*/128, /*n_cols=*/256,
                     /*seed=*/7300, /*use_row_map=*/false, /*bf16_out=*/true);
}

// W6 DETERMINISTIC SPLIT-K on the small-P (decode) naive path: P=6 with N=8 gives
// only 6 thread blocks, so MoeSplitKCount picks splits = K/kMoeSplitKMinChunk(256)
// = 4 -> four f32 partials reduced in fixed ascending split order. Also asserts
// the split reduction is RUN-TO-RUN BIT-REPRODUCIBLE (no atomicAdd), which the
// greedy token-exact gate depends on.
TEST_CASE("CUDA moe_grouped_gemm_bf16 split-K decode path matches the reference and is exact") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  RunGroupedBf16Case(/*e_count=*/5, /*t_rows=*/3, /*top_k=*/2, /*k_dim=*/1024, /*n_cols=*/8,
                     /*seed=*/7400, /*use_row_map=*/true, /*bf16_out=*/false);
  CHECK(GroupedBf16Bitwise(/*e_count=*/5, /*t_rows=*/3, /*top_k=*/2, /*k_dim=*/1024, /*n_cols=*/8,
                           /*seed=*/7400, /*reps=*/3));
}

// Identity routing (row_map == nullptr) + bf16 output — the DOWN projection's exact
// call shape (act = the per-pair silu buffer, one act row per output row).
TEST_CASE("CUDA moe_grouped_gemm_bf16 identity row-map + bf16 out matches the reference") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping");
    return;
  }
  // Naive regime (P=12) and decode-tile regime (P=128), both bf16 out.
  RunGroupedBf16Case(/*e_count=*/4, /*t_rows=*/6, /*top_k=*/2, /*k_dim=*/48, /*n_cols=*/32,
                     /*seed=*/8000, /*use_row_map=*/false, /*bf16_out=*/true);
  RunGroupedBf16Case(/*e_count=*/6, /*t_rows=*/16, /*top_k=*/8, /*k_dim=*/64, /*n_cols=*/96,
                     /*seed=*/8100, /*use_row_map=*/false, /*bf16_out=*/true);
}
