// vt::MlaDecodeAttention unit tests (MLA campaign W4).
//
// Upstream test modules ported per .agents/test-porting.md:
//   * vllm/tests/kernels/attention/test_mla_decode_cpu.py @ pin e24d1b24 — the
//     DEVICE-INDEPENDENT numerical oracle the spike named as the one to port
//     FIRST (.agents/specs/mla-deepseek-campaign.md §7). Its `ref_mla`
//     (`:13-33`) gathers the paged latent, flattens it to `seq_len` rows, takes
//     `v = kv[:, :, :v_head_dim]` and runs one plain
//     `scaled_dot_product_attention(..., enable_gqa=True)`. `RefMla` below is
//     that function, re-expressed: a TWO-PASS softmax (max, then exp-sum, then
//     the weighted sum) — deliberately a DIFFERENT algorithm from the streaming
//     online-softmax both of our impls use, so a bug in the streaming rescale
//     cannot hide behind a matching reference.
//     Its parametrization is ported too: `bs=4`, `mean_seq_len=256`, `h_q=16`,
//     `d=576`, `dv=512`, `block_size=16`, and BOTH `varlen` arms (`:36-45`).
//     Its NaN-PADDING trick (`:71-73`: every cache row past `seq_len` is set to
//     NaN, then `assert not out_mla.isnan().any(), "Likely read out of bounds"`)
//     is ported verbatim — it is a real out-of-bounds detector and it is kept
//     IN ADDITION to the compute-sanitizer run, not instead of it.
//   * vllm/tests/v1/attention/test_mla_backends.py — "MLA backend correctness vs
//     a reference across batch/seq shapes"; its shape sweep is what the ragged /
//     multi-block / split-boundary / single-block cases below cover.
//
// Kernels under port: vllm/v1/attention/ops/triton_decode_attention.py:278-458
//   (`_fwd_grouped_kernel_stage1`, IS_MLA branch) + `:575-639`
//   (`_fwd_kernel_stage2`), driven by
//   vllm/v1/attention/backends/mla/triton_mla.py:189-260 (`forward_mqa`).
//
// GATE DIMENSIONS ARE THE REAL ONES. DeepSeek-V2-Lite (confirmed at W0):
// head_size 576 = kv_lora_rank 512 + qk_rope_head_dim 64, num_kv_heads 1,
// num_attention_heads 16, block_size 16, and a `scale` that carries the YaRN
// mscale^2 correction. DeepSeek-V3's 128-head shape is covered too (it exercises
// head_tiles > 1, which V2-Lite's 16 heads does not).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::MlaDecodeAttentionArgs;
using vt::Queue;
using vt::Tensor;

namespace {

// ─── DeepSeek-V2-Lite geometry (config.json, confirmed at W0) ───────────────
constexpr int kKvLoraRank = 512;   // == v_head_dim
constexpr int kRopeDim = 64;       // qk_rope_head_dim
constexpr int kHeadSize = kKvLoraRank + kRopeDim;  // 576
constexpr int kBlockSize = 16;     // page = 16 * 576 * 2 B = 18,432 B
constexpr int kHeadsLite = 16;     // num_attention_heads

// `self.scale` as MLAAttentionImpl builds it: head_size^-0.5 times the YaRN
// mscale^2 correction. DeepSeek-V2-Lite's rope_scaling is
// {factor: 40, mscale: 0.707, mscale_all_dim: 0.707}, and upstream's
// `yarn_get_mscale(scale, mscale) = 0.1 * mscale * log(scale) + 1.0` gives
// mscale = 0.1*0.707*ln(40) + 1 = 1.26082..., so the multiplier is mscale^2.
// The kernel knows nothing about mscale — it is a plain float — but gating with
// the REAL value (not 1/sqrt(576)) is what makes this the production geometry.
double LiteScale() {
  const double mscale = 0.1 * 0.707 * std::log(40.0) + 1.0;
  return (1.0 / std::sqrt(static_cast<double>(kHeadSize))) * mscale * mscale;
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

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

// Deterministic uniform noise in [-1, 1] (the same LCG the W3 tests use).
std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
  }
  return v;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::F32ToBF16(v[i]);
  return o;
}

std::vector<float> FromBf16(const std::vector<uint16_t>& v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::BF16ToF32(v[i]);
  return o;
}

// ─── the ported oracle: test_mla_decode_cpu.py:13-33 `ref_mla` ──────────────
// Two-pass softmax (max, exp-sum, weighted sum) over the GATHERED latent rows.
// V is the leading `v_head_dim` slice of the same row (`:29 v = kv[:,:,:dv]`).
// Deliberately NOT the streaming form our two impls use.
void RefMla(std::vector<float>& out, std::vector<float>* lse, const std::vector<float>& q,
            const std::vector<float>& cache, const std::vector<int32_t>& block_table,
            int max_blocks, const std::vector<int32_t>& seq_lens, int heads, int head_size,
            int v_head_dim, int block_size, float scale) {
  const int bs = static_cast<int>(seq_lens.size());
  out.assign(static_cast<size_t>(bs) * heads * v_head_dim, 0.0f);
  if (lse != nullptr) lse->assign(static_cast<size_t>(bs) * heads, 0.0f);
  std::vector<double> logits;
  for (int b = 0; b < bs; ++b) {
    const int n = seq_lens[static_cast<size_t>(b)];
    for (int h = 0; h < heads; ++h) {
      const float* qp = q.data() + (static_cast<size_t>(b) * heads + h) * head_size;
      logits.assign(static_cast<size_t>(n), 0.0);
      double mx = -std::numeric_limits<double>::infinity();
      for (int j = 0; j < n; ++j) {
        const int blk = block_table[static_cast<size_t>(b) * max_blocks + j / block_size];
        const float* kp =
            cache.data() + (static_cast<size_t>(blk) * block_size + j % block_size) * head_size;
        double dot = 0.0;
        for (int d = 0; d < head_size; ++d) dot += static_cast<double>(qp[d]) * kp[d];
        logits[static_cast<size_t>(j)] = dot * scale;
        if (logits[static_cast<size_t>(j)] > mx) mx = logits[static_cast<size_t>(j)];
      }
      double denom = 0.0;
      for (int j = 0; j < n; ++j) {
        logits[static_cast<size_t>(j)] = std::exp(logits[static_cast<size_t>(j)] - mx);
        denom += logits[static_cast<size_t>(j)];
      }
      float* op = out.data() + (static_cast<size_t>(b) * heads + h) * v_head_dim;
      for (int j = 0; j < n; ++j) {
        const int blk = block_table[static_cast<size_t>(b) * max_blocks + j / block_size];
        const float* kp =
            cache.data() + (static_cast<size_t>(blk) * block_size + j % block_size) * head_size;
        const double w = logits[static_cast<size_t>(j)] / denom;
        for (int d = 0; d < v_head_dim; ++d) op[d] += static_cast<float>(w * kp[d]);
      }
      if (lse != nullptr) {
        (*lse)[static_cast<size_t>(b) * heads + h] =
            n > 0 ? static_cast<float>(mx + std::log(denom))
                  : -std::numeric_limits<float>::infinity();
      }
    }
  }
}

// One test case: geometry + host buffers. `max_blocks` block-table columns per
// request; block ids are SHUFFLED across requests so a kernel that assumes
// contiguous pages fails (upstream's test uses a plain arange, which cannot
// catch that).
struct Case {
  int bs = 0;
  int heads = 0;
  int head_size = kHeadSize;
  int v_head_dim = kKvLoraRank;
  int block_size = kBlockSize;
  int max_blocks = 0;
  int num_blocks = 0;
  std::vector<int32_t> seq_lens;
  std::vector<int32_t> block_table;
  std::vector<float> cache;  // [num_blocks, block_size, head_size]
  std::vector<float> q;      // [bs, heads, head_size]
};

Case MakeCase(const std::vector<int32_t>& seq_lens, int heads, uint32_t seed,
              int head_size = kHeadSize, int v_head_dim = kKvLoraRank,
              int block_size = kBlockSize) {
  Case c;
  c.bs = static_cast<int>(seq_lens.size());
  c.heads = heads;
  c.head_size = head_size;
  c.v_head_dim = v_head_dim;
  c.block_size = block_size;
  c.seq_lens = seq_lens;
  int max_len = 1;
  for (int32_t s : seq_lens) max_len = std::max(max_len, static_cast<int>(s));
  c.max_blocks = (max_len + block_size - 1) / block_size;
  c.num_blocks = c.bs * c.max_blocks + 3;  // +3 pages that no request ever names
  c.block_table.resize(static_cast<size_t>(c.bs) * c.max_blocks);
  // Reverse-interleaved page assignment: request b's page i is
  // num_blocks-1 - (b*max_blocks + i), so pages are non-contiguous AND
  // descending — a stride assumption anywhere is caught.
  for (int b = 0; b < c.bs; ++b) {
    for (int i = 0; i < c.max_blocks; ++i) {
      c.block_table[static_cast<size_t>(b) * c.max_blocks + i] =
          c.num_blocks - 1 - (b * c.max_blocks + i);
    }
  }
  c.cache = RandF32(static_cast<size_t>(c.num_blocks) * block_size * head_size, seed);
  c.q = RandF32(static_cast<size_t>(c.bs) * heads * head_size, seed + 7919u);
  // test_mla_decode_cpu.py:71-73 — poison every cache row past this request's
  // seq_len with NaN, so ANY read past the sequence end poisons the output.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int b = 0; b < c.bs; ++b) {
    for (int j = seq_lens[static_cast<size_t>(b)]; j < c.max_blocks * block_size; ++j) {
      const int blk = c.block_table[static_cast<size_t>(b) * c.max_blocks + j / block_size];
      float* row =
          c.cache.data() + (static_cast<size_t>(blk) * block_size + j % block_size) * head_size;
      for (int d = 0; d < head_size; ++d) row[d] = nan;
    }
  }
  return c;
}

// Run the op on the CPU backend.
void RunCpu(const Case& c, std::vector<float>& out, std::vector<float>* lse,
            const MlaDecodeAttentionArgs& args) {
  Queue q = CpuQ();
  out.assign(static_cast<size_t>(c.bs) * c.heads * c.v_head_dim, 0.0f);
  std::vector<float> lse_buf(static_cast<size_t>(c.bs) * c.heads, 0.0f);
  auto cache = c.cache;
  auto qq = c.q;
  auto bt = c.block_table;
  auto sl = c.seq_lens;
  Tensor t_out = Contig(out.data(), DType::kF32, Cpu(), {c.bs, c.heads, c.v_head_dim});
  Tensor t_q = Contig(qq.data(), DType::kF32, Cpu(), {c.bs, c.heads, c.head_size});
  Tensor t_c =
      Contig(cache.data(), DType::kF32, Cpu(), {c.num_blocks, c.block_size, c.head_size});
  Tensor t_bt = Contig(bt.data(), DType::kI32, Cpu(), {c.bs, c.max_blocks});
  Tensor t_sl = Contig(sl.data(), DType::kI32, Cpu(), {c.bs});
  Tensor t_lse = Contig(lse_buf.data(), DType::kF32, Cpu(), {c.bs, c.heads});
  vt::MlaDecodeAttention(q, t_out, lse != nullptr ? &t_lse : nullptr, t_q, t_c, t_bt, t_sl,
                         args);
  if (lse != nullptr) *lse = lse_buf;
}

// Run the op on CUDA. `bf16` converts every float operand to bf16 first.
void RunCuda(const Case& c, std::vector<float>& out, std::vector<float>* lse,
             const MlaDecodeAttentionArgs& args, bool bf16) {
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);
  const std::vector<int64_t> out_shape{c.bs, c.heads, c.v_head_dim};
  const std::vector<int64_t> q_shape{c.bs, c.heads, c.head_size};
  const std::vector<int64_t> c_shape{c.num_blocks, c.block_size, c.head_size};
  const DType dt = bf16 ? DType::kBF16 : DType::kF32;

  // Poison the output with NaN so a kernel that fails to write an element is
  // caught (the assertion below rejects any NaN) — the positive "this ran and
  // wrote every element" signal.
  const size_t out_n = static_cast<size_t>(c.bs) * c.heads * c.v_head_dim;
  std::vector<float> nanf(out_n, std::numeric_limits<float>::quiet_NaN());
  std::vector<uint16_t> nanb = ToBf16(nanf);

  const std::vector<uint16_t> qb = ToBf16(c.q);
  const std::vector<uint16_t> cb = ToBf16(c.cache);
  DeviceTensor d_out(b, g.q, dt, out_shape, bf16 ? static_cast<const void*>(nanb.data())
                                                 : static_cast<const void*>(nanf.data()));
  DeviceTensor d_q(b, g.q, dt, q_shape,
                   bf16 ? static_cast<const void*>(qb.data())
                        : static_cast<const void*>(c.q.data()));
  DeviceTensor d_c(b, g.q, dt, c_shape,
                   bf16 ? static_cast<const void*>(cb.data())
                        : static_cast<const void*>(c.cache.data()));
  DeviceTensor d_bt(b, g.q, DType::kI32, {c.bs, c.max_blocks}, c.block_table.data());
  DeviceTensor d_sl(b, g.q, DType::kI32, {c.bs}, c.seq_lens.data());
  DeviceTensor d_lse(b, g.q, DType::kF32, {c.bs, c.heads});

  vt::MlaDecodeAttention(g.q, d_out.tensor(), lse != nullptr ? &d_lse.tensor() : nullptr,
                         d_q.tensor(), d_c.tensor(), d_bt.tensor(), d_sl.tensor(), args);
  b.Synchronize(g.q);
  if (bf16) {
    std::vector<uint16_t> raw(out_n);
    d_out.Download(g.q, raw.data());
    out = FromBf16(raw);
  } else {
    out.assign(out_n, 0.0f);
    d_out.Download(g.q, out.data());
  }
  if (lse != nullptr) {
    lse->assign(static_cast<size_t>(c.bs) * c.heads, 0.0f);
    d_lse.Download(g.q, lse->data());
  }
}

// Max |a-b| plus an explicit "no NaN anywhere" assertion — the ported
// `assert not out_mla.isnan().any(), "Likely read out of bounds"`.
double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE_MESSAGE(!std::isnan(a[i]), "NaN in result — likely read out of bounds");
    REQUIRE_MESSAGE(!std::isnan(b[i]), "NaN in reference — likely read out of bounds");
    worst = std::max(worst, std::abs(static_cast<double>(a[i]) - b[i]));
  }
  return worst;
}

}  // namespace

// ───────────────────────────────────────────────────────────────────────────
// CPU op vs the ported `ref_mla` oracle, at the REAL V2-Lite dimensions.
// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("mla_decode CPU matches ref_mla at DeepSeek-V2-Lite dims (varlen=False)") {
  // test_mla_decode_cpu.py:36-45 — bs=4, mean_seq_len=256, h_q=16, d=576,
  // dv=512, block_size=16, varlen=False.
  const Case c = MakeCase({256, 256, 256, 256}, kHeadsLite, 11u);
  MlaDecodeAttentionArgs args;
  args.scale = static_cast<float>(LiteScale());
  std::vector<float> got;
  std::vector<float> got_lse;
  RunCpu(c, got, &got_lse, args);

  std::vector<float> want;
  std::vector<float> want_lse;
  RefMla(want, &want_lse, c.q, c.cache, c.block_table, c.max_blocks, c.seq_lens, c.heads,
         c.head_size, c.v_head_dim, c.block_size, args.scale);

  CHECK(MaxAbsDiff(got, want) < 1e-4);
  CHECK(MaxAbsDiff(got_lse, want_lse) < 1e-3);
}

TEST_CASE("mla_decode CPU matches ref_mla at V2-Lite dims (varlen=True, ragged)") {
  // The varlen=True arm. Lengths deliberately straddle every boundary that
  // matters: 1 (single token), 15 (part block), 16 (exact block), 17 (block+1),
  // 255/256/257 (around the split heuristic's 512/2 work unit), 300.
  const Case c = MakeCase({1, 15, 16, 17, 255, 256, 257, 300}, kHeadsLite, 23u);
  MlaDecodeAttentionArgs args;
  args.scale = static_cast<float>(LiteScale());
  std::vector<float> got;
  std::vector<float> got_lse;
  RunCpu(c, got, &got_lse, args);

  std::vector<float> want;
  std::vector<float> want_lse;
  RefMla(want, &want_lse, c.q, c.cache, c.block_table, c.max_blocks, c.seq_lens, c.heads,
         c.head_size, c.v_head_dim, c.block_size, args.scale);

  CHECK(MaxAbsDiff(got, want) < 1e-4);
  CHECK(MaxAbsDiff(got_lse, want_lse) < 1e-3);
}

TEST_CASE("mla_decode CPU single block, single token") {
  const Case c = MakeCase({1}, kHeadsLite, 31u);
  REQUIRE(c.max_blocks == 1);
  MlaDecodeAttentionArgs args;
  args.scale = static_cast<float>(LiteScale());
  std::vector<float> got;
  RunCpu(c, got, nullptr, args);
  // With one key the softmax is 1.0, so the output IS that key's V slice.
  const int blk = c.block_table[0];
  for (int h = 0; h < c.heads; ++h) {
    for (int d = 0; d < c.v_head_dim; d += 97) {
      CHECK(got[static_cast<size_t>(h) * c.v_head_dim + d] ==
            doctest::Approx(c.cache[static_cast<size_t>(blk) * c.block_size * c.head_size + d])
                .epsilon(1e-6));
    }
  }
}

// ───────────────────────────────────────────────────────────────────────────
// CUDA (the two-stage split-KV port) vs the CPU reference.
// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("CUDA mla_decode matches the CPU reference at V2-Lite dims (f32, ragged)") {
  if (!HasCuda()) return;
  const Case c = MakeCase({1, 15, 16, 17, 255, 256, 257, 300}, kHeadsLite, 23u);
  MlaDecodeAttentionArgs args;
  args.scale = static_cast<float>(LiteScale());
  args.max_seq_len = 300;  // exercise the ComputeNumKvSplits heuristic path

  std::vector<float> want;
  std::vector<float> want_lse;
  RunCpu(c, want, &want_lse, args);
  std::vector<float> got;
  std::vector<float> got_lse;
  RunCuda(c, got, &got_lse, args, /*bf16=*/false);

  CHECK(MaxAbsDiff(got, want) < 2e-4);
  CHECK(MaxAbsDiff(got_lse, want_lse) < 2e-3);
}

TEST_CASE("CUDA mla_decode matches the CPU reference at V2-Lite dims (bf16, ragged)") {
  if (!HasCuda()) return;
  // bf16 is the production dtype (triton_mla.py:82 supported_dtypes). The CPU
  // reference runs on the SAME bf16-rounded values, so the only difference is
  // f32 summation order plus the bf16 rounding of the stored output.
  Case c = MakeCase({1, 15, 16, 17, 255, 256, 257, 300}, kHeadsLite, 23u);
  for (auto& x : c.cache) {
    if (!std::isnan(x)) x = vt::BF16ToF32(vt::F32ToBF16(x));
  }
  for (auto& x : c.q) x = vt::BF16ToF32(vt::F32ToBF16(x));
  MlaDecodeAttentionArgs args;
  args.scale = static_cast<float>(LiteScale());
  args.max_seq_len = 300;

  std::vector<float> want;
  RunCpu(c, want, nullptr, args);
  std::vector<float> got;
  RunCuda(c, got, nullptr, args, /*bf16=*/true);
  // bf16 has ~8 bits of mantissa; |out| <= 1 here, so one ulp is ~2^-8.
  CHECK(MaxAbsDiff(got, want) < 6e-3);
}

TEST_CASE("CUDA mla_decode: every num_kv_splits agrees (split boundaries)") {
  if (!HasCuda()) return;
  // The split partition is cdiv(seq_len, S) — the boundary cases are S == 1
  // (no split), S == seq_len (one key per split), S > seq_len (empty splits that
  // BOTH stages must skip), and S that does not divide seq_len.
  const Case c = MakeCase({1, 16, 17, 63, 64, 65, 300}, kHeadsLite, 47u);
  MlaDecodeAttentionArgs base;
  base.scale = static_cast<float>(LiteScale());

  std::vector<float> want;
  std::vector<float> want_lse;
  RunCpu(c, want, &want_lse, base);

  for (int splits : {1, 2, 3, 4, 5, 8, 16, 17, 64, 300, 512}) {
    MlaDecodeAttentionArgs args = base;
    args.num_kv_splits = splits;
    std::vector<float> got;
    std::vector<float> got_lse;
    RunCuda(c, got, &got_lse, args, /*bf16=*/false);
    CAPTURE(splits);
    CHECK(MaxAbsDiff(got, want) < 2e-4);
    CHECK(MaxAbsDiff(got_lse, want_lse) < 2e-3);
  }
}

TEST_CASE("CUDA mla_decode is run-to-run bit-reproducible") {
  if (!HasCuda()) return;
  const Case c = MakeCase({37, 128, 511, 512}, kHeadsLite, 59u);
  MlaDecodeAttentionArgs args;
  args.scale = static_cast<float>(LiteScale());
  args.num_kv_splits = 4;  // fixed split count → fixed f32 summation order

  std::vector<float> a;
  std::vector<float> b;
  RunCuda(c, a, nullptr, args, /*bf16=*/false);
  for (int i = 0; i < 4; ++i) {
    RunCuda(c, b, nullptr, args, /*bf16=*/false);
    REQUIRE(a.size() == b.size());
    for (size_t k = 0; k < a.size(); ++k) {
      REQUIRE(a[k] == b[k]);  // BIT-exact, no atomicAdd anywhere
    }
  }
}

TEST_CASE("CUDA mla_decode at DeepSeek-V3 dims (128 heads → head_tiles > 1)") {
  if (!HasCuda()) return;
  // V3/R1: num_attention_heads = 128 with the same 576/512 latent geometry. 128
  // heads is 8 tiles of BLOCK_H=16, which V2-Lite's 16 heads never exercises.
  // Dimensions are free even though V3's weights are not (spike §5.2).
  const Case c = MakeCase({1, 33, 200}, 128, 71u);
  MlaDecodeAttentionArgs args;
  args.scale = static_cast<float>(LiteScale());
  args.max_seq_len = 200;

  std::vector<float> want;
  RunCpu(c, want, nullptr, args);
  std::vector<float> got;
  RunCuda(c, got, nullptr, args, /*bf16=*/false);
  CHECK(MaxAbsDiff(got, want) < 2e-4);
}

TEST_CASE("CUDA mla_decode with a head count that does not fill a BLOCK_H tile") {
  if (!HasCuda()) return;
  // Hq = 1, 3, 17 — the mask_h path (triton_decode_attention.py:321-323).
  for (int heads : {1, 3, 17}) {
    const Case c = MakeCase({5, 40}, heads, 83u);
    MlaDecodeAttentionArgs args;
    args.scale = static_cast<float>(LiteScale());
    std::vector<float> want;
    RunCpu(c, want, nullptr, args);
    std::vector<float> got;
    RunCuda(c, got, nullptr, args, /*bf16=*/false);
    CAPTURE(heads);
    CHECK(MaxAbsDiff(got, want) < 2e-4);
  }
}

TEST_CASE("CUDA mla_decode: non-V2-Lite geometry still matches (generality)") {
  if (!HasCuda()) return;
  // head_size 288 / v_head_dim 256 is upstream's OTHER hardcoded MLA tile
  // (triton_decode_attention.py:497-499), and block_size 32 is a legal
  // MultipleOf(16) page (triton_mla.py:96-103).
  const Case c = MakeCase({7, 96, 129}, 8, 97u, /*head_size=*/288, /*v_head_dim=*/256,
                          /*block_size=*/32);
  MlaDecodeAttentionArgs args;
  args.scale = static_cast<float>(1.0 / std::sqrt(288.0));
  std::vector<float> want;
  RunCpu(c, want, nullptr, args);
  std::vector<float> got;
  RunCuda(c, got, nullptr, args, /*bf16=*/false);
  CHECK(MaxAbsDiff(got, want) < 2e-4);
}

// ───────────────────────────────────────────────────────────────────────────
// Contract validation (ops.cpp) — the shapes MLA makes easy to get wrong.
// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("mla_decode rejects malformed operands") {
  Queue q = CpuQ();
  const int bs = 2;
  const int heads = 4;
  std::vector<float> out(static_cast<size_t>(bs) * heads * kKvLoraRank, 0.0f);
  std::vector<float> qq(static_cast<size_t>(bs) * heads * kHeadSize, 0.1f);
  std::vector<float> cache(static_cast<size_t>(4) * kBlockSize * kHeadSize, 0.1f);
  std::vector<int32_t> bt{0, 1, 2, 3};
  std::vector<int32_t> sl{16, 16};
  std::vector<float> lse(static_cast<size_t>(bs) * heads, 0.0f);

  Tensor t_out = Contig(out.data(), DType::kF32, Cpu(), {bs, heads, kKvLoraRank});
  Tensor t_q = Contig(qq.data(), DType::kF32, Cpu(), {bs, heads, kHeadSize});
  Tensor t_c = Contig(cache.data(), DType::kF32, Cpu(), {4, kBlockSize, kHeadSize});
  Tensor t_bt = Contig(bt.data(), DType::kI32, Cpu(), {bs, 2});
  Tensor t_sl = Contig(sl.data(), DType::kI32, Cpu(), {bs});
  Tensor t_lse = Contig(lse.data(), DType::kF32, Cpu(), {bs, heads});
  MlaDecodeAttentionArgs args;
  args.scale = 0.04f;

  CHECK_NOTHROW(vt::MlaDecodeAttention(q, t_out, &t_lse, t_q, t_c, t_bt, t_sl, args));

  SUBCASE("scale must be set") {
    MlaDecodeAttentionArgs bad = args;
    bad.scale = 0.0f;
    CHECK_THROWS_AS(vt::MlaDecodeAttention(q, t_out, nullptr, t_q, t_c, t_bt, t_sl, bad),
                    std::runtime_error);
  }
  SUBCASE("a 4-D K/V-split cache is refused (MLA has no K/V axis)") {
    Tensor bad = Contig(cache.data(), DType::kF32, Cpu(), {2, 2, kBlockSize, kHeadSize});
    CHECK_THROWS_AS(vt::MlaDecodeAttention(q, t_out, nullptr, t_q, bad, t_bt, t_sl, args),
                    std::runtime_error);
  }
  SUBCASE("cache entry width must equal the query head_size") {
    Tensor bad = Contig(cache.data(), DType::kF32, Cpu(), {4, kBlockSize, kKvLoraRank});
    CHECK_THROWS_AS(vt::MlaDecodeAttention(q, t_out, nullptr, t_q, bad, t_bt, t_sl, args),
                    std::runtime_error);
  }
  SUBCASE("v_head_dim must not exceed head_size") {
    std::vector<float> big(static_cast<size_t>(bs) * heads * (kHeadSize + 8), 0.0f);
    Tensor bad = Contig(big.data(), DType::kF32, Cpu(), {bs, heads, kHeadSize + 8});
    CHECK_THROWS_AS(vt::MlaDecodeAttention(q, bad, nullptr, t_q, t_c, t_bt, t_sl, args),
                    std::runtime_error);
  }
  SUBCASE("block_table / seq_lens must be i32") {
    std::vector<float> f(4, 0.0f);
    Tensor bad = Contig(f.data(), DType::kF32, Cpu(), {bs, 2});
    CHECK_THROWS_AS(vt::MlaDecodeAttention(q, t_out, nullptr, t_q, t_c, bad, t_sl, args),
                    std::runtime_error);
  }
  SUBCASE("lse must be f32 [batch, heads]") {
    Tensor bad = Contig(lse.data(), DType::kF32, Cpu(), {bs, heads - 1});
    CHECK_THROWS_AS(vt::MlaDecodeAttention(q, t_out, &bad, t_q, t_c, t_bt, t_sl, args),
                    std::runtime_error);
  }
  SUBCASE("query / cache / out must share one float dtype") {
    std::vector<uint16_t> h(cache.size(), 0);
    Tensor bad = Contig(h.data(), DType::kBF16, Cpu(), {4, kBlockSize, kHeadSize});
    CHECK_THROWS_AS(vt::MlaDecodeAttention(q, t_out, nullptr, t_q, bad, t_bt, t_sl, args),
                    std::runtime_error);
  }
  SUBCASE("negative num_kv_splits is refused") {
    MlaDecodeAttentionArgs bad = args;
    bad.num_kv_splits = -1;
    CHECK_THROWS_AS(vt::MlaDecodeAttention(q, t_out, nullptr, t_q, t_c, t_bt, t_sl, bad),
                    std::runtime_error);
  }
}
