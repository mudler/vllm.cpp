// MLA chunked-context prefill unit tests (MLA campaign W5) — the metadata
// builder, vt::GatherMlaCache, vt::MergeAttnStates, and the whole loop.
//
// Upstream test modules ported per .agents/test-porting.md:
//   * vllm/tests/v1/attention/test_mla_backends.py @ pin e24d1b24 — the MLA
//     backend correctness module. Its chunked-prefill shapes are what the
//     multi-chunk / chunk-boundary cases below cover; the campaign spec assigns
//     it to W4 (decode) and W5 (prefill).
//   * vllm/tests/v1/attention/test_mla_prefill_quant_output.py — the MLA prefill
//     output-vs-reference module (its fp8 arms are unreachable on sm_121, see
//     test_ops_mla_prefill.cpp).
//
// Kernels/logic under port:
//   mla_attention.py:1422-1451 (workspace sizing), :1667-1745 + :1837-1855 (the
//   chunk grid), :2094-2199 `_compute_prefill_context`, :2344-2425 `forward_mha`;
//   csrc/libtorch_stable/cache_kernels.cu:992-1064 (the gather) and
//   csrc/libtorch_stable/attention/merge_attn_states.cu:18-192 (the merge).
//
// THE ORACLE IS A DIFFERENT ALGORITHM. The whole point of the chunked loop is
// that attending to [context ++ new] in one shot must equal merging the
// per-chunk partials by their log-sum-exps. So the gate computes the SINGLE-SHOT
// answer — one causal varlen attention over the full concatenated K/V, in
// double precision — and requires the chunked path to reproduce it. A bug in the
// chunk grid, the gather indexing, the LSE algebra or the ping-pong shows up as
// a mismatch; none of them can hide, because the oracle never chunks.
//
// ADVERSARIAL BLOCK TABLES. Every paged case here uses a REVERSE-INTERLEAVED
// page assignment (request b's logical page p maps to a physical block far from
// b*max_blocks + p), so any assumption that pages are contiguous or ascending
// fails loudly. Upstream's own tests use `arange` block tables, which cannot
// catch that class of bug — this is the same hardening W4 applied to decode.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/layers/attention/mla_chunked_context.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm::mla::BuildMlaChunkedContext;
using vllm::mla::DetermineChunkedPrefillWorkspaceSize;
using vllm::mla::ForwardMlaPrefillMha;
using vllm::mla::MlaChunkDeviceMetadata;
using vllm::mla::MlaContextChunkKv;
using vllm::mla::MlaPrefillContextBuffers;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

// ─── DeepSeek-V2-Lite geometry (confirmed at W0) ────────────────────────────
constexpr int kKvLoraRank = 512;
constexpr int kQkRope = 64;
constexpr int kLatentDim = kKvLoraRank + kQkRope;  // 576, the MLA cache entry
constexpr int kQkNope = 128;
constexpr int kQkHeadDim = kQkNope + kQkRope;  // 192
constexpr int kVHeadDim = 128;
constexpr int kBlockSize = 16;

double LiteScale() {
  const double mscale = 0.1 * 0.707 * std::log(40.0) + 1.0;
  return (1.0 / std::sqrt(static_cast<double>(kQkHeadDim))) * mscale * mscale;
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
  void Upload(Queue& q, const void* src) { b_.Copy(q, p_, src, bytes_); }
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

std::vector<float> Bf16Round(const std::vector<float>& v) { return FromBf16(ToBf16(v)); }

std::vector<int32_t> Cumsum(const std::vector<int32_t>& lens) {
  std::vector<int32_t> cu(lens.size() + 1, 0);
  for (size_t i = 0; i < lens.size(); ++i) cu[i + 1] = cu[i] + lens[i];
  return cu;
}

// The independent double-precision two-pass oracle (same one the sibling
// test_ops_mla_prefill.cpp uses; FlashAttention's causal is BOTTOM-RIGHT
// aligned).
void RefPrefill(std::vector<float>& out, const std::vector<float>& q,
                const std::vector<float>& k, const std::vector<float>& v,
                const std::vector<int32_t>& cu_q, const std::vector<int32_t>& cu_k, int heads,
                int dqk, int dv, double scale, bool causal) {
  const int nreq = static_cast<int>(cu_q.size()) - 1;
  const int total_q = cu_q.back();
  out.assign(static_cast<size_t>(total_q) * heads * dv, 0.0f);
  std::vector<double> logits;
  for (int b = 0; b < nreq; ++b) {
    const int q0 = cu_q[static_cast<size_t>(b)];
    const int lq = cu_q[static_cast<size_t>(b) + 1] - q0;
    const int k0 = cu_k[static_cast<size_t>(b)];
    const int lk = cu_k[static_cast<size_t>(b) + 1] - k0;
    for (int iq = 0; iq < lq; ++iq) {
      const int t = q0 + iq;
      const int visible = causal ? std::min(lk, std::max(0, iq + (lk - lq) + 1)) : lk;
      for (int h = 0; h < heads; ++h) {
        const float* qp = q.data() + (static_cast<size_t>(t) * heads + h) * dqk;
        logits.assign(static_cast<size_t>(std::max(visible, 0)), 0.0);
        double mx = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < visible; ++j) {
          const float* kp = k.data() + (static_cast<size_t>(k0 + j) * heads + h) * dqk;
          double dot = 0.0;
          for (int d = 0; d < dqk; ++d) dot += static_cast<double>(qp[d]) * kp[d];
          dot *= scale;
          logits[static_cast<size_t>(j)] = dot;
          mx = std::max(mx, dot);
        }
        double denom = 0.0;
        for (int j = 0; j < visible; ++j) {
          logits[static_cast<size_t>(j)] = std::exp(logits[static_cast<size_t>(j)] - mx);
          denom += logits[static_cast<size_t>(j)];
        }
        std::vector<double> acc(static_cast<size_t>(dv), 0.0);
        for (int j = 0; j < visible; ++j) {
          const float* vp = v.data() + (static_cast<size_t>(k0 + j) * heads + h) * dv;
          const double p = logits[static_cast<size_t>(j)];
          for (int d = 0; d < dv; ++d) acc[static_cast<size_t>(d)] += p * vp[d];
        }
        float* op = out.data() + (static_cast<size_t>(t) * heads + h) * dv;
        for (int d = 0; d < dv; ++d) {
          op[d] = denom > 0.0 ? static_cast<float>(acc[static_cast<size_t>(d)] / denom) : 0.0f;
        }
      }
    }
  }
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE_FALSE(std::isnan(a[i]));
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return m;
}

// ─── the "kv_b_proj" up-projection, mla_attention.py:2160-2170 ──────────────
// kv_c (the leading kv_lora_rank of the latent) -> [num_heads, qk_nope + v_head_dim],
// split into k_nope / v, then k = concat(k_nope, the SHARED rope part broadcast
// across heads). The weight is fixed test data; the real GEMM is W6's job.
struct KvBProj {
  int heads;
  std::vector<float> w;  // [kv_lora_rank, heads * (qk_nope + v_head_dim)]
  explicit KvBProj(int h, uint32_t seed) : heads(h) {
    const size_t n = static_cast<size_t>(kKvLoraRank) * h * (kQkNope + kVHeadDim);
    w = RandF32(n, seed);
    // Keep the projected magnitudes near 1 so bf16 has usable precision.
    const float s = 1.0f / std::sqrt(static_cast<float>(kKvLoraRank));
    for (auto& x : w) x *= s;
  }

  // latent [n, kLatentDim] -> k [n, heads, kQkHeadDim], v [n, heads, kVHeadDim],
  // both bf16-rounded (that is what the kernel will consume).
  void Apply(const std::vector<float>& latent, int64_t n, std::vector<float>& k,
             std::vector<float>& v) const {
    k.assign(static_cast<size_t>(n) * heads * kQkHeadDim, 0.0f);
    v.assign(static_cast<size_t>(n) * heads * kVHeadDim, 0.0f);
    const int inner = kQkNope + kVHeadDim;
    for (int64_t t = 0; t < n; ++t) {
      const float* row = latent.data() + static_cast<size_t>(t) * kLatentDim;
      for (int h = 0; h < heads; ++h) {
        for (int j = 0; j < inner; ++j) {
          double acc = 0.0;
          const size_t col = static_cast<size_t>(h) * inner + j;
          for (int c = 0; c < kKvLoraRank; ++c) {
            acc += static_cast<double>(row[c]) * w[static_cast<size_t>(c) * heads * inner + col];
          }
          if (j < kQkNope) {
            k[(static_cast<size_t>(t) * heads + h) * kQkHeadDim + j] =
                static_cast<float>(acc);
          } else {
            v[(static_cast<size_t>(t) * heads + h) * kVHeadDim + (j - kQkNope)] =
                static_cast<float>(acc);
          }
        }
        // The decoupled rope part is SHARED across heads (deepseek_v2.py:588,
        // mla_attention.py:2159 `workspace[...][..., kv_lora_rank:].unsqueeze(1)`).
        for (int j = 0; j < kQkRope; ++j) {
          k[(static_cast<size_t>(t) * heads + h) * kQkHeadDim + kQkNope + j] =
              row[kKvLoraRank + j];
        }
      }
    }
    k = Bf16Round(k);
    v = Bf16Round(v);
  }
};

}  // namespace

// ─── the metadata builder ───────────────────────────────────────────────────

TEST_CASE("MLA chunked-prefill workspace size mirrors determine_chunked_prefill_workspace_size") {
  // mla_attention.py:1426-1451. The 64k clamp is the one whose reason is spelled
  // out inline (144 MB latent vs a 3 GB up-projected context).
  CHECK(DetermineChunkedPrefillWorkspaceSize(163840, 256, 16) == 64 * 1024);
  // 8 * max_model_len wins under the clamp.
  CHECK(DetermineChunkedPrefillWorkspaceSize(4096, 8, 16) == 8 * 4096);
  // 4 * max_num_seqs * block_size wins.
  CHECK(DetermineChunkedPrefillWorkspaceSize(64, 256, 16) == 4 * 256 * 16);
  // The trailing "enough for at least 1 page per request" clamp (:1445-1449) is
  // the ONLY way the result exceeds the 64k cap, and it must.
  CHECK(DetermineChunkedPrefillWorkspaceSize(1, 1024, 128) == 1024 * 128);
  CHECK(DetermineChunkedPrefillWorkspaceSize(1, 1024, 64) == 64 * 1024);
}

TEST_CASE("MLA chunk grid mirrors the upstream builder, boundaries included") {
  // Upstream's own worked example, quoted at mla_attention.py:1696-1700:
  // "if max_context_chunk = 256, num_chunks = 3, and num_prefills_with_context
  //  = 4, create a tensor that looks like
  //  [[0,0,0,0],[256,256,256,256],[512,512,512,512]]".
  {
    const std::vector<int32_t> ctx{700, 512, 256, 600};
    const auto qsl = Cumsum(std::vector<int32_t>{4, 4, 4, 4});
    // workspace 1024 over 4 with-context prefills -> 256, already page-aligned.
    const auto m = BuildMlaChunkedContext(ctx, qsl, 1024, 16);
    CHECK(m.max_context_chunk == 256);
    CHECK(m.num_chunks == 3);
    CHECK(m.num_prefills == 4);
    for (int b = 0; b < 4; ++b) {
      CHECK(m.starts[0 * 4 + b] == 0);
      CHECK(m.starts[1 * 4 + b] == 256);
      CHECK(m.starts[2 * 4 + b] == 512);
    }
    // chunk_seq_lens = clamp(min(ctx, start+256) - start, 0):
    //   chunk0 -> 256,256,256,256   chunk1 -> 256,256,0,256   chunk2 -> 188,0,0,88
    CHECK(m.chunk_seq_lens[0 * 4 + 0] == 256);
    CHECK(m.chunk_seq_lens[1 * 4 + 2] == 0);  // BOUNDARY: ctx exactly 256
    CHECK(m.chunk_seq_lens[1 * 4 + 1] == 256);
    CHECK(m.chunk_seq_lens[2 * 4 + 1] == 0);  // BOUNDARY: ctx exactly 512
    CHECK(m.chunk_seq_lens[2 * 4 + 0] == 188);
    CHECK(m.chunk_seq_lens[2 * 4 + 3] == 88);
    // cu_seq_lens is the per-chunk varlen prefix sum with a leading 0.
    CHECK(m.cu_seq_lens[2 * 5 + 0] == 0);
    CHECK(m.cu_seq_lens[2 * 5 + 1] == 188);
    CHECK(m.cu_seq_lens[2 * 5 + 2] == 188);
    CHECK(m.cu_seq_lens[2 * 5 + 3] == 188);
    CHECK(m.cu_seq_lens[2 * 5 + 4] == 276);
    CHECK(m.seq_tot[2] == 276);
    CHECK(m.max_seq_lens[2] == 188);
    CHECK(m.chunk_total_token[2] == 276);
    // token_to_seq back-maps token -> request (cache_kernels.cu:1015).
    const int row = m.max_token_num_over_chunk;
    CHECK(m.token_to_seq[static_cast<size_t>(2) * row + 0] == 0);
    CHECK(m.token_to_seq[static_cast<size_t>(2) * row + 187] == 0);
    CHECK(m.token_to_seq[static_cast<size_t>(2) * row + 188] == 3);
    // All four requests have context, so every prefill query token merges.
    CHECK(m.prefill_tokens_with_context == 16);
  }

  // PAGE ALIGNMENT: 1000 / 3 = 333 must round DOWN to 320 with block_size 64
  // (mla_attention.py:1687-1690 — the gather kernel cannot handle an unaligned
  // `seq_starts`). Getting this wrong silently gathers the wrong pages.
  {
    const std::vector<int32_t> ctx{900, 700, 100};
    const auto qsl = Cumsum(std::vector<int32_t>{2, 2, 2});
    const auto m = BuildMlaChunkedContext(ctx, qsl, 1000, 64);
    CHECK(m.max_context_chunk == 320);
    CHECK(m.max_context_chunk % 64 == 0);
    CHECK(m.num_chunks == 3);  // cdiv(900, 320)
  }

  // NO CONTEXT AT ALL -> no chunked metadata; forward_mha skips the loop
  // entirely (mla_attention.py:1666, :2394).
  {
    const std::vector<int32_t> ctx{0, 0};
    const auto qsl = Cumsum(std::vector<int32_t>{5, 5});
    const auto m = BuildMlaChunkedContext(ctx, qsl, 1024, 16);
    CHECK(m.num_chunks == 0);
  }

  // MIXED: only the FIRST request has context, so the workspace is not split
  // three ways and `prefill_tokens_with_context` covers only its query tokens.
  {
    const std::vector<int32_t> ctx{300, 0, 0};
    const auto qsl = Cumsum(std::vector<int32_t>{7, 11, 13});
    const auto m = BuildMlaChunkedContext(ctx, qsl, 512, 16);
    CHECK(m.max_context_chunk == 512);
    CHECK(m.num_chunks == 1);
    CHECK(m.chunk_seq_lens[0] == 300);
    CHECK(m.chunk_seq_lens[1] == 0);
    CHECK(m.prefill_tokens_with_context == 7);
  }

  // A workspace too small for one page per with-context request is upstream's
  // `assert max_context_chunk > 0` (:1692) — refuse loudly, never mis-gather.
  {
    const std::vector<int32_t> ctx{64, 64, 64, 64};
    const auto qsl = Cumsum(std::vector<int32_t>{1, 1, 1, 1});
    CHECK_THROWS(BuildMlaChunkedContext(ctx, qsl, 32, 64));
  }
}

// ─── vt::GatherMlaCache ─────────────────────────────────────────────────────

TEST_CASE("GatherMlaCache gathers the right pages through an adversarial block table") {
  const int num_prefills = 3;
  const std::vector<int32_t> chunk_lens{35, 0, 17};
  const std::vector<int32_t> starts{32, 0, 16};  // page-aligned chunk starts
  const auto cu = Cumsum(chunk_lens);
  const int total = cu.back();
  const int max_blocks = 8;
  const int num_blocks = num_prefills * max_blocks;

  // REVERSE-INTERLEAVED pages: request b's logical page p is physical block
  // (num_blocks - 1 - (p * num_prefills + b)). Nothing about this is contiguous
  // or ascending, so a page-stride assumption fails.
  std::vector<int32_t> block_table(static_cast<size_t>(num_prefills) * max_blocks, 0);
  for (int b = 0; b < num_prefills; ++b) {
    for (int p = 0; p < max_blocks; ++p) {
      block_table[static_cast<size_t>(b) * max_blocks + p] =
          num_blocks - 1 - (p * num_prefills + b);
    }
  }

  auto cache = RandF32(static_cast<size_t>(num_blocks) * kBlockSize * kLatentDim, 77u);
  std::vector<int32_t> token_to_seq(static_cast<size_t>(std::max(total, 1)), 0);
  {
    int t = 0;
    for (int b = 0; b < num_prefills; ++b)
      for (int j = 0; j < chunk_lens[static_cast<size_t>(b)]; ++j) token_to_seq[t++] = b;
  }

  // The expected gather, computed independently from the block table.
  std::vector<float> want(static_cast<size_t>(std::max(total, 1)) * kLatentDim, 0.0f);
  for (int t = 0; t < total; ++t) {
    const int b = token_to_seq[static_cast<size_t>(t)];
    const int within = t - cu[static_cast<size_t>(b)] + starts[static_cast<size_t>(b)];
    const int blk = block_table[static_cast<size_t>(b) * max_blocks + within / kBlockSize];
    const float* src =
        cache.data() +
        (static_cast<size_t>(blk) * kBlockSize + within % kBlockSize) * kLatentDim;
    std::copy(src, src + kLatentDim, want.begin() + static_cast<size_t>(t) * kLatentDim);
  }

  // CPU
  {
    std::vector<float> dst(static_cast<size_t>(std::max(total, 1)) * kLatentDim,
                           std::numeric_limits<float>::quiet_NaN());
    auto bt = block_table;
    auto cuv = cu;
    auto t2s = token_to_seq;
    auto st = starts;
    Tensor tdst = Contig(dst.data(), DType::kF32, Cpu(), {total, kLatentDim});
    Tensor tsrc =
        Contig(cache.data(), DType::kF32, Cpu(), {num_blocks, kBlockSize, kLatentDim});
    Tensor tbt = Contig(bt.data(), DType::kI32, Cpu(), {num_prefills, max_blocks});
    Tensor tcu = Contig(cuv.data(), DType::kI32, Cpu(), {num_prefills + 1});
    Tensor tt2s = Contig(t2s.data(), DType::kI32, Cpu(), {total});
    Tensor tst = Contig(st.data(), DType::kI32, Cpu(), {num_prefills});
    Queue q0 = CpuQ();
    vt::GatherMlaCache(q0, tdst, tsrc, tbt, tcu, tt2s, &tst, total);
    CHECK(MaxAbsDiff(dst, want) == 0.0);
  }

  // CUDA — bit-identical (it is a pure copy).
  if (HasCuda()) {
    Backend& b = vt::GetBackend(DeviceType::kCUDA);
    QueueGuard g(b);
    auto cache_b = ToBf16(cache);
    auto want_b = ToBf16(want);
    std::vector<uint16_t> poison(static_cast<size_t>(std::max(total, 1)) * kLatentDim,
                                 vt::F32ToBF16(std::numeric_limits<float>::quiet_NaN()));
    auto bt = block_table;
    auto cuv = cu;
    auto t2s = token_to_seq;
    auto st = starts;
    DeviceTensor ddst(b, g.q, DType::kBF16, {total, kLatentDim}, poison.data());
    DeviceTensor dsrc(b, g.q, DType::kBF16, {num_blocks, kBlockSize, kLatentDim},
                      cache_b.data());
    DeviceTensor dbt(b, g.q, DType::kI32, {num_prefills, max_blocks}, bt.data());
    DeviceTensor dcu(b, g.q, DType::kI32, {num_prefills + 1}, cuv.data());
    DeviceTensor dt2s(b, g.q, DType::kI32, {total}, t2s.data());
    DeviceTensor dst_starts(b, g.q, DType::kI32, {num_prefills}, st.data());
    vt::GatherMlaCache(g.q, ddst.tensor(), dsrc.tensor(), dbt.tensor(), dcu.tensor(),
                       dt2s.tensor(), &dst_starts.tensor(), total);
    b.Synchronize(g.q);
    std::vector<uint16_t> got(poison.size());
    ddst.Download(g.q, got.data());
    CHECK(got == want_b);
  }
}

// ─── vt::MergeAttnStates ────────────────────────────────────────────────────

TEST_CASE("MergeAttnStates reproduces the LSE algebra and BOTH -inf edge cases") {
  const int tokens = 6;
  const int heads = 4;
  const int dim = 32;
  auto p_out = RandF32(static_cast<size_t>(tokens) * heads * dim, 3u);
  auto s_out = RandF32(static_cast<size_t>(tokens) * heads * dim, 5u);
  std::vector<float> p_lse(static_cast<size_t>(heads) * tokens);
  std::vector<float> s_lse(static_cast<size_t>(heads) * tokens);
  for (size_t i = 0; i < p_lse.size(); ++i) {
    p_lse[i] = static_cast<float>(i % 7) * 0.5f - 1.0f;
    s_lse[i] = static_cast<float>(i % 5) * 0.25f + 0.125f;
  }
  // Token 1 / head 0: prefix has NO keys (-inf) — the chunk-boundary case.
  p_lse[0 * tokens + 1] = -std::numeric_limits<float>::infinity();
  // Token 2 / head 1: BOTH -inf — merge_attn_states.cu:100-134, which would be
  // 0/0 and must instead emit the PREFIX output.
  p_lse[1 * tokens + 2] = -std::numeric_limits<float>::infinity();
  s_lse[1 * tokens + 2] = -std::numeric_limits<float>::infinity();
  // Token 3 / head 2: +inf, which `:97-98` normalizes to -inf FIRST. FA-2
  // actually writes +INFINITY for an empty-K row, so this is not hypothetical.
  p_lse[2 * tokens + 3] = std::numeric_limits<float>::infinity();
  // Tokens >= 4 have no context at all: prefill_tokens_with_context == 4.
  const int64_t ptwc = 4;

  // Independent double-precision reference.
  std::vector<float> want(p_out.size(), 0.0f);
  std::vector<float> want_lse(p_lse.size(), 0.0f);
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < heads; ++h) {
      const size_t base = (static_cast<size_t>(t) * heads + h) * dim;
      if (t >= ptwc) {
        for (int d = 0; d < dim; ++d) want[base + d] = s_out[base + d];
        want_lse[static_cast<size_t>(h) * tokens + t] = s_lse[static_cast<size_t>(h) * tokens + t];
        continue;
      }
      double pl = p_lse[static_cast<size_t>(h) * tokens + t];
      double sl = s_lse[static_cast<size_t>(h) * tokens + t];
      if (std::isinf(pl)) pl = -std::numeric_limits<double>::infinity();
      if (std::isinf(sl)) sl = -std::numeric_limits<double>::infinity();
      const double mx = std::max(pl, sl);
      if (std::isinf(mx)) {
        for (int d = 0; d < dim; ++d) want[base + d] = p_out[base + d];
        want_lse[static_cast<size_t>(h) * tokens + t] = static_cast<float>(mx);
        continue;
      }
      const double ps = std::exp(pl - mx);
      const double ss = std::exp(sl - mx);
      const double tot = ps + ss;
      for (int d = 0; d < dim; ++d) {
        want[base + d] =
            static_cast<float>(p_out[base + d] * (ps / tot) + s_out[base + d] * (ss / tot));
      }
      want_lse[static_cast<size_t>(h) * tokens + t] = static_cast<float>(std::log(tot) + mx);
    }
  }

  auto check = [&](const std::vector<float>& got, const std::vector<float>& got_lse,
                   double tol) {
    CHECK(MaxAbsDiff(got, want) < tol);
    for (size_t i = 0; i < got_lse.size(); ++i) {
      if (std::isinf(want_lse[i])) {
        CHECK(std::isinf(got_lse[i]));
        CHECK(got_lse[i] < 0.0f);
      } else {
        CHECK(std::fabs(got_lse[i] - want_lse[i]) < tol);
      }
    }
  };

  // CPU
  {
    std::vector<float> out(p_out.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> out_lse(p_lse.size(), std::numeric_limits<float>::quiet_NaN());
    Tensor to = Contig(out.data(), DType::kF32, Cpu(), {tokens, heads, dim});
    Tensor tol_ = Contig(out_lse.data(), DType::kF32, Cpu(), {heads, tokens});
    Tensor tp = Contig(p_out.data(), DType::kF32, Cpu(), {tokens, heads, dim});
    Tensor tpl = Contig(p_lse.data(), DType::kF32, Cpu(), {heads, tokens});
    Tensor ts = Contig(s_out.data(), DType::kF32, Cpu(), {tokens, heads, dim});
    Tensor tsl = Contig(s_lse.data(), DType::kF32, Cpu(), {heads, tokens});
    Queue q0 = CpuQ();
    vt::MergeAttnStates(q0, to, &tol_, tp, tpl, ts, tsl, ptwc);
    check(out, out_lse, 1e-5);
  }

  // CUDA
  if (HasCuda()) {
    Backend& b = vt::GetBackend(DeviceType::kCUDA);
    QueueGuard g(b);
    std::vector<float> poison(p_out.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> poison_lse(p_lse.size(), std::numeric_limits<float>::quiet_NaN());
    DeviceTensor dout(b, g.q, DType::kF32, {tokens, heads, dim}, poison.data());
    DeviceTensor dout_lse(b, g.q, DType::kF32, {heads, tokens}, poison_lse.data());
    DeviceTensor dp(b, g.q, DType::kF32, {tokens, heads, dim}, p_out.data());
    DeviceTensor dpl(b, g.q, DType::kF32, {heads, tokens}, p_lse.data());
    DeviceTensor ds(b, g.q, DType::kF32, {tokens, heads, dim}, s_out.data());
    DeviceTensor dsl(b, g.q, DType::kF32, {heads, tokens}, s_lse.data());
    vt::MergeAttnStates(g.q, dout.tensor(), &dout_lse.tensor(), dp.tensor(), dpl.tensor(),
                        ds.tensor(), dsl.tensor(), ptwc);
    b.Synchronize(g.q);
    std::vector<float> got(poison.size());
    std::vector<float> got_lse(poison_lse.size());
    dout.Download(g.q, got.data());
    dout_lse.Download(g.q, got_lse.data());
    check(got, got_lse, 1e-5);
  }
}

// ─── the whole chunked-context loop vs a single-shot oracle ─────────────────

namespace {

// One end-to-end chunked-prefill scenario: build a paged cache holding each
// request's context behind an ADVERSARIAL block table, run
// ForwardMlaPrefillMha, and require it to reproduce the single-shot causal
// attention over [context ++ new tokens].
void RunChunkedCase(Backend& b, Queue& q, const std::vector<int32_t>& ctx_lens,
                    const std::vector<int32_t>& q_lens, int heads, int64_t workspace_tokens,
                    const char* what) {
  CAPTURE(what);
  const int num_prefills = static_cast<int>(ctx_lens.size());
  const auto cu_q = Cumsum(q_lens);
  const int total_q = cu_q.back();
  const double scale = LiteScale();
  const KvBProj proj(heads, 909u);

  // Full per-request latent: context rows FIRST, then this step's new tokens.
  std::vector<std::vector<float>> latent(num_prefills);
  int max_seq = 0;
  for (int i = 0; i < num_prefills; ++i) {
    const int n = ctx_lens[static_cast<size_t>(i)] + q_lens[static_cast<size_t>(i)];
    latent[static_cast<size_t>(i)] =
        Bf16Round(RandF32(static_cast<size_t>(n) * kLatentDim, 1000u + 17u * i));
    max_seq = std::max(max_seq, n);
  }

  // The paged MLA cache, with a REVERSE-INTERLEAVED block table. Only the
  // CONTEXT rows live in the cache: the chunked loop only ever reads
  // [0, context_len) (mla_attention.py:1663 context = seq_len - query_len).
  const int max_blocks = (max_seq + kBlockSize - 1) / kBlockSize + 1;
  const int num_blocks = num_prefills * max_blocks;
  std::vector<int32_t> block_table(static_cast<size_t>(num_prefills) * max_blocks, 0);
  for (int i = 0; i < num_prefills; ++i) {
    for (int p = 0; p < max_blocks; ++p) {
      block_table[static_cast<size_t>(i) * max_blocks + p] =
          num_blocks - 1 - (p * num_prefills + i);
    }
  }
  std::vector<float> cache(static_cast<size_t>(num_blocks) * kBlockSize * kLatentDim, 0.0f);
  for (int i = 0; i < num_prefills; ++i) {
    for (int j = 0; j < ctx_lens[static_cast<size_t>(i)]; ++j) {
      const int blk = block_table[static_cast<size_t>(i) * max_blocks + j / kBlockSize];
      float* dst =
          cache.data() +
          (static_cast<size_t>(blk) * kBlockSize + j % kBlockSize) * kLatentDim;
      const float* src =
          latent[static_cast<size_t>(i)].data() + static_cast<size_t>(j) * kLatentDim;
      std::copy(src, src + kLatentDim, dst);
    }
  }

  // Queries (bf16-rounded) and the new tokens' own K/V.
  auto qf = Bf16Round(RandF32(static_cast<size_t>(total_q) * heads * kQkHeadDim, 4242u));
  std::vector<float> new_latent;
  for (int i = 0; i < num_prefills; ++i) {
    const int c = ctx_lens[static_cast<size_t>(i)];
    const int n = q_lens[static_cast<size_t>(i)];
    const float* src = latent[static_cast<size_t>(i)].data() + static_cast<size_t>(c) * kLatentDim;
    new_latent.insert(new_latent.end(), src, src + static_cast<size_t>(n) * kLatentDim);
  }
  std::vector<float> new_k;
  std::vector<float> new_v;
  proj.Apply(new_latent, total_q, new_k, new_v);

  // ── the ORACLE: one causal attention over [context ++ new] ──────────────
  std::vector<int32_t> full_lens(num_prefills);
  for (int i = 0; i < num_prefills; ++i)
    full_lens[static_cast<size_t>(i)] =
        ctx_lens[static_cast<size_t>(i)] + q_lens[static_cast<size_t>(i)];
  const auto cu_full = Cumsum(full_lens);
  std::vector<float> full_latent;
  for (int i = 0; i < num_prefills; ++i) {
    full_latent.insert(full_latent.end(), latent[static_cast<size_t>(i)].begin(),
                       latent[static_cast<size_t>(i)].end());
  }
  std::vector<float> full_k;
  std::vector<float> full_v;
  proj.Apply(full_latent, cu_full.back(), full_k, full_v);
  std::vector<float> want;
  RefPrefill(want, qf, full_k, full_v, cu_q, cu_full, heads, kQkHeadDim, kVHeadDim, scale,
             /*causal=*/true);

  // ── the CHUNKED path ────────────────────────────────────────────────────
  const auto meta = BuildMlaChunkedContext(ctx_lens, cu_q, workspace_tokens, kBlockSize);
  int32_t max_query_len = 0;
  for (int32_t l : q_lens) max_query_len = std::max(max_query_len, l);

  auto qb = ToBf16(qf);
  auto kb = ToBf16(new_k);
  auto vb = ToBf16(new_v);
  auto cache_b = ToBf16(cache);
  auto bt = block_table;
  auto cu_q_v = cu_q;
  std::vector<uint16_t> poison(static_cast<size_t>(total_q) * heads * kVHeadDim,
                               vt::F32ToBF16(std::numeric_limits<float>::quiet_NaN()));

  DeviceTensor dq(b, q, DType::kBF16, {total_q, heads, kQkHeadDim}, qb.data());
  DeviceTensor dk(b, q, DType::kBF16, {std::max(total_q, 1), heads, kQkHeadDim}, kb.data());
  DeviceTensor dv(b, q, DType::kBF16, {std::max(total_q, 1), heads, kVHeadDim}, vb.data());
  DeviceTensor dcache(b, q, DType::kBF16, {num_blocks, kBlockSize, kLatentDim},
                      cache_b.data());
  DeviceTensor dbt(b, q, DType::kI32, {num_prefills, max_blocks}, bt.data());
  DeviceTensor dcq(b, q, DType::kI32, {static_cast<int64_t>(cu_q_v.size())}, cu_q_v.data());
  DeviceTensor dout(b, q, DType::kBF16, {total_q, heads, kVHeadDim}, poison.data());

  const int64_t ws_rows = std::max<int64_t>(meta.max_token_num_over_chunk, 1);
  MlaPrefillContextBuffers bufs;
  DeviceTensor dws(b, q, DType::kBF16, {ws_rows, kLatentDim});
  DeviceTensor dchunk(b, q, DType::kBF16, {total_q, heads, kVHeadDim}, poison.data());
  DeviceTensor dchunk_lse(b, q, DType::kF32, {heads, total_q});
  DeviceTensor daccum(b, q, DType::kBF16, {total_q, heads, kVHeadDim}, poison.data());
  DeviceTensor daccum_lse(b, q, DType::kF32, {heads, total_q});
  DeviceTensor dmerge(b, q, DType::kBF16, {total_q, heads, kVHeadDim}, poison.data());
  DeviceTensor dmerge_lse(b, q, DType::kF32, {heads, total_q});
  DeviceTensor dsuffix(b, q, DType::kBF16, {total_q, heads, kVHeadDim}, poison.data());
  DeviceTensor dsuffix_lse(b, q, DType::kF32, {heads, total_q});
  bufs.workspace = dws.tensor();
  bufs.chunk_output = dchunk.tensor();
  bufs.chunk_lse = dchunk_lse.tensor();
  bufs.accum_output = daccum.tensor();
  bufs.accum_lse = daccum_lse.tensor();
  bufs.merge_output = dmerge.tensor();
  bufs.merge_lse = dmerge_lse.tensor();

  // Upload the per-chunk metadata once, exactly as upstream's build() does.
  std::vector<MlaChunkDeviceMetadata> chunks;
  std::vector<std::unique_ptr<DeviceTensor>> keep;
  const int row = std::max<int32_t>(meta.max_token_num_over_chunk, 1);
  for (int i = 0; i < meta.num_chunks; ++i) {
    std::vector<int32_t> cu(meta.cu_seq_lens.begin() +
                                static_cast<size_t>(i) * (num_prefills + 1),
                            meta.cu_seq_lens.begin() +
                                static_cast<size_t>(i + 1) * (num_prefills + 1));
    std::vector<int32_t> st(meta.starts.begin() + static_cast<size_t>(i) * num_prefills,
                            meta.starts.begin() + static_cast<size_t>(i + 1) * num_prefills);
    std::vector<int32_t> t2s(meta.token_to_seq.begin() + static_cast<size_t>(i) * row,
                             meta.token_to_seq.begin() + static_cast<size_t>(i + 1) * row);
    keep.push_back(std::unique_ptr<DeviceTensor>(
        new DeviceTensor(b, q, DType::kI32, {num_prefills + 1}, cu.data())));
    Tensor tcu = keep.back()->tensor();
    keep.push_back(std::unique_ptr<DeviceTensor>(
        new DeviceTensor(b, q, DType::kI32, {num_prefills}, st.data())));
    Tensor tst = keep.back()->tensor();
    keep.push_back(
        std::unique_ptr<DeviceTensor>(new DeviceTensor(b, q, DType::kI32, {row}, t2s.data())));
    Tensor tt2s = keep.back()->tensor();
    MlaChunkDeviceMetadata c;
    c.cu_seq_lens = tcu;
    c.token_to_seq = tt2s;
    c.starts = tst;
    c.total_tokens = meta.chunk_total_token[static_cast<size_t>(i)];
    c.max_seq_len = meta.max_seq_lens[static_cast<size_t>(i)];
    chunks.push_back(c);
  }
  b.Synchronize(q);

  // The up-projection callback (mla_attention.py:2141-2170). The real kv_b_proj
  // GEMM is W6's; here it runs on the host against the SAME weights the oracle
  // used, so any mismatch is the loop's, not the projection's.
  std::vector<std::unique_ptr<DeviceTensor>> chunk_kv_keep;
  auto up_project = [&](Queue& qq, const Tensor& ws, int64_t toks) -> MlaContextChunkKv {
    std::vector<uint16_t> host(static_cast<size_t>(std::max<int64_t>(toks, 1)) * kLatentDim, 0);
    b.Synchronize(qq);
    b.Copy(qq, host.data(), ws.data, host.size() * sizeof(uint16_t));
    b.Synchronize(qq);
    const auto rows = FromBf16(host);
    std::vector<float> k;
    std::vector<float> v;
    proj.Apply(rows, std::max<int64_t>(toks, 1), k, v);
    auto kbb = ToBf16(k);
    auto vbb = ToBf16(v);
    chunk_kv_keep.push_back(std::unique_ptr<DeviceTensor>(new DeviceTensor(
        b, qq, DType::kBF16, {std::max<int64_t>(toks, 1), heads, kQkHeadDim}, kbb.data())));
    Tensor tk = chunk_kv_keep.back()->tensor();
    chunk_kv_keep.push_back(std::unique_ptr<DeviceTensor>(new DeviceTensor(
        b, qq, DType::kBF16, {std::max<int64_t>(toks, 1), heads, kVHeadDim}, vbb.data())));
    Tensor tv = chunk_kv_keep.back()->tensor();
    b.Synchronize(qq);
    tk.shape[0] = toks;
    tv.shape[0] = toks;
    return MlaContextChunkKv{tk, tv};
  };

  ForwardMlaPrefillMha(q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(),
                       dcache.tensor(), dbt.tensor(), dcq.tensor(), chunks, up_project,
                       static_cast<float>(scale), max_query_len,
                       meta.prefill_tokens_with_context, bufs, dsuffix.tensor(),
                       dsuffix_lse.tensor());
  b.Synchronize(q);

  std::vector<uint16_t> got_b(poison.size());
  dout.Download(q, got_b.data());
  const auto got = FromBf16(got_b);
  CHECK(MaxAbsDiff(got, want) < 4e-2);
}

}  // namespace

TEST_CASE("CUDA chunked-context prefill reproduces the single-shot oracle") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);

  // Workspace 64 tokens over N with-context requests forces MANY chunks at these
  // context lengths — the point is to exercise the loop, not the sizing.
  //
  // SINGLE chunk (context fits in one chunk for every request).
  RunChunkedCase(b, g.q, {32}, {8}, 16, 256, "single request, single chunk");
  // MULTI-chunk: 96 context over a 32-token chunk grid = 3 chunks, so the LSE
  // merge and the ping-pong both run twice.
  RunChunkedCase(b, g.q, {96}, {8}, 16, 32, "single request, 3 chunks");
  // CHUNK BOUNDARY: one context is an EXACT multiple of the chunk size (so its
  // last chunk contributes ZERO tokens), one is one over, one is one under.
  RunChunkedCase(b, g.q, {64, 65, 63}, {4, 4, 4}, 16, 96, "exact / +1 / -1 chunk boundaries");
  // MIXED context: request 1 has NO context at all, so its query tokens must
  // take the suffix verbatim via `prefill_tokens_with_context`.
  RunChunkedCase(b, g.q, {80, 0}, {6, 6}, 16, 64, "one request with no context");
  // Ragged everything, several chunks, page-aligned starts.
  RunChunkedCase(b, g.q, {160, 48, 96, 16}, {3, 9, 1, 17}, 16, 128, "ragged, multi-chunk");
  // DeepSeek-V3's 128 heads over a multi-chunk context.
  RunChunkedCase(b, g.q, {80, 32}, {4, 4}, 128, 64, "V3 128-head, multi-chunk");
}
