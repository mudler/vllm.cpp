// Qwen3-VL vision tower (`Qwen3_VisionTransformer`) forward — M2a + speed pass.
//
// Ported 1:1 from vllm/model_executor/models/qwen3_vl.py @ e24d1b24:
//   forward (:800-841), Qwen3_VisionPatchEmbed (:347-373),
//   Qwen3_VisionBlock (:413-464), Qwen3_VisionMLP (:376-410),
//   Qwen3_VisionPatchMerger (:467-516), pos_embed_interpolate_native (:277-344),
//   rot_pos_ids (:640-665) + rot_pos_emb (:667-683),
//   vision attention Qwen2_5_VisionAttention.forward (qwen2_5_vl.py:397-460),
//   ApplyRotaryEmb.forward_static (rotary_embedding/common.py:151-186).
//
// Composed from the public vt:: ops (Matmul/Add/LayerNorm/RopeFromCache/
// Attention/GeluTanh/GeluErf). All GEMMs run in the production model dtype bf16;
// softmax/norm accumulate in f32. The pos-embed bilinear interp and the vision
// rope cos|sin are deterministic host precomputes (f32) consumed on device — vLLM
// computes them on GPU (a Triton bilinear kernel + a rope cache), gated within a
// stated bf16 tolerance in the M2a unit test.
//
// SPEED PASS (CLAIM-MULTIMODAL-SPEED-TOWER): the tower weights are converted to
// bf16 + uploaded ONCE via PrepareVisionDeviceWeights and kept device-resident
// (mirroring vLLM's already-loaded nn.Linears); the per-image forward then does
// only the tiny pixel/pos-embed/rope uploads + the ViT GEMMs/attention. The old
// host-weights overload is preserved as a thin prepare-then-forward wrapper
// (BIT-IDENTICAL: same bf16 weight bytes, same GEMM order) so every unit gate is
// unchanged. Set VLLM_MM_TOWER_PROFILE=1 to print the prepare(marshal) vs
// forward(compute) split on stderr.
#include "vllm/model_executor/models/qwen3_vl_vision.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/merged_qkv_fold.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm::multimodal {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// --- RAII device buffer (mirror of the tests' DeviceTensor helper). ----------
struct Buf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t;
  Buf(Backend& backend, Queue& q, DType dt, std::vector<int64_t> shape,
      const void* host = nullptr)
      : b(backend) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p = b.Alloc(bytes == 0 ? 1 : bytes);
    t.data = p;
    t.dtype = dt;
    t.device = q.device;
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    if (host != nullptr) b.Copy(q, p, host, bytes);
  }
  ~Buf() { b.Free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Tensor& tensor() { return t; }
  void Download(Queue& q, void* dst) {
    b.Copy(q, dst, p, bytes);
    b.Synchronize(q);
  }
};

std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> o(f.size());
  for (size_t i = 0; i < f.size(); ++i) o[i] = vt::F32ToBF16(f[i]);
  return o;
}

// out[M,N] = x[M,K] @ W[N,K]^T + bias[N]  (bias optional). All bf16.
void LinearBias(Queue& q, Buf& out, Tensor x, Tensor w, const Tensor* bias) {
  vt::MatmulBT(q, out.tensor(), x, w);
  if (bias != nullptr) vt::Add(q, out.tensor(), out.tensor(), *bias);
}

}  // namespace

// --- device-resident weight holder (owns one bf16 buffer, device-global) ------
struct DevW {
  Backend* b = nullptr;
  void* p = nullptr;
  Tensor t{};
  DevW() = default;
  DevW(const DevW&) = delete;
  DevW& operator=(const DevW&) = delete;
  DevW(DevW&& o) noexcept { *this = std::move(o); }
  DevW& operator=(DevW&& o) noexcept {
    if (this != &o) {
      Reset();
      b = o.b;
      p = o.p;
      t = o.t;
      o.b = nullptr;
      o.p = nullptr;
    }
    return *this;
  }
  void Reset() {
    if (b != nullptr && p != nullptr) b->Free(p);
    b = nullptr;
    p = nullptr;
  }
  ~DevW() { Reset(); }
  const Tensor& tensor() const { return t; }
};

namespace {

// Convert host f32 -> bf16 and upload once as a device-resident buffer.
DevW MakeDevBf16(Backend& b, Queue& q, const std::vector<float>& f, std::vector<int64_t> shape) {
  DevW d;
  d.b = &b;
  int64_t numel = 1;
  for (auto s : shape) numel *= s;
  const size_t bytes = static_cast<size_t>(numel) * vt::SizeOf(DType::kBF16);
  d.p = b.Alloc(bytes == 0 ? 1 : bytes);
  d.t.data = d.p;
  d.t.dtype = DType::kBF16;
  d.t.device = q.device;
  d.t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    d.t.shape[i] = shape[static_cast<size_t>(i)];
    d.t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  const auto bf = ToBf16(f);
  if (bytes != 0) b.Copy(q, d.p, bf.data(), bytes);
  return d;
}

// (Former SubRows/SubVec qkv row-slicers removed: the qkv projection now folds
// to ONE MatmulBT over the resident merged qkv_w [3H,H] + merged-bias epilogue
// + contiguous QkvSplit — see models::FusedMergedQkvBiasSplit, Tier C2.)

struct DevBlock {
  DevW norm1_w, norm1_b, norm2_w, norm2_b;
  DevW qkv_w, qkv_b;  // qkv_w [3H,H], qkv_b [3H]
  DevW proj_w, proj_b;
  DevW fc1_w, fc1_b, fc2_w, fc2_b;
};

struct DevMerger {
  bool use_postshuffle_norm = false;
  DevW norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b;
};

DevMerger MakeDevMerger(Backend& b, Queue& q, const VisionMergerWeights& mw,
                        const Qwen3VLVisionConfig& cfg) {
  const int64_t H = cfg.hidden_size;
  const int64_t ctx4 = H * cfg.merge_unit();
  const int64_t D = cfg.out_hidden_size;
  DevMerger dm;
  dm.use_postshuffle_norm = mw.use_postshuffle_norm;
  const int64_t nd = mw.use_postshuffle_norm ? ctx4 : H;
  dm.norm_w = MakeDevBf16(b, q, mw.norm_w, {nd});
  dm.norm_b = MakeDevBf16(b, q, mw.norm_b, {nd});
  dm.fc1_w = MakeDevBf16(b, q, mw.fc1_w, {ctx4, ctx4});
  dm.fc1_b = MakeDevBf16(b, q, mw.fc1_b, {ctx4});
  dm.fc2_w = MakeDevBf16(b, q, mw.fc2_w, {D, ctx4});
  dm.fc2_b = MakeDevBf16(b, q, mw.fc2_b, {D});
  return dm;
}

}  // namespace

// The device-resident tower weights (opaque to callers; built once).
struct Qwen3VLVisionDeviceWeights {
  DevW patch_proj_w, patch_proj_b;
  std::vector<float> pos_embed_w;  // host f32 kept for the per-grid bilinear interp
  std::vector<DevBlock> blocks;
  DevMerger merger;
  std::vector<DevMerger> deepstack_mergers;
};

// --- host precompute: pos-embed bilinear interp + spatial-merge reorder -------
// pos_embed_interpolate_native (qwen3_vl.py:277-344) for a single (t,h,w).
std::vector<float> VisionPosEmbedInterpolate(const std::vector<float>& pos_embed_w,
                                             const std::array<int64_t, 3>& grid_thw,
                                             const Qwen3VLVisionConfig& cfg) {
  const int64_t t = grid_thw[0], h = grid_thw[1], w = grid_thw[2];
  const int64_t H = cfg.hidden_size;
  const int64_t G = cfg.num_grid_per_side();
  const int64_t m = cfg.spatial_merge_size;
  // linspace(0, G-1, n) in f32.
  auto linspace = [](int64_t n, int64_t g) {
    std::vector<float> v(static_cast<size_t>(n));
    if (n == 1) {
      v[0] = 0.0f;
      return v;
    }
    for (int64_t i = 0; i < n; ++i)
      v[static_cast<size_t>(i)] =
          static_cast<float>(i) * static_cast<float>(g - 1) / static_cast<float>(n - 1);
    return v;
  };
  std::vector<float> h_idx = linspace(h, G), w_idx = linspace(w, G);
  const int64_t hm = h / m, wm = w / m;
  // one frame [h*w, H] in spatial-merge order, then repeated t times.
  std::vector<float> frame(static_cast<size_t>(h) * w * H);
  // r enumerates (bi,bj,li,lj) C-order; source (i=bi*m+li, j=bj*m+lj).
  for (int64_t bi = 0; bi < hm; ++bi)
    for (int64_t bj = 0; bj < wm; ++bj)
      for (int64_t li = 0; li < m; ++li)
        for (int64_t lj = 0; lj < m; ++lj) {
          const int64_t i = bi * m + li, j = bj * m + lj;
          const int64_t r = ((bi * wm + bj) * m + li) * m + lj;
          const float hf = h_idx[static_cast<size_t>(i)], wf = w_idx[static_cast<size_t>(j)];
          const int64_t h_floor = static_cast<int64_t>(std::floor(hf));
          const int64_t w_floor = static_cast<int64_t>(std::floor(wf));
          const int64_t h_ceil = std::min(h_floor + 1, G - 1);
          const int64_t w_ceil = std::min(w_floor + 1, G - 1);
          const float dh = hf - static_cast<float>(h_floor);
          const float dw = wf - static_cast<float>(w_floor);
          const float w11 = dh * dw, w10 = dh - w11, w01 = dw - w11, w00 = 1.0f - dh - w01;
          const int64_t i00 = h_floor * G + w_floor, i01 = h_floor * G + w_ceil;
          const int64_t i10 = h_ceil * G + w_floor, i11 = h_ceil * G + w_ceil;
          float* dst = &frame[static_cast<size_t>(r) * H];
          const float* e00 = &pos_embed_w[static_cast<size_t>(i00) * H];
          const float* e01 = &pos_embed_w[static_cast<size_t>(i01) * H];
          const float* e10 = &pos_embed_w[static_cast<size_t>(i10) * H];
          const float* e11 = &pos_embed_w[static_cast<size_t>(i11) * H];
          for (int64_t d = 0; d < H; ++d)
            dst[d] = w00 * e00[d] + w01 * e01[d] + w10 * e10[d] + w11 * e11[d];
        }
  std::vector<float> out(static_cast<size_t>(t) * h * w * H);
  for (int64_t f = 0; f < t; ++f)
    std::memcpy(&out[static_cast<size_t>(f) * h * w * H], frame.data(),
                frame.size() * sizeof(float));
  return out;
}

// --- host precompute: vision rope cos|sin ([L, head_dim/2] each) --------------
// rot_pos_ids (:640-665) + rot_pos_emb (:667-683). partial_rotary_factor 0.5:
// rotary_dim = head_dim/2; inv_freq over rotary_dim/2 = head_dim/4 freqs, each
// spatial axis (h,w) contributes head_dim/4 → cos|sin width = head_dim/2.
void VisionRopeCosSin(const std::array<int64_t, 3>& grid_thw, const Qwen3VLVisionConfig& cfg,
                      std::vector<float>* cos, std::vector<float>* sin) {
  const int64_t t = grid_thw[0], h = grid_thw[1], w = grid_thw[2];
  const int64_t m = cfg.spatial_merge_size;
  const int64_t head_dim = cfg.head_dim();
  const int64_t rotary_dim = head_dim / 2;   // partial 0.5
  const int64_t nfreq = rotary_dim / 2;      // per-axis frequency count (=head_dim/4)
  const int64_t half = head_dim / 2;         // cos|sin width
  const double base = 10000.0;
  std::vector<double> inv_freq(static_cast<size_t>(nfreq));
  for (int64_t i = 0; i < nfreq; ++i)
    inv_freq[static_cast<size_t>(i)] =
        1.0 / std::pow(base, static_cast<double>(2 * i) / static_cast<double>(rotary_dim));
  const int64_t hm = h / m, wm = w / m;
  const int64_t L = t * h * w;
  cos->assign(static_cast<size_t>(L) * half, 0.0f);
  sin->assign(static_cast<size_t>(L) * half, 0.0f);
  // per-frame pos_ids [(bi,bj,li,lj)] -> hpos=bi*m+li, wpos=bj*m+lj; cos[r] =
  // concat(cos(hpos*inv_freq), cos(wpos*inv_freq)).
  for (int64_t f = 0; f < t; ++f)
    for (int64_t bi = 0; bi < hm; ++bi)
      for (int64_t bj = 0; bj < wm; ++bj)
        for (int64_t li = 0; li < m; ++li)
          for (int64_t lj = 0; lj < m; ++lj) {
            const int64_t hpos = bi * m + li, wpos = bj * m + lj;
            const int64_t rframe = ((bi * wm + bj) * m + li) * m + lj;
            const int64_t r = f * (h * w) + rframe;
            float* cr = &(*cos)[static_cast<size_t>(r) * half];
            float* sr = &(*sin)[static_cast<size_t>(r) * half];
            for (int64_t i = 0; i < nfreq; ++i) {
              const double ah = static_cast<double>(hpos) * inv_freq[static_cast<size_t>(i)];
              const double aw = static_cast<double>(wpos) * inv_freq[static_cast<size_t>(i)];
              cr[i] = static_cast<float>(std::cos(ah));
              sr[i] = static_cast<float>(std::sin(ah));
              cr[nfreq + i] = static_cast<float>(std::cos(aw));
              sr[nfreq + i] = static_cast<float>(std::sin(aw));
            }
          }
}

// --- build the resident device weights (the ONE-TIME conversion + upload) -----
std::shared_ptr<Qwen3VLVisionDeviceWeights> PrepareVisionDeviceWeights(
    const Qwen3VLVisionWeights& w, const Qwen3VLVisionConfig& cfg, Backend& b) {
  Queue q = b.CreateQueue();
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const int64_t patch_dim =
      cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
  auto dw = std::make_shared<Qwen3VLVisionDeviceWeights>();
  dw->patch_proj_w = MakeDevBf16(b, q, w.patch_proj_w, {H, patch_dim});
  dw->patch_proj_b = MakeDevBf16(b, q, w.patch_proj_b, {H});
  dw->pos_embed_w = w.pos_embed_w;  // host f32 kept for per-grid interp
  dw->blocks.resize(w.blocks.size());
  for (size_t l = 0; l < w.blocks.size(); ++l) {
    const VisionBlockWeights& bw = w.blocks[l];
    DevBlock& db = dw->blocks[l];
    db.norm1_w = MakeDevBf16(b, q, bw.norm1_w, {H});
    db.norm1_b = MakeDevBf16(b, q, bw.norm1_b, {H});
    db.norm2_w = MakeDevBf16(b, q, bw.norm2_w, {H});
    db.norm2_b = MakeDevBf16(b, q, bw.norm2_b, {H});
    db.qkv_w = MakeDevBf16(b, q, bw.qkv_w, {3 * H, H});  // fused, sliced at forward
    db.qkv_b = MakeDevBf16(b, q, bw.qkv_b, {3 * H});
    db.proj_w = MakeDevBf16(b, q, bw.proj_w, {H, H});
    db.proj_b = MakeDevBf16(b, q, bw.proj_b, {H});
    db.fc1_w = MakeDevBf16(b, q, bw.fc1_w, {I, H});
    db.fc1_b = MakeDevBf16(b, q, bw.fc1_b, {I});
    db.fc2_w = MakeDevBf16(b, q, bw.fc2_w, {H, I});
    db.fc2_b = MakeDevBf16(b, q, bw.fc2_b, {H});
  }
  dw->merger = MakeDevMerger(b, q, w.merger, cfg);
  dw->deepstack_mergers.reserve(w.deepstack_mergers.size());
  for (const auto& dm : w.deepstack_mergers)
    dw->deepstack_mergers.push_back(MakeDevMerger(b, q, dm, cfg));
  b.Synchronize(q);  // resident + ready for any later queue
  b.DestroyQueue(q);
  return dw;
}

namespace {

// One patch-merger (main or deepstack) on resident weights. in = current hidden
// [L, hidden] device bf16; returns [Nmerge, out_hidden] device bf16 into `out`.
void RunMerger(Backend& b, Queue& q, const DevMerger& mw, const Qwen3VLVisionConfig& cfg,
               Tensor hidden, int64_t L, Buf& out) {
  const int64_t H = cfg.hidden_size;
  const int64_t ctx4 = H * cfg.merge_unit();  // 4*context
  const int64_t Nm = L / cfg.merge_unit();
  const float eps = cfg.norm_eps;

  Buf normed(b, q, DType::kBF16, {L, H});
  Buf fc1(b, q, DType::kBF16, {Nm, ctx4});

  if (mw.use_postshuffle_norm) {
    // x.view(-1, ctx4) THEN norm over ctx4.
    Tensor xv = hidden;  // [L,H] contiguous == [Nm,ctx4] reinterpret
    xv.rank = 2; xv.shape[0] = Nm; xv.shape[1] = ctx4; xv.stride[0] = ctx4; xv.stride[1] = 1;
    Buf nrm(b, q, DType::kBF16, {Nm, ctx4});
    vt::LayerNorm(q, nrm.tensor(), xv, &mw.norm_w.tensor(), &mw.norm_b.tensor(),
                  vt::LayerNormArgs{eps});
    LinearBias(q, fc1, nrm.tensor(), mw.fc1_w.tensor(), &mw.fc1_b.tensor());
  } else {
    // norm over context_dim (H) THEN view(-1, ctx4).
    vt::LayerNorm(q, normed.tensor(), hidden, &mw.norm_w.tensor(), &mw.norm_b.tensor(),
                  vt::LayerNormArgs{eps});
    Tensor nv = normed.tensor();  // [L,H] -> [Nm,ctx4]
    nv.rank = 2; nv.shape[0] = Nm; nv.shape[1] = ctx4; nv.stride[0] = ctx4; nv.stride[1] = 1;
    LinearBias(q, fc1, nv, mw.fc1_w.tensor(), &mw.fc1_b.tensor());
  }
  vt::GeluErf(q, fc1.tensor(), fc1.tensor());
  LinearBias(q, out, fc1.tensor(), mw.fc2_w.tensor(), &mw.fc2_b.tensor());
}

}  // namespace

// --- the resident-weights forward (the fast/production path) ------------------
std::vector<float> Qwen3VLVisionForward(const std::vector<uint16_t>& pixel_values_bf16,
                                        const std::array<int64_t, 3>& grid_thw,
                                        const Qwen3VLVisionDeviceWeights& dw,
                                        const Qwen3VLVisionConfig& cfg, Backend& b,
                                        Qwen3VLVisionCapture* cap) {
  Queue q = b.CreateQueue();
  const int64_t H = cfg.hidden_size;
  const int64_t nh = cfg.num_heads;
  const int64_t hd = cfg.head_dim();
  const int64_t I = cfg.intermediate_size;
  const int64_t patch_dim =
      cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
  const int64_t L = grid_thw[0] * grid_thw[1] * grid_thw[2];
  const int64_t half = hd / 2;
  const float eps = cfg.norm_eps;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  // --- inputs (the only per-image uploads) ------------------------------------
  Buf pix(b, q, DType::kBF16, {L, patch_dim}, pixel_values_bf16.data());

  // patch_embed: [L,patch_dim] @ proj_w[H,patch_dim]^T + bias -> [L,H].
  Buf hidden(b, q, DType::kBF16, {L, H});
  LinearBias(q, hidden, pix.tensor(), dw.patch_proj_w.tensor(), &dw.patch_proj_b.tensor());
  if (cap != nullptr) {
    cap->patch_embed_out.resize(static_cast<size_t>(L) * H);
    std::vector<uint16_t> tmp(static_cast<size_t>(L) * H);
    hidden.Download(q, tmp.data());
    for (size_t i = 0; i < tmp.size(); ++i) cap->patch_embed_out[i] = vt::BF16ToF32(tmp[i]);
  }

  // + pos_embeds (host interp, uploaded bf16).
  std::vector<float> pos = VisionPosEmbedInterpolate(dw.pos_embed_w, grid_thw, cfg);
  {
    Buf pe(b, q, DType::kBF16, {L, H});
    const auto pe_bf = ToBf16(pos);
    b.Copy(q, pe.tensor().data, pe_bf.data(), pe_bf.size() * sizeof(uint16_t));
    vt::Add(q, hidden.tensor(), hidden.tensor(), pe.tensor());
  }

  // vision rope cache [L,hd] bf16 = [cos(half)|sin(half)]; positions [0..L-1].
  std::vector<float> rcos, rsin;
  VisionRopeCosSin(grid_thw, cfg, &rcos, &rsin);
  std::vector<float> cache_f(static_cast<size_t>(L) * hd);
  for (int64_t r = 0; r < L; ++r) {
    std::memcpy(&cache_f[static_cast<size_t>(r) * hd], &rcos[static_cast<size_t>(r) * half],
                static_cast<size_t>(half) * sizeof(float));
    std::memcpy(&cache_f[static_cast<size_t>(r) * hd + half],
                &rsin[static_cast<size_t>(r) * half], static_cast<size_t>(half) * sizeof(float));
  }
  Buf cache(b, q, DType::kBF16, {L, hd});
  {
    const auto cache_bf = ToBf16(cache_f);
    b.Copy(q, cache.tensor().data, cache_bf.data(), cache_bf.size() * sizeof(uint16_t));
  }
  std::vector<int32_t> pos_ids(static_cast<size_t>(L));
  for (int64_t i = 0; i < L; ++i) pos_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  Buf posb(b, q, DType::kI32, {L}, pos_ids.data());
  if (cap != nullptr) {
    cap->rotary_cos = rcos;
    cap->rotary_sin = rsin;
    cap->pos_embeds = pos;
  }

  // --- ViT blocks -------------------------------------------------------------
  vt::RopeArgs ra;
  ra.rotary_dim = static_cast<int>(hd);
  ra.is_neox_style = true;
  if (cap != nullptr) cap->deepstack_out.clear();
  std::vector<std::vector<float>> ds_features(cfg.deepstack_visual_indexes.size());

  // Per-block SCRATCH buffers hoisted OUT of the loop and reused every block —
  // every block writes them fresh (LayerNorm/GEMM outputs), so reuse is
  // BIT-IDENTICAL. This is THE tower lever: nsys attributed ~100% of the ~1.6 s
  // forward to per-op cudaMalloc/cudaFree (each Buf alloc/free synchronizes the
  // device on GB10; the ViT kernels themselves are ~1.6 ms total). Allocating
  // once collapses ~250 allocs/forward to a handful (mirror torch's caching
  // allocator, which makes vLLM's per-op tensors free).
  Buf n1(b, q, DType::kBF16, {L, H});
  Buf qb(b, q, DType::kBF16, {L, H}), kb(b, q, DType::kBF16, {L, H}),
      vb(b, q, DType::kBF16, {L, H});
  // C2 fold scratch: merged qkv GEMM output [L, 3H] (view-split into qb/kb/vb).
  Buf qkv(b, q, DType::kBF16, {L, 3 * H});
  Buf ao(b, q, DType::kBF16, {L, nh, hd});
  Buf attn(b, q, DType::kBF16, {L, H});
  Buf n2(b, q, DType::kBF16, {L, H});
  Buf f1(b, q, DType::kBF16, {L, I});
  Buf f2(b, q, DType::kBF16, {L, H});

  // Vision self-attention kernel selection (per-forward; the forward runs once per
  // image so getenv cost is nil). DEFAULT: vt::AttentionDenseFlash — the
  // SHARED-MEMORY-TILED flash kernel (spec §14/§16). Its per-warp online-softmax
  // recurrence is BYTE-FOR-BYTE the AttentionDenseFast (AttentionWarpKernel) math —
  // identical per-lane head_dim grouping, identical butterfly reduction, identical
  // sequential j-order — only K/V are read from shared-memory tiles reused across a
  // block of query-warps instead of re-streamed from global per (query,head). So the
  // output is BIT-IDENTICAL to the warp kernel (token-identical, goldens unchanged),
  // while killing AttentionDenseFast's O(t^2) redundant global K/V reads over the 784
  // non-causal patches. Same-binary A/B knobs: VT_QWEN3VL_ATTN_WARP=1 → warp
  // AttentionDenseFast (the pre-§16 default); VT_QWEN3VL_ATTN_EAGER=1 → naive
  // kAttention. kAttention (text/audio decode) is untouched by all three.
  const int vis_attn = [] {
    const char* e = std::getenv("VT_QWEN3VL_ATTN_EAGER");
    if (e != nullptr && e[0] == '1') return 0;  // naive kAttention
    const char* w = std::getenv("VT_QWEN3VL_ATTN_WARP");
    if (w != nullptr && w[0] == '1') return 1;  // warp AttentionDenseFast
    return 2;                                   // flash-tiled AttentionDenseFlash (default)
  }();

  for (int64_t l = 0; l < cfg.depth; ++l) {
    const DevBlock& db = dw.blocks[static_cast<size_t>(l)];
    // norm1
    vt::LayerNorm(q, n1.tensor(), hidden.tensor(), &db.norm1_w.tensor(), &db.norm1_b.tensor(),
                  vt::LayerNormArgs{eps});
    // qkv (C2 fold): ONE MatmulBT over the resident merged qkv_w [3H,H] + a fused
    // merged [3H] BIAS epilogue, then a contiguous QkvSplit into qb/kb/vb [L,H].
    // BIT-identical to the prior 3x {LinearBias(row-slice)} (merged GEMM math ==
    // per-slice GEMM math; [3H] bias add broadcasts per column == 3x [H] adds;
    // QkvSplit is a pure contiguous copy). The merged-bias epilogue is the NEW
    // C2 piece vs the text bf16 merged-QKV (which carries no qkv bias).
    {
      Tensor qbias = db.qkv_b.tensor();
      models::FusedMergedQkvBiasSplit(q, qkv.tensor(), qb.tensor(), kb.tensor(),
                                      vb.tensor(), n1.tensor(), db.qkv_w.tensor(),
                                      &qbias);
    }
    // rope on q,k viewed [L,nh,hd].
    Tensor q3 = qb.tensor(); q3.rank = 3; q3.shape[0] = L; q3.shape[1] = nh; q3.shape[2] = hd;
    q3.stride[0] = nh * hd; q3.stride[1] = hd; q3.stride[2] = 1;
    Tensor k3 = kb.tensor(); k3.rank = 3; k3.shape[0] = L; k3.shape[1] = nh; k3.shape[2] = hd;
    k3.stride[0] = nh * hd; k3.stride[1] = hd; k3.stride[2] = 1;
    vt::RopeFromCache(q, q3, &k3, posb.tensor(), cache.tensor(), ra);
    // Non-causal attention, WINDOWED PER FRAME (image grid_t==1 == single window).
    Tensor v3 = vb.tensor(); v3.rank = 3; v3.shape[0] = L; v3.shape[1] = nh; v3.shape[2] = hd;
    v3.stride[0] = nh * hd; v3.stride[1] = hd; v3.stride[2] = 1;
    {
      const int64_t nframes = grid_thw[0];
      const int64_t hw = grid_thw[1] * grid_thw[2];  // patches per frame
      const size_t elt = vt::SizeOf(DType::kBF16);
      auto frame_slice = [&](const Tensor& src, int64_t f) -> Tensor {
        Tensor s = src;
        s.shape[0] = hw;
        s.data = static_cast<char*>(src.data) +
                 static_cast<size_t>(f * hw * nh * hd) * elt;
        return s;
      };
      for (int64_t f = 0; f < nframes; ++f) {
        Tensor qf = frame_slice(q3, f), kf = frame_slice(k3, f),
               vf = frame_slice(v3, f), aof = frame_slice(ao.tensor(), f);
        // Dense non-causal attention, head_dim 72, windowed per frame (image
        // grid_t==1 == single 784-patch window). Default flash-tiled (byte-identical
        // to warp; see vis_attn above). kAttention (text/audio) is untouched.
        const vt::AttentionArgs aargs{scale, /*causal=*/false};
        // VT-ATTN-NAIVE: the EAGER rung of the same-binary A/B above, reachable
        // only with VT_QWEN3VL_ATTN_EAGER=1, which nothing in the tree sets. The
        // default is the flash-tiled rung two branches down. Rerouting this arm
        // would delete the baseline the other two are measured against (#1544).
        if (vis_attn == 0)
          vt::Attention(q, aof, qf, kf, vf, aargs);
        else if (vis_attn == 1)
          vt::AttentionDenseFast(q, aof, qf, kf, vf, aargs);
        else
          vt::AttentionDenseFlash(q, aof, qf, kf, vf, aargs);
      }
    }
    // proj + residual.
    Tensor ao2 = ao.tensor(); ao2.rank = 2; ao2.shape[0] = L; ao2.shape[1] = H;
    ao2.stride[0] = H; ao2.stride[1] = 1;
    LinearBias(q, attn, ao2, db.proj_w.tensor(), &db.proj_b.tensor());
    vt::Add(q, hidden.tensor(), hidden.tensor(), attn.tensor());
    // norm2 + MLP + residual.
    vt::LayerNorm(q, n2.tensor(), hidden.tensor(), &db.norm2_w.tensor(), &db.norm2_b.tensor(),
                  vt::LayerNormArgs{eps});
    LinearBias(q, f1, n2.tensor(), db.fc1_w.tensor(), &db.fc1_b.tensor());
    vt::GeluTanh(q, f1.tensor(), f1.tensor());
    LinearBias(q, f2, f1.tensor(), db.fc2_w.tensor(), &db.fc2_b.tensor());
    vt::Add(q, hidden.tensor(), hidden.tensor(), f2.tensor());

    if (cap != nullptr && l == 0) {
      cap->block0_out.resize(static_cast<size_t>(L) * H);
      std::vector<uint16_t> tmp(static_cast<size_t>(L) * H);
      hidden.Download(q, tmp.data());
      for (size_t i = 0; i < tmp.size(); ++i) cap->block0_out[i] = vt::BF16ToF32(tmp[i]);
    }
    // deepstack tap after this block?
    for (size_t di = 0; di < cfg.deepstack_visual_indexes.size(); ++di) {
      if (cfg.deepstack_visual_indexes[di] == static_cast<int>(l)) {
        const int64_t Nm = L / cfg.merge_unit();
        Buf dsout(b, q, DType::kBF16, {Nm, cfg.out_hidden_size});
        RunMerger(b, q, dw.deepstack_mergers[di], cfg, hidden.tensor(), L, dsout);
        std::vector<uint16_t> tmp(static_cast<size_t>(Nm) * cfg.out_hidden_size);
        dsout.Download(q, tmp.data());
        std::vector<float> f(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) f[i] = vt::BF16ToF32(tmp[i]);
        ds_features[di] = f;
        if (cap != nullptr) {
          if (cap->deepstack_out.size() <= di) cap->deepstack_out.resize(di + 1);
          cap->deepstack_out[di] = std::move(f);
        }
      }
    }
  }

  // --- merger + deepstack concat -> [Nm, out_hidden*(1+ndeep)] -----------------
  const int64_t Nm = L / cfg.merge_unit();
  const int64_t D = cfg.out_hidden_size;
  const int64_t ndeep = static_cast<int64_t>(cfg.deepstack_visual_indexes.size());
  Buf mout(b, q, DType::kBF16, {Nm, D});
  RunMerger(b, q, dw.merger, cfg, hidden.tensor(), L, mout);
  std::vector<float> merger_f(static_cast<size_t>(Nm) * D);
  {
    std::vector<uint16_t> tmp(static_cast<size_t>(Nm) * D);
    mout.Download(q, tmp.data());
    for (size_t i = 0; i < tmp.size(); ++i) merger_f[i] = vt::BF16ToF32(tmp[i]);
  }
  if (cap != nullptr) cap->merger_out = merger_f;

  std::vector<std::vector<float>>& ds = ds_features;

  // concat: [merger | ds0 | ds1 | ds2] along dim1.
  const int64_t W = D * (1 + ndeep);
  std::vector<float> tower(static_cast<size_t>(Nm) * W);
  for (int64_t r = 0; r < Nm; ++r) {
    std::memcpy(&tower[static_cast<size_t>(r) * W], &merger_f[static_cast<size_t>(r) * D],
                static_cast<size_t>(D) * sizeof(float));
    for (int64_t di = 0; di < ndeep; ++di)
      std::memcpy(&tower[static_cast<size_t>(r) * W + (di + 1) * D],
                  &ds[static_cast<size_t>(di)][static_cast<size_t>(r) * D],
                  static_cast<size_t>(D) * sizeof(float));
  }

  b.DestroyQueue(q);
  return tower;
}

// --- host-weights overload: prepare-then-forward (BIT-IDENTICAL wrapper) -------
std::vector<float> Qwen3VLVisionForward(const std::vector<uint16_t>& pixel_values_bf16,
                                        const std::array<int64_t, 3>& grid_thw,
                                        const Qwen3VLVisionWeights& w,
                                        const Qwen3VLVisionConfig& cfg, Backend& b,
                                        Qwen3VLVisionCapture* cap) {
  using clock = std::chrono::steady_clock;
  const bool prof = std::getenv("VLLM_MM_TOWER_PROFILE") != nullptr;
  const auto t0 = clock::now();
  const std::shared_ptr<Qwen3VLVisionDeviceWeights> dw =
      PrepareVisionDeviceWeights(w, cfg, b);  // synchronizes internally
  const auto t1 = clock::now();
  std::vector<float> out = Qwen3VLVisionForward(pixel_values_bf16, grid_thw, *dw, cfg, b, cap);
  const auto t2 = clock::now();
  if (prof) {
    const double marshal_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double compute_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::fprintf(stderr,
                 "[VLLM_MM_TOWER_PROFILE] prepare(marshal)=%.1f ms  forward(compute)=%.1f ms  "
                 "total=%.1f ms\n",
                 marshal_ms, compute_ms, marshal_ms + compute_ms);
  }
  return out;
}

}  // namespace vllm::multimodal
