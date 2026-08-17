// Nemotron-H (`NemotronHForCausalLM`) — A2-R, the DEVICE arm
// ([spec](../../../../.agents/specs/nemotron-h-abi-e2e.md), issue #810; parent
// row #517).
//
// ─── WHAT RUNS WHERE, WHICH IS THE WHOLE SCOPE OF THIS FILE ─────────────────
//
//   DEVICE  the embedding lookup, all 52 layer norms + norm_f, and the 6 GQA
//           attention blocks. The residual stream is device-resident for the
//           entire forward.
//   HOST    the 23 Mamba2 blocks, the 23 MoE blocks, and lm_head.
//
// The line is drawn by the MEMORY FORMAT THE CHECKPOINT SHIPS, not by taste.
// The released checkpoint is `quant_algo: MIXED_PRECISION` and only 216 of its
// tensors are plain bf16 (gated: test_nemotron_h_loader.cpp:254). Everything
// this file puts on the device is from that bf16 population. Everything it
// leaves on the host is not:
//
//   * the 23 Mamba2 blocks are entered through `mixer.in_proj`, which is FP8
//     W8A8 static. The block is NOT splittable — `in_proj` produces the fused
//     zxbcdt that the conv and the scan both consume — so it moves as a unit,
//     after the shared FP8 W8A8 linear seam is extracted out of qwen3_5.cpp
//     (`ResidentFp8` :1456, `MatmulFp8CutlassD` :1495) into a real shared
//     header. That extraction is issue #940 and is deliberately NOT done here:
//     its own gate is Qwen3.5 byte-identity, which has no business landing
//     inside a NemotronH row.
//   * the 23 MoE blocks' 5888 routed + 23 shared expert projections and
//     `lm_head` are NVFP4 W4A16 g16 — 30.19e9 parameters, 15.8 GiB packed and
//     56.2 GiB dequantized to bf16. There is NO device NVFP4->bf16 dequant
//     kernel in vt (the only standalone device dequant is
//     `vt::DequantFp8ChannelBf16`, fused_ops.h:31, ROCm-only and per-channel
//     FP8), so putting them on the device in bf16 would mean a host dequant
//     plus a 56.2 GiB upload. Both gate hosts are unified-memory, so that is a
//     REBOOT rather than an OOM. nemotron_h_loader.h:36-46 rejected the same
//     design for the load; this file does not re-open it.
//
// So 46 of 52 layers still compute on the host, and each costs one download of
// the normed hidden and one upload of the mixer output. THAT BOUNCE IS
// SCAFFOLD, NOT ARCHITECTURE: every later unit deletes one pair of it. It is
// also why this file makes NO SPEED CLAIM OF ANY KIND — it is slower than the
// host reference, and that is expected and irrelevant to what it gates.
//
// ─── WHY NOT `dense_attn::AttnBlock` (issue #941) ──────────────────────────
//
// The spec's §2 names `dense_attn::AttnBlock` as this model's seam. That is
// wrong, and routing through it would reintroduce the exact defect #810 just
// removed from the runner — a shared function reading HF-config fields this
// architecture does not ship. Three measured reasons:
//
//   1. It takes `Qwen3DenseAttnWeights` (dense_attn_block.h:335), not this
//      model's separate q/k/v/o.
//   2. It reads `cfg.rms_norm_eps`. NemotronH's config ships
//      `layer_norm_epsilon` and `norm_eps` and NO `rms_norm_eps`, which
//      `hf_config.cpp:551` defaults to **0.0** — a silent eps=0 normalization.
//   3. Its default path calls `vt::RopeNeox` unconditionally
//      (dense_attn_block.h:496). NemotronH has no positional embedding at all
//      (`kNemotronHAttentionHasNoRope`; nemotron_h.py:473-486 @ 555967922).
//
// The tree's own idiom for exactly this case is a MODEL-LOCAL block —
// `granite.cpp:84 GraniteAttnBlock`, `gemma.cpp:42`, `gemma2.cpp:123`,
// `gemma3.cpp:108`, `glm4.cpp:80`, `gemma4.cpp:206`, none of them allowlisted.
// `NemotronHAttnBlock` below follows it and documents its deltas.
//
// ─── G-SAFE IS UNTOUCHED ────────────────────────────────────────────────────
//
// Nothing in this file consumes `attn_kv`, `gdn_state`, `gdn_meta`,
// `gdn_state_slots` or `num_reqs`. This arm is NON-PAGED and SINGLE-REQUEST: it
// recomputes Q/K/V over the whole sequence every call, exactly as the host
// reference does. So it does not create the capability the interlock at
// `nemotron_h_registry.cpp:161-170` guards, and all three of that interlock's
// clauses stay exactly as they are. A2-P narrows them; A2-R does not.
#include "vllm/model_executor/models/nemotron_h_forward.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// The SHARED device glue: Dev, DBuf, MakeTensor, Reshape and — the reason this
// header rather than dense_device_glue.h — `ResidentWeight`, the lazy
// upload-once seam this row converts NemotronH's dense weights onto.
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"  // kFusedAddRmsNormStd

// DSR-ALLOW(A2-Q2a): TYPES, not behaviour -- vt/cuda/marlin_repack.h is BUILT ONLY under VT_MARLIN_NVFP4 (its own header note), so this include cannot be resolved by a runtime query. This is the platform leg for the CUDA Marlin repack.
#ifdef VT_MARLIN_NVFP4
// A2-Q2a: the load-time NVFP4 -> Marlin repack primitives. DELIBERATELY NOT
// `dense_nvfp4_gemm.h`: that header's `MarlinDenseResidentFor` (:379) keys its
// repack cache on the WEIGHT'S ADDRESS, which is issue #984, and including it
// here would put the unsafe accessor one unqualified call away from a reviewer's
// eye. These four functions are the same primitives it and qwen3_5.cpp both
// drive, with no cache attached.
#include "vt/cuda/marlin_repack.h"
#endif

namespace vllm {
namespace {

using vt::DType;
using vt::Queue;
using vt::Tensor;

// Same reuse the other model-local blocks take (granite.cpp:52): Dev / DBuf /
// ResidentWeight / MakeTensor / Reshape verbatim, and NOT AttnBlock.
using namespace dense_attn;

int64_t NumelOf(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// VT_FUSED_CHAIN_ADOPT, read exactly as the host arm reads it
// (nemotron_h.cpp:55). This is NOT an incidental duplicate: the whole point of
// A2-R's gate is that the two arms compose the IDENTICAL vt:: op sequence and
// differ only in which backend runs it. A device arm that hand-called
// `vt::RmsNorm(..., &residual)` while the host arm routed the same chain
// through `vt::FusedChain` would be comparing two different compositions and
// calling the result an equivalence — and `scripts/check-fusion-consistency.py`
// refuses it outright (AGENTS.md, "Route model fusion through `vt::FusedChain`").
//
// PRECISELY WHAT IS AND IS NOT GUARANTEED. These are TWO file-local
// function-local statics with byte-identical predicates — this one and
// `nemotron_h.cpp:56` — each latching on its own first call. They read the same
// variable with the same default, so within one process they resolve the same
// way and the equivalence gate below compares like with like. What is NOT true,
// and an earlier draft of this comment claimed, is that they "can never
// straddle": a caller that changed `VT_FUSED_CHAIN_ADOPT` between the two first
// calls would latch two different answers. Nothing in this tree does that, and
// the duplication is the tree's idiom for this env read (qwen3_5.cpp:1699), but
// the property is a convention rather than an impossibility. Hoisting the
// predicate into one shared reader would make it an impossibility, and that is
// a tree-wide change, not this row's.
bool FusedChainAdoptEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// The residual-add + RMSNorm preamble, through the declared fusion recipe.
// `residual` is updated in place (res += x) and `out` receives the norm, which
// is the contract `kFusedAddRmsNormStd` encodes and the Tier-0 composite
// dispatches to the same `vt::RmsNorm(..., &residual)` primitive, so the two
// branches are bit-identical (tests/vt/test_ops_fused_chain.cpp).
void AddRmsNorm(Dev d, Tensor& out, Tensor& x, const Tensor& w, Tensor& residual,
                const vt::RmsNormArgs& nargs, double eps) {
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, out, x, w, &residual, vt::kFusedAddRmsNormStd,
                   static_cast<float>(eps));
  } else {
    vt::RmsNorm(d.q, out, x, w, nargs, &residual);
  }
}

// Pack a host f32 vector into `dt` and upload it. The staging buffer is kept
// alive across an explicit Synchronize: `DBuf`'s constructor issues an ASYNC
// `cudaMemcpyAsync` (dense_device_glue.h:126, cuda_backend.cu:90) and does not
// wait, so a staging buffer that died at the end of this function would be a
// use-after-free that pageable-memory semantics happen to hide most of the
// time. This arm records no throughput number, so an explicit sync per upload
// costs nothing it is measuring.
DBuf UploadAs(Dev d, const std::vector<float>& v, DType dt,
              const std::vector<int64_t>& shape) {
  const int64_t n = NumelOf(shape);
  VT_CHECK(static_cast<int64_t>(v.size()) == n,
           "NemotronH device: upload element count does not match the shape");
  std::vector<uint8_t> staging(static_cast<size_t>(n) * vt::SizeOf(dt));
  if (dt == DType::kF32) {
    std::memcpy(staging.data(), v.data(), staging.size());
  } else {
    auto* dst = reinterpret_cast<uint16_t*>(staging.data());
    for (int64_t i = 0; i < n; ++i) dst[i] = vt::F32ToBF16(v[static_cast<size_t>(i)]);
  }
  DBuf b(d, dt, shape, staging.data());
  d.b.Synchronize(d.q);
  return b;
}

// Download a device buffer and widen it to f32, the comparison currency every
// host-side entry point in nemotron_h_forward.h already speaks.
std::vector<float> DownloadF32(Dev d, DBuf& b, DType dt, int64_t n) {
  std::vector<uint8_t> staging(static_cast<size_t>(n) * vt::SizeOf(dt));
  b.Download(d, staging.data());  // Copy + Synchronize (dense_device_glue.h:158)
  std::vector<float> out(static_cast<size_t>(n));
  if (dt == DType::kF32) {
    std::memcpy(out.data(), staging.data(), staging.size());
  } else {
    const auto* src = reinterpret_cast<const uint16_t*>(staging.data());
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  }
  return out;
}

// Refuse by name when a weight the device arm is about to upload is absent or
// mis-shaped. Mirrors the host arm's `RequireWeight` (nemotron_h.cpp:180) —
// same three properties, so a defect that would refuse on the host refuses here
// too rather than reaching a kernel with a null pointer.
void RequireDeviceWeight(const OwnedTensor& w, const char* what, DType want,
                         const std::vector<int64_t>& shape) {
  VT_CHECK(!w.Empty(),
           std::string("NemotronH device forward: weight '") + what +
               "' is not materialized");
  VT_CHECK(w.dtype == want, std::string("NemotronH device forward: weight '") + what +
                                "' has the wrong dtype for this arm");
  VT_CHECK(w.rank == static_cast<int>(shape.size()) &&
               std::equal(shape.begin(), shape.end(), w.shape),
           std::string("NemotronH device forward: weight '") + what +
               "' has the wrong shape");
}

// ─── the model-local attention block ────────────────────────────────────────
//
// One NemotronH GQA self-attention block (nemotron_h.py:415-486 @ 555967922).
// `normed` is the already-normed hidden [T,H] in `adt` on the device; returns
// the o_proj output [T,H] in `adt` on the device.
//
// DELTAS vs `dense_attn::AttnBlock`, each one measured, not assumed:
//   (a) NO POSITIONAL EMBEDDING. `nemotron_h.py` contains zero occurrences of
//       `rope`/`rotary`/`Rotary`; `.forward` (:473-486) is qkv -> split -> attn
//       -> o_proj with no rotation step. `kNemotronHAttentionHasNoRope` states
//       it and the mutation gate proves the property is armed NUMERICALLY —
//       applying a rotation here changes no tensor shape and need not move a
//       token on a short prompt.
//   (b) SEPARATE q/k/v, not a merged qkv owner. The checkpoint ships
//       `q_proj`/`k_proj`/`v_proj` as three tensors (upstream fuses them at
//       load through its stacked-params mapping); `EnumerateNemotronHTensors`
//       already claims them separately, so there is nothing to merge here.
//   (c) NO per-head q/k RMSNorm. `NemotronHAttention.__init__` (:415-471)
//       builds `qkv_proj`, `o_proj` and `Attention` and nothing else.
//   (d) NO biases — `attention_bias=false` on this checkpoint and the loader
//       ships no q/k/v/o bias tensor.
//   (e) NON-PAGED. No `PagedKvCache`, no `ReshapeAndCache`, no slot mapping: a
//       dense causal `vt::Attention` over the whole [T,·], which is exactly
//       what the host reference does (nemotron_h.cpp:615-626). That is what
//       keeps this arm inside G-SAFE, and it is A2-P that makes it paged.
//   (f) Every geometry and epsilon comes from `NemotronHParams`, never from
//       `HfConfig` — the defect class of #810 and #941.
DBuf NemotronHAttnBlock(Dev d, const NemotronHAttentionWeights& w,
                        const NemotronHParams& params, const Tensor& normed,
                        int64_t T, DType adt) {
  const int64_t H = params.hidden_size;
  const int64_t Hq = params.num_attention_heads;
  const int64_t Hkv = params.num_key_value_heads;
  const int64_t Dh = params.head_dim;
  const int64_t qdim = params.q_proj_out_features();
  const int64_t kvdim = params.kv_proj_out_features();

  // The same two refusals the host arm raises, in the same order, so an
  // unsupported checkpoint fails identically on both arms.
  VT_CHECK(!params.attention_bias,
           "NemotronH device forward: attention_bias is not ported (the "
           "checkpoint has attention_bias=false and ships no q/k/v/o bias)");
  VT_CHECK(!params.sliding_window.has_value(),
           "NemotronH device forward: per-layer sliding_window is not ported "
           "(this checkpoint ships sliding_window=null)");

  RequireDeviceWeight(w.q_proj, "mixer.q_proj", adt, {qdim, H});
  RequireDeviceWeight(w.k_proj, "mixer.k_proj", adt, {kvdim, H});
  RequireDeviceWeight(w.v_proj, "mixer.v_proj", adt, {kvdim, H});
  RequireDeviceWeight(w.o_proj, "mixer.o_proj", adt, {H, qdim});

  // `nk` IS CONSUMED HERE, exactly as the host arm consumes it
  // (nemotron_h.cpp:306). All four projections below go through `vt::MatmulBT`,
  // which reads `b` as [N=out, K=in] — the raw torch-Linear orientation
  // `nk = true` names. A weight recorded as [K, N] is a transposed GEMM operand
  // with the same shape and the same byte count, so nothing else here would
  // notice it.
  VT_CHECK(w.q_proj.nk && w.k_proj.nk && w.v_proj.nk && w.o_proj.nk,
           "NemotronH device forward: an attention projection is not in the "
           "[out, in] torch-Linear orientation vt::MatmulBT consumes");

  // THE RESIDENCY SEAM. Each of the four uploads ONCE, on the first step, and
  // every later step reuses the same device allocation. This is the shared
  // `dense_attn::ResidentWeight` (dense_attn_block.h:178) and not a local
  // equivalent: a `d_dev` bolted onto NemotronHOwned would have been the
  // parallel path AGENTS.md forbids, even though it is the smaller diff.
  Tensor wq = ResidentWeight(d, w.q_proj);
  Tensor wk = ResidentWeight(d, w.k_proj);
  Tensor wv = ResidentWeight(d, w.v_proj);
  Tensor wo = ResidentWeight(d, w.o_proj);

  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kvdim});
  DBuf v(d, adt, {T, kvdim});
  vt::MatmulBT(d.q, q.t(), normed, wq);
  vt::MatmulBT(d.q, k.t(), normed, wk);
  vt::MatmulBT(d.q, v.t(), normed, wv);

  // (a) NO RoPE HERE. This gap is the port, not an omission.

  DBuf attn(d, adt, {T, Hq, Dh});
  {
    Tensor qt = Reshape(q.t(), {T, Hq, Dh});
    Tensor kt = Reshape(k.t(), {T, Hkv, Dh});
    Tensor vt_ = Reshape(v.t(), {T, Hkv, Dh});
    vt::AttentionArgs args;
    // `self.scaling = self.head_dim**-0.5` (nemotron_h.py:440) — the SAME
    // expression the host arm evaluates (nemotron_h.cpp:622), in f64 before the
    // narrowing, so the two arms feed `vt::Attention` bit-identical scales.
    args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(Dh)));
    args.causal = true;
    vt::Attention(d.q, attn.t(), qt, kt, vt_, args);
  }

  DBuf out(d, adt, {T, H});
  {
    Tensor at = Reshape(attn.t(), {T, qdim});
    vt::MatmulBT(d.q, out.t(), at, wo);
  }
  return out;
}

// Refuse by name unless every expert projection is the NVFP4 W4A16 g16 form the
// arena is built from. The synthetic `BuildTiny` fixture is all `kDense`, so
// this is what keeps it on the host reference arm instead of reaching a kernel
// with garbage.
bool MoeIsNvfp4(const NemotronHMoeWeights& w) {
  auto q = [](const NemotronHOwned& t) {
    return t.form == NemotronHWeightForm::kNvfp4W4A16G16 && !t.bytes.empty() &&
           !t.scale.empty();
  };
  if (w.experts.empty()) return false;
  for (const NemotronHExpertWeights& e : w.experts) {
    if (!q(e.up_proj) || !q(e.down_proj)) return false;
  }
  if (w.has_shared && (!q(w.shared.up_proj) || !q(w.shared.down_proj))) return false;
  return true;
}

// DSR-ALLOW(A2-Q2a): TYPES, not behaviour -- the whole arena region names vt::cuda::Marlin* functions declared only in the guarded header above, so it cannot compile on a build without them. vt::OpRegistered answers availability, never declaration. Mirrors laguna.cpp:456, the same CUDA-leg arena.
#ifdef VT_MARLIN_NVFP4

// Fetch (building on first use) the resident state a weight owns. A VERBATIM
// copy of qwen3_5.cpp:680, which is file-local `static` there and so cannot be
// called from here. The duplication is deliberate and is the same call this file
// already makes for `FusedChainAdoptEnabled` (:125): hoisting a helper out of
// qwen3_5.cpp into a shared header is a tree-wide change that would put an
// unrelated diff on this row's critical path. What matters is the PROPERTY, and
// it is identical — the state is keyed on the slot the weights own, never on
// their address (issue #237).
template <typename R>
R& ResidentIn(const ResidentSlot& slot) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  if (!slot.state) slot.state = std::make_shared<R>();
  return *static_cast<R*>(slot.state.get());
}

// ─── A2-Q2a: one MoE layer's device-resident Marlin arena ───────────────────
//
// THE SHAPE IS AN ARENA, NOT A POINTER ARRAY, and that distinction is the whole
// size of this work. `vt::MoeGroupedGemmBf16` (ops.h:1642) takes an `[E]` i64
// device array of per-expert `[K,N]` pointers and needs no repack.
// `vt::MoeGroupedGemmNvfp4Marlin` (ops.h:1685) takes a rank-3 STRIDED arena
// validated at ops.cpp:884, so every expert must be repacked into Marlin's
// interleaved layout at load. The two are not interchangeable.
//
// Sizes at this checkpoint's geometry (H=2688, I=1856, E=128), per MoE layer:
//   w_up   [E, H/16=168, I*2=3712] i32  319.0 MB     s_up   [E,168,1856] i8  39.9 MB
//   w_down [E, I/16=116, H*2=5376] i32  319.3 MB     s_down [E,116,2688] i8  39.9 MB
// = 718 MB per layer x 23 layers = 16.5 GB device-resident, plus 11.2 MB per
// layer for the shared expert's E=1 slice.
//
// PEAK IS THE ARENA PLUS 2.8 MB, NOT PLUS THE RAW TOWER. qwen3_5.cpp:5751 and
// laguna.cpp:638 upload each expert's packed fp4 through `ResidentNvfp4`, which
// CACHES the device copy on the weight, so both accumulate the whole raw tower
// and free it in a tail sweep after the loop (qwen3_5.cpp:5820-5857). The A2-Q2
// spec's §3 read that as "freed as the repack proceeds"; it is not, and a
// whole-model loop written from that reading would peak at 16.5 + 15.8 = 32 GB
// on a box that reboots rather than OOM-kills. This build instead streams each
// expert through ONE REUSED 2.8 MB staging pair, so there is no tower to free
// and no accumulation to get wrong.
struct NemotronHMoeMarlinResident {
  void* w_up = nullptr;    // i32 [E, H/16, I*2]
  void* s_up = nullptr;    // fp8 [E, H/16, I]
  void* g_up = nullptr;    // f32 [E]
  void* w_down = nullptr;  // i32 [E, I/16, H*2]
  void* s_down = nullptr;  // fp8 [E, I/16, H]
  void* g_down = nullptr;  // f32 [E]
  // The shared expert, as an E=1 slice of the same machinery — the documented
  // dense route (dense_nvfp4_gemm.h:38-43: "the SINGLE-EXPERT grouped GEMM is
  // how a dense [M,K]x[N,K]^T W4A16 linear runs on the MoE Marlin entry point",
  // which is also how vLLM reaches the same csrc kernel). Using it here means
  // NO second mechanism, no `Nvfp4Weight` copy of the 23 shared pairs, and
  // therefore no change to the pinned `rep.host_bytes`.
  void* sw_up = nullptr;   // i32 [1, H/16, Is*2]
  void* ss_up = nullptr;   // fp8 [1, H/16, Is]
  void* sg_up = nullptr;   // f32 [1]
  void* sw_down = nullptr;  // i32 [1, Is/16, H*2]
  void* ss_down = nullptr;  // fp8 [1, Is/16, H]
  void* sg_down = nullptr;  // f32 [1]
  void* workspace = nullptr;  // i32 [sms*4] reduction locks
  int sms = 0;
  bool ready = false;
};


// Upload one DENSE `NemotronHOwned` to the device, refusing by name on the same
// three properties `RequireWeight` checks on the host arm (nemotron_h.cpp:180),
// so an absent or mis-shaped router weight fails identically on both arms rather
// than reaching a kernel with a null pointer. Uploaded per call, not resident:
// the router gate is 1.4 MB and A2-Q2a makes no speed claim (:39).
DBuf UploadOwned(Dev d, const NemotronHOwned& w, const char* what, DType want,
                 const std::vector<int64_t>& shape) {
  VT_CHECK(!w.Empty(),
           std::string("NemotronH device moe: weight '") + what + "' is not materialized");
  VT_CHECK(w.IsDense(),
           std::string("NemotronH device moe: weight '") + what +
               "' is not dense; the router is bf16/f32 on this checkpoint, never quantized");
  VT_CHECK(w.dtype == want, std::string("NemotronH device moe: weight '") + what +
                                "' has the wrong dtype for this arm");
  VT_CHECK(w.shape == shape,
           std::string("NemotronH device moe: weight '") + what + "' has the wrong shape");
  DBuf b(d, want, shape, w.bytes.data());
  d.b.Synchronize(d.q);  // `w.bytes` outlives this, but the copy is async; see UploadAs
  return b;
}

// Repack ONE projection into `dst_w`/`dst_s`, streaming its packed codes and
// group scales through the caller's reused staging buffers. `K`/`N` are the
// LOGICAL in/out features: the on-disk weight is [N, K/2] packed and
// [N, K/16] scales, which is exactly what `MarlinRepackExpertWeight` and
// `MarlinProcessExpertScales` read (marlin_repack.h:15,:21).
void RepackOne(Dev d, const NemotronHOwned& src, void* stage_w, void* stage_s,
               uint32_t* dst_w, uint8_t* dst_s, int K, int N, float sf) {
  const size_t packed_b = static_cast<size_t>(N) * (static_cast<size_t>(K) / 2);
  const size_t scale_b = static_cast<size_t>(N) * (static_cast<size_t>(K) / 16);
  VT_CHECK(src.bytes.size() == packed_b,
           "NemotronH MoE repack: packed byte count does not match [N, K/2]");
  VT_CHECK(src.scale.size() == scale_b,
           "NemotronH MoE repack: group-scale byte count does not match [N, K/16]");
  // Copy then repack on the SAME stream, so this expert's repack reads its
  // staging bytes before the next expert's copy overwrites them — the ordering
  // qwen3_5.cpp:5765 relies on for its fused-w13 staging.
  d.b.Copy(d.q, stage_w, src.bytes.data(), packed_b);
  d.b.Copy(d.q, stage_s, src.scale.data(), scale_b);
  vt::cuda::MarlinRepackExpertWeight(d.q.handle, d.q.device.index, dst_w,
                                     static_cast<const uint8_t*>(stage_w), K, N);
  vt::cuda::MarlinProcessExpertScales(d.q.handle, static_cast<const uint8_t*>(stage_s), dst_s,
                                      K, N, sf);
}

// `combined_scale_factor` over every expert sharing ONE Marlin GEMM — up
// together, down together, mirroring qwen3_5.cpp:5716's gu/dn split. It reads
// HOST scale buffers (marlin_repack.h:38), which is what `NemotronHOwned::scale`
// already is, so nothing round-trips.
float CombinedSfOver(const NemotronHMoeWeights& w, bool up) {
  std::vector<const uint8_t*> bufs;
  std::vector<size_t> lens;
  for (const NemotronHExpertWeights& e : w.experts) {
    const NemotronHOwned& t = up ? e.up_proj : e.down_proj;
    bufs.push_back(t.scale.data());
    lens.push_back(t.scale.size());
  }
  return vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
}

void BuildNemotronHMoeMarlinResident(Dev d, const NemotronHMoeWeights& w,
                                     const NemotronHParams& params,
                                     NemotronHMoeMarlinResident& mr) {
  if (mr.ready) return;
  const int E = static_cast<int>(params.n_routed_experts);
  const int H = static_cast<int>(params.hidden_size);
  const int I = static_cast<int>(params.moe_intermediate_size);
  const int Is = static_cast<int>(params.moe_shared_expert_intermediate_size *
                                  params.n_shared_experts);

  mr.sms = vt::cuda::MarlinDeviceSms(d.q.device.index);

  const size_t wu_i32 = static_cast<size_t>(H / 16) * (static_cast<size_t>(I) * 2);
  const size_t wd_i32 = static_cast<size_t>(I / 16) * (static_cast<size_t>(H) * 2);
  const size_t su_b = static_cast<size_t>(H / 16) * static_cast<size_t>(I);
  const size_t sd_b = static_cast<size_t>(I / 16) * static_cast<size_t>(H);

  mr.w_up = d.b.Alloc(static_cast<size_t>(E) * wu_i32 * 4);
  mr.s_up = d.b.Alloc(static_cast<size_t>(E) * su_b);
  mr.g_up = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.w_down = d.b.Alloc(static_cast<size_t>(E) * wd_i32 * 4);
  mr.s_down = d.b.Alloc(static_cast<size_t>(E) * sd_b);
  mr.g_down = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.workspace = d.b.Alloc(static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));

  // ONE staging pair, sized for the LARGEST projection this layer repacks (the
  // shared expert's, at Is > I), reused by every expert.
  const size_t stage_w_b = static_cast<size_t>(Is > I ? Is : I) * static_cast<size_t>(H) / 2;
  const size_t stage_s_b = static_cast<size_t>(Is > I ? Is : I) * static_cast<size_t>(H) / 16;
  void* stage_w = d.b.Alloc(stage_w_b > (static_cast<size_t>(H) * static_cast<size_t>(Is) / 2)
                                ? stage_w_b
                                : static_cast<size_t>(H) * static_cast<size_t>(Is) / 2);
  void* stage_s = d.b.Alloc(stage_s_b > (static_cast<size_t>(H) * static_cast<size_t>(Is) / 16)
                                ? stage_s_b
                                : static_cast<size_t>(H) * static_cast<size_t>(Is) / 16);

  const float sf_up = CombinedSfOver(w, /*up=*/true);
  const float sf_down = CombinedSfOver(w, /*up=*/false);

  std::vector<float> gu(static_cast<size_t>(E)), gd(static_cast<size_t>(E));
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    const NemotronHExpertWeights& x = w.experts[se];
    RepackOne(d, x.up_proj, stage_w, stage_s, static_cast<uint32_t*>(mr.w_up) + se * wu_i32,
              static_cast<uint8_t*>(mr.s_up) + se * su_b, H, I, sf_up);
    RepackOne(d, x.down_proj, stage_w, stage_s, static_cast<uint32_t*>(mr.w_down) + se * wd_i32,
              static_cast<uint8_t*>(mr.s_down) + se * sd_b, I, H, sf_down);
    gu[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(x.up_proj.global_scale, sf_up);
    gd[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(x.down_proj.global_scale, sf_down);
  }
  d.b.Copy(d.q, mr.g_up, gu.data(), gu.size() * sizeof(float));
  d.b.Copy(d.q, mr.g_down, gd.data(), gd.size() * sizeof(float));

  if (w.has_shared) {
    const size_t swu_i32 = static_cast<size_t>(H / 16) * (static_cast<size_t>(Is) * 2);
    const size_t swd_i32 = static_cast<size_t>(Is / 16) * (static_cast<size_t>(H) * 2);
    const size_t ssu_b = static_cast<size_t>(H / 16) * static_cast<size_t>(Is);
    const size_t ssd_b = static_cast<size_t>(Is / 16) * static_cast<size_t>(H);
    mr.sw_up = d.b.Alloc(swu_i32 * 4);
    mr.ss_up = d.b.Alloc(ssu_b);
    mr.sg_up = d.b.Alloc(sizeof(float));
    mr.sw_down = d.b.Alloc(swd_i32 * 4);
    mr.ss_down = d.b.Alloc(ssd_b);
    mr.sg_down = d.b.Alloc(sizeof(float));
    // Its own combined scale factor: it is its own one-expert GEMM, so it shares
    // the factor with nobody (the E=1 case of qwen3_5.cpp:5716).
    std::vector<const uint8_t*> ub{w.shared.up_proj.scale.data()};
    std::vector<size_t> ul{w.shared.up_proj.scale.size()};
    std::vector<const uint8_t*> db{w.shared.down_proj.scale.data()};
    std::vector<size_t> dl{w.shared.down_proj.scale.size()};
    const float sf_su = vt::cuda::MarlinNvfp4CombinedScaleFactor(ub, ul);
    const float sf_sd = vt::cuda::MarlinNvfp4CombinedScaleFactor(db, dl);
    RepackOne(d, w.shared.up_proj, stage_w, stage_s, static_cast<uint32_t*>(mr.sw_up),
              static_cast<uint8_t*>(mr.ss_up), H, Is, sf_su);
    RepackOne(d, w.shared.down_proj, stage_w, stage_s, static_cast<uint32_t*>(mr.sw_down),
              static_cast<uint8_t*>(mr.ss_down), Is, H, sf_sd);
    const float g_su = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.shared.up_proj.global_scale, sf_su);
    const float g_sd =
        vt::cuda::MarlinNvfp4ProcessGlobalScale(w.shared.down_proj.global_scale, sf_sd);
    d.b.Copy(d.q, mr.sg_up, &g_su, sizeof(float));
    d.b.Copy(d.q, mr.sg_down, &g_sd, sizeof(float));
  }

  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
  d.b.Synchronize(d.q);  // every repack has landed -> the staging pair is dead
  d.b.Free(stage_w);
  d.b.Free(stage_s);
  mr.ready = true;
}

// The E=1 grouped GEMM every dense NVFP4 projection here runs on: all `M` rows
// route to expert 0. Buffers are pooled `DBuf`s rather than a process-lifetime
// cache — A2-Q2a makes NO speed claim of any kind (the same posture A2-R took,
// :39), and a per-token-count static cache is state whose lifetime A2-P would
// have to revisit the moment this arm is reached from production.
DBuf DenseMarlinE1(Dev d, const Tensor& x, void* w, void* s, void* g, void* ws, int sms,
                   int64_t M, int64_t K, int64_t N) {
  const int Mi = static_cast<int>(M);
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(Mi, 1, 1);
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(Mi, 1, 1, block, &max_tok, &max_blk);
  DBuf ids(d, DType::kI32, {M});
  d.b.Memset(d.q, ids.t().data, 0, static_cast<size_t>(M) * sizeof(int32_t));  // all -> expert 0
  DBuf sorted(d, DType::kI32, {max_tok});
  DBuf experts(d, DType::kI32, {max_blk});
  DBuf npad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(d.q.handle, static_cast<const int32_t*>(ids.t().data), Mi, 1, 1,
                                    block, static_cast<int32_t*>(sorted.t().data),
                                    static_cast<int32_t*>(experts.t().data),
                                    static_cast<int32_t*>(npad.t().data));
  // `mul_topk_weights` is false, so these are read by nothing; the op still
  // requires an f32 tensor of the right extent.
  const std::vector<float> ones(static_cast<size_t>(M), 1.0F);
  DBuf tw(d, DType::kF32, {M}, ones.data());
  DBuf out(d, DType::kBF16, {M, N});
  Tensor wq = MakeTensor(w, DType::kI32, d.q.device, {1, K / 16, N * 2});
  Tensor sc = MakeTensor(s, DType::kI8, d.q.device, {1, K / 16, N});
  Tensor gg = MakeTensor(g, DType::kF32, d.q.device, {1});
  Tensor wst = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
  vt::MoeGroupedGemmNvfp4Marlin(
      d.q, out.t(), x, wq, sc, gg, wst, sorted.t(), experts.t(), npad.t(), tw.t(),
      vt::MoeMarlinArgs{block, 1, Mi, static_cast<int>(N), static_cast<int>(K), false});
  return out;
}

// ONE NemotronH MoE block on the device. Statement for statement the host arm's
// composition (nemotron_h.cpp:689-824), with the per-(token,slot) MatmulBT loop
// replaced by the grouped Marlin GEMM and NOTHING ELSE changed:
//
//   f32 router GEMM -> MoeRouterTopK(sigmoid, grouped, bias, factor 1.0)
//     -> moe_align -> grouped GEMM (up) -> MoeRelu2 -> grouped GEMM (down)
//     -> shared expert (E=1) -> MoeCombine(routed_scale)
//
// The three properties the whole block exists to gate, unchanged from the host
// arm and each carried by an argument rather than reimplemented:
//   * the router runs in f32 — MIRRORED from `force_fp32_compute=True`
//     (nemotron_h.py:150-156), not a local precision choice;
//   * `routed_scaling_factor` reaches `MoeCombine`'s `routed_scale`, so it
//     multiplies the ROUTED sum on the OUTPUT; the ROUTER's own factor is forced
//     to 1.0 in exactly this case (layer.py:291-300);
//   * the shared expert is added UNSCALED, after the routed sum is scaled
//     (moe_runner.py:402-406 then :722-725) — which is what `MoeCombine`'s
//     `shared` argument does (ops.h:2438-2446).
DBuf NemotronHMoeBlockDevice(Dev d, const NemotronHMoeWeights& w,
                             const NemotronHParams& params, const Tensor& dh, int64_t T) {
  const int64_t H = params.hidden_size;
  const int64_t E = params.n_routed_experts;
  const int64_t Kk = params.num_experts_per_tok;
  const int64_t I = params.moe_intermediate_size;
  const int64_t P = T * Kk;

  VT_CHECK(!params.moe_latent_size.has_value(),
           "NemotronH device moe: moe_latent_size is out of scope "
           "(fc1_latent_proj/fc2_latent_proj); it is null in the released checkpoint");
  VT_CHECK(static_cast<int64_t>(w.experts.size()) == E,
           "NemotronH device moe: expert count does not match n_routed_experts");
  VT_CHECK(dh.dtype == DType::kBF16,
           "NemotronH device moe: the Marlin arm requires a bf16 activation "
           "(ops.cpp:879), which is the released checkpoint's model dtype");

  NemotronHMoeMarlinResident& mr = ResidentIn<NemotronHMoeMarlinResident>(w.moe_marlin);
  // ── LAZY, AND EXPLICITLY TRANSITIONAL (A2-P owns moving it) ────────────────
  // The A2-Q2 spec's §4.2 puts this repack in `PrepareNemotronHForCausalLM`,
  // because a 16.5 GB allocation inside a forward would land inside a CUDA-graph
  // capture. That reason is FORWARD-LOOKING and is false today: nothing captures
  // `NemotronHDeviceForward`, which has no production caller at all
  // (`ForwardNemotronHForCausalLM` still routes to the host reference,
  // nemotron_h_registry.cpp:185-187). Putting it in `Prepare` NOW would instead
  // make every production NemotronH engine load pay 16.5 GB of device memory for
  // a path nothing reaches — `ModelRegistry::Prepare` is called unconditionally
  // from both `GPUModelRunner` constructors (runner.cpp:414, :455) — on a
  // unified-memory box that REBOOTS rather than OOM-kills.
  //
  // "Nothing lands dead" covers an unreached FORWARD, which costs nothing. It
  // does not cover an unreached ALLOCATION inside a REACHED hook. So A2-Q2a
  // builds on first use and `PrepareNemotronHForCausalLM` stays a no-op.
  //
  // THIS IS NOT THE INTENDED END STATE. A2-P moves it to `Prepare` at exactly
  // the moment §4.2's justification stops being false — when a production caller
  // and a capture both exist. Do not read a lazy build here as a decision that
  // the forward is the right home for it.
  if (!mr.ready) BuildNemotronHMoeMarlinResident(d, w, params, mr);

  // --- router. f32 END TO END, exactly as the host arm does it
  // (nemotron_h.cpp:712-733): the activation reaching the router is the
  // MODEL-DTYPE one widened, never a separately-computed f32 activation, so the
  // bf16 rounding the host arm applies first is applied here too by construction
  // (`dh` is already bf16).
  DBuf gate = UploadOwned(d, w.gate, "mixer.gate.weight", DType::kF32, {E, H});
  DBuf bias = UploadOwned(d, w.e_score_correction_bias,
                          "mixer.gate.e_score_correction_bias", DType::kF32, {E});
  DBuf hf32(d, DType::kF32, {T, H});
  vt::CastF32(d.q, hf32.t(), dh);
  DBuf logits(d, DType::kF32, {T, E});
  vt::MatmulBT(d.q, logits.t(), hf32.t(), gate.t());

  DBuf topk_w(d, DType::kF32, {T, Kk});
  DBuf topk_id(d, DType::kI32, {T, Kk});
  {
    vt::MoeRouterTopKArgs args;
    args.top_k = static_cast<int>(Kk);
    args.renormalize = params.norm_topk_prob;
    args.scoring_func = vt::MoeScoringFunc::kSigmoid;  // nemotron_h.py:225
    args.num_expert_group = static_cast<int>(params.n_group);
    args.topk_group = static_cast<int>(params.topk_group);
    // NOT params.routed_scaling_factor — see the block comment above.
    args.routed_scaling_factor = 1.0f;
    Tensor bt = bias.t();
    vt::MoeRouterTopK(d.q, topk_w.t(), topk_id.t(), logits.t(), args, &bt);
  }

  // --- moe_align over the router's top-k ids.
  const int Ti = static_cast<int>(T), Hi = static_cast<int>(H), Ii = static_cast<int>(I);
  const int tki = static_cast<int>(Kk), Ei = static_cast<int>(E), Pi = static_cast<int>(P);
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(Ti, tki, Ei);
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(Ti, tki, Ei, block, &max_tok, &max_blk);
  DBuf sorted(d, DType::kI32, {max_tok});
  DBuf expert_ids(d, DType::kI32, {max_blk});
  DBuf npad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(d.q.handle, static_cast<const int32_t*>(topk_id.t().data), Ti,
                                    tki, Ei, block, static_cast<int32_t*>(sorted.t().data),
                                    static_cast<int32_t*>(expert_ids.t().data),
                                    static_cast<int32_t*>(npad.t().data));

  Tensor wst = MakeTensor(mr.workspace, DType::kI32, d.q.device, {mr.sms * 4});

  // --- up, relu², down. NON-GATED: `ckpt_names=("up_proj","down_proj","")`
  // (nemotron_h.py:220), so there is no gate half, the fused
  // `kMoeGroupedGemmBf16GateUpSilu` does not apply, and the activation is
  // `vt::MoeRelu2` — which exists FOR this architecture (ops.h:1728-1738).
  DBuf dup(d, DType::kBF16, {P, I});
  {
    Tensor wq = MakeTensor(mr.w_up, DType::kI32, d.q.device, {E, H / 16, I * 2});
    Tensor sc = MakeTensor(mr.s_up, DType::kI8, d.q.device, {E, H / 16, I});
    Tensor gg = MakeTensor(mr.g_up, DType::kF32, d.q.device, {E});
    vt::MoeGroupedGemmNvfp4Marlin(d.q, dup.t(), dh, wq, sc, gg, wst, sorted.t(), expert_ids.t(),
                                  npad.t(), topk_w.t(),
                                  vt::MoeMarlinArgs{block, tki, Ti, Ii, Hi, false});
  }
  DBuf dact(d, DType::kBF16, {P, I});
  vt::MoeRelu2(d.q, dact.t(), dup.t());
  DBuf ddown(d, DType::kBF16, {P, H});
  {
    Tensor wq = MakeTensor(mr.w_down, DType::kI32, d.q.device, {E, I / 16, H * 2});
    Tensor sc = MakeTensor(mr.s_down, DType::kI8, d.q.device, {E, I / 16, H});
    Tensor gg = MakeTensor(mr.g_down, DType::kF32, d.q.device, {E});
    vt::MoeGroupedGemmNvfp4Marlin(d.q, ddown.t(), dact.t(), wq, sc, gg, wst, sorted.t(),
                                  expert_ids.t(), npad.t(), topk_w.t(),
                                  vt::MoeMarlinArgs{block, 1, Pi, Hi, Ii, false});
  }

  // --- shared expert (nemotron_h.py:176-190): the SAME non-gated shape, at
  // moe_shared_expert_intermediate_size * n_shared_experts.
  DBuf shared_out(d, DType::kBF16, {T, H});
  bool have_shared = false;
  if (w.has_shared) {
    VT_CHECK(params.n_shared_experts > 0,
             "NemotronH device moe: shared expert weights present but n_shared_experts is 0");
    const int64_t Is = params.moe_shared_expert_intermediate_size * params.n_shared_experts;
    DBuf su = DenseMarlinE1(d, dh, mr.sw_up, mr.ss_up, mr.sg_up, mr.workspace, mr.sms, T, H, Is);
    DBuf sa(d, DType::kBF16, {T, Is});
    vt::MoeRelu2(d.q, sa.t(), su.t());
    shared_out = DenseMarlinE1(d, sa.t(), mr.sw_down, mr.ss_down, mr.sg_down, mr.workspace, mr.sms,
                               T, Is, H);
    have_shared = true;
  }

  DBuf out(d, DType::kBF16, {T, H});
  {
    Tensor eo = Reshape(ddown.t(), {T, Kk, H});
    Tensor st = shared_out.t();
    vt::MoeCombine(d.q, out.t(), eo, topk_w.t(), have_shared ? &st : nullptr,
                   static_cast<float>(params.routed_scaling_factor));
  }
  return out;
}

#endif  // VT_MARLIN_NVFP4

}  // namespace

// ─── the per-block equivalence seam ─────────────────────────────────────────

std::vector<float> NemotronHAttnBlockHostIO(const NemotronHAttentionWeights& w,
                                            const NemotronHParams& params,
                                            const std::vector<float>& hidden_normed,
                                            int64_t num_tokens, DType act_dtype,
                                            Queue& dev_queue) {
  const int64_t T = num_tokens;
  const int64_t H = params.hidden_size;
  VT_CHECK(T > 0, "NemotronH device attention: empty token sequence");
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NemotronH device attention: hidden size mismatch");
  VT_CHECK(act_dtype == DType::kBF16 || act_dtype == DType::kF32,
           "NemotronH device attention: the model dtype must be bf16 or f32");
  VT_CHECK(dev_queue.device.type != vt::DeviceType::kCPU,
           "NemotronH device attention: this is the DEVICE arm and requires a "
           "non-CPU queue; the host reference is NemotronHAttentionMixer");

  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};
  // Round the input through `act_dtype` on the way in, exactly as the host arm
  // does with `PackF32` (nemotron_h.cpp:610). Feeding the device f32 values the
  // host arm would have rounded first is the kind of "more precise" deviation
  // that makes an equivalence gate quietly meaningless.
  DBuf x = UploadAs(d, hidden_normed, act_dtype, {T, H});
  DBuf out = NemotronHAttnBlock(d, w, params, x.t(), T, act_dtype);
  return DownloadF32(d, out, act_dtype, T * H);
}

std::vector<float> NemotronHMoeBlockDeviceHostIO(const NemotronHMoeWeights& w,
                                                 const NemotronHParams& params,
                                                 const std::vector<float>& hidden_normed,
                                                 int64_t num_tokens, DType act_dtype,
                                                 Queue& dev_queue) {
  const int64_t T = num_tokens;
  const int64_t H = params.hidden_size;
  VT_CHECK(T > 0, "NemotronH device moe: empty token sequence");
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NemotronH device moe: hidden size mismatch");
  VT_CHECK(dev_queue.device.type != vt::DeviceType::kCPU,
           "NemotronH device moe: this is the DEVICE arm and requires a non-CPU "
           "queue; the host reference is NemotronHMoeMixer");
// DSR-ALLOW(A2-Q2a): TYPES, not behaviour -- this arm calls NemotronHMoeBlockDevice, which does not EXIST without the guarded region. It carries an #else that refuses by name, so a build without Marlin reports the missing arm rather than silently computing on the host.
#ifdef VT_MARLIN_NVFP4
  VT_CHECK(act_dtype == DType::kBF16,
           "NemotronH device moe: the NVFP4 Marlin arm requires act_dtype=bf16 "
           "(ops.cpp:879), which is the released checkpoint's model dtype");
  VT_CHECK(MoeIsNvfp4(w),
           "NemotronH device moe: this layer's experts are not NVFP4 W4A16 g16, "
           "which is the only form A2-Q2a's arena is built from");
  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};
  // Round the input through act_dtype on the way in, exactly as the host arm
  // does with `PackF32` (nemotron_h.cpp:759). Feeding the device f32 values the
  // host arm would have rounded first is the deviation that makes an
  // equivalence gate quietly meaningless.
  DBuf x = UploadAs(d, hidden_normed, act_dtype, {T, H});
  DBuf out = NemotronHMoeBlockDevice(d, w, params, x.t(), T);
  return DownloadF32(d, out, act_dtype, T * H);
#else
  (void)w;
  (void)params;
  (void)act_dtype;
  VT_CHECK(false,
           "NemotronH device moe: this build has no Marlin NVFP4 grouped GEMM "
           "(VT_MARLIN_NVFP4 is off), so the device MoE arm is not compiled in");
  return {};
#endif
}

// ─── the hybrid forward ─────────────────────────────────────────────────────

std::vector<float> NemotronHDeviceForward(const NemotronHHostWeights& host,
                                          const NemotronHParams& params,
                                          const std::vector<int32_t>& token_ids,
                                          const std::vector<int32_t>& logits_indices,
                                          Queue& dev_queue, Queue& host_queue,
                                          NemotronHTrace* trace) {
  VT_CHECK(host.materialized,
           "NemotronH device forward: host weights are not materialized");
  VT_CHECK(dev_queue.device.type != vt::DeviceType::kCPU,
           "NemotronH device forward: `dev_queue` must be a non-CPU queue");
  VT_CHECK(host_queue.device.type == vt::DeviceType::kCPU,
           "NemotronH device forward: `host_queue` must be a CPU queue — it is "
           "what the 23 Mamba2 blocks, the 23 MoE blocks and lm_head run on");

  const DType adt = host.act_dtype;
  VT_CHECK(adt == DType::kBF16 || adt == DType::kF32,
           "NemotronH device forward: the model dtype must be bf16 or f32");
  const int64_t H = params.hidden_size;
  const int64_t V = params.vocab_size;
  const int64_t L = params.num_hidden_layers();
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "NemotronH device forward: empty token sequence");
  VT_CHECK(static_cast<int64_t>(host.layers.size()) == L,
           "NemotronH device forward: host layer count != layers_block_type length");

  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};

  // --- the embedding lookup, on the device, off the resident table.
  RequireDeviceWeight(host.embeddings, "backbone.embeddings.weight", adt, {V, H});
  RequireDeviceWeight(host.norm_f, "backbone.norm_f.weight", adt, {H});
  DBuf residual(d, adt, {T, H});
  {
    std::vector<int32_t> ids = token_ids;
    for (int32_t id : ids) {
      VT_CHECK(id >= 0 && id < V, "NemotronH device forward: token id out of range");
    }
    DBuf it(d, DType::kI32, {T}, ids.data());
    d.b.Synchronize(d.q);  // `ids` is a local; see UploadAs for why this waits.
    Tensor tab = ResidentWeight(d, host.embeddings);
    vt::Embedding(d.q, residual.t(), tab, it.t());
  }

  vt::RmsNormArgs nargs;
  nargs.eps = static_cast<float>(params.layer_norm_epsilon);
  nargs.gemma = false;

  if (trace != nullptr && trace->capture) {
    trace->normed.assign(static_cast<size_t>(L), {});
    trace->mixer.assign(static_cast<size_t>(L), {});
    trace->hidden.assign(static_cast<size_t>(L), {});
  }

  // The single-branch pre-norm stream (nemotron_h.py:625-640). Layer 0 sees
  // `residual is None`, so the embedding IS the residual and the norm is
  // un-fused; every later layer folds the previous mixer output into the
  // residual inside the norm. Identical control flow to the host arm
  // (nemotron_h.cpp:872-...), which is what makes the two comparable.
  DBuf carry(d, adt, {T, H});
  for (int64_t l = 0; l < L; ++l) {
    const NemotronHLayerWeights& lw = host.layers[static_cast<size_t>(l)];
    VT_CHECK(lw.block == params.layers_block_type[static_cast<size_t>(l)],
             "NemotronH device forward: host layer block kind disagrees with "
             "layers_block_type");
    RequireDeviceWeight(lw.norm, "layer norm", adt, {H});

    DBuf normed(d, adt, {T, H});
    {
      Tensor wt = ResidentWeight(d, lw.norm);
      if (l == 0) {
        // `residual is None` (nemotron_h.py:627-631): the embedding IS the
        // residual, so there is no add to fuse and this is not a fusion site.
        vt::RmsNorm(d.q, normed.t(), residual.t(), wt, nargs, nullptr);
      } else {
        Tensor rt = residual.t();
        Tensor xt = carry.t();
        Tensor ot = normed.t();
        AddRmsNorm(d, ot, xt, wt, rt, nargs, params.layer_norm_epsilon);
      }
    }

    // The 6 attention layers stay on the device end to end. The other 46
    // bounce: download the normed hidden, run the HOST mixer on `host_queue`,
    // upload the result. One helper, one place, so the scaffold is visible and
    // deletable rather than scattered through the loop.
    // A2-Q2a: a MoE layer whose experts are NVFP4 runs on the DEVICE now, so it
    // needs no host bounce. A `kDense` MoE layer (the synthetic `BuildTiny`
    // fixture, and any future unquantized NemotronH) still bounces, because the
    // arena is built from the NVFP4 form alone — the fallback is stated here
    // rather than discovered as a silent slow path.
    // NO `#ifdef` HERE. This site only SELECTS a path, and every term it reads
    // is available in every build: `MoeIsNvfp4` names only NemotronHWeightForm,
    // and `vt::OpRegistered` IS the op/provider table's own answer to "is the
    // Marlin arm realized for this device". Asking the table rather than the
    // preprocessor is what `check-device-leakage.py` asks for, and it is
    // correct by construction here -- a build without VT_MARLIN_NVFP4 does not
    // register kMoeGroupedGemmNvfp4Marlin, so this resolves false on exactly the
    // builds the guard used to exclude.
    const bool moe_on_device =
        lw.block == NemotronHBlock::kMoe && adt == DType::kBF16 && MoeIsNvfp4(lw.moe) &&
        vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type);
    std::vector<float> nvec;
    const bool needs_host =
        lw.block != NemotronHBlock::kAttention && !moe_on_device;
    if (needs_host || (trace != nullptr && trace->capture)) {
      nvec = DownloadF32(d, normed, adt, T * H);
    }

    // Assigned STRAIGHT INTO `carry`, which is the only thing that reads the
    // mixer output. A `DBuf mixer_out(d, adt, {T, H})` declared here and
    // move-assigned over on both branches was one dead device allocation per
    // layer per step; the previous `carry` block still returns to the pool at
    // exactly the same statement it did before, after every enqueue for this
    // layer, so the lifetimes are unchanged. No speed claim is made or implied.
    std::vector<float> mvec;
    if (lw.block == NemotronHBlock::kAttention) {
      carry = NemotronHAttnBlock(d, lw.attn, params, normed.t(), T, adt);
      if (trace != nullptr && trace->capture) {
        mvec = DownloadF32(d, carry, adt, T * H);
      }
// DSR-ALLOW(A2-Q2a): TYPES, not behaviour -- same call, same reason: the symbol is absent without the guarded region. The SELECTION is already a runtime op-table query (moe_on_device above); only the call site needs the build guard. Mirrors laguna.cpp:1187, a dispatch branch with an else fallback.
#ifdef VT_MARLIN_NVFP4
    } else if (moe_on_device) {
      carry = NemotronHMoeBlockDevice(d, lw.moe, params, normed.t(), T);
      if (trace != nullptr && trace->capture) {
        mvec = DownloadF32(d, carry, adt, T * H);
      }
#endif
    } else {
      switch (lw.block) {
        case NemotronHBlock::kMamba:
          mvec = NemotronHMamba2Mixer(lw.mamba, params, nvec, T, adt, host_queue);
          break;
        case NemotronHBlock::kMoe:
          mvec = NemotronHMoeMixer(lw.moe, params, nvec, T, adt, host_queue);
          break;
        case NemotronHBlock::kMlp:
          mvec = NemotronHMlpMixer(lw.mlp, params, nvec, T, adt, host_queue);
          break;
        case NemotronHBlock::kAttention:
          break;  // handled above
      }
      carry = UploadAs(d, mvec, adt, {T, H});
    }

    if (trace != nullptr && trace->capture) {
      trace->normed[static_cast<size_t>(l)] = std::move(nvec);
      // The residual AFTER this layer is what the next norm folds `carry` into.
      std::vector<float> h = DownloadF32(d, residual, adt, T * H);
      for (size_t i = 0; i < h.size(); ++i) h[i] += mvec[i];
      trace->mixer[static_cast<size_t>(l)] = std::move(mvec);
      trace->hidden[static_cast<size_t>(l)] = std::move(h);
    }
  }

  // `hidden_states, _ = self.norm_f(hidden_states, residual)` (nemotron_h.py:641).
  DBuf final_normed(d, adt, {T, H});
  {
    Tensor wt = ResidentWeight(d, host.norm_f);
    Tensor rt = residual.t();
    Tensor xt = carry.t();
    Tensor ot = final_normed.t();
    AddRmsNorm(d, ot, xt, wt, rt, nargs, params.layer_norm_epsilon);
  }
  const std::vector<float> fvec = DownloadF32(d, final_normed, adt, T * H);
  if (trace != nullptr && trace->capture) trace->final_normed = fvec;

  // --- lm_head, on the HOST: it is NVFP4 W4A16 g16 on the released
  // checkpoint. Both arms therefore end in the IDENTICAL host projection, which
  // is what makes A2-R's token gate attributable: a token difference can only
  // have come from the 6 device attention blocks and the device residual
  // stream, never from the output projection.
  std::vector<int64_t> want;
  if (logits_indices.empty()) {
    want.resize(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) want[static_cast<size_t>(i)] = i;
  } else {
    for (int32_t idx : logits_indices) {
      VT_CHECK(idx >= 0 && idx < T,
               "NemotronH device forward: logits index out of range");
      want.push_back(idx);
    }
  }
  std::vector<float> gathered(want.size() * static_cast<size_t>(H));
  for (size_t r = 0; r < want.size(); ++r) {
    std::memcpy(gathered.data() + r * static_cast<size_t>(H),
                fvec.data() + static_cast<size_t>(want[r]) * static_cast<size_t>(H),
                static_cast<size_t>(H) * sizeof(float));
  }
  return NemotronHHostLmHead(host, params, gathered,
                             static_cast<int64_t>(want.size()), host_queue);
}

}  // namespace vllm
