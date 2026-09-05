// GLM-5.3-Flash — W5: the 288 routed + 1 shared expert MoE block.
// Contract, port anchors and the five silent-failure notes: glm5_next_moe.h.
#include "vllm/model_executor/models/glm5_next_moe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_moe.h"  // deepseek_v4::ClampedSwiGLU
#include "vllm/model_executor/models/dense_attn_block.h"  // dense_attn::ResidentWeight
#include "vllm/model_executor/models/dense_device_glue.h"  // dense_attn::Dev, DBuf
#include "vllm/platforms/interface.h"                    // platforms::GetPlatform
#include "vt/backend.h"                                  // vt::Backend
#include "vt/dtype.h"                                    // VT_CHECK

namespace vllm::glm5_next {
namespace {

// `[out, in] x [num_tokens, in] -> [num_tokens, out]`, the torch `nn.Linear`
// layout. The accumulator is `double` for the same reason every other host
// reference in this model uses one: it removes reduction ORDER from the
// comparison against the oracle, so a residual difference is the formula and
// not the summation.
std::vector<float> MatVecRows(const std::vector<float>& w, const std::vector<float>& x,
                              int64_t out_dim, int64_t in_dim, int64_t num_tokens,
                              const char* what) {
  VT_CHECK(static_cast<int64_t>(w.size()) == out_dim * in_dim,
           std::string("glm5_next moe: ") + what + " expects [" +
               std::to_string(out_dim) + ", " + std::to_string(in_dim) +
               "] = " + std::to_string(out_dim * in_dim) + " floats, got " +
               std::to_string(w.size()));
  VT_CHECK(static_cast<int64_t>(x.size()) == num_tokens * in_dim,
           std::string("glm5_next moe: ") + what + " input expects [" +
               std::to_string(num_tokens) + ", " + std::to_string(in_dim) +
               "] floats, got " + std::to_string(x.size()));
  std::vector<float> y(static_cast<size_t>(num_tokens * out_dim));
  for (int64_t t = 0; t < num_tokens; ++t) {
    const float* xt = &x[static_cast<size_t>(t * in_dim)];
    for (int64_t o = 0; o < out_dim; ++o) {
      const float* wo = &w[static_cast<size_t>(o * in_dim)];
      double acc = 0.0;
      for (int64_t i = 0; i < in_dim; ++i) acc += static_cast<double>(wo[i]) * xt[i];
      y[static_cast<size_t>(t * out_dim + o)] = static_cast<float>(acc);
    }
  }
  return y;
}

// A host-memory `vt::Tensor` view, as in `glm5_next_kda.cpp`. Row-major strides.
vt::Tensor MakeT(void* data, vt::DType dt, vt::Device dev,
                 const std::vector<int64_t>& shape) {
  vt::Tensor t;
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


// ─── the routed-expert GEMM on the shared keep-quant seam (W9a) ──────────────
//
// Replaces the per-expert host-f32 matvec pair with two grouped ops. The
// argument for it, and what it does and does not change numerically, is the
// `MoeQuantBanks` block in glm5_next_moe.h.
//
// P IS THE NUMBER OF ROUTING SLOTS, `num_tokens * num_experts_per_tok`, and
// every slot is one row of the grouped call. `expert_ids[p]` is `topk_ids[p]`
// and the activation row is `hidden[p / K]` — the token that slot belongs to.
//
// THE DIVISION IS THE ONE ERROR HERE THAT IS FLUENT RATHER THAN A CRASH.
// `p % K` is in range for every p and indexes a real token, so a port that
// writes it evaluates real experts at real router weights over the WRONG
// tokens, and the model keeps generating. Nothing about the shapes, the dtypes
// or the finiteness of the result can see it; only a fixture with more than one
// token and more than one distinct hidden row can.
//
// There is no per-expert grouping here, and that is not an omission. "Grouped"
// is what these ops are: they take `expert_ids` and index the stacked tower per
// row themselves. The hand-rolled `hit`/`slots` walk in the f32 arms exists
// only because those arms hold one weight pointer at a time.
// THE FIT GUARD's margin. A shared fleet box carries the engine's KV pool, the
// CUDA context and whatever else the process already staged, and `cudaMemGetInfo`
// answers about the whole device rather than about us. Staging a 2.25 GiB bank
// that leaves the allocator with nothing does not fail here -- it fails later, in
// somebody else's allocation, on a lease that cannot be retaken cheaply. So the
// guard keeps a floor free and falls back to the host arm by name instead.
//
// 2 GiB, and the number is a judgement rather than a measurement: it is roughly
// one sparse layer's three banks at this model's published geometry, so a device
// that cannot hold one more layer stops staging one layer early.
constexpr size_t kGlm5NextDeviceStagingFloorBytes = size_t{2} << 30;

// Whether this layer's three banks can be staged. Already-resident banks cost
// nothing and are always admitted; a bank that still needs uploading is charged
// against the device's free memory.
//
// It answers per LAYER and not per model on purpose. The tower is 94.6758 GiB
// across 42 sparse layers and is staged lazily as the first step walks them, so
// a model that fits gets every layer and a model that does not gets a prefix of
// them plus the host arm for the rest -- which is a degraded but correct run,
// and is what an OOM is not.
bool DeviceBanksFit(const MoeQuantBanks& b, const dense_attn::Dev& d) {
  // A CPU "device" allocates nothing. `ResidentWeight`'s host-alias arm
  // (`dense_attn_block.h:196`) hands back a view over the tensor's own bytes,
  // so there is no upload to charge and no free-memory question to ask. This is
  // also what makes the device ARM runnable on a CPU-only host, which is where
  // its operand construction is gated: without this clause the guard would read
  // `DeviceMemoryInfo`'s `false` default and route every such run to the host
  // arm, and the gate would measure the arm it was not testing.
  if (vllm::platforms::GetPlatform(d.q.device.type).is_cpu()) return true;
  size_t need = 0;
  const OwnedTensor* srcs[3] = {b.gate_src, b.up_src, b.down_src};
  for (const OwnedTensor* t : srcs) {
    if (t->d_dev) continue;  // already resident: this call uploads nothing
    need += t->bytes.size();
  }
  if (need == 0) return true;
  size_t free_b = 0;
  size_t total_b = 0;
  // A backend that cannot answer is NOT a backend with room. `Backend::
  // DeviceMemoryInfo` returns false by default (`vt/backend.h:104`) and CUDA
  // overrides it with `cudaMemGetInfo` (`cuda_backend.cu:93`), so a false here
  // means an unmeasured device, and staging 94.7 GiB against an unmeasured
  // device is exactly the bet this guard exists to decline.
  if (!d.b.DeviceMemoryInfo(&free_b, &total_b)) return false;
  return free_b >= need + kGlm5NextDeviceStagingFloorBytes;
}

// The POSITIVE counterpart of the fallback warning below, and it is here for
// reachability rather than for telemetry.
//
// AGENTS.md "Nothing lands dead" asks what reaches a capability from a
// production entry point. On a device this cannot be answered by a value: the
// device arm and the host arm compute the same block, and on a CPU-backed `Dev`
// they agree bit-for-bit, which is exactly what makes them safe to swap and
// exactly what makes a numeric gate blind to WHICH one ran. One line on stderr,
// naming the device, is the observable that separates them -- so the end-to-end
// `--device cuda` job greps for it, and a mutation that stops threading `dev`
// through the layer removes it and produces the fallback line instead.
void AnnounceDeviceArmOnce(const vt::Device& dev) {
  static bool said = false;
  if (said) return;
  said = true;
  std::fprintf(stderr,
               "[glm5-next] the routed-expert keep-quant GEMM is running on "
               "DEVICE type %d index %d; every other primitive of this model is "
               "on the host (spec section W9c-3a, O43, issue #2464).\n",
               static_cast<int>(dev.type), static_cast<int>(dev.index));
}

// Said ONCE per process, not once per layer per step. 42 sparse layers x every
// decode step is a log that hides the run it is describing.
void WarnDeviceFallbackOnce(const char* why) {
  static bool said = false;
  if (said) return;
  said = true;
  std::fprintf(stderr,
               "[glm5-next] the routed-expert GEMM is running on the HOST arm: %s. "
               "The run is correct and slower; see .agents/specs/glm5-next-flash.md "
               "section W9c-3a and issue #2464.\n",
               why);
}

void MoeExpertsKeepQuant(const MoeDims& d, const MoeQuantBanks& b,
                         const std::vector<float>& hidden, const MoeRouting& r,
                         int64_t num_tokens, vt::Queue& queue,
                         dense_attn::Dev* dev,
                         std::vector<float>* expert_out) {
  const int64_t H = d.hidden_size;
  const int64_t I = d.moe_intermediate_size;
  const int64_t K = d.num_experts_per_tok;
  const int64_t P = num_tokens * K;
  VT_CHECK(static_cast<int64_t>(r.topk_ids.size()) == P,
           "glm5_next moe: the keep-quant arm expects one expert id per routing "
           "slot, [" + std::to_string(num_tokens) + ", " + std::to_string(K) +
               "] = " + std::to_string(P) + ", got " +
               std::to_string(r.topk_ids.size()));
  VT_CHECK(static_cast<int64_t>(expert_out->size()) == P * H,
           "glm5_next moe: the keep-quant arm writes the combine's [" +
               std::to_string(num_tokens) + ", " + std::to_string(K) + ", " +
               std::to_string(H) + "] operand, " + std::to_string(P * H) +
               " floats, got " + std::to_string(expert_out->size()));

  // `act[p] = hidden[p / K]`. The seam has a BROADCAST arm (activation rows ==
  // 1) and it is not usable here: it repeats ONE hidden row across every slot,
  // which is only correct when `num_tokens == 1`. Taking it would make every
  // decode step right and every prefill wrong, which is the shape of bug a
  // one-token fixture cannot see, so the gather is unconditional.
  std::vector<float> act(static_cast<size_t>(P * H));
  for (int64_t p = 0; p < P; ++p) {
    const int64_t t = p / K;
    std::copy(hidden.begin() + static_cast<std::ptrdiff_t>(t * H),
              hidden.begin() + static_cast<std::ptrdiff_t>((t + 1) * H),
              act.begin() + static_cast<std::ptrdiff_t>(p * H));
  }

  const int64_t E = d.n_routed_experts;

  // --- W9c-3a: THE DEVICE ARM -----------------------------------------------
  //
  // The ONE arm of this model that computes on a GPU. Everything above and below
  // it -- the router that produced `r`, the combine that consumes `expert_out`,
  // and every other primitive in this forward -- runs on the host, and the row's
  // spec records that as O43 rather than leaving it to be inferred from a
  // `--device cuda` that returns a token.
  //
  // Nothing is retagged. The three banks become DEVICE tensors by being
  // uploaded (`ResidentWeight`), and the four activations are real device
  // allocations (`DBuf`), so the ops' `a.device == b.device == out.device ==
  // q.device` check passes because it is TRUE and not because it was made to
  // look true.
  if (dev != nullptr && b.HasSources() && DeviceBanksFit(b, *dev)) {
    AnnounceDeviceArmOnce(dev->q.device);
    // `ResidentWeight` uploads `bytes` verbatim and keeps the block dtype, so
    // these stay Q2_K / IQ2_XS / IQ3_XXS / IQ4_XS on the device. The shapes are
    // the ones the ops declare; `OwnedTensor` records `[E, N, K]` and the tower
    // the seam wants is `[E * N, K]`, which is a rank change over the same
    // bytes. It also drops the three load-time repack markers, which is safe
    // here ONLY because `AdmitMoeQuantBanks` refuses a bank carrying any of
    // them (`glm5_next_bridge.cpp`).
    const vt::Tensor g_dev = dense_attn::ResidentWeight(*dev, *b.gate_src, {E * I, H});
    const vt::Tensor u_dev = dense_attn::ResidentWeight(*dev, *b.up_src, {E * I, H});
    const vt::Tensor d_dev = dense_attn::ResidentWeight(*dev, *b.down_src, {E * H, I});
    dense_attn::DBuf d_act(*dev, vt::DType::kF32, {P, H}, act.data());
    dense_attn::DBuf d_mid(*dev, vt::DType::kF32, {P, I});
    dense_attn::DBuf d_ids(*dev, vt::DType::kI32, {P}, r.topk_ids.data());
    dense_attn::DBuf d_out(*dev, vt::DType::kF32, {P, H});
    vt::MoeGateUpSwiGLUGrouped(dev->q, d_mid.t(), d_act.t(), g_dev, u_dev, d_ids.t(),
                               d.swiglu_limit);
    vt::MatmulBTQuantGrouped(dev->q, d_out.t(), d_mid.t(), d_dev, d_ids.t());
    // `Download` synchronises, so the combine below reads a completed buffer
    // rather than a stream the caller would have to remember to drain -- and
    // `act` and `r.topk_ids`, whose bytes the two uploads above copy
    // asynchronously on this queue, are still alive at that point.
    d_out.Download(*dev, expert_out->data());
    return;
  }
  if (dev != nullptr) {
    WarnDeviceFallbackOnce(b.HasSources()
                               ? "the device has no room to stage this layer's "
                                 "expert banks (see DeviceBanksFit)"
                               : "this layer's keep-quant banks carry no source "
                                 "tensors, so nothing can be made resident");
  }

  std::vector<float> mid(static_cast<size_t>(P * I));
  const vt::Device hdev = queue.device;
  // HOST pointers on a HOST queue. This arm is unchanged and is the operand the
  // device arm above is gated against.
  VT_CHECK(hdev.type == vt::DeviceType::kCPU,
           "glm5_next moe: the keep-quant HOST arm was handed a non-CPU queue. "
           "Its four operands are `std::vector` buffers, and the grouped ops "
           "require every operand on the queue's device, so this would be a "
           "refusal or a crash and never a fallback. The device arm is the "
           "branch above; see .agents/specs/glm5-next-flash.md section W9c-3a.");
  vt::Tensor t_act = MakeT(act.data(), vt::DType::kF32, hdev, {P, H});
  vt::Tensor t_mid = MakeT(mid.data(), vt::DType::kF32, hdev, {P, I});
  vt::Tensor t_ids = MakeT(const_cast<int32_t*>(r.topk_ids.data()),
                           vt::DType::kI32, hdev, {P});

  // The bank views are passed THROUGH, with their own `device` untouched. Both
  // ops require every operand to be on the queue's device and refuse otherwise,
  // and that refusal is the guard that keeps a host-resident bank out of a CUDA
  // kernel. Overwriting `.device` here to make the check pass would hand a
  // device kernel host pointers, which is what `minimax_h3_device.cpp:339-342`
  // records as reading "ALL ZEROS on the GPU". The device arm above does not
  // retag anything: it UPLOADS, and gets a device tensor because the bytes are
  // on the device.
  vt::MoeGateUpSwiGLUGrouped(queue, t_mid, t_act, b.gate, b.up, t_ids,
                             d.swiglu_limit);

  // `expert_out` is the combine's `[T, K, H]` and `P * H == T * K * H`, so slot
  // p IS row p of a `[P, H]` view. The reshape is the identity on the bytes.
  vt::Tensor t_eo = MakeT(expert_out->data(), vt::DType::kF32, hdev, {P, H});
  vt::MatmulBTQuantGrouped(queue, t_eo, t_mid, b.down, t_ids);
}

}  // namespace

// ── the resolved geometry ────────────────────────────────────────────────────

void MoeDims::Validate() const {
  VT_CHECK(hidden_size > 0 && n_routed_experts > 0 && num_experts_per_tok > 0 &&
               moe_intermediate_size > 0 && n_group > 0 && topk_group > 0,
           "glm5_next moe: every routing dimension must be positive; got "
           "hidden_size=" +
               std::to_string(hidden_size) + " n_routed_experts=" +
               std::to_string(n_routed_experts) + " num_experts_per_tok=" +
               std::to_string(num_experts_per_tok) + " moe_intermediate_size=" +
               std::to_string(moe_intermediate_size) + " n_group=" +
               std::to_string(n_group) + " topk_group=" + std::to_string(topk_group));
  // `n_shared_experts` may legitimately be 0 (a checkpoint with no shared
  // expert), but a NEGATIVE count would make `shared_intermediate_size()`
  // negative and the shared MLP would be sized from it.
  VT_CHECK(n_shared_experts >= 0,
           "glm5_next moe: n_shared_experts must be >= 0, got " +
               std::to_string(n_shared_experts));
  VT_CHECK(num_experts_per_tok <= n_routed_experts,
           "glm5_next moe: num_experts_per_tok " +
               std::to_string(num_experts_per_tok) +
               " exceeds n_routed_experts " + std::to_string(n_routed_experts));
  // `vt::MoeRouterTopK`'s own admission rule, checked here so the refusal names
  // this model rather than the op.
  VT_CHECK(n_routed_experts % n_group == 0,
           "glm5_next moe: n_group " + std::to_string(n_group) +
               " does not divide n_routed_experts " +
               std::to_string(n_routed_experts));
  VT_CHECK(topk_group <= n_group,
           "glm5_next moe: topk_group " + std::to_string(topk_group) +
               " exceeds n_group " + std::to_string(n_group));
  // A non-positive clamp limit is not "no clamp": `clamp(max=0)` zeroes every
  // positive gate, so the layer would emit zeros through silu. Upstream's
  // default is 10.0 and the published checkpoint's per-layer
  // `swiglu_clamp_exp` / `swiglu_clamp_shexp` are both 10.0.
  VT_CHECK(swiglu_limit > 0.0f,
           "glm5_next moe: swiglu_limit must be positive, got " +
               std::to_string(swiglu_limit));
}

MoeDims MoeDimsFrom(const Glm5NextParams& p) {
  MoeDims d;
  d.hidden_size = p.hidden_size;
  d.n_routed_experts = p.moe.n_routed_experts;
  d.n_shared_experts = p.moe.n_shared_experts;
  d.num_experts_per_tok = p.moe.num_experts_per_tok;
  d.moe_intermediate_size = p.moe.moe_intermediate_size;
  d.n_group = p.moe.n_group;
  d.topk_group = p.moe.topk_group;
  d.routed_scaling_factor = p.moe.routed_scaling_factor;
  d.norm_topk_prob = p.moe.norm_topk_prob;
  d.swiglu_limit = static_cast<float>(p.swiglu_limit);
  d.Validate();
  return d;
}

// ── the router ───────────────────────────────────────────────────────────────

std::vector<float> RouterLogits(const MoeDims& d, const std::vector<float>& hidden,
                                const std::vector<float>& router_weight,
                                int64_t num_tokens) {
  d.Validate();
  // fp32 on BOTH operands, unconditionally (`:160`). This host reference is
  // float-in/float-out, so the upcast is already satisfied; what the annotation
  // records is that a DEVICE arm of this GEMM may not inherit the model dtype
  // here the way every other projection in this model does.
  return MatVecRows(router_weight, hidden, d.n_routed_experts, d.hidden_size,
                    num_tokens, "router_weight");
}

MoeRouting RouteTopk(const MoeDims& d, const MoeLayerWeights& w,
                     const std::vector<float>& hidden, int64_t num_tokens,
                     vt::Queue& queue) {
  d.Validate();
  VT_CHECK(num_tokens > 0,
           "glm5_next moe: RouteTopk needs at least one token, got " +
               std::to_string(num_tokens));
  // `vt::MoeRouterTopK` dispatches on the queue's device; handing it host
  // pointers on a CUDA queue is a crash, not a fallback. The device arm of this
  // block belongs to the assembled text forward.
  VT_CHECK(queue.device.type == vt::DeviceType::kCPU,
           "glm5_next moe: RouteTopk is the HOST reference and needs a CPU "
           "queue; the device arm is the assembled text forward's "
           "(.agents/specs/glm5-next-flash.md)");
  const bool has_bias = !w.e_score_correction_bias.empty();
  VT_CHECK(!has_bias || static_cast<int64_t>(w.e_score_correction_bias.size()) ==
                            d.n_routed_experts,
           "glm5_next moe: e_score_correction_bias has " +
               std::to_string(w.e_score_correction_bias.size()) +
               " entries, expected " + std::to_string(d.n_routed_experts) +
               " (one per routed expert)");

  MoeRouting r;
  r.router_logits = RouterLogits(d, hidden, w.router_weight, num_tokens);
  r.topk_ids.assign(static_cast<size_t>(num_tokens * d.num_experts_per_tok), 0);
  r.topk_weights.assign(static_cast<size_t>(num_tokens * d.num_experts_per_tok), 0.0f);

  vt::MoeRouterTopKArgs args{};
  args.top_k = static_cast<int>(d.num_experts_per_tok);
  args.renormalize = d.norm_topk_prob;
  // SIGMOID, not softmax (`:161`). The scores are elementwise and are NOT
  // normalised across experts, which is why the renormalize step exists at all.
  args.scoring_func = vt::MoeScoringFunc::kSigmoid;
  args.num_expert_group = static_cast<int>(d.n_group);
  args.topk_group = static_cast<int>(d.topk_group);
  // Folded into the ROUTER WEIGHTS (`:182`), which is why `vt::MoeCombine` below
  // keeps its default `routed_scale = 1.0f`. Passing it in both places squares
  // it, and the result is still smooth and still routes to the same experts.
  args.routed_scaling_factor = static_cast<float>(d.routed_scaling_factor);

  const vt::Device dev = queue.device;
  vt::Tensor t_w = MakeT(r.topk_weights.data(), vt::DType::kF32, dev,
                         {num_tokens, d.num_experts_per_tok});
  vt::Tensor t_i = MakeT(r.topk_ids.data(), vt::DType::kI32, dev,
                         {num_tokens, d.num_experts_per_tok});
  vt::Tensor t_l = MakeT(r.router_logits.data(), vt::DType::kF32, dev,
                         {num_tokens, d.n_routed_experts});
  vt::Tensor t_b;
  if (has_bias) {
    t_b = MakeT(const_cast<float*>(w.e_score_correction_bias.data()), vt::DType::kF32,
                dev, {d.n_routed_experts});
  }
  vt::MoeRouterTopK(queue, t_w, t_i, t_l, args, has_bias ? &t_b : nullptr);
  return r;
}

// ── the clamped SwiGLU epilogue ──────────────────────────────────────────────

std::vector<float> ExpertGate(const std::vector<float>& gate_up, int64_t intermediate,
                              float limit) {
  VT_CHECK(intermediate > 0,
           "glm5_next moe: ExpertGate needs a positive intermediate width, got " +
               std::to_string(intermediate));
  VT_CHECK(static_cast<int64_t>(gate_up.size()) == 2 * intermediate,
           "glm5_next moe: ExpertGate expects a fused [2 * " +
               std::to_string(intermediate) + "] row, got " +
               std::to_string(gate_up.size()) + " floats");
  VT_CHECK(limit > 0.0f,
           "glm5_next moe: ExpertGate needs a positive clamp limit, got " +
               std::to_string(limit));
  // alpha = 1, beta = 0 reduces V4's `gate * sigmoid(alpha*gate) * (up + beta)`
  // to `silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)`, which is
  // `_apply_gate`'s "Simple swiglu instead of alpha" line for line.
  return deepseek_v4::ClampedSwiGLU(gate_up, intermediate, limit, /*alpha=*/1.0f,
                                    /*beta=*/0.0f);
}

// ── the dense feed-forward, and the shared expert ────────────────────────────

std::vector<float> DenseMlpForward(const DenseMlpWeights& w,
                                   const std::vector<float>& hidden,
                                   int64_t hidden_size, int64_t intermediate,
                                   int64_t num_tokens, float limit) {
  VT_CHECK(hidden_size > 0 && intermediate > 0 && num_tokens > 0,
           "glm5_next moe: DenseMlpForward needs positive dims; got hidden_size=" +
               std::to_string(hidden_size) + " intermediate=" +
               std::to_string(intermediate) + " num_tokens=" +
               std::to_string(num_tokens));
  VT_CHECK(limit > 0.0f,
           "glm5_next moe: DenseMlpForward needs a positive clamp limit, got " +
               std::to_string(limit));
  const std::vector<float> gate =
      MatVecRows(w.gate_proj, hidden, intermediate, hidden_size, num_tokens, "gate_proj");
  const std::vector<float> up =
      MatVecRows(w.up_proj, hidden, intermediate, hidden_size, num_tokens, "up_proj");
  // Re-fuse into the [gate | up] row the shared epilogue takes, so the DENSE
  // path and the EXPERT path clamp through exactly one implementation. The
  // checkpoint stores the dense projections separately and the expert ones
  // fused; the activation is the same function of both.
  std::vector<float> act(static_cast<size_t>(num_tokens * intermediate));
  std::vector<float> row(static_cast<size_t>(2 * intermediate));
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t i = 0; i < intermediate; ++i) {
      row[static_cast<size_t>(i)] = gate[static_cast<size_t>(t * intermediate + i)];
      row[static_cast<size_t>(intermediate + i)] =
          up[static_cast<size_t>(t * intermediate + i)];
    }
    const std::vector<float> a = ExpertGate(row, intermediate, limit);
    for (int64_t i = 0; i < intermediate; ++i) {
      act[static_cast<size_t>(t * intermediate + i)] = a[static_cast<size_t>(i)];
    }
  }
  return MatVecRows(w.down_proj, act, hidden_size, intermediate, num_tokens, "down_proj");
}

// ── the composed sparse block ────────────────────────────────────────────────

std::vector<float> MoeForward(const MoeDims& d, const MoeLayerWeights& w,
                              const std::vector<float>& hidden, int64_t num_tokens,
                              vt::Queue& queue, dense_attn::Dev* dev) {
  d.Validate();
  const int64_t H = d.hidden_size;
  const int64_t E = d.n_routed_experts;
  const int64_t I = d.moe_intermediate_size;
  const int64_t K = d.num_experts_per_tok;
  VT_CHECK(static_cast<int64_t>(hidden.size()) == num_tokens * H,
           "glm5_next moe: MoeForward hidden expects [" +
               std::to_string(num_tokens) + ", " + std::to_string(H) +
               "] floats, got " + std::to_string(hidden.size()));
  // EXACTLY ONE of the two expert residencies, refused by name on the other two
  // states. `resident` is true when EITHER bank carries values, so a layer that
  // filled one and forgot the other is a named error and not a half-read bank.
  const bool resident = !w.expert_gate_up.empty() || !w.expert_down.empty();
  // THREE residencies now, not two, and only the two f32 ones exclude each
  // other. The keep-quant banks are the SAME weights the other two carry, in
  // the encoding the file already holds, so a layer that offers them together
  // with an `ExpertSource` is not ambiguous — it is a layer whose bridge built
  // both and the cheaper one wins. What stays ambiguous, and is still refused,
  // is BOTH f32 shapes at once.
  VT_CHECK(!(resident && w.expert_source != nullptr),
           "glm5_next moe: this layer carries BOTH resident expert banks and an "
           "`expert_source`, and there is no rule for which one wins. Populate "
           "the banks or set the source, never both (glm5_next_moe.h).");
  // MEASURED, not asserted defensively: removing this check makes the loop
  // below dereference a NULL `expert_source` and the process dies with SIGSEGV
  // (mutation M5, rc=139). The message says that rather than the softer "would
  // read as zeros", because a reader who lands here needs to know which failure
  // the guard is standing in front of.
  VT_CHECK(resident || w.expert_source != nullptr || w.has_quant_banks,
           "glm5_next moe: this layer carries NEITHER resident expert banks nor "
           "an `expert_source` nor keep-quant banks, so the router would select " +
               std::to_string(K) + " of " + std::to_string(E) +
               " experts and there would be nothing to evaluate them with -- the "
               "loop below would dereference a null source. At the published "
               "geometry the banks are 27.0 GiB per sparse layer in f32 against "
               "a ~119.63 GiB box, so the on-demand `ExpertSource` is the shape "
               "that fits; see glm5_next_moe.h.");
  if (resident) {
    VT_CHECK(static_cast<int64_t>(w.expert_gate_up.size()) == E * 2 * I * H,
             "glm5_next moe: expert_gate_up expects the STACKED, FUSED [" +
                 std::to_string(E) + ", 2 * " + std::to_string(I) + ", " +
                 std::to_string(H) + "] = " + std::to_string(E * 2 * I * H) +
                 " floats, got " + std::to_string(w.expert_gate_up.size()));
    VT_CHECK(static_cast<int64_t>(w.expert_down.size()) == E * H * I,
             "glm5_next moe: expert_down expects the STACKED [" +
                 std::to_string(E) + ", " + std::to_string(H) + ", " +
                 std::to_string(I) + "] = " + std::to_string(E * H * I) +
                 " floats, got " + std::to_string(w.expert_down.size()));
  }

  const MoeRouting r = RouteTopk(d, w, hidden, num_tokens, queue);

  // The per-slot expert MLP outputs, [T, K, H] — what `vt::MoeCombine` reduces.
  // Upstream loops over the HIT experts and `index_add_`s each one's tokens; the
  // two orders differ only in the summation order of the same terms, and the
  // combine below accumulates in f32 exactly as upstream's `final` does.
  std::vector<float> expert_out(static_cast<size_t>(num_tokens * K * H), 0.0f);
  // THE KEEP-QUANT ARM, preferred whenever the bridge could build it. It reads
  // the checkpoint's blocks where they lie and runs the two shared grouped ops;
  // the f32 arms below decode the same weights to host floats first. See
  // `MoeQuantBanks` in glm5_next_moe.h for why this is a binding and not new
  // numerics, and for the one thing it does change (the activation is quantized
  // to Q8_K, so this arm and the f32 arms agree to a band and not exactly).
  //
  // The f32 arms stay reachable BELOW this branch deliberately, the way
  // `nemotron_h_registry.cpp:204-213` keeps its host reference below the device
  // fold: they are the operand the parity gate compares against, and deleting
  // them deletes the gate.
  if (w.has_quant_banks) {
    MoeExpertsKeepQuant(d, w.quant_banks, hidden, r, num_tokens, queue, dev,
                        &expert_out);
  } else {
    std::vector<float> gate_up(static_cast<size_t>(2 * I));

    // Group the `[T, K]` routing slots by EXPERT, then visit each hit expert
    // ONCE. Upstream does the same (`:120-135` iterates the hit experts, not the
    // tokens), and with an `ExpertSource` it is also the residency bound: a
    // second visit would be a second decode of the same 96 MiB. `hit` is sorted
    // so the visit order is a property of the SET and not of the router's
    // emission order.
    std::vector<std::vector<int64_t>> slots(static_cast<size_t>(E));
    std::vector<int64_t> hit;
    for (int64_t t = 0; t < num_tokens; ++t) {
      for (int64_t j = 0; j < K; ++j) {
        const int32_t e = r.topk_ids[static_cast<size_t>(t * K + j)];
        VT_CHECK(e >= 0 && static_cast<int64_t>(e) < E,
                 "glm5_next moe: the router selected expert " + std::to_string(e) +
                     ", which is outside [0, " + std::to_string(E) + ")");
        if (slots[static_cast<size_t>(e)].empty()) hit.push_back(e);
        slots[static_cast<size_t>(e)].push_back(t * K + j);
      }
    }
    std::sort(hit.begin(), hit.end());

    // ONE pair of buffers for the whole call, reused across experts: the peak is
    // one expert's `[2I, H]` plus `[H, I]` and never K of them.
    std::vector<float> src_gate_up;
    std::vector<float> src_down;
    for (int64_t e : hit) {
      const float* gu = nullptr;
      const float* dw = nullptr;
      if (resident) {
        gu = &w.expert_gate_up[static_cast<size_t>(e) * static_cast<size_t>(2 * I * H)];
        dw = &w.expert_down[static_cast<size_t>(e) * static_cast<size_t>(H * I)];
      } else {
        w.expert_source->Expert(e, src_gate_up, src_down);
        VT_CHECK(static_cast<int64_t>(src_gate_up.size()) == 2 * I * H,
                 "glm5_next moe: the expert source returned " +
                     std::to_string(src_gate_up.size()) +
                     " gate_up floats for expert " + std::to_string(e) +
                     ", expected the FUSED [2 * " + std::to_string(I) + ", " +
                     std::to_string(H) + "] = " + std::to_string(2 * I * H));
        VT_CHECK(static_cast<int64_t>(src_down.size()) == H * I,
                 "glm5_next moe: the expert source returned " +
                     std::to_string(src_down.size()) +
                     " down floats for expert " + std::to_string(e) +
                     ", expected [" + std::to_string(H) + ", " +
                     std::to_string(I) + "] = " + std::to_string(H * I));
        gu = src_gate_up.data();
        dw = src_down.data();
      }
      for (int64_t slot : slots[static_cast<size_t>(e)]) {
        const int64_t t = slot / K;
        const float* xt = &hidden[static_cast<size_t>(t * H)];
        for (int64_t o = 0; o < 2 * I; ++o) {
          double acc = 0.0;
          const float* wo = gu + static_cast<size_t>(o * H);
          for (int64_t i = 0; i < H; ++i) acc += static_cast<double>(wo[i]) * xt[i];
          gate_up[static_cast<size_t>(o)] = static_cast<float>(acc);
        }
        const std::vector<float> act = ExpertGate(gate_up, I, d.swiglu_limit);
        float* out = &expert_out[static_cast<size_t>(slot * H)];
        for (int64_t o = 0; o < H; ++o) {
          double acc = 0.0;
          const float* wo = dw + static_cast<size_t>(o * I);
          for (int64_t i = 0; i < I; ++i) acc += static_cast<double>(wo[i]) * act[static_cast<size_t>(i)];
          out[o] = static_cast<float>(acc);
        }
      }
    }

  }

  // The shared expert, at `moe_intermediate_size * n_shared_experts` and NOT at
  // `intermediate_size`. It reads the block's INPUT (`residuals`, `:201`), not
  // the routed output.
  std::vector<float> shared;
  if (d.n_shared_experts > 0) {
    shared = DenseMlpForward(w.shared, hidden, H, d.shared_intermediate_size(),
                             num_tokens, d.swiglu_limit);
  }

  std::vector<float> out(static_cast<size_t>(num_tokens * H), 0.0f);
  // `cdev` and not `dev`: the parameter of that name is the DEVICE ARM, and
  // this is the combine's HOST device. The combine reads `expert_out` and
  // `topk_weights`, both host buffers, whichever arm filled them -- the device
  // arm downloads before it returns -- so the combine stays on the host queue
  // and W9c-3a does not move it. O43 lists it among the ten arms still there.
  const vt::Device cdev = queue.device;
  vt::Tensor t_out = MakeT(out.data(), vt::DType::kF32, cdev, {num_tokens, H});
  vt::Tensor t_eo = MakeT(expert_out.data(), vt::DType::kF32, cdev, {num_tokens, K, H});
  vt::Tensor t_w = MakeT(const_cast<float*>(r.topk_weights.data()), vt::DType::kF32, cdev,
                         {num_tokens, K});
  vt::Tensor t_sh;
  if (!shared.empty()) {
    t_sh = MakeT(shared.data(), vt::DType::kF32, cdev, {num_tokens, H});
  }
  // `routed_scale` stays 1.0f: `routed_scaling_factor` is already in the router
  // weights (`MoeRouterTopKArgs::routed_scaling_factor` above), which is
  // upstream's polarity — the SHARED term is added unscaled (`:206`).
  vt::MoeCombine(queue, t_out, t_eo, t_w, shared.empty() ? nullptr : &t_sh,
                 /*routed_scale=*/1.0f);
  return out;
}

}  // namespace vllm::glm5_next
