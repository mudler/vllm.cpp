// vllm.cpp original (vt runtime env-flag plumbing); the kernels these flags select
// are BIT-IDENTICAL-to-shipped ports, see below.
//
// Env-flag plumbing for the GDN PREFILL causal-conv1d forward (CausalConv1dFwd) and
// the fused post-conv preparation (GdnPostConv) kernel-efficiency selection, split
// into this pure-C++ (CUDA-free) header so the flag predicates are unit-testable on
// the CPU tier (tests/vt/test_gdn_prefill_conv.cpp). The kernels they select are
// CUDA-only and live in cuda_gdn.cu.
//
// (1) VT_CONV_REG — CausalConv1dFwdRegKernel: a register-resident sliding-window
//     port of the prefill causal_conv1d forward that mirrors vLLM's FLA Triton
//     kernel (vllm/model_executor/layers/mamba/ops/causal_conv1d.py
//     _causal_conv1d_fwd_kernel:397-452). The shipped tiled kernel
//     (CausalConv1dFwdTiledKernel) stages an x-tile through shared memory with two
//     __syncthreads() per token-tile and RE-LOADS the k conv weights from global
//     memory on every tap of every output token. The register kernel — exactly like
//     FLA — pre-loads the k per-channel weights into registers ONCE, keeps the
//     (k-1) history taps in a register sliding window (each x element read from
//     global EXACTLY ONCE, coalesced across the channel dimension), and parallelizes
//     the token axis into BLOCK_M chunks (grid.z) so the single-sequence prefill
//     saturates the GPU. It is BIT-IDENTICAL (0-ulp) to the shipped tiled and scalar
//     kernels: for every output (token t, channel c) it accumulates acc = bias; for
//     j in [0,k): acc += w[c*k+j] * x_tap[j] over the SAME tap ORDER (j=0 oldest ..
//     j=k-1 newest) over the SAME f32 values (the register window is filled by the
//     identical Load(x,...) / conv_state reads the scalar kernel does), then the
//     SAME silu/identity epilogue + round-to-store, and the SAME (K-1) state
//     write-back (ascending j, reading raw x / shifted conv_state directly, never a
//     value the window mutated). A conv is a fixed sum of k<=~4 f32 products;
//     preserving its accumulation order keeps the razor-thin greedy argmax from
//     flipping (a875397 lesson).
//
// SHIPPING (measured on DGX GB10, 35B NVFP4, nsys --cuda-graph-trace=node):
//   * VT_CONV_REG ships DEFAULT ON — bit-identical AND consistently faster on the
//     conv kernel (35B prefill c1 −4.7%, c6 −7.3%), a bandwidth-bound win.
//   * VT_GDN_POSTCONV_SPLIT ships DEFAULT OFF (OPT-IN) — bit-identical but measured
//     near-neutral / mildly concurrency-dependent (see its predicate), so per the
//     house "neutral ⇒ opt-in" convention the production default is unchanged.
//
// (2) VT_GDN_POSTCONV_SPLIT — GdnPostConvSplitKernel: the fused post-conv prep with
//     the V-copy + g/beta gating split across HV grid.y blocks (grid (T, Hk+Hv)),
//     mirroring vLLM's grid (ceil(L,BLOCK_T), H+HV) in
//     vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py
//     _fused_post_conv_kernel:57-149 where each V head is its own program. The
//     shipped GdnPostConvKernel packs the ENTIRE value_dim = Hv*Dv copy plus all Hv
//     gating scalars into a SINGLE grid.y block per token (head == Hk), serializing
//     ~Hv× more work behind 1/Hv the blocks; the split gives each (token, v-head)
//     its own block. It is BIT-IDENTICAL (0-ulp): the q/k L2-norm branch is byte-for
//     -byte the shipped code; the V copy is the same Load/Store per element; the
//     gating computes the same -exp(A_log)*softplus(a+dt_bias) and sigmoid(b) per
//     head over the same f32 inputs. No arithmetic reorders, so the output bytes are
//     unchanged by construction.
//
// VT_CONV_REG DEFAULT ON (bit-identical ⇒ never-slower + token-safe AND measured
// faster; '0'-rollback), mirroring the house VT_CONV_UPDATE_FAST / VT_RMSNORM_*_FAST
// default-ON convention. VT_GDN_POSTCONV_SPLIT DEFAULT OFF (opt-in; enabled by a
// non-'0' value) because it measured near-neutral. The launchers read getenv per call
// (prefill dispatch is coarse — one launch/step — so the getenv is negligible and
// in-process CUDA tests can flip the selection); the parse is factored here so it is
// regression-covered on every platform, not just DGX.
#ifndef VT_CUDA_GDN_PREFILL_CONV_H_
#define VT_CUDA_GDN_PREFILL_CONV_H_

#include <cstdint>

namespace vt::cuda {

// Pure predicate for the VT_CONV_REG contract: DEFAULT ON. The register-window
// kernel (CausalConv1dFwdRegKernel) is BIT-IDENTICAL (0-ulp) to the shipped tiled
// and scalar prefill conv kernels by construction. OFF only when the environment
// value is present AND its first character is '0'. nullptr (unset) and every
// non-'0'-leading value select the register kernel.
inline bool ConvRegFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

// Exact upstream (sequence, token-chunk) dispatch: DEFAULT ON, with `0` as the
// same-binary rollback to the legacy sequence-serial / rectangular-grid mapping.
// On sm_120 Qwen3.5-4B c32 the paired graph-node trace measured the causal-conv
// family at 720.047 -> 234.607 ms (3.07x), with byte-identical output tokens.
inline bool ConvExactChunksFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

// Three-arm same-binary experiment for the remaining prefill causal-conv gap.
// Arm 0 is the sealed runtime-width kernel. Arms 1 and 2 are valid only for the
// production width K=4: respectively one and two channels per 128-thread lane.
// Unset and every spelling except the exact strings "1" and "2" preserve arm 0.
enum class ConvChannelTileArm : uint8_t {
  kRuntimeWidth = 0,
  kWidthFour = 1,
  kWidthFourTwoChannels = 2,
};

inline constexpr ConvChannelTileArm ConvChannelTileArmFromEnv(const char* env_value) {
  if (env_value != nullptr && env_value[0] == '1' && env_value[1] == '\0') {
    return ConvChannelTileArm::kWidthFour;
  }
  if (env_value != nullptr && env_value[0] == '2' && env_value[1] == '\0') {
    return ConvChannelTileArm::kWidthFourTwoChannels;
  }
  return ConvChannelTileArm::kRuntimeWidth;
}

inline constexpr ConvChannelTileArm ResolveConvChannelTileArm(ConvChannelTileArm requested,
                                                              int64_t channels,
                                                              int64_t kernel_width) {
  if (requested != ConvChannelTileArm::kRuntimeWidth &&
      (channels <= 0 || kernel_width != 4)) {
    return ConvChannelTileArm::kRuntimeWidth;
  }
  return requested;
}

struct ConvChannelTileLaunchContract {
  ConvChannelTileArm arm;
  int64_t feature_blocks;
  int64_t threads_per_block;
};

inline constexpr int64_t kConvChannelTileThreads = 128;

inline constexpr ConvChannelTileLaunchContract ConvChannelTileLaunchContractFor(
    const char* env_value, int64_t channels, int64_t kernel_width) {
  const ConvChannelTileArm arm = ResolveConvChannelTileArm(
      ConvChannelTileArmFromEnv(env_value), channels, kernel_width);
  const int64_t channels_per_block =
      arm == ConvChannelTileArm::kWidthFourTwoChannels ? 256 : 128;
  return ConvChannelTileLaunchContract{
      arm,
      channels > 0 ? (channels + channels_per_block - 1) / channels_per_block : 0,
      kConvChannelTileThreads,
  };
}

// One shared dispatch seam for both the CUDA launcher and portable tests. Keeping
// the arm selection here means the tests exercise the exact branch logic used in
// production without adding a launch counter or other debug state to the hot path.
// The callbacks inline away at each call site and receive the already-resolved
// geometry, including runtime-width fallback for unsupported shapes.
template <typename RuntimeWidthLaunch, typename WidthFourLaunch,
          typename WidthFourTwoChannelsLaunch>
inline decltype(auto) DispatchConvChannelTileLaunch(
    const char* env_value, int64_t channels, int64_t kernel_width,
    RuntimeWidthLaunch&& runtime_width_launch,
    WidthFourLaunch&& width_four_launch,
    WidthFourTwoChannelsLaunch&& width_four_two_channels_launch) {
  const ConvChannelTileLaunchContract contract =
      ConvChannelTileLaunchContractFor(env_value, channels, kernel_width);
  if (contract.arm == ConvChannelTileArm::kWidthFour) {
    return width_four_launch(contract);
  }
  if (contract.arm == ConvChannelTileArm::kWidthFourTwoChannels) {
    return width_four_two_channels_launch(contract);
  }
  return runtime_width_launch(contract);
}

// Pure predicate for the VT_GDN_POSTCONV_SPLIT contract: DEFAULT OFF (OPT-IN). The
// split post-conv kernel (GdnPostConvSplitKernel) is BIT-IDENTICAL (0-ulp) to the
// shipped GdnPostConvKernel by construction, but the DGX nsys A/B measured it
// near-NEUTRAL and mildly concurrency-dependent (35B c1 prefill −3.8% on the kernel,
// c6 +4.7%) — because GdnPostConv time is dominated by the per-head q/k L2-norm
// reduction (already grid-parallel), not the single V-megablock the split targets.
// Per the house "neutral ⇒ opt-in" convention it ships OFF and is enabled only when
// the environment value is present AND its first character is not '0' (e.g. "1").
// nullptr (unset) and a '0'-leading value keep the shipped single-megablock kernel.
inline bool GdnPostConvSplitFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] != '0';
}

// Pure predicate for the VT_GDN_POSTCONV_FAST contract: DEFAULT ON ('0'-leading
// value rolls back to the shipped megablock). The fast post-conv kernel
// (GdnPostConvFastKernel) keeps the shipped single-megablock grid (T, Hk+1) — the
// low-launch-overhead layout the split failed to beat — but makes two provably
// BYTE-IDENTICAL changes that target the measured bottleneck (the per-head q/k
// L2-norm reduction, plus the V-megablock memory pass); it is BIT-IDENTICAL (0-ulp)
// to GdnPostConvKernel yet measured -24.3% (27B) / -24.8% (35B) per-call on GB10, so
// per the parity-enabler policy (byte-exact ⇒ never-slower + token-safe) it ships ON:
//   (a) it launches 128 threads/block instead of 256. For the Dk==Dv==128 gate
//       dims every lane owns exactly one element, so the 256-thread tree merely
//       added a leading `partial[t] += partial[t+128]` step over `partial[128..255]`
//       which are all +0 (those lanes never entered the `j<Dk` load loop). Dropping
//       the wasted half-block removes that no-op sync round and doubles the resident
//       block count per SM (better latency hiding on the reduction) with the SAME
//       summation tree over the SAME 128 squared values — 0-ulp identical.
//   (b) the V copy (mixed_qkv[:, 2*key_dim:] -> v_out, the largest memory pass) is
//       staged in 128-bit vector transactions (raw int4 when conv/out dtypes match;
//       per-element __bfloat162float / __float2bfloat16 — the SAME converts the
//       scalar Load/Store use — when they differ) instead of one transaction per
//       element. A pure copy/convert reorders no arithmetic, so the bytes are
//       unchanged. Gated to Dk==Dv==128 (guarantees 16B alignment + value_dim%8==0);
//       any other shape keeps the shipped megablock/split path. ON unless the
//       environment value is present AND its first char is '0' (rollback).
inline bool GdnPostConvFastFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

// Experimental spelling of vLLM's fused post-conv work partition: 16 tokens per
// block, one Q/K or V head per grid.y program, four warps. It remains opt-in until
// same-binary correctness and performance gates close; unset and '0' keep the
// byte-identical fast megablock default above.
inline bool GdnPostConvTokenTileFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] != '0';
}

// Shared production eligibility contract for the token-tile kernel. The
// explicit split experiment retains priority, and the token-tile indexing owns
// exactly four 32-element feature groups per lane, so both Q/K and V head
// widths must independently be 128. Unsupported shapes keep the existing
// megablock/split dispatch.
inline bool GdnPostConvTokenTileEligible(bool split, const char* env_value,
                                         int64_t dk, int64_t dv) {
  return !split && GdnPostConvTokenTileFlagIsOn(env_value) && dk == 128 &&
         dv == 128;
}

inline constexpr int64_t kGdnPostConvTokenTileTokens = 16;

inline constexpr int64_t GdnPostConvTokenTileGridX(int64_t tokens) {
  return (tokens + kGdnPostConvTokenTileTokens - 1) / kGdnPostConvTokenTileTokens;
}

}  // namespace vt::cuda

#endif  // VT_CUDA_GDN_PREFILL_CONV_H_
