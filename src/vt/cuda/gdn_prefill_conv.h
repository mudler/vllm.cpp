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

namespace vt::cuda {

// Pure predicate for the VT_CONV_REG contract: DEFAULT ON. The register-window
// kernel (CausalConv1dFwdRegKernel) is BIT-IDENTICAL (0-ulp) to the shipped tiled
// and scalar prefill conv kernels by construction. OFF only when the environment
// value is present AND its first character is '0'. nullptr (unset) and every
// non-'0'-leading value select the register kernel.
inline bool ConvRegFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
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

}  // namespace vt::cuda

#endif  // VT_CUDA_GDN_PREFILL_CONV_H_
