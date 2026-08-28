// EXL3 (exllamav3 trellis) — the quantization scheme, on the shared linear seam.
//
// UPSTREAM, and the two halves come from DIFFERENT places on purpose:
//   THE SEAM is vLLM's. `vllm/model_executor/layers/quantization/base_config.py:87-180`
//     (`QuantizationConfig` + `get_quant_method`) and
//     `vllm/model_executor/layers/linear.py:141-181` (`LinearMethodBase`) define
//     where a scheme plugs in, and this header mirrors them exactly as fp8.h and
//     compressed_tensors/schemes/nvfp4.h do.
//   THE FORMAT is exllamav3's, because vLLM registers no EXL3 at the parity pin
//     `5559679229bc961848b121ccdeaa8fa5d79bec98` — the fallback case AGENTS.md
//     admits, with the pin recorded in `.agents/oracles/exllamav3.md`
//     (`2398c05635fbbad01a0a51dce63c85c6c8a8450e`, tag v1.4.3, MIT).
//     `exllamav3/modules/quant/exl3.py:16-40` owns the four tensors and
//     `:183-214` the runtime form this method computes.
//
// WHY THIS FILE EXISTS (QUANT-EXL3 W1, #2181). The trellis kernels have existed
// since `MODEL-DSV4-EXL3` W2 and are device-proven on GB10, but their only
// consumer was `src/vllm/model_executor/models/deepseek_v4.cpp` — a model-private
// arm, so no other architecture could reach the scheme and no stock EXL3
// checkpoint could load. That is the parallel-path shape AGENTS.md forbids.
//
// The compute is ONE `vt::Exl3Gemm`, which is already the whole fused linear:
//   C = had_r_128( had_r_128(A, pre_scale=suh) @ reconstruct(trellis), post_scale=svh )
// algebraically `A @ Exl3DequantLinear(trellis, suh, svh)` (`exl3.py:183-214` vs
// `:227-237`). Nothing here re-derives the format; this header is the BINDING.
#pragma once

#include <memory>
#include <string>

#include "vllm/model_executor/layers/linear.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace layers {

// One EXL3-quantized linear's storage. THREE tensors, not four: the `mcg` int32
// marker each linear may also carry is a codebook SELECTOR that is never read at
// inference (`exl3_lib/quantize.py:1414-1424`), the loader resolves it to
// `codebook` below, and the stock `turboderp/*-exl3` checkpoints ship no `mcg`
// tensor at all — `Linear.is_exl3_storage` requires only `{key}.trellis` with
// `suh|su` and `svh|sv` (`modules/linear.py:385-389`).
//
// There are NO SCALES. `exl3.py:38` says so in as many words ("scale is no
// longer used"), and a reader that goes looking for one is reading a different
// format.
struct Exl3Weight {
  // I8 [k/16, n/16, 32*bits] — the SAME BYTES the checkpoint stores as
  // `I16 [k/16, n/16, 16*bits]`, held at byte width because that is the shape
  // `vt::Exl3Gemm` reads (`ops.h`: "trellis i8 [k/16, n/16, 32*bits] (bytes)")
  // and because `vt::DType` has no 16-bit integer. The loader does the widening
  // once, at load, rather than every call site doing it again.
  OwnedTensor trellis;
  OwnedTensor suh;      // F16 [k]   input-side Hadamard sign vector
  OwnedTensor svh;      // F16 [n]   output-side Hadamard sign vector
  int codebook = 1;     // cb; 1 == MCG, `LinearEXL3`'s own default

  bool Empty() const { return trellis.bytes.empty(); }

  // k and n, recovered from the trellis geometry rather than from a config: a
  // 16x16 tile packs 256 weights, so dim 0 counts input tiles and dim 1 output
  // tiles (`exl3.py:47`).
  int64_t InFeatures() const { return trellis.shape[0] * 16; }
  int64_t OutFeatures() const { return trellis.shape[1] * 16; }

  // BITS ARE PER TENSOR, and `quantization_config.bits` is NOT this number.
  //
  // Measured on `turboderp/Llama-3.2-1B-Instruct-exl3` @ `3.0bpw`
  // (`f8f438c290680b15622270eff03bef23a458b1cf`): the body is 3-bit
  // (`mlp.gate_proj.trellis [128, 512, 48]`, 48 = 16*3) while `lm_head.trellis`
  // is `[128, 8016, 96]`, 96 = 16*6 — a SIX-bit head under a config that says
  // `bits: 3.0`. A reader that trusts the config scalar decodes the head at the
  // wrong width, and no shape check anywhere catches it, because the tensor is
  // self-consistent at either reading: the bytes are there either way and only
  // the values come out wrong. So the width is derived HERE, from the tensor,
  // and the config scalar is only ever a cross-check.
  int Bits() const {
    VT_CHECK(trellis.rank == 3,
             "exl3: trellis must be 3-D [k/16, n/16, 16*bits] (exl3.py:47), got rank " +
                 std::to_string(trellis.rank));
    const int64_t last = trellis.shape[2];
    VT_CHECK(last > 0 && last % 32 == 0,
             "exl3: trellis last dim must be 32*bits BYTES (16*bits i16 words on disk), got " +
                 std::to_string(last));
    const int64_t bits = last / 32;
    VT_CHECK(bits >= 1 && bits <= 8,
             "exl3: bits must be in [1, 8]; the trellis last dim " + std::to_string(last) +
                 " implies " + std::to_string(bits));
    return static_cast<int>(bits);
  }
};

// The EXL3 linear method. `Apply` is one `vt::Exl3Gemm`, with the activation
// staged to fp16 on the way in.
//
// WHY THE STAGING IS NOT OPTIONAL. `Exl3Gemm` reads `a` as fp16 and nothing
// else — the CPU arm calls `HadRows(HadIo::kHalfHalf, ...)`
// (`cpu_exl3_kernels.cpp:205`) and the device arm stages `a_had` in fp16 —
// because exllamav3 runs the whole linear in fp16. A residual stream in bf16 or
// f32 therefore pays one `vt::CastF16` per call. That cast is a general op
// rather than a private helper here, and it is the third sibling of the
// `CastBf16`/`CastF32` pair the tree already had.
//
// THE OUTPUT DTYPE IS THE CALLER'S, never inherited from the kernel. `Exl3Gemm`
// writes f16 or f32 (`ops.h`), so an f32 request is written straight and a bf16
// request is written f32 and cast once — the destination the model dtype names,
// which is the polarity AGENTS.md §"Inherit vLLM defaults" requires and which a
// token gate cannot check for you.
class Exl3LinearMethod : public LinearMethodBase {
 public:
  explicit Exl3LinearMethod(const Exl3Weight* w) : w_(w) {}

  DBuf Apply(Dev d, const vt::Tensor& x, vt::DType out_dtype) const override {
    const int64_t M = x.shape[0];
    const int64_t K = w_->InFeatures();
    const int64_t N = w_->OutFeatures();
    VT_CHECK(x.rank == 2 && x.shape[1] == K,
             "exl3 linear: activation is [" + std::to_string(x.shape[0]) + "," +
                 std::to_string(x.rank == 2 ? x.shape[1] : -1) + "] but the weight needs K=" +
                 std::to_string(K));
    VT_CHECK(out_dtype == vt::DType::kF32 || out_dtype == vt::DType::kBF16 ||
                 out_dtype == vt::DType::kF16,
             "exl3 linear: out_dtype must be f32, bf16 or f16");

    // 1. the activation, in fp16. An already-fp16 caller pays no copy.
    DBuf a_owned;
    vt::Tensor a = x;
    if (x.dtype != vt::DType::kF16) {
      a_owned = DBuf(d, vt::DType::kF16, {M, K});
      vt::CastF16(d.q, a_owned.t(), x);
      a = a_owned.t();
    }
    DBuf a_had(d, vt::DType::kF16, {M, K});

    // 2. the three weight tensors, resident on this device.
    vt::Tensor trellis = ResidentWeight(d, w_->trellis);
    vt::Tensor suh = ResidentWeight(d, w_->suh);
    vt::Tensor svh = ResidentWeight(d, w_->svh);

    vt::Exl3GemmArgs args;
    args.bits = w_->Bits();
    args.codebook = w_->codebook;

    // 3. the GEMM. f16 out is written straight; anything else goes through f32,
    // which the kernel writes natively.
    if (out_dtype == vt::DType::kF16) {
      DBuf c(d, vt::DType::kF16, {M, N});
      vt::Exl3Gemm(d.q, c.t(), a, trellis, suh, svh, a_had.t(), args);
      return c;
    }
    DBuf c32(d, vt::DType::kF32, {M, N});
    vt::Exl3Gemm(d.q, c32.t(), a, trellis, suh, svh, a_had.t(), args);
    if (out_dtype == vt::DType::kF32) return c32;
    DBuf cbf(d, vt::DType::kBF16, {M, N});
    vt::CastBf16(d.q, cbf.t(), c32.t());
    return cbf;
  }

  const char* Name() const override { return "exl3-trellis"; }

 private:
  const Exl3Weight* w_;
};

// get_quant_method analogue, same shape as the fp8 and NVFP4 factories and
// overloaded on the weight type: a non-empty EXL3 weight selects the trellis
// method, everything else falls to bf16. The scheme is chosen ONCE, at load,
// from the checkpoint's populated weights — never per forward call by a
// tensor-name probe (base_config.h records why that matters).
inline std::unique_ptr<LinearMethodBase> MakeLinearMethod(const OwnedTensor& bf16_w,
                                                          const Exl3Weight& exl3_w) {
  if (!exl3_w.Empty()) return std::make_unique<Exl3LinearMethod>(&exl3_w);
  return std::make_unique<UnquantizedLinearMethod>(&bf16_w);
}

}  // namespace layers
}  // namespace vllm
