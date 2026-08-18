// Laguna-NVFP4 shared-expert KEEP-QUANT loader (VT_LAGUNA_SHARED_FP4).
//
// The XS-NVFP4 checkpoint (poolside/Laguna-XS-2.1-NVFP4) quantizes the per-layer
// SHARED expert (`mlp.shared_expert.{gate,up,down}_proj`) to W4A4 NVFP4 —
// weight_packed U8 [N,K/2] + weight_scale F8_E4M3 [N,K/16] + weight_global_scale.
// The default NVFP4 loader (laguna_weights.cpp LnLoadSharedExpertBf16) DEQUANTIZES
// it to BF16, so the resident/graph decode reads 2 bytes/weight — 4x the DRAM
// traffic of vLLM, which keeps the shared expert fp4 (0.5 byte/weight) and runs it
// through flashinfer-cutlass. The shared expert is a LARGE per-layer MLP, so on a
// DRAM-bound M=1 decode GEMV those extra bytes are a measured matmul-time gap.
//
// This TU keeps the shared expert fp4 RESIDENT instead: it re-reads the on-disk
// fp4 bytes into the (else-dead) `LagunaMoeWeights::shared_{gate,up,down}_proj_fp4`
// fields so the decode can route the shared expert through the SAME Marlin W4A16
// grouped-GEMM kernel that WINS on the routed experts (dense_nvfp4 single-expert
// path). alpha is left 0 (IsTrueW4A4()==false) so the weight is consumed W4A16 —
// bf16 activation, fp4 weight, scale2 — exactly the routed-expert Marlin regime
// (vLLM's forced-Marlin `use_a16` config). Loading here (a NEW additive TU driven
// from the gen driver BEFORE the shards are released) avoids touching the
// SACRED/concurrent laguna_weights.cpp; the fp4-raw read mirrors that file's
// LnLoadCtNvfp4Raw byte-for-byte, minus the (unused, W4A16) input-scale/alpha.
//
// Gated OFF by default (VT_LAGUNA_SHARED_FP4=1 opts in) for a same-binary A/B.
// When ON we free the now-dead fused router|shared_gate|shared_up projection (a
// DECODE-ONLY tensor; the decode re-reads the split `moe.router` for the router-only
// GEMV). The bf16 shared_gate/up/down copies are KEPT — the T>1 PREFILL forward
// (LagunaFfnBlock) still reads them per-token; only the T=1 decode goes fp4.
#include "vllm/model_executor/models/laguna.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"  // SafetensorsFile, StTensor
#include "vllm/model_executor/models/dense_weight_loaders.h"  // ReadF32Scalar (#1181)
#include "vllm/model_executor/models/dense_weight_loaders.h"      // dense_loaders::MakeOwned
#include "vllm/model_executor/models/qwen3_5_weights.h"           // Nvfp4Weight, OwnedTensor
#include "vt/dtype.h"                                             // VT_CHECK, vt::DType

namespace vllm {

// VT_LAGUNA_SHARED_FP4 (default ON): keep the shared expert fp4-resident and route
// it through the Marlin W4A16 grouped GEMM instead of dequantizing to bf16 — matching
// vLLM's own fp4 shared expert (parity-enablers-ship-as-defaults) at ~4x less shared-
// tower DRAM traffic on the M=1 decode GEMV. Read once per process. `=0` opts back out
// to the bf16 shared-expert GEMV (same-binary A/B). DGX-gated on ~/laguna-xs-nvfp4:
// coherent, distributional-PASS (ours' first-40 ids ∈ vLLM's 8-run greedy set, 40/40),
// and faster (GPU 27.24->26.53 ms/step, wall 35.8->36.3 tok/s).
bool LagunaSharedFp4Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LAGUNA_SHARED_FP4");
    return !(e != nullptr && e[0] == '0');  // default ON; =0 opts out
  }();
  return on;
}

// True iff the (loaded) weights carry the fp4 shared expert (the XS-NVFP4 tower;
// the S-2.1-NVFP4 tower keeps the shared expert bf16, so the flag is a no-op there).
bool LagunaHasFp4SharedExpert(const LagunaWeights& w) {
  for (const LagunaLayerWeights& lw : w.layers)
    if (!lw.is_dense) return !lw.moe.shared_gate_fp4.Empty();
  return false;
}

namespace {

// F32 scalar read, from the shared seam. This was a hand-copied mirror of
// laguna_weights.cpp's own copy, and both bounded the size with `nbytes >= 4`,
// a FLOOR, so a scale ARRAY was read as element 0 (#1181).
using dense_loaders::ReadF32Scalar;

// W4A16 fp4-raw read of one shared-expert projection. Byte-identical to
// laguna_weights.cpp LnLoadCtNvfp4Raw for the weight fields (n/k/scale2/packed/
// scale); the on-disk input_global_scale + alpha are DELIBERATELY not read, so
// IsTrueW4A4()==false and the weight is consumed W4A16 (bf16 activation) — the
// same way the routed-expert Marlin path ignores the activation-quant fields.
Nvfp4Weight ShLoadSharedNvfp4W4A16(
    const std::function<const StTensor&(const std::string&)>& get, const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8" && packed.shape.size() == 2,
           "laguna shared-fp4: U8 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0, "laguna shared-fp4: in_dim must be %16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3", "laguna shared-fp4: F8_E4M3 weight_scale for " + proj);
  const float wgs = ReadF32Scalar(get, proj + ".weight_global_scale");
  VT_CHECK(wgs != 0.0F, "laguna shared-fp4: zero weight_global_scale for " + proj);
  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.weight_global_scale_inv = wgs;
  r.scale2 = 1.0F / wgs;
  // input_global_scale_inv / alpha left 0 => W4A16 (IsTrueW4A4() == false).
  r.packed = dense_loaders::MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  VT_CHECK(packed.nbytes == r.packed.bytes.size(), "laguna shared-fp4: packed size " + proj);
  std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
  r.scale = dense_loaders::MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  VT_CHECK(ws.nbytes == r.scale.bytes.size(), "laguna shared-fp4: scale size " + proj);
  std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
  return r;
}

}  // namespace

// Populate the fp4 shared-expert fields from the checkpoint shards when
// VT_LAGUNA_SHARED_FP4=1 AND the tower quantizes the shared expert (XS-NVFP4).
// Called from the gen driver AFTER LoadLagunaForCausalLMWeights and BEFORE the
// shards are released (the fp4 bytes are copied out here). When it loads, the
// now-dead bf16 shared copies + the fused router|shared|shared projection are
// freed (Empty() true) — the decode re-reads the split `moe.router` for the
// router-only GEMV and the fp4 fields for the shared expert. No-op otherwise.
void LagunaLoadSharedExpertFp4(const std::vector<SafetensorsFile>& shards, LagunaWeights& w) {
  if (!LagunaSharedFp4Enabled() || !w.has_nvfp4_weights) return;

  // name -> shard resolver (mirror laguna_weights.cpp LoadLagunaForCausalLMWeights).
  auto where = std::make_shared<std::unordered_map<std::string, const SafetensorsFile*>>();
  for (const SafetensorsFile& s : shards)
    for (const std::string& n : s.Names()) (*where)[n] = &s;
  const std::function<const StTensor&(const std::string&)> get =
      [where](const std::string& name) -> const StTensor& {
    auto it = where->find(name);
    VT_CHECK(it != where->end(), "laguna shared-fp4: tensor not found: " + name);
    return it->second->Get(name);
  };
  const auto has = [where](const std::string& name) {
    return where->find(name) != where->end();
  };

  int64_t loaded = 0;
  for (size_t l = 0; l < w.layers.size(); ++l) {
    LagunaLayerWeights& lw = w.layers[l];
    if (lw.is_dense) continue;
    const std::string se = "model.layers." + std::to_string(l) + ".mlp.shared_expert.";
    if (!has(se + "gate_proj.weight_packed")) return;  // bf16 shared tower (S-2.1) => no-op
    lw.moe.shared_gate_fp4 = ShLoadSharedNvfp4W4A16(get, se + "gate_proj");
    lw.moe.shared_up_fp4 = ShLoadSharedNvfp4W4A16(get, se + "up_proj");
    lw.moe.shared_down_fp4 = ShLoadSharedNvfp4W4A16(get, se + "down_proj");
    // Free the fused router|shared_gate|shared_up projection: it is a DECODE-ONLY
    // tensor (the resident/graph decode), and with SHARED_FP4 the decode uses the
    // split `moe.router` for the router-only GEMV + the fp4 fields for the shared
    // expert, so nothing reads it. Assigning an empty OwnedTensor deallocates the
    // vector AND makes Empty() true (unlike ReleaseHost). The bf16 shared_gate/up/
    // down are KEPT — the T>1 PREFILL forward (LagunaFfnBlock) still reads them
    // per-token; only the T=1 decode goes fp4 here.
    lw.moe.router_shared_gu = OwnedTensor{};
    ++loaded;
  }
  std::fprintf(stderr, "[gen] VT_LAGUNA_SHARED_FP4: kept %lld shared experts fp4-resident "
                       "for decode (freed fused router-shared projection; bf16 shared kept "
                       "for prefill)\n",
               static_cast<long long>(loaded));
}

}  // namespace vllm
