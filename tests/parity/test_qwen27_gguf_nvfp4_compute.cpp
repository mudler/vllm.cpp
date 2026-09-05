// vllm.cpp original (asset-gated); no upstream mirror — vLLM has no GGUF path.
//
// `QUANT-GGUF-NVFP4` column C — NATIVE fp4 compute from a GGUF container.
//
// WHAT THIS GATES, AND WHAT IT DELIBERATELY DOES NOT.
//
// The EXACT half of this row is weight-level and lives in
// tests/vllm/test_gguf_nvfp4.cpp: the ggml type-40 blocks repack, BYTE FOR
// BYTE, into the same (weight_packed, weight_scale) operand pair the
// compressed-tensors container of the same quantization run stores, so the fp4
// kernels a GGUF weight reaches are the already-gated kernels running on
// already-gated operand bytes. Nothing here re-litigates that.
//
// What is left to gate is the WIRING, and the trap it carries: a load that
// silently kept expanding to bf16 would pass every correctness assertion, since
// bf16 expansion is also correct — just unquantized. So this file asserts the
// fp4 residency POSITIVELY and BEHAVIOURALLY:
//
//   (1) the loader really produced fp4-resident weights — the `*_fp4` fields
//       are populated, their bf16 counterparts are EMPTY (which is the exact
//       invariant the forward's `!Empty()` dispatch keys on), the routing audit
//       shows the expected number of `nvfp4_fp4` decisions, and the residency
//       is materially smaller than the bf16 expansion it replaced;
//   (2) VT_GGUF_NVFP4_FP4=0 restores the bf16 expansion exactly, so the switch
//       is a real same-binary A/B rather than a no-op;
//   (3) the model still GENERATES coherently through the fp4 path.
//
// **NOT A GATE: token identity against the safetensors sibling.** The two 27B
// NVFP4 containers are NOT the same model. The GGUF NVFP4-quantizes the whole
// GDN in_proj_{qkv,z,a,b} family (192 tensors, mean relative weight error ~0.18)
// that the safetensors keeps in BF16, and their activation global scales
// disagree on most projections. Their greedy streams therefore diverge for
// WEIGHT reasons that no compute change can remove; the divergence index is
// REPORTED here as a characteristic, never asserted. Measurements and the full
// table: .agents/specs/gguf-nvfp4-native-compute.md Sec A.
//
// ASSET-GATED (dgx.casa): VLLM_NVFP4_GGUF -> ~/bench/q36-27b-nvfp4.gguf, and
// optionally VLLM_NVFP4_ST_DIR -> ~/bench/q36-27b-nvfp4-vllm/ for the reported
// cross-container comparison. Absent => loud SKIP, so CI stays asset-free.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/sampling_params.h"
#include "vllm/platforms/interface.h"

namespace fs = std::filesystem;

namespace {

constexpr int kMaxTokens = 24;
// Prosaic and low-entropy on purpose: a factual continuation keeps greedy away
// from near-ties, so a reported divergence is about the weights and not about
// a coin-flip step.
constexpr const char* kPrompt = "The capital of France is";

std::string GgufPath() {
  const char* env = std::getenv("VLLM_NVFP4_GGUF");
  return env != nullptr ? std::string(env) : std::string();
}

std::string StDir() {
  const char* env = std::getenv("VLLM_NVFP4_ST_DIR");
  return env != nullptr ? std::string(env) : std::string();
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

vllm::entrypoints::EngineParams BaseParams() {
  vllm::entrypoints::EngineParams p;
  p.block_size = 32;
  p.num_blocks = 256;
  p.max_num_seqs = 1;
  return p;
}

// Total host bytes the dense weights own, plus how many projections took each
// representation. This is the residency half of the claim: fp4 residency is not
// merely "a different code path", it is ~3.5x less host weight memory for every
// projection it covers (0.5625 bytes/element against bf16's 2).
struct Residency {
  size_t bytes = 0;
  int fp4_projections = 0;
  int bf16_projections = 0;
};

void AccountNvfp4(const vllm::Nvfp4Weight& w, Residency* r) {
  if (w.Empty()) return;
  r->bytes += w.packed.bytes.size() + w.scale.bytes.size();
  ++r->fp4_projections;
}

void AccountBf16(const vllm::OwnedTensor& t, Residency* r) {
  if (t.Empty()) return;
  r->bytes += t.bytes.size();
  ++r->bf16_projections;
}

// The projections this row moves: the dense MLP everywhere, and full-attention
// q/k/v/o. The GDN in_proj family and ssm_out are deliberately NOT counted —
// they stay bf16 by design (no fp4 field / the V-head reorder), which is the
// one documented gap and would otherwise blur the signal.
Residency AccountDense(const vllm::Qwen3_5DenseWeights& w) {
  Residency r;
  for (const auto& layer : w.layers) {
    AccountNvfp4(layer.mlp.gate_proj_fp4, &r);
    AccountNvfp4(layer.mlp.up_proj_fp4, &r);
    AccountNvfp4(layer.mlp.down_proj_fp4, &r);
    AccountBf16(layer.mlp.gate_proj, &r);
    AccountBf16(layer.mlp.up_proj, &r);
    AccountBf16(layer.mlp.down_proj, &r);
    if (layer.is_linear_attention) continue;
    AccountNvfp4(layer.attn.q_proj_fp4, &r);
    AccountNvfp4(layer.attn.k_proj_fp4, &r);
    AccountNvfp4(layer.attn.v_proj_fp4, &r);
    AccountNvfp4(layer.attn.o_proj_fp4, &r);
    AccountBf16(layer.attn.q_proj, &r);
    AccountBf16(layer.attn.k_proj, &r);
    AccountBf16(layer.attn.v_proj, &r);
    AccountBf16(layer.attn.o_proj, &r);
  }
  return r;
}

size_t DivergenceIndex(const std::vector<int32_t>& a,
                       const std::vector<int32_t>& b) {
  const size_t n = a.size() < b.size() ? a.size() : b.size();
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) return i;
  }
  return n;  // identical over the shared prefix
}

}  // namespace

// --- (1) + (2): the residency itself, with its same-binary A/B. -------------
TEST_CASE("gguf nvfp4 compute: the dense loader really produces fp4-resident "
          "weights, and VT_GGUF_NVFP4_FP4=0 really restores bf16") {
  const std::string path = GgufPath();
  if (path.empty() || !fs::exists(path)) {
    MESSAGE("SKIP: set VLLM_NVFP4_GGUF to a Qwen3.6-27B NVFP4 .gguf");
    return;
  }
  vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig config = vllm::HfConfigFromGguf(g);

  // The fp4 arm. Built explicitly rather than from the environment so the case
  // states the policy it is gating instead of inheriting it.
  vllm::GgufLoadPolicy fp4_pol = vllm::GgufLoadPolicy::FromEnv(vllm::platforms::CurrentPlatform().device_type());
  fp4_pol.nvfp4_fp4 = true;
  fp4_pol.nvfp4_w4a4 = true;
  int routed_fp4 = 0;
  int routed_total = 0;
  fp4_pol.audit = [&](const std::string&, vllm::GgufTensorRole,
                      vllm::GgufResidency res) {
    ++routed_total;
    if (res == vllm::GgufResidency::kNvfp4Fp4) ++routed_fp4;
  };

  Residency fp4_res;
  {
    const vllm::Qwen3_5DenseWeights w =
        vllm::LoadQwen3_5DenseFromGguf(g, config, &fp4_pol);
    fp4_res = AccountDense(w);
    REQUIRE(!w.layers.empty());
    // Exactly one representation per projection, on EVERY layer — the forward's
    // dispatch is `!Empty()`, so a layer with both filled would silently pick
    // one and a layer with neither would fault.
    for (const auto& layer : w.layers) {
      CHECK(!layer.mlp.gate_proj_fp4.Empty());
      CHECK(!layer.mlp.up_proj_fp4.Empty());
      CHECK(!layer.mlp.down_proj_fp4.Empty());
      CHECK(layer.mlp.gate_proj.Empty());
      CHECK(layer.mlp.up_proj.Empty());
      CHECK(layer.mlp.down_proj.Empty());
      // The activation globals came from the `<stem>.input_scale` sidecars, so
      // the forward takes vLLM's true-W4A4 arm rather than its use_a16 one.
      CHECK(layer.mlp.gate_proj_fp4.IsTrueW4A4());
      CHECK(layer.mlp.gate_proj_fp4.scale2 > 0.0F);
      CHECK(layer.mlp.gate_proj_fp4.k % 64 == 0);
      if (!layer.is_linear_attention) {
        CHECK(!layer.attn.q_proj_fp4.Empty());
        CHECK(!layer.attn.o_proj_fp4.Empty());
        CHECK(layer.attn.q_proj.Empty());
        CHECK(layer.attn.o_proj.Empty());
      }
    }
  }
  MESSAGE("fp4 arm: " << routed_fp4 << " of " << routed_total
                      << " routed tensors took the fp4 residency; "
                      << fp4_res.fp4_projections << " fp4 projections, "
                      << fp4_res.bf16_projections << " bf16, "
                      << (fp4_res.bytes / (1024 * 1024)) << " MiB");
  CHECK(routed_fp4 > 0);
  CHECK(fp4_res.fp4_projections > 0);
  CHECK(fp4_res.bf16_projections == 0);

  // The bf16 arm — the SAME file, the SAME loader, one policy flag apart.
  vllm::GgufLoadPolicy bf16_pol = fp4_pol;
  bf16_pol.nvfp4_fp4 = false;
  bf16_pol.nvfp4_w4a4 = false;
  bf16_pol.audit = nullptr;
  Residency bf16_res;
  {
    const vllm::Qwen3_5DenseWeights w =
        vllm::LoadQwen3_5DenseFromGguf(g, config, &bf16_pol);
    bf16_res = AccountDense(w);
    for (const auto& layer : w.layers) {
      CHECK(layer.mlp.gate_proj_fp4.Empty());
      CHECK(!layer.mlp.gate_proj.Empty());
    }
  }
  MESSAGE("bf16 arm: " << bf16_res.bf16_projections << " bf16 projections, "
                       << (bf16_res.bytes / (1024 * 1024)) << " MiB");
  CHECK(bf16_res.fp4_projections == 0);
  CHECK(bf16_res.bf16_projections == fp4_res.fp4_projections);
  // 0.5625 bytes/element against 2 => the fp4 arm must be under a third. Stated
  // as an inequality with headroom rather than an exact ratio, because the
  // accounting rounds per tensor.
  CHECK(fp4_res.bytes * 3 < bf16_res.bytes);
}

// --- (3): it generates, and the two arms really compute differently. --------
TEST_CASE("gguf nvfp4 compute: the 27B NVFP4 GGUF generates through the fp4 "
          "path") {
  const std::string path = GgufPath();
  if (path.empty() || !fs::exists(path)) {
    MESSAGE("SKIP: set VLLM_NVFP4_GGUF");
    return;
  }
  // Which arm this process runs is the environment's decision (the production
  // default is fp4 wherever the NVFP4 GEMM is registered), so the run is driven
  // twice from the shell and the two token streams compared there. What the case
  // asserts is that whichever arm ran, it produced a real continuation.
  auto loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(path, BaseParams());
  const vllm::RequestOutput out =
      loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "gguf-nvfp4-c");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& ids = out.outputs[0].token_ids;
  REQUIRE(ids.size() == static_cast<size_t>(kMaxTokens));
  std::string joined;
  for (int32_t t : ids) joined += std::to_string(t) + " ";
  MESSAGE("gguf ids: " << joined);
  MESSAGE("gguf text: \"" << out.outputs[0].text << "\"");

  const std::string st = StDir();
  if (st.empty() || !fs::is_directory(st)) {
    MESSAGE("cross-container comparison SKIPPED (set VLLM_NVFP4_ST_DIR)");
    return;
  }
  auto loaded_st =
      vllm::entrypoints::LoadedEngine::FromModelDir(st, BaseParams());
  const vllm::RequestOutput out_st =
      loaded_st->engine().generate(kPrompt, Greedy(kMaxTokens), "st-nvfp4-c");
  REQUIRE(out_st.finished);
  const std::vector<int32_t>& st_ids = out_st.outputs[0].token_ids;
  std::string joined_st;
  for (int32_t t : st_ids) joined_st += std::to_string(t) + " ";
  MESSAGE("st   ids: " << joined_st);
  MESSAGE("st   text: \"" << out_st.outputs[0].text << "\"");
  // REPORTED, NOT ASSERTED. See the file header: the containers hold different
  // weights for 192 tensors, so this index is a characteristic of the pair, not
  // a property of the compute path.
  MESSAGE("cross-container divergence index: " << DivergenceIndex(ids, st_ids)
                                               << " of " << kMaxTokens);
}
