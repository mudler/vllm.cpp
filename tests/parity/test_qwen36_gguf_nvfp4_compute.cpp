// vllm.cpp original (asset-gated); no upstream mirror - vLLM has no GGUF path.
//
// `QUANT-GGUF-NVFP4` column C, the MoE (STACKED-EXPERT) arm - the 35B A3B.
//
// The dense sibling is tests/parity/test_qwen27_gguf_nvfp4_compute.cpp; read its
// header for what the C column gates and why. This file covers the ONE thing the
// dense arm cannot: a stacked `[E, out, in]` NVFP4 expert tensor becoming E
// fp4-resident `Nvfp4Weight`s, each with its OWN slab and its OWN
// `<stem>.scale[e]`. Both of those can be wrong SILENTLY (a wrong slab or a
// wrong scale index still yields legal fp4 operands and plausible logits), so
// this file asserts them positively:
//
//   (1) every routed expert stack is fp4-resident, its bf16 counterpart EMPTY,
//       with E entries carrying DISTINCT per-expert scale2 values taken from the
//       file's own `[256]` sidecar - not one value replicated;
//   (2) the per-expert operands are byte-identical to a direct repack of that
//       expert's own row range of the file (the byte-level half; the
//       cross-CONTAINER half, against the modelopt safetensors' unstacked
//       `mlp.experts.<e>.<proj>`, lives in tests/vllm/test_gguf_nvfp4.cpp);
//   (3) the model GENERATES through the path.
//
// WHAT DIFFERS FROM THE 27B, and it matters when reading the token streams:
//
//   * this checkpoint is MODELOPT, not compressed-tensors. Its `weight_scale_2`
//     is already the multiply form, so the GGUF `<stem>.scale` equals it
//     DIRECTLY rather than through a reciprocal (the loader is unaffected - it
//     only ever reads the GGUF side - but the cross-container check is);
//   * our SAFETENSORS 35B loader (LoadNvfp4Raw) deliberately runs this
//     checkpoint as W4A16: it never reads `input_scale`, so `IsTrueW4A4()` is
//     false and the forward takes vLLM's `use_a16` arm. The GGUF file DOES carry
//     `<stem>.input_scale`, so the GGUF arm defaults to TRUE W4A4 and is a
//     DIFFERENT arithmetic path from the safetensors arm of the same model.
//     `VT_GGUF_NVFP4_W4A4=0` is the switch that puts the two containers on the
//     same arm; a cross-container token comparison is only meaningful there;
//   * the file's GDN in_proj family, full-attention q/k/v/o and the whole MTP
//     (`blk.40`, `nextn.*`) layer are BF16 in this GGUF, so the fp4 residency
//     covers exactly the MoE experts + shared experts. `output.weight` is NVFP4
//     but is the lm_head, which has no fp4 field and expands by design.
//
// ASSET-GATED (dgx.casa): VLLM_NVFP4_MOE_GGUF -> ~/bench/q36-35b-a3b-nvfp4.gguf.
// Optional: VLLM_NVFP4_MOE_ST_DIR -> the modelopt safetensors dir, for the
// REPORTED cross-container stream; VLLM_NVFP4_MOE_ST_ONLY=1 to capture that
// reference in its own process (the 35B does not fit twice in the unified pool);
// VLLM_NVFP4_MOE_BF16=1 to additionally run the heavy bf16-expansion arm, which
// is ~3.5x the weight residency and is therefore opt-in.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/platforms/interface.h"

namespace fs = std::filesystem;

namespace {

constexpr int kMaxTokens = 24;
constexpr const char* kPrompt = "The capital of France is";

std::string Env(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr ? std::string(v) : std::string();
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

size_t DivergenceIndex(const std::vector<int32_t>& a,
                       const std::vector<int32_t>& b) {
  const size_t n = a.size() < b.size() ? a.size() : b.size();
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) return i;
  }
  return n;
}

std::string Joined(const std::vector<int32_t>& ids) {
  std::string s;
  for (int32_t t : ids) s += std::to_string(t) + " ";
  return s;
}

}  // namespace

// --- (1) + (2): the stacked-expert residency, per expert. -------------------
TEST_CASE("gguf nvfp4 moe: the 35B loader produces fp4-resident stacked "
          "experts, each with its OWN slab and scale") {
  const std::string path = Env("VLLM_NVFP4_MOE_GGUF");
  if (path.empty() || !fs::exists(path)) {
    MESSAGE("SKIP: set VLLM_NVFP4_MOE_GGUF to a Qwen3.6-35B-A3B NVFP4 .gguf");
    return;
  }
  vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig config = vllm::HfConfigFromGguf(g);

  vllm::GgufLoadPolicy pol = vllm::GgufLoadPolicy::FromEnv(vllm::platforms::CurrentPlatform().device_type());
  pol.nvfp4_fp4 = true;
  int routed_fp4 = 0;
  int routed_total = 0;
  int routed_stacked_fp4 = 0;
  pol.audit = [&](const std::string&, vllm::GgufTensorRole role,
                  vllm::GgufResidency res) {
    ++routed_total;
    if (res != vllm::GgufResidency::kNvfp4Fp4) return;
    ++routed_fp4;
    if (role == vllm::GgufTensorRole::kStackedExpertWeight) ++routed_stacked_fp4;
  };

  const vllm::Qwen3_5MoeWeights w =
      vllm::LoadQwen3_5MoeFromGguf(g, config, &pol);
  REQUIRE(!w.layers.empty());

  const int64_t E = config.num_experts;
  size_t fp4_bytes = 0;
  size_t bf16_bytes = 0;
  int fp4_stacks = 0;
  int bf16_stacks = 0;
  int layers_with_fp4 = 0;
  bool w4a4 = false;

  for (size_t li = 0; li < w.layers.size(); ++li) {
    const vllm::MoeBlockWeights& m = w.layers[li].moe;
    const std::vector<vllm::Nvfp4Weight>* stacks[] = {
        &m.expert_gate_fp4, &m.expert_up_fp4, &m.expert_down_fp4};
    const std::vector<vllm::OwnedTensor>* bf16[] = {
        &m.expert_gate, &m.expert_up, &m.expert_down};
    const char* names[] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    bool any = false;
    for (int s = 0; s < 3; ++s) {
      INFO("layer " << li << " stack " << std::string(names[s]));
      if (stacks[s]->empty()) {
        // The MTP layer's expert tensors are BF16 in this file and expand.
        ++bf16_stacks;
        for (const vllm::OwnedTensor& t : *bf16[s]) bf16_bytes += t.bytes.size();
        continue;
      }
      any = true;
      ++fp4_stacks;
      // Exactly one representation: the forward dispatches on `!empty()`.
      CHECK(bf16[s]->empty());
      REQUIRE(stacks[s]->size() == static_cast<size_t>(E));

      std::set<uint32_t> scale_bits;
      for (int64_t e = 0; e < E; ++e) {
        const vllm::Nvfp4Weight& x = (*stacks[s])[static_cast<size_t>(e)];
        REQUIRE(!x.Empty());
        REQUIRE(x.k % 64 == 0);
        REQUIRE(x.scale2 > 0.0F);
        REQUIRE(x.packed.bytes.size() ==
                static_cast<size_t>(x.n * x.k / 2));
        REQUIRE(x.scale.bytes.size() == static_cast<size_t>(x.n * x.k / 16));
        fp4_bytes += x.packed.bytes.size() + x.scale.bytes.size();
        uint32_t bits = 0;
        std::memcpy(&bits, &x.scale2, sizeof(bits));
        scale_bits.insert(bits);
        if (x.IsTrueW4A4()) w4a4 = true;
      }
      // THE per-expert scale-INDEX signal on the real file: a loader that read
      // `scale[0]` for every expert would leave exactly ONE distinct value here.
      // The file's own sidecars carry 100+ distinct scales per stack, so this is
      // not a coincidence of the data.
      CHECK(scale_bits.size() > static_cast<size_t>(E) / 4);
    }
    if (any) ++layers_with_fp4;
  }

  MESSAGE("35B fp4 arm: " << routed_fp4 << " of " << routed_total
                          << " routed tensors took the fp4 residency ("
                          << routed_stacked_fp4 << " stacked-expert); "
                          << fp4_stacks << " fp4 expert stacks over "
                          << layers_with_fp4 << " layers, " << bf16_stacks
                          << " bf16 stacks; expert residency "
                          << (fp4_bytes / (1024 * 1024)) << " MiB fp4 + "
                          << (bf16_bytes / (1024 * 1024)) << " MiB bf16; "
                          << "true-W4A4=" << (w4a4 ? "yes" : "no"));
  CHECK(routed_stacked_fp4 > 0);
  CHECK(fp4_stacks > 0);
  CHECK(layers_with_fp4 > 1);

  // (2) The byte-level half, sampled: expert `e`'s operands ARE the repack of
  // its own row range of the file. Sampled rather than exhaustive because the
  // exhaustive cross-container form is the asset-gated sweep in
  // tests/vllm/test_gguf_nvfp4.cpp; here the point is that the LOADER wired the
  // offsets, on the real stacked shapes.
  const int kSampleExperts[] = {0, 1, 255};
  int slabs_checked = 0;
  for (int layer = 0; layer < 4; ++layer) {
    const std::string gname =
        "blk." + std::to_string(layer) + ".ffn_gate_exps.weight";
    bool present = false;
    for (const vllm::GgufTensorInfo& t : g.Tensors()) {
      if (t.name == gname) present = true;
    }
    if (!present) continue;
    const vllm::GgufTensorInfo& t = g.Get(gname);
    if (t.ggml_type != 40) continue;
    const int64_t out_dim = t.shape[1];
    const int64_t in_dim = t.shape[2];
    const int64_t slab_blocks = out_dim * (in_dim / 64);
    const auto& got = w.layers[static_cast<size_t>(layer)].moe.expert_gate_fp4;
    REQUIRE(got.size() == static_cast<size_t>(E));
    for (int e : kSampleExperts) {
      if (e >= E) continue;
      INFO("layer " << layer << " expert " << e);
      std::vector<uint8_t> packed(static_cast<size_t>(out_dim * in_dim / 2));
      std::vector<uint8_t> scale(static_cast<size_t>(out_dim * in_dim / 16));
      vllm::RepackGgufNvfp4Rows(t.data + e * slab_blocks * 36, out_dim, in_dim,
                                packed.data(), scale.data());
      const vllm::Nvfp4Weight& x = got[static_cast<size_t>(e)];
      REQUIRE(x.packed.bytes.size() == packed.size());
      size_t pdiff = 0;
      const auto* xp =
          reinterpret_cast<const uint8_t*>(x.packed.bytes.data());
      for (size_t i = 0; i < packed.size(); ++i) {
        if (xp[i] != packed[i]) ++pdiff;
      }
      size_t sdiff = 0;
      const auto* xs = reinterpret_cast<const uint8_t*>(x.scale.bytes.data());
      for (size_t i = 0; i < scale.size(); ++i) {
        if (xs[i] != scale[i]) ++sdiff;
      }
      REQUIRE(pdiff == 0);
      REQUIRE(sdiff == 0);
      ++slabs_checked;
    }
  }
  MESSAGE("per-expert slab byte-identity: " << slabs_checked << " slabs");
  CHECK(slabs_checked > 0);

  // The ROUTER-GATE ORIENTATION this arm forced into the fused MoE block.
  //
  // `expand_nk` (default ON wherever the quantized GEMM is registered, which now
  // includes CUDA) keeps the GGUF router gate in the file's own [E, H] with
  // nk = true, where the safetensors loader hands over a transposed [H, E]. The
  // two fp4 fused MoE blocks hardcoded a vt::Matmul on that weight, so the FIRST
  // 35B NVFP4 GGUF forward through them threw "vt: matmul: inner dims mismatch";
  // MoeRouterLogits (qwen3_5.cpp) now branches on the flag. Recorded as a
  // structural assertion so the layout this arm exercises stays visible without
  // a GPU forward.
  const vllm::OwnedTensor& rg = w.layers.front().moe.router_gate;
  MESSAGE("router_gate: nk=" << (rg.nk ? 1 : 0) << " shape [" << rg.shape[0]
                             << ", " << rg.shape[1] << "]"
                             << "  (policy keep_quant=" << (pol.keep_quant ? 1 : 0)
                             << " expand_nk=" << (pol.expand_nk ? 1 : 0)
                             << " nvfp4_fp4=" << (pol.nvfp4_fp4 ? 1 : 0)
                             << " nvfp4_w4a4=" << (pol.nvfp4_w4a4 ? 1 : 0) << ")");
  REQUIRE(rg.rank == 2);
  if (rg.nk) {
    CHECK(rg.shape[0] == E);
    CHECK(rg.shape[1] == config.hidden_size);
  } else {
    CHECK(rg.shape[0] == config.hidden_size);
    CHECK(rg.shape[1] == E);
  }
}

// --- The bf16 arm of the same file (OPT-IN: ~3.5x the residency). -----------
TEST_CASE("gguf nvfp4 moe: VT_GGUF_NVFP4_FP4=0 restores the bf16 expansion") {
  const std::string path = Env("VLLM_NVFP4_MOE_GGUF");
  if (path.empty() || !fs::exists(path) || Env("VLLM_NVFP4_MOE_BF16").empty()) {
    MESSAGE("SKIP: set VLLM_NVFP4_MOE_GGUF and VLLM_NVFP4_MOE_BF16=1");
    return;
  }
  vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig config = vllm::HfConfigFromGguf(g);
  vllm::GgufLoadPolicy pol = vllm::GgufLoadPolicy::FromEnv(vllm::platforms::CurrentPlatform().device_type());
  pol.nvfp4_fp4 = false;
  pol.nvfp4_w4a4 = false;
  const vllm::Qwen3_5MoeWeights w =
      vllm::LoadQwen3_5MoeFromGguf(g, config, &pol);
  size_t bytes = 0;
  int stacks = 0;
  for (const auto& layer : w.layers) {
    const vllm::MoeBlockWeights& m = layer.moe;
    CHECK(m.expert_gate_fp4.empty());
    CHECK(m.expert_up_fp4.empty());
    CHECK(m.expert_down_fp4.empty());
    for (const auto* v : {&m.expert_gate, &m.expert_up, &m.expert_down}) {
      if (v->empty()) continue;
      ++stacks;
      for (const vllm::OwnedTensor& t : *v) bytes += t.bytes.size();
    }
  }
  MESSAGE("35B bf16 arm: " << stacks << " expert stacks, "
                           << (bytes / (1024 * 1024)) << " MiB");
  CHECK(stacks > 0);
}

// --- (3): it generates, and the arms really compute differently. ------------
TEST_CASE("gguf nvfp4 moe: the 35B NVFP4 GGUF generates through the fp4 path") {
  const std::string path = Env("VLLM_NVFP4_MOE_GGUF");
  const std::string st = Env("VLLM_NVFP4_MOE_ST_DIR");
  const bool st_only = !Env("VLLM_NVFP4_MOE_ST_ONLY").empty();
  if ((path.empty() || !fs::exists(path)) && !st_only) {
    MESSAGE("SKIP: set VLLM_NVFP4_MOE_GGUF");
    return;
  }
  // Which arm runs is the environment's decision (VT_GGUF_NVFP4_FP4 /
  // VT_GGUF_NVFP4_W4A4), so the streams are compared from the shell across
  // processes. What this case asserts is that whichever arm ran produced a real
  // continuation of the requested length.
  std::vector<int32_t> ids;
  if (!st_only) {
    auto loaded =
        vllm::entrypoints::LoadedEngine::FromModelDir(path, BaseParams());
    const vllm::RequestOutput out =
        loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "gguf-moe-c");
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    ids = out.outputs[0].token_ids;
    REQUIRE(ids.size() == static_cast<size_t>(kMaxTokens));
    MESSAGE("gguf ids: " << Joined(ids));
    MESSAGE("gguf text: \"" << out.outputs[0].text << "\"");
  }

  if (st.empty() || !fs::is_directory(st)) {
    MESSAGE("cross-container comparison SKIPPED (set VLLM_NVFP4_MOE_ST_DIR)");
    return;
  }
  auto loaded_st =
      vllm::entrypoints::LoadedEngine::FromModelDir(st, BaseParams());
  const vllm::RequestOutput out_st =
      loaded_st->engine().generate(kPrompt, Greedy(kMaxTokens), "st-moe-c");
  REQUIRE(out_st.finished);
  const std::vector<int32_t>& st_ids = out_st.outputs[0].token_ids;
  MESSAGE("st   ids: " << Joined(st_ids));
  MESSAGE("st   text: \"" << out_st.outputs[0].text << "\"");
  if (st_only) return;
  // REPORTED, NOT ASSERTED. The two containers are only on the same arithmetic
  // arm when VT_GGUF_NVFP4_W4A4=0 (see the file header), and even then the GGUF
  // stores the GDN/attention families as BF16 where the safetensors keeps FP8,
  // so identity is not guaranteed by construction.
  MESSAGE("cross-container divergence index: " << DivergenceIndex(ids, st_ids)
                                               << " of " << kMaxTokens);
}
