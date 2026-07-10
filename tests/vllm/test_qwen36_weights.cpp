// Task 3 real-tensor loader check for the Qwen3.6-35B-A3B-NVFP4 MoE gate.
//
// GATED on the real checkpoint: if the pinned snapshot is not present (CI,
// laptops) every case skips and passes. On dgx it opens shard 1 (which holds
// embed_tokens + layers 0-16, so both a GDN layer (0) and a full-attn layer (3)
// are fully resolvable from one file) and proves the loader RESOLVES real
// tensors by name, verifies the standalone dequant helpers against python/torch
// golden bf16 bit patterns, and checks the full-layer loader's current native
// resident contract (FP8/NVFP4 raw weights by default; snapshot 491c2f1e; see
// .agents/specs/qwen36-forward-notes.md §6). The full-model load is exercised in Task 5.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"

namespace fs = std::filesystem;

namespace {

// Resolve the pinned 35B snapshot's shard 1, or "" if unavailable.
std::string FindShard1() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/"
      "models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    const fs::path shard = e.path() / "model-00001-of-00003.safetensors";
    if (fs::exists(shard, ec)) return shard.string();
  }
  return "";
}

uint16_t Bf16At(const vllm::OwnedTensor& t, int64_t i) {
  return reinterpret_cast<const uint16_t*>(t.bytes.data())[i];
}

}  // namespace

TEST_CASE("Qwen3.6-35B loader: real-tensor resolve + dequant + transpose") {
  const std::string shard1_path = FindShard1();
  if (shard1_path.empty()) {
    MESSAGE("SKIP: nvidia/Qwen3.6-35B-A3B-NVFP4 shard 1 not present");
    return;
  }
  const vllm::SafetensorsFile shard =
      vllm::SafetensorsFile::Open(shard1_path);
  const vllm::TensorResolver get =
      [&shard](const std::string& name) -> const vllm::StTensor& {
    return shard.Get(name);
  };

  SUBCASE("tensor naming + shapes match the config-derived dims") {
    // embed_tokens bf16 [vocab, H].
    const vllm::StTensor& embed =
        get("model.language_model.embed_tokens.weight");
    CHECK(embed.dtype == "BF16");
    REQUIRE(embed.shape.size() == 2);
    CHECK(embed.shape[0] == 248320);
    CHECK(embed.shape[1] == 2048);

    // GDN layer 0: in_proj_qkv FP8 [conv_dim=8192, H=2048].
    const vllm::StTensor& qkv = get(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
    CHECK(qkv.dtype == "F8_E4M3");
    CHECK(qkv.shape[0] == 8192);
    CHECK(qkv.shape[1] == 2048);

    // Full-attn layer 3: q_proj FP8 [2*Hq*Dh=8192, H], k_proj [Hkv*Dh=512, H].
    const vllm::StTensor& qp =
        get("model.language_model.layers.3.self_attn.q_proj.weight");
    CHECK(qp.dtype == "F8_E4M3");
    CHECK(qp.shape[0] == 8192);
    const vllm::StTensor& kp =
        get("model.language_model.layers.3.self_attn.k_proj.weight");
    CHECK(kp.shape[0] == 512);

    // MoE expert 0 NVFP4: gate_proj U8 [I=512, H/2=1024], down_proj [H, I/2].
    const vllm::StTensor& eg = get(
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight");
    CHECK(eg.dtype == "U8");
    CHECK(eg.shape[0] == 512);
    CHECK(eg.shape[1] == 1024);
    const vllm::StTensor& ed = get(
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight");
    CHECK(ed.shape[0] == 2048);
    CHECK(ed.shape[1] == 256);

    // Router gate bf16 [E=256, H=2048] (unquantized).
    const vllm::StTensor& rg = get("model.language_model.layers.0.mlp.gate.weight");
    CHECK(rg.dtype == "BF16");
    CHECK(rg.shape[0] == 256);
    CHECK(rg.shape[1] == 2048);
  }

  SUBCASE("NVFP4 expert dequant matches torch golden (pre-transpose)") {
    // layers.0.experts.0.down_proj: U8 [2048,256] -> bf16 [2048,512].
    const std::string p =
        "model.language_model.layers.0.mlp.experts.0.down_proj";
    const vllm::StTensor& w = get(p + ".weight");
    const vllm::StTensor& ws = get(p + ".weight_scale");
    const vllm::StTensor& ws2 = get(p + ".weight_scale_2");
    const int64_t out_dim = w.shape[0];       // 2048
    const int64_t in_dim = w.shape[1] * 2;    // 512
    float ws2v = 0.0F;
    std::memcpy(&ws2v, ws2.data, sizeof(float));

    std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
    vllm::DequantNvfp4ToBf16(w.data, ws.data, ws2v, out_dim, in_dim, dq.data());
    // Golden bf16 bit patterns from torch dequant (snapshot 491c2f1e).
    CHECK(dq[0 * in_dim + 0] == 0x3C7E);
    CHECK(dq[5 * in_dim + 17] == 0x3BCB);
    // Transpose correctness (dequant[o,i] -> transposed[i,o]) is checked on the
    // owned buffer in the full-layer SUBCASE below.
  }

  SUBCASE("FP8 attn dequant matches torch golden (pre-transpose)") {
    // layers.3.self_attn.q_proj: F8 [8192,2048], per-tensor weight_scale.
    const std::string p = "model.language_model.layers.3.self_attn.q_proj";
    const vllm::StTensor& w = get(p + ".weight");
    const vllm::StTensor& sc = get(p + ".weight_scale");
    const int64_t out_dim = w.shape[0];  // 8192
    const int64_t in_dim = w.shape[1];   // 2048
    float scale = 0.0F;
    std::memcpy(&scale, sc.data, sizeof(float));
    CHECK(scale == doctest::Approx(0.0007629395F).epsilon(1e-4));

    std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
    vllm::DequantFp8ToBf16(w.data, scale, out_dim * in_dim, dq.data());
    CHECK(dq[0 * in_dim + 0] == 0x3C0A);
    CHECK(dq[1 * in_dim + 5] == 0x3CE1);
  }

  SUBCASE("full-layer load: GDN (layer 0) and full-attn (layer 3)") {
    // A small expert count keeps the test fast; the loop/shared-expert/router
    // path is what we're proving, not all 256 experts (that is Task 5).
    const int64_t kExperts = 4;

    const vllm::Qwen3_5MoeLayerWeights gdn =
        vllm::LoadQwen3_5MoeLayer(get, "linear_attention", 0, kExperts);
    CHECK(gdn.is_linear_attention);
    // The default CUDA-parity loader keeps 35B W8A8 projections resident in
    // raw FP8 [N=out,K=in] orientation. VT_DENSE_NATIVE=0 is the legacy
    // dequant-to-bf16 path; the gate default should exercise the native fields.
    CHECK(gdn.gdn.in_proj_qkv.Empty());
    REQUIRE_FALSE(gdn.gdn.in_proj_qkv_fp8.Empty());
    CHECK(gdn.gdn.in_proj_qkv_fp8.n == 8192);
    CHECK(gdn.gdn.in_proj_qkv_fp8.k == 2048);
    CHECK(gdn.gdn.in_proj_qkv_fp8.packed.shape[0] == 8192);
    CHECK(gdn.gdn.in_proj_qkv_fp8.packed.shape[1] == 2048);
    CHECK(gdn.gdn.in_proj_qkv_fp8.weight_scale > 0.0F);
    CHECK(gdn.gdn.in_proj_qkv_fp8.input_scale > 0.0F);
    CHECK(gdn.gdn.in_proj_qkv_fp8.alpha > 0.0F);
    // conv1d collapsed to [conv_dim, K].
    CHECK(gdn.gdn.conv1d_weight.shape[0] == 8192);
    CHECK(gdn.gdn.conv1d_weight.shape[1] == 4);
    // a_log / dt_bias upcast to f32 [Hv=32].
    CHECK(gdn.gdn.a_log.dtype == vt::DType::kF32);
    CHECK(gdn.gdn.a_log.shape[0] == 32);
    // norm over Dv=128.
    CHECK(gdn.gdn.norm_weight.shape[0] == 128);
    // Router gate transposed to [H, E].
    CHECK(gdn.moe.router_gate.shape[0] == 2048);
    CHECK(gdn.moe.router_gate.shape[1] == 256);
    // Router gate transpose is value-preserving: gate[0,0] on disk == [0,0]
    // after transpose (corner is invariant). Golden 0x3BB3.
    CHECK(Bf16At(gdn.moe.router_gate, 0) == 0x3BB3);
    // NVFP4 expert weights are also resident now: raw packed [N=out,K/2] plus
    // fp8 group scales [N,K/16]. The helper-level dequant goldens above cover
    // the numeric bit patterns.
    CHECK(gdn.moe.expert_down.empty());
    REQUIRE(gdn.moe.expert_down_fp4.size() == static_cast<size_t>(kExperts));
    CHECK(gdn.moe.expert_down_fp4[0].n == 2048);
    CHECK(gdn.moe.expert_down_fp4[0].k == 512);
    CHECK(gdn.moe.expert_down_fp4[0].packed.shape[0] == 2048);
    CHECK(gdn.moe.expert_down_fp4[0].packed.shape[1] == 256);
    CHECK(gdn.moe.expert_down_fp4[0].scale.shape[0] == 2048);
    CHECK(gdn.moe.expert_down_fp4[0].scale.shape[1] == 32);
    CHECK(gdn.moe.expert_down_fp4[0].scale2 > 0.0F);

    const vllm::Qwen3_5MoeLayerWeights attn =
        vllm::LoadQwen3_5MoeLayer(get, "full_attention", 3, kExperts);
    CHECK_FALSE(attn.is_linear_attention);
    CHECK(attn.attn.q_proj.Empty());
    REQUIRE_FALSE(attn.attn.q_proj_fp8.Empty());
    CHECK(attn.attn.q_proj_fp8.n == 8192);
    CHECK(attn.attn.q_proj_fp8.k == 2048);
    CHECK(attn.attn.q_proj_fp8.packed.shape[0] == 8192);
    CHECK(attn.attn.q_proj_fp8.packed.shape[1] == 2048);
    CHECK(attn.attn.q_proj_fp8.weight_scale > 0.0F);
    CHECK(attn.attn.q_proj_fp8.input_scale > 0.0F);
    CHECK(attn.attn.q_proj_fp8.alpha > 0.0F);
    // qk-norm over head_dim=256.
    CHECK(attn.attn.q_norm.shape[0] == 256);
    CHECK(attn.attn.k_norm.shape[0] == 256);
    CHECK(attn.attn.o_proj.Empty());
    REQUIRE_FALSE(attn.attn.o_proj_fp8.Empty());
    CHECK(attn.attn.o_proj_fp8.n == 2048);
    CHECK(attn.attn.o_proj_fp8.k == 4096);
  }
}
