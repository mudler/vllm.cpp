// The NATIVE-layout EXL3 reader — QUANT-EXL3 W1b (#2181).
//
// The reader's job is to turn `{prefix}.{trellis,suh,svh}` into an `Exl3Weight`
// with the right geometry AND the right codebook. The second half is what this
// suite exists for.
//
// THE CODEBOOK IS SELECTED BY TENSOR PRESENCE, AND THE POLARITY IS THE OPPOSITE
// OF THE OBVIOUS GUESS. `LinearEXL3` sets `self.mcg = (self.mcg_tensor is not
// None)` and passes that BOOLEAN to `ext.reconstruct` (`exl3.py:74-77,197,223`),
// so a checkpoint shipping NO `mcg` tensor is NOT MCG — it is cb 0, the
// original QTIP 3INST. The first draft of this reader read absence as MCG.
//
// That mistake is invisible to every check a loader can make. The wrong
// multiplier yields a codebook with the SAME DISTRIBUTION and no relation to
// the right one, so the weight decodes to the correct RMS, every shape check
// passes, and the model emits fluent nonsense. MEASURED on
// `turboderp/Llama-3.2-1B-Instruct-exl3` @ 3.0bpw, layer 0 `q_proj`, against the
// unquantized `Llama-3.2-1B-Instruct` tensor fetched by range request:
//
//     cb 1 (mcg, WRONG here):  RMS 0.038454   cosine -0.0006
//     cb 0 (3INST, correct):   RMS 0.035941   cosine +0.9896
//     reference:               RMS 0.036056
//
// Same distribution, opposite verdict. Only a correlation against real
// exllamav3-produced data separates them, which is why the fixtures below gate
// the SELECTION and `test_exl3_dequant` gates the decode.
#include <doctest/doctest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/llama.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/models/dsv4_exl3_fixture.h"  // WriteSafetensors/StEntry, reused

namespace {

using vllm::StTensor;
using vllm::dense_loaders::IsExl3Projection;
using vllm::dense_loaders::LoadExl3;

// A hermetic stand-in for a shard: names -> tensors, with bytes this fixture
// owns. No file, no mmap — the reader takes a resolver and a probe, so the
// suite can hand it exactly the tensor set a checkpoint would carry.
struct FakeShard {
  std::map<std::string, StTensor> t;
  std::vector<std::vector<uint8_t>> storage;

  void Add(const std::string& name, const std::string& dtype,
           const std::vector<int64_t>& shape, size_t bytes) {
    storage.push_back(std::vector<uint8_t>(bytes, 0x5A));
    StTensor s;
    s.dtype = dtype;
    s.shape = shape;
    s.data = storage.back().data();
    s.nbytes = bytes;
    t[name] = s;
  }
  // One EXL3 projection at [k, n] and `bits`, without any codebook marker.
  void AddProjection(const std::string& proj, int64_t k, int64_t n, int bits) {
    Add(proj + ".trellis", "I16", {k / 16, n / 16, 16 * bits},
        static_cast<size_t>(k / 16) * (n / 16) * 16 * bits * 2);
    Add(proj + ".suh", "F16", {k}, static_cast<size_t>(k) * 2);
    Add(proj + ".svh", "F16", {n}, static_cast<size_t>(n) * 2);
  }
  vllm::TensorResolver Get() const {
    return [this](const std::string& n) -> const StTensor& {
      auto it = t.find(n);
      VT_CHECK(it != t.end(), "fake shard: tensor not found: " + n);
      return it->second;
    };
  }
  std::function<bool(const std::string&)> Has() const {
    return [this](const std::string& n) { return t.find(n) != t.end(); };
  }
};

}  // namespace

TEST_CASE("exl3 native loader: NO marker means codebook 0, not MCG") {
  FakeShard s;
  s.AddProjection("model.layers.0.mlp.gate_proj", 2048, 8192, 3);
  REQUIRE(IsExl3Projection(s.Has(), "model.layers.0.mlp.gate_proj"));

  const vllm::Exl3Weight w = LoadExl3(s.Get(), s.Has(), "model.layers.0.mlp.gate_proj");
  // THE ASSERTION THIS FILE EXISTS FOR. Reading absence as MCG is the defect
  // that decoded a real checkpoint to fluent nonsense.
  CHECK(w.codebook == 0);
  CHECK(w.InFeatures() == 2048);
  CHECK(w.OutFeatures() == 8192);
  CHECK(w.Bits() == 3);
}

TEST_CASE("exl3 native loader: an mcg marker means codebook 1") {
  FakeShard s;
  s.AddProjection("lm_head", 2048, 128256, 6);
  s.Add("lm_head.mcg", "I32", {1}, 4);

  const vllm::Exl3Weight w = LoadExl3(s.Get(), s.Has(), "lm_head");
  CHECK(w.codebook == 1);
  // The same fixture pins the per-tensor width: this head is SIX-bit, which is
  // what the published 3.0bpw artifact ships over a 3-bit body.
  CHECK(w.Bits() == 6);
}

TEST_CASE("exl3 native loader: an unported codebook REFUSES rather than decoding") {
  FakeShard s;
  s.AddProjection("p", 128, 128, 3);
  s.Add("p.mul1", "I32", {1}, 4);
  // cb 2 is upstream's dp4a byte-sum variant. Decoding it as 0 or 1 would be
  // silently wrong in exactly the way this suite's header documents, so the
  // reader refuses by name instead.
  CHECK_THROWS(LoadExl3(s.Get(), s.Has(), "p"));
}

TEST_CASE("exl3 native loader: the storage predicate is upstream's, all three tensors") {
  FakeShard s;
  s.Add("p.trellis", "I16", {8, 8, 48}, 8 * 8 * 48 * 2);
  // `Linear.is_exl3_storage` requires trellis WITH suh|su AND svh|sv
  // (`modules/linear.py:385-389`). A trellis alone is not EXL3 storage, and
  // answering yes here would route a half-written projection into this reader
  // instead of letting it fall through to the dense loader.
  CHECK_FALSE(IsExl3Projection(s.Has(), "p"));
  s.Add("p.suh", "F16", {128}, 256);
  CHECK_FALSE(IsExl3Projection(s.Has(), "p"));
  s.Add("p.svh", "F16", {128}, 256);
  CHECK(IsExl3Projection(s.Has(), "p"));
}

TEST_CASE("exl3 native loader: a transposed sign vector REFUSES BY NAME") {
  // suh is the INPUT side and svh the OUTPUT side. Swapping them loads, runs
  // and returns a confidently wrong answer on a square projection, so the
  // lengths are checked against the trellis geometry rather than each other.
  FakeShard s;
  s.Add("p.trellis", "I16", {8, 32, 48}, 8 * 32 * 48 * 2);  // k=128, n=512
  s.Add("p.suh", "F16", {512}, 1024);                       // swapped
  s.Add("p.svh", "F16", {128}, 256);                        // swapped
  std::string what;
  try {
    LoadExl3(s.Get(), s.Has(), "p");
    FAIL("exl3 native loader: swapped suh/svh did NOT throw");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("refusal: " << what);
  CHECK(what.find("suh") != std::string::npos);
}

// ── the loader, driven from an actual file ──────────────────────────────────
//
// Everything above exercises `LoadExl3` through a hand-built resolver. This
// case goes through `LoadLlamaForCausalLMWeights`, which is the production
// entry point, and it exists for one thing the resolver cases cannot reach:
// `LoadF16AsBf16Direct`, the F16 -> BF16 widening applied to the UNQUANTIZED
// remainder of an EXL3 checkpoint — the layernorms, the final norm and the
// whole embedding table.
//
// A fresh review replaced that conversion with a bare bit-copy — reinterpreting
// every F16 pattern as BF16, corrupting every norm and the embedding table —
// and the entire declared gate stayed GREEN. Nothing executed the function.
namespace {

std::vector<uint8_t> F16Bytes(const std::vector<float>& v) {
  std::vector<uint8_t> b(v.size() * 2);
  auto* p = reinterpret_cast<uint16_t*>(b.data());
  for (size_t i = 0; i < v.size(); ++i) p[i] = vt::F32ToF16(v[i]);
  return b;
}

std::vector<uint8_t> TrellisBytes(int64_t k, int64_t n, int bits, uint32_t seed) {
  std::vector<uint8_t> b(static_cast<size_t>(k / 16) * (n / 16) * 16 * bits * 2);
  uint32_t s = seed | 1u;
  for (auto& x : b) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    x = static_cast<uint8_t>(s & 0xffu);
  }
  return b;
}

}  // namespace

TEST_CASE("exl3 native loader: the F16 remainder is CONVERTED to bf16, not reinterpreted") {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "exl3_native_loader_fixture";
  fs::remove_all(dir);
  fs::create_directories(dir);

  const int64_t H = 128, I = 128, V = 128, Hq = 4, Hkv = 2, Dh = 64;
  const int64_t qdim = Hq * Dh, kvdim = Hkv * Dh;

  // Values chosen so a REINTERPRET is visibly wrong: each survives F16 exactly,
  // and its F16 bit pattern read as BF16 is a different number.
  const std::vector<float> norm_vals = [&] {
    std::vector<float> v(static_cast<size_t>(H));
    for (size_t i = 0; i < v.size(); ++i) v[i] = 0.5f + 0.015625f * static_cast<float>(i % 32);
    return v;
  }();

  std::vector<dsv4_exl3_fixture::StEntry> e;
  const auto add_proj = [&](const std::string& p, int64_t k, int64_t n, uint32_t seed) {
    e.push_back({p + ".trellis", "I16", {k / 16, n / 16, 48}, TrellisBytes(k, n, 3, seed)});
    e.push_back({p + ".suh", "F16", {k}, F16Bytes(std::vector<float>(static_cast<size_t>(k), 1.0f))});
    e.push_back({p + ".svh", "F16", {n}, F16Bytes(std::vector<float>(static_cast<size_t>(n), -1.0f))});
  };
  e.push_back({"model.embed_tokens.weight", "F16", {V, H},
               F16Bytes(std::vector<float>(static_cast<size_t>(V * H), 0.25f))});
  e.push_back({"model.norm.weight", "F16", {H}, F16Bytes(norm_vals)});
  e.push_back({"model.layers.0.input_layernorm.weight", "F16", {H}, F16Bytes(norm_vals)});
  e.push_back({"model.layers.0.post_attention_layernorm.weight", "F16", {H}, F16Bytes(norm_vals)});
  add_proj("model.layers.0.self_attn.q_proj", H, qdim, 11);
  add_proj("model.layers.0.self_attn.k_proj", H, kvdim, 12);
  add_proj("model.layers.0.self_attn.v_proj", H, kvdim, 13);
  add_proj("model.layers.0.self_attn.o_proj", qdim, H, 14);
  add_proj("model.layers.0.mlp.gate_proj", H, I, 15);
  add_proj("model.layers.0.mlp.up_proj", H, I, 16);
  add_proj("model.layers.0.mlp.down_proj", I, H, 17);
  add_proj("lm_head", H, V, 18);
  const std::string st =
      dsv4_exl3_fixture::WriteSafetensors(dir / "model.safetensors", e);

  {
    std::ofstream cfg(dir / "config.json");
    cfg << R"({"architectures":["LlamaForCausalLM"],"model_type":"llama",)"
        << R"("hidden_size":128,"num_hidden_layers":1,"num_attention_heads":4,)"
        << R"("num_key_value_heads":2,"head_dim":64,"intermediate_size":128,)"
        << R"("vocab_size":128,"rms_norm_eps":1e-5,"rope_theta":500000.0,)"
        << R"("torch_dtype":"bfloat16","tie_word_embeddings":true,)"
        << R"("quantization_config":{"quant_method":"exl3","bits":3.0}})";
  }

  const vllm::HfConfig config = vllm::LoadHfConfig((dir / "config.json").string());
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(st));
  const vllm::LlamaWeights w = vllm::LoadLlamaForCausalLMWeights(shards, config);

  // The EXL3 arm was taken by the PRODUCTION loader, not by a hand-built struct.
  REQUIRE(w.layers.size() == 1);
  CHECK(w.layers[0].attn.IsExl3());
  CHECK(w.layers[0].mlp.IsExl3());
  CHECK_FALSE(w.lm_head_exl3.Empty());
  // Absence of a marker means codebook 0, on the production path too.
  CHECK(w.layers[0].attn.q_proj_exl3.codebook == 0);

  // THE ASSERTION THIS CASE EXISTS FOR. Every value must be the bf16 ROUNDING
  // of the F16 value, which is what a conversion produces and what a
  // reinterpret cannot: reading the F16 pattern as BF16 changes the exponent
  // field and yields a wildly different number.
  REQUIRE(w.final_norm.dtype == vt::DType::kBF16);
  REQUIRE(w.final_norm.bytes.size() == static_cast<size_t>(H) * 2);
  const auto* got = reinterpret_cast<const uint16_t*>(w.final_norm.bytes.data());
  int reinterpreted = 0;
  for (int64_t i = 0; i < H; ++i) {
    CHECK(got[i] == vt::F32ToBF16(vt::F16ToF32(vt::F32ToF16(norm_vals[i]))));
    if (got[i] == vt::F32ToF16(norm_vals[i])) ++reinterpreted;
  }
  // Not vacuous: the two readings must actually DIFFER for these values, or the
  // check above would pass on a bit-copy too.
  CHECK(reinterpreted == 0);

  // The embedding table takes the same path and is the largest thing that would
  // be silently corrupted.
  REQUIRE(w.embed_tokens.dtype == vt::DType::kBF16);
  const auto* emb = reinterpret_cast<const uint16_t*>(w.embed_tokens.bytes.data());
  CHECK(emb[0] == vt::F32ToBF16(0.25f));

  fs::remove_all(dir);
}
