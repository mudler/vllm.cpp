// vllm.cpp original; no upstream mirror. BACKEND-TENSTORRENT-KEEPQUANT W3
// (#2959) golden recipe.
//
// THE DEQUANTIZED-ARTIFACT DUMP: the q4km vehicle's teacher-forcing oracle
// input. The row spec fixes the oracle chain for the Tenstorrent keep-quant
// gate: teacher-force `transformers` on the DEQUANTIZED artifact, never on
// the bf16 safetensors checkpoint ("those logits are a different model's").
// The pinned transformers cannot parse the `qwen35` GGUF architecture
// (measured 2026-09-06, vllm-neartie venv: "GGUF model with architecture
// qwen35 is not supported yet"), so the spec's second arm applies: OUR OWN
// bit-exact decoder writes a safetensors dir first.
//
// This case IS that writer, driven through the PRODUCTION GGUF loader —
// `LoadQwen3_5DenseFromGguf`, the same entry the engine takes, under the
// all-expand policy (struct default: keep_quant off — the W3 keep-quant
// residency changes WHERE weights live, never their decoded values, and the
// oracle wants plain dense tensors transformers can load). The loader
// recovers raw-HF values and layouts (its contract: "matching the
// safetensors loader's layouts (transposes) and semantics (raw-HF
// values)"), so this file only RENAMES and re-orients each tensor to the
// checkpoint spelling, read from the real Qwen3.5-0.8B safetensors keys:
//
//   model.language_model.{embed_tokens,norm}          (tied head: no lm_head)
//   model.language_model.layers.N.{input,post_...}norm.weight
//   linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b}.weight  (SPLIT,
//       NOT the merged in_proj_qkvz spelling — measured on the 0.8B shard)
//   linear_attn.{A_log,dt_bias}        (NO .weight suffix; A_log F32)
//   linear_attn.conv1d.weight          [conv_dim, 1, K]
//   linear_attn.norm.weight            [Dv], F32 in the checkpoint
//   linear_attn.out_proj.weight, self_attn.{q,k,v,o}_proj.weight,
//   self_attn.{q,k}_norm.weight, mlp.{gate,up,down}_proj.weight
//
// Matmul weights carry the loader's own `nk` flag: nk=true is already torch
// [out, in] and is written verbatim; nk=false is Matmul-B [in, out] and is
// transposed here. The bf16 BYTES are never rounded through f64 — the
// loader's RNE values land on disk verbatim.
//
// The recipe (recorded in tests/parity/goldens/qwen35_gguf_q4km/
// manifest.json): run this case with VLLM_CPP_QWEN35_Q4KM_GGUF and
// VLLM_CPP_QWEN35_Q4KM_DUMP_DIR set, copy config.json + tokenizer files
// from the pinned bf16 snapshot into the dump dir, VERIFY every dumped
// tensor against that checkpoint (names, shapes, and values inside the
// q4km quantization noise — the layout proof), then teacher-force with
// scripts/qwen3-neartie-gap-transformers.py --model <dump dir>.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vt/dtype.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"

namespace fs = std::filesystem;

namespace {

bool HasTensor(const vllm::GgufFile& g, const std::string& name) {
  for (const vllm::GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

// One output tensor, already in checkpoint dtype/layout.
struct StEntry {
  std::string name;
  std::string dtype;  // "BF16" | "F32"
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

// The tensor as torch [out, in] bf16, transposing Matmul-B storage when the
// loader left it nk=false.
std::vector<uint16_t> ToTorchRows(const vllm::OwnedTensor& t, int64_t* out_rows,
                                  int64_t* out_cols) {
  const uint16_t* src = reinterpret_cast<const uint16_t*>(t.bytes.data());
  const int64_t a = t.shape[0], b = t.shape[1];
  std::vector<uint16_t> rows(static_cast<size_t>(a * b));
  if (t.nk) {  // already [N=out, K=in]
    std::memcpy(rows.data(), src, rows.size() * sizeof(uint16_t));
    *out_rows = a;
    *out_cols = b;
  } else {  // Matmul-B [K=in, N=out] -> transpose to [out, in]
    for (int64_t i = 0; i < a; ++i)
      for (int64_t j = 0; j < b; ++j)
        rows[static_cast<size_t>(j * a + i)] =
            src[static_cast<size_t>(i * b + j)];
    *out_rows = b;
    *out_cols = a;
  }
  return rows;
}

void EmitMatmul(const std::string& name, const vllm::OwnedTensor& t,
                std::vector<StEntry>* out) {
  REQUIRE(t.rank == 2);
  REQUIRE(t.dtype == vt::DType::kBF16);
  int64_t rows = 0, cols = 0;
  std::vector<uint16_t> r = ToTorchRows(t, &rows, &cols);
  StEntry e;
  e.name = name;
  e.dtype = "BF16";
  e.shape = {rows, cols};
  e.bytes.resize(r.size() * sizeof(uint16_t));
  std::memcpy(e.bytes.data(), r.data(), e.bytes.size());
  out->push_back(std::move(e));
}

// Row-slice of a merged raw-NK [2I, H] owner: rows [lo, hi) -> one tensor.
void EmitRowSlice(const std::string& name, const vllm::OwnedTensor& t,
                  int64_t lo, int64_t hi, std::vector<StEntry>* out) {
  REQUIRE(t.nk);  // the merged owner is raw torch orientation by contract
  REQUIRE(t.rank == 2);
  REQUIRE(t.dtype == vt::DType::kBF16);
  const int64_t cols = t.shape[1];
  const uint16_t* src = reinterpret_cast<const uint16_t*>(t.bytes.data());
  StEntry e;
  e.name = name;
  e.dtype = "BF16";
  e.shape = {hi - lo, cols};
  e.bytes.resize(static_cast<size_t>((hi - lo) * cols) * sizeof(uint16_t));
  std::memcpy(e.bytes.data(), src + static_cast<size_t>(lo * cols),
              e.bytes.size());
  out->push_back(std::move(e));
}

// 1-D or already-oriented tensor written verbatim (embed table, norms,
// conv1d). `shape` overrides the stored shape (the conv1d [C, K] -> [C, 1, K]
// reshape); empty means "use the tensor's own shape".
void EmitVerbatim(const std::string& name, const vllm::OwnedTensor& t,
                  std::vector<int64_t> shape, std::vector<StEntry>* out) {
  const bool f32 = t.dtype == vt::DType::kF32;
  const bool bf16 = t.dtype == vt::DType::kBF16;
  REQUIRE(f32 != bf16);  // exactly one of F32 / BF16
  StEntry e;
  e.name = name;
  e.dtype = f32 ? "F32" : "BF16";
  e.shape = shape.empty() ? std::vector<int64_t>(t.shape, t.shape + t.rank)
                          : std::move(shape);
  const size_t n = t.bytes.size();
  e.bytes.resize(n);
  std::memcpy(e.bytes.data(), t.bytes.data(), n);
  out->push_back(std::move(e));
}

void WriteSafetensors(const std::vector<StEntry>& entries,
                      const std::string& path) {
  nlohmann::json header = nlohmann::json::object();
  size_t off = 0;
  for (const StEntry& e : entries) {
    size_t n = 1;
    for (int64_t s : e.shape) n *= static_cast<size_t>(s);
    const size_t w = e.dtype == "F32" ? 4u : 2u;
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {off, off + n * w}}};
    off += n * w;
  }
  const std::string hs = header.dump();
  std::ofstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), "cannot open " << path << " for writing");
  const uint64_t hlen = hs.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  for (const StEntry& e : entries) f.write(
      reinterpret_cast<const char*>(e.bytes.data()),
      static_cast<std::streamsize>(e.bytes.size()));
  REQUIRE_MESSAGE(f.good(), "short write on " << path);
}

}  // namespace

TEST_CASE("qwen3.5 GGUF q4km dequantized-artifact dump (KEEPQUANT W3 oracle recipe)") {
  const char* gguf_env = std::getenv("VLLM_CPP_QWEN35_Q4KM_GGUF");
  const char* dump_env = std::getenv("VLLM_CPP_QWEN35_Q4KM_DUMP_DIR");
  if (gguf_env == nullptr || gguf_env[0] == '\0' || dump_env == nullptr ||
      dump_env[0] == '\0') {
    MESSAGE("SKIPPED: set VLLM_CPP_QWEN35_Q4KM_GGUF (the q4km vehicle) and "
            "VLLM_CPP_QWEN35_Q4KM_DUMP_DIR (an EMPTY output dir) to dump the "
            "dequantized teacher-forcing input; see the q4km golden manifest");
    return;
  }

  vllm::GgufFile g = vllm::GgufFile::Open(gguf_env);
  REQUIRE(vllm::IsQwen3_5Gguf(g));
  const vllm::HfConfig cfg = vllm::HfConfigFromGguf(g);
  // Struct default: keep_quant off — the historical all-expand load. The
  // oracle wants plain dense tensors; residency is a runtime concern.
  vllm::GgufLoadPolicy pol{};
  const vllm::Qwen3_5DenseWeights w = vllm::LoadQwen3_5DenseFromGguf(g, cfg, &pol);

  std::vector<StEntry> entries;
  const std::string p = "model.language_model.";
  EmitVerbatim(p + "embed_tokens.weight", w.embed_tokens, {}, &entries);
  EmitVerbatim(p + "norm.weight", w.final_norm, {}, &entries);
  // The q4km vehicle has NO output.weight (verified: the tied-head probe in
  // LoadEmbedAndHead aliases the head onto token_embd). The checkpoint spells
  // the tie through config.json tie_word_embeddings=true and ships no lm_head
  // key, so none is emitted. The loader leaves tied_lm_head unset on this
  // path, so the FILE, not the struct field, is the authority here.
  const bool tied = !HasTensor(g, "output.weight");
  if (!tied) {
    REQUIRE(!w.lm_head.Empty());
    EmitMatmul("lm_head.weight", w.lm_head, &entries);
  }

  const int64_t nl = static_cast<int64_t>(w.layers.size());
  for (int64_t il = 0; il < nl; ++il) {
    const vllm::Qwen3_5DenseLayerWeights& l = w.layers[static_cast<size_t>(il)];
    const std::string lp = p + "layers." + std::to_string(il) + ".";
    EmitVerbatim(lp + "input_layernorm.weight", l.input_layernorm, {}, &entries);
    EmitVerbatim(lp + "post_attention_layernorm.weight",
                 l.post_attention_layernorm, {}, &entries);
    if (l.is_linear_attention) {
      const vllm::GdnLayerWeights& gn = l.gdn;
      const std::string gp = lp + "linear_attn.";
      EmitMatmul(gp + "in_proj_qkv.weight", gn.in_proj_qkv, &entries);
      EmitMatmul(gp + "in_proj_z.weight", gn.in_proj_z, &entries);
      EmitMatmul(gp + "in_proj_b.weight", gn.in_proj_b, &entries);
      EmitMatmul(gp + "in_proj_a.weight", gn.in_proj_a, &entries);
      // [conv_dim, K] -> [conv_dim, 1, K], the checkpoint's conv1d spelling.
      EmitVerbatim(gp + "conv1d.weight", gn.conv1d_weight,
                   {gn.conv1d_weight.shape[0], 1, gn.conv1d_weight.shape[1]},
                   &entries);
      EmitVerbatim(gp + "A_log", gn.a_log, {}, &entries);
      EmitVerbatim(gp + "dt_bias", gn.dt_bias, {}, &entries);
      EmitVerbatim(gp + "norm.weight", gn.norm_weight, {}, &entries);
      EmitMatmul(gp + "out_proj.weight", gn.out_proj, &entries);
    } else {
      const vllm::FullAttnLayerWeights& fa = l.attn;
      const std::string ap = lp + "self_attn.";
      EmitMatmul(ap + "q_proj.weight", fa.q_proj, &entries);
      EmitMatmul(ap + "k_proj.weight", fa.k_proj, &entries);
      EmitMatmul(ap + "v_proj.weight", fa.v_proj, &entries);
      EmitMatmul(ap + "o_proj.weight", fa.o_proj, &entries);
      if (!fa.q_norm.Empty()) EmitVerbatim(ap + "q_norm.weight", fa.q_norm, {}, &entries);
      if (!fa.k_norm.Empty()) EmitVerbatim(ap + "k_norm.weight", fa.k_norm, {}, &entries);
    }
    // The GGUF loader fills the SPLIT mlp fields (gate/up/down); the merged
    // raw-NK owner is the safetensors production arm. Support both, keyed on
    // which is populated.
    const vllm::DenseMlpWeights& m = l.mlp;
    if (!m.gate_up_proj.Empty()) {
      const int64_t half = m.gate_up_proj.nk ? m.gate_up_proj.shape[0] / 2
                                             : m.gate_up_proj.shape[1] / 2;
      EmitRowSlice(lp + "mlp.gate_proj.weight", m.gate_up_proj, 0, half, &entries);
      EmitRowSlice(lp + "mlp.up_proj.weight", m.gate_up_proj, half, 2 * half, &entries);
    } else {
      EmitMatmul(lp + "mlp.gate_proj.weight", m.gate_proj, &entries);
      EmitMatmul(lp + "mlp.up_proj.weight", m.up_proj, &entries);
    }
    EmitMatmul(lp + "mlp.down_proj.weight", m.down_proj, &entries);
  }

  const fs::path dir(dump_env);
  fs::create_directories(dir);
  const std::string out = (dir / "model.safetensors").string();
  WriteSafetensors(entries, out);
  size_t total = 0;
  for (const StEntry& e : entries) total += e.bytes.size();
  MESSAGE("dumped " << entries.size() << " tensors, " << (total >> 20)
                    << " MiB -> " << out);
  MESSAGE("recipe: copy config.json + tokenizer files from the pinned bf16 "
          "snapshot into the dump dir, verify the tensors against the "
          "checkpoint (see manifest.json), then teacher-force with "
          "scripts/qwen3-neartie-gap-transformers.py --model " << dump_env);
}
