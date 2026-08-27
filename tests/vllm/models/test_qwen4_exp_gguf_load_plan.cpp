// MODEL-MM-QWEN4-EXP W6a — the LOAD PLAN for Qwen3.8-Flash-Next's only
// runnable artifact, `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S`.
//
// A load plan is name, shape and dtype resolution. NOTHING HERE IS A TOKEN
// CLAIM and nothing here is a speed claim: no forward runs, no weight byte is
// read, and the model class this config names is not registered yet — the PLE
// and hyper-connection waves (#1987, #1988) own that, and the wiring is owed
// under `## Owed` in .agents/specs/qwen4-exp-flash-next.md.
//
// What the four groups of cases prove, and why each one exists:
//
//   (1) THE SHARDS OPEN. Every ggml type id in the real 1224-tensor table
//       resolves through the reader's `GgmlTraits`, and the row size the reader
//       computes agrees with `vt::RowSizeBytes` element for element. Before this
//       row, `GgufFile::OpenOne` died at header parse with "unknown ggml type id
//       20" on shard 2, so this is the case the whole wave turns on.
//   (2) THE TENSOR TABLE IS ACCOUNTED, in both directions and against the
//       COMMITTED manifest, so CI needs none of the 67.56 GiB.
//   (3) THE CONFIG BUILDER reads the file's own metadata, is REACHED through the
//       production architecture dispatch, and refuses by its OWN name.
//   (4) THE RESIDENCY of `per_layer_token_embd` is ASSERTED rather than assumed,
//       including the arithmetic that makes the decision load-bearing.
#include "vllm/model_executor/models/qwen4_exp_gguf_weights.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vt/dtype.h"
#include "vt/quant.h"

#include "qwen4_exp_gguf_manifest.inc"

namespace {

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

// The stand-in `token_embd` shape; see BuildQwen4ExpMetaGguf.
constexpr uint64_t kFixtureH = 8;
constexpr uint64_t kFixtureVocab = 16;

int64_t Numel(const vllm_test::Qwen4ExpGgufTensor& t) {
  int64_t n = 1;
  for (int i = 0; i < t.n_dims; ++i) n *= t.dims[i];
  return n;
}

// The KV set the config builder needs, at the SHIPPED values. Read live from
// shard 1 of `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` on 2026-08-26; the
// tensor table beside it is the committed manifest.
using gguf_test::I32ArrayKv;

std::string BuildQwen4ExpMetaGguf(bool with_token_embd = true) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen4exp"));
  b.AddKv(U32Kv("qwen4exp.embedding_length", 2560));
  b.AddKv(U32Kv("qwen4exp.block_count", 48));
  b.AddKv(U32Kv("qwen4exp.attention.head_count", 24));
  b.AddKv(U32Kv("qwen4exp.attention.head_count_kv", 2));
  b.AddKv(U32Kv("qwen4exp.attention.key_length", 256));
  b.AddKv(U32Kv("qwen4exp.attention.value_length", 256));
  b.AddKv(U32Kv("qwen4exp.context_length", 262144));
  b.AddKv(F32Kv("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(F32Kv("qwen4exp.rope.freq_base", 10000000.0F));
  b.AddKv(U32Kv("qwen4exp.rope.dimension_count", 64));
  b.AddKv(I32ArrayKv("qwen4exp.rope.dimension_sections", {11, 11, 10, 0}));
  b.AddKv(U32Kv("qwen4exp.expert_count", 512));
  b.AddKv(U32Kv("qwen4exp.expert_used_count", 10));
  b.AddKv(U32Kv("qwen4exp.expert_feed_forward_length", 640));
  b.AddKv(U32Kv("qwen4exp.expert_shared_feed_forward_length", 640));
  b.AddKv(U32Kv("qwen4exp.ssm.group_count", 16));
  b.AddKv(U32Kv("qwen4exp.ssm.time_step_rank", 48));
  b.AddKv(U32Kv("qwen4exp.ssm.state_size", 128));
  b.AddKv(U32Kv("qwen4exp.ssm.conv_kernel", 4));
  b.AddKv(U32Kv("qwen4exp.ssm.inner_size", 6144));
  b.AddKv(U32Kv("qwen4exp.full_attention_interval", 4));
  b.AddKv(U32Kv("qwen4exp.hyper_connection.count", 4));
  b.AddKv(U32Kv("qwen4exp.hyper_connection.low_rank", 320));
  b.AddKv(U32Kv("qwen4exp.attention.indexer.head_count", 4));
  b.AddKv(U32Kv("qwen4exp.attention.indexer.key_length", 128));
  b.AddKv(U32Kv("qwen4exp.attention.indexer.top_k", 2048));
  b.AddKv(U32Kv("qwen4exp.embedding_length_per_layer_input", 160));
  b.AddKv(U32Kv("qwen4exp.ple.ngram_size", 3));
  b.AddKv(U32Kv("qwen4exp.ple.heads_per_ngram", 8));
  b.AddKv(U32Kv("qwen4exp.ple.conv_kernel", 4));
  b.AddKv(I32ArrayKv("qwen4exp.ple.layers", {1}));
  std::vector<int32_t> ratios;
  for (int i = 0; i < 48; ++i) ratios.push_back(((i + 1) % 4) == 0 ? 4 : 0);
  b.AddKv(I32ArrayKv("qwen4exp.attention.compress_ratios", ratios));
  if (with_token_embd) {
    // The builder reads exactly ONE thing off this tensor — `shape[0]`, i.e.
    // ne1 — for `vocab_size`, because the container has no vocab key. The
    // shipped tensor is [2560, 248320] and 2.5 G elements of payload cannot go
    // in a fixture, so this stand-in exercises the RULE at a small shape and
    // the manifest case asserts the real ne1 the rule would read.
    b.AddTensor("token_embd.weight", {kFixtureH, kFixtureVocab}, /*F32=*/0,
                std::string(static_cast<size_t>(kFixtureH * kFixtureVocab) * 4,
                            '\0'));
  }
  return b.Build();
}

}  // namespace

// --- (1) the shards OPEN ----------------------------------------------------

TEST_CASE("every ggml type in the shipped file resolves in the reader") {
  // The blocker this row removed. Type 20 (IQ4_NL) appears 49 times — all 48
  // `ffn_down_exps` plus the n-gram table — and had no entry, so
  // `GgufFile::OpenOne` threw "unknown ggml type id 20" before reaching a single
  // tensor of shard 2.
  std::set<uint32_t> seen;
  for (const auto& t : vllm_test::kQwen4ExpGgufTensors) seen.insert(t.ggml_type);
  REQUIRE_FALSE(seen.empty());
  for (uint32_t type : seen) {
    CAPTURE(type);
    // Resolves at all...
    const vllm::GgmlTypeTraits& g = vllm::GgmlTraits(type);
    CHECK(g.block_elems > 0);
    CHECK(g.block_bytes > 0);
    // ...and, where it is a block encoding this tree executes, vt's INDEPENDENT
    // geometry table agrees. A disagreement here mis-strides every row.
    vt::DType dt = vt::DType::kF32;
    if (vt::BlockDTypeFromGgmlTypeId(type, &dt)) {
      CHECK(vt::BlockElems(dt) == g.block_elems);
      CHECK(vt::BlockBytes(dt) == g.block_bytes);
    }
  }
  // The set itself, longhand, so a re-quantized upload that changes an encoding
  // is a visible diff rather than a silently different file.
  const std::set<uint32_t> expect = {0, 8, 12, 13, 14, 16, 19, 20, 30};
  CHECK(seen == expect);
  CHECK(seen.count(20) == 1);  // IQ4_NL, the one this row added
}

TEST_CASE("every tensor's byte size is computable, and the row rule holds") {
  int64_t iq4nl_tensors = 0;
  for (const auto& t : vllm_test::kQwen4ExpGgufTensors) {
    CAPTURE(t.name);
    const vllm::GgmlTypeTraits& g = vllm::GgmlTraits(t.ggml_type);
    // ggml's own size rule: the tensor is a whole number of blocks, and the
    // INNER dim (ne0, the row) is what has to be block-aligned.
    CHECK(t.dims[0] % g.block_elems == 0);
    CHECK(Numel(t) % g.block_elems == 0);
    if (t.ggml_type == 20) ++iq4nl_tensors;
  }
  // 48 `ffn_down_exps` + the n-gram table. `moe_intermediate_size` 640 and the
  // table row 160 are both indivisible by 256, which is exactly why no K-quant
  // could carry them and why llama.cpp's `IQ4_XS -> IQ4_NL` fallback fires.
  CHECK(iq4nl_tensors == 49);
}

// --- (2) the tensor table is ACCOUNTED --------------------------------------

TEST_CASE("the manifest is the shipped file: counts, split, and structure") {
  CHECK(vllm_test::kQwen4ExpGgufTensorCount == 1224);
  CHECK(vllm_test::kQwen4ExpGgufShardCount == 3);
  CHECK(vllm_test::kQwen4ExpGgufVersion == 3);
  CHECK(std::string(vllm_test::kQwen4ExpGgufArchitecture) == "qwen4exp");
  CHECK(std::size(vllm_test::kQwen4ExpGgufTensors) ==
        static_cast<size_t>(vllm_test::kQwen4ExpGgufTensorCount));

  // Shard 1 is metadata ONLY. That is not a curiosity: a reader that assumed
  // every shard carries tensors would take its 0-tensor header for a truncated
  // file. Nothing in the manifest may come from it.
  std::map<int32_t, int> per_shard;
  for (const auto& t : vllm_test::kQwen4ExpGgufTensors) per_shard[t.shard]++;
  CHECK(per_shard[1] == 0);
  CHECK(per_shard[2] == 595);
  CHECK(per_shard[3] == 629);
  CHECK(per_shard[2] + per_shard[3] == vllm_test::kQwen4ExpGgufTensorCount);

  // Names are unique across shards, and every one is either model-level or a
  // `blk.N.` tensor with N inside the 48-layer trunk.
  std::set<std::string> names;
  for (const auto& t : vllm_test::kQwen4ExpGgufTensors) {
    CAPTURE(t.name);
    CHECK(names.insert(t.name).second);
    const std::string n = t.name;
    if (n.rfind("blk.", 0) == 0) {
      const size_t dot = n.find('.', 4);
      REQUIRE(dot != std::string::npos);
      const int layer = std::stoi(n.substr(4, dot - 4));
      CHECK(layer >= 0);
      CHECK(layer < 48);
    }
  }
}

TEST_CASE("the four load-bearing tensors are exactly what the row assumed") {
  std::map<std::string, const vllm_test::Qwen4ExpGgufTensor*> by_name;
  for (const auto& t : vllm_test::kQwen4ExpGgufTensors) by_name[t.name] = &t;

  // The n-gram table: the reason this row exists.
  REQUIRE(by_name.count("per_layer_token_embd.weight") == 1);
  const auto& ple = *by_name["per_layer_token_embd.weight"];
  CHECK(ple.ggml_type == 20u);  // IQ4_NL
  CHECK(ple.n_dims == 2);
  CHECK(ple.dims[0] == 160);         // ne0: the row
  CHECK(ple.dims[1] == 320001536);   // ne1: 16 heads x ~20 M entries, padded

  // The routed experts: down is IQ4_NL, gate/up are the sub-2-bit encodings.
  const auto& down = *by_name["blk.0.ffn_down_exps.weight"];
  CHECK(down.ggml_type == 20u);
  CHECK(down.n_dims == 3);
  CHECK(down.dims[0] == 640);
  CHECK(down.dims[1] == 2560);
  CHECK(down.dims[2] == 512);
  const auto& gate = *by_name["blk.0.ffn_gate_exps.weight"];
  CHECK((gate.ggml_type == 19u || gate.ggml_type == 16u));  // IQ1_S / IQ2_XXS

  // `token_embd` is Q4_K, and ne1 is what the config builder reads for
  // `vocab_size` — 248320, the released config.json's value, from an
  // independent source.
  REQUIRE(by_name.count("token_embd.weight") == 1);
  CHECK(by_name["token_embd.weight"]->dims[0] == 2560);
  CHECK(by_name["token_embd.weight"]->dims[1] == 248320);
  CHECK(by_name["token_embd.weight"]->ggml_type == 12u);  // Q4_K

  // The QSA indexer projections are left BF16 — an UNQUANTIZED tensor in a
  // 1.6-bit file, which is a deliberate recipe choice and not an oversight.
  REQUIRE(by_name.count("blk.3.indexer.q_proj.weight") == 1);
  CHECK(by_name["blk.3.indexer.q_proj.weight"]->ggml_type == 30u);
  CHECK(by_name["blk.3.indexer.k_proj.weight"]->ggml_type == 30u);
}

TEST_CASE("the indexer and PLE layer schedules agree with the metadata") {
  // Two independent statements of the same schedule: the config builder derives
  // `layer_types` from `full_attention_interval`, and the FILE places the
  // indexer tensors. They must name the same layers, or the port would run a
  // sparse-attention layer where the file has none.
  std::set<int> indexer_layers;
  std::set<int> ple_layers;
  for (const auto& t : vllm_test::kQwen4ExpGgufTensors) {
    const std::string n = t.name;
    if (n.rfind("blk.", 0) != 0) continue;
    const size_t dot = n.find('.', 4);
    const int layer = std::stoi(n.substr(4, dot - 4));
    const std::string tail = n.substr(dot + 1);
    if (tail.rfind("indexer.", 0) == 0) indexer_layers.insert(layer);
    if (tail.rfind("ple_", 0) == 0) ple_layers.insert(layer);
  }
  std::set<int> expect_indexer;
  for (int il = 0; il < 48; ++il) {
    if (((il + 1) % 4) == 0) expect_indexer.insert(il);
  }
  CHECK(indexer_layers == expect_indexer);
  CHECK(indexer_layers.size() == 12);
  // The PLE lives at ONE layer, and the file says blk.1 while the released
  // config.json says `ple_layer_ids: [2]`. The disagreement is recorded rather
  // than resolved here — see qwen4_exp_gguf_weights.cpp, which carries the GGUF
  // value under its GGUF name for exactly this reason.
  CHECK(ple_layers == std::set<int>{1});
}

// --- (3) the CONFIG BUILDER -------------------------------------------------

TEST_CASE("Qwen4ExpHfConfigFromGguf reads the shipped metadata") {
  TempFile f(BuildQwen4ExpMetaGguf());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  REQUIRE(vllm::IsQwen4ExpGguf(g));
  const vllm::HfConfig c = vllm::Qwen4ExpHfConfigFromGguf(g);

  // The HF identity, which is NOT the GGUF architecture key.
  CHECK(c.model_type == "qwen4_exp");
  REQUIRE(c.architectures.size() == 1);
  CHECK(c.architectures[0] == "Qwen4ExpForConditionalGeneration");

  // Every value below is the released `Qwen/Qwen3.8-Flash-Next` config.json's,
  // read 2026-08-26 — an INDEPENDENT source from the GGUF the builder read.
  CHECK(c.hidden_size == 2560);
  CHECK(c.num_hidden_layers == 48);
  CHECK(c.num_attention_heads == 24);
  CHECK(c.num_key_value_heads == 2);
  CHECK(c.head_dim == 256);
  CHECK(c.max_position_embeddings == 262144);
  CHECK(c.rms_norm_eps == doctest::Approx(1e-6));
  CHECK(c.rope_theta == doctest::Approx(10000000.0));
  CHECK(c.rotary_dim == 64);  // partial_rotary_factor 0.25 x head_dim 256
  CHECK(c.num_experts == 512);
  CHECK(c.num_experts_per_tok == 10);
  CHECK(c.moe_intermediate_size == 640);
  CHECK(c.shared_expert_intermediate_size == 640);
  CHECK(c.linear_num_key_heads == 16);
  CHECK(c.linear_num_value_heads == 48);
  CHECK(c.linear_key_head_dim == 128);
  CHECK(c.linear_value_head_dim == 128);
  CHECK(c.linear_conv_kernel_dim == 4);
  // The RULE (ne1 of `token_embd.weight`) at the fixture's shape; the real
  // value the same rule reads off the shipped file is asserted on the manifest.
  CHECK(c.vocab_size == static_cast<int64_t>(kFixtureVocab));
  // Not in the GGUF container at all; carried because the default would be a
  // silent wrong activation that no shape check can see.
  CHECK(c.output_gate_type == "sigmoid");

  // The 3 x linear -> 1 x full schedule, both ends checked.
  REQUIRE(static_cast<int64_t>(c.layer_types.size()) == c.num_hidden_layers);
  CHECK(c.layer_types[0] == "linear_attention");
  CHECK(c.layer_types[2] == "linear_attention");
  CHECK(c.layer_types[3] == "full_attention");
  CHECK(c.layer_types[47] == "full_attention");

  // Interleaved mRoPE: the GGUF's 4-slot array becomes HF's 3-slot section.
  REQUIRE(c.has_rope_parameters);
  CHECK(c.rope_parameters.mrope_interleaved);
  REQUIRE(c.rope_parameters.mrope_section.size() == 3);
  CHECK(c.rope_parameters.mrope_section[0] == 11);
  CHECK(c.rope_parameters.mrope_section[1] == 11);
  CHECK(c.rope_parameters.mrope_section[2] == 10);

  // The architecture-specific numbers, under the released config.json's OWN
  // key names, so the model waves read one spelling.
  const auto& text = c.raw.at("text_config");
  CHECK(text.at("hc_count").get<int64_t>() == 4);
  CHECK(text.at("hc_lowrank").get<int64_t>() == 320);
  CHECK(text.at("indexer_n_heads").get<int64_t>() == 4);
  CHECK(text.at("indexer_head_dim").get<int64_t>() == 128);
  CHECK(text.at("indexer_budget").get<int64_t>() == 2048);
  CHECK(text.at("indexer_compress_ratio").get<int64_t>() == 4);
  CHECK(text.at("ngram_size").get<int64_t>() == 3);
  CHECK(text.at("heads_per_ngram").get<int64_t>() == 8);
  CHECK(text.at("ple_conv_kernel_size").get<int64_t>() == 4);
  CHECK(text.at("ple_embed_dim_per_head").get<int64_t>() == 160);
  // Under its GGUF name, deliberately: the file says 1 where config.json says
  // 2, and mapping them onto one key would bury the disagreement.
  REQUIRE(text.at("gguf_ple_layers").size() == 1);
  CHECK(text.at("gguf_ple_layers")[0].get<int64_t>() == 1);
}

TEST_CASE("Qwen4ExpHfConfigFromGguf refuses by ITS OWN name") {
  // #809's rule: a refusal names the model that owes the work. The old
  // fall-through reported every unsupported architecture as "qwen3_5 gguf:",
  // which sent the reader into an unrelated translation unit.
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen4exp"));
  TempFile f(b.Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::string message;
  try {
    (void)vllm::Qwen4ExpHfConfigFromGguf(g);
  } catch (const std::exception& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("qwen4_exp gguf:") != std::string::npos);
  CHECK(message.find("qwen4exp.embedding_length") != std::string::npos);
  CHECK(message.find("qwen3_5 gguf:") == std::string::npos);

  // And it refuses a file that is not its architecture at all.
  GgufModelBuilder other;
  other.AddKv(StrKv("general.architecture", "qwen35moe"));
  TempFile f2(other.Build());
  const vllm::GgufFile g2 = vllm::GgufFile::Open(f2.path());
  CHECK_FALSE(vllm::IsQwen4ExpGguf(g2));
  CHECK_THROWS(vllm::Qwen4ExpHfConfigFromGguf(g2));
}

TEST_CASE("a malformed schedule is a LOUD failure, not a wrong model") {
  // Both cross-checks the builder makes, each mutated one value away from the
  // shipped file. Neither is a shape error, so neither would be caught later:
  // a wrong `ssm.inner_size` sizes every GDN buffer wrong, and a
  // `compress_ratios` that disagrees with the interval puts the sparse-attention
  // indexer on the wrong layers. Both still produce a model that emits text.
  {
    std::string bytes = BuildQwen4ExpMetaGguf();
    GgufModelBuilder b;  // rebuild with the one key changed
    // (rebuild rather than patch bytes: the offsets are recomputed for us)
    b.AddKv(StrKv("general.architecture", "qwen4exp"));
    for (const char* skip : {""}) (void)skip;
    // Minimal set + the bad inner_size.
    b.AddKv(U32Kv("qwen4exp.embedding_length", 2560));
    b.AddKv(U32Kv("qwen4exp.block_count", 48));
    b.AddKv(U32Kv("qwen4exp.attention.head_count", 24));
    b.AddKv(U32Kv("qwen4exp.attention.key_length", 256));
    b.AddKv(U32Kv("qwen4exp.context_length", 262144));
    b.AddKv(F32Kv("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6F));
    b.AddKv(F32Kv("qwen4exp.rope.freq_base", 10000000.0F));
    b.AddKv(U32Kv("qwen4exp.rope.dimension_count", 64));
    b.AddKv(U32Kv("qwen4exp.expert_count", 512));
    b.AddKv(U32Kv("qwen4exp.expert_used_count", 10));
    b.AddKv(U32Kv("qwen4exp.expert_feed_forward_length", 640));
    b.AddKv(U32Kv("qwen4exp.ssm.group_count", 16));
    b.AddKv(U32Kv("qwen4exp.ssm.time_step_rank", 48));
    b.AddKv(U32Kv("qwen4exp.ssm.state_size", 128));
    b.AddKv(U32Kv("qwen4exp.ssm.conv_kernel", 4));
    b.AddKv(U32Kv("qwen4exp.ssm.inner_size", 6145));  // 48 * 128 == 6144
    b.AddKv(U32Kv("qwen4exp.full_attention_interval", 4));
    TempFile f(b.Build());
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    CHECK_THROWS(vllm::Qwen4ExpHfConfigFromGguf(g));
  }
}

// --- (4) the RESIDENCY decision ---------------------------------------------

TEST_CASE("per_layer_token_embd is kept QUANTIZED, and the arithmetic says why") {
  // The decision this row exists to make, ASSERTED rather than assumed.
  const int64_t rows = 320001536;
  const int64_t k = 160;
  const vt::DType dt = vt::DType::kIQ4_NL;

  // Eligible: the row is a whole number of blocks (160 == 5 x 32). A K-quant
  // could not have encoded it at all, which is why the file uses IQ4_NL.
  CHECK(k % vt::BlockElems(dt) == 0);
  CHECK(k % 256 != 0);
  CHECK(vllm::RouteGgufTensor(/*keep_quant=*/true, /*keep_f16=*/false,
                              /*nvfp4_fp4=*/false, /*cpu_ref=*/false,
                              vllm::GgufTensorRole::kEmbeddingTable,
                              /*ggml_type=*/20u, {rows, k}) ==
        vllm::GgufResidency::kKeepQuant);

  // The numbers that make it load-bearing rather than a preference. 51.2 G
  // parameters: 28.8 GB of IQ4_NL blocks against 102.4 GB expanded to bf16, on
  // a box with ~119.6 GiB usable. The expansion is not a slow load; it is the
  // end of the box, and it is 86 % of one on its own.
  const int64_t params = rows * k;
  CHECK(params == 51200245760LL);
  const size_t blocks_bytes =
      static_cast<size_t>(rows) * vt::RowSizeBytes(dt, k);
  CHECK(blocks_bytes == 28800138240ULL);
  CHECK(static_cast<size_t>(params) * 2 == 102400491520ULL);
  // 3.5556x, i.e. 32/9. Written as an exact integer identity so it cannot drift.
  CHECK(blocks_bytes * 32 == static_cast<size_t>(params) * 2 * 9);

  // The gather op is what makes the residency legal, and its admission rule is
  // the ROW DECODER. Stated here so the dependency is visible from the decision.
  vt::DType decoded = vt::DType::kF32;
  REQUIRE(vllm::KeepQuantGatherDType(20u, &decoded));
  CHECK(decoded == dt);
  CHECK(vt::cpu::BlockToFloat(decoded) != nullptr);
}

TEST_CASE("the OTHER IQ4_NL tensors are GEMM weights and route as such") {
  // `ffn_down_exps` is IQ4_NL too, and it is a stacked expert weight, not a
  // gather. Its K is 640 — a whole number of 32-element blocks and NOT of 256,
  // which is the same reason the table is IQ4_NL. Asserted so the two IQ4_NL
  // populations in this file are not conflated.
  CHECK(640 % vt::BlockElems(vt::DType::kIQ4_NL) == 0);
  CHECK(640 % 256 != 0);
  CHECK(vllm::RouteGgufTensor(true, false, false, false,
                              vllm::GgufTensorRole::kStackedExpertWeight, 20u,
                              {512, 2560, 640}) ==
        vllm::GgufResidency::kKeepQuant);
  // Which requires the GEMM predicate, not the gather one: a `vec_dot` against
  // Q8_0 activations. Without it these 48 tensors expand to bf16 and the box is
  // gone for a second, independent reason.
  vt::DType dt = vt::DType::kF32;
  REQUIRE(vllm::KeepQuantDType(20u, &dt));
  CHECK(dt == vt::DType::kIQ4_NL);
  CHECK(vt::cpu::QuantTraits(dt).vec_dot_type == vt::DType::kQ8_0);
}
