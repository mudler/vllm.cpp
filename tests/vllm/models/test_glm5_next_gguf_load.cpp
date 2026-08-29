// MODEL-MM-GLM53-FLASH W5c — the weight tower and `load_weights` for
// `Glm5NextForConditionalGeneration`.
//
// Issue [#2242](https://github.com/mudler/vllm.cpp/issues/2242), spec
// `.agents/specs/glm5-next-flash.md` §W5c.
//
// NOTHING HERE IS A TOKEN CLAIM and nothing here is a speed claim. No forward
// runs; the forward and the KV-cache spec still refuse by name and W5b
// ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)) owns them. What the
// four groups of cases prove, and why each one exists:
//
//   (1) THE PUBLISHED ARTIFACT'S TABLE IS ACCOUNTED, in both directions, out of
//       the committed 1412-tensor header manifest — so CI gates the name map
//       against the real 101.2535 GiB checkpoint with no asset. This is the
//       only instrument in the suite with authority over NAMES: we wrote the
//       fixture, we did not write the checkpoint, and a name the loader invents
//       is invisible to a fixture that would simply carry it.
//   (2) `blk.45` IS NOT A DECODER LAYER. The highest-value assertion in the
//       wave, asserted three ways because each one alone is satisfiable by a
//       wrong loader.
//   (3) THE LOAD RUNS THROUGH THE PRODUCTION ENTRY POINT and its RESULT is
//       read — structure, shapes, dtypes and bytes — never `REQUIRE_NOTHROW`
//       plus `!= nullptr`, which a hook returning `Glm5NextWeights{}` passes.
//   (4) EVERY REFUSAL IS BY NAME, entered through the same builder the happy
//       path uses.
#include "vllm/model_executor/models/glm5_next_loader.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "support/glm5_next_gguf_fixture.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/glm5_next_weights.h"
#include "vt/dtype.h"
#include "vt/quant.h"

#include "glm5_next_gguf_manifest.inc"

namespace {

using gguf_test::TempFile;
using namespace glm5_next_fixture;  // NOLINT(build/namespaces) — the fixture IS this suite's vocabulary

std::vector<int64_t> ShapeOf(const vllm::OwnedTensor& t) {
  return std::vector<int64_t>(t.shape, t.shape + t.rank);
}

float Bf16At(const vllm::OwnedTensor& t, int64_t i) {
  REQUIRE(t.dtype == vt::DType::kBF16);
  const auto* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
  return vt::BF16ToF32(p[i]);
}

float F32At(const vllm::OwnedTensor& t, int64_t i) {
  REQUIRE(t.dtype == vt::DType::kF32);
  const auto* p = reinterpret_cast<const float*>(t.bytes.data());
  return p[i];
}

// The fixture's ramps run past 256, where bf16 stops representing consecutive
// integers exactly (the step is 2 by 256 and 16 by 2048). An expectation has to
// be rounded the same way the loader's store rounds it, or the comparison is
// testing bf16 and not the loader.
float Rounded(float f) { return vt::BF16ToF32(vt::F32ToBF16(f)); }

std::set<std::string> FileNames(const vllm::GgufFile& g) {
  std::set<std::string> out;
  for (const auto& t : g.Tensors()) out.insert(t.name);
  return out;
}

// The PUBLISHED checkpoint's resolved params, built from the manifest's own
// metadata rather than from a transcription of it. Only the fields
// `Glm5NextExpectedGgufTensors` reads are set, and each is named where it comes
// from.
vllm::Glm5NextParams PublishedParams() {
  vllm::Glm5NextParams p;
  // `block_count - nextn_predict_layers` = 46 - 1 = 45 (#2269), the same
  // subtraction `Glm5NextHfConfigFromGguf` does.
  p.num_hidden_layers =
      vllm_test::kGlm5NextGgufBlockCount - vllm_test::kGlm5NextGgufNextnPredictLayers;
  p.tie_word_embeddings = false;  // the file carries `output.weight`
  p.has_vision = false;           // the text container declares no vision block
  // The schedule, READ out of the file's own 46-entry `attention.head_count_kv`
  // and TRUNCATED to the backbone — never synthesized from `idx % 4 == 3`.
  for (int64_t i = 0; i < p.num_hidden_layers; ++i) {
    p.layer_types.push_back(vllm_test::kGlm5NextGgufHeadCountKv[i] == 0
                                ? vllm::Glm5NextLayerKind::kLinearAttention
                                : vllm::Glm5NextLayerKind::kDeepseekSparseAttention);
    // `leading_dense_block_count` is 3 on the published artifact, which is
    // upstream's own `min(3, num_hidden_layers)` default.
    p.mlp_layer_types.push_back(i < 3 ? vllm::Glm5NextMlpKind::kDense
                                      : vllm::Glm5NextMlpKind::kSparse);
  }
  return p;
}

std::map<std::string, const vllm_test::Glm5NextGgufTensor*> ManifestByName() {
  std::map<std::string, const vllm_test::Glm5NextGgufTensor*> m;
  for (const auto& t : vllm_test::kGlm5NextGgufTensors) m[t.name] = &t;
  return m;
}

const vllm::Glm5NextLoadedModel& Open(const std::unique_ptr<vllm::LoadedModel>& m) {
  // `ModelAs`, never a `static_cast`: the checked form establishes the dynamic
  // type first (#775, #730).
  return vllm::ModelAs<vllm::Glm5NextLoadedModel>(
      *m, "Glm5NextForConditionalGeneration");
}

}  // namespace

// --- (1) the name map, against the REAL 1412-tensor table -------------------

TEST_CASE("glm5_next GGUF: the manifest is the published artifact's own table") {
  CHECK(std::string(vllm_test::kGlm5NextGgufArchitecture) == "glm5next");
  CHECK(vllm_test::kGlm5NextGgufVersion == 3);
  CHECK(vllm_test::kGlm5NextGgufShardCount == 4);
  CHECK(vllm_test::kGlm5NextGgufBlockCount == 46);
  CHECK(vllm_test::kGlm5NextGgufNextnPredictLayers == 1);
  const auto n = static_cast<int64_t>(std::size(vllm_test::kGlm5NextGgufTensors));
  REQUIRE(n == vllm_test::kGlm5NextGgufTensorCount);
  CHECK(n == 1412);
  // The schedule is 46 entries long — one per BLOCK, not one per layer — and
  // its last entry is non-zero, which is what makes the MTP block MLA-shaped
  // and therefore mistakable for a twelfth DSA layer.
  REQUIRE(std::size(vllm_test::kGlm5NextGgufHeadCountKv) == 46u);
  CHECK(vllm_test::kGlm5NextGgufHeadCountKv[45] != 0);
  int64_t kda = 0;
  int64_t dsa = 0;
  for (int64_t i = 0; i < 45; ++i) {
    (vllm_test::kGlm5NextGgufHeadCountKv[i] == 0 ? kda : dsa) += 1;
  }
  CHECK(kda == 34);
  CHECK(dsa == 11);
}

TEST_CASE("glm5_next GGUF: every ggml type in the shipped file resolves") {
  // A type the reader cannot size is a file that will not open, and this
  // artifact carries ten distinct encodings including the two (IQ2_XS 17 and
  // IQ4_XS 23) that #2245 and #2247 had to port before it could be read at all.
  std::set<uint32_t> seen;
  for (const auto& t : vllm_test::kGlm5NextGgufTensors) {
    seen.insert(t.ggml_type);
    CAPTURE(t.name);
    CAPTURE(t.ggml_type);
    // `GgmlTraits` THROWS, naming the id, on an untabulated type — which is
    // exactly how a file with an unported encoding fails to open. Before #2245
    // this reddened on ids 17 and 23.
    const vllm::GgmlTypeTraits* tr = nullptr;
    REQUIRE_NOTHROW(tr = &vllm::GgmlTraits(t.ggml_type));
    REQUIRE(tr != nullptr);
    CHECK(tr->block_elems > 0);
    CHECK(tr->block_bytes > 0);
    // ne0 is the innermost extent, and the reader sizes a row in whole blocks;
    // a tensor whose row is not a whole number of them cannot be read at all.
    CHECK(t.dims[0] % tr->block_elems == 0);
  }
  // Non-vacuous: the artifact really does carry ten encodings, so a loop that
  // silently saw none would be visible here.
  CHECK(seen.size() == 10u);
  CHECK(seen.count(17u) == 1u);  // IQ2_XS — 82 tensors
  CHECK(seen.count(23u) == 1u);  // IQ4_XS — 3 tensors
}

TEST_CASE("glm5_next GGUF: the name map accounts the shipped file BOTH WAYS") {
  // The strongest instrument this wave has, and the only one with authority
  // over names: we wrote the fixture, we did not write the checkpoint. A name
  // the loader invents is invisible to the fixture — the fixture would simply
  // carry it — and fatal here.
  const vllm::Glm5NextParams p = PublishedParams();
  const std::vector<std::string> enumerated =
      vllm::EnumerateGlm5NextGgufTensors(p);
  const std::set<std::string> ours(enumerated.begin(), enumerated.end());
  CHECK(ours.size() == enumerated.size());  // no name enumerated twice

  std::set<std::string> shipped;
  for (const auto& t : vllm_test::kGlm5NextGgufTensors) shipped.insert(t.name);
  REQUIRE(shipped.size() ==
          static_cast<size_t>(vllm_test::kGlm5NextGgufTensorCount));

  // FORWARD: every name we expect is in the file.
  std::vector<std::string> missing;
  for (const std::string& n : ours) {
    if (shipped.count(n) == 0) missing.push_back(n);
  }
  CAPTURE(missing.size());
  if (!missing.empty()) CAPTURE(missing.front());
  CHECK(missing.empty());

  // BACKWARD: every name in the file is either expected or is an MTP-block
  // tensor this port deliberately drops. There is no third bucket, and the
  // absence of one is the assertion: a tensor the file carries that nothing
  // accounts for is either a name we got wrong or a module we have not ported.
  std::vector<std::string> unexplained;
  const std::string mtp_prefix = "blk." + std::to_string(p.num_hidden_layers) + ".";
  for (const std::string& n : shipped) {
    if (ours.count(n) != 0) continue;
    if (n.rfind(mtp_prefix, 0) == 0) continue;
    unexplained.push_back(n);
  }
  CAPTURE(unexplained.size());
  if (!unexplained.empty()) CAPTURE(unexplained.front());
  CHECK(unexplained.empty());

  // And the arithmetic, so the two `empty()` checks above cannot both pass on
  // an enumeration that produced nothing: 1412 shipped = 1383 enumerated + 29
  // MTP-block tensors.
  int64_t mtp = 0;
  for (const std::string& n : shipped) {
    if (n.rfind(mtp_prefix, 0) == 0) ++mtp;
  }
  CHECK(mtp == 29);
  CHECK(static_cast<int64_t>(ours.size()) + mtp ==
        vllm_test::kGlm5NextGgufTensorCount);
}

TEST_CASE("glm5_next GGUF: the two SPLIT MLA halves are the file's own names") {
  // #2242's finding: the name map called this one tensor, `attn_kv_b.weight`,
  // and the published artifact carries two — because llama.cpp #27752's
  // converter splits `kv_b_proj` and transposes the k half. A map that named
  // the fused tensor refuses every DSA layer of the only file that exists.
  const auto by_name = ManifestByName();
  for (int64_t il : {int64_t{3}, int64_t{43}}) {
    const std::string blk = "blk." + std::to_string(il) + ".";
    CAPTURE(il);
    CHECK(by_name.count(blk + "attn_kv_b.weight") == 0u);
    REQUIRE(by_name.count(blk + "attn_k_b.weight") == 1u);
    REQUIRE(by_name.count(blk + "attn_v_b.weight") == 1u);
    // ne is reversed against torch, so k_b torch [64, 512, 256] reads
    // ne [256, 512, 64] and v_b torch [64, 256, 512] reads ne [512, 256, 64].
    // The two are DIFFERENT shapes, which is what the transpose produces.
    const auto* kb = by_name.at(blk + "attn_k_b.weight");
    const auto* vb = by_name.at(blk + "attn_v_b.weight");
    CHECK(std::vector<int64_t>(kb->dims, kb->dims + kb->n_dims) ==
          std::vector<int64_t>{256, 512, 64});
    CHECK(std::vector<int64_t>(vb->dims, vb->dims + vb->n_dims) ==
          std::vector<int64_t>{512, 256, 64});
  }
  // And `ssm_dt.bias`, the other half of the same finding: the converter
  // renames `.dt_bias` to `.dt_proj.bias`, so the file carries the `.bias`
  // suffix and no bare `ssm_dt`.
  CHECK(by_name.count("blk.0.ssm_dt") == 0u);
  REQUIRE(by_name.count("blk.0.ssm_dt.bias") == 1u);
  CHECK(by_name.at("blk.0.ssm_dt.bias")->dims[0] == 8192);
}

// --- (2) `blk.45` IS NOT A DECODER LAYER ------------------------------------

TEST_CASE("glm5_next GGUF: the MTP block is NOT loaded as a decoder layer") {
  // Asserted THREE ways, because each one alone is satisfiable by a wrong
  // loader:
  //
  //   * a depth of 45 is equally true of a stack built from blocks 0..44 and
  //     one built from 1..45;
  //   * "no `blk.45.*` name is enumerated" is equally true of a file that never
  //     had an MTP block;
  //   * so the loader also COUNTS what it skipped, positively, and the fixture
  //     carries a real MTP block for it to skip.
  const vllm::Glm5NextParams p = PublishedParams();
  REQUIRE(p.num_hidden_layers == 45);
  const std::vector<std::string> enumerated =
      vllm::EnumerateGlm5NextGgufTensors(p);
  for (const std::string& n : enumerated) {
    CAPTURE(n);
    REQUIRE(n.rfind("blk.45.", 0) != 0);
  }
  // The file DOES carry one, so the absence above is an exclusion and not a
  // vacuous truth. `.agents/specs/glm5-next-flash.md` records the same fact.
  const auto by_name = ManifestByName();
  CHECK(by_name.count("blk.45.attn_norm.weight") == 1u);
  CHECK(by_name.count("blk.45.nextn.eh_proj.weight") == 1u);
  // The MTP block carries NO hyper-connection parameters, unlike every backbone
  // block. A loader that built it would look for six tensors that are not
  // there.
  CHECK(by_name.count("blk.45.hc_attn_fn.weight") == 0u);
  CHECK(by_name.count("blk.44.hc_attn_fn.weight") == 1u);

  // And on a real load, through the production entry point.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  const vllm::Glm5NextWeights& w = Open(model).weights();
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));
  // POSITIVELY: the loader saw an MTP block and declined to build it.
  // 29 tensors, the same count `blk.45` carries in the published artifact:
  // two norms, the fifteen NoPE-MLA-plus-indexer tensors, the eight sparse-MoE
  // ones and the four `nextn.*` ones. It is a count of TENSORS rather than of
  // blocks so it adds up against `enumerated_tensors` below.
  CHECK(w.mtp_block_tensors_dropped == 29);
  // The stack starts at block 0, not at block 1. Layer 0's `attn_norm` is
  // written with `NormTag(0, 0)` and no other tensor in the file carries that
  // sequence, so a stack shifted by one reads a different first value.
  CHECK(Bf16At(w.layers[0].input_layernorm, 0) ==
        doctest::Approx(NormValue(0, NormTag(0, 0))));
  CHECK(Bf16At(w.layers[3].input_layernorm, 0) ==
        doctest::Approx(NormValue(0, NormTag(3, 0))));
}

// --- (3) the load, through the production entry point ------------------------

TEST_CASE("glm5_next GGUF: the production load_weights hook LOADS the file") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);
  const vllm::Glm5NextWeights& w = Open(model).weights();

  // STRUCTURE, then BYTES. The counts a stub reports as zero.
  //
  // `enumerated` counts the backbone; the file also carries the MTP block, so
  // the two differ by exactly what the loader dropped.
  CHECK(w.enumerated_tensors > 0);
  CHECK(w.accounted_tensors == w.enumerated_tensors);
  CHECK(w.enumerated_tensors + w.mtp_block_tensors_dropped ==
        static_cast<int64_t>(FileNames(g).size()));
  CHECK(w.mtp_block_tensors_dropped == 29);
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));

  // THE SCHEDULE WAS READ, NOT SYNTHESIZED. The fixture's DSA layer is at index
  // 2; `idx % 4 == 3` would put it at index 3. A synthesizing loader gets both
  // of these backwards.
  CHECK(w.layers[0].is_linear_attention);
  CHECK(w.layers[1].is_linear_attention);
  CHECK_FALSE(w.layers[2].is_linear_attention);
  CHECK(w.layers[3].is_linear_attention);
  CHECK(w.num_kda_layers() == 3);
  CHECK(w.num_dsa_layers() == 1);
  CHECK(w.layers[0].is_dense_mlp);
  CHECK_FALSE(w.layers[1].is_dense_mlp);

  // Model level.
  CHECK_FALSE(w.tied_word_embeddings);
  REQUIRE(ShapeOf(w.embed_tokens) == std::vector<int64_t>{kVocab, kH});
  CHECK(w.embed_tokens.dtype == vt::DType::kBF16);
  CHECK_FALSE(w.embed_tokens.nk);  // a GATHER, not a MatmulBT operand
  for (int64_t i : {int64_t{0}, int64_t{1}, int64_t{37}}) {
    CAPTURE(i);
    CHECK(Bf16At(w.embed_tokens, i) ==
          doctest::Approx(Rounded(1.0F + static_cast<float>(i))));
  }
  REQUIRE(ShapeOf(w.norm) == std::vector<int64_t>{kH});
  CHECK(Bf16At(w.norm, 0) == doctest::Approx(NormValue(0, 101)));
  REQUIRE(ShapeOf(w.lm_head) == std::vector<int64_t>{kVocab, kH});
  CHECK(w.lm_head.nk);
  CHECK(Bf16At(w.lm_head, 0) == doctest::Approx(Rounded(5000.0F)));
}

TEST_CASE("glm5_next GGUF: the KDA tower loads at the reference's own shapes") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const vllm::Glm5NextWeights& w = Open(model).weights();
  // REQUIRE, not CHECK: a hook that returned a default-constructed
  // `Glm5NextWeights{}` has no layers at all, and every index below would then
  // be undefined behaviour rather than a named failure. The reachability
  // mutation (delete the `LoadGlm5NextFromGguf` call site) is exactly that
  // case, and it crashed this suite before this line existed.
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));
  const int64_t L = 3;  // a KDA layer that is NOT layer 0, so the loop is real
  const vllm::Glm5NextKdaWeights& k = w.layers[L].kda;

  CHECK(ShapeOf(k.q_proj) == std::vector<int64_t>{kKdaQkv, kH});
  CHECK(ShapeOf(k.k_proj) == std::vector<int64_t>{kKdaQkv, kH});
  CHECK(ShapeOf(k.v_proj) == std::vector<int64_t>{kKdaQkv, kH});
  CHECK(ShapeOf(k.o_proj) == std::vector<int64_t>{kH, kKdaQkv});
  // The middle axis of `nn.Conv1d`'s `[C, 1, K]` is DROPPED, so the loaded
  // shape is what `Glm5NextMixedQkvConvWeight` concatenates.
  CHECK(ShapeOf(k.q_conv1d) == std::vector<int64_t>{kKdaQkv, kConvKernel});
  CHECK(ShapeOf(k.k_conv1d) == std::vector<int64_t>{kKdaQkv, kConvKernel});
  CHECK(ShapeOf(k.v_conv1d) == std::vector<int64_t>{kKdaQkv, kConvKernel});
  CHECK(ShapeOf(k.f_a_proj) == std::vector<int64_t>{kKdaHeadDim, kH});
  CHECK(ShapeOf(k.f_b_proj) == std::vector<int64_t>{kKdaQkv, kKdaHeadDim});
  CHECK(ShapeOf(k.g_a_proj) == std::vector<int64_t>{kKdaHeadDim, kH});
  CHECK(ShapeOf(k.g_b_proj) == std::vector<int64_t>{kKdaQkv, kKdaHeadDim});
  // ONE ROW PER HEAD. A port that sized this `[qkv_dim, hidden]` reads 16x too
  // many rows on this fixture and 128x on the published checkpoint.
  CHECK(ShapeOf(k.b_proj) == std::vector<int64_t>{kKdaHeads, kH});
  // `A_log` is [num_heads] and `dt_bias` is [qkv_dim] — declared one line apart
  // at DIFFERENT widths, and the pair most likely to be sized alike.
  CHECK(ShapeOf(k.a_log) == std::vector<int64_t>{kKdaHeads});
  CHECK(ShapeOf(k.dt_bias) == std::vector<int64_t>{kKdaQkv});
  CHECK(ShapeOf(k.o_norm) == std::vector<int64_t>{kKdaHeadDim});

  // THE ANNOTATED f32 EXCEPTIONS, asserted rather than described.
  CHECK(k.a_log.dtype == vt::DType::kF32);
  CHECK(k.dt_bias.dtype == vt::DType::kF32);
  CHECK(k.o_norm.dtype == vt::DType::kBF16);

  // TRANSFORM 1, and this is the assertion that separates a loader that
  // inverts `-exp` from one that copies the file's bytes through. The fixture's
  // `ssm_a[h]` is `-exp(L - 0.25h)`, so the recovered `A_log` is `L - 0.25h`,
  // which shares no value with the stored one.
  for (int64_t hd = 0; hd < kKdaHeads; ++hd) {
    CAPTURE(hd);
    const float want = static_cast<float>(L) - 0.25F * static_cast<float>(hd);
    CHECK(F32At(k.a_log, hd) == doctest::Approx(want).epsilon(1e-5));
    // Non-vacuous: the recovered value is nowhere near the stored one, so a
    // pass-through loader cannot satisfy the check above.
    CHECK(std::abs(SsmAValue(hd, L) - want) > 1.0F);
  }

  // Bytes, from the far end of the per-layer loop, so a hook that loaded only
  // the prologue is visible.
  CHECK(Bf16At(k.q_proj, 0) == doctest::Approx(Rounded(Base(L, 8))));
  CHECK(Bf16At(k.b_proj, 0) == doctest::Approx(Rounded(Base(L, 19))));
  CHECK(F32At(k.dt_bias, 0) == doctest::Approx(Base(L, 20)));
  // And the three convs are NOT the same tensor read three times.
  CHECK(Bf16At(k.q_conv1d, 0) == doctest::Approx(Rounded(Base(L, 12))));
  CHECK(Bf16At(k.k_conv1d, 0) == doctest::Approx(Rounded(Base(L, 13))));
  CHECK(Bf16At(k.v_conv1d, 0) == doctest::Approx(Rounded(Base(L, 14))));
}

TEST_CASE("glm5_next GGUF: the NoPE MLA and its k-pool indexer load") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const vllm::Glm5NextWeights& w = Open(model).weights();
  // REQUIRE, not CHECK: a hook that returned a default-constructed
  // `Glm5NextWeights{}` has no layers at all, and every index below would then
  // be undefined behaviour rather than a named failure. The reachability
  // mutation (delete the `LoadGlm5NextFromGguf` call site) is exactly that
  // case, and it crashed this suite before this line existed.
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));
  const vllm::Glm5NextMlaWeights& m = w.layers[2].mla;

  CHECK(ShapeOf(m.q_a_proj) == std::vector<int64_t>{kQLora, kH});
  CHECK(ShapeOf(m.q_a_layernorm) == std::vector<int64_t>{kQLora});
  CHECK(ShapeOf(m.q_b_proj) == std::vector<int64_t>{kHeads * kQkNope, kQLora});
  // `kv_lora_rank + qk_rope_head_dim`, and `qk_rope_head_dim` is ZERO here.
  CHECK(ShapeOf(m.kv_a_proj_with_mqa) == std::vector<int64_t>{kKvLora, kH});
  CHECK(ShapeOf(m.kv_a_layernorm) == std::vector<int64_t>{kKvLora});
  // TRANSFORM 3: two tensors, and only the k half is transposed. Both head dims
  // are 16 on this fixture, so a loader that read them at the same shape would
  // still be reading one of them wrong.
  CHECK(ShapeOf(m.k_b_proj) == std::vector<int64_t>{kHeads, kKvLora, kQkNope});
  CHECK(ShapeOf(m.v_b_proj) == std::vector<int64_t>{kHeads, kVHead, kKvLora});
  CHECK(ShapeOf(m.o_proj) == std::vector<int64_t>{kH, kHeads * kVHead});
  // The two halves carry DIFFERENT bytes, so they are not one tensor read
  // twice.
  CHECK(Bf16At(m.k_b_proj, 0) == doctest::Approx(Rounded(Base(2, 11))));
  CHECK(Bf16At(m.v_b_proj, 0) == doctest::Approx(Rounded(Base(2, 12))));

  const vllm::Glm5NextIndexerWeights& ix = m.indexer;
  CHECK(ShapeOf(ix.wq_b) == std::vector<int64_t>{kIdxHeads * kIdxHeadDim, kQLora});
  CHECK(ShapeOf(ix.wk) == std::vector<int64_t>{kIdxHeadDim, kH});
  CHECK(ShapeOf(ix.k_norm_weight) == std::vector<int64_t>{kIdxHeadDim});
  // The BIAS is what makes `k_norm` a LayerNorm and not an RMSNorm.
  CHECK(ShapeOf(ix.k_norm_bias) == std::vector<int64_t>{kIdxHeadDim});
  CHECK(Bf16At(ix.k_norm_weight, 0) == doctest::Approx(NormValue(0, NormTag(2, 5))));
  CHECK(Bf16At(ix.k_norm_bias, 0) == doctest::Approx(NormValue(0, NormTag(2, 6))));
  // ONE ROW PER INDEXER HEAD (`index_n_heads`), not per MLA head.
  CHECK(ShapeOf(ix.weights_proj) == std::vector<int64_t>{kIdxHeads, kH});
  // The k-pool stage. `index_kpool` leads, and it is 4 — the published value,
  // against a class default of 16.
  CHECK(ShapeOf(ix.kpool_ape) == std::vector<int64_t>{kKpool, kIdxHeadDim});
  CHECK(ShapeOf(ix.kpool_gate) == std::vector<int64_t>{kIdxHeadDim, kH});
}

TEST_CASE("glm5_next GGUF: the mHC pair loads at (2 + hc_mult) * hc_mult") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const vllm::Glm5NextWeights& w = Open(model).weights();
  // REQUIRE, not CHECK: a hook that returned a default-constructed
  // `Glm5NextWeights{}` has no layers at all, and every index below would then
  // be undefined behaviour rather than a named failure. The reachability
  // mutation (delete the `LoadGlm5NextFromGguf` call site) is exactly that
  // case, and it crashed this suite before this line existed.
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));
  for (int64_t L = 0; L < kLayers; ++L) {
    CAPTURE(L);
    for (int pass = 0; pass < 2; ++pass) {
      const vllm::Glm5NextMhcWeights& hc =
          pass == 0 ? w.layers[L].attn_hc : w.layers[L].mlp_hc;
      // 24, not 4 and not 16. `mix` is nonlinear in `hc_mult`, so a port that
      // used `hc_mult` or `hc_mult * hc_mult` gets a shape mismatch here and
      // not a wrong answer later.
      CHECK(ShapeOf(hc.fn) == std::vector<int64_t>{kHcMix, kStream});
      CHECK(ShapeOf(hc.base) == std::vector<int64_t>{kHcMix});
      // THREE: `pre`, `post`, `comb`, one learned gain each.
      CHECK(ShapeOf(hc.scale) == std::vector<int64_t>{3});
      // The annotated f32 exception: the Sinkhorn denominators add `hc_eps`
      // 1e-6, which a bf16 store near 1.0 would make arithmetically invisible.
      CHECK(hc.base.dtype == vt::DType::kF32);
      CHECK(hc.scale.dtype == vt::DType::kF32);
      const int64_t slot = pass == 0 ? 2 : 5;
      CHECK(Bf16At(hc.fn, 0) == doctest::Approx(Rounded(Base(L, slot))));
      CHECK(F32At(hc.base, 0) == doctest::Approx(Base(L, slot + 1)));
      CHECK(F32At(hc.scale, 0) == doctest::Approx(Base(L, slot + 2)));
    }
    // The attention site and the MLP site are DIFFERENT tensors. Reading one
    // into both is the cross-wiring this pair invites.
    CHECK(F32At(w.layers[L].attn_hc.base, 0) !=
          doctest::Approx(F32At(w.layers[L].mlp_hc.base, 0)));
  }
}

TEST_CASE("glm5_next GGUF: the stacked expert banks keep their blocks") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const vllm::Glm5NextWeights& w = Open(model).weights();
  // REQUIRE, not CHECK: a hook that returned a default-constructed
  // `Glm5NextWeights{}` has no layers at all, and every index below would then
  // be undefined behaviour rather than a named failure. The reachability
  // mutation (delete the `LoadGlm5NextFromGguf` call site) is exactly that
  // case, and it crashed this suite before this line existed.
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));
  const int64_t L = 2;  // a sparse layer that is also the DSA one
  const vllm::Glm5NextMoeWeights& moe = w.layers[L].moe;

  // The router GEMM is f32 UPSTREAM (`F.linear(hidden.type(float32),
  // self.weight.type(float32))`), so f32 here is mirroring and not widening.
  CHECK(ShapeOf(moe.router) == std::vector<int64_t>{kExperts, kH});
  CHECK(moe.router.dtype == vt::DType::kF32);
  CHECK(moe.router.nk);
  CHECK(F32At(moe.router, 0) == doctest::Approx(Base(L, 24)));
  CHECK(ShapeOf(moe.e_score_correction_bias) == std::vector<int64_t>{kExperts});
  CHECK(moe.e_score_correction_bias.dtype == vt::DType::kF32);
  CHECK(F32At(moe.e_score_correction_bias, 0) == doctest::Approx(Base(L, 25)));

  // The three banks, at [E, I, H] / [E, H, I]. The 3-D shape is what a
  // consumer slices by expert, and it survives the keep-quant flattening.
  CHECK(ShapeOf(moe.gate_exps) == std::vector<int64_t>{kExperts, kMoeI, kH});
  CHECK(ShapeOf(moe.up_exps) == std::vector<int64_t>{kExperts, kMoeI, kH});
  CHECK(ShapeOf(moe.down_exps) == std::vector<int64_t>{kExperts, kH, kMoeI});

  // RESIDENCY. Under the production policy the Q8_0 banks KEEP their blocks —
  // which is the property the whole 101.14 GiB result on the published artifact
  // rests on — and the loaded tensor is Q8_0 rather than bf16.
  const vllm::GgufLoadPolicy pol = vllm::GgufLoadPolicy::FromEnv();
  const vllm::GgufResidency r = vllm::PeekRoute(
      pol, g.Get("blk.2.ffn_gate_exps.weight"),
      vllm::GgufTensorRole::kStackedExpertWeight);
  if (r == vllm::GgufResidency::kKeepQuant) {
    CHECK(moe.gate_exps.dtype == vt::DType::kQ8_0);
    // Kept blocks are BYTES, so the check that they are the right ones is a
    // decode of the first block against what the fixture encoded.
    CHECK(moe.gate_exps.bytes.size() ==
          static_cast<size_t>(kExperts * kMoeI * (kH / 32)) * 34u);
  } else {
    CHECK(moe.gate_exps.dtype == vt::DType::kBF16);
    CHECK(Bf16At(moe.gate_exps, 0) ==
          doctest::Approx(Rounded(Q8_0ValueAt(0, kH, 10 * L + 1))));
  }

  // The shared expert is sized `moe_intermediate_size * n_shared_experts`, NOT
  // `intermediate_size` — 32 here against the dense layer's 64.
  CHECK(ShapeOf(moe.shared.gate_proj) == std::vector<int64_t>{kMoeI, kH});
  CHECK(ShapeOf(moe.shared.up_proj) == std::vector<int64_t>{kMoeI, kH});
  CHECK(ShapeOf(moe.shared.down_proj) == std::vector<int64_t>{kH, kMoeI});
  CHECK(Bf16At(moe.shared.gate_proj, 0) == doctest::Approx(Rounded(Base(L, 26))));

  // And the DENSE layer's MLP is the OTHER width, so the two are not confused.
  const vllm::Glm5NextMlpWeights& dense = w.layers[0].dense_mlp;
  CHECK(ShapeOf(dense.gate_proj) == std::vector<int64_t>{kDenseInter, kH});
  CHECK(ShapeOf(dense.down_proj) == std::vector<int64_t>{kH, kDenseInter});
  CHECK(Bf16At(dense.gate_proj, 0) == doctest::Approx(Rounded(Base(0, 21))));
}

// --- (4) refusals, by name ---------------------------------------------------

TEST_CASE("glm5_next GGUF: a missing tensor is refused BY NAME") {
  // One per module family, so a family the loader never reads would show up as
  // a load that SUCCEEDS with a tensor missing.
  for (const char* name : {"token_embd.weight", "output_norm.weight",
                           "blk.0.attn_norm.weight", "blk.0.hc_attn_fn.weight",
                           "blk.3.ssm_dt.bias", "blk.3.ssm_a",
                           "blk.2.attn_k_b.weight", "blk.2.attn_v_b.weight",
                           "blk.2.indexer.k_norm.bias",
                           "blk.2.indexer_compressor_ape.weight",
                           "blk.2.ffn_gate_exps.weight",
                           "blk.2.ffn_down_shexp.weight",
                           "blk.0.ffn_gate.weight"}) {
    CAPTURE(name);
    FixtureOpts o;
    o.drop = name;
    TempFile f(BuildFixture(o));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    // The refusal must NAME the tensor. A generic "load failed" would send the
    // reader to the wrong file.
    bool named = false;
    try {
      auto m = LoadThroughRegistry(g);
      FAIL_CHECK("loading succeeded with " << name << " missing");
    } catch (const std::exception& e) {
      named = std::string(e.what()).find(name) != std::string::npos;
      if (!named) MESSAGE("message was: " << e.what());
    }
    CHECK(named);
  }
}

TEST_CASE("glm5_next GGUF: a wrong shape is refused BY NAME") {
  for (const char* name : {"blk.0.attn_q.weight", "blk.0.ssm_beta.weight",
                           "blk.2.attn_k_b.weight",
                           "blk.2.indexer_compressor_ape.weight",
                           "blk.2.ffn_gate_exps.weight"}) {
    CAPTURE(name);
    FixtureOpts o;
    o.bad_shape = name;
    TempFile f(BuildFixture(o));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    bool named = false;
    try {
      auto m = LoadThroughRegistry(g);
      FAIL_CHECK("loading succeeded with " << name << " at a wrong shape");
    } catch (const std::exception& e) {
      named = std::string(e.what()).find(name) != std::string::npos;
      if (!named) MESSAGE("message was: " << e.what());
    }
    CHECK(named);
  }
}

TEST_CASE("glm5_next GGUF: a non-negative `ssm_a` is refused, not made NaN") {
  // The converter writes `-exp(A_log)`, so every entry is strictly negative and
  // `log(-x)` is defined. A file written WITHOUT that transform would take
  // `log` of a positive number — NaN — and poison every decay in the layer,
  // which reads downstream as a diverged sequence and never as a bad file.
  FixtureOpts o;
  o.positive_ssm_a = true;
  TempFile f(BuildFixture(o));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  bool ok = false;
  try {
    auto m = LoadThroughRegistry(g);
    FAIL_CHECK("loading succeeded with a positive `ssm_a`");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    ok = msg.find("blk.0.ssm_a") != std::string::npos &&
         msg.find("-exp(A_log)") != std::string::npos;
    if (!ok) MESSAGE("message was: " << msg);
  }
  CHECK(ok);
}

TEST_CASE("glm5_next GGUF: a TIED head is read off the FILE") {
  // llama.cpp's writer omits `output.weight` exactly when the head is tied, so
  // the file is the authority. The published artifact carries it and is untied;
  // this proves the other branch exists and is not dead.
  FixtureOpts o;
  o.tie_lm_head = true;
  TempFile f(BuildFixture(o));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  const vllm::Glm5NextWeights& w = Open(model).weights();
  CHECK(w.tied_word_embeddings);
  CHECK(w.lm_head.rank == 0);
  CHECK(w.lm_head.bytes.empty());
}

TEST_CASE("glm5_next GGUF: a GGUF source with no file is refused BY NAME") {
  // A null `gguf` reaches the hook from a caller that set the KIND without the
  // FILE. Refused by name rather than dereferenced: the alternative is a
  // segmentation fault inside a loader the reader is entitled to read as "GGUF
  // is not supported here".
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Glm5NextHfConfigFromGguf(g);
  vllm::ModelSource source = vllm::ModelSource::FromGguf(g);
  source.gguf = nullptr;
  bool named = false;
  try {
    auto m = vllm::ModelRegistry::Load(config, source);
    FAIL_CHECK("loading succeeded with a null GGUF file");
  } catch (const std::exception& e) {
    named = std::string(e.what()).find("carries no file") != std::string::npos;
    if (!named) MESSAGE("message was: " << e.what());
  }
  CHECK(named);
}

TEST_CASE("glm5_next: the safetensors arm still refuses, and says why") {
  // Every published safetensors artifact is larger than every device this
  // project owns, so the arm is deferred rather than merely unwritten, and the
  // refusal has to say which.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Glm5NextHfConfigFromGguf(g);
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(config);
  vllm::ModelSource source;
  source.kind = vllm::ModelSource::Kind::kSafetensors;
  bool ok = false;
  try {
    auto m = reg.factory->load_weights(reg, config, source);
    FAIL_CHECK("the safetensors arm loaded");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    ok = msg.find("safetensors") != std::string::npos &&
         msg.find("305.78") != std::string::npos;
    if (!ok) MESSAGE("message was: " << msg);
  }
  CHECK(ok);
}
