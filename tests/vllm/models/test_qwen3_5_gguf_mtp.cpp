// SPEC-MTP-GGUF: the MTP head read back out of a head-carrying GGUF.
//
// TWO ARMS, and the split is the point (issue #1454). The trunk-depth
// ARITHMETIC — llama.cpp's converter folds the head blocks into `block_count`,
// so the decoder depth is `block_count - nextn_predict_layers` — is pinned
// HERMETICALLY here, on KV-only synthetic files carrying no weight bytes, so
// CI checks it on every run. The head's WEIGHT layout, which no synthetic
// fixture can honestly stand in for, stays gated on VLLM_MTP_GGUF_MODEL
// pointing at a Qwen3.5/3.6 GGUF converted WITH the head (i.e. NOT --no-mtp),
// because the whole point of that gate is to hold the loader against a REAL
// file produced by llama.cpp's converter rather than against our own
// assumptions about its naming.
//
// Before #1454 there was only the second arm, and both of its cases opened with
// a bare `return` on the unset variable. A bare `return` from a doctest case is
// a PASS: with the variable unset this file reported `test cases: 2 | 2 passed`,
// `assertions: 0`, `Status: SUCCESS!`, exit 0 — which is every CI run of this
// repository, because the variable is set nowhere in `.github/workflows/`. The
// skips now print a MESSAGE naming the variable, as
// `tests/vllm/entrypoints/test_gguf_mmproj_reach.cpp` does.
//
// Reference file used to develop this (llama.cpp Qwen3.5-2B, Q8_0 body):
//   qwen35.block_count            = 25
//   qwen35.nextn_predict_layers   = 1        => trunk L = 24, head at blk.24
//   blk.24.nextn.eh_proj.weight   [2048, 4096]  (torch [N, K])
//   blk.24.nextn.enorm.weight     [2048]
//   blk.24.nextn.hnorm.weight     [2048]
//   blk.24.nextn.shared_head_norm.weight [2048]
//   blk.24.{attn_*,ffn_*,attn_norm,post_attention_norm}   the head's own block
// Note there is NO blk.24.nextn.embed_tokens / .shared_head_head: Qwen3.5's
// head SHARES the target's embedding and lm_head, so the converter emits
// neither, and the loader must not ask for them.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/transformers_utils/hf_config.h"

namespace {

const char* MtpGgufPath() { return std::getenv("VLLM_MTP_GGUF_MODEL"); }

// Reads one integer metadata value out of a GGUF. The variant alternatives in
// GgufValue are listed in wire type-id order, so the tag IS the GGUF type id.
// Kept here rather than reused from the loader because the loader's reader is
// in an anonymous namespace, and because a case that re-derives `block_count`
// through the same helper the production path uses cannot disagree with it.
int64_t KvInt64(const vllm::GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case 0: return std::get<uint8_t>(v.v);
    case 1: return std::get<int8_t>(v.v);
    case 2: return std::get<uint16_t>(v.v);
    case 3: return std::get<int16_t>(v.v);
    case 4: return std::get<uint32_t>(v.v);
    case 5: return std::get<int32_t>(v.v);
    case 10: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case 11: return std::get<int64_t>(v.v);
    default: throw std::runtime_error("kv is not an integer: " + key);
  }
}

int64_t RequiredKvInt(const vllm::GgufFile& g, const std::string& key) {
  const vllm::GgufValue* v = g.FindKv(key);
  if (v == nullptr) throw std::runtime_error("missing metadata key " + key);
  return KvInt64(*v, key);
}

// A KV-ONLY GGUF — no tensors, no weight bytes — carrying exactly the scalars
// `HfConfigFromGguf` requires, so the PRODUCTION config builder is what derives
// the depth here rather than this test restating its arithmetic. `vocab_size`
// is supplied as a kv because the alternative is `token_embd.weight`'s shape,
// and a hermetic fixture has no such tensor. `nextn < 0` omits the
// nextn_predict_layers key entirely, which is the head-less export.
std::string BuildTrunkDepthGguf(int64_t block_count, int64_t nextn) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "qwen35"));
  b.AddKv(gguf_test::U32Kv("qwen35.embedding_length", 64));
  b.AddKv(gguf_test::U32Kv("qwen35.block_count",
                           static_cast<uint32_t>(block_count)));
  if (nextn >= 0) {
    b.AddKv(gguf_test::U32Kv("qwen35.nextn_predict_layers",
                             static_cast<uint32_t>(nextn)));
  }
  b.AddKv(gguf_test::U32Kv("qwen35.attention.head_count", 4));
  b.AddKv(gguf_test::U32Kv("qwen35.attention.head_count_kv", 2));
  b.AddKv(gguf_test::U32Kv("qwen35.attention.key_length", 16));
  b.AddKv(gguf_test::U32Kv("qwen35.vocab_size", 128));
  b.AddKv(gguf_test::F32Kv("qwen35.attention.layer_norm_rms_epsilon", 1e-6F));
  return b.Build();
}

}  // namespace

// THE INVARIANT, hermetic. `num_hidden_layers + mtp_num_hidden_layers ==
// block_count`, checked on files this test writes, so it runs in CI with no
// asset. Until #1454 the only statement of this rule in this file was a comment
// above `CHECK(c.num_hidden_layers > 0)` — true of every valid model — so a
// regression that spent the whole of `block_count` on the trunk passed.
//
// Both the 1-block and the 3-block head are checked, because `- 1` satisfies
// the first alone and is a different (wrong) rule.
TEST_CASE("gguf mtp: the trunk depth EXCLUDES the head blocks") {
  struct Arm {
    int64_t block_count;
    int64_t nextn;
  };
  // 65/1 is Qwen3.8-27B-Q4_K_M's own pair; 25/1 is the Qwen3.5-2B reference
  // file this suite was developed against; 28/3 separates `- nextn` from `- 1`.
  const Arm arms[] = {{65, 1}, {25, 1}, {28, 3}};

  for (const Arm& a : arms) {
    CAPTURE(a.block_count);
    CAPTURE(a.nextn);
    gguf_test::TempFile f(BuildTrunkDepthGguf(a.block_count, a.nextn));
    const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
    const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

    // The depth key is what ResolveSpecConfig reads (via NumMtpLayers). Before
    // this row HfConfigFromGguf spent nextn_predict_layers on the trunk layer
    // count and discarded it, so a head-carrying GGUF looked head-less to the
    // spec resolver and silently fell back to depth 1.
    REQUIRE(c.raw.contains("mtp_num_hidden_layers"));
    CHECK(vllm::NumMtpLayers(c) == a.nextn);
    CHECK(c.num_hidden_layers == a.block_count - a.nextn);
    CHECK(c.num_hidden_layers + vllm::NumMtpLayers(c) == a.block_count);
    // A trunk of the right SIZE made of the wrong layers is the same defect one
    // level down: layer_types covers the trunk only, never the head block.
    CHECK(static_cast<int64_t>(c.layer_types.size()) ==
          a.block_count - a.nextn);
  }
}

// The head-less export, which is the OTHER half of the same arithmetic and the
// reason the invariant cannot be written with NumMtpLayers alone: that helper
// answers 1 for an absent key (`qwen3_5_mtp.cpp`, "how deep is the head we are
// running", not "does one exist"). With no `nextn_predict_layers` the whole of
// block_count is the trunk and NOTHING is published, which is what
// `model_loader`'s head-less-GGUF refusal reads.
TEST_CASE("gguf mtp: a head-less GGUF publishes no depth and keeps every block") {
  gguf_test::TempFile f(BuildTrunkDepthGguf(24, -1));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  CHECK_FALSE(c.raw.contains("mtp_num_hidden_layers"));
  CHECK(c.num_hidden_layers == 24);
  CHECK(static_cast<int64_t>(c.layer_types.size()) == 24);
}

TEST_CASE("gguf mtp: the head depth reaches config.raw") {
  const char* path = MtpGgufPath();
  if (path == nullptr) {
    MESSAGE(
        "SKIPPED: set VLLM_MTP_GGUF_MODEL to a Qwen3.5/3.6 GGUF converted WITH "
        "the MTP head (llama.cpp convert_hf_to_gguf.py, NOT --no-mtp) to run "
        "this against a real converter output; docs/USAGE.md pins the files. "
        "The arithmetic this case re-derives is also gated hermetically above, "
        "so an unset run is not an unchecked one");
    return;
  }

  vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  REQUIRE(c.raw.contains("mtp_num_hidden_layers"));
  const int64_t depth = vllm::NumMtpLayers(c);
  CHECK(depth > 0);

  // The trunk count must EXCLUDE the head blocks, re-derived from the file's
  // OWN block_count rather than from a number written here — the same rule the
  // hermetic case above pins, now against a real converter's bytes.
  const int64_t block_count =
      RequiredKvInt(g, c.model_type + ".block_count");
  CHECK(c.num_hidden_layers + depth == block_count);
  CHECK(c.num_hidden_layers > 0);
}

TEST_CASE("gguf mtp: the head loads with trunk-consistent conventions") {
  const char* path = MtpGgufPath();
  if (path == nullptr) {
    MESSAGE(
        "SKIPPED: set VLLM_MTP_GGUF_MODEL to a Qwen3.5/3.6 GGUF converted WITH "
        "the MTP head (llama.cpp convert_hf_to_gguf.py, NOT --no-mtp) to load "
        "the head and check its weight layout; docs/USAGE.md pins the files. "
        "No synthetic fixture stands in for this one: it is about what the "
        "converter really emits");
    return;
  }

  vllm::GgufFile g = vllm::GgufFile::Open(path);
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  REQUIRE(c.raw.contains("mtp_num_hidden_layers"));

  const vllm::Qwen3_5MTPKind kind =
      c.num_experts > 0 ? vllm::Qwen3_5MTPKind::kMoe
                        : vllm::Qwen3_5MTPKind::kDense;
  const vllm::Qwen3_5MTPWeights w = vllm::LoadQwen3_5MTPFromGguf(
      g, c, kind, vllm::GgufLoadPolicy::FromEnv());

  const int64_t H = c.hidden_size;

  // fc is the [embedding; hidden] -> hidden projection, kept in raw [N, K]
  // exactly as the safetensors path's LoadBf16RawNK leaves it. GGUF stores
  // shapes in torch [N, K] order already, so this is the VERBATIM path; loading
  // it through the transposing helper would silently produce [2H, H] and only
  // fail much later, inside the draft forward.
  REQUIRE(w.fc.rank == 2);
  CHECK(w.fc.shape[0] == H);
  CHECK(w.fc.shape[1] == 2 * H);
  // The ORIENTATION FLAG, not just the shape. The draft forward requires
  // `fc.nk` set ("fc must be raw bf16 [H,2H]"); GGUF already stores [N, K] so
  // the shape assertions above pass either way, and an unset flag fails only
  // later inside the forward. The G4 token gate caught exactly that, which is
  // why it is asserted here too.
  CHECK(w.fc.nk);

  // The three RMSNorm weights are [H]. They also carry the GGUF (w + 1) storage
  // convention, which is why the loader routes them through the same
  // OwnNormMinus1 helper the trunk uses; a plain read would leave every norm
  // weight off by one and poison every draft proposal.
  REQUIRE(w.pre_fc_norm_embedding.rank == 1);
  CHECK(w.pre_fc_norm_embedding.shape[0] == H);
  REQUIRE(w.pre_fc_norm_hidden.rank == 1);
  CHECK(w.pre_fc_norm_hidden.shape[0] == H);
  REQUIRE(w.final_norm.rank == 1);
  CHECK(w.final_norm.shape[0] == H);

  // One transformer block per head layer, and it is ALWAYS full attention: the
  // head is not in config.layer_types (which covers the trunk only), so a
  // loader that consulted it would run off the end or mis-type the block.
  CHECK(w.NumLayers() == vllm::NumMtpLayers(c));
  if (kind == vllm::Qwen3_5MTPKind::kDense) {
    REQUIRE(!w.dense_layers.empty());
    CHECK(!w.dense_layers[0].is_linear_attention);
    CHECK(w.dense_layers[0].input_layernorm.shape[0] == H);
    CHECK(w.dense_layers[0].mlp.gate_proj.rank == 2);
  } else {
    REQUIRE(!w.moe_layers.empty());
    CHECK(!w.moe_layers[0].is_linear_attention);
    CHECK(w.moe_layers[0].input_layernorm.shape[0] == H);
  }
}
