// SPEC-MTP-K-GT-1 (#81) — MTP speculation DEPTH, through the production loader.
//
// This is the reachability gate for the row. It builds a real engine stack
// (LoadedEngine -> EngineCore -> Scheduler -> GPUModelRunner) over a small
// synthetic Qwen3.5 DENSE model on CPU, exactly as
// tests/vllm/entrypoints/test_loaded_engine_dense.cpp does, attaches a real
// `mtp.*` draft head through the production head loader (LoadQwen3_5MTP), and
// configures depth the way a user does: through EngineParams::speculative_config.
// Nothing here constructs a proposer by hand, because a test that does proves the
// class works and never that the configured depth reaches it.
//
// ─── WHY IDENTITY IS NOT ENOUGH, AND WHAT IS ────────────────────────────────
// Greedy target plus accept-iff-equal rejection makes the emitted token sequence
// INDEPENDENT of k. That is the property that lets depth be a pure throughput
// lever, and it is also why a token-identity gate CANNOT see a silently clamped
// depth: a drafter pinned to k=1 passes every identity check at k=3. That was the
// exact defect this row closes.
//
// So every depth case pairs the identity with a POSITIVE WITNESS of depth. The
// first witness this file used was `spec_drafts_proposed_by_depth().size()`, and
// A FRESH REVIEW PROVED IT CANNOT SEE THE REGRESSION IT EXISTS TO CATCH.
//
// The mutation: make `MtpProposeDrafts` return straight after the prefill and pad
// all k columns with the step-0 draft. No decode step runs and ONE forward
// happens in total. The suite stayed green, 5/5, 34/34, `Status: SUCCESS`, with
// `compile_err=0` and the mutation demonstrably applied. The reason is
// mechanical: the per-depth vectors are grown from
// `step.num_draft_tokens_per_req[i]` (`runner.cpp`), which is the LENGTH of the
// emitted draft list, and that length is decided by the runner's slicing of
// whatever the proposer returned. Anything that hands the verify path k tokens
// satisfies it. The reviewer also probed the per-depth counters in both arms at
// k=3 and got IDENTICAL output, `proposed: 7 7 7 accepted: 0 0 0` on the real
// loop and `proposed: 7 7 7 accepted: 0 0 0` on the padded fake.
//
// Note the second half of that: ACCEPTANCE IS ZERO AT EVERY DEPTH on this
// synthetic model, in both arms. So no acceptance figure can be the witness here
// either, and the accept path at depth is UNEXERCISED by this suite. What the CPU
// tier establishes is that k drafts are PROPOSED and VERIFIED, not that a draft
// is ever accepted at depth. The owed DGX gate demands non-zero acceptance at
// every depth for exactly this reason.
//
// ─── TWO WITNESSES, BECAUSE ONE DOES NOT COVER BOTH FAILURES ────────────────
// The first is counted where the work happens: `spec_mtp_draft_decode_forwards()`,
// incremented after each draft decode forward RETURNS, against
// `spec_mtp_propose_calls()`. Upstream runs `num_speculative_steps - 1` decode
// forwards per propose, so the assertion is the exact equality
//
//     forwards == calls * (k - 1)
//
// and no draft-list shape can produce it. Under the mutation described above the
// forward count is 0 while `calls` is not, so at every k above 1 the equality
// fails and the suite goes RED.
//
// THAT EQUALITY DOES NOT SEE PADDING, and an earlier revision of this header
// claimed it did. A SECOND fresh review wrote the mutation that proves it: let
// the loop run all k-1 forwards and count them honestly, then discard what they
// sampled and pad every column with the step-0 draft. The equality holds exactly
// on that, and the suite was fully green, 5 passed / 0 failed, 47/47 assertions.
// The forwards witness answers "did the work run", never "did its results reach
// the caller", and those are different questions.
//
// The second witness answers the second question, and it is read at the CONSUMER
// rather than inside the loop: `spec_mtp_proposals_with_varied_drafts()` counts
// the propose calls whose DELIVERED row was not a pure function of its own first
// column. A padded row is exactly such a function, so the counter is 0 at every k
// under that mutation while the real loop leaves it non-zero.
//
// What NEITHER shows is per-column provenance: that column j came from forward j.
// An off-by-one in the `update_draft_inputs` column index, or a broken carry,
// still satisfies both. That is the same bound as the zero-acceptance gap above
// and it closes in the same place: acceptance AT DEPTH on real weights, which is
// what the owed DGX gate demands.
//
// PER-CALL distinctness of the k drafts was rejected and stays rejected: a
// correct drafter may legitimately repeat a token, and on this 24-entry
// vocabulary it does, a `2 2 2` row at k=3 being MEASURED here. The counter above
// is the AGGREGATE form of it, asserted over a run and never per call.
//
// RED-first for this file: before the multi-step propose, the k=3 case built an
// engine that reserved KV for 3 and drafted ONE token, so the depth witness read
// 1 against 3. RED again for the forwards witness: with a propose that returns
// straight after the prefill, `forwards` reads 0 against `calls * 2`. RED again
// for the result witness: with the padding mutation that keeps the forwards
// honest, `varied` reads 0 against a required non-zero.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/speculative.h"
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"  // W6 (#1374) dispatch counters
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

using nlohmann::json;
using vllm::DenseMlpWeights;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseWeights;
using vllm::Qwen3_5MTPKind;
using vllm::Qwen3_5MTPWeights;
using vllm::StTensor;
using vllm::TensorResolver;
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vt::DType;

namespace {

// ─── Synthetic dense weights (mirrors test_loaded_engine_dense.cpp) ──────────
uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}
OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

constexpr int kVocab = 24;      // == the tiny BPE fixture's ids 0..23, no holes.
constexpr int kMaxModelLen = 32;

HfConfig MakeDenseConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = kVocab;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = kMaxModelLen;
  // mtp_num_hidden_layers == 1, which is what both gate checkpoints ship and
  // what LoadedEngine::ResolveSpecConfig reads to resolve n_predict.
  c.raw = json::object();
  c.raw["mtp_num_hidden_layers"] = 1;
  return c;
}

DenseMlpWeights MakeMlp(const HfConfig& c, uint64_t s) {
  DenseMlpWeights m;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  m.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 1);
  m.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 2);
  m.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 3);
  return m;
}

Qwen3_5DenseWeights MakeDenseWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.mlp = MakeMlp(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// ─── The synthetic `mtp.*` draft head ───────────────────────────────────────
// Built through the PRODUCTION head loader (LoadQwen3_5MTP), not by filling
// Qwen3_5MTPWeights by hand, so the tensor names, the BF16-only rule and the
// dedicated-embedding refusal are all the ones a real checkpoint meets. The head
// is one full_attention decoder layer, which is what qwen3_5_mtp.py:105-112
// declares and what both gate checkpoints ship (mtp_num_hidden_layers == 1).
class MtpTensorStore {
 public:
  void Add(const std::string& name, std::vector<int64_t> shape, uint64_t seed) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    Stored& st = tensors_[name];
    st.values.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
      st.values[static_cast<size_t>(i)] =
          vt::F32ToBF16(RandV(seed * 977 + static_cast<uint64_t>(i)));
    st.view.dtype = "BF16";
    st.view.shape = std::move(shape);
    st.view.data = reinterpret_cast<const uint8_t*>(st.values.data());
    st.view.nbytes = st.values.size() * sizeof(uint16_t);
  }
  TensorResolver Resolver() const {
    return [this](const std::string& name) -> const StTensor& {
      return tensors_.at(name).view;
    };
  }

 private:
  struct Stored {
    std::vector<uint16_t> values;
    StTensor view;
  };
  std::map<std::string, Stored> tensors_;
};

Qwen3_5MTPWeights MakeMtpHead(const HfConfig& c) {
  static MtpTensorStore store;
  static bool filled = false;
  if (!filled) {
    const int64_t H = c.hidden_size;
    const int64_t Dh = c.head_dim;
    const int64_t q_out = 2 * c.num_attention_heads * Dh;  // gated Q projection
    const int64_t kv_out = c.num_key_value_heads * Dh;
    const int64_t I = c.intermediate_size;
    store.Add("mtp.fc.weight", {H, 2 * H}, 1);
    store.Add("mtp.pre_fc_norm_embedding.weight", {H}, 2);
    store.Add("mtp.pre_fc_norm_hidden.weight", {H}, 3);
    store.Add("mtp.layers.0.input_layernorm.weight", {H}, 4);
    store.Add("mtp.layers.0.self_attn.q_proj.weight", {q_out, H}, 5);
    store.Add("mtp.layers.0.self_attn.k_proj.weight", {kv_out, H}, 6);
    store.Add("mtp.layers.0.self_attn.v_proj.weight", {kv_out, H}, 7);
    store.Add("mtp.layers.0.self_attn.o_proj.weight",
              {H, c.num_attention_heads * Dh}, 8);
    store.Add("mtp.layers.0.self_attn.q_norm.weight", {Dh}, 9);
    store.Add("mtp.layers.0.self_attn.k_norm.weight", {Dh}, 10);
    store.Add("mtp.layers.0.post_attention_layernorm.weight", {H}, 11);
    store.Add("mtp.norm.weight", {H}, 12);
    store.Add("mtp.layers.0.mlp.gate_proj.weight", {I, H}, 13);
    store.Add("mtp.layers.0.mlp.up_proj.weight", {I, H}, 14);
    store.Add("mtp.layers.0.mlp.down_proj.weight", {H, I}, 15);
    filled = true;
  }
  return vllm::LoadQwen3_5MTP(store.Resolver(), c, Qwen3_5MTPKind::kDense);
}

// The tiny oracle-verified BPE fixture (ids 0..23, no holes).
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_mtp_depth_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  json vocab = {{"h", 0},   {"e", 1},   {"l", 2},     {"o", 3},   {"w", 4},
                {"r", 5},   {"d", 6},   {"Ġ", 7},     {"1", 8},   {"2", 9},
                {"ll", 10}, {"he", 11}, {"llo", 12},  {"hello", 13},
                {"Ġw", 14}, {"or", 15}, {"orld", 16}, {"Ġworld", 17},
                {"ld", 18}};
  vocab[MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 23;
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

SamplingParams Greedy(int max_tokens) {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax) -> deterministic and depth-neutral.
  sp.max_tokens = max_tokens;
  sp.output_kind = RequestOutputKind::kCumulative;
  return sp;
}

// The GDN state dtype the CPU tier can run a spec VERIFY in.
//
// The gate checkpoints are GDN hybrids and their state storage is bf16 (the
// model dtype), but `vt::CausalConv1dSpecUpdate` requires an f32 conv state and
// rejects bf16 off CUDA (src/vt/ops.cpp:1806). That is a property of the GDN
// speculative ROLLBACK path, which this row does not touch: the MTP head is
// itself declared layer_type="full_attention" (qwen3_5_mtp.py:105-112) and every
// draft decode step reads and writes only the draft's own paged full-attention
// KV layer. So the CPU depth gate runs the f32-state arm through the existing
// same-binary escape MakeQwen3_5KVCacheSpec already reads
// (qwen3_5_common.cpp:54-63), and the bf16 arm stays covered by the DGX gate
// this row records as owed. Set once per process, before any engine is built.
struct F32GdnState {
  F32GdnState() { setenv("VT_GDN_STATE_BF16", "0", /*overwrite=*/1); }
};
const F32GdnState kF32GdnState;

EngineParams SpecParams(int k) {
  EngineParams p;  // defaults: block_size 32 == max_model_len 32.
  p.speculative_config = vllm::ParseSpeculativeConfigJson(
      R"({"method":"mtp","num_speculative_tokens":)" + std::to_string(k) + "}");
  return p;
}

// The deepest draft the rejection sampler has ever verified on this engine. It
// is the SIZE of the per-depth counter, which the verify path grows on demand.
// It reports the LENGTH OF THE EMITTED DRAFT LIST and NOT the work the propose
// did, so it is a necessary condition and never a sufficient one: see the file
// header for the mutation it passes. 0 means nothing ever speculated.
int VerifiedDepth(const LoadedEngine& eng) {
  return static_cast<int>(eng.runner().spec_drafts_proposed_by_depth().size());
}

// WITNESS 1, THE WORK. `k - 1` draft decode forwards run per propose call,
// counted after each forward returns, so this equality holds exactly when the
// multi-step loop RAN to depth on every call and fails on any propose that
// short-circuits or clamps.
//
// It does NOT fail on a propose that pads, and an earlier revision of this
// comment said it did. The reviewer's mutation runs every forward, counts them
// honestly, then discards the sampled tokens and writes the step-0 draft into
// all k columns: this equality holds exactly on it. Witness 2 is the one that
// does not.
void CheckDraftDecodeForwards(const LoadedEngine& eng, int k) {
  const int64_t calls = eng.runner().spec_mtp_propose_calls();
  const int64_t forwards = eng.runner().spec_mtp_draft_decode_forwards();
  CAPTURE(k);
  CAPTURE(calls);
  CAPTURE(forwards);
  // A zero-call engine would make the equality below hold vacuously at every k,
  // which is the "a gate that cannot say how many things it examined has not
  // reported" shape. Require the work to have happened first.
  REQUIRE(calls > 0);
  CHECK(forwards == calls * static_cast<int64_t>(k - 1));
}

// WITNESS 2, THE RESULT. Read at the CONSUMER, on the array the propose handed
// the runner: how many calls delivered a row that was not a pure function of its
// own first column. A padded propose leaves this 0 at every k, because every
// column of a padded row IS its first column, and no amount of honest forward
// counting changes that.
//
// The bound, stated because the assertion is easy to over-read. `varied > 0`
// says the delivered array carries information the step-0 draft alone does not
// determine. It does NOT say column j came from forward j, so an off-by-one in
// the `update_draft_inputs` column index or a broken carry still passes. A non-zero
// acceptance count AT DEPTH does not prove it either, here or on real weights: a
// padded row `t0 t0 ...` is accepted at column 1 exactly when the target's greedy
// continuation repeats `t0`, which is a property of the target rather than of the
// drafter. What separates them is an acceptance-RATE comparison against a PADDED
// CONTROL, so per-column provenance stays owed to the DGX gate in that shape, exactly
// as the zero-acceptance gap below does.
//
// The assertion is an AGGREGATE over the run and never a per-call one, because a
// correct drafter may resample the same token: a `2 2 2` row at k=3 is MEASURED
// on this fixture. Measured over a whole run it never happens on every call, and
// the two buckets are captured below so a report says how many of how many.
void CheckDraftsCarryDepth(const LoadedEngine& eng, int k) {
  const int64_t calls = eng.runner().spec_mtp_propose_calls();
  const int64_t varied = eng.runner().spec_mtp_proposals_with_varied_drafts();
  CAPTURE(k);
  CAPTURE(calls);
  CAPTURE(varied);
  CAPTURE(calls - varied);  // the constant-row bucket. The two SUM to calls.
  REQUIRE(calls > 0);
  CHECK(varied <= calls);
  if (k == 1) {
    // A one-column row has nothing to differ from, so the counter is 0 BY
    // CONSTRUCTION here. Asserted rather than skipped, because a counter that
    // fired at k=1 would be counting something other than depth.
    CHECK(varied == 0);
  } else {
    CHECK(varied > 0);
  }
}

}  // namespace

TEST_CASE("mtp depth: a configured depth of 3 actually drafts 3") {
  const HfConfig c = MakeDenseConfig();
  const std::string prompt = "hello";
  const int kN = 8;

  LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), SpecParams(3),
                   MakeMtpHead(c));
  const RequestOutput out = eng.engine().generate(prompt, Greedy(kN), "req");

  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  CHECK(static_cast<int>(out.outputs[0].token_ids.size()) == kN);

  // THE WITNESSES. Two draft decode forwards ran per propose call, which only a
  // loop that went to depth 3 can produce, AND what those forwards sampled
  // reached the array the runner was handed. Every list-shape assertion below
  // passes on a propose that satisfies neither.
  CheckDraftDecodeForwards(eng, 3);
  CheckDraftsCarryDepth(eng, 3);

  // The list-shape assertions, kept as NECESSARY conditions. They say the verify
  // path received three drafts per request. They do not say three were drafted,
  // which is why the forward count above leads.
  const std::vector<int64_t>& by_depth =
      eng.runner().spec_drafts_proposed_by_depth();
  REQUIRE(VerifiedDepth(eng) == 3);
  CHECK(by_depth[0] > 0);
  CHECK(by_depth[1] > 0);
  CHECK(by_depth[2] > 0);
  // Every request drafts the full depth every step, so the three depths are
  // proposed EQUALLY often. A drafter that ran out of steps early would show a
  // decreasing profile here even with a non-zero deepest count.
  CHECK(by_depth[0] == by_depth[1]);
  CHECK(by_depth[1] == by_depth[2]);

  // The aggregate must agree with the split, or one of the two is lying.
  int64_t summed = 0;
  for (int64_t n : by_depth) summed += n;
  CHECK(summed == eng.runner().spec_drafts_proposed());
  int64_t summed_accepted = 0;
  for (int64_t n : eng.runner().spec_drafts_accepted_by_depth())
    summed_accepted += n;
  CHECK(summed_accepted == eng.runner().spec_drafts_accepted());
  // Acceptance is a prefix, so it can never grow with depth. MEASURED HERE: this
  // profile is `0 0 0`, on the real loop and on the padded mutation alike, so
  // these two assertions are VACUOUS and the accept path at depth is unexercised
  // by the CPU tier. Kept because the invariant is the right one to state, and
  // recorded because it bounds what this suite proves: k drafts are proposed and
  // verified, not accepted. The owed DGX gate demands non-zero acceptance at
  // every depth on real weights, which is the only place that can be shown.
  const std::vector<int64_t>& acc =
      eng.runner().spec_drafts_accepted_by_depth();
  REQUIRE(acc.size() == 3);
  CHECK(acc[0] >= acc[1]);
  CHECK(acc[1] >= acc[2]);
  CHECK(eng.runner().spec_drafts_accepted() <=
        eng.runner().spec_drafts_proposed());
}

TEST_CASE("mtp depth: k=1, k=3 and spec-OFF emit the SAME greedy tokens") {
  // Greedy target plus accept-iff-equal rejection makes the emitted sequence
  // independent of k, so depth is a pure throughput lever that cannot move a
  // token gate. The equality is exact, not a tolerance.
  //
  // On its own this passes on a drafter clamped to k=1, which is why each arm
  // also asserts the depth it actually reached. Identity plus witness is the
  // pair; either alone proves the wrong thing.
  const HfConfig c = MakeDenseConfig();
  const std::string prompt = "hello world";
  const int kN = 10;

  std::vector<int32_t> off_ids;
  std::vector<int32_t> k1_ids;
  std::vector<int32_t> k3_ids;
  {
    EngineParams p;  // no speculative_config: the production default.
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), p);
    off_ids = eng.engine().generate(prompt, Greedy(kN), "req")
                  .outputs[0].token_ids;
    // Nothing speculated, so no propose ran and no counter was ever grown.
    CHECK(VerifiedDepth(eng) == 0);
    CHECK(eng.runner().spec_mtp_propose_calls() == 0);
    CHECK(eng.runner().spec_mtp_draft_decode_forwards() == 0);
    CHECK(eng.runner().spec_mtp_proposals_with_varied_drafts() == 0);
  }
  {
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), SpecParams(1),
                     MakeMtpHead(c));
    k1_ids = eng.engine().generate(prompt, Greedy(kN), "req")
                 .outputs[0].token_ids;
    CHECK(VerifiedDepth(eng) == 1);
    // k=1 is the early exit: the propose ran, and it ran NO decode forward. The
    // same equality as every other arm, at k-1 == 0, and the same result witness
    // at its own k=1 value of 0.
    CheckDraftDecodeForwards(eng, 1);
    CheckDraftsCarryDepth(eng, 1);
  }
  {
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), SpecParams(3),
                     MakeMtpHead(c));
    k3_ids = eng.engine().generate(prompt, Greedy(kN), "req")
                 .outputs[0].token_ids;
    CHECK(VerifiedDepth(eng) == 3);
    CheckDraftDecodeForwards(eng, 3);
    CheckDraftsCarryDepth(eng, 3);
  }

  REQUIRE(static_cast<int>(off_ids.size()) == kN);
  CHECK(k1_ids == off_ids);
  CHECK(k3_ids == off_ids);
}

TEST_CASE("mtp depth: k=2 and k=4 both reach their configured depth") {
  // Two more depths, because a loop that runs "more than once" and a loop that
  // runs exactly k times are different implementations and only the second is
  // configurable depth. #81's M1 asks for k=2..4.
  const HfConfig c = MakeDenseConfig();
  const std::string prompt = "hello";

  for (const int k : {2, 4}) {
    CAPTURE(k);
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), SpecParams(k),
                     MakeMtpHead(c));
    const RequestOutput out = eng.engine().generate("hello", Greedy(8), "req");
    REQUIRE(out.finished);
    CHECK(static_cast<int>(out.outputs[0].token_ids.size()) == 8);
    CHECK(VerifiedDepth(eng) == k);
    // k-1 forwards per call at k=2 and at k=4, which is what separates "the loop
    // ran more than once" from "the loop ran exactly k times", plus the result
    // witness at both depths.
    CheckDraftDecodeForwards(eng, k);
    CheckDraftsCarryDepth(eng, k);
  }
}

TEST_CASE("mtp depth: a depth engine still refuses nothing it can serve") {
  // The Phase 1 refusal existed only between commits. A configured depth above 1
  // now CONSTRUCTS, which is the observable difference between "removed" and
  // "widened": a widened bound would still throw somewhere above it.
  const HfConfig c = MakeDenseConfig();
  for (const int k : {1, 2, 3, 4, 8}) {
    CAPTURE(k);
    CHECK_NOTHROW(LoadedEngine(c, MakeDenseWeights(c), BuildFixture(),
                               SpecParams(k), MakeMtpHead(c)));
  }
}

TEST_CASE("mtp depth: the ngram proposer is unaffected by the MTP depth work") {
  // The n-gram proposer returns 0..k drafts per request per step through the SAME
  // DraftTokenIds seam the MTP propose now fills with k tokens
  // (runner.cpp:2273-2282). Its depth must keep working across that change.
  const HfConfig c = MakeDenseConfig();
  EngineParams p;
  p.speculative_config = vllm::ParseSpeculativeConfigJson(
      R"({"method":"ngram","num_speculative_tokens":3})");
  LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), p);
  const RequestOutput out = eng.engine().generate("hello", Greedy(4), "req");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  CHECK(static_cast<int>(out.outputs[0].token_ids.size()) == 4);
}

// ───────────────────────────────────────────────────────────────────────────
// ENG-CUDAGRAPH-BREAK W6 (#1374): THE GRAPH-ELIGIBILITY PREDICATE, reached from
// the production engine. Closes the predicate half of
// [#1020](https://github.com/mudler/vllm.cpp/issues/1020), which this row owned
// under `## Owed` until W6 took it.
//
// WHY IT LIVES HERE AND NOT IN ITS OWN FILE. The predicate runs in
// `GPUModelRunner::execute_model`, so seeing it work needs a real engine driving
// real speculative steps -- LoadedEngine, EngineCore, Scheduler, runner -- which
// is exactly the stack this file already builds. A second copy of that fixture
// would be the "two copies of a harness diverge invisibly" shape the sibling
// decode-graph gates were consolidated to avoid, and a test that called
// `ActualUniformDecodeQueryLen` directly would prove the arithmetic and never
// that a step reaches it.
//
// WHAT #1020 IS, IN ONE SENTENCE. The predicate compared the batch's uniform
// query length against `1 + num_speculative_tokens`, the CONFIGURED width, so a
// step the scheduler had clamped to a shorter -- but still perfectly uniform --
// draft prefix missed it and ran its verify eager, with no log and no counter.
//
// WHY `clamped_spec_steps` AND NOT `uniform_spec_steps`. The total moves on an
// unclamped engine too, so it cannot witness the widening: it is above zero
// before and after. Only the strictly-between-1-and-`1 + k` bucket is the
// population the old predicate refused, and it is zero unless something actually
// clamped. MEASURED on this fixture at `max_model_len == 32`: the context tail
// clamps the last verify steps, and the bucket reads 0 at k=1, 0 at k=2, 1 at
// k=3, 2 at k=4 and 4 at k=6. The k<=2 zeros are not a gap, they are the
// control: at k=1 there is no length strictly between 1 and 2 for a clamp to
// land on, so a counter that fired there would be counting something else.
TEST_CASE("W6: a CLAMPED spec verify is graph-eligible at its actual depth") {
  const HfConfig c = MakeDenseConfig();

  // k=1 and k=2: the control. Nothing can be clamped into the open interval.
  for (const int k : {1, 2}) {
    CAPTURE(k);
    vllm::v1::ResetGraphDispatchStats();
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), SpecParams(k),
                     MakeMtpHead(c));
    const RequestOutput out = eng.engine().generate("hello", Greedy(29), "req");
    REQUIRE(out.finished);
    const vllm::v1::GraphDispatchStats st = vllm::v1::GetGraphDispatchStats();
    CAPTURE(st.uniform_steps);
    CAPTURE(st.uniform_spec_steps);
    CAPTURE(st.clamped_spec_steps);
    // The engine ran and the predicate saw it. A zero here would make every
    // assertion below hold vacuously, which is the shape a broken instrument
    // reports as a pass.
    REQUIRE(st.uniform_steps > 0);
    CHECK(st.uniform_spec_steps > 0);
    CHECK(st.clamped_spec_steps == 0);
  }

  // k >= 3: the context tail clamps, and those steps are now eligible. This is
  // the assertion the mutation moves.
  for (const int k : {3, 4, 6}) {
    CAPTURE(k);
    vllm::v1::ResetGraphDispatchStats();
    LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), SpecParams(k),
                     MakeMtpHead(c));
    const RequestOutput out = eng.engine().generate("hello", Greedy(29), "req");
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    CHECK(static_cast<int>(out.outputs[0].token_ids.size()) == 29);
    const vllm::v1::GraphDispatchStats st = vllm::v1::GetGraphDispatchStats();
    CAPTURE(st.uniform_steps);
    CAPTURE(st.uniform_spec_steps);
    CAPTURE(st.clamped_spec_steps);
    REQUIRE(st.uniform_steps > 0);
    CHECK(st.clamped_spec_steps > 0);
    // A clamped step is a subset of the speculative ones, or one of the two
    // counters is counting the wrong population.
    CHECK(st.clamped_spec_steps < st.uniform_spec_steps);
    CHECK(st.uniform_spec_steps <= st.uniform_steps);
  }
}

// The conjunct that keeps a PREFILL out of the widened arm, and it is not
// hypothetical: without it this fixture reported 27 "uniform spec" steps out of
// 28 on a 29-token run, because a single request prefilling n tokens is uniform
// at query length n by every arithmetic test vLLM applies. The runner
// additionally requires that every request in the step is verifying at exactly
// `q - 1` drafts, read off the scheduler's own per-request counts.
//
// A spec-OFF engine is the cleanest statement of it: `num_spec()` is 0, so
// `1 + k` is 1 and nothing above 1 can be admitted at all.
TEST_CASE("W6: a spec-OFF engine admits query length 1 and nothing else") {
  const HfConfig c = MakeDenseConfig();
  vllm::v1::ResetGraphDispatchStats();
  LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), EngineParams{});
  const RequestOutput out = eng.engine().generate("hello", Greedy(8), "req");
  REQUIRE(out.finished);
  const vllm::v1::GraphDispatchStats st = vllm::v1::GetGraphDispatchStats();
  CAPTURE(st.uniform_steps);
  CAPTURE(st.uniform_spec_steps);
  CAPTURE(st.ragged_steps);
  REQUIRE(st.uniform_steps > 0);
  CHECK(st.uniform_spec_steps == 0);
  CHECK(st.clamped_spec_steps == 0);
  // No driver on this CPU platform captures, so nothing opened a ring. Asserted
  // rather than left unsaid: a non-zero here would mean a CPU step reached a
  // capture path, which `support_static_graph_mode()` is supposed to refuse.
  CHECK(st.capture_shapes == 0);
}

// A MULTI-TOKEN PREFILL ON A SPECULATING ENGINE. "hello world" is two tokens in
// this fixture's BPE, so at k=3 the prefill step is uniform at query length 2 by
// arithmetic, inside the `1 + k` bound, and strictly below the configured 4 --
// the shape a bare uniformity test would put in the CLAMPED bucket and hand to a
// decode capture.
//
// WHICH CONJUNCT ACTUALLY REFUSES IT HERE, because the two are easy to confuse
// and only measurement separates them. On this GDN hybrid it is the FIRST one:
// the model has linear-attention layers, so `gdn_group_id_ >= 0` and a prefill
// step carries `gdn_meta.num_prefill_tokens > 0`. The per-request draft conjunct
// inside `GraphEligibleQueryLen` is therefore REDUNDANT on every model that
// reads `uniform_query_len` today, and a mutation deleting it left this suite
// GREEN at 104/104 -- which is why that conjunct is gated in
// `test_cudagraph_dispatch.cpp` on the function itself, and why the spec records
// it as defence in depth for the next model rather than as something measured
// here. Recorded rather than claimed away.
//
// One request generating ONE token is one step, and that step is the prefill.
// So every speculative bucket must be empty and the step must be reported as
// serving no captured shape at all.
TEST_CASE("W6: a multi-token PREFILL is not a verify shape") {
  const HfConfig c = MakeDenseConfig();
  vllm::v1::ResetGraphDispatchStats();
  LoadedEngine eng(c, MakeDenseWeights(c), BuildFixture(), SpecParams(3),
                   MakeMtpHead(c));
  const RequestOutput out = eng.engine().generate("hello world", Greedy(1), "req");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  CHECK(static_cast<int>(out.outputs[0].token_ids.size()) == 1);
  const vllm::v1::GraphDispatchStats st = vllm::v1::GetGraphDispatchStats();
  CAPTURE(st.uniform_steps);
  CAPTURE(st.uniform_spec_steps);
  CAPTURE(st.clamped_spec_steps);
  CAPTURE(st.ragged_steps);
  // A step ran, so nothing below is vacuous.
  REQUIRE(st.uniform_steps + st.ragged_steps > 0);
  CHECK(st.uniform_spec_steps == 0);
  CHECK(st.clamped_spec_steps == 0);
  // And it was reported as serving no captured shape, which is the positive
  // half: "not admitted" has to be OBSERVABLE or the two counters above are
  // satisfied by a step that was never counted at all.
  CHECK(st.ragged_steps >= 1);
}
