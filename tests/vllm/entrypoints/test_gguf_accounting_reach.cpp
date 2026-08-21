// QUANT-QWEN38-27B-GGUF-ARM (issue #821, W2) — the REACHABILITY gate for the
// tensor accounting.
//
// `test_qwen38_27b_gguf_manifest.cpp` gates the two enumerations by calling
// them. That proves the enumerations are right and proves nothing about
// anything reaching them, which is the shape `.agents/reachability.md` exists
// for: an accounting function nobody calls is a class, not a capability, and a
// checkpoint with unread tensors still loads silently.
//
// So this file drives `LoadedEngine::FromModelDir` — the entry point the C ABI,
// the OpenAI server and the CLI all arrive through — and asserts the thrown
// MESSAGE.
//
// HOW THE PERMITTING CASES ARE MEANINGFUL. The synthetic language GGUF carries
// no tokenizer, so a load that gets PAST the accounting dies at
// `tokenizer: GGUF missing kv "tokenizer.ggml.model"` — the step after it.
// Asserting that later message POSITIVELY is the point: a case that only looked
// for the absence of a refusal would also pass if the loader had died earlier
// for an unrelated reason.
//
// THE MUTATION THIS FILE ANSWERS TO. Delete the `RefuseUnaccountedQwen3_5Gguf`
// call in `model_loader.cpp` and the stray-tensor case below stops seeing its
// refusal and sees the tokenizer error instead, which is red here. Same for
// `RefuseUnaccountedClipMmproj` and the projector case.
#include <doctest/doctest.h>

#include <string>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/gguf_builder.h"
#include "vllm/models/clip_mmproj_fixture.h"

namespace {

using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

// A synthetic `qwen35` DENSE language GGUF: exactly the hparams
// `HfConfigFromGguf` and `ModelRegistry::Resolve` need, and no tokenizer, so
// the load stops one step after the accounting. Same shape as
// `test_gguf_mmproj_reach.cpp`'s fixture, which measured that stopping point.
//
// `block_count = 5` with `full_attention_interval = 4` makes layers 0,1,2,4 GDN
// and layer 3 full attention — the same alternation the 27B carries at 65x the
// depth, so an enumeration that got the layer kinds backwards is visible here.
struct LanguageOptions {
  // How many TRAILING blocks carry the drafter shape: a full-attention block
  // plus the four `nextn.*` tensors. This is a property of the TENSOR TABLE.
  uint32_t nextn_tensors = 0;
  // Whether the file DECLARES `qwen35.nextn_predict_layers`. Separate from the
  // shape on purpose: the defect this row exists for is a file that ships the
  // drafter and a reader that does not know it, and the only way to build that
  // pair is to move the two independently.
  bool declare_nextn = false;
  // A tensor no loader reads.
  std::string stray;
};

std::string BuildLanguageGguf(const LanguageOptions& o = LanguageOptions{}) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35"));
  b.AddKv(U32Kv("qwen35.embedding_length", 64));
  b.AddKv(U32Kv("qwen35.block_count", 5));
  if (o.declare_nextn) {
    b.AddKv(U32Kv("qwen35.nextn_predict_layers", o.nextn_tensors));
  }
  b.AddKv(U32Kv("qwen35.attention.head_count", 4));
  b.AddKv(U32Kv("qwen35.attention.head_count_kv", 2));
  b.AddKv(U32Kv("qwen35.attention.key_length", 16));
  b.AddKv(U32Kv("qwen35.feed_forward_length", 32));
  b.AddKv(U32Kv("qwen35.ssm.group_count", 2));
  b.AddKv(U32Kv("qwen35.ssm.time_step_rank", 4));
  b.AddKv(U32Kv("qwen35.ssm.state_size", 8));
  b.AddKv(U32Kv("qwen35.ssm.conv_kernel", 4));
  b.AddKv(U32Kv("qwen35.full_attention_interval", 4));
  b.AddKv(U32Kv("qwen35.context_length", 256));
  b.AddKv(gguf_test::F32Kv("qwen35.rope.freq_base", 1000000.0F));
  b.AddKv(gguf_test::F32Kv("qwen35.attention.layer_norm_rms_epsilon", 1e-6F));
  // Only the names matter to the accounting, so every tensor is a 1-element
  // F32: this fixture never reaches a weight read.
  auto tiny = [&b](const std::string& name) {
    b.AddTensor(name, {1}, /*ggml_type=*/0, std::string(4, '\0'));
  };
  // token_embd is 2-D because HfConfigFromGguf reads the vocabulary off it.
  b.AddTensor("token_embd.weight", {64, 64}, /*ggml_type=*/0,
              std::string(4096 * 4, '\0'));
  tiny("output.weight");
  tiny("output_norm.weight");
  const uint32_t blocks = 5;
  for (uint32_t il = 0; il < blocks; ++il) {
    // The MTP block is always full attention, and so is every 4th trunk layer.
    const bool mtp = o.nextn_tensors > 0 && il >= blocks - o.nextn_tensors;
    const bool full = mtp || ((il + 1) % 4 == 0);
    const std::string p = "blk." + std::to_string(il) + ".";
    tiny(p + "attn_norm.weight");
    tiny(p + "post_attention_norm.weight");
    if (full) {
      for (const char* n : {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                            "attn_output.weight", "attn_q_norm.weight",
                            "attn_k_norm.weight"}) {
        tiny(p + n);
      }
    } else {
      for (const char* n : {"attn_qkv.weight", "attn_gate.weight",
                            "ssm_beta.weight", "ssm_alpha.weight",
                            "ssm_conv1d.weight", "ssm_out.weight", "ssm_a",
                            "ssm_dt.bias", "ssm_norm.weight"}) {
        tiny(p + n);
      }
    }
    for (const char* n : {"ffn_gate.weight", "ffn_up.weight",
                          "ffn_down.weight"}) {
      tiny(p + n);
    }
    if (mtp) {
      for (const char* n : {"nextn.eh_proj.weight", "nextn.enorm.weight",
                            "nextn.hnorm.weight",
                            "nextn.shared_head_norm.weight"}) {
        tiny(p + n);
      }
    }
  }
  if (!o.stray.empty()) tiny(o.stray);
  return b.Build();
}

// What `FromModelDir` threw, as text. A message is the only thing that can say
// WHICH step refused.
std::string Load(const std::string& model_path,
                 const std::string& mmproj_path = std::string()) {
  vllm::entrypoints::EngineParams params;
  params.mmproj_path = mmproj_path;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(model_path, params);
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

constexpr const char* kTokenizerStop = "tokenizer: GGUF missing kv";

}  // namespace

TEST_CASE("accounting reach: a fully-accounted GGUF is NOT refused") {
  TempFile model(BuildLanguageGguf());
  const std::string message = Load(model.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // Past the accounting: the throw is the NEXT step's.
  CHECK(message.find(kTokenizerStop) != std::string::npos);
  CHECK(message.find("NOTHING in this loader reads") == std::string::npos);
}

TEST_CASE("accounting reach: a tensor no loader reads is refused BY NAME") {
  TempFile model(BuildLanguageGguf(
      {/*nextn_tensors=*/0, /*declare_nextn=*/false,
       "blk.0.ffn_gate_exps.weight"}));
  const std::string message = Load(model.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("blk.0.ffn_gate_exps.weight") != std::string::npos);
  CHECK(message.find("NOTHING in this loader reads") != std::string::npos);
  // And it fired BEFORE the tokenizer, therefore before any weight I/O: on the
  // real artifact that is a message instead of a 17 GB map.
  CHECK(message.find(kTokenizerStop) == std::string::npos);
}

// THE ROW'S OWN DEFECT, at the loader. `Qwen3.8-27B-Q4_K_M.gguf` ships
// `block_count = 65` with `nextn_predict_layers = 1`. This fixture is the same
// shape at depth 5: drop the `nextn_predict_layers` key and the last block stops
// being the drafter and becomes trunk layer 4 — a GDN layer by the interval
// rule, so its six full-attention projections and its four `nextn.*` tensors
// become tensors nothing reads, and its nine GDN tensors become tensors the file
// does not have.
//
// Both halves of that are the failure, and only ONE of them has ever had a
// detector: the missing side already refuses at `GgufFile::Get`, deep in the
// weight load, after the tokenizer. The unaccounted side had none, which is why
// the loader checks it here and why a fluent, wrong 65-layer model was the
// realistic outcome.
TEST_CASE("accounting reach: the MTP block, read as a decoder layer, is caught") {
  // The file the artifact IS: five blocks, the last one the drafter, and the
  // key that says so.
  TempFile drafter(
      BuildLanguageGguf({/*nextn_tensors=*/1, /*declare_nextn=*/true, ""}));
  const std::string ok = Load(drafter.path());
  CAPTURE(ok);
  CHECK(ok.find(kTokenizerStop) != std::string::npos);

  // The same tensor table, with the key that says so REMOVED — which is exactly
  // what a loader that ignored `nextn_predict_layers` would see.
  TempFile misread(
      BuildLanguageGguf({/*nextn_tensors=*/1, /*declare_nextn=*/false, ""}));
  const std::string message = Load(misread.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("NOTHING in this loader reads") != std::string::npos);
  CHECK(message.find("blk.4.nextn.eh_proj.weight") != std::string::npos);
  CHECK(message.find(kTokenizerStop) == std::string::npos);
}

TEST_CASE("accounting reach: a projector with an unread tensor is refused") {
  TempFile model(BuildLanguageGguf());
  clip_fixture::Options o;
  o.stray_tensor = "v.blk.0.ffn_gate.weight";
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}, o));

  const std::string message = Load(model.path(), mmproj.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("v.blk.0.ffn_gate.weight") != std::string::npos);
  CHECK(message.find("NEVER reads") != std::string::npos);
  CHECK(message.find(kTokenizerStop) == std::string::npos);
}

TEST_CASE("accounting reach: a fully-accounted projector is NOT refused") {
  TempFile model(BuildLanguageGguf());
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}));
  const std::string message = Load(model.path(), mmproj.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find(kTokenizerStop) != std::string::npos);
  CHECK(message.find("NEVER reads") == std::string::npos);
}

TEST_CASE("accounting reach: a DeepStack projector is accounted, not refused") {
  TempFile model(BuildLanguageGguf());
  clip_fixture::Options o;
  o.deepstack_layer = 1;
  TempFile mmproj(clip_fixture::Build(clip_fixture::Dims{}, o));
  const std::string message = Load(model.path(), mmproj.path());
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // The six DeepStack tensors are READ, so they must not read as unaccounted:
  // an enumeration blind to the tap would refuse a file the reader handles.
  CHECK(message.find(kTokenizerStop) != std::string::npos);
  CHECK(message.find("NEVER reads") == std::string::npos);
}
