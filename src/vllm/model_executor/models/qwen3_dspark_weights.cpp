// DSpark draft weight loader (SPEC-DSPARK W3). Ported from
// Qwen3DSparkForCausalLM.load_weights (qwen3_dspark.py:149-185 @ 555967922) plus
// the shared DFlash backbone loader it inherits.
//
// The DSpark-specific tensors are exactly three: markov_head.markov_w1,
// markov_head.markov_w2 and (reduced draft vocab only) d2t. Everything else is
// the DFlash backbone, loaded by the landed LoadQwen3DFlash unchanged.
#include <cstring>
#include <string>
#include <vector>

#include <unordered_map>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_dspark.h"
#include "vt/dtype.h"
#include "vt/unaligned.h"

namespace vllm {
namespace {

// Resolve `name`, then the "model."-prefixed spelling upstream's loader writes
// (qwen3_dspark.py:157-159 prepends "model." to every non-lm_head/d2t tensor).
const StTensor* TryGet(const TensorResolver& get, const std::string& name) {
  try {
    return &get(name);
  } catch (const std::runtime_error&) {
  }
  try {
    return &get("model." + name);
  } catch (const std::runtime_error&) {
    return nullptr;
  }
}

OwnedTensor LoadBf16(const StTensor& tensor, const std::string& name, bool nk) {
  VT_CHECK(tensor.dtype == "BF16", "qwen3_dspark: expected BF16 for " + name);
  OwnedTensor out;
  out.dtype = vt::DType::kBF16;
  out.rank = static_cast<int>(tensor.shape.size());
  VT_CHECK(out.rank <= vt::kMaxRank, "qwen3_dspark: rank exceeds kMaxRank for " + name);
  int64_t numel = 1;
  for (int i = 0; i < out.rank; ++i) {
    out.shape[i] = tensor.shape[static_cast<size_t>(i)];
    numel *= out.shape[i];
  }
  out.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  VT_CHECK(tensor.nbytes == out.bytes.size(),
           "qwen3_dspark: byte-size mismatch for " + name);
  std::memcpy(out.bytes.data(), tensor.data, tensor.nbytes);
  out.nk = nk;
  return out;
}

}  // namespace

Qwen3DSparkWeights LoadQwen3DSpark(const TensorResolver& get, const HfConfig& config,
                                   int64_t num_taps, int32_t mask_token_id) {
  Qwen3DSparkWeights w;
  // markov_rank / vocab / draft_vocab / logit_scale from the (already
  // native-shaped) draft config.
  Qwen3DSparkModel::ResolveDsparkDims(config, w);

  // The inherited DFlashQwen3Model backbone, unchanged
  // (Qwen3DSparkModel(DFlashQwen3Model), qwen3_dspark.py:70-92).
  w.backbone = LoadQwen3DFlash(get, config, num_taps, mask_token_id);
  w.backbone.draft_vocab_size = w.draft_vocab_size;

  // markov_w1: the [vocab_size, markov_rank] embedding of the previously sampled
  // TARGET token (a VocabParallelEmbedding, so NOT an nk Linear).
  const StTensor* w1 = TryGet(get, "markov_head.markov_w1.weight");
  VT_CHECK(w1 != nullptr,
           "qwen3_dspark: markov_head.markov_w1.weight missing — not a DSpark "
           "draft checkpoint");
  w.markov_w1 = LoadBf16(*w1, "markov_head.markov_w1.weight", /*nk=*/false);
  VT_CHECK(w.markov_w1.rank == 2 && w.markov_w1.shape[0] == w.vocab_size &&
               w.markov_w1.shape[1] == w.markov_rank,
           "qwen3_dspark: markov_w1 must be [vocab_size, markov_rank]");

  // markov_w2: the [draft_vocab_size, markov_rank] ParallelLMHead applied through
  // the logits processor, i.e. an nk Linear like lm_head.
  const StTensor* w2 = TryGet(get, "markov_head.markov_w2.weight");
  VT_CHECK(w2 != nullptr,
           "qwen3_dspark: markov_head.markov_w2.weight missing — not a DSpark "
           "draft checkpoint");
  w.markov_w2 = LoadBf16(*w2, "markov_head.markov_w2.weight", /*nk=*/true);
  VT_CHECK(w.markov_w2.rank == 2 && w.markov_w2.shape[0] == w.draft_vocab_size &&
               w.markov_w2.shape[1] == w.markov_rank,
           "qwen3_dspark: markov_w2 must be [draft_vocab_size, markov_rank] — the "
           "checkpoint's draft vocab disagrees with the config");

  // d2t -> draft_id_to_target_id (qwen3_dspark.py:154-156). Present exactly when
  // the draft vocab is reduced; I64 on disk, one OFFSET per draft id. `t2d` is
  // training-only and deliberately skipped (:152-153).
  const StTensor* d2t = TryGet(get, "d2t");
  if (d2t != nullptr) {
    VT_CHECK(d2t->dtype == "I64", "qwen3_dspark: expected I64 for d2t");
    VT_CHECK(d2t->shape.size() == 1 && d2t->shape[0] == w.draft_vocab_size,
             "qwen3_dspark: d2t must be [draft_vocab_size]");
    // Unaligned: `d2t->data` is an arbitrary byte offset into the mmap and I64
    // wants 8-byte alignment, so a `const int64_t*` onto it is undefined to form
    // or load through (issue #627).
    const uint8_t* src = d2t->data;
    w.draft_id_to_target_id.resize(static_cast<size_t>(w.draft_vocab_size));
    for (int64_t i = 0; i < w.draft_vocab_size; ++i) {
      const int64_t off = vt::LoadUnaligned<int64_t>(src + i * 8);
      const int64_t target = i + off;
      VT_CHECK(target >= 0 && target < w.vocab_size,
               "qwen3_dspark: d2t maps a draft id outside the target vocab");
      w.draft_id_to_target_id[static_cast<size_t>(i)] = static_cast<int32_t>(off);
    }
    // Gather markov_w1's rows into DRAFT order once, here, so the sequential
    // sampler can index it with the raw device argmax and never round-trip the
    // draft->target map through the host mid-chain. Pure data movement: row j of
    // the result IS row (j + d2t[j]) of markov_w1.
    {
      const int64_t R = w.markov_rank;
      VT_CHECK(w.markov_w1.rank == 2 && w.markov_w1.shape[0] == w.vocab_size &&
                   w.markov_w1.shape[1] == R,
               "qwen3_dspark: markov_w1 must be [vocab_size, markov_rank] before "
               "the draft-order gather");
      OwnedTensor t;
      t.dtype = vt::DType::kBF16;
      t.rank = 2;
      t.shape[0] = w.draft_vocab_size;
      t.shape[1] = R;
      t.nk = w.markov_w1.nk;
      t.bytes.resize(static_cast<size_t>(w.draft_vocab_size) *
                     static_cast<size_t>(R) * sizeof(uint16_t));
      const auto* src_rows = reinterpret_cast<const uint16_t*>(w.markov_w1.bytes.data());
      auto* dst_rows = reinterpret_cast<uint16_t*>(t.bytes.data());
      for (int64_t j = 0; j < w.draft_vocab_size; ++j) {
        const int64_t target =
            j + w.draft_id_to_target_id[static_cast<size_t>(j)];
        std::memcpy(dst_rows + static_cast<size_t>(j * R),
                    src_rows + static_cast<size_t>(target * R),
                    static_cast<size_t>(R) * sizeof(uint16_t));
      }
      w.markov_w1_draft = std::move(t);
    }
  } else {
    VT_CHECK(w.draft_vocab_size == w.vocab_size,
             "qwen3_dspark: a reduced draft vocab needs the d2t remap table");
  }
  return w;
}

Qwen3DSparkWeights LoadQwen3DSpark(const std::vector<SafetensorsFile>& shards,
                                   const HfConfig& config, int64_t num_taps,
                                   int32_t mask_token_id) {
  // The same bare-then-"model."-prefixed resolver the DFlash shards overload
  // builds (qwen3_dflash_weights.cpp:191-210).
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;
  const TensorResolver get = [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    std::string key = name;
    if (it == where.end()) {
      key = "model." + name;
      it = where.find(key);
    }
    VT_CHECK(it != where.end(), "qwen3_dspark: tensor not found: " + name);
    return it->second->Get(key);
  };
  return LoadQwen3DSpark(get, config, num_taps, mask_token_id);
}

}  // namespace vllm
