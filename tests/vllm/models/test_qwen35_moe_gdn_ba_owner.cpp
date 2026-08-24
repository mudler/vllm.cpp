// GDN-MOE-PACKED-BA (#1169): the MoE safetensors loader builds the ONE merged
// `in_proj_ba` owner, exactly as the dense loader does.
//
// vLLM owns one physical `in_proj_ba` on every Qwen3.5/3.6 GDN layer, dense and
// MoE alike: `packed_modules_mapping["in_proj_ba"] = ["in_proj_b", "in_proj_a"]`
// sits on `Qwen3_5ForCausalLMBase` (vllm/model_executor/models/qwen3_5.py:297 @
// 555967922), which both `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`
// derive from, and the stacked mapping puts `in_proj_b` at shard 0 and
// `in_proj_a` at shard 1 (:217-218), so the merged row order is [b; a].
//
// Locally the consumer `ProjectGdnBA` (qwen3_5.cpp) and the packed-decode
// eligibility `ShouldUsePackedGdnDecode` are already general: a populated
// `in_proj_ba` is what makes `has_packed_ba` true. At the base SHA the MoE
// loader (`LoadGdn`, qwen3_5_weights.cpp) still loaded the two shards split,
// so the merged owner was written at ONE site in the tree, the dense loader,
// and packed GDN decode was unreachable on every MoE checkpoint.
//
// This suite enters through `vllm::LoadQwen3_5MoeLayer`, the production
// per-layer loader entry (`LoadQwen3_5Moe` calls the same `LoadLayerImpl`),
// over a synthetic in-memory resolver that serves the full tensor set of one
// MoE `linear_attention` layer. It asserts the owner's shape, dtype,
// orientation, row order and byte content, and that the split fields stay
// empty so nothing retains duplicate bytes.
//
// RED at the base: `gdn.in_proj_ba` is empty. The reviewer's mutation restores
// the split pair in `LoadGdn`; this suite must go red again.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vt/dtype.h"

namespace {

// Small shapes that keep the case well under a second while exercising every
// tensor family `LoadGdn` + `LoadMoe` resolve for a linear-attention layer.
constexpr int64_t kHidden = 64;
constexpr int64_t kKeyHeads = 2;
constexpr int64_t kKeyHeadDim = 8;
constexpr int64_t kValueHeads = 4;  // Hv
constexpr int64_t kValueHeadDim = 8;
constexpr int64_t kConvK = 4;
constexpr int64_t kExperts = 2;
constexpr int64_t kInter = 16;
constexpr int64_t kSharedInter = 16;
constexpr int64_t kLayer = 0;
const char* const kPrefix = "model.language_model.";

// An in-memory safetensors "shard": every tensor is a BF16 byte buffer this
// fixture owns, handed out through the same `TensorResolver` signature the
// real `SafetensorsFile`-backed resolver has. `mapping` stays null (the
// header documents that as the test shape), so no loader path can BORROW these
// bytes and the copy helpers always materialize an owned mirror.
class InMemoryShard {
 public:
  // Fill pattern: every bf16 element of `name` is `Pattern(name_seed, i)`,
  // deterministic so the test can recompute the expected bytes without reading
  // them back from a buffer the loader may have MADV_DONTNEED'd (the windowed
  // page release runs on any whole page inside a consumed source range).
  void Add(const std::string& name, std::vector<int64_t> shape,
           uint16_t seed) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::vector<uint8_t> bytes(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = Pattern(seed, i);
    Entry& e = tensors_[name];
    e.bytes = std::move(bytes);
    e.tensor.dtype = "BF16";
    e.tensor.shape = std::move(shape);
    // `data` points at the buffer the map node OWNS; std::map nodes are
    // address-stable, so the pointer stays valid for the fixture's lifetime.
    e.tensor.data = e.bytes.data();
    e.tensor.nbytes = e.bytes.size();
  }

  static uint16_t Pattern(uint16_t seed, int64_t i) {
    // Avoid NaN/Inf bf16 patterns so a dequant path, if any, stays finite; the
    // loader compares nothing numerically here, but the bytes are distinct per
    // (seed, index) which is what the row-order assertion needs.
    const uint64_t mixed =
        static_cast<uint64_t>(seed) * 131u + static_cast<uint64_t>(i) * 7u;
    return static_cast<uint16_t>(0x3C00u ^ (mixed & 0x03FFu));
  }

  vllm::TensorResolver Resolver() const {
    return [this](const std::string& name) -> const vllm::StTensor& {
      const auto it = tensors_.find(name);
      if (it == tensors_.end()) {
        throw std::runtime_error("tensor not found: " + name);
      }
      return it->second.tensor;
    };
  }

  size_t Count() const { return tensors_.size(); }

 private:
  struct Entry {
    std::vector<uint8_t> bytes;
    vllm::StTensor tensor;
  };
  std::map<std::string, Entry> tensors_;
};

// The complete request set of `LoadLayerImpl` for a `linear_attention` layer on
// the all-bf16 tower (`MoeProjDtype::kBf16` everywhere, stacked bf16 experts):
// exactly the names `PlanQwen3_5MoeLoad` plans for such a layer.
InMemoryShard MakeGdnLayerShard(uint16_t b_seed, uint16_t a_seed) {
  const std::string base = std::string(kPrefix) + "layers." +
                           std::to_string(kLayer) + ".";
  const std::string la = base + "linear_attn.";
  const std::string mlp = base + "mlp.";
  const int64_t qkv = 2 * kKeyHeads * kKeyHeadDim + kValueHeads * kValueHeadDim;
  const int64_t v_dim = kValueHeads * kValueHeadDim;
  InMemoryShard s;
  s.Add(base + "input_layernorm.weight", {kHidden}, 1);
  s.Add(base + "post_attention_layernorm.weight", {kHidden}, 2);
  s.Add(la + "in_proj_qkv.weight", {qkv, kHidden}, 3);
  s.Add(la + "in_proj_z.weight", {v_dim, kHidden}, 4);
  s.Add(la + "out_proj.weight", {kHidden, v_dim}, 5);
  s.Add(la + "in_proj_b.weight", {kValueHeads, kHidden}, b_seed);
  s.Add(la + "in_proj_a.weight", {kValueHeads, kHidden}, a_seed);
  s.Add(la + "conv1d.weight", {qkv, 1, kConvK}, 6);
  s.Add(la + "A_log", {kValueHeads}, 7);
  s.Add(la + "dt_bias", {kValueHeads}, 8);
  s.Add(la + "norm.weight", {kValueHeadDim}, 9);
  s.Add(mlp + "gate.weight", {kExperts, kHidden}, 10);
  s.Add(mlp + "shared_expert_gate.weight", {1, kHidden}, 11);
  s.Add(mlp + "experts.gate_up_proj", {kExperts, 2 * kInter, kHidden}, 12);
  s.Add(mlp + "experts.down_proj", {kExperts, kHidden, kInter}, 13);
  s.Add(mlp + "shared_expert.gate_proj.weight", {kSharedInter, kHidden}, 14);
  s.Add(mlp + "shared_expert.up_proj.weight", {kSharedInter, kHidden}, 15);
  s.Add(mlp + "shared_expert.down_proj.weight", {kHidden, kSharedInter}, 16);
  return s;
}

vllm::Qwen3_5MoeTowerDtypes Bf16Tower() {
  vllm::Qwen3_5MoeTowerDtypes t;
  t.gdn = vllm::MoeProjDtype::kBf16;
  t.attn = vllm::MoeProjDtype::kBf16;
  t.shared_expert = vllm::MoeProjDtype::kBf16;
  t.lm_head = vllm::MoeProjDtype::kBf16;
  return t;
}

}  // namespace

TEST_CASE("qwen35 MoE loader: LoadGdn builds the one merged in_proj_ba owner, rows [b; a], split fields empty") {
  constexpr uint16_t kBSeed = 0x51;
  constexpr uint16_t kASeed = 0xA3;
  const InMemoryShard shard = MakeGdnLayerShard(kBSeed, kASeed);
  REQUIRE(shard.Count() == 18u);

  const vllm::Qwen3_5MoeLayerWeights layer = vllm::LoadQwen3_5MoeLayer(
      shard.Resolver(), "linear_attention", kLayer, kExperts, kPrefix,
      vllm::MoeExpertLayout::kStackedBf16, kHidden, Bf16Tower());
  REQUIRE(layer.is_linear_attention);
  const vllm::GdnLayerWeights& gdn = layer.gdn;

  // THE OWNER. This is the assertion that is red at the base SHA: the MoE
  // loader wrote the split pair and left `in_proj_ba` empty, so `has_packed_ba`
  // was false on every MoE checkpoint (#1169).
  REQUIRE_FALSE(gdn.in_proj_ba.Empty());
  CHECK(gdn.in_proj_ba.dtype == vt::DType::kBF16);
  REQUIRE(gdn.in_proj_ba.rank == 2);
  CHECK(gdn.in_proj_ba.shape[0] == 2 * kValueHeads);
  CHECK(gdn.in_proj_ba.shape[1] == kHidden);
  // nk=true: raw torch-Linear [N,K] orientation for vt::MatmulBT, which is what
  // `ProjectGdnBA` VT_CHECKs before it slices the owner.
  CHECK(gdn.in_proj_ba.nk);
  REQUIRE(gdn.in_proj_ba.HasHostBytes());
  REQUIRE(gdn.in_proj_ba.bytes.size() ==
          static_cast<size_t>(2 * kValueHeads * kHidden) * 2);

  // ROW ORDER AND BYTES. The first Hv rows are the `in_proj_b` source bytes
  // verbatim, the next Hv rows the `in_proj_a` source bytes verbatim. The
  // expectation is recomputed from the fill pattern, not read back from the
  // served buffer (see InMemoryShard::Add).
  const auto* owner = reinterpret_cast<const uint16_t*>(gdn.in_proj_ba.bytes.data());
  const int64_t half = kValueHeads * kHidden;
  int64_t b_mismatch = 0, a_mismatch = 0;
  for (int64_t i = 0; i < half; ++i) {
    if (owner[i] != InMemoryShard::Pattern(kBSeed, i)) ++b_mismatch;
    if (owner[half + i] != InMemoryShard::Pattern(kASeed, i)) ++a_mismatch;
  }
  CHECK(b_mismatch == 0);
  CHECK(a_mismatch == 0);
  // ...and the two halves are distinguishable, so a swapped [a; b] order would
  // have been caught above rather than passing by coincidence.
  CHECK(InMemoryShard::Pattern(kBSeed, 0) != InMemoryShard::Pattern(kASeed, 0));

  // THE SPLIT FIELDS STAY EMPTY. The consumer slices the owner where it needs
  // the legacy pair; retaining both would double the resident bytes and give
  // `ResidentWeight` two things to upload. This mirrors the dense loader's
  // one-owner invariant (test_qwen35_plain_weights asserts the same on the 4B).
  CHECK(gdn.in_proj_b.Empty());
  CHECK(gdn.in_proj_a.Empty());

  // The rest of the layer still loaded the way it did before: the owner change
  // is independent of the tower branch and touches no other field.
  CHECK_FALSE(gdn.in_proj_qkv.Empty());
  CHECK_FALSE(gdn.in_proj_z.Empty());
  CHECK_FALSE(gdn.out_proj.Empty());
  CHECK_FALSE(gdn.conv1d_weight.Empty());
  CHECK(gdn.a_log.dtype == vt::DType::kF32);
  CHECK(gdn.dt_bias.dtype == vt::DType::kF32);
  CHECK_FALSE(gdn.norm_weight.Empty());
  CHECK(gdn.in_proj_qkvz.Empty());  // out of scope for this row (#1169 note)
}
