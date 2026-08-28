// Qwen4-Exp W5c (#2031) — the KV-cache spec, read through the PRODUCTION hook.
//
// Every case here enters through `ModelRegistry::Resolve(config).factory->
// make_kv_cache`, which is the entry point `LoadedEngine::MakeKVCacheMaybeSpec`
// calls (`src/vllm/entrypoints/model_loader.cpp:1465` via
// `ModelRegistry::MakeKVCache`). Constructing the spec structs by hand would
// prove that `MambaSpec` works, never that `qwen4_exp` publishes one — the
// distinction AGENTS.md "Nothing lands dead" draws.
//
// ORACLE. vLLM registers no `qwen4_exp` at any revision, so the grouping shape
// is mirrored from vLLM's GENERAL recurrent contract at the parity pin
// `5559679229bc961848b121ccdeaa8fa5d79bec98` rather than from a `qwen4_exp`
// implementation, and every anchor named in a comment below was read there:
//   * `vllm/model_executor/models/interfaces.py:809-812` —
//     `get_mamba_state_shape_from_config(cls, vllm_config)`, a CLASSMETHOD with
//     no `layer_idx`. 18 implementations at the pin; not one takes a layer.
//   * `vllm/v1/worker/mamba_utils.py:441` — `assert all(mamba_specs[0] == spec
//     for spec in mamba_specs)`: every recurrent spec in the model is EQUAL.
//   * `vllm/v1/core/kv_cache_utils.py:1101-1109` — a smaller `MambaSpec` is
//     PADDED to the max page size, not split into its own group.
//   * `vllm/v1/kv_cache_interface.py:386` and `:393-395` — `compress_ratio` and
//     `storage_block_size = block_size // compress_ratio`, an integer division.
//   * `:424-435` — `MLAAttentionSpec.merge` asserts ONE `compress_ratio` per
//     group.
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

using vllm::HfConfig;
using vllm::LoadHfConfig;
using vllm::ModelRegistry;
using vllm::v1::FullAttentionSpec;
using vllm::v1::KVCacheConfig;
using vllm::v1::KVCacheSpecKind;
using vllm::v1::MambaSpec;
using vllm::v1::MLAAttentionSpec;
using vt::DType;

namespace {

const char* FixtureDir() {
#ifdef QWEN4_EXP_CKPT_FIXTURE_DIR
  return QWEN4_EXP_CKPT_FIXTURE_DIR;
#else
  return "tests/vllm/models/fixtures/qwen4_exp";
#endif
}

std::filesystem::path UniqueTempDir(const std::string& stem) {
  static const std::string kToken = [] {
    std::random_device rd;
    std::ostringstream os;
    os << std::hex << rd() << "_"
       << std::chrono::steady_clock::now().time_since_epoch().count();
    return os.str();
  }();
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         (stem + kToken + "_" + std::to_string(counter++));
}

class TempConfig {
 public:
  explicit TempConfig(const nlohmann::json& doc) {
    dir_ = UniqueTempDir("qwen4_exp_kv_");
    std::filesystem::create_directories(dir_);
    std::ofstream(dir_ / "config.json") << doc.dump();
  }
  ~TempConfig() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string path() const { return (dir_ / "config.json").string(); }

 private:
  std::filesystem::path dir_;
};

nlohmann::json FixtureDoc() {
  std::ifstream in(std::string(FixtureDir()) + "/config.json");
  REQUIRE_MESSAGE(in.good(), "fixture config.json missing under " << FixtureDir());
  nlohmann::json doc;
  in >> doc;
  return doc;
}

// THE PRODUCTION HOOK AND NOTHING ELSE.
KVCacheConfig MakeThroughRegistry(const nlohmann::json& doc, int block_size,
                                  int num_blocks) {
  TempConfig cfg(doc);
  const HfConfig config = LoadHfConfig(cfg.path());
  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->make_kv_cache != nullptr);
  return reg.factory->make_kv_cache(config, block_size, num_blocks);
}

// The same hook, reporting what it threw. "" means it did not throw, so a
// silent acceptance fails the same substring check a wrong message does.
std::string ThrowText(const nlohmann::json& doc, int block_size,
                      int num_blocks) {
  try {
    (void)MakeThroughRegistry(doc, block_size, num_blocks);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

// ─── 1. THREE groups, real per-layer names, and the shapes that follow ───────

TEST_CASE("qwen4_exp: the KV spec publishes THREE groups over REAL layer names") {
  const KVCacheConfig kv = MakeThroughRegistry(FixtureDoc(), 16, 8);
  CHECK(kv.num_blocks == 8);
  REQUIRE(kv.kv_cache_groups.size() == 3);
  // The bf16 default. Named rather than assumed, so a changed default is a red
  // REQUIRE and not silently different arithmetic below.
  REQUIRE(vllm::v1::ResolveKvCacheDType() == DType::kBF16);

  SUBCASE("group 0: the 12 QSA layers' paged K+V") {
    const auto& g = kv.kv_cache_groups[0];
    REQUIRE(g.kv_cache_spec != nullptr);
    CHECK(g.kv_cache_spec->kind() == KVCacheSpecKind::kFullAttention);
    // 48 layers on a 3:1 linear:sparse schedule, so the sparse layers are
    // 3, 7, ... 47.
    REQUIRE(g.layer_names.size() == 12);
    CHECK(g.layer_names.front() == "model.layers.3.self_attn.attn");
    CHECK(g.layer_names.back() == "model.layers.47.self_attn.attn");
    const auto* spec = dynamic_cast<const FullAttentionSpec*>(g.kv_cache_spec.get());
    REQUIRE(spec != nullptr);
    CHECK(spec->num_kv_heads == 2);
    CHECK(spec->head_size == 256);
    CHECK(spec->dtype == DType::kBF16);
    // block * Hkv * (Dh + Dh_v) * 2 = 16 * 2 * 512 * 2.
    CHECK(spec->page_size_bytes() == 32768);
  }

  SUBCASE("group 1: ONE uniform recurrent group over all 36 linear layers") {
    const auto& g = kv.kv_cache_groups[1];
    REQUIRE(g.kv_cache_spec != nullptr);
    CHECK(g.kv_cache_spec->kind() == KVCacheSpecKind::kMamba);
    REQUIRE(g.layer_names.size() == 36);
    CHECK(g.layer_names.front() == "model.layers.0.linear_attn");
    CHECK(g.layer_names.back() == "model.layers.46.linear_attn");
    const auto* spec = dynamic_cast<const MambaSpec*>(g.kv_cache_spec.get());
    REQUIRE(spec != nullptr);

    // FOUR states, in the order `GdnStateCache` reads them:
    // [gdn_conv, temporal, ple_conv, ngram]. Slots 0 and 1 are load-bearing
    // NAMES (`conv_state` / `ssm_state`), which is why the temporal state sits
    // between the three conv states rather than after them as upstream's
    // `number_of_conv_states = 3` ordering would put it.
    REQUIRE(spec->shapes.size() == 4);
    REQUIRE(spec->dtypes.size() == 4);
    // conv_dim = 2 * (16 * 128) + 48 * 128 = 10240; taps = kernel - 1 = 3.
    CHECK(spec->shapes[0] == std::vector<int64_t>{10240, 3});
    CHECK(spec->shapes[1] == std::vector<int64_t>{48, 128, 128});
    // The PLE conv is DILATED by ngram_size, so (4 - 1) * 3 = 9 columns deep,
    // over the FULL hyper-connection stream width 4 * 2560 = 10240.
    CHECK(spec->shapes[2] == std::vector<int64_t>{10240, 9});
    // The n-gram token history: ngram_size - 1 = 2 ids.
    CHECK(spec->shapes[3] == std::vector<int64_t>{2});

    CHECK(spec->dtypes[0] == DType::kBF16);
    // The fixture states `mamba_ssm_dtype: float32`, resolved by the SAME
    // helper every other hybrid uses.
    CHECK(spec->dtypes[1] == DType::kF32);
    CHECK(spec->dtypes[2] == DType::kBF16);
    // TOKEN IDS. `kI64` is the point of ENG-RECURRENT-MULTISTATE's dtype
    // widening (#2131): a float state would round a token id, and the id feeds
    // a uint64 hash multiply where a rounded value diverges in silence.
    CHECK(spec->dtypes[3] == DType::kI64);

    // 10240*3*2 + 48*128*128*4 + 10240*9*2 + 2*8
    CHECK(spec->page_size_bytes() == 61440 + 3145728 + 184320 + 16);
    CHECK(spec->page_size_bytes() == 3391504);
  }

  SUBCASE("group 2: the QSA indexer side cache, one key per FOUR tokens") {
    const auto& g = kv.kv_cache_groups[2];
    REQUIRE(g.kv_cache_spec != nullptr);
    // MLA, not full attention. A `FullAttentionSpec` third group is absorbed by
    // the runner's leftover scan as the single `fa_draft` draft-KV slot, so
    // `multi_cache_topology` stays false and this cache gets no buffer at all
    // — published and silently unallocated.
    CHECK(g.kv_cache_spec->kind() == KVCacheSpecKind::kMlaAttention);
    REQUIRE(g.layer_names.size() == 12);
    CHECK(g.layer_names.front() ==
          "model.layers.3.self_attn.indexer.k_cache");
    CHECK(g.layer_names.back() ==
          "model.layers.47.self_attn.indexer.k_cache");
    const auto* spec = dynamic_cast<const MLAAttentionSpec*>(g.kv_cache_spec.get());
    REQUIRE(spec != nullptr);
    CHECK(spec->num_kv_heads == 1);
    CHECK(spec->head_size == 128);
    CHECK(spec->compress_ratio == 4);
    CHECK(spec->storage_block_size() == 4);
    // storage_block * 1 * 128 * 2. NO factor 2 for a V that does not exist:
    // 64 B per token per layer, a quarter of a per-token index cache.
    CHECK(spec->page_size_bytes() == 1024);
    CHECK(spec->page_size_bytes() / 16 == 64);
  }

  SUBCASE("every published name resolves to a distinct in-range layer index") {
    // The runner reads group membership BY NAME and refuses a group whose names
    // "do not all resolve to distinct in-range layer indices"
    // (`gpu/runner.cpp`, the multi-cache admission check). A placeholder name
    // resolves to nullopt, which is exactly what this asserts against.
    std::set<int64_t> attn_layers;
    std::set<int64_t> recurrent_layers;
    for (size_t g = 0; g < kv.kv_cache_groups.size(); ++g) {
      CAPTURE(g);
      const auto& group = kv.kv_cache_groups[g];
      std::set<int64_t> seen;
      for (const std::string& name : group.layer_names) {
        CAPTURE(name);
        const auto l = vllm::v1::KVCacheLayerIndexOfName(name);
        REQUIRE(l.has_value());
        CHECK(*l >= 0);
        CHECK(*l < 48);
        CHECK(seen.insert(*l).second);  // distinct WITHIN the group
        if (group.kv_cache_spec->kind() == KVCacheSpecKind::kMamba) {
          recurrent_layers.insert(*l);
        } else {
          attn_layers.insert(*l);
        }
      }
    }
    // No layer is named by both an attention and the recurrent group — the
    // runner asserts this too, and would throw rather than mis-allocate.
    for (int64_t l : recurrent_layers) CHECK(attn_layers.count(l) == 0);
    CHECK(recurrent_layers.size() + attn_layers.size() == 48);
  }

  SUBCASE("the loader's placeholder rewrite leaves it alone") {
    // `ResolveKVCacheGroupLayerNames` returns untouched as soon as ONE name
    // resolves, and its fallback would `clear()` a THIRD attention group's
    // names — which is precisely how a placeholder-named side cache would end
    // up unnamed and then refused. Idempotence is what makes publishing real
    // names sufficient.
    KVCacheConfig rewritten = kv;
    std::vector<std::string> layer_types(48, "linear_attention");
    for (int i = 3; i < 48; i += 4) layer_types[static_cast<size_t>(i)] =
        "qwen_sparse_attention";
    vllm::v1::ResolveKVCacheGroupLayerNames(rewritten, 48, layer_types);
    REQUIRE(rewritten.kv_cache_groups.size() == 3);
    for (size_t g = 0; g < 3; ++g) {
      CAPTURE(g);
      CHECK(rewritten.kv_cache_groups[g].layer_names ==
            kv.kv_cache_groups[g].layer_names);
    }
  }
}

// ─── 2. What the uniform group COSTS, and why it is still the mirror ─────────

TEST_CASE("qwen4_exp: the uniform recurrent group is 49.2 MiB of deliberate slack") {
  const nlohmann::json with_ple = FixtureDoc();
  nlohmann::json without_ple = FixtureDoc();
  without_ple["text_config"].erase("ple_layer_ids");

  const KVCacheConfig a = MakeThroughRegistry(with_ple, 16, 8);
  const KVCacheConfig b = MakeThroughRegistry(without_ple, 16, 8);

  const auto* with_spec =
      dynamic_cast<const MambaSpec*>(a.kv_cache_groups[1].kv_cache_spec.get());
  const auto* without_spec =
      dynamic_cast<const MambaSpec*>(b.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(with_spec != nullptr);
  REQUIRE(without_spec != nullptr);

  // `number_of_conv_states` is 3 with a PLE layer and 1 without, mirroring
  // upstream. Two states is therefore not a fallback shape: it is what a config
  // with no PLE layer genuinely needs.
  CHECK(with_spec->shapes.size() == 4);
  CHECK(without_spec->shapes.size() == 2);

  // 10240 * 9 * 2 (the PLE conv) + 2 * 8 (the n-gram history).
  const int64_t per_layer_surcharge =
      with_spec->page_size_bytes() - without_spec->page_size_bytes();
  CHECK(per_layer_surcharge == 184336);

  // Every recurrent layer pays it; exactly ONE of the 36 uses it, because
  // `ple_layer_ids` is [2] one-indexed, i.e. 0-based layer 1.
  constexpr int kMaxNumSeqs = 8;  // the engine default
  const int64_t paid =
      vllm::v1::recurrent_state_bytes(a, kMaxNumSeqs) -
      vllm::v1::recurrent_state_bytes(b, kMaxNumSeqs);
  CHECK(paid == 36 * per_layer_surcharge * kMaxNumSeqs);
  // The slack: 35 of the 36 layers carry a state they never read. 49.2 MiB.
  const int64_t slack = 35 * per_layer_surcharge * kMaxNumSeqs;
  CHECK(slack == 51614080);

  // And the whole recurrent allocation the runner will make, from the same
  // shared accessor the engine's own budget check reads.
  CHECK(vllm::v1::recurrent_state_bytes(a, kMaxNumSeqs) ==
        36 * 3391504LL * kMaxNumSeqs);

  // The PAGED divisor the engine sizes its block pool with counts BOTH
  // attention groups, weighted by the layers each covers — the whole point of
  // FIX-KV-GROUP-LAYER-COUNT (#1963, #1966), which is why the groups above
  // publish real per-layer names rather than one placeholder each. 12 QSA
  // layers at 32768 B plus 12 side caches at 1024 B; the recurrent group
  // contributes nothing here because its state is sized per sequence slot and
  // not per block.
  CHECK(vllm::v1::KVBytesPerBlock(a) == 12 * 32768LL + 12 * 1024LL);
  CHECK(vllm::v1::KVBytesPerBlock(a) == 405504);
}

// ─── 3. The refusals, each naming what it refuses ────────────────────────────

TEST_CASE("qwen4_exp: the KV spec refuses BY NAME what it cannot size") {
  SUBCASE("a block size the compress ratio does not divide") {
    // `storage_block_size()` is INTEGER division
    // (`kv_cache_interface.py:393-395`). At block 18 / ratio 4 the page is
    // sized for 4 states while the block still covers 18 tokens, so the last
    // partial state has nowhere to go: a short cache, not a crash.
    const std::string msg = ThrowText(FixtureDoc(), 18, 8);
    CHECK(msg.find("qwen4_exp KV spec") != std::string::npos);
    CHECK(msg.find("indexer_compress_ratio") != std::string::npos);
    CHECK(msg.find("storage_block_size") != std::string::npos);
    CHECK(msg.find("18") != std::string::npos);
    // 16 and 4 divide, so the same config at the production block size does not
    // throw — the refusal is scoped to the defect and not to the model.
    CHECK(ThrowText(FixtureDoc(), 16, 8).empty());
    CHECK(ThrowText(FixtureDoc(), 4, 8).empty());
  }

  SUBCASE("sparse-attention layers with no indexer group") {
    // QSA is optional as a WHOLE in the config layer (all five `indexer_*`
    // fields or none), while the `full_attention` -> `qwen_sparse_attention`
    // rewrite is unconditional. That combination parses and has no side cache
    // to size.
    nlohmann::json doc = FixtureDoc();
    for (const char* f : {"indexer_n_heads", "indexer_kv_heads",
                          "indexer_head_dim", "indexer_budget",
                          "indexer_compress_ratio"}) {
      doc["text_config"].erase(f);
    }
    const std::string msg = ThrowText(doc, 16, 8);
    CHECK(msg.find("qwen4_exp KV spec") != std::string::npos);
    CHECK(msg.find("indexer_") != std::string::npos);
    CHECK(msg.find("side cache cannot be sized") != std::string::npos);
  }

  SUBCASE("a non-positive block size") {
    const std::string msg = ThrowText(FixtureDoc(), 0, 8);
    CHECK(msg.find("block_size must be positive") != std::string::npos);
  }
}

// ─── 4. What publishing an MLA group COSTS the model, stated as a gate ───────
//
// Not a defect of this wave and not fixed by it: recorded as an executable
// consequence, because the alternative is that the first person to type
// `--kv-cache-dtype fp8` at this model discovers it from a refusal whose reason
// lives in another row.
TEST_CASE("qwen4_exp: publishing an MLA group makes --kv-cache-dtype fp8 REFUSE") {
  KVCacheConfig kv = MakeThroughRegistry(FixtureDoc(), 16, 8);
  // `auto` is the production default and is INERT: `ApplyCacheDType` returns
  // before it touches a spec, so the three groups above are byte-identical.
  const int64_t before = kv.kv_cache_groups[2].kv_cache_spec->page_size_bytes();
  CHECK_NOTHROW(vllm::v1::ApplyCacheDType(
      kv, vllm::v1::ParseCacheDType("auto", DType::kBF16), 1.0F, 1.0F));
  CHECK(kv.kv_cache_groups[2].kv_cache_spec->page_size_bytes() == before);

  // `fp8` refuses the WHOLE model, because an MLA page has its own quantized
  // formula upstream (`fp8_ds_mla`, `kv_cache_interface.py:398-410`) and this
  // tree has landed that formula with no fp8_ds_mla store or read
  // (`src/vllm/v1/kv_cache_interface.cpp`, `RetypeAttentionSpec`). Sizing the
  // page for bytes nothing writes is wrong tokens rather than a crash, so the
  // refusal is the right direction — and it now covers a model whose OTHER two
  // groups an fp8 cache would have been fine for.
  std::string msg;
  try {
    vllm::v1::ApplyCacheDType(
        kv, vllm::v1::ParseCacheDType("fp8", DType::kBF16), 1.0F, 1.0F);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  CHECK(msg.find("MLA KV cache") != std::string::npos);
  CHECK(msg.find("fp8_ds_mla") != std::string::npos);
}
