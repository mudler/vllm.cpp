// MODEL-FP8-BLOCK-WEIGHT — #1189 milestone M3, spec
// `.agents/specs/model-fp8-block-weight.md`.
//
// Block-wise (fine-grained 128x128) FP8 LOADS: `Fp8BlockWeight`, the
// `weight_scale_inv` loader rung, and the quantization-config reader that
// cross-checks the config against the tensors.
//
// EVERY case here enters through a PRODUCTION entry point and never through
// `LoadFp8BlockRaw`. `ModelRegistry::Load` is the registry seam
// `src/vllm/entrypoints/model_loader.cpp:1706` calls; `LoadQwen3_5Dense` is the
// dense loader it reaches at `src/vllm/model_executor/models/qwen3_5_dense.cpp:101`
// and is the only one of the two that hands back an inspectable
// `Qwen3_5DenseWeights`. Deleting the block rung in `load_projection`
// (`qwen3_5_dense_weights.cpp`) must red this file: without it a block-wise
// projection falls into the per-tensor arm and the load dies on
// `tensor not found: ...q_proj.weight_scale`, which is exactly issue #1166.
//
// No checkpoint download, no GPU, no snapshot. The fixture is a complete but
// tiny `Qwen3_5ForConditionalGeneration` dense checkpoint written to a temp
// directory, following the safetensors byte layout pinned by
// `tests/vllm/test_safetensors.cpp:57-89`.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

using vllm::Fp8BlockWeight;
using vllm::HfConfig;
using vllm::ModelSource;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseWeights;
using vllm::SafetensorsFile;

constexpr const char* kArch = "Qwen3_5ForConditionalGeneration";

// ---------------------------------------------------------------------------
// The synthetic checkpoint
// ---------------------------------------------------------------------------

struct FixtureTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  return n;
}

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

// u64-LE header length + JSON header + payload, the safetensors container.
std::string BuildSafetensors(const std::vector<FixtureTensor>& tensors) {
  nlohmann::json header = nlohmann::json::object();
  std::string payload;
  for (const FixtureTensor& t : tensors) {
    const size_t begin = payload.size();
    payload.append(reinterpret_cast<const char*>(t.bytes.data()),
                   t.bytes.size());
    nlohmann::json entry = nlohmann::json::object();
    entry["dtype"] = t.dtype;
    entry["shape"] = t.shape;
    entry["data_offsets"] = nlohmann::json::array({begin, payload.size()});
    header[t.name] = std::move(entry);
  }
  const std::string head = header.dump();
  return U64Le(head.size()) + head + payload;
}

class TempCheckpoint {
 public:
  explicit TempCheckpoint(const std::vector<FixtureTensor>& tensors) {
    static std::atomic<uint64_t> counter{0};
    static const uint64_t nonce = [] {
      std::random_device rd;
      return (static_cast<uint64_t>(rd()) << 32) ^ rd();
    }();
    dir_ = std::filesystem::temp_directory_path() /
           ("vllm_fp8_block_" + std::to_string(nonce) + "_" +
            std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(dir_);
    path_ = dir_ / "model.safetensors";
    const std::string bytes = BuildSafetensors(tensors);
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("failed to write fixture checkpoint");
  }
  ~TempCheckpoint() {
    std::error_code ignored;
    std::filesystem::remove_all(dir_, ignored);
  }
  TempCheckpoint(const TempCheckpoint&) = delete;
  TempCheckpoint& operator=(const TempCheckpoint&) = delete;
  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path dir_;
  std::filesystem::path path_;
};

std::vector<uint8_t> Bf16Filled(const std::vector<int64_t>& shape,
                                uint16_t pattern) {
  std::vector<uint8_t> bytes(static_cast<size_t>(Numel(shape)) * 2);
  for (size_t i = 0; i + 1 < bytes.size(); i += 2) {
    bytes[i] = static_cast<uint8_t>(pattern & 0xff);
    bytes[i + 1] = static_cast<uint8_t>(pattern >> 8);
  }
  return bytes;
}

// Raw fp8-e4m3fn bytes, a walking pattern so a truncated or misaligned copy is
// visible rather than uniform.
std::vector<uint8_t> Fp8Walk(const std::vector<int64_t>& shape) {
  const size_t n = static_cast<size_t>(Numel(shape));
  std::vector<uint8_t> bytes(n);
  for (size_t i = 0; i < n; ++i) bytes[i] = static_cast<uint8_t>((i * 7) & 0x7f);
  return bytes;
}

std::vector<uint8_t> Bytes16(const std::vector<uint16_t>& values) {
  std::vector<uint8_t> bytes(values.size() * 2);
  for (size_t i = 0; i < values.size(); ++i) {
    bytes[2 * i] = static_cast<uint8_t>(values[i] & 0xff);
    bytes[2 * i + 1] = static_cast<uint8_t>(values[i] >> 8);
  }
  return bytes;
}

std::vector<uint8_t> Bytes32(const std::vector<float>& values) {
  std::vector<uint8_t> bytes(values.size() * 4);
  for (size_t i = 0; i < values.size(); ++i)
    std::memcpy(bytes.data() + 4 * i, &values[i], 4);
  return bytes;
}

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// The four bf16 scale bit patterns every block fixture uses. Each is an EXACT
// f32 value with its low 16 bits zero, so the expected f32 is a literal and a
// four-byte REINTERPRETATION of two adjacent bf16 halves lands nowhere near it.
//   0x3F80 -> 1.0     0x3E00 -> 0.125
//   0xBF00 -> -0.5    0x3DCD -> 0.10009765625 (bf16(0.1), deliberately not 0.1)
const std::vector<uint16_t>& BlockScaleBf16Pattern() {
  static const std::vector<uint16_t> p = {0x3F80, 0x3E00, 0xBF00, 0x3DCD};
  return p;
}
const std::vector<float>& BlockScaleExpected() {
  static const std::vector<float> v = {1.0F, 0.125F, -0.5F, 0.10009765625F};
  return v;
}

std::vector<uint8_t> BlockScaleBytesBf16(int64_t rows, int64_t cols) {
  const std::vector<uint16_t>& pat = BlockScaleBf16Pattern();
  std::vector<uint16_t> out(static_cast<size_t>(rows * cols));
  for (size_t i = 0; i < out.size(); ++i) out[i] = pat[i % pat.size()];
  return Bytes16(out);
}

std::vector<uint8_t> BlockScaleBytesF32(int64_t rows, int64_t cols) {
  const std::vector<float>& pat = BlockScaleExpected();
  std::vector<float> out(static_cast<size_t>(rows * cols));
  for (size_t i = 0; i < out.size(); ++i) out[i] = pat[i % pat.size()];
  return Bytes32(out);
}

float ExpectedScaleAt(int64_t index) {
  return BlockScaleExpected()[static_cast<size_t>(index) %
                              BlockScaleExpected().size()];
}

// How a projection's tensors are written into the fixture.
enum class ProjArm {
  kBlockBf16Scale,   // F8_E4M3 weight + BF16 weight_scale_inv
  kBlockF32Scale,    // F8_E4M3 weight + F32 weight_scale_inv
  kBlockF16Scale,    // F8_E4M3 weight + F16 weight_scale_inv (must refuse)
  kBlockFloorScale,  // scale sized by FLOOR instead of cdiv (must refuse)
  kBlockNoScale,     // F8_E4M3 weight with NO weight_scale_inv
  kBlockPlusInput,   // block-wise plus a stray input_scale (must refuse)
  kPerTensor,        // F8_E4M3 weight + F32 weight_scale + F32 input_scale
  kBf16,             // plain BF16 weight
};

void AppendProjection(std::vector<FixtureTensor>& out, const std::string& proj,
                      int64_t n, int64_t k, ProjArm arm, int64_t block_n = 128,
                      int64_t block_k = 128) {
  const std::vector<int64_t> wshape = {n, k};
  if (arm == ProjArm::kBf16) {
    out.push_back({proj + ".weight", "BF16", wshape, Bf16Filled(wshape, 0x3F80)});
    return;
  }
  out.push_back({proj + ".weight", "F8_E4M3", wshape, Fp8Walk(wshape)});
  if (arm == ProjArm::kPerTensor) {
    out.push_back({proj + ".weight_scale", "F32", {}, Bytes32({0.25F})});
    out.push_back({proj + ".input_scale", "F32", {}, Bytes32({0.5F})});
    return;
  }
  if (arm == ProjArm::kBlockNoScale) return;
  int64_t rows = CDiv(n, block_n);
  int64_t cols = CDiv(k, block_k);
  if (arm == ProjArm::kBlockFloorScale) {
    rows = n / block_n;
    cols = k / block_k;
  }
  const std::vector<int64_t> sshape = {rows, cols};
  switch (arm) {
    case ProjArm::kBlockF32Scale:
      out.push_back({proj + ".weight_scale_inv", "F32", sshape,
                     BlockScaleBytesF32(rows, cols)});
      break;
    case ProjArm::kBlockF16Scale:
      out.push_back({proj + ".weight_scale_inv", "F16", sshape,
                     BlockScaleBytesBf16(rows, cols)});
      break;
    default:
      out.push_back({proj + ".weight_scale_inv", "BF16", sshape,
                     BlockScaleBytesBf16(rows, cols)});
      break;
  }
  if (arm == ProjArm::kBlockPlusInput)
    out.push_back({proj + ".input_scale", "F32", {}, Bytes32({0.5F})});
}

// Geometry of the fixture model. Small, and every projection's N and K are
// independent because the loader loads tensors rather than validating a model.
struct FixtureShape {
  int64_t hidden = 256;
  int64_t vocab = 32;
  int64_t q_n = 256;
  int64_t q_k = 256;
  int64_t kv_n = 128;
  int64_t inter = 512;
};

// One `full_attention` layer, tied lm_head. `arm` selects how the four
// self_attn projections and the three MLP projections are written.
std::vector<FixtureTensor> DenseFixture(ProjArm arm,
                                        const FixtureShape& s = {}) {
  std::vector<FixtureTensor> t;
  t.push_back({"model.embed_tokens.weight", "BF16", {s.vocab, s.hidden},
               Bf16Filled({s.vocab, s.hidden}, 0x3F80)});
  t.push_back({"model.norm.weight", "BF16", {s.hidden},
               Bf16Filled({s.hidden}, 0x3F80)});
  const std::string base = "model.layers.0.";
  t.push_back({base + "input_layernorm.weight", "BF16", {s.hidden},
               Bf16Filled({s.hidden}, 0x3F80)});
  t.push_back({base + "post_attention_layernorm.weight", "BF16", {s.hidden},
               Bf16Filled({s.hidden}, 0x3F80)});
  AppendProjection(t, base + "self_attn.q_proj", s.q_n, s.q_k, arm);
  AppendProjection(t, base + "self_attn.k_proj", s.kv_n, s.hidden, arm);
  AppendProjection(t, base + "self_attn.v_proj", s.kv_n, s.hidden, arm);
  AppendProjection(t, base + "self_attn.o_proj", s.hidden, s.q_n, arm);
  t.push_back({base + "self_attn.q_norm.weight", "BF16", {64},
               Bf16Filled({64}, 0x3F80)});
  t.push_back({base + "self_attn.k_norm.weight", "BF16", {64},
               Bf16Filled({64}, 0x3F80)});
  AppendProjection(t, base + "mlp.gate_proj", s.inter, s.hidden, arm);
  AppendProjection(t, base + "mlp.up_proj", s.inter, s.hidden, arm);
  AppendProjection(t, base + "mlp.down_proj", s.hidden, s.inter, arm);
  return t;
}

// ---------------------------------------------------------------------------
// The config
// ---------------------------------------------------------------------------

nlohmann::json BlockQuantJson(const std::vector<int64_t>& block = {128, 128},
                              const std::string& scheme = "dynamic",
                              const std::string& method = "fp8") {
  nlohmann::json q = nlohmann::json::object();
  q["quant_method"] = method;
  q["fmt"] = "e4m3";
  q["activation_scheme"] = scheme;
  q["weight_block_size"] = block;
  return q;
}

// `quant` is null for a checkpoint that declares no quantization config.
HfConfig DenseConfig(const nlohmann::json& quant) {
  HfConfig config;
  config.architectures = {kArch};
  config.model_type = "qwen3_5";
  config.num_hidden_layers = 1;
  config.layer_types = {"full_attention"};
  config.vocab_size = 32;
  nlohmann::json doc = nlohmann::json::object();
  doc["architectures"] = nlohmann::json::array({kArch});
  if (!quant.is_null()) doc["quantization_config"] = quant;
  config.raw = std::move(doc);
  return config;
}

// ---------------------------------------------------------------------------
// Driving the production paths
// ---------------------------------------------------------------------------

// `LoadQwen3_5Dense` is the loader `ModelRegistry::Load` reaches through
// `LoadQwen3_5DenseModel` (`qwen3_5_dense.cpp:101`). It is used where a case
// has to INSPECT what was loaded, which the type-erased `LoadedModel` cannot
// hand back.
Qwen3_5DenseWeights LoadDense(const TempCheckpoint& ckpt,
                              const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  return vllm::LoadQwen3_5Dense(shards, config, /*load_queue=*/nullptr);
}

std::string LoadDenseFailure(const TempCheckpoint& ckpt,
                             const HfConfig& config) {
  try {
    const Qwen3_5DenseWeights weights = LoadDense(ckpt, config);
    (void)weights;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

// The registry seam, the entry point a user actually arrives through.
std::string RegistryLoadFailure(const TempCheckpoint& ckpt,
                                const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  const ModelSource source = ModelSource::FromSafetensors(shards);
  try {
    std::unique_ptr<vllm::LoadedModel> model =
        vllm::ModelRegistry::Load(config, source);
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

bool Names(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

float ScaleAt(const Fp8BlockWeight& w, int64_t r, int64_t c) {
  const auto* p = reinterpret_cast<const float*>(w.scale.bytes.data());
  return p[r * w.scale.shape[1] + c];
}

}  // namespace

// G1 -----------------------------------------------------------------------
TEST_CASE("fp8 block weight: the loader rung is selected for a block-wise checkpoint") {
  const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockBf16Scale));
  const HfConfig config = DenseConfig(BlockQuantJson());

  // Through the REGISTRY first. Without the rung this throws
  // `tensor not found: ...q_proj.weight_scale`, which is issue #1166 and is the
  // RED this case was written against.
  const std::string registry = RegistryLoadFailure(ckpt, config);
  CHECK_MESSAGE(registry.empty(), "ModelRegistry::Load failed: " << registry);

  const Qwen3_5DenseWeights w = LoadDense(ckpt, config);
  REQUIRE(w.layers.size() == 1);
  const vllm::Qwen3_5DenseLayerWeights& layer = w.layers[0];
  REQUIRE_FALSE(layer.is_linear_attention);

  // The block slot is populated and the two arms it must not have taken are
  // empty. Checking only the block slot would pass for a rung that ALSO ran the
  // per-tensor arm.
  CHECK_FALSE(layer.attn.q_proj_fp8_block.Empty());
  CHECK(layer.attn.q_proj_fp8.Empty());
  CHECK(layer.attn.q_proj.Empty());
  CHECK(layer.attn.q_proj_fp4.Empty());
  CHECK_FALSE(layer.attn.k_proj_fp8_block.Empty());
  CHECK_FALSE(layer.attn.v_proj_fp8_block.Empty());
  CHECK_FALSE(layer.attn.o_proj_fp8_block.Empty());
  // The MLP had NO fp8 rung before this row; without one a block-wise MLP goes
  // to LoadMergedBf16RawNK and dies on "expected BF16".
  CHECK_FALSE(layer.mlp.gate_proj_fp8_block.Empty());
  CHECK_FALSE(layer.mlp.up_proj_fp8_block.Empty());
  CHECK_FALSE(layer.mlp.down_proj_fp8_block.Empty());
  CHECK(layer.mlp.gate_up_proj.Empty());

  // The geometry travels ON the weight, so a consumer cannot pair it with a
  // block shape from somewhere else.
  const Fp8BlockWeight& q = layer.attn.q_proj_fp8_block;
  CHECK(q.n == 256);
  CHECK(q.k == 256);
  CHECK(q.block_n == 128);
  CHECK(q.block_k == 128);
  CHECK(q.packed.dtype == vt::DType::kI8);
  REQUIRE(q.packed.rank == 2);
  CHECK(q.packed.shape[0] == 256);
  CHECK(q.packed.shape[1] == 256);
  // The fp8 bytes are copied verbatim, not dequantized.
  const std::vector<uint8_t> expect = Fp8Walk({256, 256});
  REQUIRE(q.packed.bytes.size() == expect.size());
  CHECK(std::memcmp(q.packed.bytes.data(), expect.data(), expect.size()) == 0);
}

// G2 -----------------------------------------------------------------------
TEST_CASE("fp8 block weight: the BF16 scale is WIDENED to f32 rather than reinterpreted") {
  // Upstream allocates the scale parameter f32 (`fp8_utils.py:1276`) and loads
  // the checkpoint tensor with `self.data.copy_()` (`parameter.py:97`), which
  // CONVERTS. `Qwen/Qwen3.8-27B-FP8` ships the tensor BF16. So the resident
  // scale is f32 and the conversion is a value conversion.
  //
  // The expected values are literals with their low 16 bits zero. Reading four
  // bytes as one f32 — the #1181 defect — splices two adjacent bf16 halves and
  // cannot land on any of them.
  SUBCASE("BF16 on disk") {
    const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockBf16Scale));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, DenseConfig(BlockQuantJson()));
    const Fp8BlockWeight& q = w.layers[0].attn.q_proj_fp8_block;
    CHECK(q.scale.dtype == vt::DType::kF32);
    REQUIRE(q.scale.rank == 2);
    CHECK(q.scale.shape[0] == 2);  // cdiv(256, 128)
    CHECK(q.scale.shape[1] == 2);  // cdiv(256, 128)
    REQUIRE(q.scale.bytes.size() == 4u * 4u);
    CHECK(ScaleAt(q, 0, 0) == 1.0F);
    CHECK(ScaleAt(q, 0, 1) == 0.125F);
    CHECK(ScaleAt(q, 1, 0) == -0.5F);
    // bf16(0.1) is 0.10009765625 exactly, NOT 0.1. A loader that rounded or
    // rewrote the value would produce 0.1 and fail here.
    CHECK(ScaleAt(q, 1, 1) == 0.10009765625F);
  }
  SUBCASE("F32 on disk") {
    // The other dtype upstream's converting copy accepts without change.
    const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockF32Scale));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, DenseConfig(BlockQuantJson()));
    const Fp8BlockWeight& q = w.layers[0].attn.q_proj_fp8_block;
    CHECK(q.scale.dtype == vt::DType::kF32);
    CHECK(ScaleAt(q, 0, 0) == 1.0F);
    CHECK(ScaleAt(q, 1, 1) == 0.10009765625F);
  }
  SUBCASE("any other dtype is refused BY NAME rather than reinterpreted") {
    const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockF16Scale));
    const std::string message =
        LoadDenseFailure(ckpt, DenseConfig(BlockQuantJson()));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "weight_scale_inv"));
    CHECK(Names(message, "F16"));
    CHECK(Names(message, "BF16"));
    CHECK(Names(message, "F32"));
  }
}

// G3 -----------------------------------------------------------------------
TEST_CASE("fp8 block weight: a ragged dimension tiles by cdiv on both axes") {
  // `N=576` is `4*128 + 64` and `K=3884` is `30*128 + 44`, upstream's own
  // non-round shapes (`tests/kernels/quantization/test_block_fp8.py:49-50`).
  // The target checkpoint has no ragged edge; the SCHEME does, and M2 measured
  // that a grid of round shapes stays green through two floor-vs-ceil defects.
  FixtureShape s;
  s.q_n = 576;    // ragged N alone
  s.q_k = 256;
  s.kv_n = 128;
  s.inter = 512;
  const TempCheckpoint ragged_n(DenseFixture(ProjArm::kBlockBf16Scale, s));
  const Qwen3_5DenseWeights wn =
      LoadDense(ragged_n, DenseConfig(BlockQuantJson()));
  const Fp8BlockWeight& qn = wn.layers[0].attn.q_proj_fp8_block;
  CHECK(qn.scale.shape[0] == 5);  // cdiv(576, 128) == 5, floor would be 4
  CHECK(qn.scale.shape[1] == 2);
  CHECK(qn.n == 576);

  FixtureShape both;
  both.q_n = 576;
  both.q_k = 3884;  // ragged K as well
  both.hidden = 256;
  both.kv_n = 128;
  both.inter = 512;
  const TempCheckpoint ragged_both(DenseFixture(ProjArm::kBlockBf16Scale, both));
  const Qwen3_5DenseWeights wb =
      LoadDense(ragged_both, DenseConfig(BlockQuantJson()));
  const Fp8BlockWeight& qb = wb.layers[0].attn.q_proj_fp8_block;
  CHECK(qb.scale.shape[0] == 5);
  CHECK(qb.scale.shape[1] == 31);  // cdiv(3884, 128) == 31, floor would be 30
  CHECK(qb.k == 3884);
  // The last row and column of the scale still carry their fixture value, so a
  // short final block was not dropped.
  CHECK(ScaleAt(qb, 4, 30) ==
        ExpectedScaleAt(4 * 31 + 30));

  SUBCASE("a FLOOR-sized scale for the same weight is refused by name") {
    const TempCheckpoint floor(DenseFixture(ProjArm::kBlockFloorScale, both));
    const std::string message =
        LoadDenseFailure(floor, DenseConfig(BlockQuantJson()));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "weight_scale_inv"));
    // BOTH shapes, so the reader does not have to derive the expected one.
    CHECK(Names(message, "[4, 30]"));
    CHECK(Names(message, "[5, 31]"));
  }
}

// G4 -----------------------------------------------------------------------
TEST_CASE("fp8 block weight: a config and tensor disagreement is refused by name") {
  // A dtype probe alone cannot see a DISAGREEMENT: it sees a tensor and picks
  // an arm. This is where a silent-wrong-scale bug lives.
  SUBCASE("weight_scale_inv present but the config declares no weight_block_size") {
    const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockBf16Scale));
    nlohmann::json quant = BlockQuantJson();
    quant.erase("weight_block_size");
    const std::string message = LoadDenseFailure(ckpt, DenseConfig(quant));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "weight_scale_inv"));
    CHECK(Names(message, "weight_block_size"));
    CHECK(Names(message, "q_proj"));
  }
  SUBCASE("the config declares weight_block_size but the tensor has no scale") {
    const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockNoScale));
    const std::string message =
        LoadDenseFailure(ckpt, DenseConfig(BlockQuantJson()));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "weight_scale_inv"));
    CHECK(Names(message, "weight_block_size"));
    CHECK(Names(message, "q_proj"));
  }
  SUBCASE("a module listed in modules_to_not_convert still ships a block scale") {
    const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockBf16Scale));
    nlohmann::json quant = BlockQuantJson();
    quant["modules_to_not_convert"] =
        nlohmann::json::array({"model.layers.0.self_attn.q_proj"});
    const std::string message = LoadDenseFailure(ckpt, DenseConfig(quant));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "modules_to_not_convert"));
    CHECK(Names(message, "q_proj"));
    CHECK(Names(message, "weight_scale_inv"));
  }
  SUBCASE("an input_scale beside a dynamic activation scheme") {
    // The target checkpoint ships ZERO input_scale tensors, and upstream
    // registers one only when `act_q_static` (`fp8.py:381-384`), which block
    // quant asserts against at `fp8.py:367`.
    const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockPlusInput));
    const std::string message =
        LoadDenseFailure(ckpt, DenseConfig(BlockQuantJson()));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "input_scale"));
    CHECK(Names(message, "dynamic"));
    CHECK(Names(message, "q_proj"));
  }
  SUBCASE("a module genuinely excluded and genuinely unquantized loads") {
    // The other half of the exclusion rule. Without this the gate passes for a
    // reader that refuses every listed module outright.
    FixtureShape s;
    std::vector<FixtureTensor> t = DenseFixture(ProjArm::kBlockBf16Scale, s);
    // Rewrite o_proj as plain BF16 and list it as not converted.
    std::vector<FixtureTensor> filtered;
    for (FixtureTensor& e : t) {
      if (e.name.find("o_proj") == std::string::npos)
        filtered.push_back(std::move(e));
    }
    AppendProjection(filtered, "model.layers.0.self_attn.o_proj", s.hidden,
                     s.q_n, ProjArm::kBf16);
    const TempCheckpoint ckpt(filtered);
    nlohmann::json quant = BlockQuantJson();
    quant["modules_to_not_convert"] =
        nlohmann::json::array({"model.layers.0.self_attn.o_proj"});
    const Qwen3_5DenseWeights w = LoadDense(ckpt, DenseConfig(quant));
    CHECK(w.layers[0].attn.o_proj_fp8_block.Empty());
    CHECK_FALSE(w.layers[0].attn.o_proj.Empty());
    CHECK_FALSE(w.layers[0].attn.q_proj_fp8_block.Empty());
  }
}

// G5 -----------------------------------------------------------------------
TEST_CASE("fp8 block weight: an unsupported block config is refused by name at the registry") {
  const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockBf16Scale));

  SUBCASE("a static activation scheme") {
    // Upstream refuses it too (`fp8.py:127-131`).
    const std::string message = RegistryLoadFailure(
        ckpt, DenseConfig(BlockQuantJson({128, 128}, "static")));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "activation_scheme"));
    CHECK(Names(message, "static"));
    CHECK(Names(message, "dynamic"));
    CHECK(Names(message, "1189"));
  }
  SUBCASE("a weight_block_size that is not two dimensions") {
    // Upstream refuses it too (`fp8.py:121-126`).
    const std::string message =
        RegistryLoadFailure(ckpt, DenseConfig(BlockQuantJson({128})));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "weight_block_size"));
    CHECK(Names(message, "2 dimensions"));
  }
  SUBCASE("a block shape other than 128x128") {
    // OUR limit, not upstream's: M5's kernel is 128x128 and nothing here can
    // execute a 64x128 weight.
    const std::string message =
        RegistryLoadFailure(ckpt, DenseConfig(BlockQuantJson({64, 128})));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "weight_block_size"));
    CHECK(Names(message, "[64, 128]"));
    CHECK(Names(message, "128"));
    CHECK(Names(message, "1189"));
  }
  SUBCASE("a quant_method that is not fp8") {
    // Mirrors `is_checkpoint_fp8_serialized` (`fp8.py:117-120`).
    const std::string message = RegistryLoadFailure(
        ckpt, DenseConfig(BlockQuantJson({128, 128}, "dynamic", "awq")));
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "quant_method"));
    CHECK(Names(message, "awq"));
  }
  SUBCASE("the supported config is NOT refused") {
    // Without this the gate passes for a reader that refuses every block-wise
    // checkpoint, which is exactly the state this row replaces.
    const std::string message =
        RegistryLoadFailure(ckpt, DenseConfig(BlockQuantJson()));
    CHECK_MESSAGE(message.empty(), "supported block config refused: " << message);
  }
}

// G6 -----------------------------------------------------------------------
TEST_CASE("fp8 block weight: the per-tensor and bf16 arms are unchanged") {
  SUBCASE("a per-tensor fp8 checkpoint still lands in Fp8Weight") {
    const TempCheckpoint ckpt(DenseFixture(ProjArm::kPerTensor));
    const Qwen3_5DenseWeights w =
        LoadDense(ckpt, DenseConfig(nlohmann::json()));
    const vllm::FullAttnLayerWeights& a = w.layers[0].attn;
    CHECK(a.q_proj_fp8_block.Empty());
    CHECK_FALSE(a.q_proj_fp8.Empty());
    CHECK(a.q_proj_fp8.weight_scale == 0.25F);
    CHECK(a.q_proj_fp8.input_scale == 0.5F);
    CHECK(a.q_proj_fp8.alpha == 0.125F);
  }
  SUBCASE("a bf16 checkpoint is untouched") {
    const TempCheckpoint ckpt(DenseFixture(ProjArm::kBf16));
    const Qwen3_5DenseWeights w =
        LoadDense(ckpt, DenseConfig(nlohmann::json()));
    const vllm::FullAttnLayerWeights& a = w.layers[0].attn;
    CHECK(a.q_proj_fp8_block.Empty());
    CHECK(a.q_proj_fp8.Empty());
    CHECK_FALSE(a.q_proj.Empty());
    CHECK_FALSE(w.layers[0].mlp.gate_up_proj.Empty());
  }
}

// G7 -----------------------------------------------------------------------
TEST_CASE("fp8 block weight: nothing consumes it yet and Prepare says so by name") {
  // The M3/M4 seam. `ModelRegistry::Load` succeeds so the loader rung is
  // reachable from a production entry point at this merge commit; nothing reads
  // an `Fp8BlockWeight` yet, so `ModelRegistry::Prepare` — which every runner
  // calls before the first forward (`src/vllm/v1/worker/gpu/runner.cpp:414`) —
  // refuses rather than letting the dense `project` lambda fall through to an
  // empty bf16 tensor.
  const TempCheckpoint ckpt(DenseFixture(ProjArm::kBlockBf16Scale));
  const HfConfig config = DenseConfig(BlockQuantJson());
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  const ModelSource source = ModelSource::FromSafetensors(shards);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);

  vt::Queue queue = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  std::string message;
  try {
    vllm::ModelRegistry::Prepare(*model, config, queue);
  } catch (const std::exception& e) {
    message = e.what();
  }
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "block-wise"));
  CHECK(Names(message, "q_proj"));
  CHECK(Names(message, "1189"));
}
