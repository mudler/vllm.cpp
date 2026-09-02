// MODEL-QWEN35-EXL3 — the EXL3 (exllamav3 trellis) arm of the DENSE Qwen3.5
// text model, #2495 items 3 and 5.
//
// `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` declares
// `architectures: ["Qwen3_5ForConditionalGeneration"]`, which this tree already
// registers and already runs in bf16, FP8 and NVFP4. Before this row
// `grep -c exl3 src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`
// returned 0: an EXL3 checkpoint died on a missing-tensor lookup, because an
// EXL3 projection ships NO `.weight` at all and every probe in the resolver
// reads one.
//
// FOUR QUESTIONS, and only the first is about the arm that was added.
//
//   G1  Does an EXL3 checkpoint LOAD, with the right codebook, the right
//       per-tensor width and the right geometry? The codebook is the half a
//       shape check cannot see: the wrong multiplier decodes to the correct RMS
//       and uncorrelated values (`test_exl3_native_loader.cpp` records the
//       measurement), so it is asserted directly.
//
//   G2  Does a PRODUCTION entry point REACH the arm? The case enters at
//       `ModelRegistry::Forward` over the weights the LOADER produced, and
//       compares against the SAME trellis bytes decoded into the bf16 fields.
//       Deleting an EXL3 call site does not merely move the numbers: the arm
//       falls through to a bf16 field an EXL3 load leaves EMPTY and
//       `dense_attn::ResidentWeight` refuses it by name, so the case reds
//       either way. A unit test that constructs an `Exl3Weight` by hand would
//       prove the class works and never that anything reaches it
//       (`.agents/reachability.md`).
//
//   G3  Does the GDN half REFUSE BY NAME? 48 of the real model's 64 layers are
//       `linear_attention` and NOTHING in `ProjectGdnQkvz`/`ProjectGdnBA`/
//       `ProjectGdnOut` consumes an `Exl3Weight`. That arm is #2495 item 4.
//       Half-loading is the failure this prevents, and a refusal is a gateable
//       behaviour where "tensor not found" is not.
//
//   G4  Are the bf16, per-tensor FP8 and NVFP4 arms UNCHANGED? A fourth rung in
//       a presence-based resolver is exactly where an existing arm gets
//       mis-selected, and a mis-selection is silent: every one of these
//       checkpoints still loads and still produces plausible numbers.
//
// No checkpoint download, no GPU, no snapshot. The fixture is a complete but
// tiny `Qwen3_5ForConditionalGeneration` dense checkpoint written to a temp
// directory, in the safetensors byte layout pinned by
// `tests/vllm/test_safetensors.cpp`.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace {

using vllm::Exl3Weight;
using vllm::ForwardLogits;
using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::ModelForwardInput;
using vllm::ModelRegistry;
using vllm::ModelSource;
using vllm::PagedKvCache;
using vllm::Qwen3_5DenseWeights;
using vllm::SafetensorsFile;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

constexpr const char* kArch = "Qwen3_5ForConditionalGeneration";

// The published artifact's width and codebook. 270 of the 272 quantized tensors
// of `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` are FOUR-bit, and every one of them
// carries a `mul1` marker, which is codebook 2 (`exl3.py:74-77`).
constexpr int kBits = 4;
constexpr int kCodebook = 2;
// `torch.tensor(0x83DCD12D, uint32).view(torch.int)` (`quantize.py:1421-1424`):
// one I32 holding the codebook's own multiplier, exactly as `mcg` is written.
constexpr uint32_t kMul1Multiplier = 0x83DCD12DU;

// ---------------------------------------------------------------------------
// Deterministic bytes
// ---------------------------------------------------------------------------

struct Rng {
  uint32_t s = 1u;
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float unit() { return static_cast<float>(next() % 2000) / 1000.0F - 1.0F; }
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  return n;
}

struct FixtureTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

std::vector<uint8_t> Bytes16(const std::vector<uint16_t>& values) {
  std::vector<uint8_t> b(values.size() * 2);
  for (size_t i = 0; i < values.size(); ++i) {
    b[2 * i] = static_cast<uint8_t>(values[i] & 0xff);
    b[2 * i + 1] = static_cast<uint8_t>(values[i] >> 8);
  }
  return b;
}

std::vector<uint8_t> Bytes32F(const std::vector<float>& values) {
  std::vector<uint8_t> b(values.size() * 4);
  for (size_t i = 0; i < values.size(); ++i)
    std::memcpy(b.data() + 4 * i, &values[i], 4);
  return b;
}

std::vector<uint8_t> Bytes32U(uint32_t v) {
  std::vector<uint8_t> b(4);
  std::memcpy(b.data(), &v, 4);
  return b;
}

// The UNQUANTIZED REMAINDER, written at either dtype from ONE set of values.
//
// An EXL3 artifact stores its remainder at F16, because exllamav3 runs the
// linear in fp16; the config's own `torch_dtype` is bfloat16, so the loader
// materializes it to bf16 -- the MODEL dtype every layer inherits. The BF16 arm
// here therefore writes `F32ToBF16(F16ToF32(F32ToF16(v)))` rather than
// `F32ToBF16(v)`: after loading, the two checkpoints hold BIT-IDENTICAL norms
// and embeddings, so the forward comparison in G2 is measuring the projection
// arms and nothing else. Writing the obvious `F32ToBF16(v)` would put a second
// difference into the comparison and quietly loosen it.
std::vector<uint8_t> RemainderBytes(const std::vector<int64_t>& shape,
                                    uint32_t seed, float scale, bool as_f16) {
  Rng r;
  r.s = seed | 1u;
  const int64_t n = Numel(shape);
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const uint16_t half = vt::F32ToF16(r.unit() * scale);
    v[static_cast<size_t>(i)] =
        as_f16 ? half : vt::F32ToBF16(vt::F16ToF32(half));
  }
  return Bytes16(v);
}

// ---------------------------------------------------------------------------
// One EXL3 projection, and the SAME weights decoded
// ---------------------------------------------------------------------------

// The trellis bytes plus the two sign vectors, held once so the EXL3 fixture
// and its decoded twin are the same bytes read two ways. That is what makes the
// forward comparison an EQUIVALENCE rather than a tolerance between two
// different models.
struct Exl3Proj {
  int64_t k = 0;
  int64_t n = 0;
  std::vector<uint16_t> trellis;
  std::vector<uint16_t> suh;
  std::vector<uint16_t> svh;
};

Exl3Proj MakeProj(int64_t k, int64_t n, uint32_t seed) {
  Rng r;
  r.s = seed | 1u;
  Exl3Proj p;
  p.k = k;
  p.n = n;
  p.trellis.resize(static_cast<size_t>(k / 16) * static_cast<size_t>(n / 16) *
                   16 * kBits);
  for (auto& w : p.trellis) w = static_cast<uint16_t>(r.next() & 0xffffu);
  p.suh.resize(static_cast<size_t>(k));
  p.svh.resize(static_cast<size_t>(n));
  // Sign vectors, which is what they are: +-1 in fp16 (`exl3.py:48-49`).
  for (auto& v : p.suh) v = vt::F32ToF16((r.next() & 1u) != 0u ? 1.0F : -1.0F);
  for (auto& v : p.svh) v = vt::F32ToF16((r.next() & 1u) != 0u ? 1.0F : -1.0F);
  return p;
}

void AppendExl3(std::vector<FixtureTensor>& out, const std::string& proj,
                const Exl3Proj& p) {
  out.push_back({proj + ".trellis", "I16",
                 {p.k / 16, p.n / 16, 16 * kBits}, Bytes16(p.trellis)});
  out.push_back({proj + ".suh", "F16", {p.k}, Bytes16(p.suh)});
  out.push_back({proj + ".svh", "F16", {p.n}, Bytes16(p.svh)});
  out.push_back({proj + ".mul1", "I32", {1}, Bytes32U(kMul1Multiplier)});
}

// `Exl3DequantLinear` yields [k, n]; a torch Linear stores [out, in] = [n, k],
// which is what the bf16 loader rung reads. This transpose is the ONE place the
// two arms' orientations are reconciled, and getting it wrong shows up
// immediately as a failed equivalence rather than as a plausible wrong number.
std::vector<uint8_t> DecodedTorchWeight(const Exl3Proj& p) {
  std::vector<float> w(static_cast<size_t>(p.k) * static_cast<size_t>(p.n), 0.0F);
  vt::Exl3DequantLinear(p.trellis.data(), p.suh.data(), p.svh.data(), p.k, p.n,
                        kBits, kCodebook, w.data());
  std::vector<uint16_t> bf(static_cast<size_t>(p.k) * static_cast<size_t>(p.n));
  for (int64_t i = 0; i < p.n; ++i)
    for (int64_t j = 0; j < p.k; ++j)
      bf[static_cast<size_t>(i) * static_cast<size_t>(p.k) + static_cast<size_t>(j)] =
          vt::F32ToBF16(w[static_cast<size_t>(j) * static_cast<size_t>(p.n) +
                          static_cast<size_t>(i)]);
  return Bytes16(bf);
}

void AppendDecodedBf16(std::vector<FixtureTensor>& out, const std::string& proj,
                       const Exl3Proj& p) {
  out.push_back({proj + ".weight", "BF16", {p.n, p.k}, DecodedTorchWeight(p)});
}

// ---------------------------------------------------------------------------
// The checkpoint container
// ---------------------------------------------------------------------------

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

std::string BuildSafetensors(const std::vector<FixtureTensor>& tensors, size_t header_pad = 0) {
  nlohmann::json header = nlohmann::json::object();
  std::string payload;
  for (const FixtureTensor& t : tensors) {
    const size_t begin = payload.size();
    payload.append(reinterpret_cast<const char*>(t.bytes.data()), t.bytes.size());
    nlohmann::json entry = nlohmann::json::object();
    entry["dtype"] = t.dtype;
    entry["shape"] = t.shape;
    entry["data_offsets"] = nlohmann::json::array({begin, payload.size()});
    header[t.name] = std::move(entry);
  }
  std::string head = header.dump();
  head.append(header_pad, ' ');  // Pad header to control payload parity
  return U64Le(head.size()) + head + payload;
}

void WriteSafetensorsFile(const std::filesystem::path& path,
                          const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!out) throw std::runtime_error("failed to write fixture checkpoint");
}

// Write the checkpoint twice — unpadded, and padded by one space — and keep
// whichever lands the payload on an ODD byte. This makes the test deterministic
// rather than dependent on JSON length luck.
std::string WriteOddOffsetSafetensors(const std::filesystem::path& dir,
                                      const std::vector<FixtureTensor>& tensors) {
  const std::filesystem::path path_a = dir / "model_a.safetensors";
  const std::filesystem::path path_b = dir / "model_b.safetensors";

  const std::string bytes_a = BuildSafetensors(tensors, 0);
  const std::string bytes_b = BuildSafetensors(tensors, 1);

  WriteSafetensorsFile(path_a, bytes_a);
  WriteSafetensorsFile(path_b, bytes_b);

  // Payload starts after 8-byte length header
  uint64_t header_len_a;
  std::memcpy(&header_len_a, bytes_a.data(), 8);
  uint64_t header_len_b;
  std::memcpy(&header_len_b, bytes_b.data(), 8);

  const size_t payload_offset_a = 8 + header_len_a;
  const size_t payload_offset_b = 8 + header_len_b;

  // Verify the two spellings differ in parity
  REQUIRE((payload_offset_a % 2) != (payload_offset_b % 2));

  // Return the path with odd payload offset
  return (payload_offset_a % 2 == 1) ? path_a.string() : path_b.string();
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
           ("vllm_qwen35_exl3_" + std::to_string(nonce) + "_" +
            std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(dir_);

    // Use odd-offset safetensors to deterministically trigger the alignment issue
    const std::string odd_path = WriteOddOffsetSafetensors(dir_, tensors);
    path_ = odd_path;
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

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

// EVERY EXL3 projection's k AND n is a multiple of 128, because each side was
// Hadamard-128 transformed at quantization time (`exl3_lib/quantize.py:15`) and
// the reference dequant refuses anything else. That constrains this model more
// than a bf16 one: `num_key_value_heads * head_dim` is a projection width and
// cannot be the usual small GQA number, and the vocabulary is a projection
// width too because the head is quantized.
HfConfig DenseConfig(bool with_exl3_quant_config) {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {kArch};
  c.hidden_size = 128;
  c.num_hidden_layers = 2;
  c.vocab_size = 128;          // lm_head's N
  c.num_attention_heads = 2;   // q_proj N = 2*Hq*Dh = 256 (output gate doubled)
  c.num_key_value_heads = 2;   // k/v_proj N = Hkv*Dh = 128
  c.head_dim = 64;             // o_proj K = Hq*Dh = 128
  c.layer_types = {"linear_attention", "full_attention"};
  c.intermediate_size = 128;
  c.num_experts = 0;
  c.linear_num_key_heads = 1;
  c.linear_num_value_heads = 2;
  c.linear_key_head_dim = 32;    // conv_dim == 192
  c.linear_value_head_dim = 64;  // value_dim == 128
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 32;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  nlohmann::json doc = nlohmann::json::object();
  doc["architectures"] = nlohmann::json::array({kArch});
  if (with_exl3_quant_config) {
    // What the artifact actually declares. It is written here NOT because the
    // loader reads it -- the scheme is resolved per Linear from the tensors, as
    // `Linear.is_exl3_storage` does upstream -- but because a config this tree
    // has never seen must not derail the compressed-tensors, ModelOpt and
    // block-FP8 readers that run over `config.raw` before any weight loads.
    nlohmann::json q = nlohmann::json::object();
    q["quant_method"] = "exl3";
    q["bits"] = 3.5;
    doc["quantization_config"] = q;
  }
  c.raw = std::move(doc);
  return c;
}

// How the four self_attn projections, the three MLP projections and the head
// are written. Every arm shares one geometry, so a mis-selection is the only
// difference a case can be reading.
enum class Arm {
  kExl3,        // .trellis + .suh + .svh + .mul1
  kBf16,        // .weight BF16, the SAME weights decoded
  kFp8,         // .weight F8_E4M3 + .weight_scale + .input_scale (per-tensor)
  kNvfp4,       // .weight U8 + .weight_scale F8 + .weight_scale_2 (ModelOpt)
};

// The projections of the model, in one place so every arm writes the same set.
struct Geometry {
  int64_t hidden = 128;
  int64_t vocab = 128;
  int64_t q_n = 256;
  int64_t kv_n = 128;
  int64_t inter = 128;
  int64_t head_dim = 64;
  int64_t attn_out_k = 128;  // Hq * Dh
  int64_t conv_dim = 192;
  int64_t value_dim = 128;
  int64_t num_v_heads = 2;
  int64_t v_head_dim = 64;
  int64_t conv_k = 4;
};

std::vector<uint8_t> Fp8Walk(int64_t n) {
  std::vector<uint8_t> b(static_cast<size_t>(n));
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<uint8_t>((i * 7) & 0x7f);
  return b;
}

void AppendProjection(std::vector<FixtureTensor>& out, const std::string& proj,
                      int64_t n, int64_t k, Arm arm, uint32_t seed) {
  if (arm == Arm::kExl3 || arm == Arm::kBf16) {
    const Exl3Proj p = MakeProj(k, n, seed);
    if (arm == Arm::kExl3) {
      AppendExl3(out, proj, p);
    } else {
      AppendDecodedBf16(out, proj, p);
    }
    return;
  }
  if (arm == Arm::kFp8) {
    out.push_back({proj + ".weight", "F8_E4M3", {n, k}, Fp8Walk(n * k)});
    out.push_back({proj + ".weight_scale", "F32", {}, Bytes32F({0.25F})});
    out.push_back({proj + ".input_scale", "F32", {}, Bytes32F({0.5F})});
    return;
  }
  out.push_back({proj + ".weight", "U8", {n, k / 2}, Fp8Walk(n * k / 2)});
  out.push_back({proj + ".weight_scale", "F8_E4M3", {n, k / 16},
                 Fp8Walk(n * k / 16)});
  out.push_back({proj + ".weight_scale_2", "F32", {}, Bytes32F({0.5F})});
}

// The GDN tower. `exl3_gdn` writes it as EXL3, which is what
// `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships and what this row REFUSES; every
// other case writes it BF16, which is the shape whose dense half loads.
void AppendGdnLayer(std::vector<FixtureTensor>& out, const std::string& base,
                    const Geometry& g, bool exl3_gdn, bool f16_remainder) {
  const std::string la = base + "linear_attn.";
  if (exl3_gdn) {
    AppendExl3(out, la + "in_proj_qkv", MakeProj(g.hidden, g.conv_dim, 4001));
    AppendExl3(out, la + "in_proj_z", MakeProj(g.hidden, g.value_dim, 4002));
    AppendExl3(out, la + "out_proj", MakeProj(g.value_dim, g.hidden, 4003));
  } else {
    // The GDN tower stays BF16 in EVERY arm of this fixture, including the EXL3
    // one. That is the point of items 3 and 5 being separable from item 4: the
    // dense half of a checkpoint can be quantized while this tower is not, and
    // when it IS quantized the loader must say so rather than half-load.
    out.push_back({la + "in_proj_qkv.weight", "BF16", {g.conv_dim, g.hidden},
                   RemainderBytes({g.conv_dim, g.hidden}, 4001, 0.1F, false)});
    out.push_back({la + "in_proj_z.weight", "BF16", {g.value_dim, g.hidden},
                   RemainderBytes({g.value_dim, g.hidden}, 4002, 0.1F, false)});
    out.push_back({la + "out_proj.weight", "BF16", {g.hidden, g.value_dim},
                   RemainderBytes({g.hidden, g.value_dim}, 4003, 0.1F, false)});
  }
  (void)f16_remainder;
  out.push_back({la + "in_proj_b.weight", "BF16", {g.num_v_heads, g.hidden},
                 RemainderBytes({g.num_v_heads, g.hidden}, 4004, 0.1F, false)});
  out.push_back({la + "in_proj_a.weight", "BF16", {g.num_v_heads, g.hidden},
                 RemainderBytes({g.num_v_heads, g.hidden}, 4005, 0.1F, false)});
  out.push_back({la + "conv1d.weight", "BF16", {g.conv_dim, 1, g.conv_k},
                 RemainderBytes({g.conv_dim, 1, g.conv_k}, 4006, 0.1F, false)});
  out.push_back({la + "A_log", "BF16", {g.num_v_heads},
                 RemainderBytes({g.num_v_heads}, 4007, 0.5F, false)});
  out.push_back({la + "dt_bias", "BF16", {g.num_v_heads},
                 RemainderBytes({g.num_v_heads}, 4008, 0.5F, false)});
  out.push_back({la + "norm.weight", "BF16", {g.v_head_dim},
                 RemainderBytes({g.v_head_dim}, 4009, 0.5F, false)});
}

// The whole checkpoint. Layer 0 is `linear_attention`, layer 1 is
// `full_attention`, and BOTH carry a dense MLP -- the same split the real model
// has, at 48/16 instead of 1/1.
std::vector<FixtureTensor> DenseCheckpoint(Arm arm, const Geometry& g = {},
                                           bool exl3_gdn = false) {
  // `f16` is a property of the checkpoint FAMILY, not of a single tensor: an
  // EXL3 artifact stores its whole unquantized remainder at F16.
  const bool f16 = (arm == Arm::kExl3);
  const char* rdt = f16 ? "F16" : "BF16";
  std::vector<FixtureTensor> t;
  t.push_back({"model.embed_tokens.weight", rdt, {g.vocab, g.hidden},
               RemainderBytes({g.vocab, g.hidden}, 11, 0.5F, f16)});
  t.push_back({"model.norm.weight", rdt, {g.hidden},
               RemainderBytes({g.hidden}, 12, 0.5F, f16)});

  for (int layer = 0; layer < 2; ++layer) {
    const std::string base = "model.layers." + std::to_string(layer) + ".";
    const uint32_t seed = 1000 + static_cast<uint32_t>(layer) * 500;
    t.push_back({base + "input_layernorm.weight", rdt, {g.hidden},
                 RemainderBytes({g.hidden}, seed + 1, 0.5F, f16)});
    t.push_back({base + "post_attention_layernorm.weight", rdt, {g.hidden},
                 RemainderBytes({g.hidden}, seed + 2, 0.5F, f16)});
    if (layer == 0) {
      AppendGdnLayer(t, base, g, exl3_gdn, f16);
    } else {
      const std::string sa = base + "self_attn.";
      AppendProjection(t, sa + "q_proj", g.q_n, g.hidden, arm, seed + 10);
      AppendProjection(t, sa + "k_proj", g.kv_n, g.hidden, arm, seed + 20);
      AppendProjection(t, sa + "v_proj", g.kv_n, g.hidden, arm, seed + 30);
      AppendProjection(t, sa + "o_proj", g.hidden, g.attn_out_k, arm, seed + 40);
      t.push_back({sa + "q_norm.weight", rdt, {g.head_dim},
                   RemainderBytes({g.head_dim}, seed + 50, 0.5F, f16)});
      t.push_back({sa + "k_norm.weight", rdt, {g.head_dim},
                   RemainderBytes({g.head_dim}, seed + 60, 0.5F, f16)});
    }
    const std::string mlp = base + "mlp.";
    AppendProjection(t, mlp + "gate_proj", g.inter, g.hidden, arm, seed + 101);
    AppendProjection(t, mlp + "up_proj", g.inter, g.hidden, arm, seed + 102);
    AppendProjection(t, mlp + "down_proj", g.hidden, g.inter, arm, seed + 103);
  }

  // The output head. An EXL3 artifact ships a REAL quantized head, and it is
  // preferred over a tied embedding table even when the config declares
  // `tie_word_embeddings: true` (`llama_weights.cpp:152-161`).
  AppendProjection(t, "lm_head", g.vocab, g.hidden, arm, 9001);
  return t;
}

// ---------------------------------------------------------------------------
// Driving the production paths
// ---------------------------------------------------------------------------

Qwen3_5DenseWeights LoadDense(const TempCheckpoint& ckpt,
                              const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  return vllm::LoadQwen3_5Dense(shards, config, /*load_queue=*/nullptr);
}

std::string LoadFailure(const TempCheckpoint& ckpt, const HfConfig& config) {
  try {
    const Qwen3_5DenseWeights weights = LoadDense(ckpt, config);
    (void)weights;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

// The REGISTRY seam, the entry point a user actually arrives through.
std::string RegistryLoadFailure(const TempCheckpoint& ckpt,
                                const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  const ModelSource source = ModelSource::FromSafetensors(shards);
  try {
    std::unique_ptr<vllm::LoadedModel> model =
        vllm::ModelRegistry::Load(config, source);
    (void)model;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

bool Names(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

vt::Queue Cpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// The caches one prefill needs. Sized in floats throughout; a bf16 KV cache
// simply uses half of each buffer.
struct CachePool {
  std::vector<std::vector<float>> attn_buf;
  std::vector<std::vector<float>> ssm_buf;
  std::vector<std::vector<float>> conv_buf;
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;

  CachePool(const HfConfig& c, const Geometry& g, int64_t nb, int64_t bs) {
    ssm_buf.emplace_back(
        static_cast<size_t>(nb * g.num_v_heads * g.v_head_dim *
                            c.linear_key_head_dim),
        0.0F);
    conv_buf.emplace_back(
        static_cast<size_t>(nb * g.conv_dim * (g.conv_k - 1)), 0.0F);
    attn_buf.emplace_back(
        static_cast<size_t>(nb * 2 * bs * c.num_key_value_heads * c.head_dim),
        0.0F);
    const vt::Device cpu{vt::DeviceType::kCPU, 0};
    for (auto& b : attn_buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kBF16;
      kv.num_blocks = nb;
      kv.block_size = bs;
      kv.num_kv_heads = c.num_key_value_heads;
      kv.head_size = c.head_dim;
      attn_kv.push_back(kv);
    }
    for (size_t i = 0; i < ssm_buf.size(); ++i) {
      GdnStateCache gs;
      gs.ssm_state = vt::Tensor::Contiguous(
          ssm_buf[i].data(), DType::kF32, cpu,
          {nb, g.num_v_heads, g.v_head_dim, c.linear_key_head_dim});
      gs.conv_state = vt::Tensor::Contiguous(conv_buf[i].data(), DType::kF32,
                                             cpu, {nb, g.conv_dim, g.conv_k - 1});
      gdn_state.push_back(gs);
    }
  }
};

CommonAttentionMetadata PrefillAttnMeta(int64_t T, int64_t block_size) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t)
    m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

GDNAttentionMetadata PrefillGdnMeta(int64_t T) {
  GDNAttentionMetadata g;
  g.num_prefills = 1;
  g.num_prefill_tokens = static_cast<int>(T);
  g.num_decodes = 0;
  g.num_decode_tokens = 0;
  g.num_actual_tokens = static_cast<int>(T);
  g.has_initial_state = std::vector<uint8_t>{0};
  g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_state_indices = std::vector<int32_t>{0};
  g.prefill_has_initial_state = std::vector<uint8_t>{0};
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}

// THE PRODUCTION ENTRY POINT. `ModelRegistry::Forward` over the weights the
// LOADER produced -- not a hand-constructed container, and not
// `Qwen3_5DenseModel::Forward` -- so deleting an EXL3 call site in the dense
// forward reds this rather than leaving a class-level gate green.
std::vector<float> RegistryForward(const HfConfig& c, const Geometry& g,
                                   const Qwen3_5DenseWeights& w) {
  const int64_t T = 4;
  const std::vector<int32_t> ids = {5, 9, 2, 17};
  const std::vector<int32_t> pos = {0, 1, 2, 3};
  const std::vector<int32_t> logits_indices;
  CachePool pool(c, g, /*nb=*/4, /*bs=*/8);
  const CommonAttentionMetadata am = PrefillAttnMeta(T, 8);
  const GDNAttentionMetadata gm = PrefillGdnMeta(T);
  vt::Queue q = Cpu();
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  ModelForwardInput in{ids, pos, am,       gm, pool.attn_kv,
                       pool.gdn_state,     c,  q,  logits_indices};
  in.num_reqs = 1;
  in.gather_logits = false;
  const ForwardLogits out = ModelRegistry::Forward(*model, in);
  REQUIRE(out.host.size() == static_cast<size_t>(T * c.vocab_size));
  return out.host;
}

}  // namespace

// G1 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: an EXL3 dense checkpoint LOADS, with the codebook and the width it stores") {
  const Geometry g;
  const TempCheckpoint ckpt(DenseCheckpoint(Arm::kExl3, g));
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);

  const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
  REQUIRE(w.layers.size() == 2);

  // The full-attention tower.
  const vllm::FullAttnLayerWeights& a = w.layers[1].attn;
  REQUIRE(a.IsExl3());
  const Exl3Weight* attn[4] = {&a.q_proj_exl3, &a.k_proj_exl3, &a.v_proj_exl3,
                               &a.o_proj_exl3};
  for (const Exl3Weight* e : attn) {
    // THE ASSERTION A SHAPE CHECK CANNOT MAKE. The wrong multiplier yields a
    // codebook with the SAME DISTRIBUTION and no relation to the right one, so
    // the weight decodes to the correct RMS and uncorrelated values and every
    // geometry check below still passes.
    CHECK(e->codebook == kCodebook);
    CHECK(e->Bits() == kBits);
  }
  CHECK(a.q_proj_exl3.InFeatures() == g.hidden);
  CHECK(a.q_proj_exl3.OutFeatures() == g.q_n);
  CHECK(a.k_proj_exl3.OutFeatures() == g.kv_n);
  CHECK(a.v_proj_exl3.OutFeatures() == g.kv_n);
  CHECK(a.o_proj_exl3.InFeatures() == g.attn_out_k);
  CHECK(a.o_proj_exl3.OutFeatures() == g.hidden);

  // The dense MLP, on BOTH layers -- the GDN layer has one too.
  for (const auto& layer : w.layers) {
    REQUIRE(layer.mlp.IsExl3());
    CHECK(layer.mlp.gate_proj_exl3.codebook == kCodebook);
    CHECK(layer.mlp.gate_proj_exl3.InFeatures() == g.hidden);
    CHECK(layer.mlp.gate_proj_exl3.OutFeatures() == g.inter);
    CHECK(layer.mlp.up_proj_exl3.OutFeatures() == g.inter);
    CHECK(layer.mlp.down_proj_exl3.InFeatures() == g.inter);
    CHECK(layer.mlp.down_proj_exl3.OutFeatures() == g.hidden);
  }

  // The head (#2495 item 5), preferred over the tied embedding table.
  REQUIRE_FALSE(w.lm_head_exl3.Empty());
  CHECK(w.lm_head_exl3.codebook == kCodebook);
  CHECK(w.lm_head_exl3.Bits() == kBits);
  CHECK(w.lm_head_exl3.InFeatures() == g.hidden);
  CHECK(w.lm_head_exl3.OutFeatures() == g.vocab);
  CHECK_FALSE(w.tied_lm_head);

  // EXACTLY ONE representation is populated. Without this the case passes for a
  // loader that filled the trellis fields AND dequantized into the bf16 ones,
  // which is numerically better and therefore invisible to every value check.
  CHECK(w.lm_head.Empty());
  CHECK(w.lm_head_fp4.Empty());
  CHECK(a.q_proj.Empty());
  CHECK(a.k_proj.Empty());
  CHECK(a.v_proj.Empty());
  CHECK(a.o_proj.Empty());
  CHECK(a.q_proj_fp8.Empty());
  CHECK(a.q_proj_fp4.Empty());
  CHECK(a.q_proj_fp8_block.Empty());
  CHECK(w.layers[1].mlp.gate_up_proj.Empty());
  CHECK(w.layers[1].mlp.down_proj.Empty());

  // The trellis is still trellis BYTES. A silent dequant-at-load would be
  // numerically better and invisible above; only the byte count sees it.
  const int64_t k = g.hidden, n = g.inter;
  CHECK(w.layers[1].mlp.gate_proj_exl3.trellis.bytes.size() ==
        static_cast<size_t>(k / 16) * static_cast<size_t>(n / 16) * 32 * kBits);
  CHECK(w.layers[1].mlp.gate_proj_exl3.trellis.dtype == DType::kI8);

  // The GDN tower loaded BF16 beside it: the dense half being quantized does
  // not disturb the half that is not.
  CHECK(w.layers[0].is_linear_attention);
  CHECK_FALSE(w.layers[0].gdn.in_proj_qkvz.Empty());
  CHECK_FALSE(w.layers[0].gdn.out_proj.Empty());

  // And the direct-device staging path must not claim this model.
  CHECK_FALSE(vllm::IsPlainBf16Qwen3_5Dense(w));

  // The registry seam accepts it too, which is the entry a user arrives through.
  CHECK(RegistryLoadFailure(ckpt, c).empty());
}

// G2 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: ModelRegistry::Forward REACHES the arm and agrees with the decoded twin") {
  const Geometry g;
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);
  const HfConfig c_plain = DenseConfig(/*with_exl3_quant_config=*/false);

  const TempCheckpoint exl3_ckpt(DenseCheckpoint(Arm::kExl3, g));
  const TempCheckpoint bf16_ckpt(DenseCheckpoint(Arm::kBf16, g));
  const Qwen3_5DenseWeights we = LoadDense(exl3_ckpt, c);
  const Qwen3_5DenseWeights wb = LoadDense(bf16_ckpt, c_plain);
  REQUIRE(we.layers[1].attn.IsExl3());
  REQUIRE_FALSE(wb.layers[1].attn.IsExl3());

  const std::vector<float> le = RegistryForward(c, g, we);
  const std::vector<float> lb = RegistryForward(c_plain, g, wb);
  REQUIRE(le.size() == lb.size());
  REQUIRE(!le.empty());

  for (const float x : le) REQUIRE(std::isfinite(x));
  for (const float x : lb) REQUIRE(std::isfinite(x));

  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < le.size(); ++i) {
    const double d = static_cast<double>(le[i]) - lb[i];
    num += d * d;
    den += static_cast<double>(lb[i]) * lb[i];
  }
  // NOT VACUOUS: a forward returning zeros satisfies any relative bound whose
  // reference is also zero. Asserted BEFORE the ratio is formed.
  REQUIRE(den > 0.0);
  const double rel = std::sqrt(num / den);
  MESSAGE("exl3 forward vs decoded-bf16 forward: rel_rms = ", rel);
  // A BOUND, not an equality: the two arms are the same weights through
  // different kernels. EXL3 rides the Hadamards on the activations while the
  // decoded twin has them baked in, and the twin rounds the decode to bf16.
  CHECK(rel <= 5.0e-2);
}

// G3 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: an EXL3 GDN tower REFUSES BY NAME rather than half-loading") {
  const Geometry g;
  const TempCheckpoint ckpt(
      DenseCheckpoint(Arm::kExl3, g, /*exl3_gdn=*/true));
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/true);

  const std::string message = LoadFailure(ckpt, c);
  REQUIRE_FALSE(message.empty());
  // The projection, so the reader knows this is a real stored weight and not a
  // config guess; the row and the issue, so the refusal names its owner.
  CHECK(Names(message, "in_proj_qkv"));
  CHECK(Names(message, "EXL3"));
  CHECK(Names(message, "2495"));
  CHECK(Names(message, "item 4"));
  // NOT the accident this replaces. Before the refusal the loader fell through
  // to `get(la + "in_proj_qkv.weight")` and died on "tensor not found", a
  // sentence about a checkpoint that is complete.
  CHECK_FALSE(Names(message, "tensor not found"));

  // Through the registry seam as well, which is where a user meets it.
  const std::string routed = RegistryLoadFailure(ckpt, c);
  REQUIRE_FALSE(routed.empty());
  CHECK(Names(routed, "in_proj_qkv"));
  CHECK(Names(routed, "2495"));

  // AND IT IS NOT UNCONDITIONAL. The same dense arm with a BF16 GDN tower
  // loads, so this case measures the GDN projections rather than the presence
  // of any EXL3 weight at all.
  const TempCheckpoint ok(DenseCheckpoint(Arm::kExl3, g, /*exl3_gdn=*/false));
  CHECK(LoadFailure(ok, c).empty());
}

// G4 -------------------------------------------------------------------------
TEST_CASE("qwen3_5 exl3: the bf16, per-tensor FP8 and NVFP4 arms are UNCHANGED") {
  // A fourth rung in a presence-based resolver is exactly where an existing arm
  // gets mis-selected, and the mis-selection is silent: each of these
  // checkpoints still loads and still produces plausible numbers. So the arm
  // each one lands in is asserted directly.
  const Geometry g;
  const HfConfig c = DenseConfig(/*with_exl3_quant_config=*/false);

  SUBCASE("bf16 stays bf16") {
    const TempCheckpoint ckpt(DenseCheckpoint(Arm::kBf16, g));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
    const vllm::FullAttnLayerWeights& a = w.layers[1].attn;
    CHECK_FALSE(a.IsExl3());
    CHECK_FALSE(w.layers[1].mlp.IsExl3());
    CHECK_FALSE(w.layers[0].mlp.IsExl3());
    CHECK(w.lm_head_exl3.Empty());
    CHECK_FALSE(a.q_proj.Empty());
    CHECK(a.q_proj.nk);
    CHECK_FALSE(a.o_proj.Empty());
    CHECK_FALSE(w.layers[1].mlp.gate_up_proj.Empty());
    CHECK_FALSE(w.layers[1].mlp.down_proj.Empty());
    CHECK_FALSE(w.lm_head.Empty());
    // The direct-device staging path still claims a plain bf16 model, which is
    // the predicate the EXL3 rung had to be added to WITHOUT disturbing.
    CHECK(vllm::IsPlainBf16Qwen3_5Dense(w));
  }

  SUBCASE("per-tensor FP8 stays per-tensor FP8") {
    const TempCheckpoint ckpt(DenseCheckpoint(Arm::kFp8, g));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
    const vllm::FullAttnLayerWeights& a = w.layers[1].attn;
    CHECK_FALSE(a.IsExl3());
    CHECK_FALSE(a.q_proj_fp8.Empty());
    CHECK(a.q_proj_fp8.n == g.q_n);
    CHECK(a.q_proj_fp8.k == g.hidden);
    CHECK(a.q_proj.Empty());
    CHECK(a.q_proj_fp8_block.Empty());
    CHECK(a.q_proj_fp4.Empty());
    CHECK_FALSE(vllm::IsPlainBf16Qwen3_5Dense(w));
  }

  SUBCASE("ModelOpt NVFP4 stays NVFP4") {
    const TempCheckpoint ckpt(DenseCheckpoint(Arm::kNvfp4, g));
    const Qwen3_5DenseWeights w = LoadDense(ckpt, c);
    const vllm::FullAttnLayerWeights& a = w.layers[1].attn;
    CHECK_FALSE(a.IsExl3());
    CHECK_FALSE(a.q_proj_fp4.Empty());
    CHECK(a.q_proj_fp4.n == g.q_n);
    CHECK(a.q_proj_fp4.k == g.hidden);
    CHECK(a.q_proj.Empty());
    CHECK(a.q_proj_fp8.Empty());
    CHECK_FALSE(w.layers[1].mlp.gate_proj_fp4.Empty());
    CHECK_FALSE(vllm::IsPlainBf16Qwen3_5Dense(w));
  }
}
