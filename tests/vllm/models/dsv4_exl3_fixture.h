// MODEL-DSV4-EXL3 — the hermetic rank-sliced EXL3 checkpoint both this row's
// suites drive the PRODUCTION loader over.
//
// WHY IT IS SHARED, AND WHY THAT IS THE POINT OF W1c. Until #1923 the loader
// suite drove `vllm::LoadDeepseekV4ForCausalLMWeights` over a fixture whose
// carried tensors were one-element stubs, and the FORWARD suite did not use the
// loader at all: it built `DeepseekV4Weights` by hand and set
// `has_host_weights = true` itself. So the loader could set that flag nowhere,
// three reviews and a mutation pass could not see it, and a real `vllm-server`
// load generated zero tokens. One fixture, written at REAL dtypes and REAL
// shapes and read by BOTH suites, is what makes "the tower is reachable" a thing
// a mutation can falsify: delete the loader's materialization and the forward
// suite reds.
//
// Every dimension respects what the format actually requires: the trellis tile
// is 16x16 so both features are multiples of 16, and BOTH sides were
// Hadamard-128 transformed at quantization time (`exl3_lib/quantize.py:15`), so
// both the COALESCED and the PER-RANK features are multiples of 128. The carried
// half mirrors the real artifact's own dtypes, measured from
// `0xSero/deepseek-v4-flash-0731-spark` @ `22f28d32` on 2026-08-25 (safetensors
// header + a 16-byte range read, no download): BF16 norms / embeddings / router
// gate, F32 MHC mixing / attention sinks / noaux_tc bias, I64 `tid2eid`, and
// block-wise FP8 (`F8_E4M3` weight + `F8_E8M0` scale over 128x128 blocks) for
// every MLA and shared-expert linear.
#pragma once

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace dsv4_exl3_fixture {

// ── the fixture's own tiny safetensors writer ──────────────────────────────

struct StEntry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

// The residue, mod 8, that `WriteSafetensors` forces the payload base to. ODD
// on purpose: see the comment inside the writer. Every entry at an even
// data_offset then starts at an address that satisfies NO alignment above 1.
constexpr size_t kMisalignedPayloadBase = 1;

inline std::string WriteSafetensors(const std::filesystem::path& path,
                                    const std::vector<StEntry>& entries) {
  nlohmann::json header = nlohmann::json::object();
  header["__metadata__"] = {{"format", "pt"}};
  uint64_t offset = 0;
  std::vector<uint8_t> data;
  for (const StEntry& e : entries) {
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {offset, offset + e.bytes.size()}}};
    data.insert(data.end(), e.bytes.begin(), e.bytes.end());
    offset += e.bytes.size();
  }
  std::string text = header.dump();
  // DELIBERATELY MISALIGN THE PAYLOAD (#1923 follow-up). A safetensors payload
  // begins at `8 + header_bytes`, and nothing in the format makes that a
  // multiple of any element's alignment — HuggingFace's own writer pads the
  // header with spaces to reach 8, but a file that does not is well formed and
  // the reader mmaps it at a page boundary either way. So a loader that forms a
  // `const uint16_t*` or a `const int64_t*` into the payload is undefined, which
  // x86 executes anyway and only UBSan reports.
  //
  // A fixture whose payload happened to land aligned would make that bug
  // invisible to every local run. This pads the header with trailing SPACES
  // (JSON whitespace, so the file stays readable by any safetensors reader)
  // until the payload base is ODD. An odd base is misaligned for every element
  // type wider than one byte, so each entry at an even offset is a live test of
  // the loader's alignment contract. `kMisalignedPayloadBase` above is the
  // guarantee a test asserts against, so this cannot silently become a no-op.
  while ((8 + text.size()) % 8 != kMisalignedPayloadBase) text.push_back(' ');
  std::ofstream out(path, std::ios::binary);
  const uint64_t n = text.size();
  for (int i = 0; i < 8; ++i) {
    const char byte = static_cast<char>((n >> (8 * i)) & 0xff);
    out.write(&byte, 1);
  }
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
  if (!out) throw std::runtime_error("failed to write fixture safetensors");
  const auto u8 = path.u8string();
  return std::string(u8.begin(), u8.end());
}

template <typename T>
std::vector<uint8_t> Raw(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
  return b;
}

class TempDir {
 public:
  TempDir() {
    static std::atomic<uint64_t> counter{0};
    static const uint64_t nonce = [] {
      std::random_device rd;
      return (static_cast<uint64_t>(rd()) << 32) ^ rd();
    }();
    path_ = std::filesystem::temp_directory_path() /
            ("dsv4_exl3_" + std::to_string(nonce) + "_" +
             std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

// ── the fixture's geometry ─────────────────────────────────────────────────
constexpr int64_t kHidden = 256;    // hidden_size = w1/w3 in, w2 out
constexpr int64_t kInter = 512;     // moe_intermediate_size = w1/w3 out, w2 in
constexpr int kTp = 4;              // 512 / 4 = 128 per rank: one Hadamard block
constexpr int kBits = 3;            // 3.0 bpw -> K = 3, last trellis dim = 48
constexpr int kExperts = 2;
constexpr int kLayers = 1;
constexpr int64_t kVocab = 32;
constexpr int64_t kHeadDim = 512;   // ParseDeepseekV4Params scopes the 512-wide MLA only
constexpr int64_t kQLora = 128;
constexpr int64_t kOLora = 128;
constexpr int64_t kOGroups = 1;
constexpr int64_t kHeads = 1;
constexpr int64_t kHcMult = 2;
// The artifact's own `weight_block_size` for the carried FP8 half.
constexpr int64_t kBlockN = 128;
constexpr int64_t kBlockK = 128;

inline int64_t TrellisElems(int64_t k, int64_t n) {
  return (k / 16) * (n / 16) * (16 * kBits);
}

// Deterministic, position-dependent contents so a misplaced slice cannot alias.
inline uint16_t TrellisWord(int expert, int proj, int rank, int64_t index) {
  const uint64_t h = 0x9E3779B97F4A7C15ull *
                     (static_cast<uint64_t>(index) * 131u + rank * 7919u +
                      proj * 104729u + expert * 1299709u + 1u);
  return static_cast<uint16_t>((h >> 27) & 0xffffu);
}
// `suh`/`svh` carry the per-channel scale, not a bare sign (`exl3.py:38`: "scale
// is no longer used"). BOUNDED on purpose: an earlier form of this generator
// masked a hash to `0x7bff`, which is any finite fp16 up to 65504, and a tower
// built from those decodes to weights large enough to take a forward to NaN in
// one MoE block. 32 distinct magnitudes over two signs still make a misplaced
// rank slice astronomically unlikely to alias, which is all the byte-parity case
// needs from this function.
inline uint16_t SignWord(int expert, int proj, int rank, int64_t index, int side) {
  const uint64_t h = 0xD6E8FEB86659FD93ull *
                     (static_cast<uint64_t>(index) * 31u + rank * 1237u +
                      proj * 7717u + expert * 65537u + side * 4099u + 1u);
  const float mag = 0.5f + static_cast<float>((h >> 31) & 0xfu) / 32.0f;
  const float sign = ((h >> 35) & 1u) != 0u ? -1.0f : 1.0f;
  return vt::F32ToF16(sign * mag);
}

// ── the carried half's deterministic contents ──────────────────────────────
//
// Keyed on the TENSOR NAME as well as the index, so no two carried tensors hold
// the same bytes and a routing that swapped two of them cannot pass. Exposed
// rather than private because the loader suite recomputes them to check what the
// materialization wrote, and a fixture that could not be recomputed would leave
// the materialization gated only by "it did not throw".
inline uint32_t NameHash(const std::string& name, int64_t index) {
  uint64_t h = 1469598103934665603ull;
  for (char c : name) {
    h ^= static_cast<unsigned char>(c);
    h *= 1099511628211ull;
  }
  h ^= static_cast<uint64_t>(index) + 0x9E3779B97F4A7C15ull;
  h *= 1099511628211ull;
  return static_cast<uint32_t>(h >> 29);
}

// A value in [-scale, scale), plus `center` — norms want ~1.0, everything else 0.
inline float CarriedValue(const std::string& name, int64_t index, float scale,
                          float center) {
  const uint32_t h = NameHash(name, index);
  const float u = (static_cast<float>(h & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
  return center + u * scale;
}

// One safe IEEE fp8-e4m3fn byte: exponent field 5..7 (magnitude 0.25 .. 1.875),
// never the subnormal field 0 and never the 0x7F/0xFF NaN encodings.
inline uint8_t CarriedFp8Byte(const std::string& name, int64_t index) {
  const uint32_t h = NameHash(name, index);
  const uint32_t sign = (h >> 20) & 1u;
  const uint32_t exp = 5u + ((h >> 3) % 3u);
  const uint32_t man = h & 7u;
  return static_cast<uint8_t>((sign << 7) | (exp << 3) | man);
}
// One UE8M0 block-scale byte: 2^(byte-127) over {0.0625 .. 1}, so a decoded
// weight lands around 0.015 .. 1.9 — the magnitude range the hand-built forward
// fixture this replaces used for its dense projections.
//
// FIVE values, not three, and the reason is a MEASUREMENT rather than taste: at
// three the two 128-block scales of `layers.0.attn.wq_a` collided, and the
// mutation that makes the decode read block [0] for every element left the
// value-parity case GREEN (ninja rc=0, 5 steps; loader 10/10 131/131 SUCCESS).
// The case now asserts the precondition it depends on, so a future narrowing
// fails as a broken instrument rather than as a silent pass.
inline uint8_t CarriedScaleByte(const std::string& name, int64_t index) {
  return static_cast<uint8_t>(123 + (NameHash(name, index) % 5u));
}

inline int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

inline StEntry Bf16Entry(const std::string& name, const std::vector<int64_t>& shape,
                         float scale, float center) {
  const int64_t n = Numel(shape);
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = vt::F32ToBF16(CarriedValue(name, i, scale, center));
  return {name, "BF16", shape, Raw(v)};
}
inline StEntry F32Entry(const std::string& name, const std::vector<int64_t>& shape,
                        float scale, float center) {
  const int64_t n = Numel(shape);
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = CarriedValue(name, i, scale, center);
  return {name, "F32", shape, Raw(v)};
}
inline StEntry I64Entry(const std::string& name, const std::vector<int64_t>& shape,
                        int64_t modulo) {
  const int64_t n = Numel(shape);
  std::vector<int64_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = static_cast<int64_t>(NameHash(name, i) % modulo);
  return {name, "I64", shape, Raw(v)};
}
// `<base>.weight` F8_E4M3 [N,K] plus `<base>.scale` F8_E8M0 over 128x128 blocks.
inline std::vector<StEntry> Fp8BlockEntries(const std::string& base, int64_t N,
                                            int64_t K) {
  const std::string wname = base + ".weight";
  const std::string sname = base + ".scale";
  std::vector<uint8_t> w(static_cast<size_t>(N * K));
  for (int64_t i = 0; i < N * K; ++i) w[static_cast<size_t>(i)] = CarriedFp8Byte(wname, i);
  const int64_t nb = (N + kBlockN - 1) / kBlockN;
  const int64_t kb = (K + kBlockK - 1) / kBlockK;
  std::vector<uint8_t> s(static_cast<size_t>(nb * kb));
  for (int64_t i = 0; i < nb * kb; ++i) s[static_cast<size_t>(i)] = CarriedScaleByte(sname, i);
  return {{wname, "F8_E4M3", {N, K}, w}, {sname, "F8_E8M0", {nb, kb}, s}};
}

inline std::string Base(int layer, int expert, const char* proj) {
  return "layers." + std::to_string(layer) + ".ffn.experts." +
         std::to_string(expert) + "." + proj;
}

// w1 and w3 split on OUT features; w2 splits on IN (exl3.py:296-313).
inline bool SplitsOut(const char* proj) { return std::strcmp(proj, "w2") != 0; }

struct FixtureOptions {
  std::string version = "rank-sliced-deepseek-v4-v1";
  std::string codebook = "mcg";
  double bits = 3.0;
  int tp = kTp;
  int ranks_written = kTp;         // < tp leaves a rank missing
  std::string drop_tensor;         // one EXL3 tensor to omit entirely
  std::string extra_carried;       // one unroutable carried tensor to add
  bool swap_w2_slice_axis = false; // write w2 sliced on OUT instead of IN
  // ── the NEGATIVE direction of the detection predicate ────────────────────
  // `quantization_config.quant_method`. Anything but "exl3" must take the
  // pre-existing dense arm, and the vehicle that proves it has to be a REAL
  // one: `deepseek_v4_fp8` carries a `quantization_config` of its own.
  std::string quant_method = "exl3";
  bool omit_quant_config = false;      // write no `quantization_config` block
  bool dense_routed_experts = false;   // dense NVFP4 experts, no rank shards
  // ── W1c: the shape of the model the carried half describes ───────────────
  int layers = kLayers;
  int64_t num_hash_layers = 0;
  int64_t topk = 1;
  std::vector<int64_t> compress_ratios{0};
  int64_t index_n_heads = 0;
  int64_t index_head_dim = 0;
  int64_t index_topk = 0;
  // Drop `base_quantization_config`, so the loader has no recipe for the
  // carried FP8 half and must refuse rather than assume a block size.
  bool omit_base_quant_config = false;
  // Write the DSA family at the REAL artifact's PER-LAYER geometry instead of
  // the collapsed one the host forward indexes (#1970).
  //
  // Upstream derives the whole thing from one line,
  // `coff = 1 + (compress_ratio == 4)`
  // (`vllm/models/deepseek_v4/compressor.py:247-248` at the parity pin
  // `5559679229bc961848b121ccdeaa8fa5d79bec98`), and spends it on the APE table
  // (`:270-277`) and on the fused `wkv|wgate` projection (`:279-287`) — and NOT
  // on the norm, which is `RMSNorm(self.head_dim, self.rms_norm_eps)` (`:288`).
  // The indexer carries
  // its OWN compressor at `head_dim = index_head_dim` and the same ratio
  // (`attention.py:768-776`), so its family doubles too; and its `wq_b` is
  // `ReplicatedLinear(q_lora_rank, head_dim * n_head)` (`:721-726`), whose K is
  // `q_lora_rank` rather than `hidden_size` — not a width at all, but the
  // q-LoRA input space our forward does not project from.
  //
  // This REPLACES `real_compressor_width`, which doubled UNCONDITIONALLY. The
  // one case that used it wrote a doubled compressor on a `cr == 128` layer,
  // where upstream's `coff` is 1 — a width the artifact does not store and the
  // pin does not produce. Keying on `compress_ratio` is what stops this fixture
  // from describing a geometry no oracle would emit.
  //
  // LEAVING THIS FALSE AT `cr == 4` IS NOT A NEUTRAL DEFAULT. It writes the
  // collapsed, UNDOUBLED family, which upstream's `coff` cannot produce at that
  // ratio and which the loader therefore refuses (#1970). A case that wants a
  // compressor layer the host forward can actually run uses `cr == 128`, where
  // `coff` is 1 and the derived width IS the collapsed one.
  bool real_dsa_geometry = false;
  // A THIRD width, which is neither upstream's `coff` width nor the collapsed
  // one. The loader DERIVES one width and refuses everything else, and a flag
  // that only ever toggled between `coff` and collapsed could not tell a derived
  // rule from a two-value allow-list — so this writes a width no oracle produces
  // and no forward indexes, and the loader must still refuse it BY NAME.
  bool bogus_dsa_width = false;
  // `indexer.wq_b` at `K = hidden_size` while the REST of the family is at the
  // real geometry. Its K is `q_lora_rank` upstream
  // (`attention.py:721-726`, called on `qr` at `:835`) and is NOT a `coff`
  // width, so it is derived by a DIFFERENT rule from every other DSA tensor and
  // needs its own case: without this the compressor refuses first and the
  // `wq_b` check is never reached, which makes it unfalsifiable.
  // `kHidden` (256) and `kQLora` (128) differ, so this is a real distinction.
  bool collapsed_indexer_wq_b = false;
  // The indexer's OWN `DeepseekCompressor` KV projection at the COLLAPSED
  // `index_head_dim` while the rest of the family is at the real geometry.
  // Without it the main compressor refuses FIRST and the loader's
  // `coff * index_head_dim` derivation is never read, which leaves that
  // derivation's message unfalsifiable — deleting the check kept both suites
  // green. Same shape of hole as `collapsed_indexer_wq_b`, one tensor over.
  bool collapsed_indexer_wkv = false;

  int64_t compress_ratio(int layer) const {
    return layer < static_cast<int>(compress_ratios.size())
               ? compress_ratios[static_cast<size_t>(layer)]
               : 0;
  }
  bool has_compressor(int layer) const { return compress_ratio(layer) != 0; }
  bool has_indexer(int layer) const { return compress_ratio(layer) == 4; }
  // `compressor.py:247-248`, verbatim. 1 unless the real geometry is requested
  // AND this layer is one of the overlapping `cr == 4` ones.
  int64_t coff(int layer) const {
    if (bogus_dsa_width) return 3;  // neither width; must refuse
    return (real_dsa_geometry && compress_ratio(layer) == 4) ? 2 : 1;
  }
  bool is_hash_layer(int layer) const { return layer < num_hash_layers; }
};

inline nlohmann::json FixtureConfigJson(const FixtureOptions& opt) {
  nlohmann::json raw = nlohmann::json::object();
  raw["architectures"] = nlohmann::json::array({"DeepseekV4ForCausalLM"});
  raw["model_type"] = "deepseek_v4";
  raw["hidden_size"] = kHidden;
  raw["num_hidden_layers"] = opt.layers;
  raw["vocab_size"] = kVocab;
  raw["num_attention_heads"] = kHeads;
  raw["num_key_value_heads"] = 1;
  raw["head_dim"] = kHeadDim;
  raw["qk_rope_head_dim"] = 64;
  raw["q_lora_rank"] = kQLora;
  raw["o_lora_rank"] = kOLora;
  raw["o_groups"] = kOGroups;
  raw["rms_norm_eps"] = 1e-6;
  raw["tie_word_embeddings"] = false;
  raw["max_position_embeddings"] = 128;
  raw["num_nextn_predict_layers"] = 1;
  raw["n_routed_experts"] = kExperts;
  raw["num_experts_per_tok"] = opt.topk;
  raw["moe_intermediate_size"] = kInter;
  raw["n_shared_experts"] = 1;
  raw["norm_topk_prob"] = true;
  raw["routed_scaling_factor"] = 1.5;
  raw["swiglu_limit"] = 10.0;
  raw["scoring_func"] = "sqrtsoftplus";
  raw["topk_method"] = "noaux_tc";
  raw["num_hash_layers"] = opt.num_hash_layers;
  raw["expert_dtype"] = "fp4";
  raw["hc_mult"] = kHcMult;
  raw["hc_sinkhorn_iters"] = 20;
  raw["hc_eps"] = 1e-6;
  raw["index_head_dim"] = opt.index_head_dim;
  raw["index_n_heads"] = opt.index_n_heads;
  raw["index_topk"] = opt.index_topk;
  raw["compress_ratios"] = opt.compress_ratios;
  // The carried (non-expert) half's own recipe, copied in shape from the real
  // artifact's `quantization_config.base_quantization_config`.
  const nlohmann::json base_quant = {
      {"quant_method", "fp8"},
      {"activation_scheme", "dynamic"},
      {"fmt", "e4m3"},
      {"scale_fmt", "ue8m0"},
      {"weight_block_size", nlohmann::json::array({kBlockN, kBlockK})}};
  if (opt.omit_quant_config) {
    // No block at all: the unquantized vehicle, and the one shape a predicate
    // that dropped its null guard would misread.
  } else if (opt.quant_method == "exl3") {
    raw["quantization_config"] = {
        {"quant_method", "exl3"},
        {"version", opt.version},
        {"bits", opt.bits},
        {"codebook", opt.codebook},
        {"source_format", "packed_e2m1_fp4_with_ue8m0_scales"}};
    if (!opt.omit_base_quant_config)
      raw["quantization_config"]["base_quantization_config"] = base_quant;
  } else {
    // The plain `deepseek_v4_fp8` block, copied in shape from the REAL
    // artifact's own `quantization_config.base_quantization_config`
    // (`0xSero/deepseek-v4-flash-0731-spark` @ `22f28d32`, config.json):
    // `{activation_scheme: dynamic, fmt: e4m3, quant_method: fp8,
    //   scale_fmt: ue8m0, weight_block_size: [128, 128]}`. It carries NO
    // `version` and NO `codebook`, which is exactly why a widened detection
    // predicate would carry it into the EXL3 arm and die there instead of
    // loading.
    raw["quantization_config"] = base_quant;
    raw["quantization_config"]["quant_method"] = opt.quant_method;
  }
  raw["hybrid_tr3_tail"] = {
      {"tp", opt.tp},
      {"format", "exl3-trellis"},
      {"tensor_schema",
       "layers.{L}.ffn.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"}};
  return raw;
}

inline vllm::HfConfig FixtureConfig(const FixtureOptions& opt) {
  vllm::HfConfig c;
  c.model_type = "deepseek_v4";
  c.architectures = {"DeepseekV4ForCausalLM"};
  c.hidden_size = kHidden;
  c.num_hidden_layers = opt.layers;
  c.vocab_size = kVocab;
  c.num_attention_heads = kHeads;
  c.head_dim = kHeadDim;
  c.torch_dtype = "bfloat16";
  c.raw = FixtureConfigJson(opt);
  return c;
}

// The carried half: the un-requantized DeepSeek-V4 source tensors, by the exact
// names AND the exact dtypes the real `carried-*.safetensors` use. W1b wrote a
// one-element stub per tensor, which was enough for an accounting pass and is
// NOT enough for a loader that materializes — the shapes below are what the host
// forward indexes, and getting one wrong is what the loader's shape refusal
// exists to catch.
inline std::vector<StEntry> CarriedEntries(const FixtureOptions& opt) {
  std::vector<StEntry> e;
  const int64_t H = kHidden;
  const int64_t hc = kHcMult;
  const int64_t hc3 = (2 + hc) * hc;
  const int64_t hcH = hc * H;
  const int64_t in_per_group = kHeads * kHeadDim / kOGroups;
  const auto push = [&](const StEntry& x) { e.push_back(x); };
  const auto push_all = [&](const std::vector<StEntry>& xs) {
    e.insert(e.end(), xs.begin(), xs.end());
  };

  push(Bf16Entry("embed.weight", {kVocab, H}, 0.8f, 0.0f));
  push(Bf16Entry("norm.weight", {H}, 0.1f, 1.0f));
  push(Bf16Entry("head.weight", {kVocab, H}, 0.5f, 0.0f));
  push(F32Entry("hc_head_base", {hc}, 0.2f, 0.0f));
  push(F32Entry("hc_head_fn", {hc, hcH}, 0.2f, 0.0f));
  push(F32Entry("hc_head_scale", {1}, 0.1f, 0.5f));
  for (int l = 0; l < opt.layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    push(Bf16Entry(b + "attn_norm.weight", {H}, 0.1f, 1.0f));
    push(Bf16Entry(b + "ffn_norm.weight", {H}, 0.1f, 1.0f));
    for (const char* h : {"hc_attn_base", "hc_ffn_base"})
      push(F32Entry(b + h, {hc3}, 0.2f, 0.0f));
    for (const char* h : {"hc_attn_fn", "hc_ffn_fn"})
      push(F32Entry(b + h, {hc3, hcH}, 0.2f, 0.0f));
    for (const char* h : {"hc_attn_scale", "hc_ffn_scale"})
      push(F32Entry(b + h, {3}, 0.5f, 0.0f));

    const std::string a = b + "attn.";
    push_all(Fp8BlockEntries(a + "wq_a", kQLora, H));
    push_all(Fp8BlockEntries(a + "wq_b", kHeads * kHeadDim, kQLora));
    push_all(Fp8BlockEntries(a + "wkv", kHeadDim, H));
    push_all(Fp8BlockEntries(a + "wo_a", kOGroups * kOLora, in_per_group));
    push_all(Fp8BlockEntries(a + "wo_b", H, kOGroups * kOLora));
    push(Bf16Entry(a + "q_norm.weight", {kQLora}, 0.1f, 1.0f));
    push(Bf16Entry(a + "kv_norm.weight", {kHeadDim}, 0.1f, 1.0f));
    push(F32Entry(a + "attn_sink", {kHeads}, 0.7f, 0.0f));

    if (opt.has_compressor(l)) {
      // The REAL artifact stores the compressor family at `coff * head_dim`,
      // where `coff = 1 + (compress_ratio == 4)`
      // (`vllm/models/deepseek_v4/compressor.py:247-248`); the collapsed
      // synthetic geometry the host forward indexes is `head_dim`.
      // `real_dsa_geometry` writes the former ON THE cr == 4 LAYERS ONLY, which
      // is where upstream's `overlap` is true and where the real artifact's four
      // refusing tensors live.
      //
      // `norm.weight` is DELIBERATELY not widened: upstream is
      // `RMSNorm(self.head_dim, self.rms_norm_eps)` (`:288`), so a doubled norm
      // would be a shape the artifact does not carry and the loader must not
      // learn to expect.
      const int64_t cw = opt.coff(l) * kHeadDim;
      push(F32Entry(a + "compressor.ape", {opt.compress_ratio(l), cw}, 0.2f, 0.0f));
      push(Bf16Entry(a + "compressor.norm.weight", {kHeadDim}, 0.1f, 1.0f));
      push(Bf16Entry(a + "compressor.wgate.weight", {cw, H}, 0.3f, 0.0f));
      push(Bf16Entry(a + "compressor.wkv.weight", {cw, H}, 0.3f, 0.0f));
    }
    if (opt.has_indexer(l)) {
      const int64_t ihd = opt.index_head_dim;
      const int64_t inh = opt.index_n_heads;
      // The indexer's own `DeepseekCompressor` runs at `head_dim =
      // index_head_dim` with the SAME ratio (`attention.py:768-776`), so the
      // same `coff` applies to its family and its norm is likewise undoubled.
      const int64_t iw = opt.coff(l) * ihd;
      push(F32Entry(a + "indexer.compressor.ape", {4, iw}, 0.2f, 0.0f));
      push(Bf16Entry(a + "indexer.compressor.norm.weight", {ihd}, 0.1f, 1.0f));
      push(Bf16Entry(a + "indexer.compressor.wgate.weight", {iw, H}, 0.3f, 0.0f));
      push(Bf16Entry(a + "indexer.compressor.wkv.weight",
                     {opt.collapsed_indexer_wkv ? ihd : iw, H}, 0.3f, 0.0f));
      push(Bf16Entry(a + "indexer.weights_proj.weight", {inh, H}, 0.3f, 0.0f));
      // NOT a width. `wq_b` is `ReplicatedLinear(q_lora_rank, head_dim *
      // n_head)` (`attention.py:721-726`) called on `qr` (`:835`), so its K is
      // `q_lora_rank`. The collapsed fixture writes `H` because our forward
      // feeds it the hidden state.
      push_all(Fp8BlockEntries(
          a + "indexer.wq_b", inh * ihd,
          (opt.real_dsa_geometry && !opt.collapsed_indexer_wq_b) ? kQLora : H));
    }

    const std::string f = b + "ffn.";
    push(Bf16Entry(f + "gate.weight", {kExperts, H}, 0.4f, 0.0f));
    if (opt.is_hash_layer(l))
      push(I64Entry(f + "gate.tid2eid", {kVocab, opt.topk}, kExperts));
    else
      push(F32Entry(f + "gate.bias", {kExperts}, 0.3f, 0.0f));
    push_all(Fp8BlockEntries(f + "shared_experts.w1", kInter, H));
    push_all(Fp8BlockEntries(f + "shared_experts.w2", H, kInter));
    push_all(Fp8BlockEntries(f + "shared_experts.w3", kInter, H));
  }
  // The real artifact carries an MTP tail. vLLM's DeepSeek-V4 loader skips it
  // (`AutoWeightsLoader(skip_substrs=["mtp."])`, nvidia/model.py:1474) and so
  // must this arm, WITHOUT reporting the tensors as unroutable.
  push(F32Entry("mtp.0.attn_norm.weight", {1}, 0.25f, 0.0f));
  push(F32Entry("mtp.0.ffn.experts.0.w1.weight", {1}, 0.25f, 0.0f));
  // The NON-EXL3 vehicle's routed experts: dense NVFP4, the four suffixes the
  // pre-existing arm's name-map requires for `expert_dtype == "fp4"` (the
  // `expert_suffixes` vector in `LoadDeepseekV4ForCausalLMWeights`,
  // `src/vllm/model_executor/models/deepseek_v4_weights.cpp`). Cited by SYMBOL
  // deliberately: the commit that first wrote this comment cited a line range,
  // and the SAME commit inserted 71 lines above it, so the anchor was stale
  // before it was ever read. Present only for the negative-direction cases,
  // where there are no rank shards at all and nothing materializes.
  if (opt.dense_routed_experts) {
    for (int l = 0; l < opt.layers; ++l) {
      const std::string f = "layers." + std::to_string(l) + ".ffn.experts.";
      for (int x = 0; x < kExperts; ++x)
        for (const char* w : {"w1", "w2", "w3"})
          for (const char* suf : {".weight", ".weight_scale", ".weight_scale_2",
                                  ".input_scale"})
            push(F32Entry(f + std::to_string(x) + "." + w + suf, {1}, 0.25f, 0.0f));
    }
  }
  if (!opt.extra_carried.empty())
    push(F32Entry(opt.extra_carried, {1}, 0.25f, 0.0f));
  return e;
}

// One rank shard, sliced exactly as `tp_import_split` would have produced it.
inline std::vector<StEntry> RankEntries(int rank, const FixtureOptions& opt) {
  std::vector<StEntry> e;
  const std::string suffix = ".rank" + std::to_string(rank);
  for (int l = 0; l < opt.layers; ++l) {
    for (int x = 0; x < kExperts; ++x) {
      int proj_index = 0;
      for (const char* proj : {"w1", "w2", "w3"}) {
        const std::string base = Base(l, x, proj) + suffix;
        const bool out_split = opt.swap_w2_slice_axis ? true : SplitsOut(proj);
        const int64_t full_in = SplitsOut(proj) ? kHidden : kInter;
        const int64_t full_out = SplitsOut(proj) ? kInter : kHidden;
        const int64_t k = out_split ? full_in : full_in / opt.tp;
        const int64_t n = out_split ? full_out / opt.tp : full_out;

        std::vector<uint16_t> trellis(static_cast<size_t>(TrellisElems(k, n)));
        for (size_t i = 0; i < trellis.size(); ++i)
          trellis[i] = TrellisWord(x + l * kExperts, proj_index, rank,
                                   static_cast<int64_t>(i));
        std::vector<uint16_t> suh(static_cast<size_t>(k)), svh(static_cast<size_t>(n));
        // The INVARIANT side is identical across ranks (every rank received the
        // whole vector); the SPLIT side differs per rank.
        for (int64_t i = 0; i < k; ++i)
          suh[static_cast<size_t>(i)] =
              SignWord(x + l * kExperts, proj_index, out_split ? 0 : rank, i, 0);
        for (int64_t j = 0; j < n; ++j)
          svh[static_cast<size_t>(j)] =
              SignWord(x + l * kExperts, proj_index, out_split ? rank : 0, j, 1);

        const std::vector<std::pair<std::string, StEntry>> four = {
            {base + ".trellis",
             {base + ".trellis", "I16", {k / 16, n / 16, 16 * kBits}, Raw(trellis)}},
            {base + ".suh", {base + ".suh", "F16", {k}, Raw(suh)}},
            {base + ".svh", {base + ".svh", "F16", {n}, Raw(svh)}},
            {base + ".mcg",
             {base + ".mcg", "I32", {}, Raw(std::vector<int32_t>{-877912083})}},
        };
        for (const auto& [name, entry] : four)
          if (name != opt.drop_tensor) e.push_back(entry);
        ++proj_index;
      }
    }
  }
  return e;
}

// The bytes the COALESCED TP1 tower holds, in one place so the byte-parity case
// and the residency case cannot drift apart. Per expert: w1 and w3 are
// [kHidden, kInter] and w2 is [kInter, kHidden]; each linear is its trellis plus
// its two fp16 sign vectors, every element 16-bit.
inline int64_t ExpectedTowerBytes(int layers = kLayers) {
  int64_t bytes = 0;
  for (int x = 0; x < kExperts; ++x) {
    bytes += (TrellisElems(kHidden, kInter) + kHidden + kInter) * 2 * 2;
    bytes += (TrellisElems(kInter, kHidden) + kInter + kHidden) * 2;
  }
  return bytes * layers;
}

struct Fixture {
  TempDir dir;
  std::vector<vllm::SafetensorsFile> shards;
  vllm::HfConfig config;
};

inline std::unique_ptr<Fixture> BuildFixture(const FixtureOptions& opt = {}) {
  auto f = std::make_unique<Fixture>();
  f->config = FixtureConfig(opt);
  f->shards.push_back(vllm::SafetensorsFile::Open(
      WriteSafetensors(f->dir.path() / "carried-001.safetensors", CarriedEntries(opt))));
  const int rank_shards = opt.dense_routed_experts ? 0 : opt.ranks_written;
  for (int r = 0; r < rank_shards; ++r) {
    f->shards.push_back(vllm::SafetensorsFile::Open(WriteSafetensors(
        f->dir.path() / ("exl3-layer-000-tp4-rank" + std::to_string(r) + ".safetensors"),
        RankEntries(r, opt))));
  }
  return f;
}

// ── shared assertion plumbing ──────────────────────────────────────────────

// doctest::Contains has no `operator&&`, and a refusal is only useful if the
// message names BOTH the offending tensor and the row that owes the arm — so
// capture the message once and assert over it.
inline std::string ThrowMessage(const std::function<void()>& body) {
  try {
    body();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}
inline bool Mentions(const std::string& message, const std::string& needle) {
  return message.find(needle) != std::string::npos;
}

// REAL fd 2 by dup/dup2, not a `std::cerr` rdbuf swap: the residency line is
// written with `std::fprintf(stderr, ...)`, which an rdbuf swap cannot see. A
// capture that could not see the line it exists to read would return an empty
// string and be indistinguishable from "the loader never emitted it", which is
// the instrument failing toward a verdict about the code
// (`.agents/verification.md`; the same reasoning as
// `tests/vllm/v1/spec_decode/dflash2_runner_fixture.h:468`).
inline std::string CaptureStderr(const std::function<void()>& body) {
  std::FILE* cap = std::tmpfile();
  REQUIRE(cap != nullptr);
  std::fflush(stderr);
  const int saved = ::dup(STDERR_FILENO);
  REQUIRE(saved >= 0);
  REQUIRE(::dup2(::fileno(cap), STDERR_FILENO) >= 0);
  body();
  std::fflush(stderr);
  const int restored = ::dup2(saved, STDERR_FILENO);
  ::close(saved);
  std::rewind(cap);
  std::string out;
  char buf[4096];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), cap)) > 0) out.append(buf, n);
  std::fclose(cap);
  REQUIRE(restored >= 0);
  return out;
}

// The GiB figure the LOAD printed after `host MemAvailable=`, or -1.0 when the
// line took the unknown branch (or was never emitted). Parsed rather than
// string-matched so the assertion can compare the reported budget against the
// pool it claims to have measured — a substring check for "MemAvailable" alone
// matches both branches of the report, which is exactly the hole MINOR-1 found.
inline double ReportedMemAvailableGiB(const std::string& log) {
  static const std::string kKey = "host MemAvailable=";
  const size_t at = log.find(kKey);
  if (at == std::string::npos) return -1.0;
  return std::strtod(log.c_str() + at + kKey.size(), nullptr);
}

}  // namespace dsv4_exl3_fixture
