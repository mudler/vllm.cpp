// LTX-2.5 IC-LoRA — the adapter format, its metadata, and the fusion arithmetic.
//
// Row LTX25-IC-LORA, .agents/specs/ltx25-ic-lora.md, issue #923.
//
// WHAT GATES WHAT, because two of these look alike and are not:
//
//  * THE ARITHMETIC. `W + (B * strength) @ A`, checked against values computed
//    from upstream's own expression (`ltx-core loader/fuse_loras.py:99-116`) at
//    Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca. There is no
//    upstream test to port — the pinned repository ships NONE, measured with a
//    positive control in the row's spec §5.1 — so this is a source-derived value
//    gate and is labelled as one rather than called a ported test.
//
//  * THE ACCUMULATOR'S DTYPE. Upstream aggregates in BF16 in all four of its
//    fuse rules. A token gate cannot see an accumulator that is too wide, and
//    neither can the arithmetic case above at ordinary magnitudes — so one case
//    here is built specifically so that f32 and bf16 accumulation DISAGREE in
//    the stored result, and it is the only thing standing between this port and
//    a silently more precise path than the one it mirrors.
//
// The quantized arms are gated in test_ltx2_loader.cpp, where the synthetic FP8
// and NVFP4 DiT builders live, and reachability in test_ltx2_video.cpp, where
// the engine fixture does.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "vllm/model_executor/models/ltx2_lora.h"
#include "vt/dtype.h"

namespace {

// ── a minimal safetensors writer ─────────────────────────────────────────────
//
// Local rather than shared: a LoRA file is two rank-2 BF16 tensors and a
// `__metadata__` object, and the fixtures that build whole DiTs are far heavier
// than that. Bytes are laid out exactly as the reader expects, so a malformed
// case here is malformed for the reason the case names.

struct LoraEntry {
  std::string name;
  std::string dtype;  // "BF16" | "F32" | "U8"
  std::vector<int64_t> shape;
  std::vector<float> values;
};

std::string TempPath(const char* tag) {
  static int counter = 0;
  return std::string("/tmp/ltx2_lora_test_") + tag + "_" + std::to_string(getpid()) + "_" +
         std::to_string(++counter) + ".safetensors";
}

void WriteLoraFile(const std::vector<LoraEntry>& entries,
                   const std::map<std::string, std::string>& metadata,
                   const std::string& path) {
  std::string header = "{";
  bool first = true;
  if (!metadata.empty()) {
    header += "\"__metadata__\":{";
    bool mfirst = true;
    for (const auto& kv : metadata) {
      if (!mfirst) header += ",";
      mfirst = false;
      header += "\"" + kv.first + "\":\"" + kv.second + "\"";
    }
    header += "}";
    first = false;
  }
  std::string payload;
  for (const LoraEntry& e : entries) {
    const size_t at = payload.size();
    size_t bytes = 0;
    if (e.dtype == "F32") {
      bytes = e.values.size() * sizeof(float);
      payload.resize(at + bytes);
      std::memcpy(&payload[at], e.values.data(), bytes);
    } else if (e.dtype == "BF16") {
      bytes = e.values.size() * sizeof(uint16_t);
      for (const float v : e.values) {
        const uint16_t b = vt::F32ToBF16(v);
        payload.append(reinterpret_cast<const char*>(&b), sizeof(b));
      }
    } else {
      bytes = e.values.size();
      for (const float v : e.values) payload.push_back(static_cast<char>(v));
    }
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      header += (i != 0 ? "," : "") + std::to_string(e.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(at) + "," +
              std::to_string(at + bytes) + "]}";
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";
  const uint64_t n = header.size();
  std::string file(reinterpret_cast<const char*>(&n), sizeof(n));
  file += header;
  file += payload;
  FILE* f = std::fopen(path.c_str(), "wb");
  REQUIRE(f != nullptr);
  std::fwrite(file.data(), 1, file.size(), f);
  std::fclose(f);
}

// The one target every simple case uses. A real contract name, so the
// unknown-target refusal is genuinely testing name resolution and not a typo.
const char* const kTarget = "transformer_blocks.0.attn1.to_q.weight";
const char* const kModule = "transformer_blocks.0.attn1.to_q";

// Write a rank-`rank` adapter for `kModule`, with `b` [out, rank] and `a`
// [rank, in] given row-major.
std::string WriteAdapter(int64_t out_features, int64_t rank, int64_t in_features,
                         const std::vector<float>& b, const std::vector<float>& a,
                         const std::map<std::string, std::string>& metadata = {},
                         const std::string& module = kModule,
                         const std::string& prefix = "diffusion_model.") {
  const std::string path = TempPath("adapter");
  WriteLoraFile(
      {
          {prefix + module + ".lora_A.weight", "BF16", {rank, in_features}, a},
          {prefix + module + ".lora_B.weight", "BF16", {out_features, rank}, b},
      },
      metadata, path);
  return path;
}

std::string Caught(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool Mentions(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

std::vector<std::string> ContractWith(const std::string& name) { return {name}; }

// Fuse one adapter into a bf16 buffer and read the result back as floats.
std::vector<float> FuseBf16(const std::vector<vllm::Ltx2LoraAdapter>& adapters,
                            const std::string& target, int64_t rows, int64_t cols,
                            const std::vector<float>& weight, bool* out_fused = nullptr) {
  std::vector<uint16_t> buffer(weight.size());
  for (size_t i = 0; i < weight.size(); ++i) buffer[i] = vt::F32ToBF16(weight[i]);
  const bool fused = vllm::Ltx2FuseLoraIntoTensor(
      adapters, target, vt::DType::kBF16, rows, cols,
      reinterpret_cast<uint8_t*>(buffer.data()), buffer.size() * sizeof(uint16_t));
  if (out_fused != nullptr) *out_fused = fused;
  std::vector<float> out(buffer.size());
  for (size_t i = 0; i < buffer.size(); ++i) out[i] = vt::BF16ToF32(buffer[i]);
  return out;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// The key shape
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ltx2 lora: a factor key resolves onto the contract name") {
  std::string target;
  bool is_a = false;

  // LTXV_LORA_COMFY_RENAMING_MAP strips `diffusion_model.` (sd_ops.py:136), and
  // `_affected_weight_keys` rewrites the suffix (fuse_loras.py:185-186).
  CHECK(vllm::Ltx2LoraContractName("diffusion_model." + std::string(kModule) +
                                       ".lora_A.weight",
                                   &target, &is_a));
  CHECK(target == kTarget);
  CHECK(is_a);

  CHECK(vllm::Ltx2LoraContractName("diffusion_model." + std::string(kModule) +
                                       ".lora_B.weight",
                                   &target, &is_a));
  CHECK(target == kTarget);
  CHECK_FALSE(is_a);

  // The prefix is OPTIONAL: a PEFT-style adapter without it resolves the same.
  CHECK(vllm::Ltx2LoraContractName(std::string(kModule) + ".lora_A.weight", &target, &is_a));
  CHECK(target == kTarget);

  // Anything that is not a factor is not one. `.weight` alone is the TARGET, and
  // reading it as a factor would fuse a weight into itself.
  CHECK_FALSE(vllm::Ltx2LoraContractName(kTarget, &target, &is_a));
  CHECK_FALSE(vllm::Ltx2LoraContractName("diffusion_model.x.lora_A.bias", &target, &is_a));
  CHECK_FALSE(vllm::Ltx2LoraContractName("", &target, &is_a));
}

// ─────────────────────────────────────────────────────────────────────────────
// The arithmetic
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ltx2 lora: the fused weight is W + (B * strength) @ A") {
  // out=2, rank=2, in=2. Chosen so every product is exactly representable in
  // bf16, which makes this case about the FORMULA and leaves the rounding
  // question entirely to the dtype case below.
  //
  //   B = [[1, 2],       A = [[1, 0],      B @ A = [[1, 2],
  //        [0, 1]]            [0, 1]]              [0, 1]]
  const std::vector<float> b = {1, 2, 0, 1};
  const std::vector<float> a = {1, 0, 0, 1};
  const std::vector<float> w = {10, 20, 30, 40};

  SUBCASE("strength 1.0") {
    const std::string path = WriteAdapter(2, 2, 2, b, a);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    spec.strength = 1.0;
    std::vector<vllm::Ltx2LoraAdapter> adapters;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

    bool fused = false;
    const std::vector<float> got = FuseBf16(adapters, kTarget, 2, 2, w, &fused);
    CHECK(fused);
    CHECK(got[0] == doctest::Approx(11.0));  // 10 + 1
    CHECK(got[1] == doctest::Approx(22.0));  // 20 + 2
    CHECK(got[2] == doctest::Approx(30.0));  // 30 + 0
    CHECK(got[3] == doctest::Approx(41.0));  // 40 + 1
    std::remove(path.c_str());
  }

  SUBCASE("strength scales the delta and NOT the weight") {
    const std::string path = WriteAdapter(2, 2, 2, b, a);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    spec.strength = 0.5;
    std::vector<vllm::Ltx2LoraAdapter> adapters;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

    const std::vector<float> got = FuseBf16(adapters, kTarget, 2, 2, w);
    // Half the delta, all of the weight. A strength that scaled the sum would
    // give 5.5 / 11 / 15 / 20.5 and is what this distinguishes.
    CHECK(got[0] == doctest::Approx(10.5));
    CHECK(got[1] == doctest::Approx(21.0));
    CHECK(got[2] == doctest::Approx(30.0));
    CHECK(got[3] == doctest::Approx(40.5));
    std::remove(path.c_str());
  }

  SUBCASE("strength 0 is a no-op on the values but still counts as fused") {
    const std::string path = WriteAdapter(2, 2, 2, b, a);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    spec.strength = 0.0;
    std::vector<vllm::Ltx2LoraAdapter> adapters;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

    bool fused = false;
    const std::vector<float> got = FuseBf16(adapters, kTarget, 2, 2, w, &fused);
    // `fused` is about whether a delta was COMPUTED, not whether it was
    // non-zero: a zero-strength adapter is a legitimate request, and reporting
    // it as "fused nothing" would trip the load-time refusal for the wrong
    // reason.
    CHECK(fused);
    for (size_t i = 0; i < w.size(); ++i) CHECK(got[i] == doctest::Approx(w[i]));
    std::remove(path.c_str());
  }
}

TEST_CASE("ltx2 lora: a tensor no adapter targets is left alone") {
  const std::string path = WriteAdapter(2, 2, 2, {1, 2, 0, 1}, {1, 0, 0, 1});
  vllm::Ltx2LoraSpec spec;
  spec.path = path;
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

  const std::vector<float> w = {10, 20, 30, 40};
  bool fused = true;
  const std::vector<float> got =
      FuseBf16(adapters, "transformer_blocks.0.attn1.to_k.weight", 2, 2, w, &fused);
  CHECK_FALSE(fused);
  for (size_t i = 0; i < w.size(); ++i) CHECK(got[i] == doctest::Approx(w[i]));
  std::remove(path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// THE DTYPE GATE
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ltx2 lora: the delta accumulates in BF16, not f32") {
  // WHY THIS CASE EXISTS. Upstream sets `aggregation_dtype=torch.bfloat16` in
  // every one of its four fuse rules (fuse_loras.py:71, fp8_cast.py:239,
  // fp8_scaled_mm.py:189, nvfp4/fuse.py:50). Accumulating in f32 instead would
  // be MORE precise, so no golden and no token comparison could ever fail on it
  // - which is exactly why the choice needs a gate of its own.
  //
  // THE CONSTRUCTION, and why the obvious one does not work. `B * strength` is
  // rounded to bf16 BEFORE the matmul (fuse_loras.py:113: a bf16 tensor times a
  // Python float stays bf16), so the two arms differ by that rounding. But the
  // RESULT is stored bf16 too, and at rank 1 the final store rounds the
  // difference straight back out again - a first version of this case did
  // exactly that and passed under BOTH arms. That is the coverage hole this
  // comment exists to stop being reopened.
  //
  // So the per-term error is ACCUMULATED until it exceeds the bf16 step of the
  // sum. With B = A = 1 and rank 192:
  //
  //   strength = 1 + 2^-8, and bf16(1 * strength) = 1.0 exactly - 2^-8 is half
  //   of bf16's 2^-7 step at 1.0, so round-to-nearest-EVEN takes it down.
  //   upstream:  192 terms of 1.0  = 192.0,  stored bf16 -> 192.0
  //   f32 fold:  192 * (1 + 2^-8)  = 192.75, stored bf16 -> 193.0
  //
  // One bf16 step apart in the stored result, so the store cannot hide it.
  const int64_t kRank = 192;
  const double strength = 1.0 + 1.0 / 256.0;
  const std::vector<float> b(static_cast<size_t>(kRank), 1.0F);
  const std::vector<float> a(static_cast<size_t>(kRank), 1.0F);

  const std::string path = WriteAdapter(1, kRank, 1, b, a);
  vllm::Ltx2LoraSpec spec;
  spec.path = path;
  spec.strength = strength;
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

  const std::vector<float> got = FuseBf16(adapters, kTarget, 1, 1, {0.0F});

  // The bf16 answer, which is upstream's.
  CHECK(got[0] == doctest::Approx(192.0));
  // Checked on the BIT PATTERN too, because `doctest::Approx` carries a relative
  // epsilon and these two are only one part in 192 apart.
  CHECK(vt::F32ToBF16(got[0]) == vt::F32ToBF16(192.0F));
  // The f32 answer, which it must NOT be.
  CHECK(vt::F32ToBF16(got[0]) != vt::F32ToBF16(193.0F));
  std::remove(path.c_str());
}

TEST_CASE("ltx2 lora: the reference factors come from the adapter's metadata") {
  SUBCASE("absent is 1, which is upstream's default") {
    const std::string path = WriteAdapter(2, 2, 2, {1, 2, 0, 1}, {1, 0, 0, 1});
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    std::vector<vllm::Ltx2LoraAdapter> adapters;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));
    const vllm::Ltx2LoraReferenceFactors f = vllm::Ltx2ResolveLoraReferenceFactors(adapters);
    CHECK(f.downscale == 1);
    CHECK(f.temporal == 1);
    std::remove(path.c_str());
  }

  SUBCASE("declared values are read") {
    const std::string path =
        WriteAdapter(2, 2, 2, {1, 2, 0, 1}, {1, 0, 0, 1},
                     {{"reference_downscale_factor", "2"},
                      {"reference_temporal_scale_factor", "4"}});
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    std::vector<vllm::Ltx2LoraAdapter> adapters;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));
    const vllm::Ltx2LoraReferenceFactors f = vllm::Ltx2ResolveLoraReferenceFactors(adapters);
    // These are the two numbers the reference refusal named as unreadable.
    CHECK(f.downscale == 2);
    CHECK(f.temporal == 4);
    std::remove(path.c_str());
  }

  SUBCASE("a malformed value REFUSES rather than silently reverting to 1") {
    // The one place this port deliberately diverges from upstream, which
    // swallows every exception and returns 1 (iclora_utils.py:36-38). A factor
    // that reverts to 1 places the reference plausibly and wrongly, and no
    // output check can see that.
    for (const char* bad : {"two", "0", "-3", "2.5", ""}) {
      const std::string path = WriteAdapter(2, 2, 2, {1, 2, 0, 1}, {1, 0, 0, 1},
                                            {{"reference_downscale_factor", bad}});
      vllm::Ltx2LoraSpec spec;
      spec.path = path;
      std::vector<vllm::Ltx2LoraAdapter> adapters;
      adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));
      const std::string err =
          Caught([&] { (void)vllm::Ltx2ResolveLoraReferenceFactors(adapters); });
      INFO("value = '", bad, "' error = ", err);
      CHECK(Mentions(err, "reference_downscale_factor"));
      CHECK(Mentions(err, "not a positive integer"));
      std::remove(path.c_str());
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// The refusals
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ltx2 lora: an adapter naming a module the contract lacks refuses BY NAME") {
  // Upstream SKIPS this key (fuse_loras.py:135-137). Here it refuses, because
  // the contract is a fixed enumerated set and a skip would absorb a misnamed
  // key and an inapplicable one alike. The row's spec §4.1 argues it.
  const std::string path = WriteAdapter(2, 2, 2, {1, 2, 0, 1}, {1, 0, 0, 1}, {},
                                        "transformer_blocks.0.not_a_real_module");
  vllm::Ltx2LoraSpec spec;
  spec.path = path;
  const std::string err = Caught([&] {
    (void)vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget));
  });
  INFO("error = ", err);
  // BY NAME: the offending target is in the message, not just "a key".
  CHECK(Mentions(err, "transformer_blocks.0.not_a_real_module.weight"));
  CHECK(Mentions(err, "does not bind"));
  // And it says what upstream would have done, so the divergence is visible to
  // whoever is reading the refusal rather than only to whoever reads the spec.
  CHECK(Mentions(err, "fuse_loras.py:135-137"));
  std::remove(path.c_str());
}

TEST_CASE("ltx2 lora: a file that is not an adapter refuses rather than loading nothing") {
  const std::string path = TempPath("empty");
  WriteLoraFile({{"some.weight", "BF16", {2, 2}, {1, 2, 3, 4}}}, {}, path);
  vllm::Ltx2LoraSpec spec;
  spec.path = path;
  const std::string err =
      Caught([&] { (void)vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)); });
  INFO("error = ", err);
  CHECK(Mentions(err, "no `.lora_A.weight`"));
  std::remove(path.c_str());
}

TEST_CASE("ltx2 lora: a half pair refuses, naming the side that is missing") {
  SUBCASE("A with no B") {
    const std::string path = TempPath("half_a");
    WriteLoraFile({{std::string(kModule) + ".lora_A.weight", "BF16", {1, 2}, {1, 1}}}, {},
                  path);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    const std::string err =
        Caught([&] { (void)vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)); });
    INFO("error = ", err);
    CHECK(Mentions(err, "no matching B factor"));
    std::remove(path.c_str());
  }
  SUBCASE("B with no A") {
    const std::string path = TempPath("half_b");
    WriteLoraFile({{std::string(kModule) + ".lora_B.weight", "BF16", {2, 1}, {1, 1}}}, {},
                  path);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    const std::string err =
        Caught([&] { (void)vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)); });
    INFO("error = ", err);
    CHECK(Mentions(err, "no matching A factor"));
    std::remove(path.c_str());
  }
}

TEST_CASE("ltx2 lora: a delta whose shape disagrees with the target refuses") {
  // out=2 in=2 in the adapter, against a [4, 4] tensor. Fusing regardless would
  // read past the delta and write a corrupt weight that nothing downstream can
  // attribute.
  const std::string path = WriteAdapter(2, 2, 2, {1, 2, 0, 1}, {1, 0, 0, 1});
  vllm::Ltx2LoraSpec spec;
  spec.path = path;
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

  std::vector<uint16_t> buffer(16, 0);
  const std::string err = Caught([&] {
    (void)vllm::Ltx2FuseLoraIntoTensor(adapters, kTarget, vt::DType::kBF16, 4, 4,
                                       reinterpret_cast<uint8_t*>(buffer.data()),
                                       buffer.size() * sizeof(uint16_t));
  });
  INFO("error = ", err);
  CHECK(Mentions(err, "[2, 2] delta"));
  CHECK(Mentions(err, "[4, 4]"));
  std::remove(path.c_str());
}

TEST_CASE("ltx2 lora: more than one adapter refuses BY NAME") {
  // Upstream's ic_lora.py accepts a list; dubit.py:364-365 enforces exactly one
  // and hdr_ic_lora.py:271-272 takes exactly one. N-adapter fusion is recorded
  // as owed by the row rather than half-built, and the refusal says so.
  const std::string a = WriteAdapter(2, 2, 2, {1, 2, 0, 1}, {1, 0, 0, 1});
  const std::string b = WriteAdapter(2, 2, 2, {1, 2, 0, 1}, {1, 0, 0, 1});
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  for (const std::string& p : {a, b}) {
    vllm::Ltx2LoraSpec spec;
    spec.path = p;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));
  }
  const std::string err =
      Caught([&] { (void)vllm::Ltx2ResolveLoraReferenceFactors(adapters); });
  INFO("error = ", err);
  CHECK(Mentions(err, "exactly ONE adapter"));
  CHECK(Mentions(err, "LTX25-IC-LORA"));
  std::remove(a.c_str());
  std::remove(b.c_str());
}

TEST_CASE("ltx2 lora: an unreadable factor dtype refuses, naming the RIGHT factor") {
  // THE KEY NAME IS THE POINT, not just the dtype. A first version of this
  // reader kept ONE key per target, so it held whichever of A/B appeared last
  // in the header and reported a malformed A factor under the B factor's name.
  // "Refuse BY NAME" defeated by the message itself, and a case that checked
  // only for "U8" passed straight through it. So each subcase makes exactly one
  // side unreadable and asserts the message names THAT side and not the other.
  SUBCASE("the A factor is unreadable") {
    const std::string path = TempPath("u8_a");
    WriteLoraFile(
        {
            {std::string(kModule) + ".lora_A.weight", "U8", {1, 2}, {1, 2}},
            {std::string(kModule) + ".lora_B.weight", "BF16", {2, 1}, {1, 1}},
        },
        {}, path);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    const std::string err =
        Caught([&] { (void)vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)); });
    INFO("error = ", err);
    CHECK(Mentions(err, "U8"));
    CHECK(Mentions(err, "BF16 or F32"));
    CHECK(Mentions(err, ".lora_A.weight"));
    CHECK_FALSE(Mentions(err, ".lora_B.weight"));
    std::remove(path.c_str());
  }
  SUBCASE("the B factor is unreadable") {
    const std::string path = TempPath("u8_b");
    WriteLoraFile(
        {
            {std::string(kModule) + ".lora_A.weight", "BF16", {1, 2}, {1, 1}},
            {std::string(kModule) + ".lora_B.weight", "U8", {2, 1}, {1, 2}},
        },
        {}, path);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    const std::string err =
        Caught([&] { (void)vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)); });
    INFO("error = ", err);
    CHECK(Mentions(err, "U8"));
    CHECK(Mentions(err, ".lora_B.weight"));
    CHECK_FALSE(Mentions(err, ".lora_A.weight"));
    std::remove(path.c_str());
  }
}

TEST_CASE("ltx2 lora: an F32 adapter is NARROWED to bf16, not kept f32") {
  // fuse_loras.py:202-203 casts both factors to the rule's aggregation dtype
  // before the matmul. An F32 adapter held at F32 would widen the accumulator
  // through the back door - the same defect the dtype case above gates, arriving
  // by a different route, so it uses the same accumulate-until-it-shows shape.
  //
  //   B[k] = 1 + 2^-8 in F32, A[k] = 1, rank 192, strength 1.
  //   narrowed to bf16: 192 terms of 1.0  = 192.0
  //   kept f32:         192 * (1 + 2^-8)  = 192.75 -> bf16 -> 193.0
  const int64_t kRank = 192;
  const float unrepresentable = 1.0F + 1.0F / 256.0F;
  const std::string path = TempPath("f32_adapter");
  WriteLoraFile(
      {
          {std::string(kModule) + ".lora_A.weight",
           "F32",
           {kRank, 1},
           std::vector<float>(static_cast<size_t>(kRank), 1.0F)},
          {std::string(kModule) + ".lora_B.weight",
           "F32",
           {1, kRank},
           std::vector<float>(static_cast<size_t>(kRank), unrepresentable)},
      },
      {}, path);
  vllm::Ltx2LoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

  const std::vector<float> got = FuseBf16(adapters, kTarget, 1, 1, {0.0F});
  CHECK(vt::F32ToBF16(got[0]) == vt::F32ToBF16(192.0F));
  CHECK(vt::F32ToBF16(got[0]) != vt::F32ToBF16(193.0F));
  std::remove(path.c_str());
}

TEST_CASE("ltx2 lora: the matmul RESULT is rounded to bf16 before the weight is added") {
  // THE THIRD ROUNDING, and the one the two cases above cannot see. Upstream's
  // aggregation dtype binds three separate places, not one:
  //
  //   1. `B * strength`  -> bf16   (fuse_loras.py:113; gated above by the
  //                                 strength case and the F32-adapter case)
  //   2. the MATMUL RESULT -> bf16 (fuse_loras.py:113's `.to(dtype=dtype)`,
  //                                 which is `aggregation_dtype`)  <- THIS CASE
  //   3. `deltas.add_(weight)` -> bf16 (fuse_loras.py:67-68; gated below)
  //
  // MEASURED: widening only (2) - keeping the f32 accumulator and adding the
  // weight to it before the single store - left `test_ltx2_lora` at 13/13 and
  // `test_ltx2_loader` at 31/31. Both of the other roundings survived that
  // mutation, which is why neither of their cases moved.
  //
  // THE CONSTRUCTION. Rank 2, with the two products chosen so the SUM sits
  // exactly on a bf16 tie and the weight then pushes the two arms to different
  // sides of the next one:
  //
  //   acc  = 1.0 * 1.0 + 2^-8 * 1.0 = 1.00390625   (exact in f32)
  //   w    = 2^-9                   = 0.001953125  (exact in bf16)
  //
  //   ported: bf16(acc) = 1.0 by ties-to-even (2^-8 is half of the 2^-7 step at
  //           1.0, and 1.0's mantissa is the even one), then
  //           bf16(1.0 + 2^-9) = 1.0
  //   f32 acc: bf16(1.00390625 + 0.001953125) = bf16(1.005859375) = 1.0078125
  //
  // One bf16 step apart in the STORED result, so the final store cannot absorb
  // it - which is exactly how a first attempt at this case would fail.
  const float kHalfStep = 1.0F / 256.0F;   // 2^-8
  const float kQuarterStep = 1.0F / 512.0F;  // 2^-9
  const std::string path = WriteAdapter(/*out_features=*/1, /*rank=*/2, /*in_features=*/1,
                                        /*b=*/{1.0F, kHalfStep}, /*a=*/{1.0F, 1.0F});
  vllm::Ltx2LoraSpec spec;
  spec.path = path;
  spec.strength = 1.0;
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

  const std::vector<float> got = FuseBf16(adapters, kTarget, 1, 1, {kQuarterStep});
  CHECK(vt::F32ToBF16(got[0]) == vt::F32ToBF16(1.0F));
  CHECK(vt::F32ToBF16(got[0]) != vt::F32ToBF16(1.0F + 4.0F * kQuarterStep));
  // Stated as the value an f32 accumulator would produce, so a reader can see
  // which number this case is separating 1.0 from.
  CHECK(vt::BF16ToF32(vt::F32ToBF16(1.0F + kHalfStep + kQuarterStep)) == doctest::Approx(1.0078125));
  std::remove(path.c_str());
}

TEST_CASE("ltx2 lora: the f32 target branch rounds through the bf16 accumulator") {
  // The scale_shift tables are the only F32 tensors in the contract. Upstream's
  // `_bf16_fuse` does `deltas.add_(weight)` IN PLACE on the bf16 aggregator and
  // only then casts to the weight's dtype (fuse_loras.py:67-68) — so an f32
  // target still rounds the SUM through bf16. Mirrored rather than "improved".
  const std::string path = WriteAdapter(1, 1, 1, {1.0F}, {1.0F});
  vllm::Ltx2LoraSpec spec;
  spec.path = path;
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

  // 1 + 2^-9 in f32; adding a delta of 1.0 gives 2.001953125, which bf16 cannot
  // hold and rounds to 2.0.
  float weight = 1.0F + 1.0F / 512.0F;
  std::vector<uint8_t> buffer(sizeof(float));
  std::memcpy(buffer.data(), &weight, sizeof(float));
  const bool fused = vllm::Ltx2FuseLoraIntoTensor(adapters, kTarget, vt::DType::kF32, 1, 1,
                                                  buffer.data(), buffer.size());
  CHECK(fused);
  float got = 0.0F;
  std::memcpy(&got, buffer.data(), sizeof(float));
  CHECK(got == doctest::Approx(2.0));
  // Not 2.001953125: that is what an f32-throughout add would produce.
  CHECK(got != doctest::Approx(weight + 1.0F));
  std::remove(path.c_str());
}
