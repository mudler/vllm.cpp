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
#include "vt/op_provider.h"
#include "vt/ops.h"

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

namespace {

// Open N adapters, each declaring the metadata given, against a one-name
// contract. Every one targets `kTarget`, which is what makes them compose.
std::vector<vllm::Ltx2LoraAdapter> OpenAll(
    const std::vector<std::map<std::string, std::string>>& metadata,
    std::vector<std::string>* paths) {
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  for (const auto& md : metadata) {
    const std::string path = WriteAdapter(2, 2, 2, {1, 2, 0, 1}, {1, 0, 0, 1}, md);
    paths->push_back(path);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));
  }
  return adapters;
}

void RemoveAll(const std::vector<std::string>& paths) {
  for (const std::string& p : paths) std::remove(p.c_str());
}

}  // namespace

TEST_CASE("ltx2 lora: N adapters resolve their reference factors TOGETHER") {
  // ROW LTX25-LORA-FUSION. `ICLoraPipeline` takes a LIST (`ic_lora.py:75`) and
  // resolves both scale factors ACROSS it (`ic_lora.py:155-173`). This function
  // used to refuse a second adapter by name, which made every branch below
  // unreachable — they were written for the day the cap lifted and this is that
  // day. `dubit.py:364-365` and `hdr_ic_lora.py:271-272` still narrow their own
  // entry points to one adapter; that is those two pipelines, not the fuser.
  SUBCASE("two adapters declaring nothing resolve to upstream's 1/1") {
    std::vector<std::string> paths;
    const std::vector<vllm::Ltx2LoraAdapter> adapters = OpenAll({{}, {}}, &paths);
    const vllm::Ltx2LoraReferenceFactors f = vllm::Ltx2ResolveLoraReferenceFactors(adapters);
    CHECK(f.downscale == 1);
    CHECK(f.temporal == 1);
    RemoveAll(paths);
  }

  SUBCASE("one adapter declaring a factor carries it for the whole list") {
    // `if scale != 1` (`ic_lora.py:157`): a 1 is upstream's ABSENT value and
    // never enters the conflict test, so a silent adapter cannot veto a
    // declared factor and the order of the two cannot matter. Both orders are
    // run, because an implementation that seeded the accumulator from the FIRST
    // adapter unconditionally would pass one of them.
    for (bool declared_first : {true, false}) {
      const std::map<std::string, std::string> declares = {
          {"reference_downscale_factor", "2"}, {"reference_temporal_scale_factor", "4"}};
      std::vector<std::string> paths;
      const std::vector<vllm::Ltx2LoraAdapter> adapters =
          declared_first ? OpenAll({declares, {}}, &paths) : OpenAll({{}, declares}, &paths);
      const vllm::Ltx2LoraReferenceFactors f =
          vllm::Ltx2ResolveLoraReferenceFactors(adapters);
      INFO("declared_first = ", declared_first);
      CHECK(f.downscale == 2);
      CHECK(f.temporal == 4);
      RemoveAll(paths);
    }
  }

  SUBCASE("two adapters AGREEING on a factor is not a conflict") {
    // The second disjunct of `not in (1, scale)` (`ic_lora.py:158`). Upstream
    // combines IC-LoRAs trained at the same reference scale, and refusing that
    // would be stricter than the reference.
    std::vector<std::string> paths;
    const std::vector<vllm::Ltx2LoraAdapter> adapters =
        OpenAll({{{"reference_downscale_factor", "2"}}, {{"reference_downscale_factor", "2"}}},
                &paths);
    const vllm::Ltx2LoraReferenceFactors f = vllm::Ltx2ResolveLoraReferenceFactors(adapters);
    CHECK(f.downscale == 2);
    RemoveAll(paths);
  }

  SUBCASE("two adapters DISAGREEING refuse, naming both values") {
    // `ic_lora.py:159-163`, reachable for the first time.
    std::vector<std::string> paths;
    const std::vector<vllm::Ltx2LoraAdapter> adapters =
        OpenAll({{{"reference_downscale_factor", "2"}}, {{"reference_downscale_factor", "3"}}},
                &paths);
    const std::string err =
        Caught([&] { (void)vllm::Ltx2ResolveLoraReferenceFactors(adapters); });
    INFO("error = ", err);
    CHECK(Mentions(err, "conflicting reference_downscale_factor"));
    CHECK(Mentions(err, "already have 2"));
    CHECK(Mentions(err, "specifies 3"));
    RemoveAll(paths);
  }

  SUBCASE("the TEMPORAL factor conflicts independently of the downscale one") {
    // Two separate accumulators upstream, and two separate raises
    // (`ic_lora.py:167-172`). Sharing one would let an agreeing downscale mask a
    // disagreeing temporal scale.
    std::vector<std::string> paths;
    const std::vector<vllm::Ltx2LoraAdapter> adapters = OpenAll(
        {{{"reference_downscale_factor", "2"}, {"reference_temporal_scale_factor", "4"}},
         {{"reference_downscale_factor", "2"}, {"reference_temporal_scale_factor", "8"}}},
        &paths);
    const std::string err =
        Caught([&] { (void)vllm::Ltx2ResolveLoraReferenceFactors(adapters); });
    INFO("error = ", err);
    CHECK(Mentions(err, "conflicting reference_temporal_scale_factor"));
    CHECK(Mentions(err, "already have 4"));
    CHECK(Mentions(err, "specifies 8"));
    RemoveAll(paths);
  }
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

// ─────────────────────────────────────────────────────────────────────────────
// THE EXECUTION SEAM (row LTX25-LORA-FUSE-SEAM, issue #1202)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Deterministic values with a FULL f32 mantissa, so every one of them rounds
// when the adapter writer narrows it to bf16, and so essentially no product of
// two of them is bf16-representable. A case built from small exact integers
// cannot see a rounding difference at all, which is what makes the byte-equality
// claim below worth asserting on these values instead.
std::vector<float> Spread(size_t n, uint32_t seed) {
  std::vector<float> out(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525U + 1013904223U;  // Numerical Recipes LCG
    out[i] = static_cast<float>(s >> 8) / 8388608.0F - 1.0F;
  }
  return out;
}

// `(B * strength) @ A`, written as the scalar triple loop this fuser ran before
// LTX25-LORA-FUSE-SEAM: `fuse_loras.py:113` transcribed, with `B * strength`
// ROUNDED TO BF16 before the product, an f32 accumulator, and a bf16 store.
// Deliberately a second transcription rather than a call back into the fuser —
// it is the oracle the seam is measured against, so it must not share code with
// the thing it measures. Inputs are given as floats and narrowed here exactly
// where `WriteAdapter` narrows them, so both arms start from the same bf16 bits.
std::vector<uint16_t> ScalarDeltaPlusWeight(const std::vector<float>& b,
                                            const std::vector<float>& a,
                                            const std::vector<float>& w, int64_t rows,
                                            int64_t rank, int64_t cols, float strength) {
  std::vector<uint16_t> bs(b.size());
  for (size_t i = 0; i < b.size(); ++i) {
    bs[i] = vt::F32ToBF16(vt::BF16ToF32(vt::F32ToBF16(b[i])) * strength);
  }
  std::vector<uint16_t> ab(a.size());
  for (size_t i = 0; i < a.size(); ++i) ab[i] = vt::F32ToBF16(a[i]);

  std::vector<uint16_t> out(static_cast<size_t>(rows * cols));
  for (int64_t o = 0; o < rows; ++o) {
    const uint16_t* brow = bs.data() + static_cast<size_t>(o * rank);
    for (int64_t i = 0; i < cols; ++i) {
      float acc = 0.0F;
      for (int64_t k = 0; k < rank; ++k) {
        acc += vt::BF16ToF32(brow[k]) * vt::BF16ToF32(ab[static_cast<size_t>(k * cols + i)]);
      }
      // `deltas.add_(weight)` in the bf16 aggregator, then the bf16 store
      // (fuse_loras.py:67-68) — the same two roundings the fuser performs.
      const uint16_t delta = vt::F32ToBF16(acc);
      out[static_cast<size_t>(o * cols + i)] = vt::F32ToBF16(
          vt::BF16ToF32(delta) + vt::BF16ToF32(vt::F32ToBF16(w[static_cast<size_t>(o * cols + i)])));
    }
  }
  return out;
}

// Fuse into a bf16 buffer and hand back the BIT PATTERNS. `FuseBf16` above
// widens to float, which is lossless and so would serve — but this case claims
// byte equality, and it should read as the byte comparison it is.
std::vector<uint16_t> FuseBf16Bits(const std::vector<vllm::Ltx2LoraAdapter>& adapters,
                                   const std::string& target, int64_t rows, int64_t cols,
                                   const std::vector<float>& weight) {
  std::vector<uint16_t> buffer(weight.size());
  for (size_t i = 0; i < weight.size(); ++i) buffer[i] = vt::F32ToBF16(weight[i]);
  REQUIRE(vllm::Ltx2FuseLoraIntoTensor(adapters, target, vt::DType::kBF16, rows, cols,
                                       reinterpret_cast<uint8_t*>(buffer.data()),
                                       buffer.size() * sizeof(uint16_t)));
  return buffer;
}

}  // namespace

TEST_CASE("ltx2 lora: the delta product runs on the shared vt::Matmul seam") {
  // WHAT THIS CASE GATES, and why the arithmetic cases above cannot.
  //
  // Every case above hands the fuser a 1x1 or 2x2 problem and checks a VALUE.
  // None of them can see WHICH code produced it, and the defect this row fixes
  // (#1202) is exactly that: a correct mirror of `fuse_loras.py:103-116`
  // executed by a scalar single-threaded triple loop, measured at ~0.43 GMAC/s,
  // which put the full 21.004 B DiT's 8.53e12-MAC fusion hours away from any
  // LoRA-bearing render. So this case asserts TWO things that must both hold:
  //
  //   1. ROUTING. `vt::Matmul` — the shared `out[M,N] = a[M,K] @ b[K,N]` GEMM
  //      seam — is dispatched exactly once per fused tensor. Counted through
  //      `vt::GetOpProviderStats`, the seam's own positive signal, because a
  //      green value assertion does not prove which provider executed. Put a
  //      hand-rolled loop back and this drops to zero.
  //   2. THE ARITHMETIC IS UNCHANGED, BYTE FOR BYTE. Not a tolerance: the seam
  //      accumulates each output in f32 over K in strict increasing order and
  //      stores through the same `vt::F32ToBF16`, so it computes the same
  //      numbers the loop did and the gate can say so exactly. A tolerance here
  //      would be the place a real reduction-order change went unnoticed.
  //
  // THE SHAPES. Three, all rank-`r` with `r` small, because that is the regime
  // (the shipped distilled adapter is rank 450 and the DiT's projections are
  // thousands wide). 37 and 96 columns straddle the seam's 16-wide output block,
  // so the ragged column tail and the blocked path both run; 19 and 64 rows
  // straddle its 16-row activation tile the same way. All are tiny enough to be
  // instant — this case gates the arithmetic, not the speed, which is measured
  // in the row's spec.
  struct Shape {
    int64_t rows, rank, cols;
  };
  const Shape shapes[] = {{19, 5, 37}, {64, 13, 96}, {1, 192, 1}};
  const float kStrength = 0.75F;

  vt::EnableOpProviderCallStats(true);
  unsigned long long fused_tensors = 0;
  const unsigned long long matmuls_before =
      vt::GetOpProviderStats(vt::OpId::kMatmul, vt::DeviceType::kCPU).selections;

  for (const Shape& s : shapes) {
    INFO("shape ", s.rows, "x", s.rank, "x", s.cols);
    const std::vector<float> b = Spread(static_cast<size_t>(s.rows * s.rank), 12345U);
    const std::vector<float> a = Spread(static_cast<size_t>(s.rank * s.cols), 67890U);
    const std::vector<float> w = Spread(static_cast<size_t>(s.rows * s.cols), 24680U);

    const std::string path = WriteAdapter(s.rows, s.rank, s.cols, b, a);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    spec.strength = kStrength;
    std::vector<vllm::Ltx2LoraAdapter> adapters;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));

    const std::vector<uint16_t> got = FuseBf16Bits(adapters, kTarget, s.rows, s.cols, w);
    ++fused_tensors;
    const std::vector<uint16_t> want =
        ScalarDeltaPlusWeight(b, a, w, s.rows, s.rank, s.cols, kStrength);
    REQUIRE(got.size() == want.size());

    size_t mismatched = 0;
    size_t moved = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      if (got[i] != want[i]) ++mismatched;
      if (got[i] != vt::F32ToBF16(w[i])) ++moved;
    }
    CHECK(mismatched == 0);
    // NOT VACUOUS. `Spread` gives a dense non-degenerate delta, so a fuser that
    // wrote nothing — or one whose GEMM never ran — leaves every element equal
    // to the weight it started from and this fails. Stated as a proportion
    // rather than "> 0", because one moved element out of 703 would satisfy
    // "> 0" while 702 outputs stayed untouched.
    CHECK(moved > got.size() * 3 / 4);
    std::remove(path.c_str());
  }

  // ROUTING. Exactly one dispatch of the shared GEMM per fused tensor. An
  // EQUALITY on both sides: fewer means something did not route through the
  // seam, and more means the fuser is running a GEMM it does not need.
  const vt::OpProviderStats stats =
      vt::GetOpProviderStats(vt::OpId::kMatmul, vt::DeviceType::kCPU);
  vt::EnableOpProviderCallStats(false);
  CHECK(stats.selections == matmuls_before + fused_tensors);
  // And the dispatch bound the first-party kernel rather than falling through to
  // the portable reference tier, which would be correct and slow.
  REQUIRE(stats.last_selected != nullptr);
  CHECK(std::string(stats.last_selected) == std::string(vt::kNativeProviderName));
}

// ─────────────────────────────────────────────────────────────────────────────
// N adapters (row LTX25-LORA-FUSION, issue #932)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// The strengths and seeds the golden below was generated from. Held here so the
// generator recipe in the row's spec and the fixture cannot drift apart.
struct AdapterRecipe {
  uint32_t b_seed, a_seed;
  float strength;
};
const AdapterRecipe kThree[] = {{11111U, 22222U, 0.75F},
                                {33333U, 44444U, 0.35F},
                                {55555U, 66666U, 1.0F}};
constexpr int64_t kGoldRows = 6;
constexpr int64_t kGoldRank = 3;
constexpr int64_t kGoldCols = 20;
constexpr uint32_t kGoldWeightSeed = 24680U;

// `sum` over the adapters of `(B * strength) @ A` with the FIRST product form
// used for EVERY member — the plausible wrong port, and the reason this case
// asserts against a generated golden rather than against a transcription of the
// formula. It pre-rounds `B * strength` to bf16 before each matmul, which is
// `fuse_loras.py:113` applied where `:115` belongs.
std::vector<uint16_t> EveryProductPreScaled(const std::vector<std::vector<float>>& b,
                                            const std::vector<std::vector<float>>& a,
                                            const std::vector<float>& w) {
  std::vector<uint16_t> agg(static_cast<size_t>(kGoldRows * kGoldCols), 0);
  for (size_t n = 0; n < b.size(); ++n) {
    const std::vector<uint16_t> one =
        ScalarDeltaPlusWeight(b[n], a[n], std::vector<float>(agg.size(), 0.0F), kGoldRows,
                              kGoldRank, kGoldCols, kThree[n].strength);
    for (size_t i = 0; i < agg.size(); ++i) {
      agg[i] = vt::F32ToBF16(vt::BF16ToF32(agg[i]) + vt::BF16ToF32(one[i]));
    }
  }
  for (size_t i = 0; i < agg.size(); ++i) {
    agg[i] = vt::F32ToBF16(vt::BF16ToF32(agg[i]) + vt::BF16ToF32(vt::F32ToBF16(w[i])));
  }
  return agg;
}

}  // namespace

TEST_CASE("ltx2 lora: N adapters aggregate through upstream's SECOND product form") {
  // THE EXPECTATION WAS EXECUTED, NOT TRANSCRIBED. It is the output of
  // `ltx_core.loader.fuse_loras.aggregate_lora_products` followed by
  // `bf16_fuse_rule`, imported from Lightricks/LTX-2 at the pin
  // `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` and run under torch
  // 2.11.0+cu130 on the same bf16 inputs this case builds. The recipe is in
  // .agents/specs/ltx25-lora-fusion.md and reproduces `Spread` exactly.
  //
  // WHAT IT PINS THAT A FORMULA COULD NOT. `aggregate_lora_products` uses two
  // different products: `matmul(B * strength, A)` for the FIRST
  // (`fuse_loras.py:113`) and `addmm_(B, A, alpha=strength)` for every one after
  // it (`:115`). The difference is WHERE the strength enters — the first rounds
  // `B * strength` to bf16 BEFORE its matmul, the second applies `alpha` to an
  // f32 accumulation and rounds ONCE at the store. Three candidate models were
  // run against the pinned module; only that one matched, over 40 randomized
  // trials and over K in {16, 64, 256, 1024}.
  //
  // AND THE CASE CAN SEE THE DIFFERENCE, which is asserted rather than assumed:
  // `EveryProductPreScaled` is the wrong port, and it is separated from the
  // expectation on 26 of these 120 elements.
  const std::vector<uint16_t> want = {
      0xBF1C, 0xBE2C, 0x3EE6, 0xC004, 0x3DCC, 0xBE18, 0x3F2F, 0xBE54,
      0x3F9E, 0xBF23, 0x3FC6, 0x3F76, 0x3EA2, 0x3F1C, 0x3D30, 0xBFAA,
      0x3E45, 0xBEAA, 0x3EB7, 0xBF02, 0x3F59, 0xBF20, 0xBEAF, 0xBFAD,
      0x3F7D, 0xBFD8, 0xBD98, 0xC010, 0xBF1E, 0xBCC0, 0xBF66, 0x3CD4,
      0x3FC0, 0xC021, 0xBE30, 0xBF9A, 0x3E98, 0xBF88, 0x3F18, 0xBF14,
      0xBE25, 0xBF0A, 0xBF8F, 0x3F90, 0xBFE2, 0xBF91, 0x3FBB, 0x3FD4,
      0x3E68, 0x3ECE, 0x3FEB, 0x3F2C, 0x3FB4, 0x3F06, 0x3E9A, 0xBF04,
      0x3F3B, 0xBFBE, 0x3E80, 0x3F28, 0x3D50, 0x3F42, 0xBECE, 0xBF24,
      0x3F5F, 0x3EC6, 0xBE48, 0xBEFA, 0x3EBF, 0x3E95, 0x3F32, 0x3F6C,
      0x4009, 0xBEE9, 0x3FA5, 0xBC80, 0xBFD8, 0xBE3C, 0xBD48, 0xBEB3,
      0x3D90, 0x3E44, 0x3F1C, 0xBF82, 0x3F4C, 0x3F82, 0x3F97, 0x3DF2,
      0x3F22, 0xBF20, 0xBF24, 0x3E8A, 0x3ED6, 0x3F58, 0x3EB8, 0xBEF4,
      0xBF88, 0xBE94, 0xBFA3, 0xBF8A, 0xBFE7, 0x3EF8, 0x3E64, 0xBF8C,
      0xBF0E, 0x3FF2, 0x3F04, 0xBFF6, 0x3FB3, 0x3D0C, 0xBFAA, 0x3F41,
      0xBEE8, 0xBF6E, 0x3F88, 0x3E9C, 0x3E3C, 0x3F89, 0x3EE0, 0x3F94
  };
  REQUIRE(want.size() == static_cast<size_t>(kGoldRows * kGoldCols));

  const std::vector<float> w =
      Spread(static_cast<size_t>(kGoldRows * kGoldCols), kGoldWeightSeed);
  std::vector<std::string> paths;
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  std::vector<std::vector<float>> bs;
  std::vector<std::vector<float>> as;
  for (const AdapterRecipe& r : kThree) {
    bs.push_back(Spread(static_cast<size_t>(kGoldRows * kGoldRank), r.b_seed));
    as.push_back(Spread(static_cast<size_t>(kGoldRank * kGoldCols), r.a_seed));
    const std::string path =
        WriteAdapter(kGoldRows, kGoldRank, kGoldCols, bs.back(), as.back());
    paths.push_back(path);
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    spec.strength = r.strength;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));
  }

  const std::vector<uint16_t> got = FuseBf16Bits(adapters, kTarget, kGoldRows, kGoldCols, w);
  REQUIRE(got.size() == want.size());
  size_t mismatched = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) ++mismatched;
  }
  CHECK(mismatched == 0);

  // NOT VACUOUS, three ways.
  //
  // 1. The wrong rounding is DISTINGUISHABLE here. Without this the case would
  //    pass on a port that reused the first product form for all three
  //    adapters, and would look like it had gated the thing it was generated to
  //    gate.
  const std::vector<uint16_t> pre_scaled = EveryProductPreScaled(bs, as, w);
  size_t separated = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (pre_scaled[i] != want[i]) ++separated;
  }
  MESSAGE("the pre-scaled second form differs from upstream on " << separated << " of "
                                                                 << want.size());
  CHECK(separated == 26);

  // 2. Every adapter after the first CONTRIBUTED. One adapter alone differs from
  //    the three on all 120 elements, so a loop that stopped after the first —
  //    which is exactly what the lifted refusal used to enforce — cannot pass.
  std::vector<vllm::Ltx2LoraAdapter> only_first;
  {
    vllm::Ltx2LoraSpec spec;
    spec.path = paths[0];
    spec.strength = kThree[0].strength;
    only_first.push_back(vllm::Ltx2LoraAdapter::Open(spec, ContractWith(kTarget)));
  }
  const std::vector<uint16_t> one =
      FuseBf16Bits(only_first, kTarget, kGoldRows, kGoldCols, w);
  size_t moved_by_the_rest = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (one[i] != want[i]) ++moved_by_the_rest;
  }
  CHECK(moved_by_the_rest == want.size());

  // 3. ORDER IS UPSTREAM'S LIST ORDER. `_products_for_sd_key` yields in the
  //    order of `lora_sd_and_strengths` (`fuse_loras.py:199-204`), and the first
  //    member is the only one that takes the first product form — so reversing
  //    the list is a DIFFERENT computation, not a re-association. If this
  //    matched, the fuser would not be preserving the order the load supplied.
  std::vector<vllm::Ltx2LoraAdapter> reversed(adapters.rbegin(), adapters.rend());
  const std::vector<uint16_t> other =
      FuseBf16Bits(reversed, kTarget, kGoldRows, kGoldCols, w);
  size_t moved_by_order = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (other[i] != want[i]) ++moved_by_order;
  }
  MESSAGE("reversing the adapter order moves " << moved_by_order << " of " << want.size());
  CHECK(moved_by_order > 0);

  RemoveAll(paths);
}

TEST_CASE("ltx2 lora: an adapter that does not target a tensor is SKIPPED, not refused") {
  // `_products_for_sd_key` yields nothing for a LoRA whose state dict lacks the
  // key (`fuse_loras.py:200-201`, `continue`), and `aggregate_lora_products`
  // then sees a shorter list. So the SECOND adapter of a two-adapter load can
  // legitimately take the FIRST product form on a tensor only it targets, and
  // the aggregator must therefore be seeded per TENSOR rather than per load.
  const char* const kOther = "transformer_blocks.0.attn1.to_k.weight";
  const char* const kOtherModule = "transformer_blocks.0.attn1.to_k";
  const std::vector<float> b = {1, 2, 0, 1};
  const std::vector<float> a = {1, 0, 0, 1};
  const std::vector<float> w = {10, 20, 30, 40};

  const std::string first = WriteAdapter(2, 2, 2, b, a);
  const std::string second = WriteAdapter(2, 2, 2, b, a, {}, kOtherModule);
  std::vector<vllm::Ltx2LoraAdapter> adapters;
  for (const std::string& path : {first, second}) {
    vllm::Ltx2LoraSpec spec;
    spec.path = path;
    spec.strength = 2.0;
    adapters.push_back(vllm::Ltx2LoraAdapter::Open(spec, {kTarget, kOther}));
  }

  // Each tensor sees exactly ONE product, and it is the first form: `W + 2*B@A`.
  bool fused = false;
  const std::vector<float> q = FuseBf16(adapters, kTarget, 2, 2, w, &fused);
  CHECK(fused);
  CHECK(q[0] == doctest::Approx(12.0));
  CHECK(q[1] == doctest::Approx(24.0));
  CHECK(q[3] == doctest::Approx(42.0));

  const std::vector<float> k = FuseBf16(adapters, kOther, 2, 2, w, &fused);
  CHECK(fused);
  CHECK(k[0] == doctest::Approx(12.0));
  CHECK(k[1] == doctest::Approx(24.0));
  CHECK(k[3] == doctest::Approx(42.0));

  std::remove(first.c_str());
  std::remove(second.c_str());
}
