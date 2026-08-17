// ENG-EXPERT-STREAM, issue #1123. The arithmetic and the predicate behind the
// load-time refusal of a GGUF whose weights cannot be staged onto the target
// device. The REACHABILITY half — that the loader actually asks — is a separate
// binary, test_gguf_device_fit_reach, because it has to register a fake staging
// platform in a global registry.
//
// Why the numbers here are the ones they are: a Q8_0 block is 34 bytes per 32
// elements, so `elems * 2` (bf16) is 64 and the on-disk size is the smaller
// term; an F32 tensor is 4 bytes per element, so `elems * 2` is the smaller
// term. One file with both therefore pins BOTH arms of
// `min(gguf_bytes, elems * model_dtype_bytes)` in a single sum, and a mutation
// that drops either arm changes it.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "support/test_env.h"
#include "vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_device_fit.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"

namespace {

using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;

// One Q8_0 block: f16 scale + 32 int8 quants = 34 bytes for 32 elements.
std::string Q8Block() {
  std::string b(2, '\0');
  b[0] = '\0';
  b[1] = '\x3c';  // f16 1.0, little-endian
  for (int i = 0; i < 32; ++i) b.push_back(static_cast<char>(i));
  return b;
}

// A GGUF with exactly two tensors:
//   "t_q8"  Q8_0, 32 elements  -> 34 bytes on disk, 64 bytes expanded to bf16
//   "t_f32" F32,  8 elements   -> 32 bytes on disk, 16 bytes expanded to bf16
// So the staged lower bound is min(34,64) + min(32,16) = 34 + 16 = 50.
constexpr size_t kExpectedLowerBound = 50;
constexpr size_t kExpectedTensors = 2;
constexpr size_t kExpectedLargest = 34;  // "t_q8"

std::string BuildTwoTensorGguf() {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "llama"));
  b.AddTensor("t_q8", {32}, /*ggml_type=*/8, Q8Block());
  b.AddTensor("t_f32", {4, 2}, /*ggml_type=*/0, std::string(32, '\1'));
  return b.Build();
}

// The same file plus one tensor of the MTP / `nextn` block, which a DEFAULT load
// never stages: the head is attached only under
// `params.speculative_config.has_value() && method == "mtp"`
// (`src/vllm/entrypoints/model_loader.cpp:1452-1453`), and the main model reads
// `block_count - nextn_predict_layers` blocks
// (`qwen3_5_gguf_weights.cpp:877-878`), so the head's blocks are outside its
// range. The footprint counts it anyway, because it takes the whole tensor
// table. A second Q8_0 block, so its staged term is min(34, 64) = 34.
constexpr size_t kNextnStaged = 34;

std::string BuildGgufWithUnstagedNextnBlock() {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "llama"));
  b.AddTensor("t_q8", {32}, /*ggml_type=*/8, Q8Block());
  b.AddTensor("t_f32", {4, 2}, /*ggml_type=*/0, std::string(32, '\1'));
  b.AddTensor("blk.1.nextn.eh_proj.weight", {32}, /*ggml_type=*/8, Q8Block());
  return b.Build();
}

}  // namespace

TEST_CASE("gguf_device_fit: the footprint takes min(on-disk, expanded) per tensor") {
  TempFile f(BuildTwoTensorGguf());
  const vllm::GgufFile gguf = vllm::GgufFile::Open(f.path());
  REQUIRE(gguf.Tensors().size() == kExpectedTensors);

  const vllm::GgufStagedFootprint fp = vllm::GgufStagedWeightFootprint(gguf);
  // The count is asserted, not assumed: a bound that cannot say how many
  // tensors it examined has not reported one.
  CHECK(fp.tensor_count == kExpectedTensors);
  CHECK(fp.lower_bound_bytes == kExpectedLowerBound);
  CHECK(fp.largest_tensor_bytes == kExpectedLargest);
  CHECK(fp.largest_tensor_name == "t_q8");

  // Summing is enough BECAUSE the two arms disagree: an implementation that
  // always took the on-disk size would give 34 + 32 = 66, one that always
  // expanded would give 64 + 16 = 80, and both differ from 50. That is why the
  // fixture carries one tensor of each kind rather than two of one kind.
}

TEST_CASE("gguf_device_fit: a wider model dtype cannot raise the on-disk term") {
  TempFile f(BuildTwoTensorGguf());
  const vllm::GgufFile gguf = vllm::GgufFile::Open(f.path());
  // f32 model dtype: the Q8_0 term stays 34 (min against 128) and the F32 term
  // becomes min(32, 32) = 32. So the sum moves to 66 and NOT to 34 + 128.
  const vllm::GgufStagedFootprint fp =
      vllm::GgufStagedWeightFootprint(gguf, /*model_dtype_bytes=*/4);
  CHECK(fp.tensor_count == kExpectedTensors);
  CHECK(fp.lower_bound_bytes == 66);
}

// The bound's ONE over-count direction, made executable rather than only
// described. Every other case in this file runs on a fixture whose tensors are
// all staged, so the footprint there happens to EQUAL the true staged size and
// the boundary cases cannot tell an exact quantity from an over-counted one.
//
// This case separates them. It exists because the spec and the commit body for
// this change asserted the bound was a lower bound "so the refusal can never
// over-refuse", which is false: a tensor counted and never staged is a positive
// over-count, and one is present on every default load. On the measured
// checkpoint that is the `nextn` block, 8,940,488,704 of 397,245,341,184 bytes
// (2.2506 %). The two error directions are on DIFFERENT quantities and do not
// cancel, so "the under-count dominates" does not rescue the claim. Recorded and
// owned by issue #1136; the header states the direction, and this pins it.
//
// NOTE for whoever closes #1136 by teaching the bound which tensors this load will
// stage: this case is SUPPOSED to go red then, and it is not an obstacle. It
// characterises today's contract, so changing the contract means changing it here
// too — deliberately, in the same commit, rather than discovering later that the
// bound quietly stopped counting something.
TEST_CASE("gguf_device_fit: a tensor the loader never stages is COUNTED, so the bound can over-refuse") {
  TempFile f(BuildGgufWithUnstagedNextnBlock());
  const vllm::GgufFile gguf = vllm::GgufFile::Open(f.path());
  REQUIRE(gguf.Tensors().size() == kExpectedTensors + 1);

  const vllm::GgufStagedFootprint fp = vllm::GgufStagedWeightFootprint(gguf);
  CHECK(fp.tensor_count == kExpectedTensors + 1);
  CHECK(fp.lower_bound_bytes == kExpectedLowerBound + kNextnStaged);

  // The consequence at the boundary: a default load stages
  // `kExpectedLowerBound` bytes, the predicate compares
  // `kExpectedLowerBound + kNextnStaged`, and every budget in between refuses a
  // weight set that fits. Both ends of that window are asserted, so a change
  // that narrowed or widened the over-count moves this case.
  for (const size_t budget :
       {kExpectedLowerBound, kExpectedLowerBound + kNextnStaged - 1}) {
    CAPTURE(budget);
    const vllm::DeviceWeightFit fit =
        vllm::CheckDeviceWeightFit(gguf, "cuda", true, budget);
    CHECK(fit.refuse);
    CHECK(fit.needed_bytes == kExpectedLowerBound + kNextnStaged);
  }
  // At the counted total it does not refuse, which pins the over-count to
  // exactly this tensor and nothing more.
  const vllm::DeviceWeightFit ok = vllm::CheckDeviceWeightFit(
      gguf, "cuda", true, kExpectedLowerBound + kNextnStaged);
  CHECK_FALSE(ok.refuse);
}

TEST_CASE("gguf_device_fit: a non-staging platform is never refused, at any budget") {
  TempFile f(BuildTwoTensorGguf());
  const vllm::GgufFile gguf = vllm::GgufFile::Open(f.path());
  for (const size_t budget : {size_t{0}, size_t{1}, kExpectedLowerBound - 1,
                              kExpectedLowerBound, size_t{1} << 40}) {
    CAPTURE(budget);
    const vllm::DeviceWeightFit fit = vllm::CheckDeviceWeightFit(
        gguf, "cpu", /*needs_weight_staging=*/false, budget);
    CHECK_FALSE(fit.refuse);
    CHECK(fit.message.empty());
    // Nothing is even computed on this arm, which is what makes every CPU load
    // byte-identical to before.
    CHECK(fit.needed_bytes == 0);
  }
}

TEST_CASE("gguf_device_fit: an unknown budget is not a verdict") {
  TempFile f(BuildTwoTensorGguf());
  const vllm::GgufFile gguf = vllm::GgufFile::Open(f.path());
  const vllm::DeviceWeightFit fit = vllm::CheckDeviceWeightFit(
      gguf, "cuda", /*needs_weight_staging=*/true, /*budget_bytes=*/0);
  CHECK_FALSE(fit.refuse);
  CHECK(fit.message.empty());
  CHECK(fit.budget_bytes == 0);
}

TEST_CASE("gguf_device_fit: refuses strictly above the budget, and not at or below it") {
  TempFile f(BuildTwoTensorGguf());
  const vllm::GgufFile gguf = vllm::GgufFile::Open(f.path());

  SUBCASE("one byte under the footprint refuses") {
    const vllm::DeviceWeightFit fit = vllm::CheckDeviceWeightFit(
        gguf, "cuda", true, kExpectedLowerBound - 1);
    CHECK(fit.refuse);
    CHECK(fit.needed_bytes == kExpectedLowerBound);
    CHECK(fit.budget_bytes == kExpectedLowerBound - 1);
  }
  SUBCASE("exactly the footprint does NOT refuse") {
    const vllm::DeviceWeightFit fit =
        vllm::CheckDeviceWeightFit(gguf, "cuda", true, kExpectedLowerBound);
    CHECK_FALSE(fit.refuse);
    CHECK(fit.needed_bytes == kExpectedLowerBound);
  }
  SUBCASE("a generous budget does NOT refuse") {
    const vllm::DeviceWeightFit fit =
        vllm::CheckDeviceWeightFit(gguf, "cuda", true, size_t{1} << 40);
    CHECK_FALSE(fit.refuse);
    CHECK(fit.message.empty());
  }
}

TEST_CASE("gguf_device_fit: the refusal names the device, both numbers, the missing part and the remedy") {
  TempFile f(BuildTwoTensorGguf());
  const vllm::GgufFile gguf = vllm::GgufFile::Open(f.path());
  const vllm::DeviceWeightFit fit =
      vllm::CheckDeviceWeightFit(gguf, "cuda", true, /*budget_bytes=*/8);
  REQUIRE(fit.refuse);
  const std::string& m = fit.message;
  // A refusal that does not name what is missing is the behaviour this change
  // exists to replace, so each half is asserted rather than the message length.
  CHECK(m.find("device 'cuda'") != std::string::npos);
  CHECK(m.find(std::to_string(kExpectedLowerBound)) != std::string::npos);
  CHECK(m.find("8 bytes") != std::string::npos);
  CHECK(m.find("t_q8") != std::string::npos);
  CHECK(m.find("HOST-ONLY") != std::string::npos);
  CHECK(m.find("device=cpu") != std::string::npos);
  CHECK(m.find("#1123") != std::string::npos);
  CHECK(m.find("VT_DEVICE_WEIGHT_BUDGET_BYTES") != std::string::npos);
}

TEST_CASE("gguf_device_fit: VT_DEVICE_WEIGHT_BUDGET_BYTES overrides, and a typo does NOT disable the guard") {
  SUBCASE("unset: the platform's probe is the budget") {
    vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
    CHECK(vllm::DeviceWeightBudgetBytes(4096) == 4096);
    CHECK(vllm::DeviceWeightBudgetBytes(0) == 0);
  }
  SUBCASE("set: the override wins in both directions") {
    vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", "64");
    CHECK(vllm::DeviceWeightBudgetBytes(4096) == 64);
    CHECK(vllm::DeviceWeightBudgetBytes(0) == 64);
    vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  }
  SUBCASE("an explicit 0 disables the check, which is a documented escape") {
    vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", "0");
    CHECK(vllm::DeviceWeightBudgetBytes(4096) == 0);
    vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  }
  SUBCASE("trailing garbage is IGNORED, not read as 0") {
    // Reading "12x" as 0 would silently disable the refusal on a typo, which is
    // the invisible-fallback shape this tree refuses. The probe must survive.
    for (const char* bad : {"12x", "x", "-1", " 64", "64 ", "1e9", ""}) {
      CAPTURE(bad);
      vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", bad);
      CHECK(vllm::DeviceWeightBudgetBytes(4096) == 4096);
    }
    vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  }
}
