// LOAD-MODELOPT-NVFP4-BORROW (issue #1647), spec
// `.agents/specs/load-modelopt-nvfp4-borrow.md` — WHERE a loaded weight's bytes
// LIVE, asked through the production loader.
//
// THE DEFECT. `LoadNvfp4AnyNaming` has two arms. The compressed-tensors arm
// (`LoadCtNvfp4Raw`) has carried `BorrowStTensorBytes`-then-`MakeOwned` since
// ENG-LOAD-DIRECT-UPLOAD (#150); the ModelOpt arm went straight to `MakeOwned` +
// `std::memcpy` for BOTH `packed` and `scale`, with no borrow attempt at all.
// `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` declares `quant_method: "modelopt"`,
// so all 193 of its NVFP4 modules took that arm and 21 GiB landed as ANONYMOUS
// heap where upstream holds reclaimable page cache
// (`weight_utils.py:969-974,1247` at the pin `5559679229bc961848b121ccdeaa8fa5d79bec98`).
// The expensive consequence is the SECOND one: `AdoptDeviceBytesAsHost` is
// gated on `w.mmap_src != nullptr && w.bytes.borrowed()`, so `ResidentNvfp4`'s
// post-upload adoption — the step that releases the consumed source pages and,
// on a host-addressable device, collapses the host and device copies onto one
// buffer — was a silent no-op for every ModelOpt weight. On a GB10 (119 GiB
// UNIFIED) that is a packed host copy plus a packed device copy out of the same
// pool, and the box reboots.
//
// WHY THESE CASES USE A REAL FILE. `BorrowStTensorBytes` FAILS CLOSED when
// `StTensor::mapping` is null (`qwen3_5_weights.cpp:434`), which is exactly what
// an in-memory fake resolver produces. A fake would therefore report the defect
// as already fixed. Every case here writes a real safetensors shard to disk,
// opens it with `SafetensorsFile::Open`, and enters through the PRODUCTION
// loader `vllm::LoadQwen3_5Dense` — "the loader" of `AGENTS.md`
// `## Nothing lands dead`, not the private helper under test.
//
// NO GPU IS NEEDED and none is used. Where a weight's bytes live is an
// observable, deterministic property of the loaded container; the second half of
// the file supplies the one platform shape a CPU tier cannot otherwise reach —
// unified memory that nevertheless STAGES — through a fake platform in the
// otherwise-unused kXPU slot, the same device the `ResidentWeight` host-alias
// gate borrows for the same reason.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/platforms/interface.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace {

using vllm::Nvfp4Weight;
using vllm::OwnedTensor;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

// ── A safetensors shard, written to disk ─────────────────────────────────────

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xFF);
  return s;
}

// One tensor staged for the writer: name, dtype string, shape, payload bytes.
struct Entry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

// Deterministic filler, distinct per name, so a wrong copy shows up in the
// VALUES and not only in a pointer comparison.
std::vector<uint8_t> Fill(size_t n, uint8_t tag) {
  std::vector<uint8_t> b(n);
  for (size_t i = 0; i < n; ++i)
    b[i] = static_cast<uint8_t>((i * 31U + tag * 7U + 1U) & 0xFFU);
  return b;
}

uint8_t TagOf(const std::string& name) {
  uint32_t h = 2166136261U;
  for (const char c : name) {
    h ^= static_cast<uint8_t>(c);
    h *= 16777619U;
  }
  return static_cast<uint8_t>(h & 0xFFU);
}

void Add(std::vector<Entry>& out, const std::string& name,
         const std::string& dtype, std::vector<int64_t> shape,
         size_t elem_bytes) {
  size_t numel = 1;
  for (const int64_t d : shape) numel *= static_cast<size_t>(d);
  out.push_back({name, dtype, std::move(shape),
                 Fill(numel * elem_bytes, TagOf(name))});
}

// An F32 scalar tensor with an explicit, NON-ZERO value: every global-scale read
// in the loader refuses a zero divisor by name.
void AddF32Scalar(std::vector<Entry>& out, const std::string& name, float v) {
  std::vector<uint8_t> b(sizeof(float));
  std::memcpy(b.data(), &v, sizeof(v));
  out.push_back({name, "F32", {}, std::move(b)});
}

class TempShard {
 public:
  explicit TempShard(const std::vector<Entry>& entries) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_dense_load_residency_" + std::to_string(counter++) + "_" +
              std::to_string(static_cast<long>(::getpid())) + ".safetensors"))
                .string();
    nlohmann::json header = nlohmann::json::object();
    uint64_t offset = 0;
    std::string body;
    for (const Entry& e : entries) {
      nlohmann::json meta = nlohmann::json::object();
      meta["dtype"] = e.dtype;
      meta["shape"] = e.shape;
      meta["data_offsets"] = {offset, offset + e.bytes.size()};
      header[e.name] = meta;
      body.append(reinterpret_cast<const char*>(e.bytes.data()), e.bytes.size());
      offset += e.bytes.size();
    }
    const std::string h = header.dump();
    std::ofstream out(path_, std::ios::binary);
    const std::string prefix = U64Le(h.size());
    out.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
  }
  ~TempShard() { std::remove(path_.c_str()); }
  TempShard(const TempShard&) = delete;
  TempShard& operator=(const TempShard&) = delete;
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// ── The synthetic checkpoint ─────────────────────────────────────────────────
//
// One `full_attention` layer, tied logits (no `lm_head.*`, so the loader ties to
// the embedding table and never reaches the head dispatch). Shapes are the
// smallest legal ones: the NVFP4 in-dim must be a multiple of 16, and nothing in
// the dense loader cross-checks a projection shape against the config.

constexpr int64_t kHidden = 32;
constexpr int64_t kVocab = 8;
constexpr int64_t kOut = 32;  // NVFP4 out-dim for every projection here

// ModelOpt spelling: `weight` U8 [out, in/2] + `weight_scale` F8_E4M3
// [out, in/16] + `weight_scale_2` F32 scalar (the SCALE itself).
void AddModelOptNvfp4(std::vector<Entry>& out, const std::string& proj) {
  Add(out, proj + ".weight", "U8", {kOut, kHidden / 2}, 1);
  Add(out, proj + ".weight_scale", "F8_E4M3", {kOut, kHidden / 16}, 1);
  AddF32Scalar(out, proj + ".weight_scale_2", 0.125F);
}

// compressed-tensors spelling: `weight_packed` + `weight_scale` +
// `weight_global_scale` (a DIVISOR) + `input_global_scale`, which
// `LoadCtNvfp4Raw` reads unconditionally.
void AddCtNvfp4(std::vector<Entry>& out, const std::string& proj) {
  Add(out, proj + ".weight_packed", "U8", {kOut, kHidden / 2}, 1);
  Add(out, proj + ".weight_scale", "F8_E4M3", {kOut, kHidden / 16}, 1);
  AddF32Scalar(out, proj + ".weight_global_scale", 8.0F);
  AddF32Scalar(out, proj + ".input_global_scale", 4.0F);
}

enum class Spelling { kModelOpt, kCompressedTensors, kPlainBf16 };

std::vector<Entry> Checkpoint(Spelling spelling) {
  std::vector<Entry> e;
  Add(e, "model.embed_tokens.weight", "BF16", {kVocab, kHidden}, 2);
  Add(e, "model.norm.weight", "BF16", {kHidden}, 2);
  const std::string base = "model.layers.0.";
  Add(e, base + "input_layernorm.weight", "BF16", {kHidden}, 2);
  Add(e, base + "post_attention_layernorm.weight", "BF16", {kHidden}, 2);
  const std::string sa = base + "self_attn.";
  const std::string mlp = base + "mlp.";
  const char* kProjections[] = {"q_proj", "k_proj", "v_proj", "o_proj"};
  const char* kMlp[] = {"gate_proj", "up_proj", "down_proj"};
  for (const char* p : kProjections) {
    switch (spelling) {
      case Spelling::kModelOpt: AddModelOptNvfp4(e, sa + p); break;
      case Spelling::kCompressedTensors: AddCtNvfp4(e, sa + p); break;
      case Spelling::kPlainBf16:
        Add(e, sa + std::string(p) + ".weight", "BF16", {kOut, kHidden}, 2);
        break;
    }
  }
  Add(e, sa + "q_norm.weight", "BF16", {kHidden}, 2);
  Add(e, sa + "k_norm.weight", "BF16", {kHidden}, 2);
  for (const char* p : kMlp) {
    switch (spelling) {
      case Spelling::kModelOpt: AddModelOptNvfp4(e, mlp + p); break;
      case Spelling::kCompressedTensors: AddCtNvfp4(e, mlp + p); break;
      case Spelling::kPlainBf16:
        Add(e, mlp + std::string(p) + ".weight", "BF16", {kOut, kHidden}, 2);
        break;
    }
  }
  return e;
}

vllm::HfConfig ConfigFor() {
  vllm::HfConfig c;
  c.model_type = "qwen3_5";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = kHidden;
  c.num_hidden_layers = 1;
  c.layer_types = {"full_attention"};
  c.vocab_size = kVocab;
  c.num_attention_heads = 2;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.intermediate_size = kOut;
  // No `quantization_config`: this file gates WHERE the loaded bytes live, and
  // the declared-algorithm cross-check is `test_qwen38_27b_modelopt_mtp_arm`'s
  // subject. An empty object is what a checkpoint without the key parses to.
  c.raw = nlohmann::json::object();
  return c;
}

// Did this weight's bytes come from the mapping rather than from a fresh
// allocation? Three independent facts, because any one of them alone can be
// satisfied by accident: the borrow flag, the pointer INSIDE the mapped range,
// and the direct-upload source descriptor `AdoptDeviceBytesAsHost` keys on.
void CheckBorrowsMapping(const OwnedTensor& w, const vllm::StTensor& src,
                         const std::string& what) {
  INFO("weight: " << what);
  // Compared as `uintptr_t`, not as pointers: doctest stringifies a pointer
  // through its bool conversion, so a failing pointer CHECK prints `1 == 1` and
  // says nothing about WHICH addresses disagreed.
  const auto addr = [](const void* p) { return reinterpret_cast<uintptr_t>(p); };
  CHECK(w.bytes.borrowed());
  CHECK(addr(w.bytes.data()) == addr(src.data));
  CHECK(addr(w.mmap_src) == addr(src.data));
  CHECK(w.mmap_src_bytes == src.nbytes);
  CHECK(w.bytes.size() == src.nbytes);
}

void CheckNvfp4Borrows(const Nvfp4Weight& w, const vllm::SafetensorsFile& st,
                       const std::string& proj, const std::string& packed_name,
                       const std::string& scale_name) {
  CheckBorrowsMapping(w.packed, st.Get(proj + packed_name), proj + packed_name);
  CheckBorrowsMapping(w.scale, st.Get(proj + scale_name), proj + scale_name);
}

// Every NVFP4 projection the one-layer checkpoint produces, in load order.
std::vector<std::pair<std::string, const Nvfp4Weight*>> Nvfp4Projections(
    const vllm::Qwen3_5DenseWeights& w) {
  const vllm::Qwen3_5DenseLayerWeights& l = w.layers.at(0);
  return {
      {"model.layers.0.self_attn.q_proj", &l.attn.q_proj_fp4},
      {"model.layers.0.self_attn.k_proj", &l.attn.k_proj_fp4},
      {"model.layers.0.self_attn.v_proj", &l.attn.v_proj_fp4},
      {"model.layers.0.self_attn.o_proj", &l.attn.o_proj_fp4},
      {"model.layers.0.mlp.gate_proj", &l.mlp.gate_proj_fp4},
      {"model.layers.0.mlp.up_proj", &l.mlp.up_proj_fp4},
      {"model.layers.0.mlp.down_proj", &l.mlp.down_proj_fp4},
  };
}

// RAII around the process-wide direct-upload seam, so a failing CHECK cannot
// leak a forced decision into the next case.
class ForcedDirectUpload {
 public:
  explicit ForcedDirectUpload(bool on) {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(on);
  }
  ~ForcedDirectUpload() {
    vllm::detail::SetLoadDirectUploadOverrideForTesting(std::nullopt);
  }
};

}  // namespace

TEST_CASE(
    "dense load: the MODELOPT NVFP4 arm BORROWS the mapping, it does not copy") {
  const ForcedDirectUpload arm(true);
  const TempShard shard(Checkpoint(Spelling::kModelOpt));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(shard.path()));

  const vllm::load_stats::Counters before = vllm::load_stats::Snapshot();
  const vllm::Qwen3_5DenseWeights w =
      vllm::LoadQwen3_5Dense(shards, ConfigFor());
  const vllm::load_stats::Counters after = vllm::load_stats::Snapshot();

  REQUIRE(w.layers.size() == 1);
  REQUIRE_FALSE(w.layers[0].is_linear_attention);
  // The routing precondition, so a case that stopped reaching the NVFP4 arm
  // cannot pass by finding nothing to check.
  REQUIRE_FALSE(w.layers[0].mlp.gate_proj_fp4.Empty());
  REQUIRE(w.layers[0].attn.q_proj.Empty());

  size_t packed_and_scale_bytes = 0;
  for (const auto& [proj, fp4] : Nvfp4Projections(w)) {
    CheckNvfp4Borrows(*fp4, shards[0], proj, ".weight", ".weight_scale");
    packed_and_scale_bytes += fp4->packed.bytes.size() + fp4->scale.bytes.size();
  }
  REQUIRE(packed_and_scale_bytes > 0);

  // THE BYTE ACCOUNTING, which is the half a pointer comparison cannot make:
  // those ranges must land under `borrowed` and NOT under `host_copy`.
  CHECK(after.borrowed_bytes - before.borrowed_bytes >= packed_and_scale_bytes);
  CHECK(after.host_copy_bytes - before.host_copy_bytes < packed_and_scale_bytes);
}

TEST_CASE(
    "dense load: the COMPRESSED-TENSORS NVFP4 arm still borrows") {
  // The regression half. A repair that MOVED the defect rather than removing it
  // — by unifying the two arms onto the copying one — passes the case above and
  // fails here.
  const ForcedDirectUpload arm(true);
  const TempShard shard(Checkpoint(Spelling::kCompressedTensors));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(shard.path()));

  const vllm::Qwen3_5DenseWeights w =
      vllm::LoadQwen3_5Dense(shards, ConfigFor());
  REQUIRE(w.layers.size() == 1);
  REQUIRE_FALSE(w.layers[0].mlp.gate_proj_fp4.Empty());
  for (const auto& [proj, fp4] : Nvfp4Projections(w)) {
    CheckNvfp4Borrows(*fp4, shards[0], proj, ".weight_packed", ".weight_scale");
  }
}

TEST_CASE(
    "dense load: the ModelOpt arm FAILS CLOSED and loads the SAME bytes") {
  // The lever is a memory-residency decision and never a correctness one. With
  // the direct-upload decision forced OFF the ModelOpt arm must fall back to its
  // owned copy and produce byte-identical payloads and scalars, which is what
  // makes the borrow safe to enable by default.
  const TempShard shard(Checkpoint(Spelling::kModelOpt));

  std::vector<uint8_t> copied_packed;
  std::vector<uint8_t> copied_scale;
  float copied_scale2 = 0.0F;
  {
    const ForcedDirectUpload off(false);
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(shard.path()));
    const vllm::Qwen3_5DenseWeights w =
        vllm::LoadQwen3_5Dense(shards, ConfigFor());
    const Nvfp4Weight& g = w.layers.at(0).mlp.gate_proj_fp4;
    REQUIRE_FALSE(g.Empty());
    // The fallback owns its buffer; that is the arm under test here.
    CHECK_FALSE(g.packed.bytes.borrowed());
    CHECK_FALSE(g.scale.bytes.borrowed());
    copied_packed.assign(g.packed.bytes.data(),
                         g.packed.bytes.data() + g.packed.bytes.size());
    copied_scale.assign(g.scale.bytes.data(),
                        g.scale.bytes.data() + g.scale.bytes.size());
    copied_scale2 = g.scale2;
  }

  const ForcedDirectUpload on(true);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(shard.path()));
  const vllm::Qwen3_5DenseWeights w =
      vllm::LoadQwen3_5Dense(shards, ConfigFor());
  const Nvfp4Weight& g = w.layers.at(0).mlp.gate_proj_fp4;
  REQUIRE_FALSE(g.Empty());
  REQUIRE(g.packed.bytes.size() == copied_packed.size());
  REQUIRE(g.scale.bytes.size() == copied_scale.size());
  CHECK(std::memcmp(g.packed.bytes.data(), copied_packed.data(),
                    copied_packed.size()) == 0);
  CHECK(std::memcmp(g.scale.bytes.data(), copied_scale.data(),
                    copied_scale.size()) == 0);
  CHECK(g.scale2 == copied_scale2);
  CHECK(g.n == kOut);
  CHECK(g.k == kHidden);
}

// ── The second predicate: unified memory that nevertheless STAGES ────────────
//
// `DirectDeviceLoadEligible` required `!platform.is_unified_memory()`, so
// `StageAndReleaseLoadedDense` — the ONLY caller of
// `ReleaseResidentQwen3_5DenseHostWeights` — never ran on a GB10. Two predicates
// disagree about one device and `needs_weight_staging()` is the right one:
// `src/vllm/platforms/cuda.cpp:85` returns true unconditionally on CUDA
// "regardless of GB10 being physically unified", and
// `include/vllm/platforms/interface.h:294-298` records that
// `is_unified_memory()` "answers the OPPOSITE question".
//
// The platform this needs does not exist on a CPU tier, so the fake below
// supplies it in the otherwise-unused kXPU slot, over a backend that allocates
// with `malloc` — which is what makes a STAGED weight a real, inspectable
// allocation at an address that differs from the weight's own bytes.

namespace {

class StagingHostBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    ++allocs;
    return std::malloc(bytes == 0 ? 1 : bytes);
  }
  void Free(void* p) override { std::free(p); }
  void Memset(Queue&, void* p, int v, size_t bytes) override {
    std::memset(p, v, bytes);
  }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override {
    return Queue{Device{DeviceType::kXPU, 0}, nullptr};
  }
  void DestroyQueue(Queue&) override {}
  // TRUE, matching GB10's CUDA backend (`cuda_backend.cu:113`): host and device
  // address the same physical RAM. This is the bit the old predicate read.
  bool UnifiedMemory() const override { return true; }
  // FALSE, also matching GB10: a `cudaMalloc` pointer is not
  // host-dereferenceable there. It keeps `AdoptDeviceBytesAsHost` from folding a
  // staged weight back onto its host buffer, so "staged" stays visibly staged.
  bool DeviceMemoryIsHostAddressable() const override { return false; }

  int allocs = 0;
};

StagingHostBackend& FakeBackend() {
  static StagingHostBackend b;
  return b;
}

// Unified memory AND weight staging AND a residency policy that releases the
// host mirror after upload — the three answers a GB10 gives.
class UnifiedStagingPlatform final : public vllm::platforms::Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kXPU; }
  vt::Backend& backend() const override { return FakeBackend(); }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return {12, 1};
  }
  std::vector<DType> supported_dtypes() const override {
    return {DType::kBF16};
  }
  vllm::platforms::ResidencyPolicy residency_policy() const override {
    vllm::platforms::ResidencyPolicy p;
    p.release_host_weights_after_upload = true;
    return p;
  }
  bool needs_weight_staging() const override { return true; }
  // FALSE: `ResidentWeight`'s host-alias branch is a different lever and would
  // leave `d_dev` null, so nothing would be staged and nothing released. This
  // case is about the OTHER arm.
  bool host_memory_is_device_addressable() const override { return false; }
};

UnifiedStagingPlatform& FakePlatform() {
  static UnifiedStagingPlatform p;
  return p;
}

struct Registrar {
  Registrar() {
    vt::RegisterBackend(Device{DeviceType::kXPU, 0}, &FakeBackend());
    vllm::platforms::RegisterPlatform(DeviceType::kXPU, &FakePlatform());
  }
};
const Registrar kRegistrar;

}  // namespace

TEST_CASE(
    "dense load: UNIFIED memory that STAGES reaches stage-and-release") {
  const ForcedDirectUpload arm(true);
  const TempShard shard(Checkpoint(Spelling::kPlainBf16));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(shard.path()));

  // The precondition, narrated so a reader can see WHICH platform shape this
  // measures rather than inferring it from the fake's source.
  const vllm::platforms::Platform& p =
      vllm::platforms::GetPlatform(DeviceType::kXPU);
  REQUIRE_MESSAGE(p.is_unified_memory(),
                  "this case measures a UNIFIED-memory platform");
  REQUIRE_MESSAGE(p.needs_weight_staging(),
                  "...that nevertheless STAGES its weights");
  REQUIRE(p.residency_policy().release_host_weights_after_upload);

  Queue q{Device{DeviceType::kXPU, 0}, nullptr};
  const int allocs_before = FakeBackend().allocs;
  const vllm::Qwen3_5DenseWeights w =
      vllm::LoadQwen3_5Dense(shards, ConfigFor(), &q);

  REQUIRE(w.layers.size() == 1);
  // The routing precondition: stage-and-release runs only while the checkpoint
  // is plain bf16, so a case whose fixture stopped being one would pass while
  // measuring nothing.
  REQUIRE(vllm::IsPlainBf16Qwen3_5Dense(w));

  // STAGED: the backend really allocated — counted, so this says HOW MANY
  // allocations were observed rather than only that a pointer is non-null...
  CHECK(FakeBackend().allocs > allocs_before);
  REQUIRE(w.embed_tokens.d_dev != nullptr);
  // ...and RELEASED: the redundant host mirror is gone while the weight stays
  // dispatch-visible through its device copy.
  CHECK_FALSE(w.embed_tokens.HasHostBytes());
  CHECK_FALSE(w.embed_tokens.Empty());
  REQUIRE(w.final_norm.d_dev != nullptr);
  CHECK_FALSE(w.final_norm.HasHostBytes());
  const vllm::Qwen3_5DenseLayerWeights& l = w.layers[0];
  REQUIRE(l.attn.q_proj.d_dev != nullptr);
  CHECK_FALSE(l.attn.q_proj.HasHostBytes());
  REQUIRE(l.mlp.down_proj.d_dev != nullptr);
  CHECK_FALSE(l.mlp.down_proj.HasHostBytes());
}
