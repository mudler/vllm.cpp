// Task 3 real-tensor loader check for the Qwen3.6-35B-A3B-NVFP4 MoE gate.
//
// GATED on the real checkpoint: if the pinned snapshot is not present (CI,
// laptops) every case skips and passes. On dgx it opens shard 1 (which holds
// embed_tokens + layers 0-16, so both a GDN layer (0) and a full-attn layer (3)
// are fully resolvable from one file) and proves the loader RESOLVES real
// tensors by name, verifies the standalone dequant helpers against python/torch
// golden bf16 bit patterns, and checks the full-layer loader's current native
// resident contract (FP8/NVFP4 raw weights by default; snapshot 491c2f1e; see
// .agents/specs/qwen36-forward-notes.md §6). The full-model load is exercised in Task 5.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"

#include "hf_snapshot.h"

namespace fs = std::filesystem;

namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_old_) {
      setenv(name_.c_str(), old_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};

// Resolve the pinned 35B snapshot's shard 1, or "" if unavailable.
// GATE-PIN-UNPINNED-SNAPSHOTS (#471). Previously the first snapshot carrying a
// shard-1 file, whichever revision that was. The 35B goldens name their revision
// (`oracle.model` of goldens/qwen36_*_35b/manifest.json), so it is enforced.
std::string FindShard1() {
  const std::string snap = parity::Qwen36A3bNvfP4Snapshot();
  if (snap.empty()) return "";
  const fs::path shard = fs::path(snap) / "model-00001-of-00003.safetensors";
  std::error_code ec;
  if (!fs::exists(shard, ec)) return "";
  return shard.string();
}

uint16_t Bf16At(const vllm::OwnedTensor& t, int64_t i) {
  return reinterpret_cast<const uint16_t*>(t.bytes.data())[i];
}

}  // namespace

TEST_CASE("Qwen3.6-35B loader: real-tensor resolve + dequant + transpose") {
  const std::string shard1_path = FindShard1();
  if (shard1_path.empty()) {
    MESSAGE("SKIP: nvidia/Qwen3.6-35B-A3B-NVFP4 shard 1 not present");
    return;
  }
  const vllm::SafetensorsFile shard =
      vllm::SafetensorsFile::Open(shard1_path);
  const vllm::TensorResolver get =
      [&shard](const std::string& name) -> const vllm::StTensor& {
    return shard.Get(name);
  };

  SUBCASE("tensor naming + shapes match the config-derived dims") {
    // embed_tokens bf16 [vocab, H].
    const vllm::StTensor& embed =
        get("model.language_model.embed_tokens.weight");
    CHECK(embed.dtype == "BF16");
    REQUIRE(embed.shape.size() == 2);
    CHECK(embed.shape[0] == 248320);
    CHECK(embed.shape[1] == 2048);

    // GDN layer 0: in_proj_qkv FP8 [conv_dim=8192, H=2048].
    const vllm::StTensor& qkv = get(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
    CHECK(qkv.dtype == "F8_E4M3");
    CHECK(qkv.shape[0] == 8192);
    CHECK(qkv.shape[1] == 2048);

    // Full-attn layer 3: q_proj FP8 [2*Hq*Dh=8192, H], k_proj [Hkv*Dh=512, H].
    const vllm::StTensor& qp =
        get("model.language_model.layers.3.self_attn.q_proj.weight");
    CHECK(qp.dtype == "F8_E4M3");
    CHECK(qp.shape[0] == 8192);
    const vllm::StTensor& kp =
        get("model.language_model.layers.3.self_attn.k_proj.weight");
    CHECK(kp.shape[0] == 512);

    // MoE expert 0 NVFP4: gate_proj U8 [I=512, H/2=1024], down_proj [H, I/2].
    const vllm::StTensor& eg = get(
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight");
    CHECK(eg.dtype == "U8");
    CHECK(eg.shape[0] == 512);
    CHECK(eg.shape[1] == 1024);
    const vllm::StTensor& ed = get(
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight");
    CHECK(ed.shape[0] == 2048);
    CHECK(ed.shape[1] == 256);

    // Router gate bf16 [E=256, H=2048] (unquantized).
    const vllm::StTensor& rg = get("model.language_model.layers.0.mlp.gate.weight");
    CHECK(rg.dtype == "BF16");
    CHECK(rg.shape[0] == 256);
    CHECK(rg.shape[1] == 2048);
  }

  SUBCASE("NVFP4 expert dequant matches torch golden (pre-transpose)") {
    // layers.0.experts.0.down_proj: U8 [2048,256] -> bf16 [2048,512].
    const std::string p =
        "model.language_model.layers.0.mlp.experts.0.down_proj";
    const vllm::StTensor& w = get(p + ".weight");
    const vllm::StTensor& ws = get(p + ".weight_scale");
    const vllm::StTensor& ws2 = get(p + ".weight_scale_2");
    const int64_t out_dim = w.shape[0];       // 2048
    const int64_t in_dim = w.shape[1] * 2;    // 512
    float ws2v = 0.0F;
    std::memcpy(&ws2v, ws2.data, sizeof(float));

    std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
    vllm::DequantNvfp4ToBf16(w.data, ws.data, ws2v, out_dim, in_dim, dq.data());
    // Golden bf16 bit patterns from torch dequant (snapshot 491c2f1e).
    CHECK(dq[0 * in_dim + 0] == 0x3C7E);
    CHECK(dq[5 * in_dim + 17] == 0x3BCB);
    // Transpose correctness (dequant[o,i] -> transposed[i,o]) is checked on the
    // owned buffer in the full-layer SUBCASE below.
  }

  SUBCASE("FP8 attn dequant matches torch golden (pre-transpose)") {
    // layers.3.self_attn.q_proj: F8 [8192,2048], per-tensor weight_scale.
    const std::string p = "model.language_model.layers.3.self_attn.q_proj";
    const vllm::StTensor& w = get(p + ".weight");
    const vllm::StTensor& sc = get(p + ".weight_scale");
    const int64_t out_dim = w.shape[0];  // 8192
    const int64_t in_dim = w.shape[1];   // 2048
    float scale = 0.0F;
    std::memcpy(&scale, sc.data, sizeof(float));
    CHECK(scale == doctest::Approx(0.0007629395F).epsilon(1e-4));

    std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
    vllm::DequantFp8ToBf16(w.data, scale, out_dim * in_dim, dq.data());
    CHECK(dq[0 * in_dim + 0] == 0x3C0A);
    CHECK(dq[1 * in_dim + 5] == 0x3CE1);
  }

  SUBCASE("full-layer load: GDN (layer 0) and full-attn (layer 3)") {
    // A small expert count keeps the test fast; the loop/shared-expert/router
    // path is what we're proving, not all 256 experts (that is Task 5).
    const int64_t kExperts = 4;

    const vllm::Qwen3_5MoeLayerWeights gdn =
        vllm::LoadQwen3_5MoeLayer(get, "linear_attention", 0, kExperts);
    CHECK(gdn.is_linear_attention);
    // The default CUDA-parity loader keeps 35B W8A8 projections resident in
    // raw FP8 [N=out,K=in] orientation. VT_DENSE_NATIVE=0 is the legacy
    // dequant-to-bf16 path; the gate default should exercise the native fields.
    CHECK(gdn.gdn.in_proj_qkv.Empty());
    REQUIRE_FALSE(gdn.gdn.in_proj_qkv_fp8.Empty());
    CHECK(gdn.gdn.in_proj_qkv_fp8.n == 8192);
    CHECK(gdn.gdn.in_proj_qkv_fp8.k == 2048);
    CHECK(gdn.gdn.in_proj_qkv_fp8.packed.shape[0] == 8192);
    CHECK(gdn.gdn.in_proj_qkv_fp8.packed.shape[1] == 2048);
    CHECK(gdn.gdn.in_proj_qkv_fp8.weight_scale > 0.0F);
    CHECK(gdn.gdn.in_proj_qkv_fp8.input_scale > 0.0F);
    CHECK(gdn.gdn.in_proj_qkv_fp8.alpha > 0.0F);
    // W1 merged BA is a 27B-dense-only owner. The 35B native path keeps both
    // split projections and must never make VT_GDN_MERGED_BA selectable.
    CHECK(gdn.gdn.in_proj_ba.Empty());
    REQUIRE_FALSE(gdn.gdn.in_proj_b.Empty());
    REQUIRE_FALSE(gdn.gdn.in_proj_a.Empty());
    // W2 merged QKVZ is likewise 27B-dense-only. The 35B qkv/z are FP8 with
    // quant scales and must never populate the packed BF16 owner.
    CHECK(gdn.gdn.in_proj_qkvz.Empty());
    // conv1d collapsed to [conv_dim, K].
    CHECK(gdn.gdn.conv1d_weight.shape[0] == 8192);
    CHECK(gdn.gdn.conv1d_weight.shape[1] == 4);
    // a_log / dt_bias upcast to f32 [Hv=32].
    CHECK(gdn.gdn.a_log.dtype == vt::DType::kF32);
    CHECK(gdn.gdn.a_log.shape[0] == 32);
    // norm over Dv=128.
    CHECK(gdn.gdn.norm_weight.shape[0] == 128);
    // Router gate transposed to [H, E].
    CHECK(gdn.moe.router_gate.shape[0] == 2048);
    CHECK(gdn.moe.router_gate.shape[1] == 256);
    // Router gate transpose is value-preserving: gate[0,0] on disk == [0,0]
    // after transpose (corner is invariant). Golden 0x3BB3.
    CHECK(Bf16At(gdn.moe.router_gate, 0) == 0x3BB3);
    // NVFP4 expert weights are also resident now: raw packed [N=out,K/2] plus
    // fp8 group scales [N,K/16]. The helper-level dequant goldens above cover
    // the numeric bit patterns.
    CHECK(gdn.moe.expert_down.empty());
    REQUIRE(gdn.moe.expert_down_fp4.size() == static_cast<size_t>(kExperts));
    CHECK(gdn.moe.expert_down_fp4[0].n == 2048);
    CHECK(gdn.moe.expert_down_fp4[0].k == 512);
    CHECK(gdn.moe.expert_down_fp4[0].packed.shape[0] == 2048);
    CHECK(gdn.moe.expert_down_fp4[0].packed.shape[1] == 256);
    CHECK(gdn.moe.expert_down_fp4[0].scale.shape[0] == 2048);
    CHECK(gdn.moe.expert_down_fp4[0].scale.shape[1] == 32);
    CHECK(gdn.moe.expert_down_fp4[0].scale2 > 0.0F);

    const vllm::Qwen3_5MoeLayerWeights attn =
        vllm::LoadQwen3_5MoeLayer(get, "full_attention", 3, kExperts);
    CHECK_FALSE(attn.is_linear_attention);
    CHECK(attn.attn.q_proj.Empty());
    REQUIRE_FALSE(attn.attn.q_proj_fp8.Empty());
    CHECK(attn.attn.q_proj_fp8.n == 8192);
    CHECK(attn.attn.q_proj_fp8.k == 2048);
    CHECK(attn.attn.q_proj_fp8.packed.shape[0] == 8192);
    CHECK(attn.attn.q_proj_fp8.packed.shape[1] == 2048);
    CHECK(attn.attn.q_proj_fp8.weight_scale > 0.0F);
    CHECK(attn.attn.q_proj_fp8.input_scale > 0.0F);
    CHECK(attn.attn.q_proj_fp8.alpha > 0.0F);
    // qk-norm over head_dim=256.
    CHECK(attn.attn.q_norm.shape[0] == 256);
    CHECK(attn.attn.k_norm.shape[0] == 256);
    CHECK(attn.attn.o_proj.Empty());
    REQUIRE_FALSE(attn.attn.o_proj_fp8.Empty());
    CHECK(attn.attn.o_proj_fp8.n == 2048);
    CHECK(attn.attn.o_proj_fp8.k == 4096);
  }

  SUBCASE("35B legacy dense path also keeps BA and QKVZ split") {
    ScopedEnv legacy_dense("VT_DENSE_NATIVE", "0");
    const vllm::Qwen3_5MoeLayerWeights gdn =
        vllm::LoadQwen3_5MoeLayer(get, "linear_attention", 0, 1);
    REQUIRE(gdn.is_linear_attention);
    CHECK(gdn.gdn.in_proj_ba.Empty());
    REQUIRE_FALSE(gdn.gdn.in_proj_b.Empty());
    REQUIRE_FALSE(gdn.gdn.in_proj_a.Empty());
    CHECK(gdn.gdn.in_proj_qkvz.Empty());
    REQUIRE_FALSE(gdn.gdn.in_proj_qkv.Empty());
    REQUIRE_FALSE(gdn.gdn.in_proj_z.Empty());
  }
}

// --- Marlin host-weight residency (the 35B ~16.9 GiB peak-PSS lever) ----------
// After BuildMoeMarlinResident builds the device-resident Marlin repacked
// experts, the per-expert fp4 HOST mirror (LoadNvfp4Raw's MakeOwned copies in
// expert_*_fp4[e].{packed,scale}.bytes) is dead weight and is returned to the
// OS via OwnedTensor::ReleaseHost(). These tests pin (1) the release MECHANISM
// (CPU, always runs) and (2) the load-path INVARIANT through the public
// PrepareMarlinResident hook (DGX): freed when the Marlin resident is the
// committed path, RETAINED when the VT_NVFP4_MARLIN=0 wmma fallback
// (MoeBlockFusedCuda) will re-read them.

// --- Deferred routed-expert load: bounded load-phase coexistence -------------
// The 35B load-phase peak-PSS lever. When the safetensors loader shares the
// shards (disk path), LoadQwen3_5Moe DEFERS the routed-expert host copies and
// installs `load_layer_experts`; PrepareMarlinResident materializes ONE layer's
// experts immediately before that layer's device Marlin build + host free, so at
// most one layer's ~256 experts coexist on the host (vs all N layers before).
// This CPU test pins the CONTRACT that the interleave depends on — closure
// move-safety and the per-layer materialize→free coexistence bound — without a
// real NVFP4 checkpoint or a GPU. The end-to-end interleave on real weights is
// exercised by the DGX token gate (full model via the deferred disk load) plus
// memcheck (no use-after-free of the freed host bytes) and the peak-PSS A/B.
namespace {

// Stand-in for LoadMoeExpertsInto: fill `moe` with `e_count` routed experts,
// each carrying small resident packed+scale host bytes tagged by layer.
void FillFakeExperts(vllm::MoeBlockWeights& moe, int e_count, int layer) {
  for (int e = 0; e < e_count; ++e) {
    vllm::Nvfp4Weight g;
    g.n = 8;
    g.k = 16;
    g.packed.dtype = vt::DType::kI8;
    g.packed.rank = 2;
    g.packed.shape[0] = 8;
    g.packed.shape[1] = 8;
    g.packed.bytes.assign(64, static_cast<uint8_t>(layer + 1));
    g.scale.dtype = vt::DType::kI8;
    g.scale.bytes.assign(8, 0x3C);
    vllm::Nvfp4Weight u = g;
    vllm::Nvfp4Weight d = g;
    moe.expert_gate_fp4.push_back(std::move(g));
    moe.expert_up_fp4.push_back(std::move(u));
    moe.expert_down_fp4.push_back(std::move(d));
  }
}

// Layers whose routed-expert host bytes are currently resident (peak-tracking).
int LayersWithResidentExperts(const vllm::Qwen3_5MoeWeights& w) {
  int n = 0;
  for (const auto& layer : w.layers) {
    bool any = false;
    for (const auto& x : layer.moe.expert_gate_fp4)
      if (!x.packed.bytes.empty()) any = true;
    if (any) ++n;
  }
  return n;
}

}  // namespace

TEST_CASE("deferred routed-expert load: move-safe closure + bounded coexistence") {
  const int kLayers = 4;
  const int kExperts = 3;

  vllm::Qwen3_5MoeWeights loaded;
  loaded.layers.resize(static_cast<size_t>(kLayers));  // deferred: experts empty
  // kExperts is a const int with a constant initializer, so the lambda may name
  // it WITHOUT capturing it ([expr.prim.lambda.capture]/7 — no odr-use). The
  // redundant explicit capture drew -Wunused-lambda-capture from Clang on macOS
  // (BACKEND-METAL-MLX W0); dropped rather than suppressed, no behaviour change.
  loaded.load_layer_experts = [](int64_t layer, vllm::MoeBlockWeights& moe) {
    FillFakeExperts(moe, kExperts, static_cast<int>(layer));
  };

  // Deferred precondition: every layer's routed experts are EMPTY at load, and
  // the streaming closure is installed.
  for (const auto& layer : loaded.layers)
    CHECK(layer.moe.expert_gate_fp4.empty());
  REQUIRE(static_cast<bool>(loaded.load_layer_experts));

  // The closure MUST survive the model's move into the LoadedModel (it must not
  // capture the Qwen3_5MoeWeights by address — it takes the target MoE by ref).
  vllm::Qwen3_5MoeWeights w = std::move(loaded);
  REQUIRE(static_cast<bool>(w.load_layer_experts));

  // Drive the interleave contract: materialize layer l, then free its host bytes
  // (mirrors BuildMoeMarlinResident's per-layer ReleaseHost after the device
  // repack) BEFORE the next layer streams in.
  int peak = 0;
  for (int l = 0; l < kLayers; ++l) {
    w.load_layer_experts(l, w.layers[static_cast<size_t>(l)].moe);
    const auto& moe = w.layers[static_cast<size_t>(l)].moe;
    REQUIRE(moe.expert_gate_fp4.size() == static_cast<size_t>(kExperts));
    peak = std::max(peak, LayersWithResidentExperts(w));
    // Simulate this layer's device build + host free.
    auto& mmoe = w.layers[static_cast<size_t>(l)].moe;
    for (auto& x : mmoe.expert_gate_fp4) {
      x.packed.ReleaseHost();
      x.scale.ReleaseHost();
    }
    for (auto& x : mmoe.expert_up_fp4) {
      x.packed.ReleaseHost();
      x.scale.ReleaseHost();
    }
    for (auto& x : mmoe.expert_down_fp4) {
      x.packed.ReleaseHost();
      x.scale.ReleaseHost();
    }
  }

  // The whole-window peak host coexistence never exceeded ONE layer's experts.
  CHECK(peak == 1);
  // Every layer's experts were materialized (vectors present) yet the host bytes
  // are all returned after the per-layer free.
  for (const auto& layer : w.layers)
    CHECK(layer.moe.expert_gate_fp4.size() == static_cast<size_t>(kExperts));
  CHECK(LayersWithResidentExperts(w) == 0);
}

TEST_CASE("OwnedTensor::ReleaseHost frees host bytes but preserves logical presence") {
  vllm::OwnedTensor t;
  t.dtype = vt::DType::kI8;
  t.rank = 2;
  t.shape[0] = 64;
  t.shape[1] = 32;
  t.bytes.resize(64 * 32, 0x5A);
  REQUIRE_FALSE(t.Empty());
  REQUIRE(t.bytes.size() == 64u * 32u);

  t.ReleaseHost();

  // Dispatch metadata remains populated: the authoritative device resident may
  // still be selected after host staging is reclaimed.
  CHECK_FALSE(t.Empty());
  CHECK_FALSE(t.HasHostBytes());
  CHECK(t.bytes.empty());
  // swap-with-empty (not clear()) must actually deallocate the capacity.
  CHECK(t.bytes.capacity() == 0u);
  // Shape/dtype metadata is retained (only the host buffer is reclaimed).
  CHECK(t.rank == 2);
  CHECK(t.shape[0] == 64);
  CHECK_THROWS_WITH_AS(t.View(), doctest::Contains("released"),
                       std::runtime_error);
}

// --- Host-addressable device adoption: ONE copy of the bytes, not two --------
// BACKEND-VULKAN-LOADMEM. On a backend whose allocations the host can
// dereference (Vulkan: every buffer HOST_VISIBLE|HOST_COHERENT and persistently
// mapped), a weight that has been uploaded must not ALSO keep a host mirror --
// on GB10 both copies come out of the same unified RAM, so the model became
// resident twice (MEASURED: 16.392 GiB VmHWM against 8.622 GiB of Vulkan
// allocation for a 7.6 GiB Qwen3-4B).
//
// This pins the MECHANISM on the CPU tier, with no GPU and no checkpoint: what
// the buffer POINTS AT afterwards, that the bytes survive, that the device
// allocation is kept alive by the view, and -- the guard that keeps CUDA
// byte-identical -- that a backend which does NOT advertise host-addressable
// memory is left completely alone.
namespace {

// Minimal Backend whose "device" memory is plain host memory, so adoption is
// observable without a real accelerator. `host_addressable` is the one axis
// under test; `frees` counts Free() so the keep-alive can be proven.
class FakeHostAddressableBackend final : public vt::Backend {
 public:
  explicit FakeHostAddressableBackend(bool host_addressable)
      : host_addressable_(host_addressable) {}

  void* Alloc(size_t bytes) override {
    ++allocs;
    return std::malloc(bytes == 0 ? 1 : bytes);
  }
  void Free(void* p) override {
    ++frees;
    std::free(p);
  }
  void Memset(vt::Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  vt::Queue CreateQueue() override {
    return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  }
  bool UnifiedMemory() const override { return true; }
  bool DeviceMemoryIsHostAddressable() const override { return host_addressable_; }

  int allocs = 0;
  int frees = 0;

 private:
  bool host_addressable_;
};

// Build a weight with owned host bytes and upload it exactly the way
// ResidentWeight does, so the test exercises the real post-upload state.
vllm::OwnedTensor UploadedWeight(vt::Backend& b, vt::Queue& q, uint8_t pattern,
                                 size_t nbytes) {
  vllm::OwnedTensor w;
  w.dtype = vt::DType::kI8;
  w.rank = 1;
  w.shape[0] = static_cast<int64_t>(nbytes);
  w.bytes.resize(nbytes, pattern);
  void* p = b.Alloc(nbytes);
  b.Copy(q, p, w.bytes.data(), nbytes);
  vt::Backend* bk = &b;
  w.d_dev = std::shared_ptr<void>(p, [bk](void* x) { bk->Free(x); });
  return w;
}

}  // namespace

TEST_CASE("AdoptDeviceBytesAsHost leaves ONE copy on a host-addressable device") {
  ScopedEnv adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  FakeHostAddressableBackend b(/*host_addressable=*/true);
  vt::Queue q = b.CreateQueue();
  constexpr size_t kBytes = 4096;

  vllm::OwnedTensor w = UploadedWeight(b, q, 0xA7, kBytes);
  // Precondition: two distinct copies exist, which is the defect.
  REQUIRE(w.d_dev != nullptr);
  REQUIRE_FALSE(w.bytes.borrowed());
  REQUIRE(static_cast<const void*>(w.bytes.data()) != w.d_dev.get());

  vllm::AdoptDeviceBytesAsHost(b, w);

  // THE MECHANISM: the host buffer now IS the device allocation. Not "freed",
  // not "smaller" -- the same address, which is the only assertion that
  // distinguishes one copy from two.
  CHECK(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) == w.d_dev.get());
  CHECK(w.bytes.size() == kBytes);
  // The surviving copy holds the right bytes: every reader that used to read
  // the mirror reads these instead.
  bool all_match = true;
  for (size_t i = 0; i < kBytes; ++i) {
    if (w.bytes.data()[i] != 0xA7) { all_match = false; break; }
  }
  CHECK(all_match);
  // Logical presence is untouched -- nothing was released, so nothing may look
  // absent to dispatch.
  CHECK_FALSE(w.Empty());
  CHECK(w.HasHostBytes());
  CHECK_FALSE(w.host_released);

  // THE LIFETIME: the view keeps the allocation alive. Dropping d_dev alone
  // must NOT free the buffer the bytes still point at.
  CHECK(b.frees == 0);
  w.d_dev.reset();
  CHECK(b.frees == 0);
  CHECK(w.bytes.data()[0] == 0xA7);
  // Only when the view goes too.
  w.bytes.Reset();
  CHECK(b.frees == 1);
}

TEST_CASE("AdoptDeviceBytesAsHost is a NO-OP where device memory is not host-addressable") {
  ScopedEnv adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  // The guard that keeps every discrete-GPU path byte-identical: a CUDA
  // pointer is not host-dereferenceable, so adopting it would hand a device
  // address to a host memcpy.
  FakeHostAddressableBackend b(/*host_addressable=*/false);
  vt::Queue q = b.CreateQueue();
  constexpr size_t kBytes = 2048;

  vllm::OwnedTensor w = UploadedWeight(b, q, 0x31, kBytes);
  const void* host_before = w.bytes.data();

  vllm::AdoptDeviceBytesAsHost(b, w);

  CHECK_FALSE(w.bytes.borrowed());
  CHECK(static_cast<const void*>(w.bytes.data()) == host_before);
  CHECK(static_cast<const void*>(w.bytes.data()) != w.d_dev.get());
  CHECK(w.bytes.size() == kBytes);
}

TEST_CASE("AdoptDeviceBytesAsHost leaves a BORROWED buffer alone") {
  ScopedEnv adopt_default("VT_ADOPT_DEVICE_BYTES", "1");
  // A borrowed buffer owns no anonymous pages (a GGUF mmap, or the single
  // expansion a tied token_embd/lm_head pair shares). Adopting would reclaim
  // nothing and would break the tie, so it must be skipped.
  FakeHostAddressableBackend b(/*host_addressable=*/true);
  vt::Queue q = b.CreateQueue();
  constexpr size_t kBytes = 1024;

  auto shared = std::make_shared<std::vector<uint8_t>>(kBytes, 0x5C);
  vllm::OwnedTensor w;
  w.dtype = vt::DType::kI8;
  w.rank = 1;
  w.shape[0] = static_cast<int64_t>(kBytes);
  w.bytes = vllm::OwnedBytes::Borrow(shared->data(), kBytes, shared);
  void* p = b.Alloc(kBytes);
  b.Copy(q, p, w.bytes.data(), kBytes);
  vt::Backend* bk = &b;
  w.d_dev = std::shared_ptr<void>(p, [bk](void* x) { bk->Free(x); });

  vllm::AdoptDeviceBytesAsHost(b, w);

  CHECK(static_cast<const void*>(w.bytes.data()) == shared->data());
  CHECK(static_cast<const void*>(w.bytes.data()) != w.d_dev.get());
}

namespace {
// The Marlin runtime gate (MarlinMoeEnabled()) caches VT_NVFP4_MARLIN on first
// use, so it cannot be flipped mid-process; the effective gate equals the LAUNCH
// env. Key the assertion on that so both branches are deterministic per launch:
// a default run exercises RELEASE, a `VT_NVFP4_MARLIN=0` run exercises RETENTION.
[[maybe_unused]] bool MarlinGateOffAtLaunch() {  // used only in VT_MARLIN_NVFP4 builds
  const char* e = std::getenv("VT_NVFP4_MARLIN");
  return e != nullptr && e[0] == '0';
}

bool WeightsTestHasCuda() {
  try {
    vt::GetBackend(vt::DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}
}  // namespace

TEST_CASE(
    "PrepareMarlinResident residency: routed expert host bytes freed under "
    "Marlin, retained under the wmma fallback") {
  const std::string shard1_path = FindShard1();
  if (shard1_path.empty()) {
    MESSAGE("SKIP: nvidia/Qwen3.6-35B-A3B-NVFP4 shard 1 not present");
    return;
  }
  if (!WeightsTestHasCuda()) {
    MESSAGE("SKIP: no CUDA device (this asserts the device-resident load path)");
    return;
  }
#ifndef VT_MARLIN_NVFP4
  MESSAGE("SKIP: built without VT_MARLIN_NVFP4");
  return;
#else
  const vllm::SafetensorsFile shard = vllm::SafetensorsFile::Open(shard1_path);
  const vllm::TensorResolver get =
      [&shard](const std::string& name) -> const vllm::StTensor& {
    return shard.Get(name);
  };
  const int64_t kExperts = 4;

  vllm::Qwen3_5MoeWeights weights;
  weights.layers.push_back(
      vllm::LoadQwen3_5MoeLayer(get, "linear_attention", 0, kExperts));
  const vllm::MoeBlockWeights& moe = weights.layers[0].moe;

  // A minimal config sufficient for BuildMoeMarlinResident (E / H=K / I=N only;
  // shapes read from the real down projection [N=H, K=I]).
  vllm::HfConfig cfg;
  cfg.num_experts = kExperts;
  cfg.hidden_size = moe.expert_down_fp4[0].n;          // H
  cfg.moe_intermediate_size = moe.expert_down_fp4[0].k;  // I
  cfg.num_experts_per_tok = 8;  // unused by the repack; 35B value for realism

  // Precondition: the raw fp4 host mirror is resident straight after load.
  REQUIRE(moe.expert_gate_fp4.size() == static_cast<size_t>(kExperts));
  REQUIRE_FALSE(moe.expert_gate_fp4[0].packed.bytes.empty());
  REQUIRE_FALSE(moe.expert_down_fp4[0].scale.bytes.empty());

  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = gpu.CreateQueue();
  vllm::Qwen3_5Model::PrepareMarlinResident(weights, cfg, q);
  gpu.DestroyQueue(q);

  if (!MarlinGateOffAtLaunch()) {
    // Marlin resident is the committed compute path → every routed expert's
    // packed+scale host buffer is returned (the 35B peak-PSS lever).
    for (int64_t e = 0; e < kExperts; ++e) {
      const size_t se = static_cast<size_t>(e);
      CHECK(moe.expert_gate_fp4[se].packed.bytes.empty());
      CHECK(moe.expert_gate_fp4[se].scale.bytes.empty());
      CHECK(moe.expert_up_fp4[se].packed.bytes.empty());
      CHECK(moe.expert_up_fp4[se].scale.bytes.empty());
      CHECK(moe.expert_down_fp4[se].packed.bytes.empty());
      CHECK(moe.expert_down_fp4[se].scale.bytes.empty());
    }
  } else {
    // VT_NVFP4_MARLIN=0: PrepareMarlinResident is a no-op and the wmma fallback
    // (MoeBlockFusedCuda) re-reads these host bytes via ResidentNvfp4 on every
    // first touch, so the guard MUST retain them.
    for (int64_t e = 0; e < kExperts; ++e) {
      const size_t se = static_cast<size_t>(e);
      CHECK_FALSE(moe.expert_gate_fp4[se].packed.bytes.empty());
      CHECK_FALSE(moe.expert_up_fp4[se].packed.bytes.empty());
      CHECK_FALSE(moe.expert_down_fp4[se].scale.bytes.empty());
    }
  }
#endif  // VT_MARLIN_NVFP4
}

TEST_CASE("ReleaseHost drops the FILE DESCRIPTOR with the bytes it described") {
  // F10 of the ENG-EXPERT-STREAM wiring review (#912).
  //
  // `mmap_fd`/`mmap_file_offset` say where a BORROWED tensor's host bytes
  // physically live. `ReleaseHost` on a borrowed buffer drops the keep-alive and
  // clears `mmap_src`, and it used to leave the descriptor set. That leaves a
  // record that outlives its own subject: `bytes.data()` is now null, so an
  // expert-stream slice would pread the recorded offset -- which belongs to a
  // tensor nothing is reading any more -- and every released tower would key on
  // the same `TowerId(nullptr)`.
  std::vector<uint8_t> backing(256, 0x5A);
  auto owner = std::make_shared<std::vector<uint8_t>>(backing);

  vllm::OwnedTensor t;
  t.dtype = vt::DType::kI8;
  t.rank = 2;
  t.shape[0] = 8;
  t.shape[1] = 32;
  t.bytes = vllm::OwnedBytes::Borrow(owner->data(), owner->size(), owner);
  t.mmap_src = owner->data();
  t.mmap_src_bytes = owner->size();
  t.mmap_fd = 7;            // any plausible live descriptor
  t.mmap_file_offset = 4096;
  REQUIRE(t.bytes.borrowed());

  t.ReleaseHost();

  CHECK(t.bytes.data() == nullptr);
  CHECK(t.mmap_src == nullptr);
  CHECK(t.mmap_src_bytes == 0u);
  // THE ASSERTIONS THAT WERE MISSING. A descriptor without bytes is not a
  // source, and reading it as one is the "right offset, wrong tensor" failure.
  CHECK(t.mmap_fd == -1);
  CHECK(t.mmap_file_offset == 0u);
}

TEST_CASE("a tower's identity survives an address the allocator hands out again") {
  // #1066. `TowerUid` exists because the expert slot cache is a process-lifetime
  // singleton and an ADDRESS is not an identity across model lifetimes.
  vllm::OwnedTensor a;
  a.dtype = vt::DType::kI8;
  a.rank = 1;
  a.shape[0] = 64;
  a.bytes.resize(64, 0x11);
  const uint64_t uid_a = a.TowerUid();
  CHECK(uid_a != 0u);
  CHECK(a.TowerUid() == uid_a);  // stable while the bytes stay put

  vllm::OwnedTensor b;
  b.dtype = vt::DType::kI8;
  b.rank = 1;
  b.shape[0] = 64;
  b.bytes.resize(64, 0x22);
  // Two live tensors are always distinct, whatever addresses they happen to hold.
  CHECK(b.TowerUid() != uid_a);

  // A COPY carries the field but not the identity: its buffer is a different
  // one, so inheriting the original's id would be exactly the collision this
  // guards against.
  vllm::OwnedTensor c = a;
  CHECK(c.TowerUid() != uid_a);

  // And re-stamping is keyed on the buffer, so replacing the bytes with a
  // DIFFERENTLY ADDRESSED buffer yields a new identity rather than silently
  // reusing the old one.
  const uint64_t uid_c = c.TowerUid();
  c.bytes = vllm::OwnedBytes(std::vector<uint8_t>(128, 0x33));
  REQUIRE(c.bytes.data() != nullptr);
  CHECK(c.TowerUid() != uid_c);
}

TEST_CASE("a tower's identity is keyed on its ADDRESS, and says so") {
  // #1091 finding 5. The field comment used to promise an identity for "this
  // tensor's CURRENT bytes"; the implementation keys on `bytes.data()` and
  // re-stamps only when that address moves. The two are not the same claim, and
  // the case above cannot tell them apart because a fresh `std::vector` lands
  // somewhere else and so satisfies both readings.
  //
  // This one separates them, deterministically, by borrowing: a borrowed view
  // names an address the test controls, so "same address, different contents"
  // is constructible rather than a matter of allocator luck. #1066 was a comment
  // on this exact field that promised more than the code delivered, so the limit
  // is pinned here rather than left to be rediscovered.
  auto one = std::make_shared<std::vector<uint8_t>>(64, 0x44);
  auto two = std::make_shared<std::vector<uint8_t>>(64, 0x55);

  vllm::OwnedTensor t;
  t.dtype = vt::DType::kI8;
  t.rank = 1;
  t.shape[0] = 64;
  t.bytes = vllm::OwnedBytes::Borrow(one->data(), one->size(), one);
  const uint64_t uid = t.TowerUid();
  CHECK(uid != 0u);

  // Same address, wholly different contents. The uid does NOT move — which is
  // the documented limit, not a defect, because nothing rewrites a tower in
  // place. A future caller that did would need a different key.
  std::fill(one->begin(), one->end(), 0x99);
  CHECK(t.bytes.data() == one->data());
  CHECK(t.TowerUid() == uid);

  // A different address is a different tower, which is the half the cache
  // depends on.
  t.bytes = vllm::OwnedBytes::Borrow(two->data(), two->size(), two);
  REQUIRE(t.bytes.data() != one->data());
  CHECK(t.TowerUid() != uid);
}
