// ─── safetensors payloads have NO alignment guarantee (issue #772) ───────────
//
// A tensor's first byte sits at `8 + <JSON header length> + <sum of the
// preceding tensors' sizes>`. Not one of those three terms is required to be
// even, so a BF16 tensor beginning on an ODD address is an ordinary safetensors
// file, not a corrupt one. Four loaders formed a `const uint16_t*` over that
// address anyway — undefined behaviour on every target, and a real fault on the
// strict-alignment ones this project builds for (`build-test-cpu-arm64`,
// Jetson/Orin sm_110):
//
//   voxtral.cpp:51     StBf16ToF32      — cast then INDEXED (UBSan sees this)
//   voxtral.cpp:344    PermuteQKBf16    — cast then row-address + memcpy
//   qwen3_vl.cpp:78    LoadVisionF32    — cast then INDEXED (UBSan sees this)
//   qwen3_5_mtp.cpp:71 CopyRawNK        — cast, `+ offset`, then memcpy
//
// This is the FOURTH recurrence of one class: #301 (closed, and the source of
// the `vt::LoadUnaligned` seam), #627 (`qwen3_5_weights.cpp`) and #674 /
// PR #688 (`ltx2_loader.cpp`) are the others.
//
// TWO OF THESE FOUR ARE INVISIBLE TO THE SANITIZER, WHICH IS WHY THIS FILE
// EXISTS RATHER THAN ANOTHER UBSAN SWEEP. `PermuteQKBf16` and `CopyRawNK` form
// and do arithmetic on the misaligned `uint16_t*` but then LAUNDER every access
// through `std::memcpy`, which reads bytes — so `-fsanitize=alignment` never
// fires on them. All three previous recurrences were found by UBSan; these two
// would have survived every one of those sweeps. For them the guarantee this
// file pins is VALUE CORRECTNESS at an odd offset: the fix rewrites element
// arithmetic (`uint16_t* + n`) into byte arithmetic (`unsigned char* + 2n`), and
// dropping that factor of two is exactly the mistake a reviewer must be able to
// see fail. Every case below therefore checks the LOADED VALUES, not merely that
// the load returned.
//
// EVERY CASE FORCES THE ODD OFFSET AND THEN ASSERTS IT. Mirroring
// `test_ltx2_video.cpp`'s "ODD safetensors payload offset (#674)" case: the JSON
// header is padded by one space (trailing whitespace is legal JSON, and padding
// the header is exactly how real writers align their payloads) until the payload
// lands on an odd byte, and the mapped ADDRESS parity is REQUIREd rather than
// inferred. A fixture edit that makes the address even fails the REQUIRE instead
// of passing while covering nothing.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <doctest/doctest.h>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"
// INTERNAL header, reached through -I${CMAKE_SOURCE_DIR}/src: the two Voxtral
// loader steps are not public ABI and must not become so for a test's sake.
#include "vllm/model_executor/models/voxtral_loader_internal.h"
#include "vt/dtype.h"

namespace {

// A bare temp directory. Deliberately NOT any model's Workspace fixture: these
// cases must not depend on a fixture whose tensor sizes could silently change
// the parity they exist to force.
struct TempDir {
  std::string root;
  TempDir() {
    static int counter = 0;
    root = "/tmp/vllm_unaligned_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++);
    ::mkdir(root.c_str(), 0755);
  }
  ~TempDir() {
    const int rc = std::system(("rm -rf '" + root + "'").c_str());
    (void)rc;
  }
};

struct Spec {
  std::string name;
  std::vector<int64_t> shape;
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  return n;
}

// The BF16 bit pattern this fixture stores at flat element index `i` of tensor
// `t`. Deliberately a BIT PATTERN and not a float: the expectation is then
// `vt::BF16ToF32` of the very bits on disk, so a case can assert EQUALITY rather
// than a tolerance, and a read shifted by one byte is a hard failure instead of
// something a band could absorb. The low bits vary per element so a row-address
// error moves the value; 0x3d00 keeps every pattern a small finite positive.
uint16_t FixtureBits(size_t t, size_t i) {
  return static_cast<uint16_t>(0x3d00U + ((t * 37U + i * 7U) & 0x1ffU));
}

std::string U64Le(uint64_t v) {
  std::string s;
  for (int i = 0; i < 8; ++i) s.push_back(static_cast<char>((v >> (8 * i)) & 0xFFU));
  return s;
}

// Build an all-BF16 safetensors file from `specs`, padding the counted JSON
// header with `header_pad` spaces. Returns the file bytes; `*payload_at` gets
// the absolute file offset of the FIRST tensor's first byte.
//
// Every BF16 tensor is an even number of bytes, so the parity of the first
// payload byte is the parity of EVERY tensor in the file — which is what lets
// one space of padding put all of them on odd addresses at once.
std::string BuildBf16Safetensors(const std::vector<Spec>& specs,
                                 size_t header_pad, size_t* payload_at) {
  std::string header = "{";
  std::string body;
  uint64_t offset = 0;
  for (size_t i = 0; i < specs.size(); ++i) {
    const int64_t n = Numel(specs[i].shape);
    const auto nbytes = static_cast<uint64_t>(n) * 2;
    if (i != 0) header += ",";
    header += "\"" + specs[i].name + "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (size_t d = 0; d < specs[i].shape.size(); ++d) {
      if (d != 0) header += ",";
      header += std::to_string(specs[i].shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + nbytes) + "]}";
    offset += nbytes;

    const size_t at = body.size();
    body.resize(at + static_cast<size_t>(nbytes));
    for (size_t e = 0; e < static_cast<size_t>(n); ++e) {
      const uint16_t v = FixtureBits(i, e);
      std::memcpy(body.data() + at + e * 2, &v, 2);
    }
  }
  header += "}";
  header.append(header_pad, ' ');
  *payload_at = 8 + header.size();
  return U64Le(header.size()) + header + body;
}

void WriteFileBytes(const std::string& path, const std::string& bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  REQUIRE(f != nullptr);
  REQUIRE(std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
  REQUIRE(std::fclose(f) == 0);
}

// Write `specs` twice — unpadded, and padded by one space — and keep whichever
// lands the payload on an ODD byte. Doing it both ways means the case never
// depends on the exact length of the generated JSON above: a rename that changes
// the header length flips which file is kept, not whether the offset is odd.
// The REQUIRE proves the two spellings really do differ in parity, so a future
// change that made padding inert could not pass silently.
std::string WriteOddOffsetSafetensors(const TempDir& ws, const std::string& stem,
                                      const std::vector<Spec>& specs) {
  const std::string a = ws.root + "/" + stem + "_a.safetensors";
  const std::string b = ws.root + "/" + stem + "_b.safetensors";
  size_t off_a = 0;
  size_t off_b = 0;
  WriteFileBytes(a, BuildBf16Safetensors(specs, 0, &off_a));
  WriteFileBytes(b, BuildBf16Safetensors(specs, 1, &off_b));
  REQUIRE((off_a % 2) != (off_b % 2));
  return (off_a % 2 == 1) ? a : b;
}

// The fixture really is what every case here claims: this tensor's MAPPED
// address is odd, so no loader below can satisfy a `uint16_t`'s alignment by
// luck. mmap bases are page-aligned, so file-offset parity IS address parity —
// but assert the address rather than infer it.
void RequireOddlyMapped(const vllm::SafetensorsFile& file, const std::string& name) {
  INFO("tensor: " << name);
  REQUIRE((reinterpret_cast<uintptr_t>(file.Get(name).data) % 2) == 1);
}

// Expected f32 for element `i` of the `t`-th tensor in a spec list.
float Expected(size_t t, size_t i) { return vt::BF16ToF32(FixtureBits(t, i)); }

}  // namespace

// ─── voxtral.cpp:51 — StBf16ToF32 ───────────────────────────────────────────
//
// UBSan-VISIBLE: the cast is followed by `src[i]`, so under
// `-fsanitize=alignment` this case reports "load of misaligned address ... for
// type 'const uint16_t', which requires 2 byte alignment" before the fix.
//
// Driven through the internal declaration rather than `LoadVoxtralWeights`,
// which cannot be reached at test scale: `VoxtralEncoderConfig()` is FIXED at 32
// layers of d_model 1280 / ffn 5120, so a synthetic checkpoint satisfying it is
// ~1.2 GiB. See voxtral_loader_internal.h.
TEST_CASE("voxtral StBf16ToF32 reads a BF16 tensor at an ODD offset (#772)") {
  const TempDir ws;
  const std::vector<Spec> specs = {{"adapter.weight", {3, 5}}};
  const std::string path = WriteOddOffsetSafetensors(ws, "voxtral_st", specs);

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
  RequireOddlyMapped(file, "adapter.weight");

  const std::vector<float> got = vllm::VoxtralStBf16ToF32(file.Get("adapter.weight"));
  REQUIRE(got.size() == 15U);
  for (size_t i = 0; i < got.size(); ++i) {
    INFO("element " << i);
    CHECK(got[i] == doctest::Approx(Expected(0, i)).scale(0.0));
  }
}

// ─── voxtral.cpp:344 — PermuteQKBf16 ────────────────────────────────────────
//
// UBSan-INVISIBLE: every access is a `memcpy`, so only the VALUES can witness
// the fix. The permutation is `out(h*hd + j*hd2 + i) <- in(h*hd + 2i + j)`, so
// each output row must equal a DIFFERENT, identifiable input row — an off-by-one
// row address (the failure mode of rewriting `&src[row*K]` into a byte offset
// without the `*2`) lands on the wrong row and every element of it changes.
TEST_CASE("voxtral PermuteQKBf16 permutes a BF16 tensor at an ODD offset (#772)") {
  const TempDir ws;
  constexpr int64_t kHeads = 2;
  constexpr int64_t kHeadDim = 4;  // hd2 = 2
  constexpr int64_t kK = 3;
  const std::vector<Spec> specs = {{"wq.weight", {kHeads * kHeadDim, kK}}};
  const std::string path = WriteOddOffsetSafetensors(ws, "voxtral_permute", specs);

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
  RequireOddlyMapped(file, "wq.weight");

  const std::vector<uint16_t> got =
      vllm::VoxtralPermuteQKBf16(file.Get("wq.weight"), kHeads);
  REQUIRE(got.size() == static_cast<size_t>(kHeads * kHeadDim * kK));

  constexpr int64_t kHd2 = kHeadDim / 2;
  for (int64_t h = 0; h < kHeads; ++h) {
    for (int64_t i = 0; i < kHd2; ++i) {
      for (int64_t j = 0; j < 2; ++j) {
        const int64_t out_row = h * kHeadDim + j * kHd2 + i;
        const int64_t in_row = h * kHeadDim + 2 * i + j;
        for (int64_t c = 0; c < kK; ++c) {
          INFO("out_row " << out_row << " col " << c << " <- in_row " << in_row);
          CHECK(got[static_cast<size_t>(out_row * kK + c)] ==
                FixtureBits(0, static_cast<size_t>(in_row * kK + c)));
        }
      }
    }
  }
}

// ─── qwen3_vl.cpp:78 — LoadVisionF32 ────────────────────────────────────────
//
// UBSan-VISIBLE (cast then `p[i]` inside Bf16BitsToF32). Driven through the
// PRODUCTION entry point `LoadQwen3VLVisionWeights`: with `depth = 0` and no
// deepstack indexes the tower needs exactly the nine tensors below, which is the
// smallest checkpoint that reaches the loader without a copy of it. Qwen3.6-27B
// really does ship an EMPTY deepstack_visual_indexes, so depth aside this is a
// shape the loader is expected to accept.
TEST_CASE("qwen3-vl vision loader reads BF16 tensors at an ODD offset (#772)") {
  const TempDir ws;
  const std::string v = "model.visual.";
  const std::vector<Spec> specs = {
      {v + "patch_embed.proj.weight", {4, 3}},
      {v + "patch_embed.proj.bias", {4}},
      {v + "pos_embed.weight", {5, 2}},
      {v + "merger.norm.weight", {4}},
      {v + "merger.norm.bias", {4}},
      {v + "merger.linear_fc1.weight", {3, 4}},
      {v + "merger.linear_fc1.bias", {3}},
      {v + "merger.linear_fc2.weight", {2, 3}},
      {v + "merger.linear_fc2.bias", {2}},
  };
  const std::string path = WriteOddOffsetSafetensors(ws, "qwen3vl_vision", specs);

  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(path));
  for (const Spec& s : specs) RequireOddlyMapped(shards[0], s.name);

  vllm::multimodal::Qwen3VLVisionConfig vc;
  vc.depth = 0;
  vc.deepstack_visual_indexes.clear();
  const vllm::multimodal::Qwen3VLVisionWeights vw =
      vllm::LoadQwen3VLVisionWeights(shards, vc);

  // Spot every tensor the loader read, each against ITS OWN spec index, so a
  // read that drifted into a neighbouring tensor is caught rather than absorbed.
  struct Check {
    size_t spec_index;
    const std::vector<float>* got;
  };
  const std::vector<Check> checks = {
      {0, &vw.patch_proj_w},        {1, &vw.patch_proj_b},
      {2, &vw.pos_embed_w},         {3, &vw.merger.norm_w},
      {4, &vw.merger.norm_b},       {5, &vw.merger.fc1_w},
      {6, &vw.merger.fc1_b},        {7, &vw.merger.fc2_w},
      {8, &vw.merger.fc2_b},
  };
  for (const Check& c : checks) {
    INFO("tensor: " << specs[c.spec_index].name);
    REQUIRE(c.got->size() == static_cast<size_t>(Numel(specs[c.spec_index].shape)));
    for (size_t i = 0; i < c.got->size(); ++i) {
      INFO("element " << i);
      CHECK((*c.got)[i] == doctest::Approx(Expected(c.spec_index, i)).scale(0.0));
    }
  }
}

// ─── qwen3_5_mtp.cpp:71 — CopyRawNK ─────────────────────────────────────────
//
// THE ONE A SANITIZER SWEEP CANNOT REACH. `CopyRawNK` forms
// `reinterpret_cast<const uint16_t*>(source.data) + offset` and then memcpy's
// from it: the misaligned pointer is formed and advanced, but never
// dereferenced as a `uint16_t`, so `-fsanitize=alignment` stays silent. Only
// reading the code — or this case — finds it.
//
// What the case pins is the ELEMENT-vs-BYTE arithmetic. `offset` counts BF16
// ELEMENTS, so the byte-pointer rewrite must advance `offset * 2` bytes; the
// two-expert MoE below gives expert 1 a non-zero `offset` for all three of
// gate/up/down, so a rewrite that dropped the `* 2` reads expert 0's rows into
// expert 1 (and the up-projection slice, whose offset is `gu_base +
// intermediate*hidden`, lands on the gate rows). Each expected value is derived
// from the fixture's own stacked layout, so those confusions are visible as
// specific wrong values and not merely as "something differs".
TEST_CASE("qwen3.5 MTP stacked-expert slice at an ODD offset (#772)") {
  const TempDir ws;
  constexpr int64_t kHidden = 4;
  constexpr int64_t kHeadDim = 2;
  constexpr int64_t kHeads = 2;   // q_proj packs q|gate => [2*Q, H]
  constexpr int64_t kKvHeads = 1;
  constexpr int64_t kExperts = 2;
  constexpr int64_t kMoeInter = 3;
  constexpr int64_t kShared = 2;
  constexpr int64_t kQ = kHeads * kHeadDim;
  constexpr int64_t kKv = kKvHeads * kHeadDim;

  const std::string b = "mtp.layers.0.";
  const std::string sa = b + "self_attn.";
  const std::string mlp = b + "mlp.";
  const std::vector<Spec> specs = {
      {"mtp.fc.weight", {kHidden, 2 * kHidden}},
      {"mtp.pre_fc_norm_embedding.weight", {kHidden}},
      {"mtp.pre_fc_norm_hidden.weight", {kHidden}},
      {"mtp.norm.weight", {kHidden}},
      {b + "input_layernorm.weight", {kHidden}},
      {b + "post_attention_layernorm.weight", {kHidden}},
      {sa + "q_proj.weight", {2 * kQ, kHidden}},
      {sa + "k_proj.weight", {kKv, kHidden}},
      {sa + "v_proj.weight", {kKv, kHidden}},
      {sa + "o_proj.weight", {kHidden, kQ}},
      {sa + "q_norm.weight", {kHeadDim}},
      {sa + "k_norm.weight", {kHeadDim}},
      {mlp + "gate.weight", {kExperts, kHidden}},
      {mlp + "shared_expert_gate.weight", {1, kHidden}},
      {mlp + "experts.gate_up_proj", {kExperts, 2 * kMoeInter, kHidden}},
      {mlp + "experts.down_proj", {kExperts, kHidden, kMoeInter}},
      {mlp + "shared_expert.gate_proj.weight", {kShared, kHidden}},
      {mlp + "shared_expert.up_proj.weight", {kShared, kHidden}},
      {mlp + "shared_expert.down_proj.weight", {kHidden, kShared}},
  };
  constexpr size_t kGateUpSpec = 14;
  constexpr size_t kDownSpec = 15;
  REQUIRE(specs[kGateUpSpec].name == mlp + "experts.gate_up_proj");
  REQUIRE(specs[kDownSpec].name == mlp + "experts.down_proj");

  const std::string path = WriteOddOffsetSafetensors(ws, "qwen35_mtp", specs);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(path));
  RequireOddlyMapped(shards[0], specs[kGateUpSpec].name);
  RequireOddlyMapped(shards[0], specs[kDownSpec].name);

  vllm::HfConfig config;
  config.hidden_size = kHidden;
  config.head_dim = kHeadDim;
  config.num_attention_heads = kHeads;
  config.num_key_value_heads = kKvHeads;
  config.num_experts = kExperts;
  config.num_experts_per_tok = 1;
  config.moe_intermediate_size = kMoeInter;
  config.shared_expert_intermediate_size = kShared;

  const vllm::Qwen3_5MTPWeights w =
      vllm::LoadQwen3_5MTP(shards, config, vllm::Qwen3_5MTPKind::kMoe);
  REQUIRE(w.NumLayers() == 1);
  const vllm::MoeBlockWeights& moe = w.moe_layers[0].moe;
  REQUIRE(moe.expert_gate.size() == static_cast<size_t>(kExperts));

  // The stacked source layouts are gate_up[E, 2I, H] and down[E, H, I]; the
  // loader slices gate at `e*2*I*H`, up at `e*2*I*H + I*H`, down at `e*H*I`.
  const auto* gate_up_bytes = shards[0].Get(specs[kGateUpSpec].name).data;
  const auto* down_bytes = shards[0].Get(specs[kDownSpec].name).data;
  REQUIRE(gate_up_bytes != nullptr);
  REQUIRE(down_bytes != nullptr);

  for (int64_t e = 0; e < kExperts; ++e) {
    const auto gu_base = static_cast<size_t>(e * 2 * kMoeInter * kHidden);
    const auto down_base = static_cast<size_t>(e * kHidden * kMoeInter);
    const auto rows_gu = static_cast<size_t>(kMoeInter * kHidden);
    const auto rows_down = static_cast<size_t>(kHidden * kMoeInter);

    const auto& gate = moe.expert_gate[static_cast<size_t>(e)];
    const auto& up = moe.expert_up[static_cast<size_t>(e)];
    const auto& down = moe.expert_down[static_cast<size_t>(e)];
    REQUIRE(gate.bytes.size() == rows_gu * 2);
    REQUIRE(up.bytes.size() == rows_gu * 2);
    REQUIRE(down.bytes.size() == rows_down * 2);

    for (size_t i = 0; i < rows_gu; ++i) {
      uint16_t bits = 0;
      std::memcpy(&bits, gate.bytes.data() + i * 2, 2);
      INFO("expert " << e << " gate element " << i);
      CHECK(bits == FixtureBits(kGateUpSpec, gu_base + i));

      std::memcpy(&bits, up.bytes.data() + i * 2, 2);
      INFO("expert " << e << " up element " << i);
      CHECK(bits == FixtureBits(kGateUpSpec, gu_base + rows_gu + i));
    }
    for (size_t i = 0; i < rows_down; ++i) {
      uint16_t bits = 0;
      std::memcpy(&bits, down.bytes.data() + i * 2, 2);
      INFO("expert " << e << " down element " << i);
      CHECK(bits == FixtureBits(kDownSpec, down_base + i));
    }
  }
}

// ─── minimax_h3_vae_loader.cpp:96-101 — the seam-routed sibling ──────────────
//
// NOT a defect: this loop already did the byte-wise load, and already carried
// the reason in prose. It is here because it hand-rolled `std::memcpy` where the
// shared `vt::LoadUnaligned` seam exists — the parallel path AGENTS.md's
// shared-seam rule is about — and routing it through the seam must be inert.
// This case is the inertness proof for that rewrite: the same odd-offset load,
// through the production entry point, must still return the exact bytes on disk.
TEST_CASE("minimax-h3 audio VAE reads a BF16 tensor at an ODD offset (#772)") {
  const TempDir ws;
  // `.bias`, not `.weight`: a 3-D `.weight` takes the materialized-weight-norm
  // branch, which derives magnitudes and is not the plain read under test.
  const std::vector<Spec> specs = {{"decoder.dec_in_proj.bias", {6}}};
  const std::string path = WriteOddOffsetSafetensors(ws, "minimax_vae", specs);

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
  RequireOddlyMapped(file, "decoder.dec_in_proj.bias");

  const vllm::MiniMaxH3AudioVaeWeights w = vllm::LoadMiniMaxH3AudioVaeWeights(file);
  // `decoder.` is stripped by the loader.
  REQUIRE(w.tensors.count("dec_in_proj.bias") == 1);
  const std::vector<float>& got = w.tensors.at("dec_in_proj.bias");
  REQUIRE(got.size() == 6U);
  for (size_t i = 0; i < got.size(); ++i) {
    INFO("element " << i);
    CHECK(got[i] == doctest::Approx(Expected(0, i)).scale(0.0));
  }
}
