// MiniMax-Music3 — the AUTOREGRESSIVE half against the REAL checkpoint (#672).
//
// The companion of tests/vllm/models/test_minimax_music3_ar.cpp. That gate runs
// upstream's classes at reduced dimensions in float32 and separates an ALGEBRA
// defect from rounding; this one drives the SHIPPED bf16 weights on the SHIPPED
// inputs and proves the algebra survives contact with them.
//
// ─── WHAT IT GATES ──────────────────────────────────────────────────────────
//
// Both references are committed under tests/parity/goldens/minimax_music3_oracle/,
// captured by tools/oracle/music3_oracle.py from the pinned diffusers PR head
// c6da9936 on CPU. Nothing here regenerates them.
//
//   condition mix    frame_hiddens.npy [25, 32768]  ->  condition_chunk0.npy
//                    [86, 2048]. 176128 values.
//   depth decoder    frame_hiddens[:, :4096] (the language model's own hidden)
//                    + rvq_codes.npy rows 1.. (the frame's sampled codes) + the
//                    language model's embedding row for each semantic code
//                    ->  frame_hiddens[:, 4096:]. 716800 values.
//
// THE ROW ALIGNMENT IS THE TRAP. `rvq_codes` is [26, 8] and `frame_hiddens` is
// [25, ...]: row 0 of the codes is the PRIMING decode step, which emits no
// frame (encoders.py:342 `if frame_index > 0`). `rows[1:]` align with the 25
// frames. Getting it wrong compares two shifted sequences that are individually
// plausible, and the manifest's own note says so.
//
// ─── WHAT IT DOES NOT GATE, AND WHY ─────────────────────────────────────────
//
// Not the codes. Upstream's AR stage has no greedy path — `_sample_top_k`
// (encoders.py:94-103) ends in `torch.multinomial` against a seeded generator —
// and both stages sample from a CFG mix whose UNCONDITIONAL row is not in the
// golden set (only `[:1]` is stored, encoders.py:132,343). So `rvq_codes.npy` is
// an INPUT here. See the header of minimax_music3_ar.h; the spec's §5 claim of a
// token-exact greedy gate is not available from these artifacts.
//
// Not the language model's own forward. Producing `frame_hiddens[:, :4096]`
// means running the 8.6B `Qwen3ForCausalLM` teacher-forced on these codes, which
// is W2's remaining piece and is recorded as owed rather than skipped quietly.
//
// ─── THE TOLERANCE ──────────────────────────────────────────────────────────
//
// The goldens are bf16 values stored as float32 (asserted below, so the claim is
// checked and not assumed). This port computes in double and rounds once, torch
// computes in float32 and rounds to bf16 at every module boundary, so the two
// cannot be bit-equal and a tolerance is unavoidable. It is stated in bf16 ULPs:
//
//   kUlpTol = 1 bf16 ULP, i.e. |a - b| <= 2^-8 * max(|a|, |b|)
//
// One ULP is the TIGHTEST bound two bf16 implementations can hold, and it is
// calibrated rather than guessed: running upstream's own module over the whole
// depth sequence at once instead of incrementally moves 2752 of these 716800
// values, every one of them by exactly one ULP (max 0.015625 at |x| ~ 4). So one
// ULP is torch's own reproducibility on this tensor.
//
// A max-ULP bound alone is not enough — it is satisfied by an implementation
// that is off by one ULP EVERYWHERE, which no correct one is. So the mean
// absolute error is bounded too (torch-vs-torch measured 9.36e-06 on the same
// tensor), and both the compared count and the exactly-equal count are reported.
// A Pearson coefficient would see none of this: it is scale-invariant, so a
// uniformly scaled tensor passes it (AGENTS.md, spec §5).
//
// ─── HOW IT SKIPS ───────────────────────────────────────────────────────────
//
// Checkpoint-gated on the SAME variable W1's loader gate uses,
// VLLM_CPP_MUSIC3_CHECKPOINT (or CHECKPOINT_ROOT, whose `minimax-music3`
// subdirectory is used). Absent, every case emits a loud SKIP
// and returns, so this file compiles, links and runs in CI without the 28.5 GB
// asset.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "npy.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_music3_ar.h"
#include "vllm/model_executor/models/minimax_music3_loader.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;
namespace m3 = vllm::models::music3;

namespace {

// bf16 has SEVEN stored mantissa bits, so inside the binade [2^e, 2^(e+1))
// consecutive values are 2^(e-7) apart. The spacing is computed from the
// exponent rather than approximated by a fixed relative fraction — 2^-8 is HALF
// a bf16 ULP at the bottom of a binade, and using it as "one ULP" flags values
// that differ by exactly one (measured: 403050 of 716800 versus 217644).
//
// ─── THE CONDITION MIX: a bound with zero violations ────────────────────────
//
// Its only long reduction is the k=3 Conv1d over 3*4096 terms. Torch accumulates
// that in float32 and this port in double, so the two can differ by a rounding
// boundary — one ULP — and by more only where the output is a near-cancellation
// of large terms, which is what the absolute floor covers. One ULP at unit
// magnitude is 2^-7.
constexpr double kCondUlpAllow = 1.0;
constexpr double kCondAbsFloor = 1.0 / 128.0;  // 2^-7

// ─── THE DEPTH DECODER: a bound calibrated against a MATCHED CONTROL ─────────
//
// Zero violations is NOT achievable here and the reason is measured, not
// assumed. torch's own attention kernel is where the divergence starts: with the
// input layernorm bit-identical, `dispatch_attention_fn` on identical bf16 q/k/v
// reproduces to only 25736 of 32768 values, because the CPU kernel runs a
// blocked online softmax whose rounding no closed-form model here matched (four
// candidate models — pre-scaled q, bf16-rounded scores, bf16-rounded
// probabilities, and their combinations — were all WORSE than the plain form).
// Its bf16 Linear, by contrast, reproduces 32759 of 32768.
//
// So the calibration is torch against ITSELF with a different, equally correct
// attention kernel. Running upstream's OWN module over these exact inputs under
// `sdpa_kernel(SDPBackend.MATH)` and comparing to the committed goldens (which
// were captured on the default backend) gives:
//
//   CONTROL   46.34% bit-identical, mean|d| 1.659e-03, max|d| 0.125,
//             82 of 716800 outside 2 ULP-or-2^-6
//
// That is the floor of what any bf16 implementation can claim on this tensor,
// and chasing a specific kernel's rounding below it is not "more correct"
// (AGENTS.md's near-tie discipline).
//
// The bounds below sit just outside the control, and they are chosen so that the
// two defects this gate has actually caught are still RED:
//
//   an fp32 forward (no bf16 rounding at op boundaries)   0.0004% identical,
//                                                          mean|d| 2.649e-03
//   any algebra defect                                     ~0% identical
//
// The bit-identical FRACTION is the discriminator with four orders of margin,
// and it is asserted ALONGSIDE absolute bounds rather than instead of them — a
// count alone bounds nothing, and a correlation coefficient would see none of
// this because it is scale-invariant (spec §5).
constexpr double kDepthUlpAllow = 2.0;
constexpr double kDepthAbsFloor = 1.0 / 64.0;  // 2^-6
constexpr double kDepthIdenticalFloor = 0.40;   // control 0.4634
constexpr double kDepthMeanAbsTol = 2.2e-3;     // control 1.659e-3
constexpr double kDepthMaxAbsTol = 0.15;        // control 0.125
constexpr double kDepthOutsideFraction = 1e-3;  // control 82/716800 = 1.14e-4

// The oracle capture's own shape facts (manifest.json), asserted rather than
// assumed so a regenerated golden cannot silently change what is compared.
constexpr int64_t kFrames = 25;
constexpr int64_t kCodeRows = 26;
constexpr int64_t kCodebooks = 8;
constexpr int64_t kHidden = 4096;
constexpr int64_t kLatentLength = 86;
constexpr int64_t kOutDim = 2048;
// The shard the language model's embedding table lives in
// (language_model/model.safetensors.index.json).
constexpr const char* kEmbedShard = "model-00001-of-00004.safetensors";

std::string CheckpointRoot() {
  if (const char* direct = std::getenv("VLLM_CPP_MUSIC3_CHECKPOINT")) return direct;
  if (const char* root = std::getenv("CHECKPOINT_ROOT")) {
    return (fs::path(root) / "minimax-music3").string();
  }
  return {};
}

std::string GoldensDir() {
  return std::string(MUSIC3_GOLDENS_DIR);
}

// Returns "" when everything needed is present, otherwise the reason to SKIP.
std::string MissingReason() {
  const std::string root = CheckpointRoot();
  if (root.empty()) {
    return "VLLM_CPP_MUSIC3_CHECKPOINT / CHECKPOINT_ROOT is unset";
  }
  std::error_code ec;
  if (!fs::is_directory(root, ec)) return "checkpoint directory " + root + " is absent";
  for (const char* needed : {"condition_encoder/diffusion_pytorch_model.safetensors",
                             "rvq_depth_decoder/diffusion_pytorch_model.safetensors"}) {
    if (!fs::is_regular_file(fs::path(root) / needed, ec)) {
      return std::string("checkpoint is missing ") + needed;
    }
  }
  if (!fs::is_regular_file(fs::path(root) / "language_model" / kEmbedShard, ec)) {
    return std::string("checkpoint is missing language_model/") + kEmbedShard;
  }
  for (const char* golden : {"frame_hiddens.npy", "condition_chunk0.npy", "rvq_codes.npy"}) {
    if (!fs::is_regular_file(fs::path(GoldensDir()) / golden, ec)) {
      return std::string("golden ") + golden + " is absent under " + GoldensDir();
    }
  }
  return {};
}

bool SkipIfMissing(const char* what) {
  const std::string reason = MissingReason();
  if (reason.empty()) return false;
  std::printf("[SKIP] %s: %s\n", what, reason.c_str());
  MESSAGE("SKIPPED (" << reason << ")");
  return true;
}

// Row-major float32 from a golden. `condition_chunk0.npy` is stored FORTRAN
// order (the oracle saved a transposed view); read as C-order it would have the
// right value count and be the wrong tensor, so the transpose is explicit and
// keys off the reader's flag rather than a guess about the file.
std::vector<float> LoadF32Npy(const std::string& name, std::vector<int64_t>* shape) {
  const parity::NpyArray array =
      parity::LoadNpy((fs::path(GoldensDir()) / name).string(), /*allow_fortran_order=*/true);
  REQUIRE_MESSAGE(array.dtype == "<f4", "golden " << name << " must be float32, is " << array.dtype);
  *shape = array.shape;
  const size_t count = array.data.size() / sizeof(float);
  std::vector<float> raw(count);
  std::memcpy(raw.data(), array.data.data(), array.data.size());
  if (!array.fortran_order) return raw;
  REQUIRE_MESSAGE(array.shape.size() == 2, "only a 2-D fortran-order golden is handled: " << name);
  const int64_t rows = array.shape[0];
  const int64_t cols = array.shape[1];
  std::vector<float> out(count);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      out[static_cast<size_t>(r * cols + c)] = raw[static_cast<size_t>(c * rows + r)];
    }
  }
  return out;
}

std::vector<int32_t> LoadI32Npy(const std::string& name, std::vector<int64_t>* shape) {
  const parity::NpyArray array = parity::LoadNpy((fs::path(GoldensDir()) / name).string());
  REQUIRE_MESSAGE(!array.fortran_order, "golden " << name << " must be C-order");
  REQUIRE_MESSAGE(array.dtype == "<i4", "golden " << name << " must be int32, is " << array.dtype);
  *shape = array.shape;
  const size_t count = array.data.size() / sizeof(int32_t);
  std::vector<int32_t> out(count);
  std::memcpy(out.data(), array.data.data(), array.data.size());
  return out;
}

// One checkpoint tensor at its RUNTIME dtype, carried in a float32 buffer.
//
// The AR half runs bf16 (MiniMaxMusic3ResolveRuntimeDtypes(kBf16ArFp32Acoustic)),
// so an F32 file — the condition encoder's, per spec section 2.1 — is ROUNDED,
// not widened. Skipping that round is the exact mistake the header's dtype note
// describes: still numerically fine, still not what the checkpoint runs.
std::vector<float> AtRuntimeDtype(const vllm::StTensor& tensor) {
  const size_t count = tensor.nbytes / (tensor.dtype == "F32" ? 4 : 2);
  std::vector<float> out(count);
  if (tensor.dtype == "F32") {
    std::memcpy(out.data(), tensor.data, tensor.nbytes);
    for (float& value : out) value = vt::BF16ToF32(vt::F32ToBF16(value));
  } else if (tensor.dtype == "BF16") {
    const auto* raw = reinterpret_cast<const uint16_t*>(tensor.data);
    for (size_t i = 0; i < count; ++i) out[i] = vt::BF16ToF32(raw[i]);
  } else {
    FAIL("unexpected checkpoint dtype " << tensor.dtype);
  }
  return out;
}

// What a full-scale comparison examined, so it can REPORT rather than log.
struct UlpReport {
  int64_t compared = 0;
  int64_t identical = 0;
  int64_t outside = 0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double max_ulps = 0.0;
  double ref_absmax = 0.0;
  int64_t first_bad = -1;

  double identical_fraction() const {
    return compared > 0 ? static_cast<double>(identical) / static_cast<double>(compared) : 0.0;
  }
  double outside_fraction() const {
    return compared > 0 ? static_cast<double>(outside) / static_cast<double>(compared) : 0.0;
  }
};

UlpReport CompareUlps(const std::vector<float>& got, const std::vector<float>& want,
                      double ulp_allow, double abs_floor) {
  UlpReport report;
  REQUIRE(got.size() == want.size());
  double sum = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double a = got[i];
    const double b = want[i];
    const double diff = std::abs(a - b);
    ++report.compared;
    if (a == b) ++report.identical;
    sum += diff;
    report.max_abs = std::max(report.max_abs, diff);
    report.ref_absmax = std::max(report.ref_absmax, std::abs(b));
    const double magnitude = std::max(std::abs(a), std::abs(b));
    double ulps = 0.0;
    if (diff > 0.0) {
      if (magnitude > 0.0) {
        int exponent = 0;
        std::frexp(magnitude, &exponent);  // magnitude = m * 2^exponent, 0.5 <= m < 1
        ulps = diff / std::ldexp(1.0, exponent - 8);
      } else {
        ulps = 1e9;
      }
    }
    report.max_ulps = std::max(report.max_ulps, ulps);
    if (ulps > ulp_allow && diff > abs_floor) {
      if (report.outside == 0) report.first_bad = static_cast<int64_t>(i);
      ++report.outside;
    }
  }
  report.mean_abs = report.compared > 0 ? sum / static_cast<double>(report.compared) : 0.0;
  return report;
}

}  // namespace

TEST_CASE("music3 ar real: the goldens are the ones this gate was written against") {
  if (SkipIfMissing("music3 ar real goldens")) return;
  std::vector<int64_t> shape;
  const std::vector<float> frame_hiddens = LoadF32Npy("frame_hiddens.npy", &shape);
  REQUIRE(shape.size() == 2);
  CHECK(shape[0] == kFrames);
  CHECK(shape[1] == kCodebooks * kHidden);
  const std::vector<float> condition = LoadF32Npy("condition_chunk0.npy", &shape);
  REQUIRE(shape.size() == 2);
  CHECK(shape[0] == kLatentLength);
  CHECK(shape[1] == kOutDim);
  std::vector<int64_t> code_shape;
  const std::vector<int32_t> codes = LoadI32Npy("rvq_codes.npy", &code_shape);
  REQUIRE(code_shape.size() == 2);
  CHECK(code_shape[0] == kCodeRows);
  CHECK(code_shape[1] == kCodebooks);
  // The alignment this whole file depends on: one MORE code row than frame.
  CHECK(code_shape[0] == kFrames + 1);

  // `frame_hiddens` is stored float32 but every value must be bf16-exact — that
  // is what makes a bf16-ULP tolerance the right shape of bound. CHECKED, not
  // assumed.
  int64_t not_bf16 = 0;
  for (const float value : frame_hiddens) {
    if (vt::BF16ToF32(vt::F32ToBF16(value)) != value) ++not_bf16;
  }
  MESSAGE("frame_hiddens: " << frame_hiddens.size() << " values, " << not_bf16
                            << " not representable in bf16");
  CHECK(not_bf16 == 0);

  // Every residual code is inside the 1024-entry audio vocabulary, and the
  // semantic column is inside the language model's 16384-entry code window.
  int64_t residual_checked = 0;
  int64_t semantic_checked = 0;
  for (int64_t row = 0; row < kCodeRows; ++row) {
    const int32_t semantic = codes[static_cast<size_t>(row * kCodebooks)];
    CHECK(semantic >= 0);
    CHECK(semantic < m3::kSemanticVocabSize);
    ++semantic_checked;
    for (int64_t j = 1; j < kCodebooks; ++j) {
      const int32_t code = codes[static_cast<size_t>(row * kCodebooks + j)];
      CHECK(code >= 0);
      CHECK(code < 1024);
      ++residual_checked;
    }
  }
  MESSAGE("codes checked: " << semantic_checked << " semantic, " << residual_checked
                            << " residual, over " << kCodeRows << " rows");
  MESSAGE("condition golden: " << condition.size() << " values");
}

TEST_CASE("music3 ar real: the condition mix reproduces condition_chunk0") {
  if (SkipIfMissing("music3 condition mix (real weights)")) return;
  const std::string root = CheckpointRoot();
  std::vector<int64_t> shape;
  const std::vector<float> frame_hiddens = LoadF32Npy("frame_hiddens.npy", &shape);
  REQUIRE(shape[0] == kFrames);
  const std::vector<float> want = LoadF32Npy("condition_chunk0.npy", &shape);

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(
      (fs::path(root) / "condition_encoder" / "diffusion_pytorch_model.safetensors").string());
  m3::ConditionMixConfig config;  // the released config's values are the defaults
  m3::ConditionMixWeights weights;
  weights.layer_weight_logits = AtRuntimeDtype(file.Get("layer_weight_logits"));
  weights.layer_scale = AtRuntimeDtype(file.Get("layer_scale"));
  weights.proj_weight = AtRuntimeDtype(file.Get("proj.weight"));
  weights.proj_bias = AtRuntimeDtype(file.Get("proj.bias"));
  REQUIRE(static_cast<int64_t>(weights.layer_weight_logits.size()) == config.num_condition_layers);
  REQUIRE(static_cast<int64_t>(weights.proj_weight.size()) ==
          config.out_dim * config.condition_hidden_dim * 3);

  CHECK(m3::ConditionLatentLength(kFrames, config) == kLatentLength);
  const std::vector<float> got =
      m3::ConditionMix(frame_hiddens, kFrames, config, weights, m3::ArCompute::kBFloat16);
  const UlpReport report = CompareUlps(got, want, kCondUlpAllow, kCondAbsFloor);
  MESSAGE("condition mix vs condition_chunk0: " << report.compared << " values compared, "
                                                << report.identical << " bit-identical, "
                                                << report.outside << " beyond 1 bf16 ULP, max|d| "
                                                << report.max_abs << " (" << report.max_ulps
                                                << " ULP), mean|d| " << report.mean_abs
                                                << ", |ref|max " << report.ref_absmax);
  CHECK(report.compared == kLatentLength * kOutDim);
  CHECK(report.outside == 0);
  CHECK(report.mean_abs <= 1e-5);
  // A uniformly scaled or constant reproduction would satisfy neither.
  CHECK(report.identical_fraction() >= 0.99);
  CHECK(report.ref_absmax > 1.0);
}

TEST_CASE("music3 ar real: the depth decoder reproduces the per-frame hidden states") {
  if (SkipIfMissing("music3 depth decoder (real weights)")) return;
  const std::string root = CheckpointRoot();
  std::vector<int64_t> shape;
  const std::vector<float> frame_hiddens = LoadF32Npy("frame_hiddens.npy", &shape);
  REQUIRE(shape[0] == kFrames);
  REQUIRE(shape[1] == kCodebooks * kHidden);
  std::vector<int64_t> code_shape;
  const std::vector<int32_t> codes = LoadI32Npy("rvq_codes.npy", &code_shape);
  REQUIRE(code_shape[0] == kFrames + 1);

  const vllm::SafetensorsFile depth_file = vllm::SafetensorsFile::Open(
      (fs::path(root) / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors").string());
  m3::DepthDecoderConfig config;  // the released config's values are the defaults
  m3::DepthDecoderWeights weights;
  weights.audio_embeddings = AtRuntimeDtype(depth_file.Get("audio_embeddings.weight"));
  weights.projection = AtRuntimeDtype(depth_file.Get("projection.weight"));
  weights.pos_embedding = AtRuntimeDtype(depth_file.Get("pos_embedding.weight"));
  weights.norm = AtRuntimeDtype(depth_file.Get("norm.weight"));
  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    const std::string base = "layers." + std::to_string(layer) + ".";
    m3::DepthDecoderLayerWeights entry;
    entry.input_layernorm = AtRuntimeDtype(depth_file.Get(base + "input_layernorm.weight"));
    entry.post_attention_layernorm =
        AtRuntimeDtype(depth_file.Get(base + "post_attention_layernorm.weight"));
    entry.to_q = AtRuntimeDtype(depth_file.Get(base + "attn.to_q.weight"));
    entry.to_k = AtRuntimeDtype(depth_file.Get(base + "attn.to_k.weight"));
    entry.to_v = AtRuntimeDtype(depth_file.Get(base + "attn.to_v.weight"));
    entry.to_out = AtRuntimeDtype(depth_file.Get(base + "attn.to_out.weight"));
    entry.gate_proj = AtRuntimeDtype(depth_file.Get(base + "gate_proj.weight"));
    entry.up_proj = AtRuntimeDtype(depth_file.Get(base + "up_proj.weight"));
    entry.down_proj = AtRuntimeDtype(depth_file.Get(base + "down_proj.weight"));
    weights.layers.push_back(std::move(entry));
  }
  for (int64_t head = 0; head < config.residual_codebooks(); ++head) {
    weights.audio_heads.push_back(
        AtRuntimeDtype(depth_file.Get("audio_heads." + std::to_string(head) + ".weight")));
  }
  REQUIRE(static_cast<int64_t>(weights.audio_embeddings.size()) ==
          config.audio_vocab_size * config.residual_codebooks() * config.hidden_size);

  // The SEMANTIC codebook is embedded by the LANGUAGE MODEL, so its rows come
  // from the language model's own table — 25 rows out of 200000, read straight
  // from the shard's mmap rather than materializing 1.6 GB.
  const vllm::SafetensorsFile lm_file = vllm::SafetensorsFile::Open(
      (fs::path(root) / "language_model" / kEmbedShard).string());
  const vllm::StTensor& embed = lm_file.Get("model.embed_tokens.weight");
  REQUIRE(embed.dtype == "BF16");
  REQUIRE(embed.shape.size() == 2);
  REQUIRE(embed.shape[0] == 200000);
  REQUIRE(embed.shape[1] == config.hidden_size);
  const auto* embed_raw = reinterpret_cast<const uint16_t*>(embed.data);

  int64_t frames_run = 0;
  std::vector<float> got;
  std::vector<float> want;
  got.reserve(static_cast<size_t>(kFrames * config.residual_codebooks() * config.hidden_size));
  want.reserve(got.capacity());
  for (int64_t frame = 0; frame < kFrames; ++frame) {
    // rows[1:] align with the frames: row 0 is the priming decode.
    const int64_t code_row = frame + 1;
    const int32_t semantic = codes[static_cast<size_t>(code_row * kCodebooks)];
    const int64_t token = static_cast<int64_t>(semantic) + m3::kAudioCodeOffset;
    REQUIRE(token < embed.shape[0]);
    std::vector<float> semantic_embed(static_cast<size_t>(config.hidden_size));
    for (int64_t c = 0; c < config.hidden_size; ++c) {
      semantic_embed[static_cast<size_t>(c)] =
          vt::BF16ToF32(embed_raw[token * config.hidden_size + c]);
    }
    // c1..c6 are fed back; c7 is only ever predicted (encoders.py:139).
    std::vector<int32_t> residual;
    for (int64_t j = 1; j + 1 < kCodebooks; ++j) {
      residual.push_back(codes[static_cast<size_t>(code_row * kCodebooks + j)]);
    }
    REQUIRE(static_cast<int64_t>(residual.size()) == config.residual_codebooks() - 1);

    const std::vector<float> last_hidden(
        frame_hiddens.begin() + static_cast<int64_t>(frame) * kCodebooks * kHidden,
        frame_hiddens.begin() + static_cast<int64_t>(frame) * kCodebooks * kHidden + kHidden);
    const std::vector<float> embeds = m3::DepthSequenceEmbeds(
        last_hidden, semantic_embed, residual, config, weights, m3::ArCompute::kBFloat16);
    const std::vector<float> hidden_states = m3::DepthDecoderForward(
        embeds, config.num_codebooks, config, weights, m3::ArCompute::kBFloat16);
    REQUIRE(static_cast<int64_t>(hidden_states.size()) ==
            config.num_codebooks * config.hidden_size);
    // Depth step i is position i of the sequence, so 1..7 are what the golden
    // row carries after the language model's own 4096.
    got.insert(got.end(), hidden_states.begin() + kHidden, hidden_states.end());
    want.insert(want.end(),
                frame_hiddens.begin() + static_cast<int64_t>(frame) * kCodebooks * kHidden + kHidden,
                frame_hiddens.begin() + static_cast<int64_t>(frame + 1) * kCodebooks * kHidden);
    ++frames_run;
  }
  CHECK(frames_run == kFrames);

  const UlpReport report = CompareUlps(got, want, kDepthUlpAllow, kDepthAbsFloor);
  MESSAGE("depth decoder vs frame_hiddens[:, 4096:]: "
          << report.compared << " values compared over " << frames_run << " frames x "
          << config.residual_codebooks() << " depth steps, " << report.identical
          << " bit-identical, " << report.outside << " beyond 1 bf16 ULP, max|d| "
          << report.max_abs << " (" << report.max_ulps << " ULP), mean|d| " << report.mean_abs
          << ", |ref|max " << report.ref_absmax << ", identical fraction "
          << report.identical_fraction()
          << (report.first_bad >= 0 ? ", first beyond at " + std::to_string(report.first_bad)
                                    : std::string()));
  CHECK(report.compared == kFrames * config.residual_codebooks() * config.hidden_size);
  // Four bounds, none of which alone is a gate. See the calibration note above:
  // the bit-identical fraction separates a correct bf16 mirror from an fp32 one
  // by four orders of magnitude, and the absolute bounds are what stop a
  // uniformly scaled tensor from passing on counts.
  CHECK(report.identical_fraction() >= kDepthIdenticalFloor);
  CHECK(report.mean_abs <= kDepthMeanAbsTol);
  CHECK(report.max_abs <= kDepthMaxAbsTol);
  CHECK(report.outside_fraction() <= kDepthOutsideFraction);
  // A zeroed or constant reproduction would satisfy a bound on nothing; the
  // reference has real dynamic range and so must the reproduction.
  CHECK(report.ref_absmax > 1.0);
}

TEST_CASE("music3 ar real: the golden codes rank highly under the reproduced heads") {
  // NOT a token gate, and labelled so. The sampled code came from a CFG mix of
  // the conditional and unconditional rows (encoders.py:134-137) and only the
  // conditional row is in the goldens, so the code cannot be re-derived. What
  // CAN be said is that our reproduced hidden states put the golden code near
  // the top of the CONDITIONAL distribution — a wrong forward would not.
  //
  // The reported rank distribution is the evidence; the assertion is only that
  // every code is inside the head's vocabulary and that the mean rank is far
  // below chance (512 for a 1024-entry codebook).
  if (SkipIfMissing("music3 depth code ranks (real weights)")) return;
  const std::string root = CheckpointRoot();
  std::vector<int64_t> shape;
  const std::vector<float> frame_hiddens = LoadF32Npy("frame_hiddens.npy", &shape);
  std::vector<int64_t> code_shape;
  const std::vector<int32_t> codes = LoadI32Npy("rvq_codes.npy", &code_shape);

  const vllm::SafetensorsFile depth_file = vllm::SafetensorsFile::Open(
      (fs::path(root) / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors").string());
  m3::DepthDecoderConfig config;
  m3::DepthDecoderWeights weights;
  weights.audio_embeddings = AtRuntimeDtype(depth_file.Get("audio_embeddings.weight"));
  weights.projection = AtRuntimeDtype(depth_file.Get("projection.weight"));
  weights.pos_embedding = AtRuntimeDtype(depth_file.Get("pos_embedding.weight"));
  weights.norm = AtRuntimeDtype(depth_file.Get("norm.weight"));
  for (int64_t head = 0; head < config.residual_codebooks(); ++head) {
    weights.audio_heads.push_back(
        AtRuntimeDtype(depth_file.Get("audio_heads." + std::to_string(head) + ".weight")));
  }

  int64_t ranked = 0;
  int64_t rank_sum = 0;
  int64_t worst_rank = 0;
  for (int64_t frame = 0; frame < kFrames; ++frame) {
    for (int64_t step = 1; step < kCodebooks; ++step) {
      const std::vector<float> state(
          frame_hiddens.begin() + static_cast<int64_t>(frame) * kCodebooks * kHidden + step * kHidden,
          frame_hiddens.begin() + static_cast<int64_t>(frame) * kCodebooks * kHidden +
              (step + 1) * kHidden);
      const std::vector<float> logits =
          m3::AudioHeadLogits(state, step - 1, config, weights, m3::ArCompute::kBFloat16);
      REQUIRE(static_cast<int64_t>(logits.size()) == config.audio_vocab_size);
      const int32_t code = codes[static_cast<size_t>((frame + 1) * kCodebooks + step)];
      REQUIRE(code < config.audio_vocab_size);
      int64_t rank = 0;
      for (int64_t j = 0; j < config.audio_vocab_size; ++j) {
        if (logits[static_cast<size_t>(j)] > logits[static_cast<size_t>(code)]) ++rank;
      }
      rank_sum += rank;
      worst_rank = std::max(worst_rank, rank);
      ++ranked;
    }
  }
  const double mean_rank = static_cast<double>(rank_sum) / static_cast<double>(ranked);
  MESSAGE("golden code ranks under the CONDITIONAL head logits: " << ranked << " codes, mean rank "
                                                                  << mean_rank << ", worst rank "
                                                                  << worst_rank
                                                                  << ", chance would be 511.5");
  CHECK(ranked == kFrames * (kCodebooks - 1));
  CHECK(mean_rank < 50.0);
}
