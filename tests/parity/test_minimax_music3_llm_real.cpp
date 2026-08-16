// MiniMax-Music3 — the 8.6B LANGUAGE MODEL against the REAL checkpoint (#672).
//
// The piece spec §5 recorded as "still owed on the LLM half": reproducing
// `frame_hiddens[:, :4096]`, which is the language model's own hidden state at
// every frame. Its two companions gate the rest of the autoregressive half —
// tests/vllm/models/test_minimax_music3_ar.cpp at reduced dimensions and
// tests/parity/test_minimax_music3_ar_real.cpp at full scale — and neither could
// touch this one, because until `Qwen3DenseModel::ForwardEmbeds` existed there
// was no way to drive the model on a continuous frame-feedback embedding.
//
// ─── WHY THIS IS TEACHER-FORCED, AND WHAT THAT BUYS ─────────────────────────
//
// Spec §5 withdrew the token gate: upstream's AR stage ends every draw in
// `torch.multinomial` against a seeded `torch.Generator` (encoders.py:94-103),
// so `rvq_codes.npy` is a seeded SAMPLE and no port reproduces it. The codes are
// therefore consumed as INPUTS here, exactly as the W2/W3 full-scale gate
// consumes them: the loop is driven onto the oracle's own trajectory by feeding
// its codes back, and what is COMPARED is the model's hidden state at each step.
//
// That is a strong gate rather than a weak one. The comparison is 25 steps deep
// through 36 decoder layers with a 60-token KV history, so every layer, the
// paged cache, RoPE at theta 1e6, the per-head q/k norms, the untied 200000-row
// lm_head and the frame-feedback embedding all have to be right for step 25 to
// land anywhere near the golden — an error at step 1 is carried by 24 further
// steps of attention over the cache it poisoned.
//
// ─── WHAT EACH CASE GATES ───────────────────────────────────────────────────
//
//   prompt      the assembled prompt tokenizes to the ids the oracle's own
//               `Qwen2Tokenizer` produced (61, verified against the HF
//               tokenizer on the identical string), and the unconditional row
//               is its `[1:-2]` rewrite.
//   hidden      25 teacher-forced steps vs `frame_hiddens[:, :4096]` —
//               102400 values, bounded in bf16 ULPs against a MEASURED control.
//   rank        the oracle's OWN sampled semantic code, ranked under our
//               reproduced guided logits. Scale-free and distributional: it
//               cannot be passed by a uniformly scaled hidden state, and chance
//               over the 16384-entry semantic window would be ~8192.
//   condition   the reproduced frame hiddens, completed with the golden depth
//               rows, through the condition mix to `condition_chunk0.npy` —
//               the stage that actually consumes them.
//
// A CORRELATION COEFFICIENT IS NOT A GATE (AGENTS.md; spec §5): Pearson is
// scale-invariant, so bounds here are on absolute error, on bf16 ULPs, and on a
// bit-identical FRACTION, and every comparison reports how many values it
// examined.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "npy.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_music3_ar.h"
#include "vllm/model_executor/models/minimax_music3_llm.h"
#include "vllm/model_executor/models/minimax_music3_loader.h"
#include "vllm/model_executor/models/minimax_music3_speech.h"  // kMusic3SpeechFamily
#include "vllm/multimodal/speech_engine.h"                     // SpeechEngineDeviceType
#include "vt/backend.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;
namespace m3 = vllm::models::music3;

namespace {

// ─── The capture's own request (manifest.json "request"), quoted ────────────
//
// Byte for byte what `tools/oracle/music3_oracle.py` was given. The assembled
// prompt is rebuilt with `AssembleArPrompt`, which
// tests/vllm/models/test_minimax_music3_ar.cpp already gates string-for-string
// against upstream, so this file states the INPUTS and not the result.
constexpr const char* kDescription =
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate. Vocals: soft female "
    "lead, close and breathy. Arrangement: fingerpicked guitar and soft piano.";
constexpr const char* kLyrics = "[verse]\nMorning light filtering through the pine\n";

// MEASURED, not assumed: the same string through HF `transformers` 5.14.1's own
// `Qwen2Tokenizer` on this checkpoint's `tokenizer/` gives exactly these 61 ids,
// and `examples/tokenize` over the same text gives them too. Asserted here so a
// tokenizer regression is named at the prompt rather than as a hidden-state
// mismatch 25 steps later.
constexpr int64_t kPromptTokens = 61;

constexpr int64_t kFrames = 25;
constexpr int64_t kCodeRows = 26;
constexpr int64_t kCodebooks = 8;
constexpr int64_t kHidden = 4096;
constexpr int64_t kLatentLength = 86;
constexpr int64_t kOutDim = 2048;

// ─── The bounds, CALIBRATED AGAINST A MATCHED CONTROL ───────────────────────
//
// Bit-exactness is not on offer here and the reason is measured, not assumed.
// Upstream computes bf16 and rounds at every module boundary through 36 decoder
// layers; this port runs the same dtype through a different attention kernel and
// a different GEMM. So the floor is TORCH AGAINST ITSELF: upstream's own
// `Qwen3ForCausalLM`, the same weights, the same teacher-forced codes, under
// `sdpa_kernel(SDPBackend.MATH)` — the goldens were captured on the default
// backend — compared to the committed `frame_hiddens[:, :4096]`:
//
//   CONTROL   102400 values, 12036 bit-identical (11.75%), mean|d| 1.475e-02,
//             max|d| 5.000e-01, max 504 ULP, 29968 (29.27%) beyond 2 ULP-or-2^-6,
//             |ref|max 81, mean|ref| 0.9489
//
// THIS ARM, measured on this box 2026-08-14 at aa3643b6 + this change:
//
//   OURS      102400 values, 9337 bit-identical (9.12%), mean|d| 1.763e-02,
//             max|d| 1.000, max 509 ULP, 37572 (36.69%) beyond 2 ULP-or-2^-6
//
// Ours is 1.20x the control's mean absolute error and 1.25x its outside
// fraction: a NEAR TIE, and chasing one kernel's rounding below the control is
// not "more correct" (AGENTS.md). The bounds below sit just outside OURS, with
// under a factor of two of headroom on each, so none can be satisfied by an
// implementation materially worse than the one measured.
//
// AND THE BOUNDS ARE PROVED TO DISCRIMINATE, IN THE GATE ITSELF. A bf16 model
// this deep is noisy enough that a loose bound would be indistinguishable from
// no bound, so the case ALSO applies every bound to a ONE-STEP-SHIFTED
// alignment — our step k against the golden's step k+1 — which is a wrong answer
// of exactly the shape this gate exists to catch, and asserts that the shifted
// comparison FAILS. That is a mutation the gate performs on itself rather than
// a claim about mutations someone else ran.
constexpr double kHiddenUlpAllow = 2.0;
constexpr double kHiddenAbsFloor = 1.0 / 64.0;  // 2^-6
constexpr double kHiddenIdenticalFloor = 0.06;    // measured 0.0912, control 0.1175
constexpr double kHiddenMeanAbsTol = 2.4e-2;      // measured 1.763e-02, control 1.475e-02
constexpr double kHiddenMaxAbsTol = 1.5;          // measured 1.000, control 0.500
constexpr double kHiddenOutsideFraction = 0.45;   // measured 0.3669, control 0.2927

std::string CheckpointRoot() {
  if (const char* direct = std::getenv("VLLM_CPP_MUSIC3_CHECKPOINT")) return direct;
  if (const char* root = std::getenv("CHECKPOINT_ROOT")) {
    return (fs::path(root) / "minimax-music3").string();
  }
  return {};
}

std::string GoldensDir() { return std::string(MUSIC3_GOLDENS_DIR); }

std::string MissingReason() {
  const std::string root = CheckpointRoot();
  if (root.empty()) return "VLLM_CPP_MUSIC3_CHECKPOINT / CHECKPOINT_ROOT is unset";
  std::error_code ec;
  if (!fs::is_directory(root, ec)) return "checkpoint directory " + root + " is absent";
  for (const char* golden : {"frame_hiddens.npy", "rvq_codes.npy", "condition_chunk0.npy"}) {
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

// The 17.2 GB language model is loaded ONCE for the whole file. doctest runs the
// cases in this translation unit sequentially in one process, so a per-case load
// would read 18.5 GB four times over a network mount.
struct Staged {
  vllm::MiniMaxMusic3Paths paths;
  vllm::MiniMaxMusic3Config config;
  m3::Music3ArWeights ar;
  bool loaded = false;
};

Staged& Weights() {
  static Staged staged;
  if (!staged.loaded) {
    staged.paths = vllm::MiniMaxMusic3ResolveCheckpoint(CheckpointRoot());
    staged.config = vllm::MiniMaxMusic3LoadConfig(staged.paths);
    staged.ar = m3::Music3LoadArWeights(staged.paths, staged.config);
    staged.loaded = true;
  }
  return staged;
}

std::vector<float> LoadF32Npy(const std::string& name, std::vector<int64_t>* shape) {
  const parity::NpyArray array =
      parity::LoadNpy((fs::path(GoldensDir()) / name).string(), /*allow_fortran_order=*/true);
  REQUIRE_MESSAGE(array.dtype == "<f4", "golden " << name << " must be float32, is "
                                                  << array.dtype);
  *shape = array.shape;
  const size_t count = array.data.size() / sizeof(float);
  std::vector<float> raw(count);
  std::memcpy(raw.data(), array.data.data(), array.data.size());
  if (!array.fortran_order) return raw;
  REQUIRE(array.shape.size() == 2);
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
  REQUIRE(array.dtype == "<i4");
  *shape = array.shape;
  std::vector<int32_t> out(array.data.size() / sizeof(int32_t));
  std::memcpy(out.data(), array.data.data(), array.data.size());
  return out;
}

// The same bf16-ULP report the W2/W3 full-scale gate uses, so the two files
// state their bounds in the same unit and can be compared by eye. bf16 has SEVEN
// stored mantissa bits, so the spacing inside [2^e, 2^(e+1)) is 2^(e-7) and is
// computed from the exponent rather than approximated by a fixed fraction.
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
        std::frexp(magnitude, &exponent);
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

// One teacher-forced run of the language model: prefill the two prompt rows,
// then feed `codes` back frame by frame. Returns the CONDITIONAL row's hidden
// state after each feedback step — h_1..h_25, which is exactly
// `frame_hiddens[k][:4096]` for k = 0..24 (encoders.py:342-353: the append at
// frame_index k stores the hidden the k-th feedback produced).
//
// `out_guided` receives, per step, the guided semantic logit row the loop would
// have sampled from BEFORE that step's feedback — so `out_guided[k]` is the
// distribution the oracle drew `codes[k]` from.
struct TeacherForced {
  std::vector<float> hidden;  // [steps, kHidden]
  std::vector<std::vector<float>> guided;
  int64_t steps = 0;
};

TeacherForced RunTeacherForced(const std::vector<int32_t>& codes, int64_t code_rows,
                               int64_t steps) {
  const m3::Music3ArWeights& ar = Weights().ar;
  const int64_t vocab = ar.lm_config.vocab_size;
  const std::string prompt = m3::AssembleArPrompt(kDescription, kLyrics);
  const std::vector<int32_t> ids = ar.Encode(prompt);
  REQUIRE(static_cast<int64_t>(ids.size()) == kPromptTokens);
  const std::vector<int32_t> unconditional = m3::UnconditionalPromptIds(ids);
  const std::vector<bool> blocked = m3::SemanticVocabMask(
      vocab, m3::kAudioCodeOffset, m3::kSemanticVocabSize, m3::kAudioEndTokenId);

  // WHERE this gate runs the 8.6B forward. Default 0 = CPU, so an unset
  // environment reproduces every number this file has ever printed, byte for
  // byte. `VLLM_CPP_MUSIC3_DEVICE=1` runs the SAME comparison on the device arm
  // (#672) — the only numeric gate that arm has, because a generated waveform
  // cannot be compared to anything (spec §5: the codes are a seeded draw).
  //
  // Resolved through the SHARED `multimodal::SpeechEngineDeviceType` the engine
  // itself calls, not a private copy: a gate that resolved the device its own
  // way could pass while the engine bound a different one.
  const char* device_env = std::getenv("VLLM_CPP_MUSIC3_DEVICE");
  const int32_t device_sel = (device_env != nullptr && device_env[0] == '1') ? 1 : 0;
  const vt::DeviceType device_type =
      vllm::multimodal::SpeechEngineDeviceType(device_sel, m3::kMusic3SpeechFamily);
  vt::Queue queue = device_type == vt::DeviceType::kCPU
                        ? vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}
                        : vt::GetBackend(device_type).CreateQueue();
  // REPORTED, not assumed: a gate that cannot say which arm it examined has not
  // reported, and the two arms print different numbers below.
  const std::string arm_line =
      std::string("music3 llm real: the 8.6B forward ran on '") +
      vt::DeviceTypeName(queue.device.type) + "' (VLLM_CPP_MUSIC3_DEVICE=" +
      (device_env == nullptr ? std::string("unset") : std::string(device_env)) + ")";
  MESSAGE(arm_line);
  m3::Music3LmSession session(ar, queue, kPromptTokens + steps + 1);

  std::vector<float> hidden;
  std::vector<float> logits;
  session.Prefill(ids, unconditional, &hidden, &logits);

  TeacherForced out;
  out.hidden.reserve(static_cast<size_t>(steps * kHidden));
  for (int64_t k = 0; k < steps; ++k) {
    REQUIRE(k < code_rows);
    // The distribution the oracle's own draw came from, at this step.
    REQUIRE(static_cast<int64_t>(logits.size()) == 2 * vocab);
    std::vector<float> rounded = logits;
    for (float& value : rounded) value = vt::BF16ToF32(vt::F32ToBF16(value));
    const std::vector<float> conditional(rounded.begin(),
                                         rounded.begin() + static_cast<ptrdiff_t>(vocab));
    const std::vector<float> unconditional_row(
        rounded.begin() + static_cast<ptrdiff_t>(vocab), rounded.end());
    out.guided.push_back(m3::GuidedSemanticLogits(conditional, unconditional_row, blocked,
                                                  m3::kArCfgTopK, m3::kArCfgScale));

    const int32_t semantic = codes[static_cast<size_t>(k * kCodebooks)];
    const std::vector<int32_t> residual(
        codes.begin() + static_cast<ptrdiff_t>(k * kCodebooks + 1),
        codes.begin() + static_cast<ptrdiff_t>((k + 1) * kCodebooks));
    REQUIRE(static_cast<int64_t>(residual.size()) == kCodebooks - 1);
    const std::vector<float> feedback = m3::EmbedAudioFrame(
        ar.EmbedRow(static_cast<int64_t>(semantic) + m3::kAudioCodeOffset), residual,
        Weights().ar.depth_config, ar.depth, m3::ArCompute::kBFloat16);
    std::vector<float> both(feedback);
    both.insert(both.end(), feedback.begin(), feedback.end());
    session.Step(both, &hidden, &logits);
    REQUIRE(static_cast<int64_t>(hidden.size()) == 2 * kHidden);
    // The CONDITIONAL row, which is the only one the golden stores.
    out.hidden.insert(out.hidden.end(), hidden.begin(),
                      hidden.begin() + static_cast<ptrdiff_t>(kHidden));
    ++out.steps;
  }
  return out;
}

// The teacher-forced run, ONCE for the file. Three cases read it and it is
// deterministic (no sampler is consulted — the codes are the capture's), so
// re-running it per case would spend 10 more minutes proving the same numbers.
const TeacherForced& Forced() {
  static TeacherForced run = [] {
    std::vector<int64_t> code_shape;
    const std::vector<int32_t> codes = LoadI32Npy("rvq_codes.npy", &code_shape);
    REQUIRE(code_shape.size() == 2);
    REQUIRE(code_shape[0] == kCodeRows);
    return RunTeacherForced(codes, kCodeRows, kFrames);
  }();
  return run;
}

}  // namespace

// ---------------------------------------------------------------------------
// The prompt
// ---------------------------------------------------------------------------

TEST_CASE("music3 llm real: the capture's prompt tokenizes to the oracle's own ids") {
  if (SkipIfMissing("music3 llm prompt")) return;
  const m3::Music3ArWeights& ar = Weights().ar;

  // The two configs agree, which is what makes the rest of this file about the
  // model rather than about which model.
  CHECK(ar.lm_config.vocab_size == 200000);
  CHECK(ar.lm_config.hidden_size == kHidden);
  CHECK(ar.lm_config.num_hidden_layers == 36);
  CHECK(ar.lm_config.num_attention_heads == 32);
  CHECK(ar.lm_config.num_key_value_heads == 8);
  CHECK(ar.lm_config.head_dim == 128);
  CHECK(ar.lm_config.rope_theta == doctest::Approx(1000000.0));
  // `tie_word_embeddings: false` — the lm_head is REAL weights, and the loader
  // refuses a tied read (minimax_music3_llm.cpp) because the shapes would match.
  CHECK_FALSE(ar.lm.tie_word_embeddings);
  CHECK_FALSE(ar.lm.lm_head.Empty());

  const std::string prompt = m3::AssembleArPrompt(kDescription, kLyrics);
  const std::vector<int32_t> ids = ar.Encode(prompt);
  MESSAGE("assembled prompt: " << prompt.size() << " bytes, " << ids.size() << " tokens");
  CHECK(static_cast<int64_t>(ids.size()) == kPromptTokens);
  // The first and last two ids are the ones the CFG rewrite preserves, so they
  // are named rather than left to the count: `<|im_start|>` opens, and
  // `<|im_end|><|audio_start|>` close (encoders.py:207-210, :216-217).
  CHECK(ids.front() == 151644);
  CHECK(ids[ids.size() - 2] == 151645);
  CHECK(ids.back() == 151669);

  const std::vector<int32_t> unconditional = m3::UnconditionalPromptIds(ids);
  REQUIRE(unconditional.size() == ids.size());
  int64_t rewritten = 0;
  int64_t kept = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i == 0 || i + 2 >= ids.size()) {
      if (unconditional[i] == ids[i]) ++kept;
    } else if (unconditional[i] == m3::kAudioCfgTokenId) {
      ++rewritten;
    }
  }
  MESSAGE("unconditional row: " << kept << " tokens kept, " << rewritten << " rewritten to "
                                << m3::kAudioCfgTokenId);
  CHECK(kept == 3);
  CHECK(rewritten == kPromptTokens - 3);
}

// ---------------------------------------------------------------------------
// The hidden states
// ---------------------------------------------------------------------------
//
// ─── THE TOLERANCE, AND WHY IT IS STATED IN bf16 ULPs ───────────────────────
//
// The goldens are bf16 values stored as float32 (checked below rather than
// assumed). Upstream computes in bf16 and rounds at every module boundary
// through 36 decoder layers; this port runs the SAME dtype through a different
// attention kernel and a different GEMM, so the two cannot be bit-equal and a
// tolerance is unavoidable. It is stated in bf16 ULPs so it means the same thing
// at every magnitude, and so it reads against the W2/W3 depth-decoder gate's
// bound in the same unit.
//
// See the recorded MEASUREMENT and the matched control in the message this case
// prints and in .agents/specs/minimax-music3.md's `## Now`.
TEST_CASE("music3 llm real: 25 teacher-forced steps reproduce frame_hiddens[:, :4096]") {
  if (SkipIfMissing("music3 llm hidden states")) return;
  std::vector<int64_t> shape;
  const std::vector<float> frame_hiddens = LoadF32Npy("frame_hiddens.npy", &shape);
  REQUIRE(shape.size() == 2);
  REQUIRE(shape[0] == kFrames);
  REQUIRE(shape[1] == kCodebooks * kHidden);
  std::vector<int64_t> code_shape;
  const std::vector<int32_t> codes = LoadI32Npy("rvq_codes.npy", &code_shape);
  REQUIRE(code_shape[0] == kCodeRows);
  REQUIRE(code_shape[1] == kCodebooks);
  // The alignment this whole file depends on: ONE more code row than frame,
  // because row 0 is the priming decode that emits no frame.
  REQUIRE(code_shape[0] == kFrames + 1);

  const TeacherForced& run = Forced();
  REQUIRE(run.steps == kFrames);
  REQUIRE(static_cast<int64_t>(run.hidden.size()) == kFrames * kHidden);

  std::vector<float> want;
  want.reserve(static_cast<size_t>(kFrames * kHidden));
  for (int64_t frame = 0; frame < kFrames; ++frame) {
    const auto begin = frame_hiddens.begin() + static_cast<ptrdiff_t>(frame * kCodebooks * kHidden);
    want.insert(want.end(), begin, begin + static_cast<ptrdiff_t>(kHidden));
  }
  // Every golden value must be bf16-exact — that is what makes a bf16-ULP bound
  // the right SHAPE of bound. CHECKED, not assumed.
  int64_t not_bf16 = 0;
  for (const float value : want) {
    if (vt::BF16ToF32(vt::F32ToBF16(value)) != value) ++not_bf16;
  }
  MESSAGE("frame_hiddens[:, :4096]: " << want.size() << " values, " << not_bf16
                                      << " not representable in bf16");
  CHECK(not_bf16 == 0);

  const UlpReport report = CompareUlps(run.hidden, want, kHiddenUlpAllow, kHiddenAbsFloor);
  MESSAGE("language model vs frame_hiddens[:, :4096]: "
          << report.compared << " values compared over " << run.steps
          << " teacher-forced steps, " << report.identical << " bit-identical, "
          << report.outside << " beyond " << kHiddenUlpAllow << " bf16 ULP, max|d| "
          << report.max_abs << " (" << report.max_ulps << " ULP), mean|d| " << report.mean_abs
          << ", |ref|max " << report.ref_absmax << ", identical fraction "
          << report.identical_fraction() << ", outside fraction " << report.outside_fraction()
          << (report.first_bad >= 0 ? ", first beyond at " + std::to_string(report.first_bad)
                                    : std::string()));

  // Per-step, so a divergence that GROWS with depth is visible rather than
  // averaged away — that is the signature of a wrong KV write, which a whole-run
  // mean would hide behind 24 good steps.
  for (int64_t step = 0; step < run.steps; step += 6) {
    const std::vector<float> a(run.hidden.begin() + static_cast<ptrdiff_t>(step * kHidden),
                               run.hidden.begin() + static_cast<ptrdiff_t>((step + 1) * kHidden));
    const std::vector<float> b(want.begin() + static_cast<ptrdiff_t>(step * kHidden),
                               want.begin() + static_cast<ptrdiff_t>((step + 1) * kHidden));
    const UlpReport per = CompareUlps(a, b, kHiddenUlpAllow, kHiddenAbsFloor);
    MESSAGE("  step " << step << ": " << per.identical << "/" << per.compared
                      << " bit-identical, mean|d| " << per.mean_abs << ", max|d| "
                      << per.max_abs);
  }

  CHECK(report.compared == kFrames * kHidden);
  CHECK(report.ref_absmax > 1.0);
  // Four bounds, none of which alone is a gate: the bit-identical FRACTION is
  // what separates a correct bf16 mirror from a wider or narrower one, and the
  // absolute bounds are what stop a uniformly scaled tensor from passing on
  // counts. A Pearson coefficient would see neither (spec §5).
  CHECK(report.identical_fraction() >= kHiddenIdenticalFloor);
  CHECK(report.mean_abs <= kHiddenMeanAbsTol);
  CHECK(report.max_abs <= kHiddenMaxAbsTol);
  CHECK(report.outside_fraction() <= kHiddenOutsideFraction);

  // ── THE GATE'S OWN NEGATIVE CONTROL ──────────────────────────────────────
  //
  // Our step k against the golden's step k+1. That is a WRONG answer of exactly
  // the shape this file could produce — an off-by-one in the priming-step
  // alignment, which `rvq_codes` having one more row than `frame_hiddens` makes
  // easy — and every value in it is a real hidden state of this very model on
  // this very prompt, so it is the hardest wrong answer to tell apart. If the
  // bounds above admitted it they would be bounding nothing.
  std::vector<float> shifted_got(run.hidden.begin(),
                                 run.hidden.end() - static_cast<ptrdiff_t>(kHidden));
  std::vector<float> shifted_want(want.begin() + static_cast<ptrdiff_t>(kHidden), want.end());
  const UlpReport shifted =
      CompareUlps(shifted_got, shifted_want, kHiddenUlpAllow, kHiddenAbsFloor);
  MESSAGE("NEGATIVE CONTROL, one-step-shifted alignment: "
          << shifted.compared << " values compared, " << shifted.identical
          << " bit-identical, mean|d| " << shifted.mean_abs << ", max|d| " << shifted.max_abs
          << ", identical fraction " << shifted.identical_fraction() << ", outside fraction "
          << shifted.outside_fraction());
  const bool shifted_passes = shifted.identical_fraction() >= kHiddenIdenticalFloor &&
                              shifted.mean_abs <= kHiddenMeanAbsTol &&
                              shifted.max_abs <= kHiddenMaxAbsTol &&
                              shifted.outside_fraction() <= kHiddenOutsideFraction;
  CHECK_FALSE(shifted_passes);
}

// ---------------------------------------------------------------------------
// The rank of the oracle's own draw
// ---------------------------------------------------------------------------
//
// The discriminator that no scaling can pass. `frame_hiddens` could in principle
// be reproduced by something uniformly off; the ORDER of the guided logits could
// not. This mirrors what the W2/W3 gate does for the depth heads (mean rank 8.99
// where chance is 511.5) at the semantic stage, where the window is 16384 wide
// and chance is 8191.5.
TEST_CASE("music3 llm real: the oracle's sampled codes rank highly under our guided logits") {
  if (SkipIfMissing("music3 llm code ranks")) return;
  std::vector<int64_t> code_shape;
  const std::vector<int32_t> codes = LoadI32Npy("rvq_codes.npy", &code_shape);
  REQUIRE(code_shape[0] == kCodeRows);

  const TeacherForced& run = Forced();
  REQUIRE(static_cast<int64_t>(run.guided.size()) == kFrames);

  int64_t ranked = 0;
  int64_t worst = 0;
  int64_t top1 = 0;
  double sum_rank = 0.0;
  for (int64_t k = 0; k < kFrames; ++k) {
    const std::vector<float>& guided = run.guided[static_cast<size_t>(k)];
    const int64_t token =
        static_cast<int64_t>(codes[static_cast<size_t>(k * kCodebooks)]) + m3::kAudioCodeOffset;
    REQUIRE(token < static_cast<int64_t>(guided.size()));
    const float target = guided[static_cast<size_t>(token)];
    // `_sample_top_k` restricts to a top-50 that only FINITE candidates reach;
    // a golden code that came out -inf here would mean the guidance pipeline
    // masked away the token the oracle actually drew.
    REQUIRE(std::isfinite(target));
    int64_t rank = 0;
    for (const float value : guided) {
      if (value > target) ++rank;
    }
    if (rank == 0) ++top1;
    worst = std::max(worst, rank);
    sum_rank += static_cast<double>(rank);
    ++ranked;
  }
  MESSAGE("golden semantic code ranks under the reproduced guided logits: "
          << ranked << " codes, mean rank " << (sum_rank / static_cast<double>(ranked))
          << ", worst rank " << worst << ", " << top1
          << " at rank 0; chance over the 16384-entry semantic window would be 8191.5");
  CHECK(ranked == kFrames);
  // 50 is `_AR_SAMPLING_TOP_K`: every code the oracle drew came from ITS top 50,
  // so a code outside OUR top 50 is a distribution we could not have drawn it
  // from. MEASURED on this box: mean rank 2.48, worst 15, 10 of 25 at rank 0 —
  // three orders of magnitude inside chance, which is the strongest statement
  // anything in this row makes about the language model being RIGHT rather than
  // merely close.
  CHECK(worst < m3::kArSamplingTopK);
  CHECK(sum_rank / static_cast<double>(ranked) <= 8.0);  // measured 2.48
}

// ---------------------------------------------------------------------------
// Through the stage that consumes them
// ---------------------------------------------------------------------------

TEST_CASE("music3 llm real: the reproduced hiddens still reach condition_chunk0") {
  if (SkipIfMissing("music3 llm condition mix")) return;
  std::vector<int64_t> shape;
  const std::vector<float> frame_hiddens = LoadF32Npy("frame_hiddens.npy", &shape);
  REQUIRE(shape[0] == kFrames);
  const std::vector<float> want = LoadF32Npy("condition_chunk0.npy", &shape);
  REQUIRE(shape[0] == kLatentLength);
  REQUIRE(shape[1] == kOutDim);
  std::vector<int64_t> code_shape;
  const std::vector<int32_t> codes = LoadI32Npy("rvq_codes.npy", &code_shape);

  const TeacherForced& run = Forced();
  REQUIRE(run.steps == kFrames);

  // OUR language-model rows, the GOLDEN depth rows. The depth decoder has its
  // own full-scale gate (tests/parity/test_minimax_music3_ar_real.cpp) and
  // mixing the two failures in one tensor would make neither attributable, so
  // this case swaps in exactly the half it is not gating.
  std::vector<float> mixed(static_cast<size_t>(kFrames * kCodebooks * kHidden));
  for (int64_t frame = 0; frame < kFrames; ++frame) {
    std::copy(run.hidden.begin() + static_cast<ptrdiff_t>(frame * kHidden),
              run.hidden.begin() + static_cast<ptrdiff_t>((frame + 1) * kHidden),
              mixed.begin() + static_cast<ptrdiff_t>(frame * kCodebooks * kHidden));
    std::copy(
        frame_hiddens.begin() + static_cast<ptrdiff_t>(frame * kCodebooks * kHidden + kHidden),
        frame_hiddens.begin() + static_cast<ptrdiff_t>((frame + 1) * kCodebooks * kHidden),
        mixed.begin() + static_cast<ptrdiff_t>(frame * kCodebooks * kHidden + kHidden));
  }

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(
      (fs::path(Weights().paths.condition_encoder_dir) / "diffusion_pytorch_model.safetensors")
          .string());
  const auto at_runtime = [&file](const std::string& name) {
    const vllm::StTensor& tensor = file.Get(name);
    REQUIRE(tensor.dtype == "F32");
    std::vector<float> out(tensor.nbytes / sizeof(float));
    std::memcpy(out.data(), tensor.data, tensor.nbytes);
    for (float& value : out) value = vt::BF16ToF32(vt::F32ToBF16(value));
    return out;
  };
  m3::ConditionMixConfig mix;
  m3::ConditionMixWeights weights;
  weights.layer_weight_logits = at_runtime("layer_weight_logits");
  weights.layer_scale = at_runtime("layer_scale");
  weights.proj_weight = at_runtime("proj.weight");
  weights.proj_bias = at_runtime("proj.bias");

  const std::vector<float> got =
      m3::ConditionMix(mixed, kFrames, mix, weights, m3::ArCompute::kBFloat16);
  const UlpReport report = CompareUlps(got, want, 2.0, 1.0 / 64.0);
  MESSAGE("condition mix over OUR language-model rows vs condition_chunk0: "
          << report.compared << " values compared, " << report.identical << " bit-identical, "
          << report.outside << " beyond 2 bf16 ULP, max|d| " << report.max_abs << ", mean|d| "
          << report.mean_abs << ", |ref|max " << report.ref_absmax << ", identical fraction "
          << report.identical_fraction());
  CHECK(report.compared == kLatentLength * kOutDim);
  CHECK(report.ref_absmax > 1.0);
  // ATTRIBUTIVE rather than independent: what it shows is that the language
  // model's own bf16 spread arrives at the consuming stage as the same spread
  // and not as something larger. W3's gate over the GOLDEN rows reaches
  // 175989/176128 bit-identical at mean|d| 1.99e-07; over OURS the numbers below
  // are the language model's error carried through one softmax-weighted sum, one
  // k=3 Conv1d and a nearest interpolation. MEASURED here: 16319 bit-identical
  // (9.27%), mean|d| 3.916e-03, max|d| 4.688e-02, 1531 (0.87%) beyond 2 ULP.
  CHECK(report.identical_fraction() >= 0.06);
  CHECK(report.mean_abs <= 5.5e-3);
  CHECK(report.max_abs <= 0.1);
  CHECK(report.outside_fraction() <= 0.02);
}
