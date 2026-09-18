// vllm.cpp original test harness; no upstream mirror.
//
// BACKEND-ROCM-BF16-MOE (#3116) — the LM-head output boundary.
//
// THE QUESTION THIS FILE SETTLES. The native Qwen3-MoE forward ends with an
// UNTIED lm_head projection that stores F32 logits
// (src/vllm/model_executor/models/qwen3_moe.cpp ForwardLayers, `vt::Matmul` into
// a `DType::kF32` buffer). The pinned compiled primary stores its head output in
// BF16 and widens it to F32 only where the sampler needs it
// (.agents/specs/rocm-residual-norm.md:320). Issue #3116 asks whether mirroring
// that boundary changes the native head arithmetic, and whether that mirror
// alone moves the decode step-6 argmax.
//
// THE INSTRUMENT. `oracle-diagnostic-2/L33-C2-R0-head-6.{json,bin}` holds the
// primary's exact head input (`hidden` [2,128] bf16), its exact lm_head weight
// (`weight` [128,128] bf16 in torch Linear [vocab,hidden] order), and its exact
// BF16 head output (`logits` [2,128] bf16) for decode step 6 of the L33/C2/R0
// workload — the exact-tie step of the BF16-MoE spec. Replaying those bytes
// through the operator the forward itself calls isolates the head from the
// hidden-state divergence (#3115) that the same step also carries.
//
// FOUR ARMS, one projection, no reimplementation:
//   (a)  `vt::Matmul(src, lm) -> f32`            the CURRENT native boundary;
//   (a') (a) narrowed to bf16 on the host (RNE)  what a store-only change buys;
//   (b)  `vt::Matmul(src, lm) -> bf16` then
//        `vt::CastF32`                           the PRIMARY's boundary, mirrored;
//   (c)  `logits.bin` widened bf16 -> f32        the primary's own words.
//
// The weight is uploaded in the [H, vocab] orientation the loader builds
// (`LoadBf16Transposed("lm_head.weight")`, qwen3_moe_weights.cpp:149), because
// that is the orientation the forward's `vt::Matmul` consumes; the artifact
// stores the torch Linear [vocab, H] order.
//
// `VT_MOE_HEAD_FIXTURE` names the directory holding the artifact. The two cases
// that replay it are decorated `doctest::skip(...)` when the variable is unset,
// so doctest reports them skipped while the fixture-free call-site case below
// still runs and still gates. The process then exits 77 (CTest: Skipped) rather
// than reporting a green gate that never ran, which is the convention
// `tests/CMakeLists.txt` documents -- and it does so ONLY when nothing failed,
// so a reddened case keeps doctest's own non-zero status.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/lm_head_projection.h"
#include "vllm/model_executor/models/qwen3_moe.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#if defined(VLLM_CPP_HIP)
#include <hip/hip_runtime_api.h>
#endif

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// ─── The captured artifact ──────────────────────────────────────────────────

struct HeadArtifact {
  std::string label;
  int64_t step = 0;
  int64_t rows = 0;
  int64_t hidden = 0;
  int64_t vocab = 0;
  std::vector<uint16_t> hidden_bf16;  // [rows, hidden] — the primary's head input
  std::vector<uint16_t> weight_bf16;  // [vocab, hidden] — torch Linear lm_head.weight
  std::vector<uint16_t> logits_bf16;  // [rows, vocab]   — the primary's head output
};

std::vector<uint16_t> ReadBf16(const std::string& path, size_t words) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), "cannot open ", path);
  std::vector<uint16_t> out(words);
  in.read(reinterpret_cast<char*>(out.data()),
          static_cast<std::streamsize>(words * sizeof(uint16_t)));
  REQUIRE_MESSAGE(in.gcount() == static_cast<std::streamsize>(words * sizeof(uint16_t)),
                  "short read from ", path);
  return out;
}

HeadArtifact LoadArtifact(const std::string& dir, const std::string& label, int64_t step) {
  const std::string stem = label + "-head-" + std::to_string(step);
  const std::string meta_path = dir + "/" + stem + ".json";
  std::ifstream meta_in(meta_path);
  REQUIRE_MESSAGE(meta_in.good(), "cannot open ", meta_path);
  nlohmann::json meta;
  meta_in >> meta;

  auto tensor = [&](const char* name, std::vector<int64_t>& shape) {
    const auto& t = meta.at("tensors").at(name);
    REQUIRE(t.at("dtype").get<std::string>() == "torch.bfloat16");
    REQUIRE(t.at("element_bytes").get<int64_t>() == 2);
    shape = t.at("shape").get<std::vector<int64_t>>();
    REQUIRE(shape.size() == 2);
    return ReadBf16(dir + "/" + t.at("file").get<std::string>(),
                    static_cast<size_t>(shape[0] * shape[1]));
  };

  HeadArtifact a;
  a.label = meta.at("label").get<std::string>();
  a.step = meta.at("step").get<int64_t>();
  std::vector<int64_t> hs, ws, ls;
  a.hidden_bf16 = tensor("hidden", hs);
  a.weight_bf16 = tensor("weight", ws);
  a.logits_bf16 = tensor("logits", ls);
  REQUIRE(hs == ls);
  REQUIRE(ws[0] == ls[1]);
  REQUIRE(ws[1] == hs[1]);
  a.rows = hs[0];
  a.hidden = hs[1];
  a.vocab = ls[1];
  return a;
}

// ─── Host-side arithmetic the comparison needs ──────────────────────────────

// bf16 -> f32 is the high half of the f32 word; the widening is exact.
float Widen(uint16_t word) {
  const uint32_t bits = static_cast<uint32_t>(word) << 16;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<float> Widen(const std::vector<uint16_t>& words) {
  std::vector<float> out(words.size());
  for (size_t i = 0; i < words.size(); ++i) out[i] = Widen(words[i]);
  return out;
}

// f32 -> bf16 round-to-nearest-even, matching vt::F32ToBF16 (src/vt/dtype.cpp)
// and the bf16 store both device backends perform.
uint16_t NarrowRne(float value) {
  uint32_t u = 0;
  std::memcpy(&u, &value, sizeof(u));
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu)) {
    return static_cast<uint16_t>((u >> 16) | 0x0040u);
  }
  const uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

std::vector<uint16_t> NarrowRne(const std::vector<float>& values) {
  std::vector<uint16_t> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) out[i] = NarrowRne(values[i]);
  return out;
}

// Lowest index wins an exact tie, which is the native GreedyArgmax rule
// (src/vt/rocm/rocm_dense_basic.hip ArgmaxK, `v > best || (v == best && j < arg)`).
std::vector<int> TopOrder(const std::vector<float>& row, int64_t vocab, int count) {
  std::vector<int> order(static_cast<size_t>(vocab));
  for (int64_t j = 0; j < vocab; ++j) order[static_cast<size_t>(j)] = static_cast<int>(j);
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
    const float vx = row[static_cast<size_t>(x)], vy = row[static_cast<size_t>(y)];
    if (vx != vy) return vx > vy;
    return x < y;
  });
  order.resize(static_cast<size_t>(count));
  return order;
}

double MaxAbsDiff(const std::vector<float>& x, const std::vector<float>& y) {
  double worst = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(x[i]) - static_cast<double>(y[i])));
  return worst;
}

double MeanAbsDiff(const std::vector<float>& x, const std::vector<float>& y) {
  double sum = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
    sum += std::fabs(static_cast<double>(x[i]) - static_cast<double>(y[i]));
  return x.empty() ? 0.0 : sum / static_cast<double>(x.size());
}

size_t DifferingWords(const std::vector<uint16_t>& x, const std::vector<uint16_t>& y) {
  size_t n = 0;
  for (size_t i = 0; i < x.size(); ++i) n += x[i] != y[i];
  return n;
}

std::string Describe(const std::vector<float>& row, const std::vector<int>& order) {
  std::string out = "argmax=" + std::to_string(order[0]) + " top2=[";
  for (size_t i = 0; i < order.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(order[i]) + ": " +
           std::to_string(static_cast<double>(row[static_cast<size_t>(order[i])]));
  }
  return out + "]";
}

// ─── Device plumbing ────────────────────────────────────────────────────────

struct DeviceScope {
  int previous = 0;
  bool rocm = false;
  explicit DeviceScope(Device device) : rocm(device.type == DeviceType::kROCM) {
#if defined(VLLM_CPP_HIP)
    if (rocm) {
      REQUIRE(hipGetDevice(&previous) == hipSuccess);
      REQUIRE(hipSetDevice(device.index) == hipSuccess);
    }
#else
    (void)device;
#endif
  }
  ~DeviceScope() {
#if defined(VLLM_CPP_HIP)
    if (rocm) (void)hipSetDevice(previous);
#endif
  }
};

struct QueueHandle {
  Queue q{};
  explicit QueueHandle(Device device) {
    DeviceScope scope(device);
    q = vt::CreateQueue(device);
  }
  ~QueueHandle() {
    DeviceScope scope(q.device);
    vt::DestroyQueue(q);
  }
};

// One device-resident bf16 upload, freed on scope exit.
struct DeviceBf16 {
  vt::Backend& b;
  Queue& q;
  void* p = nullptr;
  Tensor t{};
  DeviceBf16(vt::Backend& backend, Queue& queue, const std::vector<uint16_t>& words,
             const std::vector<int64_t>& shape)
      : b(backend), q(queue) {
    DeviceScope scope(queue.device);
    p = b.Alloc(words.size() * sizeof(uint16_t));
    b.Copy(q, p, words.data(), words.size() * sizeof(uint16_t));
    b.Synchronize(q);
    t = Tensor::Contiguous(p, DType::kBF16, q.device, {shape[0], shape[1]});
  }
  ~DeviceBf16() {
    DeviceScope scope(q.device);
    b.Free(p);
  }
  std::vector<uint16_t> Read() const {
    DeviceScope scope(q.device);
    std::vector<uint16_t> out(static_cast<size_t>(t.Numel()));
    b.Copy(q, out.data(), p, out.size() * sizeof(uint16_t));
    b.Synchronize(q);
    return out;
  }
  DeviceBf16(const DeviceBf16&) = delete;
  DeviceBf16& operator=(const DeviceBf16&) = delete;
};

struct DeviceF32 {
  vt::Backend& b;
  Queue& q;
  void* p = nullptr;
  Tensor t{};
  DeviceF32(vt::Backend& backend, Queue& queue, const std::vector<int64_t>& shape)
      : b(backend), q(queue) {
    DeviceScope scope(queue.device);
    p = b.Alloc(static_cast<size_t>(shape[0] * shape[1]) * sizeof(float));
    t = Tensor::Contiguous(p, DType::kF32, q.device, {shape[0], shape[1]});
  }
  ~DeviceF32() {
    DeviceScope scope(q.device);
    b.Free(p);
  }
  std::vector<float> Read() const {
    DeviceScope scope(q.device);
    std::vector<float> out(static_cast<size_t>(t.Numel()));
    b.Copy(q, out.data(), p, out.size() * sizeof(float));
    b.Synchronize(q);
    return out;
  }
  DeviceF32(const DeviceF32&) = delete;
  DeviceF32& operator=(const DeviceF32&) = delete;
};

std::vector<Device> Devices() {
  std::vector<Device> devices{{DeviceType::kCPU, 0}};
#if defined(VLLM_CPP_HIP)
  if (vt::TryGetBackend({DeviceType::kROCM, 0}) != nullptr)
    devices.push_back({DeviceType::kROCM, 0});
#endif
  return devices;
}

std::string DeviceName(Device device) {
  return device.type == DeviceType::kROCM ? "ROCM" : "CPU";
}

// The artifact stores the torch Linear [vocab, hidden] order; the loader hands
// the untied head to `vt::Matmul` as [H, vocab]
// (`qwen3_moe_weights.cpp` `LoadBf16Transposed("lm_head.weight")`), so the
// replay uploads the same transposed bytes the model holds.
std::vector<uint16_t> TransposedWeight(const HeadArtifact& art) {
  std::vector<uint16_t> weight_hv(art.weight_bf16.size());
  for (int64_t v = 0; v < art.vocab; ++v)
    for (int64_t h = 0; h < art.hidden; ++h)
      weight_hv[static_cast<size_t>(h * art.vocab + v)] =
          art.weight_bf16[static_cast<size_t>(v * art.hidden + h)];
  return weight_hv;
}

// ─── The three arms, through the operators the forward calls ────────────────

struct Measurements {
  std::vector<float> a_f32;       // (a) native F32 head output
  std::vector<uint16_t> a_words;  // (a') that output narrowed to bf16 on the host
  std::vector<float> b_f32;       // (b) native BF16 head output, widened by vt::CastF32
  std::vector<uint16_t> b_words;  // (b) before widening
  std::vector<uint16_t> c_words;  // (c) the primary's BF16 words
};

Measurements Measure(vt::Backend& b, Queue& q, const HeadArtifact& art) {
  DeviceScope scope(q.device);
  DeviceBf16 hidden(b, q, art.hidden_bf16, {art.rows, art.hidden});
  DeviceBf16 weight(b, q, TransposedWeight(art), {art.hidden, art.vocab});
  DeviceBf16 head(b, q, std::vector<uint16_t>(static_cast<size_t>(art.rows * art.vocab), 0),
                  {art.rows, art.vocab});
  DeviceF32 f32_out(b, q, {art.rows, art.vocab});
  DeviceF32 widened(b, q, {art.rows, art.vocab});

  // (a) the current boundary: the projection stores F32.
  vt::Matmul(q, f32_out.t, hidden.t, weight.t);
  // (b) the primary's boundary: the projection stores BF16, the shared op widens.
  vt::Matmul(q, head.t, hidden.t, weight.t);
  vt::CastF32(q, widened.t, head.t);
  b.Synchronize(q);

  Measurements m;
  m.a_f32 = f32_out.Read();
  m.a_words = NarrowRne(m.a_f32);
  m.b_f32 = widened.Read();
  m.b_words = head.Read();
  m.c_words = art.logits_bf16;
  return m;
}

void Report(const std::string& device, int64_t row, const HeadArtifact& art,
            const Measurements& m) {
  const int64_t V = art.vocab;
  const size_t base = static_cast<size_t>(row * V);
  const auto slice = [&](const std::vector<float>& all) {
    return std::vector<float>(all.begin() + static_cast<ptrdiff_t>(base),
                              all.begin() + static_cast<ptrdiff_t>(base + V));
  };
  const auto word_slice = [&](const std::vector<uint16_t>& all) {
    return std::vector<uint16_t>(all.begin() + static_cast<ptrdiff_t>(base),
                                 all.begin() + static_cast<ptrdiff_t>(base + V));
  };
  const std::vector<float> a = slice(m.a_f32);
  const std::vector<float> bv = slice(m.b_f32);
  const std::vector<uint16_t> aw = word_slice(m.a_words);
  const std::vector<uint16_t> bw = word_slice(m.b_words);
  const std::vector<uint16_t> cw = word_slice(m.c_words);
  const std::vector<float> cv = Widen(cw);

  const std::vector<int> order_a = TopOrder(a, V, 2);
  const std::vector<int> order_b = TopOrder(bv, V, 2);
  const std::vector<int> order_c = TopOrder(cv, V, 2);
  const double max_ac = MaxAbsDiff(a, cv), mean_ac = MeanAbsDiff(a, cv);
  const double max_ab = MaxAbsDiff(a, bv), mean_ab = MeanAbsDiff(a, bv);
  const size_t differ_bc = DifferingWords(bw, cw);
  const size_t differ_ac = DifferingWords(aw, cw);

  std::cout << "[lm-head boundary] device=" << device << " row=" << row
            << " max|a-c|=" << max_ac << " mean|a-c|=" << mean_ac
            << " max|a-b|=" << max_ab << " mean|a-b|=" << mean_ab << "\n"
            << "  b==c: " << (differ_bc == 0 ? "EQUAL" : "DIFFER") << " (" << differ_bc << "/"
            << V << " bf16 words)"
            << "  a'==c: " << (differ_ac == 0 ? "EQUAL" : "DIFFER") << " (" << differ_ac << "/" << V
            << " bf16 words), max|a'-c|=" << MaxAbsDiff(Widen(aw), cv) << "\n"
            << "  (a) f32  " << Describe(a, order_a) << "\n"
            << "  (b) bf16 " << Describe(bv, order_b) << "\n"
            << "  (c) prim " << Describe(cv, order_c) << std::endl;

  CAPTURE(device);
  CAPTURE(row);
  // The mirror's whole claim: given the primary's own head input, the native
  // BF16 boundary reproduces the primary's words element for element, and picks
  // the primary's argmax.
  CHECK(differ_bc == 0);
  CHECK(order_b[0] == order_c[0]);
  CHECK(bw[static_cast<size_t>(order_b[0])] == cw[static_cast<size_t>(order_c[0])]);
  // ...and the F32 boundary the tree ships today does NOT reproduce them.
  CHECK(max_ac > 0.0);
}

// ─── The fixture-absent run: per-case skip, process-level 77 ────────────────
//
// The two replay cases need `VT_MOE_HEAD_FIXTURE`; the production call-site case
// at the bottom needs no directory at all. Before this repair both replay cases
// called `std::exit(77)`, and because doctest runs the cases in file order that
// stopped the binary at the FIRST one whenever the fixture was absent: the
// fixture-free case never ran, and the whole binary read as Skipped although a
// gate inside it could have run (the fresh review's finding 1).
//
// The replay cases are now decorated `doctest::skip(HeadFixtureAbsent())`, so
// doctest itself reports them as skipped in its summary while the call-site case
// still runs and still gates. A fixture-absent run still exits 77 -- CTest
// reports Skipped and a shell chain stops, the convention `tests/CMakeLists.txt`
// documents -- but ONLY when nothing failed, so a reddened call-site case keeps
// doctest's own non-zero exit instead of being hidden behind the skip.
bool HeadFixtureAbsent() {
  const char* directory = std::getenv("VT_MOE_HEAD_FIXTURE");
  return directory == nullptr || directory[0] == '\0';
}

bool g_replay_cases_skipped = false;
bool g_run_failed = false;

// A listener, not a reporter: listeners are always active whatever `-r=`
// selects, so neither the skip note nor the failure guard can be switched off
// from the command line (`tests/vt/test_ops_attention_cross.cpp:662` has the
// same shape).
struct ReplaySkipListener : public doctest::IReporter {
  explicit ReplaySkipListener(const doctest::ContextOptions&) {}
  void test_run_start() override {}
  void report_query(const doctest::QueryData&) override {}
  void test_run_end(const doctest::TestRunStats& stats) override {
    g_run_failed = stats.numTestCasesFailed != 0 || stats.numAssertsFailed != 0;
  }
  void test_case_start(const doctest::TestCaseData&) override {}
  void test_case_reenter(const doctest::TestCaseData&) override {}
  void test_case_end(const doctest::CurrentTestCaseStats&) override {}
  void test_case_exception(const doctest::TestCaseException&) override {}
  void subcase_start(const doctest::SubcaseSignature&) override {}
  void subcase_end() override {}
  void log_assert(const doctest::AssertData&) override {}
  void log_message(const doctest::MessageData&) override {}
  // This also fires for a case a filter excluded; only the decorator sets
  // `m_skip`, and only that means "the fixture is not here".
  void test_case_skipped(const doctest::TestCaseData& tc) override {
    if (!tc.m_skip) return;
    g_replay_cases_skipped = true;
    std::cout << "[lm-head boundary] SKIPPED: " << tc.m_name
              << " (set VT_MOE_HEAD_FIXTURE to run it)" << std::endl;
  }
};
DOCTEST_REGISTER_LISTENER("vt-moe-head-replay-skip", 1, ReplaySkipListener);

// Registered during static initialization, so it runs after doctest's main has
// printed its summary and returned. `std::_Exit` rather than `std::exit`: this
// IS an exit handler, and re-entering the exit sequence is undefined.
void ExitSkippedWhenTheReplayDidNotRun() {
  if (!g_replay_cases_skipped || g_run_failed) return;
  std::cout.flush();
  std::fflush(nullptr);
  std::fprintf(stderr,
               "\n*** SKIPPED (exit 77): the artifact-replay cases did not run because "
               "VT_MOE_HEAD_FIXTURE is unset. The fixture-free production case above DID "
               "run and its result stands; this status says only that the replay gate did "
               "not. ***\n\n");
  std::fflush(stderr);
  std::_Exit(77);
}

struct RegisterExitSkippedWhenTheReplayDidNotRun {
  RegisterExitSkippedWhenTheReplayDidNotRun() { std::atexit(&ExitSkippedWhenTheReplayDidNotRun); }
};
[[maybe_unused]] const RegisterExitSkippedWhenTheReplayDidNotRun
    g_exit_skipped_when_the_replay_did_not_run;

}  // namespace

TEST_CASE("qwen3 MoE LM-head BF16 boundary: primary artifact replay" *
          doctest::skip(HeadFixtureAbsent())) {
  // The decorator above covers the normal run; `--no-skip` forces the case
  // anyway, and then the absent fixture must FAIL rather than reach
  // `LoadArtifact` as a null path.
  REQUIRE_MESSAGE(!HeadFixtureAbsent(),
                  "VT_MOE_HEAD_FIXTURE must name the directory holding "
                  "L33-C2-R0-head-6.{json,hidden.bin,weight.bin,logits.bin}");
  const HeadArtifact art = LoadArtifact(std::getenv("VT_MOE_HEAD_FIXTURE"), "L33-C2-R0", 6);
  for (const Device device : Devices()) {
    CAPTURE(DeviceName(device));
    QueueHandle queue(device);
    vt::Backend& backend = vt::GetBackend(device.type);
    const Measurements m = Measure(backend, queue.q, art);
    for (int64_t row = 0; row < art.rows; ++row) Report(DeviceName(device), row, art, m);
  }
}

// THE PRODUCTION CASE. This is the red-first gate for #3116: it calls the head
// projection the forward calls, over the primary's own captured head input, and
// requires the logits it returns to be the primary's BF16 words widened to F32 —
// not merely close to them.
TEST_CASE("qwen3 MoE LM-head BF16 boundary: the production projection mirrors the primary" *
          doctest::skip(HeadFixtureAbsent())) {
  // Same decorator, same `--no-skip` guard as the replay case above.
  REQUIRE_MESSAGE(!HeadFixtureAbsent(),
                  "VT_MOE_HEAD_FIXTURE must name the directory holding "
                  "L33-C2-R0-head-6.{json,hidden.bin,weight.bin,logits.bin}");
  const HeadArtifact art = LoadArtifact(std::getenv("VT_MOE_HEAD_FIXTURE"), "L33-C2-R0", 6);
  const std::vector<float> primary = Widen(art.logits_bf16);
  for (const Device device : Devices()) {
    CAPTURE(DeviceName(device));
    QueueHandle queue(device);
    vt::Backend& backend = vt::GetBackend(device.type);
    {
      DeviceScope scope(device);
      DeviceBf16 hidden(backend, queue.q, art.hidden_bf16, {art.rows, art.hidden});
      DeviceBf16 weight(backend, queue.q, TransposedWeight(art), {art.hidden, art.vocab});
      vllm::dense_attn::Dev d{backend, queue.q};
      vllm::dense_attn::DBuf logits = vllm::lm_head::Project(d, hidden.t, weight.t, /*tied=*/false);
      backend.Synchronize(queue.q);

      // The views and host copies downstream stay valid: F32, [rows, vocab].
      REQUIRE(logits.t().dtype == DType::kF32);
      REQUIRE(logits.t().rank == 2);
      REQUIRE(logits.t().shape[0] == art.rows);
      REQUIRE(logits.t().shape[1] == art.vocab);

      std::vector<float> got(primary.size());
      logits.Download(d, got.data());
      size_t differing = 0, first_diff = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != primary[i]) {
          if (differing == 0) first_diff = i;
          ++differing;
        }
      }
      double worst = 0.0;
      for (size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - primary[i]));
      std::cout << "[lm-head boundary] device=" << DeviceName(device)
                << " production projection vs primary: differing=" << differing << "/"
                << got.size() << " max_abs=" << worst;
      if (differing != 0)
        std::cout << " first_diff_index=" << first_diff << " got=" << got[first_diff]
                  << " primary=" << primary[first_diff];
      std::cout << std::endl;

      CHECK(differing == 0);
      for (int64_t row = 0; row < art.rows; ++row) {
        const size_t base = static_cast<size_t>(row * art.vocab);
        std::vector<float> row_values(got.begin() + static_cast<ptrdiff_t>(base),
                                      got.begin() + static_cast<ptrdiff_t>(base + art.vocab));
        const std::vector<int> order = TopOrder(row_values, art.vocab, 1);
        const std::vector<int> expected = TopOrder(
            std::vector<float>(primary.begin() + static_cast<ptrdiff_t>(base),
                               primary.begin() + static_cast<ptrdiff_t>(base + art.vocab)),
            art.vocab, 1);
        CAPTURE(row);
        CHECK(order[0] == expected[0]);
      }
    }
  }
}

// ─── The production entry point ─────────────────────────────────────────────

// A deterministic BF16 weight, so this case needs no capture and no checkpoint.
vllm::OwnedTensor SyntheticBf16(const std::vector<int64_t>& shape, uint64_t seed) {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  vllm::OwnedTensor t;
  t.dtype = DType::kBF16;
  t.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
  std::vector<uint8_t> bytes(static_cast<size_t>(n) * sizeof(uint16_t));
  auto* words = reinterpret_cast<uint16_t*>(bytes.data());
  uint64_t s = seed;
  for (int64_t i = 0; i < n; ++i) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    s ^= s >> 33;
    words[i] = vt::F32ToBF16(static_cast<float>((s >> 40) & 0xFFFF) / 32768.0f - 1.0f);
  }
  t.bytes = vllm::OwnedBytes(std::move(bytes));
  return t;
}

// THE CALL-SITE CASE, and the reason it uses a degenerate config. With ZERO
// decoder layers the production entry point reduces to embed -> final RMSNorm ->
// lm_head, which is the smallest shape in which `Qwen3MoeModel::Forward` reaches
// the head call site. The focused seam cases above cannot see a reverted call
// site (they call `lm_head::Project` themselves); this one can, and it needs no
// checkpoint, no GPU-only build and no fixture directory.
//
// WHAT IT ASSERTS is the boundary the primary defines, read off the returned
// values: every logit the production forward hands to the sampler must be a BF16
// word widened to F32, i.e. its low 16 mantissa bits are zero. Nothing else in
// this path is allowed to round the logits, so the property is the boundary's
// fingerprint rather than a coincidence of the fixture.
//
// THIS CASE IS UNDECORATED AND UNCONDITIONAL: it carries no `doctest::skip`, so
// it runs and is reported in the fixture-absent run too, where the two cases
// above skip. That is what makes it the gate a reverted call site has to redden.
TEST_CASE("qwen3 MoE LM-head BF16 boundary: the production forward returns BF16 logits") {
  constexpr int64_t kH = 128;
  constexpr int64_t kV = 128;
  for (const Device device : Devices()) {
    CAPTURE(DeviceName(device));
    QueueHandle queue(device);
    vllm::Qwen3MoeWeights weights;
    weights.tie_word_embeddings = false;
    weights.embed_tokens = SyntheticBf16({kV, kH}, 0x51ed01);
    weights.final_norm = SyntheticBf16({kH}, 0x51ed02);
    weights.lm_head = SyntheticBf16({kH, kV}, 0x51ed03);
    vllm::HfConfig config;
    config.model_type = "qwen3_moe";
    config.architectures = {"Qwen3MoeForCausalLM"};
    config.hidden_size = kH;
    config.vocab_size = kV;
    config.num_hidden_layers = 0;  // embed -> final RMSNorm -> lm_head
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = kH;
    config.intermediate_size = kH;
    config.moe_intermediate_size = kH;
    config.rms_norm_eps = 1e-6;
    config.rotary_dim = 0;

    const std::vector<float> logits = vllm::Qwen3MoeModel::Forward(
        /*token_ids=*/{7}, /*positions=*/{0}, vllm::v1::CommonAttentionMetadata{},
        /*attn_kv=*/{}, weights, config, queue.q);

    REQUIRE(logits.size() == static_cast<size_t>(kV));
    size_t not_bf16 = 0;
    bool any_nonzero = false;
    for (const float value : logits) {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      not_bf16 += (bits & 0xFFFFu) != 0;
      any_nonzero = any_nonzero || value != 0.0f;
    }
    std::cout << "[lm-head boundary] device=" << DeviceName(device)
              << " production forward: logits=" << logits.size()
              << " not_bf16_representable=" << not_bf16 << " any_nonzero=" << any_nonzero
              << std::endl;
    // The projection ran...
    CHECK(any_nonzero);
    // ...and its result is the primary's dtype, not a full F32 quotient.
    CHECK(not_bf16 == 0);
  }
}
