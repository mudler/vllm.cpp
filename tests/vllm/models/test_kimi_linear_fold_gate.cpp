// ROW 7 / kimi-linear.md §20.3 — the GB10 fold-gate REFERENCE leg.
//
// The CLI-incremental battery (bf16-resident load + ForwardPrefillIncremental /
// ForwardDecodeStepIncremental greedy over the §12 golden prompts) that
// `examples/kimi_linear_gen` used to run through internal headers. The example
// is now a thin public-ABI client (ONE SURFACE B4), so the REFERENCE leg — the
// stream the engine-paged path must reproduce — lives here, where internal
// headers are legitimate. ENV-GATED: skipped unless VT_KIMI_MODEL_DIR points at
// the real 48.9B snapshot (a 91.5 GiB load has no place in the CPU suite);
// VT_KIMI_GOLDEN_DIR selects the §12 golden and is REQUIRED whenever the model
// gate is enabled.  This executable is only the CLI-incremental REFERENCE leg;
// it does not stand in for the still-owed engine/C-ABI golden or HTTP server
// smoke on the 91.5 GiB checkpoint.
//
// GB10 usage (the §19-winning config is the reference regime): with
//   VT_KIMI_DEVICE_COMPUTE=1 VT_KIMI_DEVICE_KDA=1 VT_KIMI_DEVICE_KDA_CHUNK=1
//   VT_KIMI_MODEL_DIR=... VT_KIMI_GOLDEN_DIR=... VT_KIMI_STEPS=16
// run ./test_kimi_linear_fold_gate
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kimi_linear.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace fs = std::filesystem;

namespace {

std::vector<vllm::SafetensorsFile> OpenSafetensorsDir(const std::string& dir) {
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.is_regular_file() && e.path().extension() == ".safetensors")
      paths.push_back(e.path().string());
  REQUIRE(!paths.empty());
  std::sort(paths.begin(), paths.end());
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(paths.size());
  for (const std::string& p : paths) shards.push_back(vllm::SafetensorsFile::Open(p));
  return shards;
}

std::vector<int32_t> ReadI32File(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<int32_t> out;
  if (!f) return out;
  int32_t v = 0;
  while (f.read(reinterpret_cast<char*>(&v), 4)) out.push_back(v);
  return out;
}

struct NpyInts {
  std::vector<int64_t> shape;
  std::vector<int64_t> data;
};
NpyInts ReadNpyInts(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE(static_cast<bool>(f));
  char magic[6];
  f.read(magic, 6);
  REQUIRE(std::memcmp(magic, "\x93NUMPY", 6) == 0);
  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint16_t hlen = 0;
  f.read(reinterpret_cast<char*>(&hlen), 2);
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);
  const bool i8 = header.find("<i8") != std::string::npos;
  const bool i4 = header.find("<i4") != std::string::npos;
  REQUIRE((i8 || i4));
  NpyInts out;
  const size_t sp = header.find("'shape':");
  const size_t lp = header.find('(', sp);
  const size_t rp = header.find(')', lp);
  std::string dims = header.substr(lp + 1, rp - lp - 1);
  for (size_t i = 0; i < dims.size();) {
    while (i < dims.size() && (dims[i] == ' ' || dims[i] == ',')) ++i;
    if (i >= dims.size()) break;
    size_t j = i;
    while (j < dims.size() && dims[j] >= '0' && dims[j] <= '9') ++j;
    if (j > i) out.shape.push_back(std::atoll(dims.substr(i, j - i).c_str()));
    i = j + 1;
  }
  int64_t n = 1;
  for (int64_t d : out.shape) n *= d;
  out.data.resize(static_cast<size_t>(n));
  for (int64_t k = 0; k < n; ++k) {
    if (i8) {
      int64_t v = 0;
      f.read(reinterpret_cast<char*>(&v), 8);
      out.data[static_cast<size_t>(k)] = v;
    } else {
      int32_t v = 0;
      f.read(reinterpret_cast<char*>(&v), 4);
      out.data[static_cast<size_t>(k)] = v;
    }
  }
  return out;
}

}  // namespace

// #2840 — WHAT LICENSES THE BAR, made executable.
//
// `.agents/specs/kimi-linear.md` §589-591 records "ALL 8 prompts DETERMINISTIC
// over K=3 -> STRICT token-exact gate" in prose. The capture that supports it is
// committed and machine-readable, and nothing read it. This case does, with no
// checkpoint and no GPU, so the bar the case below applies is selected by a
// measurement rather than by a sentence.
//
// `greedy_dist.npy` is [N,T,K]: K independent greedy runs of the pinned oracle.
// A cell holding more than one id is a cell the oracle cannot reproduce against
// itself. Zero such cells is what `CLAUDE.md` §Gates requires before STRICT is
// the bar, and it is equally what DENIES a distributional bar: that is licensed
// "only when the oracle's greedy decode is non-deterministic".
TEST_CASE("kimi-linear committed golden: the oracle is deterministic over K=3, so the bar is STRICT") {
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / "kimi_linear_greedy";
  REQUIRE_MESSAGE(fs::exists(gdir / "greedy_dist.npy"),
                  "kimi-linear: the committed K-run capture is what selects the "
                  "bar; without it neither STRICT nor a band is licensed");
  const NpyInts dist = ReadNpyInts((gdir / "greedy_dist.npy").string());
  REQUIRE(dist.shape.size() == 3);
  const int64_t N = dist.shape[0], T = dist.shape[1], K = dist.shape[2];
  REQUIRE(K > 1);

  int64_t multi_cells = 0;
  for (int64_t i = 0; i < N; ++i)
    for (int64_t j = 0; j < T; ++j) {
      const int64_t base = (i * T + j) * K;
      for (int64_t k = 1; k < K; ++k)
        if (dist.data[static_cast<size_t>(base + k)] !=
            dist.data[static_cast<size_t>(base)]) {
          ++multi_cells;
          break;
        }
    }
  std::fprintf(stderr,
               "[fold-gate] oracle self-determinism over K=%lld runs: "
               "%lld multi-valued (prompt,pos) cells of %lld\n",
               static_cast<long long>(K), static_cast<long long>(multi_cells),
               static_cast<long long>(N * T));
  CHECK_MESSAGE(multi_cells == 0,
                "kimi-linear: the oracle is NOT self-deterministic on this "
                "capture; re-derive the bar, do not loosen it");

  // The capture must be a capture OF the committed golden. Without this the
  // determinism count is a property of an array rather than of the oracle whose
  // ids the gate below compares against.
  const NpyInts gold = ReadNpyInts((gdir / "greedy_ids.npy").string());
  REQUIRE(gold.shape.size() == 2);
  REQUIRE(gold.shape[0] == N);
  REQUIRE(gold.shape[1] == T);
  int64_t run0_mismatch = 0;
  for (int64_t i = 0; i < N; ++i)
    for (int64_t j = 0; j < T; ++j)
      if (dist.data[static_cast<size_t>((i * T + j) * K)] !=
          gold.data[static_cast<size_t>(i * T + j)])
        ++run0_mismatch;
  CHECK_MESSAGE(run0_mismatch == 0,
                "kimi-linear: the K-run capture's first run does not reproduce "
                "greedy_ids.npy, so the two artifacts describe different runs");
}

TEST_CASE("kimi fold-gate environment enables model and golden together") {
  const bool have_model = std::getenv("VT_KIMI_MODEL_DIR") != nullptr;
  const bool have_golden = std::getenv("VT_KIMI_GOLDEN_DIR") != nullptr;
  INFO("the 91.5 GiB gate needs both VT_KIMI_MODEL_DIR and VT_KIMI_GOLDEN_DIR");
  CHECK(have_model == have_golden);
}

TEST_CASE("kimi fold-gate reference: CLI-incremental battery on the real checkpoint" *
          doctest::skip(std::getenv("VT_KIMI_MODEL_DIR") == nullptr)) {
  const char* model_dir = std::getenv("VT_KIMI_MODEL_DIR");
  REQUIRE(model_dir != nullptr);
  const char* golden_dir = std::getenv("VT_KIMI_GOLDEN_DIR");
  const char* steps_env = std::getenv("VT_KIMI_STEPS");
  const char* prompts_env = std::getenv("VT_KIMI_PROMPTS");
  const int steps = steps_env != nullptr ? std::atoi(steps_env) : 16;
  const int prompts = prompts_env != nullptr ? std::atoi(prompts_env) : 8;
  REQUIRE(golden_dir != nullptr);
  REQUIRE(steps > 1);
  REQUIRE(prompts > 0);

  // CUDA when available (the GB10 leg; context BEFORE weights — the §13 recipe),
  // CPU otherwise.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  bool cuda = false;
  try {
    q = vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue();
    cuda = true;
  } catch (...) {
  }
  std::fprintf(stderr, "[fold-gate] device=%s\n", cuda ? "CUDA" : "CPU");

  const vllm::HfConfig config =
      vllm::LoadHfConfig((fs::path(model_dir) / "config.json").string());
  std::vector<vllm::SafetensorsFile> shards = OpenSafetensorsDir(model_dir);
  const auto t0 = std::chrono::steady_clock::now();
  vllm::KimiLinearWeights w =
      vllm::LoadKimiLinearResidentBf16Weights(shards, config, cuda ? &q : nullptr);
  shards.clear();
  shards.shrink_to_fit();  // release the mmap'd shards (§13)
  const auto t1 = std::chrono::steady_clock::now();
  std::fprintf(stderr, "[fold-gate] loaded bf16-resident in %.1fs\n",
               std::chrono::duration<double>(t1 - t0).count());

  const NpyInts gold =
      ReadNpyInts((fs::path(golden_dir) / "greedy_ids.npy").string());
  REQUIRE(gold.shape.size() == 2);
  REQUIRE(gold.shape[0] >= prompts);
  REQUIRE(gold.shape[1] >= steps);

  vt::Backend& be = vt::GetBackend(q.device.type);
  const int64_t V = w.params.vocab_size;
  auto argmax_row = [&](const vllm::ForwardLogits& fl) {
    std::vector<float> row(static_cast<size_t>(V));
    be.Copy(q, row.data(), fl.device_tensor.data, row.size() * sizeof(float));
    be.Synchronize(q);
    int best = 0;
    float bv = row[0];
    for (int64_t o = 1; o < V; ++o)
      if (row[static_cast<size_t>(o)] > bv) {
        bv = row[static_cast<size_t>(o)];
        best = static_cast<int>(o);
      }
    return best;
  };

  int total = 0, matched = 0;
  int processed_prompts = 0;
  int steady_steps = 0;
  double steady_s = 0.0;
  for (int pi = 0; pi < prompts; ++pi) {
    std::vector<int32_t> prompt;
    prompt = ReadI32File(
        (fs::path(golden_dir) / ("p" + std::to_string(pi) + "_prompt.i32")).string());
    REQUIRE_FALSE(prompt.empty());
    ++processed_prompts;

    vllm::KimiDecodeCache cache;
    std::vector<int32_t> positions(prompt.size());
    for (size_t t = 0; t < prompt.size(); ++t) positions[t] = static_cast<int32_t>(t);
    const std::vector<int32_t> li = {static_cast<int32_t>(prompt.size() - 1)};
    vllm::ForwardLogits fl =
        vllm::KimiLinearModel::ForwardPrefillIncremental(prompt, positions, w, q, cache, li);
    std::vector<int32_t> gen;
    int best = argmax_row(fl);
    gen.push_back(best);
    for (int s = 1; s < steps; ++s) {
      const auto ts = std::chrono::steady_clock::now();
      vllm::ForwardLogits d =
          vllm::KimiLinearModel::ForwardDecodeStepIncremental(best, cache.seq_len, w, q, cache);
      best = argmax_row(d);
      const auto te = std::chrono::steady_clock::now();
      steady_s += std::chrono::duration<double>(te - ts).count();
      ++steady_steps;
      gen.push_back(best);
    }

    std::string got;
    for (int t = 0; t < steps; ++t)
      got += std::to_string(gen[static_cast<size_t>(t)]) + (t + 1 < steps ? "," : "");
    const int64_t T = gold.shape[1];
    int row_match = 0;
    for (int t = 0; t < steps; ++t) {
      ++total;
      if (gen[static_cast<size_t>(t)] ==
          static_cast<int32_t>(gold.data[static_cast<size_t>(pi) * T + t])) {
        ++matched;
        ++row_match;
      }
    }
    std::fprintf(stderr, "[fold-gate] p%d: %d/%d vs golden | got: %s\n", pi,
                 row_match, steps, got.c_str());
  }
  std::fprintf(stderr, "[fold-gate] TOKEN MATCH %d/%d\n", matched, total);
  std::fprintf(stderr, "[fold-gate] steady %.3f s/step (%.2f tok/s) over %d steps\n",
               steady_s / steady_steps, steady_steps / steady_s, steady_steps);

  REQUIRE(processed_prompts == prompts);
  REQUIRE(total == prompts * steps);
  REQUIRE(steady_steps == prompts * (steps - 1));
  REQUIRE(steady_s > 0.0);
  // #2840 — THE BAR IS EXACTNESS, and it used to be a floor.
  //
  // What stood here was `matched * 128 >= total * 122`, which admits six wrong
  // tokens in a 128-token battery. The case above measures the oracle
  // reproducing itself exactly over K=3, and `CLAUDE.md` §Gates licenses a
  // distributional bar "only when the oracle's greedy decode is
  // non-deterministic". So 122/128 is a FAILING STRICT GATE, which is what this
  // row's own spec has called it all along (`kimi-linear.md` §1295-1296, §1354),
  // and not an "intrinsic near-tie profile".
  //
  // Exactness scales to a deliberate diagnostic subset without an inequality,
  // which is what the floor was reaching for. The two admissible exits are
  // 128/128, or an explicit ratification of a distributional bar over an oracle
  // that reproduces itself. Do not tune this number.
  CHECK_MESSAGE(matched == total,
                "kimi-linear vs the committed golden: " << matched << "/" << total
                << " tokens. The oracle is K=3 self-deterministic, so STRICT is "
                   "the licensed bar and any mismatch fails it. Repair the "
                   "forward or ratify a different bar; do not widen this.");
}
