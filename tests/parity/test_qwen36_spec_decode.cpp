// vllm.cpp original (checkpoint-gated spec-decode gate); no upstream mirror.
//
// THE 35B k=1 MTP SPECULATIVE-DECODE THREE-WAY IDENTITY GATE (SPEC-MTP M-mtp-2,
// MODEL-SPEC-qwen3-5-mtp-qwen3-5-moe-mtp / arch Qwen3_5MoeMTP). Exact mirror of
// test_qwen27_spec_decode.cpp at 35B dims: it drives the SAME pinned M0-exit
// prompt as test_qwen36_paged_engine.cpp through the FULL paged engine, but with
// speculative decoding turned ON via EngineParams::speculative_config
// ('{"method":"mtp","num_speculative_tokens":1}'). The 35B draft head is the MoE
// MTP layer (Qwen3_5MTPKind::kMoe: 256 routed experts top-8 + shared expert),
// selected automatically for the MoE target by
// `src/vllm/entrypoints/model_loader.cpp::is_dense_model`; the GDN
// verify/rollback path is shared bit-exactly with the 27B (I4/I5a). Asserts:
//
//   (a) == (b) THREE-WAY IDENTITY. Greedy spec-decode is exactness-preserving, so
//       our spec-ON greedy continuation must equal `greedy_ids.npy` — which is
//       BOTH the vLLM-oracle greedy continuation (== vLLM's own
//       --speculative-config mtp greedy, exactness-preserving) AND our OWN
//       spec-OFF continuation (proven token-for-token by
//       test_qwen36_paged_engine.cpp). So our-ON == our-OFF == vLLM, position for
//       position, is established transitively through this one golden without a
//       second (OOM-risky on GB10 unified memory) 35B load.
//   (b) MEASURED NONZERO ACCEPTANCE. The verify/propose loop actually RAN and the
//       drafter is alive: runner().spec_drafts_accepted() > 0 (token-identity
//       alone passes on a dead drafter — the I5e lesson — so this is required
//       alongside identity, and the MoE MTP draft must actually accept tokens).
//
// Checkpoint-GATED + dgx-only: on the CPU dev box / CI the snapshot is absent, so
// the body emits a loud SKIP — the test compiles + links on CPU, runs only on
// dgx.casa (GB10) where the 35B snapshot + the CUDA spec kernels are present.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "npy.h"
#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

#include "hf_snapshot.h"

namespace fs = std::filesystem;

namespace {

// IDENTICAL resolution to test_qwen36_paged_engine.cpp's Find35BSnapshot — the
// HF cache layout for models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/<rev>/.
// GATE-PIN-UNPINNED-SNAPSHOTS (#471). This used to take the first
// `directory_iterator` entry under `<repo>/snapshots/`. The 35B goldens name the
// revision they were captured against (`oracle.model` of
// goldens/qwen36_*_35b/manifest.json), so there was never a reason not to
// enforce it. Now pinned; a cache holding another revision skips.
std::string Find35BSnapshot() { return parity::Qwen36A3bNvfP4Snapshot(); }

std::vector<int32_t> LoadI32Npy(const fs::path& p) {
  const parity::NpyArray a = parity::LoadNpy(p.string());
  REQUIRE(a.dtype == "<i4");
  const size_t n = a.data.size() / sizeof(int32_t);
  const auto* src = reinterpret_cast<const int32_t*>(a.data.data());
  return std::vector<int32_t>(src, src + n);
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

}  // namespace

TEST_CASE("qwen36 spec-decode three-way identity gate (dgx-only, 35B MoE MTP k=1)") {
  const std::string snap = Find35BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "35B checkpoint absent; skipping (dgx-only) — "
        "nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot not present");
    return;
  }

  const std::string kPrompt = "The capital of France is Paris, and the";
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_logits_35b";
  const std::vector<int32_t> want_prompt_ids =
      LoadI32Npy(golden / "token_ids.npy");
  // greedy_ids.npy: the vLLM-oracle greedy continuation == our spec-OFF
  // continuation (test_qwen36_paged_engine.cpp) == vLLM spec-ON greedy.
  const std::vector<int32_t> want_prod = LoadI32Npy(golden / "greedy_ids.npy");
  const int kGoldenTokens = static_cast<int>(want_prod.size());  // 16
  // Decode a few extra tokens so the acceptance measurement has more than the
  // 16 golden steps to observe (a short prose prompt accepts ~2/15, B5 §5); the
  // three-way identity is checked on the golden-length prefix only.
  const int kMaxTokens = kGoldenTokens + 16;

  // Turn speculative decoding ON: method mtp, k defaults to n_predict (=1).
  vllm::entrypoints::EngineParams params;
  params.speculative_config = vllm::ParseSpeculativeConfigJson(
      "{\"method\":\"mtp\",\"num_speculative_tokens\":1}");

  MESSAGE("qwen36_spec_decode: loading 35B via FromModelDir(" << snap
          << ") with speculative-config mtp k=1...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(snap, params);

  loaded->engine().add_request("spec", kPrompt, Greedy(kMaxTokens));
  std::optional<vllm::RequestOutput> final;
  auto consume = [&](std::vector<vllm::RequestOutput> batch) {
    for (vllm::RequestOutput& item : batch)
      if (item.finished) final = std::move(item);
  };
  while (loaded->engine().has_unfinished_requests())
    consume(loaded->engine().step());
  if (!final.has_value())
    throw std::runtime_error("qwen36 spec-decode produced no final output");
  const vllm::RequestOutput& out = *final;

  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& got = out.outputs[0].token_ids;
  CHECK(out.prompt_token_ids == want_prompt_ids);

  const int64_t proposed = loaded->runner().spec_drafts_proposed();
  const int64_t accepted = loaded->runner().spec_drafts_accepted();
  MESSAGE("qwen36_spec_decode: produced " << got.size() << " tokens; drafts "
          << accepted << "/" << proposed << " accepted; continuation=\""
          << out.outputs[0].text << "\"");

  // (a)==(b): three-way token identity on the golden-length prefix.
  REQUIRE(static_cast<int>(got.size()) >= kGoldenTokens);
  std::vector<int32_t> got_prefix(got.begin(), got.begin() + kGoldenTokens);
  CHECK(got_prefix == want_prod);

  // Nonzero acceptance: the loop RAN and the MoE MTP drafter is alive (identity
  // alone would pass on a dead drafter, so this is a REQUIRED second condition).
  CHECK(proposed > 0);
  CHECK(accepted > 0);
}
