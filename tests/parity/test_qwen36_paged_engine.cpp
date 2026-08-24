// vllm.cpp original (checkpoint-gated acceptance gate); no upstream mirror.
//
// THE PAGED-ENGINE 35B GREEDY ACCEPTANCE GATE — the counterpart to
// test_op_parity.cpp's RunQwen36Logits (which validates the DENSE ForwardDense
// path). This one drives the SAME M0-exit prompt through the FULL PAGED
// LLMEngine stack (InputProcessor -> Scheduler -> paged attention + KV-cache
// growth + batched GDN + Sampler -> OutputProcessor) via
// LoadedEngine::FromModelDir, and asserts the greedy (temperature-0) decode
// reproduces the pinned-oracle M0-exit continuation TOKEN-FOR-TOKEN.
//
// Checkpoint-GATED + dgx-only, mirroring RunQwen36Logits exactly: it resolves
// the real nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot under
// ~/.cache/huggingface/hub/... (same resolution as Find35BSnapshot in
// test_op_parity.cpp). On the CPU dev box / CI the snapshot is absent, so the
// body emits a loud SKIP MESSAGE and returns — the test compiles + links on CPU
// but only RUNS on dgx.casa (GB10), where scripts/dgx-bringup.sh invokes it.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef VLLM_CPP_CUDA
#include <cuda_runtime_api.h>
#include "vt/cuda/cuda_gdn_internal.h"
#endif

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/sampling_params.h"

#include "hf_snapshot.h"

namespace fs = std::filesystem;

namespace {

void CheckDeviceCacheResidency(
    const vllm::entrypoints::LoadedEngine& loaded) {
#ifdef VLLM_CPP_CUDA
  const char* fallback = std::getenv("VT_DEVICE_KV_CACHE");
  if (fallback != nullptr && fallback[0] == '0') {
    CHECK_FALSE(loaded.runner().kv_cache_backend_resident());
    return;
  }
  REQUIRE(loaded.runner().kv_cache_backend_resident());
  const auto check_device_pointer = [](const void* pointer) {
    cudaPointerAttributes attributes{};
    REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
    CHECK(attributes.type == cudaMemoryTypeDevice);
  };
  for (const vllm::PagedKvCache& cache : loaded.runner().attn_kv()) {
    check_device_pointer(cache.data);
  }
  for (const vllm::GdnStateCache& cache : loaded.runner().gdn_state()) {
    if (std::getenv("VT_GDN_STATE_BF16") == nullptr &&
        (cache.ssm_state.dtype != vt::DType::kF32 ||
         cache.conv_state.dtype != vt::DType::kBF16)) {
      throw std::runtime_error(
          "qwen36 default GDN cache ABI must be FP32 SSM + BF16 conv");
    }
    check_device_pointer(cache.ssm_state.data);
    check_device_pointer(cache.conv_state.data);
  }
#else
  CHECK_FALSE(loaded.runner().kv_cache_backend_resident());
#endif
}

// Snapshot dir of the pinned 35B checkpoint (contains config.json), or "".
// IDENTICAL resolution to test_op_parity.cpp's Find35BSnapshot — the HF cache
// layout for models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/<rev>/.
// GATE-PIN-UNPINNED-SNAPSHOTS (#471). This used to take the first
// `directory_iterator` entry under `<repo>/snapshots/`. The 35B goldens name the
// revision they were captured against (`oracle.model` of
// goldens/qwen36_*_35b/manifest.json), so there was never a reason not to
// enforce it. Now pinned; a cache holding another revision skips.
std::string Find35BSnapshot() { return parity::Qwen36A3bNvfP4Snapshot(); }

// Load an i32 (.npy "<i4") vector from the committed golden.
std::vector<int32_t> LoadI32Npy(const fs::path& p) {
  const parity::NpyArray a = parity::LoadNpy(p.string());
  REQUIRE(a.dtype == "<i4");
  const size_t n = a.data.size() / sizeof(int32_t);
  const auto* src = reinterpret_cast<const int32_t*>(a.data.data());
  return std::vector<int32_t>(src, src + n);
}

// Greedy (argmax) sampling params — temperature 0 => deterministic, matching
// the oracle's temperature-0 continuation. PostInit normalizes/validates as the
// engine's InputProcessor would.
vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// The gate's packed-decode expectation as a PURE function of env-shaped
// inputs, so the CPU tier can pin its truth table without the checkpoint (the
// gate bodies below only RUN where the 35B snapshot resolves). The gate call
// site hands it the live getenv values; the truth-table case below hands it
// literals.
bool DerivePackedExpected(const char* packed_decode_env,
                          const char* fp8_tower_env,
                          const char* fp8_in_bf16_env, bool fp8_merged) {
  // VT_GDN_PACKED_DECODE, production's FIRST eligibility term. Mirror
  // PackedGdnDecodeRuntimeEnabled (qwen3_5.cpp:3589-3595) exactly: enabled
  // unless the value's first char is '0' (unset or anything else enables).
  // Parsed INLINE because no exported pure parser carries just this term —
  // detail::PackedGdnDecodeEnvSelected conjoins five more, so reusing it here
  // would silently fold terms this derivation deliberately does not read.
  const bool runtime_enabled =
      packed_decode_env == nullptr || packed_decode_env[0] != '0';
  const bool fp8_tower_lever =
      vllm::detail::PackedGdnDecodeFp8TowerFlagIsOn(fp8_tower_env);
  const bool fp8_in_bf16 =
      fp8_in_bf16_env != nullptr && fp8_in_bf16_env[0] == '1';
  return runtime_enabled && fp8_tower_lever && fp8_merged &&
         vllm::detail::GdnFp8MergedMixedQkvDType(fp8_in_bf16, vt::DType::kBF16,
                                                 vt::DType::kBF16) ==
             vt::DType::kBF16;
}

}  // namespace

// The M0-exit prompt (pinned oracle: qwen36_logits_35b/manifest.json) and its
// greedy continuation. RunQwen36Logits proved the DENSE forward reproduces
// greedy_ids token-for-token (16/16); this proves the PAGED engine does too,
// end to end from the prompt STRING through the batched serving loop.
TEST_CASE("qwen36 paged-engine greedy acceptance gate (dgx-only, 35B)") {
  const std::string snap = Find35BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "35B checkpoint absent; skipping (dgx-only) — "
        "nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot not present");
    return;
  }

  // The pinned M0-exit prompt + its oracle greedy continuation (16 tokens). The
  // golden lives beside the RunQwen36Logits inputs; prompt_ids is the
  // tokenization the oracle used (diagnostic cross-check on the engine's own
  // tokenization), greedy_ids is the temperature-0 continuation to reproduce.
  const std::string kPrompt = "The capital of France is Paris, and the";
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_logits_35b";
  const std::vector<int32_t> want_prompt_ids =
      LoadI32Npy(golden / "token_ids.npy");
  const std::vector<int32_t> want_greedy_ids =
      LoadI32Npy(golden / "greedy_ids.npy");
  const int kMaxTokens = static_cast<int>(want_greedy_ids.size());  // 16

  MESSAGE("qwen36_paged_engine: loading full 35B via FromModelDir("
          << snap << ") — dequant + transpose + engine stack...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});
  CheckDeviceCacheResidency(*loaded);

  MESSAGE("qwen36_paged_engine: greedy-decoding "
          << kMaxTokens << " tokens through the PAGED engine...");
#ifdef VLLM_CPP_CUDA
  vt::cuda::testing::ResetGdnPackedDecodeDebugStats();
#endif
  const vllm::RequestOutput out =
      loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "gate");
  // GDN-MOE-PACKED-BA (#1169): the MoE loader now builds the merged
  // `in_proj_ba` owner, so packed GDN decode is REACHABLE on this checkpoint.
  // The previous block here pinned the pre-#1169 contract — any nonzero
  // packed_launches threw "dense-only" — which made the row's GPU gate (b)
  // (.agents/specs/gdn-moe-packed-ba.md: this test under
  // VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1, counter REQUIRED
  // nonzero) structurally impossible to pass. Derive the expectation from the
  // SAME terms `ShouldUsePackedGdnDecode` evaluates instead, exactly as
  // tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp does — and, like
  // there, deliberately NOT through `detail::PackedGdnDecodeEnvSelected`,
  // which mirrors the ENV couplings only and knows nothing about the weight
  // dtype (#470).
  //
  // On the spec's three gate arms over this checkpoint
  // (nvidia/Qwen3.6-35B-A3B-NVFP4, native FP8 GDN tower on every arm),
  // eligibility reduces to: the runtime lever (VT_GDN_PACKED_DECODE,
  // production's FIRST eligibility term), the #365 fp8-tower relaxation
  // lever, the merged fp8 qkvz arm (the split arm hardcodes F32, which the
  // activation-dtype rule rejects), and the merged arm's predicted
  // `mixed_qkv` dtype through the shared bridge helper. Spec gate (a) — the
  // default arm, no levers — expects 0 by the #365 fp8-tower term; gate (b)
  // — both levers on — expects the packed leg dispatched; and gate (b)'s
  // ROLLBACK sub-arm — the same levers plus VT_GDN_PACKED_DECODE=0 —
  // expects 0 again by the runtime term.
  //
  // WHAT THIS DERIVATION CANNOT SEE, deliberately. It does NOT derive the
  // merged-BA env couplings — the VT_GDN_MERGED_BA leaf, and
  // VT_GDN_MERGED_PROJ's master role beyond the slice of it `fp8_merged`
  // reads — nor the VT_GDN_IN_BF16 / VT_GDN_OUT_BF16 / VT_GDN_BA_OUT_BF16
  // dtype overrides (the vt::DType::kBF16 arguments below are HARDCODED
  // defaults, not parses), nor `ShouldUseMergedGdnFp8Qkvz`'s
  // checkpoint-shape terms (`shared_k`, `shared_input_scale`,
  // `shard_widths_match`). Setting any of those rollbacks while running this
  // gate makes the derivation wrong. It is valid ONLY on the spec's three
  // named arms over an FP8-tower checkpoint whose merged shards satisfy the
  // shape terms — which the revision-pinned snapshot this gate loads
  // guarantees.
  //
  // ONE deliberate difference from the 27n template: that harness steps
  // EAGERLY around a single decode step and can assert an EXACT host-dispatch
  // count. This gate calls generate() end to end, where CUDA-graph capture
  // and replay make an exact count fragile: replay performs no host dispatch,
  // and capture dispatches once. So assert the SIGN of the counter, not the
  // count.
  //
  // Hoisted outside the VLLM_CPP_CUDA block on purpose: these predicates are
  // host code, and the CPU build must keep type-checking this derivation.
  [[maybe_unused]] const char* packed_decode_env =
      std::getenv("VT_GDN_PACKED_DECODE");
  [[maybe_unused]] const bool fp8_tower_lever =
      vllm::detail::PackedGdnDecodeFp8TowerFlagIsOn(
          std::getenv("VT_GDN_PACKED_DECODE_FP8_TOWER"));
  const char* fp8_in_bf16_env = std::getenv("VT_GDN_FP8_IN_BF16");
  [[maybe_unused]] const bool fp8_in_bf16 =
      fp8_in_bf16_env != nullptr && fp8_in_bf16_env[0] == '1';
  [[maybe_unused]] const bool fp8_merged =
      vllm::detail::MergedGdnFp8QkvzEnvSelected(
          vllm::detail::GdnMergedFp8QkvzEnvConfig{
              std::getenv("VT_GDN_MERGED_PROJ"),
              std::getenv("VT_GDN_MERGED_QKVZ"),
              std::getenv("VT_GDN_MERGED_QKVZ_FP8")});
  [[maybe_unused]] const bool packed_expected = DerivePackedExpected(
      packed_decode_env, std::getenv("VT_GDN_PACKED_DECODE_FP8_TOWER"),
      fp8_in_bf16_env, fp8_merged);
#ifdef VLLM_CPP_CUDA
  const vt::cuda::testing::GdnPackedDecodeDebugStats packed_stats =
      vt::cuda::testing::GetGdnPackedDecodeDebugStats();
  const uint64_t packed_launches = packed_stats.launches;
  vt::cuda::testing::DisableGdnPackedDecodeDebugStats();
  // INTEGER rendering, deliberately — see the 27n gate's warning: doctest's
  // MESSAGE prints a string-literal ternary's FIRST branch unconditionally,
  // so `(on ? "1" : "0")` would mislabel the arm that ran.
  MESSAGE("qwen36 GDN packed-decode SELECTION (generate(), host dispatch): "
          << "packed_launches=" << packed_launches
          << " triton_launches=" << packed_stats.triton_launches
          << " packed_expected=" << (packed_expected ? 1 : 0)
          << " VT_GDN_PACKED_DECODE="
          << ((packed_decode_env == nullptr || packed_decode_env[0] != '0')
                  ? 1
                  : 0)
          << " VT_GDN_PACKED_DECODE_FP8_TOWER=" << (fp8_tower_lever ? 1 : 0)
          << " VT_GDN_FP8_IN_BF16=" << (fp8_in_bf16 ? 1 : 0));
  if (!packed_expected) {
    CHECK(packed_launches == 0);
    if (packed_launches != 0)
      throw std::runtime_error(
          "qwen36: packed GDN decode selected on an arm whose predicate "
          "implies it is off");
  } else {
    CHECK(packed_launches > 0);
    if (packed_launches == 0)
      throw std::runtime_error(
          "qwen36: the arm's predicate implies packed GDN decode and nothing "
          "dispatched it");
  }
#endif

  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& got = out.outputs[0].token_ids;

  // Diagnostic: the engine's tokenization of the prompt string must match the
  // oracle's prompt_ids — otherwise a greedy divergence would be a tokenizer
  // mismatch, not a forward-pass regression. (prompt_token_ids carries the
  // engine-tokenized prompt.)
  CHECK(out.prompt_token_ids == want_prompt_ids);

  MESSAGE("qwen36_paged_engine M0-EXIT: produced " << got.size() << "/"
          << kMaxTokens << " tokens; continuation=\""
          << out.outputs[0].text << "\"");

  // THE M0 EXIT BAR (paged engine): the batched serving loop reproduces the
  // oracle's greedy continuation EXACTLY, token-for-token. Greedy is
  // deterministic and was verified at M0 exit (dense: 16/16); this is the
  // paged-engine counterpart and MUST hold.
  REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
  CHECK(got == want_greedy_ids);
}

// BATCHED decode CUDA-graph acceptance gate (num_reqs>1). Submits N identical
// greedy requests so the engine, once all are past prefill, runs PURE-DECODE
// steps with num_reqs==N through the batched Qwen3_5DecodeGraph (padded up to the
// nearest captured size — N=6 pads to 8, exercising 2 INERT padding rows). Each
// request is independent (own KV + GDN state), so each MUST reproduce the same
// oracle greedy continuation token-for-token; a corrupted real row (padding
// leaking into a real slot) or a bad batched replay would diverge. This is the
// conc>1 counterpart to the num_reqs==1 gate above.
TEST_CASE("qwen36 paged-engine batched-graph greedy gate (dgx-only, 35B)") {
  const std::string snap = Find35BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "35B checkpoint absent; skipping (dgx-only) — "
        "nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot not present");
    return;
  }

  const std::string kPrompt = "The capital of France is Paris, and the";
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_logits_35b";
  const std::vector<int32_t> want_greedy_ids =
      LoadI32Npy(golden / "greedy_ids.npy");
  const int kMaxTokens = static_cast<int>(want_greedy_ids.size());  // 16
  const int kN = 6;  // pure-decode num_reqs==6 -> padded to captured size 8

  MESSAGE("qwen36_paged_engine(batched): loading 35B via FromModelDir...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});
  CheckDeviceCacheResidency(*loaded);

  MESSAGE("qwen36_paged_engine(batched): greedy-decoding " << kN
          << " concurrent requests x " << kMaxTokens << " tokens...");
  for (int i = 0; i < kN; ++i)
    loaded->engine().add_request("r" + std::to_string(i), kPrompt,
                                 Greedy(kMaxTokens));

  std::map<std::string, vllm::RequestOutput> finished;
  while (loaded->engine().has_unfinished_requests()) {
    for (vllm::RequestOutput& out : loaded->engine().step())
      if (out.finished) finished[out.request_id] = std::move(out);
  }

  // Every concurrent request reproduces the oracle continuation exactly.
  REQUIRE(static_cast<int>(finished.size()) == kN);
  for (int i = 0; i < kN; ++i) {
    const std::string id = "r" + std::to_string(i);
    REQUIRE(finished.count(id) == 1);
    REQUIRE(finished[id].outputs.size() == 1);
    const std::vector<int32_t>& g = finished[id].outputs[0].token_ids;
    REQUIRE(static_cast<int>(g.size()) == kMaxTokens);
    CHECK(g == want_greedy_ids);
  }
}

// CPU-TIER TRUTH TABLE for DerivePackedExpected. The two gates above are
// checkpoint-gated (dgx-only), so on the CPU tier this file used to only
// type-check the derivation; this case EXECUTES it, pinning the expectation
// for each of the spec's gate arms (.agents/specs/gdn-moe-packed-ba.md,
// gates (a) and (b)) as a pure function of the env values those arms set.
TEST_CASE("qwen36 packed-decode expectation env truth table (CPU tier)") {
  // Spec gate (b): both #365 levers on, runtime lever unset -> packed.
  CHECK(DerivePackedExpected(nullptr, "1", "1", true));
  // Spec gate (b) ROLLBACK sub-arm: the same two levers plus
  // VT_GDN_PACKED_DECODE=0 -> NOT packed. Production's FIRST eligibility
  // term is PackedGdnDecodeRuntimeEnabled (qwen3_5.cpp:3589-3595), so a
  // correct binary dispatches 0 on this arm and the derivation must agree.
  CHECK_FALSE(DerivePackedExpected("0", "1", "1", true));
  // Runtime lever explicitly on, #365 levers off -> NOT packed (spec gate
  // (a), the default arm: the fp8-tower term keeps the fp8 tower off the
  // packed leg, and the predicted mixed_qkv dtype is F32 without
  // VT_GDN_FP8_IN_BF16=1).
  CHECK_FALSE(DerivePackedExpected("1", nullptr, nullptr, true));
  // Merged fp8 qkvz deselected -> NOT packed, whatever the levers say (the
  // split arm hardcodes F32 mixed_qkv).
  CHECK_FALSE(DerivePackedExpected(nullptr, "1", "1", false));
}
