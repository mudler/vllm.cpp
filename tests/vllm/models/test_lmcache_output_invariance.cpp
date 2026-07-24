// vllm.cpp original (KV-EXTERNAL-CACHE — the LMCache connector-ON full-model
// OUTPUT-INVARIANCE + throughput gate; spec gate 4/6 of
// .agents/specs/lmcache-cpp-client-connector.md).
//
// THE PROOF THIS TEST CARRIES: the LMCacheConnector, configured and turned ON
// and pointed at a LIVE `lmcache.v1.server` (lm:// CPU remote-store, headless
// per the W2 recipe), works INSIDE A REAL GENERATION LOOP — not synthetic KV.
// A prompt whose prefix was previously STORED to the server LOADS that KV mid-
// generation and SHORTCUTS prefill, and the generated token sequence is
// BIT-IDENTICAL to the same run with the connector OFF (cold, full prefill).
// This is the gap-0 correctness requirement: loading a prefix's KV from LMCache
// must produce byte-identical decoding to computing it. A cache hit may change
// timing, never a token.
//
// The engine here is the SAME LoadedEngine + GPUModelRunner the SACRED gates
// use. The connector is selected exactly as a user would: an EngineParams
// KVTransferConfig{kv_connector="LMCacheConnector", host, port}. LoadedEngine
// builds the connector via KVConnectorFactory, injects the runner's resolved
// full-attention KV geometry, and wires it to BOTH the scheduler (prefix lookup
// / prefill shortcut, get_num_new_matched_tokens) and the worker (the runner's
// ConnectorLoadExternalKv / ConnectorStorePromptKv, which move real KV bytes
// K-plane/V-plane between the paged KV cache and the lm:// store).
//
// TWO CYCLES (both required by the arm):
//   (a) store -> "restart" -> load, WITHIN ONE PROCESS: a first engine (connector
//       ON) generates the prompt and STORES its prompt-prefix chunks; a fresh
//       second engine (connector ON, same server, empty local cache = a restart)
//       generates the SAME prompt, HITS the server, LOADS the KV, shortcuts
//       prefill, and must match the OFF run token-for-token. (mode "full")
//   (b) a genuinely COLD second process that only hits the server: run the binary
//       again with VT_LMCACHE_OI_MODE=loadonly — it does NOT store anything, yet
//       the connector still finds the prefix a PRIOR process stored and matches
//       the OFF run token-for-token. (mode "loadonly")
//
// Server + model are external inputs, so the body loudly SKIPS (compiles/links
// everywhere) unless BOTH are present:
//   * VLLM_CPP_OPT_MODEL_DIR (or ~/models/opt-125m-bf16-st) — the bf16 OPT-125m
//     checkpoint the OPT SACRED gate already uses (the smallest real model that
//     exercises the path; the connector + lm:// store are CPU, only the forward
//     needs the GPU and a 125M model is tiny — the unified-memory OOM discipline).
//   * VT_LMCACHE_OI_HOST / VT_LMCACHE_OI_PORT — a live `lmcache.v1.server`.
// Run it under scripts/lmcache/run_output_invariance.sh, which stands the server
// up and drives both modes.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/kv_offload/lmcache/lmcache_connector.h"

namespace fs = std::filesystem;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using LMConn = vllm::v1::kv_offload::lmcache::LMCacheConnector;

namespace {

std::string FindOptModelDir() {
  if (const char* env = std::getenv("VLLM_CPP_OPT_MODEL_DIR"); env != nullptr) {
    std::error_code ec;
    if (fs::exists(fs::path(env) / "config.json", ec)) return env;
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path dir = fs::path(home) / "models/opt-125m-bf16-st";
  std::error_code ec;
  if (fs::exists(dir / "config.json", ec) &&
      fs::exists(dir / "model.safetensors", ec))
    return dir.string();
  return "";
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// block_size 16 so a ~80-token prompt spans several full blocks; num_blocks and
// max_num_seqs kept tiny (one request at a time). The connector is wired only
// when connector_on.
EngineParams MakeParams(bool connector_on, const std::string& host,
                        const std::string& port) {
  EngineParams p;
  p.block_size = 16;
  p.num_blocks = 512;
  p.max_num_seqs = 1;
  if (connector_on) {
    vllm::KVTransferConfig cfg;
    cfg.kv_connector = "LMCacheConnector";
    cfg.kv_role = vllm::KVRole::kBoth;
    cfg.kv_connector_extra_config["host"] = host;
    cfg.kv_connector_extra_config["port"] = port;
    // blake3 block-aligned: our-store <-> our-load round-trip (this gate is
    // our client storing and our client loading; peer key-agreement is proven
    // separately by W4 test_lmcache_key_agreement). chunk_tokens == block_size
    // is injected by LoadedEngine from the runner geometry.
    cfg.kv_connector_extra_config["hash_algo"] = "blake3";
    p.kv_transfer_config = cfg;
  }
  return p;
}

struct GenResult {
  std::vector<int32_t> tokens;
  double seconds = 0.0;
};

GenResult Generate(LoadedEngine& engine, const std::string& prompt, int T,
                   const std::string& id) {
  const auto t0 = std::chrono::steady_clock::now();
  const vllm::RequestOutput out = engine.engine().generate(prompt, Greedy(T), id);
  const auto t1 = std::chrono::steady_clock::now();
  GenResult r;
  r.seconds = std::chrono::duration<double>(t1 - t0).count();
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  r.tokens = out.outputs[0].token_ids;
  return r;
}

int FirstDivergence(const std::vector<int32_t>& a,
                    const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return static_cast<int>(i);
  if (a.size() != b.size()) return static_cast<int>(n);
  return -1;
}

}  // namespace

TEST_CASE("LMCache connector-ON output-invariance in a real OPT-125m loop") {
  const std::string dir = FindOptModelDir();
  if (dir.empty()) {
    MESSAGE(
        "SKIP: OPT-125m bf16 checkpoint absent (~/models/opt-125m-bf16-st or "
        "$VLLM_CPP_OPT_MODEL_DIR)");
    return;
  }
  const char* host_env = std::getenv("VT_LMCACHE_OI_HOST");
  const char* port_env = std::getenv("VT_LMCACHE_OI_PORT");
  if (host_env == nullptr || port_env == nullptr) {
    MESSAGE(
        "SKIP: no live lmcache.v1.server (set VT_LMCACHE_OI_HOST / "
        "VT_LMCACHE_OI_PORT; run via scripts/lmcache/run_output_invariance.sh)");
    return;
  }
  const std::string host = host_env;
  const std::string port = port_env;
  const char* mode_env = std::getenv("VT_LMCACHE_OI_MODE");
  const std::string mode = mode_env != nullptr ? mode_env : "full";
  const bool load_only = (mode == "loadonly");

  // A prompt long enough to span several full 16-token blocks, so the external
  // prefix hit saves a meaningful, reportable number of tokens.
  const std::string prompt =
      "The history of the Roman empire spans many centuries of conquest, "
      "civil war, reform and decline, from the founding of the republic to "
      "the fall of Constantinople, and its legacy in law, language and "
      "architecture endures to this day across the whole of Europe.";
  const int T = 24;

  MESSAGE("lmcache-oi: mode=" << mode << " server=" << host << ":" << port
                              << " model=" << dir);

  // ---- OFF (cold full prefill) — the token BAR ------------------------------
  std::unique_ptr<LoadedEngine> engine_off =
      LoadedEngine::FromModelDir(dir, MakeParams(false, host, port));
  const GenResult off = Generate(*engine_off, prompt, T, "oi_off");
  REQUIRE(off.tokens.size() == static_cast<size_t>(T));

  // ---- STORE arm (connector ON, populates the server) — mode "full" only ----
  if (!load_only) {
    std::unique_ptr<LoadedEngine> engine_store =
        LoadedEngine::FromModelDir(dir, MakeParams(true, host, port));
    const GenResult store = Generate(*engine_store, prompt, T, "oi_store");
    auto* lm = dynamic_cast<LMConn*>(engine_store->kv_connector());
    REQUIRE(lm != nullptr);
    MESSAGE("lmcache-oi STORE arm: chunks_stored="
            << lm->chunks_stored()
            << " external_tokens_matched=" << lm->external_tokens_matched()
            << " (a hit here means a prior process already populated the key)");
    // Storing/loading must NEVER change the tokens vs the OFF bar.
    const int d = FirstDivergence(store.tokens, off.tokens);
    CHECK_MESSAGE(d < 0,
                  "lmcache-oi: STORE-arm tokens diverge from OFF at pos " << d);
    CHECK(lm->chunks_stored() > 0);
  }

  // ---- LOAD arm (fresh engine = restart; hits the server, shortcuts prefill)-
  std::unique_ptr<LoadedEngine> engine_load =
      LoadedEngine::FromModelDir(dir, MakeParams(true, host, port));
  const GenResult load = Generate(*engine_load, prompt, T, "oi_load");
  auto* lm_load = dynamic_cast<LMConn*>(engine_load->kv_connector());
  REQUIRE(lm_load != nullptr);
  const int64_t saved = lm_load->external_tokens_matched();

  const int d = FirstDivergence(load.tokens, off.tokens);
  if (d >= 0) {
    MESSAGE("lmcache-oi DIVERGENCE at pos "
            << d << " load=" << load.tokens[static_cast<size_t>(d)]
            << " off=" << off.tokens[static_cast<size_t>(d)]);
  }

  // ---- THE GATE -------------------------------------------------------------
  // A prefix cache HIT must have occurred (the connector genuinely loaded KV
  // from the server), and the generated tokens must be BIT-IDENTICAL to the
  // cold OFF run.
  MESSAGE("lmcache-oi LOAD arm: prefill tokens saved (external match) = "
          << saved << " of " << off.tokens.size()
          << " generated; wall OFF=" << off.seconds << "s LOAD=" << load.seconds
          << "s (tiny model: fixed connect/transfer overhead dominates the "
             "compute saved — see docs/BENCHMARKS.md)");
  CHECK_MESSAGE(saved > 0,
                "lmcache-oi: LOAD arm matched ZERO external tokens — the "
                "connector never hit the server, so the invariance below is "
                "vacuous");
  REQUIRE_MESSAGE(d < 0,
                  "lmcache-oi: connector-ON generated tokens are NOT identical "
                  "to connector-OFF — a BUG in the KV load path (gap-0 "
                  "correctness), never a near-tie");
}
