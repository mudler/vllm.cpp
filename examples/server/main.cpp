// server: an OpenAI-compatible HTTP server over the vllm.cpp LLMEngine (M3.1
// Task 4). Loads a supported model (safetensors or GGUF weights + tokenizer + a KV-cache config →
// LLMEngine), constructs the OpenAI serving handlers (chat wired with the real
// chat template via MakeChatTemplatePromptFn(LoadChatTemplateFromConfig(...)))
// and serves /v1/completions, /v1/chat/completions, /v1/models, /health,
// /version.
//
//   server --model <dir> [--host 0.0.0.0] [--port 8000]
//          [--tokenizer-config <tokenizer_config.json>]
//          [--served-model-name <name>]
//          [--block-size N] [--num-blocks N] [--max-model-len N]
//          [--max-num-seqs N] [--max-num-batched-tokens N]
//          [--enable-force-include-usage]
//          [--[no-]enable-prefix-caching]
//          [--scheduling-policy fcfs|priority]
//
// A directory holds config.json, tokenizer.json and supported safetensors
// shards. A supported GGUF file is also accepted and supplies model metadata
// plus embedded vocabulary. If --tokenizer-config is omitted for a directory it
// defaults to <dir>/tokenizer_config.json; when that file has no chat_template
// the chat endpoint falls back to the simple role-join prompt.
//
// NOTE: loading the real 35B checkpoint is a GPU/dgx concern; on the CPU CI box
// this binary is only built + smoke-tested against a synthetic engine (see
// tests/vllm/entrypoints/openai/test_api_server.cpp). The wiring below is the
// same either way.
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/chat_template.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/openai/api_server.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_completion.h"
#include "vllm/entrypoints/openai/serving_models.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/version.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

namespace fs = std::filesystem;
using vllm::HfConfig;
using vllm::Qwen3_5MoeWeights;

struct Args {
  std::string model_dir;
  std::string host = "0.0.0.0";
  int port = 8000;
  std::string tokenizer_config;  // default: <model_dir>/tokenizer_config.json
  std::string served_model_name;  // default: the model dir name
  int block_size = 32;
  int num_blocks = 256;
  int max_model_len = 0;  // 0 => config.max_position_embeddings
  int max_num_seqs = 8;
  int max_num_batched_tokens = 0;  // 0 => per-architecture default.
  std::optional<bool> enable_prefix_caching = std::nullopt;
  bool enable_force_include_usage = false;
  // Scheduling policy: "fcfs" (default) or "priority" (mirrors vLLM's
  // --scheduling-policy / SchedulerConfig.policy).
  std::string scheduling_policy = "fcfs";
};

[[noreturn]] void Usage(const char* argv0, int code) {
  std::cerr
      << "usage: " << argv0
      << " --model <dir> [--host H] [--port P] [--tokenizer-config F]\n"
         "               [--served-model-name N] [--block-size N] "
         "[--num-blocks N] [--max-model-len N]\n"
         "               [--max-num-seqs N] "
         "[--max-num-batched-tokens N]\n"
         "               [--enable-force-include-usage]\n"
         "               [--[no-]enable-prefix-caching]\n"
         "               [--scheduling-policy fcfs|priority]\n";
  std::exit(code);
}

std::string NextArg(int argc, char** argv, int& i, const char* argv0) {
  if (i + 1 >= argc) Usage(argv0, 2);
  return argv[++i];
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--model") {
      a.model_dir = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--host") {
      a.host = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--port") {
      a.port = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--tokenizer-config") {
      a.tokenizer_config = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--served-model-name") {
      a.served_model_name = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "--block-size") {
      a.block_size = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--num-blocks") {
      a.num_blocks = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--max-model-len") {
      a.max_model_len = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--max-num-seqs") {
      a.max_num_seqs = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--max-num-batched-tokens") {
      a.max_num_batched_tokens = std::stoi(NextArg(argc, argv, i, argv[0]));
    } else if (flag == "--enable-force-include-usage") {
      a.enable_force_include_usage = true;
    } else if (flag == "--enable-prefix-caching" ||
               flag == "--no-enable-prefix-caching") {
      if (a.enable_prefix_caching.has_value()) {
        std::cerr << "server: prefix-caching flag specified more than once\n";
        Usage(argv[0], 2);
      }
      a.enable_prefix_caching = flag == "--enable-prefix-caching";
    } else if (flag == "--scheduling-policy") {
      a.scheduling_policy = NextArg(argc, argv, i, argv[0]);
    } else if (flag == "-h" || flag == "--help") {
      Usage(argv[0], 0);
    } else {
      std::cerr << "server: unknown argument '" << flag << "'\n";
      Usage(argv[0], 2);
    }
  }
  if (a.model_dir.empty()) {
    std::cerr << "server: --model <dir> is required\n";
    Usage(argv[0], 2);
  }
  if (a.max_num_seqs <= 0 || a.max_num_batched_tokens < 0) {
    std::cerr << "server: scheduler capacities must be positive "
                 "(--max-num-batched-tokens may be 0 for auto)\n";
    Usage(argv[0], 2);
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);

    const fs::path dir(args.model_dir);
    const std::string config_path = (dir / "config.json").string();
    const std::string tokenizer_path = (dir / "tokenizer.json").string();
    const std::string tokenizer_config_path =
        args.tokenizer_config.empty()
            ? (dir / "tokenizer_config.json").string()
            : args.tokenizer_config;
    const std::string served_model_name =
        args.served_model_name.empty()
            ? (dir.has_filename() ? dir.filename().string()
                                  : dir.parent_path().filename().string())
            : args.served_model_name;

    // ── Load the model + build the full engine stack via the shared loader
    // (src/vllm/entrypoints/model_loader.cpp) — the same path the C ABI drives.
    // It loads config.json + tokenizer.json + *.safetensors and wires the M1.8
    // LLMEngine over Scheduler + runner + KV + processors. ─────────────────────
    std::cerr << "server: loading model from " << args.model_dir << " (config "
              << config_path << ", tokenizer " << tokenizer_path << ")\n";
    vllm::entrypoints::EngineParams engine_params;
    engine_params.block_size = args.block_size;
    engine_params.num_blocks = args.num_blocks;
    engine_params.max_model_len = args.max_model_len;  // 0 => from config.
    engine_params.max_num_seqs = args.max_num_seqs;
    engine_params.max_num_batched_tokens = args.max_num_batched_tokens;
    engine_params.enable_prefix_caching = args.enable_prefix_caching;
    // Reject an unknown policy string (mirrors upstream SchedulingPolicy(value)).
    engine_params.policy = vllm::SchedulerPolicyFromString(args.scheduling_policy);
    std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
        vllm::entrypoints::LoadedEngine::FromModelDir(args.model_dir,
                                                      engine_params);
    std::cerr << "server: prefix caching "
              << (loaded->prefix_caching_enabled() ? "enabled" : "disabled")
              << "\n";
    // W2: the production server uses AsyncLLM over EngineCoreProc's dedicated
    // engine thread. HTTP workers submit independently and stream from their
    // per-request collectors; no server-wide engine mutex remains.
    vllm::v1::AsyncLLM& engine = loaded->async_engine();
    const vllm::tok::Tokenizer& tokenizer = loaded->tokenizer();

    // ── OpenAI serving handlers. The chat handler is wired with the real chat
    // template (Task 3) when tokenizer_config.json carries one; otherwise it
    // keeps the default role-join fallback. ────────────────────────────────
    namespace oai = vllm::entrypoints::openai;
    oai::OpenAIServingModels models(served_model_name);
    oai::OpenAIServingCompletion completion(
        engine, served_model_name, args.enable_force_include_usage);

    oai::ChatPromptFn chat_prompt_fn = oai::DefaultChatPromptFallback;
    try {
      const std::string chat_template =
          vllm::entrypoints::LoadChatTemplateFromConfig(tokenizer_config_path);
      const std::string bos =
          tokenizer.BosId() >= 0 ? tokenizer.Decode({tokenizer.BosId()}) : "";
      const std::string eos =
          tokenizer.EosId() >= 0 ? tokenizer.Decode({tokenizer.EosId()}) : "";
      chat_prompt_fn =
          vllm::entrypoints::MakeChatTemplatePromptFn(chat_template, bos, eos);
      std::cerr << "server: using chat template from " << tokenizer_config_path
                << "\n";
    } catch (const std::exception& e) {
      std::cerr << "server: no chat template (" << e.what()
                << "); falling back to the simple role-join prompt\n";
    }
    oai::OpenAIServingChat chat(engine, served_model_name, chat_prompt_fn,
                                "hermes", args.enable_force_include_usage);

    oai::ApiServer server(completion, chat, models, vllm::Version());
    std::cerr << "server: listening on http://" << args.host << ":" << args.port
              << " (model '" << served_model_name << "')\n";
    if (!server.listen(args.host, args.port)) {
      std::cerr << "server: failed to bind " << args.host << ":" << args.port
                << "\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "server: fatal: " << e.what() << "\n";
    return 1;
  }
}
