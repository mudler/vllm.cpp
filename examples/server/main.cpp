// server: an OpenAI-compatible HTTP server over the vllm.cpp LLMEngine (M3.1
// Task 4). Loads a model (safetensors weights + tokenizer + a KV-cache config →
// LLMEngine), constructs the OpenAI serving handlers (chat wired with the real
// chat template via MakeChatTemplatePromptFn(LoadChatTemplateFromConfig(...)))
// and serves /v1/completions, /v1/chat/completions, /v1/models, /health,
// /version.
//
//   server --model <dir> [--host 0.0.0.0] [--port 8000]
//          [--tokenizer-config <tokenizer_config.json>]
//          [--served-model-name <name>]
//          [--block-size N] [--num-blocks N] [--max-model-len N]
//
// <dir> holds config.json, tokenizer.json and the *.safetensors shards (T0:
// safetensors only — GGUF load is M0.10). If --tokenizer-config is omitted it
// defaults to <dir>/tokenizer_config.json; when that file has no chat_template
// the chat endpoint falls back to the simple role-join prompt.
//
// NOTE: loading the real 35B checkpoint is a GPU/dgx concern; on the CPU CI box
// this binary is only built + smoke-tested against a synthetic engine (see
// tests/vllm/entrypoints/openai/test_api_server.cpp). The wiring below is the
// same either way.
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/entrypoints/chat_template.h"
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
};

[[noreturn]] void Usage(const char* argv0, int code) {
  std::cerr
      << "usage: " << argv0
      << " --model <dir> [--host H] [--port P] [--tokenizer-config F]\n"
         "               [--served-model-name N] [--block-size N] "
         "[--num-blocks N] [--max-model-len N]\n";
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
  return a;
}

// Build a hybrid KV-cache config (one full-attention group + one GDN/mamba
// group) sized from the HfConfig. Mirrors the test harness MakeKvConfig; the
// runner keys off the spec KIND (not the group-id strings), so this generalizes
// to any layer count. block_size == the caller's --block-size (unified).
vllm::v1::KVCacheConfig MakeKvConfig(const HfConfig& c, int block_size,
                                     int num_blocks) {
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);
  const int Hv = static_cast<int>(c.linear_num_value_heads);
  const int Dv = static_cast<int>(c.linear_value_head_dim);
  const int Dk = static_cast<int>(c.linear_key_head_dim);
  const int Kw = static_cast<int>(c.linear_conv_kernel_dim);
  const int key_dim = static_cast<int>(c.linear_num_key_heads) * Dk;
  const int value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;

  vllm::v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<vllm::v1::FullAttentionSpec>(block_size, Hkv, Dh,
                                                    vt::DType::kF32));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn"},
      std::make_shared<vllm::v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{{Hv, Dv, Dk}, {conv_dim, Kw - 1}},
          std::vector<vt::DType>{vt::DType::kF32, vt::DType::kF32}));
  return kv;
}

std::vector<vllm::SafetensorsFile> LoadShards(const std::string& model_dir) {
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(model_dir)) {
    if (e.is_regular_file() && e.path().extension() == ".safetensors") {
      paths.push_back(e.path().string());
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("no *.safetensors shards found in " + model_dir);
  }
  std::sort(paths.begin(), paths.end());
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(paths.size());
  for (const std::string& p : paths) {
    shards.push_back(vllm::SafetensorsFile::Open(p));
  }
  return shards;
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

    std::cerr << "server: loading config " << config_path << "\n";
    const HfConfig config = vllm::LoadHfConfig(config_path);

    std::cerr << "server: loading tokenizer " << tokenizer_path << "\n";
    const vllm::tok::Tokenizer tokenizer =
        vllm::tok::Tokenizer::FromHfJson(tokenizer_path);

    std::cerr << "server: loading weights from " << args.model_dir << "\n";
    std::vector<vllm::SafetensorsFile> shards = LoadShards(args.model_dir);
    const Qwen3_5MoeWeights weights = vllm::LoadQwen3_5Moe(shards, config);
    shards.clear();  // the mmap'd shards may be released after the load.

    const int max_model_len = args.max_model_len > 0
                                  ? args.max_model_len
                                  : static_cast<int>(config.max_position_embeddings);

    // ── Engine stack (mirrors the M1.8 wiring in test_llm_engine.cpp). ──────
    vllm::v1::init_none_hash(vllm::v1::sha256_cbor);

    vllm::SchedulerConfig sched_cfg;
    sched_cfg.max_num_seqs = 8;
    sched_cfg.max_num_batched_tokens = max_model_len * 8;
    sched_cfg.enable_chunked_prefill = true;
    sched_cfg.max_model_len = max_model_len;
    sched_cfg.watermark = 0.0;

    const vllm::v1::KVCacheConfig kv_cfg =
        MakeKvConfig(config, args.block_size, args.num_blocks);

    vllm::v1::Scheduler scheduler(sched_cfg, kv_cfg, args.block_size,
                                  /*enable_caching=*/true);
    vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    vllm::v1::GPUModelRunner runner(config, weights, kv_cfg, queue,
                                    /*max_num_reqs=*/8, max_model_len,
                                    /*max_num_batched_tokens=*/max_model_len * 8);
    vllm::v1::Executor executor(runner);
    vllm::v1::EngineCore engine_core(scheduler, executor);
    vllm::v1::InputProcessor input_processor(tokenizer, config);
    vllm::v1::OutputProcessor output_processor(&tokenizer);
    vllm::v1::LLMEngine engine(
        input_processor, engine_core, output_processor,
        vllm::v1::get_request_block_hasher(args.block_size,
                                           vllm::v1::sha256_cbor));

    // ── OpenAI serving handlers. The chat handler is wired with the real chat
    // template (Task 3) when tokenizer_config.json carries one; otherwise it
    // keeps the default role-join fallback. ────────────────────────────────
    namespace oai = vllm::entrypoints::openai;
    oai::OpenAIServingModels models(served_model_name);
    oai::OpenAIServingCompletion completion(engine, served_model_name);

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
    oai::OpenAIServingChat chat(engine, served_model_name, chat_prompt_fn);

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
