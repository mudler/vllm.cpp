// vllm-cli — a llama.cpp-style command-line client for vllm.cpp, driven ENTIRELY
// through the stable C ABI (include/vllm.h). It links libvllm and uses only the
// public `vllm_*` surface (no C++ engine headers), so it doubles as a worked
// example of the library-first packaging: load a model dir, run one completion
// (blocking or streaming), print it, free everything, report errors via
// vllm_last_error().
//
//   vllm-cli --model <dir> --prompt "<text>"
//            [--tokenizer-config <path>]
//            [--max-tokens N] [--temperature T] [--top-p P] [--top-k K]
//            [--seed S] [--stream]
//
// <dir> holds config.json, tokenizer.json and the *.safetensors shards (T0:
// safetensors only). Loading a real checkpoint is a GPU/dgx concern; on a CPU
// box `--help` / bad-args still work without a model (smoke-tested in CI).
#include "vllm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string model_dir;
  std::string tokenizer_config;  // optional override
  std::string prompt;
  bool have_prompt = false;
  int max_tokens = 16;
  float temperature = 0.0f;  // default: greedy (deterministic).
  float top_p = 1.0f;
  int top_k = 0;
  unsigned long long seed = 0;
  bool have_seed = false;
  bool stream = false;
};

void Usage(const char* argv0, std::FILE* out) {
  std::fprintf(
      out,
      "usage: %s --model <dir> --prompt \"<text>\"\n"
      "          [--tokenizer-config <path>]\n"
      "          [--max-tokens N] [--temperature T] [--top-p P] [--top-k K]\n"
      "          [--seed S] [--stream]\n"
      "\n"
      "Runs one completion over the vllm.cpp C ABI (libvllm). <dir> holds\n"
      "config.json, tokenizer.json and the *.safetensors shards.\n",
      argv0);
}

// Returns the value following `argv[i]`, advancing i; prints usage + exits on a
// missing operand.
const char* NextArg(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "vllm-cli: missing value for '%s'\n", argv[i]);
    Usage(argv[0], stderr);
    std::exit(2);
  }
  return argv[++i];
}

// Parse argv. Returns false (after printing usage) if the args are invalid or
// --help was given; *exit_code carries the process exit status in that case.
bool ParseArgs(int argc, char** argv, Args& a, int& exit_code) {
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--model") {
      a.model_dir = NextArg(argc, argv, i);
    } else if (flag == "--tokenizer-config") {
      a.tokenizer_config = NextArg(argc, argv, i);
    } else if (flag == "--prompt") {
      a.prompt = NextArg(argc, argv, i);
      a.have_prompt = true;
    } else if (flag == "--max-tokens") {
      a.max_tokens = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--temperature") {
      a.temperature = static_cast<float>(std::atof(NextArg(argc, argv, i)));
    } else if (flag == "--top-p") {
      a.top_p = static_cast<float>(std::atof(NextArg(argc, argv, i)));
    } else if (flag == "--top-k") {
      a.top_k = std::atoi(NextArg(argc, argv, i));
    } else if (flag == "--seed") {
      a.seed = std::strtoull(NextArg(argc, argv, i), nullptr, 10);
      a.have_seed = true;
    } else if (flag == "--stream") {
      a.stream = true;
    } else if (flag == "-h" || flag == "--help") {
      Usage(argv[0], stdout);
      exit_code = 0;
      return false;
    } else {
      std::fprintf(stderr, "vllm-cli: unknown argument '%s'\n", flag.c_str());
      Usage(argv[0], stderr);
      exit_code = 2;
      return false;
    }
  }
  if (a.model_dir.empty()) {
    std::fprintf(stderr, "vllm-cli: --model <dir> is required\n");
    Usage(argv[0], stderr);
    exit_code = 2;
    return false;
  }
  if (!a.have_prompt) {
    std::fprintf(stderr, "vllm-cli: --prompt \"<text>\" is required\n");
    Usage(argv[0], stderr);
    exit_code = 2;
    return false;
  }
  return true;
}

// Streaming callback: print each delta as it arrives (flushing so it streams to
// the terminal), keep going until the engine finishes. Matches
// vllm_token_callback exactly.
bool StreamPrintCb(const char* delta_text, bool finished, void* user_data) {
  (void)finished;
  (void)user_data;
  if (delta_text != nullptr) {
    std::fputs(delta_text, stdout);
    std::fflush(stdout);
  }
  return true;  // never stop early from the CLI.
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  int exit_code = 0;
  if (!ParseArgs(argc, argv, args, exit_code)) {
    return exit_code;
  }

  // ── Load the model + build the engine stack via the C ABI. ─────────────────
  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = args.model_dir.c_str();
  if (!args.tokenizer_config.empty()) {
    mp.tokenizer_config_path = args.tokenizer_config.c_str();
  }

  vllm_engine* engine = nullptr;
  std::fprintf(stderr, "vllm-cli: loading model from %s\n",
               args.model_dir.c_str());
  vllm_status st = vllm_engine_load(&mp, &engine);
  if (st != VLLM_OK) {
    std::fprintf(stderr, "vllm-cli: model load failed (status %d): %s\n",
                 static_cast<int>(st), vllm_last_error());
    return 1;
  }

  // ── Sampling params from the CLI flags. ────────────────────────────────────
  vllm_sampling_params sp = vllm_sampling_params_default();
  sp.temperature = args.temperature;  // <= 0 => greedy.
  sp.top_p = args.top_p;
  sp.top_k = args.top_k;
  sp.max_tokens = args.max_tokens;
  if (args.have_seed) {
    sp.has_seed = 1;
    sp.seed = static_cast<uint64_t>(args.seed);
  }

  int rc = 0;
  if (args.stream) {
    // ── Streaming: print deltas as they arrive. ──────────────────────────────
    st = vllm_complete_stream(engine, args.prompt.c_str(), &sp, &StreamPrintCb,
                              nullptr);
    std::fputc('\n', stdout);
    if (st != VLLM_OK) {
      std::fprintf(stderr, "vllm-cli: streaming completion failed (status %d): %s\n",
                   static_cast<int>(st), vllm_last_error());
      rc = 1;
    }
  } else {
    // ── Blocking: run to completion, then print the whole text. ──────────────
    vllm_completion out;
    st = vllm_complete(engine, args.prompt.c_str(), &sp, &out);
    if (st != VLLM_OK) {
      std::fprintf(stderr, "vllm-cli: completion failed (status %d): %s\n",
                   static_cast<int>(st), vllm_last_error());
      rc = 1;
    } else {
      std::fputs(out.text != nullptr ? out.text : "", stdout);
      std::fputc('\n', stdout);
      std::fprintf(stderr,
                   "vllm-cli: finish_reason=%s prompt_tokens=%d completion_tokens=%d\n",
                   out.finish_reason != nullptr ? out.finish_reason : "(none)",
                   out.prompt_tokens, out.completion_tokens);
      vllm_completion_free(&out);
    }
  }

  vllm_engine_free(engine);
  return rc;
}
