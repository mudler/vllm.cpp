// vllm_c.cpp — the vllm.cpp C ABI implementation (see include/vllm.h).
//
// ORIGINAL packaging layer — NOT a 1:1 upstream mirror (vLLM ships no C ABI;
// recorded as a deviation in .agents/porting-inventory.md §9). C++ internally,
// `extern "C"` at the boundary. Every entry point wraps its body in try/catch,
// stores the message in a thread-local buffer (surfaced by vllm_last_error), and
// returns a vllm_status — NOTHING throws across the ABI. The ergonomics follow
// llama.cpp's llama.h (handle-based load -> complete -> free).
#include "vllm.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "capi/chat_prompt.h"
#include "capi/engine_handle.h"
#include "vllm/config/kv_transfer.h"   // ParseKVTransferConfigJson (ABI v9)
#include "vllm/config/multimodal.h"    // ParseLimitMmPerPromptJson (ABI v19)
#include "vllm/config/scheduler.h"     // SchedulerPolicyFromString (ABI v9)
#include "vllm/v1/kv_offload/kv_connector.h"  // KVConnectorFactory (ABI v9)
#include "vllm/entrypoints/chat_template.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"  // get_tool_parser
#include "vllm/entrypoints/openai/tool_parsers/detect.h"    // DetectToolParser
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"  // get_reasoning_parser
#include "vllm/entrypoints/openai/reasoning_parsers/detect.h"  // DetectReasoningParser
#include "vllm/model_executor/models/model_registry.h"  // refuse-by-task (v11)
#include "vllm/model_executor/models/minimax_h3.h"    // mux argv (v12)
#include "vllm/multimodal/parakeet_transcription.h"     // vllm_transcribe (v11)
#include "vllm/multimodal/minimax_h3_video.h"          // vllm_video_* (v12)
#include "vllm/multimodal/video_engine.h"              // the v18 family registry
#include "vllm/entrypoints/openai/server_main.h"   // vllm_server_main (v17)
#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"  // PeekHfArchitectures (v11)
#include "vllm/version.h"
#include "vllm/v1/engine/async_llm.h"

// The opaque handle: owns the whole C++ engine stack behind LoadedEngine —
// OR, since ABI v11, a transcription stack (ParakeetTranscriber) when the
// model directory resolves to a SupportsTranscription-ONLY architecture.
// Exactly one of `loaded` / `transcriber` is set; every text entry point
// guards on `loaded` (RequireTextEngine) and vllm_transcribe on `transcriber`,
// so the two task families refuse each other cleanly instead of crashing.
struct vllm_engine {
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded;
  // ABI v11 transcription stack (the ONE library seam the server route and the
  // parakeet-transcribe example also drive). Null for text engines.
  std::unique_ptr<vllm::multimodal::ParakeetTranscriber> transcriber;
  // Monotonic per-handle request-id source. Each vllm_complete[_stream] call
  // uses a FRESH id so a request left in-flight by a mid-call exception can never
  // collide with a later call's id — a collision would make LLMEngine.add_request
  // free-and-reinsert the same key while the scheduler still holds the old
  // (now-freed) Request → heap-use-after-free. Unique ids + the RequestGuard
  // below make the engine safely reusable after ANY mid-request error.
  std::atomic<uint64_t> next_request_id{0};

  // ── Chat entry points (ABI v3) ────────────────────────────────────────────
  // The model path vllm_engine_load received; empty for a test-hook handle.
  // Used to resolve the chat template + the served model name.
  std::string model_path;
  // ABI v9 vllm_model_params.tokenizer_config_path: an explicit override for the
  // tokenizer_config.json the chat template is read from. Empty => the
  // <model_path>/tokenizer_config.json default. Ignored for a .gguf model_path
  // (its template lives in GGUF metadata).
  std::string tokenizer_config_path;
  // Test-hook override for the chat-prompt seam (MakeEngineHandle overload):
  // when set, chat_serving is built with it instead of the resolved template.
  vllm::entrypoints::openai::ChatPromptFn test_prompt_fn;
  // The caller-selected tool-call parser (ABI v4 vllm_model_params.tool_parser),
  // copied at vllm_engine_load. Empty => AUTO-detect from the chat template at
  // the first chat call; a non-empty value must name a registered parser or the
  // first chat call fails with VLLM_ERR_INVALID_ARGUMENT (checked in
  // EnsureChatServing). Tests set it via vllm::capi::SetEngineToolParser.
  std::string tool_parser;
  // ABI v5 reasoning-parser selection ("" = auto-detect from the template;
  // "none" = force-disabled). Tests set it via SetEngineReasoningParser.
  std::string reasoning_parser;
  // Lazily-built chat serving handler over the shared AsyncLLM (one per
  // handle; create_chat_completion is safe for concurrent callers, matching
  // the HTTP server's worker pool). Guarded by chat_mutex for the lazy build.
  std::mutex chat_mutex;
  std::unique_ptr<vllm::entrypoints::openai::OpenAIServingChat> chat_serving;
  // ABI v15: serialize vllm_embed batches per handle (the pooling path drives
  // the SYNCHRONOUS LLMEngine, not the AsyncLLM the text entry points share).
  std::mutex embed_mutex;
};

// One non-blocking callback-delivery request. The AsyncLLM output handler owns
// EngineCore output processing; this lightweight thread only consumes this
// request's collector and invokes the C callback.
struct vllm_request {
  vllm_engine* parent = nullptr;  // borrowed; parent must outlive this handle.
  vllm::v1::AsyncRequest async_request;
  vllm_token_callback callback = nullptr;
  void* user_data = nullptr;
  std::thread delivery_thread;
  std::mutex join_mutex;
  std::atomic<bool> done{false};
  std::atomic<bool> cancelled{false};
  vllm_status status = VLLM_OK;
  std::string error;
};

namespace {

// RAII: aborts an in-flight request on scope exit (incl. exception unwind) unless
// disarmed after a clean finish. abort_request is a safe no-op on an already-
// finished/unknown id, and we swallow any exception so the noexcept dtor can't
// std::terminate during unwind. This guarantees no request is left registered
// after a throwing callback or a mid-stream runtime error.
struct RequestGuard {
  vllm::v1::AsyncLLM& engine;
  std::string id;
  bool armed = true;
  ~RequestGuard() {
    if (!armed) return;
    try {
      engine.abort(id);
    } catch (...) {  // NOLINT(bugprone-empty-catch) — dtor must not throw
    }
  }
  void disarm() { armed = false; }
};

// Thread-local last-error string. Set on every non-OK return; read by
// vllm_last_error(). Thread-local so concurrent callers on different threads do
// not clobber each other's error.
thread_local std::string g_last_error;

void SetError(const std::string& msg) { g_last_error = msg; }
void ClearError() { g_last_error.clear(); }

// ABI v11 refuse-by-task: true when the handle owns the TEXT engine stack.
// A transcription-only handle (Parakeet) reports an actionable error instead
// of dereferencing the null LoadedEngine — the SupportsTranscription-only
// mirror of vLLM excluding "generate" from supported_tasks
// (vllm/model_executor/models/interfaces.py:1118). Since ABI v15 the SAME
// guard also refuses a POOLING (embedding) engine: its model has no
// text-generation path either (is_pooling_model && !is_text_generation_model,
// the mirror of vLLM's runner_type validation, config/model.py:607-613) —
// running generate on it would sample over hidden states.
bool RequireTextEngine(const vllm_engine* engine, const char* fn) {
  if (engine->loaded != nullptr) {
    if (engine->loaded->is_pooling_model()) {
      SetError(std::string(fn) +
               ": this engine was loaded from a pooling (embedding) "
               "checkpoint; it has no text-generation path — use vllm_embed "
               "or the server's /v1/embeddings");
      return false;
    }
    return true;
  }
  SetError(std::string(fn) +
           ": this engine was loaded from a transcription-only checkpoint "
           "(Parakeet); it has no text-generation path — use vllm_transcribe");
  return false;
}

// Heap-copy a std::string into a caller-owned NUL-terminated C string (freed via
// vllm_string_free / vllm_completion_free). Returns nullptr on allocation
// failure.
char* DupString(const std::string& s) {
  char* out = static_cast<char*>(std::malloc(s.size() + 1));
  if (out == nullptr) return nullptr;
  std::memcpy(out, s.data(), s.size());
  out[s.size()] = '\0';
  return out;
}

// Map a finish_reason string (as produced by the OutputProcessor, i.e. the
// upstream FINISH_REASON_STRINGS) to a stable static literal, so vllm_completion
// can expose it as a borrowed `const char*` the caller never frees.
const char* CanonicalFinishReason(const std::string& reason) {
  if (reason == "stop") return "stop";
  if (reason == "length") return "length";
  if (reason == "abort") return "abort";
  if (reason == "error") return "error";
  if (reason == "repetition") return "repetition";
  return "unknown";
}

// Translate the C sampling POD into vllm::SamplingParams and run PostInit()
// (mandatory normalization + validation). `output_kind` selects CUMULATIVE (the
// blocking generate() driver returns the full text) vs DELTA (the streaming
// driver gets one incremental delta per step). Throws std::runtime_error (via
// Verify/PostInit) on invalid params.
vllm::SamplingParams ToSamplingParams(const vllm_sampling_params& c,
                                      vllm::RequestOutputKind output_kind) {
  vllm::SamplingParams sp;
  sp.temperature = static_cast<double>(c.temperature);
  sp.top_p = static_cast<double>(c.top_p);
  sp.top_k = c.top_k;
  sp.min_p = static_cast<double>(c.min_p);
  sp.presence_penalty = static_cast<double>(c.presence_penalty);
  sp.frequency_penalty = static_cast<double>(c.frequency_penalty);
  sp.repetition_penalty = static_cast<double>(c.repetition_penalty);
  sp.min_tokens = c.min_tokens;
  sp.ignore_eos = c.ignore_eos != 0;
  if (c.max_tokens > 0) {
    sp.max_tokens = c.max_tokens;
  } else {
    sp.max_tokens = std::nullopt;  // unbounded (capped by max_model_len).
  }
  if (c.has_seed != 0) {
    sp.seed = static_cast<int64_t>(c.seed);
  }
  if (c.stop != nullptr && c.n_stop > 0) {
    sp.stop.reserve(static_cast<size_t>(c.n_stop));
    for (int32_t i = 0; i < c.n_stop; ++i) {
      if (c.stop[i] != nullptr) sp.stop.emplace_back(c.stop[i]);
    }
  }
  // Structured output (ABI v2): lower the POD constraint fields into
  // StructuredOutputsParams. PostInit() -> Verify() enforces the exactly-one
  // rule, so setting more than one constraint is rejected here, not deep in the
  // engine.
  const bool has_choice =
      c.structured_choice != nullptr && c.n_structured_choice > 0;
  if (c.structured_json != nullptr || c.structured_regex != nullptr ||
      has_choice || c.structured_grammar != nullptr ||
      c.structured_json_object != 0) {
    vllm::StructuredOutputsParams so;
    if (c.structured_json != nullptr) so.json = std::string(c.structured_json);
    if (c.structured_regex != nullptr)
      so.regex = std::string(c.structured_regex);
    if (has_choice) {
      std::vector<std::string> choice;
      choice.reserve(static_cast<size_t>(c.n_structured_choice));
      for (int32_t i = 0; i < c.n_structured_choice; ++i) {
        if (c.structured_choice[i] != nullptr)
          choice.emplace_back(c.structured_choice[i]);
      }
      so.choice = std::move(choice);
    }
    if (c.structured_grammar != nullptr)
      so.grammar = std::string(c.structured_grammar);
    if (c.structured_json_object != 0) so.json_object = true;
    sp.structured_outputs = std::move(so);
  }
  // Custom logits processor (ABI v8): carry the host callback + user_data onto
  // the SamplingParams. NULL fn => no processor (byte-identical default).
  if (c.logits_processor != nullptr) {
    sp.logits_processor.fn = c.logits_processor;
    sp.logits_processor.user_data = c.logits_processor_user_data;
  }
  sp.output_kind = output_kind;
  sp.PostInit();
  return sp;
}

// Resolve the chat-prompt seam for a loaded model: tokenizer_config.json's
// chat_template for a model directory, the `tokenizer.chat_template` GGUF
// metadata for a .gguf file, else the role-join fallback (mirrors the bundled
// server's resolution in examples/server/main.cpp).
//
// When `out_raw_template` is non-null it receives the RAW resolved template
// string (the tool-parser auto-detector sniffs it), or stays untouched when no
// template resolved at all. The raw string is captured regardless of whether the
// minja subset can render it: detection is a plain substring match, so an exotic
// template that degrades to the fallback for RENDERING still names its tool-call
// dialect for DETECTION.
vllm::entrypoints::openai::ChatPromptFn ResolveChatPromptFn(
    const std::string& model_path,
    const vllm::entrypoints::LoadedEngine& loaded,
    std::string* out_raw_template = nullptr,
    const std::string& tokenizer_config_path = std::string()) {
  namespace fs = std::filesystem;
  try {
    std::string tmpl;
    if (fs::is_regular_file(model_path) &&
        fs::path(model_path).extension() == ".gguf") {
      tmpl = vllm::entrypoints::LoadChatTemplateFromGguf(model_path);
    } else {
      // ABI v9: an explicit tokenizer_config_path wins over the sibling default,
      // mirroring the server's --tokenizer-config.
      tmpl = vllm::entrypoints::LoadChatTemplateFromConfig(
          tokenizer_config_path.empty()
              ? (fs::path(model_path) / "tokenizer_config.json").string()
              : tokenizer_config_path);
    }
    if (out_raw_template != nullptr) *out_raw_template = tmpl;
    const vllm::tok::Tokenizer& tok = loaded.tokenizer();
    const std::string bos =
        tok.BosId() >= 0 ? tok.Decode({tok.BosId()}) : std::string();
    const std::string eos =
        tok.EosId() >= 0 ? tok.Decode({tok.EosId()}) : std::string();
    // Probe-renders the template and degrades (with a stderr witness) to the
    // hermes-aware fallback when the minja subset cannot serve it.
    return vllm::capi::ResolveTemplatePromptFn(tmpl, bos, eos, model_path);
  } catch (const std::exception&) {
    // No template shipped with the model at all: the hermes-aware fallback
    // still primes the structural-tag tool flow, and detection sees no template.
    return vllm::capi::HermesToolsFallbackPrompt;
  }
}

// Lazily build the handle's chat serving handler over the shared AsyncLLM (same
// wiring as the bundled server: real chat template when the model ships one).
// One handler per handle; create_chat_completion is safe for concurrent callers.
//
// The tool-call parser (ABI v4) is selected here: an EXPLICIT handle->tool_parser
// wins; otherwise it is AUTO-detected from the resolved chat template
// (DetectToolParser); when no template resolved at all it defaults to "hermes".
// An explicitly-named parser that is not registered throws std::invalid_argument
// (the callers map it to VLLM_ERR_INVALID_ARGUMENT) — detection never returns an
// unregistered name, so only a bad caller-supplied name can trip this.
vllm::entrypoints::openai::OpenAIServingChat& EnsureChatServing(
    vllm_engine* engine) {
  std::lock_guard<std::mutex> lock(engine->chat_mutex);
  if (engine->chat_serving == nullptr) {
    std::string raw_template;
    vllm::entrypoints::openai::ChatPromptFn prompt_fn =
        engine->test_prompt_fn
            ? engine->test_prompt_fn
            : ResolveChatPromptFn(engine->model_path, *engine->loaded,
                                  &raw_template,
                                  engine->tokenizer_config_path);

    std::string parser_name;
    if (!engine->tool_parser.empty()) {
      parser_name = engine->tool_parser;  // explicit selection wins.
    } else if (!raw_template.empty()) {
      parser_name = vllm::entrypoints::openai::DetectToolParser(raw_template);
    } else {
      parser_name = "hermes";  // no template to detect from.
    }
    // Reject an unknown explicit name here, at the first chat call, rather than
    // silently disabling tool parsing (get_tool_parser returns nullptr for an
    // unregistered name, which MakeToolParser would treat as "no parser").
    if (!parser_name.empty() &&
        vllm::entrypoints::openai::get_tool_parser(parser_name) == nullptr) {
      throw std::invalid_argument("unknown tool parser \"" + parser_name +
                                  "\" (not a registered parser)");
    }

    // Reasoning selection (ABI v5): explicit wins ("none" disables), else
    // detect from the template; no detection => disabled (a reasoning parser
    // actively splits text, so silence is the only safe default).
    std::string reasoning_name;
    if (!engine->reasoning_parser.empty()) {
      if (engine->reasoning_parser != "none") {
        reasoning_name = engine->reasoning_parser;
      }
    } else if (!raw_template.empty()) {
      reasoning_name =
          vllm::entrypoints::openai::DetectReasoningParser(raw_template);
    }
    if (!reasoning_name.empty() &&
        vllm::entrypoints::openai::get_reasoning_parser(reasoning_name) ==
            nullptr) {
      throw std::invalid_argument("unknown reasoning parser \"" +
                                  reasoning_name +
                                  "\" (not a registered parser)");
    }

    std::string served_name =
        engine->model_path.empty()
            ? std::string("model")
            : std::filesystem::path(engine->model_path).filename().string();
    engine->chat_serving =
        std::make_unique<vllm::entrypoints::openai::OpenAIServingChat>(
            engine->loaded->async_engine(), std::move(served_name),
            std::move(prompt_fn), std::move(parser_name),
            std::move(reasoning_name));
  }
  return *engine->chat_serving;
}

// Strip the SSE framing off one serving-layer chunk ("data: {json}\n\n" →
// "{json}"; "data: [DONE]\n\n" → "[DONE]").
std::string StripSseFraming(const std::string& chunk) {
  std::string payload = chunk;
  constexpr const char kPrefix[] = "data: ";
  if (payload.rfind(kPrefix, 0) == 0) payload.erase(0, sizeof(kPrefix) - 1);
  while (!payload.empty() &&
         (payload.back() == '\n' || payload.back() == '\r')) {
    payload.pop_back();
  }
  return payload;
}

// Parse an OpenAI chat request JSON for the ABI entry points. Throws
// std::invalid_argument on malformed JSON / shape (mapped to
// VLLM_ERR_INVALID_ARGUMENT by the callers).
vllm::entrypoints::openai::ChatCompletionRequest ParseChatRequest(
    const char* request_json) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(request_json);
  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string("malformed request JSON: ") +
                                e.what());
  }
  try {
    return j.get<vllm::entrypoints::openai::ChatCompletionRequest>();
  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string("invalid chat request: ") +
                                e.what());
  }
}

void JoinRequest(vllm_request* request) {
  std::lock_guard<std::mutex> lock(request->join_mutex);
  if (request->delivery_thread.joinable()) request->delivery_thread.join();
}

void RunRequestDelivery(vllm_request* request) noexcept {
  try {
    vllm::v1::AsyncLLM& engine = request->parent->loaded->async_engine();
    for (;;) {
      vllm::RequestOutput output =
          engine.get_output(request->async_request);
      if (!request->cancelled.load()) {
        for (const vllm::CompletionOutput& completion : output.outputs) {
          const std::string delta =
              vllm::entrypoints::openai::SanitizeUtf8(completion.text);
          if (!request->callback(delta.c_str(), output.finished,
                                 request->user_data)) {
            request->cancelled.store(true);
            engine.abort(request->async_request.request_id);
            break;
          }
        }
      }
      if (output.finished || request->cancelled.load()) break;
    }
    request->status = VLLM_OK;
  } catch (const std::exception& e) {
    request->status = VLLM_ERR_RUNTIME;
    request->error = e.what();
    try {
      request->parent->loaded->async_engine().abort(
          request->async_request.request_id);
    } catch (...) {
    }
  } catch (...) {
    request->status = VLLM_ERR_UNKNOWN;
    request->error = "unknown asynchronous request error";
    try {
      request->parent->loaded->async_engine().abort(
          request->async_request.request_id);
    } catch (...) {
    }
  }
  request->done.store(true, std::memory_order_release);
}

}  // namespace

namespace {

// C++ LINKAGE, deliberately hoisted ABOVE the `extern "C"` block below.
//
// This helper returns std::string. Defined INSIDE `extern "C"` it inherits C
// linkage while returning a C++ type: GCC only warns, but Apple Clang's
// -Wreturn-type-c-linkage is an ERROR under the -Werror this project builds
// with, so the whole C ABI failed to compile there. Found downstream by the
// LocalAI vllm-cpp backend, which had to vendor a patch against a pinned SHA to
// build its metal-darwin-arm64 lane; this is that patch, upstream.
//
// Keep helpers that return or take C++ types on THIS side of the boundary.
std::string OrEmpty(const char* s) { return s == nullptr ? std::string() : std::string(s); }

}  // namespace

extern "C" {

VLLM_API vllm_model_params vllm_model_params_default(void) {
  vllm_model_params p;
  p.model_path = nullptr;
  p.tokenizer_config_path = nullptr;
  p.block_size = 32;
  p.num_blocks = 0;  // 0 => auto: sized by the v16 knobs, else 256.
  p.max_model_len = 0;
  p.max_num_seqs = 32;  // see EngineParams::max_num_seqs.
  p.tool_parser = nullptr;  // AUTO-detect from the chat template (ABI v4).
  p.reasoning_parser = nullptr;  // AUTO-detect / disabled (ABI v5).
  p.speculative_config = nullptr;  // speculation disabled (ABI v6).
  p.enable_prefix_caching = 0;     // 0 => model default (ABI v7).
  p.max_num_batched_tokens = 0;    // 0 => the per-arch default (ABI v9).
  p.scheduling_policy = nullptr;   // NULL => "fcfs" (ABI v9).
  p.kv_transfer_config = nullptr;  // NULL => no connector (ABI v9).
  p.enable_jump_forward = 0;       // 0 => env-resolved, default OFF (ABI v10).
  p.device = 0;  // 0 => auto: the accelerator-first probe (ABI v14).
  p.gpu_memory_utilization = 0.92;  // vLLM default fraction (ABI v16).
  p.kv_cache_memory_bytes = 0;      // 0 => unset (ABI v16).
  p.language_model_only = 0;        // 0 => off; limits stay at 999 (ABI v19).
  p.limit_mm_per_prompt = nullptr;  // NULL => no limits configured (ABI v19).
  return p;
}

VLLM_API vllm_sampling_params vllm_sampling_params_default(void) {
  // Defaults mirror vllm::SamplingParams (T0 subset).
  vllm_sampling_params p;
  p.temperature = 1.0f;
  p.top_p = 1.0f;
  p.top_k = 0;
  p.min_p = 0.0f;
  p.max_tokens = 16;
  p.seed = 0;
  p.has_seed = 0;
  p.presence_penalty = 0.0f;
  p.frequency_penalty = 0.0f;
  p.repetition_penalty = 1.0f;
  p.min_tokens = 0;
  p.ignore_eos = 0;
  p.stop = nullptr;
  p.n_stop = 0;
  p.structured_json = nullptr;
  p.structured_regex = nullptr;
  p.structured_choice = nullptr;
  p.n_structured_choice = 0;
  p.structured_grammar = nullptr;
  p.structured_json_object = 0;
  p.logits_processor = nullptr;
  p.logits_processor_user_data = nullptr;
  return p;
}

VLLM_API vllm_status vllm_engine_load(const vllm_model_params* params,
                                      vllm_engine** out) {
  if (out == nullptr) {
    SetError("vllm_engine_load: out handle pointer is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  *out = nullptr;
  if (params == nullptr || params->model_path == nullptr) {
    SetError("vllm_engine_load: params or params->model_path is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  try {
    vllm::entrypoints::EngineParams ep;
    if (params->block_size > 0) ep.block_size = params->block_size;
    // ABI v16: num_blocks is the OVERRIDE (> 0 pins the count); <= 0 leaves the
    // EngineParams auto default (0) so ResolveNumBlocks applies the sizing knobs.
    if (params->num_blocks > 0) ep.num_blocks = params->num_blocks;
    // ABI v16 KV sizing: gpu_memory_utilization (<= 0 keeps the 0.92 default) and
    // the absolute kv_cache_memory_bytes override (0 => unset).
    if (params->gpu_memory_utilization > 0.0)
      ep.gpu_memory_utilization = params->gpu_memory_utilization;
    if (params->kv_cache_memory_bytes > 0)
      ep.kv_cache_memory_bytes = params->kv_cache_memory_bytes;
    if (params->max_model_len > 0) ep.max_model_len = params->max_model_len;
    if (params->max_num_seqs > 0) ep.max_num_seqs = params->max_num_seqs;
    // ABI v9: chunked-prefill per-step token budget. <= 0 leaves the per-arch
    // default resolution (LoadedEngine::ResolveMaxNumBatchedTokens) untouched.
    if (params->max_num_batched_tokens > 0)
      ep.max_num_batched_tokens = params->max_num_batched_tokens;
    // The config DOCUMENTS below are caller input, so a malformed one is
    // VLLM_ERR_INVALID_ARGUMENT — what vllm.h has documented for
    // speculative_config since v6. Scoped to just the parsing: a genuine
    // std::invalid_argument out of FromModelDir (a bad checkpoint) must keep
    // reporting as VLLM_ERR_MODEL_LOAD.
    try {
      // ABI v9: scheduler admission policy. NULL/empty => the default fcfs.
      if (params->scheduling_policy != nullptr &&
          params->scheduling_policy[0] != '\0') {
        ep.policy = vllm::SchedulerPolicyFromString(params->scheduling_policy);
      }
      // ABI v6: speculative-decoding config. NULL/empty => unset => the
      // byte-identical no-speculation engine.
      if (params->speculative_config != nullptr &&
          params->speculative_config[0] != '\0') {
        ep.speculative_config =
            vllm::ParseSpeculativeConfigJson(params->speculative_config);
      }
      // ABI v9: external KV connector (LMCache). NULL/empty => no connector ==
      // the inert default path. The connector NAME is validated here, mirroring
      // the server's startup check, so a typo reports as a caller error instead
      // of surfacing later as an opaque engine-construction failure.
      if (params->kv_transfer_config != nullptr &&
          params->kv_transfer_config[0] != '\0') {
        vllm::KVTransferConfig kv_cfg =
            vllm::ParseKVTransferConfigJson(params->kv_transfer_config);
        if (kv_cfg.kv_connector.has_value() &&
            !vllm::v1::kv_offload::KVConnectorFactory::IsRegistered(
                *kv_cfg.kv_connector)) {
          std::string msg = "unknown kv_connector \"" + *kv_cfg.kv_connector +
                            "\" (registered connectors: ";
          const std::vector<std::string> names =
              vllm::v1::kv_offload::KVConnectorFactory::RegisteredNames();
          for (size_t n = 0; n < names.size(); ++n) {
            if (n != 0) msg += ", ";
            msg += names[n];
          }
          msg += ")";
          throw std::invalid_argument(msg);
        }
        ep.kv_transfer_config = std::move(kv_cfg);
      }
      // ABI v19: the multimodal input limits (#607 L2). Parsed HERE so a
      // malformed document is a CALLER error reported before any model I/O,
      // exactly as the server refuses --limit-mm-per-prompt before the load.
      // The flag half is deliberately not a second knob: language_model_only
      // zeroes every limit inside GetLimitPerPrompt, ahead of the map
      // (multimodal.py:326-327), so setting both is well-defined and the flag
      // wins — mirrored rather than reimplemented here.
      ep.multimodal.language_model_only = params->language_model_only != 0;
      if (params->limit_mm_per_prompt != nullptr &&
          params->limit_mm_per_prompt[0] != '\0') {
        ep.multimodal.limit_per_prompt = vllm::ParseLimitMmPerPromptJson(
            params->limit_mm_per_prompt, /*ignored_options=*/nullptr);
      }
    } catch (const std::invalid_argument& e) {
      SetError(std::string("vllm_engine_load: ") + e.what());
      return VLLM_ERR_INVALID_ARGUMENT;
    }
    // ABI v7: tri-state prefix-caching toggle (0=model default, 1=on, 2=off).
    // 0 leaves ep.enable_prefix_caching unset (nullopt) so the model-capability
    // default resolves — the byte-identical default. --enable-radix-attention
    // maps to state 1 (RadixAttention is fused into our APC).
    switch (params->enable_prefix_caching) {
      case 0:
        break;  // model default (nullopt).
      case 1:
        ep.enable_prefix_caching = true;
        break;
      case 2:
        ep.enable_prefix_caching = false;
        break;
      default:
        SetError(
            "vllm_engine_load: enable_prefix_caching must be 0 (model "
            "default), 1 (on), or 2 (off)");
        return VLLM_ERR_INVALID_ARGUMENT;
    }
    // ABI v10: tri-state jump-forward toggle (0=default/env-resolved-OFF, 1=on,
    // 2=off), mirroring enable_prefix_caching. 0 leaves ep.enable_jump_forward
    // unset (nullopt), so JumpForwardEnabled resolves it from the environment
    // (default OFF) — the byte-identical default. The VT_ENABLE_JUMP_FORWARD env
    // var still overrides 1/2 at resolution time.
    switch (params->enable_jump_forward) {
      case 0:
        break;  // default (nullopt) => env-resolved, OFF unless VT_ENABLE_JUMP_FORWARD.
      case 1:
        ep.enable_jump_forward = true;
        break;
      case 2:
        ep.enable_jump_forward = false;
        break;
      default:
        SetError(
            "vllm_engine_load: enable_jump_forward must be 0 (default), 1 (on), "
            "or 2 (off)");
        return VLLM_ERR_INVALID_ARGUMENT;
    }
    // ABI v14: explicit device selection (0=auto, 1=cpu, 2=cuda), mirroring
    // vLLM's DeviceConfig.device names (vllm/config/device.py:13). 0 leaves
    // ep.device at kAuto — the byte-identical accelerator-first probe. An
    // explicitly named device that is ABSENT fails inside FromModelDir, before
    // any model I/O, and reports VLLM_ERR_MODEL_LOAD with a message naming the
    // device (never a silent fallback — device.py:61-66).
    switch (params->device) {
      case 0:
        break;  // auto (the default probe) — ep.device stays kAuto.
      case 1:
        ep.device = vllm::Device::kCPU;
        break;
      case 2:
        ep.device = vllm::Device::kNamedPlatform;
        break;
      default:
        SetError(
            "vllm_engine_load: device must be 0 (auto), 1 (cpu), or 2 (cuda)");
        return VLLM_ERR_INVALID_ARGUMENT;
    }

    // ABI v11 task dispatch: a directory whose config.json architectures
    // resolve to a SupportsTranscription-ONLY registration (Parakeet
    // CTC/RNNT/TDT) gets the TRANSCRIPTION stack — the same library seam the
    // server's /v1/audio/transcriptions route and the parakeet-transcribe
    // example drive. The peek is non-throwing and narrow: every other path,
    // including unknown architectures and .gguf files, is byte-identical to
    // pre-v11 (FromModelDir owns the diagnosis).
    if (const std::vector<std::string> archs =
            vllm::PeekHfArchitectures(std::string(params->model_path) +
                                      "/config.json");
        !archs.empty()) {
      const vllm::ModelRegistration* peek = nullptr;
      try {
        peek = &vllm::ModelRegistry::Resolve(
            std::span<const std::string>(archs));
      } catch (const std::exception&) {
        peek = nullptr;
      }
      if (peek != nullptr && peek->info.supports_transcription_only) {
        // ABI v14: the transcription stack is a CPU-hosted pipeline; an
        // explicit CUDA ask cannot be served and is REFUSED rather than
        // silently downgraded (the same never-substitute rule as the text
        // engine, vllm/config/device.py:61-66).
        if (ep.device == vllm::Device::kNamedPlatform) {
          SetError(
              "vllm_engine_load: device 'cuda' was requested but this "
              "transcription-only checkpoint serves on the CPU pipeline; use "
              "device=auto or device=cpu");
          return VLLM_ERR_INVALID_ARGUMENT;
        }
        auto* handle = new vllm_engine;
        handle->transcriber =
            std::make_unique<vllm::multimodal::ParakeetTranscriber>(
                vllm::multimodal::ParakeetTranscriber::FromDir(
                    params->model_path));
        handle->model_path = params->model_path;
        *out = handle;
        ClearError();
        return VLLM_OK;
      }
    }
    auto loaded =
        vllm::entrypoints::LoadedEngine::FromModelDir(params->model_path, ep);
    auto* handle = new vllm_engine;
    handle->loaded = std::move(loaded);
    handle->model_path = params->model_path;
    // ABI v9: an explicit tokenizer_config.json override for the chat template.
    if (params->tokenizer_config_path != nullptr)
      handle->tokenizer_config_path = params->tokenizer_config_path;
    // ABI v4: copy the caller's tool-parser selection (NULL => empty => AUTO).
    if (params->tool_parser != nullptr) handle->tool_parser = params->tool_parser;
    if (params->reasoning_parser != nullptr)
      handle->reasoning_parser = params->reasoning_parser;
    *out = handle;
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_engine_load: ") + e.what());
    return VLLM_ERR_MODEL_LOAD;
  } catch (...) {
    SetError("vllm_engine_load: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API void vllm_engine_free(vllm_engine* engine) { delete engine; }

VLLM_API vllm_status vllm_complete(vllm_engine* engine, const char* prompt,
                                   const vllm_sampling_params* params,
                                   vllm_completion* out) {
  if (out == nullptr) {
    SetError("vllm_complete: out is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  out->text = nullptr;
  out->finish_reason = nullptr;
  out->prompt_tokens = 0;
  out->completion_tokens = 0;
  // Null-check the pointers. We cannot validate a non-null-but-garbage handle
  // across a C ABI; a valid handle (from vllm_engine_load / MakeEngineHandle)
  // always owns a LoadedEngine, so `engine != nullptr` is the contract.
  if (engine == nullptr || prompt == nullptr || params == nullptr) {
    SetError("vllm_complete: engine, prompt or params is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (!RequireTextEngine(engine, "vllm_complete")) return VLLM_ERR_INVALID_ARGUMENT;
  try {
    const vllm::SamplingParams sp =
        ToSamplingParams(*params, vllm::RequestOutputKind::kCumulative);
    vllm::v1::AsyncLLM& e = engine->loaded->async_engine();
    const std::string request_id =
        std::to_string(engine->next_request_id.fetch_add(1));
    // If generate() throws mid-loop the request is torn down by the guard; a
    // clean return means it already finished, so disarm.
    RequestGuard guard{e, request_id};
    const vllm::RequestOutput result = e.generate(prompt, sp, request_id);
    guard.disarm();

    if (result.outputs.empty()) {
      SetError("vllm_complete: engine produced no output sequence");
      return VLLM_ERR_RUNTIME;
    }
    const vllm::CompletionOutput& o = result.outputs[0];

    // The raw-bytes detokenizer can leave invalid/truncated UTF-8 in the text,
    // which a NUL-terminated C string cannot safely carry (embedded NULs would
    // truncate it; invalid bytes break UTF-8 consumers). Sanitize to valid UTF-8
    // (U+FFFD for invalid runs) exactly as the streaming path does — the C ABI
    // always hands out well-formed UTF-8.
    char* text = DupString(vllm::entrypoints::openai::SanitizeUtf8(o.text));
    if (text == nullptr) {
      SetError("vllm_complete: out-of-memory copying completion text");
      return VLLM_ERR_RUNTIME;
    }
    out->text = text;
    out->finish_reason = o.finish_reason.has_value()
                             ? CanonicalFinishReason(*o.finish_reason)
                             : nullptr;
    out->prompt_tokens = static_cast<int32_t>(result.prompt_token_ids.size());
    out->completion_tokens = static_cast<int32_t>(o.token_ids.size());
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_complete: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_complete: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API vllm_status vllm_complete_tokens(
    vllm_engine* engine, const int32_t* prompt_tokens, int32_t n_prompt_tokens,
    const vllm_sampling_params* params, int32_t* out_tokens,
    int32_t max_out_tokens, int32_t* n_out_tokens, vllm_completion* out) {
  if (n_out_tokens != nullptr) *n_out_tokens = 0;
  if (out != nullptr) {
    out->text = nullptr;
    out->finish_reason = nullptr;
    out->prompt_tokens = 0;
    out->completion_tokens = 0;
  }
  if (engine == nullptr || prompt_tokens == nullptr || params == nullptr ||
      n_out_tokens == nullptr) {
    SetError(
        "vllm_complete_tokens: engine, prompt_tokens, params or n_out_tokens "
        "is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (n_prompt_tokens <= 0) {
    SetError("vllm_complete_tokens: n_prompt_tokens must be > 0");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (max_out_tokens < 0) {
    SetError("vllm_complete_tokens: max_out_tokens must be >= 0");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (out_tokens == nullptr && max_out_tokens > 0) {
    SetError("vllm_complete_tokens: out_tokens is null with max_out_tokens > 0");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  // Refuse-by-task (ABI v15 tightening): v13 shipped this entry point without
  // the v11 guard, so a transcription-only handle would deref the null
  // LoadedEngine here; the same guard now also refuses pooling engines.
  if (!RequireTextEngine(engine, "vllm_complete_tokens"))
    return VLLM_ERR_INVALID_ARGUMENT;
  try {
    const vllm::SamplingParams sp =
        ToSamplingParams(*params, vllm::RequestOutputKind::kCumulative);
    vllm::v1::AsyncLLM& e = engine->loaded->async_engine();
    const std::string request_id =
        std::to_string(engine->next_request_id.fetch_add(1));
    RequestGuard guard{e, request_id};
    // The PRE-TOKENIZED add_request overload (vLLM TokensPrompt): builds the
    // request from prompt_token_ids directly, skipping tokenization.
    std::vector<int32_t> ids(prompt_tokens, prompt_tokens + n_prompt_tokens);
    vllm::v1::AsyncRequest request = e.add_request(request_id, std::move(ids), sp);
    vllm::RequestOutput result;
    for (;;) {
      result = e.get_output(request);
      if (result.finished) break;
    }
    guard.disarm();

    if (result.outputs.empty()) {
      SetError("vllm_complete_tokens: engine produced no output sequence");
      return VLLM_ERR_RUNTIME;
    }
    const vllm::CompletionOutput& o = result.outputs[0];
    const int32_t n =
        std::min<int32_t>(static_cast<int32_t>(o.token_ids.size()), max_out_tokens);
    for (int32_t i = 0; i < n; ++i) out_tokens[i] = o.token_ids[static_cast<size_t>(i)];
    *n_out_tokens = n;
    if (out != nullptr) {
      char* text = DupString(vllm::entrypoints::openai::SanitizeUtf8(o.text));
      if (text == nullptr) {
        SetError("vllm_complete_tokens: out-of-memory copying completion text");
        return VLLM_ERR_RUNTIME;
      }
      out->text = text;
      out->finish_reason = o.finish_reason.has_value()
                               ? CanonicalFinishReason(*o.finish_reason)
                               : nullptr;
      out->prompt_tokens = static_cast<int32_t>(result.prompt_token_ids.size());
      out->completion_tokens = static_cast<int32_t>(o.token_ids.size());
    }
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_complete_tokens: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_complete_tokens: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API vllm_status vllm_complete_stream(vllm_engine* engine,
                                          const char* prompt,
                                          const vllm_sampling_params* params,
                                          vllm_token_callback cb,
                                          void* user_data) {
  if (engine == nullptr || prompt == nullptr || params == nullptr ||
      cb == nullptr) {
    SetError("vllm_complete_stream: engine, prompt, params or cb is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (!RequireTextEngine(engine, "vllm_complete_stream")) return VLLM_ERR_INVALID_ARGUMENT;
  try {
    // DELTA output_kind: each step yields one incremental delta (mirrors the M3.1
    // OpenAI streaming path, serving_completion.cpp). PostInit runs inside.
    const vllm::SamplingParams sp =
        ToSamplingParams(*params, vllm::RequestOutputKind::kDelta);

    // One prompt per call -> its own FRESH request id. add_request + drive step()
    // directly (like serving_completion) — no new LLMEngine driver needed. The
    // guard tears the request down on EVERY exit path (early-stop, a throwing
    // callback, or a mid-stream step() error) so the engine stays reusable.
    vllm::v1::AsyncLLM& e = engine->loaded->async_engine();
    const std::string request_id =
        std::to_string(engine->next_request_id.fetch_add(1));
    RequestGuard guard{e, request_id};
    vllm::v1::AsyncRequest request = e.add_request(request_id, prompt, sp);

    bool stopped_by_callback = false;
    for (;;) {
      const vllm::RequestOutput res = e.get_output(request);
      for (const vllm::CompletionOutput& output : res.outputs) {
        // SanitizeUtf8: the raw-bytes detokenizer can emit invalid/truncated
        // UTF-8; the callback gets a well-formed, NUL-terminated C string
        // (embedded NULs cannot survive a C string either). See serving_utils.
        const std::string delta =
            vllm::entrypoints::openai::SanitizeUtf8(output.text);
        const bool finished = res.finished;
        // Callback owns the borrow only for this call; false => stop early.
        const bool keep_going = cb(delta.c_str(), finished, user_data);
        if (!keep_going) {
          stopped_by_callback = true;
          break;
        }
      }
      if (stopped_by_callback || res.finished) break;
    }

    // On early stop the guard aborts the in-flight request. On a natural finish
    // the request already left the engine (the DELTA path delivered a final
    // res.finished==true delta to the callback), so the guard's abort would be a
    // no-op — but disarm to skip it. If a step()/callback threw, we never reach
    // here and the guard fires during unwind.
    if (!stopped_by_callback) guard.disarm();

    ClearError();
    return VLLM_OK;
  } catch (const std::exception& ex) {
    SetError(std::string("vllm_complete_stream: ") + ex.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_complete_stream: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API vllm_status vllm_request_submit(
    vllm_engine* engine, const char* prompt,
    const vllm_sampling_params* params, vllm_token_callback cb,
    void* user_data, vllm_request** out) {
  if (out == nullptr) {
    SetError("vllm_request_submit: out is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  *out = nullptr;
  if (engine == nullptr || prompt == nullptr || params == nullptr ||
      cb == nullptr) {
    SetError("vllm_request_submit: engine, prompt, params or cb is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (!RequireTextEngine(engine, "vllm_request_submit")) return VLLM_ERR_INVALID_ARGUMENT;
  try {
    vllm::SamplingParams sp =
        ToSamplingParams(*params, vllm::RequestOutputKind::kDelta);
    vllm::v1::AsyncLLM& async = engine->loaded->async_engine();
    const std::string request_id =
        std::to_string(engine->next_request_id.fetch_add(1));
    vllm::v1::AsyncRequest async_request =
        async.add_request(request_id, prompt, std::move(sp));
    RequestGuard guard{async, request_id};

    auto request = std::make_unique<vllm_request>();
    request->parent = engine;
    request->async_request = std::move(async_request);
    request->callback = cb;
    request->user_data = user_data;
    request->delivery_thread =
        std::thread(RunRequestDelivery, request.get());
    *out = request.release();
    guard.disarm();
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_request_submit: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_request_submit: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API vllm_status vllm_request_cancel(vllm_request* request) {
  if (request == nullptr) {
    SetError("vllm_request_cancel: request is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  try {
    if (!request->done.load() && !request->cancelled.exchange(true)) {
      request->parent->loaded->async_engine().abort(
          request->async_request.request_id);
    }
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_request_cancel: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_request_cancel: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API vllm_status vllm_request_wait(vllm_request* request) {
  if (request == nullptr) {
    SetError("vllm_request_wait: request is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (request->delivery_thread.get_id() == std::this_thread::get_id()) {
    SetError("vllm_request_wait: cannot wait from the request callback");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  try {
    JoinRequest(request);
    if (request->status != VLLM_OK) {
      SetError(std::string("vllm_request_wait: ") + request->error);
    } else {
      ClearError();
    }
    return request->status;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_request_wait: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_request_wait: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API bool vllm_request_done(const vllm_request* request) {
  return request != nullptr &&
         request->done.load(std::memory_order_acquire);
}

VLLM_API const char* vllm_request_error(const vllm_request* request) {
  static const char* kEmpty = "";
  if (request == nullptr ||
      !request->done.load(std::memory_order_acquire)) {
    return kEmpty;
  }
  return request->error.c_str();
}

VLLM_API void vllm_request_free(vllm_request* request) {
  if (request == nullptr) return;
  if (request->delivery_thread.get_id() == std::this_thread::get_id()) {
    SetError("vllm_request_free: cannot free from the request callback");
    return;
  }
  try {
    if (!request->done.load()) {
      (void)vllm_request_cancel(request);
    }
    JoinRequest(request);
    delete request;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_request_free: ") + e.what());
  } catch (...) {
    SetError("vllm_request_free: unknown error");
  }
}

VLLM_API vllm_status vllm_chat(vllm_engine* engine, const char* request_json,
                               char** out_response_json) {
  if (out_response_json == nullptr) {
    SetError("vllm_chat: out_response_json is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  *out_response_json = nullptr;
  if (engine == nullptr || request_json == nullptr) {
    SetError("vllm_chat: engine or request_json is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (!RequireTextEngine(engine, "vllm_chat")) return VLLM_ERR_INVALID_ARGUMENT;
  try {
    vllm::entrypoints::openai::ChatCompletionRequest request =
        ParseChatRequest(request_json);
    request.stream = false;
    vllm::entrypoints::openai::ChatCompletionResult result =
        EnsureChatServing(engine).create_chat_completion(request);
    if (!result.response.has_value()) {
      SetError("vllm_chat: serving produced no response");
      return VLLM_ERR_RUNTIME;
    }
    const nlohmann::json j = *result.response;
    char* text = DupString(j.dump());
    if (text == nullptr) {
      SetError("vllm_chat: out-of-memory copying response");
      return VLLM_ERR_RUNTIME;
    }
    *out_response_json = text;
    ClearError();
    return VLLM_OK;
  } catch (const std::invalid_argument& e) {
    SetError(std::string("vllm_chat: ") + e.what());
    return VLLM_ERR_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_chat: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_chat: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API vllm_status vllm_chat_stream(vllm_engine* engine,
                                      const char* request_json,
                                      vllm_token_callback cb,
                                      void* user_data) {
  if (engine == nullptr || request_json == nullptr || cb == nullptr) {
    SetError("vllm_chat_stream: engine, request_json or cb is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (!RequireTextEngine(engine, "vllm_chat_stream")) return VLLM_ERR_INVALID_ARGUMENT;
  try {
    vllm::entrypoints::openai::ChatCompletionRequest request =
        ParseChatRequest(request_json);
    request.stream = true;
    vllm::entrypoints::openai::ChatCompletionResult result =
        EnsureChatServing(engine).create_chat_completion(request);

    bool stopped_by_callback = false;
    auto deliver = [&](const std::string& framed) -> bool {
      const std::string payload = StripSseFraming(framed);
      if (payload.empty() || payload == "[DONE]") return true;
      // Callback contract mirrors vllm_complete_stream: borrowed, per-chunk.
      return cb(payload.c_str(), false, user_data);
    };

    if (result.sse_stream != nullptr) {
      // Live AsyncLLM path: pull framed chunks as the engine produces them.
      std::string chunk;
      while (result.sse_stream->next(chunk)) {
        if (!deliver(chunk)) {
          stopped_by_callback = true;
          result.sse_stream->abort();
          break;
        }
      }
    } else {
      for (const std::string& chunk : result.sse_chunks) {
        if (!deliver(chunk)) {
          stopped_by_callback = true;
          break;
        }
      }
    }
    if (!stopped_by_callback) {
      // Terminal call, matching the vllm_token_callback contract.
      (void)cb("", true, user_data);
    }
    ClearError();
    return VLLM_OK;
  } catch (const std::invalid_argument& e) {
    SetError(std::string("vllm_chat_stream: ") + e.what());
    return VLLM_ERR_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_chat_stream: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_chat_stream: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

// ── Audio transcription (ABI v11) ───────────────────────────────────────────

VLLM_API vllm_transcription_params vllm_transcription_params_default(void) {
  vllm_transcription_params p;
  p.audio_path = nullptr;
  p.pcm = nullptr;
  p.n_samples = 0;
  p.sample_rate = 0;
  return p;
}

VLLM_API vllm_status vllm_transcribe(vllm_engine* engine,
                                     const vllm_transcription_params* params,
                                     vllm_transcription* out) {
  if (out == nullptr) {
    SetError("vllm_transcribe: out is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  out->text = nullptr;
  out->token_ids = nullptr;
  out->n_token_ids = 0;
  out->has_text = 0;
  if (engine == nullptr || params == nullptr) {
    SetError("vllm_transcribe: engine or params is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (engine->transcriber == nullptr) {
    SetError(
        "vllm_transcribe: this engine is a text-generation engine (no "
        "SupportsTranscription architecture); use the completion/chat entry "
        "points, or load a Parakeet checkpoint for transcription");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  const bool has_path = params->audio_path != nullptr;
  const bool has_pcm = params->pcm != nullptr;
  if (has_path == has_pcm) {
    SetError(
        "vllm_transcribe: set exactly ONE input — audio_path (a 16-bit PCM "
        "mono WAV) or pcm+n_samples+sample_rate");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (has_pcm && (params->n_samples <= 0 || params->sample_rate <= 0)) {
    SetError("vllm_transcribe: pcm requires n_samples > 0 and sample_rate > 0");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  try {
    const vllm::multimodal::ParakeetTranscription result =
        has_path ? engine->transcriber->TranscribeWavFile(params->audio_path)
                 : engine->transcriber->Transcribe(params->pcm,
                                                   params->n_samples,
                                                   params->sample_rate);
    int32_t* ids = nullptr;
    if (!result.token_ids.empty()) {
      ids = static_cast<int32_t*>(
          std::malloc(result.token_ids.size() * sizeof(int32_t)));
      if (ids == nullptr) {
        SetError("vllm_transcribe: out-of-memory copying token ids");
        return VLLM_ERR_RUNTIME;
      }
      std::memcpy(ids, result.token_ids.data(),
                  result.token_ids.size() * sizeof(int32_t));
    }
    if (result.has_text) {
      char* text = DupString(result.text);
      if (text == nullptr) {
        std::free(ids);
        SetError("vllm_transcribe: out-of-memory copying transcript");
        return VLLM_ERR_RUNTIME;
      }
      out->text = text;
      out->has_text = 1;
    }
    out->token_ids = ids;
    out->n_token_ids = static_cast<int32_t>(result.token_ids.size());
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    // One catch for the whole pipeline (unreadable/undecodable audio, wrong
    // sample rate, forward failure): VLLM_ERR_RUNTIME with the cause named,
    // matching vllm_complete's convention. The cheap argument-shape errors
    // were already reported as VLLM_ERR_INVALID_ARGUMENT above.
    SetError(std::string("vllm_transcribe: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_transcribe: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API void vllm_transcription_free(vllm_transcription* out) {
  if (out == nullptr) return;
  std::free(out->text);
  std::free(out->token_ids);
  out->text = nullptr;
  out->token_ids = nullptr;
  out->n_token_ids = 0;
  out->has_text = 0;
}

// ── Embeddings (ABI v15, ARCH-ONE-SURFACE ROW 6) ────────────────────────────
// The pooling slice of the ONE surface: the SAME registry forward +
// PoolingRunner engine step the server's /v1/embeddings drives
// (LLMEngine::embed -> pool_tokens, the mirror of LLM.embed /
// entrypoints/pooling/offline.py:65-119 with pooling_task="embed").

VLLM_API vllm_status vllm_embed(vllm_engine* engine, const char* const* texts,
                                int32_t n_texts, vllm_embedding_result* out) {
  if (out == nullptr) {
    SetError("vllm_embed: out is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  out->values = nullptr;
  out->n_embeddings = 0;
  out->dim = 0;
  out->prompt_tokens = 0;
  if (engine == nullptr || texts == nullptr) {
    SetError("vllm_embed: engine or texts is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (n_texts <= 0) {
    SetError("vllm_embed: n_texts must be > 0");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  for (int32_t i = 0; i < n_texts; ++i) {
    if (texts[i] == nullptr) {
      SetError("vllm_embed: texts[" + std::to_string(i) + "] is null");
      return VLLM_ERR_INVALID_ARGUMENT;
    }
  }
  // Refuse-by-task, the other direction of RequireTextEngine: only a POOLING
  // (embedding) engine serves this entry point — the mirror of vLLM refusing
  // `--runner pooling` on a non-pooling model (config/model.py:612-617).
  if (engine->loaded == nullptr) {
    SetError(
        "vllm_embed: this engine was loaded from a transcription-only "
        "checkpoint (Parakeet); use vllm_transcribe");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (!engine->loaded->is_pooling_model()) {
    SetError(
        "vllm_embed: this engine was loaded from a text-generation "
        "checkpoint; it has no pooling path — use vllm_complete / vllm_chat "
        "(embedding checkpoints resolve to a pooling architecture, e.g. "
        "LlamaModel)");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  try {
    // Serialize embed batches per handle: the pooling path drives the
    // SYNCHRONOUS LLMEngine step loop (async scheduling resolves OFF for
    // pooling models, config/vllm.py:1068-1073 mirror).
    std::lock_guard<std::mutex> lock(engine->embed_mutex);
    const vllm::tok::Tokenizer& tokenizer = engine->loaded->tokenizer();
    vllm::v1::LLMEngine& e = engine->loaded->engine();

    std::vector<std::vector<float>> vectors;
    vectors.reserve(static_cast<size_t>(n_texts));
    int64_t total_prompt_tokens = 0;
    for (int32_t i = 0; i < n_texts; ++i) {
      // The serving tokenization applies the template's special tokens
      // (add_special_tokens=True on the OpenAI embedding path).
      std::vector<int32_t> ids = tokenizer.EncodeWithSpecialTokens(texts[i]);
      if (ids.empty()) {
        SetError("vllm_embed: texts[" + std::to_string(i) +
                 "] tokenized to an empty prompt");
        return VLLM_ERR_INVALID_ARGUMENT;
      }
      total_prompt_tokens += static_cast<int64_t>(ids.size());
      const std::string request_id =
          "embed-" + std::to_string(engine->next_request_id.fetch_add(1));
      vllm::RequestOutput ro =
          e.embed(std::move(ids), vllm::PoolingParams{}, request_id);
      if (!ro.finished || !ro.pooling_output.has_value()) {
        SetError("vllm_embed: engine produced no pooled output");
        return VLLM_ERR_RUNTIME;
      }
      vectors.push_back(std::move(*ro.pooling_output));
    }

    const size_t dim = vectors.empty() ? 0 : vectors[0].size();
    for (const std::vector<float>& v : vectors) {
      if (v.size() != dim || dim == 0) {
        SetError("vllm_embed: inconsistent embedding dimensions");
        return VLLM_ERR_RUNTIME;
      }
    }
    float* values = static_cast<float*>(
        std::malloc(static_cast<size_t>(n_texts) * dim * sizeof(float)));
    if (values == nullptr) {
      SetError("vllm_embed: out-of-memory copying embeddings");
      return VLLM_ERR_RUNTIME;
    }
    for (int32_t i = 0; i < n_texts; ++i) {
      std::memcpy(values + static_cast<size_t>(i) * dim,
                  vectors[static_cast<size_t>(i)].data(),
                  dim * sizeof(float));
    }
    out->values = values;
    out->n_embeddings = n_texts;
    out->dim = static_cast<int32_t>(dim);
    out->prompt_tokens = static_cast<int32_t>(total_prompt_tokens);
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_embed: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_embed: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API void vllm_embedding_result_free(vllm_embedding_result* out) {
  if (out == nullptr) return;
  std::free(out->values);
  out->values = nullptr;
  out->n_embeddings = 0;
  out->dim = 0;
  out->prompt_tokens = 0;
}

// ── Video+audio generation (ABI v12; generalized at v18) ────────────────────
// Thin C wrappers over the ONE library seam — since v18 the ABSTRACT one
// (vllm::multimodal::VideoEngine + LoadVideoEngine's family registry), which
// the server's /v1/videos routes and the minimax-h3-gen example also drive.
// See include/vllm.h for the contract.

VLLM_API vllm_video_model_params vllm_video_model_params_default(void) {
  vllm_video_model_params p;
  std::memset(&p, 0, sizeof(p));
  return p;
}

VLLM_API vllm_video_params vllm_video_params_default(void) {
  vllm_video_params p;
  std::memset(&p, 0, sizeof(p));
  return p;
}

// The opaque video handle: owns the loaded checkpoint set + staged weights.
// `family` caches the resolved name so vllm_video_engine_family can hand back
// storage that lives exactly as long as the handle.
struct vllm_video_engine {
  std::unique_ptr<vllm::multimodal::VideoEngine> engine;
  std::string family;
};

namespace {

// Lower the v18 parallel key/value arrays into the seam's extras map. Returns
// false with *error set when the arrays are malformed — a NULL entry is a
// caller bug, and reading past it would be undefined rather than merely wrong.
bool VideoExtrasFromArrays(const char* const* keys, const char* const* values, int32_t n,
                           const char* field, std::map<std::string, std::string>* out,
                           std::string* error) {
  if (n == 0) return true;
  if (n < 0) {
    *error = std::string(field) + ": n_extras is negative";
    return false;
  }
  if (keys == nullptr || values == nullptr) {
    *error = std::string(field) + ": n_extras > 0 but extra_keys/extra_values is null";
    return false;
  }
  for (int32_t i = 0; i < n; ++i) {
    if (keys[i] == nullptr || values[i] == nullptr || keys[i][0] == '\0') {
      *error = std::string(field) + ": extra " + std::to_string(i) +
               " has a null/empty key or a null value";
      return false;
    }
    (*out)[keys[i]] = values[i];
  }
  return true;
}

}  // namespace

VLLM_API vllm_status vllm_video_engine_load(const vllm_video_model_params* params,
                                            vllm_video_engine** out) {
  if (out == nullptr) {
    SetError("vllm_video_engine_load: out handle pointer is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  *out = nullptr;
  if (params == nullptr || params->dit_path == nullptr || params->dit_path[0] == '\0') {
    SetError("vllm_video_engine_load: params or params->dit_path is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  vllm::multimodal::VideoModelParams mp;
  mp.dit_path = OrEmpty(params->dit_path);
  mp.encoder_path = OrEmpty(params->encoder_path);
  mp.tokenizer_path = OrEmpty(params->tokenizer_path);
  mp.video_vae_path = OrEmpty(params->video_vae_path);
  mp.video_vae_config_path = OrEmpty(params->video_vae_config_path);
  mp.audio_vae_path = OrEmpty(params->audio_vae_path);
  mp.audio_vae_config_path = OrEmpty(params->audio_vae_config_path);
  mp.prompt_embeds_path = OrEmpty(params->prompt_embeds_path);
  mp.family = OrEmpty(params->family);
  mp.device = params->device;
  mp.dequant_bf16 = params->dequant_bf16;
  mp.fp4_resident = params->fp4_resident;

  std::string extras_error;
  if (!VideoExtrasFromArrays(params->extra_keys, params->extra_values, params->n_extras,
                             "vllm_video_engine_load", &mp.extras, &extras_error)) {
    SetError(extras_error);
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  // The v12 `partition` field is the documented ALIAS for the load extra
  // "partition". Both spellings disagreeing is a caller error with no defensible
  // winner, so it is refused rather than resolved silently.
  const std::string partition = OrEmpty(params->partition);
  if (!partition.empty()) {
    const auto it = mp.extras.find("partition");
    if (it != mp.extras.end() && it->second != partition) {
      SetError("vllm_video_engine_load: params->partition ('" + partition +
               "') and the extra \"partition\" ('" + it->second +
               "') disagree; supply one spelling");
      return VLLM_ERR_INVALID_ARGUMENT;
    }
    mp.extras["partition"] = partition;
  }

  try {
    auto handle = std::make_unique<vllm_video_engine>();
    handle->engine = vllm::multimodal::LoadVideoEngine(mp);
    handle->family = handle->engine->family();
    *out = handle.release();
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_video_engine_load: ") + e.what());
    return VLLM_ERR_MODEL_LOAD;
  } catch (...) {
    SetError("vllm_video_engine_load: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API void vllm_video_engine_free(vllm_video_engine* engine) { delete engine; }

VLLM_API const char* vllm_video_engine_family(const vllm_video_engine* engine) {
  return engine == nullptr ? nullptr : engine->family.c_str();
}

namespace {

// Copy an argv into the malloc'd, NULL-terminated shape the ABI promises.
// Returns false on OOM (with everything allocated so far freed).
bool DupArgv(const std::vector<std::string>& argv, char*** out_argv, int32_t* out_argc) {
  char** arr = static_cast<char**>(std::calloc(argv.size() + 1, sizeof(char*)));
  if (arr == nullptr) return false;
  for (size_t i = 0; i < argv.size(); ++i) {
    arr[i] = DupString(argv[i]);
    if (arr[i] == nullptr) {
      for (size_t k = 0; k < i; ++k) std::free(arr[k]);
      std::free(arr);
      return false;
    }
  }
  arr[argv.size()] = nullptr;  // execvp-ready
  *out_argv = arr;
  *out_argc = static_cast<int32_t>(argv.size());
  return true;
}

}  // namespace

VLLM_API vllm_status vllm_video_generate(vllm_video_engine* engine,
                                         const vllm_video_params* params,
                                         vllm_video_result* out) {
  if (out == nullptr) {
    SetError("vllm_video_generate: out is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  std::memset(out, 0, sizeof(*out));
  if (engine == nullptr || params == nullptr) {
    SetError("vllm_video_generate: engine or params is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  if (params->output_dir == nullptr || params->output_dir[0] == '\0') {
    SetError("vllm_video_generate: output_dir is required (frames + WAV land there)");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  vllm::multimodal::VideoGenParams gen;
  {
    std::string extras_error;
    if (!VideoExtrasFromArrays(params->extra_keys, params->extra_values, params->n_extras,
                               "vllm_video_generate", &gen.extras, &extras_error)) {
      SetError(extras_error);
      return VLLM_ERR_INVALID_ARGUMENT;
    }
  }
  try {
    gen.prompt = OrEmpty(params->prompt);
    gen.width = params->width;
    gen.height = params->height;
    gen.num_frames = params->num_frames;
    gen.steps = params->steps;
    gen.seed = params->seed;
    gen.has_seed = params->has_seed != 0;
    gen.first_frame_path = OrEmpty(params->first_frame);
    gen.last_frame_path = OrEmpty(params->last_frame);
    // A zeroed struct must preserve behaviour: <= 0 resolves to the exact-pin
    // default (1.0), the value every pre-v12 consumer used.
    gen.noise_aug = params->noise_aug > 0.0f ? static_cast<double>(params->noise_aug) : 1.0;
    if (params->ref_image != nullptr && params->ref_image[0] != '\0') {
      gen.ref_image_paths.push_back(params->ref_image);
    }
    gen.ref_video_dir = OrEmpty(params->ref_video);
    gen.ref_audio_path = OrEmpty(params->ref_audio);
    gen.output_dir = params->output_dir;

    const vllm::multimodal::VideoResult result = engine->engine->Generate(gen);

    vllm_video_result r;
    std::memset(&r, 0, sizeof(r));
    r.frame_dir = DupString(result.frame_dir);
    r.audio_path = DupString(result.audio_path);
    const bool argv_ok = DupArgv(result.mux_argv, &r.mux_argv, &r.mux_argc);
    if (r.frame_dir == nullptr || r.audio_path == nullptr || !argv_ok) {
      vllm_video_result_free(&r);
      SetError("vllm_video_generate: out-of-memory copying the result");
      return VLLM_ERR_RUNTIME;
    }
    r.frame_count = static_cast<int32_t>(result.frame_count);
    r.width = static_cast<int32_t>(result.width);
    r.height = static_cast<int32_t>(result.height);
    r.fps = static_cast<int32_t>(result.fps);
    r.sample_rate = static_cast<int32_t>(result.sample_rate);
    *out = r;
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_video_generate: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_video_generate: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API void vllm_video_result_free(vllm_video_result* out) {
  if (out == nullptr) return;
  std::free(out->frame_dir);
  std::free(out->audio_path);
  if (out->mux_argv != nullptr) {
    for (int32_t i = 0; i < out->mux_argc; ++i) std::free(out->mux_argv[i]);
    std::free(out->mux_argv);
  }
  std::memset(out, 0, sizeof(*out));
}

VLLM_API vllm_video_mux_params vllm_video_mux_params_default(void) {
  vllm_video_mux_params p;
  std::memset(&p, 0, sizeof(p));
  return p;
}

VLLM_API vllm_status vllm_video_mux_argv(const vllm_video_mux_params* params,
                                         char*** out_argv, int32_t* out_argc) {
  if (out_argv == nullptr || out_argc == nullptr) {
    SetError("vllm_video_mux_argv: out_argv/out_argc is null");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  *out_argv = nullptr;
  *out_argc = 0;
  if (params == nullptr || params->frames == nullptr || params->frames[0] == '\0' ||
      params->output_path == nullptr || params->output_path[0] == '\0') {
    SetError("vllm_video_mux_argv: frames (a printf-style pattern) and output_path are required");
    return VLLM_ERR_INVALID_ARGUMENT;
  }
  try {
    vllm::MiniMaxH3MuxRequest request;  // the library's encoding contract
    request.frame_pattern = params->frames;
    request.audio_path = OrEmpty(params->audio_path);
    request.output_path = params->output_path;
    if (params->fps > 0) request.fps = params->fps;
    if (params->crf > 0) request.crf = params->crf;
    const std::vector<std::string> argv = vllm::MiniMaxH3BuildMp4MuxArgs(request);
    if (!DupArgv(argv, out_argv, out_argc)) {
      SetError("vllm_video_mux_argv: out-of-memory copying the argv");
      return VLLM_ERR_RUNTIME;
    }
    ClearError();
    return VLLM_OK;
  } catch (const std::exception& e) {
    SetError(std::string("vllm_video_mux_argv: ") + e.what());
    return VLLM_ERR_RUNTIME;
  } catch (...) {
    SetError("vllm_video_mux_argv: unknown error");
    return VLLM_ERR_UNKNOWN;
  }
}

VLLM_API void vllm_video_mux_argv_free(char** argv, int32_t argc) {
  if (argv == nullptr) return;
  for (int32_t i = 0; i < argc; ++i) std::free(argv[i]);
  std::free(argv);
}

VLLM_API void vllm_string_free(char* s) { std::free(s); }

VLLM_API void vllm_completion_free(vllm_completion* out) {
  if (out == nullptr) return;
  std::free(out->text);
  out->text = nullptr;
  out->finish_reason = nullptr;
  out->prompt_tokens = 0;
  out->completion_tokens = 0;
}

VLLM_API const char* vllm_last_error(void) { return g_last_error.c_str(); }

VLLM_API const char* vllm_version(void) {
  // Static storage: computed once, borrowed by the caller (never freed).
  static const std::string kVersion = vllm::Version();
  return kVersion.c_str();
}

VLLM_API int32_t vllm_server_main(int32_t argc, char** argv) {
#ifdef VLLM_CPP_SERVER
  // The server owns its own error reporting on stderr (it is a PROCESS entry
  // point, not a request call), so this does not set vllm_last_error. What it
  // must guarantee is that nothing throws across the C boundary.
  try {
    return static_cast<int32_t>(
        vllm::entrypoints::openai::VllmServerMain(static_cast<int>(argc), argv));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vllm_server_main: %s\n", e.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "vllm_server_main: unknown error\n");
    return 1;
  }
#else
  // The server translation unit is compiled ONLY under VLLM_CPP_SERVER (it pulls
  // in the vendored httplib transport). The SYMBOL still has to exist in every
  // build: it is part of ABI v17, and a consumer that dlopen's the library and
  // resolves entry points must find it rather than fail to load. So this arm
  // reports the missing capability instead of the symbol going missing.
  //
  // Getting this wrong broke a real downstream build: LocalAI links libvllm with
  // the server OFF, and the unconditional call left
  // vllm::entrypoints::openai::VllmServerMain undefined at dylib link time.
  (void)argc;
  (void)argv;
  SetError(
      "vllm_server_main: this library was built without VLLM_CPP_SERVER; "
      "rebuild with -DVLLM_CPP_SERVER=ON to serve HTTP");
  std::fprintf(stderr, "%s\n", vllm_last_error());
  return 1;
#endif
}

VLLM_API int32_t vllm_abi_version(void) { return VLLM_ABI_VERSION; }

}  // extern "C"

namespace vllm::capi {

vllm_engine* MakeEngineHandle(
    std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded) noexcept {
  if (loaded == nullptr) return nullptr;
  auto* handle = new (std::nothrow) vllm_engine;
  if (handle != nullptr) handle->loaded = std::move(loaded);
  return handle;
}

vllm_engine* MakeEngineHandle(
    std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded,
    vllm::entrypoints::openai::ChatPromptFn prompt_fn) noexcept {
  vllm_engine* handle = MakeEngineHandle(std::move(loaded));
  if (handle != nullptr) handle->test_prompt_fn = std::move(prompt_fn);
  return handle;
}

void SetEngineToolParser(vllm_engine* handle, const std::string& name) noexcept {
  if (handle != nullptr) handle->tool_parser = name;
}

void SetEngineReasoningParser(vllm_engine* engine, std::string name) noexcept {
  if (engine != nullptr) engine->reasoning_parser = std::move(name);
}

}  // namespace vllm::capi
