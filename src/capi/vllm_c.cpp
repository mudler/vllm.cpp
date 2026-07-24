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
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "capi/chat_prompt.h"
#include "capi/engine_handle.h"
#include "vllm/entrypoints/chat_template.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"  // get_tool_parser
#include "vllm/entrypoints/openai/tool_parsers/detect.h"    // DetectToolParser
#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/version.h"
#include "vllm/v1/engine/async_llm.h"

// The opaque handle: owns the whole C++ engine stack behind LoadedEngine.
struct vllm_engine {
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded;
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
  // Test-hook override for the chat-prompt seam (MakeEngineHandle overload):
  // when set, chat_serving is built with it instead of the resolved template.
  vllm::entrypoints::openai::ChatPromptFn test_prompt_fn;
  // The caller-selected tool-call parser (ABI v4 vllm_model_params.tool_parser),
  // copied at vllm_engine_load. Empty => AUTO-detect from the chat template at
  // the first chat call; a non-empty value must name a registered parser or the
  // first chat call fails with VLLM_ERR_INVALID_ARGUMENT (checked in
  // EnsureChatServing). Tests set it via vllm::capi::SetEngineToolParser.
  std::string tool_parser;
  // Lazily-built chat serving handler over the shared AsyncLLM (one per
  // handle; create_chat_completion is safe for concurrent callers, matching
  // the HTTP server's worker pool). Guarded by chat_mutex for the lazy build.
  std::mutex chat_mutex;
  std::unique_ptr<vllm::entrypoints::openai::OpenAIServingChat> chat_serving;
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
    std::string* out_raw_template = nullptr) {
  namespace fs = std::filesystem;
  try {
    std::string tmpl;
    if (fs::is_regular_file(model_path) &&
        fs::path(model_path).extension() == ".gguf") {
      tmpl = vllm::entrypoints::LoadChatTemplateFromGguf(model_path);
    } else {
      tmpl = vllm::entrypoints::LoadChatTemplateFromConfig(
          (fs::path(model_path) / "tokenizer_config.json").string());
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
                                  &raw_template);

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

    std::string served_name =
        engine->model_path.empty()
            ? std::string("model")
            : std::filesystem::path(engine->model_path).filename().string();
    engine->chat_serving =
        std::make_unique<vllm::entrypoints::openai::OpenAIServingChat>(
            engine->loaded->async_engine(), std::move(served_name),
            std::move(prompt_fn), std::move(parser_name));
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

extern "C" {

VLLM_API vllm_model_params vllm_model_params_default(void) {
  vllm_model_params p;
  p.model_path = nullptr;
  p.tokenizer_config_path = nullptr;
  p.block_size = 32;
  p.num_blocks = 256;
  p.max_model_len = 0;
  p.max_num_seqs = 8;
  p.tool_parser = nullptr;  // AUTO-detect from the chat template (ABI v4).
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
    if (params->num_blocks > 0) ep.num_blocks = params->num_blocks;
    if (params->max_model_len > 0) ep.max_model_len = params->max_model_len;
    if (params->max_num_seqs > 0) ep.max_num_seqs = params->max_num_seqs;

    auto loaded =
        vllm::entrypoints::LoadedEngine::FromModelDir(params->model_path, ep);
    auto* handle = new vllm_engine;
    handle->loaded = std::move(loaded);
    handle->model_path = params->model_path;
    // ABI v4: copy the caller's tool-parser selection (NULL => empty => AUTO).
    if (params->tool_parser != nullptr) handle->tool_parser = params->tool_parser;
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

}  // namespace vllm::capi
