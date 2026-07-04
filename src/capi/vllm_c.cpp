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
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

#include "capi/engine_handle.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/openai/serving_utils.h"
#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/version.h"

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
};

namespace {

// RAII: aborts an in-flight request on scope exit (incl. exception unwind) unless
// disarmed after a clean finish. abort_request is a safe no-op on an already-
// finished/unknown id, and we swallow any exception so the noexcept dtor can't
// std::terminate during unwind. This guarantees no request is left registered
// after a throwing callback or a mid-stream runtime error.
struct RequestGuard {
  vllm::v1::LLMEngine& engine;
  std::string id;
  bool armed = true;
  ~RequestGuard() {
    if (!armed) return;
    try {
      engine.abort_request(id);
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
  sp.output_kind = output_kind;
  sp.PostInit();
  return sp;
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
    auto* handle = new vllm_engine{std::move(loaded)};
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
    vllm::v1::LLMEngine& e = engine->loaded->engine();
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
    vllm::v1::LLMEngine& e = engine->loaded->engine();
    const std::string request_id =
        std::to_string(engine->next_request_id.fetch_add(1));
    RequestGuard guard{e, request_id};
    e.add_request(request_id, prompt, sp);

    bool stopped_by_callback = false;
    while (e.has_unfinished_requests()) {
      for (const vllm::RequestOutput& res : e.step()) {
        if (res.request_id != request_id) continue;
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
        if (stopped_by_callback) break;
      }
      if (stopped_by_callback) break;
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
  return new (std::nothrow) vllm_engine{std::move(loaded)};
}

}  // namespace vllm::capi
