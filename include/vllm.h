/*
 * vllm.h — the vllm.cpp stable C ABI (libvllm).
 *
 * ORIGINAL packaging layer — NOT a 1:1 upstream mirror. vLLM ships no C ABI;
 * this header is the llama.cpp-style (see llama.h) handle-based C surface over
 * the C++ LLMEngine, so downstream FFI consumers (LocalAI via purego/cgo, any
 * C/C++ host) can load + drive vllm.cpp without the C++ headers. Recorded as a
 * deviation in .agents/porting-inventory.md §9 (C-ABI packaging, alongside the
 * vt:: runtime and the cpp-httplib transport).
 *
 * ── ABI contract ────────────────────────────────────────────────────────────
 * - PURE C: this header compiles as C (C11) and as C++. It uses only C types —
 *   opaque handle typedefs, POD param/result structs, primitive scalars and
 *   `const char*`. No C++ leaks across the boundary.
 * - NO-THROW: every entry point catches all C++ exceptions internally and maps
 *   them to a `vllm_status`; nothing throws across the ABI. On a non-OK status,
 *   `vllm_last_error()` returns a human-readable message (thread-local).
 * - OWNERSHIP is documented per pointer below. In short: the caller frees
 *   heap `char*` returned in results (via vllm_string_free / vllm_completion_free)
 *   and frees engine handles (via vllm_engine_free); `const char*` returns
 *   (finish_reason, vllm_last_error, vllm_version) point to storage the library
 *   owns and the caller must NOT free.
 * - Scope (M3.5 Task 1): load a model + blocking completion. Streaming callback
 *   + sampled generation land in Task 2; the shared/static lib packaging + CLI
 *   + dlopen smoke test in Task 3.
 */
#ifndef VLLM_H_
#define VLLM_H_

#include <stddef.h>
#include <stdint.h>

/* `bool` in the streaming callback signature: native in C++, needs <stdbool.h>
 * when this header is compiled as C. */
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── ABI version ──────────────────────────────────────────────────────────────
 * Bumped on any incompatible change to the structs / signatures below. A
 * consumer can compare this against vllm_abi_version() to detect a mismatch
 * between the header it compiled against and the loaded library.
 * v2: structured-output constraint fields appended to vllm_sampling_params
 * (structured_json / structured_regex / structured_choice / structured_grammar /
 * structured_json_object).
 * v3: OpenAI-style chat entry points (vllm_chat / vllm_chat_stream) — chat
 * templating, tool_choice lowering, and streaming tool-call parsing run
 * ENGINE-SIDE.
 * v4: tool_parser field appended to vllm_model_params — selects the tool-call
 * parser for the chat entry points, or AUTO-detects it from the chat template
 * when NULL/empty. */
#define VLLM_ABI_VERSION 4

/* ── Export macro ─────────────────────────────────────────────────────────────
 * Marks the symbols that make up the stable ABI. Default visibility now; Task 3
 * hides everything else (visibility=hidden + version script) so only VLLM_API
 * symbols are exported from libvllm.so. */
#ifndef VLLM_API
#if defined(_WIN32)
#define VLLM_API __declspec(dllexport)
#else
#define VLLM_API __attribute__((visibility("default")))
#endif
#endif

/* ── Status codes ─────────────────────────────────────────────────────────────
 * Every fallible entry point returns one of these. VLLM_OK == 0. On any error,
 * vllm_last_error() (thread-local) carries the detail. */
typedef enum vllm_status {
  VLLM_OK = 0,
  VLLM_ERR_INVALID_ARGUMENT = 1, /* null/out-of-range argument */
  VLLM_ERR_MODEL_LOAD = 2,       /* config/tokenizer/weights load failed */
  VLLM_ERR_RUNTIME = 3,          /* generation / engine runtime failure */
  VLLM_ERR_UNKNOWN = 4           /* a non-std::exception escaped internally */
} vllm_status;

/* ── Opaque handles ───────────────────────────────────────────────────────────
 * The engine handle owns the whole C++ stack (LLMEngine + Scheduler + runner +
 * KV cache + processors + tokenizer). Created by vllm_engine_load, destroyed by
 * vllm_engine_free. W2 completion submissions are thread-safe and share one
 * AsyncLLM scheduler; destruction still requires all request handles freed. */
typedef struct vllm_engine vllm_engine;

/* A non-blocking request returned by vllm_request_submit. The engine MUST
 * outlive every request created from it. Free with vllm_request_free. */
typedef struct vllm_request vllm_request;

/* ── Model / load parameters ──────────────────────────────────────────────────
 * POD. Populate with vllm_model_params_default() then override. All `const char*`
 * fields are borrowed for the duration of the vllm_engine_load call only — the
 * library copies what it needs; the caller retains ownership. */
typedef struct vllm_model_params {
  /* Supported model directory or GGUF file. Required. */
  const char* model_path;
  /* Optional override for tokenizer_config.json; NULL => <model_path>/
   * tokenizer_config.json (unused at T0 blocking-complete, reserved). */
  const char* tokenizer_config_path;
  /* KV-cache block size (tokens per block). <= 0 => 32. */
  int32_t block_size;
  /* Number of KV-cache blocks to allocate. <= 0 => 256. */
  int32_t num_blocks;
  /* Max sequence length. <= 0 => config.max_position_embeddings. */
  int32_t max_model_len;
  /* Max concurrent sequences the scheduler admits. <= 0 => 8. */
  int32_t max_num_seqs;
  /* ── Tool-call parser (ABI v4) ──────────────────────────────────────────────
   * Selects the parser that turns the model's raw tool-call output into
   * structured tool_calls for the chat entry points (vllm_chat /
   * vllm_chat_stream). NULL or "" => AUTO: the parser is detected from the
   * model's chat template at the first chat call (a template that wraps calls
   * in <tool_call> selects "hermes"; the fallback when nothing is detected is
   * also "hermes"). A non-empty value MUST name a registered parser (e.g.
   * "hermes", "qwen3"); an unknown name fails the first chat call with
   * VLLM_ERR_INVALID_ARGUMENT. Borrowed for the vllm_engine_load call only. */
  const char* tool_parser;
} vllm_model_params;

/* ── Sampling parameters ──────────────────────────────────────────────────────
 * POD mirror of the T0 fields of vllm::SamplingParams. Populate with
 * vllm_sampling_params_default() then override; a zero-initialized struct is NOT
 * valid (repetition_penalty must be > 0). temperature <= 0 selects greedy.
 * `stop` is borrowed for the duration of the vllm_complete call; the library
 * copies the strings. */
typedef struct vllm_sampling_params {
  float temperature;         /* randomness; <= eps => greedy (argmax). */
  float top_p;               /* nucleus cutoff in (0, 1]. */
  int32_t top_k;             /* top-k; 0 (or -1) => all tokens. */
  float min_p;               /* min token prob relative to the max, in [0, 1]. */
  int32_t max_tokens;        /* max tokens to generate; <= 0 => unbounded. */
  uint64_t seed;             /* RNG seed; used only when has_seed != 0. */
  int32_t has_seed;          /* 0 => unseeded (nondeterministic sampling). */
  float presence_penalty;    /* new-token presence penalty. */
  float frequency_penalty;   /* new-token frequency penalty. */
  float repetition_penalty;  /* repetition penalty; must be > 0 (default 1). */
  int32_t min_tokens;        /* min tokens before EOS/stop can end generation. */
  int32_t ignore_eos;        /* 0/1: keep generating past EOS. */
  const char* const* stop;   /* array of stop strings (may be NULL). */
  int32_t n_stop;            /* number of entries in `stop`. */
  /* ── Structured output (ABI v2) ─────────────────────────────────────────────
   * POD mirror of vllm::StructuredOutputsParams (the same constraints the
   * OpenAI response_format layer lowers to). AT MOST ONE of the five
   * constraints below may be set (non-NULL / non-zero); more than one is
   * rejected with an error status (upstream's exactly-one rule). The strings
   * are borrowed for the duration of the call; the library copies them.
   * Enforcement is engine-side per-step constrained decoding (a grammar
   * bitmask over the logits), on every completion entry point.
   *   - structured_json: a JSON-Schema document, as a JSON string; the output
   *     is constrained to instances of that schema.
   *   - structured_regex: the output matches the regular expression.
   *   - structured_choice / n_structured_choice: the output is exactly one of
   *     the given strings.
   *   - structured_grammar: a GBNF (llama.cpp-style) grammar.
   *   - structured_json_object: != 0 => some valid JSON object (schema-free
   *     "JSON mode"). */
  const char* structured_json;          /* JSON-Schema string, or NULL. */
  const char* structured_regex;         /* regular expression, or NULL. */
  const char* const* structured_choice; /* array of choices (may be NULL). */
  int32_t n_structured_choice;          /* entries in structured_choice. */
  const char* structured_grammar;       /* GBNF grammar, or NULL. */
  int32_t structured_json_object;       /* 0/1: schema-free JSON-object mode. */
} vllm_sampling_params;

/* ── Completion result ────────────────────────────────────────────────────────
 * Filled by vllm_complete. OWNERSHIP:
 *   - text: heap-allocated, NUL-terminated. The CALLER owns it and must free it
 *     via vllm_completion_free(out) (or vllm_string_free(out->text)). Set to
 *     NULL on any non-OK status.
 *   - finish_reason: borrowed pointer into library-owned static storage
 *     ("stop" / "length" / "abort" / ...). The caller must NOT free it and must
 *     not use it after the string literal's program lifetime (it is static, so
 *     it is always valid). NULL if the request did not finish.
 *   - prompt_tokens / completion_tokens: token counts (prompt vs generated). */
typedef struct vllm_completion {
  char* text;
  const char* finish_reason;
  int32_t prompt_tokens;
  int32_t completion_tokens;
} vllm_completion;

/* ── Defaults ─────────────────────────────────────────────────────────────────
 * Return structs pre-filled with the upstream SamplingParams / sane load
 * defaults. Use these as the base and override fields, so future struct growth
 * stays source-compatible. */
VLLM_API vllm_model_params vllm_model_params_default(void);
VLLM_API vllm_sampling_params vllm_sampling_params_default(void);

/* ── Lifecycle ────────────────────────────────────────────────────────────────
 * vllm_engine_load: build the full engine stack from `params->model_path`.
 * On success returns VLLM_OK and stores a handle in *out (caller frees via
 * vllm_engine_free). On failure returns a VLLM_ERR_* code, sets vllm_last_error(),
 * and leaves *out == NULL. `params` and `out` must be non-NULL. */
VLLM_API vllm_status vllm_engine_load(const vllm_model_params* params,
                                      vllm_engine** out);

/* Destroy an engine handle and everything it owns. NULL is a no-op. */
VLLM_API void vllm_engine_free(vllm_engine* engine);

/* ── Completion (blocking) ────────────────────────────────────────────────────
 * Run a single blocking completion for `prompt` with `params`, filling *out.
 * Returns VLLM_OK on success (out->text is a heap string the caller frees), or a
 * VLLM_ERR_* code (out is zeroed, out->text == NULL, vllm_last_error() set).
 * `engine`, `prompt`, `params` and `out` must be non-NULL. */
VLLM_API vllm_status vllm_complete(vllm_engine* engine, const char* prompt,
                                   const vllm_sampling_params* params,
                                   vllm_completion* out);

/* ── Streaming completion (M3.5 Task 2) ───────────────────────────────────────
 * vllm_token_callback: invoked once per engine-step delta for the streaming
 * request, then once more with finished == true to carry the finish.
 *   - delta_text: the incremental text produced since the previous call, as a
 *     NUL-terminated, well-formed UTF-8 C string. It is BORROWED — valid ONLY
 *     for the duration of the call; copy it if you need to retain it. The final
 *     finished == true call may carry an empty delta_text ("").
 *   - finished: true on the terminal call (the request ended: EOS / stop /
 *     length / abort). The callback is not invoked again for this request.
 *   - user_data: the opaque pointer passed to vllm_complete_stream, round-tripped
 *     unchanged (e.g. an accumulator the callback appends to).
 * RETURN false to STOP generation early: the library aborts the in-flight
 * request (tears it down so the engine stays usable) and returns VLLM_OK. Return
 * true to keep generating. The callback is C code and MUST NOT throw across the
 * ABI (any C++ exception it raises is caught and mapped to a status). */
typedef bool (*vllm_token_callback)(const char* delta_text, bool finished,
                                    void* user_data);

/* Run a single streaming completion for `prompt` with `params`, invoking `cb`
 * per delta (see vllm_token_callback). BLOCKING: drives the engine loop to a
 * natural finish, an early stop (cb returned false), or an error before
 * returning. Returns VLLM_OK on success (including an early stop), or a
 * VLLM_ERR_* code (vllm_last_error() set). `engine`, `prompt`, `params` and `cb`
 * must be non-NULL; `user_data` may be NULL. Sampled generation (temperature > 0
 * with a seed) is supported and deterministic for a fixed seed. */
VLLM_API vllm_status vllm_complete_stream(vllm_engine* engine,
                                          const char* prompt,
                                          const vllm_sampling_params* params,
                                          vllm_token_callback cb,
                                          void* user_data);

/* ── Non-blocking streaming requests (async-serving W2) ─────────────────────
 * Submit returns after validation/enqueue. A library-owned delivery thread
 * invokes `cb` for each RequestOutput delta while the shared AsyncLLM engine
 * continues batching other requests. `out` receives an owned request handle.
 * The callback/user_data borrow follows vllm_complete_stream's contract.
 * The engine must outlive the request handle. */
VLLM_API vllm_status vllm_request_submit(
    vllm_engine* engine, const char* prompt,
    const vllm_sampling_params* params, vllm_token_callback cb,
    void* user_data, vllm_request** out);

/* Abort an in-flight request. Idempotent; its delivery thread exits after the
 * terminal abort output is consumed. */
VLLM_API vllm_status vllm_request_cancel(vllm_request* request);

/* Wait for callback delivery to finish and return its terminal status. On an
 * error, vllm_last_error() on the WAITING thread receives the request error.
 * Do not call wait/free from that request's own callback. */
VLLM_API vllm_status vllm_request_wait(vllm_request* request);

/* Non-blocking completion probe (false for NULL). */
VLLM_API bool vllm_request_done(const vllm_request* request);

/* Request-owned diagnostic string. Empty while the request is still running;
 * after vllm_request_done returns true (or wait returns), valid until
 * vllm_request_free. Never NULL. */
VLLM_API const char* vllm_request_error(const vllm_request* request);

/* Cancel if needed, join the delivery thread, and destroy the handle. NULL is
 * a no-op. The parent engine must still be alive. Must not be called from the
 * request's own callback; use cancel there and free from another thread. */
VLLM_API void vllm_request_free(vllm_request* request);

/* ── Chat completions (ABI v3) ────────────────────────────────────────────────
 * OpenAI-style chat entry points over the SAME engine handle. The heavy
 * lifting runs ENGINE-SIDE, exactly like the bundled OpenAI server:
 *   - request_json is one OpenAI /v1/chat/completions request object
 *     (messages, tools, tool_choice, temperature, top_p, max_tokens, stop,
 *     ...). The `model` and `stream` fields are ignored (the handle's model
 *     serves; streaming is selected by the entry point).
 *   - The chat template is applied by the engine's renderer. It is resolved
 *     at vllm_engine_load: <model_dir>/tokenizer_config.json `chat_template`,
 *     or the GGUF `tokenizer.chat_template` metadata for a .gguf model; when
 *     neither exists, a plain "<role>: <content>" join is used.
 *   - tools + tool_choice lower to the engine's structural-tag DECODE
 *     constraint: `auto` is LAZY (the ENGINE decides when a tool engages —
 *     text is unconstrained until the model emits the tool trigger, then the
 *     call is grammar-constrained); `required`/named force a call; `none`
 *     disables. Tool-call output is parsed engine-side (streaming-stateful
 *     Hermes-style parser) into structured tool_calls deltas.
 *
 * vllm_chat: BLOCKING non-streaming completion. On VLLM_OK, *out_response_json
 * is a heap NUL-terminated ChatCompletionResponse JSON object (choices with
 * message.content / message.tool_calls, finish_reason "stop"/"length"/
 * "tool_calls", usage) that the CALLER frees via vllm_string_free. On error,
 * *out_response_json is NULL and vllm_last_error() is set (a malformed
 * request_json maps to VLLM_ERR_INVALID_ARGUMENT). */
VLLM_API vllm_status vllm_chat(vllm_engine* engine, const char* request_json,
                               char** out_response_json);

/* vllm_chat_stream: BLOCKING streaming completion. `cb` receives ONE OpenAI
 * chat.completion.chunk JSON object per invocation in delta_text (no SSE
 * framing): first the role chunk, then content and/or tool_calls delta chunks
 * as the engine-side parser emits them, then the finish chunk; after the last
 * chunk the callback is invoked once more with finished == true and an empty
 * delta. Returning false from the callback aborts the in-flight request
 * (VLLM_OK is still returned). The borrow/threading contract of
 * vllm_token_callback applies unchanged. */
VLLM_API vllm_status vllm_chat_stream(vllm_engine* engine,
                                      const char* request_json,
                                      vllm_token_callback cb, void* user_data);

/* ── Memory helpers ───────────────────────────────────────────────────────────
 * Free a heap string returned by the library. NULL is a no-op. */
VLLM_API void vllm_string_free(char* s);

/* Free the owned members of a completion (out->text) and zero the struct. The
 * `out` struct itself is caller-provided storage and is not freed. NULL is a
 * no-op. */
VLLM_API void vllm_completion_free(vllm_completion* out);

/* ── Diagnostics / versioning ─────────────────────────────────────────────────
 * The last error on the CURRENT thread, as a NUL-terminated string owned by the
 * library (thread-local). Never NULL (empty string if no error). Valid until the
 * next C API call on the same thread; the caller must NOT free it. */
VLLM_API const char* vllm_last_error(void);

/* The library version string ("MAJOR.MINOR.PATCH[+cuda]"), static storage; do
 * not free. */
VLLM_API const char* vllm_version(void);

/* The ABI version the library was built with (compare against VLLM_ABI_VERSION). */
VLLM_API int32_t vllm_abi_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* VLLM_H_ */
