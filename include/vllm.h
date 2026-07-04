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

#ifdef __cplusplus
extern "C" {
#endif

/* ── ABI version ──────────────────────────────────────────────────────────────
 * Bumped on any incompatible change to the structs / signatures below. A
 * consumer can compare this against vllm_abi_version() to detect a mismatch
 * between the header it compiled against and the loaded library. */
#define VLLM_ABI_VERSION 1

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
 * vllm_engine_free. Not thread-safe: serialize calls on one handle. */
typedef struct vllm_engine vllm_engine;

/* ── Model / load parameters ──────────────────────────────────────────────────
 * POD. Populate with vllm_model_params_default() then override. All `const char*`
 * fields are borrowed for the duration of the vllm_engine_load call only — the
 * library copies what it needs; the caller retains ownership. */
typedef struct vllm_model_params {
  /* Directory holding config.json, tokenizer.json and the *.safetensors shards
   * (T0: safetensors only). Required. */
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
