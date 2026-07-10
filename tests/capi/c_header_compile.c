/*
 * Pure-C compile check: include/vllm.h MUST be valid C (no C++ constructs leak
 * across the ABI). Built as C11 under -Wall -Wextra -Werror, so any C++-ism
 * (bool without <stdbool.h>, C++ default args, references, namespaces, ...) or
 * unused-parameter/implicit-declaration slip fails the build. This TU is never
 * run; it only has to compile + link the declared ABI symbols.
 */
#include "vllm.h"

/* A C-typed vllm_token_callback: proves the function-pointer typedef (with its
 * `bool` params, via <stdbool.h>) is valid C. Never called; only compiled. */
static bool c_header_token_cb(const char* delta_text, bool finished,
                              void* user_data) {
  (void)delta_text;
  (void)user_data;
  return !finished; /* stop once finished (return value exercises the bool). */
}

/* Instantiate the POD structs + a status value so the C compiler actually lays
 * them out, and reference every ABI entry point so the declarations are used. */
int vllm_capi_c_header_check(vllm_engine* eng, const char* prompt) {
  vllm_model_params mp = vllm_model_params_default();
  vllm_sampling_params sp = vllm_sampling_params_default();
  vllm_completion out;
  vllm_token_callback cb = &c_header_token_cb;
  vllm_request* request = NULL;
  vllm_status st = VLLM_OK;

  st = vllm_engine_load(&mp, &eng);
  if (st == VLLM_OK) {
    st = vllm_complete(eng, prompt, &sp, &out);
    vllm_completion_free(&out);
    vllm_string_free(out.text);
    st = vllm_complete_stream(eng, prompt, &sp, cb, /*user_data=*/NULL);
    st = vllm_request_submit(eng, prompt, &sp, cb, /*user_data=*/NULL,
                             &request);
    if (request != NULL) {
      (void)vllm_request_done(request);
      (void)vllm_request_error(request);
      (void)vllm_request_cancel(request);
      st = vllm_request_wait(request);
      vllm_request_free(request);
    }
    vllm_engine_free(eng);
  }

  (void)vllm_last_error();
  (void)vllm_version();
  (void)vllm_abi_version();
  return (int)st;
}
