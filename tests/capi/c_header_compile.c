/*
 * Pure-C compile check: include/vllm.h MUST be valid C (no C++ constructs leak
 * across the ABI). Built as C11 under -Wall -Wextra -Werror, so any C++-ism
 * (bool without <stdbool.h>, C++ default args, references, namespaces, ...) or
 * unused-parameter/implicit-declaration slip fails the build. This TU is never
 * run; it only has to compile + link the declared ABI symbols.
 */
#include "vllm.h"

/* Instantiate the POD structs + a status value so the C compiler actually lays
 * them out, and reference every ABI entry point so the declarations are used. */
int vllm_capi_c_header_check(vllm_engine* eng, const char* prompt) {
  vllm_model_params mp = vllm_model_params_default();
  vllm_sampling_params sp = vllm_sampling_params_default();
  vllm_completion out;
  vllm_status st = VLLM_OK;

  st = vllm_engine_load(&mp, &eng);
  if (st == VLLM_OK) {
    st = vllm_complete(eng, prompt, &sp, &out);
    vllm_completion_free(&out);
    vllm_string_free(out.text);
    vllm_engine_free(eng);
  }

  (void)vllm_last_error();
  (void)vllm_version();
  (void)vllm_abi_version();
  return (int)st;
}
