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
 * them out, and reference every ABI entry point so the declarations are used —
 * including the v11 transcription slice, the v12 video slice, and the v14
 * device field (this file went stale between v10 and v12; the claim above is
 * only honest if new surface lands HERE in the same change). */
int vllm_capi_c_header_check(vllm_engine* eng, const char* prompt) {
  vllm_model_params mp = vllm_model_params_default();
  vllm_sampling_params sp = vllm_sampling_params_default();
  vllm_completion out;
  vllm_token_callback cb = &c_header_token_cb;
  vllm_request* request = NULL;
  vllm_status st = VLLM_OK;

  mp.device = 0; /* ABI v14: 0=auto (the accelerator-first probe default). */

  st = vllm_engine_load(&mp, &eng);
  if (st == VLLM_OK) {
    st = vllm_complete(eng, prompt, &sp, &out);
    vllm_completion_free(&out);
    vllm_string_free(out.text);
    st = vllm_complete_stream(eng, prompt, &sp, cb, /*user_data=*/NULL);
    {
      const int32_t prompt_ids[1] = {1};
      int32_t out_ids[4];
      int32_t n_out = 0;
      st = vllm_complete_tokens(eng, prompt_ids, 1, &sp, out_ids, 4, &n_out,
                                /*out=*/NULL);
    }
    st = vllm_request_submit(eng, prompt, &sp, cb, /*user_data=*/NULL,
                             &request);
    if (request != NULL) {
      (void)vllm_request_done(request);
      (void)vllm_request_error(request);
      (void)vllm_request_cancel(request);
      st = vllm_request_wait(request);
      vllm_request_free(request);
    }

    /* Chat entry points (ABI v3). */
    {
      char* response_json = NULL;
      st = vllm_chat(eng, "{}", &response_json);
      vllm_string_free(response_json);
      st = vllm_chat_stream(eng, "{}", cb, /*user_data=*/NULL);
    }

    /* Audio transcription (ABI v11). */
    {
      vllm_transcription_params tp = vllm_transcription_params_default();
      vllm_transcription transcript;
      st = vllm_transcribe(eng, &tp, &transcript);
      vllm_transcription_free(&transcript);
    }

    /* Embeddings (ABI v15). */
    {
      const char* texts[1] = {"strict-C embed reference"};
      vllm_embedding_result emb;
      st = vllm_embed(eng, texts, 1, &emb);
      vllm_embedding_result_free(&emb);
    }

    vllm_engine_free(eng);
  }

  /* Video+audio generation (ABI v12): the separate vllm_video_engine handle
   * plus the engine-free mux-argv composer. */
  {
    /* v18: the family selector + the parallel extras arrays must be spellable
     * from pure C, const-qualification and all. */
    static const char* const extra_keys[] = {"partition"};
    static const char* const extra_values[] = {"fl2va"};
    vllm_video_model_params vmp = vllm_video_model_params_default();
    vllm_video_params vp = vllm_video_params_default();
    vllm_video_engine* veng = NULL;
    vmp.family = "minimax-h3";
    vmp.extra_keys = extra_keys;
    vmp.extra_values = extra_values;
    vmp.n_extras = 1;
    vp.extra_keys = NULL;
    vp.extra_values = NULL;
    vp.n_extras = 0;
    st = vllm_video_engine_load(&vmp, &veng);
    if (st == VLLM_OK) {
      vllm_video_result vres;
      const char* fam = vllm_video_engine_family(veng);
      (void)fam;
      st = vllm_video_generate(veng, &vp, &vres);
      vllm_video_result_free(&vres);
      vllm_video_engine_free(veng);
    }
    {
      vllm_video_mux_params mux = vllm_video_mux_params_default();
      char** mux_argv = NULL;
      int32_t mux_argc = 0;
      st = vllm_video_mux_argv(&mux, &mux_argv, &mux_argc);
      vllm_video_mux_argv_free(mux_argv, mux_argc);
    }
  }

  (void)vllm_last_error();
  (void)vllm_version();
  (void)vllm_abi_version();
  return (int)st;
}
