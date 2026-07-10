// dlopen packaging smoke test (M3.5 Task 3) — THE library-first packaging DoD.
//
// Proves a purego/cgo/FFI consumer that has ONLY the built libvllm.so (no C++
// headers, no link-time symbols) can drive vllm.cpp: it dlopen()s the shared
// lib and dlsym()s every stable C ABI entry point BY NAME, then exercises the
// header-free paths — vllm_version / vllm_abi_version (no model needed) and a
// bad-path vllm_engine_load whose error surfaces through vllm_last_error(),
// entirely through the dlsym'd function pointers.
//
// This TU does NOT link libvllm; it only links the doctest main + ${CMAKE_DL_LIBS}.
// The .so path is injected as VLLM_SHARED_LIB_PATH by CMake ($<TARGET_FILE:...>).
// We include vllm.h purely for the ABI *types* (a real FFI consumer would redeclare
// them); the point is that the SYMBOLS are resolved at runtime via dlsym, not the
// link line.
#include "vllm.h"

#include <doctest/doctest.h>

#include <dlfcn.h>

#include <string>

#ifndef VLLM_SHARED_LIB_PATH
#error "VLLM_SHARED_LIB_PATH must be defined (path to the built libvllm.so)"
#endif

namespace {

// Function-pointer types for the ABI symbols we dlsym. These mirror the
// declarations in vllm.h; a header-less consumer would type them by hand.
using fn_version = const char* (*)(void);
using fn_abi_version = int32_t (*)(void);
using fn_model_params_default = vllm_model_params (*)(void);
using fn_sampling_params_default = vllm_sampling_params (*)(void);
using fn_engine_load = vllm_status (*)(const vllm_model_params*, vllm_engine**);
using fn_engine_free = void (*)(vllm_engine*);
using fn_complete = vllm_status (*)(vllm_engine*, const char*,
                                    const vllm_sampling_params*, vllm_completion*);
using fn_complete_stream = vllm_status (*)(vllm_engine*, const char*,
                                           const vllm_sampling_params*,
                                           vllm_token_callback, void*);
using fn_request_submit = vllm_status (*)(vllm_engine*, const char*,
                                          const vllm_sampling_params*,
                                          vllm_token_callback, void*,
                                          vllm_request**);
using fn_request_cancel = vllm_status (*)(vllm_request*);
using fn_request_wait = vllm_status (*)(vllm_request*);
using fn_request_done = bool (*)(const vllm_request*);
using fn_request_error = const char* (*)(const vllm_request*);
using fn_request_free = void (*)(vllm_request*);
using fn_string_free = void (*)(char*);
using fn_completion_free = void (*)(vllm_completion*);
using fn_last_error = const char* (*)(void);

// Resolve `name` from `handle`; the returned pointer must be non-null (fails the
// test otherwise). Uses a union-free reinterpret through void* (POSIX-sanctioned
// for dlsym function pointers).
template <typename Fn>
Fn Sym(void* handle, const char* name) {
  void* p = dlsym(handle, name);
  INFO("dlsym(", name, ")");
  REQUIRE(p != nullptr);
  return reinterpret_cast<Fn>(p);
}

}  // namespace

// ─── the packaging DoD: dlopen + dlsym every ABI symbol, drive header-free ────
TEST_CASE("dlopen: libvllm.so resolves the whole C ABI by name and drives it") {
  // (1) dlopen the built shared library (RTLD_NOW forces eager symbol binding —
  //     an unresolved symbol would fail here, proving the .so is self-contained).
  void* lib = dlopen(VLLM_SHARED_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
  INFO("dlopen error: ", (dlerror() != nullptr ? dlerror() : ""));
  REQUIRE(lib != nullptr);

  // (2) dlsym EVERY stable C ABI symbol by name — all must be non-null.
  auto p_version = Sym<fn_version>(lib, "vllm_version");
  auto p_abi = Sym<fn_abi_version>(lib, "vllm_abi_version");
  auto p_model_defaults = Sym<fn_model_params_default>(lib, "vllm_model_params_default");
  auto p_sampling_defaults =
      Sym<fn_sampling_params_default>(lib, "vllm_sampling_params_default");
  auto p_load = Sym<fn_engine_load>(lib, "vllm_engine_load");
  auto p_engine_free = Sym<fn_engine_free>(lib, "vllm_engine_free");
  auto p_complete = Sym<fn_complete>(lib, "vllm_complete");
  auto p_complete_stream = Sym<fn_complete_stream>(lib, "vllm_complete_stream");
  auto p_request_submit = Sym<fn_request_submit>(lib, "vllm_request_submit");
  auto p_request_cancel = Sym<fn_request_cancel>(lib, "vllm_request_cancel");
  auto p_request_wait = Sym<fn_request_wait>(lib, "vllm_request_wait");
  auto p_request_done = Sym<fn_request_done>(lib, "vllm_request_done");
  auto p_request_error = Sym<fn_request_error>(lib, "vllm_request_error");
  auto p_request_free = Sym<fn_request_free>(lib, "vllm_request_free");
  auto p_string_free = Sym<fn_string_free>(lib, "vllm_string_free");
  auto p_completion_free = Sym<fn_completion_free>(lib, "vllm_completion_free");
  auto p_last_error = Sym<fn_last_error>(lib, "vllm_last_error");

  // (3) version / abi-version work with no model loaded.
  CHECK(std::string(p_version()).size() > 0);
  CHECK(p_abi() == VLLM_ABI_VERSION);

  // The default-struct helpers round-trip through the ABI.
  vllm_model_params mp = p_model_defaults();
  vllm_sampling_params sp = p_sampling_defaults();
  CHECK(sp.repetition_penalty > 0.0f);  // a zeroed struct would be invalid.
  (void)p_complete;
  (void)p_complete_stream;
  (void)p_request_submit;
  (void)p_request_cancel;
  (void)p_request_wait;
  (void)p_request_done;
  (void)p_request_error;
  (void)p_request_free;
  (void)p_string_free;
  (void)p_completion_free;

  // (4) bad-path load through the dlsym'd pointers: returns an error status and
  //     sets the thread-local last_error — proves the FULL error path works via
  //     dlopen, with no headers/link-time symbols (the ABI is genuinely loadable).
  mp.model_path = "/nonexistent/vllm-cpp/dlopen/model/dir";
  vllm_engine* engine = reinterpret_cast<vllm_engine*>(0x1);  // must be nulled.
  vllm_status st = p_load(&mp, &engine);
  CHECK(st != VLLM_OK);
  CHECK(st == VLLM_ERR_MODEL_LOAD);
  CHECK(engine == nullptr);
  CHECK(std::string(p_last_error()).size() > 0);

  // Null-out-handle argument error also routes correctly through the pointer.
  CHECK(p_load(&mp, nullptr) == VLLM_ERR_INVALID_ARGUMENT);

  // p_engine_free on null is a no-op (exercises the free pointer safely).
  p_engine_free(nullptr);

  CHECK(dlclose(lib) == 0);
}
