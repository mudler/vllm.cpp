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

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <string>

#ifndef VLLM_SHARED_LIB_PATH
#error "VLLM_SHARED_LIB_PATH must be defined (path to the built shared library)"
#endif

namespace {

#if defined(_WIN32)
using SharedLibraryHandle = HMODULE;

std::string LastSharedLibraryError() {
  const DWORD error = GetLastError();
  return error == 0 ? std::string() : ("GetLastError=" + std::to_string(error));
}

SharedLibraryHandle OpenSharedLibrary(const char* path) {
  return LoadLibraryA(path);
}

void* LoadSymbol(SharedLibraryHandle handle, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(handle, name));
}

bool CloseSharedLibrary(SharedLibraryHandle handle) {
  return FreeLibrary(handle) != 0;
}
#else
using SharedLibraryHandle = void*;

std::string LastSharedLibraryError() {
  const char* error = dlerror();
  return error != nullptr ? std::string(error) : std::string();
}

SharedLibraryHandle OpenSharedLibrary(const char* path) {
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

void* LoadSymbol(SharedLibraryHandle handle, const char* name) {
  return dlsym(handle, name);
}

bool CloseSharedLibrary(SharedLibraryHandle handle) {
  return dlclose(handle) == 0;
}
#endif


// Function-pointer types for the ABI symbols we dlsym. These mirror the
// declarations in vllm.h; a header-less consumer would type them by hand.
using fn_version = const char* (*)(void);
using fn_abi_version = int32_t (*)(void);
using fn_embed = vllm_status (*)(vllm_engine*, const char* const*, int32_t,
                                 vllm_embedding_result*);
using fn_embed_free = void (*)(vllm_embedding_result*);
using fn_model_params_default = vllm_model_params (*)(void);
using fn_sampling_params_default = vllm_sampling_params (*)(void);
using fn_engine_load = vllm_status (*)(const vllm_model_params*, vllm_engine**);
using fn_engine_free = void (*)(vllm_engine*);
using fn_complete = vllm_status (*)(vllm_engine*, const char*,
                                    const vllm_sampling_params*, vllm_completion*);
using fn_complete_stream = vllm_status (*)(vllm_engine*, const char*,
                                           const vllm_sampling_params*,
                                           vllm_token_callback, void*);
using fn_complete_tokens = vllm_status (*)(vllm_engine*, const int32_t*, int32_t,
                                           const vllm_sampling_params*, int32_t*,
                                           int32_t, int32_t*, vllm_completion*);
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

template <typename Fn>
Fn Sym(SharedLibraryHandle handle, const char* name) {
  void* symbol = LoadSymbol(handle, name);
  INFO("resolve(", name, ")");
  REQUIRE(symbol != nullptr);
  return reinterpret_cast<Fn>(symbol);
}

}  // namespace

// ─── the packaging DoD: dlopen + dlsym every ABI symbol, drive header-free ────
TEST_CASE("shared library resolves the whole C ABI by name and drives it") {
  SharedLibraryHandle lib = OpenSharedLibrary(VLLM_SHARED_LIB_PATH);
  INFO("shared library load error: ", LastSharedLibraryError());
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
  auto p_complete_tokens = Sym<fn_complete_tokens>(lib, "vllm_complete_tokens");
  auto p_request_submit = Sym<fn_request_submit>(lib, "vllm_request_submit");
  auto p_request_cancel = Sym<fn_request_cancel>(lib, "vllm_request_cancel");
  auto p_request_wait = Sym<fn_request_wait>(lib, "vllm_request_wait");
  auto p_request_done = Sym<fn_request_done>(lib, "vllm_request_done");
  auto p_request_error = Sym<fn_request_error>(lib, "vllm_request_error");
  auto p_request_free = Sym<fn_request_free>(lib, "vllm_request_free");
  auto p_string_free = Sym<fn_string_free>(lib, "vllm_string_free");
  // ABI v15 (ARCH-ONE-SURFACE ROW 6): the embeddings entry points resolve.
  auto p_embed = Sym<fn_embed>(lib, "vllm_embed");
  auto p_embed_free = Sym<fn_embed_free>(lib, "vllm_embedding_result_free");
  (void)p_embed;
  // A NULL free is the documented no-op — drivable with no model loaded.
  p_embed_free(nullptr);
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
  (void)p_complete_tokens;
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

  CHECK(CloseSharedLibrary(lib));
}
