// vllm.cpp original (test harness); no upstream mirror.
//
// ONE portable environment-variable setter for the whole test tree.
//
// WHY IT EXISTS (issue #603). `setenv`/`unsetenv` are POSIX. MSVC provides
// neither, so a test that flips a `VT_*` knob mid-process does not fail on
// Windows — it does not COMPILE, and it takes the whole `windows-msvc-*` lane
// down with it:
//
//   tests\vt\test_backend_cross_device.cpp(1004,5): error C3861: 'setenv':
//       identifier not found
//
// Three files had already grown their own private `_putenv_s` branch before this
// header existed — `tests/vllm/test_gguf.cpp` (`SetEnvironment`),
// `tests/vllm/v1/kv_offload/lmcache/test_lmcache_client.cpp` and
// `tests/parity/test_qwen27_dense_lmhead_fp4.cpp` — which is exactly why #603
// asks for the shim to land ONCE rather than per site. New env-flipping tests
// include this header; the three private copies predate it and are left alone
// here so this repair stays reviewable, not because they are correct to keep.
//
// WHAT IS DELIBERATELY NOT DONE: `#ifdef`-ing the offending case out on MSVC.
// That turns a compile error into silently missing coverage on the one platform
// nobody builds locally, which is strictly worse than the error.
//
// SEMANTICS, and the one point the two platforms cannot be made to agree on:
// `_putenv_s(name, "")` DELETES the variable on Windows, while
// `setenv(name, "", 1)` defines it as the empty string on POSIX. `SetEnv` maps an
// empty value to a DELETE on both, so callers see one behaviour everywhere. A
// test that genuinely needs a defined-but-empty variable cannot use this helper
// and has to say so at its own call site.
#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace vllm_test {

// Removes `name` from the environment. Throws rather than returning a status: a
// test whose environment did not take effect is not running the arm it claims.
inline void UnsetEnv(const char* name) {
#if defined(_WIN32)
  const int result = ::_putenv_s(name, "");
#else
  const int result = ::unsetenv(name);
#endif
  if (result != 0) {
    throw std::runtime_error(std::string("failed to unset test environment variable ") +
                             name);
  }
}

// Defines `name` as `value`, overwriting any previous definition. An EMPTY (or
// null) `value` removes it — see the header comment. Throws on failure.
inline void SetEnv(const char* name, const char* value) {
  if (value == nullptr || value[0] == '\0') {
    UnsetEnv(name);
    return;
  }
#if defined(_WIN32)
  const int result = ::_putenv_s(name, value);
#else
  const int result = ::setenv(name, value, /*overwrite=*/1);
#endif
  if (result != 0) {
    throw std::runtime_error(std::string("failed to set test environment variable ") +
                             name);
  }
}

inline void SetEnv(const char* name, const std::string& value) {
  SetEnv(name, value.c_str());
}

}  // namespace vllm_test
