#pragma once

#include <exception>
#include <ostream>
#include <stdexcept>
#include <string>

#include "vllm/support/platform_compat.h"

#if defined(_WIN32)

inline int setenv(const char* name, const char* value, int /*overwrite*/) {
  return vllm::support::SetEnvVar(name, value) ? 0 : -1;
}

inline int unsetenv(const char* name) {
  return vllm::support::UnsetEnvVar(name) ? 0 : -1;
}

inline int getpid() { return vllm::support::CurrentProcessId(); }

#endif

namespace vllm::support::test {

inline void SetEnvOrThrow(const char* name, const char* value) {
  if (!vllm::support::SetEnvVar(name, value)) {
    throw std::runtime_error(std::string("SetEnvVar failed: ") + name);
  }
}

inline void UnsetEnvOrThrow(const char* name) {
  if (!vllm::support::UnsetEnvVar(name)) {
    throw std::runtime_error(std::string("UnsetEnvVar failed: ") + name);
  }
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    SetEnvOrThrow(name_.c_str(), value);
  }

  ~ScopedEnvVar() {
    if (!vllm::support::UnsetEnvVar(name_.c_str())) {
      std::terminate();
    }
  }

  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

 private:
  std::string name_;
};

}  // namespace vllm::support::test
