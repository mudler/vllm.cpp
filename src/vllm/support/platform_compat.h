#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace vllm::support {

#if defined(_WIN32)

inline long HostPageSize() {
  SYSTEM_INFO system_info{};
  GetSystemInfo(&system_info);
  return system_info.dwPageSize > 0
             ? static_cast<long>(system_info.dwPageSize)
             : 4096L;
}

inline int CurrentProcessId() { return static_cast<int>(::GetCurrentProcessId()); }

inline int FileDescriptorFromFile(std::FILE* file) { return _fileno(file); }

inline bool TruncateFile(int fd, std::uint64_t size) { return _chsize_s(fd, size) == 0; }

// DIVERGENT ON AN EMPTY VALUE, and deliberately not normalised: `_putenv_s(name,
// "")` REMOVES the variable, where POSIX `setenv(name, "", 1)` defines it empty.
// No caller passes an empty value today. A test that needs defined-but-empty must
// say so at its call site rather than relying on this — the same contract
// `tests/support/test_env.h` records for the test-side seam.
inline bool SetEnvVar(const char* name, const char* value) {
  return _putenv_s(name, value) == 0;
}

#else

inline long HostPageSize() {
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0 ? page_size : 4096L;
}

inline int CurrentProcessId() { return ::getpid(); }

inline int FileDescriptorFromFile(std::FILE* file) { return ::fileno(file); }

inline bool TruncateFile(int fd, std::uint64_t size) {
  return ::ftruncate(fd, static_cast<off_t>(size)) == 0;
}

inline bool SetEnvVar(const char* name, const char* value) {
  return ::setenv(name, value, 1) == 0;
}

#endif

}  // namespace vllm::support
