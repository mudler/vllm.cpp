#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <fcntl.h>
#include <io.h>
#include <malloc.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vllm::support {

inline constexpr double kPi = 3.141592653589793238462643383279502884;

#if defined(_WIN32)

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

inline int OpenFile(const char* path, int flags) {
  return _open(path, flags | _O_BINARY);
}

inline int OpenFile(const char* path, int flags, int mode) {
  return _open(path, flags | _O_BINARY, mode);
}

inline int CloseFile(int fd) { return _close(fd); }

inline std::intptr_t ReadFile(int fd, void* buffer, std::size_t size) {
  const auto chunk = static_cast<unsigned int>(
      std::min<std::size_t>(size,
                            static_cast<std::size_t>(
                                std::numeric_limits<unsigned int>::max())));
  return _read(fd, buffer, chunk);
}

inline std::intptr_t WriteFile(int fd, const void* buffer, std::size_t size) {
  const auto chunk = static_cast<unsigned int>(
      std::min<std::size_t>(size,
                            static_cast<std::size_t>(
                                std::numeric_limits<unsigned int>::max())));
  return _write(fd, buffer, chunk);
}

inline void* MapReadOnlyFile(int fd, std::size_t size) {
  if (size == 0) {
    return nullptr;
  }
  const auto os_handle = _get_osfhandle(fd);
  if (os_handle == -1) {
    return nullptr;
  }
  HANDLE mapping = CreateFileMappingA(
      reinterpret_cast<HANDLE>(os_handle), nullptr, PAGE_READONLY,
      static_cast<DWORD>(static_cast<std::uint64_t>(size) >> 32),
      static_cast<DWORD>(static_cast<std::uint64_t>(size) & 0xffffffffu),
      nullptr);
  if (mapping == nullptr) {
    return nullptr;
  }
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, size);
  CloseHandle(mapping);
  return view;
}

inline int UnmapFile(void* address, std::size_t /*size*/) {
  if (address == nullptr) {
    return 0;
  }
  return UnmapViewOfFile(address) ? 0 : -1;
}

inline long HostPageSize() {
  SYSTEM_INFO system_info{};
  GetSystemInfo(&system_info);
  return system_info.dwPageSize > 0
             ? static_cast<long>(system_info.dwPageSize)
             : 4096L;
}

inline int CurrentProcessId() { return _getpid(); }

inline int FileDescriptorFromFile(std::FILE* file) { return _fileno(file); }

inline bool TruncateFile(int fd, std::uint64_t size) { return _chsize_s(fd, size) == 0; }

inline bool SetEnvVar(const char* name, const char* value) {
  return _putenv_s(name, value) == 0;
}

inline bool UnsetEnvVar(const char* name) { return _putenv_s(name, "") == 0; }

inline void* AlignedAlloc(std::size_t alignment, std::size_t size) {
  return _aligned_malloc(size, alignment);
}

inline void AlignedFree(void* pointer) { _aligned_free(pointer); }

#else

inline int OpenFile(const char* path, int flags) { return ::open(path, flags); }

inline int OpenFile(const char* path, int flags, int mode) {
  return ::open(path, flags, mode);
}

inline int CloseFile(int fd) { return ::close(fd); }

inline ssize_t ReadFile(int fd, void* buffer, std::size_t size) {
  return ::read(fd, buffer, size);
}

inline ssize_t WriteFile(int fd, const void* buffer, std::size_t size) {
  return ::write(fd, buffer, size);
}

inline void* MapReadOnlyFile(int fd, std::size_t size) {
  void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  return mapped == MAP_FAILED ? nullptr : mapped;
}

inline int UnmapFile(void* address, std::size_t size) {
  if (address == nullptr) {
    return 0;
  }
  return ::munmap(address, size);
}

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

inline bool UnsetEnvVar(const char* name) { return ::unsetenv(name) == 0; }

inline void* AlignedAlloc(std::size_t alignment, std::size_t size) {
  return std::aligned_alloc(alignment, size);
}

inline void AlignedFree(void* pointer) { std::free(pointer); }

#endif

}  // namespace vllm::support
