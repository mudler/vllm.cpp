#ifndef VT_UNALIGNED_H_
#define VT_UNALIGNED_H_

#include <cstring>
#include <type_traits>

// GCC and Clang recognise that a small trivially-copyable memcpy is equivalent
// to a typed load and rewrite it as such under -O2. UBSan then flags the
// synthesised load as a misaligned access, even though memcpy is the correct
// way to read an unaligned address. no_sanitize("alignment") suppresses the
// false positive inside this function only.
#if defined(__GNUC__)
#define VT_NO_SANITIZE_ALIGNMENT __attribute__((no_sanitize("alignment")))
#else
#define VT_NO_SANITIZE_ALIGNMENT
#endif

namespace vt {

// Load a scalar from storage whose byte address is not required to satisfy the
// scalar type's alignment. mmap-backed tensor payloads may begin at any byte
// offset, so forming a typed pointer to them is undefined even on CPUs that
// tolerate unaligned instructions.
template <typename T>
VT_NO_SANITIZE_ALIGNMENT
T LoadUnaligned(const void* address) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value;
  std::memcpy(&value, address, sizeof(value));
  return value;
}

}  // namespace vt

#endif  // VT_UNALIGNED_H_
