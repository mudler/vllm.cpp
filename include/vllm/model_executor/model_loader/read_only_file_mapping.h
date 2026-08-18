// vllm.cpp original (portable read-only model-file mapping).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace vllm::detail {

// One immutable, whole-file mapping. The platform handles and mapped view are
// released together when the last shared owner dies, which lets model tensors
// safely borrow data() without retaining their reader object.
class ReadOnlyFileMapping final {
 public:
  static std::shared_ptr<ReadOnlyFileMapping> Open(
      const std::filesystem::path& path);

  ReadOnlyFileMapping(const ReadOnlyFileMapping&) = delete;
  ReadOnlyFileMapping& operator=(const ReadOnlyFileMapping&) = delete;
  ~ReadOnlyFileMapping() noexcept;

  const uint8_t* data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }

#if !defined(_WIN32)
  // The descriptor the mapping was made from, for a caller that wants to READ
  // the file rather than fault it in through the mapping. Streaming an expert
  // slice is such a caller: a pread lands the bytes in one syscall, where a
  // memcpy from the mapping traps every 4 KiB page on the way. Negative when
  // the mapping is not backed by a descriptor.
  int fd() const noexcept { return fd_; }
#endif

 private:
  ReadOnlyFileMapping() = default;

  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
#if defined(_WIN32)
  void* file_handle_ = nullptr;
  void* mapping_handle_ = nullptr;
#else
  int fd_ = -1;
#endif
};

}  // namespace vllm::detail
