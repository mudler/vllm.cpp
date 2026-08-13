// Ported from: vllm/v1/kv_offload/tiering/fs/io.py:32-101 @ e24d1b24
//               vllm/v1/kv_offload/file_mapper.py:112-139
// See include/vllm/v1/kv_offload/fs_io.h for scope, what is ported faithfully
// and the two deliberate omissions (O_DIRECT, the GIL-releasing extension).
#include "vllm/v1/kv_offload/fs_io.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "vllm/support/platform_compat.h"

namespace vllm::v1::kv_offload {
namespace {

std::filesystem::path NativePath(const std::string& utf8) {
#if defined(_WIN32)
  const std::u8string value(
      reinterpret_cast<const char8_t*>(utf8.data()), utf8.size());
  return std::filesystem::path(value);
#else
  return std::filesystem::path(utf8);
#endif
}

uint64_t ProcessId() {
#if defined(_WIN32)
  return static_cast<uint64_t>(GetCurrentProcessId());
#else
  return static_cast<uint64_t>(::getpid());
#endif
}

#if defined(_WIN32)
using NativeFile = HANDLE;
const NativeFile kInvalidFile = INVALID_HANDLE_VALUE;
#else
using NativeFile = int;
constexpr NativeFile kInvalidFile = -1;
#endif

std::string FileError(const char* operation, const std::string& path) {
#if defined(_WIN32)
  const DWORD error = GetLastError();
  return std::string("kv_offload: ") + operation + " '" + path +
         "' failed with Win32 error " + std::to_string(error);
#else
  return std::string("kv_offload: ") + operation + " '" + path + "': " +
         std::strerror(errno);
#endif
}

void CloseFile(NativeFile file) {
  if (file == kInvalidFile) return;
#if defined(_WIN32)
  CloseHandle(file);
#else
  ::close(file);
#endif
}

NativeFile CreateExclusiveFile(const std::string& path) {
#if defined(_WIN32)
  return CreateFileW(NativePath(path).c_str(), GENERIC_WRITE, 0, nullptr,
                     CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
  return ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_TRUNC, 0644);
#endif
}

NativeFile OpenReadFile(const std::string& path, bool* missing) {
  *missing = false;
#if defined(_WIN32)
  NativeFile file =
      CreateFileW(NativePath(path).c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == kInvalidFile) {
    const DWORD error = GetLastError();
    *missing = error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND;
  }
  return file;
#else
  NativeFile file = ::open(path.c_str(), O_RDONLY);
  if (file == kInvalidFile) *missing = errno == ENOENT;
  return file;
#endif
}

void WriteCompleteAt(NativeFile file, const std::string& path,
                     const char* data, size_t size, uint64_t offset) {
  size_t written = 0;
  while (written < size) {
#if defined(_WIN32)
    const uint64_t position = offset + written;
    LARGE_INTEGER location{};
    location.QuadPart = static_cast<LONGLONG>(position);
    if (!SetFilePointerEx(file, location, nullptr, FILE_BEGIN)) {
      throw std::runtime_error(FileError("seek", path));
    }
    const DWORD chunk = static_cast<DWORD>(
        std::min(size - written, static_cast<size_t>(MAXDWORD)));
    DWORD count = 0;
    if (!WriteFile(file, data + written, chunk, &count, nullptr)) {
      throw std::runtime_error(FileError("write", path));
    }
    if (count == 0) {
      throw std::runtime_error("kv_offload: short write on '" + path + "'");
    }
    written += count;
#else
    const ssize_t count =
        ::pwrite(file, data + written, size - written,
                 static_cast<off_t>(offset + written));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      throw std::runtime_error(FileError("short write on", path));
    }
    written += static_cast<size_t>(count);
#endif
  }
}

void ReadCompleteAt(NativeFile file, const std::string& path, char* data,
                    size_t size, uint64_t offset) {
  size_t got = 0;
  while (got < size) {
#if defined(_WIN32)
    const uint64_t position = offset + got;
    LARGE_INTEGER location{};
    location.QuadPart = static_cast<LONGLONG>(position);
    if (!SetFilePointerEx(file, location, nullptr, FILE_BEGIN)) {
      throw std::runtime_error(FileError("seek", path));
    }
    const DWORD chunk = static_cast<DWORD>(
        std::min(size - got, static_cast<size_t>(MAXDWORD)));
    DWORD count = 0;
    if (!ReadFile(file, data + got, chunk, &count, nullptr)) {
      throw std::runtime_error(FileError("read", path));
    }
    if (count == 0) {
      throw std::runtime_error("kv_offload: short read on '" + path +
                               "' (file is truncated)");
    }
    got += count;
#else
    const ssize_t count =
        ::pread(file, data + got, size - got,
                static_cast<off_t>(offset + got));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) throw std::runtime_error(FileError("read", path));
    if (count == 0) {
      throw std::runtime_error("kv_offload: short read on '" + path +
                               "' (file is truncated)");
    }
    got += static_cast<size_t>(count);
#endif
  }
}

void FlushFile(NativeFile file, const std::string& path) {
#if defined(_WIN32)
  if (!FlushFileBuffers(file)) {
    throw std::runtime_error(FileError("flush", path));
  }
#else
  if (::fsync(file) != 0) {
    throw std::runtime_error(FileError("flush", path));
  }
#endif
}

void PublishFile(const std::string& source, const std::string& destination) {
#if defined(_WIN32)
  if (!MoveFileExW(NativePath(source).c_str(), NativePath(destination).c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    throw std::runtime_error(FileError("publish", destination));
  }
#else
  std::error_code ec;
  std::filesystem::rename(source, destination, ec);
  if (ec) {
    throw std::system_error(ec, "kv_offload: cannot publish '" + destination +
                                    "'");
  }
#endif
}

std::string ToHex(const std::string& raw) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(raw.size() * 2);
  for (unsigned char c : raw) {
    out.push_back(kDigits[c >> 4]);
    out.push_back(kDigits[c & 0x0f]);
  }
  return out;
}

void WriteU32(char* p, uint32_t v) {
  p[0] = static_cast<char>(v & 0xff);
  p[1] = static_cast<char>((v >> 8) & 0xff);
  p[2] = static_cast<char>((v >> 16) & 0xff);
  p[3] = static_cast<char>((v >> 24) & 0xff);
}
uint32_t ReadU32(const char* p) {
  const auto b = [&](int i) {
    return static_cast<uint32_t>(static_cast<unsigned char>(p[i]));
  };
  return b(0) | (b(1) << 8) | (b(2) << 16) | (b(3) << 24);
}
void WriteU64(char* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    p[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  }
}
uint64_t ReadU64(const char* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i);
  }
  return v;
}

// Thread-local unique temp suffix, mirroring io.py:16-25 — two threads writing
// the same destination must not collide on the temp name.
const std::string& TmpSuffix() {
  static std::atomic<uint64_t> counter{0};
  static thread_local std::string suffix = [&] {
    std::string value = ".";
    value.append(std::to_string(ProcessId()));
    value.push_back('.');
    value.append(std::to_string(counter.fetch_add(1)));
    value.append(".tmp");
    return value;
  }();
  return suffix;
}

// Remove a file, ignoring "it was not there". Used on every failure path.
void RemoveQuietly(const std::string& path) {
  std::error_code ec;
  std::filesystem::remove(NativePath(path), ec);
}

}  // namespace

std::string BlockFileHeader::Encode() const {
  if (key.size() > kBlockHeaderKeyCapacity) {
    throw std::runtime_error("kv_offload: block key exceeds header capacity");
  }
  if (identity_digest.size() != 32) {
    throw std::runtime_error(
        "kv_offload: identity digest must be 32 raw bytes");
  }
  std::string buf(kBlockHeaderBytes, '\0');
  std::memcpy(&buf[0], kBlockHeaderMagic, sizeof(kBlockHeaderMagic));
  WriteU32(&buf[8], format_version);
  WriteU32(&buf[12], static_cast<uint32_t>(key.size()));
  WriteU64(&buf[16], payload_size);
  std::memcpy(&buf[24], identity_digest.data(), 32);
  if (!key.empty()) {
    std::memcpy(&buf[56], key.data(), key.size());
  }
  return buf;
}

BlockFileHeader BlockFileHeader::Decode(const char* data, size_t size) {
  if (size < kBlockHeaderBytes) {
    throw std::runtime_error("kv_offload: block file is shorter than a header");
  }
  if (std::memcmp(data, kBlockHeaderMagic, sizeof(kBlockHeaderMagic)) != 0) {
    throw std::runtime_error(
        "kv_offload: block file has a bad magic (not a vllm.cpp KV block)");
  }
  BlockFileHeader h;
  h.format_version = ReadU32(data + 8);
  const uint32_t key_size = ReadU32(data + 12);
  if (key_size > kBlockHeaderKeyCapacity) {
    throw std::runtime_error("kv_offload: block header key size is out of range");
  }
  h.payload_size = ReadU64(data + 16);
  h.identity_digest.assign(data + 24, 32);
  h.key.assign(data + 56, key_size);
  return h;
}

// --- FileMapper --------------------------------------------------------------

FileMapper::FileMapper(std::string root_dir, CacheIdentity identity)
    : root_dir_(std::move(root_dir)), identity_(std::move(identity)) {
  if (auto bad = identity_.Validate()) {
    throw std::runtime_error(
        "kv_offload: refusing to open a KV cache directory with an incomplete "
        "identity (field '" +
        *bad + "')");
  }
  // <root>/<safe_model_name>_<digest12> (file_mapper.py:128-139).
  std::string safe_model_name = identity_.model_name;
  for (char& c : safe_model_name) {
    if (c == '/') {
      c = '_';
    }
  }
  base_path_ = root_dir_ + "/" + safe_model_name + "_" +
               identity_.ShortDigestHex();
}

std::string FileMapper::config_file_path() const {
  return base_path_ + "/config.json";
}

std::string FileMapper::file_name(const OffloadKey& key) const {
  const std::string hash_hex = ToHex(get_offload_block_hash(key));
  const uint32_t group_idx = get_offload_group_idx(key);
  if (hash_hex.size() < 5) {
    throw std::runtime_error("kv_offload: block hash is too short to map");
  }
  return base_path_ + "_r" + std::to_string(identity_.rank) + "/" +
         hash_hex.substr(0, 3) + "/" + hash_hex.substr(3, 2) + "_g" +
         std::to_string(group_idx) + "/" + hash_hex + ".bin";
}

void FileMapper::OpenOrCreate() const {
  std::error_code ec;
  std::filesystem::create_directories(NativePath(base_path_), ec);
  if (ec) {
    throw std::runtime_error("kv_offload: cannot create '" + base_path_ +
                             "': " + ec.message());
  }
  const std::string config_path = config_file_path();

  std::ifstream in(NativePath(config_path), std::ios::binary);
  if (in.good()) {
    // THE CHECK UPSTREAM DOES NOT DO. Upstream writes config.json once and
    // never reads it (file_mapper.py:122-126); its only defence is the path
    // digest, which omits checkpoint content, weight quantization, rope config
    // and sliding_window. We read it back and compare field by field so a
    // mismatch REFUSES with the offending field named, instead of loading
    // another model's KV and producing plausible wrong tokens.
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    in.close();
    const CacheIdentity on_disk = CacheIdentity::FromCanonicalJson(text);
    if (auto field = CacheIdentity::FirstMismatch(on_disk, identity_)) {
      throw std::runtime_error(
          "kv_offload: REFUSING to use the KV cache at '" + base_path_ +
          "': it was written under a different configuration (field '" +
          *field +
          "' differs). Loading it would produce plausible but WRONG tokens. "
          "Use a different --kv-offload-dir or delete the directory.");
    }
    return;
  }

  // Absent: write it. Temp + rename so a concurrent opener never reads a
  // half-written config.
  const std::string tmp = config_path + TmpSuffix();
  NativeFile config_file = CreateExclusiveFile(tmp);
  if (config_file == kInvalidFile) {
    throw std::runtime_error(FileError("cannot create", tmp));
  }
  try {
    const std::string contents = identity_.ToCanonicalJson();
    WriteCompleteAt(config_file, tmp, contents.data(), contents.size(), 0);
    FlushFile(config_file, tmp);
    CloseFile(config_file);
    config_file = kInvalidFile;
    PublishFile(tmp, config_path);
    return;
  } catch (...) {
    CloseFile(config_file);
    RemoveQuietly(tmp);
    // A concurrent creator winning the race is fine as long as what landed
    // agrees with us — re-enter to perform the comparison.
    if (std::filesystem::exists(NativePath(config_path))) {
      OpenOrCreate();
      return;
    }
    throw;
  }
}

// --- the byte path -----------------------------------------------------------

void store_block(const std::string& dest_path, const BlockFileHeader& header,
                 const void* payload, size_t payload_size) {
  if (payload_size != header.payload_size) {
    throw std::runtime_error(
        "kv_offload: payload size does not match the header");
  }
  // Existence-skip: a content-addressed block never needs rewriting
  // (io.py:42-43).
  if (std::filesystem::exists(NativePath(dest_path))) {
    return;
  }

  std::error_code ec;
  const std::filesystem::path parent =
      NativePath(dest_path).parent_path();
  std::filesystem::create_directories(parent, ec);
  if (ec) {
    throw std::runtime_error("kv_offload: cannot create parent for '" + dest_path +
                             "': " + ec.message());
  }

  const std::string tmp_path = dest_path + TmpSuffix();
  const NativeFile file = CreateExclusiveFile(tmp_path);
  if (file == kInvalidFile) {
    throw std::runtime_error(FileError("cannot create", tmp_path));
  }

  try {
    const std::string encoded = header.Encode();
    WriteCompleteAt(file, tmp_path, encoded.data(), encoded.size(), 0);
    if (payload_size > 0) {
      WriteCompleteAt(file, tmp_path, static_cast<const char*>(payload),
                      payload_size, encoded.size());
    }
    FlushFile(file, tmp_path);
  } catch (...) {
    CloseFile(file);
    RemoveQuietly(tmp_path);
    throw;
  }
  CloseFile(file);

  // ATOMIC PUBLISH: a reader observes either no file or a complete one.
  try {
    PublishFile(tmp_path, dest_path);
  } catch (...) {
    RemoveQuietly(tmp_path);
    // Another writer publishing the identical content first is a success, not
    // a failure — the block is content-addressed.
    if (std::filesystem::exists(NativePath(dest_path))) {
      return;
    }
    throw;
  }
}

bool load_block(const std::string& source_path,
                const BlockFileHeader& expected, void* out,
                size_t out_capacity) {
  bool missing = false;
  const NativeFile file = OpenReadFile(source_path, &missing);
  if (file == kInvalidFile) {
    if (missing) {
      return false;  // an ordinary miss
    }
    throw std::runtime_error(FileError("cannot open", source_path));
  }

  try {
    if (out_capacity < expected.payload_size) {
      throw std::runtime_error(
          "kv_offload: destination buffer is smaller than the block payload");
    }
    char header_buf[kBlockHeaderBytes];
    ReadCompleteAt(file, source_path, header_buf, kBlockHeaderBytes, 0);
    const BlockFileHeader on_disk =
        BlockFileHeader::Decode(header_buf, kBlockHeaderBytes);

    // THE VERIFIED HEADER READ, on EVERY open. Each of these is a REFUSAL, not
    // a warning: proceeding would hand the engine bytes that decode into
    // plausible but wrong tokens.
    if (on_disk.format_version != expected.format_version) {
      throw std::runtime_error(
          "kv_offload: block '" + source_path +
          "' has format version " + std::to_string(on_disk.format_version) +
          ", expected " + std::to_string(expected.format_version));
    }
    if (on_disk.identity_digest != expected.identity_digest) {
      throw std::runtime_error(
          "kv_offload: block '" + source_path +
          "' was written under a DIFFERENT model/config identity; refusing to "
          "load it");
    }
    if (on_disk.payload_size != expected.payload_size) {
      throw std::runtime_error(
          "kv_offload: block '" + source_path + "' has payload size " +
          std::to_string(on_disk.payload_size) + ", expected " +
          std::to_string(expected.payload_size));
    }
    if (!expected.key.empty() && on_disk.key != expected.key) {
      // A misfiled or renamed block: the file's own key must match the slot it
      // is being read for.
      throw std::runtime_error("kv_offload: block '" + source_path +
                               "' holds a different block key; refusing");
    }

    if (expected.payload_size > 0) {
      ReadCompleteAt(file, source_path, static_cast<char*>(out),
                     static_cast<size_t>(expected.payload_size),
                     kBlockHeaderBytes);
    }
  } catch (...) {
    CloseFile(file);
    // SELF-HEALING (io.py:87-92): an unreadable, truncated or foreign file is
    // removed so the next lookup is a clean miss rather than a repeating error.
    RemoveQuietly(source_path);
    throw;
  }
  CloseFile(file);
  return true;
}

}  // namespace vllm::v1::kv_offload
