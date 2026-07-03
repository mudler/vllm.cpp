// Test-only helpers for building synthetic GGUF files byte-by-byte
// (little-endian wire format) and writing them to disposable temp files.
// Shared by test_gguf.cpp (reader hardening) and test_bpe.cpp (GGUF vocab).
#pragma once

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gguf_test {

// Little-endian scalar encoders (GGUF is little-endian on the wire).
inline std::string U32Le(uint32_t v) {
  std::string s(4, '\0');
  for (int i = 0; i < 4; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

inline std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

// GGUF string: u64 LE length + raw bytes (no NUL).
inline std::string GStr(const std::string& s) { return U64Le(s.size()) + s; }

// GGUF header for `tensor_count` tensors and `kv_count` kvs.
inline std::string Header(uint32_t version, uint64_t tensor_count,
                          uint64_t kv_count) {
  return "GGUF" + U32Le(version) + U64Le(tensor_count) + U64Le(kv_count);
}

// Pads to the alignment boundary the way writers do before the data section.
inline void PadTo(std::string& bytes, size_t alignment) {
  while (bytes.size() % alignment != 0) bytes.push_back('\0');
}

// Whole-kv encoders (key + type tag + payload). Type ids match GgufValueType.
inline std::string StrKv(const std::string& key, const std::string& val) {
  return GStr(key) + U32Le(8) + GStr(val);
}

inline std::string U32Kv(const std::string& key, uint32_t val) {
  return GStr(key) + U32Le(4) + U32Le(val);
}

inline std::string StrArrayKv(const std::string& key,
                              const std::vector<std::string>& vals) {
  std::string s = GStr(key) + U32Le(9) + U32Le(8) + U64Le(vals.size());
  for (const auto& v : vals) s += GStr(v);
  return s;
}

inline std::string I32ArrayKv(const std::string& key,
                              const std::vector<int32_t>& vals) {
  std::string s = GStr(key) + U32Le(9) + U32Le(5) + U64Le(vals.size());
  for (const int32_t v : vals) s += U32Le(static_cast<uint32_t>(v));
  return s;
}

// Writes raw bytes to a unique file under the system temp dir; removed in the
// destructor so test runs don't accumulate files.
class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    // Distinct test binaries may run concurrently under ctest -j and each
    // starts its own counter, so the pid keeps the names collision-free.
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_gguf_test_" + std::to_string(::getpid()) + "_" +
              std::to_string(counter++) + ".gguf"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

}  // namespace gguf_test
