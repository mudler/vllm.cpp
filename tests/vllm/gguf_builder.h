// Test-only helpers for building synthetic GGUF files byte-by-byte
// (little-endian wire format) and writing them to disposable temp files.
// Shared by test_gguf.cpp (reader hardening) and test_bpe.cpp (GGUF vocab).
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gguf_test {

inline std::string Utf8Path(const std::filesystem::path& path) {
  const auto bytes = path.u8string();
  return std::string(bytes.begin(), bytes.end());
}

inline std::filesystem::path UniqueTempPath(const std::string& prefix,
                                            const std::string& suffix) {
  static std::atomic<uint64_t> counter{0};
  static const uint64_t process_nonce = [] {
    std::random_device random;
    return (static_cast<uint64_t>(random()) << 32) ^ random();
  }();
  return std::filesystem::temp_directory_path() /
         (prefix + std::to_string(process_nonce) + "_" +
          std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) +
          suffix);
}

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

inline std::string F32ArrayKv(const std::string& key,
                              const std::vector<float>& vals) {
  std::string s = GStr(key) + U32Le(9) + U32Le(6) + U64Le(vals.size());
  for (const float v : vals) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    s += U32Le(bits);
  }
  return s;
}

inline std::string F32Kv(const std::string& key, float val) {
  std::string s = GStr(key) + U32Le(6);  // type 6 = f32
  uint32_t bits;
  std::memcpy(&bits, &val, 4);
  return s + U32Le(bits);
}

inline std::string U64Kv(const std::string& key, uint64_t val) {
  return GStr(key) + U32Le(10) + U64Le(val);  // type 10 = u64
}

// GGUF bool kv: type 7, one byte. Written by llama.cpp for
// `dflash.attention.causal`, which is the GGUF spelling of the HF `is_causal`.
inline std::string BoolKv(const std::string& key, bool val) {
  return GStr(key) + U32Le(7) + std::string(1, val ? '\1' : '\0');
}

inline std::string BoolArrayKv(const std::string& key,
                               const std::vector<bool>& vals) {
  std::string s = GStr(key) + U32Le(9) + U32Le(7) + U64Le(vals.size());
  for (bool b : vals) s.push_back(b ? '\1' : '\0');
  return s;
}

// Accumulates kvs + tensors and assembles a valid GGUF v3 file (LE), computing
// the aligned data-section offsets the reader validates. Tensor `dims` are in
// ggml order (ne0 = fastest/inner dim first); `data` is the raw tensor bytes in
// ggml storage order. Shared by the GGUF model-loader tests.
class GgufModelBuilder {
 public:
  void AddKv(const std::string& kv) { kvs_.push_back(kv); }
  void AddTensor(const std::string& name, const std::vector<uint64_t>& dims,
                 uint32_t ggml_type, const std::string& data) {
    tensors_.push_back({name, dims, ggml_type, data});
  }

  std::string Build(uint32_t alignment = 32) const {
    // Assign each tensor an alignment-padded offset within the data section.
    std::vector<uint64_t> offsets(tensors_.size());
    uint64_t off = 0;
    for (size_t i = 0; i < tensors_.size(); ++i) {
      offsets[i] = off;
      off += tensors_[i].data.size();
      off = (off + alignment - 1) / alignment * alignment;
    }
    std::string f = Header(3, tensors_.size(), kvs_.size() + 1);
    f += U32Kv("general.alignment", alignment);
    for (const auto& kv : kvs_) f += kv;
    for (size_t i = 0; i < tensors_.size(); ++i) {
      const T& t = tensors_[i];
      f += GStr(t.name) + U32Le(static_cast<uint32_t>(t.dims.size()));
      for (uint64_t d : t.dims) f += U64Le(d);
      f += U32Le(t.ggml_type) + U64Le(offsets[i]);
    }
    PadTo(f, alignment);
    const size_t data_start = f.size();
    for (size_t i = 0; i < tensors_.size(); ++i) {
      f.resize(data_start + offsets[i], '\0');
      f += tensors_[i].data;
    }
    return f;
  }

 private:
  struct T {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t ggml_type;
    std::string data;
  };
  std::vector<std::string> kvs_;
  std::vector<T> tensors_;
};

// Writes raw bytes to a unique file under the system temp dir; removed in the
// destructor so test runs don't accumulate files.
class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    path_ = UniqueTempPath("vllm_gguf_test_", ".gguf");
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("failed to write GGUF test file");
    utf8_path_ = Utf8Path(path_);
  }
  ~TempFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const std::string& path() const { return utf8_path_; }

 private:
  std::filesystem::path path_;
  std::string utf8_path_;
};

}  // namespace gguf_test
