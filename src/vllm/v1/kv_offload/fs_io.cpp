// Ported from: vllm/v1/kv_offload/tiering/fs/io.py:32-101 @ e24d1b24
//               vllm/v1/kv_offload/file_mapper.py:112-139
// See include/vllm/v1/kv_offload/fs_io.h for scope, what is ported faithfully
// and the two deliberate omissions (O_DIRECT, the GIL-releasing extension).
#include "vllm/v1/kv_offload/fs_io.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace vllm::v1::kv_offload {
namespace {

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
  static thread_local std::string suffix =
      "." + std::to_string(::getpid()) + "." +
      std::to_string(counter.fetch_add(1)) + ".tmp";
  return suffix;
}

// Remove a file, ignoring "it was not there". Used on every failure path.
void RemoveQuietly(const std::string& path) {
  std::error_code ec;
  std::filesystem::remove(path, ec);
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
  std::filesystem::create_directories(base_path_, ec);
  if (ec) {
    throw std::runtime_error("kv_offload: cannot create '" + base_path_ +
                             "': " + ec.message());
  }
  const std::string config_path = config_file_path();

  std::ifstream in(config_path, std::ios::binary);
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
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      throw std::runtime_error("kv_offload: cannot write '" + tmp + "'");
    }
    out << identity_.ToCanonicalJson();
    out.flush();
    if (!out.good()) {
      out.close();
      RemoveQuietly(tmp);
      throw std::runtime_error("kv_offload: short write on '" + tmp + "'");
    }
  }
  std::filesystem::rename(tmp, config_path, ec);
  if (ec) {
    RemoveQuietly(tmp);
    // A concurrent creator winning the race is fine as long as what landed
    // agrees with us — re-enter to perform the comparison.
    if (std::filesystem::exists(config_path)) {
      OpenOrCreate();
      return;
    }
    throw std::runtime_error("kv_offload: cannot publish '" + config_path +
                             "': " + ec.message());
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
  if (std::filesystem::exists(dest_path)) {
    return;
  }

  std::error_code ec;
  const std::filesystem::path parent =
      std::filesystem::path(dest_path).parent_path();
  std::filesystem::create_directories(parent, ec);
  if (ec) {
    throw std::runtime_error("kv_offload: cannot create '" + parent.string() +
                             "': " + ec.message());
  }

  const std::string tmp_path = dest_path + TmpSuffix();
  const int fd = ::open(tmp_path.c_str(),
                        O_CREAT | O_EXCL | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) {
    throw std::runtime_error("kv_offload: cannot open '" + tmp_path +
                             "': " + std::strerror(errno));
  }

  const auto write_all = [&](const char* data, size_t size) {
    size_t written = 0;
    while (written < size) {
      const ssize_t n = ::write(fd, data + written, size - written);
      if (n <= 0) {
        if (n < 0 && errno == EINTR) {
          continue;
        }
        // A SHORT WRITE IS AN ERROR, never a partial block.
        throw std::runtime_error("kv_offload: short write on '" + tmp_path +
                                 "'");
      }
      written += static_cast<size_t>(n);
    }
  };

  try {
    const std::string encoded = header.Encode();
    write_all(encoded.data(), encoded.size());
    if (payload_size > 0) {
      write_all(static_cast<const char*>(payload), payload_size);
    }
  } catch (...) {
    ::close(fd);
    RemoveQuietly(tmp_path);
    throw;
  }
  ::close(fd);

  // ATOMIC PUBLISH: a reader observes either no file or a complete one.
  std::filesystem::rename(tmp_path, dest_path, ec);
  if (ec) {
    RemoveQuietly(tmp_path);
    // Another writer publishing the identical content first is a success, not
    // a failure — the block is content-addressed.
    if (std::filesystem::exists(dest_path)) {
      return;
    }
    throw std::runtime_error("kv_offload: cannot publish '" + dest_path +
                             "': " + ec.message());
  }
}

bool load_block(const std::string& source_path,
                const BlockFileHeader& expected, void* out,
                size_t out_capacity) {
  const int fd = ::open(source_path.c_str(), O_RDONLY);
  if (fd < 0) {
    if (errno == ENOENT) {
      return false;  // an ordinary miss
    }
    throw std::runtime_error("kv_offload: cannot open '" + source_path +
                             "': " + std::strerror(errno));
  }

  const auto read_all = [&](char* data, size_t size) {
    size_t got = 0;
    while (got < size) {
      const ssize_t n = ::read(fd, data + got, size - got);
      if (n == 0) {
        // A SHORT READ IS AN ERROR: never serve a partial block.
        throw std::runtime_error("kv_offload: short read on '" + source_path +
                                 "' (file is truncated)");
      }
      if (n < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error("kv_offload: read failed on '" + source_path +
                                 "': " + std::strerror(errno));
      }
      got += static_cast<size_t>(n);
    }
  };

  try {
    if (out_capacity < expected.payload_size) {
      throw std::runtime_error(
          "kv_offload: destination buffer is smaller than the block payload");
    }
    char header_buf[kBlockHeaderBytes];
    read_all(header_buf, kBlockHeaderBytes);
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
      read_all(static_cast<char*>(out),
               static_cast<size_t>(expected.payload_size));
    }
  } catch (...) {
    ::close(fd);
    // SELF-HEALING (io.py:87-92): an unreadable, truncated or foreign file is
    // removed so the next lookup is a clean miss rather than a repeating error.
    RemoveQuietly(source_path);
    throw;
  }
  ::close(fd);
  return true;
}

}  // namespace vllm::v1::kv_offload
