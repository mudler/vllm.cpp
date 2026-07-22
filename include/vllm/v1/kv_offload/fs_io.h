// Ported from: vllm/v1/kv_offload/tiering/fs/io.py:32-101 @ e24d1b24
//               vllm/v1/kv_offload/file_mapper.py:112-139
//
// Scope: the byte path and the naming of the DISK tier — one raw file per
// block, no container and no index (EXISTENCE IS THE INDEX). Upstream's whole
// byte path is 101 lines of open/write/readv/rename/unlink with nothing
// Python-specific in it, and it ports essentially verbatim. This also
// introduces the byte WRITER the tree previously lacked entirely: before this
// change `grep -rn "ofstream|fwrite|O_WRONLY|O_CREAT" src include tools`
// returned exactly one hit, in a dev harness.
//
// PORTED FAITHFULLY:
//   * store: existence-skip -> O_CREAT|O_EXCL|O_WRONLY|O_TRUNC temp file ->
//     write -> rename (ATOMIC PUBLISH). A reader therefore never observes a
//     partially written block (io.py:42-66).
//   * load failure is SELF-HEALING: any failure to read a file unlinks it, so a
//     corrupt or truncated block becomes a miss next time instead of a
//     permanent error (io.py:87-92).
//   * a SHORT READ OR SHORT WRITE IS AN ERROR, never a partial block
//     (io.py:59-62, 88-90).
//   * the path layout <base>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin
//     (file_mapper.py:112-120) — the two-level fan-out keeps any one directory
//     from holding millions of entries.
//
// DELIBERATELY NOT PORTED, with the reason:
//   * O_DIRECT (io.py:12,53-57,87). Upstream opens both paths O_DIRECT. Our
//     block file is [fixed-size header][payload], and O_DIRECT on Linux
//     requires the buffer address, the file offset AND the length to be
//     block-aligned; a header that is not a multiple of the device block size
//     breaks all three, and the payload size is a per-spec quantity we do not
//     get to choose. Buffered IO is therefore the CORRECT first implementation.
//     Reinstating O_DIRECT means aligning the header to the device block size
//     and using an aligned bounce buffer, and is a measured optimization, not a
//     correctness item.
//   * the GIL-releasing faccessat C extension (csrc/fs_io.cpp:23,40). It exists
//     purely to batch existence checks without holding the GIL. We have no GIL,
//     so a batch lookup is an ordinary parallel loop and the extension is
//     unconditionally unnecessary (§B1).
#ifndef VLLM_V1_KV_OFFLOAD_FS_IO_H_
#define VLLM_V1_KV_OFFLOAD_FS_IO_H_

#include <cstdint>
#include <string>

#include "vllm/v1/kv_offload/base.h"
#include "vllm/v1/kv_offload/cache_identity.h"

namespace vllm::v1::kv_offload {

// The fixed-size header prefixed to every block file. READ AND VERIFIED ON
// EVERY OPEN — this is the safety property upstream does not have (its
// config.json is written and never read; see cache_identity.h).
//
// Layout on disk, little-endian, exactly kBlockHeaderBytes long:
//   [0  .. 8)    magic "VLCPPKV1"
//   [8  ..12)    format_version (u32)
//   [12 ..16)    key_size (u32)     — bytes of `key` that are meaningful
//   [16 ..24)    payload_size (u64) — MUST equal the spec's page_size_bytes()
//   [24 ..56)    identity_digest    — SHA-256 of the canonical identity JSON
//   [56 ..120)   key                — the OffloadKey this file must serve
//   [120..128)   reserved (zero)
// The payload follows immediately and is OPAQUE BYTES of exactly
// payload_size. No layout is interpreted here (see cache_identity.h §R9).
inline constexpr size_t kBlockHeaderBytes = 128;
inline constexpr size_t kBlockHeaderKeyCapacity = 64;
inline constexpr char kBlockHeaderMagic[8] = {'V', 'L', 'C', 'P',
                                              'P', 'K', 'V', '1'};

struct BlockFileHeader {
  uint32_t format_version = kCacheFormatVersion;
  uint64_t payload_size = 0;
  // 32 raw bytes.
  std::string identity_digest;
  // The OffloadKey (36 bytes for a 32-byte digest + 4-byte group id).
  OffloadKey key;

  // Serialize to exactly kBlockHeaderBytes.
  std::string Encode() const;
  // Parse; throws std::runtime_error on a bad magic, a short buffer, or an
  // over-long key.
  static BlockFileHeader Decode(const char* data, size_t size);
};

// Upstream: class FileMapper (file_mapper.py:19-139), plus the config.json
// READER upstream does not have.
class FileMapper {
 public:
  // `root_dir` is the tier root. The identity determines the directory digest,
  // exactly as upstream's field set determines its 12-hex path digest.
  FileMapper(std::string root_dir, CacheIdentity identity);

  // <root>/<safe_model_name>_<digest12>  (file_mapper.py:128-139). The model
  // name's '/' is replaced with '_' so a HuggingFace id does not nest.
  const std::string& base_path() const { return base_path_; }
  // <base>/config.json (file_mapper.py:122-126).
  std::string config_file_path() const;
  // <base>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin (file_mapper.py:112-120).
  std::string file_name(const OffloadKey& key) const;

  const CacheIdentity& identity() const { return identity_; }

  // Create the base directory and, if config.json is absent, write it. If it is
  // PRESENT, read it back and compare FIELD BY FIELD; on any difference throw
  // std::runtime_error naming the field. This is the tier-open half of the
  // two-level verification — upstream writes this file and never reads it.
  void OpenOrCreate() const;

 private:
  std::string root_dir_;
  CacheIdentity identity_;
  std::string base_path_;
};

// Write one block ATOMICALLY. `payload_size` must equal
// header.payload_size. Existing destinations are skipped (existence-skip,
// io.py:42-43). On any failure the temp file is removed and the error is
// rethrown.
void store_block(const std::string& dest_path, const BlockFileHeader& header,
                 const void* payload, size_t payload_size);

// Read one block into `out` (which must have room for
// `expected.payload_size` bytes). The header is verified against `expected`
// (magic, format version, identity digest, payload size, key) BEFORE the
// payload is trusted; a mismatch, a short read or a truncated file REFUSES and
// the file is UNLINKED so the next lookup is a clean miss.
// Returns false when the file simply does not exist (an ordinary miss); throws
// std::runtime_error on any verification or IO failure.
bool load_block(const std::string& source_path,
                const BlockFileHeader& expected, void* out,
                size_t out_capacity);

}  // namespace vllm::v1::kv_offload

#endif  // VLLM_V1_KV_OFFLOAD_FS_IO_H_
