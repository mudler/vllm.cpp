// vllm.cpp original. GGUF wire semantics follow the llama.cpp format; pinned
// vLLM e24d1b24 has no GGUF load format.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace vllm {

// The read-only mapping of ONE .gguf file, refcounted. `GgufFile` holds one
// reference; a weight that is consumed IN PLACE out of the mapping (the
// keep-quant loader's mmap residency, QUANT-GGUF-KEEPQ-LOADER L5) holds another,
// so the mapping is unmapped only when BOTH the reader and every borrowing
// tensor are gone. This is what makes a borrowed `GgufTensorInfo::data` pointer
// safe to outlive the `GgufFile` value it came from — the entrypoint's
// `GgufFile` local is destroyed as soon as the model is built.
struct GgufMapping {
  int fd = -1;
  void* addr = nullptr;
  size_t size = 0;

  GgufMapping() = default;
  GgufMapping(const GgufMapping&) = delete;
  GgufMapping& operator=(const GgufMapping&) = delete;
  ~GgufMapping();
};

// GGUF metadata value type ids (wire format).
enum GgufValueType : uint32_t {
  kGgufU8 = 0,
  kGgufI8 = 1,
  kGgufU16 = 2,
  kGgufI16 = 3,
  kGgufU32 = 4,
  kGgufI32 = 5,
  kGgufF32 = 6,
  kGgufBool = 7,
  kGgufString = 8,
  kGgufArray = 9,
  kGgufU64 = 10,
  kGgufI64 = 11,
  kGgufF64 = 12,
};

struct GgufValue;

// Array kv payload. `elem_type` is the GGUF value type id of the elements
// (kept so empty arrays stay typed); each element also carries its own tag.
struct GgufArray {
  uint32_t elem_type = 0;
  std::vector<GgufValue> elems;
};

// Tagged union for one GGUF metadata value. The variant alternatives are
// listed in wire type-id order 0..12, so v.index() == the GGUF value type id
// (see GgufValueType).
struct GgufValue {
  std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
               bool, std::string, GgufArray, uint64_t, int64_t, double>
      v;
  uint32_t TypeId() const { return static_cast<uint32_t>(v.index()); }
};

// One tensor entry from the GGUF tensor-info table. `data` points into the
// file's read-only mmap and stays valid for the lifetime of the owning
// GgufFile. `shape` holds the on-disk ggml dims REVERSED into torch
// row-major order (ggml stores the fastest-varying dim first).
struct GgufTensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  uint32_t ggml_type = 0;
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
};

// Size math per ggml tensor type: quantized types pack `block_elems`
// elements into `block_bytes` bytes; scalar types are 1-element "blocks".
struct GgmlTypeTraits {
  int64_t block_elems;
  int64_t block_bytes;
  const char* name;
};

// Traits for a ggml type id; throws std::runtime_error (naming the id) on an
// unknown/untabulated type. Standard ids mirror ggml.h's enum ggml_type;
// Task 5 extends the table with fork-specific NVFP4 ids.
const GgmlTypeTraits& GgmlTraits(uint32_t type);

// One .gguf file, mmap'd read-only. Supports little-endian GGUF v3 (and v2,
// which shares the same little-endian layout; v3 only added a big-endian
// variant, which we reject with a clear message). All header contents are
// treated as UNTRUSTED: every read is bounds-checked against the real file
// size before it happens, counts are sanity-capped before allocation, all
// size arithmetic is overflow-guarded, and every tensor span is validated
// against the data section, so Open() throws std::runtime_error (message
// includes the path) on any malformation instead of handing out an
// out-of-bounds span.
class GgufFile {
 public:
  static GgufFile Open(const std::string& path);

  GgufFile(GgufFile&& other) noexcept;
  GgufFile& operator=(GgufFile&& other) noexcept;
  GgufFile(const GgufFile&) = delete;
  GgufFile& operator=(const GgufFile&) = delete;
  ~GgufFile();

  // Metadata kv by key; nullptr if absent.
  const GgufValue* FindKv(const std::string& key) const;
  // Tensor infos in file-appearance order.
  const std::vector<GgufTensorInfo>& Tensors() const { return tensors_; }
  // Throws std::runtime_error if `name` is not present.
  const GgufTensorInfo& Get(const std::string& name) const;

  // A keep-alive reference on this file's mapping. Hand it to anything that
  // retains a `GgufTensorInfo::data` pointer past this object's lifetime; see
  // GgufMapping above. Never null on an open file.
  const std::shared_ptr<const GgufMapping>& Mapping() const { return map_; }

  // True when [data, data+nbytes) lies wholly inside this file's mapping — the
  // precondition for borrowing those bytes in place.
  bool OwnsSpan(const uint8_t* data, size_t nbytes) const;

  // Drop the resident pages of a span that has been read for the LAST time —
  // i.e. a tensor the loader EXPANDED, whose file bytes nothing will look at
  // again. This is llama.cpp's `unmap_fragment` idea
  // (src/llama-mmap.cpp:490, called from src/llama-model-loader.cpp:1676-1678,
  // where the loader releases the parts of the mapping it did NOT keep in place)
  // adapted to a MIXED file: llama.cpp can munmap two contiguous fragments
  // because its in-place tensors are contiguous, whereas ours are interleaved
  // with expanded f16/f32 ones, so the release is per-tensor and uses
  // MADV_DONTNEED rather than munmap — which keeps the mapping's address space
  // whole while dropping exactly the same physical pages. RECORDED DEVIATION.
  //
  // Purely a residency hint: the mapping is read-only and file-backed, so a
  // later read of a dropped page simply re-faults it with the same bytes. Only
  // whole INTERIOR pages are dropped, so a page shared with a neighbouring
  // kept-in-place tensor is never touched. Enabled by ReleaseExpandedPages().
  void DropSpanResidency(const uint8_t* data, size_t nbytes) const;

  // Opt the file into DropSpanResidency (default off, i.e. today's behavior).
  // Set by the loader for the duration of a load whose kept weights are
  // consumed in place, so the read-once pages of the EXPANDED tensors do not
  // accumulate into peak RSS alongside their expansions.
  void ReleaseExpandedPages(bool on) const { release_expanded_ = on; }
  bool releases_expanded_pages() const { return release_expanded_; }

 private:
  GgufFile() = default;
  void Release() noexcept;

  std::string path_;
  std::shared_ptr<const GgufMapping> map_;
  std::map<std::string, GgufValue> kvs_;
  std::vector<GgufTensorInfo> tensors_;
  std::map<std::string, size_t> index_;  // name -> position in tensors_
  // A residency hint, not state: it changes which physical pages are resident,
  // never what any read returns. Mutable so a const GgufFile (which is how every
  // loader takes it) can carry the load's policy.
  mutable bool release_expanded_ = false;
};

}  // namespace vllm
