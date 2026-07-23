// vllm.cpp original. GGUF wire semantics follow the llama.cpp format; pinned
// vLLM e24d1b24 has no GGUF load format.
#include "vllm/model_executor/model_loader/gguf_reader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace vllm {

namespace {

[[noreturn]] void Fail(const std::string& path, const std::string& what) {
  throw std::runtime_error("gguf: " + what + " in " + path);
}

// Sanity cap on kv and tensor counts (UNTRUSTED header): real checkpoints
// have thousands of entries, so 1e6 rejects hostile counts before any
// allocation is sized from them.
constexpr uint64_t kMaxCount = 1000000;
// Arrays may nest (an array element can itself be an array); bound the
// recursion so a hostile file cannot overflow the stack.
constexpr int kMaxArrayDepth = 16;

// Bounds-checked little-endian cursor over the mmap'd file. Every Need()
// call validates against the real file size BEFORE the bytes are touched.
struct Cursor {
  const uint8_t* base;
  size_t size;
  size_t pos = 0;
  const std::string& path;

  void Need(size_t n, const char* what) {
    // pos <= size always holds, so size - pos cannot underflow.
    if (n > size - pos)
      Fail(path, std::string("truncated file: ") + what + " needs " +
                     std::to_string(n) + " bytes at offset " +
                     std::to_string(pos) + " but only " +
                     std::to_string(size - pos) + " remain");
  }
  uint8_t U8(const char* what) {
    Need(1, what);
    return base[pos++];
  }
  uint16_t U16(const char* what) {
    Need(2, what);
    uint16_t v = 0;
    for (int i = 0; i < 2; ++i)
      v = static_cast<uint16_t>(v | static_cast<uint16_t>(base[pos + i])
                                        << (8 * i));
    pos += 2;
    return v;
  }
  uint32_t U32(const char* what) {
    Need(4, what);
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
      v |= static_cast<uint32_t>(base[pos + i]) << (8 * i);
    pos += 4;
    return v;
  }
  uint64_t U64(const char* what) {
    Need(8, what);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
      v |= static_cast<uint64_t>(base[pos + i]) << (8 * i);
    pos += 8;
    return v;
  }
  std::string Str(const char* what) {
    const uint64_t len = U64(what);
    // UNTRUSTED length: bound against the remaining file before use (this
    // also caps it at the file size).
    if (len > size - pos)
      Fail(path, std::string(what) + " string length " + std::to_string(len) +
                     " exceeds remaining file size " +
                     std::to_string(size - pos));
    std::string s(reinterpret_cast<const char*>(base + pos),
                  static_cast<size_t>(len));
    pos += static_cast<size_t>(len);
    return s;
  }
};

// `elems_parsed` is the running RECURSIVE total of array elements parsed so
// far in this file; it is bounded by kMaxCount so nested arrays cannot
// multiply the per-array count checks into an amplified allocation.
GgufValue ReadValue(Cursor& cur, uint32_t type, int depth,
                    uint64_t& elems_parsed) {
  GgufValue out;
  switch (type) {
    case kGgufU8:
      out.v = cur.U8("u8 kv");
      break;
    case kGgufI8:
      out.v = static_cast<int8_t>(cur.U8("i8 kv"));
      break;
    case kGgufU16:
      out.v = cur.U16("u16 kv");
      break;
    case kGgufI16:
      out.v = static_cast<int16_t>(cur.U16("i16 kv"));
      break;
    case kGgufU32:
      out.v = cur.U32("u32 kv");
      break;
    case kGgufI32:
      out.v = static_cast<int32_t>(cur.U32("i32 kv"));
      break;
    case kGgufF32: {
      const uint32_t bits = cur.U32("f32 kv");
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      out.v = f;
      break;
    }
    case kGgufBool:
      out.v = cur.U8("bool kv") != 0;
      break;
    case kGgufString:
      out.v = cur.Str("kv");
      break;
    case kGgufArray: {
      if (depth >= kMaxArrayDepth)
        Fail(cur.path, "kv array nesting exceeds depth limit " +
                           std::to_string(kMaxArrayDepth));
      GgufArray arr;
      arr.elem_type = cur.U32("array elem type");
      // Validate the element type even when the array is empty (the loop
      // below would otherwise never see an unknown type for count == 0).
      if (arr.elem_type > kGgufF64)
        Fail(cur.path, "unknown kv array element type " +
                           std::to_string(arr.elem_type));
      const uint64_t count = cur.U64("array count");
      // UNTRUSTED count: every element consumes at least 1 byte, so a count
      // beyond the remaining bytes is malformed; reject before allocating.
      if (count > cur.size - cur.pos)
        Fail(cur.path, "kv array count " + std::to_string(count) +
                           " exceeds remaining file size " +
                           std::to_string(cur.size - cur.pos));
      // Bound the recursive TOTAL number of array elements in the file: each
      // parsed element costs sizeof(GgufValue) >> 1 byte of file, so without
      // this budget a small file could amplify into ~40x its size in memory.
      // elems_parsed <= kMaxCount holds, so the subtraction cannot underflow.
      if (count > kMaxCount - elems_parsed)
        Fail(cur.path, "array element budget exceeded (more than " +
                           std::to_string(kMaxCount) +
                           " total array elements)");
      elems_parsed += count;
      arr.elems.reserve(static_cast<size_t>(count));
      for (uint64_t i = 0; i < count; ++i)
        arr.elems.push_back(
            ReadValue(cur, arr.elem_type, depth + 1, elems_parsed));
      out.v = std::move(arr);
      break;
    }
    case kGgufU64:
      out.v = cur.U64("u64 kv");
      break;
    case kGgufI64:
      out.v = static_cast<int64_t>(cur.U64("i64 kv"));
      break;
    case kGgufF64: {
      const uint64_t bits = cur.U64("f64 kv");
      double d;
      std::memcpy(&d, &bits, sizeof(d));
      out.v = d;
      break;
    }
    default:
      Fail(cur.path, "unknown kv value type " + std::to_string(type));
  }
  return out;
}

// Standard ggml type traits. Ids and block geometry mirror ggml.h's
// enum ggml_type / type_traits table (llama.cpp); recorded here so the
// reader has no ggml dependency.
//
// Ids 39-41 follow mudler's killgate llama.cpp fork
// (~/llama-phase84-attn-only-source on dgx.casa), which appends
// GGML_TYPE_NVFP4 = 40 and GGML_TYPE_Q1_0 = 41 after mainline's
// GGML_TYPE_MXFP4 = 39 (ggml/include/ggml.h:429-431). Block geometry from
// ggml/src/ggml-common.h and gguf-py/gguf/constants.py GGML_QUANT_SIZES.
// See .agents/specs/gguf-nvfp4-notes.md for the full layout writeup.
const GgmlTypeTraits* FindGgmlTraits(uint32_t type) {
  switch (type) {
    case 0: {
      static constexpr GgmlTypeTraits t{1, 4, "F32"};
      return &t;
    }
    case 1: {
      static constexpr GgmlTypeTraits t{1, 2, "F16"};
      return &t;
    }
    case 2: {
      static constexpr GgmlTypeTraits t{32, 18, "Q4_0"};
      return &t;
    }
    case 8: {
      static constexpr GgmlTypeTraits t{32, 34, "Q8_0"};
      return &t;
    }
    case 10: {
      static constexpr GgmlTypeTraits t{256, 84, "Q2_K"};
      return &t;
    }
    case 11: {
      static constexpr GgmlTypeTraits t{256, 110, "Q3_K"};
      return &t;
    }
    case 12: {
      static constexpr GgmlTypeTraits t{256, 144, "Q4_K"};
      return &t;
    }
    case 13: {
      static constexpr GgmlTypeTraits t{256, 176, "Q5_K"};
      return &t;
    }
    case 14: {
      static constexpr GgmlTypeTraits t{256, 210, "Q6_K"};
      return &t;
    }
    case 22: {
      // block_iq2_s: f16 d + QK_K/4 qs + QK_K/16 qh = 2 + 64 + 16.
      // Used by the APEX "Mini" GGUFs for expert weights.
      static constexpr GgmlTypeTraits t{256, 82, "IQ2_S"};
      return &t;
    }
    case 23: {
      // block_iq4_xs: f16 d + u16 scales_h + QK_K/64 scales_l + QK_K/2 qs
      // = 2 + 2 + 4 + 128. Used by the APEX "Quality" GGUFs.
      static constexpr GgmlTypeTraits t{256, 136, "IQ4_XS"};
      return &t;
    }
    case 24: {
      static constexpr GgmlTypeTraits t{1, 1, "I8"};
      return &t;
    }
    case 25: {
      static constexpr GgmlTypeTraits t{1, 2, "I16"};
      return &t;
    }
    case 26: {
      static constexpr GgmlTypeTraits t{1, 4, "I32"};
      return &t;
    }
    case 27: {
      static constexpr GgmlTypeTraits t{1, 8, "I64"};
      return &t;
    }
    case 28: {
      static constexpr GgmlTypeTraits t{1, 8, "F64"};
      return &t;
    }
    case 30: {
      static constexpr GgmlTypeTraits t{1, 2, "BF16"};
      return &t;
    }
    case 39: {
      // block_mxfp4: u8 E8M0 scale + 16 bytes packed 4-bit e2m1
      // (fork ggml-common.h:205-210; same id/geometry as mainline).
      static constexpr GgmlTypeTraits t{32, 17, "MXFP4"};
      return &t;
    }
    case 40: {
      // Killgate fork extension: block_nvfp4 = 4 u8 UE4M3 scales (one per
      // 16-element sub-block) + 32 bytes packed 4-bit e2m1 => 64 elems in
      // 36 bytes. No per-tensor scale tensor; blocks are self-contained.
      // Fork ggml-common.h:211-217, ggml.h:430, gguf-py constants.py
      // GGML_QUANT_SIZES: (64, 4 + 32).
      static constexpr GgmlTypeTraits t{64, 36, "NVFP4"};
      return &t;
    }
    case 41: {
      // Killgate fork extension: block_q1_0 = f16 d + QK1_0/8 bit-packed
      // quants => 128 elems in 18 bytes (fork ggml-common.h:177-182,
      // ggml.h:431).
      static constexpr GgmlTypeTraits t{128, 18, "Q1_0"};
      return &t;
    }
    default:
      return nullptr;
  }
}

}  // namespace

const GgmlTypeTraits& GgmlTraits(uint32_t type) {
  const GgmlTypeTraits* t = FindGgmlTraits(type);
  if (t == nullptr)
    throw std::runtime_error("gguf: unknown ggml type id " +
                             std::to_string(type));
  return *t;
}

GgufFile GgufFile::Open(const std::string& path) {
  GgufFile f;  // fully constructed: dtor cleans up on any throw below
  f.path_ = path;

  // The mapping is refcounted from the moment it exists, so every early Fail()
  // below unmaps through the same one owner (GgufMapping's destructor).
  auto mapping = std::make_shared<GgufMapping>();
  f.map_ = mapping;

  mapping->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (mapping->fd < 0)
    Fail(path, std::string("cannot open file: ") + std::strerror(errno));
  struct stat st{};
  if (::fstat(mapping->fd, &st) != 0)
    Fail(path, std::string("fstat failed: ") + std::strerror(errno));
  if (st.st_size <= 0) Fail(path, "empty file");
  const size_t file_size = static_cast<size_t>(st.st_size);

  void* map = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, mapping->fd, 0);
  if (map == MAP_FAILED)
    Fail(path, std::string("mmap failed: ") + std::strerror(errno));
  mapping->addr = map;
  mapping->size = file_size;

  Cursor cur{static_cast<const uint8_t*>(map), file_size, 0, path};

  // Header: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count.
  cur.Need(4, "magic");
  if (std::memcmp(cur.base, "GGUF", 4) != 0) Fail(path, "bad magic (not GGUF)");
  cur.pos = 4;
  const uint32_t version = cur.U32("version");
  if (version != 2 && version != 3) {
    // A byte-swapped version field means a big-endian GGUFv3 file.
    if (version == 0x02000000u || version == 0x03000000u)
      Fail(path, "big-endian GGUF is not supported (version field is "
                 "byte-swapped)");
    Fail(path, "unsupported GGUF version " + std::to_string(version) +
                   " (v2 and v3 little-endian are supported)");
  }
  // v2 and v3 share the little-endian layout below (v1 used u32 lengths and
  // is rejected above; v3 only added the big-endian variant over v2).

  const uint64_t tensor_count = cur.U64("tensor count");
  const uint64_t kv_count = cur.U64("kv count");
  // UNTRUSTED counts: cap before any allocation is sized from them.
  if (tensor_count > kMaxCount)
    Fail(path, "tensor count " + std::to_string(tensor_count) +
                   " exceeds sanity cap " + std::to_string(kMaxCount));
  if (kv_count > kMaxCount)
    Fail(path, "kv count " + std::to_string(kv_count) +
                   " exceeds sanity cap " + std::to_string(kMaxCount));

  // Metadata kvs. `array_elems` is the file-wide recursive total of array
  // elements parsed, budgeted at kMaxCount inside ReadValue.
  uint64_t array_elems = 0;
  for (uint64_t i = 0; i < kv_count; ++i) {
    std::string key = cur.Str("kv key");
    const uint32_t type = cur.U32("kv value type");
    GgufValue value = ReadValue(cur, type, 0, array_elems);
    auto [it, inserted] = f.kvs_.emplace(std::move(key), std::move(value));
    if (!inserted) Fail(path, "duplicate kv key \"" + it->first + "\"");
  }

  // Data-section alignment: kv "general.alignment" (u32, power of two),
  // default 32.
  uint64_t alignment = 32;
  if (auto it = f.kvs_.find("general.alignment"); it != f.kvs_.end()) {
    const uint32_t* a = std::get_if<uint32_t>(&it->second.v);
    if (a == nullptr) Fail(path, "general.alignment kv is not a u32");
    if (*a == 0 || (*a & (*a - 1)) != 0)
      Fail(path, "general.alignment " + std::to_string(*a) +
                     " is not a power of two");
    alignment = *a;
  }

  // Tensor infos: name, u32 n_dims, u64 dims (ggml order), u32 type, u64
  // offset relative to the data section start. Offsets are stashed and
  // bounds-checked after the loop, once the section start (which depends on
  // the end of this table) is known.
  // (No reserve from the UNTRUSTED count: a tiny hostile file could claim
  // the full cap; growth stays proportional to bytes actually parsed.)
  std::vector<uint64_t> offsets;
  for (uint64_t i = 0; i < tensor_count; ++i) {
    GgufTensorInfo t;
    t.name = cur.Str("tensor name");
    const uint32_t n_dims = cur.U32("tensor n_dims");
    if (n_dims > 4)  // mirrors GGML_MAX_DIMS
      Fail(path, "tensor \"" + t.name + "\" has " + std::to_string(n_dims) +
                     " dims, exceeding GGML_MAX_DIMS (4)");
    uint64_t numel = 1;
    std::vector<uint64_t> ggml_dims(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      const uint64_t dim = cur.U64("tensor dim");
      if (dim > static_cast<uint64_t>(INT64_MAX))
        Fail(path, "tensor \"" + t.name + "\" dim does not fit in int64");
      // UNTRUSTED dims: division-check before each multiply so a huge
      // declared shape throws instead of wrapping (same guard pattern as
      // vt::StepArena / the safetensors reader).
      if (dim != 0 && numel > UINT64_MAX / dim)
        Fail(path, "tensor \"" + t.name + "\" element count overflows");
      numel *= dim;
      ggml_dims[d] = dim;
    }
    // ggml stores the fastest-varying dim first; reverse into torch
    // row-major order.
    t.shape.reserve(n_dims);
    for (uint32_t d = n_dims; d > 0; --d)
      t.shape.push_back(static_cast<int64_t>(ggml_dims[d - 1]));

    t.ggml_type = cur.U32("tensor ggml type");
    const GgmlTypeTraits* traits = FindGgmlTraits(t.ggml_type);
    if (traits == nullptr)
      Fail(path, "tensor \"" + t.name + "\" has unknown ggml type id " +
                     std::to_string(t.ggml_type));
    const uint64_t block_elems = static_cast<uint64_t>(traits->block_elems);
    const uint64_t block_bytes = static_cast<uint64_t>(traits->block_bytes);
    if (numel % block_elems != 0)
      Fail(path, "tensor \"" + t.name + "\" element count " +
                     std::to_string(numel) + " is not divisible by the " +
                     traits->name + " block size " +
                     std::to_string(block_elems));
    const uint64_t blocks = numel / block_elems;
    if (blocks != 0 && block_bytes > UINT64_MAX / blocks)
      Fail(path, "tensor \"" + t.name + "\" byte size overflows");
    const uint64_t nbytes = blocks * block_bytes;

    const uint64_t offset = cur.U64("tensor offset");
    if (offset % alignment != 0)
      Fail(path, "tensor \"" + t.name + "\" offset " + std::to_string(offset) +
                     " is not a multiple of the alignment " +
                     std::to_string(alignment));
    t.nbytes = static_cast<size_t>(nbytes);
    if (!f.index_.emplace(t.name, f.tensors_.size()).second)
      Fail(path, "duplicate tensor name \"" + t.name + "\"");
    offsets.push_back(offset);
    f.tensors_.push_back(std::move(t));
  }

  // Data section starts at the alignment boundary after the tensor-info
  // table. cur.pos <= file_size and alignment <= 2^32, so this cannot
  // overflow size_t.
  const size_t data_start =
      (cur.pos + static_cast<size_t>(alignment) - 1) /
      static_cast<size_t>(alignment) * static_cast<size_t>(alignment);
  if (data_start > file_size && !f.tensors_.empty())
    Fail(path, "data section start " + std::to_string(data_start) +
                   " is beyond the file size " + std::to_string(file_size));
  const size_t data_section =
      data_start <= file_size ? file_size - data_start : 0;

  // Bind tensor spans, bounds-checking each (UNTRUSTED) offset + nbytes
  // against the data section.
  for (size_t i = 0; i < f.tensors_.size(); ++i) {
    GgufTensorInfo& t = f.tensors_[i];
    const uint64_t offset = offsets[i];
    if (offset > data_section || t.nbytes > data_section - offset)
      Fail(path, "tensor \"" + t.name + "\" span [" + std::to_string(offset) +
                     ", " + std::to_string(offset + t.nbytes) +
                     ") exceeds the data section size " +
                     std::to_string(data_section));
    t.data = cur.base + data_start + static_cast<size_t>(offset);
  }

  return f;
}

const GgufValue* GgufFile::FindKv(const std::string& key) const {
  auto it = kvs_.find(key);
  return it == kvs_.end() ? nullptr : &it->second;
}

const GgufTensorInfo& GgufFile::Get(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) Fail(path_, "no tensor named \"" + name + "\"");
  return tensors_[it->second];
}

GgufMapping::~GgufMapping() {
  if (addr != nullptr) ::munmap(addr, size);
  if (fd >= 0) ::close(fd);
}

bool GgufFile::OwnsSpan(const uint8_t* data, size_t nbytes) const {
  if (map_ == nullptr || map_->addr == nullptr) return false;
  const auto* base = static_cast<const uint8_t*>(map_->addr);
  return data >= base && nbytes <= map_->size &&
         static_cast<size_t>(data - base) <= map_->size - nbytes;
}

void GgufFile::DropSpanResidency(const uint8_t* data, size_t nbytes) const {
#if defined(__unix__)
  if (!release_expanded_ || !OwnsSpan(data, nbytes)) return;
  const long ps_l = ::sysconf(_SC_PAGESIZE);
  const auto ps = static_cast<uintptr_t>(ps_l > 0 ? ps_l : 4096);
  const auto begin = reinterpret_cast<uintptr_t>(data);
  const uintptr_t end = begin + nbytes;
  // INTERIOR whole pages only: a boundary page may also hold the first/last
  // bytes of a neighbouring tensor that IS being kept in place.
  const uintptr_t page_begin = (begin + ps - 1) & ~(ps - 1);
  const uintptr_t page_end = end & ~(ps - 1);
  if (page_end > page_begin) {
    // Best-effort by contract: a failure costs resident pages, never
    // correctness, so there is nothing to report or recover.
    (void)::madvise(reinterpret_cast<void*>(page_begin),
                    static_cast<size_t>(page_end - page_begin), MADV_DONTNEED);
  }
#else
  (void)data;
  (void)nbytes;
#endif
}

// Drops THIS object's reference. The mapping itself survives while any borrowing
// weight still holds one (GgufMapping's destructor does the munmap/close).
void GgufFile::Release() noexcept { map_.reset(); }

GgufFile::~GgufFile() = default;

GgufFile::GgufFile(GgufFile&& other) noexcept
    : path_(std::move(other.path_)),
      map_(std::move(other.map_)),
      kvs_(std::move(other.kvs_)),
      tensors_(std::move(other.tensors_)),
      index_(std::move(other.index_)) {}

GgufFile& GgufFile::operator=(GgufFile&& other) noexcept {
  if (this != &other) {
    path_ = std::move(other.path_);
    map_ = std::move(other.map_);
    kvs_ = std::move(other.kvs_);
    tensors_ = std::move(other.tensors_);
    index_ = std::move(other.index_);
  }
  return *this;
}

}  // namespace vllm
