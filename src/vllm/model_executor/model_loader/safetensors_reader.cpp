// vllm.cpp original (container reader); no upstream mirror.
#include "vllm/model_executor/model_loader/safetensors_reader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <nlohmann/json.hpp>

namespace vllm {

namespace {

[[noreturn]] void Fail(const std::string& path, const std::string& what) {
  throw std::runtime_error("safetensors: " + what + " in " + path);
}

// Byte width of the safetensors dtype string, or 0 when unknown. Unknown
// dtypes are allowed through as opaque spans (size not cross-checked against
// shape, but still bounds-checked against the data section).
size_t DtypeSize(const std::string& dtype) {
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16")
    return 2;
  if (dtype == "U8" || dtype == "I8" || dtype == "BOOL" ||
      dtype == "F8_E4M3" || dtype == "F8_E5M2")
    return 1;
  return 0;
}

}  // namespace

SafetensorsFile SafetensorsFile::Open(const std::string& path) {
  SafetensorsFile f;  // fully constructed: dtor cleans up on any throw below
  f.path_ = path;

  f.fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (f.fd_ < 0) Fail(path, "cannot open file");
  struct stat st{};
  if (::fstat(f.fd_, &st) != 0) Fail(path, "fstat failed");
  if (st.st_size <= 0) Fail(path, "empty file");
  const size_t file_size = static_cast<size_t>(st.st_size);
  if (file_size < 8) Fail(path, "file shorter than the 8-byte header prefix");

  void* map = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, f.fd_, 0);
  if (map == MAP_FAILED) Fail(path, "mmap failed");
  f.map_ = map;
  f.map_size_ = file_size;
  const uint8_t* bytes = static_cast<const uint8_t*>(map);

  // u64 little-endian header length, assembled byte-wise for portability.
  uint64_t header_len = 0;
  for (int i = 0; i < 8; ++i)
    header_len |= static_cast<uint64_t>(bytes[i]) << (8 * i);
  // UNTRUSTED: bound against the real file size before any use.
  if (header_len > file_size - 8)
    Fail(path, "header length " + std::to_string(header_len) +
                   " exceeds file size " + std::to_string(file_size));
  const uint8_t* data_base = bytes + 8 + header_len;
  const size_t data_section = file_size - 8 - static_cast<size_t>(header_len);

  // Parse the JSON header. The parse callback sees every key event (unlike
  // the resulting DOM, where duplicate keys silently collapse), so use it to
  // record header-appearance order and reject duplicate names.
  std::vector<std::string> order;
  std::set<std::string> seen;
  std::string duplicate;
  auto cb = [&](int depth, nlohmann::json::parse_event_t event,
                nlohmann::json& parsed) -> bool {
    if (depth == 1 && event == nlohmann::json::parse_event_t::key) {
      std::string key = parsed.get<std::string>();
      if (!seen.insert(key).second && duplicate.empty()) duplicate = key;
      order.push_back(std::move(key));
    }
    return true;
  };
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(bytes + 8, data_base, cb);
  } catch (const nlohmann::json::exception& e) {
    Fail(path, std::string("JSON header parse error: ") + e.what());
  }
  if (!duplicate.empty()) Fail(path, "duplicate entry \"" + duplicate + "\"");
  if (!doc.is_object()) Fail(path, "JSON header is not an object");

  // (begin, end, name) per tensor entry, collected while walking the header
  // and validated for overlaps after sorting by offset. The official reader
  // (safetensors 0.7.0) sorts entries by data offset before its layout check,
  // so JSON key order carries no meaning; matching that keeps us from
  // rejecting files the reference implementation accepts.
  std::vector<std::tuple<uint64_t, uint64_t, std::string>> spans;

  try {
    // Walk entries in header-appearance order (which fixes Names() order);
    // offset validation happens over the sorted spans below.
    for (const std::string& name : order) {
      const nlohmann::json& entry = doc.at(name);
      if (name == "__metadata__") {
        if (!entry.is_object()) Fail(path, "__metadata__ is not an object");
        for (const auto& [k, v] : entry.items()) {
          if (!v.is_string())
            Fail(path, "__metadata__ value for \"" + k + "\" is not a string");
          f.metadata_.emplace(k, v.get<std::string>());
        }
        continue;
      }
      if (!entry.is_object())
        Fail(path, "tensor entry \"" + name + "\" is not an object");
      for (const char* key : {"dtype", "shape", "data_offsets"}) {
        if (!entry.contains(key))
          Fail(path, "tensor \"" + name + "\" is missing \"" + key + "\"");
      }

      StTensor t;
      t.dtype = entry.at("dtype").get<std::string>();
      t.shape = entry.at("shape").get<std::vector<int64_t>>();

      const nlohmann::json& offs = entry.at("data_offsets");
      if (!offs.is_array() || offs.size() != 2)
        Fail(path, "tensor \"" + name + "\" data_offsets is not a 2-array");
      const uint64_t begin = offs[0].get<uint64_t>();
      const uint64_t end = offs[1].get<uint64_t>();
      // UNTRUSTED offsets: require begin <= end <= data_section so the span
      // stays inside the mapping.
      if (begin > end)
        Fail(path, "tensor \"" + name + "\" data_offsets begin > end");
      if (end > data_section)
        Fail(path, "tensor \"" + name + "\" data_offsets end " +
                       std::to_string(end) + " exceeds data section size " +
                       std::to_string(data_section));
      spans.emplace_back(begin, end, name);

      t.nbytes = static_cast<size_t>(end - begin);
      t.data = data_base + begin;

      // numel * dtype_size, division-checked before each multiply so a huge
      // declared shape throws instead of wrapping (untrusted metadata).
      size_t numel = 1;
      for (int64_t dim : t.shape) {
        if (dim < 0)
          Fail(path, "tensor \"" + name + "\" has a negative shape dim");
        const size_t d = static_cast<size_t>(dim);
        if (d != 0 && numel > SIZE_MAX / d)
          Fail(path, "tensor \"" + name + "\" shape element count overflows");
        numel *= d;
      }
      const size_t dtype_size = DtypeSize(t.dtype);
      if (dtype_size != 0) {
        if (numel > SIZE_MAX / dtype_size)
          Fail(path, "tensor \"" + name + "\" byte size overflows");
        if (numel * dtype_size != t.nbytes)
          Fail(path, "tensor \"" + name + "\" data_offsets span " +
                         std::to_string(t.nbytes) + " bytes but dtype " +
                         t.dtype + " * shape needs " +
                         std::to_string(numel * dtype_size));
      }

      f.names_.push_back(name);
      f.tensors_.emplace(name, std::move(t));
    }
  } catch (const nlohmann::json::exception& e) {
    Fail(path, std::string("bad header field type: ") + e.what());
  }

  // Overlap check over the offset-sorted spans, mirroring the official
  // reader's sort-then-validate order. The official reader additionally
  // rejects gaps (the ranges must tile [0, data_section) exactly); we
  // deliberately tolerate gaps as an accept-side-only relaxation, since some
  // third-party writers pad/align tensors and a gap cannot make a span
  // escape the mapping.
  std::sort(spans.begin(), spans.end());
  uint64_t prev_end = 0;
  const std::string* prev_name = nullptr;
  for (const auto& [begin, end, name] : spans) {
    if (begin < prev_end)
      Fail(path, "tensor \"" + name + "\" data_offsets overlap tensor \"" +
                     *prev_name + "\"");
    prev_end = end;
    prev_name = &name;
  }

  return f;
}

const StTensor& SafetensorsFile::Get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) Fail(path_, "no tensor named \"" + name + "\"");
  return it->second;
}

void SafetensorsFile::Release() noexcept {
  if (map_ != nullptr) {
    ::munmap(map_, map_size_);
    map_ = nullptr;
    map_size_ = 0;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

SafetensorsFile::~SafetensorsFile() { Release(); }

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : path_(std::move(other.path_)),
      fd_(other.fd_),
      map_(other.map_),
      map_size_(other.map_size_),
      names_(std::move(other.names_)),
      tensors_(std::move(other.tensors_)),
      metadata_(std::move(other.metadata_)) {
  // Leave the moved-from object inert so its dtor is a no-op (no double
  // munmap/close).
  other.fd_ = -1;
  other.map_ = nullptr;
  other.map_size_ = 0;
}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
  if (this != &other) {
    Release();
    path_ = std::move(other.path_);
    fd_ = other.fd_;
    map_ = other.map_;
    map_size_ = other.map_size_;
    names_ = std::move(other.names_);
    tensors_ = std::move(other.tensors_);
    metadata_ = std::move(other.metadata_);
    other.fd_ = -1;
    other.map_ = nullptr;
    other.map_size_ = 0;
  }
  return *this;
}

std::map<std::string, std::string> LoadSafetensorsIndex(
    const std::string& index_json_path) {
  std::ifstream in(index_json_path, std::ios::binary);
  if (!in) Fail(index_json_path, "cannot open index file");
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(in);
  } catch (const nlohmann::json::exception& e) {
    Fail(index_json_path, std::string("JSON parse error: ") + e.what());
  }
  auto it = doc.find("weight_map");
  if (!doc.is_object() || it == doc.end() || !it->is_object())
    Fail(index_json_path, "missing \"weight_map\" object");
  std::map<std::string, std::string> weight_map;
  for (const auto& [tensor, file] : it->items()) {
    if (!file.is_string())
      Fail(index_json_path,
           "weight_map value for \"" + tensor + "\" is not a string");
    std::string shard = file.get<std::string>();
    // UNTRUSTED index: shard names must be plain filenames next to the
    // index, never paths. Reject separators and parent references so a
    // hostile index cannot traverse outside the model directory.
    if (shard.find('/') != std::string::npos ||
        shard.find("..") != std::string::npos)
      Fail(index_json_path, "weight_map value \"" + shard + "\" for \"" +
                                tensor +
                                "\" must be a plain filename (no '/' or "
                                "\"..\")");
    weight_map.emplace(tensor, std::move(shard));
  }
  return weight_map;
}

}  // namespace vllm
