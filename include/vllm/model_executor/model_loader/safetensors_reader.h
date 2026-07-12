// vllm.cpp original (container reader); no upstream mirror.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vllm {

// One tensor entry from a .safetensors header. `data` points into the file's
// read-only mmap and stays valid for the lifetime of the owning
// SafetensorsFile. `dtype` is the raw header string ("F32", "BF16",
// "F8_E4M3", ...): unknown dtypes are kept as an opaque byte span.
struct StTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
};

// One .safetensors file, mmap'd read-only. All header metadata is treated as
// UNTRUSTED: every offset/size is bounds-checked against the actual file size
// and all size arithmetic is overflow-guarded, so Open() throws
// std::runtime_error (message includes the path) on any malformation instead
// of handing out an out-of-bounds span.
class SafetensorsFile {
 public:
  static SafetensorsFile Open(const std::string& path);

  SafetensorsFile(SafetensorsFile&& other) noexcept;
  SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  ~SafetensorsFile();

  // Tensor names in header-appearance order ("__metadata__" excluded).
  const std::vector<std::string>& Names() const { return names_; }
  // Throws std::runtime_error if `name` is not present.
  const StTensor& Get(const std::string& name) const;
  // Contents of the optional "__metadata__" entry (string -> string).
  const std::map<std::string, std::string>& Metadata() const {
    return metadata_;
  }
  // Drop resident pages without invalidating the mapping or tensor pointers.
  // Returns false when the platform/advice call cannot honor the hint.
  bool DiscardResidentPages() const noexcept;

 private:
  SafetensorsFile() = default;
  void Release() noexcept;

  std::string path_;
  int fd_ = -1;
  void* map_ = nullptr;
  size_t map_size_ = 0;
  std::vector<std::string> names_;
  std::map<std::string, StTensor> tensors_;
  std::map<std::string, std::string> metadata_;
};

// Multi-file checkpoints: parses a model.safetensors.index.json and returns
// its "weight_map" (tensor name -> shard file name). Throws
// std::runtime_error (message includes the path) on missing file, malformed
// JSON, or a missing/non-string-valued "weight_map".
std::map<std::string, std::string> LoadSafetensorsIndex(
    const std::string& index_json_path);

}  // namespace vllm
