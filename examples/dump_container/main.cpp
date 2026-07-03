// dump_container: prints a sorted per-tensor manifest of a .safetensors or
// .gguf file for byte-level verification against the Python truth side
// (tools/parity/verify_containers.py). One line per tensor:
//   <name> <dtype> [d0,d1,...] <nbytes> <sha256 of first min(nbytes, 64KiB)>
// followed by a final "TOTAL <count> tensors" line.
#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "sha256.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace {

constexpr size_t kHashPrefixBytes = 65536;

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string ShapeStr(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) out += ",";
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

std::string HashPrefix(const uint8_t* data, size_t nbytes) {
  examples::Sha256 h;
  h.Update(data, std::min(nbytes, kHashPrefixBytes));
  return h.HexDigest();
}

// (name, rest-of-line) rows, sorted by name before printing so the manifest
// is order-independent across readers.
using Rows = std::vector<std::pair<std::string, std::string>>;

Rows DumpSafetensors(const std::string& path) {
  vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
  Rows rows;
  for (const std::string& name : file.Names()) {
    const vllm::StTensor& t = file.Get(name);
    rows.emplace_back(name, t.dtype + " " + ShapeStr(t.shape) + " " +
                                std::to_string(t.nbytes) + " " +
                                HashPrefix(t.data, t.nbytes));
  }
  return rows;
}

Rows DumpGguf(const std::string& path) {
  vllm::GgufFile file = vllm::GgufFile::Open(path);
  Rows rows;
  for (const vllm::GgufTensorInfo& t : file.Tensors()) {
    rows.emplace_back(t.name, std::string(vllm::GgmlTraits(t.ggml_type).name) +
                                  " " + ShapeStr(t.shape) + " " +
                                  std::to_string(t.nbytes) + " " +
                                  HashPrefix(t.data, t.nbytes));
  }
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: dump_container <path.safetensors|path.gguf>\n";
    return 2;
  }
  const std::string path = argv[1];
  try {
    Rows rows;
    if (EndsWith(path, ".safetensors")) {
      rows = DumpSafetensors(path);
    } else if (EndsWith(path, ".gguf")) {
      rows = DumpGguf(path);
    } else {
      std::cerr << "dump_container: unsupported suffix (want .safetensors or "
                   ".gguf): "
                << path << "\n";
      return 2;
    }
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [name, rest] : rows) std::cout << name << " " << rest << "\n";
    std::cout << "TOTAL " << rows.size() << " tensors\n";
  } catch (const std::exception& e) {
    std::cerr << "dump_container: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
