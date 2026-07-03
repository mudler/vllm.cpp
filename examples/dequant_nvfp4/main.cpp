// dequant_nvfp4: dequantizes ONE modelopt W4A16_NVFP4 weight tensor from a
// .safetensors shard using our SafetensorsFile reader + DequantNvfp4ToBf16, and
// writes the raw bf16 output (little-endian uint16, row-major [out, in]) to a
// file. The Python side (tools/parity/verify_nvfp4_dequant.py) recomputes the
// same tensor with the pinned modelopt code and diffs the bf16 bit patterns.
//
// Usage:
//   dequant_nvfp4 <shard.safetensors> <weight_name> <scale_name> <scale2_name> <out_raw>
// Prints: "<out_dim> <in_dim>" on success.
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace {

void RequireDtype(const vllm::StTensor& t, const std::string& name,
                  const std::string& want_dtype) {
  if (t.dtype != want_dtype) {
    throw std::runtime_error("tensor '" + name + "' has dtype " + t.dtype +
                             ", expected " + want_dtype);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: dequant_nvfp4 <shard.safetensors> <weight> <scale> "
                 "<scale2> <out_raw>\n";
    return 2;
  }
  try {
    vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(argv[1]);
    const vllm::StTensor& w = file.Get(argv[2]);
    const vllm::StTensor& sf = file.Get(argv[3]);
    const vllm::StTensor& s2 = file.Get(argv[4]);
    RequireDtype(w, argv[2], "U8");
    RequireDtype(sf, argv[3], "F8_E4M3");
    RequireDtype(s2, argv[4], "F32");

    if (w.shape.size() != 2) {
      throw std::runtime_error("weight must be 2-D [out, in/2]");
    }
    const int64_t out_dim = w.shape[0];
    const int64_t in_dim = w.shape[1] * 2;

    // weight_scale sanity: [out, in/16].
    if (sf.shape.size() != 2 || sf.shape[0] != out_dim ||
        sf.shape[1] != in_dim / 16) {
      throw std::runtime_error("weight_scale shape mismatch");
    }
    // weight_scale_2: per-tensor f32 scalar (shape [] or [1]).
    if (s2.nbytes < sizeof(float)) {
      throw std::runtime_error("weight_scale_2 too small");
    }
    float ws2 = 0.0F;
    std::memcpy(&ws2, s2.data, sizeof(float));

    std::vector<uint16_t> out(static_cast<size_t>(out_dim) *
                              static_cast<size_t>(in_dim));
    vllm::DequantNvfp4ToBf16(w.data, sf.data, ws2, out_dim, in_dim, out.data());

    std::ofstream os(argv[5], std::ios::binary);
    if (!os) throw std::runtime_error(std::string("cannot open ") + argv[5]);
    os.write(reinterpret_cast<const char*>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(uint16_t)));
    if (!os) throw std::runtime_error("write failed");

    std::cout << out_dim << " " << in_dim << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "dequant_nvfp4: " << e.what() << "\n";
    return 1;
  }
}
