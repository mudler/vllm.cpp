// Does a REAL LTX-2.5 text encoder load? — the caption-projection arms, measured.
//
// Loads a shipped `gemma4-12b-with-proj-*.safetensors` through
// `Ltx2LoadTextEncoderFromSafetensors`, which is the exact function the engine
// calls at `src/vllm/multimodal/ltx2_video.cpp` during
// `vllm_video_engine_load`, and prints the geometry, both projections' resolved
// widths, the quantized-module inventory and the first weight and bias values of
// the video projection. It touches no product code of its own.
//
// It exists because the two shipped text encoders store the SAME projection in
// two different formats — U8 [4096, 94080] beside a scale pair in the torchao
// file, BF16 [4096, 188160] with no scale tensor at all in the bf16 one — and a
// synthetic fixture cannot prove that the real bytes of either resolve. #2140 is
// what happens when only one of the two is ever fed: the loader doubled the bf16
// file's already-logical 188160 to 376320 and refused.
//
// ─── BUILD AND RUN (there is no CMake target; this is the recorded recipe) ───
// Deliberately not a target, for the same reason its two sibling probes are not:
// a probe should not charge every configure. Written down rather than implied,
// because a reviewer cannot re-run a probe whose compile line was never recorded.
//
//   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
//   ninja -C build vllm
//   g++ -O2 -std=c++20 -Iinclude -Ithird_party
//       scripts/probe_ltx2_text_encoder_load.cpp build/libvllm.a -o /tmp/ltx2_te -pthread
//   (one line; it is split here only because a trailing backslash inside a `//`
//   comment is -Wcomment, and this file is compiled warning-clean on purpose)
//   /tmp/ltx2_te <text_encoder.safetensors>
//
// Exit 0 and a trailing `OK` is a load; exit 1 and `REFUSED: <message>` is the
// loader's own refusal, printed rather than swallowed. It reads only the header
// and the tensors the load materializes, so it costs about 2.4 GB of reads
// against a 24.5 GB file and needs no GPU.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2_loader.h"

namespace {

float Bf16ToF32(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f = 0.0F;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// Peak resident set, so a reader knows what the load actually cost.
long PeakRssKb() {
  std::ifstream status("/proc/self/status");
  std::string key;
  long value = 0;
  while (status >> key) {
    if (key == "VmHWM:") {
      status >> value;
      return value;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <text_encoder.safetensors>\n", argv[0]);
    return 2;
  }
  try {
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(argv[1]);
    std::printf("opened %s, %zu tensors\n", argv[1], file.Names().size());
    const vllm::Ltx2TextEncoderCheckpoint ck =
        vllm::Ltx2LoadTextEncoderFromSafetensors(file);
    std::printf("gemma_hidden_size       = %lld\n",
                static_cast<long long>(ck.gemma_hidden_size));
    std::printf("gemma_num_hidden_layers = %lld\n",
                static_cast<long long>(ck.gemma_num_hidden_layers));
    std::printf("geometry hidden*(L+1)   = %lld\n",
                static_cast<long long>(ck.gemma_hidden_size *
                                       (ck.gemma_num_hidden_layers + 1)));
    std::printf("video  out=%lld in=%lld weights=%zu bias=%zu\n",
                static_cast<long long>(ck.video.out_features),
                static_cast<long long>(ck.video.in_features),
                ck.video.weight_bf16.size(), ck.video.bias_bf16.size());
    std::printf("audio  out=%lld in=%lld weights=%zu bias=%zu\n",
                static_cast<long long>(ck.audio.out_features),
                static_cast<long long>(ck.audio.in_features),
                ck.audio.weight_bf16.size(), ck.audio.bias_bf16.size());
    std::printf("quantized_modules       = %zu\n", ck.quantized_modules.size());
    std::printf("tokenizer_json bytes    = %zu, has_config=%d\n",
                ck.assets.tokenizer_json.size(), static_cast<int>(ck.assets.has_config));
    if (ck.video.weight_bf16.size() >= 4 && ck.video.bias_bf16.size() >= 4) {
      std::printf("video.weight[0..3]      = %.8f %.8f %.8f %.8f\n",
                  Bf16ToF32(ck.video.weight_bf16[0]), Bf16ToF32(ck.video.weight_bf16[1]),
                  Bf16ToF32(ck.video.weight_bf16[2]), Bf16ToF32(ck.video.weight_bf16[3]));
      std::printf("video.bias[0..3]        = %.8f %.8f %.8f %.8f\n",
                  Bf16ToF32(ck.video.bias_bf16[0]), Bf16ToF32(ck.video.bias_bf16[1]),
                  Bf16ToF32(ck.video.bias_bf16[2]), Bf16ToF32(ck.video.bias_bf16[3]));
    }
    // The same widening the engine applies immediately after the load. It reads
    // no file shape and no dtype, so a wrong in_features arrives here unchanged.
    const vllm::Ltx2TextEncoderWeights w = vllm::Ltx2WidenTextProjectionsToF32(ck);
    std::printf("widened video out=%lld in=%lld weights=%zu bias=%zu\n",
                static_cast<long long>(w.video.out_features),
                static_cast<long long>(w.video.in_features), w.video.weight.size(),
                w.video.bias.size());
    std::printf("VmHWM kB                = %ld\n", PeakRssKb());
    std::printf("OK\n");
    return 0;
  } catch (const std::exception& e) {
    std::printf("REFUSED: %s\n", e.what());
    return 1;
  }
}
