// vllm.cpp — stable FlashInfer SM12 NVFP4 tactic identity.
//
// Mirrors FlashInfer 0.6.12
// `flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h:187-220`: eight CTA
// shapes, each ordered swap-AB DP, ordinary DP, swap-AB Stream-K, ordinary
// Stream-K. This metadata is CUDA-free so descriptor order remains covered by
// the CPU suite and can become W3's persistent-cache ABI.
#pragma once

#include <array>
#include <cstddef>

namespace vt::cuda::nvfp4 {

struct TacticDescriptor {
  int id;
  int tile_m;
  int tile_n;
  int tile_k;
  bool swap_ab;
  bool stream_k;
  const char* name;
};

inline constexpr std::array<TacticDescriptor, 32> kFullTacticDescriptors{{
    {0, 128, 32, 128, true, false, "128x32x128/swap/dp"},
    {1, 128, 32, 128, false, false, "128x32x128/ordinary/dp"},
    {2, 128, 32, 128, true, true, "128x32x128/swap/stream-k"},
    {3, 128, 32, 128, false, true, "128x32x128/ordinary/stream-k"},
    {4, 128, 32, 256, true, false, "128x32x256/swap/dp"},
    {5, 128, 32, 256, false, false, "128x32x256/ordinary/dp"},
    {6, 128, 32, 256, true, true, "128x32x256/swap/stream-k"},
    {7, 128, 32, 256, false, true, "128x32x256/ordinary/stream-k"},
    {8, 128, 64, 128, true, false, "128x64x128/swap/dp"},
    {9, 128, 64, 128, false, false, "128x64x128/ordinary/dp"},
    {10, 128, 64, 128, true, true, "128x64x128/swap/stream-k"},
    {11, 128, 64, 128, false, true, "128x64x128/ordinary/stream-k"},
    {12, 128, 64, 256, true, false, "128x64x256/swap/dp"},
    {13, 128, 64, 256, false, false, "128x64x256/ordinary/dp"},
    {14, 128, 64, 256, true, true, "128x64x256/swap/stream-k"},
    {15, 128, 64, 256, false, true, "128x64x256/ordinary/stream-k"},
    {16, 128, 128, 128, true, false, "128x128x128/swap/dp"},
    {17, 128, 128, 128, false, false, "128x128x128/ordinary/dp"},
    {18, 128, 128, 128, true, true, "128x128x128/swap/stream-k"},
    {19, 128, 128, 128, false, true, "128x128x128/ordinary/stream-k"},
    {20, 128, 128, 256, true, false, "128x128x256/swap/dp"},
    {21, 128, 128, 256, false, false, "128x128x256/ordinary/dp"},
    {22, 128, 128, 256, true, true, "128x128x256/swap/stream-k"},
    {23, 128, 128, 256, false, true, "128x128x256/ordinary/stream-k"},
    {24, 256, 128, 128, true, false, "256x128x128/swap/dp"},
    {25, 256, 128, 128, false, false, "256x128x128/ordinary/dp"},
    {26, 256, 128, 128, true, true, "256x128x128/swap/stream-k"},
    {27, 256, 128, 128, false, true, "256x128x128/ordinary/stream-k"},
    {28, 128, 256, 128, true, false, "128x256x128/swap/dp"},
    {29, 128, 256, 128, false, false, "128x256x128/ordinary/dp"},
    {30, 128, 256, 128, true, true, "128x256x128/swap/stream-k"},
    {31, 128, 256, 128, false, true, "128x256x128/ordinary/stream-k"},
}};

// Exact W1 candidate order. The IDs name the corresponding full-surface tile
// and orientation, while the launch functions retain W1's original builder and
// scheduler semantics behind VT_FP4_FULL_TACTICS=0.
inline constexpr std::array<int, 4> kW1TacticIds{{17, 25, 21, 29}};

inline constexpr const TacticDescriptor* TacticDescriptorForId(int id) {
  if (id < 0 || static_cast<size_t>(id) >= kFullTacticDescriptors.size()) return nullptr;
  return &kFullTacticDescriptors[static_cast<size_t>(id)];
}

}  // namespace vt::cuda::nvfp4
