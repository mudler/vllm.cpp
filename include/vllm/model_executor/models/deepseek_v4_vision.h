// DeepSeek-V4 Flash Vision ViT and downsample-3 aligner.
//
// Ported from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp at revision
// 86f746b36186f0e567729a5c06a8c918caba82a9:
//   inference/vision.py::{get_vision_cos_sin,apply_rotary,RMSNorm,PatchEmbed,
//     Attention,MLP,Block,ViT,Aligner}
//
// W2 is a standalone, config-driven composition over public vt operations. It is
// deliberately not reached by the DeepSeek-V4 registry; W4 owns that production
// call site. The same Tensor/Queue entry point is used by every device provider.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm::multimodal {

struct DeepSeekV4VisionConfig {
  int64_t patch_size = 14;
  int64_t hidden_size = 1024;
  int64_t num_heads = 16;
  int64_t depth = 32;
  int64_t intermediate_size = 2816;
  int64_t output_size = 4096;
  int64_t downsample_ratio = 3;
  float norm_epsilon = 1.0e-6f;
  double rope_theta = 10000.0;
  vt::DType compute_dtype = vt::DType::kBF16;

  int64_t patch_dim() const { return 3 * patch_size * patch_size; }
  int64_t head_dim() const { return hidden_size / num_heads; }
  int64_t aligner_input_size() const {
    return hidden_size * downsample_ratio * downsample_ratio;
  }
  int64_t aligned_rows(int64_t height, int64_t width) const;
};

// Tensor views are non-owning and use torch Linear storage order [out,in]. All
// linear weights and biases use compute_dtype. RMSNorm weights remain f32: the
// pinned module declares them f32 and widens x before the variance and affine.
struct DeepSeekV4VisionBlockWeights {
  vt::Tensor norm1_weight;       // f32 [hidden]
  vt::Tensor qkv_weight;         // model dtype [3*hidden, hidden]
  vt::Tensor qkv_bias;           // model dtype [3*hidden]
  vt::Tensor out_weight;         // model dtype [hidden, hidden]
  vt::Tensor out_bias;           // model dtype [hidden]
  vt::Tensor norm2_weight;       // f32 [hidden]
  vt::Tensor mlp_w1_weight;      // model dtype [2*intermediate, hidden]
  vt::Tensor mlp_w2_weight;      // model dtype [hidden, intermediate]
};

struct DeepSeekV4VisionWeights {
  vt::Tensor patch_weight;       // model dtype [hidden, 3*patch_size^2]
  vt::Tensor patch_bias;         // model dtype [hidden]
  std::vector<DeepSeekV4VisionBlockWeights> blocks;
  vt::Tensor final_norm_weight;  // f32 [hidden]
  vt::Tensor aligner_w1_weight;  // model dtype [output, hidden*r^2]
  vt::Tensor aligner_w1_bias;    // model dtype [output]
  vt::Tensor aligner_w2_weight;  // model dtype [output, output]
  vt::Tensor aligner_w2_bias;    // model dtype [output]
};

// One internal scratch buffer's declared storage dtype, recorded in allocation
// order. AGENTS.md, "Inherit vLLM defaults": a token gate CANNOT detect a dtype
// that is too wide, because the values still match while the path moves twice
// the bytes. Widening the attention-output buffer to f32 left every stage
// golden green, so the memory format needs its own assertion.
struct DeepSeekV4VisionScratchDType {
  const char* name = nullptr;
  vt::DType dtype = vt::DType::kBF16;
};

// Optional device-tensor captures for parity gates. Production passes nullptr
// and performs no stage copies. A non-null tensor must have the documented
// contiguous shape, the model dtype, and the queue device.
struct DeepSeekV4VisionCapture {
  vt::Tensor* patch_embedding = nullptr;       // [patches, hidden]
  std::vector<vt::Tensor*> block_outputs;      // empty, or one [patches,hidden] per layer
  vt::Tensor* final_norm = nullptr;            // [patches, hidden]
  vt::Tensor* aligner_unfold = nullptr;         // [aligned_rows, hidden*r^2]
  vt::Tensor* aligner_hidden = nullptr;         // [aligned_rows, output]
  vt::Tensor* aligner_gelu = nullptr;           // [aligned_rows, output]

  // Non-null: the forward appends one entry per internal scratch buffer it
  // allocates, so a test can assert the memory format of the model path. The
  // vision stage CLEARS it and the aligner stage APPENDS, so a whole Forward
  // records both stages as one sequence and an aligner call on its own adds to
  // whatever the caller's vector already holds.
  //
  // The sequence is allocation order WITHIN each stage. `Forward`'s own
  // `vision` buffer is allocated before either stage but recorded between them,
  // because the vision stage's clear would otherwise erase it.
  //
  // THIS LIST IS A DECLARATION, NOT A MEASUREMENT. `RecordScratch` is called by
  // hand at each allocation site, so a buffer that does not call it is invisible
  // here: a review added a hoisted f32 buffer with no call and every assertion
  // over this list stayed green. What bounds the buffers the list cannot see is
  // the pool byte cap in `test_deepseek_v4_vision.cpp`, which prices one Forward
  // in driver allocations and bytes that no call site can bypass. Read the two
  // together; neither is sufficient alone.
  std::vector<DeepSeekV4VisionScratchDType>* scratch_dtypes = nullptr;
};

// The load-time storage-layout markers the shared MlpGateUpMethodBase seam
// actually holds for one block's gate-up weight.
//
// Observable because a borrow that DROPS them is invisible to every value gate
// on this host: dtype, rank, shape and byte count are all unchanged, and
// `vt::cpu::QuantRepackActive()` is true only on an aarch64 i8mm host, so the
// wrongly-decoded weight is not even wrong here. It was wrong on `thor`: the
// shared `dense_attn::ResidentWeight` dropped `repacked` and an i8mm-interleaved
// `block_q8_0x4` buffer (136-byte blocks) was decoded as flat `q8_0` (34-byte
// blocks), which produced NaN, then all-zero logits, then token id 0, with
// nothing logged because the `lm_head` GEMM swallowed the NaN. Fixed on `main`
// at `7a937db8a` (#2031); this accessor is what keeps the same loss from being
// re-introduced by a private borrow helper here.
struct DeepSeekV4VisionStorageMarkers {
  bool repacked = false;
  bool q8_0_aligned = false;
  bool elem_kn_repacked = false;
};

// Host f32 oracle helper. For each patch row it returns head_dim/2 values per
// table: all height frequencies first, then all width frequencies, exactly as
// torch.stack([hpos,wpos]).flatten(1) in the pinned source.
void DeepSeekV4VisionRopeCosSin(int64_t height, int64_t width,
                                const DeepSeekV4VisionConfig& config,
                                std::vector<float>* cosine,
                                std::vector<float>* sine);

class DeepSeekV4Vision {
 public:
  DeepSeekV4Vision(vt::Backend& backend, DeepSeekV4VisionConfig config,
                   DeepSeekV4VisionWeights weights);
  ~DeepSeekV4Vision();

  DeepSeekV4Vision(const DeepSeekV4Vision&) = delete;
  DeepSeekV4Vision& operator=(const DeepSeekV4Vision&) = delete;
  DeepSeekV4Vision(DeepSeekV4Vision&&) noexcept;
  DeepSeekV4Vision& operator=(DeepSeekV4Vision&&) noexcept;

  const DeepSeekV4VisionConfig& config() const;

  // patches [height*width, patch_dim] -> output
  // [ceil(height/r)*ceil(width/r), output_size].
  void Forward(vt::Queue& queue, vt::Tensor& output,
               const vt::Tensor& patches, int64_t height, int64_t width,
               DeepSeekV4VisionCapture* capture = nullptr);

  // Lowest stage seams used by the W2 parity gate and by W3 composition.
  void VisionForward(vt::Queue& queue, vt::Tensor& output,
                     const vt::Tensor& patches, int64_t height, int64_t width,
                     DeepSeekV4VisionCapture* capture = nullptr);
  void AlignerForward(vt::Queue& queue, vt::Tensor& output,
                      const vt::Tensor& vision, int64_t height, int64_t width,
                      DeepSeekV4VisionCapture* capture = nullptr);

  // Observable cache size for allocation-stability tests. Geometry entries hold
  // reusable f32 RoPE data, positions, and exact unfold indices by shape/device.
  size_t cached_geometry_count() const;

  // Observable storage-layout markers for the gate-up weight this model handed
  // to the shared MLP seam. See DeepSeekV4VisionStorageMarkers.
  DeepSeekV4VisionStorageMarkers mlp_gate_up_markers(int64_t block) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm::multimodal
