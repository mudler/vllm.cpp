// GPT-2 backbone — IndexTTS-2.5's stage-0 talker (W2, #634).
//
// Ported from vllm/model_executor/models/gpt2.py @ 555967922 (the parity pin):
//   GPT2Attention.__init__/forward  gpt2.py:61-110
//   GPT2MLP.__init__/forward        gpt2.py:113-143
//   GPT2Block.forward               gpt2.py:165-180
//   GPT2Model.forward               gpt2.py:217-240
//   _transpose_conv1d               gpt2.py:242-254
//
// This is the HOST REFERENCE forward, the same shape the MiniMax-H3 and LTX-2.5
// lanes started from: a portable f32 implementation gated against upstream
// before any device path exists. It is not wired to the runner, the ABI or the
// server; that is W6a/W6b in .agents/specs/indextts-2-5.md.
//
// TWO THINGS THIS ARCHITECTURE GETS WRONG QUIETLY, both gated in test_gpt2.cpp:
//
//  1. CONV1D ORIENTATION. HF's GPT-2 uses Conv1D, not Linear, so c_attn/c_proj/
//     c_fc store their 2D weight as [in, out]. `Load` transposes to [out, in].
//     Skipping that produces a model that runs, emits plausible tokens, and is
//     wrong.
//  2. CAUSALITY. Without the upper-triangular mask every position attends to the
//     future. The output stays fluent, so only a perturbation test sees it.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vllm {
namespace gpt2 {

// The subset of GPT2Config the backbone reads (gpt2.py:184-210).
struct Params {
  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 0;
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  // `n_inner` when the config sets it, else 4 * hidden_size (gpt2.py:157).
  int64_t inner_size = 0;
  double layer_norm_eps = 1e-5;

  int64_t head_dim() const { return hidden_size / num_attention_heads; }
};

// Raw checkpoint tensors, in the orientation the file carries them.
struct CheckpointTensors {
  std::map<std::string, std::vector<float>> values;
  std::map<std::string, std::vector<int64_t>> shapes;

  void Set(const std::string& name, std::vector<int64_t> shape, std::vector<float> data);
  const std::vector<float>& Get(const std::string& name) const;
  const std::vector<int64_t>& Shape(const std::string& name) const;
  bool Has(const std::string& name) const { return values.count(name) != 0; }
};

// One transformer block's weights, already transposed for the matmul.
struct LayerWeights {
  std::vector<float> ln_1_weight, ln_1_bias;
  std::vector<float> ln_2_weight, ln_2_bias;
  std::vector<float> c_attn_weight, c_attn_bias;  // [3H, H], [3H]
  std::vector<float> c_proj_weight, c_proj_bias;  // [H, H],  [H]
  std::vector<float> c_fc_weight, c_fc_bias;      // [I, H],  [I]
  std::vector<float> mlp_c_proj_weight, mlp_c_proj_bias;  // [H, I], [H]
};

struct Weights {
  std::vector<float> wte;  // [vocab, H]
  std::vector<float> wpe;  // [positions, H]
  std::vector<float> ln_f_weight, ln_f_bias;
  std::vector<LayerWeights> layers;
};

// Materialize `Weights` from a checkpoint, applying the Conv1D transpose
// (gpt2.py:242-254). Throws BY NAME on a missing tensor rather than reading
// zeros.
Weights Load(const Params& params, const CheckpointTensors& tensors);

// gpt2.py:217-240. Returns the post-`ln_f` hidden states, [seq, hidden].
std::vector<float> ForwardHost(const Params& params, const Weights& weights,
                               const std::vector<int64_t>& input_ids,
                               const std::vector<int64_t>& positions);

// The same stack, driven from EMBEDDINGS rather than token ids.
//
// The IndexTTS talker's prompt has no token ids: it is projected conditioning
// rows concatenated with embedded text, assembled by `talker::PrepareInputs`.
// Everything after gpt2.py:225-229 is independent of where the embeddings came
// from, so `ForwardHost` now builds them and delegates here.
//
// `inputs_embeds` is [seq, hidden]. NOTE that position embeddings are NOT added
// here -- the caller owns them, because the talker uses its own mel and text
// position tables rather than `wpe`.
std::vector<float> ForwardHostEmbeds(const Params& params, const Weights& weights,
                                     const std::vector<float>& inputs_embeds);

// Tied lm_head: GPT-2 ties the output projection to `wte`. Returns [seq, vocab].
std::vector<float> LogitsHost(const Params& params, const Weights& weights,
                              const std::vector<float>& hidden);

}  // namespace gpt2
}  // namespace vllm
