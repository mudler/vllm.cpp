// vllm.cpp ORIGINAL GGUF-format Qwen3.6 loader (porting-inventory.md §9
// deviation, like gguf_reader / gguf_dequant / the safetensors reader). No
// upstream vLLM mirror.
//
// Loads a GGUF Qwen3.6-A3B MoE checkpoint into the SAME Qwen3_5MoeWeights
// owned-bf16 targets the safetensors loader (qwen3_5_weights.{h,cpp}) produces,
// so the shared M0.9 forward is unchanged. Only the SOURCE (ggml k-quant
// blocks, dequant via gguf_dequant.h) and the tensor NAMES differ.
//
// Tensor names + metadata keys mirror llama.cpp @ 237ad9b (the qwen35moe /
// qwen3next arch):
//   src/llama-arch.cpp      — LLM_TENSOR_NAMES ("token_embd", "output",
//                             "output_norm", "blk.%d.attn_{norm,q,k,v,output,
//                             q_norm,k_norm}", "blk.%d.post_attention_norm",
//                             "blk.%d.ssm_{a,conv1d,dt,alpha,beta,norm,out}",
//                             "blk.%d.attn_qkv", "blk.%d.attn_gate",
//                             "blk.%d.ffn_gate_inp[_shexp]",
//                             "blk.%d.ffn_{gate,up,down}_exps",
//                             "blk.%d.ffn_{gate,up,down}_shexp") + LLM_KV_*
//                             ("%s.block_count", "%s.attention.head_count[_kv]",
//                             "%s.embedding_length", "%s.rope.freq_base",
//                             "%s.expert_count", "%s.expert_used_count",
//                             "%s.attention.layer_norm_rms_epsilon",
//                             "%s.expert[_shared]_feed_forward_length",
//                             "%s.ssm.{conv_kernel,state_size,time_step_rank,
//                             group_count}", "%s.attention.recurrent_layers",
//                             "%s.full_attention_interval").
//   src/models/qwen35moe.cpp — the per-layer create_tensor() dims (the GDN
//                             ssm_* <-> in_proj mapping).
//   conversion/qwen.py       — the convert-time value transforms this loader
//                             INVERTS to recover the raw-HF weights the
//                             safetensors OwnedTensors hold: norm.weight is
//                             stored as (w + 1) [except the GDN ssm_norm];
//                             ssm_a is stored as -exp(A_log); and when
//                             num_v_heads != num_k_heads the V heads are
//                             reordered grouped->tiled.
#pragma once

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Build the HfConfig from a GGUF file's metadata (arch prefix qwen35moe /
// qwen3next / qwen35 [dense]). vocab_size is taken from token_embd's shape
// when the kv is absent; layer_types is derived from the recurrent-layers kv
// or the full_attention_interval (default 4: every interval-th layer is full
// attention, the rest linear/GDN). Throws std::runtime_error on a missing
// required key or an unexpected architecture.
HfConfig HfConfigFromGguf(const GgufFile& gguf);

// Load the whole model from a GGUF file into owned host bf16 tensors, matching
// the safetensors loader's layouts (transposes) and semantics (raw-HF values).
// Uses config.num_hidden_layers, layer_types, num_experts and the GDN head
// dims. MTP/nextn blocks are ignored (as in the safetensors path). Throws on a
// missing tensor or an unsupported ggml quant type (i-quants are Task 3+).
Qwen3_5MoeWeights LoadQwen3_5MoeFromGguf(const GgufFile& gguf,
                                         const HfConfig& config);

// DENSE-arch (`qwen35`, e.g. Qwen3.5-2B) analogue of LoadQwen3_5MoeFromGguf:
// same GDN / full-attention block loaders (identical tensor names + convert
// transforms — llama.cpp's Qwen3_5TextModel shares the Qwen3NextModel convert
// base with the MoE), with the per-layer MoE block replaced by the dense
// SwiGLU MLP ("blk.%d.ffn_{gate,up,down}"). Targets the same
// Qwen3_5DenseWeights the 27B safetensors loader produces (bf16 fields; the
// fp4 variants stay empty).
Qwen3_5DenseWeights LoadQwen3_5DenseFromGguf(const GgufFile& gguf,
                                             const HfConfig& config);

}  // namespace vllm
