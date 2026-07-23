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

#include <cstdint>

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// L2 (keep-quant residency, .agents/specs/gguf-keep-quant-loader.md). Take `n`
// rows of `k` elements of `tensor`'s RAW ggml blocks — starting at row
// `row_offset`, which is how a stacked [E, out, in] expert tensor is split —
// as a block-typed tensor.
//
// The result is [N = n, K = k] with `nk = true` and `dtype` the block dtype:
// GGUF's on-disk [out, in] row-major order IS ggml's src0 layout and IS our
// MatmulBT orientation, so keep-quant needs NO transpose. It could not do one
// anyway — transposing a block encoding would require requantizing.
//
// RESIDENCY (L5). `mmap_src` null (the default) COPIES the bytes into an owned
// buffer — L2's original behavior. Non-null BORROWS them in place out of that
// file's read-only mapping and takes a refcount on it, which is llama.cpp's
// `use_mmap` arm (src/llama-model-loader.cpp:1385 `load_data_for` points the
// tensor at `mapping->addr + w.offs` instead of `read_raw`-ing into it, @
// 237ad9b96). The two are byte-identical by construction: the same file bytes in
// the same order, read rather than memcpy'd first. `mmap_src` must be the file
// `tensor` came from; the span is re-validated against its mapping.
//
// CIQ G7. When `repack` is set AND the slice is repack-eligible
// (vt::cpu::QuantRepackEligible: i8mm live, q8_0, n % 4 == 0, k % 32 == 0), the
// resident bytes are REPACKED once into the CPU i8mm interleave (block_q8_0x4)
// and `repacked = true` is marked on the result. Repacking mutates the buffer,
// so it forces the COPY residency (a read-only mmap borrow cannot be rewritten):
// `mmap_src` is ignored for the tensors it touches. A non-eligible slice ignores
// the flag and keeps its normal residency. The transform is a byte permutation,
// so the GEMM stays bit-identical to the plain-block path.
//
// Throws when the tensor's encoding is not keep-quant capable, when `k` is not
// a whole number of blocks (ggml_row_size's precondition), when the requested
// rows fall outside the tensor's validated byte span, or when a borrowed span is
// not inside `mmap_src`'s mapping.
OwnedTensor OwnGgufQuantBlocks(const GgufTensorInfo& tensor, int64_t n,
                               int64_t k, int64_t row_offset = 0,
                               const GgufFile* mmap_src = nullptr,
                               bool repack = false);

// L6 (keep-f16 residency). Take `n` rows of `k` F16 elements of `tensor`'s raw
// bytes — starting at row `row_offset` (how a stacked [E, out, in] expert tensor
// is split) — as an F16 tensor with orientation `nk` ([N=n, K=k] when true; a
// [vocab, H] gather table when false). GGUF's on-disk [out, in] row-major order
// IS the MatmulBT [N, K] orientation, so a matmul weight needs NO transpose; the
// elementwise f16 GEMM consumes it directly (cpu_matmul_elem f16 vec_dot).
//
// RESIDENCY mirrors OwnGgufQuantBlocks: `mmap_src` null COPIES the bytes into an
// owned buffer; non-null BORROWS them in place out of that file's read-only
// mapping and refcounts it (llama.cpp `use_mmap`, src/llama-model-loader.cpp:1385).
// Byte-for-byte lossless in both arms — the SAME f16 bytes, read not re-encoded.
//
// Throws when the tensor is not F16, on a bad slice, when the rows fall outside
// the tensor's validated span, or when a borrowed span is not inside the mapping.
OwnedTensor OwnGgufF16(const GgufTensorInfo& tensor, int64_t n, int64_t k,
                       int64_t row_offset = 0,
                       const GgufFile* mmap_src = nullptr, bool nk = true);

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
//
// `policy` (optional) selects per-tensor residency: null reads the process
// environment (GgufLoadPolicy::FromEnv — VT_CPU_REF / VT_GGUF_KEEP_QUANT),
// which with today's defaults reproduces the historical all-bf16 load exactly.
// Tests pass an explicit policy (and may attach a routing audit hook).
Qwen3_5MoeWeights LoadQwen3_5MoeFromGguf(const GgufFile& gguf,
                                         const HfConfig& config,
                                         const GgufLoadPolicy* policy = nullptr);

// DENSE-arch (`qwen35`, e.g. Qwen3.5-2B) analogue of LoadQwen3_5MoeFromGguf:
// same GDN / full-attention block loaders (identical tensor names + convert
// transforms — llama.cpp's Qwen3_5TextModel shares the Qwen3NextModel convert
// base with the MoE), with the per-layer MoE block replaced by the dense
// SwiGLU MLP ("blk.%d.ffn_{gate,up,down}"). Targets the same
// Qwen3_5DenseWeights the 27B safetensors loader produces (bf16 fields; the
// fp4 variants stay empty).
Qwen3_5DenseWeights LoadQwen3_5DenseFromGguf(
    const GgufFile& gguf, const HfConfig& config,
    const GgufLoadPolicy* policy = nullptr);

}  // namespace vllm
