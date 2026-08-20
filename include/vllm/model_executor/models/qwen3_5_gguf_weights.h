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
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
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
// `cuda_align` (Brick 4): if set AND the encoding is Q8_0, repack the blocks into
// the CUDA coalesced-load layout (copy off the mmap into an owned buffer, drop the
// source file pages) instead of borrowing/plain-copying — see RepackQ8_0Cuda.
OwnedTensor OwnGgufQuantBlocks(const GgufTensorInfo& tensor, int64_t n,
                               int64_t k, int64_t row_offset = 0,
                               const GgufFile* mmap_src = nullptr,
                               bool repack = false, bool cuda_align = false);

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
// `elem_kn_repack` transposes an `nk` weight's bytes [N,K] -> [K,N] at load so
// `vt::MatmulBT` reaches the transpose-free CPU micro-kernels
// (KERNEL-GEMM-CPU-TILED lever 2); it marks `OwnedTensor::elem_kn_repacked` and
// is ignored for a gather table (`nk == false`). Defaulted OFF so every existing
// caller, including tests/vllm/test_gguf_keep_quant.cpp, is unchanged.
OwnedTensor OwnGgufF16(const GgufTensorInfo& tensor, int64_t n, int64_t k,
                       int64_t row_offset = 0,
                       const GgufFile* mmap_src = nullptr, bool nk = true,
                       bool elem_kn_repack = false);

// Build the HfConfig from a GGUF file's metadata (arch prefix qwen35moe /
// qwen3next / qwen35 [dense]). vocab_size is taken from token_embd's shape
// when the kv is absent; layer_types is derived from the recurrent-layers kv
// or the full_attention_interval (default 4: every interval-th layer is full
// attention, the rest linear/GDN). Throws std::runtime_error on a missing
// required key or an unexpected architecture.
HfConfig HfConfigFromGguf(const GgufFile& gguf);

// ─── Tensor accounting (QUANT-QWEN38-27B-GGUF-ARM, issue #821) ──────────────
//
// True when `general.architecture` names one of the three GGUF families this
// translation unit loads (`qwen35`, `qwen35moe`, `qwen3next`). Mirrors
// `IsNemotronHGguf`'s shape: the model that owns the family answers the
// question, so the dispatch in `model_loader.cpp` does not carry a second copy
// of the arch list.
bool IsQwen3_5Gguf(const GgufFile& gguf);

// The EXACT set of tensor names the loaders in this file read for `config`:
// the embedding and head, the trunk blocks under their `layer_types`, and — when
// `mtp_num_hidden_layers` is set — the MTP head blocks `LoadQwen3_5MTPFromGguf`
// reads at `blk.{num_hidden_layers + i}`.
//
// `has_output_weight` picks the tied-embedding arm: a GGUF that omits
// `output.weight` has the head aliased onto `token_embd.weight` (llama.cpp
// TENSOR_DUPLICATED), and `LoadEmbedAndHead` resolves it that way, so the name
// is expected only when the file ships it.
std::vector<std::string> Qwen3_5GgufExpectedTensors(const HfConfig& config,
                                                    bool has_output_weight);

// What the enumeration above says about a file's actual tensor table.
struct Qwen3_5GgufAccounting {
  std::vector<std::string> missing;      // enumerated, absent from the file
  std::vector<std::string> unaccounted;  // in the file, read by nothing
  int64_t enumerated = 0;
  int64_t present = 0;
};

// Account `present` (a file's tensor names) against the enumeration for
// `config`. An NVFP4 sidecar (`<stem>.scale`, `<stem>.input_scale`, read by
// `GgufNvfp4SidecarScalars`) is accounted whenever its `<stem>.weight` is
// enumerated, because whether it exists is a property of the ENCODING of that
// weight rather than of the architecture.
Qwen3_5GgufAccounting Qwen3_5GgufAccountTensors(
    const HfConfig& config, const std::vector<std::string>& present);

// Refuse a qwen3_5-family GGUF that carries tensors NOTHING in this file reads,
// naming them.
//
// ONE DIRECTION, deliberately. A tensor the loaders ask for and the file lacks
// already refuses by name, at `GgufFile::Get`, on every arm. A tensor the file
// carries and no loader reads has no detector at all, and it is the direction
// that is SILENT: `Qwen3.8-27B-Q4_K_M.gguf` states `qwen35.block_count = 65`
// and `qwen35.nextn_predict_layers = 1`, so a reader that spends the whole 65
// on the trunk builds a 65-layer model out of a 64-layer checkpoint plus an MTP
// drafter, loads, decodes fluently, and is the wrong graph. Under that defect
// `blk.64`'s four `nextn.*` tensors and its six full-attention projections stop
// being read, and this is what says so.
//
// The other direction is still gated, on the committed manifest rather than
// here: `tests/vllm/models/test_qwen38_27b_gguf_manifest.cpp` asserts zero in
// BOTH directions against the real 866-name table with no asset in CI.
void RefuseUnaccountedQwen3_5Gguf(const GgufFile& gguf, const HfConfig& config);

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
// Load the Multi-Token Prediction head from a head-carrying GGUF
// (`SPEC-MTP-GGUF`). llama.cpp's Qwen3.5 converter folds the HF `mtp.*` block
// into the block list at index `config.num_hidden_layers`, under the
// DeepSeek-style `nextn` names; this reads it back using the same conventions
// (norm (w+1) un-shift, quantized matmul routing, torch [N, K] shapes) as the
// trunk loaders above, so the head matches the trunk it speculates for.
// Requires `config.raw["mtp_num_hidden_layers"]`, which HfConfigFromGguf
// publishes from `<arch>.nextn_predict_layers`. The head shares the target's
// embed_tokens / lm_head and so loads neither.
Qwen3_5MTPWeights LoadQwen3_5MTPFromGguf(const GgufFile& gguf,
                                         const HfConfig& config,
                                         Qwen3_5MTPKind kind,
                                         const GgufLoadPolicy& pol);

Qwen3_5DenseWeights LoadQwen3_5DenseFromGguf(
    const GgufFile& gguf, const HfConfig& config,
    const GgufLoadPolicy* policy = nullptr);

// `SPEC-DFLASH-GGUF` B1 - the TARGET-shared bf16 embedding table + lm_head,
// read out of a GGUF target for a draft that owns neither.
//
// A DFlash draft runs the TARGET's embedding table and the TARGET's lm_head
// over its OWN hidden states; llama.cpp's DFLASH arch omits `token_embd` and
// `output` for exactly that reason, as does the z-lab safetensors checkpoint.
// When the target is a GGUF this is where those two tensors come from, and it
// is the GGUF half of `SharedHeadSource` (`model_loader.cpp`).
//
// Deliberately NOT `LoadEmbedAndHead`: that one is residency-policy driven and
// may hand back kept-F16 or block-quantized tensors shaped for the TRUNK's own
// GEMMs, whereas the draft's forward wants plain bf16 in the layout the
// safetensors path produces - `[vocab, H]` with `nk = false` for the gather
// table, the same `[vocab, H]` with `nk = true` for the MatmulBT head. What it
// DOES reuse is the tied-embedding rule (`output.weight` absent => the head IS
// `token_embd.weight`) and the sidecar-aware dequant, so an NVFP4 head cannot
// silently lose the `<stem>.scale` factor its blocks do not carry.
void LoadGgufSharedEmbedAndHeadBf16(const GgufFile& gguf, OwnedTensor* embed,
                                    OwnedTensor* head);

}  // namespace vllm
