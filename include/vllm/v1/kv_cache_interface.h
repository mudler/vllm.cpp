// Ported from: vllm/v1/kv_cache_interface.py @ e24d1b24
//
// Scope (M1.3 Task 1): the KV-cache SPEC hierarchy + config wrappers the
// KVCacheManager (M1.3 Task 2-4) and the scheduler (M1.4) build on. The gate
// models interleave GDN (mamba-state) layers and full-attention layers, so the
// two T0 specs ported here are `FullAttentionSpec` (paged K+V blocks) and
// `MambaSpec` (the SSM+conv recurrent state). Behavioral only: no CUDA, no
// model — a spec is pure allocation metadata (block_size + page_size_bytes).
//
// HIERARCHY SHAPE: upstream is a frozen-dataclass hierarchy
//   KVCacheSpec (base) -> AttentionSpec -> FullAttentionSpec
//                                      -> SlidingWindowSpec
//                                      -> ChunkedLocalAttentionSpec
//                      -> MambaSpec
// mirrored here with a virtual base class + derived structs. `page_size_bytes`
// is the abstract per-page byte cost; `AttentionSpec` splits it into
// `real_page_size_bytes` (the raw K+V bytes) + a `page_size_padded` override,
// exactly as upstream, so FullAttentionSpec can override only the raw formula.
//
// PAGE-SIZE FORMULAS (kv_quant_mode == NONE, the only mode the gate models use):
//   AttentionSpec.real_page_size_bytes =
//       2 * block_size * num_kv_heads * head_size * dtype_size          (K + V)
//   FullAttentionSpec.real_page_size_bytes =
//       block_size * num_kv_heads * (head_size + head_size_v) * dtype_size
//       (identical to the base when head_size_v == head_size, the default; the
//        split exists so MLA/asymmetric-V layers can differ)
//   MLAAttentionSpec.real_page_size_bytes =
//       storage_block_size * num_kv_heads(1) * head_size * dtype_size
//       (NO factor 2, NO separate V — see the MLAAttentionSpec comment)
//   SlidingWindowSpec.real_page_size_bytes = the same asymmetric K+V formula;
//       its window changes allocation lifetime, not bytes per stored token.
//   ChunkedLocalAttentionSpec inherits AttentionSpec's symmetric K+V formula;
//       its fixed chunk changes allocation lifetime, not bytes per token.
//   MambaSpec.page_size_bytes =
//       sum_i( prod(shapes[i]) * dtype_size(dtypes[i]) )    (SSM + conv state)
//   page_size_bytes = page_size_padded if set (>= real), else real.
//
// FIELD/METHOD NAMES are kept EXACTLY as upstream (snake_case: block_size,
// num_kv_heads, head_size, head_size_v, page_size_bytes, real_page_size_bytes,
// page_size_padded, num_speculative_blocks, mamba_cache_mode, layer_names,
// kv_cache_spec, num_blocks, kv_cache_tensors, kv_cache_groups, shared_by,
// has_mamba_layers, needs_kv_cache_zeroing) — this overrides the repo's usual
// CamelCase convention per the plan's 1:1 mandate.
//
// DEFERRED (marked stubs / omissions; the gate models never exercise these, and
// later units fill them in without reshaping the base):
//   - SlidingWindowMLASpec / SinkFullAttentionSpec / RSWASpec /
//     EncoderOnlyAttentionSpec / CrossAttentionSpec / UniformTypeKVCacheSpecs /
//     TQFullAttentionSpec / HiddenStateCacheSpec (T1/T2) — omitted. The base
//     stays extensible (virtual page_size_bytes/kind/storage_block_size).
//     `MLAAttentionSpec` LANDED (MLA campaign W1): allocation metadata only —
//     no MLA math, no MLA ops, no model consumes it yet.
//   - kv_quant_mode != NONE page-size math (per-token-head scale bytes, nvfp4 /
//     int4 packed layouts): the `kv_quant_mode` field is carried for fidelity
//     but any non-NONE mode throws in real_page_size_bytes (T1). The gate models
//     run unquantized KV cache.
//   - max_memory_usage_bytes(VllmConfig): needs the (not-yet-ported) VllmConfig
//     + parallel/cache configs. Omitted with this note (T1); no Task-1 test uses
//     it. page_size_bytes is the piece the KVCacheManager needs now.
//   - merge() / is_uniform_with_collection() / copy_with_new_block_size(): the
//     spec-grouping helpers (used when building KVCacheConfig from raw specs).
//     Omitted (T1); Task 1 constructs groups/configs directly, as upstream's
//     own tests do.
//   - MambaSpec.mamba_type (MambaAttentionBackendEnum): omitted — it selects a
//     CUDA attention backend, irrelevant to page-size accounting; porting the
//     backend enum is out of Task-1 scope.
#ifndef VLLM_V1_KV_CACHE_INTERFACE_H_
#define VLLM_V1_KV_CACHE_INTERFACE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm::v1 {

// Upstream KVCacheSpecKind (str Enum). Ported as a plain enum; the T1/T2 kinds
// are listed for fidelity; kFullAttention / kSlidingWindow / kMamba are
// currently produced here.
enum class KVCacheSpecKind {
  kFullAttention,
  kMlaAttention,
  kSlidingWindow,
  kSlidingWindowMla,
  kMamba,
  kChunkedLocalAttention,
  kSinkFullAttention,
  kEncoderOnlyAttention,
  kCrossAttention,
  kUnknown,
};

// Upstream KVQuantMode (IntEnum). Carried on AttentionSpec for field fidelity;
// only kNone participates in the ported page-size math (see header note).
enum class KVQuantMode : uint8_t {
  kNone = 0,
  kFp8PerTensor = 1,
  kInt8PerTokenHead = 2,
  kFp8PerTokenHead = 3,
  kInt4PerTokenHead = 4,
  kNvfp4 = 5,
};

// A base class for specifying the KV cache format of one layer.
// (Upstream: @dataclass(frozen=True) KVCacheSpec.)
struct KVCacheSpec {
  explicit KVCacheSpec(int block_size) : block_size(block_size) {}
  virtual ~KVCacheSpec() = default;

  // Number of tokens in a block.
  int block_size;

  // The size of a page with `block_size` tokens in bytes.
  virtual int64_t page_size_bytes() const = 0;

  // Upstream property `storage_block_size` (defaults to block_size).
  virtual int storage_block_size() const { return block_size; }

  // The spec kind (upstream: get_kv_cache_spec_kind(spec)).
  virtual KVCacheSpecKind kind() const = 0;
};

// (Upstream: @dataclass(frozen=True, kw_only=True) AttentionSpec.)
// Not instantiated directly (kind() stays pure); FullAttentionSpec is the T0
// concrete attention spec.
struct AttentionSpec : KVCacheSpec {
  AttentionSpec(int block_size, int num_kv_heads, int head_size, vt::DType dtype,
                KVQuantMode kv_quant_mode = KVQuantMode::kNone,
                std::optional<int64_t> page_size_padded = std::nullopt,
                bool indexes_kv_by_block_stride = false)
      : KVCacheSpec(block_size),
        num_kv_heads(num_kv_heads),
        head_size(head_size),
        dtype(dtype),
        kv_quant_mode(kv_quant_mode),
        page_size_padded(page_size_padded),
        indexes_kv_by_block_stride(indexes_kv_by_block_stride) {}

  int num_kv_heads;
  int head_size;
  vt::DType dtype;
  KVQuantMode kv_quant_mode;
  std::optional<int64_t> page_size_padded;
  bool indexes_kv_by_block_stride;

  int64_t page_size_bytes() const override;

  // The raw (unpadded) K+V bytes per page. Overridden by FullAttentionSpec.
  virtual int64_t real_page_size_bytes() const;
};

// When the hybrid allocator is disabled and the model mixes full + sliding
// window attention, sliding window is treated as full attention here (blocks
// allocated for all tokens) — hence sliding_window / attention_chunk_size are
// recorded on the full-attention spec. (Upstream FullAttentionSpec.)
struct FullAttentionSpec : AttentionSpec {
  // head_size_v defaults to head_size (upstream __post_init__). Pass
  // std::nullopt (the default) to inherit head_size.
  FullAttentionSpec(int block_size, int num_kv_heads, int head_size,
                    vt::DType dtype,
                    std::optional<int> head_size_v = std::nullopt,
                    KVQuantMode kv_quant_mode = KVQuantMode::kNone,
                    std::optional<int64_t> page_size_padded = std::nullopt,
                    bool indexes_kv_by_block_stride = false,
                    std::optional<int> sliding_window = std::nullopt,
                    std::optional<int> attention_chunk_size = std::nullopt,
                    bool non_causal = false)
      : AttentionSpec(block_size, num_kv_heads, head_size, dtype, kv_quant_mode,
                      page_size_padded, indexes_kv_by_block_stride),
        head_size_v(head_size_v.value_or(head_size)),
        sliding_window(sliding_window),
        attention_chunk_size(attention_chunk_size),
        non_causal(non_causal) {}

  int head_size_v;
  std::optional<int> sliding_window;
  std::optional<int> attention_chunk_size;
  bool non_causal;

  int64_t real_page_size_bytes() const override;
  KVCacheSpecKind kind() const override {
    return KVCacheSpecKind::kFullAttention;
  }
};

// The compressed-latent (Multi-head Latent Attention) paged cache.
// (Upstream: vllm/v1/kv_cache_interface.py:363 MLAAttentionSpec(FullAttentionSpec),
// page formula :380-398.)
//
// MLA stores ONE latent row per token per layer — `kv_lora_rank +
// qk_rope_head_dim` elements (512 + 64 = 576 for every DeepSeek variant and for
// Kimi Linear's MLA layers) — with `num_kv_heads == 1` and **no V tensor at
// all**: V is reconstructed on the fly from the same latent via `W_UV`. So the
// page formula DROPS the factor 2 that every other attention spec carries:
//
//   real_page_size_bytes = storage_block_size * num_kv_heads * head_size * es
//
// mirroring upstream `kv_cache_interface.py:397-398`. The out-of-scope special
// cases upstream guards above that line (`fp8_ds_mla`: V3.2 = 656 B/token,
// V4 = 584 B/token at :381-388; INT4 per-token-head at :389-390) are NOT ported
// and throw via the shared kv_quant_mode guard rather than silently mis-sizing.
//
// MLA-ness is a page-SIZE and tensor-SHAPE concern ONLY: upstream maps
// `MLAAttentionSpec -> FullAttentionManager` with
// `uniform_type_base_spec=FullAttentionSpec`
// (`vllm/v1/core/single_type_kv_cache_manager.py:1539`), and
// `vllm/v1/worker/gpu_model_runner.py` has **no `use_mla` branch at all** — so
// the block table, prefix caching and eviction are untouched. We mirror that by
// deriving from FullAttentionSpec and registering the same manager kind
// (see kv_cache_spec_registry.cpp).
//
// Upstream also asserts MLA and full-attention layers can NEVER share a KV
// group (`kv_cache_interface.py:277-279` and `:400-403`, the two `merge`
// asserts). We have not ported `merge()` (see the DEFERRED note above); the
// distinct `kind()` below is what a future `merge()` will key that assert on.
struct MLAAttentionSpec : FullAttentionSpec {
  // num_kv_heads is accepted for field fidelity but is 1 for MLA — upstream
  // states it in three places (`mla_attention.py:390`, `:1004-1009`,
  // `vllm/config/model.py:1270-1274` "When using MLA during decode it becomes
  // MQA") and `MLACommonBackend.get_kv_cache_shape` accepts and IGNORES the
  // argument (`mla_attention.py:1219`).
  MLAAttentionSpec(int block_size, int head_size, vt::DType dtype,
                   int num_kv_heads = 1,
                   KVQuantMode kv_quant_mode = KVQuantMode::kNone,
                   std::optional<int64_t> page_size_padded = std::nullopt,
                   bool indexes_kv_by_block_stride = false)
      : FullAttentionSpec(block_size, num_kv_heads, head_size, dtype,
                          /*head_size_v=*/head_size, kv_quant_mode,
                          page_size_padded, indexes_kv_by_block_stride) {}

  int64_t real_page_size_bytes() const override;
  KVCacheSpecKind kind() const override {
    return KVCacheSpecKind::kMlaAttention;
  }
};

// Sliding-window paged K+V cache. The compute path still applies the local
// attention mask; this spec controls the recycling-aware cache allocation and
// prefix policy. (Upstream SlidingWindowSpec.)
struct SlidingWindowSpec : AttentionSpec {
  // head_size_v defaults to head_size (upstream __post_init__).
  SlidingWindowSpec(int block_size, int num_kv_heads, int head_size,
                    vt::DType dtype, int sliding_window,
                    std::optional<int> head_size_v = std::nullopt,
                    KVQuantMode kv_quant_mode = KVQuantMode::kNone,
                    std::optional<int64_t> page_size_padded = std::nullopt,
                    bool indexes_kv_by_block_stride = false)
      : AttentionSpec(block_size, num_kv_heads, head_size, dtype, kv_quant_mode,
                      page_size_padded, indexes_kv_by_block_stride),
        sliding_window(sliding_window),
        head_size_v(head_size_v.value_or(head_size)) {}

  int sliding_window;
  int head_size_v;

  int64_t real_page_size_bytes() const override;

  // Per-request startup-admission bound used by both pool sizing and the
  // runtime full-sequence fit check. The +1 covers a window that starts in the
  // middle of a block.
  int max_admission_blocks_per_request(int max_num_batched_tokens,
                                       int max_model_len) const;

  KVCacheSpecKind kind() const override {
    return KVCacheSpecKind::kSlidingWindow;
  }
};

// Fixed-chunk local-attention paged K+V cache. Tokens before the current
// attention chunk are represented by null logical blocks and their physical
// pages are recycled. (Upstream ChunkedLocalAttentionSpec.)
struct ChunkedLocalAttentionSpec : AttentionSpec {
  ChunkedLocalAttentionSpec(
      int block_size, int num_kv_heads, int head_size, vt::DType dtype,
      int attention_chunk_size,
      KVQuantMode kv_quant_mode = KVQuantMode::kNone,
      std::optional<int64_t> page_size_padded = std::nullopt,
      bool indexes_kv_by_block_stride = false)
      : AttentionSpec(block_size, num_kv_heads, head_size, dtype, kv_quant_mode,
                      page_size_padded, indexes_kv_by_block_stride),
        attention_chunk_size(attention_chunk_size) {}

  int attention_chunk_size;

  // During chunked prefill, physical KV holds at most the current fixed chunk
  // plus the newly scheduled batch, clamped by max_model_len.
  int max_admission_blocks_per_request(int max_num_batched_tokens,
                                       int max_model_len) const;

  KVCacheSpecKind kind() const override {
    return KVCacheSpecKind::kChunkedLocalAttention;
  }
};

// The mamba/GDN recurrent-state spec. `shapes`/`dtypes` describe each state
// tensor in upstream order (e.g. conv state, then temporal/SSM state);
// page_size_bytes is the sum of their byte sizes — NOT the K+V paged formula.
// (Upstream MambaSpec.)
struct MambaSpec : KVCacheSpec {
  MambaSpec(int block_size, std::vector<std::vector<int64_t>> shapes,
            std::vector<vt::DType> dtypes,
            std::optional<int64_t> page_size_padded = std::nullopt,
            std::string mamba_cache_mode = "none",
            int num_speculative_blocks = 0)
      : KVCacheSpec(block_size),
        shapes(std::move(shapes)),
        dtypes(std::move(dtypes)),
        page_size_padded(page_size_padded),
        mamba_cache_mode(std::move(mamba_cache_mode)),
        num_speculative_blocks(num_speculative_blocks) {}

  std::vector<std::vector<int64_t>> shapes;
  std::vector<vt::DType> dtypes;
  std::optional<int64_t> page_size_padded;
  std::string mamba_cache_mode;
  int num_speculative_blocks;

  int64_t page_size_bytes() const override;
  KVCacheSpecKind kind() const override { return KVCacheSpecKind::kMamba; }
};

// How the workers should initialize a KV cache tensor. (Upstream KVCacheTensor.)
struct KVCacheTensor {
  int64_t size;                      // size of the KV cache tensor in bytes
  std::vector<std::string> shared_by;  // layer names sharing this tensor
  int64_t offset = 0;                // byte offset within a contiguous block
  int64_t block_stride = 0;          // bytes per block in a packed layout (0 = unpacked)
};

// A group of model layers that share the same KV cache block table; treated as
// one layer by the KV cache manager. (Upstream KVCacheGroupSpec.)
struct KVCacheGroupSpec {
  KVCacheGroupSpec(std::vector<std::string> layer_names,
                   std::shared_ptr<KVCacheSpec> kv_cache_spec,
                   bool is_eagle_group = false)
      : layer_names(std::move(layer_names)),
        kv_cache_spec(std::move(kv_cache_spec)),
        is_eagle_group(is_eagle_group) {}

  std::vector<std::string> layer_names;
  std::shared_ptr<KVCacheSpec> kv_cache_spec;
  bool is_eagle_group = false;
};

// The KV cache configuration of a model. (Upstream KVCacheConfig.)
struct KVCacheConfig {
  int num_blocks;
  std::vector<KVCacheTensor> kv_cache_tensors;
  std::vector<KVCacheGroupSpec> kv_cache_groups;

  // Upstream property has_mamba_layers.
  bool has_mamba_layers() const;
  // Upstream property needs_kv_cache_zeroing.
  bool needs_kv_cache_zeroing() const { return has_mamba_layers(); }
};

}  // namespace vllm::v1

#endif  // VLLM_V1_KV_CACHE_INTERFACE_H_
