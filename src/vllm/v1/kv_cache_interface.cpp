// Ported from: vllm/v1/kv_cache_interface.py @ e24d1b24
//
// The page_size_bytes / real_page_size_bytes math for FullAttentionSpec,
// SlidingWindowSpec and MambaSpec. See kv_cache_interface.h for the formulas
// and deferred-spec notes.
#include "vllm/v1/kv_cache_interface.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace vllm::v1 {

namespace {

// Upstream KVQuantMode.is_per_token_head / is_nvfp4 (the modes whose page-size
// math is deferred here — see header).
bool needs_deferred_quant_math(KVQuantMode mode) {
  return mode != KVQuantMode::kNone;
}

}  // namespace

int64_t AttentionSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    throw std::runtime_error(
        "AttentionSpec: kv_quant_mode != NONE page-size math is deferred (T1)");
  }
  // K + V: 2 * block_size * num_kv_heads * head_size * dtype_size.
  return 2LL * block_size * num_kv_heads * head_size *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

int64_t AttentionSpec::page_size_bytes() const {
  const int64_t real_page_size = real_page_size_bytes();
  // (Per-token-head scale bytes for quantized KV cache are deferred; the NONE
  // path adds nothing here.)
  if (page_size_padded.has_value()) {
    if (*page_size_padded < real_page_size) {
      throw std::runtime_error(
          "AttentionSpec: page_size_padded must be >= real_page_size_bytes");
    }
    return *page_size_padded;
  }
  return real_page_size;
}

int64_t FullAttentionSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    throw std::runtime_error(
        "FullAttentionSpec: kv_quant_mode != NONE page-size math is deferred "
        "(T1)");
  }
  const int64_t last_dim =
      static_cast<int64_t>(head_size) + static_cast<int64_t>(head_size_v);
  return static_cast<int64_t>(block_size) * num_kv_heads * last_dim *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

// Upstream kv_cache_interface.py:397-398 — the MLA page formula. ONE latent row
// per token (kv_lora_rank + qk_rope_head_dim wide), num_kv_heads == 1, and NO
// separate V: the factor 2 every other attention spec carries is absent.
int64_t MLAAttentionSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    // Upstream's fp8_ds_mla (V3.2 656 B/token, V4 584 B/token) and INT4
    // per-token-head layouts (kv_cache_interface.py:381-390) are OUT OF SCOPE
    // for this campaign; throw loudly rather than silently mis-size.
    throw std::runtime_error(
        "MLAAttentionSpec: kv_quant_mode != NONE page-size math (fp8_ds_mla / "
        "int4) is out of scope");
  }
  return static_cast<int64_t>(storage_block_size()) * num_kv_heads * head_size *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

int64_t SlidingWindowSpec::real_page_size_bytes() const {
  if (needs_deferred_quant_math(kv_quant_mode)) {
    throw std::runtime_error(
        "SlidingWindowSpec: kv_quant_mode != NONE page-size math is deferred "
        "(T1)");
  }
  const int64_t last_dim =
      static_cast<int64_t>(head_size) + static_cast<int64_t>(head_size_v);
  return static_cast<int64_t>(block_size) * num_kv_heads * last_dim *
         static_cast<int64_t>(vt::SizeOf(dtype));
}

int SlidingWindowSpec::max_admission_blocks_per_request(
    int max_num_batched_tokens, int max_model_len) const {
  const int num_tokens = std::min(
      sliding_window - 1 + max_num_batched_tokens, max_model_len);
  return (num_tokens + block_size - 1) / block_size + 1;
}

int ChunkedLocalAttentionSpec::max_admission_blocks_per_request(
    int max_num_batched_tokens, int max_model_len) const {
  const int num_tokens = std::min(
      attention_chunk_size + max_num_batched_tokens, max_model_len);
  return (num_tokens + block_size - 1) / block_size;
}

int64_t MambaSpec::page_size_bytes() const {
  if (shapes.size() != dtypes.size()) {
    throw std::runtime_error(
        "MambaSpec: shapes and dtypes must have the same length");
  }
  int64_t page_size = 0;
  for (size_t i = 0; i < shapes.size(); ++i) {
    int64_t numel = 1;
    for (int64_t dim : shapes[i]) {
      numel *= dim;
    }
    page_size += numel * static_cast<int64_t>(vt::SizeOf(dtypes[i]));
  }
  if (page_size_padded.has_value()) {
    if (*page_size_padded < page_size) {
      throw std::runtime_error(
          "MambaSpec: page_size_padded must be >= computed page_size");
    }
    return *page_size_padded;
  }
  return page_size;
}

bool KVCacheConfig::has_mamba_layers() const {
  for (const auto& group : kv_cache_groups) {
    if (dynamic_cast<const MambaSpec*>(group.kv_cache_spec.get()) != nullptr) {
      return true;
    }
  }
  return false;
}

int64_t KVBytesPerBlock(const KVCacheConfig& config) {
  // Heterogeneous-KV path (Gemma-4 G1b): one attention spec per non-GDN layer,
  // GDN/linear layers left null. Mirror the runner's per-layer allocation.
  if (!config.per_layer_attn_specs.empty()) {
    int64_t bytes = 0;
    for (const auto& spec : config.per_layer_attn_specs) {
      if (spec != nullptr) {
        bytes += spec->page_size_bytes();
      }
    }
    return bytes;
  }
  // Uniform path: sum each attention group's page over the layers it covers.
  // Mamba/GDN groups do not scale with the block count in our runner, so they
  // are skipped (dynamic_cast to AttentionSpec fails for MambaSpec).
  int64_t bytes = 0;
  for (const auto& group : config.kv_cache_groups) {
    const auto* attn =
        dynamic_cast<const AttentionSpec*>(group.kv_cache_spec.get());
    if (attn == nullptr) {
      continue;
    }
    bytes += attn->page_size_bytes() *
             static_cast<int64_t>(group.layer_names.size());
  }
  return bytes;
}

namespace {

// The ONE arithmetic statement W3 makes about block sizing, written where it can
// be read beside the specs it rewrites: an fp8 KV element is 1 byte where bf16
// is 2 (`torch_utils.py:38-40` maps every fp8 CacheDType to `torch.uint8`), and
// `AttentionSpec::real_page_size_bytes` is linear in `vt::SizeOf(dtype)`, so the
// page halves. If this and the store ever disagree about the element size the
// result is wrong tokens rather than a crash, so the storage dtype is asserted
// here rather than assumed.
void RetypeAttentionSpec(AttentionSpec& spec, const ResolvedCacheDType& resolved,
                         float k_scale, float v_scale) {
  VT_CHECK(dynamic_cast<const MLAAttentionSpec*>(&spec) == nullptr,
           "cache_dtype: an MLA KV cache has its own quantized page formula "
           "upstream (fp8_ds_mla, kv_cache_interface.py:398-410) and no "
           "cache_dtype override is wired for it; run the MLA model on "
           "--kv-cache-dtype auto");
  if (resolved.is_fp8) {
    VT_CHECK(resolved.storage == vt::DType::kI8,
             "cache_dtype: an fp8 KV cache stores 1 byte per element "
             "(vt::DType::kI8); any other storage dtype would size the block "
             "for one element width and store another");
    VT_CHECK(vt::SizeOf(resolved.storage) == 1,
             "cache_dtype: the fp8 KV storage dtype must be exactly 1 byte");
    VT_CHECK(resolved.fp8_kind == vt::Fp8KVCacheDataType::kFp8E4M3,
             "cache_dtype: only fp8_e4m3 is implemented on the KV store and "
             "read (fp8_e5m2 compute is a named later brick, KV-FP8 W5)");
    VT_CHECK(k_scale > 0.0F && v_scale > 0.0F,
             "cache_dtype: an fp8 KV cache needs k_scale/v_scale > 0");
  } else {
    // Explicit float overrides. bfloat16 is the storage dtype the KV path
    // already writes; float16 parses (the CacheDType surface is mirrored in
    // full) but no model's attention block casts K/V to f16 before the store,
    // so it is refused by name instead of reaching a dtype mismatch deep in an
    // op wrapper.
    VT_CHECK(resolved.storage == vt::DType::kBF16,
             "cache_dtype: only 'auto', 'bfloat16', 'fp8' and 'fp8_e4m3' are "
             "wired to the KV store; 'float16' parses but no attention block "
             "casts K/V to f16 before the store (KV-FP8 W3, owed)");
  }
  spec.dtype = resolved.storage;
  spec.fp8_kind = resolved.fp8_kind;
  spec.k_scale = k_scale;
  spec.v_scale = v_scale;
}

}  // namespace

void ApplyCacheDType(KVCacheConfig& config, const ResolvedCacheDType& resolved,
                     float k_scale, float v_scale) {
  const auto retype = [&](AttentionSpec& spec) {
    // NOTHING TO APPLY, and this is the whole default path. "auto" resolves to
    // the model dtype, which is exactly what every KV-cache factory already
    // built the spec with (`ResolveKvCacheDType()`), so the write would set the
    // field to the value it holds. Returning first keeps the default load
    // byte-identical AND keeps the refusals below out of its way: an MLA model
    // on `--kv-cache-dtype auto`, and the `VT_KV_CACHE_F32=1` A/B cache, must
    // both keep loading, and neither is asking for anything to change.
    if (!resolved.is_fp8 && spec.dtype == resolved.storage &&
        spec.fp8_kind == vt::Fp8KVCacheDataType::kAuto) {
      return;
    }
    RetypeAttentionSpec(spec, resolved, k_scale, v_scale);
  };
  for (auto& group : config.kv_cache_groups) {
    auto* attn = dynamic_cast<AttentionSpec*>(group.kv_cache_spec.get());
    // MambaSpec (recurrent conv/SSM state) is NOT the KV cache: upstream sizes
    // it from its own `mamba_cache_dtype`/`mamba_ssm_cache_dtype` knobs
    // (`config/cache.py:131-138`) and `--kv-cache-dtype` never touches it.
    if (attn == nullptr) continue;
    retype(*attn);
  }
  // The heterogeneous-per-layer seam (Gemma-4) allocates from these instead of
  // the group spec, so a rewrite that skipped them would half-size the pool and
  // leave the layers writing full-width bytes into it.
  for (auto& spec : config.per_layer_attn_specs) {
    if (spec == nullptr) continue;
    retype(*spec);
  }
}

}  // namespace vllm::v1
