// The paged KV-cache STORAGE dtype, resolved in ONE place.
//
// No single upstream twin: vLLM carries the KV storage dtype on the cache
// config (`CacheConfig.cache_dtype` -> `AttentionSpec.dtype`, consumed by
// `kv_cache_interface.py:380-398`). We mirror the SHAPE of that contract — the
// KV cache SPEC is the single source of truth for the storage dtype, and the
// allocator sizes buffers from `spec->page_size_bytes()` — while keeping our
// own `VT_KV_CACHE_F32` same-binary A/B as the thing that picks the value.
//
// DEFAULT: bf16 (vLLM's bf16 flash_attn KV store — halves KV memory vs f32).
// `VT_KV_CACHE_F32=1` selects f32 for the A/B. Zero bytes are +0.0f in both.
//
// Every producer of an attention KV-cache spec (the model KV-cache factories
// and the runner tests) MUST build its spec with this dtype, because the runner
// now derives BOTH the allocation size and the cache view from the spec.
#ifndef VLLM_V1_KV_CACHE_DTYPE_H_
#define VLLM_V1_KV_CACHE_DTYPE_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#include "vt/dtype.h"
#include "vt/fp8_kv.h"

namespace vllm::v1 {

inline vt::DType ResolveKvCacheDType() {
  const char* kv_f32_env = std::getenv("VT_KV_CACHE_F32");
  return (kv_f32_env != nullptr && kv_f32_env[0] == '1') ? vt::DType::kF32
                                                         : vt::DType::kBF16;
}

// ── The CacheDType surface as an ENUM (KV-NVFP4-TURBO W0, #2620) ─────────────
//
// Ported 1:1 from `vllm/v1/kv_cache_interface.py:39-101,126-128` @ `e126687a9a`,
// the ACTIVE parity pin (`.agents/upstream-sync.md`). Upstream added this map so
// that "attention backends and kernels dispatch quantization logic without
// string matching on `kv_cache_dtype`" (`:42-43`), and it is the shape every
// later KV-quant wave needs here: `is_fp8` cannot express NVFP4 or a
// per-token-head scale, so a tree that carries only that flag has to re-test the
// string at each site.
//
// UPSTREAM CARRIES TWO COPIES of the predicates derived from this map, and at
// this pin THEY DISAGREE. The enum copy is `kv_cache_interface.py:100-101`:
// `is_quantized_kv_cache` is `get_kv_quant_mode(...) != NONE`, so it answers
// TRUE for the four `turboquant_*` members. The older string copy is
// `vllm/utils/torch_utils.py:77-82` -- `startswith("fp8") or
// endswith("per_token_head") or startswith("nvfp4")` -- which answers FALSE for
// them, and it is the copy `vllm/config/cache.py:13-16` imports. The two agreed
// at the PRIOR pin `555967922`, where `get_kv_quant_mode` had no turboquant arm
// and fell those names through to NONE. They no longer do, and neither copy is
// dead. This tree therefore mirrors both, on purpose and separately: the string
// copy at `src/vllm/v1/attention/backend.cpp:106` (`IsQuantizedKvCacheName`) and
// `IsQuantizedKvCache` below, and the enum copy here. Do not "reconcile" them
// into one predicate; that would pick a side upstream has not picked.
//
// THE ENUM ITSELF IS NOT NEW. It landed with `AttentionSpec::kv_quant_mode`
// "for field fidelity" and lived in `kv_cache_interface.h`, which includes this
// header. What was missing is the FUNCTION that produces a value for it from a
// `cache_dtype` string, so every value on that field came from a default. The
// definition moves here, to the header that resolves the string, and the
// `uint8_t` underlying type is kept because the enum is a struct field whose
// width other code sizes.
enum class KVQuantMode : uint8_t {
  kNone = 0,               // NONE
  kFp8PerTensor = 1,       // FP8_PER_TENSOR -- per-tensor scales (the fp8 path)
  kInt8PerTokenHead = 2,   // INT8_PER_TOKEN_HEAD
  kFp8PerTokenHead = 3,    // FP8_PER_TOKEN_HEAD
  kInt4PerTokenHead = 4,   // INT4_PER_TOKEN_HEAD -- packed 2x int4/byte, RHT + zp
  kNvfp4 = 5,              // NVFP4 -- packed fp4 data + fp8 block scales
  kTurboquantK8v4 = 6,     // TURBOQUANT_K8V4
  kTurboquant4bitNc = 7,   // TURBOQUANT_4BIT_NC
  kTurboquantK3v4Nc = 8,   // TURBOQUANT_K3V4_NC
  kTurboquant3bitNc = 9,   // TURBOQUANT_3BIT_NC
};

// `kv_cache_interface.py:58-65` KVQuantMode.is_per_token_head.
inline bool KvQuantModeIsPerTokenHead(KVQuantMode mode) {
  return mode == KVQuantMode::kInt8PerTokenHead ||
         mode == KVQuantMode::kFp8PerTokenHead ||
         mode == KVQuantMode::kInt4PerTokenHead;
}

// `kv_cache_interface.py:67-70` KVQuantMode.is_nvfp4.
inline bool KvQuantModeIsNvfp4(KVQuantMode mode) { return mode == KVQuantMode::kNvfp4; }

// `kv_cache_interface.py:72-80` KVQuantMode.is_turboquant. It has NO member at
// the prior pin; the four modes and this predicate arrived with `e126687a9a`.
inline bool KvQuantModeIsTurboquant(KVQuantMode mode) {
  return mode == KVQuantMode::kTurboquantK8v4 ||
         mode == KVQuantMode::kTurboquant4bitNc ||
         mode == KVQuantMode::kTurboquantK3v4Nc ||
         mode == KVQuantMode::kTurboquant3bitNc;
}

// `kv_cache_interface.py:83-97` get_kv_quant_mode. THE ORDER IS LOAD-BEARING:
// the three `*_per_token_head` members are matched BEFORE the `startswith("fp8")`
// arm, so `fp8_per_token_head` resolves to kFp8PerTokenHead and not to
// kFp8PerTensor. The nvfp4 arm is `startswith("nvfp4")` and not equality, which
// is what folds `nvfp4_4over6` onto kNvfp4 (`:91-92`).
//
// Upstream's turboquant arm is `KVQuantMode[kv_cache_dtype.upper()]` (`:93-94`),
// which RAISES KeyError for a `turboquant_` name outside the enum. That branch
// is unreachable there because pydantic validates the `CacheDType` Literal
// first, and it is unreachable here for the same reason: `ParseCacheDType`
// checks `IsCacheDTypeName` before it looks at anything else. This function is
// total, so an unknown `turboquant_` name answers kNone rather than throwing out
// of a header predicate.
inline KVQuantMode GetKvQuantMode(std::string_view cache_dtype) {
  if (cache_dtype == "int4_per_token_head") return KVQuantMode::kInt4PerTokenHead;
  if (cache_dtype == "int8_per_token_head") return KVQuantMode::kInt8PerTokenHead;
  if (cache_dtype == "fp8_per_token_head") return KVQuantMode::kFp8PerTokenHead;
  if (cache_dtype.rfind("nvfp4", 0) == 0) return KVQuantMode::kNvfp4;
  if (cache_dtype == "turboquant_k8v4") return KVQuantMode::kTurboquantK8v4;
  if (cache_dtype == "turboquant_4bit_nc") return KVQuantMode::kTurboquant4bitNc;
  if (cache_dtype == "turboquant_k3v4_nc") return KVQuantMode::kTurboquantK3v4Nc;
  if (cache_dtype == "turboquant_3bit_nc") return KVQuantMode::kTurboquant3bitNc;
  if (cache_dtype.rfind("fp8", 0) == 0) return KVQuantMode::kFp8PerTensor;
  return KVQuantMode::kNone;
}

// `vllm/utils/torch_utils.py:546-548` nvfp4_kv_cache_full_dim — the packed last
// dim of an NVFP4 KV page, per head: `head_size/2` bytes of fp4 data (two E2M1
// values per byte) plus `head_size/16` bytes of E4M3 block scales, one scale per
// 16 elements. Consumed upstream by `FlashInferBackend.customize_spec`
// (`v1/attention/backends/flashinfer.py:394-407`), which rewrites the spec's
// `num_head_slots` to `2 * num_kv_heads` and its `state_content_bytes` to this
// dim, and by `nvfp4_split_data_scale` (`torch_utils.py:551`).
//
// It is 9/16 of a bf16 page's per-element width, not 1/4: the block scales are
// not free, and a memory estimate that forgets them is 12.5% short.
inline int Nvfp4KvCacheFullDim(int head_size) {
  return head_size / 2 + head_size / 16;
}

namespace detail {

// The `CacheDType` Literal in upstream's own order (`vllm/config/cache.py:
// 39-57`). ONE list: `IsCacheDTypeName` and the refusal below both read it, so a
// name cannot be legal in one place and unknown in the other.
inline constexpr const char* kCacheDTypeNames[] = {
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
    "nvfp4_4over6",
};

inline std::string CacheDTypeNameList() {
  std::string out;
  for (const char* n : kCacheDTypeNames) {
    if (!out.empty()) out += ", ";
    out += n;
  }
  return out;
}

}  // namespace detail

// Membership in the `CacheDType` Literal (`config/cache.py:39-57`). Upstream
// validates this with pydantic before any backend is asked whether it can serve
// the value, and the two are DIFFERENT facts: "that is not a KV cache dtype" and
// "that is a KV cache dtype this engine does not serve" send a reader to
// different places. They were one message here until #2620.
inline bool IsCacheDTypeName(std::string_view cache_dtype) {
  for (const char* n : detail::kCacheDTypeNames) {
    if (cache_dtype == n) return true;
  }
  return false;
}

namespace detail {

// The tail of the refusal for a real `CacheDType` member this engine does not
// serve. Each member names the ROW that owes it, because a refusal that does not
// is a dead end: the reader learns only which row does not own it.
inline std::string UnservedCacheDTypeReason(std::string_view cache_dtype) {
  if (KvQuantModeIsNvfp4(GetKvQuantMode(cache_dtype))) {
    return "is a vLLM CacheDType this engine does not serve. Row KV-NVFP4-TURBO "
           "owns it, issue #2620 tracks it, and .agents/specs/nvfp4-kv-cache.md "
           "records what is missing: the packed page (head_size/2 fp4 bytes plus "
           "head_size/16 fp8 block-scale bytes per head, "
           "vllm/utils/torch_utils.py:546-548, side-packed at 2*num_kv_heads head "
           "slots by flashinfer.py:394-407), the quantizing store "
           "(csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:170), and a paged "
           "attention read that takes the block-scale region as a SEPARATE cache "
           "view -- vt::PagedAttention takes two cache tensors and nvfp4 needs "
           "four. Serve this checkpoint with --kv-cache-dtype fp8, which is the "
           "quantized KV arm this engine does serve, or --kv-cache-dtype auto";
  }
  if (cache_dtype == "fp8_inc") {
    return "is the Intel Gaudi INC fp8 KV cache, which this engine does not "
           "serve. Row QUANT-KV-FP8-VENDOR owns it "
           "(.agents/quantization-matrix.md). Use --kv-cache-dtype fp8 for the "
           "portable per-tensor fp8 arm";
  }
  if (cache_dtype == "fp8_ds_mla") {
    return "is DeepSeek's compressed MLA latent KV cache, which this engine does "
           "not serve. Row KV-DSV4-MULTICACHE owns it: its W1 landed the 584-byte "
           "page formula and neither a store nor a read, so accepting the name "
           "here would size every MLA page for bytes nothing writes. Run the MLA "
           "model on --kv-cache-dtype auto";
  }
  if (KvQuantModeIsPerTokenHead(GetKvQuantMode(cache_dtype))) {
    return "needs per-token-head dynamic KV scales "
           "(vllm/v1/kv_cache_interface.py:58-65), which this engine does not "
           "compute -- every KV scale here is per-tensor. Row KV-NVFP4-TURBO owns "
           "it. Use --kv-cache-dtype fp8 for the per-tensor fp8 arm";
  }
  if (KvQuantModeIsTurboquant(GetKvQuantMode(cache_dtype))) {
    return "is a TurboQuant KV cache, which this engine does not serve. Row "
           "KV-NVFP4-TURBO owns it. Use --kv-cache-dtype fp8 for the per-tensor "
           "fp8 arm";
  }
  // Unreachable for the Literal as it stands: every member is either served by
  // ParseCacheDType above or named by an arm here. It is written out rather than
  // asserted so that a member added by an upstream-sync bump refuses honestly
  // instead of claiming a row that never agreed to own it.
  return "is a vLLM CacheDType this engine does not serve, and no row here "
         "claims it yet. Use --kv-cache-dtype auto";
}

}  // namespace detail

// Resolution of vLLM's `CacheConfig.cache_dtype` string (config/cache.py:39-57,
// 119-125) into what our KV cache + ops need: the STORAGE dtype the allocator
// sizes blocks from, plus the fp8 read/write interpretation. Mirrors the
// `CacheDType` Literal and the `is_quantized_kv_cache` contract.
//
// KV-FP8 W1 owns the fp8-e4m3 family here. Every other CacheDType member is
// refused by name, and since KV-NVFP4-TURBO W0 (#2620) each refusal names the
// ROW that owes it rather than the row that does not: nvfp4, nvfp4_4over6, the
// four turboquant_* members and the three *_per_token_head members are
// KV-NVFP4-TURBO's, fp8_inc is QUANT-KV-FP8-VENDOR's, and fp8_ds_mla is
// KV-DSV4-MULTICACHE's.
struct ResolvedCacheDType {
  bool is_fp8 = false;  // quantized fp8 KV (cache pages are 1-byte fp8 / kI8)
  vt::DType storage = vt::DType::kBF16;  // block-allocation dtype
  vt::Fp8KVCacheDataType fp8_kind = vt::Fp8KVCacheDataType::kAuto;
  // `KVQuantMode` for the resolved string (kv_cache_interface.py:83-97). It
  // carries what `is_fp8` cannot: a later wave dispatches on the enum instead of
  // testing the string again at each site. Only the served arms reach a caller,
  // because every other member is refused below.
  KVQuantMode quant_mode = KVQuantMode::kNone;
};

// The ENUM copy of is_quantized_kv_cache (`kv_cache_interface.py:100-101`):
// anything the mode map does not answer NONE for. Over the whole `CacheDType`
// Literal this is exactly "not auto/float16/bfloat16", which is how it is
// written here. It answers TRUE for the four turboquant_* members, and the
// string copy mirrored at `src/vllm/v1/attention/backend.cpp:106` answers FALSE
// for them; see the note above the enum -- upstream carries both and they
// disagree at this pin.
inline bool IsQuantizedKvCache(std::string_view cache_dtype) {
  return cache_dtype != "auto" && cache_dtype != "float16" && cache_dtype != "bfloat16";
}

// Parse a CacheDType string. `model_dtype` is the resolved model storage dtype
// used for the "auto" path (config/cache.py:76 "If auto, use model data type").
inline ResolvedCacheDType ParseCacheDType(std::string_view cache_dtype, vt::DType model_dtype) {
  // THE LITERAL GATE RUNS FIRST, as upstream's does: pydantic validates
  // `CacheDType` (config/cache.py:39-57) before any code reads the string, so
  // `get_kv_quant_mode` never sees a non-member and its KeyError arm (`:93-94`)
  // is unreachable. Keeping that order here keeps ours unreachable too.
  const std::string requested(cache_dtype);
  VT_CHECK(IsCacheDTypeName(cache_dtype),
           "cache_dtype '" + requested +
               "' is not a vLLM CacheDType (config/cache.py:39-57). The members "
               "are: " +
               detail::CacheDTypeNameList());
  ResolvedCacheDType r;
  r.quant_mode = GetKvQuantMode(cache_dtype);
  if (cache_dtype == "auto") {
    r.storage = model_dtype;
    r.fp8_kind = vt::Fp8KVCacheDataType::kAuto;
    return r;
  }
  if (cache_dtype == "float16") {
    r.storage = vt::DType::kF16;
    return r;
  }
  if (cache_dtype == "bfloat16") {
    r.storage = vt::DType::kBF16;
    return r;
  }
  // fp8 == fp8_e4m3 (config/cache.py:76 "CUDA 11.8+ supports fp8 (=fp8_e4m3)").
  if (cache_dtype == "fp8" || cache_dtype == "fp8_e4m3") {
    r.is_fp8 = true;
    r.storage = vt::DType::kI8;  // 1-byte fp8 storage
    r.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
    return r;
  }
  if (cache_dtype == "fp8_e5m2") {
    r.is_fp8 = true;
    r.storage = vt::DType::kI8;
    r.fp8_kind = vt::Fp8KVCacheDataType::kFp8E5M2;
    return r;
  }
  // TWO REFUSALS, because they are two different facts and one message made a
  // reader guess which one they had hit. The first fired above: a name outside
  // the Literal is not a KV cache dtype at all. This is the second: a member
  // inside it is a real vLLM CacheDType that THIS engine does not serve, and
  // each such member is owed by a different row.
  VT_CHECK(false, "cache_dtype '" + requested + "' " +
                      detail::UnservedCacheDTypeReason(cache_dtype));
  return r;
}

}  // namespace vllm::v1

#endif  // VLLM_V1_KV_CACHE_DTYPE_H_
