// Ported from: vllm/model_executor/layers/quantization/kv_cache.py @ 555967922
//
// `BaseKVCacheMethod`'s k/v-scale resolution, which is the half of the fp8-KV
// feature that is NOT arithmetic: deciding which scale the store and the read
// multiply by, and — the part this file exists for — deciding whether there was
// an fp8 KV cache to scale at all.
//
// THE TWO STATES A FALLTHROUGH CANNOT TELL APART. `r0b0tlab/Qwen3.8-27B-NVFP4-
// MTP-sm121` @ `36f717a2` declares `kv_cache_quant_algo: "FP8"` in
// `hf_quant_config.json` and ships ZERO `k_scale`/`v_scale` tensors — 2001
// entries in `model.safetensors.index.json`, none of them a KV scale (measured
// 2026-08-21 from the public index). It therefore serves on the DEFAULT scale
// 1.0, and so would a checkpoint that declared no KV quantization at all if the
// default were reached by falling off the end of a missing-tensor lookup. The
// two produce identical output today and stop being the same the moment a
// checkpoint declares nothing: one is a documented default, the other is an
// invented scale for a cache nobody asked to quantize.
//
// Upstream keeps them apart STRUCTURALLY rather than by a sentinel, and this
// file mirrors that. `process_weights_after_loading` reaches the scale block at
// all only under `is_quantized_kv_cache(layer.kv_cache_dtype)`
// (`kv_cache.py:100-102`); INSIDE that block, both scales still holding the
// `KVCacheScaleParameter` sentinel `-1.0` (`kv_cache.py:18-30`) is the separate
// "no scales were loaded" arm that takes 1.0 and warns (`:112-116`, `:150-156`).
// So `kNotQuantized` is not a value on the same axis as `kDeclaredButAbsent`:
// it is the state in which asking for a scale is a question that was never
// posed. `ScalesForFp8Store` refuses it by name rather than answering 1.0.
#ifndef VLLM_MODEL_EXECUTOR_LAYERS_QUANTIZATION_KV_CACHE_H_
#define VLLM_MODEL_EXECUTOR_LAYERS_QUANTIZATION_KV_CACHE_H_

#include <algorithm>
#include <string>
#include <string_view>

#include "vllm/v1/kv_cache_dtype.h"
#include "vt/dtype.h"

namespace vllm {

// kv_cache.py:18-30 `KVCacheScaleParameter.__new__` — the scalar parameter is
// initialized to -1.0, an INVALID sentinel, so "absent" is a value the resolver
// can read rather than an absence it has to infer. Mirror the constant, because
// every branch below keys on `< 0.0` exactly as upstream's does.
inline constexpr float kKvScaleUnloaded = -1.0F;

// Which of `process_weights_after_loading`'s arms produced the pair. The point
// of naming all four is that `kNotQuantized` and `kDeclaredButAbsent` carry the
// same NUMBERS and are different FACTS.
enum class KvScaleOrigin {
  // is_quantized_kv_cache(kv_cache_dtype) == false (kv_cache.py:100-102): the
  // scale block never ran. There is no fp8 cache, so there is no scale — not a
  // scale of 1.0.
  kNotQuantized,
  // kv_cache.py:104-111 — both `k_scale` and `v_scale` came from the checkpoint.
  kCheckpoint,
  // kv_cache.py:117-127 — the checkpoint carried a single `kv_scale`, remapped
  // to `k_scale` at load and duplicated to `v_scale` here.
  kCheckpointKvScale,
  // kv_cache.py:112-116 — an fp8 KV cache WAS declared and both scales are the
  // unloaded sentinel, so the documented default 1.0 applies and the
  // uncalibrated warning fires (`:150-156`). This is the arm the
  // `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` gate checkpoint takes WHEN
  // `--kv-cache-dtype fp8` is typed. It does not reach it on its own:
  // MEASURED 2026-08-22 from the live artifact @ `36f717a2`, that repository's
  // `config.json:quantization_config` carries `quant_method: "modelopt"` and
  // `quant_algo: "MIXED_PRECISION"` and NO `kv_cache_*` key at all, and only the
  // legacy `hf_quant_config.json` declares `kv_cache_quant_algo: "FP8"`. The
  // inline document wins (`transformers_utils/config.py:751-761`), on both
  // engines, so neither vLLM nor this port auto-selects an fp8 KV cache for it.
  kDeclaredButAbsent,
};

struct ResolvedKvCacheScales {
  float k_scale = 1.0F;
  float v_scale = 1.0F;
  KvScaleOrigin origin = KvScaleOrigin::kNotQuantized;
  // kv_cache.py:150-156: `k_scale == 1.0 and v_scale == 1.0 and "e5m2" not in
  // kv_cache_dtype` -> warn that the cache is being quantized against an
  // uncalibrated scale. Carried as a flag rather than printed here so the caller
  // decides once per engine instead of once per layer (upstream's own
  // `warning_once`).
  bool uncalibrated = false;
};

// kv_cache.py:74-156 `BaseKVCacheMethod.process_weights_after_loading`, reduced
// to the per-tensor k/v half this port serves. `loaded_k_scale`/`loaded_v_scale`
// are `kKvScaleUnloaded` when the checkpoint carried no such tensor, exactly as
// `KVCacheScaleParameter` leaves them.
//
// `calculate_kv_scales` is upstream's DEPRECATED dynamic path (`cache.py:111`,
// removal announced for v0.19): when it is set, upstream skips this whole block
// and computes the scales from the first forward instead. We do not implement
// that, so it is refused BY NAME rather than silently taking the static arm.
inline ResolvedKvCacheScales ResolveKvCacheScales(std::string_view kv_cache_dtype,
                                                  bool calculate_kv_scales,
                                                  float loaded_k_scale,
                                                  float loaded_v_scale) {
  ResolvedKvCacheScales r;
  // kv_cache.py:100-102 — the guard, and the reason kNotQuantized is a state
  // rather than a default. It is checked BEFORE anything reads a scale, so a
  // checkpoint that declares no KV quantization cannot reach a default value.
  if (!v1::IsQuantizedKvCache(kv_cache_dtype)) {
    r.origin = KvScaleOrigin::kNotQuantized;
    return r;
  }
  VT_CHECK(!calculate_kv_scales,
           "kv_cache scales: --calculate-kv-scales (the on-the-fly dynamic k/v "
           "scale, deprecated upstream at config/cache.py:111) is not "
           "implemented; the checkpoint scale path is (KV-FP8 W3)");
  if (loaded_k_scale > 0.0F && loaded_v_scale > 0.0F) {
    // kv_cache.py:104-111 — prefer separate k_scale and v_scale when present.
    r.k_scale = loaded_k_scale;
    r.v_scale = loaded_v_scale;
    r.origin = KvScaleOrigin::kCheckpoint;
  } else if (loaded_k_scale < 0.0F && loaded_v_scale < 0.0F) {
    // kv_cache.py:112-116 — BOTH sentinels: no scale was loaded, so the
    // documented default 1.0 applies. Reached only because the branch above
    // proved an fp8 KV cache was declared.
    r.k_scale = 1.0F;
    r.v_scale = 1.0F;
    r.origin = KvScaleOrigin::kDeclaredButAbsent;
  } else {
    // kv_cache.py:117-127 — a single `kv_scale`, remapped to k_scale at load
    // and duplicated here. Upstream asserts `layer.k_scale > 0.0` and takes
    // `max(k_scale, v_scale)`.
    VT_CHECK(loaded_k_scale > 0.0F,
             "kv_cache scales: a single checkpoint kv_scale must land on "
             "k_scale (kv_cache.py:120)");
    const float dup = std::max(loaded_k_scale, loaded_v_scale);
    r.k_scale = dup;
    r.v_scale = dup;
    r.origin = KvScaleOrigin::kCheckpointKvScale;
  }
  // kv_cache.py:150-156. e5m2's dynamic range makes 1.0 the ordinary choice, so
  // upstream suppresses the warning there; e5m2 compute is refused elsewhere in
  // this port, and the condition is mirrored anyway so the two agree.
  r.uncalibrated = r.k_scale == 1.0F && r.v_scale == 1.0F &&
                   kv_cache_dtype.find("e5m2") == std::string_view::npos;
  return r;
}

// The CONSUMER guard. `kNotQuantized` has no scale to give, and a caller that
// asks for one has decided to quantize a cache the configuration never declared
// — the exact failure the origin enum exists to make impossible. Refuse rather
// than return the 1.0 that would make the mistake invisible.
inline void ScalesForFp8Store(const ResolvedKvCacheScales& scales, float* k_out,
                              float* v_out) {
  VT_CHECK(scales.origin != KvScaleOrigin::kNotQuantized,
           "kv_cache scales: no fp8 KV cache was declared (cache_dtype is not "
           "an fp8 dtype and the checkpoint declares no kv_cache_quant_algo), "
           "so there is no k/v scale to apply; a default 1.0 here would "
           "quantize a cache nobody asked to quantize");
  VT_CHECK(scales.k_scale > 0.0F && scales.v_scale > 0.0F,
           "kv_cache scales: k_scale/v_scale must be > 0");
  *k_out = scales.k_scale;
  *v_out = scales.v_scale;
}

// The one line upstream prints when the scales defaulted (kv_cache.py:150-156),
// as a string so the caller can log it once per engine.
inline std::string UncalibratedKvScaleWarning(std::string_view kv_cache_dtype) {
  return std::string("vllm.cpp: WARNING using KV cache scaling factor 1.0 for ") +
         std::string(kv_cache_dtype) +
         ". If this is unintended, verify that k/v_scale scaling factors are "
         "properly set in the checkpoint.\n";
}

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_LAYERS_QUANTIZATION_KV_CACHE_H_
