// LTX-2.5 IC-LoRA — reading an adapter and FUSING it into the DiT at load.
//
// Ported from Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca:
//
//   ltx-core loader/primitives.py:160-167   LoraPathStrengthAndSDOps
//   ltx-core loader/fuse_loras.py:99-116    aggregate_lora_products
//   ltx-core loader/fuse_loras.py:119-150   fuse_lora_weights
//   ltx-core loader/fuse_loras.py:183-204   the key shape and the products
//   ltx-core loader/sd_ops.py:135-137       LTXV_LORA_COMFY_RENAMING_MAP
//   ltx-pipelines iclora_utils.py:30-49     the reference-scale metadata
//   ltx-pipelines ic_lora.py:150-173        resolving factors across LoRAs
//
// ─── THIS IS NOT `include/vllm/lora/` ────────────────────────────────────────
//
// That subsystem is vLLM's runtime punica path: f32, slot-indexed, applied per
// token through `LinearMethodBase` (`punica.h:15-24`, `layers.h:26-28`). LTX
// does not apply an adapter at runtime. It FUSES the delta into the weights at
// load and the adapter then has no runtime existence at all
// (`fuse_loras.py:119-150`). Mirroring vLLM's LoRA here would mirror the wrong
// upstream, so this is a separate path by decision rather than by oversight.
// The row's spec §1 records the argument.
//
// ─── THE ARITHMETIC, AND THE DTYPE THAT CARRIES IT ───────────────────────────
//
//     delta = sum over adapters of  (B * strength) @ A
//     W'    = W + delta
//
// `B` is [out, rank], `A` is [rank, in], `W` is [out, in] row-major.
//
// **The accumulator is bfloat16, not f32.** All four of upstream's fuse rules
// set `aggregation_dtype=torch.bfloat16` — bf16 (`fuse_loras.py:71`), fp8-cast
// (`quantization/fp8_cast.py:239`), fp8-scaled-mm
// (`quantization/fp8_scaled_mm.py:189`) and NVFP4
// (`quantization/nvfp4/fuse.py:50`). An f32 accumulator here would be the exact
// defect AGENTS.md names: every token would still match and every golden would
// still pass while the path carried a precision upstream does not have.
// `test_ltx2_lora` mutates this deliberately, because nothing else can see it.
//
// Upstream's own docstring records that the FIRST product rounds differently
// from the rest — `matmul(B * strength, A)` against `addmm_(B, A, alpha=...)`
// (`fuse_loras.py:103-107`) — and that pattern is preserved here. Only the
// first form is reachable: this row fuses exactly ONE adapter and refuses a
// second by name (see `Ltx2ResolveLoraReferenceFactors`), so implementing the
// second form would be landing a branch nothing can select.
//
// ─── WHERE IT IS APPLIED, AND WHY ONE PLACE SERVES EVERY ARM ─────────────────
//
// `MaterializeDitTensor` (ltx2_loader.cpp) has four dtype branches, and BOTH
// quantized ones — `DequantFp8ToBf16` and `Ltx2DequantNvfp4ToBf16` — return
// `vt::DType::kBF16`. By the time any caller sees a byte, an FP8 or NVFP4
// checkpoint is already bf16. So a single hook immediately after that call
// serves F32, BF16, FP8 and NVFP4 at once.
//
// That is a DELIBERATE DIVERGENCE from upstream's per-arm rules, and it is
// forced rather than chosen. Upstream re-quantizes after adding the delta
// because it keeps FP8/NVFP4 weights resident for its quantized kernels; this
// tree keeps bf16 (ltx2_loader.h's DTYPE note), so there is nothing to
// re-quantize into, and the tree carries no FP8 or NVFP4 quantizer to do it
// with. The consequence is that our fused weight SKIPS upstream's lossy
// round trip and is slightly more precise on those two arms. It costs no extra
// bytes — the weight was already bf16 — so this is not the too-wide-dtype
// failure the rule above guards against.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vt/dtype.h"

namespace vllm {

class SafetensorsFile;

// LoraPathStrengthAndSDOps (loader/primitives.py:160-167). `sd_ops` has no
// field here because the only map upstream ever passes for LTX is
// LTXV_LORA_COMFY_RENAMING_MAP (`sd_ops.py:135-137`, used at `args.py:172`),
// whose entire content is stripping a `diffusion_model.` prefix — so it is
// applied unconditionally by `Ltx2LoraContractName` rather than carried.
struct Ltx2LoraSpec {
  std::string path;
  // DEFAULT_LORA_STRENGTH (utils/args.py, quoted into --lora's help at :605-608).
  double strength = 1.0;
};

// The reference scale factors an IC-LoRA was TRAINED with, read from the
// adapter's own `__metadata__` (iclora_utils.py:30-49). Both default to 1, which
// is upstream's default for an absent key.
//
// These are what `Ltx2ConditionVideoByReference` needs and what the reference
// refusal in `ltx2_video.cpp` named as unreadable.
struct Ltx2LoraReferenceFactors {
  int64_t downscale = 1;
  int64_t temporal = 1;
};

// One `A`/`B` factor pair, already resolved onto the DiT contract's tensor name.
struct Ltx2LoraFactorPair {
  // The CONTRACT name this pair fuses into, e.g.
  // "transformer_blocks.0.attn1.to_q.weight".
  std::string target;
  int64_t out_features = 0;
  int64_t rank = 0;
  int64_t in_features = 0;
  // [out, rank] and [rank, in], row-major, bf16 bit patterns. Held bf16 because
  // that is the dtype the products are formed in (`fuse_loras.py:202-203` casts
  // both to the rule's `aggregation_dtype` before the matmul).
  std::vector<uint16_t> b;
  std::vector<uint16_t> a;
};

// One loaded adapter file.
class Ltx2LoraAdapter {
 public:
  // Read `spec.path`, resolve every `.lora_A.weight` / `.lora_B.weight` pair
  // onto a contract name, and keep the file's whole `__metadata__`.
  //
  // `contract` is the set of tensor names the DiT actually binds. A pair whose
  // target is not in it REFUSES BY NAME. Upstream instead SKIPS it
  // (`fuse_loras.py:135-137`, `if original_weight is None: continue`) because
  // its state dict is the whole model and a skip is unambiguous there. Here the
  // contract is a fixed enumerated set with unported modules already stripped,
  // so a skip would absorb a misnamed key and a genuinely inapplicable one
  // alike — and the first is the failure this project keeps paying for. The
  // row's spec §4.1 argues the divergence; it is the narrower behaviour.
  static Ltx2LoraAdapter Open(const Ltx2LoraSpec& spec,
                              const std::vector<std::string>& contract);

  const std::string& path() const { return path_; }
  double strength() const { return strength_; }
  const std::map<std::string, std::string>& metadata() const { return metadata_; }

  // The pair targeting `name`, or nullptr. One target at most: a duplicate is
  // refused at Open.
  const Ltx2LoraFactorPair* Find(const std::string& name) const;

 private:
  std::string path_;
  double strength_ = 1.0;
  std::map<std::string, std::string> metadata_;
  std::vector<Ltx2LoraFactorPair> pairs_;
};

// `diffusion_model.<module>.lora_A.weight` -> `<module>.weight`, mirroring
// LTXV_LORA_COMFY_RENAMING_MAP's prefix strip (`sd_ops.py:136`) composed with
// the `.lora_{A,B}.weight` -> `.weight` rewrite `_affected_weight_keys` performs
// (`fuse_loras.py:185-186`). Returns false when `key` is not a LoRA factor.
bool Ltx2LoraContractName(const std::string& key, std::string* out_target,
                          bool* out_is_a);

// One `__metadata__` integer factor, defaulting to 1 when the key is absent
// (iclora_utils.py:35, 46).
//
// Upstream swallows EVERY failure and returns 1 (`iclora_utils.py:36-38`,
// `except Exception: ... return 1`). A malformed value is refused by name here
// instead, because the two factors silently reverting to 1 place the reference
// plausibly and wrongly — which is the failure mode the reference refusal was
// written to prevent in the first place. Absent still means 1; only malformed
// differs.
int64_t Ltx2ReadLoraMetadataFactor(const std::map<std::string, std::string>& metadata,
                                   const std::string& key, const std::string& path);

// Resolve the reference factors across the adapters, mirroring ic_lora.py:150-173
// including its conflict errors (`:158-163`, `:167-172`).
//
// Refuses more than one adapter by name. Upstream's own `dubit.py` enforces
// exactly one (`dubit.py:364-365`) and `hdr_ic_lora.py` takes exactly one
// (`hdr_ic_lora.py:271-272`); N-adapter fusion is owed by the row rather than
// half-built, and this signature already takes the list it will need.
Ltx2LoraReferenceFactors Ltx2ResolveLoraReferenceFactors(
    const std::vector<Ltx2LoraAdapter>& adapters);

// Add every adapter's delta for `target` into `buffer`, which holds `rows*cols`
// elements of `dtype` as produced by `MaterializeDitTensor`. Returns true when
// anything was fused.
//
// This is `fuse_lora_weights` (`fuse_loras.py:119-150`) collapsed onto one
// already-materialized tensor: the aggregation is bf16, the add is upstream's
// `deltas.add_(weight)` and the store is its `.to(dtype=weight.dtype)`
// (`fuse_loras.py:67-68`).
bool Ltx2FuseLoraIntoTensor(const std::vector<Ltx2LoraAdapter>& adapters,
                            const std::string& target, vt::DType dtype, int64_t rows,
                            int64_t cols, uint8_t* buffer, size_t buffer_bytes);

}  // namespace vllm
