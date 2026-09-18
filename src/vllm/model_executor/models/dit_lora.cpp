// DiT LoRA adapter reading and fusion. See dit_lora.h for the upstream anchors,
// the dtype argument, and the two deliberate divergences.
//
// Generalized from ltx2_lora.cpp (row ROAD-V1-DIT-LORA). The fusion arithmetic,
// adapter loading, spec parsing, and metadata factor reading are generic. Only
// the contract name rewrite is model-specific, and it is injected via the
// `prefixes` parameter.
//
// The `(B * strength) @ A` product is `vt::Matmul`, the shared row-major GEMM
// seam, and NOT a loop in this file (LTX25-LORA-FUSE-SEAM, #1202). The seam
// changes nothing about the arithmetic — the rounding pattern the header's dtype
// note argues for is byte-identical either way, and the row's gate asserts that
// as byte equality — it changes only who executes it. The reason is written
// beside the call.
#include "vllm/model_executor/models/dit_lora.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error("dit lora: " + what);
}

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    out += (i != 0 ? ", " : "") + std::to_string(shape[i]);
  }
  return out + "]";
}

constexpr const char* kLoraASuffix = ".lora_A.weight";
constexpr const char* kLoraBSuffix = ".lora_B.weight";

bool HasSuffix(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Read a rank-2 tensor into bf16 bit patterns. The products are formed in the
// rule's aggregation dtype and upstream casts both factors to it before the
// matmul (`fuse_loras.py:202-203`), so bf16 is where they are held.
std::vector<uint16_t> ReadFactorAsBf16(const std::string& key, const StTensor& t,
                                       const std::string& path) {
  if (t.shape.size() != 2) {
    Fail("'" + key + "' in '" + path + "' is rank " + std::to_string(t.shape.size()) +
         " " + ShapeText(t.shape) + "; a LoRA factor is rank 2");
  }
  int64_t numel = t.shape[0] * t.shape[1];
  if (numel <= 0) {
    Fail("'" + key + "' in '" + path + "' is empty " + ShapeText(t.shape));
  }
  std::vector<uint16_t> out(static_cast<size_t>(numel));
  if (t.dtype == "BF16") {
    if (t.nbytes != out.size() * sizeof(uint16_t)) {
      Fail("'" + key + "' in '" + path + "' declares " + std::to_string(t.nbytes) +
           " BF16 bytes but its shape " + ShapeText(t.shape) + " needs " +
           std::to_string(out.size() * sizeof(uint16_t)));
    }
    std::memcpy(out.data(), t.data, t.nbytes);
    return out;
  }
  if (t.dtype == "F32") {
    if (t.nbytes != out.size() * sizeof(float)) {
      Fail("'" + key + "' in '" + path + "' declares " + std::to_string(t.nbytes) +
           " F32 bytes but its shape " + ShapeText(t.shape) + " needs " +
           std::to_string(out.size() * sizeof(float)));
    }
    // Narrowed to bf16 on purpose: this is the cast upstream performs at
    // `fuse_loras.py:202-203`, not a shortcut. An f32 factor kept f32 would
    // widen the accumulator through the back door.
    for (size_t i = 0; i < out.size(); ++i) {
      float v = 0.0F;
      std::memcpy(&v, t.data + i * sizeof(float), sizeof(float));
      out[i] = vt::F32ToBF16(v);
    }
    return out;
  }
  if (t.dtype == "F16") {
    Fail("'" + key + "' in '" + path +
         "' is F16. Upstream casts LoRA factors to the fuse rule's aggregation dtype "
         "(fuse_loras.py:202-203) and every LTX rule aggregates in BF16, so an F16 "
         "adapter is readable in principle — but no shipped LTX IC-LoRA is F16, so this "
         "port has never seen one and will not guess at its scaling. File an issue with "
         "the adapter.");
  }
  Fail("'" + key + "' in '" + path + "' has dtype " + t.dtype +
       ", which this reader does not read. LoRA factors are BF16 or F32.");
}

// Strip the first matching prefix from `module`. A prefix matches when
// `module` starts with it. Returns true when a prefix was stripped.
bool StripPrefix(std::string& module, const std::vector<std::string>& prefixes) {
  for (const std::string& prefix : prefixes) {
    const size_t plen = prefix.size();
    if (module.size() > plen && module.compare(0, plen, prefix) == 0) {
      module = module.substr(plen);
      return true;
    }
  }
  return false;
}

}  // namespace

bool DitLoraContractName(const std::string& key,
                          const std::vector<std::string>& prefixes,
                          std::string* out_target, bool* out_is_a) {
  bool is_a = false;
  const char* suffix = nullptr;
  if (HasSuffix(key, kLoraASuffix)) {
    is_a = true;
    suffix = kLoraASuffix;
  } else if (HasSuffix(key, kLoraBSuffix)) {
    suffix = kLoraBSuffix;
  } else {
    return false;
  }
  std::string module = key.substr(0, key.size() - std::strlen(suffix));
  // Strip any model-specific ComfyUI prefix. LTXV_LORA_COMFY_RENAMING_MAP
  // strips `diffusion_model.` (sd_ops.py:136); H3 adapters carry
  // `model.diffusion_model.`. The DiT contract's own names are already stripped
  // by the loader's plan, so the two meet.
  StripPrefix(module, prefixes);
  if (module.empty()) return false;
  if (out_target != nullptr) *out_target = module + ".weight";
  if (out_is_a != nullptr) *out_is_a = is_a;
  return true;
}

int64_t DitReadLoraMetadataFactor(const std::map<std::string, std::string>& metadata,
                                   const std::string& key, const std::string& path) {
  const auto it = metadata.find(key);
  // Absent is 1, which is upstream's default (`iclora_utils.py:35, 46`).
  if (it == metadata.end()) return 1;
  const std::string& raw = it->second;
  size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(raw, &consumed);
  } catch (const std::exception&) {
    consumed = 0;
  }
  if (consumed != raw.size() || value < 1) {
    Fail("'" + path + "' carries metadata " + key + "='" + raw +
         "', which is not a positive integer. Upstream swallows this and silently returns "
         "1 (iclora_utils.py:36-38); refusing instead, because a factor that reverts to 1 "
         "places the reference plausibly and WRONGLY, and no output check can see that.");
  }
  return static_cast<int64_t>(value);
}

DitLoraAdapter DitLoraAdapter::Open(const DitLoraSpec& spec,
                                     const std::vector<std::string>& contract,
                                     const std::vector<std::string>& prefixes) {
  if (spec.path.empty()) Fail("an adapter path is empty");
  const std::set<std::string> known(contract.begin(), contract.end());

  DitLoraAdapter out;
  out.path_ = spec.path;
  out.strength_ = spec.strength;

  const SafetensorsFile file = SafetensorsFile::Open(spec.path);
  out.metadata_ = file.Metadata();

  // Gather the A and B halves by target, so a pair missing its other half is
  // reported as the pair it is rather than as two unrelated tensors.
  std::map<std::string, const StTensor*> a_of;
  std::map<std::string, const StTensor*> b_of;
  // The A and B keys are tracked SEPARATELY. One map keyed by target held
  // whichever key was seen last, so a malformed A factor was reported under the
  // B factor's name -- which is precisely the "refuse BY NAME" property these
  // messages exist to have, defeated by the message itself.
  std::map<std::string, std::string> a_key_of;
  std::map<std::string, std::string> b_key_of;
  for (const std::string& key : file.Names()) {
    std::string target;
    bool is_a = false;
    if (!DitLoraContractName(key, prefixes, &target, &is_a)) continue;
    auto& side = is_a ? a_of : b_of;
    if (side.count(target) != 0) {
      Fail("'" + spec.path + "' carries two " + std::string(is_a ? "A" : "B") +
           " factors for '" + target + "'");
    }
    side[target] = &file.Get(key);
    (is_a ? a_key_of : b_key_of)[target] = key;
  }

  if (a_of.empty() && b_of.empty()) {
    Fail("'" + spec.path +
         "' carries no `.lora_A.weight` / `.lora_B.weight` pair at all, so it is not a "
         "LoRA adapter. Refusing rather than loading a model with no delta and reporting "
         "success.");
  }

  for (const auto& kv : a_of) {
    const std::string& target = kv.first;
    const auto b_it = b_of.find(target);
    if (b_it == b_of.end()) {
      Fail("'" + spec.path + "' has an A factor for '" + target +
           "' with no matching B factor");
    }
    if (known.count(target) == 0) {
      // The divergence argued in dit_lora.h and the row's spec §4.1: upstream
      // skips (`fuse_loras.py:135-137`), this refuses.
      Fail("'" + spec.path + "' targets '" + target +
           "', which the DiT contract does not bind. Upstream would SKIP this key "
           "(fuse_loras.py:135-137) because its state dict is the whole model; here the "
           "contract is a fixed enumerated set, so a skip would absorb a MISNAMED key and "
           "an inapplicable one alike. If this adapter was trained against a module this "
           "port does not carry, that is the thing to report.");
    }
    const StTensor& a = *kv.second;
    const StTensor& b = *b_it->second;

    DitLoraFactorPair pair;
    pair.target = target;
    // A is [rank, in], B is [out, rank] (`fuse_loras.py:196-198` pairs them for
    // a [out, in] weight).
    pair.rank = a.shape.size() == 2 ? a.shape[0] : 0;
    pair.in_features = a.shape.size() == 2 ? a.shape[1] : 0;
    pair.out_features = b.shape.size() == 2 ? b.shape[0] : 0;
    pair.a = ReadFactorAsBf16(a_key_of[target], a, spec.path);
    pair.b = ReadFactorAsBf16(b_key_of[target], b, spec.path);
    if (b.shape.size() != 2 || b.shape[1] != pair.rank) {
      Fail("'" + spec.path + "' pairs A " + ShapeText(a.shape) + " with B " +
           ShapeText(b.shape) + " for '" + target +
           "'; B's second dimension must be A's first (the rank)");
    }
    out.pairs_.push_back(std::move(pair));
  }
  for (const auto& kv : b_of) {
    if (a_of.count(kv.first) == 0) {
      Fail("'" + spec.path + "' has a B factor for '" + kv.first +
           "' with no matching A factor");
    }
  }
  return out;
}

const DitLoraFactorPair* DitLoraAdapter::Find(const std::string& name) const {
  for (const DitLoraFactorPair& p : pairs_) {
    if (p.target == name) return &p;
  }
  return nullptr;
}

DitLoraReferenceFactors DitResolveLoraReferenceFactors(
    const std::vector<DitLoraAdapter>& adapters) {
  DitLoraReferenceFactors out;
  if (adapters.empty()) return out;
  // ic_lora.py:150-173, over the WHOLE list. Row LTX25-LORA-FUSION lifted the
  // arity cap that used to refuse a second adapter here, so both conflict
  // branches below are reachable rather than written for a later day.
  //
  // `dubit.py:364-365` and `hdr_ic_lora.py:271-272` still take exactly one
  // adapter each; that is a property of those two PIPELINE ENTRY POINTS, not of
  // the fuser, and `--lora` has always been repeatable (`utils/args.py:600-611`).
  for (const DitLoraAdapter& lora : adapters) {
    const int64_t scale =
        DitReadLoraMetadataFactor(lora.metadata(), "reference_downscale_factor", lora.path());
    if (scale != 1) {
      if (out.downscale != 1 && out.downscale != scale) {
        Fail("conflicting reference_downscale_factor values in LoRAs: already have " +
             std::to_string(out.downscale) + ", but " + lora.path() + " specifies " +
             std::to_string(scale) + ". Cannot combine LoRAs with different reference scales.");
      }
      out.downscale = scale;
    }
    const int64_t temporal = DitReadLoraMetadataFactor(
        lora.metadata(), "reference_temporal_scale_factor", lora.path());
    if (temporal != 1) {
      if (out.temporal != 1 && out.temporal != temporal) {
        Fail("conflicting reference_temporal_scale_factor values in LoRAs: already have " +
             std::to_string(out.temporal) + ", but " + lora.path() + " specifies " +
             std::to_string(temporal) +
             ". Cannot combine LoRAs with different temporal scales.");
      }
      out.temporal = temporal;
    }
  }
  return out;
}

bool DitFuseLoraIntoTensor(const std::vector<DitLoraAdapter>& adapters,
                            const std::string& target, vt::DType dtype, int64_t rows,
                            int64_t cols, uint8_t* buffer, size_t buffer_bytes) {
  if (adapters.empty()) return false;

  // The aggregator. BF16 BY DECLARATION — see the dtype note in dit_lora.h.
  // `has_delta` rather than an empty vector, because a delta of exactly zero is
  // a legitimate (if useless) adapter and must still count as fused.
  std::vector<uint16_t> agg;
  bool has_delta = false;

  for (const DitLoraAdapter& lora : adapters) {
    const DitLoraFactorPair* pair = lora.Find(target);
    if (pair == nullptr) continue;
    if (pair->out_features != rows || pair->in_features != cols) {
      Fail("'" + lora.path() + "' targets '" + target + "' with a [" +
           std::to_string(pair->out_features) + ", " + std::to_string(pair->in_features) +
           "] delta, but that tensor is [" + std::to_string(rows) + ", " +
           std::to_string(cols) + "] in this checkpoint");
    }
    // THE TWO PRODUCT FORMS (`fuse_loras.py:110-116`). Upstream's aggregator
    // materializes its accumulator from the FIRST product and folds every one
    // after it in with `addmm_`, and its own docstring (`:103-107`) says the
    // difference in rounding is deliberate. Which form applies is a property of
    // THIS TENSOR and not of the load: an adapter that targets a tensor no
    // earlier adapter touched takes the first form there, because
    // `_products_for_sd_key` skips a LoRA that lacks the key
    // (`fuse_loras.py:200-201`) and the aggregator never sees it.
    //
    // MEASURED, NOT TRANSCRIBED. Three candidate models of `addmm_` on a bf16
    // accumulator were run against the pinned module itself over 49 randomized
    // trials; only the one below matched, on every one of them. The two that
    // failed differ from it on 18 and 21 of the 120 elements of this row's own
    // golden fixture — which is to say a wrong choice here is invisible to any
    // tolerance and visible only to a byte comparison. The recipe and the counts
    // are in .agents/specs/ltx25-lora-fusion.md §3.
    if (has_delta) {
      // `aggregated.addmm_(B, A, alpha=strength)` (`fuse_loras.py:115`).
      //
      // THE STRENGTH ENTERS AFTER THE GEMM AND BEFORE THE ONLY ROUNDING. torch
      // converts `alpha` to the op math type (f32 for a bf16 tensor) and applies
      // it to the f32 accumulation, so there is NO bf16 store between `B @ A`
      // and the scale. That is why the output tensor here is f32 where the first
      // form's is bf16 — the same `vt::Matmul` seam, whose contract already
      // admits "out f32 or bf16, f32 accumulation" (`vt/ops.h`), and not a
      // widening of the landed first-form arm.
      //
      // IT COSTS ONE TRANSIENT f32 BUFFER of the target's shape, live only
      // while this adapter folds in and only on a tensor a previous adapter
      // already touched. The loader's "one host buffer live at a time"
      // invariant is about the DEVICE copy and is unaffected; the peak here is
      // the bf16 aggregator plus this, freed before the next target.
      if (pair->rank > 0 && !agg.empty()) {
        std::vector<float> prod(agg.size(), 0.0F);
        vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
        const vt::Device dev{vt::DeviceType::kCPU, 0};
        vt::Tensor b_t = vt::Tensor::Contiguous(const_cast<uint16_t*>(pair->b.data()),
                                                vt::DType::kBF16, dev, {rows, pair->rank});
        vt::Tensor a_t = vt::Tensor::Contiguous(const_cast<uint16_t*>(pair->a.data()),
                                                vt::DType::kBF16, dev, {pair->rank, cols});
        vt::Tensor o_t =
            vt::Tensor::Contiguous(prod.data(), vt::DType::kF32, dev, {rows, cols});
        vt::Matmul(q, o_t, b_t, a_t);
        const float alpha = static_cast<float>(lora.strength());
        for (size_t i = 0; i < agg.size(); ++i) {
          agg[i] = vt::F32ToBF16(vt::BF16ToF32(agg[i]) + alpha * prod[i]);
        }
      }
      has_delta = true;
      continue;
    }
    agg.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);

    // `(B * strength) @ A`, upstream's FIRST product form (`fuse_loras.py:113`).
    // `B * strength` is a bf16 tensor times a Python float and stays bf16, so
    // the scaled factor is ROUNDED TO BF16 BEFORE the matmul — not folded into
    // the accumulation. That rounding is the pattern the docstring at
    // `fuse_loras.py:103-107` says is preserved, so it is preserved here.
    std::vector<uint16_t> bs(pair->b.size());
    for (size_t i = 0; i < bs.size(); ++i) {
      bs[i] = vt::F32ToBF16(vt::BF16ToF32(pair->b[i]) * static_cast<float>(lora.strength()));
    }
    // torch's bf16 matmul accumulates in f32 and stores bf16; mirrored — and
    // that is `vt::Matmul`'s written contract, not an approximation of it
    // (`vt/ops.h`: "a/b float dtypes (f32/f16/bf16), out f32 or bf16, f32
    // accumulation"). The seam's CPU kernel vectorizes ACROSS OUTPUT COLUMNS
    // rather than along K, so every output element keeps the same strictly
    // sequential f32 reduction the scalar loop this replaced had, with mul+add
    // and never an FMA (`cpu_matmul_elem.h`, the recorded deviation from ggml);
    // its store is the same `vt::F32ToBF16` through `StoreF32`. So this is the
    // same ARITHMETIC on a different execution strategy, and the row's gate
    // asserts that as byte equality rather than as a tolerance.
    //
    // The operand orientation needs no transpose: `bs` is [rows, rank] and A is
    // [rank, cols], both row-major, which is exactly kMatmul's `out[M,N] =
    // a[M,K] @ b[K,N]`. (`vt::MatmulBT` would need A as [cols, rank] and is
    // therefore the wrong member of the pair here, notwithstanding that it is
    // the one the sibling text-tower row took.)
    //
    // The guard is not reachable through `DitLoraAdapter::Open`, which is the
    // only way a pair is built: `ReadFactorAsBf16` refuses an empty factor, and
    // A is [rank, in], so `rank == 0` is already a refusal by the time anything
    // gets here — as is a zero-sized target, whose `agg` would be empty. It is
    // written anyway because the loop this replaced HANDLED both: a zero trip
    // count left every output at its zero-seeded accumulator, and `agg` is
    // zero-filled, so skipping the GEMM reproduces that exactly. That makes the
    // replacement behaviour-preserving rather than merely equivalent wherever
    // the tests happen to look, and it keeps a zero-shaped tensor away from
    // `vt::Matmul`, whose contract does not speak to one.
    if (pair->rank > 0 && !agg.empty()) {
      vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
      const vt::Device dev{vt::DeviceType::kCPU, 0};
      vt::Tensor b_t = vt::Tensor::Contiguous(bs.data(), vt::DType::kBF16, dev,
                                              {rows, pair->rank});
      vt::Tensor a_t = vt::Tensor::Contiguous(const_cast<uint16_t*>(pair->a.data()),
                                              vt::DType::kBF16, dev, {pair->rank, cols});
      vt::Tensor o_t = vt::Tensor::Contiguous(agg.data(), vt::DType::kBF16, dev,
                                              {rows, cols});
      vt::Matmul(q, o_t, b_t, a_t);
    }
    has_delta = true;
  }
  if (!has_delta) return false;

  const size_t numel = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  // `deltas.add_(weight)` then `.to(dtype=weight.dtype)` (`fuse_loras.py:67-68`).
  // The add happens IN THE AGGREGATOR'S dtype, which is bf16 even when the
  // target is f32 — that is upstream's in-place semantics, not an approximation
  // of them, and it is why the f32 branch below rounds through bf16.
  if (dtype == vt::DType::kBF16) {
    if (buffer_bytes != numel * sizeof(uint16_t)) {
      Fail("'" + target + "' is " + std::to_string(buffer_bytes) +
           " bf16 bytes but its shape needs " + std::to_string(numel * sizeof(uint16_t)));
    }
    auto* w = reinterpret_cast<uint16_t*>(buffer);
    for (size_t i = 0; i < numel; ++i) {
      w[i] = vt::F32ToBF16(vt::BF16ToF32(agg[i]) + vt::BF16ToF32(w[i]));
    }
    return true;
  }
  if (dtype == vt::DType::kF32) {
    if (buffer_bytes != numel * sizeof(float)) {
      Fail("'" + target + "' is " + std::to_string(buffer_bytes) +
           " f32 bytes but its shape needs " + std::to_string(numel * sizeof(float)));
    }
    for (size_t i = 0; i < numel; ++i) {
      float w = 0.0F;
      std::memcpy(&w, buffer + i * sizeof(float), sizeof(float));
      // add_ on the bf16 aggregator rounds the SUM to bf16 before `.to(f32)`.
      const float sum = vt::BF16ToF32(vt::F32ToBF16(vt::BF16ToF32(agg[i]) + w));
      std::memcpy(buffer + i * sizeof(float), &sum, sizeof(float));
    }
    return true;
  }
  Fail("'" + target + "' materialized as a dtype this fuser does not write. FP8 and NVFP4 "
       "never reach here: the materializer dequantizes both to BF16 before returning.");
}

// ── extras-map resolution ───────────────────────────────────────────────────

namespace {

std::string LoraIndexSuffix(int64_t index) {
  return index == 1 ? std::string() : "_" + std::to_string(index);
}

bool LoraExtraIndex(const std::string& key, int64_t* out_index) {
  for (const char* base : {kDitLoraPathExtra, kDitLoraStrengthExtra}) {
    const std::string prefix = std::string(base) + "_";
    if (key.size() <= prefix.size()) continue;
    if (key.compare(0, prefix.size(), prefix) != 0) continue;
    const std::string digits = key.substr(prefix.size());
    if (digits[0] == '0' || digits.size() > 6) return false;
    for (const char c : digits) {
      if (c < '0' || c > '9') return false;
    }
    const int64_t index = std::stoll(digits);
    if (index < 2) return false;
    if (out_index != nullptr) *out_index = index;
    return true;
  }
  return false;
}

double ParseLoraStrength(const std::string& key, const std::string& raw) {
  try {
    size_t consumed = 0;
    const double value = std::stod(raw, &consumed);
    if (consumed != raw.size()) throw std::invalid_argument("trailing");
    if (!std::isfinite(value)) throw std::invalid_argument("non-finite");
    return value;
  } catch (const std::exception&) {
    Fail("the extra '" + key + "' is '" + raw + "', which is not a finite number");
  }
}

std::string ExtraGet(const std::map<std::string, std::string>& extras,
                    const std::string& key) {
  auto it = extras.find(key);
  return it == extras.end() ? std::string() : it->second;
}

// Resolve a runtime lora adapter name to a file path, mirroring LocalAI sd.cpp's
// discover_lora_files + fallback chain (gosd.cpp:110-158, 174-330). Tries, in
// order: absolute path, exact filename in lora_dir, extension probing
// (.safetensors), case-insensitive scan of lora_dir. A name with a path
// separator that is not an absolute path is refused to prevent directory
// traversal. A name that resolves to nothing is refused by name.
std::string ResolveLoraName(const std::string& name,
                             const std::string& lora_dir) {
  namespace fs = std::filesystem;

  auto to_lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
  };
  static const std::string kExt = ".safetensors";
  auto has_ext = [&](const std::string& s) {
    return s.size() > kExt.size() &&
           s.compare(s.size() - kExt.size(), kExt.size(), kExt) == 0;
  };

  // Refuse names with path separators (except absolute paths) to prevent
  // directory traversal.
  if (!name.empty() && name[0] != '/') {
    if (name.find('/') != std::string::npos ||
        name.find('\\') != std::string::npos) {
      Fail("the lora adapter name '" + name + "' contains a path separator. Only "
           "simple filenames (resolved against lora_dir) and absolute paths are "
           "accepted, to prevent directory traversal. Refusing.");
    }
  }

  // 1. Absolute path.
  if (!name.empty() && name[0] == '/') {
    if (fs::exists(name)) return name;
    Fail("the lora adapter path '" + name + "' does not exist.");
  }

  // 2. Exact filename in lora_dir.
  {
    fs::path p = fs::path(lora_dir) / name;
    if (fs::exists(p)) return p.string();
  }

  // 3. Extension probing: append .safetensors.
  if (!has_ext(name)) {
    fs::path p = fs::path(lora_dir) / (name + kExt);
    if (fs::exists(p)) return p.string();
  }

  // 4. Case-insensitive scan of lora_dir.
  {
    std::string name_lower = to_lower(name);
    std::string with_ext_lower = name_lower;
    if (!has_ext(name)) with_ext_lower += kExt;
    std::error_code ec;
    if (fs::is_directory(lora_dir, ec)) {
      for (const auto& entry : fs::directory_iterator(lora_dir, ec)) {
        if (ec) break;
        std::string fn_lower = to_lower(entry.path().filename().string());
        if (fn_lower == name_lower || fn_lower == with_ext_lower) {
          return entry.path().string();
        }
      }
    }
  }

  Fail("the lora adapter '" + name + "' was not found. Searched: exact filename in '" +
       lora_dir + "', extension probing ('.safetensors'), and case-insensitive match "
       "in '" + lora_dir + "'. Refusing rather than loading a model with no adapter.");
}

}  // namespace

std::vector<DitLoraSpec> ResolveDitLoraSpecs(
    const std::map<std::string, std::string>& extras) {
  std::vector<DitLoraSpec> out;
  for (int64_t index = 1;; ++index) {
    const std::string path_key =
       std::string(kDitLoraPathExtra) + LoraIndexSuffix(index);
    const std::string strength_key =
       std::string(kDitLoraStrengthExtra) + LoraIndexSuffix(index);
    const std::string path = ExtraGet(extras, path_key);
    const std::string strength = ExtraGet(extras, strength_key);
    if (path.empty()) {
      if (!strength.empty()) {
       Fail("'" + strength_key + "' was given without '" + path_key +
            "'. A strength with no adapter fuses nothing, and silently doing "
            "nothing is what this refusal exists to prevent.");
      }
      break;
    }
    DitLoraSpec spec;
    spec.path = path;
    if (!strength.empty()) spec.strength = ParseLoraStrength(strength_key, strength);
    out.push_back(std::move(spec));
  }
  const int64_t next = static_cast<int64_t>(out.size()) + 1;
  for (const auto& kv : extras) {
    int64_t index = 0;
    if (!LoraExtraIndex(kv.first, &index) || index <= next) continue;
    Fail("the load carries '" + kv.first + "' but no '" +
        std::string(kDitLoraPathExtra) + LoraIndexSuffix(next) +
        "', so the adapters are not numbered 1..N with no gaps. " +
        std::to_string(out.size()) +
        " adapter(s) would be fused and the rest silently dropped. Number them "
        "from 1 — the first is '" +
        std::string(kDitLoraPathExtra) +
        "' with no index — or drop '" + kv.first + "'.");
  }
  return out;
}

bool IsDitLoraIndexedExtra(const std::string& key) {
  return LoraExtraIndex(key, nullptr);
}

bool IsDitLoraExtra(const std::string& key) {
  if (key == kDitLoraPathExtra || key == kDitLoraStrengthExtra) return true;
  return IsDitLoraIndexedExtra(key);
}

// ── per-load fusion helpers ─────────────────────────────────────────────────

std::vector<DitLoraAdapter> DitOpenLoras(
    const std::vector<DitLoraSpec>& specs,
    const std::vector<std::string>& contract_names,
    const std::vector<std::string>& prefixes) {
  if (specs.empty()) return {};
  std::vector<DitLoraAdapter> out;
  out.reserve(specs.size());
  for (const DitLoraSpec& spec : specs) {
    out.push_back(DitLoraAdapter::Open(spec, contract_names, prefixes));
  }
  return out;
}

bool DitFuseLorasIntoBuffer(
    const std::vector<DitLoraAdapter>& loras,
    const std::string& name, const std::vector<int64_t>& shape,
    vt::DType dtype, uint8_t* buffer, size_t buffer_bytes) {
  if (loras.empty()) return false;
  if (shape.size() != 2) return false;
  return DitFuseLoraIntoTensor(loras, name, dtype, shape[0], shape[1],
                                buffer, buffer_bytes);
}

void DitCheckLorasWereApplied(
    const std::vector<DitLoraAdapter>& loras, int64_t fused) {
  if (loras.empty() || fused > 0) return;
  std::string paths;
  for (const DitLoraAdapter& lora : loras) {
    paths += std::string(paths.empty() ? "" : ", ") + "'" + lora.path() + "'";
  }
  Fail("the adapter(s) " + paths +
       " fused into ZERO tensors of this checkpoint. Every A/B pair named a tensor the "
       "contract binds, so the delta was computed for none of them — which means the "
       "render would be byte-identical to loading no adapter, while reporting success. "
       "Refusing instead.");
}

// ── runtime prompt-activated LoRA (ROAD-V1-LORA-RUNTIME) ─────────────────────

DitParseLoraResult DitParseLoraTags(const std::string& prompt,
                                     const std::string& lora_dir) {
  DitParseLoraResult result;

  static const std::regex kLoraTag(R"(<lora:([^:>]+):([^>]+)>)");

  std::string clean;
  size_t last_end = 0;

  for (std::sregex_iterator it(prompt.begin(), prompt.end(), kLoraTag), end;
       it != end; ++it) {
    const std::smatch& m = *it;
    const size_t pos = static_cast<size_t>(m.position());
    clean += prompt.substr(last_end, pos - last_end);
    last_end = pos + m.length();

    const std::string name = m[1].str();
    const std::string strength_str = m[2].str();

    // Parse strength.
    double strength = 1.0;
    try {
      size_t consumed = 0;
      strength = std::stod(strength_str, &consumed);
      if (consumed != strength_str.size() || !std::isfinite(strength)) {
        throw std::invalid_argument("bad strength");
      }
    } catch (const std::exception&) {
      Fail("the lora tag '<lora:" + name + ":" + strength_str +
           ">' carries strength '" + strength_str +
           "', which is not a finite number");
    }

    // Resolve name to a file path.
    const std::string path = ResolveLoraName(name, lora_dir);

    // Accumulate strength for duplicate adapters (same resolved path).
    // Mirrors sd.cpp multiplier accumulation (gosd.cpp:300-310).
    bool found = false;
    for (auto& spec : result.loras) {
      if (spec.path == path) {
        spec.strength += strength;
        found = true;
        break;
      }
    }
    if (!found) {
      result.loras.push_back({path, strength});
    }
  }

  // Append remaining text after the last tag.
  clean += prompt.substr(last_end);

  // Collapse whitespace: replace runs of whitespace with a single space, trim
  // leading and trailing. Mirrors Go strings.TrimSpace(Join(Fields(s), " ")).
  std::string collapsed;
  bool in_ws = true;
  for (char c : clean) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (!in_ws) { collapsed += ' '; in_ws = true; }
    } else {
      collapsed += c;
      in_ws = false;
    }
  }
  if (!collapsed.empty() && collapsed.back() == ' ') collapsed.pop_back();

  result.clean_prompt = collapsed;
  return result;
}

// ── runtime LoRA delta (CPU path) ──────────────────────────────────────────

void DitApplyRuntimeLoraDelta(vt::Queue& q, const vt::Tensor& a,
                               float* out, int64_t rows, int64_t out_features,
                               const DitRuntimeLoraLayer* lora) {
  if (lora == nullptr) return;
  const int64_t rank = lora->lora_a.shape[0];
  std::vector<float> tmp_buf(static_cast<size_t>(rows * rank));
  vt::Tensor tmp = vt::Tensor::Contiguous(tmp_buf.data(), vt::DType::kF32,
                                          a.device, {rows, rank});
  vt::MatmulBT(q, tmp, a, lora->lora_a);
  std::vector<float> delta_buf(static_cast<size_t>(rows * out_features));
  vt::Tensor delta = vt::Tensor::Contiguous(delta_buf.data(), vt::DType::kF32,
                                            a.device, {rows, out_features});
  vt::MatmulBT(q, delta, tmp, lora->lora_b);
  vt::Tensor o = vt::Tensor::Contiguous(out, vt::DType::kF32, a.device,
                                        {rows, out_features});
  vt::Add(q, o, o, delta);
}

// ── runtime LoRA loading (ROAD-V1-LORA-RUNTIME phase 5) ─────────────────────

// Read a rank-2 LoRA factor into f32 values. Handles BF16, F32, and F16 storage.
// Unlike the load-time ReadFactorAsBf16 (which narrows to bf16 for the fuse
// rule's aggregation dtype), runtime LoRA computes the delta in f32 throughout.
std::vector<float> ReadFactorAsF32(const std::string& key, const StTensor& t,
                                     const std::string& path) {
  if (t.shape.size() != 2) {
    Fail("'" + key + "' in '" + path + "' is rank " + std::to_string(t.shape.size()) +
         " " + ShapeText(t.shape) + "; a LoRA factor is rank 2");
  }
  int64_t numel = t.shape[0] * t.shape[1];
  if (numel <= 0) {
    Fail("'" + key + "' in '" + path + "' is empty " + ShapeText(t.shape));
  }
  std::vector<float> out(static_cast<size_t>(numel));
  if (t.dtype == "F32") {
    if (t.nbytes != out.size() * sizeof(float)) {
      Fail("'" + key + "' in '" + path + "' declares " + std::to_string(t.nbytes) +
           " F32 bytes but its shape " + ShapeText(t.shape) + " needs " +
           std::to_string(out.size() * sizeof(float)));
    }
    std::memcpy(out.data(), t.data, t.nbytes);
    return out;
  }
  if (t.dtype == "BF16") {
    if (t.nbytes != out.size() * sizeof(uint16_t)) {
      Fail("'" + key + "' in '" + path + "' declares " + std::to_string(t.nbytes) +
           " BF16 bytes but its shape " + ShapeText(t.shape) + " needs " +
           std::to_string(out.size() * sizeof(uint16_t)));
    }
    const auto* raw = reinterpret_cast<const uint16_t*>(t.data);
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = vt::BF16ToF32(raw[i]);
    }
    return out;
  }
  if (t.dtype == "F16") {
    if (t.nbytes != out.size() * sizeof(uint16_t)) {
      Fail("'" + key + "' in '" + path + "' declares " + std::to_string(t.nbytes) +
           " F16 bytes but its shape " + ShapeText(t.shape) + " needs " +
           std::to_string(out.size() * sizeof(uint16_t)));
    }
    const auto* raw = reinterpret_cast<const uint16_t*>(t.data);
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = vt::F16ToF32(raw[i]);
    }
    return out;
  }
  Fail("'" + key + "' in '" + path + "' has dtype " + t.dtype +
       ", which this reader does not read. LoRA factors are BF16, F16, or F32.");
}

DitRuntimeLoraState DitLoadRuntimeLoras(
    const std::vector<DitRuntimeLoraSpec>& specs,
    const std::vector<std::string>& contract_names,
    const std::vector<std::string>& prefixes, vt::Device device) {
  DitRuntimeLoraState state;
  if (specs.empty()) return state;

  const std::set<std::string> known(contract_names.begin(), contract_names.end());

  for (const DitRuntimeLoraSpec& spec : specs) {
    if (spec.path.empty()) Fail("a runtime LoRA adapter path is empty");

    const SafetensorsFile file = SafetensorsFile::Open(spec.path);
    const auto& metadata = file.Metadata();

    // lora_alpha from __metadata__ (default: rank, so alpha/rank = 1).
    // Mirrors vLLM-Omni optimize() (lora_weights.py:31-41): scaling = alpha/rank,
    // folded into lora_b.
    int64_t lora_alpha = 0;
    const auto alpha_it = metadata.find("lora_alpha");
    if (alpha_it != metadata.end()) {
      try {
        lora_alpha = std::stoll(alpha_it->second);
      } catch (const std::exception&) {
        Fail("'" + spec.path + "' carries metadata lora_alpha='" +
             alpha_it->second + "', which is not an integer");
      }
      if (lora_alpha < 1) {
        Fail("'" + spec.path + "' carries metadata lora_alpha='" +
             alpha_it->second + "', which is not a positive integer");
      }
    }

    // Gather A and B halves by target, mirroring DitLoraAdapter::Open.
    std::map<std::string, const StTensor*> a_of;
    std::map<std::string, const StTensor*> b_of;
    std::map<std::string, std::string> a_key_of;
    std::map<std::string, std::string> b_key_of;
    for (const std::string& key : file.Names()) {
      std::string target;
      bool is_a = false;
      if (!DitLoraContractName(key, prefixes, &target, &is_a)) continue;
      auto& side = is_a ? a_of : b_of;
      if (side.count(target) != 0) {
        Fail("'" + spec.path + "' carries two " + std::string(is_a ? "A" : "B") +
             " factors for '" + target + "'");
      }
      side[target] = &file.Get(key);
      (is_a ? a_key_of : b_key_of)[target] = key;
    }

    if (a_of.empty() && b_of.empty()) {
      Fail("'" + spec.path +
           "' carries no `.lora_A.weight` / `.lora_B.weight` pair at all, "
           "so it is not a LoRA adapter");
    }

    for (const auto& kv : a_of) {
      const std::string& target = kv.first;
      const auto b_it = b_of.find(target);
      if (b_it == b_of.end()) {
        Fail("'" + spec.path + "' has an A factor for '" + target +
             "' with no matching B factor");
      }
      if (known.count(target) == 0) {
        Fail("'" + spec.path + "' targets '" + target +
             "', which the DiT contract does not bind");
      }
      if (state.layers.count(target) != 0) {
        Fail("'" + spec.path + "' targets '" + target +
             "', which a previous runtime LoRA adapter already bound. "
             "Multiple runtime adapters targeting the same layer are not "
             "supported; use prompt-tag strength instead");
      }

      const StTensor& a = *kv.second;
      const StTensor& b = *b_it->second;
      const int64_t rank = a.shape.size() == 2 ? a.shape[0] : 0;
      const int64_t in_features = a.shape.size() == 2 ? a.shape[1] : 0;
      const int64_t out_features = b.shape.size() == 2 ? b.shape[0] : 0;
      if (b.shape.size() != 2 || b.shape[1] != rank) {
        Fail("'" + spec.path + "' pairs A " + ShapeText(a.shape) + " with B " +
             ShapeText(b.shape) + " for '" + target +
             "'; B's second dimension must be A's first (the rank)");
      }

      // Read factors as f32.
      std::vector<float> a_data = ReadFactorAsF32(a_key_of[target], a, spec.path);
      std::vector<float> b_data = ReadFactorAsF32(b_key_of[target], b, spec.path);

      // Fold alpha/rank and strength into B.
      // effective_b = b * (alpha/rank) * strength
      const int64_t effective_alpha = lora_alpha > 0 ? lora_alpha : rank;
      const float scale = static_cast<float>(effective_alpha) /
                          static_cast<float>(rank) *
                          static_cast<float>(spec.strength);
      for (float& v : b_data) v *= scale;

      // Store in backing storage (map elements are node-based, data pointers
      // stay stable for the state's lifetime).
      state.a_storage[target] = std::move(a_data);
      state.b_storage[target] = std::move(b_data);

      // Construct tensor views into the backing storage.
      DitRuntimeLoraLayer layer;
      layer.lora_a = vt::Tensor::Contiguous(
          state.a_storage[target].data(), vt::DType::kF32, device,
          {rank, in_features});
      layer.lora_b = vt::Tensor::Contiguous(
          state.b_storage[target].data(), vt::DType::kF32, device,
          {out_features, rank});
      layer.strength = 1.0f;  // already folded into lora_b
      state.layers[target] = layer;
    }
    for (const auto& kv : b_of) {
      if (a_of.count(kv.first) == 0) {
        Fail("'" + spec.path + "' has a B factor for '" + kv.first +
             "' with no matching A factor");
      }
    }
  }
  return state;
}

}  // namespace vllm
