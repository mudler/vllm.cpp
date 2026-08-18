// LTX-2.5 IC-LoRA adapter reading and fusion. See ltx2_lora.h for the upstream
// anchors, the dtype argument, and the two deliberate divergences.
//
// A separate translation unit from `ltx2_loader.cpp` for the reason that file's
// siblings already record: `ltx2_loader.cpp` is 1500 lines that several rows
// edit concurrently, and a new family does not need to lock it.
//
// The `(B * strength) @ A` product is `vt::Matmul`, the shared row-major GEMM
// seam, and NOT a loop in this file (LTX25-LORA-FUSE-SEAM, #1202). The seam
// changes nothing about the arithmetic — the rounding pattern the header's dtype
// note argues for is byte-identical either way, and the row's gate asserts that
// as byte equality — it changes only who executes it. The reason is written
// beside the call.
#include "vllm/model_executor/models/ltx2_lora.h"

#include <algorithm>
#include <cstring>
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
  throw std::runtime_error("ltx2 lora: " + what);
}

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    out += (i != 0 ? ", " : "") + std::to_string(shape[i]);
  }
  return out + "]";
}

// LTXV_LORA_COMFY_RENAMING_MAP is a single prefix strip (`sd_ops.py:136`).
constexpr const char* kComfyPrefix = "diffusion_model.";
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

}  // namespace

bool Ltx2LoraContractName(const std::string& key, std::string* out_target,
                          bool* out_is_a) {
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
  // LTXV_LORA_COMFY_RENAMING_MAP: strip `diffusion_model.` when present
  // (`sd_ops.py:136`). The DiT contract's own names are already stripped of
  // `model.diffusion_model.` by the loader's plan, so the two meet.
  const size_t plen = std::strlen(kComfyPrefix);
  if (module.size() > plen && module.compare(0, plen, kComfyPrefix) == 0) {
    module = module.substr(plen);
  }
  if (module.empty()) return false;
  if (out_target != nullptr) *out_target = module + ".weight";
  if (out_is_a != nullptr) *out_is_a = is_a;
  return true;
}

int64_t Ltx2ReadLoraMetadataFactor(const std::map<std::string, std::string>& metadata,
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

Ltx2LoraAdapter Ltx2LoraAdapter::Open(const Ltx2LoraSpec& spec,
                                      const std::vector<std::string>& contract) {
  if (spec.path.empty()) Fail("an adapter path is empty");
  const std::set<std::string> known(contract.begin(), contract.end());

  Ltx2LoraAdapter out;
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
    if (!Ltx2LoraContractName(key, &target, &is_a)) continue;
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
      // The divergence argued in ltx2_lora.h and the row's spec §4.1: upstream
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

    Ltx2LoraFactorPair pair;
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

const Ltx2LoraFactorPair* Ltx2LoraAdapter::Find(const std::string& name) const {
  for (const Ltx2LoraFactorPair& p : pairs_) {
    if (p.target == name) return &p;
  }
  return nullptr;
}

Ltx2LoraReferenceFactors Ltx2ResolveLoraReferenceFactors(
    const std::vector<Ltx2LoraAdapter>& adapters) {
  Ltx2LoraReferenceFactors out;
  if (adapters.empty()) return out;
  if (adapters.size() > 1) {
    Fail("this port fuses exactly ONE adapter and " + std::to_string(adapters.size()) +
         " were given. Upstream's own dubit.py enforces the same (dubit.py:364-365) and "
         "hdr_ic_lora.py takes exactly one (hdr_ic_lora.py:271-272); ic_lora.py does "
         "accept a list, and N-adapter fusion is recorded as owed by row LTX25-IC-LORA "
         "rather than half-built.");
  }
  // ic_lora.py:150-173. The conflict branches are kept even though one adapter
  // cannot trip them, because they are the loop N-adapter support will use and
  // removing them would have to reinvent them.
  for (const Ltx2LoraAdapter& lora : adapters) {
    const int64_t scale =
        Ltx2ReadLoraMetadataFactor(lora.metadata(), "reference_downscale_factor", lora.path());
    if (scale != 1) {
      if (out.downscale != 1 && out.downscale != scale) {
        Fail("conflicting reference_downscale_factor values in LoRAs: already have " +
             std::to_string(out.downscale) + ", but " + lora.path() + " specifies " +
             std::to_string(scale) + ". Cannot combine LoRAs with different reference scales.");
      }
      out.downscale = scale;
    }
    const int64_t temporal = Ltx2ReadLoraMetadataFactor(
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

bool Ltx2FuseLoraIntoTensor(const std::vector<Ltx2LoraAdapter>& adapters,
                            const std::string& target, vt::DType dtype, int64_t rows,
                            int64_t cols, uint8_t* buffer, size_t buffer_bytes) {
  if (adapters.empty()) return false;

  // The aggregator. BF16 BY DECLARATION — see the dtype note in ltx2_lora.h.
  // `has_delta` rather than an empty vector, because a delta of exactly zero is
  // a legitimate (if useless) adapter and must still count as fused.
  std::vector<uint16_t> agg;
  bool has_delta = false;

  for (const Ltx2LoraAdapter& lora : adapters) {
    const Ltx2LoraFactorPair* pair = lora.Find(target);
    if (pair == nullptr) continue;
    if (pair->out_features != rows || pair->in_features != cols) {
      Fail("'" + lora.path() + "' targets '" + target + "' with a [" +
           std::to_string(pair->out_features) + ", " + std::to_string(pair->in_features) +
           "] delta, but that tensor is [" + std::to_string(rows) + ", " +
           std::to_string(cols) + "] in this checkpoint");
    }
    if (has_delta) {
      // Unreachable: Ltx2ResolveLoraReferenceFactors refuses more than one
      // adapter, so no second product can arrive. Upstream's second form is
      // `addmm_(B, A, alpha=strength)`, which rounds differently from the first
      // (`fuse_loras.py:103-116`); implementing it here would land a branch
      // nothing can select, so it refuses instead of guessing.
      Fail("two adapters both target '" + target +
           "'. Upstream aggregates them with a SECOND rounding pattern "
           "(addmm_ with alpha, fuse_loras.py:115) that this port does not implement, "
           "because only one adapter is accepted. Recorded as owed by row LTX25-IC-LORA.");
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
    // The guard is not reachable through `Ltx2LoraAdapter::Open`, which is the
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
       "never reach here: MaterializeDitTensor dequantizes both to BF16 before returning "
       "(ltx2_loader.cpp, the F8_E4M3 and U8 branches both `return vt::DType::kBF16`).");
}

}  // namespace vllm
