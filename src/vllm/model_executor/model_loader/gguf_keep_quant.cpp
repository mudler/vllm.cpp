// vllm.cpp ORIGINAL — GGUF keep-quantized residency policy. See
// gguf_keep_quant.h for the upstream (llama.cpp @ 237ad9b96) anchors and the
// totality contract.
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

#include <cstdlib>
#include <cstring>

#include "vllm/config/weight_residency.h"
#include "vllm/model_executor/device_placement.h"
#include "vt/ops.h"
#include "vt/quant.h"

namespace vllm {
namespace {

// -1 means "this role never keeps its blocks"; otherwise the K (in-features)
// dimension whose block alignment decides eligibility.
//
// NOTE the deliberate absence of a `default:` label: a new GgufTensorRole that
// forgets to state its residency is a -Werror=switch BUILD failure, which is
// the compile-time half of the totality proof (the unit tests are the other).
int64_t KeepQuantKDim(GgufTensorRole role, const std::vector<int64_t>& shape) {
  switch (role) {
    case GgufTensorRole::kMatmulWeight:
      // [out, in]; ggml's src0 orientation, our MatmulBT [N, K] with nk=true.
      return shape.size() == 2 ? shape[1] : -1;
    case GgufTensorRole::kStackedExpertWeight:
      // [E, out, in]; each expert slice is whole rows of the same K.
      return shape.size() == 3 ? shape[2] : -1;
    case GgufTensorRole::kEmbeddingTable:
      // [vocab, hidden]; a GATHER, not a GEMM. The blocks stay and
      // `vt::Embedding` decodes ONE ROW per gathered id (port of llama.cpp's
      // `ggml_compute_forward_get_rows_q`, ggml/src/ggml-cpu/ops.cpp:4850
      // @ b10451). K is the row length, and the same whole-blocks rule applies:
      // a ragged row has no stride. This returned -1 until the dequantizing
      // gather existed, and the -1 was CORRECT until then — without the op a
      // kept table is a tensor nothing can read.
      return shape.size() == 2 ? shape[1] : -1;
    case GgufTensorRole::kTransformedWeight:
    case GgufTensorRole::kConvWeight:
    case GgufTensorRole::kVector:
      return -1;
  }
  return -1;  // unreachable for a valid enumerator (see -Wswitch above)
}

// L6 keep-f16 eligibility by role. Same verbatim-bytes roles as keep-quant PLUS
// the embedding table: an F16 gather table stays F16 (the gather widens f16 to
// f32 exactly like it did bf16), which is what lets a tied token_embd/lm_head be
// ONE resident f16 vocab matrix instead of a bf16 re-expansion. A value/layout
// rewrite (kTransformedWeight), a conv filter or a 1-D vector never keeps native
// bytes, exactly as for keep-quant. Returns the K (in-features) dim, or -1.
int64_t KeepF16KDim(GgufTensorRole role, const std::vector<int64_t>& shape) {
  switch (role) {
    case GgufTensorRole::kMatmulWeight:
    case GgufTensorRole::kEmbeddingTable:
      return shape.size() == 2 ? shape[1] : -1;
    case GgufTensorRole::kStackedExpertWeight:
      return shape.size() == 3 ? shape[2] : -1;
    case GgufTensorRole::kTransformedWeight:
    case GgufTensorRole::kConvWeight:
    case GgufTensorRole::kVector:
      return -1;
  }
  return -1;  // unreachable for a valid enumerator (see -Wswitch above)
}

bool EnvOn(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return false;
  return !(std::strcmp(v, "") == 0 || std::strcmp(v, "0") == 0 ||
           std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}

// Tri-state: unset -> `fallback`; "0"/"false"/"off"/"" -> false; else true.
bool EnvOnOr(const char* name, bool fallback) {
  return std::getenv(name) == nullptr ? fallback : EnvOn(name);
}

}  // namespace

bool GgufQuantComputeAvailable(vt::DeviceType dev) {
  return vt::OpRegistered(vt::OpId::kMatmulBTQuant, dev);
}

bool GgufNvfp4ComputeAvailable(vt::DeviceType dev) {
  return vt::OpRegistered(vt::OpId::kMatmulNvfp4, dev);
}

// #2406. See the header for why `host_repack_active` is a parameter and not a
// call. The device term is the whole fix: everything else here is the
// pre-existing expression, so on `dev == kCPU` this is character-for-character
// the decision the loader made before, and a CPU load is unchanged.
bool QuantRepackForDevice(bool keep_quant, bool cpu_ref,
                          bool host_repack_active, vt::DeviceType dev) {
  return keep_quant && !cpu_ref && host_repack_active &&
         dev == vt::DeviceType::kCPU;
}

const char* Name(GgufTensorRole role) {
  switch (role) {
    case GgufTensorRole::kMatmulWeight: return "matmul_weight";
    case GgufTensorRole::kStackedExpertWeight: return "stacked_expert_weight";
    case GgufTensorRole::kTransformedWeight: return "transformed_weight";
    case GgufTensorRole::kEmbeddingTable: return "embedding_table";
    case GgufTensorRole::kConvWeight: return "conv_weight";
    case GgufTensorRole::kVector: return "vector";
  }
  return "?";
}

const char* Name(GgufResidency residency) {
  switch (residency) {
    case GgufResidency::kExpandBf16: return "expand_bf16";
    case GgufResidency::kKeepQuant: return "keep_quant";
    case GgufResidency::kKeepF16: return "keep_f16";
    case GgufResidency::kNvfp4Fp4: return "nvfp4_fp4";
  }
  return "?";
}

// ggml type id 1 is F16 (IEEE half); see gguf_dequant.cpp case 1.
bool KeepF16DType(uint32_t ggml_type) { return ggml_type == 1; }

// ggml type id 40 is the NVFP4 fork extension; see gguf_dequant.cpp case 40.
bool KeepNvfp4DType(uint32_t ggml_type) { return ggml_type == 40; }

// Device-side keep-quant capability (review sweep on #523): the master
// boolean `GgufQuantComputeAvailable()` only says the OP is registered; a
// device's kernel set can be narrower than the CPU admission list, and on a
// discrete backend with no CPU fallback tier a format the device cannot
// execute must keep its pre-existing expand-bf16 residency -- flipping it to
// a keep-quant block throws at FORWARD time with the whole model resident.
// Per-device sets name what the registered kernels actually implement.
bool DeviceKeepQuantSupported(vt::DType dt, vt::DeviceType dev) {
  switch (dev) {
    case vt::DeviceType::kROCM:
      // src/vt/rocm/rocm_grouped_gemm.hip implements exactly these on both the
      // grouped and non-grouped arms; Q4_0/Q2_K/Q3_K/IQ2_*/IQ3_*/MXFP4 are
      // owed (recorded in .agents/specs/rocm-gg-keep-quant.md).
      return dt == vt::DType::kQ8_0 || dt == vt::DType::kQ4_K ||
             dt == vt::DType::kQ5_K || dt == vt::DType::kQ6_K;
    default:
      // CUDA falls back to the CPU kernel for anything it lacks
      // (cuda_quant_dot.cu:1841-1846); the CPU list IS the CPU capability.
      return true;
  }
}

// keep-f16 needs an f16-capable MatmulBT on the running device; the ROCm
// kernel accepts bf16/bf16 and f32/f32 only, so an F16 file weight must
// expand there rather than be kept and refused at first forward (same review).
bool DeviceKeepF16Supported(vt::DeviceType dev) {
  return dev != vt::DeviceType::kROCM;
}

// The GATHER's admission rule; see the header. A block dtype with a row decoder
// is enough — `HasQuantDotKernel` is not consulted, because a gather table is
// never the `x` side of a vec_dot.
bool KeepQuantGatherDType(uint32_t ggml_type, vt::DType* out) {
  vt::DType dt = vt::DType::kF32;
  if (!vt::BlockDTypeFromGgmlTypeId(ggml_type, &dt)) return false;
  if (vt::cpu::BlockToFloat(dt) == nullptr) return false;
  if (out != nullptr) *out = dt;
  return true;
}

// Device gate for the gather arm, asked of the OP/PROVIDER TABLE and not of a
// hand-kept device list.
//
// `vt::Embedding` routes a block-quantized table to `OpId::kEmbeddingQuant`
// (vt/ops.cpp), so "can this device decode a block row" is exactly "is that op
// registered here". That is the same shape `GgufQuantComputeAvailable()` above
// uses for the GEMM arm, and it is the shape this function was owed: naming a
// device here is what `scripts/check-device-leakage.py` refuses, because the
// GGUF loader is the device-agnostic layer and a per-device set written into it
// drifts from the kernels it claims to describe.
//
// Registered today by kCPU (cpu_ops.cpp, through `BlockToFloat`) and by the
// CUDA backend (cuda_ops.cu, through cuda_quant_dequant.cuh) -- named in prose
// rather than as the enumerator on purpose, because the leakage checker greps
// the token in comments too, and rightly so: a prose mention is how the next
// hand-kept device list starts. METAL, VULKAN, ROCM and
// TENSTORRENT register only `kEmbedding`, whose kernels each assert a float
// table by name, so they answer false here and keep their pre-existing
// expand-bf16 residency -- and their gather arms are owed.
//
// What this unlocks is not one op. On a device that cannot gather quantized,
// the Qwen3.8-Flash-Next n-gram table expands from 26.822 GiB of IQ4_NL to
// 95.368 GiB of bf16, against ~119.6 GiB of usable memory on the one box that
// fits the model at all -- so the device-resident quantized table is what makes
// a GPU arm possible, and llama.cpp's own n-gram path does NOT have this shape
// (#27742 pins that table to the CPU by tensor class).
bool DeviceQuantGatherSupported(vt::DeviceType dev) {
  return vt::OpRegistered(vt::OpId::kEmbeddingQuant, dev);
}

bool KeepQuantDType(uint32_t ggml_type, vt::DType* out) {
  vt::DType dt = vt::DType::kF32;
  if (!vt::BlockDTypeFromGgmlTypeId(ggml_type, &dt)) return false;
  // Q8_K is the K-quants' ACTIVATION encoding; it never appears as a file
  // weight type and has no vec_dot, so it is not keep-quant capable.
  // TQ1_0/TQ2_0 are Vulkan-native keep-quant: no CPU vec_dot, but the Vulkan
  // backend has native TQ vec_dot shaders, so they are keep-quant capable.
  if (dt == vt::DType::kTQ1_0 || dt == vt::DType::kTQ2_0) {
    if (out != nullptr) *out = dt;
    return true;
  }
  if (!vt::cpu::HasQuantDotKernel(dt)) return false;
  if (out != nullptr) *out = dt;
  return true;
}

GgufResidency RouteGgufTensor(bool keep_quant, bool keep_f16, bool nvfp4_fp4,
                              bool cpu_ref, GgufTensorRole role,
                              uint32_t ggml_type,
                              const std::vector<int64_t>& shape,
                              vt::DeviceType dev) {
  // The oracle switch wins over everything (spec gate 2).
  if (cpu_ref) return GgufResidency::kExpandBf16;

  // 0. Native fp4 residency for NVFP4. Checked first because it is the ONLY
  // keep outcome type 40 has: it is not a vt block dtype (so it can never be
  // kKeepQuant) and it is not F16 (so never kKeepF16). The eligible roles are
  // exactly the verbatim-bytes ones keep-quant uses, and the alignment rule is
  // the ggml block, 64 elements — a ragged K cannot be repacked block-wise.
  //
  // The GATHER role is excluded EXPLICITLY rather than by sharing
  // `KeepQuantKDim`'s answer. It shared it until W6a, when that function
  // started returning a real K for the embedding table, and the fp4 arm would
  // then have claimed a gather table it cannot serve: `kNvfp4Fp4` produces an
  // `Nvfp4Weight` operand PAIR for `vt::MatmulNvfp4`, not a block-typed tensor,
  // and no gather can read one. The exclusion is by ROLE and not by capability,
  // so a future NVFP4 gather has to say so here.
  if (nvfp4_fp4 && KeepNvfp4DType(ggml_type) &&
      role != GgufTensorRole::kEmbeddingTable) {
    const int64_t k = KeepQuantKDim(role, shape);
    if (k > 0 && k % 64 == 0) return GgufResidency::kNvfp4Fp4;
  }

  // 1. Keep-quant blocks (a block encoding in a verbatim GEMM/expert/gather
  // role). The GEMM roles and the GATHER role share the block-alignment rule and
  // the device gate but NOT the encoding rule: a GEMM weight needs a `vec_dot`,
  // a gather table needs only a row decoder, so the two ask different questions.
  if (keep_quant) {
    const int64_t k = KeepQuantKDim(role, shape);
    const bool gather = role == GgufTensorRole::kEmbeddingTable;
    vt::DType dt = vt::DType::kF32;
    // ggml_row_size's precondition: a row is a whole number of blocks. A weight
    // whose K is ragged cannot be dotted OR decoded block-wise, so it expands.
    // The device gate (review #523): a format the RUNNING device cannot execute
    // keeps its pre-existing expand-bf16 residency instead of flipping to a
    // keep-quant block that throws at forward time on a card with no CPU
    // fallback tier.
    const bool encoding_ok = gather ? KeepQuantGatherDType(ggml_type, &dt)
                                    : KeepQuantDType(ggml_type, &dt);
    if (k > 0 && encoding_ok && k % vt::BlockElems(dt) == 0 &&
        (gather ? DeviceQuantGatherSupported(dev)
                : DeviceKeepQuantSupported(dt, dev))) {
      return GgufResidency::kKeepQuant;
    }
  }

  // 2. Keep-f16 (an F16 file weight in a verbatim role, incl. the gather table).
  // Independent of keep_quant: F16 is not a block encoding, so a weight is never
  // eligible for both. No block-alignment constraint — f16 is per-element.
  if (keep_f16 && KeepF16DType(ggml_type) &&
      KeepF16KDim(role, shape) > 0) {
    return GgufResidency::kKeepF16;
  }

  return GgufResidency::kExpandBf16;
}

GgufLoadPolicy GgufLoadPolicy::FromEnv(vt::DeviceType dev) {
  GgufLoadPolicy p;
  // FIRST, because every device-dependent flag below reads it. It is the
  // ENGINE's resolved device, not `CurrentPlatform()`: see the field comment.
  p.device = dev;
  p.cpu_ref = EnvOn("VT_CPU_REF");
  // CIQ G4 flipped this default: keep-quant is ON wherever the running device
  // can execute the quantized GEMM. VT_GGUF_KEEP_QUANT is the two-way
  // override that survives the flip (=0 is the opt-out the spec promised).
  p.keep_quant = EnvOnOr("VT_GGUF_KEEP_QUANT", GgufQuantComputeAvailable(dev));
  // The orientation win rides the same availability condition, and the oracle
  // switch turns it off with everything else so VT_CPU_REF=1 reproduces the
  // historical load byte for byte.
  p.expand_nk = p.keep_quant && !p.cpu_ref;
  // L6 keep-f16 — now DEFAULT ON (L7, 2026-07-23). Keeps an F16 file weight
  // resident as F16 (mmap-borrowed) and computes on it natively, instead of the
  // load-time expansion to a bf16 anonymous buffer. Mirrors llama.cpp, which keeps
  // f16 weights resident and runs `ggml_vec_dot_f16` on them.
  //
  // L6 shipped this OPT-IN because it measured RSS-NEUTRAL (3.884 -> 3.832 GiB)
  // and prefill-regressive. L7's profile found WHY, and both objections are now
  // removed: (1) the "neutrality" was a q8_0 double-count — on the mmap arm the
  // repack COPY leaves the source q8_0 blocks resident in the still-alive mapping
  // (kept alive by the f16 borrows), so the q8_0 mass counted twice. Releasing
  // the repack source (OwnGgufQuantBlocks, port of llama.cpp unmap_fragment) drops
  // that 1.0 GiB, so keep-f16 now measures 3.834 -> 2.832 GiB = 1.01x llama.cpp
  // (2.798), a real weight-residency win, NOT neutral. (2) the prefill regression
  // was borrowed-weight first-touch faults landing in the timed prefill; the
  // load-time PrefaultBorrowedSpan (port of llama.cpp's mmap prefetch) faults them
  // off the timed path, restoring prefill (was 0.72x behind). Net: RSS parity,
  // greedy tokens byte-identical (native-f16 compute, md5 d235db1... unchanged).
  //
  // THE llama.cpp DENOMINATORS ABOVE ARE SUPERSEDED, and this is the one place
  // the contamination reaches a shipped DEFAULT rather than a document. They were
  // measured against 237ad9b96, our own local-only fork, 65 of this project's
  // performance commits past upstream tag b9827, six of them in ggml/src/ggml-cpu/
  // (fused gated_delta_net and a discriminated ssm_conv, both DEFAULT-ON, neither
  // of which stock upstream carries at that point). The CPU floor arm ran a qwen35
  // model whose CPU graph reaches exactly those ops. The oracle is now stock
  // b10451, and the re-take is owed under #1003.
  //
  // TWO CORRECTIONS TO WHAT THIS COMMENT USED TO SAY, both from the #1003 sweep:
  //
  // (a) THE PREFILL FIGURES HERE WERE UNSOURCED. This comment read "~205 t/s =
  //     1.16x AHEAD of pp128 176.6". NO recorded run produces either operand:
  //     `git grep -F 176.6` finds it in four other files, none of them a
  //     llama.cpp prefill number: a MoE microbenchmark mean in MICROSECONDS
  //     (benchmark-record.md), a 176.69 in an ours-vs-vLLM ledger row
  //     (parity-ledger.md), two LTX golden floats, and #1003's own index row.
  //     Do not grep the regex `176.6`: the `.` matches any character and buries
  //     these in ~360 hits. No arm of ours recorded 205 t/s. The owning record
  //     (gguf-keep-quant-loader.md:595, benchmark-record.md:10722) says 204 t/s
  //     against pp128 173.2 = 1.18x. The numbers are NOT reconciled here, because
  //     picking one would assert an attribution nobody measured; #1003 owes a
  //     single re-measured pair. Do not quote any of these ratios.
  //
  // (b) THIS DEFAULT IS NOT INDEPENDENT OF THAT FLOOR. An earlier pass claimed it
  //     was, on the grounds that the L7 acceptance is a same-binary OURS-vs-OURS
  //     A/B. That reads one row of a three-row table. The A/B trades ~10% of
  //     prefill (224 -> 204 t/s, TTFT 571 -> 625 ms) and ~1.4% of decode for
  //     1.05 GiB of peak RSS, and the recorded reason the PREFILL loss is
  //     acceptable is stated in llama.cpp's own terms: "comfortably above the
  //     competitor floor" (gguf-keep-quant-loader.md:595). That floor is the
  //     contaminated pp128 173.2. b10451 is 624 commits past b9827 and carries
  //     upstream's own fused_gdn, so the direction is NOT established: if a
  //     re-taken stock pp128 lands above 204 t/s, that clause fails and this
  //     default's only recorded justification for its prefill regression is gone.
  //     The RSS leg would still stand alone and may well suffice. Keep-f16 stays
  //     DEFAULT ON here because this is a record pass that measured nothing;
  //     revisiting it is QUANT-GGUF-KEEPQ-LOADER's decision and needs the re-take
  //     first. See .agents/specs/oracle-llamacpp-repin-stock.md, row 12.
  //
  // VT_GGUF_KEEP_F16=0 is the opt-out; rides expand_nk so it is CPU-only and off
  // under VT_CPU_REF regardless (the oracle load stays byte-identical).
  p.keep_f16 = EnvOnOr("VT_GGUF_KEEP_F16", p.expand_nk) && p.expand_nk &&
               DeviceKeepF16Supported(dev);
  // `QUANT-GGUF-NVFP4` column C. Same shape as the keep-quant default: ON
  // wherever the running device can execute the NVFP4 GEMM (CUDA today; a CPU
  // build keeps expanding, which is correct but unquantized), with
  // VT_GGUF_NVFP4_FP4=0 the same-binary opt-out and the oracle switch forcing it
  // off so VT_CPU_REF=1 still reproduces the historical bf16 load byte for byte.
  p.nvfp4_fp4 = EnvOnOr("VT_GGUF_NVFP4_FP4", GgufNvfp4ComputeAvailable(dev)) &&
                !p.cpu_ref;
  // Both of vLLM's NVFP4 modes are mirrored; W4A4 is the default because it is
  // what the sibling compressed-tensors container of the same model runs, and
  // the GGUFs carry the `<stem>.input_scale` activation sidecars it needs.
  p.nvfp4_w4a4 = EnvOnOr("VT_GGUF_NVFP4_W4A4", true) && p.nvfp4_fp4;
  // L5. Both ride the same availability condition as the residency they refine,
  // and both are forced off by the oracle switch, so VT_CPU_REF=1 keeps
  // reproducing the historical load byte for byte and allocation for allocation.
  // ENG-RESIDENCY-CONFIG (#1110): `VT_GGUF_MMAP` is now `--offload-config`'s
  // `vllm_cpp.mmap.enabled` as well, and ResolveGgufMmap holds the precedence —
  // env var > config > this availability default. It is the SOLE reader of the
  // variable, and it applies the same whole-value polarity `EnvOnOr` did, so an
  // environment-only run resolves byte-for-byte as before. `VT_CPU_REF` still wins
  // over both: the oracle switch is not a residency preference.
  p.mmap_residency = ResolveGgufMmap(p.keep_quant) && !p.cpu_ref;
  p.share_tied_head = EnvOnOr("VT_GGUF_SHARE_TIED_HEAD", p.expand_nk) && p.expand_nk;
  // GDN split-projection orientation. Rides expand_nk (so VT_CPU_REF=1
  // reproduces the historical transpose); VT_GGUF_GDN_NK=0 is the narrow
  // same-binary A/B that reverts only the GDN projections to [K, N].
  p.gdn_expand_nk = EnvOnOr("VT_GGUF_GDN_NK", p.expand_nk) && p.expand_nk;
  // CIQ G7 repack-at-load. Rides keep_quant AND the i8mm probe (which itself
  // honors VT_CPU_QUANT_REPACK=0), and is forced off by the oracle switch so
  // VT_CPU_REF=1 reproduces the historical load. No separate default env read:
  // QuantRepackActive() is the single source of the on/off decision, so the
  // loader and the kernel can never disagree about whether repack is live.
  //
  // AND ON THE RESOLVED DEVICE (#2406), which it did not read until now. This
  // was the ONE device-dependent flag in this function still deciding from a
  // pure host probe, while its sibling `elem_kn_repack` eight lines below has
  // always carried the same term for the same reason. `QuantRepackForDevice`
  // holds the rule; see its header comment for why the ISA probe is passed in
  // rather than called there.
  p.quant_repack = QuantRepackForDevice(p.keep_quant, p.cpu_ref,
                                        vt::cpu::QuantRepackActive(), dev);
  // KERNEL-GEMM-CPU-TILED lever 2, elementwise [N,K] -> [K,N] repack-at-load.
  // OPT-IN ONLY (default false) because the repacked bytes are transposed and
  // only the CPU MatmulBTKernel honours Tensor.elem_kn_repacked today; see the
  // field comment. Forced off by the oracle switch like every other load
  // transform, so VT_CPU_REF=1 still reproduces the historical load.
  // Gated on the CPU platform, not just the env var. Only the CPU
  // MatmulBTKernel honours Tensor::elem_kn_repacked, and a staged device would
  // upload the transposed bytes and read them as [N,K]. `dev` is the same
  // seam GgufQuantComputeAvailable(dev) above already uses to decide a load
  // transform, so the policy CAN see this and should not be left to a runtime
  // backstop alone (ResidentWeight also VT_CHECKs, belt and braces).
  p.elem_kn_repack = !p.cpu_ref && EnvOnOr("VT_CPU_ELEM_KN_REPACK", false) &&
                     dev == vt::DeviceType::kCPU;
  return p;
}

// THE DEVICE A RESIDENCY DECISION IS ACTUALLY ABOUT (#2516).
//
// `GgufLoadPolicy::device` is the ENGINE's resolved device, and its own field
// comment says every device-dependent decision in the struct reads it. Hybrid
// placement (#2023, #2314) introduced a SECOND, narrower answer for exactly one
// class of tensor: `ActiveMoePlacementPlan().DeviceForLayer(l)` is the device a
// layer's routed experts are COMPUTED on, and `RunMoePlaced` hands that device
// to `MoeBlock`. A keep-quant decision asks "does the device that will execute
// this GEMM have a `vec_dot` for this encoding", so for a PLACED routed-expert
// tower it is a question about the placement device, not about the engine.
//
// Measured, on GLM-5.3 `UD-IQ1_S` on `strix:gpu0`: `DeviceKeepQuantSupported`
// serves {Q8_0, Q4_K, Q5_K, Q6_K} on ROCm, so all 228 IQ1_S/IQ3_XXS/IQ2_XXS/
// IQ4_XS/Q2_K/Q3_K towers routed `kExpandBf16` and `LoadStackedExperts` refused
// the load by name -- for towers whose bytes never reach the GPU at all (that
// model reads a tower only through `GlmExpertSlice`, never through
// `ResidentWeight`) and which the installed plan had already sent to the CPU,
// whose `vec_dot` table covers every one of those six encodings.
//
// THIS IS #1136 AND #2406 ONE SEAM FURTHER ALONG. Both were the same shape: a
// residency decision resolved against a device other than the one that would
// read the bytes. The repair is the same shape too -- ask the device that will
// run it.
//
// INERT BY CONSTRUCTION, in four terms, and the last one is the load-bearing
// one. The plan is a process-global that a load with no placement never writes,
// and a DEFAULT-CONSTRUCTED plan's `engine_device()` is `kCPU`. Returning its
// answer unconditionally would therefore route a CUDA load's experts as though
// they were on the host. So the placed answer is ADOPTED only when the plan
// actively moves THIS tensor somewhere other than the device the plan itself was
// resolved for; every other path returns the policy's own device unchanged.
vt::DeviceType GgufLoadPolicy::ComputeDeviceFor(const std::string& name,
                                                GgufTensorRole role) const {
  if (role != GgufTensorRole::kStackedExpertWeight) return device;
  const MoePlacementPlan& plan = ActiveMoePlacementPlan();
  if (!plan.PlacesAnything()) return device;
  const vt::DeviceType placed = plan.DeviceForRoutedExpertTensor(name);
  if (placed == plan.engine_device()) return device;
  return placed;
}

GgufResidency GgufLoadPolicy::Route(const GgufTensorInfo& tensor,
                                    GgufTensorRole role) const {
  const GgufResidency r = RouteGgufTensor(
      keep_quant, keep_f16, nvfp4_fp4, cpu_ref, role, tensor.ggml_type,
      tensor.shape, ComputeDeviceFor(tensor.name, role));
  if (audit) audit(tensor.name, role, r);
  return r;
}

GgufLoadPolicy NoKeepQuant(const GgufLoadPolicy& policy) {
  GgufLoadPolicy p = policy;
  p.keep_quant = false;
  return p;
}

GgufResidency PeekRoute(const GgufLoadPolicy& policy, const GgufTensorInfo& tensor,
                        GgufTensorRole role) {
  // THROUGH THE SAME RESOLVER `Route` USES, not a copy of the expression. This
  // function exists so a caller can ask what the loader WILL do without
  // notifying the audit, and `GgufExpertTowersReachSlotLane` is one such caller
  // -- it peeks the very tensors the loader then routes. Two spellings of the
  // device term is exactly how the bound and the forward come to disagree about
  // one file, which is the defect #1378 records.
  return RouteGgufTensor(policy.keep_quant, policy.keep_f16, policy.nvfp4_fp4,
                         policy.cpu_ref, role, tensor.ggml_type, tensor.shape,
                         policy.ComputeDeviceFor(tensor.name, role));
}

}  // namespace vllm
