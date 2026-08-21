// vllm.cpp ORIGINAL — GGUF keep-quantized residency policy. See
// gguf_keep_quant.h for the upstream (llama.cpp @ 237ad9b96) anchors and the
// totality contract.
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

#include <cstdlib>
#include <cstring>

#include "vllm/config/weight_residency.h"
#include "vllm/platforms/interface.h"
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
    case GgufTensorRole::kTransformedWeight:
    case GgufTensorRole::kEmbeddingTable:
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

bool GgufQuantComputeAvailable() {
  return vt::OpRegistered(vt::OpId::kMatmulBTQuant,
                   vllm::platforms::CurrentPlatform().device_type());
}

bool GgufNvfp4ComputeAvailable() {
  return vt::OpRegistered(vt::OpId::kMatmulNvfp4,
                   vllm::platforms::CurrentPlatform().device_type());
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

bool KeepQuantDType(uint32_t ggml_type, vt::DType* out) {
  vt::DType dt = vt::DType::kF32;
  if (!vt::BlockDTypeFromGgmlTypeId(ggml_type, &dt)) return false;
  // Q8_K is the K-quants' ACTIVATION encoding; it never appears as a file
  // weight type and has no vec_dot, so it is not keep-quant capable.
  if (!vt::cpu::HasQuantDotKernel(dt)) return false;
  if (out != nullptr) *out = dt;
  return true;
}

GgufResidency RouteGgufTensor(bool keep_quant, bool keep_f16, bool nvfp4_fp4,
                              bool cpu_ref, GgufTensorRole role,
                              uint32_t ggml_type,
                              const std::vector<int64_t>& shape) {
  // The oracle switch wins over everything (spec gate 2).
  if (cpu_ref) return GgufResidency::kExpandBf16;

  // 0. Native fp4 residency for NVFP4. Checked first because it is the ONLY
  // keep outcome type 40 has: it is not a vt block dtype (so it can never be
  // kKeepQuant) and it is not F16 (so never kKeepF16). The eligible roles are
  // exactly the verbatim-bytes ones keep-quant uses, and the alignment rule is
  // the ggml block, 64 elements — a ragged K cannot be repacked block-wise.
  if (nvfp4_fp4 && KeepNvfp4DType(ggml_type)) {
    const int64_t k = KeepQuantKDim(role, shape);
    if (k > 0 && k % 64 == 0) return GgufResidency::kNvfp4Fp4;
  }

  // 1. Keep-quant blocks (a block encoding in a verbatim GEMM/expert role).
  if (keep_quant) {
    const int64_t k = KeepQuantKDim(role, shape);
    vt::DType dt = vt::DType::kF32;
    // ggml_row_size's precondition: a row is a whole number of blocks. A weight
    // whose K is ragged cannot be dotted block-wise, so it expands.
    if (k > 0 && KeepQuantDType(ggml_type, &dt) &&
        k % vt::BlockElems(dt) == 0) {
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

GgufLoadPolicy GgufLoadPolicy::FromEnv() {
  GgufLoadPolicy p;
  p.cpu_ref = EnvOn("VT_CPU_REF");
  // CIQ G4 flipped this default: keep-quant is ON wherever the running device
  // can execute the quantized GEMM. VT_GGUF_KEEP_QUANT is the two-way
  // override that survives the flip (=0 is the opt-out the spec promised).
  p.keep_quant = EnvOnOr("VT_GGUF_KEEP_QUANT", GgufQuantComputeAvailable());
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
  p.keep_f16 = EnvOnOr("VT_GGUF_KEEP_F16", p.expand_nk) && p.expand_nk;
  // `QUANT-GGUF-NVFP4` column C. Same shape as the keep-quant default: ON
  // wherever the running device can execute the NVFP4 GEMM (CUDA today; a CPU
  // build keeps expanding, which is correct but unquantized), with
  // VT_GGUF_NVFP4_FP4=0 the same-binary opt-out and the oracle switch forcing it
  // off so VT_CPU_REF=1 still reproduces the historical bf16 load byte for byte.
  p.nvfp4_fp4 = EnvOnOr("VT_GGUF_NVFP4_FP4", GgufNvfp4ComputeAvailable()) &&
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
  p.quant_repack = p.keep_quant && !p.cpu_ref && vt::cpu::QuantRepackActive();
  // KERNEL-GEMM-CPU-TILED lever 2, elementwise [N,K] -> [K,N] repack-at-load.
  // OPT-IN ONLY (default false) because the repacked bytes are transposed and
  // only the CPU MatmulBTKernel honours Tensor.elem_kn_repacked today; see the
  // field comment. Forced off by the oracle switch like every other load
  // transform, so VT_CPU_REF=1 still reproduces the historical load.
  // Gated on the CPU platform, not just the env var. Only the CPU
  // MatmulBTKernel honours Tensor::elem_kn_repacked, and a staged device would
  // upload the transposed bytes and read them as [N,K]. CurrentPlatform() is
  // the same seam GgufQuantComputeAvailable() above already uses to decide a
  // load transform, so the policy CAN see this and should not be left to a
  // runtime backstop alone (ResidentWeight also VT_CHECKs, belt and braces).
  p.elem_kn_repack = !p.cpu_ref && EnvOnOr("VT_CPU_ELEM_KN_REPACK", false) &&
                     vllm::platforms::CurrentPlatform().device_type() ==
                         vt::DeviceType::kCPU;
  return p;
}

GgufResidency GgufLoadPolicy::Route(const GgufTensorInfo& tensor,
                                    GgufTensorRole role) const {
  const GgufResidency r =
      RouteGgufTensor(keep_quant, keep_f16, nvfp4_fp4, cpu_ref, role,
                      tensor.ggml_type, tensor.shape);
  if (audit) audit(tensor.name, role, r);
  return r;
}

GgufResidency PeekRoute(const GgufLoadPolicy& policy, const GgufTensorInfo& tensor,
                        GgufTensorRole role) {
  return RouteGgufTensor(policy.keep_quant, policy.keep_f16, policy.nvfp4_fp4,
                         policy.cpu_ref, role, tensor.ggml_type, tensor.shape);
}

}  // namespace vllm
