// vllm.cpp original (in-memory weight container for the Qwen3.6-35B-A3B MoE
// gate model); no 1:1 upstream mirror — the pinned load path is
// AutoWeightsLoader over nn.Module params (models/qwen3_5.py @ e24d1b24), which
// this replaces with an explicit owned-tensor struct for the M0.9 forward.
//
// Loads the real 35B checkpoint (nvidia/Qwen3.6-35B-A3B-NVFP4) into owned host
// bf16 tensors. Quant schemes per weight class (verified against the real ckpt,
// .agents/specs/qwen36-forward-notes.md §6):
//   - MoE experts + shared_expert + lm_head : NVFP4 W4A16 (DequantNvfp4ToBf16)
//   - attention (q/k/v/o) + GDN (in_proj_qkv/z, out_proj) : per-tensor FP8
//     (DequantFp8ToBf16) — NOT bf16 as the task first assumed
//   - everything else (embeds, norms, router gate, conv1d, A_log/dt_bias,
//     in_proj_a/b) : bf16 (A_log/dt_bias upcast to f32)
//
// All 2-D projection weights are stored TRANSPOSED to vt::Matmul's B layout
// [in, out] (on-disk torch layout is [out, in]). Host checkpoint ownership stays
// per logical projection. On CUDA the production 27B path additionally builds
// resident packed gate_up and full-attention QKV operands, mirroring vLLM's
// MergedColumnParallelLinear/QKVParallelLinear topology at TP=1; diagnostic
// toggles retain the split residents.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/owned_bytes.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"  // MoE vision tower (#891)
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vt {
class Backend;
}  // namespace vt

namespace vllm {

// One owned, contiguous, host tensor: heap bytes + shape/dtype. View() builds a
// fresh vt::Tensor over the current buffer, so it stays valid across moves/
// reallocations of the owning struct (no cached raw pointer to dangle).
struct OwnedTensor {
  // Heap bytes, OR a read-only borrowed view (GGUF mmap / a bf16 expansion
  // shared with a second tensor) that carries its own keep-alive. See
  // owned_bytes.h; the read API is the std::vector<uint8_t> subset this tree
  // uses, so readers are unaffected by which residency a weight has.
  OwnedBytes bytes;
  vt::DType dtype = vt::DType::kF32;
  int rank = 0;
  int64_t shape[vt::kMaxRank] = {0, 0, 0, 0};

  // Matmul-weight orientation. false (default): Matmul-B [K=in, N=out] (the
  // loader transposed the disk tensor; row-major x row-major vt::Matmul).
  // true: raw torch Linear [N=out, K=in] as on disk (LoadBf16Raw) — consumed
  // via vt::MatmulBT, the cuBLASLt TN fast path (see ops.h MatmulBT).
  bool nk = false;

  // CIQ G7: the block-quant bytes were REPACKED at load into the CPU i8mm
  // interleave (q8_0 -> block_q8_0x4). Propagated to vt::Tensor::repacked by
  // View(); the CPU quant GEMM keys on it. Only set on an i8mm process, only for
  // repack-eligible q8_0 weights, and only on the copy residency (the transform
  // mutates the buffer, so it cannot ride the read-only mmap borrow).
  bool repacked = false;

  // Brick 4 (DeepSeek-V4 last-mile): the Q8_0 blocks were repacked at load into
  // the CUDA coalesced-load layout ([all qs | all scales], 16-byte-aligned qs).
  // Carried to vt::Tensor::q8_0_aligned by View(); the CUDA Q8_0 GEMM keys on it.
  // Same total bytes; set only on the owned (copy) residency (the transform
  // rewrites the buffer, so it cannot ride the read-only mmap borrow).
  bool q8_0_aligned = false;
  // KERNEL-GEMM-CPU-TILED lever 2: the [N,K] elementwise bytes were transposed
  // to [K,N] at load. Carried to vt::Tensor::elem_kn_repacked; shape stays
  // [N,K]. Opt-in (VT_CPU_ELEM_KN_REPACK=1) and CPU-only, see the flag comment.
  bool elem_kn_repacked = false;

  // A synchronized direct-device load may discard the host staging buffer while
  // retaining an authoritative d_dev/d_dev_f32 copy. Empty() is used as model
  // dispatch metadata, so host reclamation must not make a populated weight look
  // absent. Mutable because ReleaseHost is logically const, like lazy residency.
  mutable bool host_released = false;

  // ENG-LOAD-DIRECT-UPLOAD (issue #150). Non-null when `bytes` BORROWS a
  // read-only safetensors mmap taken verbatim from the checkpoint instead of
  // being copied into an owned buffer, and records the exact source range so
  // the windowed page release can run once a device copy exists. It is the
  // discriminator between the two borrow producers: this one may be adopted
  // onto the device allocation after upload, whereas a GGUF-mmap or
  // tied-expansion borrow must not be (see AdoptDeviceBytesAsHost).
  mutable const void* mmap_src = nullptr;
  mutable size_t mmap_src_bytes = 0;
  // Where those borrowed bytes physically live, for a consumer that wants to
  // READ them rather than fault them in: the owning shard's descriptor and the
  // byte offset of this weight within that shard. Set only on the mmap-borrow
  // path; -1 means "no descriptor, read through the mapping".
  //
  // Expert streaming is the consumer. Filling a slot by memcpy from the mapping
  // traps every 4 KiB page on the way, which is the whole reason W4 moved no
  // faster than the mmap path it replaced; a pread lands the slice in one call.
  mutable int mmap_fd = -1;
  mutable size_t mmap_file_offset = 0;

  // A process-unique identity for the BUFFER this tensor currently points at,
  // for a cache that outlives the model.
  //
  // READ THAT LITERALLY: the identity is keyed on `bytes.data()`, so it is an
  // identity for the address, not for the contents. Replacing a buffer's bytes
  // IN PLACE — same address, different weights — keeps the old uid, and the
  // cache would then serve the old entries for the new contents. Nothing does
  // that today: a tower's `bytes` is assigned once when the model loads and is
  // only ever replaced wholesale, which moves the address. This comment says
  // where the guarantee stops rather than rounding it up, because #1066 was
  // caused by a comment on this exact field that rounded it up (it claimed a
  // base pointer was a stable identity, which is true for one model's life and
  // false for the cache's). `test_qwen36_weights` pins both halves.
  //
  // The expert slot cache is a process-lifetime singleton keyed by (tower,
  // expert), and it used to derive the tower half from the buffer's ADDRESS.
  // That is stable for one model's life, which is what its comment claimed, but
  // the cache is not scoped to one model's life: free a model, load another, and
  // the allocator hands a new tower an address the old one used. Every entry the
  // cache still holds for that address is then served to a DIFFERENT model's
  // expert, silently, as a hit that moves no bytes. Measured on two synthetic
  // models in one process: 21 distinct addresses for 24 towers, and 20 of 222
  // slices returned another tower's bytes.
  //
  // A counter cannot collide because it never goes backwards, and re-stamping
  // when `bytes` moves keeps a COPY of a tensor from inheriting the original's
  // identity along with a different buffer. Assigned lazily, so a weight that is
  // never streamed never pays for one.
  mutable uint64_t tower_uid = 0;
  mutable const uint8_t* tower_uid_for = nullptr;

  // `tower_uid`, assigned on first use and re-assigned if `bytes` has moved.
  // Never returns 0, so a caller can treat 0 as "no identity".
  uint64_t TowerUid() const;

  bool Empty() const { return bytes.empty() && !host_released; }
  bool HasHostBytes() const { return !bytes.empty(); }
  int64_t Numel() const;
  // Contiguous view over the current buffer (host/CPU device).
  vt::Tensor View() const;

  // Return the host byte buffer to the OS once an authoritative device-resident
  // copy exists — the platform residency behavior
  // `residency_policy().release_host_weights_after_upload`
  // (platforms/interface.h; BACKEND-PLATFORM item 2). Logically const: the
  // tensor's VALUE is unchanged, only the now-dead host mirror is freed. This
  // mirrors the existing mutable lazy-device-upload residency design (d_dev/
  // d_packed above are populated on a const weight). swap-with-empty (not
  // clear()) guarantees the std::vector capacity is actually deallocated. After
  // release View()/bytes must not be read; shape/dtype metadata is retained and
  // Empty() remains false so device-resident dispatch continues to see it.
  void ReleaseHost() const;

  // Lazily-populated device-resident copies (CUDA forward only; null on host or
  // before first use). Uploaded ONCE and reused across every forward step so the
  // model's bf16/f32 weights (embed table, norms, attention/GDN projections,
  // router) stop re-uploading per op. d_dev holds the raw-dtype bytes; d_dev_f32
  // holds a bf16->f32 upcast (the CUDA norm/conv kernels want f32 when the
  // activation is f32). The shared_ptr deleter frees through the vt Backend.
  mutable std::shared_ptr<void> d_dev;
  mutable std::shared_ptr<void> d_dev_f32;
};

// ADOPT the device-resident copy AS the host buffer, where the backend says its
// allocations are host-addressable (vt::Backend::DeviceMemoryIsHostAddressable).
//
// THE DEFECT THIS CLOSES (BACKEND-VULKAN-LOADMEM). `ResidentWeight` uploads a
// weight and keeps `bytes` as well, so on Vulkan the model became resident
// TWICE. That is invisible on a discrete GPU -- the two copies are in different
// memories -- but GB10 is unified, so both come out of the same 119 GiB of
// system RAM. MEASURED on Qwen3-4B (7.6 GiB on disk): 8.622 GiB of Vulkan
// allocation and 16.392 GiB of process VmHWM, i.e. a whole second copy, and
// 17.1 GiB off MemAvailable. Extrapolated to the 27B (50.89 GiB) that is over
// 100 GiB, which is why loading it could take the machine down rather than fail.
// `src/vllm/platforms/vulkan.cpp` even REASONED that there is "exactly one copy
// of the bytes"; this is what makes that true.
//
// It is an ADOPTION, not a free: `bytes` is re-pointed at the device allocation
// (persistently mapped, host-coherent) and keeps it alive through `d_dev`, so
// every existing `.bytes` reader -- `ResidentWeightF32`'s upcast, the portable
// CPU reference tier, `Numel`/`View` -- reads the SAME bytes it read before,
// from the surviving copy. Nothing is dropped that anyone could still want, so
// unlike `ReleaseHost()` this needs no "is the device path committed" proof.
//
// A no-op unless the backend opts in, and a no-op on an already-BORROWED buffer
// (a GGUF mmap or a tied-weight expansion): those own no anonymous pages, and a
// tied pair must keep sharing one keep-alive.
//
// `VT_ADOPT_DEVICE_BYTES=0` is the same-binary A/B back to the two-copy
// behavior (house convention for a default-on residency change).
void AdoptDeviceBytesAsHost(vt::Backend& backend, const OwnedTensor& w);

// Lazily-built per-weight DEVICE-RESIDENT state, OWNED BY THE WEIGHT (issue
// #237).
//
// The forward paths build per-weight constants once — arrays of per-expert
// device pointers, Marlin repacks, token row maps — and reuse them on every
// subsequent step. That state used to live in `static` maps keyed on the ADDRESS
// of the weight, built on first touch and never erased, on the assumption that a
// process holds one engine for its lifetime.
//
// The assumption is false and it corrupts output. Destroy a `LoadedEngine` and
// build another in the same process, and the allocator can hand the new weights
// the address the old ones had; the new weights then inherit an entry marked
// ready whose device pointers were `cudaFree`d with the old engine. Nothing in
// this tree destroys the CUDA context, so those pointers stay *mapped* — they
// just now belong to whatever the new engine allocated there, which is how it
// surfaced: not a crash, but zeroed and corrupted output token ids, only in a
// test ordering that builds a second engine.
//
// Holding the state here ties it to the weights it describes, so it cannot
// outlive them and an address cannot be inherited. Deliberately opaque: the
// resident types are CUDA-path implementation details of the model .cpp files,
// and `shared_ptr<void>` keeps them out of this header while still running the
// correct destructor.
//
// This does NOT change the lifetime of the DEVICE allocations those types point
// at. They were leaked for the process before and are leaked now; freeing them
// is a separate question about backend teardown order, and widening this fix to
// touch that would put a shutdown-ordering hazard on the critical path of a
// correctness fix.
struct ResidentSlot {
  // Mutable because building the resident state is logically const: it is a
  // cache of what the weight already contains, populated on first use from a
  // const forward.
  mutable std::shared_ptr<void> state;
};

// Device-resident NVFP4 W4A16 weight (M2.2b). The modelopt packed fp4 codes +
// fp8-e4m3 group scales + per-tensor scale, kept RAW in the ORIGINAL torch
// [N=out_features, K=in_features] orientation vt::MatmulNvfp4 expects (NOT
// transposed to Matmul-B [in,out], and NOT dequanted to bf16). Keeping the
// ~22GB fp4 as fp4 avoids the M2.2-profile CPU dequant (~40 min) + the ~70GB
// bf16 host tensors. On the CUDA path the forward uploads packed+scale to the
// GPU ONCE (lazily, on first use — the mutable device handles below) and reads
// them in place across every step; on the host path it dequants for reference.
struct Nvfp4Weight {
  OwnedTensor packed;   // i8 [N, K/2]   two 4-bit E2M1 codes per byte
  OwnedTensor scale;    // i8 [N, K/16]  one fp8-e4m3 scale per 16-elem group
  float scale2 = 0.0F;  // per-tensor weight global scale (1/divisor), multiplied
  int64_t n = 0;        // out_features
  int64_t k = 0;        // in_features (K % 16 == 0)
  bool Empty() const { return packed.Empty(); }

  // Block-scale FORMAT. Default = NVFP4: group_size 16, fp8-e4m3 `scale`, a
  // per-tensor `scale2` global. is_mxfp4 selects compressed-tensors MXFP4
  // (`mxfp4-pack-quantized`): group_size 32, E8M0 (UE8M0) `scale` [N, K/32], NO
  // global (scale2 unused). Same E2M1 `packed`. Set ONLY by the dense MXFP4
  // loader; the 27B/35B NVFP4 gate paths never touch it (default false) so their
  // Marlin repack/GEMM are byte-unchanged.
  int group_size = 16;
  bool is_mxfp4 = false;

  // TRUE W4A4 fields (27B compressed-tensors NVFP4; notes §7). Populated ONLY on
  // the 27B CT load (LoadCtNvfp4Raw); left 0 for the 35B modelopt W4A16 weights
  // (which have no activation quant) so `IsTrueW4A4()` gates the 27B alone.
  //   input_global_scale_inv = the ON-DISK activation divisor (2688/amax_act),
  //     passed DIRECTLY to vt::ScaledFp4Quant.
  //   alpha = (1/input_divisor)·(1/weight_divisor) = scale2/input_global_scale_inv,
  //     the single scalar vt::MatmulNvfp4Fp4 multiplies the fp4xfp4 accumulator by.
  // Keep the original on-disk weight divisor as well as its reciprocal. Fused
  // logical linears (qkv/gate_up) take max(divisors) BEFORE reciprocating in
  // vLLM; reconstructing the divisor from scale2 can lose one float ULP.
  float weight_global_scale_inv = 0.0F;
  float input_global_scale_inv = 0.0F;
  float alpha = 0.0F;
  // True when the activation-quant globals were loaded (27B true-W4A4 path).
  bool IsTrueW4A4() const { return alpha > 0.0F; }

  // Lazily-populated device-resident copies (CUDA forward only; null on host or
  // before first use). The shared_ptr deleter frees through the vt Backend.
  mutable std::shared_ptr<void> d_packed;
  mutable std::shared_ptr<void> d_scale;
  // Lazily-populated SWIZZLED weight block scale for the cutlass sm120a fp4 GEMM
  // path (VT_NVFP4_CUTLASS): [round_up(n,128), round_up(k/16,4)] in the cutlass
  // atom layout, computed once from d_scale via vt::SwizzleBlockscale.
  mutable std::shared_ptr<void> d_scale_sw;
  // vLLM/FlashInfer-compatible model-owned f32 alpha for the true-W4A4 CUTLASS
  // path. Uploaded once from the persistent `alpha` member; the diagnostic host
  // scalar path leaves this null.
  mutable std::shared_ptr<void> d_alpha;
  // OPT-IN lifetime residency for the DEQUANTIZED bf16 [K=in, N=out] Matmul-B
  // operand the backends with NO fp4 GEMM multiply against (CPU / Vulkan / Metal /
  // HIP / Tenstorrent; CUDA never dequantizes). Default OFF, and it must stay a
  // per-WEIGHT opt-in: the operand is a bf16 expansion of ~4x the packed bytes, so
  // holding one per tower projection is the double-residency that OOM-reboots a
  // Spark on Vulkan (#203). The dense loader opts in the OUTPUT HEAD alone — one
  // weight, re-read whole every step (~2.54 GB a step rebuilt per call at the
  // 27B's 248320x5120); everything else keeps a per-call copy.
  bool keep_dequant_b = false;
  mutable std::shared_ptr<void> d_dequant_b;

  // Resident Marlin constants (issue #237; see ResidentSlot). `resident_marlin`
  // is this weight's own repack; `resident_marlin_pair` is the fused gate+up
  // repack, held on the GATE weight of the pair (it is the pair's cache key).
  ResidentSlot resident_marlin;
  ResidentSlot resident_marlin_pair;
};

// Device-resident per-tensor FP8 (W8A8) weight — the 35B attn q/k/v/o + GDN
// in_proj_qkv/z/out_proj projections (checkpoint: weight F8_E4M3 + f32
// weight_scale + f32 input_scale, activations W8A8 static per-tensor). Raw IEEE
// fp8-e4m3fn bytes kept in the ORIGINAL torch [N=out_features, K=in_features]
// orientation the cutlass W8A8 GEMM reads directly (NOT transposed to Matmul-B
// [in,out], NOT dequanted to bf16 — halves the projection's device memory vs the
// bf16 field and defers all scaling into the GEMM). On the CUDA path the forward
// uploads the bytes ONCE (lazily; the mutable device handle below) and reads them
// in place across every step. Populated only on the real 35B CUDA load with the
// cutlass W8A8 path enabled (LoadFp8Raw); the bf16 field it replaces is left
// EMPTY. `alpha = input_scale * weight_scale` is precomputed at load (both scales
// are per-tensor scalars) — the single fused scalar vt::MatmulFp8Cutlass applies.
struct Fp8Weight {
  OwnedTensor packed;         // i8 [N, K]  one fp8-e4m3fn byte per element
  float weight_scale = 0.0F;  // per-tensor weight_scale (dequant(w) = f8(w)*this)
  float input_scale = 0.0F;   // per-tensor activation scale (quant a = a/this)
  float alpha = 0.0F;         // input_scale * weight_scale (folded GEMM scalar)
  int64_t n = 0;              // out_features
  int64_t k = 0;              // in_features
  bool Empty() const { return packed.Empty(); }

  // Lazily-populated device-resident copy (CUDA forward only; null on host or
  // before first use). The shared_ptr deleter frees through the vt Backend.
  mutable std::shared_ptr<void> d_packed;
};

// Block-wise (fine-grained) FP8 weight — MODEL-FP8-BLOCK-WEIGHT, #1189 M3, spec
// `.agents/specs/model-fp8-block-weight.md`. One fp8-e4m3fn scale per
// `block_n` x `block_k` tile of the weight, the layout `Qwen/Qwen3.8-27B-FP8`
// ships and vLLM registers as `weight_scale_inv`
// (`vllm/model_executor/layers/quantization/fp8.py:378-379,511` @ `555967922`).
//
// A SIBLING of `Fp8Weight` above, deliberately, not an extension of it. That
// struct is three host floats whose whole point is the `alpha = input_scale *
// weight_scale` folded once at load; a block scheme has NO `input_scale` at all
// (the activation scheme is `dynamic`, and the target checkpoint ships zero such
// tensors) and its weight scale is a 2-D tensor, so there is no value `alpha`
// could take. Adding an optional tensor to `Fp8Weight` would make every existing
// reader of `alpha` — the cutlass and cuBLASLt fp8 wrappers, the merged-QKV alpha
// vector, `PrepareGdnFp8Resident` — carry a silent which-arm branch, and the one
// that forgets it returns a plausible number instead of an error. A distinct type
// makes the wrong call site fail to COMPILE. The shape mirrors `Nvfp4Weight`
// above: an OwnedTensor scale beside the packed values plus lazy device handles.
//
// `scale` is f32 and that is the MIRROR rather than a widening
// (`.agents/porting.md` §"Mirror the memory format"). Upstream allocates the
// parameter `torch.float32` (`utils/fp8_utils.py:1276,1283-1296`; `scale_dtype`
// is None unless `is_scale_e8m0`, which `Fp8Config` does not define) and loads
// the checkpoint tensor into it with `self.data.copy_()`
// (`vllm/model_executor/parameter.py:97`), a CONVERTING copy. A `BF16` tensor on
// disk is therefore widened to f32 once, at load, losslessly. `LoadFp8BlockRaw`
// switches on the on-disk dtype explicitly and has no default branch, because
// #1181 landed a guard for a reader that memcpy'd four bytes whatever the dtype
// was.
//
// `block_n`/`block_k` are carried ON the weight rather than looked up from the
// config at use time: the consumer needs them per GEMM, and a weight that knows
// its own geometry cannot be paired with the wrong one.
//
// NOTHING CONSUMES THIS YET. #1189 milestone M4 owns `Fp8BlockLinearMethod` and
// the forward wiring; `PrepareQwen3_5Dense` refuses a populated one by name so a
// block-wise checkpoint declines to run rather than running wrong.
struct Fp8BlockWeight {
  OwnedTensor packed;  // i8  [N, K]  one fp8-e4m3fn byte per element, verbatim
  OwnedTensor scale;   // f32 [cdiv(N, block_n), cdiv(K, block_k)]
  int64_t n = 0;       // out_features
  int64_t k = 0;       // in_features
  int64_t block_n = 0;
  int64_t block_k = 0;
  bool Empty() const { return packed.Empty(); }

  // Lazily-populated device-resident copies (CUDA forward only; null on host or
  // before first use). Declared here so M4/M5 upload through the same seam every
  // other quantized weight uses; owed and unpopulated at this merge commit.
  mutable std::shared_ptr<void> d_packed;
  mutable std::shared_ptr<void> d_scale;
};

// The N-concatenated device operand of a MERGED block-wise FP8 group —
// MODEL-FP8-BLOCK-MERGED (#1189 milestone M6, spec
// `.agents/specs/model-fp8-block-merged.md`).
//
// vLLM loads `gate_proj`/`up_proj` into ONE MergedColumnParallelLinear and
// `q`/`k`/`v` into ONE QKVParallelLinear, so one GEMM runs where this tree ran
// two and three. Unlike the per-tensor fp8 case beside it, no alpha vector is
// needed: a block scale is indexed by `n / block_n`, so the shard scale grids
// row-concatenate exactly when each shard's rows begin on a block boundary,
// which is upstream's own merged-partition rule (`fp8_utils.py:1229-1244`).
//
// Built lazily-once by `dense_fp8_block::ResidentFp8BlockMerged`, exactly like
// `Fp8BlockWeight::d_packed`. The per-shard residents are then never built, so
// the merged arm costs no duplicate device bytes. Empty on every non-block
// owner.
struct Fp8BlockMergedResident {
  mutable std::shared_ptr<void> d_packed;  // i8  [sum N_i, K]
  mutable std::shared_ptr<void> d_scale;   // f32 [sum cdiv(N_i,bn), cdiv(K,bk)]
};

// Gated-DeltaNet (linear_attention) layer weights. Projections in Matmul-B
// layout [in, out]; conv1d [conv_dim, K]; a_log/dt_bias f32 [Hv]; norm bf16.
struct GdnLayerWeights {
  OwnedTensor in_proj_qkv;    // bf16 [H, conv_dim]  (FP8 dequant + T)
  OwnedTensor in_proj_z;      // bf16 [H, value_dim] (FP8 dequant + T)
  OwnedTensor in_proj_b;      // bf16 [H, Hv]        (bf16 + T)
  OwnedTensor in_proj_a;      // bf16 [H, Hv]        (bf16 + T)
  // Qwen3.6-27B production owner for vLLM's MergedColumnParallelLinear
  // `in_proj_ba`: raw torch Linear orientation [2*Hv,H], rows [b,a], nk=true.
  // The real dense loader populates this and leaves in_proj_b/a empty. The
  // split rollback slices this owner; 35B/GGUF/synthetic paths retain the
  // legacy fields above and leave this empty.
  OwnedTensor in_proj_ba;
  // Qwen3.6-27B production owner for vLLM's MergedColumnParallelLinear
  // `in_proj_qkvz` (W2): raw torch Linear orientation [conv_dim+value_dim, H],
  // rows in exact [q,k,v,z] order (the checkpoint's in_proj_qkv already stacks
  // q|k|v per vLLM's stacked mapping (0,1,2) qwen3_5.py:203-207; in_proj_z
  // appends the z rows as shard 3), nk=true. The real dense loader populates
  // this and leaves in_proj_qkv/in_proj_z empty; the split rollback slices
  // this owner. 35B (FP8 qkv/z) / GGUF / synthetic paths retain the legacy
  // fields above and leave this empty.
  OwnedTensor in_proj_qkvz;
  OwnedTensor conv1d_weight;  // bf16 [conv_dim, K]  (bf16, NOT transposed)
  OwnedTensor a_log;          // f32  [Hv]
  OwnedTensor dt_bias;        // f32  [Hv]
  OwnedTensor norm_weight;    // bf16 [Dv]           (RMSNormGated)
  OwnedTensor out_proj;       // bf16 [value_dim, H] (FP8 dequant + T)

  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): block-wise FP8 GDN projections. The
  // target checkpoint lists the GDN small tensors under
  // `modules_to_not_convert`, so these stay empty for it; the rung exists
  // because the SITE probes `F8_E4M3` and a block-wise weight is one.
  Fp8BlockWeight in_proj_qkv_fp8_block;
  Fp8BlockWeight in_proj_z_fp8_block;
  Fp8BlockWeight out_proj_fp8_block;

  // 27B W4A4 fp4-resident variant of out_proj (compressed-tensors NVFP4, notes
  // §3.6). When populated (real 27B CUDA load) the forward calls vt::MatmulNvfp4
  // on it and out_proj above is left EMPTY; the 35B / synthetic loaders populate
  // the bf16 out_proj and leave this empty. Exactly one is filled.
  Nvfp4Weight out_proj_fp4;   // [N=H, K=value_dim]

  // 35B fp8-resident W8A8 variants (per-tensor FP8). Populated BY DEFAULT on the
  // real 35B CUDA+cutlass load (VT_DENSE_NATIVE); the bf16 in_proj_qkv/z/out_proj
  // above are then left EMPTY and the forward calls the native fp8 GEMM (cuBLASLt
  // fp8 by default, or cutlass fp8 under VT_DENSE_CUBLASLT_FP8=0). VT_DENSE_NATIVE
  // =0 flips back to the bf16 fields. The 27B (bf16 in_proj + fp4 out_proj) and
  // GGUF/synthetic (bf16) loaders leave these empty. Forward checks fp8, fp4, bf16.
  Fp8Weight in_proj_qkv_fp8;  // [N=conv_dim, K=H]
  Fp8Weight in_proj_z_fp8;    // [N=value_dim, K=H]
  Fp8Weight out_proj_fp8;     // [N=H, K=value_dim]

  // PERF-27B-GDN-FP8-QKVZ: the FP8 analogue of `in_proj_qkvz`. vLLM runs ONE
  // merged qkvz GEMM per GDN layer, so the two RAW fp8 shards above are
  // N-concatenated ONCE into a single device operand — i8 [conv_dim+value_dim,
  // H], qkv rows first — and the forward issues one fp8 GEMM instead of two.
  // Built lazily-once (and eagerly, PRE-CAPTURE, by
  // Qwen3_5DenseModel::PrepareGdnFp8Resident) exactly like Fp8Weight::d_packed;
  // the shard-local `d_packed` residents are then never built, so the merged
  // arm costs no duplicate device bytes. `d_qkvz_fp8_alpha` is the f32
  // [conv_dim+value_dim] per-output-column folded alpha and stays NULL in the
  // common case where both shards fold the SAME alpha (it is then folded into
  // the GEMM scalar instead). Empty on every non-fp8 owner.
  mutable std::shared_ptr<void> d_qkvz_fp8_packed;
  mutable std::shared_ptr<void> d_qkvz_fp8_alpha;
};

// Full (dense causal) attention layer weights.
struct FullAttnLayerWeights {
  OwnedTensor q_proj;   // bf16 [H, 2*Hq*Dh]  (FP8 dequant + T; output-gate doubled)
  OwnedTensor k_proj;   // bf16 [H, Hkv*Dh]
  OwnedTensor v_proj;   // bf16 [H, Hkv*Dh]
  OwnedTensor o_proj;   // bf16 [Hq*Dh, H]    (FP8 dequant + T)
  OwnedTensor q_norm;   // bf16 [Dh]
  OwnedTensor k_norm;   // bf16 [Dh]

  // 27B W4A4 fp4-resident variants of q/k/v/o_proj (compressed-tensors NVFP4,
  // notes §3.6). Populated on the real 27B CUDA load; the 35B / synthetic
  // loaders populate the bf16 fields above and leave these empty (exactly one
  // set filled). Each kept in the on-disk [N=out, K=in] orientation MatmulNvfp4
  // reads directly.
  Nvfp4Weight q_proj_fp4;  // [N=2*Hq*Dh, K=H]
  Nvfp4Weight k_proj_fp4;  // [N=Hkv*Dh,  K=H]
  Nvfp4Weight v_proj_fp4;  // [N=Hkv*Dh,  K=H]
  Nvfp4Weight o_proj_fp4;  // [N=H,       K=Hq*Dh]

  // CUDA resident for vLLM's QKVParallelLinear. The checkpoint owns logical
  // Q/K/V shards separately; production concatenates their packed rows and
  // linear block scales once, then keeps the combined packed operand and
  // combined swizzled scale resident. The split weights remain available for
  // VT_FP4_MERGED_QKV=0 and non-CUTLASS diagnostics.
  mutable std::shared_ptr<void> d_qkv_packed;
  mutable std::shared_ptr<void> d_qkv_scale_sw;
  // Merged QKV owns the max-before-reciprocal alpha as one physical device
  // scalar, matching its one physical projection. The host member is persistent
  // storage for the asynchronous H2D copy.
  mutable float qkv_alpha = 0.0F;
  mutable std::shared_ptr<void> d_qkv_alpha;

  // 35B fp8-resident W8A8 variants (per-tensor FP8). Populated BY DEFAULT on the
  // real 35B CUDA+cutlass load (VT_DENSE_NATIVE); the bf16 q/k/v/o_proj above are
  // then left EMPTY and the forward calls the native fp8 GEMM (cuBLASLt fp8 by
  // default, or cutlass fp8 under VT_DENSE_CUBLASLT_FP8=0). VT_DENSE_NATIVE=0
  // flips back to the bf16 fields. The 27B (fp4) and GGUF/synthetic (bf16)
  // loaders leave these empty. Forward checks fp8, then fp4, then bf16.
  Fp8Weight q_proj_fp8;  // [N=2*Hq*Dh, K=H]
  Fp8Weight k_proj_fp8;  // [N=Hkv*Dh,  K=H]
  Fp8Weight v_proj_fp8;  // [N=Hkv*Dh,  K=H]
  Fp8Weight o_proj_fp8;  // [N=H,       K=Hq*Dh]

  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): the block-wise (128x128) FP8 variants,
  // populated by the `weight_scale_inv` rung in `qwen3_5_dense_weights.cpp`
  // BEFORE the per-tensor rung, because a block-wise weight is also `F8_E4M3`
  // (#1166). The bf16, fp4 and per-tensor fp8 slots are left EMPTY when these
  // are populated, and vice versa. Nothing reads them yet -- M4 owns the linear
  // method and `PrepareQwen3_5Dense` refuses a populated one by name.
  Fp8BlockWeight q_proj_fp8_block;  // [N=2*Hq*Dh, K=H]
  Fp8BlockWeight k_proj_fp8_block;  // [N=Hkv*Dh,  K=H]
  Fp8BlockWeight v_proj_fp8_block;  // [N=Hkv*Dh,  K=H]
  Fp8BlockWeight o_proj_fp8_block;  // [N=H,       K=Hq*Dh]

  // CUDA resident for the FP8 (W8A8) analog of QKVParallelLinear (VT_FP8_MERGED
  // _QKV, opt-in). The checkpoint owns logical Q/K/V shards separately with a
  // shared per-tensor input_scale but per-projection weight_scale; production
  // N-concatenates their RAW fp8 bytes into one [Nq+Nk+Nv, K] operand kept
  // resident (below), runs ONE fp8 GEMM with alpha=1 (raw f32 accumulation), and
  // applies each output column's folded scalar (input_scale * that shard's
  // weight_scale) through the resident per-column alpha vector — byte-identical
  // to the three separate per-shard GEMMs when the GEMM accumulation matches.
  // The split fp8 weights above remain available for VT_FP8_MERGED_QKV=0.
  mutable std::shared_ptr<void> d_qkv_fp8_packed;  // i8 [Nq+Nk+Nv, K] raw e4m3fn
  mutable std::shared_ptr<void> d_qkv_fp8_alpha;   // f32 [Nq+Nk+Nv] per-column

  // MODEL-FP8-BLOCK-MERGED (#1189 M6): the BLOCK-wise analog of the resident
  // above, and a much smaller one. Block scales concatenate losslessly along N,
  // so there is no alpha vector and no shared-input_scale guard; the merged
  // operand is the three shards' bytes and scale grids end to end. Default ON
  // wherever the site exposes packed q/k/v views, because there is nothing to
  // trade off. Empty on every non-block owner.
  Fp8BlockMergedResident qkv_fp8_block_merged;
};

// Exact scalar processing for the three-shard CT NVFP4 QKVParallelLinear.
// Mirrors compressed_tensors_w4a4_nvfp4.py:95-138 and QKVParallelLinear's
// logical-shard loader: take each maximum divisor before reciprocating once.
struct FullAttnQkvGlobals {
  float input_global_scale_inv = 0.0F;  // max on-disk input divisor
  float weight_global_scale = 0.0F;     // reciprocal of max weight divisor
  float alpha = 0.0F;
};

FullAttnQkvGlobals MergeFullAttnQkvGlobals(const Nvfp4Weight& q,
                                           const Nvfp4Weight& k,
                                           const Nvfp4Weight& v);

// Sparse-MoE block (router + per-expert MLP + shared expert). Per-expert and
// shared projections are NVFP4-dequant'd and stored separately (gate/up/down),
// all in Matmul-B layout.
struct MoeBlockWeights {
  OwnedTensor router_gate;   // bf16 [H, E]  (bf16 + T)
  OwnedTensor shared_gate;   // bf16 [H, 1]  (bf16 + T)
  std::vector<OwnedTensor> expert_gate;  // E * bf16 [H, I]
  std::vector<OwnedTensor> expert_up;    // E * bf16 [H, I]
  std::vector<OwnedTensor> expert_down;  // E * bf16 [I, H]
  // A3 keep-quant grouped-MoE fold: when the GGUF experts stay block-quant (the
  // keep-quant arm of LoadExpertsT slices WITHOUT transpose), the whole stacked
  // [E*N, K] block tensor is loaded ONCE here (memory-neutral: replaces the E
  // per-expert `expert_*` copies, not additive) so the MoE forward can issue ONE
  // vt::MatmulBTQuantGrouped per {gate,up,down} instead of E per-expert matvecs
  // (mirror of laguna W9). EMPTY on fp4 / bf16-expand (those keep the vectors above).
  OwnedTensor expert_gate_kq;  // keep-quant stacked [E*I, H]
  OwnedTensor expert_up_kq;    // keep-quant stacked [E*I, H]
  OwnedTensor expert_down_kq;  // keep-quant stacked [E*H, I]
  OwnedTensor shared_gate_proj;  // bf16 [H, Is]
  OwnedTensor shared_up_proj;    // bf16 [H, Is]
  OwnedTensor shared_down_proj;  // bf16 [Is, H]

  // M2.2b fp4-resident variants of the NVFP4 expert/shared projections. When
  // populated (real-checkpoint CUDA load) the forward calls vt::MatmulNvfp4 on
  // these and the bf16 fields above are left EMPTY; the synthetic / GGUF loaders
  // populate the bf16 fields and leave these empty. Exactly one set is filled.
  std::vector<Nvfp4Weight> expert_gate_fp4;  // E * [N=I, K=H]
  std::vector<Nvfp4Weight> expert_up_fp4;    // E * [N=I, K=H]
  std::vector<Nvfp4Weight> expert_down_fp4;  // E * [N=H, K=I]
  Nvfp4Weight shared_gate_proj_fp4;  // [N=Is, K=H]
  Nvfp4Weight shared_up_proj_fp4;    // [N=Is, K=H]
  Nvfp4Weight shared_down_proj_fp4;  // [N=H, K=Is]

  // Resident MoE constants, one slot per forward path (issue #237; see
  // ResidentSlot). Exactly one is populated on a given engine — whichever path
  // this block's experts route through — and all three die with the block.
  ResidentSlot resident_fused;   // MoeFusedResident   (fp4 fused)
  ResidentSlot resident_bf16;    // MoeBf16Resident    (bf16 fast)
  ResidentSlot resident_marlin;  // MoeMarlinResident  (Marlin grouped)
};

// One decoder layer: input/post norms + one attention variant + the MoE block.
struct Qwen3_5MoeLayerWeights {
  bool is_linear_attention = false;
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  GdnLayerWeights gdn;                    // valid iff is_linear_attention
  FullAttnLayerWeights attn;             // valid iff !is_linear_attention
  MoeBlockWeights moe;                   // all 35B layers are MoE
};

// Whole-model weights.
struct Qwen3_5MoeWeights {
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (NOT transposed; embed lookup)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab]  (bf16/GGUF path; empty when fp4)
  Nvfp4Weight lm_head_fp4;   // [N=vocab, K=H]   (M2.2b fp4-resident; else empty)
  std::vector<Qwen3_5MoeLayerWeights> layers;

  // Deferred routed-expert host materialization — the 35B load-phase peak-PSS
  // interleave (ENG-EXPERT-STREAM / ENG-MOE-HOSTFREE follow-up). When set, the
  // per-layer `moe.expert_*_fp4` host copies are NOT loaded at Load time; instead
  // PrepareMarlinResident calls this per layer IMMEDIATELY BEFORE the device
  // Marlin build + host free, so at most ONE layer's routed-expert host copies
  // coexist (peak host residency ~ one layer, not all N layers ×256 experts). The
  // closure owns the mmap'd safetensors shards (its keepalive); resetting it after
  // the whole model is built returns them. Null on the GGUF / synthetic / Borrow
  // paths (experts already materialized eagerly at load). Non-marlin / non-CUDA /
  // VT_NVFP4_MARLIN=0 falls back to a bulk host materialization (correctness for
  // the wmma/CPU forward that reads these host bytes). `mutable`: materialization
  // is a logically const residency op, like the lazy device-upload handles above.
  mutable std::function<void(int64_t /*layer*/, MoeBlockWeights& /*moe*/)>
      load_layer_experts;
};

// Resolves a tensor name to its StTensor (across shards). Throws if absent.
using TensorResolver = std::function<const StTensor&(const std::string&)>;

// --- Backbone weight namespace (MODEL-TEXT-qwen3-5-*-for-causal-lm) -----------
//
// A Qwen3.5-family checkpoint publishes its text backbone under ONE of two
// spellings. The multimodal wrappers we already gate (Qwen3.6-27B / 35B-A3B /
// Coder) nest it under `model.language_model.`; the TEXT-ONLY arms
// (`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`, e.g.
// `Qwen/Qwen3.8-2.4T-A95B`) publish it flat under `model.`. The BACKBONE names
// are otherwise identical — same `mlp.shared_expert_gate.weight`, same
// top-level `lm_head`.
//
// THE PREFIX IS NOT THE ONLY THING BETWEEN THIS LOADER AND A PUBLISHED
// CHECKPOINT, and an earlier revision of this comment wrongly implied it was.
// The published Qwen3.5-family MoE repos ship 3-D STACKED routed experts
// (`...mlp.experts.gate_up_proj` / `.down_proj`) and carry no quantization
// scales at all, while `LoadQwen3_5Moe` reads ONLY per-expert NVFP4. That arm
// is OWED and is refused by name (`CheckMoeExpertLayoutSupported`,
// `qwen3_5_weights.cpp`). The DENSE loader is different: it routes BF16 vs FP8
// vs NVFP4 per projection by tensor presence, so it may genuinely read a flat
// bf16 checkpoint. Resolving the namespace is what THIS seam does; it is not a
// support claim for either published checkpoint.
//
// Upstream normalizes the two with ONE mapper —
//   WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
//   (vllm/model_executor/models/qwen3_5.py:296-300 @ `ad5d29db7`, PR #50210,
//    which is AHEAD OF our `555967922` parity pin and recorded as such) —
// so `model.` is canonical and the VL spelling is its accepted alias.
inline constexpr std::string_view kQwen3_5VlBackbonePrefix =
    "model.language_model.";
inline constexpr std::string_view kQwen3_5TextBackbonePrefix = "model.";

// Decides which of the two the checkpoint uses, ONCE, from the shard index, so
// every subsequent lookup in a load uses one namespace. Deliberately NOT a
// per-lookup fallback: a fallback would let a checkpoint bind half its tensors
// from one namespace and half from the other and still appear to load.
//
// Only BACKBONE spellings vote — `<prefix>embed_tokens.weight`,
// `<prefix>norm.weight` and `<prefix>layers.`. `model.visual.*` (the
// vision-inclusive 27B/35B towers) and the top-level `lm_head.*` / `mtp.*`
// therefore cast no vote, which is what keeps a vision checkpoint from looking
// like a flat text one.
//
// Throws std::runtime_error when BOTH namespaces carry backbone tensors (a
// mixed index is refused, never half-loaded) and when NEITHER does.
std::string ResolveQwen3_5BackbonePrefix(
    const std::vector<std::string>& tensor_names);

// --- ENG-LOAD-DIRECT-UPLOAD (issue #150) -------------------------------------
//
// THE DEFECT THIS CLOSES. Loading a checkpoint copies the weights TWICE: the
// loader `memcpy`s each tensor out of the read-only safetensors mmap into an
// owned anonymous buffer, and `ResidentWeight` later copies that buffer into a
// device allocation and drops it. For a 27B that is two full passes over tens
// of GiB where one would do, and the intermediate pass also costs the kernel a
// fresh anonymous page (and its zero-fill) for every page of the model.
//
// THE MECHANISM. For a tensor the device consumes VERBATIM, the owned buffer is
// never allocated at all: `bytes` borrows the mapping (OwnedBytes::Borrow, with
// StTensor::mapping as the keep-alive, so the mapping cannot be unmapped out
// from under it), and the device upload reads straight from the file mapping.
//
// WHAT QUALIFIES, BY CONSTRUCTION. Only a call site that would have performed a
// plain `memcpy` of the WHOLE source range into a freshly allocated destination
// of the SAME size may call this — no transpose, no dtype conversion, no
// dequant, no concatenation of several sources into one buffer, and no
// load-time repack (`repacked`/`q8_0_aligned`/`elem_kn_repacked`, which mutate
// the buffer and are set only on the GGUF path). The size identity is re-checked
// here (`numel(shape) * sizeof(dtype) == t.nbytes`) and the call FAILS CLOSED,
// returning false so the caller runs its normal copy, whenever anything does
// not line up. A reshape is fine: it changes no byte.
//
// Returns true when `o` was made to borrow (its dtype/rank/shape are then set
// from `dtype`/`shape` and its bytes are the mapping's); false when the caller
// must fall back to its existing copy. `VT_LOAD_DIRECT_UPLOAD=0` forces false
// (same-binary A/B, house convention).
bool BorrowStTensorBytes(OwnedTensor& o, const StTensor& t, vt::DType dtype,
                         const std::vector<int64_t>& shape);

// Process-cached gate behind `BorrowStTensorBytes`. Exposed so a test can assert
// which arm it is measuring.
bool LoadDirectUploadEnabled();

namespace detail {
// Test-only override of the direct-upload decision, bypassing the env cache so
// one test binary can exercise both arms. std::nullopt restores the default.
void SetLoadDirectUploadOverrideForTesting(std::optional<bool> value);
}  // namespace detail

// How a checkpoint spells its ROUTED experts (issue #740). Resolved ONCE per
// checkpoint from the shard index and threaded, exactly as the backbone
// namespace is: a per-lookup fallback would let one checkpoint bind half its
// experts from each layout and still appear to load.
//
//  kPerExpertNvfp4  `<layer>.mlp.experts.<e>.{gate,up,down}_proj.weight` U8 +
//                   `.weight_scale` F8_E4M3 + `.weight_scale_2` -- what an
//                   NVFP4 requant (nvidia/Qwen3.6-35B-A3B-NVFP4) ships and what
//                   every gated row reads today. Populates `expert_*_fp4`.
//  kStackedBf16     `<layer>.mlp.experts.{gate_up_proj,down_proj}`, ONE 3-D bf16
//                   tensor per projection holding every expert -- what the
//                   PUBLISHED repos (Qwen/Qwen3.8-2.4T-A95B,
//                   Qwen/Qwen3.6-35B-A3B) ship. Populates the bf16 `expert_*`.
enum class MoeExpertLayout { kPerExpertNvfp4, kStackedBf16 };

// Decides which of the two a checkpoint uses, ONCE, from its shard index —
// the routed-expert sibling of `ResolveQwen3_5BackbonePrefix`, and public for
// the same reason: a caller that wants to know what a published index implies
// must ask the same question the loader asks, not a paraphrase of it.
//
// Only names under `<backbone_prefix>layers.` vote. The top-level `mtp.` draft
// head carries the STACKED spelling even in `nvidia/Qwen3.6-35B-A3B-NVFP4`,
// whose model is per-expert NVFP4, so a scan that counted every `.mlp.experts.`
// name would flip that checkpoint's whole model onto the wrong arm.
//
// Throws when BOTH spellings appear under the backbone (a mixed index is
// refused, never half-bound). An index with NEITHER resolves to the per-expert
// arm — the status quo — and the load then fails at its first missing tensor.
MoeExpertLayout ResolveQwen3_5MoeExpertLayout(
    const std::vector<std::string>& tensor_names,
    const std::string& backbone_prefix);

// --- Everything that is NOT a routed expert (issue #864) ---------------------
//
// #740 gave the MoE loader the published repos' routed experts. It did not make
// a published repo LOAD: `Qwen/Qwen3.6-35B-A3B` and `Qwen/Qwen3.8-2.4T-A95B`
// carry ZERO `weight_scale` / `input_scale` / `scale_inv` tensors anywhere, and
// the loader additionally hard-required per-tensor FP8 for the GDN and
// attention towers and NVFP4 for the shared expert and `lm_head`. Those four
// components are what this enum and struct route by tensor PRESENCE.
//
// `DenseNativeEnabled()` does NOT cover this and is deliberately not widened:
// it switches fp8-RESIDENT against fp8-DEQUANT and BOTH of its branches assume
// an fp8 input, so it is a build/env A/B lever with recorded evidence attached,
// not a layout probe.
enum class MoeProjDtype {
  kBf16,   // `<proj>.weight` BF16, no scales -- what a published repo ships
  kFp8,    // `<proj>.weight` F8_E4M3 + `.weight_scale` (+ `.input_scale`)
  kNvfp4,  // ModelOpt `<proj>.weight` U8 + `.weight_scale` F8 + `.weight_scale_2`
};

const char* MoeProjDtypeName(MoeProjDtype dtype);

// The safetensors dtype string of `name` ("BF16", "F8_E4M3", "U8", ...), or an
// EMPTY string when the checkpoint index has no such tensor.
//
// One callback rather than the dense loader's `TensorResolver` + `TensorExists`
// pair, because presence is `!dtype_of(name).empty()` and the dense ladder needs
// exactly those two questions -- so a caller that has only an INDEX (a manifest,
// a plan audit) can answer both without materializing an `StTensor`.
using TensorDtypeProbe = std::function<std::string(const std::string&)>;

// THE DENSE ARM'S LADDER, AND IT MUST STAY THE DENSE ARM'S LADDER.
// `qwen3_5_dense_weights.cpp`'s `load_projection` (:475-484) asks, in order:
//   1. `IsNvfp4Projection`  = `has(<proj>.weight_packed) || has(<proj>.weight_scale_2)`
//      -- compressed-tensors spells the packed weight `weight_packed`; ModelOpt
//      spells the global scale `weight_scale_2`. Probing only one missed
//      `nvidia/Qwen3.6-27B-NVFP4` entirely (:342-356).
//   2. `get(<proj>.weight).dtype == "F8_E4M3"` -- the `modelopt_mixed` tower.
//   3. otherwise BF16.
// This is that ladder and nothing else. If the two ever disagree about one
// projection, a checkpoint could route differently through two loaders in the
// same build, so `test_qwen3_8_text_only` binds them by loading the SAME
// synthetic projection through the dense loader and comparing which slot it
// filled.
MoeProjDtype ClassifyQwen3_5Projection(const TensorDtypeProbe& dtype_of,
                                       const std::string& proj);

// True iff the checkpoint carries this projection at all, under either spelling
// (mirror of `DenseCheckpointHasLmHead`: a compressed-tensors projection's only
// weight tensor is `<proj>.weight_packed`).
bool Qwen3_5ProjectionPresent(const TensorDtypeProbe& dtype_of,
                              const std::string& proj);

// The four non-routed-expert components, each resolved ONCE per checkpoint and
// THREADED -- the same discipline `ResolveQwen3_5BackbonePrefix` and
// `ResolveQwen3_5MoeExpertLayout` follow, and for the same reason: a per-lookup
// probe would let one checkpoint bind some layers quantized and some bf16 and
// still appear to load.
//
// The defaults are the STATUS QUO before #864 -- per-tensor FP8 towers, NVFP4
// shared expert and head -- so every existing caller of the defaulted seams
// below is unchanged by construction.
//
// WHY FOUR INDEPENDENT DECISIONS AND NOT ONE. A checkpoint that is quantized in
// one component and bf16 in another is ordinary upstream, not a defect:
// `nvidia/Qwen3.6-27B-NVFP4` is `modelopt_mixed` (FP8 attention tower next to
// NVFP4 MLP and a BF16 GDN in-projection) and the dense arm reads it by asking
// per projection. Collapsing the four into one decision would refuse that shape
// and diverge from the ladder above, which is the one thing the spec's stop
// condition forbids. What IS refused is a component that disagrees with ITSELF
// -- see `ResolveQwen3_5MoeTowerDtypes`.
struct Qwen3_5MoeTowerDtypes {
  MoeProjDtype gdn = MoeProjDtype::kFp8;     // linear_attn in_proj_qkv/z, out_proj
  MoeProjDtype attn = MoeProjDtype::kFp8;    // self_attn q/k/v/o_proj
  MoeProjDtype shared_expert = MoeProjDtype::kNvfp4;  // mlp.shared_expert.*
  MoeProjDtype lm_head = MoeProjDtype::kNvfp4;
};

// Resolves all four from the checkpoint index, walking EVERY layer of
// `layer_types` (GDN projections on `linear_attention` layers, attention
// projections on `full_attention` ones, the shared expert on both) plus the
// top-level `lm_head`.
//
// Throws when one component disagrees with itself -- layer 0's `q_proj` BF16
// against layer 4's F8_E4M3, or `gate_proj` BF16 against `down_proj` NVFP4 --
// naming both projections and both dtypes. That is the "half from each" failure
// the once-per-checkpoint discipline exists to prevent, and unlike a missing
// tensor it would otherwise produce wrong logits rather than an error.
//
// A component with no projections present at all keeps its default, so an index
// that simply lacks a tensor still fails at the reader with `tensor not found`.
Qwen3_5MoeTowerDtypes ResolveQwen3_5MoeTowerDtypes(
    const TensorDtypeProbe& dtype_of, const std::string& backbone_prefix,
    const std::vector<std::string>& layer_types);

// --- The load PLAN (issue #740, .agents/specs/moe-bf16-stacked-experts.md) ---
//
// WHAT PROBLEM THIS SOLVES. `Qwen/Qwen3.8-2.4T-A95B` is ~4.8 TB over 213 shards.
// Nothing here can hold it, so "the reader works at 35B" is the only byte-level
// evidence available — and on its own it does not show that the 2.4T's OWN
// names, shapes and dtypes resolve, nor that the per-expert offset arithmetic
// survives its dimensions (one layer's `gate_up_proj` is 34,359,738,368 bytes,
// which overflows int32 by four orders of magnitude).
//
// So: walk the whole load for a config WITHOUT allocating or reading a single
// weight byte, and report every tensor it would fetch. Checked against the
// published `model.safetensors.index.json`, that answers "would it load on
// hardware that can hold it?" for exactly the part that can be answered without
// the hardware.
//
// WHAT IT DELIBERATELY DOES NOT CLAIM: a generated token, throughput, memory
// headroom, or that any allocation path survives at that scale.
//
// THE PLAN IS ONLY WORTH ANYTHING IF IT IS A PROJECTION OF THE LOADER RATHER
// THAN A SECOND MODEL OF IT. `PlanQwen3_5MoeLoad` therefore mirrors
// `LoadQwen3_5Moe` helper for helper, including the `DenseNativeEnabled()`
// decision that adds `.input_scale` to every FP8 projection, and the test suite
// binds the two: it builds a synthetic checkpoint from the plan ALONE, requires
// the production loader to read it, and then removes each planned tensor in turn
// and requires the load to fail naming exactly that tensor. A plan entry the
// loader does not want, or a tensor it wants that the plan omits, fails there.
struct PlannedTensor {
  std::string name;
  // The safetensors dtype string the load hard-requires (`LoadBf16Direct` wants
  // BF16, `LoadFp8Raw` F8_E4M3, `LoadNvfp4Raw` U8 + F8_E4M3 + F32, ...).
  std::string dtype;
  // The shape the CONFIG implies. EMPTY when neither the loader nor the config
  // determines it: the attention projections' output width depends on
  // `attn_output_gate`, which `HfConfig` does not carry, so this planner states
  // no shape for them rather than a plausible one that is wrong on every real
  // checkpoint (the 2.4T's `q_proj` is [32768, 8192], twice heads*head_dim).
  std::vector<int64_t> shape;
  // True when the LOADER ITSELF checks this shape against config, rather than
  // reading it off the header. Only the 3-D stacked routed experts do — which is
  // exactly the arithmetic this row added, so it is the one shape whose
  // agreement with the published index is a statement about the reader and not
  // about this planner.
  bool shape_enforced = false;
};

// Every tensor `LoadQwen3_5Moe` would fetch for `config`, in load order, without
// touching a weight byte. `backbone_prefix` and `layout` are what
// `ResolveQwen3_5BackbonePrefix` / `ResolveQwen3_5MoeExpertLayout` resolved from
// the index. Does NOT include `mtp.*`: `LoadQwen3_5MTP` loads that optional
// draft head separately, and only when speculative decoding is enabled.
//
// Identical for the eager and DEFERRED (`load_layer_experts`) residency paths —
// deferring changes WHEN the routed experts are read, never WHICH.
//
// `tower` is what `ResolveQwen3_5MoeTowerDtypes` resolved for the four
// non-routed-expert components; it changes the REQUEST SET (an FP8 projection
// asks for two or three tensors where a BF16 one asks for a single `.weight`),
// so a plan built with the wrong one is not a projection of the loader.
std::vector<PlannedTensor> PlanQwen3_5MoeLoad(
    const HfConfig& config, const std::string& backbone_prefix,
    MoeExpertLayout layout, Qwen3_5MoeTowerDtypes tower = {});

// Load one decoder layer's weights from real tensors. `layer_type` is
// "linear_attention" or "full_attention"; `num_experts` drives the expert loop.
// Exercised on real data by the Task 3 unit test (both layer types live in
// shard 1). Prefix is "{backbone_prefix}layers.{layer_idx}.", and the default
// is the VL spelling every checkpoint we gate today uses, so this seam is
// byte-identical for the 27B/35B/Coder callers. `LoadQwen3_5Moe` passes the
// prefix it resolved once from the shard index.
//
// `layout` likewise defaults to the arm every gated caller uses, so this seam
// stays byte-identical for them. `hidden` is read ONLY by the stacked arm, which
// needs it to resolve the 3-D tensors' orientation the way upstream does; it is
// unused (and may be 0) for the per-expert arm.
//
// `tower` likewise defaults to the arm every gated caller reads (FP8 towers,
// NVFP4 shared expert), so this seam stays byte-identical for them.
Qwen3_5MoeLayerWeights LoadQwen3_5MoeLayer(
    const TensorResolver& get, const std::string& layer_type, int64_t layer_idx,
    int64_t num_experts,
    const std::string& backbone_prefix = std::string(kQwen3_5VlBackbonePrefix),
    MoeExpertLayout layout = MoeExpertLayout::kPerExpertNvfp4,
    int64_t hidden = 0, Qwen3_5MoeTowerDtypes tower = {});

// Full-model load: resolves every param across the given shards (name -> shard
// looked up from each file's own header), dequantizes/transposes, and returns
// owned host bf16 tensors. Uses config.num_hidden_layers, config.layer_types,
// config.num_experts.
//
// `shards_owner` (optional): when non-null it SHARES ownership of `shards`
// (`shards_owner.get() == &shards`), which lets the loader DEFER the routed-MoE
// experts' host copies — the returned weights carry a `load_layer_experts`
// closure (holding `shards_owner` as its keepalive) that PrepareMarlinResident
// drives per layer to bound peak host residency (35B load-phase peak-PSS lever).
// When null the experts are loaded EAGERLY (the mmap'd shards may then be
// released as soon as this returns) — used by GGUF/synthetic/borrowed paths and
// any caller that will drop the shards before Prepare.
Qwen3_5MoeWeights LoadQwen3_5Moe(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    std::shared_ptr<const std::vector<SafetensorsFile>> shards_owner = nullptr);

// ── The MoE arm's VISION TOWER (issue #891, .agents/specs/moe-vision-tower.md) ─
//
// `LoadQwen3_5Moe` above reads the TEXT backbone only. `Qwen/Qwen3.6-35B-A3B`
// ships 333 `model.visual.*` tensors alongside it, and until this seam existed
// they were simply not read — the load succeeded and produced a text-only model,
// which is the silent-drop failure class this project keeps rediscovering.
//
// Upstream composes the SAME tower on both arms: `Qwen3_5MoeForConditionalGener
// ation` and `Qwen3_5ForConditionalGeneration` each hold a `Qwen3_VisionTransfor
// mer` (pinned vLLM `qwen3_5.py`), so this is the SHARED
// `LoadQwen3VLVisionWeights` reader (`qwen3_vl.h`) the dense 27B arm already
// gates at image 32/32 + video 32/32 — deliberately NOT a second tower.

// The tower geometry for a Qwen3.6 conditional-generation checkpoint. Mirrors
// the checkpoint's `config.json::vision_config`; `Qwen/Qwen3.6-35B-A3B` and
// `Qwen/Qwen3.6-27B` publish the SAME tower (depth 27, hidden 1152, heads 16,
// intermediate 4304, patch 16, temporal patch 2, spatial merge 2, 2304 position
// embeddings, EMPTY `deepstack_visual_indexes`) and differ only in
// `out_hidden_size`, which is the text backbone's hidden size (2048 on the 35B
// MoE, 5120 on the 27B dense) because the merger writes straight into the text
// residual stream. That one field is therefore taken from `config.hidden_size`
// rather than hardcoded. `HfConfig` does not parse `vision_config` (the dense
// arm's gate hardcodes the same numbers in-test).
//
// NO DeepStack: `deepstack_visual_indexes: []` compiles that path out for this
// family upstream (`qwen3_vl.py:1709-1716`), so the tower output is exactly
// [N, out_hidden_size].
multimodal::Qwen3VLVisionConfig Qwen3_5MoeVisionConfig(const HfConfig& config);

// Load the MoE arm's vision tower from the SAME shards the text backbone came
// from, through the shared `LoadQwen3VLVisionWeights`.
//
// REFUSES BY NAME when the checkpoint carries no `model.visual.*` tensor at all:
// a Qwen3.5-family `*ForConditionalGeneration` checkpoint without a tower cannot
// answer an image or video prompt, and returning an empty tower would let it
// answer from text alone and still emit plausible tokens. AGENTS.md: an arm that
// is not implemented "is refused with a message naming the missing piece ...
// never left to be discovered later". `nvidia/Qwen3.6-35B-A3B-NVFP4` is exactly
// such a checkpoint (`vision_config` declared, `visual.*` weights absent).
multimodal::Qwen3VLVisionWeights LoadQwen3_5MoeVision(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// True iff any shard carries a `model.visual.` tensor. Exposed so a caller can
// route to the text-only path deliberately instead of discovering the refusal.
bool HasQwen3_5MoeVisionTower(const std::vector<SafetensorsFile>& shards);

}  // namespace vllm
