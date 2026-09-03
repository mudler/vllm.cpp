// Vulkan backend — shared GLSL prelude for every compute shader in this
// directory (BACKEND-VULKAN, W0 skeleton).
//
// PORT PROVENANCE. vLLM has no Vulkan path at all, so per the standing
// mirror-upstream rule the port source is llama.cpp's Vulkan backend,
// `ggml/src/ggml-vulkan/` @ pin 237ad9b96. What comes from where:
//   * the DISPATCH SHAPE (one workgroup per row for reducing kernels, a
//     shared-memory halving tree reduction guarded by barrier(), one thread per
//     element for flat kernels) is llama.cpp
//     `vulkan-shaders/rms_norm.comp:37-83` and `vulkan-shaders/norm.comp:9-35`;
//   * the PER-ELEMENT MATH is ported 1:1 from OUR OWN CPU reference kernels,
//     which are the vLLM-parity goldens, exactly as the Metal skeleton did
//     (src/vt/metal/metal_msl.h). Cited per kernel in each .comp file.
//
// TWO DELIBERATE DIVERGENCES FROM llama.cpp, both numeric:
//   1. llama.cpp uses `inversesqrt()` (rms_norm.comp:86, norm.comp:39). GLSL
//      only requires ~2 ULP from `inversesqrt` and drivers implement it with the
//      hardware reciprocal-sqrt approximation. We use `1.0 / sqrt(x)` because
//      that is literally what the CPU reference computes
//      (src/vt/cpu/cpu_ops.cpp:243) and this backend is gated against it. This
//      is the Vulkan analogue of the Metal skeleton's MTLMathModeSafe pin.
//   2. llama.cpp's norm.comp:37-38 computes the variance as E[x^2]-E[x]^2 (one
//      pass, catastrophic cancellation possible). Our CPU LayerNorm is the
//      two-pass stable form (src/vt/cpu/cpu_layernorm.cpp:49-73) and we keep it.
//
// STORAGE MODEL — why every buffer is bound TWICE.
// `vt::Tensor::data` is a raw pointer and Slice/View hand out INTERIOR pointers,
// while Vulkan binds (buffer, offset) descriptors whose offset must respect
// `minStorageBufferOffsetAlignment` (llama.cpp asserts exactly this in
// `ggml_vk_tensor_subbuffer`, ggml-vulkan.cpp:7448-7451). Rather than inherit
// that alignment constraint we bind every buffer WHOLE at offset 0 and carry the
// element offset as a BYTE offset in the push constants, so any 2-byte-aligned
// interior pointer works.
// That in turn means one shader has to address the same memory as 32-bit words
// (f32) and as 16-bit words (f16/bf16). A single `uint[]` view would force a
// read-modify-write for 16-bit stores, and two threads writing the two halves of
// one 32-bit word would RACE and silently lose data. So each operand gets TWO
// descriptor bindings onto the SAME VkBuffer — a `uint32_t[]` view at 2*k and a
// `uint16_t[]` view at 2*k+1 — and each store goes through the view that matches
// the element width. No read-modify-write, no race. This needs
// VK_KHR_16bit_storage (`storageBuffer16BitAccess`), which the backend probes
// and refuses to register without (src/vt/vulkan/vulkan_context.cpp).

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

// Opt-in only: a module that does not define VT_SUBGROUP_REDUCE must NOT declare
// these, because declaring them makes glslang emit the GroupNonUniform*
// capabilities and a device without them cannot create a pipeline from the
// module at all. See § vt_tg_sum below.
#ifdef VT_SUBGROUP_REDUCE
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#endif

// Storage dtype codes. Mirror the three FLOAT entries of vt::DType; translated
// host-side by DtypeCode() in vulkan_ops.cpp — the shader never sees vt::DType.
#define VT_DT_F32  0u
#define VT_DT_F16  1u
#define VT_DT_BF16 2u

// Workgroup size, fixed for every kernel in this backend.
// 128 is the Vulkan-GUARANTEED minimum of `maxComputeWorkGroupInvocations`
// (and of maxComputeWorkGroupSize.x), so no capability probe or per-device
// pipeline specialization is needed and the same SPIR-V runs on GB10 and on
// llvmpipe. llama.cpp instead hardcodes 512 (rms_norm.comp:33) and 512/1024
// elsewhere, which is faster but not universally valid; W0 chooses portability.
// It is a power of two, which the halving tree reduction below requires.
//
// IT STAYS A #define, AND THAT IS A MEASURED DECISION — do not "improve" it into
// a specialization constant (VK-A1, 2026-08-06). The workgroup size is written
// down three times (here, each .comp's `local_size_x`, and kWorkgroupSize on the
// host, which computes the workgroup COUNT from it), which is exactly the shape
// a specialization constant is supposed to collapse. It cannot, here:
//
//   `layout(local_size_x_id = 0, ...)` makes glslang emit
//   `ExecutionMode LocalSize 1 1 1` plus the LEGACY `BuiltIn WorkgroupSize`
//   spec-constant vector. The modern `LocalSizeId` execution mode that would
//   carry the real value needs SPIR-V 1.2 with VK_KHR_maintenance4 (core in
//   Vulkan 1.3), and this backend targets vulkan1.1 on purpose. MEASURED
//   consequence on llvmpipe: the literal LocalSize 1 wins, every workgroup runs
//   ONE thread while the host still dispatches ceil(n/128) groups, and roughly
//   1/128 of each tensor is computed — cross-device NMSE went from ~1e-14 to
//   ~0.99 on ops as simple as kAdd.
//
// Making VT_TG a specialization constant WITHOUT specializing local_size_x is
// worse than leaving it alone: it advertises a host-settable knob that silently
// corrupts, since the shared-memory extent and the reduction tree would follow it
// while the actual workgroup size would not. The specialization machinery
// (vulkan_spirv.h spec_ids, GetPipeline) is still the right variant mechanism —
// it is simply for axes like dtype, quant format and coopmat tier, where the
// constant does not have to agree with the launch geometry.
//
// IT IS OVERRIDABLE AT COMPILE TIME, AND ONLY THAT WAY (VK-RMSNORM, 2026-08-09).
// A shader that wants a wider workgroup `#define VT_TG` BEFORE including this
// file and declares the SAME literal in its own `local_size_x`; the two are then
// baked together into one SPIR-V module, so the "host-settable knob that
// silently corrupts" failure above cannot occur — a mismatch is a different
// module, not a different launch of the same one. The host picks the module by
// NAME after probing maxComputeWorkGroupInvocations, so the 128-wide module
// stays the always-valid fallback on any conformant device. See
// vt_rms_norm_wide.comp.
#ifndef VT_TG
#define VT_TG 128u
#endif

// Operand k is declared by each shader as a PAIR of blocks onto the same
// VkBuffer, at bindings 2*k (uint32_t view) and 2*k+1 (uint16_t view):
//
//   layout(binding = 0) readonly buffer Xb32 { uint32_t v[]; } X32;
//   layout(binding = 1) readonly buffer Xb16 { uint16_t v[]; } X16;
//
// They are spelled out rather than generated by a macro because the GLSL
// preprocessor has NO token-pasting operator (`##` is explicitly unsupported,
// GLSL 4.60 § 3.4), so a `VT_DECL(k, X)` cannot build the block names. The
// accessors below therefore take BOTH instance names.

// --- bf16 codec, bit-identical to src/vt/dtype.cpp:224-233 -------------------
// BF16ToF32 is a pure 16-bit left shift; F32ToBF16 is round-to-nearest-EVEN with
// the NaN-quieting special case (truncate then force the mantissa MSB). Ported
// verbatim so a bf16 round trip through a Vulkan kernel rounds EXACTLY the way
// the CPU reference does — the cross-device harness gates this with memcmp, not
// NMSE (tests/vt/test_backend_cross_device.cpp).
float vt_bf16_to_f32(uint b) { return uintBitsToFloat(b << 16); }

uint vt_f32_to_bf16(float f) {
  uint u = floatBitsToUint(f);
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) {
    return ((u >> 16) | 0x0040u) & 0xFFFFu;  // nan: keep quiet, truncate
  }
  uint rounding = 0x7FFFu + ((u >> 16) & 1u);  // round to nearest even
  return ((u + rounding) >> 16) & 0xFFFFu;
}

// --- f16 codec, bit-identical to src/vt/dtype.cpp:176-220 --------------------
// Deliberately NOT GLSL's packHalf2x16/unpackHalf2x16: those delegate to the
// driver, whose subnormal handling and rounding are not contractually the same
// as vt::F32ToF16's. Transcribing our own codec keeps f16 storage on the same
// bit-exact footing as bf16.
float vt_f16_to_f32(uint h) {
  uint sign = (h & 0x8000u) << 16;
  uint expo = (h >> 10) & 0x1Fu;
  uint mant = h & 0x3FFu;
  if (expo == 0x1Fu) { return uintBitsToFloat(sign | 0x7F800000u | (mant << 13)); }
  if (expo == 0u) {
    if (mant == 0u) { return uintBitsToFloat(sign); }  // signed zero
    uint shift = 0u;                                   // subnormal: normalize
    while ((mant & 0x400u) == 0u) { mant <<= 1; shift += 1u; }
    mant &= 0x3FFu;
    return uintBitsToFloat(sign | ((113u - shift) << 23) | (mant << 13));
  }
  return uintBitsToFloat(sign | ((expo + 112u) << 23) | (mant << 13));
}

uint vt_f32_to_f16(float f) {
  uint u = floatBitsToUint(f);
  uint sign = (u >> 16) & 0x8000u;
  uint raw_exp = (u >> 23) & 0xFFu;
  int expo = int(raw_exp) - 127 + 15;
  uint mant = u & 0x7FFFFFu;
  if (raw_exp == 0xFFu) {  // inf/nan
    return sign | 0x7C00u | (mant != 0u ? (0x200u | (mant >> 13)) : 0u);
  }
  if (expo >= 0x1F) { return sign | 0x7C00u; }  // overflow -> inf
  if (expo <= 0) {
    if (expo < -10) { return sign; }  // underflow -> zero
    mant |= 0x800000u;                // subnormal
    uint shift = uint(14 - expo);
    uint half_ = mant >> shift;
    uint rem = mant & ((1u << shift) - 1u);
    uint mid = 1u << (shift - 1u);
    if (rem > mid || (rem == mid && (half_ & 1u) != 0u)) { half_ += 1u; }
    return sign | half_;
  }
  uint half_ = (uint(expo) << 10) | (mant >> 13);
  uint rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half_ & 1u) != 0u)) { half_ += 1u; }
  return sign | half_;
}

float vt_from16(uint w, uint dt) {
  return dt == VT_DT_F16 ? vt_f16_to_f32(w) : vt_bf16_to_f32(w);
}
uint vt_to16(float f, uint dt) {
  return dt == VT_DT_F16 ? vt_f32_to_f16(f) : vt_f32_to_bf16(f);
}

// --- dtype-erased element access, mirroring src/vt/cpu/cpu_ops.cpp:27-43
// LoadF32/StoreF32. `BOFF` is a BYTE offset into the whole buffer (see § STORAGE
// MODEL); `I` is an ELEMENT index from there. Reduced-width outputs round ON
// STORE, exactly once, like the CPU reference.
#define VT_LOAD(B32, B16, DT, BOFF, I)                            \
  ((DT) == VT_DT_F32 ? uintBitsToFloat(B32.v[((BOFF) >> 2) + (I)]) \
                     : vt_from16(uint(B16.v[((BOFF) >> 1) + (I)]), (DT)))

#define VT_STORE(B32, B16, DT, BOFF, I, V)                        \
  do {                                                            \
    if ((DT) == VT_DT_F32) {                                      \
      B32.v[((BOFF) >> 2) + (I)] = floatBitsToUint(V);            \
    } else {                                                      \
      B16.v[((BOFF) >> 1) + (I)] = uint16_t(vt_to16(V, (DT)));    \
    }                                                             \
  } while (false)

// silu/sigmoid in f32, matching src/vt/cpu/cpu_ops.cpp:1646 FSigmoid and the
// `gate / (1 + exp(-gate))` spelling of SiluAndMulKernel (cpu_ops.cpp:252-264).
float vt_sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

// Workgroup tree reduction over VT_TG lanes. Shape ported from llama.cpp
// `vulkan-shaders/rms_norm.comp:73-83` (shared array, halving loop, barrier()
// each step). VT_TG is a power of two so the halving is exact.
//
// THE LEADING BARRIER IS LOAD-BEARING and is our addition, not llama.cpp's:
// llama.cpp's reduction is inlined once per kernel, ours is a reusable function
// that vt_layer_norm calls TWICE (mean, then variance) and vt_fused_chain calls
// once per kRmsNorm step. Without it a thread racing ahead into the next call
// would overwrite smem[0] while a slower thread was still reading the previous
// call's result. The Metal skeleton shipped this bug briefly and it passed the
// tests anyway because small threadgroups run near-lockstep
// (.agents/specs/backend-fanout-metal-vulkan-xpu.md § W0 landed, defect 1) — so
// it is written deliberately here rather than rediscovered.
shared float vt_smem[VT_TG];

#ifdef VT_SUBGROUP_REDUCE
// SUBGROUP variant of the same reduction, selected by defining
// VT_SUBGROUP_REDUCE before including this file. Ported from llama.cpp
// `vulkan-shaders/rms_norm_partials.comp:41-47` @ 237ad9b96 (`subgroupAdd` over a
// per-lane partial, then a second pass over the per-subgroup results) and
// `vulkan-shaders/flash_attn_base.comp`'s use of `gl_NumSubgroups` /
// `subgroupElect` for the shared-memory hand-off. Both extensions are CORE in
// Vulkan 1.1, which this backend already requires, but the CAPABILITY still has
// to be present on the device — the host only creates a pipeline from a module
// containing these instructions when VkPhysicalDeviceSubgroupProperties reports
// ARITHMETIC and BASIC in the COMPUTE stage (vulkan_context.cpp
// § SUBGROUP REDUCTION).
//
// WHY IT IS FASTER HERE AND WAS A WASH IN THE GEMV. The halving tree costs
// log2(VT_TG) barriers, and a barrier's cost grows with the number of resident
// subgroups, so at VT_TG = 1024 it is 10 barriers across 32 subgroups. This
// collapses that to ONE barrier plus one `subgroupAdd`. In the GEMV the same
// substitution read as a 4/8 wash because the reduction was a negligible slice
// of a bandwidth-bound kernel; in a norm the reduction IS the kernel, so the
// earlier negative result does not transfer (.agents/ negative results are
// regime-dependent).
//
// NOTHING IS ASSUMED ABOUT THE SUBGROUP SIZE. `gl_NumSubgroups` is read from the
// shader rather than pinned host-side, so the same module is correct at
// subgroupSize 4 (llvmpipe) and 32 (GB10). VT_TG >= gl_NumSubgroups always, so
// vt_smem is large enough by construction.
float vt_tg_sum(uint tid, float v) {
  const float sg = subgroupAdd(v);
  barrier();  // load-bearing, see below: protects vt_smem across CALLS
  if (subgroupElect()) { vt_smem[gl_SubgroupID] = sg; }
  barrier();
  float total = 0.0;
  for (uint i = 0u; i < gl_NumSubgroups; ++i) { total += vt_smem[i]; }
  return total;
}
#else
float vt_tg_sum(uint tid, float v) {
  barrier();
  vt_smem[tid] = v;
  barrier();
  for (uint s = VT_TG / 2u; s > 0u; s >>= 1) {
    if (tid < s) { vt_smem[tid] += vt_smem[tid + s]; }
    barrier();
  }
  return vt_smem[0];
}
#endif
