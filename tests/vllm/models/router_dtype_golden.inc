// The MoE router gate's OUTPUT dtype, and one GEMM at that dtype. W4 of
// `.agents/specs/glm-dsa-latest-deepseek.md` §3.7, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// ─── PART 1: WHERE THE DTYPE ANSWERS COME FROM ───────────────────────────────
// They are not transcribed. `_get_moe_router_dtype`
// (`vllm/model_executor/models/deepseek_v2.py:123-133`) was EXTRACTED from the
// parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98` with
// `git show <pin>:<path> | sed -n '123,133p'`, sha256
// `c8ca0f6368f694bcc5a5c771e64bb32023741e789adce8e98fc5a99edafed5d7`, compiled
// and CALLED on each config below with torch 2.11.0+cu130 on 2026-08-30. The
// answers are upstream's own return values, so this fixture cannot agree with a
// transcription of the rule — there is no transcription. (The whole `vllm`
// package does not import in this environment: `vllm.utils.network_utils` needs
// `zmq`, which is not installed. The function is pure and depends only on
// `torch`, so executing its own bytes is the closest reachable oracle run and it
// is recorded here as exactly that.)
//
// THE ORDER OF THE TWO ARMS IS WHAT THIS GATES. `:127` returns f32 for
// `model_type == "glm_moe_dsa"` BEFORE `:131` looks at `moe_router_dtype`, so
// `glm_moe_dsa_router_bfloat16` is f32 — a rule written in the other order
// would return None there and pass every other case in this table.
//
// ─── PART 2: THE GEMM ────────────────────────────────────────────────────────
// `a` and `b` are small INTEGER matrices. Every entry is bf16-exact, and every
// partial sum of every dot product is an exact integer below 2^24, so the f32
// result is EXACT FOR ANY REDUCTION ORDER — there is no tolerance to widen and
// no accumulation-order artefact to hide behind. 13 of the 24 products are odd
// integers in (256, 512), which bf16 cannot represent, so a bf16 STORE is
// detectable rather than merely suspected. `f32_exact` is the answer a
// `DType::kF32` output buffer must reproduce bit for bit; `bf16_stored` is what
// the same GEMM produces through a `DType::kBF16` output, and the two differ.
// Cross-checked with torch 2.11.0+cu130.
//
// Regenerate: see the commit that added this file. Do not edit by hand.
#pragma once

#include <cstdint>

namespace router_dtype_fixture {

// One row per config handed to the pinned `_get_moe_router_dtype`.
struct RouterDtypeCase {
  const char* name;
  const char* model_type;      // nullptr == the key is ABSENT
  const char* moe_router_dtype;// nullptr == the key is ABSENT
  bool upstream_returns_f32;   // torch.float32 -> true; None -> false
};

inline constexpr RouterDtypeCase kRouterDtypeCases[] = {
    {"glm53_shipped", "glm_moe_dsa", "float32", true},
    {"glm_moe_dsa_no_moe_router_dtype", "glm_moe_dsa", nullptr, true},
    {"glm_moe_dsa_router_bfloat16", "glm_moe_dsa", "bfloat16", true},
    {"deepseek_v2_plain", "deepseek_v2", nullptr, false},
    {"deepseek_v3_plain", "deepseek_v3", nullptr, false},
    {"deepseek_v3_router_float32", "deepseek_v3", "float32", true},
    {"deepseek_v3_router_bfloat16", "deepseek_v3", "bfloat16", false},
    {"no_model_type_router_float32", nullptr, "float32", true},
};
inline constexpr int kNumRouterDtypeCases =
    static_cast<int>(sizeof(kRouterDtypeCases) / sizeof(kRouterDtypeCases[0]));

// ─── the GEMM ───────────────────────────────────────────────────────────────
inline constexpr int64_t kGemmT = 4;
inline constexpr int64_t kGemmH = 8;
inline constexpr int64_t kGemmE = 6;
inline constexpr float kGemmA[] = {0.0f, 3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 1.0f, 4.0f, 7.0f, 10.0f, 13.0f, 16.0f, 2.0f, 5.0f, 8.0f, 11.0f, 14.0f, 0.0f, 3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 1.0f, 4.0f, 7.0f, 10.0f, 13.0f, 16.0f, 2.0f, 5.0f, 8.0f};
inline constexpr float kGemmB[] = {-4.0f, 7.0f, -1.0f, 10.0f, 2.0f, 13.0f, 1.0f, 12.0f, 4.0f, -4.0f, 7.0f, -1.0f, 6.0f, -2.0f, 9.0f, 1.0f, 12.0f, 4.0f, 11.0f, 3.0f, 14.0f, 6.0f, -2.0f, 9.0f, -3.0f, 8.0f, 0.0f, 11.0f, 3.0f, 14.0f, 2.0f, 13.0f, 5.0f, -3.0f, 8.0f, 0.0f, 7.0f, -1.0f, 10.0f, 2.0f, 13.0f, 5.0f, 12.0f, 4.0f, -4.0f, 7.0f, -1.0f, 10.0f};
// EXACT for any reduction order; assert with ==, never a tolerance.
inline constexpr float kGemmF32Exact[] = {187.0f, 357.0f, 261.0f, 165.0f, 240.0f, 315.0f, 428.0f, 308.0f, 435.0f, 239.0f, 347.0f, 455.0f, 142.0f, 327.0f, 303.0f, 279.0f, 369.0f, 459.0f, 281.0f, 312.0f, 324.0f, 336.0f, 272.0f, 531.0f};
// The same products rounded to bf16. 13 of 24 differ from kGemmF32Exact.
inline constexpr float kGemmBf16Stored[] = {187.0f, 356.0f, 260.0f, 165.0f, 240.0f, 316.0f, 428.0f, 308.0f, 436.0f, 239.0f, 348.0f, 456.0f, 142.0f, 328.0f, 304.0f, 280.0f, 368.0f, 460.0f, 280.0f, 312.0f, 324.0f, 336.0f, 272.0f, 532.0f};

}  // namespace router_dtype_fixture
