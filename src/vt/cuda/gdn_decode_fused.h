// vllm.cpp original. CUDA-free selector and launch-geometry contract for the
// fused single-token GDN recurrence implemented in cuda_gdn.cu.
#ifndef VT_CUDA_GDN_DECODE_FUSED_H_
#define VT_CUDA_GDN_DECODE_FUSED_H_

#include <cstddef>
#include <cstdint>

namespace vt::cuda {

enum class GdnDecodeValueTile : uint8_t {
  kBv16 = 16,
  kBv32 = 32,
};

// BV16 is an exact opt-in discriminator. Every spelling except exactly "16"
// preserves the shipped BV32 schedule, including removed sweep values.
inline constexpr GdnDecodeValueTile GdnDecodeValueTileFromEnv(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1' && env_value[1] == '6' &&
                 env_value[2] == '\0'
             ? GdnDecodeValueTile::kBv16
             : GdnDecodeValueTile::kBv32;
}

// The shared-bank swizzle is an exact opt-in. In particular, numeric prefixes,
// whitespace, and truthy words must leave the incumbent layout unchanged.
inline constexpr bool GdnDecodeSwizzleFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1' && env_value[1] == '\0';
}

// Register-resident recurrence state is an exact opt-in. Selection is further
// constrained by GdnDecodeLaunchContractFor to the production BV16+SWIZZLE
// geometry; the parser alone never widens eligibility.
inline constexpr bool GdnDecodeRegstateFlagIsOn(const char* env_value) {
  return env_value != nullptr && env_value[0] == '1' && env_value[1] == '\0';
}

inline constexpr int64_t GdnDecodeStateStride(bool swizzled, int64_t dk,
                                              int lanes_per_row) {
  return dk + (swizzled ? lanes_per_row : 1);
}

// Maps a logical Dk column to its shared-memory column. Production only selects
// the swizzled arm for Dk=128/NW=8, where ck=16.
inline constexpr int64_t GdnDecodeSharedColumn(bool swizzled, int64_t logical_column,
                                               int64_t dk, int lanes_per_row) {
  if (!swizzled) return logical_column;
  const int64_t columns_per_lane = dk / lanes_per_row;
  return (logical_column % columns_per_lane) * lanes_per_row +
         logical_column / columns_per_lane;
}

inline constexpr int64_t GdnDecodeRegisterLogicalColumn(int lane, int slot) {
  return static_cast<int64_t>(lane) * 16 + slot;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
inline constexpr int64_t GdnDecodeRegisterSharedColumn(int lane, int slot) {
  return static_cast<int64_t>(slot) * 8 + lane;
}

struct GdnDecodeLaunchContract {
  GdnDecodeValueTile selected_tile;
  int64_t value_tile;
  int64_t value_tiles;
  int lanes_per_row;
  int64_t block_threads;
  size_t shared_bytes;
  bool should_launch;
  bool swizzled;
  bool regstate;
};

inline constexpr GdnDecodeLaunchContract GdnDecodeLaunchContractFor(
    const char* bv_env_value, const char* swizzle_env_value,
    const char* regstate_env_value, int64_t dv, int64_t dk,
    int requested_nw) {
  const GdnDecodeValueTile selected = GdnDecodeValueTileFromEnv(bv_env_value);
  if (dv <= 0 || dk <= 0) {
    return GdnDecodeLaunchContract{selected, 0, 0, 0, 0, 0, false, false,
                                   false};
  }
  const int64_t requested_bv = static_cast<int64_t>(selected);
  const int64_t bv = dv < requested_bv ? dv : requested_bv;
  const int64_t nv = (dv + bv - 1) / bv;
  const int nw_eff = dv >= 32 ? requested_nw : 1;
  const int64_t block_threads = bv * nw_eff;
  const bool swizzled = GdnDecodeSwizzleFlagIsOn(swizzle_env_value) &&
                         selected == GdnDecodeValueTile::kBv16 && dv == 128 &&
                         dk == 128 && nw_eff == 8;
  const bool regstate = GdnDecodeRegstateFlagIsOn(regstate_env_value) &&
                        swizzled;
  const int64_t state_stride = GdnDecodeStateStride(swizzled, dk, nw_eff);
  const size_t shared_bytes =
      (2 * static_cast<size_t>(dk) +
       static_cast<size_t>(bv) * static_cast<size_t>(state_stride)) *
      sizeof(float);
  return GdnDecodeLaunchContract{selected, bv, nv, nw_eff, block_threads,
                                 shared_bytes, true, swizzled, regstate};
}

inline constexpr GdnDecodeLaunchContract GdnDecodeLaunchContractFor(
    const char* bv_env_value, const char* swizzle_env_value, int64_t dv,
    int64_t dk, int requested_nw) {
  return GdnDecodeLaunchContractFor(bv_env_value, swizzle_env_value, nullptr,
                                    dv, dk, requested_nw);
}

inline constexpr GdnDecodeLaunchContract GdnDecodeLaunchContractFor(
    const char* bv_env_value, int64_t dv, int64_t dk, int requested_nw) {
  return GdnDecodeLaunchContractFor(bv_env_value, nullptr, nullptr, dv, dk,
                                    requested_nw);
}

// Storage callback dispatcher used by production and portable tests. The
// resolved contract selects exactly one compile-time kernel specialization.
template <typename SharedLaunch, typename RegisterLaunch>
inline decltype(auto) DispatchGdnDecodeStateStorage(
    const GdnDecodeLaunchContract& contract, SharedLaunch&& shared_launch,
    RegisterLaunch&& register_launch) {
  if (contract.regstate) return register_launch(contract);
  return shared_launch(contract);
}

// Shared callback dispatcher used by production and portable tests. The
// callbacks receive fully resolved geometry and exactly one is invoked.
template <typename Bv16Launch, typename Bv32Launch>
inline decltype(auto) DispatchGdnDecodeValueTile(const char* env_value, int64_t dv,
                                                 int64_t dk, int requested_nw,
                                                 Bv16Launch&& bv16_launch,
                                                 Bv32Launch&& bv32_launch) {
  const GdnDecodeLaunchContract contract =
      GdnDecodeLaunchContractFor(env_value, dv, dk, requested_nw);
  if (contract.selected_tile == GdnDecodeValueTile::kBv16) {
    return bv16_launch(contract);
  }
  return bv32_launch(contract);
}

template <typename Bv16Launch, typename Bv32Launch>
inline decltype(auto) DispatchGdnDecodeValueTile(
    const char* bv_env_value, const char* swizzle_env_value, int64_t dv,
    int64_t dk, int requested_nw, Bv16Launch&& bv16_launch,
    Bv32Launch&& bv32_launch) {
  const GdnDecodeLaunchContract contract = GdnDecodeLaunchContractFor(
      bv_env_value, swizzle_env_value, dv, dk, requested_nw);
  if (contract.selected_tile == GdnDecodeValueTile::kBv16) {
    return bv16_launch(contract);
  }
  return bv32_launch(contract);
}

template <typename Bv16Launch, typename Bv32Launch>
inline decltype(auto) DispatchGdnDecodeValueTile(
    const char* bv_env_value, const char* swizzle_env_value,
    const char* regstate_env_value, int64_t dv, int64_t dk, int requested_nw,
    Bv16Launch&& bv16_launch, Bv32Launch&& bv32_launch) {
  const GdnDecodeLaunchContract contract = GdnDecodeLaunchContractFor(
      bv_env_value, swizzle_env_value, regstate_env_value, dv, dk,
      requested_nw);
  if (contract.selected_tile == GdnDecodeValueTile::kBv16) {
    return bv16_launch(contract);
  }
  return bv32_launch(contract);
}

}  // namespace vt::cuda

#endif  // VT_CUDA_GDN_DECODE_FUSED_H_
