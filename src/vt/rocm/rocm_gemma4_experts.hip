// ROCm upload of Gemma-4 MoE expert stacks (BF16 fused or FP8→BF16 dequant).
#include <cstring>
#include "vt/rocm/rocm_gemma4_prefill_dequant_cache.h"
// Packs GPU0 first (compute device), then GPU1. FP8 streams per-expert to
// avoid ~30G host OOM from permanent BF16 caches / full-layer host buffers.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <mutex>
#include <vector>

#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/gemma4_moe.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/fused_ops.h"
#include "vt/ops.h"

namespace vllm {
namespace {
// hipDeviceEnablePeerAccess leaves hipErrorPeerAccessAlreadyEnabled sticky;
// later Check(hipGetLastError()) would false-fail (seen on peer-act GeluAndMul).
inline void EnablePeerAccessQuiet(int peer_dev) {
  const hipError_t pe = hipDeviceEnablePeerAccess(peer_dev, 0);
  if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
    // best-effort; unsupported P2P falls back to staging copies elsewhere
  }
  (void)hipGetLastError();  // clear sticky
}


void Check(hipError_t e, const char* w) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string("gemma4 resident: ") + w + ": " + hipGetErrorString(e));
}

bool DeviceHasRoom(int dev, size_t need_bytes) {
  Check(hipSetDevice(dev), "hipSetDevice");
  size_t free_b = 0, tot_b = 0;
  if (hipMemGetInfo(&free_b, &tot_b) != hipSuccess) return true;
  // GPU0 is compute: dense weights + KV + activations need real headroom.
  // Default 12 GiB (lab dual-GPU). Override: VT_GEMMA4_GPU0_HEADROOM_GB (e.g. 4 for
  // single-GPU resident + short KV / graph experiments).
  static const size_t gpu0_headroom = []() -> size_t {
    const char* e = std::getenv("VT_GEMMA4_GPU0_HEADROOM_GB");
    if (e && e[0]) {
      const int g = std::atoi(e);
      if (g >= 0 && g <= 24) return static_cast<size_t>(g) << 30;
    }
    return static_cast<size_t>(12ull << 30);
  }();
  const size_t headroom = (dev == 0) ? gpu0_headroom : (512ull << 20);
  return free_b >= need_bytes + headroom;
}

bool AllocLayerDev(int dev, size_t gu_bytes, size_t dn_bytes, void** gu, void** dn) {
  if (!DeviceHasRoom(dev, gu_bytes + dn_bytes)) return false;
  Check(hipSetDevice(dev), "hipSetDevice");
  *gu = nullptr;
  *dn = nullptr;
  if (hipMalloc(gu, gu_bytes) != hipSuccess) return false;
  if (hipMalloc(dn, dn_bytes) != hipSuccess) {
    (void)hipFree(*gu);
    *gu = nullptr;
    return false;
  }
  return true;
}

bool AllocFour(int dev, size_t a, size_t b, size_t c, size_t d, void** pa, void** pb, void** pc,
               void** pd) {
  if (!DeviceHasRoom(dev, a + b + c + d)) return false;
  Check(hipSetDevice(dev), "hipSetDevice");
  *pa = *pb = *pc = *pd = nullptr;
  if (hipMalloc(pa, a) != hipSuccess) return false;
  if (hipMalloc(pb, b) != hipSuccess) {
    (void)hipFree(*pa);
    *pa = nullptr;
    return false;
  }
  if (hipMalloc(pc, c) != hipSuccess) {
    (void)hipFree(*pa);
    (void)hipFree(*pb);
    *pa = *pb = nullptr;
    return false;
  }
  if (hipMalloc(pd, d) != hipSuccess) {
    (void)hipFree(*pa);
    (void)hipFree(*pb);
    (void)hipFree(*pc);
    *pa = *pb = *pc = nullptr;
    return false;
  }
  return true;
}

// Native FP8 resident: keep u8 weights + BF16 channel scales on device (no BF16 expand).
// ~½ VRAM vs UploadFp8LayerStreaming BF16 path. Per-expert pointers alias layer packs.
bool UploadFp8NativeLayer(Gemma4FusedExperts& ex, int dev) {
  const int64_t E = ex.num_experts;
  const int64_t I = ex.intermediate;
  const int64_t H = ex.hidden;
  if (E <= 0 || I <= 0 || H <= 0 || ex.fp8.size() != static_cast<size_t>(E)) return false;

  const size_t gu_one = static_cast<size_t>(2 * I * H);       // u8
  const size_t dn_one = static_cast<size_t>(H * I);           // u8
  const size_t sgu_one = static_cast<size_t>(2 * I) * 2;      // bf16
  const size_t sdn_one = static_cast<size_t>(H) * 2;          // bf16
  const size_t gu_bytes = gu_one * static_cast<size_t>(E);
  const size_t dn_bytes = dn_one * static_cast<size_t>(E);
  const size_t sgu_bytes = sgu_one * static_cast<size_t>(E);
  const size_t sdn_bytes = sdn_one * static_cast<size_t>(E);

  void *gu = nullptr, *dn = nullptr, *sgu = nullptr, *sdn = nullptr;
  if (!AllocFour(dev, gu_bytes, dn_bytes, sgu_bytes, sdn_bytes, &gu, &dn, &sgu, &sdn))
    return false;

  std::vector<uint8_t> gu_pack(gu_one);
  for (int64_t e = 0; e < E; ++e) {
    auto& fex = ex.fp8[static_cast<size_t>(e)];
    VT_CHECK(fex.gate_w.HasHostBytes() && fex.up_w.HasHostBytes() && fex.down_w.HasHostBytes(),
             "fp8 native resident: missing weights");
    VT_CHECK(fex.gate_s.HasHostBytes() && fex.up_s.HasHostBytes() && fex.down_s.HasHostBytes(),
             "fp8 native resident: missing scales");
    // Pack gate|up FP8 contiguous [2I, H]
    const size_t half = static_cast<size_t>(I * H);
    std::memcpy(gu_pack.data(), fex.gate_w.bytes.data(), half);
    std::memcpy(gu_pack.data() + half, fex.up_w.bytes.data(), half);
    Check(hipMemcpy(static_cast<char*>(gu) + static_cast<size_t>(e) * gu_one, gu_pack.data(),
                    gu_one, hipMemcpyHostToDevice),
          "H2D fp8 gu");
    Check(hipMemcpy(static_cast<char*>(dn) + static_cast<size_t>(e) * dn_one,
                    fex.down_w.bytes.data(), dn_one, hipMemcpyHostToDevice),
          "H2D fp8 dn");
    // scales: gate then up → [2I]
    Check(hipMemcpy(static_cast<char*>(sgu) + static_cast<size_t>(e) * sgu_one,
                    fex.gate_s.bytes.data(), static_cast<size_t>(I) * 2, hipMemcpyHostToDevice),
          "H2D sgu g");
    Check(hipMemcpy(static_cast<char*>(sgu) + static_cast<size_t>(e) * sgu_one +
                        static_cast<size_t>(I) * 2,
                    fex.up_s.bytes.data(), static_cast<size_t>(I) * 2, hipMemcpyHostToDevice),
          "H2D sgu u");
    Check(hipMemcpy(static_cast<char*>(sdn) + static_cast<size_t>(e) * sdn_one,
                    fex.down_s.bytes.data(), sdn_one, hipMemcpyHostToDevice),
          "H2D sdn");

    // Drop host BF16 caches if any
    fex.cached_gu.clear();
    fex.cached_gu.shrink_to_fit();
    fex.cached_dn.clear();
    fex.cached_dn.shrink_to_fit();

    fex.dev_fp8_gu = static_cast<char*>(gu) + static_cast<size_t>(e) * gu_one;
    fex.dev_fp8_dn = static_cast<char*>(dn) + static_cast<size_t>(e) * dn_one;
    fex.dev_s_gu = static_cast<char*>(sgu) + static_cast<size_t>(e) * sgu_one;
    fex.dev_s_dn = static_cast<char*>(sdn) + static_cast<size_t>(e) * sdn_one;
    // Clear BF16 device expert slots (owned by LRU if any)
    fex.dev_gu = nullptr;
    fex.dev_dn = nullptr;
  }

  ex.fp8_gu_base = gu;
  ex.fp8_dn_base = dn;
  ex.fp8_sgu_base = sgu;
  ex.fp8_sdn_base = sdn;
  ex.fp8_native_resident = true;
  ex.gate_up_dev = nullptr;
  ex.down_dev = nullptr;
  ex.dev_id = dev;
  return true;
}

// Legacy: FP8 → BF16 expand on device (2× VRAM). Opt-in VT_GEMMA4_RESIDENT_BF16=1.
bool UploadFp8LayerStreaming(Gemma4FusedExperts& ex, int dev) {
  const int64_t E = ex.num_experts;
  const int64_t I = ex.intermediate;
  const int64_t H = ex.hidden;
  const size_t gu_one = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_one = static_cast<size_t>(H * I) * sizeof(uint16_t);
  const size_t gu_bytes = gu_one * static_cast<size_t>(E);
  const size_t dn_bytes = dn_one * static_cast<size_t>(E);

  void* gu = nullptr;
  void* dn = nullptr;
  if (!AllocLayerDev(dev, gu_bytes, dn_bytes, &gu, &dn)) return false;

  std::vector<uint16_t> gu_tmp(static_cast<size_t>(2 * I * H));
  std::vector<uint16_t> dn_tmp(static_cast<size_t>(H * I));
  for (int64_t e = 0; e < E; ++e) {
    DequantGemma4Fp8ExpertToBf16Ephemeral(ex.fp8[static_cast<size_t>(e)], I, H, gu_tmp.data(),
                                          dn_tmp.data());
    // Drop any accidental permanent cache on this expert (decode may have set it).
    ex.fp8[static_cast<size_t>(e)].cached_gu.clear();
    ex.fp8[static_cast<size_t>(e)].cached_gu.shrink_to_fit();
    ex.fp8[static_cast<size_t>(e)].cached_dn.clear();
    ex.fp8[static_cast<size_t>(e)].cached_dn.shrink_to_fit();

    Check(hipMemcpy(static_cast<char*>(gu) + static_cast<size_t>(e) * gu_one, gu_tmp.data(),
                    gu_one, hipMemcpyHostToDevice),
          "H2D gu e");
    Check(hipMemcpy(static_cast<char*>(dn) + static_cast<size_t>(e) * dn_one, dn_tmp.data(),
                    dn_one, hipMemcpyHostToDevice),
          "H2D dn e");
  }
  ex.gate_up_dev = gu;
  ex.down_dev = dn;
  ex.dev_id = dev;
  return true;
}

bool UploadBf16Layer(Gemma4FusedExperts& ex, int dev) {
  const int64_t E = ex.num_experts;
  const int64_t I = ex.intermediate;
  const int64_t H = ex.hidden;
  const size_t gu_bytes = static_cast<size_t>(E * 2 * I * H) * 2;
  const size_t dn_bytes = static_cast<size_t>(E * H * I) * 2;
  VT_CHECK(ex.gate_up.bytes.size() == gu_bytes && ex.down.bytes.size() == dn_bytes,
           "gemma4 resident: bf16 fused size");
  void* gu = nullptr;
  void* dn = nullptr;
  if (!AllocLayerDev(dev, gu_bytes, dn_bytes, &gu, &dn)) return false;
  Check(hipMemcpy(gu, ex.gate_up.bytes.data(), gu_bytes, hipMemcpyHostToDevice), "H2D gu");
  Check(hipMemcpy(dn, ex.down.bytes.data(), dn_bytes, hipMemcpyHostToDevice), "H2D dn");
  ex.gate_up_dev = gu;
  ex.down_dev = dn;
  ex.dev_id = dev;
  return true;
}

}  // namespace

size_t UploadGemma4ExpertsResident(std::vector<Gemma4MoeLayerWeights>& layers,
                                   int num_gpus) {
  if (num_gpus < 1) num_gpus = 1;
  int available = 0;
  Check(hipGetDeviceCount(&available), "hipGetDeviceCount");
  if (available < 1) return 0;
  if (num_gpus > available) num_gpus = available;

  // Enable peer access so off-device resident experts can D2D into compute GPU.
  for (int a = 0; a < num_gpus; ++a) {
    for (int b = 0; b < num_gpus; ++b) {
      if (a == b) continue;
      int can = 0;
      if (hipDeviceCanAccessPeer(&can, a, b) == hipSuccess && can) {
        Check(hipSetDevice(a), "peer set");
        hipError_t pe = hipDeviceEnablePeerAccess(b, 0);
        if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
          std::fprintf(stderr, "gemma4 moe: peer access %d->%d failed: %s\n", a, b,
                       hipGetErrorString(pe));
        }
      }
    }
  }

  size_t total = 0;
  int ok_layers = 0;
  int max_layers = 100000;
  if (const char* ml = std::getenv("VT_GEMMA4_RESIDENT_MAX_LAYERS"))
    max_layers = std::max(0, std::atoi(ml));

  int fill_dev = 0;

  // FP8 resident mode:
  // Default: native FP8 packs (½ VRAM + fast fused ExpertGeGLU FP8 kernels).
  // VT_GEMMA4_RESIDENT_BF16=1 forces BF16 expand (hipBLAS reference / A/B).
  static const bool force_bf16 = [] {
    const char* e = std::getenv("VT_GEMMA4_RESIDENT_BF16");
    return e && e[0] == '1';
  }();
  static const bool force_native = [] {
    const char* e = std::getenv("VT_GEMMA4_RESIDENT_NATIVE");
    // Default ON when unset. =0 disables native (use BF16 expand).
    if (e == nullptr) return true;
    return e[0] == '1';
  }();

  for (size_t li = 0; li < layers.size(); ++li) {
    if (ok_layers >= max_layers) break;
    auto& moe = layers[li];
    if (!moe.enabled || moe.experts.Empty()) continue;
    auto& ex = moe.experts;
    const int64_t E = ex.num_experts;
    const int64_t I = ex.intermediate;
    const int64_t H = ex.hidden;
    const bool want_native = ex.is_fp8 && !force_bf16 && force_native;
    const size_t layer_bytes =
        want_native
            ? (static_cast<size_t>(E) * (static_cast<size_t>(2 * I * H) + static_cast<size_t>(H * I) +
                                         static_cast<size_t>(2 * I) * 2 + static_cast<size_t>(H) * 2))
            : (static_cast<size_t>(E * 2 * I * H) * 2 + static_cast<size_t>(E * H * I) * 2);

    bool ok = false;
    if (ex.is_fp8) {
      if (want_native)
        ok = UploadFp8NativeLayer(ex, fill_dev);
      else
        ok = UploadFp8LayerStreaming(ex, fill_dev);
    } else {
      ok = UploadBf16Layer(ex, fill_dev);
    }

    if (!ok && fill_dev + 1 < num_gpus) {
      ++fill_dev;
      std::fprintf(stderr, "gemma4 moe: GPU%d full after %d layers; continuing on GPU%d\n",
                   fill_dev - 1, ok_layers, fill_dev);
      if (ex.is_fp8) {
        if (want_native)
          ok = UploadFp8NativeLayer(ex, fill_dev);
        else
          ok = UploadFp8LayerStreaming(ex, fill_dev);
      } else {
        ok = UploadBf16Layer(ex, fill_dev);
      }
    }
    if (!ok) {
      std::fprintf(stderr, "gemma4 moe: resident upload stopped at layer %zu (OOM)\n", li);
      break;
    }
    total += layer_bytes;
    ++ok_layers;
    std::fprintf(stderr,
                 "gemma4 moe: resident layer %zu -> gpu %d %s (%.2f GiB cumulative)\n", li,
                 ex.dev_id, ex.fp8_native_resident ? "fp8-native" : "bf16",
                 total / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stderr);
  }
  Check(hipSetDevice(0), "hipSetDevice0");
  std::fprintf(stderr, "gemma4 moe: RESIDENT DONE gpus=%d layers=%d %.2f GiB\n", num_gpus,
               ok_layers, total / (1024.0 * 1024.0 * 1024.0));
  std::fflush(stderr);
  return total;
}

size_t UploadGemma4ExpertsResidentForWeights(Gemma4Weights& weights, int num_gpus) {
  std::vector<Gemma4MoeLayerWeights> flat;
  std::vector<Gemma4MoeLayerWeights*> ptrs;
  for (auto& layer : weights.layers) {
    if (!layer.moe.enabled) continue;
    ptrs.push_back(&layer.moe);
    flat.push_back(std::move(layer.moe));
  }
  size_t n = UploadGemma4ExpertsResident(flat, num_gpus);
  for (size_t i = 0; i < ptrs.size(); ++i) *ptrs[i] = std::move(flat[i]);
  return n;
}

bool PeerCopyGemma4ExpertSlice(int src_dev, const void* gate_up_base, const void* down_base,
                               int expert_id, int64_t I, int64_t H, int compute_dev,
                               void* gate_up_dst, void* down_dst) {
  if (!gate_up_base || !down_base || !gate_up_dst || !down_dst) return false;
  if (src_dev < 0 || compute_dev < 0) return false;
  const size_t gu_one = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_one = static_cast<size_t>(H * I) * sizeof(uint16_t);
  const char* gu_src =
      static_cast<const char*>(gate_up_base) + static_cast<size_t>(expert_id) * gu_one;
  const char* dn_src =
      static_cast<const char*>(down_base) + static_cast<size_t>(expert_id) * dn_one;
  if (src_dev == compute_dev) {
    Check(hipSetDevice(compute_dev), "peer same set");
    Check(hipMemcpy(gate_up_dst, gu_src, gu_one, hipMemcpyDeviceToDevice), "D2D gu");
    Check(hipMemcpy(down_dst, dn_src, dn_one, hipMemcpyDeviceToDevice), "D2D dn");
    return true;
  }
  hipError_t e1 =
      hipMemcpyPeer(gate_up_dst, compute_dev, const_cast<char*>(gu_src), src_dev, gu_one);
  hipError_t e2 =
      hipMemcpyPeer(down_dst, compute_dev, const_cast<char*>(dn_src), src_dev, dn_one);
  if (e1 != hipSuccess || e2 != hipSuccess) {
    // Fallback: staged through host (still avoids FP8 dequant)
    std::vector<uint16_t> gu_tmp(gu_one / 2), dn_tmp(dn_one / 2);
    Check(hipSetDevice(src_dev), "peer fb src");
    Check(hipMemcpy(gu_tmp.data(), gu_src, gu_one, hipMemcpyDeviceToHost), "D2H gu");
    Check(hipMemcpy(dn_tmp.data(), dn_src, dn_one, hipMemcpyDeviceToHost), "D2H dn");
    Check(hipSetDevice(compute_dev), "peer fb dst");
    Check(hipMemcpy(gate_up_dst, gu_tmp.data(), gu_one, hipMemcpyHostToDevice), "H2D gu");
    Check(hipMemcpy(down_dst, dn_tmp.data(), dn_one, hipMemcpyHostToDevice), "H2D dn");
  }
  Check(hipSetDevice(compute_dev), "peer restore compute");
  return true;
}

bool PeerCopyGemma4Fp8ExpertSlice(int src_dev, const void* fp8_gu, const void* fp8_dn,
                                  const void* s_gu, const void* s_dn, int64_t I, int64_t H,
                                  int compute_dev, void* fp8_gu_dst, void* fp8_dn_dst,
                                  void* s_gu_dst, void* s_dn_dst) {
  if (!fp8_gu || !fp8_dn || !s_gu || !s_dn || !fp8_gu_dst || !fp8_dn_dst || !s_gu_dst ||
      !s_dn_dst)
    return false;
  if (src_dev < 0 || compute_dev < 0) return false;
  const size_t gu_b = static_cast<size_t>(2 * I * H);
  const size_t dn_b = static_cast<size_t>(H * I);
  const size_t sgu_b = static_cast<size_t>(2 * I) * 2;
  const size_t sdn_b = static_cast<size_t>(H) * 2;
  auto copy1 = [&](void* dst, const void* src, size_t n) -> bool {
    if (src_dev == compute_dev) {
      Check(hipSetDevice(compute_dev), "fp8 peer same");
      return hipMemcpy(dst, src, n, hipMemcpyDeviceToDevice) == hipSuccess;
    }
    if (hipMemcpyPeer(dst, compute_dev, const_cast<void*>(src), src_dev, n) == hipSuccess)
      return true;
    std::vector<char> tmp(n);
    Check(hipSetDevice(src_dev), "fp8 peer fb src");
    if (hipMemcpy(tmp.data(), src, n, hipMemcpyDeviceToHost) != hipSuccess) return false;
    Check(hipSetDevice(compute_dev), "fp8 peer fb dst");
    return hipMemcpy(dst, tmp.data(), n, hipMemcpyHostToDevice) == hipSuccess;
  };
  if (!copy1(fp8_gu_dst, fp8_gu, gu_b)) return false;
  if (!copy1(fp8_dn_dst, fp8_dn, dn_b)) return false;
  if (!copy1(s_gu_dst, s_gu, sgu_b)) return false;
  if (!copy1(s_dn_dst, s_dn, sdn_b)) return false;
  Check(hipSetDevice(compute_dev), "fp8 peer restore");
  return true;
}

bool RunGemma4Fp8TopKOnExpertDevice(vt::Queue& compute_q, int expert_dev, void* ysum_compute,
                                    const void* x_compute, const void* const* fp8_gu,
                                    const void* const* s_gu, const void* const* fp8_dn,
                                    const void* const* s_dn, const float* wts, int G, int I,
                                    int H) {
  const int compute_dev = compute_q.device.index;
  if (expert_dev < 0 || compute_dev < 0 || !ysum_compute || !x_compute || !fp8_gu || !s_gu ||
      !fp8_dn || !s_dn || !wts || G <= 0 || I <= 0 || H <= 0)
    return false;

  struct Tls {
    int cdev = -1;
    int edev = -1;
    vt::Queue cq{};  // only used when expert==compute and we need a stable q
    vt::Queue eq{};
    void* x = nullptr;
    void* y = nullptr;
    hipEvent_t ev_c = nullptr;
    hipEvent_t ev_e = nullptr;
    int H = 0;
    bool eq_live = false;
    bool cq_live = false;
    bool peer_ok = false;
  };
  static thread_local Tls tls;
  const size_t hb = static_cast<size_t>(H) * 2;  // bf16
  hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);

  // Same device: run on compute_q directly (no Create/Destroy per call).
  if (expert_dev == compute_dev) {
    return vt::ExpertGeGLUFp8TopKM1(compute_q, ysum_compute, x_compute, fp8_gu, s_gu, fp8_dn, s_dn,
                                    wts, G, I, H);
  }

  vt::Device edev{vt::DeviceType::kROCM, expert_dev};
  if (tls.edev != expert_dev || tls.cdev != compute_dev || tls.H != H || !tls.x || !tls.y) {
    if (tls.x) {
      Check(hipSetDevice(tls.edev >= 0 ? tls.edev : expert_dev), "free x");
      (void)hipFree(tls.x);
      tls.x = nullptr;
    }
    if (tls.y) {
      Check(hipSetDevice(tls.edev >= 0 ? tls.edev : expert_dev), "free y");
      (void)hipFree(tls.y);
      tls.y = nullptr;
    }
    if (tls.eq_live) {
      vt::DestroyQueue(tls.eq);
      tls.eq_live = false;
    }
    if (tls.ev_c) {
      (void)hipEventDestroy(tls.ev_c);
      tls.ev_c = nullptr;
    }
    if (tls.ev_e) {
      (void)hipEventDestroy(tls.ev_e);
      tls.ev_e = nullptr;
    }
    Check(hipSetDevice(expert_dev), "expert alloc");
    if (hipMalloc(&tls.x, hb) != hipSuccess) return false;
    if (hipMalloc(&tls.y, hb) != hipSuccess) {
      (void)hipFree(tls.x);
      tls.x = nullptr;
      return false;
    }
    tls.eq = vt::CreateQueue(edev);
    tls.eq_live = true;
    Check(hipSetDevice(compute_dev), "ev c");
    Check(hipEventCreateWithFlags(&tls.ev_c, hipEventDisableTiming), "ev_c");
    Check(hipSetDevice(expert_dev), "ev e");
    Check(hipEventCreateWithFlags(&tls.ev_e, hipEventDisableTiming), "ev_e");
    // Best-effort P2P enable (ignore already-enabled / unsupported).
    Check(hipSetDevice(compute_dev), "p2p c");
    EnablePeerAccessQuiet(expert_dev);
    Check(hipSetDevice(expert_dev), "p2p e");
    EnablePeerAccessQuiet(compute_dev);
    tls.edev = expert_dev;
    tls.cdev = compute_dev;
    tls.H = H;
    tls.peer_ok = true;
  }

  hipStream_t est = static_cast<hipStream_t>(tls.eq.handle);

  // Steady-state: stream-ordered peer, no per-hop hipSetDevice.
  if (hipEventRecord(tls.ev_c, cst) != hipSuccess) return false;
  if (hipStreamWaitEvent(est, tls.ev_c, 0) != hipSuccess) return false;

  bool x_ok =
      hipMemcpyPeerAsync(tls.x, expert_dev, const_cast<void*>(x_compute), compute_dev, hb, est) ==
      hipSuccess;
  if (!x_ok) {
    // Host staging fallback (sync).
    std::vector<char> tmp(hb);
    Check(hipSetDevice(compute_dev), "x d2h");
    if (hipStreamSynchronize(cst) != hipSuccess) return false;
    if (hipMemcpy(tmp.data(), x_compute, hb, hipMemcpyDeviceToHost) != hipSuccess) return false;
    Check(hipSetDevice(expert_dev), "x h2d");
    if (hipMemcpyAsync(tls.x, tmp.data(), hb, hipMemcpyHostToDevice, est) != hipSuccess)
      return false;
  }

  if (!vt::ExpertGeGLUFp8TopKM1(tls.eq, tls.y, tls.x, fp8_gu, s_gu, fp8_dn, s_dn, wts, G, I, H))
    return false;

  // expert → compute: record expert done, wait on compute stream, async peer y.
  if (hipEventRecord(tls.ev_e, est) != hipSuccess) return false;
  if (hipStreamWaitEvent(cst, tls.ev_e, 0) != hipSuccess) return false;

  bool y_ok =
      hipMemcpyPeerAsync(ysum_compute, compute_dev, tls.y, expert_dev, hb, cst) == hipSuccess;
  if (!y_ok) {
    std::vector<char> tmp(hb);
    Check(hipSetDevice(expert_dev), "y d2h");
    if (hipStreamSynchronize(est) != hipSuccess) return false;
    if (hipMemcpy(tmp.data(), tls.y, hb, hipMemcpyDeviceToHost) != hipSuccess) return false;
    Check(hipSetDevice(compute_dev), "y h2d");
    if (hipMemcpyAsync(ysum_compute, tmp.data(), hb, hipMemcpyHostToDevice, cst) != hipSuccess)
      return false;
  }
  if (hipSetDevice(compute_dev) != hipSuccess) return false;
  return true;
}

// Decode T=1 indexed: bases stay on expert_dev; peer only x + tiny idx/wts + y.
bool RunGemma4Fp8TopKIndexedOnExpertDevice(vt::Queue& compute_q, int expert_dev, void* ysum_compute,
                                           const void* x_compute, const void* gu_base,
                                           const void* dn_base, const void* sgu_base,
                                           const void* sdn_base, const int32_t* idx_compute,
                                           const float* wts_compute, int G, int I, int H) {
  const int compute_dev = compute_q.device.index;
  if (expert_dev < 0 || compute_dev < 0 || !ysum_compute || !x_compute || !gu_base || !dn_base ||
      !sgu_base || !sdn_base || !idx_compute || !wts_compute || G <= 0 || I <= 0 || H <= 0)
    return false;

  if (expert_dev == compute_dev) {
    return vt::ExpertGeGLUFp8TopKIndexed(compute_q, ysum_compute, x_compute, gu_base, dn_base,
                                         sgu_base, sdn_base, idx_compute, wts_compute, G, I, H);
  }

  struct Tls {
    int cdev = -1, edev = -1, H = 0, Gcap = 0;
    vt::Queue eq{};
    void* x = nullptr;
    void* y = nullptr;
    int32_t* idx = nullptr;
    float* wts = nullptr;
    hipEvent_t ev_c = nullptr, ev_e = nullptr;
    bool eq_live = false;
  };
  static thread_local Tls tls;
  const size_t hb = static_cast<size_t>(H) * 2;
  hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
  vt::Device edev{vt::DeviceType::kROCM, expert_dev};

  if (tls.edev != expert_dev || tls.cdev != compute_dev || tls.H != H || tls.Gcap < G || !tls.x) {
    if (tls.x) {
      Check(hipSetDevice(tls.edev >= 0 ? tls.edev : expert_dev), "free");
      (void)hipFree(tls.x);
      (void)hipFree(tls.y);
      (void)hipFree(tls.idx);
      (void)hipFree(tls.wts);
      tls.x = tls.y = nullptr;
      tls.idx = nullptr;
      tls.wts = nullptr;
    }
    if (tls.eq_live) {
      vt::DestroyQueue(tls.eq);
      tls.eq_live = false;
    }
    if (tls.ev_c) {
      (void)hipEventDestroy(tls.ev_c);
      tls.ev_c = nullptr;
    }
    if (tls.ev_e) {
      (void)hipEventDestroy(tls.ev_e);
      tls.ev_e = nullptr;
    }
    Check(hipSetDevice(expert_dev), "idx alloc");
    if (hipMalloc(&tls.x, hb) != hipSuccess) return false;
    if (hipMalloc(&tls.y, hb) != hipSuccess) {
      (void)hipFree(tls.x);
      tls.x = nullptr;
      return false;
    }
    if (hipMalloc(&tls.idx, sizeof(int32_t) * 8) != hipSuccess) return false;
    if (hipMalloc(&tls.wts, sizeof(float) * 8) != hipSuccess) return false;
    tls.eq = vt::CreateQueue(edev);
    tls.eq_live = true;
    Check(hipSetDevice(compute_dev), "ev");
    Check(hipEventCreateWithFlags(&tls.ev_c, hipEventDisableTiming), "ev_c");
    Check(hipSetDevice(expert_dev), "ev e");
    Check(hipEventCreateWithFlags(&tls.ev_e, hipEventDisableTiming), "ev_e");
    Check(hipSetDevice(compute_dev), "p2p");
    EnablePeerAccessQuiet(expert_dev);
    Check(hipSetDevice(expert_dev), "p2p e");
    EnablePeerAccessQuiet(compute_dev);
    tls.edev = expert_dev;
    tls.cdev = compute_dev;
    tls.H = H;
    tls.Gcap = 8;
  }

  hipStream_t est = static_cast<hipStream_t>(tls.eq.handle);
  // Steady-state: no hipSetDevice between stream ops (graph-prep + less driver tax).
  if (hipEventRecord(tls.ev_c, cst) != hipSuccess) return false;
  if (hipStreamWaitEvent(est, tls.ev_c, 0) != hipSuccess) return false;

  auto peer = [&](void* dst, const void* src, size_t n) -> bool {
    return hipMemcpyPeerAsync(dst, expert_dev, const_cast<void*>(src), compute_dev, n, est) ==
           hipSuccess;
  };
  if (!peer(tls.x, x_compute, hb)) return false;
  if (!peer(tls.idx, idx_compute, sizeof(int32_t) * static_cast<size_t>(G))) return false;
  if (!peer(tls.wts, wts_compute, sizeof(float) * static_cast<size_t>(G))) return false;

  if (!vt::ExpertGeGLUFp8TopKIndexed(tls.eq, tls.y, tls.x, gu_base, dn_base, sgu_base, sdn_base,
                                     tls.idx, tls.wts, G, I, H))
    return false;

  if (hipEventRecord(tls.ev_e, est) != hipSuccess) return false;
  if (hipStreamWaitEvent(cst, tls.ev_e, 0) != hipSuccess) return false;
  if (hipMemcpyPeerAsync(ysum_compute, compute_dev, tls.y, expert_dev, hb, cst) != hipSuccess)
    return false;
  if (hipSetDevice(compute_dev) != hipSuccess) return false;
  return true;
}

// Prefill: run ExpertGeGLU for M rows on expert_dev (weights resident there).
// Peer only activations M×H bf16 — not multi-GB FP8 expert packs.
// #839: wrapper → Launch/Finish + pin until host-observed retirement.

struct HipPrefillCacheHooks {
  vt::Queue* q = nullptr;
  const void* fp8_gu = nullptr;
  const void* s_gu = nullptr;
  const void* fp8_dn = nullptr;
  const void* s_dn = nullptr;
  int I = 0, H = 0;
  struct Event {
    hipEvent_t h = nullptr;
  };
  bool SetDevice(int d) { return hipSetDevice(d) == hipSuccess; }
  void* Malloc(size_t n) {
    void* p = nullptr;
    return hipMalloc(&p, n) == hipSuccess ? p : nullptr;
  }
  void Free(void* p) {
    if (p) (void)hipFree(p);
  }
  void DestroyEvent(Event& e) {
    if (e.h) {
      (void)hipEventDestroy(e.h);
      e.h = nullptr;
    }
  }
  bool Fill(void* gu, void* dn) {
    if (!q || !fp8_gu || !s_gu || !fp8_dn || !s_dn || I <= 0 || H <= 0 || !gu || !dn) return false;
    vt::DequantFp8ChannelBf16(*q, gu, fp8_gu, s_gu, 2 * I, H);
    vt::DequantFp8ChannelBf16(*q, dn, fp8_dn, s_dn, H, I);
    return true;
  }
  bool RecordReady(Event& e) {
    if (!q) return false;
    if (!e.h) {
      if (hipEventCreateWithFlags(&e.h, hipEventDisableTiming) != hipSuccess) return false;
    }
    return hipEventRecord(e.h, static_cast<hipStream_t>(q->handle)) == hipSuccess;
  }
  bool WaitReady(Event& e) {
    if (!q || !e.h) return false;
    return hipStreamWaitEvent(static_cast<hipStream_t>(q->handle), e.h, 0) == hipSuccess;
  }
};

using PrefillDequantCache = vt::rocm::PrefillDequantCacheT<HipPrefillCacheHooks>;

inline bool BindPrefillCacheFill(PrefillDequantCache& cache, vt::Queue& q, const void* fp8_gu,
                                 const void* s_gu, const void* fp8_dn, const void* s_dn, int I,
                                 int H) {
  cache.hooks.q = &q;
  cache.hooks.fp8_gu = fp8_gu;
  cache.hooks.s_gu = s_gu;
  cache.hooks.fp8_dn = fp8_dn;
  cache.hooks.s_dn = s_dn;
  cache.hooks.I = I;
  cache.hooks.H = H;
  return true;
}

struct DequantCacheSlot {
  PrefillDequantCache cache;
};

DequantCacheSlot& DequantCacheSlotFor(int device) {
  static DequantCacheSlot slots[8];
  if (device < 0 || device >= 8) {
    static DequantCacheSlot fallback;
    return fallback;
  }
  return slots[device];
}

struct PeerSlot {
  int cdev = -1, edev = -1;
  int I = 0, H = 0, Mcap = 0;
  int pending_M = 0;
  vt::Queue eq{};
  bool eq_live = false;
  void* x = nullptr;
  void* y = nullptr;
  void* gu_act = nullptr;
  void* act = nullptr;
  hipEvent_t ev_c = nullptr, ev_e = nullptr;
  int cache_pin = -1;
  int cache_dev = -1;
  bool ev_e_recorded = false;
  bool this_gen_ev_e = false;
  bool work_on_compute = false;
  bool work_on_expert = false;
  bool rollback_armed = false;
  bool quarantined = false;
  bool fill_lease = false;
};

struct PeerPipe {
  PeerSlot s[1];  // slot 0 only; overlap slot-1 is a later row
};

inline PeerPipe& PeerPipeTls() {
  static thread_local PeerPipe pipe;
  return pipe;
}

vt::rocm::PrefillPeerLife PeerLifeView(const PeerSlot& tls) { return vt::rocm::LifeFromSlot(tls); }

void RestoreComputeOrThrow(int compute_dev) {
  if (hipSetDevice(compute_dev) != hipSuccess) throw vt::rocm::RestoreFailed{};
}

// Unpin only after a successful event/stream retirement. Failure quarantines the pin.
// ChoosePrefillRetire forbids syncing a leftover prior ev_e during current-gen rollback.
bool RetirePinThenUnpin(PeerSlot& tls, hipStream_t expert_stream, hipStream_t compute_stream) {
  const auto target = vt::rocm::ChoosePrefillRetire(PeerLifeView(tls));
  bool retire_ok = true;
  if (target == vt::rocm::PrefillRetireTarget::RecordedEvent && tls.ev_e) {
    retire_ok = hipEventSynchronize(tls.ev_e) == hipSuccess;
  } else if (target == vt::rocm::PrefillRetireTarget::ExpertStream && expert_stream) {
    retire_ok = hipStreamSynchronize(expert_stream) == hipSuccess;
  } else if (target == vt::rocm::PrefillRetireTarget::ComputeStream && compute_stream) {
    retire_ok = hipStreamSynchronize(compute_stream) == hipSuccess;
  } else if (target == vt::rocm::PrefillRetireTarget::ComputeStream && expert_stream) {
    // no compute handle (should not happen after first enqueue) — do not fake-succeed
    retire_ok = false;
  }
  if (!retire_ok) {
    tls.quarantined = true;
    return false;
  }
  if (tls.cache_pin >= 0 && tls.cache_dev >= 0) {
    auto& wslot = DequantCacheSlotFor(tls.cache_dev);
    std::lock_guard<std::mutex> lk(wslot.cache.mu);
    if (!wslot.cache.ReleaseObservedPinLocked(tls.cache_pin, tls.fill_lease)) {
      tls.quarantined = true;
      return false;
    }
  }
  tls.cache_pin = -1;
  tls.cache_dev = -1;
  tls.rollback_armed = false;
  tls.ev_e_recorded = false;
  tls.this_gen_ev_e = false;
  tls.work_on_compute = false;
  tls.work_on_expert = false;
  tls.fill_lease = false;
  return true;
}

void FailLaunchRestore(PeerSlot& tls, hipStream_t est, hipStream_t cst, int compute_dev) {
  if (!RetirePinThenUnpin(tls, est, cst)) tls.quarantined = true;
  RestoreComputeOrThrow(compute_dev);
}

bool LaunchGemma4Fp8ExpertGeGLUPrefillPeer(vt::Queue& compute_q, int slot, int expert_dev,
                                           const void* x_compute, const void* fp8_gu,
                                           const void* s_gu, const void* fp8_dn, const void* s_dn,
                                           int M, int I, int H);
bool FinishGemma4Fp8ExpertGeGLUPrefillPeer(vt::Queue& compute_q, int slot, void* y_compute, int M,
                                           int H);

bool RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice(vt::Queue& compute_q, int expert_dev,
                                                  void* y_compute, const void* x_compute,
                                                  const void* fp8_gu, const void* s_gu,
                                                  const void* fp8_dn, const void* s_dn, int M,
                                                  int I, int H) {
  const int compute_dev = compute_q.device.index;
  if (expert_dev < 0 || compute_dev < 0 || !y_compute || !x_compute || !fp8_gu || !s_gu ||
      !fp8_dn || !s_dn || M <= 0 || I <= 0 || H <= 0)
    return false;

  if (expert_dev == compute_dev) {
    vt::Device dev = compute_q.device;
    struct SameTls {
      int dev = -1;
      int I = 0, H = 0, Mcap = 0;
      void* gu_act = nullptr;
      void* act = nullptr;
      vt::rocm::SameDevLife life{};
    };
    static thread_local SameTls tls_slots[2];
    SameTls& tls = tls_slots[(compute_dev == 1) ? 1 : 0];
    if (!tls.life.CanEnter()) return false;
    if ((tls.dev != compute_dev || tls.I != I || tls.H != H || tls.Mcap < M || !tls.gu_act) &&
        !tls.life.CanReconfigure())
      return false;
    if (tls.dev != compute_dev || tls.I != I || tls.H != H || tls.Mcap < M || !tls.gu_act) {
      auto freep = [&](void*& p) {
        if (p) {
          (void)hipFree(p);
          p = nullptr;
        }
      };
      freep(tls.gu_act);
      freep(tls.act);
      Check(hipSetDevice(compute_dev), "same alloc");
      size_t gu_act_cap = static_cast<size_t>(std::max(M, 64)) * static_cast<size_t>(2 * I) * 2;
      size_t act_cap = static_cast<size_t>(std::max(M, 64)) * static_cast<size_t>(I) * 2;
      void* ngu = nullptr;
      void* nact = nullptr;
      if (hipMalloc(&ngu, gu_act_cap) != hipSuccess) return false;
      if (hipMalloc(&nact, act_cap) != hipSuccess) {
        (void)hipFree(ngu);
        return false;
      }
      tls.gu_act = ngu;
      tls.act = nact;
      tls.dev = compute_dev;
      tls.I = I;
      tls.H = H;
      tls.Mcap = std::max(M, 64);
    }
    auto& wslot = DequantCacheSlotFor(compute_dev);
    void* gu_w = nullptr;
    void* dn_w = nullptr;
    int pin = -1;
    hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
    {
      std::lock_guard<std::mutex> lk(wslot.cache.mu);
      if (!wslot.cache.Ensure(compute_dev, I, H)) return false;
      BindPrefillCacheFill(wslot.cache, compute_q, fp8_gu, s_gu, fp8_dn, s_dn, I, H);
      if (!wslot.cache.GetLocked(fp8_gu, &gu_w, &dn_w, &pin)) {
        if (pin >= 0) tls.life.PersistPin(pin, compute_dev);
      }
    }
    if (pin >= 0 && (gu_w == nullptr || dn_w == nullptr)) {
      const bool ok = hipStreamSynchronize(cst) == hipSuccess;
      if (!ok) {
        tls.life.Quarantine(pin, compute_dev);
        return false;
      }
      {
        std::lock_guard<std::mutex> lk(wslot.cache.mu);
        if (!wslot.cache.RetireFillLocked(pin, true)) {
          tls.life.Quarantine(pin, compute_dev);
          return false;
        }
      }
      tls.life.ClearPin();
      return false;
    }
    if (gu_w == nullptr || dn_w == nullptr) return false;
    tls.life.PersistPin(pin, compute_dev);
    try {
      vt::Tensor x = vt::Tensor::Contiguous(const_cast<void*>(x_compute), vt::DType::kBF16, dev, {M, H});
      vt::Tensor gu_wt = vt::Tensor::Contiguous(gu_w, vt::DType::kBF16, dev, {2 * I, H});
      vt::Tensor gu_act = vt::Tensor::Contiguous(tls.gu_act, vt::DType::kBF16, dev, {M, 2 * I});
      vt::Tensor act = vt::Tensor::Contiguous(tls.act, vt::DType::kBF16, dev, {M, I});
      vt::MatmulBT(compute_q, gu_act, x, gu_wt);
      vt::GeluAndMul(compute_q, act, gu_act);
      vt::MatmulBTAlphaBeta(compute_q, y_compute, act.data, dn_w, M, H, I, 1.f, 0.f, vt::DType::kBF16);
      if (hipStreamSynchronize(cst) != hipSuccess) {
        tls.life.Quarantine(pin, compute_dev);
        RestoreComputeOrThrow(compute_dev);
        return false;
      }
      {
        std::lock_guard<std::mutex> lk(wslot.cache.mu);
        wslot.cache.UnpinLocked(pin);
      }
      tls.life.ClearPin();
      RestoreComputeOrThrow(compute_dev);
      return true;
    } catch (...) {
      if (hipStreamSynchronize(cst) != hipSuccess) tls.life.Quarantine(pin, compute_dev);
      else {
        std::lock_guard<std::mutex> lk(wslot.cache.mu);
        wslot.cache.UnpinLocked(pin);
        tls.life.ClearPin();
      }
      RestoreComputeOrThrow(compute_dev);
      throw;
    }
  }

  if (!LaunchGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0, expert_dev, x_compute, fp8_gu,
                                             s_gu, fp8_dn, s_dn, M, I, H))
    return false;
  return FinishGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0, y_compute, M, H);
}

bool LaunchGemma4Fp8ExpertGeGLUPrefillPeer(vt::Queue& compute_q, int slot, int expert_dev,
                                           const void* x_compute, const void* fp8_gu,
                                           const void* s_gu, const void* fp8_dn, const void* s_dn,
                                           int M, int I, int H) {
  try {
  if (slot != 0 || !x_compute || !fp8_gu || !s_gu || !fp8_dn || !s_dn || M <= 0 ||
      I <= 0 || H <= 0 || expert_dev < 0)
    return false;
  const int compute_dev = compute_q.device.index;
  if (compute_dev < 0 || expert_dev == compute_dev) return false;

  PeerSlot& tls = PeerPipeTls().s[slot];
  auto freep = [](int dev, void*& p) {
    if (!p) return;
    (void)hipSetDevice(dev);
    (void)hipFree(p);
    p = nullptr;
  };
  const size_t row_b = static_cast<size_t>(H) * 2;
  const size_t xy_need = static_cast<size_t>(M) * row_b;
  vt::Device edev{vt::DeviceType::kROCM, expert_dev};
  hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);

  if (!vt::rocm::SlotReusable(tls.pending_M > 0, tls.rollback_armed, tls.quarantined)) return false;

  if (tls.edev != expert_dev || tls.cdev != compute_dev || tls.I != I || tls.H != H ||
      tls.Mcap < M || !tls.x) {
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.x);
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.y);
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.gu_act);
    freep(tls.edev >= 0 ? tls.edev : expert_dev, tls.act);
    if (tls.eq_live) {
      vt::DestroyQueue(tls.eq);
      tls.eq_live = false;
    }
    if (tls.ev_c) {
      (void)hipEventDestroy(tls.ev_c);
      tls.ev_c = nullptr;
    }
    if (tls.ev_e) {
      (void)hipEventDestroy(tls.ev_e);
      tls.ev_e = nullptr;
    }
    const int Mcap = std::max(M, 64);
    void* nx = nullptr;
    void* ny = nullptr;
    void* ngu = nullptr;
    void* nact = nullptr;
    hipEvent_t nev_c = nullptr;
    hipEvent_t nev_e = nullptr;
    vt::Queue nq{};
    bool nq_live = false;
    auto drop = [&]() {
      if (nx) (void)hipFree(nx);
      if (ny) (void)hipFree(ny);
      if (ngu) (void)hipFree(ngu);
      if (nact) (void)hipFree(nact);
      if (nev_c) (void)hipEventDestroy(nev_c);
      if (nev_e) (void)hipEventDestroy(nev_e);
      if (nq_live) vt::DestroyQueue(nq);
    };
    try {
      Check(hipSetDevice(expert_dev), "peer-pipe expert alloc");
      if (hipMalloc(&nx, static_cast<size_t>(Mcap) * row_b) != hipSuccess ||
          hipMalloc(&ny, static_cast<size_t>(Mcap) * row_b) != hipSuccess ||
          hipMalloc(&ngu, static_cast<size_t>(Mcap) * static_cast<size_t>(2 * I) * 2) != hipSuccess ||
          hipMalloc(&nact, static_cast<size_t>(Mcap) * static_cast<size_t>(I) * 2) != hipSuccess) {
        drop();
        RestoreComputeOrThrow(compute_dev);
        return false;
      }
      nq = vt::CreateQueue(edev);
      nq_live = true;
      Check(hipSetDevice(compute_dev), "peer-pipe ev_c");
      Check(hipEventCreateWithFlags(&nev_c, hipEventDisableTiming), "ev_c");
      Check(hipSetDevice(expert_dev), "peer-pipe ev_e");
      Check(hipEventCreateWithFlags(&nev_e, hipEventDisableTiming), "ev_e");
      Check(hipSetDevice(compute_dev), "p2p c");
      EnablePeerAccessQuiet(expert_dev);
      Check(hipSetDevice(expert_dev), "p2p e");
      EnablePeerAccessQuiet(compute_dev);
    } catch (...) {
      drop();
      RestoreComputeOrThrow(compute_dev);
      throw;
    }
    tls.x = nx;
    tls.y = ny;
    tls.gu_act = ngu;
    tls.act = nact;
    tls.eq = nq;
    tls.eq_live = true;
    tls.ev_c = nev_c;
    tls.ev_e = nev_e;
    tls.edev = expert_dev;
    tls.cdev = compute_dev;
    tls.I = I;
    tls.H = H;
    tls.Mcap = Mcap;
    tls.cache_pin = -1;
    tls.cache_dev = -1;
    tls.ev_e_recorded = false;
    tls.this_gen_ev_e = false;
    tls.work_on_compute = false;
    tls.work_on_expert = false;
    tls.rollback_armed = false;
  }

  hipStream_t est = static_cast<hipStream_t>(tls.eq.handle);
  if (hipEventRecord(tls.ev_c, cst) != hipSuccess) {
    RestoreComputeOrThrow(compute_dev); return false;
  }
  // First successful current-gen enqueue is on the compute stream.
  tls.rollback_armed = true;
  tls.work_on_compute = true;
  tls.this_gen_ev_e = false;
  if (hipStreamWaitEvent(est, tls.ev_c, 0) != hipSuccess) {
    FailLaunchRestore(tls, est, cst, compute_dev); return false;
  }
  tls.work_on_expert = true;

  bool x_ok = hipMemcpyPeerAsync(tls.x, expert_dev, const_cast<void*>(x_compute), compute_dev,
                                 xy_need, est) == hipSuccess;
  if (!x_ok) {
    std::vector<char> tmp(xy_need);
    Check(hipSetDevice(compute_dev), "x d2h");
    if (hipStreamSynchronize(cst) != hipSuccess) {
      FailLaunchRestore(tls, est, cst, compute_dev); return false;
    }
    if (hipMemcpy(tmp.data(), x_compute, xy_need, hipMemcpyDeviceToHost) != hipSuccess) {
      FailLaunchRestore(tls, est, cst, compute_dev); return false;
    }
    Check(hipSetDevice(expert_dev), "x h2d");
    if (hipMemcpyAsync(tls.x, tmp.data(), xy_need, hipMemcpyHostToDevice, est) != hipSuccess) {
      FailLaunchRestore(tls, est, cst, compute_dev); return false;
    }
  }

  auto& wslot = DequantCacheSlotFor(expert_dev);
  void* gu_w = nullptr;
  void* dn_w = nullptr;
  int pin = -1;
  bool got = false;
  {
    std::lock_guard<std::mutex> lk(wslot.cache.mu);
    if (!wslot.cache.Ensure(expert_dev, I, H)) {
      got = false;
    } else {
      Check(hipSetDevice(expert_dev), "dequant setdev");
      BindPrefillCacheFill(wslot.cache, tls.eq, fp8_gu, s_gu, fp8_dn, s_dn, I, H);
      got = wslot.cache.GetLocked(fp8_gu, &gu_w, &dn_w, &pin);
      if (!got && pin >= 0) {
        tls.cache_pin = pin;
        tls.cache_dev = expert_dev;
        tls.fill_lease = true;
      }
    }
  }
  if (!got) {
    FailLaunchRestore(tls, est, cst, compute_dev); return false;
  }
  tls.cache_pin = pin;
  tls.cache_dev = expert_dev;
  tls.fill_lease = false;
  tls.ev_e_recorded = false;

  vt::Tensor x = vt::Tensor::Contiguous(tls.x, vt::DType::kBF16, edev, {M, H});
  vt::Tensor gu_wt = vt::Tensor::Contiguous(gu_w, vt::DType::kBF16, edev, {2 * I, H});
  vt::Tensor gu_act = vt::Tensor::Contiguous(tls.gu_act, vt::DType::kBF16, edev, {M, 2 * I});
  vt::Tensor act = vt::Tensor::Contiguous(tls.act, vt::DType::kBF16, edev, {M, I});
  vt::MatmulBT(tls.eq, gu_act, x, gu_wt);
  vt::GeluAndMul(tls.eq, act, gu_act);
  vt::MatmulBTAlphaBeta(tls.eq, tls.y, act.data, dn_w, M, H, I, 1.f, 0.f, vt::DType::kBF16);

  if (hipEventRecord(tls.ev_e, est) != hipSuccess) {
    FailLaunchRestore(tls, est, cst, compute_dev); return false;
  }
  tls.this_gen_ev_e = true;
  tls.ev_e_recorded = true;
  vt::rocm::PublishThenRestoreOrThrow(
      tls, M, [&] { RestoreComputeOrThrow(compute_dev); },
      [&] { return RetirePinThenUnpin(tls, est, cst); });
  return true;
  } catch (const vt::rocm::RestoreFailed&) {
    PeerSlot& tls = PeerPipeTls().s[slot];
    hipStream_t est = tls.eq_live ? static_cast<hipStream_t>(tls.eq.handle) : nullptr;
    hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
    if (tls.cache_pin >= 0 || tls.rollback_armed || tls.pending_M > 0)
      (void)RetirePinThenUnpin(tls, est, cst);
    tls.pending_M = 0;
    throw;
  } catch (...) {
    const int compute_dev = compute_q.device.index;
    PeerSlot& tls = PeerPipeTls().s[slot];
    hipStream_t est = tls.eq_live ? static_cast<hipStream_t>(tls.eq.handle) : nullptr;
    hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
    if (tls.rollback_armed || tls.cache_pin >= 0) (void)RetirePinThenUnpin(tls, est, cst);
    RestoreComputeOrThrow(compute_dev);
    throw;
  }
}

bool FinishGemma4Fp8ExpertGeGLUPrefillPeer(vt::Queue& compute_q, int slot, void* y_compute, int M,
                                           int H) {
  try {
  if (slot != 0 || !y_compute || M <= 0 || H <= 0) return false;
  PeerSlot& tls = PeerPipeTls().s[slot];
  if (tls.pending_M <= 0 || !tls.y || !tls.ev_e) return false;
  if (M > tls.pending_M) return false;
  const int compute_dev = compute_q.device.index;
  const int expert_dev = tls.edev;
  const size_t xy_need = static_cast<size_t>(M) * static_cast<size_t>(H) * 2;
  hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
  hipStream_t est = tls.eq_live ? static_cast<hipStream_t>(tls.eq.handle) : nullptr;

  if (hipStreamWaitEvent(cst, tls.ev_e, 0) != hipSuccess) {
    if (!RetirePinThenUnpin(tls, est, cst)) tls.quarantined = true;
    else tls.pending_M = 0;
    RestoreComputeOrThrow(compute_dev); return false;
  }
  bool y_ok =
      hipMemcpyPeerAsync(y_compute, compute_dev, tls.y, expert_dev, xy_need, cst) == hipSuccess;
  if (!y_ok) {
    std::vector<char> tmp(xy_need);
    Check(hipSetDevice(expert_dev), "y d2h");
    if (est && hipStreamSynchronize(est) != hipSuccess) {
      tls.quarantined = true;
      RestoreComputeOrThrow(compute_dev); return false;
    }
    if (hipMemcpy(tmp.data(), tls.y, xy_need, hipMemcpyDeviceToHost) != hipSuccess) {
      if (!RetirePinThenUnpin(tls, est, cst)) tls.quarantined = true;
      else tls.pending_M = 0;
      RestoreComputeOrThrow(compute_dev); return false;
    }
    Check(hipSetDevice(compute_dev), "y h2d");
    if (hipMemcpyAsync(y_compute, tmp.data(), xy_need, hipMemcpyHostToDevice, cst) != hipSuccess) {
      if (!RetirePinThenUnpin(tls, est, cst)) tls.quarantined = true;
      else tls.pending_M = 0;
      RestoreComputeOrThrow(compute_dev); return false;
    }
  }
  if (tls.ev_e && hipEventSynchronize(tls.ev_e) != hipSuccess) {
    tls.quarantined = true;
    RestoreComputeOrThrow(compute_dev); return false;
  }
  tls.ev_e_recorded = true;
  // Output copy is on cst. Host-observe that stream before pending/reuse.
  if (hipStreamSynchronize(cst) != hipSuccess) {
    tls.quarantined = true;
    RestoreComputeOrThrow(compute_dev); return false;
  }
  if (!RetirePinThenUnpin(tls, est, cst)) {
    tls.quarantined = true;
    RestoreComputeOrThrow(compute_dev); return false;
  }
  tls.pending_M = 0;
  RestoreComputeOrThrow(compute_dev); return true;
  } catch (const vt::rocm::RestoreFailed&) {
    PeerSlot& tls = PeerPipeTls().s[slot];
    hipStream_t est = tls.eq_live ? static_cast<hipStream_t>(tls.eq.handle) : nullptr;
    hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
    if (tls.cache_pin >= 0 || tls.rollback_armed) (void)RetirePinThenUnpin(tls, est, cst);
    tls.pending_M = 0;
    throw;
  } catch (...) {
    const int compute_dev = compute_q.device.index;
    PeerSlot& tls = PeerPipeTls().s[slot];
    hipStream_t est = tls.eq_live ? static_cast<hipStream_t>(tls.eq.handle) : nullptr;
    hipStream_t cst = static_cast<hipStream_t>(compute_q.handle);
    if (tls.cache_pin >= 0 || tls.rollback_armed) (void)RetirePinThenUnpin(tls, est, cst);
    RestoreComputeOrThrow(compute_dev);
    throw;
  }
}

void PinGemma4Fp8ExpertHostCache(const Gemma4Fp8ExpertMats& ex) {
  if (ex.host_pinned) return;
  if (ex.cached_gu.empty() || ex.cached_dn.empty()) return;
  const size_t gu_b = ex.cached_gu.size() * sizeof(uint16_t);
  const size_t dn_b = ex.cached_dn.size() * sizeof(uint16_t);
  hipError_t e1 =
      hipHostRegister(const_cast<uint16_t*>(ex.cached_gu.data()), gu_b, hipHostRegisterDefault);
  hipError_t e2 =
      hipHostRegister(const_cast<uint16_t*>(ex.cached_dn.data()), dn_b, hipHostRegisterDefault);
  if ((e1 == hipSuccess || e1 == hipErrorHostMemoryAlreadyRegistered) &&
      (e2 == hipSuccess || e2 == hipErrorHostMemoryAlreadyRegistered)) {
    ex.host_pinned = true;
  }
}

void UnpinGemma4Fp8ExpertHostCache(const Gemma4Fp8ExpertMats& ex) {
  if (!ex.host_pinned) return;
  if (!ex.cached_gu.empty()) {
    (void)hipHostUnregister(const_cast<uint16_t*>(ex.cached_gu.data()));
  }
  if (!ex.cached_dn.empty()) {
    (void)hipHostUnregister(const_cast<uint16_t*>(ex.cached_dn.data()));
  }
  ex.host_pinned = false;
}

}  // namespace vllm
