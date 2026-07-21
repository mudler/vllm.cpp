// Shared pooled device-scratch glue for dense-transformer forwards.
//
// Extracted VERBATIM (behavior-preserving) from include/vllm/model_executor/
// models/dense_attn_block.h so the NVFP4 W4A16 GEMM helpers
// (dense_nvfp4_gemm.h) can sit BENEATH the attention block that consumes them
// rather than above it — dense_attn_block.h now includes both this header and
// dense_nvfp4_gemm.h, which would otherwise be an include cycle. The definitions
// below are byte-for-byte the dense_attn_block.h originals and stay in the SAME
// namespace (`vllm::dense_attn`), so every existing `using namespace
// dense_attn;` consumer (qwen3.cpp, qwen3_moe.cpp, opt.cpp) resolves exactly as
// before and every dense forward remains BYTE-IDENTICAL.
//
// Contents (all `vllm::dense_attn`):
//   Dev                     — {Backend&, Queue&} device handle pair.
//   MakeTensor / Reshape    — non-owning strided views.
//   DevicePoolPolicy        — platform residency policy for the scratch pool.
//   DBuf                    — move-only pooled device allocation + tensor view.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/device_pool.h"       // DevicePool/Pool/ActivePool
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace dense_attn {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

struct Dev {
  Backend& b;
  Queue& q;
};

inline Tensor MakeTensor(void* data, DType dt, vt::Device dev,
                         const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t acc = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = acc;
    acc *= t.shape[i];
  }
  return t;
}

inline Tensor Reshape(const Tensor& src, const std::vector<int64_t>& shape) {
  return MakeTensor(src.data, src.dtype, src.device, shape);
}

// The device-scratch residency policy (BACKEND-PLATFORM item 2), resolved from
// the running device's platform. The DevicePool soft cap is platform data (0 ==
// uncapped, GB10 today ⇒ pool behavior byte-for-byte unchanged). Memoized in a
// function-local static: DBuf is a per-op hot path and the process runs on ONE
// device, so the virtual dispatch is paid exactly once. Mirrors qwen3_5.cpp.
struct DevicePoolPolicy {
  size_t cap_bytes = 0;  // residency_policy().device_pool_cap_bytes (0 == uncapped)
};
inline DevicePoolPolicy ResolveDevicePoolPolicy(const Dev& d) {
  static const DevicePoolPolicy p = [&] {
    const auto rp =
        vllm::platforms::GetPlatform(d.q.device.type).residency_policy();
    return DevicePoolPolicy{rp.device_pool_cap_bytes};
  }();
  return p;
}

// Owned device allocation + tensor view, routed through the SHARED DevicePool so
// the buffer's storage is reused rather than freed to the driver (avoiding the
// per-op cudaMalloc/cudaFree sync). Move-only, RAII. Ported verbatim from the
// qwen3_5.cpp pooled DBuf (device_pool.h Pool()/ActivePool()).
class DBuf {
 public:
  DBuf(Dev d, DType dt, const std::vector<int64_t>& shape,
       const void* host = nullptr)
      : b_(&d.b) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    alloc_bytes_ = bytes_ == 0 ? 1 : bytes_;
    cap_ = ResolveDevicePoolPolicy(d).cap_bytes;
    pool_ = ActivePool();
    p_ = pool_->Get(*b_, alloc_bytes_);
    t_ = MakeTensor(p_, dt, d.q.device, shape);
    if (host != nullptr && bytes_ > 0) b_->Copy(d.q, p_, host, bytes_);
  }
  ~DBuf() { if (p_ != nullptr) pool_->Put(*b_, alloc_bytes_, p_, cap_); }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  DBuf(DBuf&& o) noexcept
      : b_(o.b_), pool_(o.pool_), p_(o.p_), bytes_(o.bytes_),
        alloc_bytes_(o.alloc_bytes_), cap_(o.cap_), t_(o.t_) {
    o.p_ = nullptr;
  }
  DBuf& operator=(DBuf&& o) noexcept {
    if (this != &o) {
      if (p_ != nullptr) pool_->Put(*b_, alloc_bytes_, p_, cap_);
      b_ = o.b_;
      pool_ = o.pool_;
      p_ = o.p_;
      bytes_ = o.bytes_;
      alloc_bytes_ = o.alloc_bytes_;
      cap_ = o.cap_;
      t_ = o.t_;
      o.p_ = nullptr;
    }
    return *this;
  }

  Tensor& t() { return t_; }
  const Tensor& t() const { return t_; }
  void* ptr() { return p_; }
  size_t bytes() const { return bytes_; }
  size_t alloc_bytes() const { return alloc_bytes_; }
  void Zero(Dev d) { b_->Memset(d.q, p_, 0, bytes_); }
  void Download(Dev d, void* host) {
    b_->Copy(d.q, host, p_, bytes_);
    b_->Synchronize(d.q);
  }
  // Relinquish the pool block WITHOUT returning it (dtor becomes a no-op); the
  // caller takes over the Pool().Put obligation for alloc_bytes().
  void* Release() {
    void* p = p_;
    p_ = nullptr;
    return p;
  }

 private:
  Backend* b_;
  DevicePool* pool_ = &Pool();
  void* p_ = nullptr;
  size_t bytes_ = 0;
  size_t alloc_bytes_ = 0;
  size_t cap_ = 0;
  Tensor t_;
};

}  // namespace dense_attn
}  // namespace vllm
