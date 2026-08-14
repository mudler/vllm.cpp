// OUR arm of the #442 standalone Marlin harness.
//
// Mirrors scripts/marlin-moe-standalone.py exactly: same 35B-A3B decode shapes
// (hidden 2048, moe_intermediate 512, E=256, top_k=8, moe_block_size 8), same
// gate_up GEMM, same expert-pool control over the occupied block count. Prints
// us/call and us/block so our plateau can be laid against upstream's 5.2-5.5.
//
// Not a test: no assertions, no goldens. It measures the kernel only.
//
// RUN IT UNDER THE BOX LOCK: `flock $HOME/gpu.lock ...`, NOT /tmp/gpu.lock,
// which coordinates with nothing. `nvidia-smi` showing no compute apps does
// not mean the GPU is unreserved, so check `fuser -v $HOME/gpu.lock` first.
// Absolute timings taken unlocked are upper bounds; only interleaved RATIOS
// survive contention.
//
// NOT WIRED INTO ANY BUILD TARGET (#442). Nothing compiles this file, so it
// carries no -Werror and no CI, and it will rot against
// vt::MoeGroupedGemmNvfp4Marlin's signature. The recorded measurements were
// taken from an out-of-tree build. Wiring it into examples/CMakeLists.txt the
// way benchmarks/vulkan_gemm_ab.cpp is wired is owed.
//
// Its routing RNG is a DIFFERENT stream from the python arm's, so the two
// arms occupy different block counts at the same --experts pool. Comparisons
// between them are NORMALISED by blocks, not matched on them; neither arm can
// yet take an externally supplied routing tensor.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/cuda/marlin_repack.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

class Dev {
 public:
  Dev(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
      const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeT(p_, dt, Gpu(), shape);
  }
  ~Dev() { b_.Free(p_); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  Tensor& tensor() { return t_; }
  void* ptr() { return p_; }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

int IntArg(int argc, char** argv, const char* name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const int pool_arg = IntArg(argc, argv, "--experts", 0);
  const int iters = IntArg(argc, argv, "--iters", 80);
  const int warmup = IntArg(argc, argv, "--warmup", 20);
  const int M = IntArg(argc, argv, "--m", 9);
  const int zero_ws = IntArg(argc, argv, "--zero-ws", 1);

  const int E = 256, K = 2048, N = 512, top_k = 8;
  const int pool = pool_arg > 0 ? pool_arg : E;
  const int size_n = 2 * N;  // gate_up
  const int size_k = K;

  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  Queue q{Gpu(), nullptr};
  void* stream = nullptr;
  const int dev_id = 0;

  // Weights: random packed nibbles, repacked per expert into Marlin layout.
  // Marlin's runtime is data independent, so random bits time like real ones.
  std::mt19937 rng(1234);
  const size_t raw_bytes = static_cast<size_t>(size_n) * size_k / 2;
  std::vector<uint8_t> raw(raw_bytes);
  for (auto& x : raw) x = static_cast<uint8_t>(rng() & 0xFF);
  const size_t scale_bytes = static_cast<size_t>(size_n) * size_k / 16;
  std::vector<uint8_t> raw_s(scale_bytes);
  for (auto& x : raw_s) x = 0x38;  // fp8-e4m3 ~ 0.5, safely positive

  Dev staging(b, q, DType::kI8, {size_n, size_k / 2}, raw.data());
  Dev staging_s(b, q, DType::kI8, {size_n, size_k / 16}, raw_s.data());

  Dev wq(b, q, DType::kI32, {E, size_k / 16, size_n * 2});
  Dev sc(b, q, DType::kI8, {E, size_k / 16, size_n});
  const float sf = 1.0f;
  std::vector<float> gs(static_cast<size_t>(E),
                        vt::cuda::MarlinNvfp4ProcessGlobalScale(1.0f, sf));

  const size_t wq_expert_words = static_cast<size_t>(size_n) * size_k / 2 / 4;
  for (int e = 0; e < E; ++e) {
    vt::cuda::MarlinRepackExpertWeight(
        stream, dev_id,
        static_cast<uint32_t*>(wq.ptr()) + static_cast<size_t>(e) * wq_expert_words,
        static_cast<const uint8_t*>(staging.ptr()), size_k, size_n);
    vt::cuda::MarlinProcessExpertScales(
        stream, static_cast<const uint8_t*>(staging_s.ptr()),
        static_cast<uint8_t*>(sc.ptr()) + static_cast<size_t>(e) * scale_bytes,
        size_k, size_n, sf);
  }
  b.Synchronize(q);
  Dev dgs(b, q, DType::kF32, {E}, gs.data());

  // Routing, drawn from `pool` distinct experts -- the block-count control.
  const int P = M * top_k;
  std::vector<int32_t> topk_ids(static_cast<size_t>(P));
  std::vector<float> topk_w(static_cast<size_t>(P), 1.0f);
  for (int i = 0; i < P; ++i)
    topk_ids[static_cast<size_t>(i)] = static_cast<int32_t>(rng() % static_cast<unsigned>(pool));

  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(M, top_k, E);
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(M, top_k, E, block, &max_tok, &max_blk);
  Dev dtid(b, q, DType::kI32, {M, top_k}, topk_ids.data());
  Dev dtw(b, q, DType::kF32, {M, top_k}, topk_w.data());
  Dev sorted_ids(b, q, DType::kI32, {max_tok});
  Dev expert_ids(b, q, DType::kI32, {max_blk});
  Dev num_pad(b, q, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(stream, static_cast<const int32_t*>(dtid.ptr()), M,
                                    top_k, E, block,
                                    static_cast<int32_t*>(sorted_ids.ptr()),
                                    static_cast<int32_t*>(expert_ids.ptr()),
                                    static_cast<int32_t*>(num_pad.ptr()));
  b.Synchronize(q);
  int32_t past = 0;
  b.Copy(q, &past, num_pad.ptr(), sizeof(int32_t));
  b.Synchronize(q);

  const int sms = vt::cuda::MarlinDeviceSms(dev_id);
  Dev ws(b, q, DType::kI32, {sms * 4});
  Dev dact(b, q, DType::kBF16, {M, K});
  Dev dout(b, q, DType::kBF16, {P, size_n});

  vt::MoeMarlinArgs args{};
  args.moe_block_size = block;
  args.top_k = top_k;
  args.size_m = M;
  args.size_n = size_n;
  args.size_k = size_k;
  args.mul_topk_weights = false;

  // vLLM's arm does NOT re-zero the workspace per call (the kernel leaves it
  // reset), so timing ours WITH a per-call memset adds a launch upstream never
  // pays. --zero-ws 0 removes that asymmetry.
  b.Memset(q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  auto once = [&]() {
    if (zero_ws) b.Memset(q, ws.ptr(), 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
    vt::MoeGroupedGemmNvfp4Marlin(q, dout.tensor(), dact.tensor(), wq.tensor(),
                                  sc.tensor(), dgs.tensor(), ws.tensor(),
                                  sorted_ids.tensor(), expert_ids.tensor(),
                                  num_pad.tensor(), dtw.tensor(), args);
  };

  for (int i = 0; i < warmup; ++i) once();
  b.Synchronize(q);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) once();
  b.Synchronize(q);
  const auto t1 = std::chrono::steady_clock::now();

  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  const int blocks = past / block;
  std::printf("OURS gate_up M=%d pool=%d zero_ws=%d blocks=%d us_per_call=%.3f us_per_block=%.4f\n",
              M, pool, zero_ws, blocks, us, us / (blocks > 0 ? blocks : 1));
  return 0;
}
