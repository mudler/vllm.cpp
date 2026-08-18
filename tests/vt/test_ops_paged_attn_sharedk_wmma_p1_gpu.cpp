// #785 P1 GPU product-seam witness. Calls vt::PagedAttention on kROCM.
// Does NOT launch PagedAttnPrefillSharedKWmma directly.
// Not registered with CTest. Run only via tests/scripts/run-785-p1.sh
// after Researcher GPU GO.
//
// Missing VT_785_P1_GPU, missing device, env conflict, or hash mismatch
// => exit 5 (fail closed). Runner treats 77/nonzero as P1 failure.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vt/sharedk_wmma_p1_fixture.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_runtime.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::PagedAttentionArgs;
using vt::Queue;
using vt::Tensor;
using namespace vt_785_p1;

namespace {

[[noreturn]] void FailClosed(const char* why) {
  std::fprintf(stderr, "\n*** P1 GPU FAIL CLOSED (exit 5) ***\n%s\n", why);
  std::exit(5);
}

const char* EnvOr(const char* key) {
  const char* e = std::getenv(key);
  return e ? e : "";
}

bool EnvIs1(const char* key) {
  const char* e = std::getenv(key);
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}
bool EnvIs0(const char* key) {
  const char* e = std::getenv(key);
  return e != nullptr && e[0] == '0' && e[1] == '\0';
}

Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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

struct DeviceBuf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t;
  DeviceBuf(Backend& b_, Queue& q, DType dt, const std::vector<int64_t>& shape, const void* host)
      : b(b_) {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    bytes = static_cast<size_t>(n) * vt::SizeOf(dt);
    p = b.Alloc(bytes == 0 ? 1 : bytes);
    if (host != nullptr) b.Copy(q, p, host, bytes);
    t = Contig(p, dt, Device{DeviceType::kROCM, 0}, shape);
  }
  ~DeviceBuf() {
    if (p) b.Free(p);
  }
  void Download(Queue& q, void* dst) {
    b.Copy(q, dst, p, bytes);
    b.Synchronize(q);
  }
};

}  // namespace

TEST_CASE("P1 product seam vt::PagedAttention d=256 SharedK") {
  if (std::getenv("VT_785_P1_GPU") == nullptr) {
    FailClosed("VT_785_P1_GPU unset");
  }
  if (!vt::rocm::DeviceAvailable()) {
    FailClosed("no ROCm device");
  }
  const char* outdir = std::getenv("VT_785_P1_OUT");
  if (outdir == nullptr || outdir[0] == '\0') {
    FailClosed("VT_785_P1_OUT unset");
  }

  if (EnvIs1("VT_ROCM_ATTN_CPU_REF") || EnvIs1("VT_CPU_REF")) {
    FailClosed("CPU-ref must be off");
  }
  if (EnvIs0("VT_ATTN_DECODE_OPT") || EnvIs0("VT_ATTN_DECODE_GQA") ||
      EnvIs0("VT_ATTN_PREFILL_FLASH_SHAREDK")) {
    FailClosed("decode-opt / decode-GQA / SharedK must be on");
  }
  if (!EnvIs1("VT_ATTN_DECODE_OPT") || !EnvIs1("VT_ATTN_DECODE_GQA") ||
      !EnvIs1("VT_ATTN_PREFILL_FLASH_SHAREDK")) {
    FailClosed("decode-opt / decode-GQA / SharedK must be explicitly 1");
  }
  const bool wmma1 = EnvIs1("VT_ATTN_PREFILL_SHAREDK_WMMA");
  const bool wmma0 = EnvIs0("VT_ATTN_PREFILL_SHAREDK_WMMA");
  if (wmma1 == wmma0) {
    FailClosed("VT_ATTN_PREFILL_SHAREDK_WMMA must be explicit 1 or 0");
  }

  const auto f = MakeFixture();
  const std::string qh = Sha256U16Le(f.q_bf16);
  const std::string kh = Sha256U16Le(f.k_bf16);
  const std::string vh = Sha256U16Le(f.v_bf16);
  if (qh != kQHash || kh != kKHash || vh != kVHash) {
    FailClosed("fixture Q/K/V SHA-256 mismatch vs preregistered");
  }
  {
    std::ofstream h(std::string(outdir) + "/fixture-hashes.txt");
    if (!h) FailClosed("cannot write fixture-hashes.txt");
    h << "q_bf16=" << qh << "\nk_bf16=" << kh << "\nv_bf16=" << vh << "\n";
  }
  {
    std::ofstream e(std::string(outdir) + "/env.txt");
    if (!e) FailClosed("cannot write env.txt");
    e << "VT_785_P1_GPU=" << EnvOr("VT_785_P1_GPU") << "\n"
      << "VT_ATTN_PREFILL_SHAREDK_WMMA=" << EnvOr("VT_ATTN_PREFILL_SHAREDK_WMMA") << "\n"
      << "VT_ATTN_PREFILL_FLASH_SHAREDK=" << EnvOr("VT_ATTN_PREFILL_FLASH_SHAREDK") << "\n"
      << "VT_ATTN_DECODE_OPT=" << EnvOr("VT_ATTN_DECODE_OPT") << "\n"
      << "VT_ATTN_DECODE_GQA=" << EnvOr("VT_ATTN_DECODE_GQA") << "\n"
      << "VT_ROCM_ATTN_CPU_REF=" << EnvOr("VT_ROCM_ATTN_CPU_REF") << "\n"
      << "VT_CPU_REF=" << EnvOr("VT_CPU_REF") << "\n";
  }

  const auto ref = Oracle(f);
  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();
  DeviceBuf dq(rocm, q, DType::kBF16, {kT, kHq, kD}, f.q_bf16.data());
  DeviceBuf dk(rocm, q, DType::kBF16, {f.num_blocks, kBlock, kHk, kD}, f.k_bf16.data());
  DeviceBuf dv(rocm, q, DType::kBF16, {f.num_blocks, kBlock, kHk, kD}, f.v_bf16.data());
  DeviceBuf dbt(rocm, q, DType::kI32, {1, f.num_blocks}, f.block_table.data());
  DeviceBuf dsl(rocm, q, DType::kI32, {1}, f.seq_lens.data());
  DeviceBuf dqsl(rocm, q, DType::kI32, {2}, f.qsl.data());
  DeviceBuf dout(rocm, q, DType::kBF16, {kT, kHq, kD}, nullptr);

  PagedAttentionArgs args{kScale, /*causal=*/true};
  args.window_size = vt::AttentionWindow{static_cast<int32_t>(kWindowLeft),
                                         static_cast<int32_t>(kWindowRight)};
  args.query_start_loc_host = f.qsl.data();
  args.max_seq_len = static_cast<int32_t>(kT);
  vt::PagedAttention(q, dout.t, dq.t, dk.t, dv.t, dbt.t, dsl.t, dqsl.t, args);

  std::vector<uint16_t> got(f.q_bf16.size(), 0);
  dout.Download(q, got.data());
  std::vector<float> got_f(got.size());
  for (size_t i = 0; i < got.size(); ++i) got_f[i] = Bf16BitsToF32(got[i]);
  const auto st = Score(got_f, ref);

  {
    std::ofstream o(std::string(outdir) + "/out.bf16", std::ios::binary);
    if (!o) FailClosed("cannot write out.bf16");
    o.write(reinterpret_cast<const char*>(got.data()),
            static_cast<std::streamsize>(got.size() * sizeof(uint16_t)));
  }
  {
    std::ofstream m(std::string(outdir) + "/metrics.txt");
    if (!m) FailClosed("cannot write metrics.txt");
    m << "max_abs=" << st.max_abs << "\ncorr=" << st.corr << "\nnonfinite=" << st.nonfinite
      << "\nviolations=" << st.violations << "\noracle_ok=" << (st.ok ? 1 : 0)
      << "\nn=" << got.size() << "\n";
  }
  if (!st.ok) FailClosed("oracle bar miss");
}
