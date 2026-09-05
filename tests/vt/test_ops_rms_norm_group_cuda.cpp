// `vt::RmsNormGroup` — THE CUDA ARM. Row MODEL-MM-QWEN4-EXP W6-CUDA-B, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT IS UNDER TEST, AND WHAT IT IS COMPARED AGAINST. Two things, and the
// distinction is the point of the file, exactly as
// `tests/vllm/models/test_qwen4_exp_cuda.cpp` states it for the W6-CUDA arms:
//
//   1. THE ORACLE. `vllm/models/qwen4_exp_hc_goldens.inc`, dumped by
//      `scripts/gen-qwen4-exp-hc-goldens.py`, which lifts `Qwen4ExpTextRMSNorm`
//      VERBATIM by line range out of transformers v5.16.0 (the lane pin,
//      sha256 77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459)
//      and EXECUTES it under torch. `tests/vt/test_ops_rms_norm_group.cpp` holds
//      the CPU arm to the same file, so the two arms answer to ONE oracle
//      instead of to each other.
//   2. THE CPU ARM, BITWISE, on identical inputs. A DEVICE ARM GATED ONLY
//      AGAINST ITSELF WOULD BE WORTHLESS, and a device arm gated only against a
//      4-case golden of 24-element rows would not see a defect those widths
//      cannot express.
//
// THE BOUND FOR (2) IS ZERO, AND THAT IS AN ANALYSIS AND NOT AN ASPIRATION. The
// op has no transcendental. The CUDA arm walks each group with ONE thread in the
// host's ascending index order, spells every multiply and add with
// `__fmul_rn`/`__fadd_rn` because the host provider is pinned to
// `-ffp-contract=off` while nvcc's `-fmad` is ON and unpinned, and takes its
// divide and square root from `__fdiv_rn`/`__frcp_rn`/`sqrtf`, all of which
// IEEE-754 requires to be correctly rounded on both sides. There is then no
// operation left that CAN differ, so anything but bitwise equality is a defect.
// **NEVER WIDEN THIS TO A TOLERANCE.** A tolerance is the wrong instrument here
// for the reason `cuda_qwen4_exp.cu` gives: a group extent read as a row, or a
// weight indexed per-group instead of per-row, lands well inside any epsilon
// anyone would write — and the SEPARATION cases below measure exactly how far
// outside a real defect sits.
//
// THE ORACLE BOUND IS NOT THE ARM-VS-ARM BOUND. `kTol` is 1e-5, which is what
// `tests/vt/test_ops_rms_norm_group.cpp` already justifies for THESE shapes
// against a golden that is an INDEPENDENT f32 computation out of torch. Holding
// a device arm to one ulp of a torch golden is a bound about the bound; the
// bitwise backstop is what makes the loose oracle bound harmless, because every
// oracle case below is ALSO compared to the CPU arm on the same input.
//
// SCOPE, HONESTLY. NOTHING IN PRODUCTION REACHES THIS KERNEL YET. This op's only
// non-test callers are three lines in
// `src/vllm/model_executor/models/qwen4_exp_ple_block.cpp` (:489, :499, :538),
// `ModelRegistry::Forward` is all-or-nothing, and at the time this file was
// written `EmbeddingKernelCuda` still refused a block-quantized table by name,
// so no `qwen4_exp` step can reach a CUDA queue. The op unblocks NO other model:
// it is a shared `vt::` seam and any model may call it, but the tree has exactly
// those three call sites. No token claim and no speed claim.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/max_abs_diff.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm_test::MaxAbsDiff;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormGroupArgs;
using vt::Tensor;

namespace {

#include "vllm/models/qwen4_exp_hc_goldens.inc"  // NOLINT — golden literals

// The same value the CPU suite justifies for these shapes, against a golden that
// is an independent f32 computation out of torch.
constexpr double kTol = 1e-5;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

// LOUD, because a silent skip on a CPU box is how a device arm goes un-gated for
// a release.
bool SkipNoCuda(const char* what) {
  if (HasCuda()) return false;
  std::printf("[SKIP] no CUDA backend: %s NOT exercised\n", what);
  return true;
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

// Narrow to `dt` and back, so both arms are handed the SAME bits at the reduced
// width. Without this the two arms would be compared on inputs that differ, and
// a bitwise claim over them would be meaningless.
std::vector<float> RoundTrip(const std::vector<float>& v, DType dt) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    switch (dt) {
      case DType::kF16: out[i] = vt::F16ToF32(vt::F32ToF16(v[i])); break;
      case DType::kBF16: out[i] = vt::BF16ToF32(vt::F32ToBF16(v[i])); break;
      default: out[i] = v[i]; break;
    }
  }
  return out;
}

// Pack an f32 vector into `dt`'s storage bytes.
std::vector<uint8_t> Pack(const std::vector<float>& v, DType dt) {
  std::vector<uint8_t> out(v.size() * vt::SizeOf(dt));
  for (size_t i = 0; i < v.size(); ++i) {
    if (dt == DType::kF32) {
      std::memcpy(out.data() + i * 4, &v[i], 4);
    } else if (dt == DType::kF16) {
      const uint16_t h = vt::F32ToF16(v[i]);
      std::memcpy(out.data() + i * 2, &h, 2);
    } else {
      const uint16_t h = vt::F32ToBF16(v[i]);
      std::memcpy(out.data() + i * 2, &h, 2);
    }
  }
  return out;
}

std::vector<float> Unpack(const std::vector<uint8_t>& b, size_t n, DType dt) {
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    if (dt == DType::kF32) {
      std::memcpy(&out[i], b.data() + i * 4, 4);
    } else {
      uint16_t h = 0;
      std::memcpy(&h, b.data() + i * 2, 2);
      out[i] = (dt == DType::kF16) ? vt::F16ToF32(h) : vt::BF16ToF32(h);
    }
  }
  return out;
}

const char* DName(DType d) {
  switch (d) {
    case DType::kF32: return "f32";
    case DType::kF16: return "f16";
    default: return "bf16";
  }
}

// Drive the op on either device over an arbitrary dtype triple. The RAW OUTPUT
// BYTES come back, so a bitwise comparison compares what was STORED and not a
// widened view of it.
std::vector<uint8_t> RunOp(DeviceType dev, const std::vector<float>& x,
                           const std::vector<float>& w, int64_t rows, int64_t width,
                           int64_t group_size, float eps, bool gemma, DType xdt, DType wdt,
                           DType odt) {
  RmsNormGroupArgs args;
  args.eps = eps;
  args.gemma = gemma;
  args.group_size = group_size;
  std::vector<uint8_t> xb = Pack(x, xdt);
  std::vector<uint8_t> wb = Pack(w, wdt);
  std::vector<uint8_t> ob(static_cast<size_t>(rows * width) * vt::SizeOf(odt), 0);
  if (dev == DeviceType::kCPU) {
    Queue q = CpuQ();
    Tensor tx = MakeTensor(xb.data(), xdt, Cpu(), {rows, width});
    Tensor tw = MakeTensor(wb.data(), wdt, Cpu(), {width});
    Tensor to = MakeTensor(ob.data(), odt, Cpu(), {rows, width});
    vt::RmsNormGroup(q, to, tx, tw, args);
    return ob;
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  DeviceTensor dx(b, qg.q, xdt, {rows, width}, xb.data());
  DeviceTensor dw(b, qg.q, wdt, {width}, wb.data());
  DeviceTensor doo(b, qg.q, odt, {rows, width});
  vt::RmsNormGroup(qg.q, doo.tensor(), dx.tensor(), dw.tensor(), args);
  b.Synchronize(qg.q);
  doo.Download(qg.q, ob.data());
  return ob;
}

// BOTH REPORTERS PRINT ON SUCCESS, not only on failure. doctest's INFO/CAPTURE
// are emitted only when an assertion fails, so a passing run of a numeric gate
// says nothing about HOW closely it passed — and "0 bytes differ" versus "one
// ulp" is the difference between this file's byte-identity claim and a weaker
// one. These numbers are the wave's evidence and have to survive a green run.
void CheckBitwise(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want,
                  const char* what) {
  REQUIRE(got.size() == want.size());
  size_t differ = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) ++differ;
  }
  std::printf("[MEASURED] %-46s BYTE gate: %zu/%zu bytes differ\n", what, differ, want.size());
  INFO(what << ": " << differ << " of " << want.size() << " STORED bytes differ from the CPU "
                                                          "arm (0 is the contract)");
  CHECK(differ == 0);
}

std::vector<float> RandomF32(size_t n, uint32_t seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// The local double-precision transcription of :167-178 with one flag per
// plausible single-character defect, copied in shape from the CPU suite. USED
// ONLY to measure how far each defect sits from the oracle. The op is never
// compared against it — a transcription cannot gate the function it transcribes.
struct Variant {
  bool full_row = false;
  bool no_fold = false;
  bool no_eps = false;
};

std::vector<double> Reference(const float* x, const float* w, int64_t rows, int64_t width,
                              int64_t group_size, double eps, Variant v) {
  const int64_t extent = v.full_row ? width : group_size;
  std::vector<double> out(static_cast<size_t>(rows * width), 0.0);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t g = 0; g < width / extent; ++g) {
      const int64_t base = g * extent;
      double sumsq = 0.0;
      for (int64_t j = 0; j < extent; ++j) {
        const double t = x[r * width + base + j];
        sumsq += t * t;
      }
      const double inv =
          1.0 / std::sqrt(sumsq / static_cast<double>(extent) + (v.no_eps ? 0.0 : eps));
      for (int64_t j = 0; j < extent; ++j) {
        const int64_t idx = base + j;
        const double wj = v.no_fold ? w[idx] : 1.0 + w[idx];
        out[static_cast<size_t>(r * width + idx)] = x[r * width + idx] * inv * wj;
      }
    }
  }
  return out;
}

double MaxAbs(const std::vector<float>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return m;
}

struct Case {
  std::string name;
  int64_t hidden, hc, T;
  float eps;
  const float* norm_w_hf;
  const float* hyper;
  const float* normed;
};

// hidden IS the group_size and hc*hidden the row width — `hc_norm =
// Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, ...)`
// (modeling_qwen4_exp.py:947), the same shape the three PLE norms take
// (:1138-1140).
const Case kCases[] = {
    {"A", 6, 4, 3, 1e-6f, kA_norm_w_hf, kA_hyper, kA_normed},
    {"B", 5, 3, 2, 1e-5f, kB_norm_w_hf, kB_hyper, kB_normed},
    {"C", 6, 4, 2, 1e-6f, kC_norm_w_hf, kC_hyper, kC_normed},
    // D is the eps case: `hyper_scale = 0.01`, so the mean square is ~1e-4 and
    // eps is 1% of it rather than 5e-7 of it.
    {"D", 6, 4, 2, 1e-6f, kD_norm_w_hf, kD_hyper, kD_normed},
};

}  // namespace

TEST_CASE("vt::RmsNormGroup is REGISTERED for kCUDA and refuses by name elsewhere") {
  // The portable CPU reference tier CANNOT rescue a missing CUDA arm — it is
  // gated on `Backend::DeviceMemoryIsHostAddressable()` and `CudaBackend` leaves
  // that at the base `false` (#844, #1435) — so an unregistered (op, device)
  // must THROW rather than fall through and dereference device pointers.
  CHECK_NOTHROW(vt::GetOp(vt::OpId::kRmsNormGroup, DeviceType::kCPU));
#ifdef VLLM_CPP_CUDA
  // This assertion is the RED this wave was written against: before
  // `src/vt/cuda/cuda_rms_norm_group.cu` existed it was `CHECK_THROWS`.
  CHECK_NOTHROW(vt::GetOp(vt::OpId::kRmsNormGroup, DeviceType::kCUDA));
#endif
  CHECK_THROWS(vt::GetOp(vt::OpId::kRmsNormGroup, DeviceType::kMETAL));
}

TEST_CASE("vt::RmsNormGroup CUDA reproduces the pinned oracle AND the CPU arm bitwise") {
  if (SkipNoCuda("vt::RmsNormGroup CUDA arm vs the transformers oracle")) return;
  for (const Case& c : kCases) {
    INFO("case ", c.name);
    const int64_t width = c.hidden * c.hc;
    const size_t n = static_cast<size_t>(c.T * width);
    const std::vector<float> x(c.hyper, c.hyper + n);
    const std::vector<float> w(c.norm_w_hf, c.norm_w_hf + width);
    const std::vector<uint8_t> gpu = RunOp(DeviceType::kCUDA, x, w, c.T, width, c.hidden,
                                           c.eps, true, DType::kF32, DType::kF32,
                                           DType::kF32);
    const std::vector<uint8_t> cpu = RunOp(DeviceType::kCPU, x, w, c.T, width, c.hidden, c.eps,
                                           true, DType::kF32, DType::kF32, DType::kF32);
    const std::vector<float> got = Unpack(gpu, n, DType::kF32);
    const double d = MaxAbsDiff(got, c.normed, n);
    std::printf("[MEASURED] rms_norm_group case %-4s vs ORACLE max|diff| = %.9g  bound = %g\n",
                c.name.c_str(), d, kTol);
    CHECK(d < kTol);
    // The backstop that makes the loose oracle bound harmless.
    CheckBitwise(gpu, cpu, ("rms_norm_group oracle case " + c.name).c_str());
  }
}

TEST_CASE("vt::RmsNormGroup CUDA: the goldens SEPARATE the group reduction from the row") {
  // The claim is about the FIXTURE, not the kernel: a full-row reduction must
  // land far outside kTol on every golden, or the case above could not tell a
  // grouped norm from an ungrouped one on a DEVICE either. Without this, a CUDA
  // kernel that ignored `group_size` would pass the oracle case above.
  if (SkipNoCuda("vt::RmsNormGroup CUDA group-vs-row separation")) return;
  for (const Case& c : kCases) {
    INFO("case ", c.name);
    const int64_t width = c.hidden * c.hc;
    const size_t n = static_cast<size_t>(c.T * width);
    const std::vector<double> wrong = Reference(c.hyper, c.norm_w_hf, c.T, width, c.hidden,
                                                c.eps, Variant{true, false, false});
    const std::vector<float> golden(c.normed, c.normed + n);
    const double sep = MaxAbs(golden, wrong);
    std::printf("[MEASURED] rms_norm_group case %-4s FULL-ROW separation = %.9g (bound %g)\n",
                c.name.c_str(), sep, kTol);
    CHECK(sep > kTol);
  }
}

TEST_CASE("vt::RmsNormGroup CUDA: the `+ 1` fold and eps, on the device") {
  if (SkipNoCuda("vt::RmsNormGroup CUDA fold/eps probes")) return;
  const Case& a = kCases[0];
  const Case& d = kCases[3];
  {
    // Dropping the `+ 1` is the #2218 defect: a gamma centred on 0 applied
    // without the fold scales the stream by ~0, which is a plausible tensor and
    // not a crash. The gemma=false arm of the CUDA kernel IS that defect's
    // output, so this measures the separation on the device rather than assuming
    // the flag is read.
    const int64_t width = a.hidden * a.hc;
    const size_t n = static_cast<size_t>(a.T * width);
    const std::vector<float> x(a.hyper, a.hyper + n);
    const std::vector<float> w(a.norm_w_hf, a.norm_w_hf + width);
    const std::vector<float> folded = Unpack(
        RunOp(DeviceType::kCUDA, x, w, a.T, width, a.hidden, a.eps, true, DType::kF32,
              DType::kF32, DType::kF32), n, DType::kF32);
    const std::vector<float> unfolded = Unpack(
        RunOp(DeviceType::kCUDA, x, w, a.T, width, a.hidden, a.eps, false, DType::kF32,
              DType::kF32, DType::kF32), n, DType::kF32);
    double sep = 0.0;
    for (size_t i = 0; i < n; ++i) {
      sep = std::max(sep, std::fabs(static_cast<double>(folded[i]) - unfolded[i]));
    }
    std::printf("[MEASURED] rms_norm_group CUDA gemma-fold separation = %.9g\n", sep);
    CHECK(sep > kTol);  // the flag is READ on the device
    CHECK(MaxAbsDiff(folded, a.normed, n) < kTol);
  }
  {
    // eps is scale-dependent and a previous wave got this wrong: at case A's
    // `hyper_scale = 1.7` the mean square is O(1) and an eps of 1e-6 moves the
    // answer by 4.1e-6, BELOW kTol, so an eps probe run there is a mute switch.
    // Case D exists for this — the generator says so in its own comment.
    const int64_t width = d.hidden * d.hc;
    const size_t n = static_cast<size_t>(d.T * width);
    const std::vector<double> no_eps = Reference(d.hyper, d.norm_w_hf, d.T, width, d.hidden,
                                                 d.eps, Variant{false, false, true});
    const std::vector<float> golden(d.normed, d.normed + n);
    const double sep = MaxAbs(golden, no_eps);
    std::printf("[MEASURED] rms_norm_group case D DROP-EPS separation = %.9g\n", sep);
    CHECK(sep > kTol);
    const std::vector<float> got = Unpack(
        RunOp(DeviceType::kCUDA, std::vector<float>(d.hyper, d.hyper + n),
              std::vector<float>(d.norm_w_hf, d.norm_w_hf + width), d.T, width, d.hidden,
              d.eps, true, DType::kF32, DType::kF32, DType::kF32), n, DType::kF32);
    CHECK(MaxAbsDiff(got, d.normed, n) < kTol);
  }
}

TEST_CASE("vt::RmsNormGroup CUDA agrees with the CPU arm BITWISE on every admitted dtype") {
  if (SkipNoCuda("vt::RmsNormGroup CUDA dtype matrix")) return;
  // The kernel carries the dtype as a RUNTIME TAG rather than a template
  // parameter (its header argues why), so every operand/dtype pairing goes
  // through ONE kernel and a wrong tag mapping is invisible to an f32-only gate.
  // 3 x 3 x 2 = 18 triples.
  const DType kAll[3] = {DType::kF32, DType::kBF16, DType::kF16};
  const DType kOut[2] = {DType::kF32, DType::kBF16};
  constexpr int64_t T = 5, width = 24, group = 6;
  const std::vector<float> x = RandomF32(static_cast<size_t>(T * width), 11u);
  const std::vector<float> w = RandomF32(static_cast<size_t>(width), 12u);
  for (DType xdt : kAll) {
    for (DType wdt : kAll) {
      for (DType odt : kOut) {
        CAPTURE(DName(xdt));
        CAPTURE(DName(wdt));
        CAPTURE(DName(odt));
        // Both arms must be handed the SAME bits; the pack/unpack round trip is
        // what makes that true rather than assumed.
        const std::vector<float> xr = RoundTrip(x, xdt);
        const std::vector<float> wr = RoundTrip(w, wdt);
        char nm[96];
        std::snprintf(nm, sizeof nm, "rms_norm_group x=%s w=%s out=%s", DName(xdt), DName(wdt),
                      DName(odt));
        CheckBitwise(RunOp(DeviceType::kCUDA, xr, wr, T, width, group, 1e-6f, true, xdt, wdt,
                           odt),
                     RunOp(DeviceType::kCPU, xr, wr, T, width, group, 1e-6f, true, xdt, wdt,
                           odt),
                     nm);
      }
    }
  }
}

TEST_CASE("vt::RmsNormGroup CUDA: many rows and many groups, still bitwise") {
  if (SkipNoCuda("vt::RmsNormGroup CUDA wide fixture")) return;
  // The goldens are 24- and 15-element rows over 2 or 3 tokens. This fixture
  // widens that to a group extent that crosses a warp, which separates "the
  // indexing is right at a non-warp-multiple extent" from "the four golden rows
  // happened to fit one block".
  //
  // **THE SENTENCE THAT STOOD HERE WAS FALSE AND IS CORRECTED.** It claimed this
  // fixture is "4096 threads' worth of (row, group) pairs". It is 37 x 7 = 259
  // pairs, which is ONE block of 256 threads plus three, and it therefore takes
  // exactly ONE trip of the kernel's grid-stride loop. The case below is the one
  // that crosses it; this one covers extent, not scale.
  constexpr int64_t T = 37, group = 33, groups = 7, width = group * groups;
  const std::vector<float> x = RandomF32(static_cast<size_t>(T * width), 21u, -8.0f, 8.0f);
  const std::vector<float> w = RandomF32(static_cast<size_t>(width), 22u);
  CheckBitwise(RunOp(DeviceType::kCUDA, x, w, T, width, group, 1e-6f, true, DType::kF32,
                     DType::kF32, DType::kF32),
               RunOp(DeviceType::kCPU, x, w, T, width, group, 1e-6f, true, DType::kF32,
                     DType::kF32, DType::kF32),
               "rms_norm_group 37x231 group=33");
}

TEST_CASE("vt::RmsNormGroup CUDA: the GRID STRIDE takes more than one trip") {
  if (SkipNoCuda("vt::RmsNormGroup CUDA grid stride")) return;
  // The launcher caps the grid at 4096 blocks of 256 threads, so the kernel's
  // `idx += gridDim.x * blockDim.x` loop runs once for any fixture under
  // 1,048,576 (row, group) pairs. EVERY other case in this file is under it --
  // the goldens are 12 and 24 pairs, and the "wide" case above is 259 -- so a
  // collapsed stride would have been invisible to all of them.
  //
  // This fixture is 4200 x 256 = 1,075,200 pairs, so the second trip is
  // mandatory and the last 26,624 pairs are written only by it.
  constexpr int64_t T = 4200, group = 4, groups = 256, width = group * groups;  // width 1024
  constexpr int64_t pairs = T * groups;
  static_assert(pairs > 4096 * 256, "the fixture must cross the grid cap");
  const std::vector<float> x = RandomF32(static_cast<size_t>(T * width), 51u, -4.0f, 4.0f);
  const std::vector<float> w = RandomF32(static_cast<size_t>(width), 52u);
  const std::vector<uint8_t> gpu = RunOp(DeviceType::kCUDA, x, w, T, width, group, 1e-6f,
                                         true, DType::kF32, DType::kF32, DType::kF32);
  const std::vector<uint8_t> cpu = RunOp(DeviceType::kCPU, x, w, T, width, group, 1e-6f, true,
                                         DType::kF32, DType::kF32, DType::kF32);
  std::printf("[MEASURED] rms_norm_group grid stride: %lld pairs over a %d-pair cap\n",
              static_cast<long long>(pairs), 4096 * 256);
  CheckBitwise(gpu, cpu, "rms_norm_group 1075200 pairs / grid stride");
  // A collapsed stride leaves the tail as allocated, and two zeros agree. Assert
  // the tail was actually written.
  const float* g = reinterpret_cast<const float*>(gpu.data());
  double tail = 0.0;
  for (int64_t i = 4096 * 256 * group; i < T * width; ++i) {
    tail = std::max(tail, std::fabs(static_cast<double>(g[i])));
  }
  std::printf("[MEASURED] rms_norm_group grid-stride tail max = %.9g (0 means never written)\n",
              tail);
  CHECK(tail > 0.0);
}

TEST_CASE("vt::RmsNormGroup CUDA: refusals name the caller's mistake") {
  if (SkipNoCuda("vt::RmsNormGroup CUDA refusals")) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard qg(b);
  constexpr int64_t T = 2, width = 12;
  DeviceTensor dx(b, qg.q, DType::kF32, {T, width});
  DeviceTensor dw(b, qg.q, DType::kF32, {width});
  DeviceTensor doo(b, qg.q, DType::kF32, {T, width});
  RmsNormGroupArgs args;
  args.gemma = true;
  // ZERO IS REFUSED rather than meaning "the whole row". Whole-row is what
  // `vt::RmsNorm` already does; letting a forgotten field silently produce it
  // would make the single most likely caller mistake indistinguishable from
  // success. The refusal is the DISPATCHER's and must hold on a CUDA queue too.
  args.group_size = 0;
  CHECK_THROWS(vt::RmsNormGroup(qg.q, doo.tensor(), dx.tensor(), dw.tensor(), args));
  args.group_size = 5;  // does not divide 12; upstream raises at :164-165
  CHECK_THROWS(vt::RmsNormGroup(qg.q, doo.tensor(), dx.tensor(), dw.tensor(), args));
  args.group_size = 6;
  args.eps = 0.0f;
  CHECK_THROWS(vt::RmsNormGroup(qg.q, doo.tensor(), dx.tensor(), dw.tensor(), args));
}
