// vllm.cpp original (parity harness); no upstream mirror.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "npy.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace fs = std::filesystem;
using nlohmann::json;
using vt::Backend;
using vt::DType;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

struct Loaded {
  parity::NpyArray raw;
  DType dtype;
  Tensor tensor;  // views raw.data (host memory)
};

DType DtypeFromName(const std::string& n) {
  if (n == "f32") return DType::kF32;
  if (n == "bf16") return DType::kBF16;
  if (n == "f16") return DType::kF16;
  if (n == "i32") return DType::kI32;
  if (n == "i64") return DType::kI64;
  throw std::runtime_error("unknown dtype " + n);
}

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t acc = 1;
  for (int d = t.rank - 1; d >= 0; --d) {
    t.shape[d] = shape[static_cast<size_t>(d)];
    t.stride[d] = acc;
    acc *= t.shape[d];
  }
  return t;
}

Loaded LoadTensor(const fs::path& dir, const json& spec) {
  const std::string file = spec["file"].get<std::string>();
  Loaded l{parity::LoadNpy((dir / file).string()),
           DtypeFromName(spec["dtype"].get<std::string>()), {}};
  std::vector<int64_t> shape = spec["shape"].get<std::vector<int64_t>>();
  if (shape.size() < 1 || shape.size() > 4)
    throw std::runtime_error("rank out of range for vt::Tensor: " + file +
                             " has rank " + std::to_string(shape.size()));
  // Cross-check the npy header against the manifest: shape must match
  // element-wise, and the descr element size must match the manifest dtype.
  if (l.raw.shape != shape) {
    auto fmt = [](const std::vector<int64_t>& s) {
      std::ostringstream os;
      os << "[";
      for (size_t i = 0; i < s.size(); ++i) os << (i ? "," : "") << s[i];
      os << "]";
      return os.str();
    };
    throw std::runtime_error("shape mismatch for " + file + ": npy header " +
                             fmt(l.raw.shape) + " vs manifest " + fmt(shape));
  }
  // descr is e.g. "<f4": element size is the digits after endian + type char.
  size_t npy_esize = 0;
  for (size_t i = 2; i < l.raw.dtype.size(); ++i)
    npy_esize = npy_esize * 10 + static_cast<size_t>(l.raw.dtype[i] - '0');
  if (npy_esize != vt::SizeOf(l.dtype))
    throw std::runtime_error(
        "dtype size mismatch for " + file + ": npy descr '" + l.raw.dtype +
        "' (" + std::to_string(npy_esize) + " bytes) vs manifest dtype '" +
        spec["dtype"].get<std::string>() + "' (" +
        std::to_string(vt::SizeOf(l.dtype)) + " bytes)");
  l.tensor = MakeTensor(l.raw.data.data(), l.dtype, Cpu(), shape);
  return l;
}

float AsF32(const Tensor& t, int64_t i) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[i];
    case DType::kBF16: return vt::BF16ToF32(t.Ptr<uint16_t>()[i]);
    case DType::kF16: return vt::F16ToF32(t.Ptr<uint16_t>()[i]);
    default: throw std::runtime_error("AsF32: bad dtype");
  }
}

// Returns the first-failure description, or nullopt if all elements match.
// Non-finite-loud: NaN or Inf anywhere (got or want) is a failure — `>`
// comparisons are silently false for NaN, so the check is `!(diff <= tol)`
// plus !isfinite.
std::optional<std::string> CompareTensors(const Tensor& got, const Tensor& want,
                                          double atol, double rtol) {
  if (got.Numel() != want.Numel()) {
    return "element count mismatch: got " + std::to_string(got.Numel()) +
           " want " + std::to_string(want.Numel());
  }
  for (int64_t i = 0; i < got.Numel(); ++i) {
    double g = AsF32(got, i), w = AsF32(want, i);
    double tol = atol + rtol * std::abs(w);
    const bool g_bad = !std::isfinite(g), w_bad = !std::isfinite(w);
    if (g_bad || w_bad || !(std::abs(g - w) <= tol)) {
      std::ostringstream os;
      os << "[" << i << "]: got " << g << " want " << w;
      if (g_bad || w_bad) {
        os << " (non-finite in " << (g_bad ? "got" : "want") << ")";
      } else {
        os << " (tol " << tol << ")";
      }
      return os.str();
    }
  }
  return std::nullopt;
}

void RequireMatch(const std::string& name, const Tensor& got, const Tensor& want,
                  double atol, double rtol) {
  auto err = CompareTensors(got, want, atol, rtol);
  if (err) FAIL(name << *err);
  // Opt-in margin report (VLLM_PARITY_PRINT_MARGINS=1): max abs diff per tensor,
  // to judge how much headroom a passing case has under its tolerance.
  if (std::getenv("VLLM_PARITY_PRINT_MARGINS") != nullptr) {
    double max_abs = 0.0;
    for (int64_t i = 0; i < got.Numel(); ++i)
      max_abs = std::max(max_abs, std::abs(static_cast<double>(AsF32(got, i)) - AsF32(want, i)));
    std::printf("margin %s: max_abs_diff=%.3e (atol=%.0e rtol=%.0e)\n", name.c_str(), max_abs,
                atol, rtol);
  }
}

// Buffer allocated on the pass's device through its Backend, viewed as a
// Tensor. On the CPU pass the backend is the CPU backend (Alloc/Copy are
// plain malloc/memcpy), so both passes run the exact same runner code; the
// CUDA pass gets real h2d/d2h transfers on the queue's stream. Downloads
// synchronize the queue before returning so the host buffer is readable.
class DeviceBuf {
 public:
  DeviceBuf(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
            const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, q.device, shape);
  }
  ~DeviceBuf() { b_.Free(p_); }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  Tensor& tensor() { return t_; }
  // Downloads into a host tensor of the same dtype/shape (backed by `store`).
  Tensor Download(Queue& q, std::vector<uint8_t>& store) {
    store.resize(bytes_);
    b_.Copy(q, store.data(), p_, bytes_);
    b_.Synchronize(q);
    Tensor host = t_;
    host.data = store.data();
    host.device = Cpu();
    return host;
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

std::vector<int64_t> ShapeOf(const Tensor& t) {
  return std::vector<int64_t>(t.shape, t.shape + t.rank);
}

void RunRmsNorm(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto x = LoadTensor(dir, m["tensors"]["x"]);
  auto w = LoadTensor(dir, m["tensors"]["weight"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  DeviceBuf dx(b, q, x.dtype, ShapeOf(x.tensor), x.raw.data.data());
  DeviceBuf dw(b, q, w.dtype, ShapeOf(w.tensor), w.raw.data.data());
  DeviceBuf dout(b, q, DType::kF32, ShapeOf(x.tensor));
  vt::RmsNormArgs args{m["args"]["eps"].get<float>(), m["args"]["gemma"].get<bool>()};
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();
  if (m["args"]["fused_residual"].get<bool>()) {
    auto res = LoadTensor(dir, m["tensors"]["residual_in"]);
    auto res_want = LoadTensor(dir, m["tensors"]["residual_out"]);
    VT_CHECK(res.dtype == DType::kF32, "fused-residual runner requires f32 residual_in");
    // The op mutates the residual stream in place; run on a device copy and
    // copy it back for comparison.
    DeviceBuf dres(b, q, DType::kF32, ShapeOf(res.tensor), res.raw.data.data());
    vt::RmsNorm(q, dout.tensor(), dx.tensor(), dw.tensor(), args, &dres.tensor());
    std::vector<uint8_t> res_host;
    RequireMatch("residual", dres.Download(q, res_host), res_want.tensor, atol, rtol);
  } else {
    vt::RmsNorm(q, dout.tensor(), dx.tensor(), dw.tensor(), args);
  }
  std::vector<uint8_t> out_host;
  RequireMatch("out", dout.Download(q, out_host), want.tensor, atol, rtol);
}

void RunMatmul(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto a = LoadTensor(dir, m["tensors"]["a"]);
  auto bt = LoadTensor(dir, m["tensors"]["b"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  const int64_t rows = a.tensor.shape[0], cols = bt.tensor.shape[1];
  DeviceBuf da(b, q, a.dtype, ShapeOf(a.tensor), a.raw.data.data());
  DeviceBuf db(b, q, bt.dtype, ShapeOf(bt.tensor), bt.raw.data.data());
  DeviceBuf dout(b, q, DType::kF32, {rows, cols});
  vt::Matmul(q, dout.tensor(), da.tensor(), db.tensor());
  std::vector<uint8_t> out_host;
  RequireMatch("out", dout.Download(q, out_host), want.tensor,
               m["tol"]["atol"].get<double>(), m["tol"]["rtol"].get<double>());
}

void RunSiluAndMul(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto x = LoadTensor(dir, m["tensors"]["x"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  const int64_t t = x.tensor.shape[0], d = x.tensor.shape[1] / 2;
  DeviceBuf dx(b, q, x.dtype, ShapeOf(x.tensor), x.raw.data.data());
  DeviceBuf dout(b, q, DType::kF32, {t, d});
  vt::SiluAndMul(q, dout.tensor(), dx.tensor());
  std::vector<uint8_t> out_host;
  RequireMatch("out", dout.Download(q, out_host), want.tensor,
               m["tol"]["atol"].get<double>(), m["tol"]["rtol"].get<double>());
}

void RunEmbedding(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto table = LoadTensor(dir, m["tensors"]["table"]);
  auto ids = LoadTensor(dir, m["tensors"]["ids"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  const int64_t t = ids.tensor.shape[0], h = table.tensor.shape[1];
  DeviceBuf dtab(b, q, table.dtype, ShapeOf(table.tensor), table.raw.data.data());
  DeviceBuf dids(b, q, ids.dtype, ShapeOf(ids.tensor), ids.raw.data.data());
  DeviceBuf dout(b, q, DType::kF32, {t, h});
  vt::Embedding(q, dout.tensor(), dtab.tensor(), dids.tensor());
  std::vector<uint8_t> out_host;
  RequireMatch("out", dout.Download(q, out_host), want.tensor,
               m["tol"]["atol"].get<double>(), m["tol"]["rtol"].get<double>());
}

// The dump stores q/k flattened [T, H*D] (upstream forward_native convention);
// vt::RopeNeox is in-place on f32 [T,H,D], so run on shaped device copies of
// the inputs and copy both back for comparison.
void RunRope(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto q_in = LoadTensor(dir, m["tensors"]["q_in"]);
  auto k_in = LoadTensor(dir, m["tensors"]["k_in"]);
  auto pos = LoadTensor(dir, m["tensors"]["positions"]);
  auto q_want = LoadTensor(dir, m["tensors"]["q_out"]);
  auto k_want = LoadTensor(dir, m["tensors"]["k_out"]);
  const int64_t t = q_in.tensor.shape[0];
  const int64_t hq = m["args"]["num_q_heads"].get<int64_t>();
  const int64_t hk = m["args"]["num_kv_heads"].get<int64_t>();
  const int64_t d = m["args"]["head_size"].get<int64_t>();
  VT_CHECK(q_in.dtype == DType::kF32 && k_in.dtype == DType::kF32,
           "rope runner requires f32 q_in/k_in (in-place copy buffers are f32)");
  VT_CHECK(q_in.tensor.Numel() == t * hq * d && k_in.tensor.Numel() == t * hk * d,
           "rope: q_in/k_in element counts do not match args num_heads*head_size");
  DeviceBuf dq(b, q, DType::kF32, {t, hq, d}, q_in.raw.data.data());
  DeviceBuf dk(b, q, DType::kF32, {t, hk, d}, k_in.raw.data.data());
  DeviceBuf dpos(b, q, pos.dtype, ShapeOf(pos.tensor), pos.raw.data.data());
  vt::RopeArgs args{m["args"]["base"].get<float>(), m["args"]["rotary_dim"].get<int>()};
  vt::RopeNeox(q, dq.tensor(), dk.tensor(), dpos.tensor(), args);
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();
  std::vector<uint8_t> q_host, k_host;
  RequireMatch("q_out", dq.Download(q, q_host), q_want.tensor, atol, rtol);
  RequireMatch("k_out", dk.Download(q, k_host), k_want.tensor, atol, rtol);
}

// Runs every golden case on `dev` and returns how many ran. Both passes use
// the same manifests and the same tolerances — the committed goldens are the
// bar for every backend.
int RunGoldenPass(Device dev) {
  fs::path root = PARITY_GOLDENS_DIR;
  REQUIRE(fs::exists(root));
  Backend& b = vt::GetBackend(dev.type);
  Queue q = b.CreateQueue();
  int cases = 0;
  for (const auto& entry : fs::directory_iterator(root)) {
    if (!entry.is_directory()) continue;
    fs::path mf = entry.path() / "manifest.json";
    // Non-op golden dirs (e.g. tokenizer_qwen36, owned by
    // test_tokenizer_parity) carry no manifest.json; the `cases >= 9` floor
    // in the callers still guards against op cases silently disappearing.
    if (!fs::exists(mf)) continue;
    json m = json::parse(std::ifstream(mf));
    std::string op = m["op"];
    INFO("case " << entry.path().filename().string());
    if (std::getenv("VLLM_PARITY_PRINT_MARGINS") != nullptr)
      std::printf("case %s (%s)\n", entry.path().filename().string().c_str(),
                  dev.type == DeviceType::kCUDA ? "cuda" : "cpu");
    if (op == "rmsnorm") {
      RunRmsNorm(b, q, entry.path(), m);
    } else if (op == "matmul") {
      RunMatmul(b, q, entry.path(), m);
    } else if (op == "silu_and_mul") {
      RunSiluAndMul(b, q, entry.path(), m);
    } else if (op == "embedding") {
      RunEmbedding(b, q, entry.path(), m);
    } else if (op == "rope") {
      RunRope(b, q, entry.path(), m);
    } else {
      FAIL("no runner for op '" << op << "' — add one before committing goldens");
    }
    ++cases;
  }
  b.DestroyQueue(q);
  return cases;
}

}  // namespace

TEST_CASE("CompareTensors is NaN- and Inf-loud and catches mismatches") {
  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor ta = Tensor::Contiguous(a, DType::kF32, Cpu(), {4});
  Tensor tb = Tensor::Contiguous(b, DType::kF32, Cpu(), {4});

  // Exact match → no error.
  CHECK(!CompareTensors(ta, tb, 1e-6, 1e-6).has_value());

  // Plain mismatch → error naming the offending index.
  b[2] = 3.5f;
  auto err = CompareTensors(ta, tb, 1e-6, 1e-6);
  REQUIRE(err.has_value());
  CHECK(err->find("[2]") != std::string::npos);
  b[2] = 3.0f;

  // NaN in got → error, reported explicitly (a plain |g-w| > tol comparator
  // is silently false for NaN and would pass).
  a[1] = std::nanf("");
  auto nan_got = CompareTensors(ta, tb, 1e-6, 1e-6);
  REQUIRE(nan_got.has_value());
  CHECK(nan_got->find("non-finite in got") != std::string::npos);
  a[1] = 2.0f;

  // NaN in want → also an error.
  b[3] = std::nanf("");
  auto nan_want = CompareTensors(ta, tb, 1e-6, 1e-6);
  REQUIRE(nan_want.has_value());
  CHECK(nan_want->find("non-finite in want") != std::string::npos);
  b[3] = 4.0f;

  // Inf in got → error (Inf-Inf would be NaN and Inf vs finite passes a plain
  // `>` check only by accident of tol; make it loud regardless).
  a[0] = INFINITY;
  auto inf_got = CompareTensors(ta, tb, 1e-6, 1e-6);
  REQUIRE(inf_got.has_value());
  CHECK(inf_got->find("non-finite in got") != std::string::npos);
  a[0] = 1.0f;

  // Inf in want → also an error.
  b[2] = INFINITY;
  auto inf_want = CompareTensors(ta, tb, 1e-6, 1e-6);
  REQUIRE(inf_want.has_value());
  CHECK(inf_want->find("non-finite in want") != std::string::npos);
}

TEST_CASE("op parity vs upstream goldens (CPU)") {
  int cases = RunGoldenPass(Cpu());
  CHECK(cases >= 9);
}

// Same cases, same tolerances, on the GPU: inputs are uploaded through the
// CUDA backend, ops run on a CUDA stream, outputs come back d2h for the host
// compare. Skips cleanly on CPU-only builds and on CUDA builds without a GPU
// (the CUDA registrar only registers kCUDA when a device is present).
TEST_CASE("op parity vs upstream goldens (CUDA)") {
  bool has_cuda = true;
  try {
    vt::GetBackend(DeviceType::kCUDA);
  } catch (const std::runtime_error&) {
    has_cuda = false;
  }
  if (!has_cuda) {
    MESSAGE("no CUDA backend registered; skipping CUDA parity pass");
    return;
  }
  int cases = RunGoldenPass(Device{DeviceType::kCUDA, 0});
  CHECK(cases >= 9);
}
