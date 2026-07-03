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
#include "vt/ops.h"

namespace fs = std::filesystem;
using nlohmann::json;
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
  Tensor tensor;  // views raw.data
};

DType DtypeFromName(const std::string& n) {
  if (n == "f32") return DType::kF32;
  if (n == "bf16") return DType::kBF16;
  if (n == "f16") return DType::kF16;
  if (n == "i32") return DType::kI32;
  if (n == "i64") return DType::kI64;
  throw std::runtime_error("unknown dtype " + n);
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
  Tensor t;
  t.data = l.raw.data.data();
  t.dtype = l.dtype;
  t.device = Cpu();
  t.rank = static_cast<int>(shape.size());
  int64_t acc = 1;
  for (int d = t.rank - 1; d >= 0; --d) {
    t.shape[d] = shape[static_cast<size_t>(d)];
    t.stride[d] = acc;
    acc *= t.shape[d];
  }
  l.tensor = t;
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
// NaN-loud: NaN anywhere (got or want) is a failure — `>` comparisons are
// silently false for NaN, so the check is `!(diff <= tol)` plus isnan.
std::optional<std::string> CompareTensors(const Tensor& got, const Tensor& want,
                                          double atol, double rtol) {
  if (got.Numel() != want.Numel()) {
    return "element count mismatch: got " + std::to_string(got.Numel()) +
           " want " + std::to_string(want.Numel());
  }
  for (int64_t i = 0; i < got.Numel(); ++i) {
    double g = AsF32(got, i), w = AsF32(want, i);
    double tol = atol + rtol * std::abs(w);
    const bool g_nan = std::isnan(g), w_nan = std::isnan(w);
    if (g_nan || w_nan || !(std::abs(g - w) <= tol)) {
      std::ostringstream os;
      os << "[" << i << "]: got " << g << " want " << w;
      if (g_nan || w_nan) {
        os << " (nan in " << (g_nan ? "got" : "want") << ")";
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

void RunRmsNorm(const fs::path& dir, const json& m) {
  auto x = LoadTensor(dir, m["tensors"]["x"]);
  auto w = LoadTensor(dir, m["tensors"]["weight"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  std::vector<float> outbuf(static_cast<size_t>(x.tensor.Numel()));
  Tensor out = Tensor::Contiguous(outbuf.data(), DType::kF32, Cpu(),
                                  {x.tensor.shape[0], x.tensor.shape[1]});
  vt::RmsNormArgs args{m["args"]["eps"].get<float>(), m["args"]["gemma"].get<bool>()};
  Queue q{Cpu(), nullptr};
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();
  if (m["args"]["fused_residual"].get<bool>()) {
    auto res = LoadTensor(dir, m["tensors"]["residual_in"]);
    auto res_want = LoadTensor(dir, m["tensors"]["residual_out"]);
    VT_CHECK(m["tensors"]["residual_in"]["dtype"].get<std::string>() == "f32",
             "fused-residual runner requires f32 residual_in");
    std::vector<float> resbuf(outbuf.size());
    VT_CHECK(res.raw.data.size() == resbuf.size() * sizeof(float),
             "residual_in byte size does not match the mutable copy buffer");
    std::memcpy(resbuf.data(), res.raw.data.data(), res.raw.data.size());
    Tensor rest = Tensor::Contiguous(resbuf.data(), DType::kF32, Cpu(),
                                     {x.tensor.shape[0], x.tensor.shape[1]});
    vt::RmsNorm(q, out, x.tensor, w.tensor, args, &rest);
    RequireMatch("residual", rest, res_want.tensor, atol, rtol);
  } else {
    vt::RmsNorm(q, out, x.tensor, w.tensor, args);
  }
  RequireMatch("out", out, want.tensor, atol, rtol);
}

void RunMatmul(const fs::path& dir, const json& m) {
  auto a = LoadTensor(dir, m["tensors"]["a"]);
  auto b = LoadTensor(dir, m["tensors"]["b"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  const int64_t rows = a.tensor.shape[0], cols = b.tensor.shape[1];
  std::vector<float> outbuf(static_cast<size_t>(rows * cols));
  Tensor out = Tensor::Contiguous(outbuf.data(), DType::kF32, Cpu(), {rows, cols});
  Queue q{Cpu(), nullptr};
  vt::Matmul(q, out, a.tensor, b.tensor);
  RequireMatch("out", out, want.tensor, m["tol"]["atol"].get<double>(),
               m["tol"]["rtol"].get<double>());
}

void RunSiluAndMul(const fs::path& dir, const json& m) {
  auto x = LoadTensor(dir, m["tensors"]["x"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  const int64_t t = x.tensor.shape[0], d = x.tensor.shape[1] / 2;
  std::vector<float> outbuf(static_cast<size_t>(t * d));
  Tensor out = Tensor::Contiguous(outbuf.data(), DType::kF32, Cpu(), {t, d});
  Queue q{Cpu(), nullptr};
  vt::SiluAndMul(q, out, x.tensor);
  RequireMatch("out", out, want.tensor, m["tol"]["atol"].get<double>(),
               m["tol"]["rtol"].get<double>());
}

void RunEmbedding(const fs::path& dir, const json& m) {
  auto table = LoadTensor(dir, m["tensors"]["table"]);
  auto ids = LoadTensor(dir, m["tensors"]["ids"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  const int64_t t = ids.tensor.shape[0], h = table.tensor.shape[1];
  std::vector<float> outbuf(static_cast<size_t>(t * h));
  Tensor out = Tensor::Contiguous(outbuf.data(), DType::kF32, Cpu(), {t, h});
  Queue q{Cpu(), nullptr};
  vt::Embedding(q, out, table.tensor, ids.tensor);
  RequireMatch("out", out, want.tensor, m["tol"]["atol"].get<double>(),
               m["tol"]["rtol"].get<double>());
}

// The dump stores q/k flattened [T, H*D] (upstream forward_native convention);
// vt::RopeNeox is in-place on f32 [T,H,D], so copy into shaped buffers first.
void RunRope(const fs::path& dir, const json& m) {
  auto q_in = LoadTensor(dir, m["tensors"]["q_in"]);
  auto k_in = LoadTensor(dir, m["tensors"]["k_in"]);
  auto pos = LoadTensor(dir, m["tensors"]["positions"]);
  auto q_want = LoadTensor(dir, m["tensors"]["q_out"]);
  auto k_want = LoadTensor(dir, m["tensors"]["k_out"]);
  const int64_t t = q_in.tensor.shape[0];
  const int64_t hq = m["args"]["num_q_heads"].get<int64_t>();
  const int64_t hk = m["args"]["num_kv_heads"].get<int64_t>();
  const int64_t d = m["args"]["head_size"].get<int64_t>();
  VT_CHECK(q_in.tensor.dtype == DType::kF32 && k_in.tensor.dtype == DType::kF32,
           "rope runner requires f32 q_in/k_in (in-place copy buffers are f32)");
  VT_CHECK(q_in.tensor.Numel() == t * hq * d && k_in.tensor.Numel() == t * hk * d,
           "rope: q_in/k_in element counts do not match args num_heads*head_size");
  std::vector<float> qbuf(static_cast<size_t>(t * hq * d));
  std::vector<float> kbuf(static_cast<size_t>(t * hk * d));
  std::memcpy(qbuf.data(), q_in.raw.data.data(), qbuf.size() * sizeof(float));
  std::memcpy(kbuf.data(), k_in.raw.data.data(), kbuf.size() * sizeof(float));
  Tensor qs = Tensor::Contiguous(qbuf.data(), DType::kF32, Cpu(), {t, hq, d});
  Tensor ks = Tensor::Contiguous(kbuf.data(), DType::kF32, Cpu(), {t, hk, d});
  vt::RopeArgs args{m["args"]["base"].get<float>(), m["args"]["rotary_dim"].get<int>()};
  Queue q{Cpu(), nullptr};
  vt::RopeNeox(q, qs, ks, pos.tensor, args);
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();
  RequireMatch("q_out", qs, q_want.tensor, atol, rtol);
  RequireMatch("k_out", ks, k_want.tensor, atol, rtol);
}

}  // namespace

TEST_CASE("CompareTensors is NaN-loud and catches mismatches") {
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
  CHECK(nan_got->find("nan") != std::string::npos);
  a[1] = 2.0f;

  // NaN in want → also an error.
  b[3] = std::nanf("");
  auto nan_want = CompareTensors(ta, tb, 1e-6, 1e-6);
  REQUIRE(nan_want.has_value());
  CHECK(nan_want->find("nan") != std::string::npos);
}

TEST_CASE("op parity vs upstream goldens") {
  fs::path root = PARITY_GOLDENS_DIR;
  REQUIRE(fs::exists(root));
  int cases = 0;
  for (const auto& entry : fs::directory_iterator(root)) {
    if (!entry.is_directory()) continue;
    fs::path mf = entry.path() / "manifest.json";
    REQUIRE(fs::exists(mf));
    json m = json::parse(std::ifstream(mf));
    std::string op = m["op"];
    INFO("case " << entry.path().filename().string());
    if (std::getenv("VLLM_PARITY_PRINT_MARGINS") != nullptr)
      std::printf("case %s\n", entry.path().filename().string().c_str());
    if (op == "rmsnorm") {
      RunRmsNorm(entry.path(), m);
    } else if (op == "matmul") {
      RunMatmul(entry.path(), m);
    } else if (op == "silu_and_mul") {
      RunSiluAndMul(entry.path(), m);
    } else if (op == "embedding") {
      RunEmbedding(entry.path(), m);
    } else if (op == "rope") {
      RunRope(entry.path(), m);
    } else {
      FAIL("no runner for op '" << op << "' — add one before committing goldens");
    }
    ++cases;
  }
  CHECK(cases >= 9);
}
