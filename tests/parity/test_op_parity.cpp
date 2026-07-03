// vllm.cpp original (parity harness); no upstream mirror.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
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
  Loaded l{parity::LoadNpy((dir / spec["file"].get<std::string>()).string()),
           DtypeFromName(spec["dtype"].get<std::string>()), {}};
  std::vector<int64_t> shape = spec["shape"].get<std::vector<int64_t>>();
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

void CompareTensors(const std::string& name, const Tensor& got, const Tensor& want,
                    double atol, double rtol) {
  REQUIRE(got.Numel() == want.Numel());
  for (int64_t i = 0; i < got.Numel(); ++i) {
    double g = AsF32(got, i), w = AsF32(want, i);
    double tol = atol + rtol * std::abs(w);
    if (std::abs(g - w) > tol) {
      FAIL(name << "[" << i << "]: got " << g << " want " << w << " (tol " << tol << ")");
    }
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
    std::vector<float> resbuf(outbuf.size());
    std::memcpy(resbuf.data(), res.raw.data.data(), resbuf.size() * 4);
    Tensor rest = Tensor::Contiguous(resbuf.data(), DType::kF32, Cpu(),
                                     {x.tensor.shape[0], x.tensor.shape[1]});
    vt::RmsNorm(q, out, x.tensor, w.tensor, args, &rest);
    CompareTensors("residual", rest, res_want.tensor, atol, rtol);
  } else {
    vt::RmsNorm(q, out, x.tensor, w.tensor, args);
  }
  CompareTensors("out", out, want.tensor, atol, rtol);
}

}  // namespace

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
    if (op == "rmsnorm") {
      RunRmsNorm(entry.path(), m);
    } else {
      FAIL("no runner for op '" << op << "' — add one before committing goldens");
    }
    ++cases;
  }
  CHECK(cases >= 4);
}
