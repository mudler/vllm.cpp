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
#include <set>
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

// Stores an f32 value into a host tensor at its declared dtype (f32 direct,
// bf16 rounded). Mirror of the kernel-side StoreF32 for runner-composed
// intermediates (shared-expert output).
void StoreAs(const Tensor& t, int64_t i, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[i] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[i] = vt::F32ToBF16(v); break;
    default: throw std::runtime_error("StoreAs: bad dtype");
  }
}

// Exact i32 comparison — router indices are an integer contract, never toleranced.
void RequireExactI32(const std::string& name, const Tensor& got, const Tensor& want) {
  REQUIRE(got.Numel() == want.Numel());
  for (int64_t i = 0; i < got.Numel(); ++i) {
    if (got.Ptr<int32_t>()[i] != want.Ptr<int32_t>()[i])
      FAIL(name << "[" << i << "]: got " << got.Ptr<int32_t>()[i] << " want "
                << want.Ptr<int32_t>()[i]);
  }
}

// Transposes a row-major [r,c] byte buffer to [c,r], preserving element bytes.
// Used to feed weight matrices (stored [out,in]) to vt::Matmul, which needs the
// b operand as [in,out] (the .T that upstream F.linear applies).
std::vector<uint8_t> TransposeBytes(const char* src, int64_t r, int64_t c, size_t esize) {
  std::vector<uint8_t> out(static_cast<size_t>(r * c) * esize);
  for (int64_t i = 0; i < r; ++i)
    for (int64_t j = 0; j < c; ++j)
      std::memcpy(out.data() + (static_cast<size_t>(j * r + i)) * esize,
                  src + (static_cast<size_t>(i * c + j)) * esize, esize);
  return out;
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

// --- GDN runners (M0.7). Goldens are pinned-oracle dumps (Task 1); shapes
// and tensor names are bound by the manifests under tests/parity/goldens/.

// causal_conv1d_fwd: x [T,C] token-major, conv_state [N,C,K-1] gathered
// per-sequence slices, query_start_loc [N+1], has_initial_state [N]. The op
// mutates conv_state in place → run on a device copy of conv_state_in and
// compare against conv_state_out. bias is null in all committed goldens
// (Qwen GDN conv has bias=False); a bias tensor is honored if present.
void RunCausalConv1dFwd(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto x = LoadTensor(dir, m["tensors"]["x"]);
  auto w = LoadTensor(dir, m["tensors"]["weight"]);
  auto st = LoadTensor(dir, m["tensors"]["conv_state_in"]);
  auto qsl = LoadTensor(dir, m["tensors"]["query_start_loc"]);
  auto his = LoadTensor(dir, m["tensors"]["has_initial_state"]);
  auto want_out = LoadTensor(dir, m["tensors"]["out"]);
  auto want_st = LoadTensor(dir, m["tensors"]["conv_state_out"]);
  DeviceBuf dx(b, q, x.dtype, ShapeOf(x.tensor), x.raw.data.data());
  DeviceBuf dw(b, q, w.dtype, ShapeOf(w.tensor), w.raw.data.data());
  DeviceBuf dst(b, q, st.dtype, ShapeOf(st.tensor), st.raw.data.data());
  DeviceBuf dqsl(b, q, qsl.dtype, ShapeOf(qsl.tensor), qsl.raw.data.data());
  DeviceBuf dhis(b, q, his.dtype, ShapeOf(his.tensor), his.raw.data.data());
  DeviceBuf dout(b, q, DType::kF32, ShapeOf(x.tensor));
  vt::CausalConv1dArgs args{m["args"]["activation"].get<std::string>() == "silu"};
  std::optional<Loaded> bias;
  std::optional<DeviceBuf> dbias;
  if (m["tensors"].contains("bias")) {
    bias.emplace(LoadTensor(dir, m["tensors"]["bias"]));
    dbias.emplace(b, q, bias->dtype, ShapeOf(bias->tensor), bias->raw.data.data());
  }
  vt::CausalConv1dFwd(q, dout.tensor(), dx.tensor(), dw.tensor(),
                      dbias ? &dbias->tensor() : nullptr, dst.tensor(), dqsl.tensor(),
                      dhis.tensor(), args);
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();
  std::vector<uint8_t> out_host, st_host;
  RequireMatch("out", dout.Download(q, out_host), want_out.tensor, atol, rtol);
  RequireMatch("conv_state_out", dst.Download(q, st_host), want_st.tensor, atol, rtol);
}

// causal_conv1d_update: x [B,C] one token per sequence, conv_state [B,C,K-1]
// rolled in place.
void RunCausalConv1dUpdate(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto x = LoadTensor(dir, m["tensors"]["x"]);
  auto w = LoadTensor(dir, m["tensors"]["weight"]);
  auto st = LoadTensor(dir, m["tensors"]["conv_state_in"]);
  auto want_out = LoadTensor(dir, m["tensors"]["out"]);
  auto want_st = LoadTensor(dir, m["tensors"]["conv_state_out"]);
  DeviceBuf dx(b, q, x.dtype, ShapeOf(x.tensor), x.raw.data.data());
  DeviceBuf dw(b, q, w.dtype, ShapeOf(w.tensor), w.raw.data.data());
  DeviceBuf dst(b, q, st.dtype, ShapeOf(st.tensor), st.raw.data.data());
  DeviceBuf dout(b, q, DType::kF32, ShapeOf(x.tensor));
  vt::CausalConv1dArgs args{m["args"]["activation"].get<std::string>() == "silu"};
  std::optional<Loaded> bias;
  std::optional<DeviceBuf> dbias;
  if (m["tensors"].contains("bias")) {
    bias.emplace(LoadTensor(dir, m["tensors"]["bias"]));
    dbias.emplace(b, q, bias->dtype, ShapeOf(bias->tensor), bias->raw.data.data());
  }
  vt::CausalConv1dUpdate(q, dout.tensor(), dx.tensor(), dw.tensor(),
                         dbias ? &dbias->tensor() : nullptr, dst.tensor(), args);
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();
  std::vector<uint8_t> out_host, st_host;
  RequireMatch("out", dout.Download(q, out_host), want_out.tensor, atol, rtol);
  RequireMatch("conv_state_out", dst.Download(q, st_host), want_st.tensor, atol, rtol);
}

void RunL2Norm(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto x = LoadTensor(dir, m["tensors"]["x"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  DeviceBuf dx(b, q, x.dtype, ShapeOf(x.tensor), x.raw.data.data());
  DeviceBuf dout(b, q, DType::kF32, ShapeOf(x.tensor));
  vt::L2Norm(q, dout.tensor(), dx.tensor(), vt::L2NormArgs{m["args"]["eps"].get<float>()});
  std::vector<uint8_t> out_host;
  RequireMatch("out", dout.Download(q, out_host), want.tensor,
               m["tol"]["atol"].get<double>(), m["tol"]["rtol"].get<double>());
}

void RunRmsNormGated(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto x = LoadTensor(dir, m["tensors"]["x"]);
  auto gate = LoadTensor(dir, m["tensors"]["gate"]);
  auto w = LoadTensor(dir, m["tensors"]["weight"]);
  auto want = LoadTensor(dir, m["tensors"]["out"]);
  // The op bakes in the only configuration Qwen GDN uses (gdn-semantics.md §5).
  VT_CHECK(m["args"]["norm_before_gate"].get<bool>(), "rmsnorm_gated golden must be norm-first");
  VT_CHECK(m["args"]["group_size"].is_null(), "rmsnorm_gated golden must be single-group");
  DeviceBuf dx(b, q, x.dtype, ShapeOf(x.tensor), x.raw.data.data());
  DeviceBuf dg(b, q, gate.dtype, ShapeOf(gate.tensor), gate.raw.data.data());
  DeviceBuf dw(b, q, w.dtype, ShapeOf(w.tensor), w.raw.data.data());
  DeviceBuf dout(b, q, DType::kF32, ShapeOf(x.tensor));
  vt::RmsNormGatedArgs args{m["args"]["eps"].get<float>(),
                            m["args"]["activation"].get<std::string>() == "sigmoid"};
  vt::RmsNormGated(q, dout.tensor(), dx.tensor(), dg.tensor(), dw.tensor(), args);
  std::vector<uint8_t> out_host;
  RequireMatch("out", dout.Download(q, out_host), want.tensor,
               m["tol"]["atol"].get<double>(), m["tol"]["rtol"].get<double>());
}

// gdn_prefill: q/k arrive already l2-normalized in these goldens
// (q_k_prenormalized, matching the upstream prefill path where
// fused_post_conv_prep normalizes before the chunk kernel). state is
// [N,Hv,Dv,Dk] f32 in/out in place.
void RunGdnPrefill(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto qi = LoadTensor(dir, m["tensors"]["q"]);
  auto k = LoadTensor(dir, m["tensors"]["k"]);
  auto v = LoadTensor(dir, m["tensors"]["v"]);
  auto g = LoadTensor(dir, m["tensors"]["g"]);
  auto beta = LoadTensor(dir, m["tensors"]["beta"]);
  auto st = LoadTensor(dir, m["tensors"]["state_in"]);
  auto qsl = LoadTensor(dir, m["tensors"]["query_start_loc"]);
  auto want_out = LoadTensor(dir, m["tensors"]["out"]);
  auto want_st = LoadTensor(dir, m["tensors"]["state_out"]);
  DeviceBuf dq(b, q, qi.dtype, ShapeOf(qi.tensor), qi.raw.data.data());
  DeviceBuf dk(b, q, k.dtype, ShapeOf(k.tensor), k.raw.data.data());
  DeviceBuf dv(b, q, v.dtype, ShapeOf(v.tensor), v.raw.data.data());
  DeviceBuf dg(b, q, g.dtype, ShapeOf(g.tensor), g.raw.data.data());
  DeviceBuf dbeta(b, q, beta.dtype, ShapeOf(beta.tensor), beta.raw.data.data());
  DeviceBuf dst(b, q, st.dtype, ShapeOf(st.tensor), st.raw.data.data());
  DeviceBuf dqsl(b, q, qsl.dtype, ShapeOf(qsl.tensor), qsl.raw.data.data());
  DeviceBuf dout(b, q, DType::kF32, ShapeOf(v.tensor));
  vt::GdnArgs args{m["args"]["scale"].get<float>()};
  vt::GdnPrefill(q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), dg.tensor(),
                 dbeta.tensor(), dst.tensor(), dqsl.tensor(), args);
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();
  std::vector<uint8_t> out_host, st_host;
  RequireMatch("out", dout.Download(q, out_host), want_out.tensor, atol, rtol);
  RequireMatch("state_out", dst.Download(q, st_host), want_st.tensor, atol, rtol);
}

// gdn_decode: the golden carries BOTH the raw set (q/k/a/b/A_log/dt_bias, as
// the fused upstream decode kernel consumes) and the derived set
// (q_l2/k_l2/g/beta, dumped from the pinned l2norm/gating math). The M0.7 op
// consumes the DERIVED set; the raw-set prep math (softplus gating from
// a/b/A_log/dt_bias, gdn-semantics.md §6) is M0.9 layer assembly. The runner
// exercises the decomposed chain M0.9 will use: vt::L2Norm on raw q/k
// (compared against the dumped q_l2/k_l2), then vt::GdnDecode on the
// normalized tensors + dumped g/beta.
void RunGdnDecode(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto qi = LoadTensor(dir, m["tensors"]["q"]);
  auto k = LoadTensor(dir, m["tensors"]["k"]);
  auto v = LoadTensor(dir, m["tensors"]["v"]);
  auto g = LoadTensor(dir, m["tensors"]["g"]);
  auto beta = LoadTensor(dir, m["tensors"]["beta"]);
  auto st = LoadTensor(dir, m["tensors"]["state_in"]);
  auto want_ql2 = LoadTensor(dir, m["tensors"]["q_l2"]);
  auto want_kl2 = LoadTensor(dir, m["tensors"]["k_l2"]);
  auto want_out = LoadTensor(dir, m["tensors"]["out"]);
  auto want_st = LoadTensor(dir, m["tensors"]["state_out"]);
  DeviceBuf dq(b, q, qi.dtype, ShapeOf(qi.tensor), qi.raw.data.data());
  DeviceBuf dk(b, q, k.dtype, ShapeOf(k.tensor), k.raw.data.data());
  DeviceBuf dv(b, q, v.dtype, ShapeOf(v.tensor), v.raw.data.data());
  DeviceBuf dg(b, q, g.dtype, ShapeOf(g.tensor), g.raw.data.data());
  DeviceBuf dbeta(b, q, beta.dtype, ShapeOf(beta.tensor), beta.raw.data.data());
  DeviceBuf dst(b, q, st.dtype, ShapeOf(st.tensor), st.raw.data.data());
  DeviceBuf dql2(b, q, DType::kF32, ShapeOf(qi.tensor));
  DeviceBuf dkl2(b, q, DType::kF32, ShapeOf(k.tensor));
  DeviceBuf dout(b, q, DType::kF32, ShapeOf(v.tensor));
  // In-kernel l2norm eps is 1e-6 (gdn-semantics.md §4).
  vt::L2NormArgs l2args{1e-6f};
  vt::L2Norm(q, dql2.tensor(), dq.tensor(), l2args);
  vt::L2Norm(q, dkl2.tensor(), dk.tensor(), l2args);
  vt::GdnArgs args{m["args"]["scale"].get<float>()};
  vt::GdnDecode(q, dout.tensor(), dql2.tensor(), dkl2.tensor(), dv.tensor(), dg.tensor(),
                dbeta.tensor(), dst.tensor(), args);
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();
  std::vector<uint8_t> ql2_host, kl2_host, out_host, st_host;
  RequireMatch("q_l2", dql2.Download(q, ql2_host), want_ql2.tensor, atol, rtol);
  RequireMatch("k_l2", dkl2.Download(q, kl2_host), want_kl2.tensor, atol, rtol);
  RequireMatch("out", dout.Download(q, out_host), want_out.tensor, atol, rtol);
  RequireMatch("state_out", dst.Download(q, st_host), want_st.tensor, atol, rtol);
}

// --- MoE runners (M0.8). Goldens are pinned-oracle dumps (Task 1);
// .agents/moe-semantics.md is the formula reference.

// moe_router_topk: logits [T,E] -> weights [T,K] f32 + ids [T,K] i32
// (moe-semantics.md §3). weights are toleranced, ids are an exact match.
void RunMoeRouterTopK(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto logits = LoadTensor(dir, m["tensors"]["logits"]);
  auto w_want = LoadTensor(dir, m["tensors"]["topk_weights"]);
  auto id_want = LoadTensor(dir, m["tensors"]["topk_ids"]);
  const int64_t t = logits.tensor.shape[0];
  const int top_k = m["args"]["top_k"].get<int>();
  const bool renorm = m["args"]["renormalize"].get<bool>();
  DeviceBuf dlog(b, q, logits.dtype, ShapeOf(logits.tensor), logits.raw.data.data());
  DeviceBuf dw(b, q, DType::kF32, {t, top_k});
  DeviceBuf did(b, q, DType::kI32, {t, top_k});
  vt::MoeRouterTopK(q, dw.tensor(), did.tensor(), dlog.tensor(),
                    vt::MoeRouterTopKArgs{top_k, renorm});
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();
  std::vector<uint8_t> w_host, id_host;
  RequireMatch("topk_weights", dw.Download(q, w_host), w_want.tensor, atol, rtol);
  RequireExactI32("topk_ids", did.Download(q, id_host), id_want.tensor);
}

// moe_block: full Qwen3NextSparseMoeBlock (moe-semantics.md §1-§6). Composes
// the router (gate F.linear -> MoeRouterTopK), the per-expert silu-mul MLP
// (Matmul + SiluAndMul + Matmul, §4), the shared expert with sigmoid gate (§5)
// and the weighted combine (MoeCombine, §6). Intermediates dumped by the oracle
// (router_logits/topk_*/routed_out/shared_out) are cross-checked; `out` is the
// layer-level contract. Intermediate buffers use the case (activation) dtype so
// the bf16 case mirrors the oracle's bf16 activation arithmetic.
void RunMoeBlock(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto x = LoadTensor(dir, m["tensors"]["x"]);
  auto gate_w = LoadTensor(dir, m["tensors"]["gate_w"]);
  auto w13 = LoadTensor(dir, m["tensors"]["w13"]);
  auto w2 = LoadTensor(dir, m["tensors"]["w2"]);
  auto wsgu = LoadTensor(dir, m["tensors"]["w_shared_gate_up"]);
  auto wsd = LoadTensor(dir, m["tensors"]["w_shared_down"]);
  auto wsg = LoadTensor(dir, m["tensors"]["w_shared_gate"]);
  auto rl_want = LoadTensor(dir, m["tensors"]["router_logits"]);
  auto tw_want = LoadTensor(dir, m["tensors"]["topk_weights"]);
  auto tid_want = LoadTensor(dir, m["tensors"]["topk_ids"]);
  auto routed_want = LoadTensor(dir, m["tensors"]["routed_out"]);
  auto shared_want = LoadTensor(dir, m["tensors"]["shared_out"]);
  auto out_want = LoadTensor(dir, m["tensors"]["out"]);

  const DType act = x.dtype;  // activation/model dtype (§2)
  const int64_t T = x.tensor.shape[0], H = x.tensor.shape[1];
  const int64_t E = m["args"]["num_experts"].get<int64_t>();
  const int64_t I = m["args"]["moe_intermediate"].get<int64_t>();
  const int64_t Is = m["args"]["shared_intermediate"].get<int64_t>();
  const int top_k = m["args"]["top_k"].get<int>();
  const bool renorm = m["args"]["renormalize"].get<bool>();
  const size_t es = vt::SizeOf(act);
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();

  DeviceBuf dx(b, q, act, {T, H}, x.raw.data.data());

  // Router: logits = x @ gate_w.T ; softmax/top-k (§2/§3).
  auto gate_wT = TransposeBytes(gate_w.raw.data.data(), E, H, es);
  DeviceBuf dgwT(b, q, act, {H, E}, gate_wT.data());
  DeviceBuf dlogits(b, q, act, {T, E});
  vt::Matmul(q, dlogits.tensor(), dx.tensor(), dgwT.tensor());
  DeviceBuf dtw(b, q, DType::kF32, {T, top_k});
  DeviceBuf dtid(b, q, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(q, dtw.tensor(), dtid.tensor(), dlogits.tensor(),
                    vt::MoeRouterTopKArgs{top_k, renorm});
  std::vector<uint8_t> rl_host, tw_host, tid_host;
  RequireMatch("router_logits", dlogits.Download(q, rl_host), rl_want.tensor, atol, rtol);
  RequireMatch("topk_weights", dtw.Download(q, tw_host), tw_want.tensor, atol, rtol);
  Tensor tid_got = dtid.Download(q, tid_host);
  RequireExactI32("topk_ids", tid_got, tid_want.tensor);

  // Per-expert silu-mul MLP (§4). For each expert e compute the MLP over ALL
  // tokens, then scatter its rows into the slots routed to e.
  std::vector<uint8_t> expert_out(static_cast<size_t>(T * top_k * H) * es, 0);
  const char* w13p = w13.raw.data.data();
  const char* w2p = w2.raw.data.data();
  for (int64_t e = 0; e < E; ++e) {
    auto w13T = TransposeBytes(w13p + static_cast<size_t>(e * 2 * I * H) * es, 2 * I, H, es);
    auto w2T = TransposeBytes(w2p + static_cast<size_t>(e * H * I) * es, H, I, es);
    DeviceBuf dw13T(b, q, act, {H, 2 * I}, w13T.data());
    DeviceBuf dw2T(b, q, act, {I, H}, w2T.data());
    DeviceBuf dh1(b, q, act, {T, 2 * I});
    vt::Matmul(q, dh1.tensor(), dx.tensor(), dw13T.tensor());
    DeviceBuf da(b, q, act, {T, I});
    vt::SiluAndMul(q, da.tensor(), dh1.tensor());
    DeviceBuf dy(b, q, act, {T, H});
    vt::Matmul(q, dy.tensor(), da.tensor(), dw2T.tensor());
    std::vector<uint8_t> y_host;
    dy.Download(q, y_host);
    for (int64_t row = 0; row < T; ++row)
      for (int k = 0; k < top_k; ++k)
        if (tid_got.Ptr<int32_t>()[row * top_k + k] == e)
          std::memcpy(expert_out.data() + (static_cast<size_t>((row * top_k + k) * H)) * es,
                      y_host.data() + static_cast<size_t>(row * H) * es,
                      static_cast<size_t>(H) * es);
  }

  // Routed combine (§4 weighted sum, f32 accumulation).
  DeviceBuf de(b, q, act, {T, top_k, H}, expert_out.data());
  DeviceBuf droute(b, q, DType::kF32, {T, H});
  vt::MoeCombine(q, droute.tensor(), de.tensor(), dtw.tensor(), nullptr);
  std::vector<uint8_t> route_host;
  RequireMatch("routed_out", droute.Download(q, route_host), routed_want.tensor, atol, rtol);

  // Shared expert (§5): s = silu(x@Wgate)*x@Wup ; sd = s@Wdown ;
  // shared = sigmoid(x@Wseg) * sd.
  auto wsguT = TransposeBytes(wsgu.raw.data.data(), 2 * Is, H, es);
  DeviceBuf dwsguT(b, q, act, {H, 2 * Is}, wsguT.data());
  DeviceBuf dgu(b, q, act, {T, 2 * Is});
  vt::Matmul(q, dgu.tensor(), dx.tensor(), dwsguT.tensor());
  DeviceBuf dsact(b, q, act, {T, Is});
  vt::SiluAndMul(q, dsact.tensor(), dgu.tensor());
  auto wsdT = TransposeBytes(wsd.raw.data.data(), H, Is, es);
  DeviceBuf dwsdT(b, q, act, {Is, H}, wsdT.data());
  DeviceBuf dsd(b, q, act, {T, H});
  vt::Matmul(q, dsd.tensor(), dsact.tensor(), dwsdT.tensor());
  auto wsgT = TransposeBytes(wsg.raw.data.data(), 1, H, es);
  DeviceBuf dwsgT(b, q, act, {H, 1}, wsgT.data());
  DeviceBuf dgl(b, q, act, {T, 1});
  vt::Matmul(q, dgl.tensor(), dx.tensor(), dwsgT.tensor());
  std::vector<uint8_t> sd_host, gl_host;
  Tensor tsd = dsd.Download(q, sd_host);
  Tensor tgl = dgl.Download(q, gl_host);
  std::vector<uint8_t> shared_host(static_cast<size_t>(T * H) * es, 0);
  Tensor t_shared = MakeTensor(shared_host.data(), act, Cpu(), {T, H});
  for (int64_t row = 0; row < T; ++row) {
    const float gate = 1.0f / (1.0f + std::exp(-AsF32(tgl, row)));  // sigmoid(gate logit)
    for (int64_t col = 0; col < H; ++col)
      StoreAs(t_shared, row * H + col, gate * AsF32(tsd, row * H + col));
  }
  RequireMatch("shared_out", t_shared, shared_want.tensor, atol, rtol);

  // Final block output (§6): out = shared + routed.
  DeviceBuf dsh(b, q, act, {T, H}, shared_host.data());
  DeviceBuf dout(b, q, DType::kF32, {T, H});
  vt::MoeCombine(q, dout.tensor(), de.tensor(), dtw.tensor(), &dsh.tensor());
  std::vector<uint8_t> out_host;
  RequireMatch("out", dout.Download(q, out_host), out_want.tensor, atol, rtol);
}

// dense_attention: the op-level golden for the Qwen3NextAttention core
// (qwen36-forward-notes.md §5). Composes the full core exactly as the Task 4
// full-attn layer will: per-head gemma-RMSNorm on q and k (vt::RmsNorm over
// head_dim), partial NeoX RoPE (vt::RopeNeox on positions), causal GQA
// scaled-dot-product (the new vt::Attention op), then the sigmoid output gate
// (elementwise, host-composed like RunMoeBlock's shared-expert gate). q/gate/k/v
// are the PRE-norm projection split; `attn` (pre-gate) and `out` (gated) are the
// checked contracts. All f32 — the op-level golden isolates the attention math.
void RunDenseAttention(Backend& b, Queue& q, const fs::path& dir, const json& m) {
  auto qin = LoadTensor(dir, m["tensors"]["q"]);
  auto gate = LoadTensor(dir, m["tensors"]["gate"]);
  auto k = LoadTensor(dir, m["tensors"]["k"]);
  auto v = LoadTensor(dir, m["tensors"]["v"]);
  auto pos = LoadTensor(dir, m["tensors"]["positions"]);
  auto qnw = LoadTensor(dir, m["tensors"]["q_norm_weight"]);
  auto knw = LoadTensor(dir, m["tensors"]["k_norm_weight"]);
  auto attn_want = LoadTensor(dir, m["tensors"]["attn"]);
  auto out_want = LoadTensor(dir, m["tensors"]["out"]);

  const int64_t T = qin.tensor.shape[0];
  const int64_t Hq = m["args"]["num_q_heads"].get<int64_t>();
  const int64_t Hk = m["args"]["num_kv_heads"].get<int64_t>();
  const int64_t D = m["args"]["head_dim"].get<int64_t>();
  const float eps = m["args"]["eps"].get<float>();
  const float base = m["args"]["base"].get<float>();
  const int rotary_dim = m["args"]["rotary_dim"].get<int>();
  const float scale = m["args"]["scale"].get<float>();
  const bool causal = m["args"]["causal"].get<bool>();
  VT_CHECK(qin.dtype == DType::kF32, "dense_attention runner requires f32 q (in-place rope)");
  double atol = m["tol"]["atol"].get<double>(), rtol = m["tol"]["rtol"].get<double>();

  // Per-head gemma-RMSNorm: normalize over head_dim, so view q/k as [T*H, D].
  DeviceBuf dq(b, q, DType::kF32, {T * Hq, D}, qin.raw.data.data());
  DeviceBuf dk(b, q, DType::kF32, {T * Hk, D}, k.raw.data.data());
  DeviceBuf dqnw(b, q, DType::kF32, ShapeOf(qnw.tensor), qnw.raw.data.data());
  DeviceBuf dknw(b, q, DType::kF32, ShapeOf(knw.tensor), knw.raw.data.data());
  DeviceBuf dqn(b, q, DType::kF32, {T * Hq, D});
  DeviceBuf dkn(b, q, DType::kF32, {T * Hk, D});
  vt::RmsNormArgs nargs{eps, /*gemma=*/true};
  vt::RmsNorm(q, dqn.tensor(), dq.tensor(), dqnw.tensor(), nargs);
  vt::RmsNorm(q, dkn.tensor(), dk.tensor(), dknw.tensor(), nargs);

  // Partial NeoX RoPE in place on the normed q/k, viewed as [T,H,D].
  Tensor qn3 = MakeTensor(dqn.tensor().data, DType::kF32, q.device, {T, Hq, D});
  Tensor kn3 = MakeTensor(dkn.tensor().data, DType::kF32, q.device, {T, Hk, D});
  DeviceBuf dpos(b, q, pos.dtype, ShapeOf(pos.tensor), pos.raw.data.data());
  vt::RopeNeox(q, qn3, kn3, dpos.tensor(), vt::RopeArgs{base, rotary_dim});

  // Causal GQA scaled-dot-product attention (the new op). v is raw [T,Hk,D].
  DeviceBuf dv(b, q, DType::kF32, {T, Hk, D}, v.raw.data.data());
  DeviceBuf dattn(b, q, DType::kF32, {T, Hq, D});
  vt::Attention(q, dattn.tensor(), qn3, kn3, dv.tensor(), vt::AttentionArgs{scale, causal});
  std::vector<uint8_t> attn_host;
  Tensor tattn = dattn.Download(q, attn_host);
  RequireMatch("attn", tattn, attn_want.tensor, atol, rtol);

  // Sigmoid output gate (elementwise, host-composed): out = attn * sigmoid(gate).
  std::vector<uint8_t> gated(static_cast<size_t>(T * Hq * D) * sizeof(float), 0);
  Tensor tgated = MakeTensor(gated.data(), DType::kF32, Cpu(), {T, Hq, D});
  for (int64_t i = 0; i < T * Hq * D; ++i) {
    const float g = 1.0f / (1.0f + std::exp(-AsF32(gate.tensor, i)));
    StoreAs(tgated, i, AsF32(tattn, i) * g);
  }
  RequireMatch("out", tgated, out_want.tensor, atol, rtol);
}

// Ops whose goldens are committed but whose runners are not implemented yet.
// The parity harness scans every golden directory eagerly, so a milestone that
// lands goldens before their runners (the standard "dump first, implement next"
// split) would otherwise hard-FAIL here. Listing an op here makes its cases
// SKIP loudly (a visible MESSAGE) instead — while ANY op NOT listed still
// hard-FAILs, preserving the anti-stale-golden gate. Each entry MUST be removed
// as its runner lands; the milestone close-out asserts this set is empty of the
// milestone's ops (M0.8: moe_router_topk → Task 2, moe_block → Task 2 runner).
// M0.9: the qwen36 layer/model goldens landed in Task 1 ahead of the Task 2/4
// runners (dense-attention op + forward assembly); qwen36_fullattn_layer →
// Task 2, qwen36_{embed,gdn_layer,norm,logits} → Task 4 layer/forward runners.
const std::set<std::string>& PendingRunnerOps() {
  static const std::set<std::string> kPending = {
      "qwen36_embed",   "qwen36_gdn_layer", "qwen36_fullattn_layer",
      "qwen36_norm",    "qwen36_logits"};
  return kPending;
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
    // test_tokenizer_parity) carry no manifest.json; the `cases >= 24` floor
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
    } else if (op == "causal_conv1d_fwd") {
      RunCausalConv1dFwd(b, q, entry.path(), m);
    } else if (op == "causal_conv1d_update") {
      RunCausalConv1dUpdate(b, q, entry.path(), m);
    } else if (op == "l2norm") {
      RunL2Norm(b, q, entry.path(), m);
    } else if (op == "rmsnorm_gated") {
      RunRmsNormGated(b, q, entry.path(), m);
    } else if (op == "gdn_prefill") {
      RunGdnPrefill(b, q, entry.path(), m);
    } else if (op == "gdn_decode") {
      RunGdnDecode(b, q, entry.path(), m);
    } else if (op == "moe_router_topk") {
      RunMoeRouterTopK(b, q, entry.path(), m);
    } else if (op == "moe_block") {
      RunMoeBlock(b, q, entry.path(), m);
    } else if (op == "dense_attention") {
      RunDenseAttention(b, q, entry.path(), m);
    } else if (PendingRunnerOps().count(op)) {
      MESSAGE("SKIP op '" << op << "' case '" << entry.path().filename().string()
                          << "': runner pending (see PendingRunnerOps)");
      continue;  // does not count toward the case floor
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
  CHECK(cases >= 31);  // 24 pre-M0.8 + 5 MoE + 2 dense_attention (M0.9 Task 2)
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
  CHECK(cases >= 26);
}
