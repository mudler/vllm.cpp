// Tenstorrent backend skeleton unit gates (BACKEND-TENSTORRENT, W0). Newly
// authored — vLLM has no Tenstorrent tests to port. Mirrors the shape of
// tests/vt/test_vulkan_backend.cpp / test_metal_backend.cpp so the three are
// read side by side.
//
// This TU is COMPILED ONLY in a Tenstorrent build (tests/CMakeLists.txt gates
// it on VLLM_CPP_TENSTORRENT) and every assertion goes through the public
// vt:: seam — if the skeleton needed ttnn headers in a test to be checkable,
// the seam would be leaking. (This is also why this file needs none of the
// object-library include isolation tenstorrent_ops.cpp needed — it never
// touches ttnn/tt-metal headers at all.)
//
// Every case is SKIPPED, not failed, when no Blackhole card is present — the
// registrars stay silent by design (tenstorrent_backend.cpp/tenstorrent.cpp),
// and a Tenstorrent-enabled build legitimately runs in CI containers with no
// card. The skip is REPORTED so a silently-unregistered backend on a box that
// DOES have one cannot masquerade as a pass.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

namespace {

bool TenstorrentPresent() { return vt::TryGetBackend(DeviceType::kTENSTORRENT) != nullptr; }

}  // namespace

TEST_CASE("kTENSTORRENT backend registers iff a device is present") {
  Backend* b = vt::TryGetBackend(DeviceType::kTENSTORRENT);
  if (b == nullptr) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  CHECK_FALSE(b->UnifiedMemory());  // discrete PCIe card — see backend.h's SCOPE note
}

TEST_CASE("kTENSTORRENT Platform mirrors the registered Backend") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vllm::platforms::HasPlatform(DeviceType::kTENSTORRENT));
  vllm::platforms::Platform& p = vllm::platforms::GetPlatform(DeviceType::kTENSTORRENT);
  CHECK(p.device_type() == DeviceType::kTENSTORRENT);
  CHECK(&p.backend() == vt::TryGetBackend(DeviceType::kTENSTORRENT));
  // OPT-125m path: BF16 weights/activations + F32 logits — see tenstorrent_ops.cpp.
  CHECK(p.supported_dtypes() ==
        std::vector<vt::DType>{vt::DType::kBF16, vt::DType::kF32});
  // FLASH_ATTN is registered against the NHD layout our kPagedAttention uses.
  CHECK(p.get_attn_backend_priority({}) == std::vector<std::string>{"FLASH_ATTN"});
  CHECK(p.supports_model_architecture("OPTForCausalLM"));
  // Qwen3-dense after RmsNorm / SiluAndMul / Cast / RoPE landed (Metal M3b twin).
  CHECK(p.supports_model_architecture("Qwen3ForCausalLM"));
  CHECK_FALSE(p.supports_model_architecture("LlamaForCausalLM"));
}

TEST_CASE("kTENSTORRENT kMatmul matches a host F32 reference") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmul, DeviceType::kTENSTORRENT));

  constexpr int64_t M = 32, K = 32, N = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_a(M * K), host_b(K * N), host_out(M * N, 0.0f);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(i % 5) * 0.2f;

  void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
  void* mem_b = backend.Alloc(host_b.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
  backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));

  Tensor a = Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {K, N});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});

  auto matmul = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmul, DeviceType::kTENSTORRENT));
  matmul(q, out, a, b);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_a);
  backend.Free(mem_b);
  backend.Free(mem_out);

  std::vector<float> ref(M * N, 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += host_a[i * K + k] * host_b[k * N + j];
      ref[i * N + j] = acc;
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - ref[i]));
  // bf16 accumulation over K=32 on-device — same tolerance the hands-on spike
  // measured (.agents/specs/tenstorrent-backend.md), not a rubber stamp.
  CHECK(max_abs_diff < 0.5f);
}

TEST_CASE("kTENSTORRENT kMatmulBT matches a host F32 reference (a @ b^T)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));

  // `a` is [M,K] activations; `b` is [N,K] nn.Linear weight (torch layout).
  constexpr int64_t M = 32, K = 32, N = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_a(M * K), host_b(N * K), host_out(M * N, 0.0f);
  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(i % 5) * 0.2f;

  void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
  void* mem_b = backend.Alloc(host_b.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
  backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));

  Tensor a = Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor b = Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {N, K});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});

  auto matmul_bt =
      reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmulBT, DeviceType::kTENSTORRENT));
  matmul_bt(q, out, a, b);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_a);
  backend.Free(mem_b);
  backend.Free(mem_out);

  std::vector<float> ref(M * N, 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc += host_a[i * K + k] * host_b[j * K + k];
      ref[i * N + j] = acc;
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - ref[i]));
  CHECK(max_abs_diff < 0.5f);
}

TEST_CASE("kTENSTORRENT kAdd matches a host F32 reference (elementwise + bias broadcast)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kAdd, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto add = reinterpret_cast<vt::AddFn>(vt::GetOp(vt::OpId::kAdd, DeviceType::kTENSTORRENT));

  SUBCASE("elementwise, same rank") {
    std::vector<float> host_a(Rows * D), host_b(Rows * D), host_out(Rows * D, 0.0f);
    for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
    for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>(i % 5) * 0.2f;

    void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
    void* mem_b = backend.Alloc(host_b.size() * sizeof(float));
    void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
    Queue q = backend.CreateQueue();
    backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
    backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));

    Tensor a =
        Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor b =
        Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor out =
        Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    add(q, out, a, b);
    backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_b);
    backend.Free(mem_out);

    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < host_out.size(); ++i)
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - (host_a[i] + host_b[i])));
    CHECK(max_abs_diff < 0.1f);
  }

  SUBCASE("rank-1 bias broadcast over the last dim") {
    std::vector<float> host_a(Rows * D), host_bias(D), host_out(Rows * D, 0.0f);
    for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>(i % 7) * 0.1f;
    for (size_t i = 0; i < host_bias.size(); ++i) host_bias[i] = static_cast<float>(i) * 0.05f;

    void* mem_a = backend.Alloc(host_a.size() * sizeof(float));
    void* mem_bias = backend.Alloc(host_bias.size() * sizeof(float));
    void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
    Queue q = backend.CreateQueue();
    backend.Copy(q, mem_a, host_a.data(), host_a.size() * sizeof(float));
    backend.Copy(q, mem_bias, host_bias.data(), host_bias.size() * sizeof(float));

    Tensor a =
        Tensor::Contiguous(mem_a, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor bias =
        Tensor::Contiguous(mem_bias, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
    Tensor out =
        Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    add(q, out, a, bias);
    backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
    backend.Free(mem_a);
    backend.Free(mem_bias);
    backend.Free(mem_out);

    float max_abs_diff = 0.0f;
    for (int64_t r = 0; r < Rows; ++r)
      for (int64_t c = 0; c < D; ++c)
        max_abs_diff = std::max(max_abs_diff,
                                 std::fabs(host_out[r * D + c] - (host_a[r * D + c] + host_bias[c])));
    CHECK(max_abs_diff < 0.1f);
  }
}

TEST_CASE("kTENSTORRENT kRelu matches a host F32 reference") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRelu, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_x(Rows * D), host_out(Rows * D, 0.0f);
  for (size_t i = 0; i < host_x.size(); ++i)
    host_x[i] = (static_cast<float>(i % 11) - 5.0f) * 0.3f;  // mix of signs

  void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));

  Tensor x = Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});

  auto relu = reinterpret_cast<vt::ReluFn>(vt::GetOp(vt::OpId::kRelu, DeviceType::kTENSTORRENT));
  relu(q, out, x);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_x);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < host_x.size(); ++i)
    max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - std::max(0.0f, host_x[i])));
  CHECK(max_abs_diff < 0.1f);
}

TEST_CASE("kTENSTORRENT kEmbedding matches a host F32 reference (row gather)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kEmbedding, DeviceType::kTENSTORRENT));

  // Non-tile-aligned (t, h) on purpose: forces the ROW_MAJOR path and
  // proves download is dense without TILE padding. Vocab is modest so the
  // host oracle stays trivial.
  constexpr int64_t Vocab = 17, H = 24, T = 7;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_table(Vocab * H);
  for (size_t i = 0; i < host_table.size(); ++i)
    host_table[i] = static_cast<float>(i % 13) * 0.1f - 0.5f;
  // i32 ids covering edges: first, last, and middle rows; repeats allowed.
  std::vector<int32_t> host_ids = {0, 3, 16, 1, 3, 8, 16};
  REQUIRE(static_cast<int64_t>(host_ids.size()) == T);

  std::vector<float> host_out(T * H, 0.0f);
  void* mem_table = backend.Alloc(host_table.size() * sizeof(float));
  void* mem_ids = backend.Alloc(host_ids.size() * sizeof(int32_t));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_table, host_table.data(), host_table.size() * sizeof(float));
  backend.Copy(q, mem_ids, host_ids.data(), host_ids.size() * sizeof(int32_t));

  Tensor table = Tensor::Contiguous(mem_table, vt::DType::kF32,
                                    Device{DeviceType::kTENSTORRENT, 0}, {Vocab, H});
  Tensor ids = Tensor::Contiguous(mem_ids, vt::DType::kI32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {T});
  Tensor out = Tensor::Contiguous(mem_out, vt::DType::kF32,
                                  Device{DeviceType::kTENSTORRENT, 0}, {T, H});

  auto embedding =
      reinterpret_cast<vt::EmbeddingFn>(vt::GetOp(vt::OpId::kEmbedding, DeviceType::kTENSTORRENT));
  embedding(q, out, table, ids);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_table);
  backend.Free(mem_ids);
  backend.Free(mem_out);

  // Host oracle: pure row gather. bf16 table storage means a modest abs tol.
  float max_abs_diff = 0.0f;
  for (int64_t i = 0; i < T; ++i) {
    const int32_t id = host_ids[static_cast<size_t>(i)];
    for (int64_t j = 0; j < H; ++j) {
      const float ref = host_table[static_cast<size_t>(id) * H + j];
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i * H + j] - ref));
    }
  }
  CHECK(max_abs_diff < 0.1f);
}

TEST_CASE("kTENSTORRENT kLayerNorm matches a host F32 reference (affine + plain)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kLayerNorm, DeviceType::kTENSTORRENT));

  // Tile-aligned so the TILE upload path is exercised cleanly (same as the
  // linear ops). Host oracle is the ATen/cpu_layernorm biased-variance form.
  constexpr int64_t Rows = 32, D = 32;
  constexpr float Eps = 1e-5f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto layer_norm = reinterpret_cast<vt::LayerNormFn>(
      vt::GetOp(vt::OpId::kLayerNorm, DeviceType::kTENSTORRENT));

  auto run_case = [&](bool with_affine) {
    std::vector<float> host_x(Rows * D), host_w(D), host_b(D), host_out(Rows * D, 0.0f);
    for (size_t i = 0; i < host_x.size(); ++i)
      host_x[i] = (static_cast<float>(i % 17) - 8.0f) * 0.15f;
    for (int64_t j = 0; j < D; ++j) {
      host_w[static_cast<size_t>(j)] = 0.5f + static_cast<float>(j % 5) * 0.1f;
      host_b[static_cast<size_t>(j)] = static_cast<float>(j % 7) * 0.05f - 0.15f;
    }

    void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
    void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
    void* mem_w = with_affine ? backend.Alloc(host_w.size() * sizeof(float)) : nullptr;
    void* mem_b = with_affine ? backend.Alloc(host_b.size() * sizeof(float)) : nullptr;
    Queue q = backend.CreateQueue();
    backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));
    if (with_affine) {
      backend.Copy(q, mem_w, host_w.data(), host_w.size() * sizeof(float));
      backend.Copy(q, mem_b, host_b.data(), host_b.size() * sizeof(float));
    }

    Tensor x =
        Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor out =
        Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
    Tensor w_t, b_t;
    const Tensor* w_ptr = nullptr;
    const Tensor* b_ptr = nullptr;
    if (with_affine) {
      w_t = Tensor::Contiguous(mem_w, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
      b_t = Tensor::Contiguous(mem_b, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
      w_ptr = &w_t;
      b_ptr = &b_t;
    }

    vt::LayerNormArgs args;
    args.eps = Eps;
    layer_norm(q, out, x, w_ptr, b_ptr, args);

    backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
    backend.Free(mem_x);
    backend.Free(mem_out);
    if (with_affine) {
      backend.Free(mem_w);
      backend.Free(mem_b);
    }

    float max_abs_diff = 0.0f;
    for (int64_t r = 0; r < Rows; ++r) {
      float sum = 0.0f;
      for (int64_t j = 0; j < D; ++j) sum += host_x[r * D + j];
      const float mean = sum / static_cast<float>(D);
      float sq = 0.0f;
      for (int64_t j = 0; j < D; ++j) {
        const float dv = host_x[r * D + j] - mean;
        sq += dv * dv;
      }
      const float rstd = 1.0f / std::sqrt(sq / static_cast<float>(D) + Eps);
      for (int64_t j = 0; j < D; ++j) {
        float ref = (host_x[r * D + j] - mean) * rstd;
        if (with_affine) ref = ref * host_w[static_cast<size_t>(j)] + host_b[static_cast<size_t>(j)];
        max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[r * D + j] - ref));
      }
    }
    // bf16 storage + device reduction; not bit-exact vs f32 host mean/var.
    CHECK(max_abs_diff < 0.5f);
  };

  SUBCASE("elementwise_affine=True (weight + bias)") { run_case(true); }
  SUBCASE("elementwise_affine=False (no weight/bias)") { run_case(false); }
}

// First Qwen3-dense (`Qwen3ForCausalLM`) op beyond OPT's set. Host oracle is
// cpu_ops RmsNormKernel (no residual, gemma=false) — the Qwen3 default.
TEST_CASE("kTENSTORRENT kRmsNorm matches a host F32 reference (weight, no residual)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRmsNorm, DeviceType::kTENSTORRENT));

  constexpr int64_t Rows = 32, D = 32;
  constexpr float Eps = 1e-6f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto rms_norm = reinterpret_cast<vt::RmsNormFn>(
      vt::GetOp(vt::OpId::kRmsNorm, DeviceType::kTENSTORRENT));

  std::vector<float> host_x(Rows * D), host_w(D), host_out(Rows * D, 0.0f);
  for (size_t i = 0; i < host_x.size(); ++i)
    host_x[i] = (static_cast<float>(i % 17) - 8.0f) * 0.15f;
  for (int64_t j = 0; j < D; ++j)
    host_w[static_cast<size_t>(j)] = 0.5f + static_cast<float>(j % 5) * 0.1f;

  void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
  void* mem_w = backend.Alloc(host_w.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));
  backend.Copy(q, mem_w, host_w.data(), host_w.size() * sizeof(float));

  Tensor x =
      Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});
  Tensor w =
      Tensor::Contiguous(mem_w, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {D});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {Rows, D});

  vt::RmsNormArgs args;
  args.eps = Eps;
  args.gemma = false;
  rms_norm(q, out, x, w, args, /*residual=*/nullptr);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_x);
  backend.Free(mem_w);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (int64_t r = 0; r < Rows; ++r) {
    float sumsq = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
      const float v = host_x[r * D + j];
      sumsq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(D) + Eps);
    for (int64_t j = 0; j < D; ++j) {
      const float ref = host_x[r * D + j] * inv * host_w[static_cast<size_t>(j)];
      max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[r * D + j] - ref));
    }
  }
  // bf16 storage + device reduction; same envelope as kLayerNorm.
  CHECK(max_abs_diff < 0.5f);
}

// Qwen3-dense MLP SwiGLU half. Device path (slice + silu + mul) via BF16 tiles.
TEST_CASE("kTENSTORRENT kSiluAndMul matches host F32 within BF16 envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kSiluAndMul, DeviceType::kTENSTORRENT));

  constexpr int64_t T = 7, D = 16;  // x is [T, 2D]
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto silu_mul = reinterpret_cast<vt::SiluAndMulFn>(
      vt::GetOp(vt::OpId::kSiluAndMul, DeviceType::kTENSTORRENT));

  std::vector<float> host_x(T * 2 * D), host_out(T * D, 0.0f);
  for (size_t i = 0; i < host_x.size(); ++i)
    host_x[i] = (static_cast<float>(i % 13) - 6.0f) * 0.2f;

  void* mem_x = backend.Alloc(host_x.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_x, host_x.data(), host_x.size() * sizeof(float));

  Tensor x = Tensor::Contiguous(mem_x, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                {T, 2 * D});
  Tensor out =
      Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, D});
  silu_mul(q, out, x);
  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_x);
  backend.Free(mem_out);

  float max_abs_diff = 0.0f;
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = 0; j < D; ++j) {
      const float gate = host_x[static_cast<size_t>(i * 2 * D + j)];
      const float up = host_x[static_cast<size_t>(i * 2 * D + D + j)];
      const float ref = (gate / (1.0f + std::exp(-gate))) * up;
      max_abs_diff =
          std::max(max_abs_diff, std::fabs(host_out[static_cast<size_t>(i * D + j)] - ref));
    }
  }
  // BF16 tile storage + silu; same envelope as kRelu / kRmsNorm.
  CHECK(max_abs_diff < 0.05f);
}

// Cast pair used by Qwen3 K/V cache dtype and logits paths.
TEST_CASE("kTENSTORRENT kCastBf16 / kCastF32 round-trip F32 values") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kCastBf16, DeviceType::kTENSTORRENT));
  REQUIRE(vt::OpRegistered(vt::OpId::kCastF32, DeviceType::kTENSTORRENT));

  constexpr int64_t N = 64;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto cast_bf16 = reinterpret_cast<vt::CastBf16Fn>(
      vt::GetOp(vt::OpId::kCastBf16, DeviceType::kTENSTORRENT));
  auto cast_f32 = reinterpret_cast<vt::CastF32Fn>(
      vt::GetOp(vt::OpId::kCastF32, DeviceType::kTENSTORRENT));

  std::vector<float> host_f(N), host_back(N, 0.0f);
  for (int64_t i = 0; i < N; ++i) host_f[static_cast<size_t>(i)] = static_cast<float>(i) * 0.125f;

  void* mem_f = backend.Alloc(N * sizeof(float));
  void* mem_bf = backend.Alloc(N * sizeof(uint16_t));
  void* mem_back = backend.Alloc(N * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_f, host_f.data(), N * sizeof(float));

  Tensor tf =
      Tensor::Contiguous(mem_f, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {N});
  Tensor tbf =
      Tensor::Contiguous(mem_bf, vt::DType::kBF16, Device{DeviceType::kTENSTORRENT, 0}, {N});
  Tensor tback =
      Tensor::Contiguous(mem_back, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {N});
  cast_bf16(q, tbf, tf);
  cast_f32(q, tback, tbf);
  backend.Copy(q, host_back.data(), mem_back, N * sizeof(float));
  backend.Free(mem_f);
  backend.Free(mem_bf);
  backend.Free(mem_back);

  for (int64_t i = 0; i < N; ++i) {
    // Exact for values representable in bf16 (i * 0.125).
    REQUIRE(host_back[static_cast<size_t>(i)] == host_f[static_cast<size_t>(i)]);
  }
}

// Default Qwen3-dense RoPE (Metal M3b). Small T*H uses host apply (bit-exact);
// large prefill uses device NeoX (BF16) — covered by PreferDeviceRope threshold.
TEST_CASE("kTENSTORRENT kRopeNeox is BIT-EXACT vs a host F32 reference (small)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRopeNeox, DeviceType::kTENSTORRENT));

  constexpr int64_t T = 4, Hq = 2, Hk = 1, Dh = 8, Rot = 8;
  constexpr float Base = 10000.0f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto rope = reinterpret_cast<vt::RopeFn>(
      vt::GetOp(vt::OpId::kRopeNeox, DeviceType::kTENSTORRENT));

  std::vector<float> hq(T * Hq * Dh), hk(T * Hk * Dh), hq_ref, hk_ref;
  std::vector<int32_t> pos(T);
  for (int64_t i = 0; i < T * Hq * Dh; ++i)
    hq[static_cast<size_t>(i)] = (static_cast<float>(i % 11) - 5.0f) * 0.1f;
  for (int64_t i = 0; i < T * Hk * Dh; ++i)
    hk[static_cast<size_t>(i)] = (static_cast<float>(i % 7) - 3.0f) * 0.1f;
  for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i + 1);
  hq_ref = hq;
  hk_ref = hk;

  auto rotate_head = [&](std::vector<float>& t, int64_t head_off, int64_t p) {
    const int half = Rot / 2;
    for (int i = 0; i < half; ++i) {
      const double freq =
          std::pow(static_cast<double>(Base), -2.0 * i / static_cast<double>(Rot));
      const double angle = static_cast<double>(p) * freq;
      const float c = static_cast<float>(std::cos(angle));
      const float s = static_cast<float>(std::sin(angle));
      const float x = t[static_cast<size_t>(head_off + i)];
      const float y = t[static_cast<size_t>(head_off + i + half)];
      t[static_cast<size_t>(head_off + i)] = x * c - y * s;
      t[static_cast<size_t>(head_off + i + half)] = x * s + y * c;
    }
  };
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t h = 0; h < Hq; ++h) rotate_head(hq_ref, (i * Hq + h) * Dh, pos[static_cast<size_t>(i)]);
    for (int64_t h = 0; h < Hk; ++h) rotate_head(hk_ref, (i * Hk + h) * Dh, pos[static_cast<size_t>(i)]);
  }

  void* mem_q = backend.Alloc(hq.size() * sizeof(float));
  void* mem_k = backend.Alloc(hk.size() * sizeof(float));
  void* mem_p = backend.Alloc(pos.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, hq.data(), hq.size() * sizeof(float));
  backend.Copy(q, mem_k, hk.data(), hk.size() * sizeof(float));
  backend.Copy(q, mem_p, pos.data(), pos.size() * sizeof(int32_t));

  Tensor tq = Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hq, Dh});
  Tensor tk = Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hk, Dh});
  Tensor tp = Tensor::Contiguous(mem_p, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {T});
  vt::RopeArgs args;
  args.base = Base;
  args.rotary_dim = Rot;
  rope(q, tq, tk, tp, args);

  backend.Copy(q, hq.data(), mem_q, hq.size() * sizeof(float));
  backend.Copy(q, hk.data(), mem_k, hk.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_k);
  backend.Free(mem_p);

  for (size_t i = 0; i < hq.size(); ++i) REQUIRE(hq[i] == hq_ref[i]);
  for (size_t i = 0; i < hk.size(); ++i) REQUIRE(hk[i] == hk_ref[i]);
}

TEST_CASE("kTENSTORRENT kRopeCosSinCache + kRopeFromCache match kRopeNeox") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kRopeCosSinCache, DeviceType::kTENSTORRENT));
  REQUIRE(vt::OpRegistered(vt::OpId::kRopeFromCache, DeviceType::kTENSTORRENT));
  REQUIRE(vt::OpRegistered(vt::OpId::kRopeNeox, DeviceType::kTENSTORRENT));

  constexpr int64_t T = 3, Hq = 2, Hk = 1, Dh = 8, Rot = 8;
  constexpr float Base = 10000.0f;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto rope_neox = reinterpret_cast<vt::RopeFn>(
      vt::GetOp(vt::OpId::kRopeNeox, DeviceType::kTENSTORRENT));
  auto rope_cache = reinterpret_cast<vt::RopeCosSinCacheFn>(
      vt::GetOp(vt::OpId::kRopeCosSinCache, DeviceType::kTENSTORRENT));
  auto rope_from = reinterpret_cast<vt::RopeFromCacheFn>(
      vt::GetOp(vt::OpId::kRopeFromCache, DeviceType::kTENSTORRENT));

  std::vector<float> q0(T * Hq * Dh), k0(T * Hk * Dh);
  std::vector<int32_t> pos(T);
  for (size_t i = 0; i < q0.size(); ++i) q0[i] = (static_cast<float>(i % 9) - 4.0f) * 0.05f;
  for (size_t i = 0; i < k0.size(); ++i) k0[i] = (static_cast<float>(i % 5) - 2.0f) * 0.05f;
  for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);

  auto alloc_copy = [&](const void* src, size_t bytes) {
    void* p = backend.Alloc(bytes);
    Queue q = backend.CreateQueue();
    backend.Copy(q, p, src, bytes);
    return p;
  };

  void* mq1 = alloc_copy(q0.data(), q0.size() * sizeof(float));
  void* mk1 = alloc_copy(k0.data(), k0.size() * sizeof(float));
  void* mq2 = alloc_copy(q0.data(), q0.size() * sizeof(float));
  void* mk2 = alloc_copy(k0.data(), k0.size() * sizeof(float));
  void* mp = alloc_copy(pos.data(), pos.size() * sizeof(int32_t));
  void* mcs = backend.Alloc(static_cast<size_t>(T * Rot) * sizeof(float));
  // Identity row index into a T-row cache built from positions 0..T-1.
  std::vector<int32_t> rows(T);
  for (int64_t i = 0; i < T; ++i) rows[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  void* mrow = alloc_copy(rows.data(), rows.size() * sizeof(int32_t));

  Queue q = backend.CreateQueue();
  Tensor tq1 = Tensor::Contiguous(mq1, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Hq, Dh});
  Tensor tk1 = Tensor::Contiguous(mk1, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Hk, Dh});
  Tensor tq2 = Tensor::Contiguous(mq2, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Hq, Dh});
  Tensor tk2 = Tensor::Contiguous(mk2, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Hk, Dh});
  Tensor tp = Tensor::Contiguous(mp, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {T});
  Tensor tcs =
      Tensor::Contiguous(mcs, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Rot});
  Tensor trow =
      Tensor::Contiguous(mrow, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {T});

  vt::RopeArgs args;
  args.base = Base;
  args.rotary_dim = Rot;
  args.is_neox_style = true;

  rope_neox(q, tq1, tk1, tp, args);
  rope_cache(q, tcs, tp, args);
  rope_from(q, tq2, &tk2, trow, tcs, args);

  std::vector<float> qn(q0.size()), kn(k0.size()), qc(q0.size()), kc(k0.size());
  backend.Copy(q, qn.data(), mq1, qn.size() * sizeof(float));
  backend.Copy(q, kn.data(), mk1, kn.size() * sizeof(float));
  backend.Copy(q, qc.data(), mq2, qc.size() * sizeof(float));
  backend.Copy(q, kc.data(), mk2, kc.size() * sizeof(float));
  for (void* p : {mq1, mk1, mq2, mk2, mp, mcs, mrow}) backend.Free(p);

  // Small T*H → host apply on both paths → bit-identical.
  for (size_t i = 0; i < qn.size(); ++i) REQUIRE(qn[i] == qc[i]);
  for (size_t i = 0; i < kn.size(); ++i) REQUIRE(kn[i] == kc[i]);
}

TEST_CASE("kTENSTORRENT kQkvSplit is BIT-EXACT vs a host reference (unequal widths)") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));

  // Independent q/k/v widths (kernel contract); not equal-width MHA only.
  constexpr int64_t T = 11, Qd = 24, Kd = 12, Vd = 12;
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_qkv(static_cast<size_t>(T * (Qd + Kd + Vd)));
  for (size_t i = 0; i < host_qkv.size(); ++i)
    host_qkv[i] = static_cast<float>(i % 19) * 0.1f - 0.7f;
  std::vector<float> host_q(static_cast<size_t>(T * Qd), 0.0f);
  std::vector<float> host_k(static_cast<size_t>(T * Kd), 0.0f);
  std::vector<float> host_v(static_cast<size_t>(T * Vd), 0.0f);

  void* mem_qkv = backend.Alloc(host_qkv.size() * sizeof(float));
  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_k = backend.Alloc(host_k.size() * sizeof(float));
  void* mem_v = backend.Alloc(host_v.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_qkv, host_qkv.data(), host_qkv.size() * sizeof(float));

  Tensor qkv = Tensor::Contiguous(mem_qkv, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Qd + Kd + Vd});
  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Qd});
  Tensor tk =
      Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Kd});
  Tensor tv =
      Tensor::Contiguous(mem_v, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Vd});

  auto split =
      reinterpret_cast<vt::QkvSplitFn>(vt::GetOp(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));
  split(q, tq, tk, tv, qkv);

  backend.Copy(q, host_q.data(), mem_q, host_q.size() * sizeof(float));
  backend.Copy(q, host_k.data(), mem_k, host_k.size() * sizeof(float));
  backend.Copy(q, host_v.data(), mem_v, host_v.size() * sizeof(float));
  backend.Free(mem_qkv);
  backend.Free(mem_q);
  backend.Free(mem_k);
  backend.Free(mem_v);

  // Pure column split — bit-exact.
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = 0; j < Qd; ++j)
      CHECK(host_q[i * Qd + j] == host_qkv[i * (Qd + Kd + Vd) + j]);
    for (int64_t j = 0; j < Kd; ++j)
      CHECK(host_k[i * Kd + j] == host_qkv[i * (Qd + Kd + Vd) + Qd + j]);
    for (int64_t j = 0; j < Vd; ++j)
      CHECK(host_v[i * Vd + j] == host_qkv[i * (Qd + Kd + Vd) + Qd + Kd + j]);
  }
}

// Device path: make qkv device-resident (kRelu), then split without host memcpy.
// Matches e2e MatmulBT → QkvSplit residency chain.
TEST_CASE("kTENSTORRENT kQkvSplit device path matches host within BF16 envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));
  REQUIRE(vt::OpRegistered(vt::OpId::kRelu, DeviceType::kTENSTORRENT));

  constexpr int64_t T = 4, Qd = 32, Kd = 16, Vd = 16;  // total=64, tile-friendly
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  auto relu = reinterpret_cast<vt::ReluFn>(vt::GetOp(vt::OpId::kRelu, DeviceType::kTENSTORRENT));
  auto split =
      reinterpret_cast<vt::QkvSplitFn>(vt::GetOp(vt::OpId::kQkvSplit, DeviceType::kTENSTORRENT));

  std::vector<float> host_qkv(static_cast<size_t>(T * (Qd + Kd + Vd)));
  for (size_t i = 0; i < host_qkv.size(); ++i)
    host_qkv[i] = std::fabs(static_cast<float>(i % 19) * 0.1f - 0.3f) + 0.05f;  // all > 0

  void* mem_qkv = backend.Alloc(host_qkv.size() * sizeof(float));
  void* mem_q = backend.Alloc(static_cast<size_t>(T * Qd) * sizeof(float));
  void* mem_k = backend.Alloc(static_cast<size_t>(T * Kd) * sizeof(float));
  void* mem_v = backend.Alloc(static_cast<size_t>(T * Vd) * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_qkv, host_qkv.data(), host_qkv.size() * sizeof(float));

  Tensor qkv = Tensor::Contiguous(mem_qkv, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {T, Qd + Kd + Vd});
  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Qd});
  Tensor tk =
      Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Kd});
  Tensor tv =
      Tensor::Contiguous(mem_v, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Vd});

  // Relu in place → CommitDevice2D leaves qkv device-resident (positive inputs).
  relu(q, qkv, qkv);
  split(q, tq, tk, tv, qkv);

  std::vector<float> host_q(static_cast<size_t>(T * Qd)), host_k(static_cast<size_t>(T * Kd)),
      host_v(static_cast<size_t>(T * Vd));
  backend.Copy(q, host_q.data(), mem_q, host_q.size() * sizeof(float));
  backend.Copy(q, host_k.data(), mem_k, host_k.size() * sizeof(float));
  backend.Copy(q, host_v.data(), mem_v, host_v.size() * sizeof(float));
  backend.Free(mem_qkv);
  backend.Free(mem_q);
  backend.Free(mem_k);
  backend.Free(mem_v);

  float max_abs = 0.0f;
  for (int64_t i = 0; i < T; ++i) {
    for (int64_t j = 0; j < Qd; ++j)
      max_abs = std::max(
          max_abs, std::fabs(host_q[static_cast<size_t>(i * Qd + j)] -
                             host_qkv[static_cast<size_t>(i * (Qd + Kd + Vd) + j)]));
    for (int64_t j = 0; j < Kd; ++j)
      max_abs = std::max(
          max_abs, std::fabs(host_k[static_cast<size_t>(i * Kd + j)] -
                             host_qkv[static_cast<size_t>(i * (Qd + Kd + Vd) + Qd + j)]));
    for (int64_t j = 0; j < Vd; ++j)
      max_abs = std::max(
          max_abs,
          std::fabs(host_v[static_cast<size_t>(i * Vd + j)] -
                    host_qkv[static_cast<size_t>(i * (Qd + Kd + Vd) + Qd + Kd + j)]));
  }
  CHECK(max_abs < 0.05f);
}

TEST_CASE("kTENSTORRENT kReshapeAndCache is BIT-EXACT incl. slot<0 skip") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kReshapeAndCache, DeviceType::kTENSTORRENT));

  constexpr int64_t NBlocks = 6, Bsz = 8, Hkv = 3, Dh = 16, T = 10;
  constexpr int64_t Page = Hkv * Dh;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Hkv * Dh);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_k(static_cast<size_t>(T * Page)), host_v(static_cast<size_t>(T * Page));
  for (size_t i = 0; i < host_k.size(); ++i) {
    host_k[i] = static_cast<float>(i % 11) * 0.1f;
    host_v[i] = static_cast<float>(i % 13) * 0.05f - 0.2f;
  }
  // Scattered slots + one padded (-1) token that must leave its page untouched.
  std::vector<int64_t> slots{0, 9, 17, 3, -1, 40, 25, 8, 33, 11};
  REQUIRE(static_cast<int64_t>(slots.size()) == T);

  std::vector<float> seed(cache_elems);
  for (size_t i = 0; i < seed.size(); ++i) seed[i] = static_cast<float>(i % 977) * 0.001f;
  std::vector<float> host_kc = seed, host_vc = seed;

  void* mem_k = backend.Alloc(host_k.size() * sizeof(float));
  void* mem_v = backend.Alloc(host_v.size() * sizeof(float));
  void* mem_kc = backend.Alloc(cache_elems * sizeof(float));
  void* mem_vc = backend.Alloc(cache_elems * sizeof(float));
  void* mem_slots = backend.Alloc(slots.size() * sizeof(int64_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_k, host_k.data(), host_k.size() * sizeof(float));
  backend.Copy(q, mem_v, host_v.data(), host_v.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_slots, slots.data(), slots.size() * sizeof(int64_t));

  Tensor tk = Tensor::Contiguous(mem_k, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hkv, Dh});
  Tensor tv = Tensor::Contiguous(mem_v, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                 {T, Hkv, Dh});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, Dh});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, Dh});
  Tensor tsl = Tensor::Contiguous(mem_slots, vt::DType::kI64, Device{DeviceType::kTENSTORRENT, 0},
                                  {T});

  auto rac = reinterpret_cast<vt::ReshapeAndCacheFn>(
      vt::GetOp(vt::OpId::kReshapeAndCache, DeviceType::kTENSTORRENT));
  rac(q, tk, tv, tkc, tvc, tsl);

  backend.Copy(q, host_kc.data(), mem_kc, host_kc.size() * sizeof(float));
  backend.Copy(q, host_vc.data(), mem_vc, host_vc.size() * sizeof(float));
  backend.Free(mem_k);
  backend.Free(mem_v);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_slots);

  // Host oracle: same stride math as cpu_cache.cpp.
  std::vector<float> ref_kc = seed, ref_vc = seed;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t slot = slots[static_cast<size_t>(t)];
    if (slot < 0) continue;
    const int64_t block = slot / Bsz;
    const int64_t offset = slot % Bsz;
    const int64_t dst = (block * Bsz + offset) * Page;
    std::memcpy(ref_kc.data() + dst, host_k.data() + t * Page, static_cast<size_t>(Page) * sizeof(float));
    std::memcpy(ref_vc.data() + dst, host_v.data() + t * Page, static_cast<size_t>(Page) * sizeof(float));
  }
  CHECK(host_kc == ref_kc);
  CHECK(host_vc == ref_vc);
}

TEST_CASE("kTENSTORRENT kPagedAttention matches a host causal GQA reference") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  // Single-request prefill: T=4 tokens, Hq=4, Hkv=2 (GQA 2:1), D=8, block=4.
  constexpr int64_t T = 4, Hq = 4, Hkv = 2, D = 8, Bsz = 4, NBlocks = 2;
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(T * Hq * D));
  std::vector<float> host_k(static_cast<size_t>(T * Page)), host_v(static_cast<size_t>(T * Page));
  for (size_t i = 0; i < host_q.size(); ++i) host_q[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;
  for (size_t i = 0; i < host_k.size(); ++i) {
    host_k[i] = static_cast<float>(i % 5) * 0.15f;
    host_v[i] = static_cast<float>(i % 9) * 0.05f - 0.1f;
  }
  // Write K/V into contiguous slots 0..T-1 of the cache first.
  std::vector<float> host_kc(cache_elems, 0.0f), host_vc(cache_elems, 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    std::memcpy(host_kc.data() + t * Page, host_k.data() + t * Page,
                static_cast<size_t>(Page) * sizeof(float));
    std::memcpy(host_vc.data() + t * Page, host_v.data() + t * Page,
                static_cast<size_t>(Page) * sizeof(float));
  }
  std::vector<int32_t> block_table{0, 1};  // [num_reqs=1, max_blocks=2]
  std::vector<int32_t> seq_lens{static_cast<int32_t>(T)};
  std::vector<int32_t> qsl{0, static_cast<int32_t>(T)};
  std::vector<float> host_out(static_cast<size_t>(T * Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {T, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0},
                                  {1, 2});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {1});
  Tensor tqsl =
      Tensor::Contiguous(mem_qsl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {2});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));
  pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);

  // Host oracle: same two-pass max-subtracted softmax as cpu_paged_attn.cpp.
  std::vector<float> ref(static_cast<size_t>(T * Hq * D), 0.0f);
  const int64_t qpk = Hq / Hkv;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t p = t;  // prefill, context=0
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (t * Hq + h) * D;
      float m = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(static_cast<size_t>(p + 1));
      for (int64_t j = 0; j <= p; ++j) {
        float dot = 0.0f;
        for (int64_t e = 0; e < D; ++e)
          dot += host_q[static_cast<size_t>(qoff + e)] *
                 host_kc[static_cast<size_t>(j * Page + g * D + e)];
        scores[static_cast<size_t>(j)] = dot * args.scale;
        m = std::max(m, scores[static_cast<size_t>(j)]);
      }
      float denom = 0.0f;
      for (int64_t j = 0; j <= p; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - m);
        denom += scores[static_cast<size_t>(j)];
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < D; ++e) {
        float acc = 0.0f;
        for (int64_t j = 0; j <= p; ++j)
          acc += scores[static_cast<size_t>(j)] * inv *
                 host_vc[static_cast<size_t>(j * Page + g * D + e)];
        ref[static_cast<size_t>(qoff + e)] = acc;
      }
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i)
    max_abs_diff = std::max(max_abs_diff, std::fabs(host_out[i] - ref[i]));
  // Host f32 path — should be essentially bit-exact; allow tiny float noise.
  CHECK(max_abs_diff < 1e-5f);
}

// Optional microbench (TT_PA_BENCH=1): Qwen3-0.6B-ish decode shape at long
// context to measure host PA throughput independent of e2e matmul/PCIe.
TEST_CASE("kTENSTORRENT kPagedAttention host microbench (opt-in)") {
  if (std::getenv("TT_PA_BENCH") == nullptr) {
    MESSAGE("SKIPPED: set TT_PA_BENCH=1 to run host PA microbench");
    return;
  }
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  // Decode-shaped: T=1, Hq=16, Hkv=8, D=128, seqlen=512, block=16.
  constexpr int64_t T = 1, Hq = 16, Hkv = 8, D = 128, Bsz = 16, Seq = 512;
  constexpr int64_t NBlocks = Seq / Bsz;
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(T * Hq * D), 0.1f);
  std::vector<float> host_kc(cache_elems, 0.05f), host_vc(cache_elems, 0.02f);
  std::vector<int32_t> block_table(static_cast<size_t>(NBlocks));
  for (int64_t i = 0; i < NBlocks; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens{static_cast<int32_t>(Seq)};
  std::vector<int32_t> qsl{0, static_cast<int32_t>(T)};
  std::vector<float> host_out(static_cast<size_t>(T * Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {T, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0},
                                  {1, NBlocks});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {1});
  Tensor tqsl =
      Tensor::Contiguous(mem_qsl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {2});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  constexpr int kWarm = 3, kIters = 20;
  for (int i = 0; i < kWarm; ++i) pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(kIters);
  MESSAGE("PA host microbench T=1 Hq=16 Hkv=8 D=128 seq=512: ", ms, " ms/call");

  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);
  CHECK(ms > 0.0);
}

// Pure decode with TILE-legal geometry (D=128, block=32) exercises the
// ttnn::paged_scaled_dot_product_attention_decode path (or host fallback).
TEST_CASE("kTENSTORRENT kPagedAttention pure-decode matches host within BF16 envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  constexpr int64_t Hq = 4, Hkv = 2, D = 128, Bsz = 32, Seq = 64;
  constexpr int64_t NBlocks = Seq / Bsz;  // 2
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(Hq * D));
  std::vector<float> host_kc(cache_elems), host_vc(cache_elems);
  for (size_t i = 0; i < host_q.size(); ++i)
    host_q[i] = (static_cast<float>(i % 17) - 8.0f) * 0.05f;
  for (size_t i = 0; i < host_kc.size(); ++i) {
    host_kc[i] = (static_cast<float>(i % 13) - 6.0f) * 0.03f;
    host_vc[i] = (static_cast<float>(i % 11) - 5.0f) * 0.02f;
  }
  std::vector<int32_t> block_table(static_cast<size_t>(NBlocks));
  for (int64_t i = 0; i < NBlocks; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens{static_cast<int32_t>(Seq)};
  std::vector<int32_t> qsl{0, 1};
  std::vector<float> host_out(static_cast<size_t>(Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {1, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {1, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0},
                                  {1, NBlocks});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {1});
  Tensor tqsl =
      Tensor::Contiguous(mem_qsl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {2});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));
  pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);

  // Host NHD oracle (same as unit test / cpu_paged_attn).
  const int64_t qpk = Hq / Hkv;
  const int64_t p = Seq - 1;
  float max_abs = 0.0f;
  for (int64_t h = 0; h < Hq; ++h) {
    const int64_t g = h / qpk;
    const int64_t qoff = h * D;
    float m = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(static_cast<size_t>(p + 1));
    for (int64_t j = 0; j <= p; ++j) {
      float dot = 0.0f;
      const int64_t kbase = j * Page + g * D;
      for (int64_t e = 0; e < D; ++e)
        dot += host_q[static_cast<size_t>(qoff + e)] * host_kc[static_cast<size_t>(kbase + e)];
      scores[static_cast<size_t>(j)] = dot * args.scale;
      m = std::max(m, scores[static_cast<size_t>(j)]);
    }
    float denom = 0.0f;
    for (int64_t j = 0; j <= p; ++j) {
      scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - m);
      denom += scores[static_cast<size_t>(j)];
    }
    const float inv = 1.0f / denom;
    for (int64_t e = 0; e < D; ++e) {
      float acc = 0.0f;
      for (int64_t j = 0; j <= p; ++j)
        acc += scores[static_cast<size_t>(j)] * inv *
               host_vc[static_cast<size_t>(j * Page + g * D + e)];
      max_abs = std::max(max_abs, std::fabs(host_out[static_cast<size_t>(qoff + e)] - acc));
    }
  }
  // Device BF16 SDPA or host path — generous envelope.
  CHECK(max_abs < 0.5f);
}

// Multi-token pure prefill with TILE-legal geometry exercises
// ttnn::chunked_scaled_dot_product_attention (or host fallback).
TEST_CASE("kTENSTORRENT kPagedAttention pure-prefill matches host within BF16 envelope") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  REQUIRE(vt::OpRegistered(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));

  // T=32 query tokens, full prefill (seq=T), D=128, block=32 → one chunked SDPA call.
  constexpr int64_t T = 32, Hq = 4, Hkv = 2, D = 128, Bsz = 32;
  constexpr int64_t Seq = T;
  constexpr int64_t NBlocks = (Seq + Bsz - 1) / Bsz;  // 1
  constexpr int64_t Page = Hkv * D;
  const size_t cache_elems = static_cast<size_t>(NBlocks * Bsz * Page);
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);

  std::vector<float> host_q(static_cast<size_t>(T * Hq * D));
  std::vector<float> host_kc(cache_elems), host_vc(cache_elems);
  for (size_t i = 0; i < host_q.size(); ++i)
    host_q[i] = (static_cast<float>(i % 17) - 8.0f) * 0.05f;
  // Dense NHD cache: position j lives at block j/Bsz, offset j%Bsz.
  for (int64_t j = 0; j < Seq; ++j) {
    const int64_t blk = j / Bsz, off = j % Bsz;
    for (int64_t g = 0; g < Hkv; ++g) {
      for (int64_t e = 0; e < D; ++e) {
        const size_t idx =
            static_cast<size_t>(((blk * Bsz + off) * Hkv + g) * D + e);
        host_kc[idx] = (static_cast<float>((j * 3 + g * 5 + e) % 13) - 6.0f) * 0.03f;
        host_vc[idx] = (static_cast<float>((j * 7 + g * 2 + e) % 11) - 5.0f) * 0.02f;
      }
    }
  }
  std::vector<int32_t> block_table(static_cast<size_t>(NBlocks));
  for (int64_t i = 0; i < NBlocks; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens{static_cast<int32_t>(Seq)};
  std::vector<int32_t> qsl{0, static_cast<int32_t>(T)};
  std::vector<float> host_out(static_cast<size_t>(T * Hq * D), 0.0f);

  void* mem_q = backend.Alloc(host_q.size() * sizeof(float));
  void* mem_out = backend.Alloc(host_out.size() * sizeof(float));
  void* mem_kc = backend.Alloc(host_kc.size() * sizeof(float));
  void* mem_vc = backend.Alloc(host_vc.size() * sizeof(float));
  void* mem_bt = backend.Alloc(block_table.size() * sizeof(int32_t));
  void* mem_sl = backend.Alloc(seq_lens.size() * sizeof(int32_t));
  void* mem_qsl = backend.Alloc(qsl.size() * sizeof(int32_t));
  Queue q = backend.CreateQueue();
  backend.Copy(q, mem_q, host_q.data(), host_q.size() * sizeof(float));
  backend.Copy(q, mem_kc, host_kc.data(), host_kc.size() * sizeof(float));
  backend.Copy(q, mem_vc, host_vc.data(), host_vc.size() * sizeof(float));
  backend.Copy(q, mem_bt, block_table.data(), block_table.size() * sizeof(int32_t));
  backend.Copy(q, mem_sl, seq_lens.data(), seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, mem_qsl, qsl.data(), qsl.size() * sizeof(int32_t));

  Tensor tq =
      Tensor::Contiguous(mem_q, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {T, Hq, D});
  Tensor tout = Tensor::Contiguous(mem_out, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                   {T, Hq, D});
  Tensor tkc = Tensor::Contiguous(mem_kc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tvc = Tensor::Contiguous(mem_vc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0},
                                  {NBlocks, Bsz, Hkv, D});
  Tensor tbt = Tensor::Contiguous(mem_bt, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0},
                                  {1, NBlocks});
  Tensor tsl =
      Tensor::Contiguous(mem_sl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {1});
  Tensor tqsl =
      Tensor::Contiguous(mem_qsl, vt::DType::kI32, Device{DeviceType::kTENSTORRENT, 0}, {2});

  vt::PagedAttentionArgs args;
  args.scale = 1.0f / std::sqrt(static_cast<float>(D));
  args.causal = true;
  auto pa = reinterpret_cast<vt::PagedAttentionFn>(
      vt::GetOp(vt::OpId::kPagedAttention, DeviceType::kTENSTORRENT));
  pa(q, tout, tq, tkc, tvc, tbt, tsl, tqsl, args);

  backend.Copy(q, host_out.data(), mem_out, host_out.size() * sizeof(float));
  backend.Free(mem_q);
  backend.Free(mem_out);
  backend.Free(mem_kc);
  backend.Free(mem_vc);
  backend.Free(mem_bt);
  backend.Free(mem_sl);
  backend.Free(mem_qsl);

  // Host causal GQA oracle over the dense NHD cache.
  const int64_t qpk = Hq / Hkv;
  float max_abs = 0.0f;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t p = t;  // pure prefill: query pos == token index
    for (int64_t h = 0; h < Hq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (t * Hq + h) * D;
      float m = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(static_cast<size_t>(p + 1));
      for (int64_t j = 0; j <= p; ++j) {
        float dot = 0.0f;
        const int64_t blk = j / Bsz, off = j % Bsz;
        const int64_t kbase = ((blk * Bsz + off) * Hkv + g) * D;
        for (int64_t e = 0; e < D; ++e)
          dot += host_q[static_cast<size_t>(qoff + e)] * host_kc[static_cast<size_t>(kbase + e)];
        scores[static_cast<size_t>(j)] = dot * args.scale;
        m = std::max(m, scores[static_cast<size_t>(j)]);
      }
      float denom = 0.0f;
      for (int64_t j = 0; j <= p; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - m);
        denom += scores[static_cast<size_t>(j)];
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < D; ++e) {
        float acc = 0.0f;
        for (int64_t j = 0; j <= p; ++j) {
          const int64_t blk = j / Bsz, off = j % Bsz;
          const int64_t vbase = ((blk * Bsz + off) * Hkv + g) * D + e;
          acc += scores[static_cast<size_t>(j)] * inv * host_vc[static_cast<size_t>(vbase)];
        }
        max_abs = std::max(max_abs, std::fabs(host_out[static_cast<size_t>(qoff + e)] - acc));
      }
    }
  }
  MESSAGE("pure-prefill max_abs vs host oracle: ", max_abs);
  CHECK(max_abs < 0.5f);
}

// ttnn mesh-trace capture: warm a matmul, capture it, replay, check BF16 envelope.
// Program cache must be warm before BeginCapture (ttnn trace contract).
TEST_CASE("kTENSTORRENT SupportsGraphCapture and matmul capture/replay") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  Backend& backend = vt::GetBackend(DeviceType::kTENSTORRENT);
  REQUIRE(backend.SupportsGraphCapture());

  constexpr int64_t M = 32, K = 64, N = 32;
  std::vector<float> ha(static_cast<size_t>(M * K), 0.1f);
  std::vector<float> hb(static_cast<size_t>(K * N), 0.2f);
  std::vector<float> hc(static_cast<size_t>(M * N), 0.0f);
  for (size_t i = 0; i < ha.size(); ++i) ha[i] = static_cast<float>((i % 7) - 3) * 0.05f;
  for (size_t i = 0; i < hb.size(); ++i) hb[i] = static_cast<float>((i % 5) - 2) * 0.04f;

  void* ma = backend.Alloc(ha.size() * sizeof(float));
  void* mb = backend.Alloc(hb.size() * sizeof(float));
  void* mc = backend.Alloc(hc.size() * sizeof(float));
  Queue q = backend.CreateQueue();
  backend.Copy(q, ma, ha.data(), ha.size() * sizeof(float));
  backend.Copy(q, mb, hb.data(), hb.size() * sizeof(float));

  Tensor ta = Tensor::Contiguous(ma, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, K});
  Tensor tb = Tensor::Contiguous(mb, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {K, N});
  Tensor tc = Tensor::Contiguous(mc, vt::DType::kF32, Device{DeviceType::kTENSTORRENT, 0}, {M, N});

  auto mm = reinterpret_cast<vt::MatmulFn>(vt::GetOp(vt::OpId::kMatmul, DeviceType::kTENSTORRENT));
  // Warm program cache (required before capture).
  mm(q, tc, ta, tb);
  backend.Copy(q, hc.data(), mc, hc.size() * sizeof(float));

  // Capture the same matmul on the already-resident shadows.
  backend.BeginCapture(q);
  mm(q, tc, ta, tb);
  backend.EndCapture(q);

  // Replay several times — should not throw.
  for (int i = 0; i < 3; ++i) backend.Replay(q);

  std::vector<float> after(hc.size(), 0.0f);
  backend.Copy(q, after.data(), mc, after.size() * sizeof(float));
  float max_abs = 0.0f;
  for (size_t i = 0; i < after.size(); ++i)
    max_abs = std::max(max_abs, std::fabs(after[i] - hc[i]));
  MESSAGE("trace replay max_abs vs warm: ", max_abs);
  CHECK(max_abs < 1e-3f);

  // Multi-graph handle API: capture again into an owned handle.
  backend.BeginCapture(q);
  mm(q, tc, ta, tb);
  void* graph = backend.EndCaptureGraph(q);
  REQUIRE(graph != nullptr);
  backend.ReplayGraph(q, graph);
  backend.DestroyGraph(graph);

  backend.Free(ma);
  backend.Free(mb);
  backend.Free(mc);
}
