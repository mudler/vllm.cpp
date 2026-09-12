// Ported from vLLM tests/kernels/ir/test_layernorm.py::TestFusedAddRMSNorm
// at e126687a9a828d513c01a07cd69f025f27d63280. The compiled boundary is
// additionally pinned by the executing row-zero witness in #3103.
#include <doctest/doctest.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>
#include <vector>
#include "support/residual_norm_fixture.h"
#include "support/residual_norm_later_fixture.h"
#include "support/residual_norm_test.h"
#include "vt/ops.h"

TEST_CASE("residual RMS norm preserves the compiled primary row zero") {
  using namespace residual_norm_fixture;
  const vt::Device device{vt::DeviceType::kCPU, 0};
  vt::Queue q{device, nullptr};
  auto a = kAttention, r = kResidual, w = kGamma;
  std::vector<uint16_t> out(128);
  auto ta = vt::Tensor::Contiguous(a.data(), vt::DType::kBF16, device, {1, 128});
  auto tr = vt::Tensor::Contiguous(r.data(), vt::DType::kBF16, device, {1, 128});
  auto tw = vt::Tensor::Contiguous(w.data(), vt::DType::kBF16, device, {128});
  auto to = vt::Tensor::Contiguous(out.data(), vt::DType::kBF16, device, {1, 128});
  vt::ResidualRmsNorm(q, to, ta, tr, nullptr, tw, vt::ResidualRmsNormArgs{});
  for (size_t j = 0; j < out.size(); ++j) {
    CAPTURE(j);
    CHECK(out[j] == kPostNorm[j]);
  }
  CHECK(a == kAttention);
  CHECK(r == kResidual);
  CHECK(w == kGamma);
}

TEST_CASE("residual RMS norm replays the actual later compiled primary outputs") {
  using namespace residual_norm_fixture;
  using namespace residual_norm_test;
  for (auto device : Devices()) for (bool final : {false, true}) for (int mode : {0, 1, 2}) {
    CAPTURE(device.type); CAPTURE(final); CAPTURE(mode);
    Queue queue(device); auto& q = queue.q;
    auto vector = [](const auto& values) { return std::vector<uint16_t>(values.begin(), values.end()); };
    const auto a = vector(final ? kFinalAttention : kNextAttention);
    const auto base = vector(final ? kFinalBase : kNextBase);
    const auto delta = vector(final ? kFinalMoe : kNextMoe);
    const auto gamma = vector(final ? kFinalGamma : kNextGamma);
    Buffer da(q, a, 1, 128), dr(q, base, 1, 128), dm(q, delta, 1, 128), dw(q, gamma, -1, 128);
    Buffer out(q, std::vector<uint16_t>(128), 1, 128), res(q, std::vector<uint16_t>(128), 1, 128);
    vt::ResidualRmsNormArgs args{1e-6f, {vt::ResidualNormExpr::kDeltaPlusAdd, !final}};
    Run(mode, q, out.t, da.t, dr.t, &dm.t, dw.t, args, final ? nullptr : &res.t);
    CHECK(out.Read() == vector(final ? kFinalNorm : kNextNorm));
    if (!final) CHECK(res.Read() == vector(kNextResidual));
    CHECK(da.Read() == a); CHECK(dr.Read() == base); CHECK(dm.Read() == delta); CHECK(dw.Read() == gamma);
  }
}

TEST_CASE("residual RMS norm ordered expressions preserve every observable boundary") {
  using namespace residual_norm_test;
  for (const auto device : Devices()) for (int64_t width : {8, 128, 769, 8192})
  for (bool materialize : {false, true}) for (bool alias : {false, true})
  for (int witness : {0, 1, 2}) for (int mode : {0, 1, 2}) {
    CAPTURE(device.type); CAPTURE(width); CAPTURE(materialize);
    CAPTURE(alias); CAPTURE(witness); CAPTURE(mode);
    constexpr int64_t rows = 3;
    const int64_t stride = width + 7;
    std::vector<uint16_t> a(rows * stride, Buffer::kGuard), base(a), delta(a);
    std::vector<uint16_t> gamma(width), blank(a.size(), Buffer::kGuard);
    for (int64_t row = 0; row < rows; ++row) for (int64_t j = 0; j < width; ++j) {
      const auto at = static_cast<size_t>(row * stride + j);
      if (witness == 0) { a[at] = vt::F32ToBF16(256); base[at] = vt::F32ToBF16(-256); delta[at] = vt::F32ToBF16(0x1p-17f); }
      if (witness == 1) { a[at] = vt::F32ToBF16(1); base[at] = delta[at] = vt::F32ToBF16(0x1p-8f); }
      if (witness == 2) {
        const size_t k = static_cast<size_t>(j % 128);
        a[at] = residual_norm_fixture::kAttention[k];
        base[at] = residual_norm_fixture::kResidual[k]; delta[at] = 0;
      }
      gamma[j] = witness == 2 ? residual_norm_fixture::kGamma[j % 128] : vt::F32ToBF16(0.75f);
    }
    std::vector<uint16_t> expected_residual;
    const auto expected = Reference(a, base, &delta, gamma, rows, width, stride, 1e-6f, &expected_residual);
    Queue queue(device); auto& q = queue.q;
    Buffer da(q, a, rows, width, stride), dr(q, base, rows, width, stride);
    Buffer dm(q, delta, rows, width, stride), dw(q, gamma, -1, width);
    Buffer output(q, blank, rows, width, stride), residual(q, blank, rows, width, stride);
    vt::ResidualRmsNormArgs args{1e-6f, {vt::ResidualNormExpr::kDeltaPlusAdd, materialize, alias, materialize && alias}};
    vt::Tensor& out = alias ? dm.t : output.t;
    vt::Tensor* res = materialize ? (alias ? &dr.t : &residual.t) : nullptr;
    Run(mode, q, out, da.t, dr.t, &dm.t, dw.t, args, res);
    const auto actual = alias ? dm.Read() : output.Read();
    // The constant witnesses have exact reductions. The measured row also
    // guards gamma rounding and normalization from the stored residual.
    CHECK(actual == expected);
    CHECK(da.Read() == a); CHECK(dw.Read() == gamma);
    if (!alias) CHECK(dm.Read() == delta);
    if (!(alias && materialize)) CHECK(dr.Read() == base);
    if (materialize) CHECK((alias ? dr.Read() : residual.Read()) == expected_residual);
    else CHECK(residual.Read() == blank);
    for (const auto* buffer : {&da, &dr, &dm, &dw, &output, &residual}) buffer->CheckGuard();
  }
}

TEST_CASE("residual RMS norm descriptors reject invalid operands and overlaps before dispatch") {
  using namespace residual_norm_test;
  const vt::Device cpu{vt::DeviceType::kCPU, 0};
  vt::Queue q{cpu, nullptr};
  std::vector<uint16_t> storage(128, vt::F32ToBF16(1));
  auto a = vt::Tensor::Contiguous(storage.data(), vt::DType::kBF16, cpu, {2, 8});
  auto base = a, delta = a, gamma = vt::Tensor::Contiguous(storage.data() + 48, vt::DType::kBF16, cpu, {8});
  auto out = a, residual = a;
  base.data = storage.data() + 16; delta.data = storage.data() + 32;
  out.data = storage.data() + 64; residual.data = storage.data() + 80;
  const auto good = vt::ResidualRmsNormArgs{};
  auto run = [&](const vt::ResidualRmsNormArgs& args, const vt::Tensor* m = nullptr, vt::Tensor* r = nullptr) {
    vt::ResidualRmsNorm(q, out, a, base, m, gamma, args, r);
  };
  CHECK_NOTHROW(run(good));
  auto args = good; args.descriptor.expression = static_cast<vt::ResidualNormExpr>(255);
  CHECK_THROWS_AS(run(args), std::runtime_error);
  CHECK_THROWS_AS(run(good, &delta), std::runtime_error);
  args = good; args.descriptor.expression = vt::ResidualNormExpr::kDeltaPlusAdd;
  CHECK_THROWS_AS(run(args), std::runtime_error);
  args = good; args.descriptor.materialize_residual = true;
  CHECK_THROWS_AS(run(args), std::runtime_error);
  CHECK_THROWS_AS(run(good, nullptr, &residual), std::runtime_error);
  args = good; args.descriptor.output_alias_delta = true;
  CHECK_THROWS_AS(run(args), std::runtime_error);
  args = good; args.descriptor.residual_alias_base = true;
  CHECK_THROWS_AS(run(args), std::runtime_error);
  for (float epsilon : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    args = good; args.eps = epsilon; CHECK_THROWS_AS(run(args), std::runtime_error);
  }
  for (auto* tensor : {&a, &base, &gamma, &out}) {
    const auto original = *tensor;
    for (auto dtype : {vt::DType::kF16, vt::DType::kF32, vt::DType::kI32}) {
      tensor->dtype = dtype; CHECK_THROWS_AS(run(good), std::runtime_error);
    }
    *tensor = original; tensor->device.index = 1;
    CHECK_THROWS_AS(run(good), std::runtime_error);
    *tensor = original; tensor->data = nullptr;
    CHECK_THROWS_AS(run(good), std::runtime_error);
    *tensor = original; tensor->data = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(original.data) + 1);
    CHECK_THROWS_AS(run(good), std::runtime_error);
    *tensor = original; tensor->shape[0] += 1;
    CHECK_THROWS_AS(run(good), std::runtime_error);
    *tensor = original; tensor->stride[tensor->rank - 1] = 2;
    CHECK_THROWS_AS(run(good), std::runtime_error);
    *tensor = original;
  }
  auto original_out = out;
  for (const auto* input : {&a, &base, &delta, &gamma}) {
    out.data = input->data;
    CHECK_THROWS_AS(run(good, input == &delta ? &delta : nullptr), std::runtime_error);
  }
  out = original_out; out.data = storage.data() + 1;
  CHECK_THROWS_AS(run(good), std::runtime_error);
  out = original_out; args = {1e-6f, {vt::ResidualNormExpr::kDeltaPlusAdd, true, true, true}};
  residual.data = out.data; CHECK_THROWS_AS(run(args, &delta, &residual), std::runtime_error);
  residual.data = storage.data() + 80; out.data = delta.data;
  args.descriptor.output_alias_delta = false;
  CHECK_THROWS_AS(run(args, &delta, &residual), std::runtime_error);
  out = original_out; args.descriptor.output_alias_delta = true; residual.data = base.data;
  args.descriptor.residual_alias_base = false;
  CHECK_THROWS_AS(run(args, &delta, &residual), std::runtime_error);
  residual.data = storage.data() + 80;
  const auto original_a = a; a.stride[0] = 7;
  CHECK_THROWS_AS(run(good), std::runtime_error);
  a = original_a; a.stride[0] = std::numeric_limits<int64_t>::max();
  CHECK_THROWS_AS(run(good), std::runtime_error);
  a = original_a; q.device.index = -1;
  CHECK_THROWS_AS(run(good), std::runtime_error); q.device.index = 0;
  a.shape[0] = base.shape[0] = out.shape[0] = 0;
  a.data = base.data = out.data = nullptr;
  CHECK_NOTHROW(run(good));
  CHECK(vt::GetBackend(cpu).GetResidualNormPolicy() == vt::ResidualNormPolicy::kMaterialized);
}

TEST_CASE("residual RMS norm matches the complete pinned upstream BF16 fixture grid") {
  using namespace residual_norm_test;
  const char* directory = std::getenv("VT_RESIDUAL_NORM_UPSTREAM");
  if (directory == nullptr) {
    MESSAGE("Upstream fixture gate not executed: set VT_RESIDUAL_NORM_UPSTREAM to the complete pinned export");
    return;
  }
  const std::filesystem::path path(directory);
  nlohmann::json manifest; std::ifstream(path / "cases.json") >> manifest;
  REQUIRE(manifest["pin"] == "e126687a9a828d513c01a07cd69f025f27d63280");
  REQUIRE(manifest["complete"] == true);
  REQUIRE(manifest["cases"].size() >= 192);  // one-device core + complete IR grid
  const auto read = [&](const nlohmann::json& record) {
    const auto file = path / record["file"].get<std::string>();
    const size_t bytes = record["bytes"];
    REQUIRE(record["dtype"] == "torch.bfloat16");
    REQUIRE(std::filesystem::file_size(file) == bytes);
    std::vector<uint16_t> values(bytes / sizeof(uint16_t));
    std::ifstream stream(file, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    REQUIRE(stream.good());
    return values;
  };
  size_t executed = 0;
  for (const auto& record : manifest["cases"]) {
    CAPTURE(record["id"].get<std::string>());
    REQUIRE(record["seed"] == 0);
    const int64_t rows = record["rows"], width = record["width"];
    const int64_t stride = record["files"]["a"]["stride"][0];
    const bool materialize = record["add_residual"];
    const auto gamma = read(record["files"]["gamma"]);
    const auto logical_a = read(record["files"]["a"]), logical_base = read(record["files"]["base"]);
    const auto expected = read(record["files"]["norm"]), unit_expected = read(record["files"]["unit_norm"]);
    const auto expected_residual = materialize ? read(record["files"]["residual"]) : std::vector<uint16_t>{};
    std::vector<uint16_t> a(rows * stride, Buffer::kGuard), base(a), blank(a);
    for (int64_t row = 0; row < rows; ++row) for (int64_t j = 0; j < width; ++j) {
      a[row * stride + j] = logical_a[row * width + j];
      base[row * stride + j] = logical_base[row * width + j];
    }
    auto devices = Devices();
#if defined(VLLM_CPP_HIP)
    devices.back().index = record["device"];
#endif
    const float atol = record["tolerance"]["atol"], rtol = record["tolerance"]["rtol"];
    const auto compare = [&](const std::vector<uint16_t>& actual, const std::vector<uint16_t>& reference) {
      size_t bad = 0, first = 0;
      float maximum = 0;
      for (int64_t row = 0; row < rows; ++row) for (int64_t j = 0; j < width; ++j) {
        const auto at = static_cast<size_t>(row * width + j);
        const float target = vt::BF16ToF32(reference[at]);
        const float error = std::abs(vt::BF16ToF32(actual[row * stride + j]) - target);
        if (!(error <= atol + rtol * std::abs(target))) { if (bad == 0) first = at; ++bad; }
        maximum = std::max(maximum, error);
      }
      CAPTURE(first); CAPTURE(maximum); CHECK(bad == 0);
    };
    for (const auto device : devices) {
      CAPTURE(device.type); CAPTURE(device.index);
      Queue queue(device); auto& q = queue.q;
      Buffer da(q, a, rows, width, stride), dr(q, base, rows, width, stride), dw(q, gamma, -1, width);
      Buffer out(q, blank, rows, width, stride), res(q, blank, rows, width, stride);
      vt::ResidualRmsNormArgs args{record["epsilon"], {vt::ResidualNormExpr::kAdd, materialize}};
      Run(0, q, out.t, da.t, dr.t, nullptr, dw.t, args, materialize ? &res.t : nullptr);
      const auto direct = out.Read(); compare(direct, expected);
      const auto direct_residual = res.Read();
      if (materialize) compare(direct_residual, expected_residual);
      Run(1, q, out.t, da.t, dr.t, nullptr, dw.t, args, materialize ? &res.t : nullptr);
      CHECK(out.Read() == direct); CHECK(res.Read() == direct_residual);
      Buffer unit(q, std::vector<uint16_t>(width, vt::F32ToBF16(1)), -1, width);
      Run(0, q, out.t, da.t, dr.t, nullptr, unit.t, args, materialize ? &res.t : nullptr);
      compare(out.Read(), unit_expected);
      CHECK(da.Read() == a); CHECK(dr.Read() == base); CHECK(dw.Read() == gamma);
      out.CheckGuard(); res.CheckGuard();
      ++executed;
    }
  }
  MESSAGE("Executed pinned upstream normalization cases across native/reference devices: ", executed);
  CHECK(executed >= manifest["cases"].size());
}

TEST_CASE("residual RMS norm fusion validates counts order and materialization") {
  using namespace residual_norm_test;
  const vt::Device cpu{vt::DeviceType::kCPU, 0};
  Queue queue(cpu); auto& q = queue.q;
  std::vector<uint16_t> values(8, vt::F32ToBF16(1));
  Buffer a(q, values, 1, 8), base(q, values, 1, 8), delta(q, values, 1, 8);
  Buffer gamma(q, values, -1, 8), out(q, values, 1, 8), residual(q, values, 1, 8);
  vt::ResidualRmsNormArgs args{1e-6f, {vt::ResidualNormExpr::kDeltaPlusAdd, true}};
  vt::FusedBinding binding{};
  binding.n = 6;
  binding.op[0] = &a.t; binding.op[1] = &base.t; binding.op[2] = &delta.t;
  binding.op[3] = &gamma.t; binding.op[4] = &out.t; binding.op[5] = &residual.t;
  const auto good = vt::ResidualRmsNormRecipe(args.descriptor);
  auto recipe = good;
  const auto run = [&] { vt::FusedChainComposite(q, recipe, binding, {}); };
  CHECK_NOTHROW(run());
  for (uint8_t count : {0, 2, 3, 5}) {
    recipe = good; recipe.steps[0].nin = count; CHECK_THROWS_AS(run(), std::runtime_error);
  }
  recipe = good; recipe.steps[0].residual_norm.expression = static_cast<vt::ResidualNormExpr>(255);
  CHECK_THROWS_AS(run(), std::runtime_error);
  recipe = good; recipe.steps[0].out2 = vt::kNoOperand;
  CHECK_THROWS_AS(run(), std::runtime_error);
  recipe = good; recipe.steps[0].gemma = true;
  CHECK_THROWS_AS(run(), std::runtime_error);
  recipe = good; binding.op[2] = nullptr;
  CHECK_THROWS_AS(run(), std::runtime_error);
  binding.op[2] = &delta.t;
  CHECK_THROWS_AS(vt::FusedChain(q, out.t, a.t, base.t, &delta.t, gamma.t, vt::ResidualRmsNormArgs{}), std::runtime_error);
  CHECK_THROWS_AS(vt::FusedChain(q, out.t, a.t, base.t, nullptr, gamma.t, args, &residual.t), std::runtime_error);
  CHECK_THROWS_AS(vt::FusedChain(q, out.t, a.t, base.t, nullptr, gamma.t, vt::ResidualRmsNormArgs{}, &residual.t), std::runtime_error);
  // The typed expression is composite-only. Both tier selections must retain
  // the explicit opcode rather than reinterpret it as a legacy RMSNorm recipe.
  const char* previous = std::getenv("VT_FUSED_TIER");
  const std::string saved = previous == nullptr ? "" : previous;
  std::vector<uint16_t> first;
  for (const char* tier : {"0", "1"}) {
    setenv("VT_FUSED_TIER", tier, 1);
    vt::FusedChain(q, out.t, a.t, base.t, &delta.t, gamma.t, args, &residual.t);
    if (first.empty()) first = out.Read(); else CHECK(out.Read() == first);
  }
  if (previous == nullptr) unsetenv("VT_FUSED_TIER"); else setenv("VT_FUSED_TIER", saved.c_str(), 1);
}

#if defined(VLLM_CPP_HIP)
TEST_CASE("ROCm residual RMS norm uses its queue across streams devices and graph replay") {
  using namespace residual_norm_test;
  int count = 0;
  REQUIRE(hipGetDeviceCount(&count) == hipSuccess);
  REQUIRE(count >= 2);  // This hardware gate owes the two-device upstream arm.
  for (int index : {0, 1}) {
    const vt::Device device{vt::DeviceType::kROCM, index};
    Queue queue(device), other(device); auto& q = queue.q;
    constexpr int64_t width = 769, rows = 3;
    std::vector<uint16_t> a(rows * width, vt::F32ToBF16(1));
    std::vector<uint16_t> base(a.size(), vt::F32ToBF16(0x1p-8f)), delta(base);
    std::vector<uint16_t> gamma(width, vt::F32ToBF16(0.75f)), blank(a.size(), 0);
    Buffer da(q, a, rows, width), dr(q, base, rows, width), dm(q, delta, rows, width);
    Buffer dw(q, gamma, -1, width), out(q, blank, rows, width), res(q, blank, rows, width);
    vt::ResidualRmsNormArgs args{1e-6f, {vt::ResidualNormExpr::kDeltaPlusAdd, true}};
    std::vector<uint16_t> expected_residual;
    const auto expected = Reference(a, base, &delta, gamma, rows, width, width, args.eps, &expected_residual);
    auto run = [&](vt::Queue& target) { vt::FusedChain(target, out.t, da.t, dr.t, &dm.t, dw.t, args, &res.t); };
    int ambient = -1;
    REQUIRE(hipSetDevice(1 - index) == hipSuccess);
    run(q);
    REQUIRE(hipGetDevice(&ambient) == hipSuccess); CHECK(ambient == 1 - index);
    CHECK(out.Read() == expected); CHECK(res.Read() == expected_residual);
    run(other.q);
    vt::GetBackend(device).Synchronize(other.q);
    CHECK(out.Read() == expected);
    // A valid tensor device descriptor with a stream owned by another device
    // must fail before launch, not silently select the ambient device.
    Queue foreign({vt::DeviceType::kROCM, 1 - index});
    vt::Queue wrong{device, foreign.q.handle};
    CHECK_THROWS_AS(run(wrong), std::runtime_error);
    REQUIRE(hipGetDevice(&ambient) == hipSuccess); CHECK(ambient == 1 - index);

    DeviceScope capture_device(device);
    auto& backend = vt::GetBackend(device);
    REQUIRE(backend.SupportsGraphCapture());
    backend.BeginCapture(q);
    run(q);
    void* graph = backend.EndCaptureGraph(q);
    REQUIRE(graph != nullptr);
    for (int repeat = 0; repeat < 3; ++repeat) {
      backend.Memset(q, out.t.data, 0, blank.size() * sizeof(uint16_t));
      backend.Memset(q, res.t.data, 0, blank.size() * sizeof(uint16_t));
      backend.ReplayGraph(q, graph);
      CHECK(out.Read() == expected); CHECK(res.Read() == expected_residual);
    }
    backend.DestroyGraph(graph);
    CHECK(da.Read() == a); CHECK(dr.Read() == base); CHECK(dm.Read() == delta);
    out.CheckGuard(); res.CheckGuard();
  }
}

TEST_CASE("ROCm residual RMS norm cannot run ahead of a blocked nondefault stream") {
  using namespace residual_norm_test;
  const vt::Device device{vt::DeviceType::kROCM, 0};
  DeviceScope scope(device);
  // The default backend queue uses hipStreamCreate. This witness needs an
  // independent stream so synchronizing the default stream cannot release it.
  // The device scope outlives both the owned stream and its queued work.
  struct Stream {
    hipStream_t handle = nullptr;
    ~Stream() { if (handle != nullptr) (void)hipStreamDestroy(handle); }
  } stream;
  REQUIRE(hipStreamCreateWithFlags(&stream.handle, hipStreamNonBlocking) == hipSuccess);
  vt::Queue q{device, stream.handle};
  unsigned flags = 0;
  REQUIRE(hipStreamGetFlags(static_cast<hipStream_t>(q.handle), &flags) == hipSuccess);
  REQUIRE((flags & hipStreamNonBlocking) != 0);
  constexpr int64_t width = 128;
  std::vector<uint16_t> zeros(width), gamma(width, vt::F32ToBF16(1));
  Buffer a(q, zeros, 1, width), base(q, zeros, 1, width), w(q, gamma, -1, width), out(q, zeros, 1, width);
  struct Release {
    hipStream_t stream;
    std::atomic<bool> ready{false};
    ~Release() {
      ready.store(true);
      // Drain the callback before its atomic flag or any operand is destroyed,
      // including when an assertion or the operation throws before release.
      (void)hipStreamSynchronize(stream);
    }
  } release{stream.handle};
  REQUIRE(hipLaunchHostFunc(static_cast<hipStream_t>(q.handle), [](void* pointer) {
    auto* ready = static_cast<std::atomic<bool>*>(pointer);
    while (!ready->load()) std::this_thread::yield();
  }, &release.ready) == hipSuccess);
  // Uniform BF16 0x3f3f is nonzero. Correct execution waits for this fill;
  // a mutated default-stream launch sees zeros and completes before release.
  REQUIRE(hipMemsetAsync(a.t.data, 0x3f, width * sizeof(uint16_t), static_cast<hipStream_t>(q.handle)) == hipSuccess);
  vt::ResidualRmsNorm(q, out.t, a.t, base.t, nullptr, w.t, {});
  REQUIRE(hipStreamSynchronize(nullptr) == hipSuccess);
  release.ready.store(true);
  CHECK(out.Read() == gamma);
}
#endif
