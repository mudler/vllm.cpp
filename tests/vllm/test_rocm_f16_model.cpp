// Real-artifact public-ABI gate for BACKEND-ROCM-F16-WEIGHTS (#3092).
// Invoked explicitly with the pinned GGUF, committed token fixture and result path.
#include <vllm.h>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/backend.h"
#include "rocm_f16_native_observer.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_runtime.h"
#include "../../src/vt/rocm/rocm_f16_conversion.h"

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using vt::DType;
using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::Tensor;

double Seconds(Clock::time_point since) {
  return std::chrono::duration<double>(Clock::now() - since).count();
}
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
struct Observation {
  std::mutex mutex;
  vt::EmbeddingFn embedding;
  vt::MatmulFn nn, bt;
  std::vector<Json> forward_batches;
  std::map<uintptr_t, Json> weights;
  size_t marked_gemms = 0, raw_f16_gemms = 0;
  Clock::time_point origin = Clock::now();
  static Observation* active;
  static void Record(const Tensor& weight, const char* operation) {
    auto& self = *active;
    std::lock_guard<std::mutex> lock(self.mutex);
    const uintptr_t address = reinterpret_cast<uintptr_t>(weight.data);
    if (!self.weights.contains(address)) {
      self.weights[address] = { {"operation", operation}, {"storage_dtype", vt::Name(weight.dtype)},
          {"weight_value_dtype", weight.weight_value_dtype ? vt::Name(*weight.weight_value_dtype) : "unset"},
          {"rows", weight.shape[0]}, {"columns", weight.shape[1]}, {"bytes", weight.Bytes()} };
    }
  }
  static void Gather(Queue& queue, Tensor& out, const Tensor& table, const Tensor& ids) {
    Require(queue.device.type == DeviceType::kROCM, "public automatic device must resolve ROCm");
    Record(table, "embedding");
    {
      std::lock_guard<std::mutex> lock(active->mutex);
      active->forward_batches.push_back({{"tokens", ids.shape[0]},
          {"time_seconds", Seconds(active->origin)}, {"queue", queue.id},
          {"table_storage", vt::Name(table.dtype)},
          {"table_values", table.weight_value_dtype ? vt::Name(*table.weight_value_dtype) : "unset"}});
    }
    active->embedding(queue, out, table, ids);
  }
  static void Gemm(Queue& queue, Tensor& out, const Tensor& a, const Tensor& b, bool transpose) {
    Require(queue.device.type == DeviceType::kROCM, "ordinary GEMM must resolve ROCm");
    Record(b, transpose ? "matmul_bt" : "matmul");
    {
      std::lock_guard<std::mutex> lock(active->mutex);
      if (b.weight_value_dtype) ++active->marked_gemms;
      else if (b.dtype == DType::kF16) ++active->raw_f16_gemms;
    }
    (transpose ? active->bt : active->nn)(queue, out, a, b);
  }
  static void Nn(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) { Gemm(q, out, a, b, false); }
  static void Bt(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) { Gemm(q, out, a, b, true); }
  Observation() {
    embedding = rocm_f16_test::RealEmbedding;
    Require(vt::GetOp(OpId::kEmbedding, DeviceType::kROCM) == reinterpret_cast<void*>(rocm_f16_test::WrapEmbedding), "native embedding interposition missing");
    nn = rocm_f16_test::RealNn;
    Require(vt::GetOp(OpId::kMatmul, DeviceType::kROCM) == reinterpret_cast<void*>(rocm_f16_test::WrapNn), "native NN interposition missing");
    bt = rocm_f16_test::RealBt;
    Require(vt::GetOp(OpId::kMatmulBT, DeviceType::kROCM) == reinterpret_cast<void*>(rocm_f16_test::WrapBt), "native BT interposition missing");
    for (auto op : {OpId::kEmbedding, OpId::kMatmul, OpId::kMatmulBT}) {
      const auto provider = vt::GetOpProviderStats(op, DeviceType::kROCM);
      Require(provider.last_selected && std::string(provider.last_selected) == vt::kNativeProviderName,
              "model gate requires native ROCm providers");
    }
    active = this;
    rocm_f16_test::on_embedding = Gather;
    rocm_f16_test::on_nn = Nn;
    rocm_f16_test::on_bt = Bt;
  }
  ~Observation() {
    rocm_f16_test::on_embedding = nullptr;
    rocm_f16_test::on_nn = nullptr;
    rocm_f16_test::on_bt = nullptr;
    active = nullptr;
  }
};
Observation* Observation::active = nullptr;
}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: test_rocm_f16_model MODEL.gguf TOKEN_FIXTURE.json CAPACITY OUTPUT.json\n";
    return 77;
  }
  Json report{{"workload_results", Json::array()}, {"complete", false}};
  const std::string output_path = argv[4];
  const auto save = [&] { std::ofstream(output_path) << report.dump(2) << '\n'; };
  try {
    Require(vt::rocm::DeviceAvailable(), "physical ROCm device required");
    const std::string model_path = argv[1], fixture_path = argv[2];
    const int capacity = std::stoi(argv[3]);
    Require(capacity == 1 || capacity == 4, "capacity must match the committed workload");
    std::ifstream input(fixture_path);
    const Json fixture = Json::parse(input);
    Require(fixture.at("row") == "BACKEND-ROCM-F16-WEIGHTS", "wrong workload fixture");
    const int vocabulary = fixture.at("checkpoint").at("vocab_size");
    Observation observe;
    vllm_model_params model = vllm_model_params_default();
    model.model_path = model_path.c_str(); model.device = 0;
    model.max_num_seqs = capacity; model.max_model_len = 256;
    vllm_engine* engine = nullptr;
    const auto load_start = Clock::now();
    const auto load_status = vllm_engine_load(&model, &engine);
    Require(load_status == VLLM_OK, vllm_last_error());
    struct EngineOwner { vllm_engine* engine; ~EngineOwner() { vllm_engine_free(engine); } } owner{engine};
    report.update({{"arm", "vllm.cpp-public-complete-tokens"}, {"capacity", capacity},
                {"load_seconds", Seconds(load_start)}, {"model_path", model_path},
                {"workload_results", Json::array()}, {"version", vllm_version()}});
    save();
    for (int repeat = 0; repeat < fixture.at("repeats_per_arm").get<int>(); ++repeat) {
      for (const auto& workload : fixture.at("workloads")) {
        if (workload.at("sequence_capacity").get<int>() != capacity) continue;
        const auto prompts = workload.at("input_ids").get<std::vector<std::vector<int32_t>>>();
        for (const auto& prompt : prompts)
          for (int32_t id : prompt) Require(id >= 0 && id < vocabulary, "prompt ID outside vocabulary");
        vllm_sampling_params sampling = vllm_sampling_params_default();
        sampling.temperature = 0.f; sampling.top_p = 1.f;
        sampling.repetition_penalty = 1.f; sampling.presence_penalty = 0.f;
        sampling.frequency_penalty = 0.f; sampling.seed = 0; sampling.has_seed = 1;
        sampling.ignore_eos = 1; sampling.stop = nullptr; sampling.n_stop = 0;
        sampling.max_tokens = workload.at("max_tokens"); sampling.min_tokens = sampling.max_tokens;
        const size_t begin_batch = observe.forward_batches.size();
        std::vector<Json> requests(prompts.size());
        std::vector<std::string> errors(prompts.size());
        std::barrier start_gate(static_cast<std::ptrdiff_t>(prompts.size() + 1));
        std::vector<std::thread> threads;
        for (size_t r = 0; r < prompts.size(); ++r) {
          threads.emplace_back([&, r] {
            std::vector<int32_t> tokens(static_cast<size_t>(sampling.max_tokens));
            int32_t count = 0;
            start_gate.arrive_and_wait();
            const auto started = Clock::now();
            const auto status = vllm_complete_tokens(engine, prompts[r].data(),
                static_cast<int32_t>(prompts[r].size()), &sampling, tokens.data(),
                sampling.max_tokens, &count, nullptr);
            requests[r] = {{"request_index", workload.at("request_indices").at(r)},
                           {"input_ids", prompts[r]}, {"output_ids", std::vector<int32_t>(tokens.begin(), tokens.begin() + std::clamp(count, 0, sampling.max_tokens))},
                           {"elapsed_seconds", Seconds(started)}, {"status", status}, {"count", count}};
            if (status != VLLM_OK) { errors[r] = vllm_last_error(); return; }
            if (count != sampling.max_tokens) { errors[r] = "generated length mismatch"; return; }
            requests[r] = {{"request_index", workload.at("request_indices").at(r)},
                           {"input_ids", prompts[r]}, {"output_ids", tokens},
                           {"elapsed_seconds", Seconds(started)}};
          });
        }
        const auto started = Clock::now();
        start_gate.arrive_and_wait();
        for (auto& thread : threads) thread.join();
        Json batches = Json::array();
        for (size_t i = begin_batch; i < observe.forward_batches.size(); ++i)
          batches.push_back(observe.forward_batches[i]);
        report["workload_results"].push_back({{"workload", workload.at("id")}, {"repeat", repeat},
            {"elapsed_seconds", Seconds(started)}, {"requests", requests}, {"scheduled_forward_tokens", batches}});
        report["workload_results"].back()["request_errors"] = errors;
        save();
        for (const auto& error : errors) Require(error.empty(), error);
        Require(!batches.empty(), "public completion did not execute native ROCm embedding");
        std::cout << "F16_PUBLIC_RESULT " << report["workload_results"].back().dump() << std::endl;
      }
    }
    report["observed_marked_gemms"] = observe.marked_gemms;
    report["observed_weights"] = Json::array();
    for (const auto& [ptr, weight] : observe.weights) report["observed_weights"].push_back(weight);
    const auto scratch = vt::rocm::GetF16ScratchStats();
    report["scratch"] = {{"capacity_bytes", scratch.capacity_bytes}, {"retained_bytes", scratch.retained_bytes},
                         {"high_water_bytes", scratch.high_water_bytes}, {"stream_count", scratch.stream_count}};
    const auto loads = vllm::load_stats::Snapshot();
    report["load_counters"] = {{"host_copy_bytes", loads.host_copy_bytes}, {"borrowed_bytes", loads.borrowed_bytes},
                               {"device_upload_bytes", loads.device_upload_bytes}};
    save();
    for (const auto& workload : fixture.at("workloads")) {
      std::vector<Json> matching;
      for (const auto& result : report.at("workload_results"))
        if (result.at("workload") == workload.at("id")) matching.push_back(result);
      if (matching.empty()) continue;
      Require(matching.size() == 2, "repeat count differs from committed fixture");
      for (size_t r = 0; r < matching[0].at("requests").size(); ++r)
        Require(matching[0].at("requests").at(r).at("output_ids") ==
                    matching[1].at("requests").at(r).at("output_ids"),
                "public model gate is not repeatable for " + workload.at("id").get<std::string>());
    }
    report["repeatable"] = true;
    Require(observe.raw_f16_gemms == 0, "model forwarded raw F16 weights without resolved values");
    report["complete"] = true;
    save();
    return 0;
  } catch (const std::exception& error) {
    report["error"] = error.what();
    save();
    std::cerr << "F16_PUBLIC_ERROR " << error.what() << '\n';
    return 1;
  }
}
