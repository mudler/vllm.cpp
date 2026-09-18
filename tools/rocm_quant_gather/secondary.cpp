// BACKEND-ROCM-QUANT-GATHER (#3093), external oracle harness.
// Stock pin 10bf611e533d81f739128304991c5e133c6aebd8 and IQ1_XXXS fork pin
// 36fe8e1cc7f2b3b8c92fdda0ab07600141921786. No decoder is transcribed here.
// GET_ROWS enters the upstream CPU graph. Q8_K calls its upstream decoder
// directly because the stock GET_ROWS dispatch explicitly excludes that type.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "llama.h"

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<char> Read(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error(std::string("cannot read ") + path);
  const auto size = file.tellg();
  if (size < 0) throw std::runtime_error("cannot determine file size");
  std::vector<char> bytes(static_cast<size_t>(size));
  file.seekg(0);
  file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("incomplete input read");
  return bytes;
}

void Write(const char* path, const void* data, size_t bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!file) throw std::runtime_error(std::string("cannot write ") + path);
}

void Gather(int argc, char** argv) {
  if (argc != 8) throw std::runtime_error("gather TYPE ROWS WIDTH PACKED IDS_I32 OUT_F32");
  const auto type = static_cast<ggml_type>(std::stoi(argv[2]));
  const int64_t rows = std::stoll(argv[3]);
  const int64_t width = std::stoll(argv[4]);
  const auto packed = Read(argv[5]);
  const auto ids_bytes = Read(argv[6]);
  if (rows <= 0 || width <= 0 || width % ggml_blck_size(type) != 0 || ids_bytes.size() % 4 != 0)
    throw std::runtime_error("invalid gather geometry");
  const size_t row_bytes = ggml_row_size(type, width);
  if (packed.size() != static_cast<size_t>(rows) * row_bytes) throw std::runtime_error("packed byte count mismatch");
  std::vector<int32_t> ids(ids_bytes.size() / 4);
  std::memcpy(ids.data(), ids_bytes.data(), ids_bytes.size());
  for (auto id : ids) if (id < 0 || id >= rows) throw std::runtime_error("oracle input ID out of range");
  std::vector<float> output(ids.size() * static_cast<size_t>(width));
  if (type == GGML_TYPE_Q8_K) {
    for (size_t token = 0; token < ids.size(); ++token) {
      dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(
          packed.data() + static_cast<size_t>(ids[token]) * row_bytes),
          output.data() + token * static_cast<size_t>(width), width);
    }
    std::puts("ORACLE operation=dequantize_row_q8_K adaptation=stock_GET_ROWS_excludes_Q8_K");
  } else {
    ggml_init_params params{};
    params.mem_size = packed.size() + output.size() * sizeof(float) + ids_bytes.size() + 8 * 1024 * 1024;
    params.no_alloc = false;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), &ggml_free);
    if (!context) throw std::runtime_error("ggml_init failed");
    ggml_tensor* table = ggml_new_tensor_2d(context.get(), type, width, rows);
    ggml_tensor* indices = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, static_cast<int64_t>(ids.size()));
    std::memcpy(table->data, packed.data(), packed.size());
    std::memcpy(indices->data, ids.data(), ids_bytes.size());
    ggml_tensor* selected = ggml_get_rows(context.get(), table, indices);
    ggml_cgraph* graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, selected);
    if (ggml_graph_compute_with_ctx(context.get(), graph, 4) != GGML_STATUS_SUCCESS)
      throw std::runtime_error("upstream GET_ROWS graph failed");
    if (selected->type != GGML_TYPE_F32 || ggml_nbytes(selected) != output.size() * sizeof(float))
      throw std::runtime_error("unexpected GET_ROWS output format");
    std::memcpy(output.data(), selected->data, output.size() * sizeof(float));
    std::puts("ORACLE operation=ggml_get_rows graph=ggml_graph_compute_with_ctx threads=4 output=f32");
  }
  Write(argv[7], output.data(), output.size() * sizeof(float));
  std::printf("ORACLE type=%s rows=%lld width=%lld selected=%zu table_bytes=%zu output_bytes=%zu\n",
              ggml_type_name(type), static_cast<long long>(rows), static_cast<long long>(width),
              ids.size(), packed.size(), output.size() * sizeof(float));
}

void Model(int argc, char** argv, bool capture) {
  const int first_token = capture ? 4 : 3;
  if (argc <= first_token) throw std::runtime_error("model[-capture] MODEL_GGUF [OUTPUT_PREFIX] TOKEN_ID...");
  std::vector<llama_token> prompt;
  for (int i = first_token; i < argc; ++i) prompt.push_back(std::stoi(argv[i]));
  llama_backend_init();
  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
      llama_model_load_from_file(argv[2], model_params), &llama_model_free);
  if (!model) throw std::runtime_error("pinned oracle refused bounded GGUF model");
  auto context_params = llama_context_default_params();
  context_params.n_ctx = 64;
  context_params.n_batch = 64;
  context_params.n_ubatch = 64;
  context_params.n_seq_max = 1;
  context_params.n_threads = 4;
  context_params.n_threads_batch = 4;
  context_params.type_k = GGML_TYPE_BF16;
  context_params.type_v = GGML_TYPE_BF16;
  std::unique_ptr<llama_context, decltype(&llama_free)> context(
      llama_init_from_model(model.get(), context_params), &llama_free);
  if (!context) throw std::runtime_error("pinned oracle refused bf16 KV context");
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), &llama_sampler_free);
  auto batch = llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()));
  std::vector<llama_token> generated;
  std::vector<float> logits;
  const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
  if (vocab != 128) throw std::runtime_error("bounded model vocabulary changed");
  generated.reserve(4);
  logits.reserve(4 * static_cast<size_t>(vocab));
  for (int step = 0; step < 4; ++step) {
    if (llama_decode(context.get(), batch) != 0) throw std::runtime_error("pinned oracle decode failed");
    const float* values = llama_get_logits_ith(context.get(), -1);
    if (!values) throw std::runtime_error("oracle returned no logits");
    for (int32_t i = 0; i < vocab; ++i) {
      if (!std::isfinite(values[i])) throw std::runtime_error("oracle returned nonfinite logits");
      logits.push_back(values[i]);
    }
    const llama_token token = llama_sampler_sample(sampler.get(), context.get(), -1);
    generated.push_back(token);
    batch = llama_batch_get_one(&generated.back(), 1);
  }
  std::printf("ORACLE model=%s requested_context=64 physical_context=%u kv_k=bf16 kv_v=bf16 concurrency=1 greedy=true ignore_eos=true tokens=",
              argv[2], llama_n_ctx(context.get()));
  for (auto token : generated) std::printf(" %d", token);
  std::printf("\n");
  std::printf("ORACLE system=%s\n", llama_print_system_info());
  if (capture) {
    const std::string prefix(argv[3]);
    Write((prefix + "-logits-f32.bin").c_str(), logits.data(), logits.size() * sizeof(float));
    std::string ids;
    for (auto token : generated) ids += std::to_string(token) + "\n";
    Write((prefix + "-tokens.txt").c_str(), ids.data(), ids.size());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "gather") Gather(argc, argv);
    else if (argc > 1 && std::string(argv[1]) == "model") Model(argc, argv, false);
    else if (argc > 1 && std::string(argv[1]) == "model-capture") Model(argc, argv, true);
    else throw std::runtime_error("choose gather, model, or model-capture");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ORACLE FAILED: %s\n", error.what());
    return 1;
  }
}
