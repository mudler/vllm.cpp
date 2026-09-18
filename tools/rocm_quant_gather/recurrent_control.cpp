// Controlled recurrent-state consumption for both pinned secondary libraries.
// Stock 10bf611e and fork 36fe8e1c, llama-graph.cpp::build_rs consume
// state_copy_main with GET_ROWS. Keep the graph and its allocations across
// changing copy indices and state values, as the real recurrent graph does.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("OUTPUT_F32");
    ggml_init_params params{4 * 1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) throw std::runtime_error("ggml_init failed");
    auto* state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 3);
    auto* copy = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 3);
    auto* delta = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 3);
    ggml_set_name(copy, "s_copy");
    ggml_set_input(state);
    ggml_set_input(copy);
    ggml_set_input(delta);
    auto* next = ggml_add(ctx.get(), ggml_get_rows(ctx.get(), state, copy), delta);
    ggml_set_output(next);
    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, next);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) throw std::runtime_error("CPU backend initialization failed");
    ggml_backend_t backends[] = {backend.get()};
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> sched(
        ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true),
        ggml_backend_sched_free);
    if (!sched || !ggml_backend_sched_alloc_graph(sched.get(), graph))
      throw std::runtime_error("recurrent graph allocation failed");
    if (!copy->buffer || !copy->data) throw std::runtime_error("consumed recurrent copy has no storage");
    const void* copy_storage = copy->data;
    std::array<float, 12> values{0,1,2,3, 10,11,12,13, 20,21,22,23};
    const std::array<float, 12> increment{1,2,3,4, 1,2,3,4, 1,2,3,4};
    const std::array<std::array<int32_t, 3>, 3> indices{{{2,0,1}, {1,2,0}, {2,2,0}}};
    std::ofstream output(argv[1], std::ios::binary);
    for (size_t step = 0; step < indices.size(); ++step) {
      std::array<float, 12> expected{};
      for (size_t row = 0; row < 3; ++row)
        for (size_t col = 0; col < 4; ++col)
          expected[row * 4 + col] = values[static_cast<size_t>(indices[step][row]) * 4 + col] + increment[row * 4 + col];
      ggml_backend_tensor_set(state, values.data(), 0, sizeof(values));
      ggml_backend_tensor_set(copy, indices[step].data(), 0, sizeof(indices[step]));
      ggml_backend_tensor_set(delta, increment.data(), 0, sizeof(increment));
      if (ggml_backend_sched_graph_compute(sched.get(), graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("reused recurrent graph failed");
      ggml_backend_tensor_get(next, values.data(), 0, sizeof(values));
      if (values != expected || copy->data != copy_storage)
        throw std::runtime_error("recurrent state or reused copy allocation changed");
      output.write(reinterpret_cast<const char*>(values.data()), sizeof(values));
      std::printf("RECURRENT step=%zu consumed_s_copy=true reused_graph=true values=", step);
      for (float value : values) std::printf(" %.0f", value);
      std::puts("");
    }
    if (!output) throw std::runtime_error("cannot write recurrent state evidence");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "RECURRENT FAILED: %s\n", error.what());
    return 1;
  }
}
