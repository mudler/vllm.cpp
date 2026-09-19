// GLiNER2.5 zero-shot NER — a THIN CLIENT of the public C ABI
// (include/vllm.h), per the ONE SURFACE directive (MODEL-GLINER25).
//
// Loads a GLiNER2.5 checkpoint, runs zero-shot NER on the given text with the
// given entity labels, and prints the extracted entities. This file owns only
// argv parsing and printing — the entire NER pipeline lives in the library
// behind vllm_engine_load + vllm_gliner_ner (ABI v27).
//
//   gliner-cli <hf-gliner-dir> <text> <label1> [label2 ...] [--threshold T] [--max-width W]
//
// `<hf-gliner-dir>` is any HF-format GLiNER2.5 checkpoint (config.json +
// model.safetensors, architecture "BoundaryExtractor"). The text is the
// string to extract entities from. The labels are the entity types to find
// (e.g. "person" "organization" "location"). Optional flags: --threshold
// (default 0.5), --max-width (default 12).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vllm.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <hf-gliner-dir> <text> <label1> [label2 ...] "
                 "[--threshold T] [--max-width W]\n",
                 argv[0]);
    return 2;
  }

  const std::string ckpt = argv[1];
  const std::string text = argv[2];

  std::vector<std::string> label_storage;
  std::vector<const char*> labels;
  float threshold = 0.5F;
  int32_t max_width = 12;

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--threshold" && i + 1 < argc) {
      threshold = static_cast<float>(std::atof(argv[++i]));
    } else if (arg == "--max-width" && i + 1 < argc) {
      max_width = std::atoi(argv[++i]);
    } else {
      label_storage.push_back(arg);
    }
  }
  if (label_storage.empty()) {
    std::fprintf(stderr, "at least one entity label is required\n");
    return 2;
  }
  for (const std::string& l : label_storage) {
    labels.push_back(l.c_str());
  }

  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = ckpt.c_str();
  vllm_engine* engine = nullptr;
  if (vllm_engine_load(&mp, &engine) != VLLM_OK) {
    std::fprintf(stderr, "load failed: %s\n", vllm_last_error());
    return 1;
  }

  vllm_ner_result result;
  const auto t0 = std::chrono::steady_clock::now();
  const vllm_status st = vllm_gliner_ner(
      engine, text.c_str(), labels.data(),
      static_cast<int32_t>(labels.size()), threshold, max_width, &result);
  const auto t1 = std::chrono::steady_clock::now();

  if (st != VLLM_OK) {
    std::fprintf(stderr, "ner failed: %s\n", vllm_last_error());
    vllm_engine_free(engine);
    return 1;
  }

  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("{\"model\": \"%s\", \"text\": \"%s\", \"entities\": [",
              ckpt.c_str(), text.c_str());
  for (int32_t i = 0; i < result.n_entities; ++i) {
    const vllm_ner_entity& e = result.entities[i];
    std::printf("%s{\"label\": \"%s\", \"text\": \"%s\", \"start\": %d, "
                "\"end\": %d, \"confidence\": %.4f}",
                i ? ", " : "", e.label, e.text, e.char_start, e.char_end,
                e.confidence);
  }
  std::printf("], \"count\": %d, \"ms\": %.1f}\n", result.n_entities, ms);

  vllm_ner_result_free(&result);
  vllm_engine_free(engine);
  return 0;
}
