#pragma once

#include <memory>
#include <optional>

#include "vllm/model_executor/models/gemma3.h"

namespace vllm {
// The graph owns stable step buffers and pins every captured scratch address.
class Gemma3DecodeGraph {
 public:
  Gemma3DecodeGraph(const Gemma3Weights&, const HfConfig&, vt::Queue);
  ~Gemma3DecodeGraph();
  Gemma3DecodeGraph(const Gemma3DecodeGraph&) = delete;
  Gemma3DecodeGraph& operator=(const Gemma3DecodeGraph&) = delete;
  ForwardLogits Step(const ModelForwardInput&);
  bool UsesQueue(const vt::Queue&) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
std::optional<ForwardLogits> Gemma3DecodeGraphForward(std::unique_ptr<Gemma3DecodeGraph>& graph,
                                                      const Gemma3Weights& weights,
                                                      const ModelForwardInput& input);
}  // namespace vllm
