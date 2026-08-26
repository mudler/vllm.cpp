// See include/vllm/model_executor/device_placement.h for what this is, what it is
// deliberately NOT (a sharding concept), and the two upstream semantics it
// transcribes. Row `ENG-HYBRID-PLACEMENT`, issue #2023.
#include "vllm/model_executor/device_placement.h"

#include <stdexcept>
#include <map>
#include <mutex>
#include <utility>

namespace vllm {

DevicePlacement DevicePlacement::FromOverrides(
    const std::vector<PlacementOverride>& overrides,
    vt::DeviceType engine_device) {
  DevicePlacement out(engine_device);
  out.compiled_.reserve(overrides.size());
  for (const PlacementOverride& o : overrides) {
    vt::DeviceType device{};
    if (!vt::DeviceTypeFromName(o.device.c_str(), &device)) {
      // W1's parser already refuses this at startup, so reaching it means a
      // caller built the list by hand. Failing loudly beats dropping an entry the
      // operator asked for, which would place fewer tensors than the install line
      // says it placed.
      throw std::invalid_argument("device placement: \"" + o.device +
                                  "\" is not a device (expected one of: " +
                                  vt::DeviceTypeNameList() + ")");
    }
    Compiled c;
    try {
      c.re = std::regex(o.pattern);
    } catch (const std::regex_error& err) {
      throw std::invalid_argument("device placement: pattern \"" + o.pattern +
                                  "\" is not a valid regex: " + err.what());
    }
    c.pattern = o.pattern;
    c.device = device;
    // TRIVIAL means nothing moves, and an override naming the engine's OWN device
    // moves nothing. That case is not a curiosity: `cpu_moe` on a CPU engine is
    // exactly it, and it is what a user gets for pasting a llama.cpp command line
    // at a CPU build. Treating it as non-trivial would take the engine off its
    // single-device path to place everything back where it already was.
    if (device != engine_device) out.trivial_ = false;
    out.compiled_.push_back(std::move(c));
  }
  return out;
}

vt::DeviceType DevicePlacement::DeviceFor(const std::string& tensor_name) const {
  // FIRST MATCH WINS and the scan stops there — `llama-model-loader.cpp:1180`.
  // `regex_search`, not `regex_match`, so a pattern hits anywhere inside the name.
  for (const Compiled& c : compiled_) {
    if (std::regex_search(tensor_name, c.re)) return c.device;
  }
  return engine_device_;
}

std::string DevicePlacement::Describe() const {
  // Silent when trivial. A line about a placement that changes nothing is noise,
  // and an operator who learns to skip this line will skip the one that matters.
  if (trivial_) return "";
  std::string out = "engine device ";
  out += vt::DeviceTypeName(engine_device_);
  out += ", ";
  out += std::to_string(compiled_.size());
  out += compiled_.size() == 1 ? " override" : " overrides";

  // Name the devices actually placed to, in the order they first appear. The
  // COUNT alone cannot distinguish "40 layers to the CPU" from "40 layers back to
  // the device I am already on", and those differ by the whole point of the row.
  std::string devices;
  for (const Compiled& c : compiled_) {
    if (c.device == engine_device_) continue;
    const std::string name = vt::DeviceTypeName(c.device);
    if (devices.find(name) != std::string::npos) continue;
    if (!devices.empty()) devices += " ";
    devices += name;
  }
  if (!devices.empty()) {
    out += " placing to ";
    out += devices;
  }
  return out;
}

std::vector<std::string> RoutedExpertTensorNames(int64_t layer) {
  // llama.cpp's GGUF spelling, which is what `LLM_FFN_EXPS_REGEX` is written
  // against (`common/common.h:1113` @ `b10451`): the alternation is
  // `up|down|gate|gate_up`, so these three cover every routed-expert tensor a
  // `-cmoe` pattern can name. `gate_up` is the FUSED spelling and appears
  // INSTEAD of `gate` plus `up` in an export that merges them, so a name list
  // that omitted it would miss a merged checkpoint entirely.
  const std::string blk = "blk." + std::to_string(layer) + ".";
  return {blk + "ffn_gate_exps.weight", blk + "ffn_up_exps.weight",
          blk + "ffn_down_exps.weight"};
}

MoePlacementPlan MoePlacementPlan::Resolve(const DevicePlacement& placement,
                                           int64_t num_hidden_layers) {
  MoePlacementPlan plan;
  plan.engine_device_ = placement.engine_device();
  if (num_hidden_layers <= 0) return plan;
  plan.per_layer_.reserve(static_cast<size_t>(num_hidden_layers));

  for (int64_t l = 0; l < num_hidden_layers; ++l) {
    const std::vector<std::string> names = RoutedExpertTensorNames(l);
    const vt::DeviceType first = placement.DeviceFor(names.front());
    for (size_t i = 1; i < names.size(); ++i) {
      const vt::DeviceType other = placement.DeviceFor(names[i]);
      if (other == first) continue;
      // A PARTIAL placement. Legal to write with `-ot`, and not implemented
      // here: the MoE block runs one grouped GEMM over gate, up and down, so
      // splitting them across devices is a different kernel rather than a
      // scheduling decision. Refuse by name — picking one of the three would
      // move weights the operator never asked to move, silently.
      throw std::invalid_argument(
          "device placement: layer " + std::to_string(l) +
          " splits its routed experts across devices (\"" + names.front() +
          "\" -> " + vt::DeviceTypeName(first) + ", \"" + names[i] + "\" -> " +
          vt::DeviceTypeName(other) +
          "); the MoE block runs one grouped GEMM over gate, up and down, so "
          "they must share a device. Widen the pattern to cover all three");
    }
    plan.per_layer_.push_back(first);
    if (first != plan.engine_device_) ++plan.placed_;
  }
  return plan;
}

vt::DeviceType MoePlacementPlan::DeviceForLayer(int64_t l) const {
  if (l < 0 || static_cast<size_t>(l) >= per_layer_.size()) {
    // The inert answer rather than an exception: this is read on the decode
    // path, and a caller asking about a layer the model does not have is a bug
    // that should surface as unchanged behaviour, not as a throw mid-token.
    return engine_device_;
  }
  return per_layer_[static_cast<size_t>(l)];
}

std::string MoePlacementPlan::Describe() const {
  if (placed_ == 0) return "";
  std::string out = std::to_string(placed_);
  out += placed_ == 1 ? " layer runs its routed experts on "
                      : " layers run their routed experts on ";
  // Name the destinations, once each, in first-appearance order.
  std::string devices;
  for (const vt::DeviceType d : per_layer_) {
    if (d == engine_device_) continue;
    const std::string name = vt::DeviceTypeName(d);
    if (devices.find(name) != std::string::npos) continue;
    if (!devices.empty()) devices += " ";
    devices += name;
  }
  out += devices;
  out += ", the rest on ";
  out += vt::DeviceTypeName(engine_device_);
  return out;
}

namespace {

struct PlacementGlobals {
  std::mutex mu;
  std::map<vt::DeviceType, vt::Queue> queues;
  MoePlacementPlan plan;
};

PlacementGlobals& Globals() {
  static PlacementGlobals g;
  return g;
}

}  // namespace

vt::Queue& PlacementQueue(vt::DeviceType device) {
  PlacementGlobals& g = Globals();
  std::lock_guard<std::mutex> lk(g.mu);
  auto it = g.queues.find(device);
  if (it != g.queues.end()) return it->second;

  // The POLICY refusal comes BEFORE `GetBackend`, and the order is load-bearing
  // rather than tidy: `GetBackend` throws "no backend registered" for a device
  // this build does not carry, so asking it first would report a build gap for
  // what is actually a rule. A CPU-only build must still say why a CUDA
  // destination is refused.
  //
  // Refuse a target whose queues must be released rather than leak a stream for
  // the process's life. CPU is the only placement target this row ships, and its
  // `DestroyQueue` is a no-op, so this holds today and fails loudly the moment
  // somebody points a placement at an accelerator without giving the queue an
  // owner.
  if (device != vt::DeviceType::kCPU) {
    throw std::invalid_argument(
        std::string("device placement: cannot host a placed group on \"") +
        vt::DeviceTypeName(device) +
        "\": only \"cpu\" is supported as a placement TARGET today, because a "
        "process-lifetime queue on an accelerator would leak its stream. The "
        "engine may run on any device; it is the destination that is limited");
  }
  vt::Backend& b = vt::GetBackend(device);
  auto [pos, inserted] = g.queues.emplace(device, b.CreateQueue());
  (void)inserted;
  return pos->second;
}

void SetActiveMoePlacementPlan(const MoePlacementPlan& plan) {
  PlacementGlobals& g = Globals();
  std::lock_guard<std::mutex> lk(g.mu);
  g.plan = plan;
}

const MoePlacementPlan& ActiveMoePlacementPlan() {
  // No lock on the READ. It is taken once per MoE layer per token, and the plan
  // is written once at model build before any forward exists. A mutex on the
  // decode path to guard a value that never changes after load would serialise
  // the lane this row exists to widen — the same reasoning
  // `weight_residency.cpp` records for its own hot read.
  return Globals().plan;
}

void ResetActiveMoePlacementPlanForTesting() {
  PlacementGlobals& g = Globals();
  std::lock_guard<std::mutex> lk(g.mu);
  g.plan = MoePlacementPlan{};
}

}  // namespace vllm
