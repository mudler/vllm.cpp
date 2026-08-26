// See include/vllm/model_executor/device_placement.h for what this is, what it is
// deliberately NOT (a sharding concept), and the two upstream semantics it
// transcribes. Row `ENG-HYBRID-PLACEMENT`, issue #2023.
#include "vllm/model_executor/device_placement.h"

#include <stdexcept>
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

}  // namespace vllm
