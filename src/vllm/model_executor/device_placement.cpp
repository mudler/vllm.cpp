// See include/vllm/model_executor/device_placement.h for what this is, what it is
// deliberately NOT (a sharding concept), and the two upstream semantics it
// transcribes. Row `ENG-HYBRID-PLACEMENT`, issue #2023.
#include "vllm/model_executor/device_placement.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>
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

int64_t GgufBlockIndexFromTensorName(const std::string& name) {
  // `blk.<N>.` is llama.cpp's GGUF spelling for a decoder block, and it is the
  // ONLY key a loader has: `GgufTensorInfo` carries a name, never a layer.
  // `RoutedExpertTensorNames` composes the same prefix and
  // `LlmFfnExpsBlockRegex` matches it, so this is the inverse of two functions
  // that already exist rather than a new naming convention.
  //
  // ANCHORED AT THE START, deliberately. `regex_search`'s unanchored semantics
  // are right for an operator's `-ot` pattern and wrong here: a name that merely
  // CONTAINS `blk.3.` is not a block-3 tensor, and answering as though it were
  // would move a weight no plan claimed.
  static constexpr char kPrefix[] = "blk.";
  const size_t plen = sizeof(kPrefix) - 1;
  if (name.size() <= plen || name.compare(0, plen, kPrefix) != 0) return -1;
  size_t i = plen;
  int64_t idx = 0;
  bool any = false;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
    // A count that cannot be a layer index is not a layer index. Refusing
    // instead of wrapping keeps a malformed name out of `per_layer_`'s range
    // rather than aliasing it onto a real layer.
    if (idx > (INT64_MAX - 9) / 10) return -1;
    idx = idx * 10 + (name[i] - '0');
    any = true;
    ++i;
  }
  if (!any || i >= name.size() || name[i] != '.') return -1;
  return idx;
}

vt::DeviceType MoePlacementPlan::DeviceForRoutedExpertTensor(
    const std::string& name) const {
  const int64_t layer = GgufBlockIndexFromTensorName(name);
  // A name carrying no block index belongs to no layer, so no placement claims
  // it. The engine's own device is the inert answer, which is `DeviceForLayer`'s
  // own out-of-range behaviour rather than a second convention.
  if (layer < 0) return engine_device_;
  return DeviceForLayer(layer);
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

void MaybeDumpMoeBlockOutput(int64_t layer_index, vt::Backend& b, vt::Queue& q,
                             const void* data, int64_t elems, bool data_is_host,
                             vt::DType dtype) {
  // Latched once. An unset variable must cost a load and a branch, not a getenv
  // per layer per token.
  static const char* const path = std::getenv("VT_PLACEMENT_DUMP_MOE");
  if (path == nullptr || path[0] == '\0') return;

  // WHICH LAYER, and it must be selectable rather than pinned to 0.
  //
  // `--fit` places TRAILING layers, so with the dump fixed at layer 0 a fit run
  // compared an UNPLACED layer against an unplaced layer and reported
  // NMSE=0.000e+00 over 12800 bitwise-identical values. That is a vacuous pass
  // wearing a perfect score: the dump region and the placement region simply did
  // not intersect. A gate can only compare a placed layer if it can ASK for one.
  //
  // Latched once, like the path: unset means layer 0, which keeps every existing
  // invocation byte-identical.
  static const int64_t want_layer = [] {
    const char* e = std::getenv("VT_PLACEMENT_DUMP_MOE_LAYER");
    if (e == nullptr || e[0] == '\0') return int64_t{0};
    const long long v = std::atoll(e);
    return v >= 0 ? static_cast<int64_t>(v) : int64_t{0};
  }();
  if (layer_index != want_layer || data == nullptr || elems <= 0) return;
  // FIRST matching call only. A decode writes this layer once per step, and the
  // gate compares one step against one step; appending every step would compare
  // arms that have already diverged in TOKENS and so no longer share an input.
  static bool done = false;
  if (done) return;
  done = true;

  // The block's dtype is NOT assumed. The seam used to hardcode bf16 and a
  // caller hands it f32; a dump that widened f32 bits as bf16 would print
  // plausible-looking garbage and the gate would compute an NMSE over it.
  if (dtype != vt::DType::kBF16 && dtype != vt::DType::kF32) {
    std::fprintf(stderr,
                 "engine: VT_PLACEMENT_DUMP_MOE cannot render dtype %s; the "
                 "gate would otherwise compare misread bytes\n",
                 vt::Name(dtype));
    return;
  }
  const size_t n = static_cast<size_t>(elems);
  const size_t esz = vt::SizeOf(dtype);
  std::vector<uint8_t> raw(n * esz);
  if (data_is_host) {
    std::memcpy(raw.data(), data, n * esz);
  } else {
    b.Copy(q, raw.data(), data, n * esz);
    b.Synchronize(q);
  }

  std::FILE* f = std::fopen(path, "w");
  if (f == nullptr) {
    std::fprintf(stderr,
                 "engine: VT_PLACEMENT_DUMP_MOE is set but '%s' cannot be "
                 "opened; the placement gate would compare nothing and read it "
                 "as agreement, so this says so instead\n",
                 path);
    return;
  }
  // bf16 -> f32 is an exact widening: the value is the high 16 bits of the
  // float. Text, because the consumer is a gate script and a binary format
  // would need a reader that could disagree with this writer.
  for (size_t i = 0; i < n; ++i) {
    float v;
    if (dtype == vt::DType::kF32) {
      std::memcpy(&v, raw.data() + i * esz, sizeof(v));
    } else {
      uint16_t h;
      std::memcpy(&h, raw.data() + i * esz, sizeof(h));
      // bf16 -> f32 is an exact widening: the value is the high 16 bits.
      const uint32_t bits = static_cast<uint32_t>(h) << 16;
      std::memcpy(&v, &bits, sizeof(v));
    }
    std::fprintf(f, "%.9g\n", static_cast<double>(v));
  }
  std::fclose(f);
  std::fprintf(stderr, "engine: wrote %zu MoE block values for layer 0 to %s\n",
               n, path);
}

const char* PlacementOriginName(PlacementOrigin origin) {
  switch (origin) {
    case PlacementOrigin::kStated: return "stated";
    case PlacementOrigin::kFit: return "fit";
    case PlacementOrigin::kNone: break;
  }
  return "none";
}

MoeFitResolution ResolveMoeFitFromSizes(
    size_t footprint_bytes, size_t budget_bytes,
    const std::vector<size_t>& moe_bytes_per_layer) {
  MoeFitResolution r;
  r.budget_bytes = budget_bytes;
  r.footprint_bytes = footprint_bytes;

  // UNKNOWN is not "nothing fits". `device_memory_total_bytes` is 0 on every
  // platform that does not probe one, and comparing against 0 would place every
  // layer on a box that was merely not measured — a wrong placement that looks
  // exactly like a working resolver.
  if (budget_bytes == 0) {
    r.reason =
        "the device memory budget is UNKNOWN (the platform reports no total), "
        "so there is nothing to fit against; placing nothing rather than "
        "placing everything against a budget of zero";
    return r;
  }
  if (footprint_bytes == 0) {
    r.reason =
        "the model's weight footprint could not be priced, so the resolver has "
        "no left-hand side; placing nothing rather than guessing";
    return r;
  }

  // A placeable layer is one that actually HAS routed experts. A dense layer
  // contributes nothing, and counting it as placed would claim a saving of zero
  // while telling the operator a layer moved.
  int64_t placeable = 0;
  for (const size_t bytes : moe_bytes_per_layer)
    if (bytes > 0) ++placeable;
  if (placeable == 0) {
    r.reason =
        "the model has no routed-expert layers, so a placement has nothing to "
        "move";
    return r;
  }

  r.resolved = true;
  if (footprint_bytes <= budget_bytes) return r;  // already fits; place nothing

  // Fill from the LAST layer backwards, mirroring `common/fit.cpp`'s
  // back-to-front order, so the layers nearest the output leave the device
  // first. Whole layers only: a boundary layer that would need splitting is
  // taken entirely, which is the coarser granularity this row's spec sanctions.
  const size_t must_free = footprint_bytes - budget_bytes;
  for (auto it = moe_bytes_per_layer.rbegin(); it != moe_bytes_per_layer.rend();
       ++it) {
    if (*it == 0) continue;  // dense: nothing to move, and not counted as placed
    r.placed_bytes += *it;
    ++r.placed_layers;
    if (r.placed_bytes >= must_free) return r;
  }

  // Every placeable layer is on the CPU and it still does not fit. Upstream
  // would also reduce the context here; this resolver does not, so it reports
  // the shortfall instead of implying success.
  r.still_exceeds = true;
  return r;
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
