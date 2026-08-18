// vllm.cpp ORIGINAL — see include/vllm/config/weight_residency.h for the schema,
// the precedence rule, why this is not part of the mirrored offload config, and
// why the latch throws. Row `ENG-RESIDENCY-CONFIG`, issue #1110.
#include "vllm/config/weight_residency.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {
namespace {

// The tree's existing environment polarity, transcribed from
// `EnvOn` (src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:61-66) rather
// than re-invented: an environment-only run must resolve byte-for-byte the way it
// did before this file existed.
bool EnvTruth(const char* value) {
  return !(std::strcmp(value, "") == 0 || std::strcmp(value, "0") == 0 ||
           std::strcmp(value, "false") == 0 || std::strcmp(value, "off") == 0);
}

// The COUNT variable's value, if and only if it would beat a configured field.
// atoll, and anything non-positive ignored, is what the existing readers do
// (qwen3_5.cpp's store constructor), kept verbatim so an environment-only run
// resolves exactly as before — which means an empty, garbage or non-positive value
// is NOT an override at all: it falls through to the config.
//
// ONE rule with two callers: the resolver below, and `DescribeEnvOverrides`, which
// used to announce mere presence and therefore reported `SLOTS=banana` as an
// override the resolver then ignored (#1122 L7).
std::optional<int64_t> EnvCountThatWins(const char* env_name) {
  const char* v = std::getenv(env_name);
  if (v == nullptr || *v == '\0') return std::nullopt;
  const long long parsed = std::atoll(v);
  if (parsed <= 0) return std::nullopt;
  return static_cast<int64_t>(parsed);
}

// `VT_DEVICE_WEIGHT_BUDGET_BYTES`'s value, if and only if it would beat a
// configured budget. The grammar is transcribed verbatim from the reader this row
// took over (`DeviceWeightBudgetBytes`, gguf_device_fit.cpp @ #1123): one or more
// DECIMAL DIGITS and nothing else, no sign and no space. Anything else is
// IGNORED, so an environment-only run resolves exactly as before.
//
// It is NOT `EnvCountThatWins`. That helper drops a non-positive value, and `0` is
// this knob's suppression spelling — the operator asking for the refusal to be
// switched off — so dropping it would silently restore the guard the operator
// turned off. `strtoull` alone is not enough for the grammar either: it skips
// leading whitespace, and it ACCEPTS a leading '-' and wraps it to ULLONG_MAX, so
// "-1" would parse as an effectively infinite budget.
//
// ONE rule with two callers, for the same reason the count rule has two: the
// resolver below, and `DescribeEnvOverrides`, which must not announce a value the
// resolver ignores (#1122 L7).
std::optional<size_t> DeviceWeightBudgetEnvThatWins() {
  const char* v = std::getenv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  if (v == nullptr || v[0] < '0' || v[0] > '9') return std::nullopt;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed =  // NOLINT(runtime/int) strtoull's type
      std::strtoull(v, &end, 10);
  if (*end != '\0' || errno != 0) return std::nullopt;
  return static_cast<size_t>(parsed);
}

struct Global {
  std::mutex mu;
  WeightResidencyConfig config;

  // TWO DECISIONS, NOT ONE, and mmap and prefault are in NEITHER. A single flag set
  // by every resolver refused a legal second load: `GgufLoadPolicy::FromEnv()` runs
  // per load and this change removed the prefault site's static, so those two knobs
  // freeze nothing, yet the coarse flag let an ordinary first load block a second
  // engine's whole document (#1122 M1, measured through `vllm_engine_load`). What
  // genuinely freezes is the streaming ANSWER — a function-local static in
  // `ResolveExpertStreamRequested` — and the slot store's GEOMETRY, because the
  // store is built once per process and its reservation cannot be resized.
  //
  // ATOMIC, not a bool under `mu`, because `ResolveExpertStreamRequested` marks the
  // first one on EVERY call and that function sits on the per-expert-slice decode
  // path (`KqExpertSlice` -> `Qwen35ExpertStream::Get`). A process-wide mutex there
  // would serialise the lane this row exists to make configurable, and a row whose
  // whole claim is "changes no kernel, no allocation and no perf axis" must not
  // quietly add a lock to a hot loop.
  //
  // RELAXED is enough, and the reason is NOT a synchronises-with edge. A relaxed
  // store made outside `mu` is not ordered by a later acquire of `mu`, so the
  // earlier claim to that effect was wrong (#1122 L2). Three facts carry it instead.
  //
  // NOTHING IS PUBLISHED THROUGH a flag, even though each decision now carries its
  // answer. The streaming answer is a value IN the tri-state atomic itself, read by
  // the same single relaxed load that reads the fact; the geometry's two numbers live
  // in `built_geometry`, written and read under `mu` by the same function that sets
  // the geometry flag. So there is no second location a reader could see out of order
  // with the flag.
  //
  // Each store is IDEMPOTENT. `ResolveExpertStreamRequested` writes the value of a
  // function-local static, and thread-safe static initialisation — which is a real
  // synchronising edge — makes that value identical on every call, so a concurrent
  // reader sees either "nothing decided" or the one final answer.
  //
  // And each is MONOTONIC: it moves off its initial value once and never back except
  // by the test reset, so no ordering between the two can be observed.
  //
  // WHAT IS NOT GUARANTEED, stated because it is a real window rather than a
  // theoretical one now that a second engine may install while a first decodes: an
  // install running concurrently with the very first expert slice of another engine
  // may not observe that latch and may accept the config. Release/acquire would not
  // close it either — no ordering makes a concurrent store visible — only holding
  // `mu` across the decode path would, and that is the lock this row must not add.
  // The consequence is the same as arriving a microsecond earlier, which is legal:
  // whoever got there first decides. Every production path installs before the model
  // exists, so the window needs a caller that loads a second engine at the instant
  // the first one starts streaming.
  //
  // THE STREAMING FLAG IS TRI-STATE, and that is what lets the refusal compare an
  // incoming document against THE DECISION rather than against the stored document.
  // 0 = nothing decided, 1 = decided OFF, 2 = decided ON. One atomic carries both
  // the fact and the answer, so a reader takes ONE relaxed load and there is no
  // ordering question between a "latched" flag and a separate "value" — which there
  // would be if the answer lived in a second atomic. It still transitions exactly
  // once (0 -> 1 or 0 -> 2, never back except by the test reset), so the monotonicity
  // argument above is unchanged.
  std::atomic<int> expert_stream_decision{0};
  std::atomic<bool> latched_geometry{false};

  // The geometry the store was built with, under `mu`. It is the geometry ANSWER,
  // the counterpart of the tri-state above, and it lives here rather than in a
  // file-static so that `FrozenFields` — which already holds `mu` — can read it.
  ExpertStreamGeometry built_geometry;

  bool Latched(ResidencyLatch knob) const {
    switch (knob) {
      case ResidencyLatch::kExpertStream:
        return expert_stream_decision.load(std::memory_order_relaxed) != 0;
      case ResidencyLatch::kExpertStreamGeometry:
        return latched_geometry.load(std::memory_order_relaxed);
    }
    return false;
  }

  // The streaming answer this process actually cached. Only meaningful once
  // `Latched(kExpertStream)`; it reads false before that, which no caller relies on
  // because every caller checks the latch first.
  bool DecidedExpertStream() const {
    return expert_stream_decision.load(std::memory_order_relaxed) == 2;
  }

  bool AnyLatched() const {
    return Latched(ResidencyLatch::kExpertStream) ||
           Latched(ResidencyLatch::kExpertStreamGeometry);
  }
};

Global& State() {
  static Global g;
  return g;
}

// The key's path in the DOCUMENT, which is the only path an operator can act on.
// `parent_path` is empty at the top level; a hardcoded prefix here is what made
// `{"vllm_cpp": 5}` report `"vllm_cpp.vllm_cpp" must be a JSON object`.
std::string Dotted(const char* parent_path, const std::string& key) {
  if (parent_path == nullptr || *parent_path == '\0') return key;
  return std::string(parent_path) + "." + key;
}

const nlohmann::json* ExtObject(const nlohmann::json& parent, const char* key,
                                const char* parent_path) {
  auto it = parent.find(key);
  if (it == parent.end() || it->is_null()) return nullptr;
  if (!it->is_object()) {
    throw std::invalid_argument(std::string("offload config: \"") +
                                Dotted(parent_path, key) +
                                "\" must be a JSON object");
  }
  return &*it;
}

// Refuse a key nobody reads, at EVERY level of the document including the top.
// `parse_offload_config_json` ignores what it does not know, which is what lets
// this extension live in the same document — and what made `{"vllm-cpp":{...}}`
// parse to an empty config and start a server running this tier at its DEFAULTS —
// prefault ON and streaming OFF, the two the 370 GiB case exists to change — so the
// typo is discovered as an out-of-memory kill rather than as an error. The hyphen is the
// likeliest spelling of all, because every flag around it is hyphenated.
//
// Refusing is the MIRROR-FAITHFUL polarity rather than a local invention: upstream
// has no `--offload-config` flag at all at the pin (no such string anywhere in the
// tree), and vLLM builds its config dataclasses with the `@config` decorator, whose
// body sets `ConfigDict(extra="forbid")` under the comment "Extra fields are
// forbidden by default" (vllm/config/utils.py:68-69 @ 555967922). `OffloadConfig`
// carries that decorator (offload.py:80) and so does `KVTransferConfig`
// (kv_transfer.py:22-23), so `--kv-transfer-config`, the JSON-document flag next
// door, refuses an unknown key. The mirrored `parse_offload_config_json` stays untouched;
// this parser reads the same string at both production entry points, so enumerating
// here closes the document without editing the transcription.
void RejectUnknownKeys(const nlohmann::json& obj, const char* path,
                       const std::vector<std::string>& known) {
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    bool ok = false;
    for (const std::string& k : known) {
      if (it.key() == k) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      std::string msg = std::string("offload config: unknown key \"") +
                        Dotted(path, it.key()) + "\" (expected one of:";
      for (const std::string& k : known) msg += " " + k;
      msg += ")";
      throw std::invalid_argument(msg);
    }
  }
}

std::optional<bool> ExtBool(const nlohmann::json& obj, const char* key,
                            const char* path) {
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return std::nullopt;
  if (!it->is_boolean()) {
    throw std::invalid_argument(std::string("offload config: \"") + path + "." +
                                key + "\" must be a boolean");
  }
  return it->get<bool>();
}

std::optional<int64_t> ExtPositiveInt(const nlohmann::json& obj, const char* key,
                                      const char* path) {
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return std::nullopt;
  if (!it->is_number_integer()) {
    throw std::invalid_argument(std::string("offload config: \"") + path + "." +
                                key + "\" must be an integer");
  }
  const int64_t v = it->get<int64_t>();
  if (v <= 0) {
    // The environment readers TOLERATE a zero or negative value and fall back to
    // the default, because they parse with atol and have no way to report. A
    // config document is parsed at startup where a message still reaches the
    // operator, so it is refused instead: a zero slot count that silently became
    // 64 is a cache the operator does not have.
    throw std::invalid_argument(std::string("offload config: \"") + path + "." +
                                key + "\" must be positive (got " +
                                std::to_string(v) + ")");
  }
  return v;
}

// The SAME type check, with ZERO accepted. It exists for exactly one field, and
// the difference from `ExtPositiveInt` is the field's meaning rather than a
// looser rule. `slots` and `slot_bytes` are sizes: a zero there is a degenerate
// cache, so the parser refuses it where the operator can still read the message.
// `device_fit.weight_budget_bytes` is a BUDGET that `CheckDeviceWeightFit` reads
// as UNKNOWN when it is zero, so `0` means "make no fit decision" — the same
// suppression `VT_DEVICE_WEIGHT_BUDGET_BYTES=0` has meant since #1123. Refusing
// it would delete the escape hatch the key exists to give. A NEGATIVE budget is
// still refused, and the message says "must not be negative" rather than "must be
// positive", because the second sentence would be false about a value this parser
// accepts.
std::optional<int64_t> ExtNonNegativeInt(const nlohmann::json& obj,
                                         const char* key, const char* path) {
  auto it = obj.find(key);
  if (it == obj.end() || it->is_null()) return std::nullopt;
  if (!it->is_number_integer()) {
    throw std::invalid_argument(std::string("offload config: \"") + path + "." +
                                key + "\" must be an integer");
  }
  const int64_t v = it->get<int64_t>();
  if (v < 0) {
    throw std::invalid_argument(std::string("offload config: \"") + path + "." +
                                key + "\" must not be negative (got " +
                                std::to_string(v) +
                                "); 0 means \"suppress the device-fit refusal\"");
  }
  return v;
}

}  // namespace

bool WeightResidencyConfig::empty() const {
  return !mmap.has_value() && !prefault.has_value() &&
         !expert_stream.has_value() && !expert_stream_slots.has_value() &&
         !expert_stream_slot_bytes.has_value() &&
         !device_weight_budget_bytes.has_value();
}

bool WeightResidencyConfig::operator==(
    const WeightResidencyConfig& other) const {
  return mmap == other.mmap && prefault == other.prefault &&
         expert_stream == other.expert_stream &&
         expert_stream_slots == other.expert_stream_slots &&
         expert_stream_slot_bytes == other.expert_stream_slot_bytes &&
         device_weight_budget_bytes == other.device_weight_budget_bytes;
}

std::string WeightResidencyConfig::Describe() const {
  if (empty()) return "";
  std::string out;
  const auto add_bool = [&out](const char* name, std::optional<bool> v) {
    if (!v.has_value()) return;
    if (!out.empty()) out += " ";
    out += name;
    out += *v ? "=on" : "=off";
  };
  const auto add_int = [&out](const char* name, std::optional<int64_t> v) {
    if (!v.has_value()) return;
    if (!out.empty()) out += " ";
    out += name;
    out += "=";
    out += std::to_string(*v);
  };
  add_bool("mmap", mmap);
  add_bool("prefault", prefault);
  add_bool("expert_stream", expert_stream);
  add_int("expert_stream_slots", expert_stream_slots);
  add_int("expert_stream_slot_bytes", expert_stream_slot_bytes);
  // Under the FIELD name rather than the document path, like its five siblings,
  // so the line reads as one list of knobs. A budget of 0 prints as
  // `device_weight_budget_bytes=0`, which is what the operator asked for and is
  // distinguishable from the field being absent, where nothing prints at all.
  add_int("device_weight_budget_bytes", device_weight_budget_bytes);
  return out;
}

std::string WeightResidencyConfig::DescribeEnvOverrides() const {
  // A variable is reported when it would WIN, not when it is merely set. For a
  // boolean any value wins, so presence is exact. For a COUNT the tolerant parse the
  // existing readers use means an empty, garbage or non-positive value falls THROUGH
  // to the config: `VT_MOE_EXPERT_STREAM_SLOTS=banana` overrides nothing, and
  // announcing it as an override sent the operator after a line the resolver ignores
  // (#1122 L7). `EnvCountThatWins` is the same predicate the resolver uses, so the
  // two cannot drift.
  //
  // No Resolve* call and no latch is marked, which matters for the ORDERING this
  // header is careful about: the line prints at install time, ahead of the weight
  // load that takes the decisions.
  struct Pair {
    bool set_by_operator;
    bool env_wins;
    const char* env_name;
    const char* field;
  };
  const auto bool_set = [](const char* name) {
    return std::getenv(name) != nullptr;
  };
  const auto count_wins = [](const char* name) {
    return EnvCountThatWins(name).has_value();
  };
  const Pair pairs[] = {
      {mmap.has_value(), bool_set("VT_GGUF_MMAP"), "VT_GGUF_MMAP", "mmap"},
      {prefault.has_value(), bool_set("VT_GGUF_PREFAULT"), "VT_GGUF_PREFAULT",
       "prefault"},
      {expert_stream.has_value(), bool_set("VT_MOE_EXPERT_STREAM"),
       "VT_MOE_EXPERT_STREAM", "expert_stream"},
      {expert_stream_slots.has_value(),
       count_wins("VT_MOE_EXPERT_STREAM_SLOTS"), "VT_MOE_EXPERT_STREAM_SLOTS",
       "expert_stream_slots"},
      {expert_stream_slot_bytes.has_value(),
       count_wins("VT_MOE_EXPERT_STREAM_SLOT_BYTES"),
       "VT_MOE_EXPERT_STREAM_SLOT_BYTES", "expert_stream_slot_bytes"},
      // The budget asks its OWN predicate, not `count_wins`: `0` wins here and is
      // dropped there, and `0` is the one value an operator is most surprised to
      // have inherited from an exported variable, because it switches the
      // device-fit refusal off entirely.
      {device_weight_budget_bytes.has_value(),
       DeviceWeightBudgetEnvThatWins().has_value(),
       "VT_DEVICE_WEIGHT_BUDGET_BYTES", "device_weight_budget_bytes"},
  };
  std::string out;
  for (const Pair& p : pairs) {
    if (!p.set_by_operator || !p.env_wins) continue;
    if (!out.empty()) out += ", ";
    out += p.env_name;
    out += " (";
    out += p.field;
    out += ")";
  }
  return out;
}

WeightResidencyConfig parse_weight_residency_extension_json(
    const std::string& json_text) {
  WeightResidencyConfig cfg;

  // An empty or blank document is the inert default, exactly as for the mirrored
  // parser this shares a flag with.
  const auto first = json_text.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) return cfg;

  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("offload config: malformed JSON: ") +
                                e.what());
  }
  if (!doc.is_object()) {
    throw std::invalid_argument("offload config: must be a JSON object");
  }

  // THE DOCUMENT'S OWN KEYS FIRST. These four are every key `--offload-config`
  // accepts: the three the mirrored parser reads by name
  // (src/vllm/config/offload.cpp, `offload_backend`/`uva`/`prefetch`) and this
  // extension. A fifth spelling is a typo, and a typo here is the failure the
  // refusal exists for — `{"vllm-cpp":{...}}` used to disable the whole tier in
  // silence. Keeping the mirrored names in this list is what lets one parser close
  // the document while `parse_offload_config_json` stays a byte-faithful
  // transcription; the two run on the same string at both entry points.
  RejectUnknownKeys(doc, "",
                    {"offload_backend", "uva", "prefetch", "vllm_cpp"});

  // ONE LEVEL INTO THE MIRRORED SUB-OBJECTS TOO, for the same reason and on the same
  // authority. Closing only the top level left the identical hole one level down:
  // `parse_offload_config_json` reads `uva.cpu_offload_gb` and the four `prefetch.*`
  // fields BY NAME with a fallback (src/vllm/config/offload.cpp:272-281), so
  // `{"uva":{"cpu_offload_GB":10}}` yielded a 0 GiB budget and
  // `{"prefetch":{"offload_groupsize":8}}` a group size of 0 — no offloading at all,
  // under a document the operator believes configures it, and under a public ABI
  // comment that said an unknown key anywhere is refused (#1133 H3).
  //
  // Refusing them is MIRROR-FAITHFUL, not extra strictness: `UVAOffloadConfig`
  // (vllm/config/offload.py:15-16 @ 555967922) and `PrefetchOffloadConfig` (:47-48)
  // each carry `@config`, whose body sets `ConfigDict(extra="forbid")`
  // (vllm/config/utils.py:68-69). Upstream refuses a nested typo, so the tolerance
  // was the deviation.
  //
  // The names below are the mirrored parser's own — `cpu_offload_gb` (offload.py:23),
  // `cpu_offload_params` (:34), `offload_group_size` (:54), `offload_num_in_group`
  // (:62), `offload_prefetch_step` (:66), `offload_params` (:70) — and enumerating
  // them HERE is exactly what keeps that parser a byte-faithful transcription. This
  // one checks NAMES only; the mirrored parser keeps sole ownership of their types,
  // defaults and bounds, so there is no second place a type rule could drift.
  if (const nlohmann::json* u = ExtObject(doc, "uva", "")) {
    RejectUnknownKeys(*u, "uva", {"cpu_offload_gb", "cpu_offload_params"});
  }
  if (const nlohmann::json* p = ExtObject(doc, "prefetch", "")) {
    RejectUnknownKeys(*p, "prefetch",
                      {"offload_group_size", "offload_num_in_group",
                       "offload_prefetch_step", "offload_params"});
  }

  const nlohmann::json* ext = ExtObject(doc, "vllm_cpp", "");
  if (ext == nullptr) return cfg;
  RejectUnknownKeys(*ext, "vllm_cpp", {"mmap", "expert_stream", "device_fit"});

  if (const nlohmann::json* m = ExtObject(*ext, "mmap", "vllm_cpp")) {
    RejectUnknownKeys(*m, "vllm_cpp.mmap", {"enabled", "prefault"});
    cfg.mmap = ExtBool(*m, "enabled", "vllm_cpp.mmap");
    cfg.prefault = ExtBool(*m, "prefault", "vllm_cpp.mmap");
  }
  if (const nlohmann::json* s = ExtObject(*ext, "expert_stream", "vllm_cpp")) {
    RejectUnknownKeys(*s, "vllm_cpp.expert_stream",
                      {"enabled", "slots", "slot_bytes"});
    cfg.expert_stream = ExtBool(*s, "enabled", "vllm_cpp.expert_stream");
    cfg.expert_stream_slots =
        ExtPositiveInt(*s, "slots", "vllm_cpp.expert_stream");
    cfg.expert_stream_slot_bytes =
        ExtPositiveInt(*s, "slot_bytes", "vllm_cpp.expert_stream");
  }
  // A THIRD knob family: the load-time device-fit check (#1123), whose budget
  // used to be reachable only as `VT_DEVICE_WEIGHT_BUDGET_BYTES` (#1127). It is an
  // OBJECT rather than a scalar beside `mmap` and `expert_stream` for two reasons.
  // `vllm_cpp` maps a name to an object today and `ExtObject` walks it on that
  // assumption, so one scalar sibling would make the level heterogeneous. And this
  // is a family, not a member of either existing one — the field name drops the
  // family prefix exactly as `VT_GGUF_MMAP` -> `mmap.enabled` and
  // `VT_MOE_EXPERT_STREAM_SLOTS` -> `expert_stream.slots` already do.
  if (const nlohmann::json* d = ExtObject(*ext, "device_fit", "vllm_cpp")) {
    RejectUnknownKeys(*d, "vllm_cpp.device_fit", {"weight_budget_bytes"});
    cfg.device_weight_budget_bytes =
        ExtNonNegativeInt(*d, "weight_budget_bytes", "vllm_cpp.device_fit");
  }
  return cfg;
}

namespace {

// Which fields of an incoming document a taken decision has already fixed. `mmap`
// and `prefault` are in no branch here, and that is the whole point of the
// function: they resolve per load, so a later engine may still set them.
//
// TWO PREDICATES CHANGED HERE, and they are the same root cause seen twice: absent
// means UNCHANGED (#1133 H1).
//
// A field is scored only when the DOCUMENT SETS IT. `in.expert_stream !=
// g.config.expert_stream` made `nullopt != engaged` true, so a second engine whose
// document merely OMITTED the latched field was refused — the same hard failure on a
// legal two-model load that #1122 M1 narrowed rather than removed, now reached by a
// partial document instead of by a full one.
//
// And the comparison is against THE DECISION ACTUALLY TAKEN, resolved through the
// same rule the production resolver applies, rather than against the stored document.
// That fixes two things at once. A document asking for exactly what the process
// resolved is accepted, where before it was refused by a message asserting the engine
// was not running it — the engine was. And the environment comes out right in both
// directions: `VT_MOE_EXPERT_STREAM=1` decided the answer, so a document that
// variable overrides cannot change anything and is not refused, while a document that
// WOULD change the answer still is.
std::string FrozenFields(const Global& g, const WeightResidencyConfig& in) {
  std::string out;
  const auto note = [&out](const char* field) {
    if (!out.empty()) out += ", ";
    out += field;
  };
  if (g.Latched(ResidencyLatch::kExpertStream) && in.expert_stream.has_value()) {
    // The same function the resolver calls, with the same variable, so the two
    // cannot disagree about what accepting this document would mean.
    const bool would_resolve = ExpertStreamRequestedFrom(
        std::getenv("VT_MOE_EXPERT_STREAM"), in.expert_stream);
    if (would_resolve != g.DecidedExpertStream()) note("expert_stream");
  }
  if (g.Latched(ResidencyLatch::kExpertStreamGeometry)) {
    // `EnvCountThatWins` is the same predicate `ResolveResidencyCount` uses, so a
    // count the variable overrides is not a change either.
    if (in.expert_stream_slots.has_value()) {
      const int64_t would_resolve =
          EnvCountThatWins("VT_MOE_EXPERT_STREAM_SLOTS")
              .value_or(*in.expert_stream_slots);
      if (would_resolve != g.built_geometry.slots) note("expert_stream_slots");
    }
    if (in.expert_stream_slot_bytes.has_value()) {
      const int64_t would_resolve =
          EnvCountThatWins("VT_MOE_EXPERT_STREAM_SLOT_BYTES")
              .value_or(*in.expert_stream_slot_bytes);
      if (would_resolve != g.built_geometry.slot_bytes) {
        note("expert_stream_slot_bytes");
      }
    }
  }
  return out;
}

// What the engine DECIDED, for the refusal message. It used to print the stored
// document, which is the wrong thing to show beside "the engine is not running this":
// the operator has to reconcile their document against the values in force, and after
// the merge the stored document is not those values either.
std::string DecisionSummary(const Global& g) {
  std::string out;
  if (g.Latched(ResidencyLatch::kExpertStream)) {
    out += std::string("expert_stream=") +
           (g.DecidedExpertStream() ? "on" : "off");
  }
  if (g.Latched(ResidencyLatch::kExpertStreamGeometry)) {
    if (!out.empty()) out += " ";
    out += "expert_stream_slots=" + std::to_string(g.built_geometry.slots) +
           " expert_stream_slot_bytes=" +
           std::to_string(g.built_geometry.slot_bytes);
  }
  // Unreachable from the one caller, which runs only when a decision was taken.
  if (out.empty()) out = "nothing decided yet";
  return out;
}

// THE SETTER REFUSES WHAT THE PARSER REFUSES, because there are TWO doors into
// this struct and only one of them had range rules.
// `parse_weight_residency_extension_json` refuses a non-positive `slots` or
// `slot_bytes` and a negative budget, and every production caller goes through
// it. But `SetWeightResidencyConfig` is declared in a PUBLIC header and takes the
// struct, so a hand-built config reaches the process-global having passed no
// parser at all.
//
// The budget is the dangerous one rather than merely the wrong one.
// `ResolveDeviceWeightBudgetBytes` casts the configured `int64_t` to `size_t`, so
// an installed `-1` resolves to SIZE_MAX: an effectively infinite budget that
// switches the load-time device-fit refusal OFF and says nothing — precisely the
// "a budget the operator believes is set" failure that refusal's own text names.
// The resolver's comment justified that cast by trusting a parser that this door
// does not run, so the trust is made true here instead of being narrated there.
//
// `0` still installs. It is this field's suppression spelling, and a guard that
// refused it would delete the escape hatch the key exists to give.
void RejectOutOfRangeFields(const WeightResidencyConfig& c) {
  const auto positive = [](const char* name, std::optional<int64_t> v) {
    if (v.has_value() && *v <= 0) {
      throw std::invalid_argument(std::string("weight residency config: ") +
                                  name + " must be positive (got " +
                                  std::to_string(*v) + ")");
    }
  };
  positive("expert_stream_slots", c.expert_stream_slots);
  positive("expert_stream_slot_bytes", c.expert_stream_slot_bytes);
  if (c.device_weight_budget_bytes.has_value() &&
      *c.device_weight_budget_bytes < 0) {
    throw std::invalid_argument(
        "weight residency config: device_weight_budget_bytes must not be "
        "negative (got " +
        std::to_string(*c.device_weight_budget_bytes) +
        "); 0 means \"suppress the device-fit refusal\", and a negative value "
        "would resolve to SIZE_MAX and switch that refusal off silently");
  }
}

}  // namespace

void SetWeightResidencyConfig(const WeightResidencyConfig& config) {
  // BEFORE the lock and before the latch check: a value this struct may not hold
  // is refused whatever the process has already decided, and the refusal reads the
  // argument only.
  RejectOutOfRangeFields(config);
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  // REFUSE ONLY WHAT CANNOT BE HONOURED. A document that sets a decided field to a
  // value that would resolve differently from the decision is refused, because
  // recording it would publish a configuration the engine is not running — the
  // invisible-fallback shape this tree refuses everywhere else. Everything else
  // passes: an equal re-install, a document touching only unfrozen knobs (the
  // ordinary two-engine load), a document that OMITS a decided field, and one that
  // asks for what was decided. The empty document needs no special case at all now:
  // it sets nothing, so `FrozenFields` scores nothing and the merge below copies
  // nothing.
  const std::string frozen = FrozenFields(g, config);
  if (!frozen.empty()) {
    throw std::logic_error(
        "weight residency config: " + frozen +
        " cannot be changed after this process already decided it (" +
        DecisionSummary(g) +
        "). The streaming answer is cached on first read and the slot store is "
        "built once, so accepting this would record a configuration the engine "
        "is not running. Install it before any weight I/O. A document that OMITS "
        "a decided field, or asks for what was decided, is accepted; so are "
        "`mmap` and `prefault`, which no decision fixes");
  }
  // MERGE, FIELD BY FIELD, because an absent field means UNCHANGED. `g.config =
  // config` replaced wholesale, so a second engine's partial document silently
  // dropped the first engine's fields: `expert_stream=on` with 8000 slots became OFF
  // with 64, with no diagnostic, and the slot store reads those values LAZILY — on
  // the first slice taken, which can be long after the second engine loaded (#1133
  // H2). The wholesale replace predates the per-field refusal, but that narrowing
  // widened its reach: before it, any differing document after a decision threw, so
  // the drop could not happen once anything had been decided.
  if (config.mmap.has_value()) g.config.mmap = config.mmap;
  if (config.prefault.has_value()) g.config.prefault = config.prefault;
  if (config.expert_stream.has_value()) {
    g.config.expert_stream = config.expert_stream;
  }
  if (config.expert_stream_slots.has_value()) {
    g.config.expert_stream_slots = config.expert_stream_slots;
  }
  if (config.expert_stream_slot_bytes.has_value()) {
    g.config.expert_stream_slot_bytes = config.expert_stream_slot_bytes;
  }
  // `has_value()`, not a truth test on the number: `0` is a value the operator
  // SET, and it is this field's suppression spelling. A merge that skipped it
  // would keep an older budget in force while the operator believed the refusal
  // was off.
  if (config.device_weight_budget_bytes.has_value()) {
    g.config.device_weight_budget_bytes = config.device_weight_budget_bytes;
  }
}

// BY VALUE, and copied under the lock. Returning a reference and then releasing
// the mutex gave the caller an unsynchronised read behind a lock that looked like
// it covered one (#1122 L3). The copy is six optionals; the callers are per load,
// per prefaulted span (against megabytes of pages) and once per store.
WeightResidencyConfig ActiveWeightResidencyConfig() {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  return g.config;
}

bool WeightResidencyLatched() { return State().AnyLatched(); }

bool WeightResidencyLatched(ResidencyLatch knob) {
  return State().Latched(knob);
}

void ResetWeightResidencyConfigForTesting() {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  g.config = WeightResidencyConfig{};
  g.expert_stream_decision.store(0, std::memory_order_relaxed);
  g.latched_geometry.store(false, std::memory_order_relaxed);
  // The geometry ANSWER goes with its flag. Leaving it behind would make
  // `BuiltExpertStreamGeometry()` report a store this process no longer claims to
  // have built, which contradicts the accessor's own "both zero until something
  // builds one". NO TEST OBSERVES THIS, and the mutation that removes the line stays
  // GREEN: the numbers are read only while the geometry latch is set, and the reset
  // clears that too. It is coherence between an accessor and its documented contract,
  // not a gated guarantee, and it is recorded as such rather than claimed as tested.
  g.built_geometry = ExpertStreamGeometry{};
}

// NEITHER of these two helpers marks a latch. A latch is a property of the SITE
// that caches the answer, not of reading a variable, and marking it here is what
// made an ordinary `GgufLoadPolicy::FromEnv()` refuse a second engine's document.
bool ResolveResidencyBool(const char* env_name, std::optional<bool> configured,
                          bool builtin_default) {
  const char* v = std::getenv(env_name);
  if (v != nullptr) return EnvTruth(v);
  if (configured.has_value()) return *configured;
  return builtin_default;
}

int64_t ResolveResidencyCount(const char* env_name,
                              std::optional<int64_t> configured,
                              int64_t builtin_default) {
  if (const std::optional<int64_t> winner = EnvCountThatWins(env_name)) {
    return *winner;
  }
  if (configured.has_value()) return *configured;
  return builtin_default;
}

bool ResolveGgufMmap(bool builtin_default) {
  return ResolveResidencyBool("VT_GGUF_MMAP",
                              ActiveWeightResidencyConfig().mmap,
                              builtin_default);
}

namespace {
std::atomic<uint64_t>& PrefaultedSpans() {
  static std::atomic<uint64_t> n{0};
  return n;
}
}  // namespace

uint64_t GgufPrefaultedSpanCount() {
  return PrefaultedSpans().load(std::memory_order_relaxed);
}

void ResetGgufPrefaultedSpanCountForTesting() {
  PrefaultedSpans().store(0, std::memory_order_relaxed);
}

void NoteGgufPrefaultedSpan() {
  PrefaultedSpans().fetch_add(1, std::memory_order_relaxed);
}

ExpertStreamGeometry BuiltExpertStreamGeometry() {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  return g.built_geometry;
}

void NoteExpertStreamGeometry(int64_t slots, int64_t slot_bytes) {
  Global& g = State();
  std::lock_guard<std::mutex> lk(g.mu);
  // Into `Global` rather than a file-static, so `FrozenFields` — which already holds
  // `mu` — can compare an incoming document against what was actually built instead
  // of against what was stored.
  g.built_geometry.slots = slots;
  g.built_geometry.slot_bytes = slot_bytes;
  // THE GEOMETRY LATCH, and this is the moment it happens: the store now holds a
  // `slots x slot_bytes` reservation for the life of the process and cannot be
  // resized. Reading the two sizes freezes nothing; building the store does. Marked
  // here rather than in the resolvers so a run that never streams never freezes
  // anything.
  g.latched_geometry.store(true, std::memory_order_relaxed);
}

bool ResolveGgufPrefault() {
  return ResolveResidencyBool("VT_GGUF_PREFAULT",
                              ActiveWeightResidencyConfig().prefault,
                              /*builtin_default=*/true);
}

bool ExpertStreamRequestedFrom(const char* env_value,
                               std::optional<bool> configured) {
  // THE FIRST-CHARACTER RULE, transcribed rather than normalised. The site this
  // replaces read `v != nullptr && v[0] != '0' && v[0] != '\0'`, so
  // `VT_MOE_EXPERT_STREAM=false` is ON, and docs/ENVIRONMENT.md says so. It is
  // not routed through ResolveResidencyBool for exactly that reason: that helper
  // applies the tree's WHOLE-VALUE polarity and would silently flip this one.
  if (env_value != nullptr) return env_value[0] != '0' && env_value[0] != '\0';
  return configured.value_or(false);
}

bool ResolveExpertStreamRequested() {
  static const bool on = [] {
    return ExpertStreamRequestedFrom(
        std::getenv("VT_MOE_EXPERT_STREAM"),
        ActiveWeightResidencyConfig().expert_stream);
  }();
  // Record the decision AND its answer even on a cached call. Whether the value came
  // from this call or an earlier one, the process's answer is fixed from here, and
  // that pair is what SetWeightResidencyConfig compares a late document against. The
  // store is idempotent: `on` is a static, so every call writes the same enumerator.
  //
  // THIS IS WHY IT IS AN ATOMIC. This function is reached once per expert slice
  // through `KqExpertSlice`, so taking the process-wide mutex here would put a lock
  // in the decode loop of the lane the row is about. A relaxed store costs nothing
  // measurable and carries the only guarantee the install needs.
  State().expert_stream_decision.store(on ? 2 : 1, std::memory_order_relaxed);
  return on;
}

int64_t ResolveExpertStreamSlots() {
  return ResolveResidencyCount("VT_MOE_EXPERT_STREAM_SLOTS",
                               ActiveWeightResidencyConfig().expert_stream_slots,
                               /*builtin_default=*/64);
}

int64_t ResolveExpertStreamSlotBytes(int64_t computed_default) {
  return ResolveResidencyCount(
      "VT_MOE_EXPERT_STREAM_SLOT_BYTES",
      ActiveWeightResidencyConfig().expert_stream_slot_bytes, computed_default);
}

size_t ResolveDeviceWeightBudgetBytes(size_t probed_total_bytes) {
  if (const std::optional<size_t> winner = DeviceWeightBudgetEnvThatWins()) {
    return *winner;
  }
  const std::optional<int64_t> configured =
      ActiveWeightResidencyConfig().device_weight_budget_bytes;
  // `has_value()`, not `> 0`. A configured `0` is the operator suppressing the
  // refusal, and `CheckDeviceWeightFit` reads a zero budget as UNKNOWN. The parser
  // already refused a negative value at startup, so the cast is in range.
  if (configured.has_value()) return static_cast<size_t>(*configured);
  return probed_total_bytes;
}

}  // namespace vllm
