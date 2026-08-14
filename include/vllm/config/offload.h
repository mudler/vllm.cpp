// Ported from: vllm/config/offload.py @ 555967922 (OffloadBackend:12;
//               UVAOffloadConfig:16-44; PrefetchOffloadConfig:48-76;
//               OffloadConfig:80-93; validate_offload_config:96-136).
//
// Scope (ENG-WEIGHT-OFFLOAD W0, spec .agents/specs/weight-offload-uva.md, issue
// #797): the weight-offload SELECTION + config surface ONLY. No offloader is
// constructed here and no weight moves; this is the analogue of kv_transfer.h,
// where POLICY (which backend, what budget, which parameters) is chosen by
// config and the mechanism lives elsewhere. Default is `auto` with both budgets
// at zero == NO offloading == zero behaviour change, mirroring upstream's
// `create_offloader` returning NoopOffloader (offloader/base.py:126-162).
//
// WHY A MIRROR AND NOT A DESIGN: vLLM ships this whole surface at the pin, so
// every field name, default, bound, error and warning below is TRANSCRIBED. The
// one thing this row may not do is invent a knob.
//
// PORT NOTES (deviations, recorded):
//   - Upstream's bounds are pydantic `Field(ge=...)` constraints checked at
//     construction (:23,54,62,66). C++ has no such decorator, so they are
//     enforced in Validate() and are therefore SEPARATE errors from the two
//     cross-field errors upstream raises in its @model_validator. Both kinds
//     throw std::invalid_argument; the messages name which bound was violated.
//   - Upstream emits its three backend/field-mismatch cases as Python
//     `warnings.warn` (:120-135). A C++ library has no warnings module, so they
//     are COLLECTED into `warnings` for the caller to log. They are warnings and
//     never errors, exactly as upstream: a mismatched pair is honoured with the
//     backend winning, not refused.
//   - `compute_hash` (:138) is not ported: we have no config-hash surface (the
//     same reason recorded in tests/vllm/config/test_multimodal_config.cpp).
//
// THE ASYMMETRY THAT IS EASY TO GET WRONG: an empty `cpu_offload_params`
// (:34-44) means "offload NON-SELECTIVELY until the byte limit"; an empty
// `offload_params` (:70-76) means "ALL parameters OF EACH OFFLOADED LAYER".
// They are not the same default and the docstrings say so explicitly.
#ifndef VLLM_CONFIG_OFFLOAD_H_
#define VLLM_CONFIG_OFFLOAD_H_

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace vllm {

// Upstream OffloadBackend (offload.py:12): Literal["auto", "uva", "prefetch"].
enum class OffloadBackend {
  kAuto,      // "auto" (default): select from which sub-config is non-default
  kUva,       // "uva": Unified Virtual Addressing zero-copy offloading
  kPrefetch,  // "prefetch": async prefetch with group-based layer offloading
};

// Upstream @config class UVAOffloadConfig (offload.py:16-44).
struct UVAOffloadConfig {
  // offload.py:23 `cpu_offload_gb: float = Field(default=0, ge=0)`. The space in
  // GiB to offload to CPU, PER GPU. 0 == no offloading.
  double cpu_offload_gb = 0.0;

  // offload.py:34 `cpu_offload_params: set[str] = Field(default_factory=set)`.
  // Parameter-name SEGMENTS to target. EMPTY => offload non-selectively until
  // `cpu_offload_gb` is reached (:36-38).
  std::set<std::string> cpu_offload_params;

  // offload.py:39-43 is the authority on matching: for "mlp.experts.w2_weight",
  // "experts" and "experts.w2_weight" MATCH; "expert" and "w2" do NOT. Upstream
  // implements it as `f".{param}." in f".{name}."` (offloader/uva.py:91-93) —
  // the dot-wrapping on BOTH sides is what makes it a segment match rather than
  // a substring match, which is what distinguishes "w2_weight" from
  // "w2_weight_scale".
  static bool MatchesSegment(const std::string& param_name,
                             const std::string& segment);

  // True iff `param_name` should be offloaded under this config's targeting.
  // An empty `cpu_offload_params` matches EVERYTHING (the byte budget, not the
  // name set, is then the only limit).
  bool Targets(const std::string& param_name) const;

  // The byte budget, mirroring create_offloader's
  // `int(uva.cpu_offload_gb * 1024**3)` (offloader/base.py:155).
  int64_t cpu_offload_max_bytes() const;
};

// Upstream @config class PrefetchOffloadConfig (offload.py:48-76).
struct PrefetchOffloadConfig {
  // offload.py:54 `offload_group_size: int = Field(default=0, ge=0)`. Group
  // every N layers; offload the LAST `offload_num_in_group` of each group.
  // 0 == disabled. Worked example (:57): group_size=8, num_in_group=2 offloads
  // layers 6,7,14,15,22,23,...
  int64_t offload_group_size = 0;

  // offload.py:62 `offload_num_in_group: int = Field(default=1, ge=1)`.
  int64_t offload_num_in_group = 1;

  // offload.py:66 `offload_prefetch_step: int = Field(default=1, ge=0)`. Layers
  // to prefetch ahead. NOTE the bound is ge=0 on the FIELD but the validator
  // additionally requires >= 1 whenever prefetch is enabled (:106-112) — the
  // two are not the same check and both are ported.
  int64_t offload_prefetch_step = 1;

  // offload.py:70 `offload_params: set[str]`. EMPTY => ALL parameters OF EACH
  // OFFLOADED LAYER (:72-73) — NOT the same default as cpu_offload_params.
  std::set<std::string> offload_params;

  // True iff layer `layer_idx` is offloaded under this grouping. Mirrors the
  // worked example at :57. Returns false when disabled (group_size == 0).
  bool OffloadsLayer(int64_t layer_idx) const;

  // True iff `param_name` is targeted. Empty set matches everything.
  bool Targets(const std::string& param_name) const;
};

// Upstream @config class OffloadConfig (offload.py:80-93).
struct OffloadConfig {
  // offload.py:83, default "auto".
  OffloadBackend offload_backend = OffloadBackend::kAuto;

  // offload.py:90,93.
  UVAOffloadConfig uva;
  PrefetchOffloadConfig prefetch;

  // Non-fatal diagnostics produced by Validate(), mirroring upstream's three
  // `warnings.warn` calls (:120-135). Cleared and repopulated on every call.
  std::vector<std::string> warnings;

  // Mirrors validate_offload_config (:96-136) plus the pydantic field bounds.
  // Throws std::invalid_argument on:
  //   * any negative/out-of-bounds field (the ge= constraints), and
  //   * offload_num_in_group > offload_group_size            (:100-106), and
  //   * offload_prefetch_step < 1 when prefetch is enabled   (:107-112).
  // Populates `warnings` for the three backend/field mismatches. Idempotent.
  void Validate();

  // The backend SELECTED, mirroring create_offloader (offloader/base.py:139-149)
  // exactly: under "auto", prefetch if offload_group_size > 0, elif uva if
  // cpu_offload_gb > 0, else NoopOffloader (nullopt here). Under an EXPLICIT
  // backend, that backend is returned even when its budget is zero — upstream
  // constructs the offloader either way, and it is the budget, not the
  // selection, that then makes it move nothing.
  std::optional<OffloadBackend> ResolvedBackend() const;

  // True iff this config would actually move any weight. This is a DIFFERENT
  // question from ResolvedBackend(): an explicit `uva` with cpu_offload_gb == 0
  // selects a backend that offloads nothing, and the engine must still take its
  // existing byte-identical path. Keeping the two apart is what stops a
  // zero-budget explicit backend from being read as "offloading is on".
  bool is_offloading_enabled() const;
};

// "auto" | "uva" | "prefetch" -> enum; nullopt on an unknown name.
std::optional<OffloadBackend> parse_offload_backend(const std::string& s);
const char* offload_backend_str(OffloadBackend backend);

// Parse the JSON document vLLM's OffloadConfig takes, mirroring how
// kv_transfer_config accepts `--kv-transfer-config` verbatim. An empty or blank
// document yields a default (inert) config. Throws std::invalid_argument on a
// malformed document, an unknown backend name, or a field of the wrong type.
// Does NOT call Validate(); the caller does, so parse errors and validation
// errors stay distinguishable.
OffloadConfig parse_offload_config_json(const std::string& json_text);

}  // namespace vllm

#endif  // VLLM_CONFIG_OFFLOAD_H_
