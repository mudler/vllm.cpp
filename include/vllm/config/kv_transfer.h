// Ported from: vllm/config/kv_transfer.py @ e24d1b24 (KVTransferConfig:22-121;
//               KVProducer/KVConsumer/KVRole:11-13).
//
// Scope (KV-CONNECTORS W5): the connector SELECTION + config surface. This is the
// analogue of `LinearMethod`'s config seam — POLICY (which connector, in which
// role, with what extra config) is chosen HERE by config; the connector itself
// (kv_connector.h) implements the transport. Default is `kv_connector` empty ==
// NO connector == zero behaviour change (offloading is opt-in and inert off).
//
// PORT NOTES (Python-incidental fields dropped, recorded):
//   - `kv_connector_module_path` (:62) is a Python `importlib` dynamic-load path;
//     it has no C++ equivalent worth having and is NOT ported. The C++ analogue is
//     the compile-time KVConnectorFactory registration (kv_connector.h), mirroring
//     our REGISTER_VLLM_MODEL seam. Recorded deviation.
//   - `kv_buffer_device`/`kv_buffer_size`/`kv_rank`/`kv_parallel_size`/`kv_ip`/
//     `kv_port`/`enable_permute_local_kv` are distributed-transfer (NIXL/P-D) knobs
//     for connectors NOT SCHEDULED on our boxes (external RDMA/store deps). They are
//     carried as fields for shape-fidelity so a future NIXL port is additive, but
//     the disk-offload and lmcache connectors this row ships do not read them.
//
// LOAD-BEARING here: `kv_connector` (selection), `kv_role` (required whenever
// `kv_connector` is set, :102-106), `kv_connector_extra_config` (per-connector
// config — e.g. the disk tier's root_dir/byte-budget), and
// `kv_load_failure_policy` (default `fail`, :69 — a `recompute` default would hide
// a torn-block bug as plausible-but-wrong output; §Risks R6).
#ifndef VLLM_CONFIG_KV_TRANSFER_H_
#define VLLM_CONFIG_KV_TRANSFER_H_

#include <map>
#include <optional>
#include <string>

namespace vllm {

// Upstream KVRole (kv_transfer.py:11-13): kv_producer, kv_consumer, kv_both.
enum class KVRole {
  kProducer,  // "kv_producer"
  kConsumer,  // "kv_consumer"
  kBoth,      // "kv_both"
};

// Upstream kv_load_failure_policy (kv_transfer.py:69). Default `fail`.
enum class KVLoadFailurePolicy {
  kRecompute,  // "recompute": reschedule the request to recompute failed blocks
  kFail,       // "fail" (default): fail the request with an error finish reason
};

// Upstream: @config class KVTransferConfig (kv_transfer.py:22-121).
struct KVTransferConfig {
  // The KV connector name selecting the concrete transport. Empty (nullopt) ==
  // NO connector == zero behaviour change. Mirrors `kv_connector: str | None`.
  std::optional<std::string> kv_connector;

  // The engine id for KV transfers. Upstream defaults to a fresh uuid4 in
  // __post_init__ (:93-94); Validate() fills a deterministic placeholder when
  // empty so a config with no id is still usable.
  std::optional<std::string> engine_id;

  // Whether this instance produces, consumes, or both. REQUIRED whenever
  // kv_connector is set (:102-106).
  std::optional<KVRole> kv_role;

  // Any extra config the connector needs (the disk tier reads root_dir,
  // cpu_bytes_to_use, eviction_policy, etc. from here — mirrors upstream putting
  // all offloading-specific config inside kv_connector_extra_config).
  std::map<std::string, std::string> kv_connector_extra_config;

  // Policy for handling KV load failures. Default `fail` (:69).
  KVLoadFailurePolicy kv_load_failure_policy = KVLoadFailurePolicy::kFail;

  // --- validation + predicates (mirrors __post_init__ + the @property set) ----

  // Mirrors __post_init__ (:92-106): kv_role is required whenever kv_connector
  // is set; fills engine_id when absent. Throws std::invalid_argument on a
  // kv_connector-without-kv_role config. Idempotent.
  void Validate();

  // kv_transfer.py:108-110. True iff a connector is configured with a role.
  bool is_kv_transfer_instance() const {
    return kv_connector.has_value() && kv_role.has_value();
  }
  // kv_transfer.py:112-114.
  bool is_kv_producer() const {
    return kv_connector.has_value() && kv_role.has_value() &&
           (*kv_role == KVRole::kProducer || *kv_role == KVRole::kBoth);
  }
  // kv_transfer.py:116-118.
  bool is_kv_consumer() const {
    return kv_connector.has_value() && kv_role.has_value() &&
           (*kv_role == KVRole::kConsumer || *kv_role == KVRole::kBoth);
  }
  // kv_transfer.py:120-121.
  std::string get_from_extra_config(const std::string& key,
                                    const std::string& default_value) const {
    auto it = kv_connector_extra_config.find(key);
    return it == kv_connector_extra_config.end() ? default_value : it->second;
  }
};

// String <-> enum helpers mirroring the upstream Literal string values, so a CLI
// flag / JSON config round-trips through the exact upstream tokens.
std::optional<KVRole> parse_kv_role(const std::string& s);
const char* kv_role_str(KVRole role);
std::optional<KVLoadFailurePolicy> parse_kv_load_failure_policy(
    const std::string& s);
const char* kv_load_failure_policy_str(KVLoadFailurePolicy policy);

}  // namespace vllm

#endif  // VLLM_CONFIG_KV_TRANSFER_H_
