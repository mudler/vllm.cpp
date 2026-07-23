// Ported from: vllm/config/kv_transfer.py:92-121 @ e24d1b24
#include "vllm/config/kv_transfer.h"

#include <stdexcept>

namespace vllm {

std::optional<KVRole> parse_kv_role(const std::string& s) {
  if (s == "kv_producer") return KVRole::kProducer;
  if (s == "kv_consumer") return KVRole::kConsumer;
  if (s == "kv_both") return KVRole::kBoth;
  return std::nullopt;
}

const char* kv_role_str(KVRole role) {
  switch (role) {
    case KVRole::kProducer:
      return "kv_producer";
    case KVRole::kConsumer:
      return "kv_consumer";
    case KVRole::kBoth:
      return "kv_both";
  }
  return "";
}

std::optional<KVLoadFailurePolicy> parse_kv_load_failure_policy(
    const std::string& s) {
  if (s == "recompute") return KVLoadFailurePolicy::kRecompute;
  if (s == "fail") return KVLoadFailurePolicy::kFail;
  return std::nullopt;
}

const char* kv_load_failure_policy_str(KVLoadFailurePolicy policy) {
  switch (policy) {
    case KVLoadFailurePolicy::kRecompute:
      return "recompute";
    case KVLoadFailurePolicy::kFail:
      return "fail";
  }
  return "";
}

// kv_transfer.py:92-106. kv_role is REQUIRED whenever kv_connector is set; a
// missing engine_id is filled (upstream uses uuid4 — we use a deterministic
// placeholder so an unconfigured id is still valid and reproducible).
void KVTransferConfig::Validate() {
  if (!engine_id.has_value() || engine_id->empty()) {
    engine_id = "vllm-cpp-engine";
  }
  if (kv_connector.has_value() && !kv_role.has_value()) {
    throw std::invalid_argument(
        "Please specify kv_role when kv_connector is set (supported roles: "
        "kv_producer, kv_consumer, kv_both).");
  }
}

}  // namespace vllm
