// Ported from: vllm/config/scheduler.py @ e24d1b24
//
// Scope (M1.4 Task 1): the T0 SUBSET of SchedulerConfig that the V1 Scheduler
// algorithm actually reads — the token-budget / admission knobs — plus the
// __post_init__ derivation and the verify_max_model_len checks that apply to
// those fields. Behavioral only.
//
// Ported T0 fields (names + defaults mirror upstream):
//   max_num_batched_tokens        (default 2048, >= 1)  -> the per-step budget
//   max_num_scheduled_tokens      (default None)        -> resolves to
//                                    max_num_batched_tokens (see below)
//   max_num_seqs                  (default 128, >= 1)
//   long_prefill_token_threshold  (default 0)           -> chunked-prefill cap
//   enable_chunked_prefill        (default true)
//   policy                        (default fcfs)
//   max_model_len                 (upstream InitVar, see DEVIATIONS)
//
// Derived in __post_init__ (mirrored):
//   max_num_encoder_input_tokens = max_num_batched_tokens
//   encoder_cache_size           = max_num_batched_tokens
//   (encoder path itself is DEFERRED to T3; the two derived budgets are set
//    faithfully so the config shape matches upstream.)
//
// DEFERRED upstream fields, intentionally omitted here (T1+):
//   runner_type, max_num_partial_prefills / max_long_partial_prefills (and the
//   long_prefill_token_threshold = int(max_model_len * 0.04) derivation they
//   gate — in T0 max_num_partial_prefills == 1 so long_prefill_token_threshold
//   stays 0), is_multimodal_model, disable_chunked_mm_input, scheduler_cls,
//   disable_hybrid_kv_cache_manager (watermark + scheduler_reserve_full_isl are
//   NOT deferred — modeled above, read by T0 schedule()),
//   prefill_schedule_interval, async_scheduling, stream_interval,
//   compute_hash / get_scheduler_cls.
//   The partial-prefill and max_long_partial_prefills verify branches are
//   likewise deferred (they are no-ops while max_num_partial_prefills == 1).
//
// DEVIATIONS, recorded:
//   - Upstream max_model_len (and is_encoder_decoder) are InitVars stored on
//     ModelConfig; ModelConfig is not ported yet, so max_model_len is stored on
//     SchedulerConfig for the T0 subset (the scheduler needs it) and both are
//     passed to PostInit() exactly as upstream passes them to __post_init__.
//   - max_num_scheduled_tokens stays a std::optional 1:1 with the upstream
//     `None` default; the fallback to max_num_batched_tokens happens in the
//     Scheduler ctor upstream (scheduler.py). ResolvedMaxNumScheduledTokens()
//     exposes that same fallback for callers/tests.
//   - The `max_num_batched_tokens > max_num_seqs * max_model_len` case logs a
//     warning upstream; there is no logger here, so it is a no-op (comment).
#pragma once

#include <optional>
#include <string>

namespace vllm {

// SchedulerPolicy = Literal["fcfs", "priority"]. String values are the wire
// form ("fcfs" is the default).
enum class SchedulerPolicy {
  kFCFS,      // "fcfs": first come first served (arrival order).
  kPriority,  // "priority": handled by (priority, arrival_time) — lower first.
};

// The wire string for a policy ("fcfs" / "priority").
const char* SchedulerPolicyToString(SchedulerPolicy policy);

// Parse a policy wire string into the enum. Mirrors upstream's Literal
// validation + Scheduler.__init__ `SchedulingPolicy(self.scheduler_config.policy)`
// (scheduler.py:175-178), which raises ValueError on an unknown value; here we
// throw std::invalid_argument("Unknown scheduling policy: <value>").
SchedulerPolicy SchedulerPolicyFromString(const std::string& value);

struct SchedulerConfig {
  // ClassVar defaults from upstream.
  static constexpr int kDefaultMaxNumBatchedTokens = 2048;
  static constexpr int kDefaultMaxNumSeqs = 128;

  // Maximum number of tokens processed in a single iteration (the per-step
  // token budget). Field(default=2048, ge=1).
  int max_num_batched_tokens = kDefaultMaxNumBatchedTokens;

  // Maximum number of tokens the scheduler may issue in a single iteration.
  // Field(default=None, ge=0); resolves to max_num_batched_tokens when unset.
  std::optional<int> max_num_scheduled_tokens = std::nullopt;

  // Maximum number of sequences processed in a single iteration.
  // Field(default=128, ge=1).
  int max_num_seqs = kDefaultMaxNumSeqs;

  // For chunked prefill, a request is "long" if its prompt exceeds this many
  // tokens. 0 disables the cap (default). Field(default=0, ge=0).
  int long_prefill_token_threshold = 0;

  // If true, prefill requests can be chunked based on the remaining
  // max_num_batched_tokens.
  bool enable_chunked_prefill = true;

  // The scheduling policy (fcfs default).
  SchedulerPolicy policy = SchedulerPolicy::kFCFS;

  // NOT deferred — read by the T0 schedule() core (M1.4 Task 3):
  //   watermark → passed to the KVCacheManager ctor (scheduler.py:275).
  //   scheduler_reserve_full_isl → passed as `full_sequence_must_fit` into the
  //     waiting-loop allocate_slots admission call (scheduler.py:914). Its
  //     upstream default is TRUE, while allocate_slots defaults
  //     full_sequence_must_fit to false — Task 3 MUST source it from here.
  double watermark = 0.0;
  bool scheduler_reserve_full_isl = true;

  // Upstream InitVar (stored on ModelConfig); kept here for the T0 subset since
  // ModelConfig is not ported yet. default_factory uses 8192.
  int max_model_len = 8192;

  // Derived in __post_init__ (= max_num_batched_tokens). Encoder path deferred.
  int max_num_encoder_input_tokens = kDefaultMaxNumBatchedTokens;
  int encoder_cache_size = kDefaultMaxNumBatchedTokens;

  // __post_init__(max_model_len, is_encoder_decoder): applies the
  // encoder-decoder disabling, sets the derived encoder budgets, then runs
  // verify_max_model_len. Stores max_model_len into the field (see DEVIATIONS).
  void PostInit(int max_model_len, bool is_encoder_decoder = false);

  // verify_max_model_len: the T0-applicable ValueError checks. Throws
  // std::invalid_argument on violation (mirrors upstream ValueError).
  void VerifyMaxModelLen(int max_model_len) const;

  // scheduler.py: max_num_scheduled_tokens if set else max_num_batched_tokens.
  int ResolvedMaxNumScheduledTokens() const {
    return max_num_scheduled_tokens.value_or(max_num_batched_tokens);
  }
};

}  // namespace vllm
