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

// AsyncSchedulingEnabled: apply the house rollback convention VT_ASYNC_SCHED on
// top of the config resolution (SchedulerConfig::ResolveAsyncScheduling). Read at
// engine-construction time (NOT per step — engine wiring calls it once):
//   VT_ASYNC_SCHED unset   -> `resolved` (the config resolution) is used as-is;
//   VT_ASYNC_SCHED=0        -> force OFF (restore the synchronous depth-1 path in
//                              the same binary — the rollback arm of the DGX A/B);
//   VT_ASYNC_SCHED=<other>  -> force ON.
// Mirrors vLLM defaulting async_scheduling ON when compatible plus its CLI/env
// override (--no-async-scheduling / async_scheduling=False).
bool AsyncSchedulingEnabled(bool resolved);

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

  // async_scheduling (scheduler.py:158-162): tri-state "asynchronous scheduling"
  // toggle. std::nullopt mirrors upstream's `async_scheduling: bool | None = None`
  // — "resolve to the default when compatible". vLLM's VllmConfig.__post_init__
  // resolves None -> True on a single-GPU MRV2 setup (vllm/config/vllm.py:990-1038,
  // the "Asynchronous scheduling is enabled" log), then disables it again for the
  // incompatible cases (pooling, non-EAGLE/MTP spec decode, unsupported runners).
  // ResolveAsyncScheduling() below performs that resolution for the T0 subset.
  // Set false to force it off, true to force it on.
  std::optional<bool> async_scheduling = std::nullopt;

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

  // ResolveAsyncScheduling: mirror vLLM's default-ON-when-compatible resolution
  // (vllm/config/vllm.py:990-1038). Returns the effective async_scheduling for
  // this config given the compatibility inputs relevant to our T0 subset:
  //   * async_scheduling explicitly false  -> off (user forced it off).
  //   * async_scheduling explicitly true   -> on  (user forced it on; upstream
  //       still errors on hard-incompatible combos, deferred here).
  //   * async_scheduling nullopt (default) -> ON when compatible, else off.
  // Compatibility gates ported for T0: pooling models and spec-decode methods
  // other than EAGLE/MTP disable async scheduling (vllm/config/vllm.py:959-1021).
  // `runner_supports_async` gates the whole thing on the MRV2 runner advertising
  // the placeholder-aware device-input path (our GPUModelRunner does not yet, so
  // the production engine keeps the synchronous Scheduler until that DGX-gated
  // leaf lands; the CPU test harness passes true). VT_ASYNC_SCHED=0 in the
  // environment force-disables it regardless (house rollback convention), read
  // by the engine wiring, not here.
  bool ResolveAsyncScheduling(bool runner_supports_async = true,
                              bool is_pooling_model = false,
                              bool spec_decode_incompatible = false) const {
    if (async_scheduling.has_value() && !async_scheduling.value()) {
      return false;  // explicitly forced off.
    }
    const bool forced_on = async_scheduling.value_or(false);
    if (!runner_supports_async || is_pooling_model || spec_decode_incompatible) {
      // Incompatible: upstream logs "Asynchronous scheduling is not supported ...
      // disabling" and turns it off even when the default would have enabled it.
      // (A user-forced `true` on a hard-incompatible combo raises upstream; that
      // hard-error path is deferred — we conservatively disable.)
      (void)forced_on;
      return false;
    }
    // Default (nullopt) or forced-on and compatible -> ON.
    return true;
  }

  // max_concurrent_batches (vllm/config/vllm.py:490-501): the engine's batch
  // queue depth. MRV2 single-GPU returns 2 under async scheduling (pp_size + 1
  // with pp_size == 1), which enables the depth-2 batch queue + flips step_fn to
  // step_with_batch_queue (core.py:196-223). 1 otherwise (no overlap).
  int MaxConcurrentBatches(bool async_scheduling_effective) const {
    return async_scheduling_effective ? 2 : 1;
  }
};

}  // namespace vllm
