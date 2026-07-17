// vllm.cpp original (vt runtime env-flag plumbing); the async device path it
// selects is a 1:1 port of upstream vLLM's async scheduling, see the ENG-ASYNC-SCHED
// row and .agents/specs/async-serving.md.
//
// Env-flag plumbing for the async-scheduling runner opt-in, split into this
// pure-C++ header so the flag predicate is unit-testable on the CPU tier
// (tests/vllm/v1/worker/test_async_runner_flag.cpp) without touching the
// environment. VT_ASYNC_RUNNER turns ON the runner's async device path
// (combine_sampled_and_draft_tokens device-input + AsyncGPUModelRunnerOutput D2H),
// which makes GPUModelRunner::runner_supports_async() TRUE so LoadedEngine resolves
// an AsyncScheduler + max_concurrent_batches=2 (depth-2 step_with_batch_queue);
// VT_ASYNC_SCHED=0 then rolls the SCHEDULER back to synchronous in the same binary.
//
// DEFAULT ON since the 2026-07-17 flip (owner CLAIM-ASYNC-SCHED-W3): the W3 async
// stack was DGX-proven token-exact across default/W3-on/W3-off on both models
// (the f086b64 5/5-gate proof and the 6ea7856 discriminator's 6/6 token gates),
// and vLLM itself defaults async scheduling ON when compatible
// (vllm/config/vllm.py:992-1044) — a MIRROR obligation, paying the same TTFT
// premium vLLM does for the TPOT/ITL-tail win (both binding ITL-tail anomalies flip
// to PASS under W3-on). VT_ASYNC_RUNNER=0 is the runner-level rollback to the
// synchronous host input/output path (bit-exact reproduction of pre-flip token
// streams); VT_ASYNC_SCHED=0 is the scheduler-level rollback. Read at runner
// CONSTRUCTION (honoring the live env per build so the DGX A/B and the CPU
// construction-matrix test flip it per engine); the parse itself is factored here
// so it is regression-covered on every platform, not just DGX, mirroring the house
// VT_RMSNORM_DECODE_FAST / VT_GDN_PACKED_DECODE default-ON / '0'-rollback convention.
#ifndef VLLM_V1_WORKER_GPU_ASYNC_RUNNER_FLAG_H_
#define VLLM_V1_WORKER_GPU_ASYNC_RUNNER_FLAG_H_

namespace vllm::v1 {

// Pure predicate for the VT_ASYNC_RUNNER contract: the runner's async device path
// is ON by default (mirrors vLLM's async-scheduling default; DGX-proven token-exact
// — see header comment); it is OFF only when the environment value is present AND
// its first character is '0' (runner-level rollback to the synchronous host path).
// nullptr (unset) and every non-'0'-leading value are ON. Kept separate from the
// getenv call in the runner so the parse is unit-testable without touching the
// environment.
inline bool AsyncRunnerFlagIsOn(const char* env_value) {
  return env_value == nullptr || env_value[0] != '0';
}

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_ASYNC_RUNNER_FLAG_H_
