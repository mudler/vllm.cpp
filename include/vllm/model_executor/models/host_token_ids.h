// THE HOST ARM of `ModelForwardInput::device_token_ids`.
//
// Row `ENG-ASYNC-DEVICE-IDS-2544`, spec
// `.agents/specs/eng-async-device-ids-2544.md`, issue
// [#2544](https://github.com/mudler/vllm.cpp/issues/2544). The contract itself
// is [#1305](https://github.com/mudler/vllm.cpp/issues/1305)'s and the
// measurement that convicted the first model is
// [#2496](https://github.com/mudler/vllm.cpp/issues/2496)'s.
//
// ─── WHAT THE CONTRACT SAYS ─────────────────────────────────────────────────
//
// `ModelForwardInput::device_token_ids` (see `model_registry.h`) means: the
// `[token_ids.size()]` input identifiers for this step ALREADY live in that
// device buffer, and `token_ids` is stale for decode rows. The asynchronous
// runner's combine splices each decode row's sampled token into the device
// buffer on the main queue and deliberately never writes it back
// (`src/vllm/v1/worker/gpu/runner.cpp`, the mirror arm), because materialising
// it on the host is the synchronise that path exists to remove. And
// `async_device_mirror()` is the DEFAULT on CUDA — integrated parts included.
//
// A forward that ignores the field therefore embeds a host array the runner
// never wrote. `token_ids_cpu` is zero-initialised, so the model is fed **token
// id 0** at every decode step: the prefill agrees, the first token looks right,
// and every token after it is generated from 0 — silently, at `rc=0`, with
// plausible-looking output.
//
// ─── WHY THIS IS A SECOND ENTRY POINT AND NOT A SECOND COPY ─────────────────
//
// The seam already has two arms and this is the third:
//
//   * the EAGER DEVICE arm, `detail::ApplyDeviceTokenIds`
//     (`src/vllm/model_executor/models/qwen3_5_internal.h`), which splices the
//     published device pointer over a DEVICE buffer an embed has already filled
//     from the host;
//   * the DECODE-GRAPH SLOT arm, `vllm::StepTokenIds`
//     (`step_token_ids.h`), which re-reads a graph slot's persistent device
//     destination over the real prefix.
//
// Both destinations are device memory. `Glm5NextForConditionalGeneration`,
// `DeepseekV4ForCausalLM` and `LagunaForCausalLM` have neither: each takes the
// identifiers as a host `std::vector<int32_t>` and gathers the embedding rows on
// the host, so there is no `dst` to splice. Handing a host vector's `data()` to
// the device consumer would be a device-to-host copy through an interface
// documented as device-to-device AND — because that copy is enqueued and never
// awaited — it would race the host gather that reads it. So the existing seam
// cannot express these three, which is the condition AGENTS.md §"Shared seams"
// gives for extending one rather than routing through it.
//
// ─── WHAT IT COSTS, NAMED RATHER THAN HIDDEN ────────────────────────────────
//
// This DOWNLOADS and SYNCHRONISES, which is the very thing the asynchronous path
// exists to avoid. That is honest here and would not be elsewhere: all three
// callers are host gathers that dereference host memory before any device work,
// so they have no asynchrony left to protect and were already synchronous end to
// end. The trade is not "slower for faster"; it is correct tokens on a path that
// is otherwise silently wrong. A caller that later grows a device embed must move
// to `detail::ApplyDeviceTokenIds` instead of keeping this.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {

struct ModelForwardInput;

// Resolve the input identifiers a HOST-GATHER forward must embed from.
//
// Returns `input.token_ids` unchanged, and touches nothing else, whenever
// `input.device_token_ids` is null — which is every path except the asynchronous
// CUDA runner, so those builds stay byte-identical by construction.
//
// Otherwise it copies `input.token_ids` into `*storage`, overwrites its first
// `input.token_ids.size()` entries from the device buffer with one
// `vt::Backend::Copy` enqueued ON `input.queue`, synchronises that queue, and
// returns `*storage`.
//
// THE QUEUE IS THE POINT. `input.queue` is the main queue, which is where the
// runner's combine wrote the source, so the copy is ordered AFTER the combine
// rather than racing it. That ordering is why a host read of
// `ModelForwardInput::token_ids` cannot substitute for this call.
//
// `storage` is the CALLER's buffer, never a function-local static: a returned
// reference into a static would alias across nested forwards (the multimodal
// generate helper reaches a second embed inside one call), and caller ownership
// makes the lifetime exactly the forward's.
//
// `what` names the caller in the refusal a shape disagreement raises. The device
// buffer is the runner's statement about THIS step; a step whose host vector is
// empty while the runner published identifiers can only mean the two disagree
// about the step's shape, and that is refused rather than embedded past the end.
const std::vector<int32_t>& ResolveHostTokenIds(const ModelForwardInput& input,
                                                std::vector<int32_t>* storage,
                                                const char* what);

}  // namespace vllm
