// DeepSeek-V4 EXL3 tower: DEVICE RESIDENCY (MODEL-DSV4-EXL3 W2, #2442).
//
// THE DEFECT THIS CLOSES. `DeepseekV4Exl3Linear` holds the TP1-coalesced trellis
// tower in host `std::vector<uint16_t>`s, and `deepseek_v4.cpp` built its
// `vt::Tensor`s straight over those host pointers while LABELLING them with the
// forward's device. Handing that to a CUDA kernel is the #844 / #1435 crash, so
// both `Exl3Linear` and the fused MoE arm refused any non-CPU queue by name.
// Every routed expert of the real 216-expert artifact therefore ran on a CPU
// queue, which is why the 44-47 tok/s target was out of reach for a reason that
// had nothing to do with the drafter.
//
// It is NOT a CUDA-specific defect and this is NOT a CUDA-specific fix. A
// discrete ROCm or Metal device cannot dereference host pointers either, and on
// a unified-memory box (GB10) reaching host memory from a kernel costs the ATS
// penalty this tree already moved Kimi's weights off. Device allocations are the
// answer on every accelerator.
//
// THE SEAM, not a parallel path. Staging goes through
// `dense_attn::ResidentWeight`, the same helper that makes every other model's
// weights device-resident: it allocates through `vt::Backend`, copies once,
// caches on `OwnedTensor::d_dev` and carries the #1953/#1946 reasoning about
// empty weights and about host-pointer aliasing being a CPU property rather than
// a not-CUDA one. Writing a second uploader here is what AGENTS.md
// §"Shared seams" forbids.
//
// PEAK RESIDENCY is why this stages per tensor and frees as it goes. The
// measured tower is 81.952 GiB; uploading it while keeping the host copy would
// need ~164 GiB and OOM a 119 GiB GB10. `StageDeepseekV4Exl3LinearToDevice`
// releases each host vector immediately after its upload, so the peak is the
// tower plus one tensor -- the same stage-then-release shape
// `StageKimiResidentBf16` uses, and for the same reason.
#pragma once

#include "vllm/model_executor/models/deepseek_v4.h"
#include "vt/device.h"

namespace vllm {

// Stage ONE coalesced EXL3 linear into device memory and free its host vectors.
//
// A CPU queue is a deliberate NO-OP: `ResidentWeight` aliases host bytes when
// the "device" is the host, so there is nothing to upload and the host vectors
// must stay. `device_staged` therefore stays false on CPU, and the forward's
// host arm keeps serving it byte-identically.
//
// Idempotent: a linear that is already staged is left alone.
void StageDeepseekV4Exl3LinearToDevice(vt::Queue& q, const DeepseekV4Exl3Linear& lin);

// Stage every routed expert of every layer. Returns the bytes uploaded, which is
// 0 on a CPU queue and on an already-staged tower.
//
// Called once, from the forward, the first time it sees a device queue -- the
// loader has no queue in hand, and a load-time hook would have to invent one.
int64_t StageDeepseekV4Exl3TowerToDevice(vt::Queue& q, const DeepseekV4Exl3Weights& w);

// Is every routed expert of this tower device-resident? The forward reads this
// to decide whether a device queue is admissible; a PARTIALLY staged tower
// answers false, because one host expert in a device kernel is the same crash as
// all of them.
bool DeepseekV4Exl3TowerIsDeviceStaged(const DeepseekV4Exl3Weights& w);

}  // namespace vllm
