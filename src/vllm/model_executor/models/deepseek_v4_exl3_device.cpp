// MODEL-DSV4-EXL3 W2 (#2442): make the coalesced trellis tower device-resident.
// Design, and why this is a seam rather than a new uploader, in the header.
#include "vllm/model_executor/models/deepseek_v4_exl3_device.h"

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev, ResidentWeight
#include "vllm/platforms/interface.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace vllm {
namespace {

using dense_attn::Dev;
using dense_attn::ResidentWeight;

// Wrap one host vector as an OwnedTensor that BORROWS its bytes, so
// `ResidentWeight` can upload it without a second host copy. The keep-alive
// owner is a no-op deleter over the vector's own storage: the vector outlives
// this call, and the borrow is dropped in the same function.
OwnedTensor BorrowU16(const std::vector<uint16_t>& v, std::vector<int64_t> shape,
                      vt::DType dt) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
  // A borrow needs an owner. `v` is alive for the whole call, so an aliasing
  // shared_ptr with an empty deleter is the honest expression of "someone else
  // owns these bytes" -- not a leak and not a second reference count on them.
  std::shared_ptr<const void> keep(static_cast<const void*>(v.data()), [](const void*) {});
  t.bytes = OwnedBytes::Borrow(reinterpret_cast<const uint8_t*>(v.data()),
                               v.size() * sizeof(uint16_t), std::move(keep));
  return t;
}

// Free a vector's storage outright. `clear()` keeps capacity, which on an 82 GiB
// tower means the whole point of staging is lost and nothing says so.
void FreeStorage(std::vector<uint16_t>& v) { std::vector<uint16_t>().swap(v); }

}  // namespace

void StageDeepseekV4Exl3LinearToDevice(vt::Queue& q, const DeepseekV4Exl3Linear& lin) {
  if (lin.device_staged) return;
  // The host IS the device here: `ResidentWeight` aliases rather than uploads,
  // so there is nothing to stage and the host vectors must survive.
  if (vllm::platforms::GetPlatform(q.device.type).is_cpu()) return;
  // Nothing to stage. An empty linear is a loader defect rather than a state to
  // paper over, but this function is not where it is diagnosed -- `Exl3Linear`
  // already refuses a shape mismatch by name, and `ResidentWeight` refuses an
  // empty weight by name. Staying silent here keeps ONE refusal per fact.
  if (lin.trellis.empty()) return;

  Dev d{vt::GetBackend(q.device), q};

  // The trellis is I8 [k/16, n/16, 32*bits], which is the SAME BYTES the
  // checkpoint stores as I16 [k/16, n/16, 16*bits] -- held at byte width because
  // that is what `vt::Exl3Gemm` reads and because `vt::DType` has no 16-bit
  // integer. This mirrors `Exl3Weight::trellis` exactly; a different spelling
  // here would be a second layout convention for one buffer.
  const int64_t k_tiles = lin.in_features / 16;
  const int64_t n_tiles = lin.out_features / 16;
  lin.d_trellis = BorrowU16(lin.trellis, {k_tiles, n_tiles, 32 * lin.bits}, vt::DType::kI8);
  lin.d_suh = BorrowU16(lin.suh, {lin.in_features}, vt::DType::kF16);
  lin.d_svh = BorrowU16(lin.svh, {lin.out_features}, vt::DType::kF16);

  // The upload itself. `ResidentWeight` allocates through the backend, copies
  // once and memoizes on `d_dev`; the return value is discarded because the
  // point here is the SIDE EFFECT, and the forward re-reads the memo.
  (void)ResidentWeight(d, lin.d_trellis);
  (void)ResidentWeight(d, lin.d_suh);
  (void)ResidentWeight(d, lin.d_svh);

  // Drop the borrows BEFORE freeing what they point at. A borrowed OwnedBytes
  // holds a raw pointer, so leaving it set across the free below leaves three
  // dangling views on a struct the forward reads -- and `d_dev` is what the
  // forward wants, so there is nothing to keep.
  lin.d_trellis.bytes = OwnedBytes{};
  lin.d_suh.bytes = OwnedBytes{};
  lin.d_svh.bytes = OwnedBytes{};

  FreeStorage(const_cast<std::vector<uint16_t>&>(lin.trellis));
  FreeStorage(const_cast<std::vector<uint16_t>&>(lin.suh));
  FreeStorage(const_cast<std::vector<uint16_t>&>(lin.svh));

  // LAST, and that order is the contract: `device_staged` means "the host
  // vectors are gone AND the device copies exist". Setting it before the frees
  // would make a throw in between publish a linear that is neither.
  lin.device_staged = true;
}

int64_t StageDeepseekV4Exl3TowerToDevice(vt::Queue& q, const DeepseekV4Exl3Weights& w) {
  // A REDUNDANT early-out, and named as such because a mutation proves it:
  // deleting this line leaves every case in
  // `test_deepseek_v4_exl3_device_residency.cpp` green, since the per-linear
  // entry carries the same guard and the walk would simply no-op 28k times.
  // It is kept for the cost, not the semantics -- ~28k predicate reads per
  // forward on the CPU arm buys nothing -- and a reader who deletes it changes
  // no behaviour. The guard that MATTERS is the one in
  // `StageDeepseekV4Exl3LinearToDevice`, and that one does red under mutation.
  if (vllm::platforms::GetPlatform(q.device.type).is_cpu()) return 0;
  int64_t uploaded = 0;
  for (const DeepseekV4Exl3LayerWeights& layer : w.layers) {
    for (const DeepseekV4Exl3Expert& e : layer.experts) {
      for (const DeepseekV4Exl3Linear* lin : {&e.w1, &e.w2, &e.w3}) {
        if (lin->device_staged) continue;
        const int64_t before = lin->Bytes();
        StageDeepseekV4Exl3LinearToDevice(q, *lin);
        if (lin->device_staged) uploaded += before;
      }
    }
  }
  return uploaded;
}

bool DeepseekV4Exl3TowerIsDeviceStaged(const DeepseekV4Exl3Weights& w) {
  bool any = false;
  for (const DeepseekV4Exl3LayerWeights& layer : w.layers) {
    for (const DeepseekV4Exl3Expert& e : layer.experts) {
      for (const DeepseekV4Exl3Linear* lin : {&e.w1, &e.w2, &e.w3}) {
        if (!lin->device_staged) return false;
        any = true;
      }
    }
  }
  // An EMPTY tower is not a staged tower. Answering true would tell the forward
  // a device queue is safe on a model that has no experts at all, which is the
  // vacuous-truth shape that makes a refusal stop refusing.
  return any;
}

}  // namespace vllm
