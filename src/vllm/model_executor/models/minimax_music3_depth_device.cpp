// MiniMax-Music3 — the DEVICE-RESIDENT RVQ depth decoder. See
// minimax_music3_depth_device.h for what is ported onto which shared op, why the
// dtype is bf16 and why the narrowing is lossless, and the three named reasons
// this arm is close to but not bit-identical to the host reference.
//
// ─── THE ONE STRUCTURAL DECISION IN THIS FILE ────────────────────────────────
//
// `vt::AttentionCross` requires CONTIGUOUS operands, so a row's key history must
// be contiguous ACROSS positions. That fixes the cache as one [capacity, H]
// block per (layer, row) and means the batched K/V projection — which produces
// [batch, 3H] with the two rows interleaved — cannot be written into the cache
// by the GEMM itself. It is copied in, H bf16 values per row per tensor.
//
// The alternative was a per-row K/V GEMM writing straight into the slot, and it
// is the wrong trade by an order of magnitude: it would sweep the [H, H] weight
// once PER ROW instead of once per call, which is exactly the weight-streaming
// cost §16.4 measured and this arm exists to remove. An 8 KB copy against a
// 2.28 GB sweep is not a trade at all.
#include "vllm/model_executor/models/minimax_music3_depth_device.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/layers/linear.h"  // layers::UnquantizedMlpGateUpMethod
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"  // GetOp — the stage-time provider refusal
#include "vt/ops.h"

namespace vllm {
namespace models {
namespace music3 {

namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using vt::DType;
using vt::Tensor;

[[noreturn]] void Fail(const std::string& message) { throw std::runtime_error(message); }

// See `Music3DepthDeviceForwardCount`. Relaxed because this is an instrument
// read after a synchronizing call, never a synchronization primitive itself.
std::atomic<uint64_t> g_forward_count{0};

// See `Music3DepthDeviceResidentDtypes`. Same ordering argument as the counter
// above: an instrument, not a synchronization primitive.
std::atomic<uint64_t> g_resident_dtypes{0};

// Record the dtype of a buffer the forward made resident. `DType` is a `uint8_t`
// enum of 20 values, so one bit each fits a `uint64_t` with room to spare, and
// the static_assert makes adding the 65th dtype a build error rather than a
// silently truncated instrument.
constexpr uint64_t DtypeBit(DType dt) {
  return uint64_t{1} << static_cast<unsigned>(dt);
}
static_assert(static_cast<unsigned>(DType::kMXFP4) < 64,
              "the resident-dtype mask has one bit per DType and DType outgrew 64");

void RequireStageSize(const std::vector<float>& values, int64_t expected, const char* what) {
  if (static_cast<int64_t>(values.size()) != expected) {
    Fail(std::string("MiniMax-Music3 depth stage: ") + what + " is " +
         std::to_string(values.size()) + " values, expected " + std::to_string(expected));
  }
}

// A bf16 device view over a fresh block, owned by `storage`, built from one or
// more f32 host sources CONCATENATED in order — which is how the two merged
// projections are assembled without a second host buffer.
//
// THE NARROWING IS EXACT AND THAT IS LOAD-BEARING, not a rounding we tolerate.
// `AtRuntimeDtype` already rounded every AR-half weight through bf16 into its
// f32 carrier, so `F32ToBF16` here is the inverse of the widening the loader
// did, value for value. See the header.
Tensor UploadBf16(vt::Backend& backend, vt::Queue& queue,
                  const std::vector<std::vector<float>*>& sources,
                  const std::vector<int64_t>& shape, bool release,
                  std::vector<std::shared_ptr<void>>* storage) {
  int64_t numel = 1;
  for (int64_t s : shape) numel *= s;
  std::vector<uint16_t> packed(static_cast<size_t>(numel));
  size_t at = 0;
  for (std::vector<float>* src : sources) {
    for (float v : *src) packed[at++] = vt::F32ToBF16(v);
  }
  if (at != packed.size()) {
    Fail("MiniMax-Music3 depth stage: staged " + std::to_string(at) + " values into a " +
         std::to_string(packed.size()) + "-value tensor");
  }
  const size_t bytes = packed.size() * sizeof(uint16_t);
  void* p = backend.Alloc(bytes);
  std::shared_ptr<void> owner(p, [&backend](void* q) { backend.Free(q); });
  backend.Copy(queue, p, packed.data(), bytes);
  // The copy must have LANDED before `packed` leaves scope. On a CPU queue this
  // is a memcpy and the sync is free; on CUDA the source is pageable host
  // memory, so returning under an unsynchronized async copy is exactly the
  // use-after-free that reads as a plausible-looking wrong tensor.
  backend.Synchronize(queue);
  if (release) {
    for (std::vector<float>* src : sources) std::vector<float>().swap(*src);
  }
  storage->push_back(std::move(owner));
  return MakeTensor(p, DType::kBF16, queue.device, shape);
}

// The same lossless f32 -> bf16 concatenation as `UploadBf16`, but left on the
// HOST as an `OwnedTensor` so the shared merged-GEMM seam owns its residency.
// `nk = true` is the torch Linear [N=out, K=in] orientation `vt::MatmulBT`
// consumes, which is the orientation these weights already have.
OwnedTensor PackBf16Owned(const std::vector<std::vector<float>*>& sources,
                          const std::vector<int64_t>& shape, bool release) {
  int64_t numel = 1;
  for (int64_t s : shape) numel *= s;
  std::vector<uint8_t> raw(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* packed = reinterpret_cast<uint16_t*>(raw.data());
  size_t at = 0;
  for (std::vector<float>* src : sources) {
    for (float v : *src) packed[at++] = vt::F32ToBF16(v);
  }
  if (at != static_cast<size_t>(numel)) {
    Fail("MiniMax-Music3 depth stage: packed " + std::to_string(at) + " values into a " +
         std::to_string(numel) + "-value tensor");
  }
  if (release) {
    for (std::vector<float>* src : sources) std::vector<float>().swap(*src);
  }
  OwnedTensor out;
  out.bytes = OwnedBytes(std::move(raw));
  out.dtype = DType::kBF16;
  out.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) out.shape[i] = shape[i];
  out.nk = true;
  return out;
}

// A contiguous sub-view of a bf16 [rows, width] buffer starting at `row`.
Tensor RowView(const Tensor& base, int64_t row, const std::vector<int64_t>& shape) {
  auto* p = static_cast<uint16_t*>(base.data) + row * base.shape[base.rank - 1];
  return MakeTensor(p, DType::kBF16, base.device, shape);
}

}  // namespace

uint64_t Music3DepthDeviceForwardCount() { return g_forward_count.load(std::memory_order_relaxed); }

uint64_t Music3DepthDeviceResidentDtypes() {
  return g_resident_dtypes.load(std::memory_order_relaxed);
}

Music3DepthDeviceWeights StageMusic3DepthWeights(vt::Queue& queue,
                                                 const DepthDecoderConfig& config,
                                                 DepthDecoderWeights& weights,
                                                 bool release_host) {
  const int64_t hidden = config.hidden_size;
  const int64_t inter = config.intermediate_size;
  const int64_t heads = config.num_attention_heads;
  if (heads <= 0 || hidden % heads != 0) {
    Fail("MiniMax-Music3 depth stage: hidden_size " + std::to_string(hidden) +
         " is not divisible by num_attention_heads " + std::to_string(heads));
  }
  if (static_cast<int64_t>(weights.layers.size()) != config.num_layers) {
    Fail("MiniMax-Music3 depth stage: the weights carry " +
         std::to_string(weights.layers.size()) + " layers, the config declares " +
         std::to_string(config.num_layers));
  }

  // REFUSE UP FRONT, before 1.3 GB moves. `vt::GetOp` throws naming the op and
  // the device, so a backend without (say) a cross-attention provider is a
  // one-line refusal at stage time rather than a failure four layers into the
  // first of 808 calls. Every op the forward below calls is listed.
  for (vt::OpId op : {vt::OpId::kMatmulBT, vt::OpId::kAdd, vt::OpId::kRmsNorm,
                      vt::OpId::kSiluAndMul, vt::OpId::kAttentionCross}) {
    (void)vt::GetOp(op, queue.device.type);
  }

  // EVERY size is checked BEFORE anything is uploaded, and therefore before
  // `release_host` destroys anything. A size error found after nine layers had
  // been released would be a correct refusal that had already consumed the
  // caller's weights.
  RequireStageSize(weights.pos_embedding, config.max_position_embeddings * hidden,
                   "pos_embedding");
  RequireStageSize(weights.norm, hidden, "norm");
  for (size_t l = 0; l < weights.layers.size(); ++l) {
    const DepthDecoderLayerWeights& layer = weights.layers[l];
    const std::string at = "layer " + std::to_string(l) + " ";
    RequireStageSize(layer.input_layernorm, hidden, (at + "input_layernorm").c_str());
    RequireStageSize(layer.post_attention_layernorm, hidden,
                     (at + "post_attention_layernorm").c_str());
    RequireStageSize(layer.to_q, hidden * hidden, (at + "to_q").c_str());
    RequireStageSize(layer.to_k, hidden * hidden, (at + "to_k").c_str());
    RequireStageSize(layer.to_v, hidden * hidden, (at + "to_v").c_str());
    RequireStageSize(layer.to_out, hidden * hidden, (at + "to_out").c_str());
    RequireStageSize(layer.gate_proj, inter * hidden, (at + "gate_proj").c_str());
    RequireStageSize(layer.up_proj, inter * hidden, (at + "up_proj").c_str());
    RequireStageSize(layer.down_proj, hidden * inter, (at + "down_proj").c_str());
  }

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  Music3DepthDeviceWeights out;
  out.pos_embedding = UploadBf16(backend, queue, {&weights.pos_embedding},
                                 {config.max_position_embeddings, hidden}, release_host,
                                 &out.storage);
  out.norm = UploadBf16(backend, queue, {&weights.norm}, {hidden}, release_host, &out.storage);
  out.layers.resize(static_cast<size_t>(config.num_layers));
  for (size_t l = 0; l < out.layers.size(); ++l) {
    DepthDecoderLayerWeights& src = weights.layers[l];
    Music3DepthDeviceLayer& dst = out.layers[l];
    dst.input_layernorm =
        UploadBf16(backend, queue, {&src.input_layernorm}, {hidden}, release_host, &out.storage);
    dst.post_attention_layernorm = UploadBf16(backend, queue, {&src.post_attention_layernorm},
                                              {hidden}, release_host, &out.storage);
    // to_q | to_k | to_v, in THAT order — the order the forward's re-views read
    // them back in. Each is [H, H] row-major (out, in), so stacking them is a
    // concatenation along the OUTPUT axis and needs no permutation.
    dst.qkv = UploadBf16(backend, queue, {&src.to_q, &src.to_k, &src.to_v}, {3 * hidden, hidden},
                         release_host, &out.storage);
    dst.to_out =
        UploadBf16(backend, queue, {&src.to_out}, {hidden, hidden}, release_host, &out.storage);
    // gate_proj | up_proj, in THAT order, because `vt::SiluAndMul` computes
    // `silu(x[:, :D]) * x[:, D:]` and upstream computes `silu(gate) * up`.
    // Swapping these two blocks silently computes `silu(up) * gate`, which is a
    // finite and plausible tensor — the gate for it is a mutation, not an
    // assertion.
    // Held as HOST bf16 bytes: `layers::UnquantizedMlpGateUpMethod` consumes an
    // `OwnedTensor` and `ResidentWeight` does the one upload (or aliases, on a
    // CPU queue). Same [2I, H] gate-then-up order, same lossless narrowing.
    dst.gate_up = PackBf16Owned({&src.gate_proj, &src.up_proj}, {2 * inter, hidden},
                                release_host);
    dst.down_proj = UploadBf16(backend, queue, {&src.down_proj}, {hidden, inter}, release_host,
                               &out.storage);
  }
  return out;
}

std::vector<float> DepthDecoderAppendDevice(vt::Queue& queue, const DepthDecoderConfig& config,
                                            const Music3DepthDeviceWeights& weights,
                                            const std::vector<float>& inputs_embeds,
                                            int64_t batch, Music3DepthDeviceCache* cache) {
  const int64_t hidden = config.hidden_size;
  const int64_t inter = config.intermediate_size;
  const int64_t heads = config.num_attention_heads;
  const int64_t head_dim = config.head_dim();
  // The SAME refusals `DepthDecoderAppend` raises, in the same order and with
  // the same words where the words still apply — a caller that swaps arms must
  // not have to learn a second vocabulary of errors.
  if (cache == nullptr) Fail("MiniMax-Music3: the incremental depth decode needs a cache");
  if (batch <= 0) Fail("MiniMax-Music3: the incremental depth decode needs a positive batch");
  if (static_cast<int64_t>(inputs_embeds.size()) != batch * hidden) {
    Fail("MiniMax-Music3: the incremental depth decode got " +
         std::to_string(inputs_embeds.size()) + " input values, expected batch*hidden = " +
         std::to_string(batch * hidden));
  }
  if (!weights.staged()) {
    Fail("MiniMax-Music3: the device depth decode was given unstaged weights");
  }
  if (static_cast<int64_t>(weights.layers.size()) != config.num_layers) {
    Fail("MiniMax-Music3: the depth decoder has " + std::to_string(weights.layers.size()) +
         " layer weight sets, expected " + std::to_string(config.num_layers));
  }
  if (heads * head_dim != hidden) {
    Fail("MiniMax-Music3: hidden_size " + std::to_string(hidden) +
         " is not divisible by num_attention_heads " + std::to_string(heads));
  }

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  Dev dev{backend, queue};
  if (cache->positions == 0 && cache->keys.empty()) {
    // Sized ONCE at `max_position_embeddings` rather than grown per position:
    // the ceiling is 16 rows, so the whole history is 16 * H * 2 bytes per
    // (layer, row) — 1 MB at the shipped 4-layer batch-2 geometry — and a
    // re-allocation mid-sequence would invalidate the views the attention holds.
    cache->batch = batch;
    cache->hidden = hidden;
    cache->layers = config.num_layers;
    cache->capacity = config.max_position_embeddings;
    const size_t slots = static_cast<size_t>(config.num_layers * batch);
    cache->keys.resize(slots);
    cache->values.resize(slots);
    for (size_t s = 0; s < 2 * slots; ++s) {
      // POOLED, through `DBuf`, and NOT `backend.Alloc` — because this cache is
      // a local of `Music3DepthStage` and is therefore built and destroyed once
      // per FRAME. At the shipped 4-layer batch-2 geometry a raw pair would be
      // 16 `cudaMalloc` plus 16 `cudaFree` per frame, and `cudaFree`
      // synchronizes the device; `DBuf`'s own comment says the pool exists for
      // exactly that. `ReleaseShared` hands the block a carrier that returns it
      // to the pool it came from, which the hand-written deleter this replaced
      // could not name. The staged WEIGHTS stay on `backend.Alloc` on purpose:
      // they are allocated once and held for the process, and a pool block held
      // forever is a pool block withdrawn from the pool.
      DBuf block(dev, DType::kBF16, {cache->capacity, hidden});
      const Tensor t = block.t();
      cache->storage.push_back(block.ReleaseShared());
      (s < slots ? cache->keys : cache->values)[s % slots] = t;
    }
  }
  if (cache->batch != batch || cache->hidden != hidden || cache->layers != config.num_layers) {
    Fail("MiniMax-Music3: the depth cache holds batch " + std::to_string(cache->batch) +
         " / hidden " + std::to_string(cache->hidden) + " / layers " +
         std::to_string(cache->layers) + ", asked for " + std::to_string(batch) + " / " +
         std::to_string(hidden) + " / " + std::to_string(config.num_layers));
  }
  const int64_t position = cache->positions;
  if (position >= config.max_position_embeddings) {
    Fail("MiniMax-Music3: the depth decoder was given position " + std::to_string(position) +
         " but max_position_embeddings is " + std::to_string(config.max_position_embeddings) +
         "; pos_embedding has no row for the rest");
  }
  const int64_t seq = position + 1;

  // The upload is EXACT: `Store(..., kBFloat16)` already rounded every value the
  // host schedule hands across this boundary, so `F32ToBF16` loses nothing.
  std::vector<uint16_t> staged(inputs_embeds.size());
  for (size_t i = 0; i < inputs_embeds.size(); ++i) staged[i] = vt::F32ToBF16(inputs_embeds[i]);
  DBuf hidden_states(dev, DType::kBF16, {batch, hidden}, staged.data());

  // :138-139 — the SAME learned row `DepthDecoderForward` adds at this position,
  // broadcast over the batch by `vt::Add`'s rank-1 form.
  const Tensor pos_row = RowView(weights.pos_embedding, position, {hidden});
  vt::Add(queue, hidden_states.t(), hidden_states.t(), pos_row);

  DBuf normed(dev, DType::kBF16, {batch, hidden});
  DBuf qkv(dev, DType::kBF16, {batch, 3 * hidden});
  DBuf attended(dev, DType::kBF16, {batch, hidden});
  DBuf projected(dev, DType::kBF16, {batch, hidden});
  DBuf down(dev, DType::kBF16, {batch, hidden});

  // #1131's SECOND instrument. Every buffer this call makes resident reports its
  // own dtype into one mask, read back by the gate — because the ULP band above
  // cannot see a buffer that is too WIDE, and a review proved it by widening
  // `normed` to `kF32` and watching all 35 cases stay green. Read the buffers
  // rather than restating the constant, or the assertion tests this line instead
  // of the allocation.
  uint64_t resident = DtypeBit(hidden_states.t().dtype) | DtypeBit(normed.t().dtype) |
                      DtypeBit(qkv.t().dtype) | DtypeBit(attended.t().dtype) |
                      DtypeBit(projected.t().dtype) | DtypeBit(down.t().dtype) |
                      DtypeBit(cache->keys[0].dtype) | DtypeBit(cache->values[0].dtype);
  const vt::RmsNormArgs norm_args{1e-6f, /*gemma=*/false};
  const vt::AttentionCrossArgs attn_args{
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)))};
  const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(uint16_t);

  for (int64_t l = 0; l < config.num_layers; ++l) {
    const Music3DepthDeviceLayer& layer = weights.layers[static_cast<size_t>(l)];
    vt::RmsNorm(queue, normed.t(), hidden_states.t(), layer.input_layernorm, norm_args);
    // ONE sweep of the merged [3H, H] weight where the host arm sweeps to_q,
    // to_k and to_v separately. Row `b` of the result is [q(H) | k(H) | v(H)],
    // so each of the three is a CONTIGUOUS H-value slice and needs no copy to be
    // viewed — which is what `vt::AttentionCross`'s contiguity requirement asks
    // of the query.
    vt::MatmulBT(queue, qkv.t(), normed.t(), layer.qkv);
    auto* qkv_rows = static_cast<uint16_t*>(qkv.t().data);
    for (int64_t b = 0; b < batch; ++b) {
      const size_t slot = static_cast<size_t>(l * batch + b);
      Tensor& kc = cache->keys[slot];
      Tensor& vc = cache->values[slot];
      uint16_t* row = qkv_rows + b * 3 * hidden;
      // The K/V of this position, copied into the row's contiguous history. See
      // the file header for why the GEMM cannot write here directly, and why an
      // 8 KB copy is the right side of that trade.
      backend.Copy(queue, static_cast<uint16_t*>(kc.data) + position * hidden, row + hidden,
                   row_bytes);
      backend.Copy(queue, static_cast<uint16_t*>(vc.data) + position * hidden, row + 2 * hidden,
                   row_bytes);
      // ONE query row against `seq` cached positions that are ALL at or before
      // it, so a causal mask would mask nothing and the non-causal op computes
      // upstream's causal result exactly. See the header.
      Tensor query = MakeTensor(row, DType::kBF16, queue.device, {1, heads, head_dim});
      Tensor key = MakeTensor(kc.data, DType::kBF16, queue.device, {seq, heads, head_dim});
      Tensor value = MakeTensor(vc.data, DType::kBF16, queue.device, {seq, heads, head_dim});
      Tensor out_row = RowView(attended.t(), b, {1, heads, head_dim});
      vt::AttentionCross(queue, out_row, query, key, value, /*bias=*/nullptr, attn_args);
    }
    vt::MatmulBT(queue, projected.t(), attended.t(), layer.to_out);
    vt::Add(queue, hidden_states.t(), hidden_states.t(), projected.t());

    vt::RmsNorm(queue, normed.t(), hidden_states.t(), layer.post_attention_layernorm, norm_args);
    // ONE sweep of the merged [2I, H] weight through the SHARED merged-GEMM
    // seam, laid out gate-then-up so its `vt::SiluAndMul` computes upstream's
    // `silu(gate) * up` with no permutation and no half swap.
    const layers::UnquantizedMlpGateUpMethod mlp(&layer.gate_up, inter);
    DBuf activated = mlp.Apply(dev, normed.t());
    resident |= DtypeBit(activated.t().dtype) | DtypeBit(layer.qkv.dtype) |
                DtypeBit(layer.gate_up.dtype) | DtypeBit(layer.down_proj.dtype) |
                DtypeBit(layer.to_out.dtype);
    vt::MatmulBT(queue, down.t(), activated.t(), layer.down_proj);
    vt::Add(queue, hidden_states.t(), hidden_states.t(), down.t());
  }

  // :142
  DBuf normed_out(dev, DType::kBF16, {batch, hidden});
  vt::RmsNorm(queue, normed_out.t(), hidden_states.t(), weights.norm, norm_args);
  resident |= DtypeBit(normed_out.t().dtype) | DtypeBit(weights.norm.dtype) |
              DtypeBit(weights.pos_embedding.dtype);
  std::vector<uint16_t> host(static_cast<size_t>(batch * hidden));
  normed_out.Download(dev, host.data());

  cache->positions = seq;
  // OR-ed ONCE, at the end, so a call that threw part way through never reports
  // a partial residency as if it were the whole forward's.
  g_resident_dtypes.fetch_or(resident, std::memory_order_relaxed);
  g_forward_count.fetch_add(1, std::memory_order_relaxed);
  std::vector<float> out(host.size());
  for (size_t i = 0; i < host.size(); ++i) out[i] = vt::BF16ToF32(host[i]);
  return out;
}

}  // namespace music3
}  // namespace models
}  // namespace vllm
