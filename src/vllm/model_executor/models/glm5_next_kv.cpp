// GLM-5.3-Flash W5b-2c — the engine's multi-KV index, consumed (#2348).
// Header carries the argument; this file is the arithmetic.

#include "vllm/model_executor/models/glm5_next_kv.h"

#include <cstdint>
#include <cstring>  // std::memcpy, the CPU-queue arm of PageIo
#include <stdexcept>
#include <string>
#include <utility>  // std::pair, in the per-cache geometry loop
#include <vector>

#include "vllm/model_executor/models/glm5_next_attn.h"  // IndexerRoleFor
#include "vllm/v1/attention/backend.h"                  // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"        // GDNAttentionMetadata
#include "vt/backend.h"  // vt::Backend, vt::TryGetBackend — W9c-3b (#2480)
#include "vt/dtype.h"

namespace vllm::glm5_next {
namespace {

[[noreturn]] void Fail(const std::string& why) {
  throw std::runtime_error(
      "glm5_next KV binding: " + why +
      " See .agents/specs/glm5-next-flash.md and issue #2348.");
}

// The published name of a DSA layer's MLA latent and of its indexer side cache.
// ONE definition each, shared with `MakeGlm5NextKVCache` by being the same two
// strings — a second spelling here would be a mapping that resolves nothing and
// says the channel is empty.
std::string LatentName(int64_t l) {
  return "model.layers." + std::to_string(l) + ".self_attn.attn";
}
std::string IndexerName(int64_t l) {
  return "model.layers." + std::to_string(l) + ".self_attn.indexer.k_cache";
}
// W5b-2d (#2445): the KDA layer's recurrent state, which the channel CAN now
// address by name. `MakeGlm5NextKVCache` publishes it at
// `glm5_next_registry.cpp:311`, the same one-definition rule as the two above.
std::string RecurrentName(int64_t l) {
  return "model.layers." + std::to_string(l) + ".linear_attn";
}

// The first few names the channel actually carried, for a diagnostic that says
// what was there instead of only what was missing.
std::string ChannelSummary(const MultiKvCacheIndex& mk) {
  // W5b-2d (#2445): `size()` and `num_groups()` are counts over EVERY published
  // cache and not over `attn_kv` — `9e7621efc` widened both and left
  // `num_groups()`'s own comment saying otherwise (#2459). The wording here says
  // what the numbers are, because a diagnostic that mislabels its denominators
  // is how the 45-against-22 confusion reads as a topology error.
  std::string s = std::to_string(mk.size()) + " published cache(s) (" +
                  std::to_string(mk.num_paged()) + " paged, " +
                  std::to_string(mk.num_recurrent()) + " recurrent) from " +
                  std::to_string(mk.num_groups()) + " group(s), " +
                  std::to_string(mk.num_published_groups()) +
                  " with a block table; names begin: ";
  if (mk.layer_names == nullptr || mk.layer_names->empty()) return s + "(none)";
  const size_t n = mk.layer_names->size() < 4 ? mk.layer_names->size() : 4;
  for (size_t i = 0; i < n; ++i) {
    if (i != 0) s += ", ";
    s += "'" + (*mk.layer_names)[i] + "'";
  }
  if (n < mk.layer_names->size()) s += ", ...";
  return s;
}

// Read one element out of a HOST-side span of `dt` storage. `kAuto` fp8 means
// the dtype IS the storage type; anything else is refused before this is
// reached.
//
// W9c-3b (#2480): these four take a raw host pointer rather than the
// `PagedKvCache` or the `vt::Tensor`, because the buffer they address is no
// longer necessarily the engine's own. On a device queue it is the staging span
// `PageIo` below filled, and giving these functions the cache would let a
// future edit reach past the staging and dereference the engine's pointer again
// -- which is the whole defect.
float ReadSpanElem(const void* host, vt::DType dt, int64_t i) {
  if (dt == vt::DType::kF32) return static_cast<const float*>(host)[i];
  return vt::BF16ToF32(static_cast<const uint16_t*>(host)[i]);
}

void WriteSpanElem(void* host, vt::DType dt, int64_t i, float v) {
  if (dt == vt::DType::kF32) {
    static_cast<float*>(host)[i] = v;
    return;
  }
  static_cast<uint16_t*>(host)[i] = vt::F32ToBF16(v);
}

// ─── W9c-3b (#2480): THE ENGINE'S PAGES ARE NOT ALWAYS HOST MEMORY ───────────
//
// EVERY reader and writer in this file used to dereference `PagedKvCache::data`
// and `GdnStateCache`'s tensors directly, and on `--device cuda` those are
// `cudaMalloc` pointers. `GPUModelRunner::initialize_kv_cache` resolves
// `kv_cache_backend_resident_` from `!platform.is_cpu()`
// (`v1/worker/gpu/runner.cpp:1119-1122`) and `CacheBuffer` then allocates every
// paged cache and every recurrent state through `vt::Alloc(device, ...)`
// (`:575-593`); a `cudaMalloc` pointer is NOT host-dereferenceable, on GB10
// included, and `src/vt/cuda/cuda_backend.cu:354-391` holds CUDA to
// `DeviceMemoryIsHostAddressable() == false` with a `static_assert` naming #844
// and #1435 as the two SIGSEGVs that came from believing otherwise.
//
// So this was a host store into device memory, and it is why `--device cuda`
// died with SIGSEGV after a forward that had already completed: `LoadCaches`
// returns before touching a page on a fresh sequence, so the FIRST such access
// on step 1 is in `StoreCaches`, after the whole stack has run and before any
// token is emitted. Spec `## Owed` O49 carries the bisect and falsifies O46's
// mixed-residency reading of the same evidence.
//
// COPY THE SPAN, DO NOT ASK WHERE IT LIVES. `vt::Backend::Copy` is
// direction-agnostic on CUDA (`cudaMemcpyAsync` with `cudaMemcpyDefault`,
// `cuda_backend.cu:116-118`), so one path is correct whether the pages are
// device memory or host memory. That is deliberate and it is what keeps this
// file from re-deriving the runner's residency policy: `VT_DEVICE_KV_CACHE=0`
// moves the pages back to the host and needs no branch here.
//
// A CPU QUEUE KEEPS THE DIRECT PATH, byte-for-byte. There is no backend to ask
// and the host IS the device, so every `--device cpu` run copies exactly the
// bytes it copied before, through `std::memcpy` rather than through a backend.
class PageIo {
 public:
  explicit PageIo(vt::Queue& q) : q_(q) {
    if (q.device.type == vt::DeviceType::kCPU) return;
    b_ = vt::TryGetBackend(q.device);
    if (b_ == nullptr) {
      Fail("this step arrived on a non-CPU queue whose backend is not "
           "registered in this build, so the engine's KV pages cannot be "
           "staged to the host. Every reader and writer of a cache on this "
           "model's forward is a host loop, and the runner allocates those "
           "pages on the queue's device.");
    }
  }

  // `bytes` from `base + byte_off` into the host buffer `dst`.
  void Read(const void* base, size_t byte_off, size_t bytes, void* dst) {
    const uint8_t* src = static_cast<const uint8_t*>(base) + byte_off;
    if (b_ == nullptr) {
      std::memcpy(dst, src, bytes);
      return;
    }
    b_->Copy(q_, dst, src, bytes);
    // SYNCHRONISE ON EVERY SPAN, not once at the end. `Copy` is asynchronous on
    // this queue and the staging buffer is REUSED by the next span, so a
    // deferred wait would read one row while the driver is still writing the
    // one before it -- and would hand the driver a pageable source that the
    // next iteration has already overwritten.
    b_->Synchronize(q_);
  }

  // `bytes` from the host buffer `src` into `base + byte_off`.
  void Write(void* base, size_t byte_off, size_t bytes, const void* src) {
    uint8_t* dst = static_cast<uint8_t*>(base) + byte_off;
    if (b_ == nullptr) {
      std::memcpy(dst, src, bytes);
      return;
    }
    b_->Copy(q_, dst, src, bytes);
    b_->Synchronize(q_);
  }

 private:
  vt::Queue& q_;
  vt::Backend* b_ = nullptr;
};

// The host staging buffer for one span, grown to fit and never shrunk. One per
// `LoadCaches` / `StoreCaches` call, so the largest span a step touches is the
// whole cost.
uint8_t* Staging(std::vector<uint8_t>* buf, size_t bytes) {
  if (buf->size() < bytes) buf->resize(bytes);
  return buf->data();
}

// Every storage type this row can address. A quantized or fp8 page is refused
// by name rather than reinterpreted: `PagedKvCache::fp8_kind` says what a `kI8`
// byte MEANS, and a reader that ignored it would index a half-width page at
// full width (`qwen3_5.h:85-97`).
void RequirePlainFloatPage(const PagedKvCache& kv, const std::string& what) {
  if (kv.fp8_kind != vt::Fp8KVCacheDataType::kAuto) {
    Fail(what +
         " arrived as an fp8 page, and every buffer on this row's forward is "
         "host f32. Dequantizing it here would be an invention rather than a "
         "port; run this model without --kv-cache-dtype fp8.");
  }
  if (kv.dtype != vt::DType::kF32 && kv.dtype != vt::DType::kBF16) {
    Fail(what + " has storage dtype " + std::to_string(static_cast<int>(kv.dtype)) +
         ", and this forward can address only f32 and bf16 pages.");
  }
}

void RequirePlainFloatState(const vt::Tensor& t, const std::string& what) {
  if (t.data == nullptr) Fail(what + " has no allocated storage.");
  if (t.dtype != vt::DType::kF32 && t.dtype != vt::DType::kBF16) {
    Fail(what + " has storage dtype " +
         std::to_string(static_cast<int>(t.dtype)) +
         ", and this forward can address only f32 and bf16 states.");
  }
}

// One published attention cache, checked against the geometry `MakeGlm5NextKVCache`
// published it at. `expect_head` is the whole check that separates the MLA
// latent from a K+V pair at the value level: the pair's page is twice as long
// and its rows are half as wide, so a reader that took the pair layout would
// read the right number of finite floats from the wrong places.
int64_t ResolveAttnCache(const MultiKvCacheIndex& mk,
                         const std::vector<PagedKvCache>& attn_kv,
                         const std::string& name, int64_t expect_head,
                         const std::string& what, int32_t* group_out) {
  // W5b-2d (#2445). `Find` answers a FLAT index over EVERY published cache,
  // paged and recurrent together, and NOT a slot in `attn_kv` — see the header's
  // "THE FLAT INDEX IS NOT THE PAGED SLOT". Using it as one is what this wave
  // repaired, and the flat index is still what `group_ids` is parallel to, so
  // both values are kept and each is used against the vector it indexes.
  const int64_t flat = mk.Find(name);
  if (flat < 0) {
    Fail("the engine published no cache named '" + name + "' (" + what +
         "). The channel carried " + ChannelSummary(mk) +
         ". `MakeGlm5NextKVCache` publishes this name for every "
         "deepseek_sparse_attention layer, so a channel without it is a cache "
         "topology this forward cannot address.");
  }
  KvCachePayload kind = KvCachePayload::kPaged;
  int32_t slot = -1;
  if (!mk.PayloadAt(flat, &kind, &slot)) {
    Fail("'" + name + "' (" + what + ") is named at flat index " +
         std::to_string(flat) +
         " but the channel carries no payload locator for it. Without one there "
         "is no way to turn a published name into a slot in `attn_kv`, and the "
         "flat index is NOT that slot on any topology carrying a recurrent "
         "group. The channel carried " + ChannelSummary(mk) + ".");
  }
  // A RECURRENT PAYLOAD UNDER AN ATTENTION NAME. `attn_kv` and `gdn_state` are
  // two different containers, so a slot read against the wrong one is an
  // unrelated buffer with no shape error — the same wrong-answer-not-a-crash
  // shape the MLA-latent refusal below exists for.
  if (kind != KvCachePayload::kPaged) {
    Fail("'" + name + "' (" + what +
         ") was published as a RECURRENT cache, and this model keeps it in a "
         "PAGED one. `MakeGlm5NextKVCache` publishes it under an "
         "`MLAAttentionSpec` group; reading it out of `gdn_state` instead would "
         "return an unrelated buffer with no shape error.");
  }
  const int64_t i = slot;
  if (i < 0 || static_cast<size_t>(i) >= attn_kv.size()) {
    Fail("'" + name + "' is at flat index " + std::to_string(flat) +
         " and resolved to attn_kv slot " + std::to_string(i) + " but only " +
         std::to_string(attn_kv.size()) +
         " cache(s) arrived; the payload locator and attn_kv disagree.");
  }
  const PagedKvCache& kv = attn_kv[static_cast<size_t>(i)];
  RequirePlainFloatPage(kv, "'" + name + "' (" + what + ")");
  // THE MLA-LATENT-NOT-A-PAIR REFUSAL. `MLAAttentionSpec` fixes num_kv_heads at
  // 1 and stores ONE vector per token (`kv_cache_interface.h:27-29`,
  // `:282-288`). A `FullAttentionSpec` in this position would carry the model's
  // real head count and a per-head width, and this forward would still run.
  if (kv.num_kv_heads != 1) {
    Fail("'" + name + "' (" + what + ") published num_kv_heads " +
         std::to_string(kv.num_kv_heads) +
         ", but this cache is an MLA-style ONE-VECTOR-PER-TOKEN page and not a "
         "K+V pair: `MLAAttentionSpec` fixes num_kv_heads at 1 and stores no "
         "separate V (glm5_next_registry.cpp, 'GROUP 0 IS AN MLA LATENT'). "
         "Reading a pair layout here would produce finite, correctly shaped, "
         "WRONG values and a fluent wrong model.");
  }
  if (kv.head_size != expect_head) {
    Fail("'" + name + "' (" + what + ") published head_size " +
         std::to_string(kv.head_size) + ", but the resolved config makes it " +
         std::to_string(expect_head) +
         ". A row width that disagrees with the config addresses a different "
         "cache than the one the layer fills.");
  }
  if (kv.block_size <= 0 || kv.num_blocks <= 0 || kv.data == nullptr) {
    Fail("'" + name + "' (" + what + ") arrived with block_size " +
         std::to_string(kv.block_size) + ", num_blocks " +
         std::to_string(kv.num_blocks) + " and " +
         (kv.data == nullptr ? "no" : "a") + " data pointer.");
  }
  // BY THE FLAT INDEX, deliberately. `group_ids` is parallel to `layer_names`
  // and the other three locator vectors — one entry per PUBLISHED cache — and
  // not to `attn_kv` (`runner.cpp`, the by-name index pass). Reading it at the
  // paged slot would answer with a different cache's group on any topology
  // whose recurrent group is not published last.
  if (mk.group_ids == nullptr ||
      static_cast<size_t>(flat) >= mk.group_ids->size()) {
    Fail("'" + name + "' has no published group id; the name index is out of "
         "sync with the group index.");
  }
  *group_out = (*mk.group_ids)[static_cast<size_t>(flat)];
  return i;
}

// W5b-2d (#2445). The recurrent twin of `ResolveAttnCache`, and the reason it
// exists at all: before `9e7621efc` the by-name channel described `attn_kv`
// only, so this row derived a KDA layer's state slot POSITIONALLY and said so in
// the header. That channel now carries every published cache with a payload
// locator, which makes the derivation expressible and therefore no longer a
// stated limit.
int64_t ResolveRecurrentCache(const MultiKvCacheIndex& mk,
                              const std::vector<GdnStateCache>& gdn,
                              const std::string& name, int64_t layer) {
  const int64_t flat = mk.Find(name);
  if (flat < 0) {
    Fail("the engine published no cache named '" + name +
         "' (layer " + std::to_string(layer) +
         "'s KDA recurrent state). The channel carried " + ChannelSummary(mk) +
         ". `MakeGlm5NextKVCache` publishes this name for every "
         "linear_attention layer.");
  }
  KvCachePayload kind = KvCachePayload::kPaged;
  int32_t slot = -1;
  if (!mk.PayloadAt(flat, &kind, &slot)) {
    Fail("'" + name + "' is named at flat index " + std::to_string(flat) +
         " but the channel carries no payload locator for it, so its slot in "
         "`gdn_state` cannot be found. The channel carried " +
         ChannelSummary(mk) + ".");
  }
  // The mirror of the paged refusal. A KDA state read out of `attn_kv` would be
  // a paged buffer of the wrong rank, and the layer forward would zero-fill or
  // reshape it rather than notice.
  if (kind != KvCachePayload::kRecurrent) {
    Fail("'" + name +
         "' was published as a PAGED cache, and this model's KDA state is a "
         "RECURRENT one. `MakeGlm5NextKVCache` publishes it under a "
         "`MambaSpec` group; reading it out of `attn_kv` would return an "
         "unrelated buffer with no shape error.");
  }
  if (slot < 0 || static_cast<size_t>(slot) >= gdn.size()) {
    Fail("'" + name + "' is at flat index " + std::to_string(flat) +
         " and resolved to gdn_state slot " + std::to_string(slot) +
         " but only " + std::to_string(gdn.size()) +
         " recurrent state set(s) arrived; the payload locator and gdn_state "
         "disagree.");
  }
  return slot;
}

}  // namespace

int64_t PagedRowOffset(const std::vector<int32_t>& blocks, int64_t block_size,
                       int64_t head_size, int64_t pos) {
  const int64_t page = pos / block_size;
  if (page < 0 || static_cast<size_t>(page) >= blocks.size()) {
    Fail("logical position " + std::to_string(pos) + " needs page " +
         std::to_string(page) + " but the gathered block table holds " +
         std::to_string(blocks.size()) + " page(s).");
  }
  const int64_t block = blocks[static_cast<size_t>(page)];
  if (block < 0) {
    Fail("logical position " + std::to_string(pos) +
         " maps to block id " + std::to_string(block) +
         "; the block table row is not committed for this step.");
  }
  return (block * block_size + pos % block_size) * head_size;
}

KvBinding ResolveKvBinding(const Glm5NextParams& p,
                           const ModelForwardInput& input) {
  if (input.multi_kv == nullptr) {
    Fail("this forward was handed no multi-KV channel. `MakeGlm5NextKVCache` "
         "publishes three groups, which is a MULTI-CACHE topology, so the "
         "runner sets `ModelForwardInput::multi_kv` on every step. A null "
         "channel means the runner classified this model's topology as "
         "uniform, and the positional `attn_kv` convention cannot say which of "
         "a DSA layer's two caches an entry is.");
  }
  const MultiKvCacheIndex& mk = *input.multi_kv;
  const std::vector<PagedKvCache>& attn_kv = input.attn_kv;
  const std::vector<GdnStateCache>& gdn = input.gdn_state;
  const int64_t L = p.num_hidden_layers;
  if (static_cast<int64_t>(p.layer_types.size()) != L) {
    Fail("the resolved config declares " + std::to_string(L) +
         " layers but carries " + std::to_string(p.layer_types.size()) +
         " layer_types.");
  }

  KvBinding b;
  b.new_tokens = static_cast<int64_t>(input.token_ids.size());

  // ── how much history request 0 already has ───────────────────────────────
  const v1::CommonAttentionMetadata& am = input.attn_meta;
  if (am.num_computed_tokens_cpu.empty()) {
    Fail("the step carries no `num_computed_tokens_cpu`, so there is no way to "
         "say how many tokens of history the cache holds. A cached forward "
         "cannot guess it: reading zero would re-attend an empty prefix and "
         "emit fluent wrong text.");
  }
  b.cached_len = am.num_computed_tokens_cpu[0];
  if (b.cached_len < 0) {
    Fail("`num_computed_tokens_cpu[0]` is " + std::to_string(b.cached_len) + ".");
  }
  if (!am.seq_lens_cpu.empty() &&
      static_cast<int64_t>(am.seq_lens_cpu[0]) != b.cached_len + b.new_tokens) {
    Fail("the step says request 0 has " + std::to_string(b.cached_len) +
         " computed token(s) and carries " + std::to_string(b.new_tokens) +
         " new one(s), but its seq_len is " + std::to_string(am.seq_lens_cpu[0]) +
         ". The two must sum, or this forward would write history rows the "
         "engine does not believe exist.");
  }

  // ── the per-group block tables, by GROUP ID ──────────────────────────────
  const int published = mk.num_published_groups();
  b.group_blocks.assign(static_cast<size_t>(published < 0 ? 0 : published), {});

  // ── one binding per layer, EVERY attention cache found BY NAME ───────────
  b.layers.assign(static_cast<size_t>(L), LayerKvBinding{});
  int64_t kda_seen = 0;
  const int64_t latent_row = p.mla.kv_lora_rank + p.mla.qk_rope_head_dim;
  const int64_t indexer_row = 2 * p.indexer.head_dim + 1;
  for (int64_t l = 0; l < L; ++l) {
    LayerKvBinding& lb = b.layers[static_cast<size_t>(l)];
    lb.kind = p.layer_types[static_cast<size_t>(l)];
    if (lb.kind == Glm5NextLayerKind::kLinearAttention) {
      // W5b-2d (#2445): BY NAME, like every other cache on this model. The
      // ordinal `kda_seen` is still counted, and it is now a CHECK rather than
      // the answer: it is what the positional convention would have said, and a
      // disagreement with the channel's own slot is a topology this forward
      // must not guess at.
      lb.recurrent = ResolveRecurrentCache(mk, gdn, RecurrentName(l), l);
      if (lb.recurrent != kda_seen) {
        Fail("layer " + std::to_string(l) +
             "'s recurrent state is published at gdn_state slot " +
             std::to_string(lb.recurrent) + ", and it is this model's " +
             std::to_string(kda_seen) +
             "th linear_attention layer in ascending order. The runner "
             "allocates one state set per recurrent layer in layer order "
             "(`alloc_recurrent_layer_states`), so the two agree on every "
             "topology this tree builds; a disagreement means the channel and "
             "the config describe different models.");
      }
      ++kda_seen;
      continue;
    }
    lb.latent = ResolveAttnCache(mk, attn_kv, LatentName(l), latent_row,
                                 "the MLA latent, group 0", &lb.latent_group);
    lb.indexer =
        ResolveAttnCache(mk, attn_kv, IndexerName(l), indexer_row,
                         "the DSA indexer side cache, group 2", &lb.indexer_group);
    if (lb.latent_group == lb.indexer_group) {
      Fail("layer " + std::to_string(l) +
           " published its MLA latent and its indexer side cache under the "
           "SAME group id " + std::to_string(lb.latent_group) +
           ". They are two groups at two page sizes (" +
           std::to_string(latent_row) + " and " + std::to_string(indexer_row) +
           " wide); one group cannot hold both.");
    }
    lb.has_own_indexer = !IndexerRoleFor(p, l).skip_topk;
    if (b.block_size == 0) {
      b.block_size = attn_kv[static_cast<size_t>(lb.latent)].block_size;
    }
    // Every attention cache on this model shares one block size, because
    // `MakeGlm5NextKVCache` builds all three specs from the one `block_size`
    // the engine hands it. A disagreement means the pages this walks and the
    // pages the engine allocated are not the same pages.
    for (const auto& pr : {std::pair<int64_t, const char*>{lb.latent, "MLA latent"},
                           std::pair<int64_t, const char*>{lb.indexer, "indexer side cache"}}) {
      const int64_t got = attn_kv[static_cast<size_t>(pr.first)].block_size;
      if (got != b.block_size) {
        Fail(std::string("layer ") + std::to_string(l) + "'s " + pr.second +
             " published block_size " + std::to_string(got) +
             " against " + std::to_string(b.block_size) + " on this model's "
             "other caches.");
      }
    }
  }
  if (kda_seen != p.num_kda_layers()) {
    Fail("counted " + std::to_string(kda_seen) +
         " linear_attention layer(s) against `num_kda_layers()` = " +
         std::to_string(p.num_kda_layers()) + ".");
  }
  // The count STILL runs, and it is no longer the only check. Each state was
  // already resolved by name above; this catches the other direction — a
  // `gdn_state` carrying MORE sets than the config declares layers, which no
  // per-layer lookup can see because it never asks about the extras.
  if (static_cast<int64_t>(gdn.size()) != kda_seen) {
    Fail("the engine handed " + std::to_string(gdn.size()) +
         " recurrent state set(s) but the config declares " +
         std::to_string(kda_seen) +
         " linear_attention layer(s). Every one of those layers resolved its "
         "own state BY NAME through the channel's payload locator, so this "
         "counts the sets nothing claimed rather than the correspondence.");
  }
  if (b.block_size <= 0) {
    Fail("the config declares no deepseek_sparse_attention layer, so no paged "
         "block size reached this binding.");
  }

  // ── the gathered tables, and the pages this step needs ───────────────────
  const int64_t total = b.cached_len + b.new_tokens;
  const int64_t pages = (total + b.block_size - 1) / b.block_size;
  for (const LayerKvBinding& lb : b.layers) {
    if (lb.kind == Glm5NextLayerKind::kLinearAttention) continue;
    for (const int32_t g : {lb.latent_group, lb.indexer_group}) {
      if (g < 0 || static_cast<size_t>(g) >= b.group_blocks.size()) {
        Fail("group id " + std::to_string(g) +
             " is outside the " + std::to_string(b.group_blocks.size()) +
             " published group(s).");
      }
      std::vector<int32_t>& dst = b.group_blocks[static_cast<size_t>(g)];
      if (!dst.empty()) continue;
      int cols = 0;
      const std::vector<int32_t>* bt = mk.BlockTableForGroup(g, &cols);
      if (bt == nullptr || cols <= 0) {
        Fail("published group " + std::to_string(g) +
             " has no gathered block table this step, so there is no map from "
             "a logical position to the page holding it. Its cache was "
             "allocated and cannot be read.");
      }
      if (static_cast<int64_t>(cols) < pages) {
        Fail("published group " + std::to_string(g) + "'s block table has " +
             std::to_string(cols) + " column(s) but this step spans " +
             std::to_string(total) + " token(s) at block_size " +
             std::to_string(b.block_size) + ", which needs " +
             std::to_string(pages) + " page(s).");
      }
      if (bt->size() < static_cast<size_t>(pages)) {
        Fail("published group " + std::to_string(g) +
             "'s gathered block table holds " + std::to_string(bt->size()) +
             " entr(ies), short of the " + std::to_string(pages) +
             " page(s) request 0 needs.");
      }
      // Row 0 is request 0's; `num_reqs <= 1` is refused above this call.
      dst.assign(bt->begin(), bt->begin() + static_cast<std::ptrdiff_t>(pages));
    }
  }

  // ── CHECK THIS ROW'S PAGE ARITHMETIC AGAINST THE ENGINE'S OWN ────────────
  //
  // The runner computes `slot_mapping` for the TARGET attention group with its
  // own block-table walk, so it is an independent second opinion on the map
  // this file just built. When it is present it must agree, and a disagreement
  // is a refusal rather than a wrong token on every step from here on.
  if (!am.slot_mapping.empty() && b.new_tokens > 0) {
    for (const LayerKvBinding& lb : b.layers) {
      if (lb.kind == Glm5NextLayerKind::kLinearAttention) continue;
      const std::vector<int32_t>& blocks =
          b.group_blocks[static_cast<size_t>(lb.latent_group)];
      if (static_cast<int64_t>(am.slot_mapping.size()) < b.new_tokens) break;
      for (int64_t t = 0; t < b.new_tokens; ++t) {
        const int64_t pos = b.cached_len + t;
        const int64_t page = pos / b.block_size;
        const int64_t want =
            static_cast<int64_t>(blocks[static_cast<size_t>(page)]) *
                b.block_size + pos % b.block_size;
        if (am.slot_mapping[static_cast<size_t>(t)] != want) {
          Fail("this forward maps new token " + std::to_string(t) +
               " (logical position " + std::to_string(pos) + ") to KV slot " +
               std::to_string(want) + ", but the engine's own slot_mapping "
               "says " + std::to_string(am.slot_mapping[static_cast<size_t>(t)]) +
               ". The two walks of the block table disagree, so one of them "
               "addresses the wrong page.");
        }
      }
      break;  // one group is enough: the tables are the same width and shape.
    }
  }

  // ── the recurrent state slot, from the GDN metadata's REMAPPED index ─────
  if (kda_seen > 0) {
    const v1::GDNAttentionMetadata& gm = input.gdn_meta;
    if (!gm.non_spec_state_indices_tensor.has_value() ||
        gm.non_spec_state_indices_tensor->empty()) {
      Fail("the step carries " + std::to_string(kda_seen) +
           " linear_attention layer(s) but no "
           "`non_spec_state_indices_tensor`. That vector is the runner's "
           "COMPACT per-sequence state slot (remap_gdn_state_slots); the raw "
           "block id would index a [gdn_state_slots, ...] buffer with an "
           "attention block number.");
    }
    b.state_slot = (*gm.non_spec_state_indices_tensor)[0];
    if (b.state_slot < 0) {
      Fail("the recurrent state slot for request 0 is " +
           std::to_string(b.state_slot) + ".");
    }
    for (int64_t j = 0; j < kda_seen; ++j) {
      const GdnStateCache& gs = gdn[static_cast<size_t>(j)];
      if (gs.conv_state.data == nullptr || gs.ssm_state.data == nullptr) {
        Fail("recurrent state set " + std::to_string(j) +
             " has no allocated conv or ssm storage.");
      }
      RequirePlainFloatState(gs.conv_state,
                             "recurrent conv state " + std::to_string(j));
      RequirePlainFloatState(gs.ssm_state,
                             "recurrent ssm state " + std::to_string(j));
      const int64_t slots = gs.conv_state.shape[0];
      if (b.state_slot >= slots || b.state_slot >= gs.ssm_state.shape[0]) {
        Fail("the recurrent state slot for request 0 is " +
             std::to_string(b.state_slot) + " but state set " +
             std::to_string(j) + " holds " + std::to_string(slots) + " slot(s).");
      }
    }
  }
  return b;
}

void LoadCaches(const Glm5NextParams& p, const KvBinding& b,
                const ModelForwardInput& input, std::vector<LayerCache>* out) {
  const int64_t L = p.num_hidden_layers;
  out->assign(static_cast<size_t>(L), LayerCache{});
  // A FRESH SEQUENCE READS NOTHING. Every consumer treats an empty vector as
  // its own zero state, and a recurrent slot is reused across sequences without
  // being pre-zeroed, so reading one here would carry another request's history
  // into this one (`gdn_attn.h`, the caller obligation).
  if (b.cached_len <= 0) return;

  const std::vector<PagedKvCache>& attn_kv = input.attn_kv;
  const std::vector<GdnStateCache>& gdn = input.gdn_state;
  const int64_t latent_row = p.mla.kv_lora_rank + p.mla.qk_rope_head_dim;
  const int64_t indexer_row = 2 * p.indexer.head_dim + 1;
  const glm5_next_kda::Glm5NextKdaDims kd = KdaDimsFrom(p);
  const int64_t conv_elems = kd.conv_dim() * kd.conv_kernel_size;
  const int64_t rec_elems = kd.num_heads * kd.head_dim * kd.head_dim;

  // W9c-3b (#2480): every span below is STAGED through the backend rather than
  // dereferenced, because on a device queue these pointers are the runner's
  // `vt::Alloc` allocations. See `PageIo`.
  PageIo io(input.queue);
  std::vector<uint8_t> span;

  for (int64_t l = 0; l < L; ++l) {
    const LayerKvBinding& lb = b.layers[static_cast<size_t>(l)];
    LayerCache& c = (*out)[static_cast<size_t>(l)];
    if (lb.kind == Glm5NextLayerKind::kLinearAttention) {
      const GdnStateCache& gs = gdn[static_cast<size_t>(lb.recurrent)];
      // ONE entry, because `num_reqs <= 1` is refused above and the KDA
      // recurrence is single-sequence (`glm5_next_layer.h:191-196`).
      c.kda.assign(1, glm5_next_kda::Glm5NextKdaCache{});
      glm5_next_kda::Glm5NextKdaCache& kc = c.kda[0];
      kc.conv_state.resize(static_cast<size_t>(conv_elems));
      const size_t conv_elt = vt::SizeOf(gs.conv_state.dtype);
      const int64_t cbase = b.state_slot * conv_elems;
      uint8_t* cbuf = Staging(&span, static_cast<size_t>(conv_elems) * conv_elt);
      io.Read(gs.conv_state.data, static_cast<size_t>(cbase) * conv_elt,
              static_cast<size_t>(conv_elems) * conv_elt, cbuf);
      for (int64_t i = 0; i < conv_elems; ++i)
        kc.conv_state[static_cast<size_t>(i)] =
            ReadSpanElem(cbuf, gs.conv_state.dtype, i);
      kc.recurrent_state.resize(static_cast<size_t>(rec_elems));
      const size_t ssm_elt = vt::SizeOf(gs.ssm_state.dtype);
      const int64_t rbase = b.state_slot * rec_elems;
      uint8_t* rbuf = Staging(&span, static_cast<size_t>(rec_elems) * ssm_elt);
      io.Read(gs.ssm_state.data, static_cast<size_t>(rbase) * ssm_elt,
              static_cast<size_t>(rec_elems) * ssm_elt, rbuf);
      for (int64_t i = 0; i < rec_elems; ++i)
        kc.recurrent_state[static_cast<size_t>(i)] =
            ReadSpanElem(rbuf, gs.ssm_state.dtype, i);
      continue;
    }
    c.dsa.cached_len = b.cached_len;
    const PagedKvCache& lat = attn_kv[static_cast<size_t>(lb.latent)];
    const std::vector<int32_t>& lblocks =
        b.group_blocks[static_cast<size_t>(lb.latent_group)];
    const size_t lat_elt = vt::SizeOf(lat.dtype);
    c.dsa.k_pass.resize(static_cast<size_t>(b.cached_len * latent_row));
    for (int64_t t = 0; t < b.cached_len; ++t) {
      const int64_t off = PagedRowOffset(lblocks, b.block_size, latent_row, t);
      uint8_t* row = Staging(&span, static_cast<size_t>(latent_row) * lat_elt);
      io.Read(lat.data, static_cast<size_t>(off) * lat_elt,
              static_cast<size_t>(latent_row) * lat_elt, row);
      for (int64_t i = 0; i < latent_row; ++i)
        c.dsa.k_pass[static_cast<size_t>(t * latent_row + i)] =
            ReadSpanElem(row, lat.dtype, i);
    }
    // A `shared` layer never appends to its side cache and never validates it
    // (`glm5_next_attn.cpp:353-366`), so its stored rows do not exist and
    // hydrating `cached_len` of them would describe history nothing wrote.
    if (!lb.has_own_indexer) continue;
    const PagedKvCache& ix = attn_kv[static_cast<size_t>(lb.indexer)];
    const std::vector<int32_t>& iblocks =
        b.group_blocks[static_cast<size_t>(lb.indexer_group)];
    const size_t ix_elt = vt::SizeOf(ix.dtype);
    c.dsa.indexer_packed.resize(static_cast<size_t>(b.cached_len * indexer_row));
    for (int64_t t = 0; t < b.cached_len; ++t) {
      const int64_t off = PagedRowOffset(iblocks, b.block_size, indexer_row, t);
      uint8_t* row = Staging(&span, static_cast<size_t>(indexer_row) * ix_elt);
      io.Read(ix.data, static_cast<size_t>(off) * ix_elt,
              static_cast<size_t>(indexer_row) * ix_elt, row);
      for (int64_t i = 0; i < indexer_row; ++i)
        c.dsa.indexer_packed[static_cast<size_t>(t * indexer_row + i)] =
            ReadSpanElem(row, ix.dtype, i);
    }
  }
}

void StoreCaches(const Glm5NextParams& p, const KvBinding& b,
                 const std::vector<LayerCache>& caches,
                 const ModelForwardInput& input) {
  const int64_t L = p.num_hidden_layers;
  if (static_cast<int64_t>(caches.size()) != L) {
    Fail("the forward returned " + std::to_string(caches.size()) +
         " layer state(s) for a " + std::to_string(L) + "-layer model.");
  }
  std::vector<PagedKvCache>& attn_kv = input.attn_kv;
  std::vector<GdnStateCache>& gdn = input.gdn_state;
  const int64_t latent_row = p.mla.kv_lora_rank + p.mla.qk_rope_head_dim;
  const int64_t indexer_row = 2 * p.indexer.head_dim + 1;
  const int64_t total = b.cached_len + b.new_tokens;
  const glm5_next_kda::Glm5NextKdaDims kd = KdaDimsFrom(p);
  const int64_t conv_elems = kd.conv_dim() * kd.conv_kernel_size;
  const int64_t rec_elems = kd.num_heads * kd.head_dim * kd.head_dim;
  // W9c-3b (#2480): the row is BUILT in the staging buffer and then copied
  // through the backend. Writing straight into `lat.data` was a host store into
  // a `cudaMalloc` allocation on every `--device cuda` step, and it is where the
  // three SIGSEGV legs of O46 died. See `PageIo` and spec `## Owed` O49.
  PageIo io(input.queue);
  std::vector<uint8_t> span;

  for (int64_t l = 0; l < L; ++l) {
    const LayerKvBinding& lb = b.layers[static_cast<size_t>(l)];
    const LayerCache& c = caches[static_cast<size_t>(l)];
    if (lb.kind == Glm5NextLayerKind::kLinearAttention) {
      if (c.kda.size() != 1) {
        Fail("layer " + std::to_string(l) + " returned " +
             std::to_string(c.kda.size()) +
             " recurrent state(s) for one sequence.");
      }
      const glm5_next_kda::Glm5NextKdaCache& kc = c.kda[0];
      if (static_cast<int64_t>(kc.conv_state.size()) != conv_elems) {
        Fail("layer " + std::to_string(l) + " returned a conv state of " +
             std::to_string(kc.conv_state.size()) + " float(s) against the " +
             std::to_string(conv_elems) +
             " the published MambaSpec allocates. A width of "
             "`conv_kernel_dim - 1` is SILENTLY accepted by the conv "
             "(glm5_next_kda.cpp:256-261) and would change the cache geometry "
             "rather than throw.");
      }
      if (static_cast<int64_t>(kc.recurrent_state.size()) != rec_elems) {
        Fail("layer " + std::to_string(l) + " returned a recurrent state of " +
             std::to_string(kc.recurrent_state.size()) + " float(s) against " +
             std::to_string(rec_elems) + ".");
      }
      const GdnStateCache& gs = gdn[static_cast<size_t>(lb.recurrent)];
      const size_t conv_elt = vt::SizeOf(gs.conv_state.dtype);
      const int64_t cbase = b.state_slot * conv_elems;
      uint8_t* cbuf = Staging(&span, static_cast<size_t>(conv_elems) * conv_elt);
      for (int64_t i = 0; i < conv_elems; ++i)
        WriteSpanElem(cbuf, gs.conv_state.dtype, i,
                      kc.conv_state[static_cast<size_t>(i)]);
      io.Write(gs.conv_state.data, static_cast<size_t>(cbase) * conv_elt,
               static_cast<size_t>(conv_elems) * conv_elt, cbuf);
      const size_t ssm_elt = vt::SizeOf(gs.ssm_state.dtype);
      const int64_t rbase = b.state_slot * rec_elems;
      uint8_t* rbuf = Staging(&span, static_cast<size_t>(rec_elems) * ssm_elt);
      for (int64_t i = 0; i < rec_elems; ++i)
        WriteSpanElem(rbuf, gs.ssm_state.dtype, i,
                      kc.recurrent_state[static_cast<size_t>(i)]);
      io.Write(gs.ssm_state.data, static_cast<size_t>(rbase) * ssm_elt,
               static_cast<size_t>(rec_elems) * ssm_elt, rbuf);
      continue;
    }
    if (c.dsa.cached_len != total) {
      Fail("layer " + std::to_string(l) + " advanced its cache to " +
           std::to_string(c.dsa.cached_len) + " token(s) against the " +
           std::to_string(total) + " this step spans.");
    }
    if (static_cast<int64_t>(c.dsa.k_pass.size()) != total * latent_row) {
      Fail("layer " + std::to_string(l) + " returned " +
           std::to_string(c.dsa.k_pass.size()) + " latent float(s) against " +
           std::to_string(total * latent_row) + ".");
    }
    const PagedKvCache& lat = attn_kv[static_cast<size_t>(lb.latent)];
    const std::vector<int32_t>& lblocks =
        b.group_blocks[static_cast<size_t>(lb.latent_group)];
    const size_t lat_elt = vt::SizeOf(lat.dtype);
    for (int64_t t = b.cached_len; t < total; ++t) {
      const int64_t off = PagedRowOffset(lblocks, b.block_size, latent_row, t);
      uint8_t* row = Staging(&span, static_cast<size_t>(latent_row) * lat_elt);
      for (int64_t i = 0; i < latent_row; ++i)
        WriteSpanElem(row, lat.dtype, i,
                      c.dsa.k_pass[static_cast<size_t>(t * latent_row + i)]);
      io.Write(lat.data, static_cast<size_t>(off) * lat_elt,
               static_cast<size_t>(latent_row) * lat_elt, row);
    }
    if (!lb.has_own_indexer) {
      if (!c.dsa.indexer_packed.empty()) {
        Fail("layer " + std::to_string(l) +
             " is a `shared` DSA layer and must build no indexer rows, but it "
             "returned " + std::to_string(c.dsa.indexer_packed.size()) +
             " float(s).");
      }
      continue;
    }
    if (static_cast<int64_t>(c.dsa.indexer_packed.size()) !=
        total * indexer_row) {
      Fail("layer " + std::to_string(l) + " returned " +
           std::to_string(c.dsa.indexer_packed.size()) +
           " packed indexer float(s) against " +
           std::to_string(total * indexer_row) + ".");
    }
    const PagedKvCache& ix = attn_kv[static_cast<size_t>(lb.indexer)];
    const std::vector<int32_t>& iblocks =
        b.group_blocks[static_cast<size_t>(lb.indexer_group)];
    const size_t ix_elt = vt::SizeOf(ix.dtype);
    for (int64_t t = b.cached_len; t < total; ++t) {
      const int64_t off = PagedRowOffset(iblocks, b.block_size, indexer_row, t);
      uint8_t* row = Staging(&span, static_cast<size_t>(indexer_row) * ix_elt);
      for (int64_t i = 0; i < indexer_row; ++i)
        WriteSpanElem(row, ix.dtype, i,
                      c.dsa.indexer_packed[static_cast<size_t>(t * indexer_row + i)]);
      io.Write(ix.data, static_cast<size_t>(off) * ix_elt,
               static_cast<size_t>(indexer_row) * ix_elt, row);
    }
  }
}

}  // namespace vllm::glm5_next
