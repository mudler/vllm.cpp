// Ported from: vllm/v1/worker/gpu/model_runner.py @ e24d1b24
// (initialize_kv_cache / execute_model / sample_tokens / sample /
// postprocess_sampled — the T0 slice) + the decode-first reorder from
// vllm/v1/attention/backends/utils.py::reorder_batch_to_split_decodes_and_prefills.
// See include/vllm/v1/worker/gpu/runner.h for scope, the V1-algorithm / MRV2-
// contract composition, the four-way ordering contract, and the deferred paths.
#include "vllm/v1/worker/gpu/runner.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"  // SPEC-MTP I5d-pre: Qwen3_5MTPModel complete type for the owned draft member
#include "vllm/platforms/interface.h"  // GetPlatform(device.type) per-tensor memory-model seam
#include "vllm/v1/attention/backend.h"  // AttentionBackend / get_kv_cache_shape (M3)
#include "vllm/v1/attention/registry.h"  // SelectAttentionBackendName / MakeAttentionBackend (M3)
#include "vllm/v1/kv_cache_dtype.h"  // ResolveKvCacheDType (VT_KV_CACHE_F32 A/B)
#include "vllm/v1/kv_offload/lmcache/lmcache_connector.h"  // KV-EXTERNAL-CACHE worker store/load
#include "vllm/v1/sample/ops/bad_words.h"  // apply_allowed_token_ids (-inf mask)
#include "vllm/v1/worker/gpu/async_runner_flag.h"  // VT_ASYNC_RUNNER predicate
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"  // W6 (#1374) the graph-eligibility predicate
#include "vllm/v1/spec_decode/rejection_sampler.h"  // SPEC-REJECTION I3 verify half
#include "vllm/v1/worker/gpu/spec_decode/mtp/speculator.h"  // SPEC-MTP I5d MtpProposePrefill
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"  // SPEC-DFLASH D5 SampleDflashBlockDrafts
#include "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h"  // SPEC-DFLASH2 W3 Dflash2SelectCandidates
#include "vllm/v1/worker/gpu/spec_decode/dspark/speculator.h"  // SPEC-DSPARK W5 SampleDsparkBlockDrafts
#include "vllm/v1/spec_decode/ngram_proposer.h"  // SPEC-NGRAM D3 NgramPropose
#include "vt/backend.h"  // vt::Backend / GetBackend (VT_GPU_SAMPLE=0 download)
#include "vt/dtype.h"  // VT_CHECK
#include "vt/tensor.h"
#ifdef VLLM_CPP_CUDA
#include "vt/cuda/combine_tokens.h"  // W3 device combine/scatter (removes the sync)
#endif

namespace vllm::v1 {

// Logits-gather A/B toggle (perf). Default ON: the forward gathers the
// per-request last-token hidden rows BEFORE lm_head (prefill/mixed), so lm_head
// runs on num_reqs rows and only [num_reqs,vocab] is Downloaded. VT_LOGITS_GATHER=0
// restores the old full [num_actual_tokens,vocab] path (lm_head over all tokens,
// full D2H, host re-gather in sample_tokens). Both are token-for-token identical.
static bool LogitsGatherEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_LOGITS_GATHER");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// GPU-sampling A/B toggle (perf). Default ON: the forward keeps the [num_reqs,
// vocab] logits ON DEVICE and the sampler (argmax / temperature / top-k/top-p —
// all device kernels, mirroring vllm/v1/sample/sampler.py which never copies the
// full logits to host) reads them directly; only the sampled token ids (~num_reqs
// * 4 bytes) cross to host. VT_GPU_SAMPLE=0 restores the OLD path: Download the
// full [num_reqs, vocab] logits to host, then sample from the host copy — for the
// A/B on the same binary. Token-for-token identical (same on-device sampler
// kernels either way; only the logits' residence differs). Requires the
// gather-before-lm_head path (default); with VT_LOGITS_GATHER=0 the forward
// already returns host logits, so this toggle is a no-op there.
static bool GpuSampleEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GPU_SAMPLE");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// Async-scheduling device-input default (ENG-ASYNC-SCHED W3 runner leaf).
// VT_ASYNC_RUNNER gates the combine_sampled_and_draft_tokens device-input +
// async sampler-output path at runner construction, which advertises
// runner_supports_async() so LoadedEngine resolves an AsyncScheduler + mcb=2.
// DEFAULT ON since the 2026-07-17 flip (mirror vLLM's async-scheduling default,
// vllm/config/vllm.py:992-1044; DGX-proven token-exact); VT_ASYNC_RUNNER=0 is the
// runner-level rollback to the synchronous host path (byte-identical pre-flip
// streams). Read at CONSTRUCTION (not per-step), honoring the env value live at
// each runner build so the DGX A/B — and the CPU construction-matrix test — can
// flip it per engine. The parse is factored into the pure, CPU-unit-tested
// AsyncRunnerFlagIsOn predicate (async_runner_flag.h). Tests may also toggle the
// flag directly via set_async_input_combine.
static bool AsyncRunnerEnvDefault() {
  return AsyncRunnerFlagIsOn(std::getenv("VT_ASYNC_RUNNER"));
}

// Async input-combine reads the sampled token id back on the host between
// steps. Whether that read is VALID is a backend CAPABILITY, not a device name:
// ask the backend (vt::Backend::SupportsAsyncSampledTokenReadback, backend.h),
// which answers true for CPU (host and device memory are one allocation) and
// CUDA (the sampled id is device-mirrored, async_device_mirror()), and false for
// a DISCRETE non-CUDA GPU (e.g. ROCm gfx1201) whose sample_tokens_async leg
// host-dereferences a device Alloc — the root cause of the "!" tokens on the lab
// R9700 (2026-08-07). An absent backend (device not built into this binary)
// yields nullptr and therefore false, which also subsumes the old
// #ifdef VLLM_CPP_CUDA guard. Keeping the question on the backend is what stops
// this device-agnostic shared layer from naming a device (check-device-leakage).
static bool QueueSupportsAsyncInputCombine(const vt::Queue& queue) {
  const vt::Backend* backend = vt::TryGetBackend(queue.device.type);
  return backend != nullptr && backend->SupportsAsyncSampledTokenReadback();
}

// GDN step-geometry diagnostic (default OFF). When VT_GDN_DIAG_STEP_LOG=1, each
// execute_model step prints the request count and the live/free recurrent-state
// slot geometry to std::cerr. Read ONCE (never per-step getenv); bounded to the
// packed-GDN c16 diagnostic checkpoint (see .agents/specs/gdn-packed-decode.md).
static bool GdnDiagStepLogEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GDN_DIAG_STEP_LOG");
    return e != nullptr && e[0] == '1' && e[1] == '\0';
  }();
  return on;
}

// ─── Decode-first reorder (utils.py::reorder_batch_to_split_decodes_and_prefills)
bool reorder_batch_to_split_decodes_and_prefills(
    InputBatch& input_batch, const SchedulerOutput& scheduler_output,
    int decode_threshold) {
  const int num_reqs = input_batch.num_reqs();
  if (num_reqs <= 1) {
    return false;
  }

  // Per-request classification, in the current (dense) input_batch order.
  std::vector<int32_t> req_regions(static_cast<size_t>(num_reqs), 0);
  int num_decodes = 0, num_short = 0, num_long = 0, num_prefills = 0;
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = *input_batch.req_ids[static_cast<size_t>(i)];
    const int num_scheduled = scheduler_output.num_scheduled_tokens.at(req_id);
    const int num_computed =
        input_batch.num_computed_tokens_cpu[static_cast<size_t>(i)];
    const int num_prompt =
        input_batch.num_prompt_tokens[static_cast<size_t>(i)];
    const bool has_context = num_computed > 0;
    const bool is_below = num_scheduled <= decode_threshold;
    const bool done_prefilling = num_computed >= num_prompt;
    // Mutually exclusive (exactly one True). Desired order:
    // decode(0) -> short_extend(1) -> long_extend(2) -> prefill(3).
    if (!has_context) {
      req_regions[static_cast<size_t>(i)] = 3;  // pure prefill (first chunk)
      ++num_prefills;
    } else if (!is_below) {
      req_regions[static_cast<size_t>(i)] = 2;  // long_extend
      ++num_long;
    } else if (!done_prefilling) {
      req_regions[static_cast<size_t>(i)] = 1;  // short_extend
      ++num_short;
    } else {
      req_regions[static_cast<size_t>(i)] = 0;  // decode
      ++num_decodes;
    }
  }

  // target_regions = repeat([0,1,2,3], [nd, ns, nl, np]).
  std::vector<int32_t> target_regions(static_cast<size_t>(num_reqs), 0);
  {
    int off = 0;
    const int counts[4] = {num_decodes, num_short, num_long, num_prefills};
    for (int region = 0; region < 4; ++region) {
      for (int k = 0; k < counts[region]; ++k) {
        target_regions[static_cast<size_t>(off++)] =
            static_cast<int32_t>(region);
      }
    }
  }

  // orig_indices = ascending indices whose region != target (need to move).
  std::vector<int> orig_indices;
  for (int i = 0; i < num_reqs; ++i) {
    if (req_regions[static_cast<size_t>(i)] !=
        target_regions[static_cast<size_t>(i)]) {
      orig_indices.push_back(i);
    }
  }
  if (orig_indices.empty()) {
    return false;
  }

  // src_indices = orig_indices sorted (stable) by their region — the source
  // order the swap chains consume. Stable keeps ascending index within a region.
  std::vector<int> src_indices = orig_indices;
  std::stable_sort(src_indices.begin(), src_indices.end(), [&](int a, int b) {
    return req_regions[static_cast<size_t>(a)] <
           req_regions[static_cast<size_t>(b)];
  });

  // src_dest_map = {src: dst} in src_indices insertion order (dst = orig_indices
  // by position). Iterate in that insertion order, following each swap chain.
  std::unordered_map<int, int> dest;
  for (size_t k = 0; k < src_indices.size(); ++k) {
    dest[src_indices[k]] = orig_indices[k];
  }
  for (int src : src_indices) {
    int dst = dest[src];
    while (src != dst) {
      input_batch.swap_states(src, dst);
      const auto it = dest.find(dst);
      const int next_dst = (it != dest.end()) ? it->second : dst;
      dest[dst] = dst;  // mark dst done
      dst = next_dst;
    }
  }
  return true;
}

// ─── apply_grammar_bitmask ──────────────────────────────────────────────────
// Ported from: vllm/v1/structured_output/utils.py::apply_grammar_bitmask @
// e24d1b24. See runner.h for the compacted-vs-dense contract + the bit sense.
void apply_grammar_bitmask(
    const GrammarOutput& grammar_output,
    const std::vector<std::string>& req_ids,
    const std::map<std::string, std::vector<int32_t>>&
        scheduled_spec_decode_tokens,
    vt::Queue& queue, vt::Tensor& logits) {
  const TokenBitmask& bitmask = grammar_output.grammar_bitmask;
  // No structured request scheduled this step => nothing to mask (no-op).
  if (grammar_output.structured_output_request_ids.empty() ||
      bitmask.num_seqs == 0) {
    return;
  }
  const int64_t num_logits = logits.shape[0];
  const int64_t vocab = logits.shape[1];

  // struct_out_req_batch_indices: for each structured req, the logit row = its
  // dense batch index + the cumulative spec-token offset ahead of it
  // (utils.py:112-120). At T0 scheduled_spec_decode_tokens is empty, so the
  // offset stays 0 and logit_index == batch_index.
  const std::set<std::string> struct_out_req_ids(
      grammar_output.structured_output_request_ids.begin(),
      grammar_output.structured_output_request_ids.end());
  std::unordered_map<std::string, int> struct_out_req_batch_indices;
  {
    int cumulative_offset = 0;
    for (int batch_index = 0;
         batch_index < static_cast<int>(req_ids.size()); ++batch_index) {
      const std::string& req_id = req_ids[static_cast<size_t>(batch_index)];
      const int logit_index = batch_index + cumulative_offset;
      const auto sit = scheduled_spec_decode_tokens.find(req_id);
      if (sit != scheduled_spec_decode_tokens.end()) {
        cumulative_offset += static_cast<int>(sit->second.size());
      }
      if (struct_out_req_ids.count(req_id) != 0) {
        struct_out_req_batch_indices[req_id] = logit_index;
      }
    }
  }

  // Reorder the compacted bitmask onto the dense logits rows and unpack it into a
  // per-row EXCLUDE mask (utils.py:124-140 reorder + the bit unpack). Rows for
  // non-structured requests stay all-false (all tokens allowed). The
  // apply_allowed_token_ids op reads TRUE == "exclude this token" (-> -inf), so a
  // token is excluded exactly when its grammar bit is CLEAR (forbidden).
  std::vector<std::vector<uint8_t>> exclude(
      static_cast<size_t>(num_logits),
      std::vector<uint8_t>(static_cast<size_t>(vocab), 0));
  int cumulative_index = 0;
  for (const std::string& req_id :
       grammar_output.structured_output_request_ids) {
    int num_spec_tokens = 0;
    const auto sit = scheduled_spec_decode_tokens.find(req_id);
    if (sit != scheduled_spec_decode_tokens.end()) {
      num_spec_tokens = static_cast<int>(sit->second.size());
    }
    const auto bit = struct_out_req_batch_indices.find(req_id);
    if (bit != struct_out_req_batch_indices.end()) {
      const int logit_idx = bit->second;
      for (int i = 0; i < 1 + num_spec_tokens; ++i) {
        const int bitmask_row = cumulative_index + i;
        const int logit_row = logit_idx + i;
        if (logit_row < 0 || logit_row >= static_cast<int>(num_logits) ||
            bitmask_row < 0 || bitmask_row >= bitmask.num_seqs) {
          continue;
        }
        const int32_t* words =
            bitmask.data.data() +
            static_cast<size_t>(bitmask_row) *
                static_cast<size_t>(bitmask.num_words);
        std::vector<uint8_t>& row = exclude[static_cast<size_t>(logit_row)];
        for (int64_t t = 0; t < vocab; ++t) {
          const int32_t word = words[static_cast<size_t>(t >> 5)];
          const bool allowed =
              ((word >> static_cast<int>(t & 31)) & 1) != 0;
          if (!allowed) row[static_cast<size_t>(t)] = 1;
        }
      }
    }
    cumulative_index += 1 + num_spec_tokens;
  }

  // Set every forbidden token's logit to -inf (reuse the M1.7 sampler op; CPU +
  // CUDA counterparts both exist — CUDA path is dgx-pending like the sampler).
  apply_allowed_token_ids(queue, logits, exclude);
}

// ─── GPUModelRunner ─────────────────────────────────────────────────────────

namespace {
// Per-group block sizes from the KVCacheConfig groups, in group order (the
// InputBatch / MultiGroupBlockTable expect one entry per KV cache group).
std::vector<int> group_block_sizes(const KVCacheConfig& cfg) {
  std::vector<int> sizes;
  sizes.reserve(cfg.kv_cache_groups.size());
  for (const auto& g : cfg.kv_cache_groups) {
    sizes.push_back(g.kv_cache_spec->block_size);
  }
  return sizes;
}

// ── #810: per-layer KV membership from the GROUP the model published ────────
//
// Upstream never parses a per-layer config string to decide what a layer's
// cache is. `get_kv_cache_spec()` walks the instantiated modules
// (`gpu_model_runner.py:7785-7787`), groups layers by their SPEC OBJECT
// (`kv_cache_utils.py:1209-1211` `same_type_layers[layer_spec].append(name)`),
// and then fans metadata out BY LAYER NAME (`gpu_model_runner.py:2548-2549`).
// `KVCacheGroupSpec.layer_names` (`kv_cache_interface.py:938-947`, mirrored at
// `include/vllm/v1/kv_cache_interface.h:343`) is the durable signal; a config
// spelling such as `layer_types` is not, and a hybrid whose config does not
// speak Qwen3.5's dialect (NemotronH ships `layers_block_type`, no
// `layer_types`) has an EMPTY one.
//
// Our runner still indexes buffers by layer POSITION, so a published name has
// to be resolved back to an index. `LayerIndexOfName` does exactly that and
// nothing else: the integer of the `.layers.<N>.` segment of an upstream-style
// module path ("backbone.layers.5.mixer", "model.layers.12.self_attn").
//
// It deliberately returns nullopt for a PLACEHOLDER group name — "fa", "gdn",
// "mla", "kda", "fa_draft", the single-name convention every other registry
// uses today — because such a name carries no layer identity at all. That is
// what keeps this additive: a group that does not publish per-layer names falls
// back to the historical `config_.layer_types` predicate, byte for byte.
std::optional<int64_t> LayerIndexOfName(std::string_view name) {
  constexpr std::string_view kSep = ".layers.";
  const size_t at = name.find(kSep);
  if (at == std::string_view::npos) return std::nullopt;
  size_t i = at + kSep.size();
  const size_t start = i;
  int64_t value = 0;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
    value = value * 10 + (name[i] - '0');
    if (value > (1 << 20)) return std::nullopt;  // not a layer index
    ++i;
  }
  if (i == start) return std::nullopt;              // ".layers.mixer"
  if (i < name.size() && name[i] != '.') return std::nullopt;  // ".layers.5x"
  return value;
}

// The per-layer membership mask of one KV cache group, or nullopt when the
// group does not publish per-layer names.
//
// ALL-OR-NOTHING on purpose: a group is only read by name when EVERY one of its
// names resolves to a distinct in-range layer index. A partially-parseable
// group would silently drop the layers whose names did not parse, which is the
// silent-wrong-answer shape this whole row exists to remove.
std::optional<std::vector<bool>> GroupLayerMask(const KVCacheGroupSpec& group,
                                                int64_t num_layers) {
  if (group.layer_names.empty()) return std::nullopt;
  std::vector<bool> mask(static_cast<size_t>(num_layers), false);
  for (const std::string& name : group.layer_names) {
    const std::optional<int64_t> l = LayerIndexOfName(name);
    if (!l.has_value() || *l < 0 || *l >= num_layers) return std::nullopt;
    if (mask[static_cast<size_t>(*l)]) return std::nullopt;  // duplicate index
    mask[static_cast<size_t>(*l)] = true;
  }
  return mask;
}
}  // namespace

GPUModelRunner::GPUModelRunner(
    const HfConfig& config, LoadedModel& model,
    const KVCacheConfig& kv_cache_config, vt::Queue queue, int max_num_reqs,
    int max_model_len, int max_num_batched_tokens,
    std::optional<vllm::SpeculativeConfig> spec_config,
    std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model,
    std::vector<PagedKvCache> draft_kv)
    : config_(config),
      model_(&model),
      spec_config_(std::move(spec_config)),
      draft_model_(std::move(draft_model)),
      draft_attn_kv_(std::move(draft_kv)),
      queue_(queue),
      input_batch_(max_num_reqs, max_model_len, max_num_batched_tokens,
                   static_cast<int>(config.vocab_size),
                   group_block_sizes(kv_cache_config),
                   group_block_sizes(kv_cache_config)) {
  max_num_reqs_ = max_num_reqs;
  max_num_batched_tokens_ = max_num_batched_tokens;
  // SPEC-MTP I5e: the async input-combine splices the device-resident
  // last_sampled token over each decode row's input id with
  // num_new_sampled_tokens==1; it is NOT spec-aware and would overwrite the
  // draft token at a verify step's draft position with the committed token.
  // Speculative decode already forces SYNC scheduling and gets its drafts
  // spliced into token_ids_cpu by update_req_spec_token_ids + prepare_inputs,
  // so force the sync host input path here. Byte-identical for non-spec
  // (spec_config_ is nullopt there, so this is AsyncRunnerEnvDefault()).
  async_input_combine_ = AsyncRunnerEnvDefault() && !spec_config_.has_value() &&
                         QueueSupportsAsyncInputCombine(queue_);
  // ARCH-ONE-SURFACE ROW 6 (mirror gpu/model_runner.py:368-369): a POOLING
  // model's runner pools instead of sampling — build the PoolingRunner over
  // the model-owned Pooler. Null for every text arch (byte-identical).
  if (model_->registration().info.is_pooling_model &&
      model_->pooler() != nullptr) {
    pooling_runner_ = std::make_unique<vllm::PoolingRunner>(*model_->pooler());
  }
  initialize_kv_cache(kv_cache_config);
  ModelRegistry::Prepare(*model_, config_, queue_);
}

GPUModelRunner::GPUModelRunner(
    const HfConfig& config, std::unique_ptr<LoadedModel> owned_model,
    const KVCacheConfig& kv_cache_config, vt::Queue queue, int max_num_reqs,
    int max_model_len, int max_num_batched_tokens,
    std::optional<vllm::SpeculativeConfig> spec_config,
    std::unique_ptr<vllm::Qwen3_5MTPModel> draft_model,
    std::vector<PagedKvCache> draft_kv)
    : config_(config),
      owned_model_(std::move(owned_model)),
      model_(owned_model_.get()),
      spec_config_(std::move(spec_config)),
      draft_model_(std::move(draft_model)),
      draft_attn_kv_(std::move(draft_kv)),
      queue_(queue),
      input_batch_(max_num_reqs, max_model_len, max_num_batched_tokens,
                   static_cast<int>(config.vocab_size),
                   group_block_sizes(kv_cache_config),
                   group_block_sizes(kv_cache_config)) {
  max_num_reqs_ = max_num_reqs;
  max_num_batched_tokens_ = max_num_batched_tokens;
  // SPEC-MTP I5e: the async input-combine splices the device-resident
  // last_sampled token over each decode row's input id with
  // num_new_sampled_tokens==1; it is NOT spec-aware and would overwrite the
  // draft token at a verify step's draft position with the committed token.
  // Speculative decode already forces SYNC scheduling and gets its drafts
  // spliced into token_ids_cpu by update_req_spec_token_ids + prepare_inputs,
  // so force the sync host input path here. Byte-identical for non-spec
  // (spec_config_ is nullopt there, so this is AsyncRunnerEnvDefault()).
  async_input_combine_ = AsyncRunnerEnvDefault() && !spec_config_.has_value() &&
                         QueueSupportsAsyncInputCombine(queue_);
  // ARCH-ONE-SURFACE ROW 6 (mirror gpu/model_runner.py:368-369): a POOLING
  // model's runner pools instead of sampling — build the PoolingRunner over
  // the model-owned Pooler. Null for every text arch (byte-identical).
  if (model_->registration().info.is_pooling_model &&
      model_->pooler() != nullptr) {
    pooling_runner_ = std::make_unique<vllm::PoolingRunner>(*model_->pooler());
  }
  initialize_kv_cache(kv_cache_config);
  ModelRegistry::Prepare(*model_, config_, queue_);
}

GPUModelRunner::GPUModelRunner(const HfConfig& config,
                               const Qwen3_5MoeWeights& weights,
                               const KVCacheConfig& kv_cache_config,
                               vt::Queue queue, int max_num_reqs,
                               int max_model_len, int max_num_batched_tokens)
    : GPUModelRunner(config, BorrowQwen3_5MoeLoadedModel(weights),
                     kv_cache_config, queue, max_num_reqs, max_model_len,
                     max_num_batched_tokens) {}

GPUModelRunner::GPUModelRunner(const HfConfig& config,
                               const Qwen3_5DenseWeights& weights,
                               const KVCacheConfig& kv_cache_config,
                               vt::Queue queue, int max_num_reqs,
                               int max_model_len, int max_num_batched_tokens)
    : GPUModelRunner(config, BorrowQwen3_5DenseLoadedModel(weights),
                     kv_cache_config, queue, max_num_reqs, max_model_len,
                     max_num_batched_tokens) {}

GPUModelRunner::CacheBuffer::CacheBuffer(vt::Device device, vt::Queue& queue,
                                         size_t bytes,
                                         bool backend_resident)
    : device_(device), backend_resident_(backend_resident) {
  if (!backend_resident_) {
    host_data_.assign(bytes, uint8_t{0});
    return;
  }

  backend_data_ = vt::Alloc(device_, std::max<size_t>(bytes, 1));
  try {
    if (bytes != 0) {
      vt::GetBackend(device_.type).Memset(queue, backend_data_, 0, bytes);
    }
  } catch (...) {
    vt::Free(device_, backend_data_);
    backend_data_ = nullptr;
    throw;
  }
}

GPUModelRunner::CacheBuffer::~CacheBuffer() {
  if (backend_data_ != nullptr) {
    vt::Free(device_, backend_data_);
  }
}

void GPUModelRunner::initialize_kv_cache(const KVCacheConfig& kv_cache_config) {
  num_blocks_ = kv_cache_config.num_blocks;
  // ENGINE-LEVEL ATTENTION-BACKEND SELECTION (M3, issue #41) happens INSIDE the
  // full-attention region below, never here: a pure-GDN / pooling model that
  // caches no paged KV must not pay selection, and a platform whose priority
  // list yields no dense backend must fail loudly only for models that actually
  // need one (the empty-list loud-throw design, rocm.cpp W0). The resolution
  // block lives in the full-attn region; the per-group validation in the view
  // loop below.
  // Resolved LAZILY per group kind, on first use in the view loop below: a
  // pure-MLA model never resolves (or validates) a dense backend, and a dense
  // model never resolves MLA. `dense_backend` throws loudly if the platform has
  // no registered dense backend (the empty-list loud-throw design); `mla_backend`
  // stays empty on a device with no registered MLA backend (op-driven MLA).
  std::string dense_backend;
  std::string mla_backend;
  bool dense_backend_resolved = false;
  bool mla_backend_resolved = false;
  // GDN mamba-state slots = max concurrent sequences (one recurrent state per
  // sequence), decoupled from the attention num_blocks. Guard against a 0 (e.g.
  // a test path that skipped the ctor arg) by falling back to num_blocks.
  //
  // SPEC-MTP I5d: under speculation each sequence needs num_spec+1 CONSECUTIVE
  // GDN state slots (the k+1 draft-timestep snapshots the recurrent rollback
  // selects among, spec §3). Size the compact pool max_num_reqs*(num_spec+1) and
  // let remap_gdn_state_slots hand each sequence a base of num_spec+1 slots.
  const int spec_cols = spec_on() ? num_spec() + 1 : 1;
  const int64_t base_slots = max_num_reqs_ > 0 ? max_num_reqs_ : num_blocks_;
  gdn_state_slots_ = base_slots * spec_cols;
  gdn_slot_of_req_.clear();
  gdn_free_slots_.clear();
  gdn_free_slots_.reserve(static_cast<size_t>(base_slots));
  // The free list holds each sequence's BASE state slot. Under speculation a
  // base owns spec_cols (== num_spec+1) consecutive slots, so bases step by
  // spec_cols; without speculation spec_cols==1 and this is the pre-spec pool.
  for (int64_t b = base_slots - 1; b >= 0; --b)
    gdn_free_slots_.push_back(static_cast<int32_t>(b * spec_cols));

  // Resolve the full-attn + GDN(mamba) KV group ids (T0 gate models: exactly one
  // of each). The block-table group order == kv_cache_groups order.
  //
  // SPEC-MTP I5d-pre LATENT-BUG FIX: the previous loop assigned
  // `full_attn_group_id_` on EVERY full-attn/MLA group, so it kept the LAST one.
  // With num_spec>0 the MTP draft adds a THIRD group (`fa_draft`, appended at
  // index num_hidden_layers — qwen3_5_common.cpp:99-104), which the old code
  // would then wrongly select as the TARGET attention group. Select the TARGET
  // deterministically instead: the FIRST (index 0) non-eagle full-attn/MLA group,
  // so a later-appended draft group can never displace it. With num_spec==0
  // (every run today) there is exactly ONE full-attn group, so this is
  // byte-identical to the old behavior — the first and only group is chosen.
  for (int g = 0; g < static_cast<int>(kv_cache_config.kv_cache_groups.size());
       ++g) {
    const auto& group = kv_cache_config.kv_cache_groups[static_cast<size_t>(g)];
    const KVCacheSpecKind kind = group.kv_cache_spec->kind();
    // MLA campaign W7: an `MLAAttentionSpec` group IS the model's attention
    // group. Upstream registers MLA against the ordinary FullAttentionManager
    // (`vllm/v1/core/single_type_kv_cache_manager.py:1539`), so block table,
    // prefix caching and eviction are identical and the ONLY difference is the
    // page cost/shape — which the spec-driven allocation below already reads
    // off the spec. Additive condition: no existing arch changes group.
    if (kind == KVCacheSpecKind::kFullAttention ||
        kind == KVCacheSpecKind::kMlaAttention) {
      // Skip a draft (eagle) group and keep the FIRST match as the target.
      if (!group.is_eagle_group && full_attn_group_id_ < 0) {
        full_attn_group_id_ = g;
      }
    } else if (kind == KVCacheSpecKind::kMamba) {
      gdn_group_id_ = g;
    }
  }

  // Allocate one PagedKvCache per full-attn layer and one GdnStateCache per GDN
  // layer, in LAYER ORDER (matches Qwen3_5Model::Forward's per-layer fa_idx /
  // gdn_idx indexing). As in upstream, MambaSpec is the source of truth for the
  // recurrent tensors' order, shapes, dtypes, and page bytes.
  //
  // #810: SPEC-DRIVEN, exactly like the attention half below. This block used
  // to rebuild the geometry from `config_.linear_num_key_heads` /
  // `linear_conv_kernel_dim` and then REFUSE when the model's own published
  // spec disagreed with that reconstruction — so the comment above stated a
  // polarity the code did not have, and every hybrid whose config does not
  // speak Qwen3.5's `linear_*` dialect (NemotronH ships `mamba_num_heads`,
  // `mamba_head_dim`, `ssm_state_size`, `conv_kernel`) was refused by Qwen3.5's
  // name for shapes it never claimed.
  //
  // Upstream cannot express that cross-check at all. Its runner allocates raw
  // BYTES (`gpu_model_runner.py:7311-7313` `torch.zeros(..., dtype=torch.int8)`)
  // and hands the Mamba layer an untyped page view (`:7429-7441`); the unpack
  // into typed conv/SSM views happens in the LAYER
  // (`mamba/abstract.py:38-43`: `nbytes = prod(shape) * get_dtype_size(dtype)`).
  // The runner never holds a conv shape, so it has nothing to compare. That is
  // why the check is DELETED rather than widened or put behind a per-arch
  // switch: it was a SECOND, independent derivation of a number the model had
  // already published, and a second derivation is the thing that can disagree.
  // It is also why `MambaSpec` (`kv_cache_interface.py:690-703`, mirrored at
  // `include/vllm/v1/kv_cache_interface.h:302-323`) deliberately carries no
  // num_heads/head_dim/conv_dim: the geometry IS `shapes`.
  //
  // SPEC-MTP I5d is now free: the conv row widened to (K-1)+num_spec under
  // speculation is simply what `MakeQwen3_5KVCacheSpec` publishes
  // (mamba_utils.py:226), so reading the spec picks it up with no spec_on()
  // branch here at all.
  std::vector<int64_t> conv_state_shape;  // per SLOT, spec order [0]
  std::vector<int64_t> ssm_state_shape;   // per SLOT, spec order [1]
  int64_t conv_row_elems = 0;
  int64_t ssm_row_elems = 0;

  const MambaSpec* mamba_spec = nullptr;
  if (gdn_group_id_ >= 0) {
    mamba_spec = dynamic_cast<const MambaSpec*>(
        kv_cache_config.kv_cache_groups[static_cast<size_t>(gdn_group_id_)]
            .kv_cache_spec.get());
    VT_CHECK(mamba_spec != nullptr,
             "runner: recurrent cache group must carry a MambaSpec");
    VT_CHECK(mamba_spec->shapes.size() == 2 &&
                 mamba_spec->dtypes.size() == 2,
             "runner: recurrent MambaSpec must contain conv then temporal state");
    conv_state_shape = mamba_spec->shapes[0];
    ssm_state_shape = mamba_spec->shapes[1];
    // The GdnStateCache view prepends the SLOT dim, so a state shape may carry
    // at most kMaxRank-1 dims. Refuse rather than silently truncate.
    VT_CHECK(!conv_state_shape.empty() && !ssm_state_shape.empty() &&
                 conv_state_shape.size() < static_cast<size_t>(vt::kMaxRank) &&
                 ssm_state_shape.size() < static_cast<size_t>(vt::kMaxRank),
             "runner: MambaSpec state shapes must be 1..kMaxRank-1 dims");
    const auto row_elems = [](const std::vector<int64_t>& shape) {
      int64_t n = 1;
      for (int64_t d : shape) n *= d;
      return n;
    };
    conv_row_elems = row_elems(conv_state_shape);
    ssm_row_elems = row_elems(ssm_state_shape);
    VT_CHECK(conv_row_elems > 0 && ssm_row_elems > 0,
             "runner: MambaSpec state shapes must be positive");
    gdn_conv_cache_dtype_ = mamba_spec->dtypes[0];
    gdn_ssm_cache_dtype_ = mamba_spec->dtypes[1];
    const auto supported_state_dtype = [](vt::DType dtype) {
      return dtype == vt::DType::kF16 || dtype == vt::DType::kBF16 ||
             dtype == vt::DType::kF32;
    };
    VT_CHECK(supported_state_dtype(gdn_conv_cache_dtype_) &&
                 supported_state_dtype(gdn_ssm_cache_dtype_),
             "runner: recurrent MambaSpec state dtypes must be floating");
    // The per-slot byte cost the allocator will use IS the spec's page size —
    // upstream's `MambaSpec.page_size_bytes` is the sum of `prod(shape) *
    // dtype_size` over the state tensors (`kv_cache_interface.py:699-703`).
    // Assert the identity rather than re-deriving it anywhere else.
    VT_CHECK(conv_row_elems *
                     static_cast<int64_t>(vt::SizeOf(gdn_conv_cache_dtype_)) +
                 ssm_row_elems *
                     static_cast<int64_t>(vt::SizeOf(gdn_ssm_cache_dtype_)) ==
                 mamba_spec->page_size_bytes(),
             "runner: MambaSpec page_size_bytes disagrees with its own "
             "shapes and dtypes");
  }

  // SPEC-DRIVEN attention-cache sizing and layout (MLA campaign W1).
  //
  // Upstream never reconstructs the KV shape from the HF config: every layer's
  // cache bytes come from its SPEC (`vllm/v1/kv_cache_interface.py:380-398`
  // `real_page_size_bytes` -> `page_size_bytes`), and the tensor shape comes
  // from the BACKEND (`get_kv_cache_shape`). That abstraction is exactly why
  // `vllm/v1/worker/gpu_model_runner.py` contains **no `use_mla` branch at all**
  // (only an import at :58, an isinstance at :977 and comments at :1085/:7133)
  // even though MLA's cache is 1-head, 576-wide and has NO separate V — MLA is
  // registered against the ordinary `FullAttentionManager`
  // (`vllm/v1/core/single_type_kv_cache_manager.py:1539`), so block table,
  // prefix caching and eviction are untouched and the whole cost lands in the
  // ALLOCATOR and the ops.
  //
  // We previously hardcoded `num_blocks * 2 * block * Hkv * Dh` with Hkv/Dh
  // read from `config_` — the factor 2 (K+V) and the HF-config reconstruction
  // are the two things MLA cannot express. Now: bytes come from
  // `spec->page_size_bytes()` and the cache VIEW comes from the spec's own
  // fields. Behaviour-preserving by construction for every existing model —
  // `FullAttentionSpec::real_page_size_bytes` is
  // `block * num_kv_heads * (head_size + head_size_v) * es`, which with
  // `head_size_v == head_size` is byte-for-byte the old expression.
  int64_t fa_block_size = 0;
  int64_t Hkv = 0;
  int64_t Dh = 0;
  int64_t fa_page_bytes = 0;
  vt::DType kv_dtype = ResolveKvCacheDType();
  if (full_attn_group_id_ >= 0) {
    const KVCacheSpec* fa_spec =
        kv_cache_config.kv_cache_groups[static_cast<size_t>(full_attn_group_id_)]
            .kv_cache_spec.get();
    const auto* attn_spec = dynamic_cast<const AttentionSpec*>(fa_spec);
    VT_CHECK(attn_spec != nullptr,
             "runner: full-attention cache group must carry an AttentionSpec");
    fa_block_size = attn_spec->block_size;
    Hkv = attn_spec->num_kv_heads;
    Dh = attn_spec->head_size;
    kv_dtype = attn_spec->dtype;
    fa_page_bytes = attn_spec->page_size_bytes();
    // The PagedKvCache view carries ONE head_size, so an asymmetric-V full
    // attention layer cannot be viewed by it. MLA's own view (a later W) is a
    // sibling struct; until then, refuse rather than mis-view.
    if (const auto* full_spec = dynamic_cast<const FullAttentionSpec*>(fa_spec)) {
      VT_CHECK(full_spec->head_size_v == full_spec->head_size,
               "runner: asymmetric head_size_v is not expressible in the "
               "PagedKvCache view");
    }
    VT_CHECK(fa_page_bytes > 0,
             "runner: full-attention spec reported a non-positive page size");

    // ENGINE-LEVEL ATTENTION-BACKEND SELECTION (M3, issue #41) is the first
    // runtime call of the selection seam, and it happens PER GROUP in the view
    // loop below — lazily per kind, inside the full-attn region, never for a
    // pure-GDN / pooling model (which has no full-attn groups and therefore no
    // paged KV to validate).
    //
    // Dense: LOUD. A model with dense full-attention groups needs a dense
    // backend; a platform whose priority list yields none (how Vulkan and ROCm
    // started) fails at init instead of silently running unlabelled. On ROCm
    // this resolves "ROCM_ATTN" (backend.cpp, M3); on CPU "CPU_ATTN"
    // (cpu_attn.cpp, issue #1371 — upstream's own CPU answer at cpu.py:75-87);
    // on CUDA/Metal/Vulkan "FLASH_ATTN". All four names report the same NHD KV
    // layout, which is the layout every one of those device kernels reads — the
    // name changed on CPU, the geometry validated below did not.
    // Mirrors upstream resolving get_attn_backend_cls per attention layer
    // (gpu_model_runner.py:6994-7099); we group by KV-cache kind because this
    // engine allocates exactly one layout per kind.
    //
    // MLA: TOLERANT. The engine executes MLA through TritonMLAImpl on a fused
    // 3-dim cache regardless of the registry (deepseek_v2.cpp:576-578), so on a
    // device with no registered MLA backend (CPU, ROCm today) the name stays
    // empty and the group keeps running op-driven — a loud throw would regress
    // working MLA paths. On CUDA this resolves "TRITON_MLA", whose
    // get_kv_cache_shape is exactly the fused view the engine allocates.

    // Positive signal that the SPEC (not the HF config) drove this allocation:
    // opt-in, one line, never on the hot path.
    if (const char* dbg = std::getenv("VT_KV_ALLOC_LOG");
        dbg != nullptr && dbg[0] == '1') {
      std::fprintf(stderr,
                   "[kv-alloc] source=spec kind=%d block_size=%lld "
                   "num_kv_heads=%lld head_size=%lld dtype=%d "
                   "page_size_bytes=%lld num_blocks=%lld\n",
                   static_cast<int>(fa_spec->kind()),
                   static_cast<long long>(fa_block_size),
                   static_cast<long long>(Hkv), static_cast<long long>(Dh),
                   static_cast<int>(kv_dtype),
                   static_cast<long long>(fa_page_bytes),
                   static_cast<long long>(num_blocks_));
    }
  }
  // Recorded for the gates: the exact per-block byte cost the allocator used,
  // sourced from the spec. `fa_page_size_bytes() > 0` is the runtime proof that
  // the spec-driven path RAN (a compiled-but-unexercised path leaves it 0).
  fa_page_size_bytes_ = fa_page_bytes;

  const vt::Device dev = queue_.device;
  const char* device_cache_env = std::getenv("VT_DEVICE_KV_CACHE");
  // W0b-1 / work row M3a: this read `is_cuda()`, which is the SAME defect the
  // `dense_attn_block.h` ResidentWeight fix corrected — "not NVIDIA" was being
  // used to mean "no device memory". On kMETAL (and kVULKAN, kXPU) the KV cache
  // fell into `host_data_`, and the first device kernel to touch it —
  // vt::ReshapeAndCache — was handed a HOST pointer. On Metal that surfaces as
  // "k_cache points outside every Metal allocation"; on a discrete-memory
  // backend it would be a silent wrong answer or a fault.
  //
  // Device residency is a property of HAVING a device, not of the vendor, so the
  // predicate is `!is_cpu()`. kCPU keeps the host vector (it has no device pool
  // and the host IS the device) and kCUDA behaviour is bit-identical — the dgx
  // regression set is the evidence, not this comment.
  kv_cache_backend_resident_ =
      !vllm::platforms::GetPlatform(dev.type).is_cpu() &&
      (device_cache_env == nullptr || device_cache_env[0] != '0');
  full_attn_buf_.clear();
  ssm_buf_.clear();
  conv_buf_.clear();
  // A full-attention-only model (e.g. dense Qwen3ForCausalLM) has NO
  // linear-attention (GDN/Mamba) KV group and an EMPTY layer_types — indexing
  // layer_types[l] would be out of bounds. Drive "is this layer GDN?" off the
  // resolved KV-group structure (gdn_group_id_ >= 0 ⇔ the model has a mamba
  // group), then the per-layer layer_types tag. This is model-shape-agnostic:
  // the hybrid gate models keep a GDN group, so their path is byte-identical.
  const bool has_mamba_group = gdn_group_id_ >= 0;
  // #810: PER-LAYER MEMBERSHIP FROM THE PUBLISHED GROUP, when the model
  // publishes one. `KVCacheGroupSpec::layer_names` is what upstream keys
  // per-layer KV on end to end (see LayerIndexOfName above); the runner keyed
  // on `config_.layer_types[l] == "linear_attention"` instead, which is a
  // Qwen3.5 config spelling. NemotronH's `layer_types` is EMPTY, so all 52
  // layers classified as full attention: zero recurrent buffers and ~8.7x the
  // attention pages actually needed (52 against 6 real GQA layers).
  //
  // BYTE-NEUTRALITY CONTRACT, mirroring the one `per_layer_attn_specs` states
  // at `include/vllm/v1/kv_cache_interface.h:354-374`: the by-name path is
  // entered ONLY when the recurrent group — and the target attention group, if
  // there is one — publish per-layer names that all resolve to distinct
  // in-range layer indices. Every registry shipping today publishes a single
  // PLACEHOLDER name per group ("fa"/"gdn", "mla"/"kda", "fa_draft"), which
  // resolves to nothing, so every existing model takes the `layer_types`
  // fallback below and gets byte-identical allocation, view, indexing and
  // kernel dispatch to before this field was read. This is a capability probe
  // on the record the model published, NOT a per-architecture switch: any
  // future hybrid that publishes real names is routed correctly with no new
  // branch, which is the whole point (`hf_config.cpp:484-528` synthesizing
  // Qwen3.5's dialect for Kimi-Linear is the anti-pattern this replaces).
  const int64_t num_layers = config_.num_hidden_layers;
  std::optional<std::vector<bool>> gdn_layer_mask;
  std::optional<std::vector<bool>> attn_layer_mask;
  if (has_mamba_group) {
    gdn_layer_mask = GroupLayerMask(
        kv_cache_config.kv_cache_groups[static_cast<size_t>(gdn_group_id_)],
        num_layers);
    if (gdn_layer_mask.has_value() && full_attn_group_id_ >= 0) {
      attn_layer_mask =
          GroupLayerMask(kv_cache_config
                             .kv_cache_groups[static_cast<size_t>(
                                 full_attn_group_id_)],
                         num_layers);
      // A recurrent group that names its layers next to an attention group that
      // does not leaves the non-recurrent layers unclassifiable. Fall back
      // wholesale rather than guess.
      if (!attn_layer_mask.has_value()) gdn_layer_mask.reset();
    }
  }
  const bool membership_by_name = gdn_layer_mask.has_value();
  if (membership_by_name && attn_layer_mask.has_value()) {
    for (int64_t l = 0; l < num_layers; ++l) {
      VT_CHECK(!((*gdn_layer_mask)[static_cast<size_t>(l)] &&
                 (*attn_layer_mask)[static_cast<size_t>(l)]),
               "runner: a layer is named by BOTH the attention and the "
               "recurrent KV cache group");
    }
  }
  // PER-LAYER KV head_dim (Gemma-4 G1b). When the model publishes a per-layer
  // attention spec (heterogeneous head_dim: sliding 256 / global 512), each
  // non-GDN layer allocates + views its OWN head_size/num_kv_heads/page_size.
  // EMPTY for every uniform-KV model => `has_per_layer` is false => the loop
  // uses the single group spec (Hkv/Dh/fa_page_bytes) for every layer, which is
  // byte-identical to before this field existed (proven by the CPU paged-engine
  // suite staying green). The block table / manager are head_dim-independent, so
  // a single group + per-layer allocation is correct with no per-group changes.
  const bool has_per_layer = !kv_cache_config.per_layer_attn_specs.empty();
  if (has_per_layer) {
    VT_CHECK(kv_cache_config.per_layer_attn_specs.size() ==
                 static_cast<size_t>(config_.num_hidden_layers),
             "runner: per_layer_attn_specs must have one entry per hidden layer");
  }
  // Per-full-attn-buffer view geometry, parallel to full_attn_buf_ (built here
  // so the views loop below need not re-derive per-layer dims). In the uniform
  // case every entry is {Hkv, Dh, kv_dtype} — identical to today.
  struct FaDims {
    int64_t num_kv_heads;
    int64_t head_size;
    vt::DType dtype;
  };
  std::vector<FaDims> fa_dims;
  // Parallel to fa_dims: 1 when the layer's spec kind is kMlaAttention (the
  // fused 3-dim cache view) vs 0 for a dense NHD layer.
  std::vector<char> mla_layer_mask;
  layer_kv_class_.assign(static_cast<size_t>(num_layers), LayerKvClass::kNone);
  for (int64_t l = 0; l < num_layers; ++l) {
    bool is_gdn = false;
    bool is_full_attn = false;
    if (membership_by_name) {
      is_gdn = (*gdn_layer_mask)[static_cast<size_t>(l)];
      is_full_attn = !is_gdn && attn_layer_mask.has_value() &&
                     (*attn_layer_mask)[static_cast<size_t>(l)];
    } else {
      // The historical predicate, unchanged. A full-attention-only model (e.g.
      // dense Qwen3ForCausalLM) has NO recurrent group and an EMPTY
      // layer_types — indexing layer_types[l] would be out of bounds.
      is_gdn = has_mamba_group && !config_.layer_types.empty() &&
               config_.layer_types[static_cast<size_t>(l)] ==
                   "linear_attention";
      is_full_attn = !is_gdn;
    }
    layer_kv_class_[static_cast<size_t>(l)] =
        is_gdn ? LayerKvClass::kRecurrent
               : (is_full_attn ? LayerKvClass::kFullAttention
                               : LayerKvClass::kNone);
    if (is_gdn) {
      VT_CHECK(mamba_spec != nullptr,
               "runner: linear-attention layer has no MambaSpec");
      // Raw buffers use their independent cache dtypes. Zero bytes are +0.0f
      // for every supported floating storage type. Per-slot element counts come
      // from the SPEC's shapes (#810), not from HF-config arithmetic.
      const size_t ssm_es = vt::SizeOf(gdn_ssm_cache_dtype_);
      const size_t conv_es = vt::SizeOf(gdn_conv_cache_dtype_);
      ssm_buf_.push_back(std::make_unique<CacheBuffer>(
          dev, queue_,
          static_cast<size_t>(gdn_state_slots_ * ssm_row_elems) * ssm_es,
          kv_cache_backend_resident_));
      conv_buf_.push_back(std::make_unique<CacheBuffer>(
          dev, queue_,
          static_cast<size_t>(gdn_state_slots_ * conv_row_elems) * conv_es,
          kv_cache_backend_resident_));
    } else if (is_full_attn) {
      // Bytes come from the SPEC, not from HF-config arithmetic: exactly
      // `num_blocks * spec->page_size_bytes()`, mirroring upstream's
      // `kv_cache_interface.py:380-398` sizing contract. For a symmetric
      // FullAttentionSpec this is byte-identical to the old
      // `num_blocks * 2 * block * Hkv * Dh * sizeof(kv_dtype)`; for a future
      // MLAAttentionSpec it drops the factor 2 with no allocator change.
      // 0 bytes == 0.0 in both bf16 and f32.
      //
      // PER-LAYER (Gemma-4 G1b): the layer's own spec supplies its page bytes +
      // view geometry when published; otherwise the single group spec (uniform
      // path). `l_page` collapses to `fa_page_bytes` and `{l_Hkv,l_Dh,l_dtype}`
      // to `{Hkv,Dh,kv_dtype}` when `has_per_layer` is false — byte-identical.
      int64_t l_Hkv = Hkv;
      int64_t l_Dh = Dh;
      int64_t l_page = fa_page_bytes;
      vt::DType l_dtype = kv_dtype;
      if (has_per_layer) {
        const std::shared_ptr<AttentionSpec>& sp =
            kv_cache_config.per_layer_attn_specs[static_cast<size_t>(l)];
        VT_CHECK(sp != nullptr,
                 "runner: non-GDN layer has no per-layer attention spec");
        l_Hkv = sp->num_kv_heads;
        l_Dh = sp->head_size;
        l_dtype = sp->dtype;
        l_page = sp->page_size_bytes();
        // Same guard as the group spec: the PagedKvCache view carries ONE
        // head_size, so an asymmetric-V layer is not expressible in it.
        if (const auto* full_sp = dynamic_cast<const FullAttentionSpec*>(sp.get())) {
          VT_CHECK(full_sp->head_size_v == full_sp->head_size,
                   "runner: asymmetric head_size_v is not expressible in the "
                   "per-layer PagedKvCache view");
        }
        VT_CHECK(l_page > 0,
                 "runner: per-layer attention spec reported a non-positive page");
      }
      full_attn_buf_.push_back(std::make_unique<CacheBuffer>(
          dev, queue_,
          static_cast<size_t>(num_blocks_) * static_cast<size_t>(l_page),
          kv_cache_backend_resident_));
      fa_dims.push_back(FaDims{l_Hkv, l_Dh, l_dtype});
      // Per-layer MLA flag, parallel to fa_dims: the view loop picks the right
      // backend name (TRITON_MLA for an MLA group) and the right expected KV
      // shape (fused 3-dim, not the NHD 5-dim) per group.
      const KVCacheSpecKind layer_kind = has_per_layer
          ? kv_cache_config
                .per_layer_attn_specs[static_cast<size_t>(l)]->kind()
          : kv_cache_config
                .kv_cache_groups[static_cast<size_t>(full_attn_group_id_)]
                .kv_cache_spec->kind();
      mla_layer_mask.push_back(layer_kind == KVCacheSpecKind::kMlaAttention);
    }
    // else: this layer is named by NO KV cache group, so it caches nothing.
    // Reachable only on the by-name path, and it is the correct answer there:
    // NemotronH's 52 blocks are 6 attention + 23 Mamba2 + 23 MoE, and an MoE
    // block registers no attention module at all, so upstream's
    // `get_kv_cache_spec()` yields no entry for it
    // (`gpu_model_runner.py:7785-7801` walks `AttentionLayerBase` instances).
    // The `layer_types` fallback cannot express this third class — every
    // non-recurrent layer there is an attention layer — which is why it
    // allocated 52 attention pages for a model that needs 6.
  }

  // Build the views over the (now stable) backing storage. `fa_dims` is parallel
  // to `full_attn_buf_`; in the uniform case every entry is {Hkv, Dh, kv_dtype}.
  VT_CHECK(fa_dims.size() == full_attn_buf_.size(),
           "runner: per-layer KV view geometry out of sync with buffers");
  attn_kv_.clear();
  attn_backend_names_.clear();
  for (size_t i = 0; i < full_attn_buf_.size(); ++i) {
    PagedKvCache kv;
    kv.data = full_attn_buf_[i]->data();
    kv.dtype = fa_dims[i].dtype;
    kv.num_blocks = num_blocks_;
    kv.block_size = fa_block_size;
    kv.num_kv_heads = fa_dims[i].num_kv_heads;
    kv.head_size = fa_dims[i].head_size;
    // M3: the backend selection resolved for THIS group must describe the view
    // geometry the engine allocates + KvSlice reads — the NHD 5-dim
    // (num_blocks, 2, block_size, num_kv_heads, head_size) for a dense group,
    // the fused MLA 3-dim (num_blocks, block_size, head_size) for an MLA group
    // (vllm::v1::CheckKvCacheShape). An empty name (MLA on a device with no
    // registered MLA backend) means op-driven execution — nothing to validate.
    // A future backend with a different layout fails LOUDLY here at init.
    const bool is_mla =
        mla_layer_mask[static_cast<size_t>(i)] != 0;
    // #1332 M1: the selector now applies the full validate_configuration
    // capability surface, so the request it is asked has to BE the request. The
    // three fields this site can answer come straight from the geometry it just
    // resolved. `dtype` (the model/query dtype) is NOT available here — the
    // runner resolves only ResolveKvCacheDType() — so it keeps its bf16 default;
    // that is owed to #1332 M4 and recorded under `## Owed` in
    // .agents/specs/attn-validate-configuration.md.
    //
    // AND READ THIS BEFORE READING A GREEN SELECTION AS A WORKING BACKEND: the
    // name resolved here still reaches only attn_backend_names_, the
    // VT_ATTN_SELECT_LOG print below and CheckKvCacheShape. dense_attn::AttnBlock
    // calls vt::PagedAttention unconditionally. Nothing DISPATCHES on this. #1332
    // M4 owns that, and until it lands a valid name is a claim, not a route.
    vllm::platforms::AttnSelectorConfig cfg;
    cfg.head_size = static_cast<int>(fa_dims[i].head_size);
    cfg.num_heads = static_cast<int>(fa_dims[i].num_kv_heads);
    cfg.block_size = static_cast<int>(fa_block_size);
    cfg.kv_cache_dtype = vllm::v1::KvCacheDTypeName(fa_dims[i].dtype);
    cfg.quantized_kv_cache = vllm::v1::IsQuantizedKvCacheName(cfg.kv_cache_dtype);

    std::string name;
    if (is_mla) {
      if (!mla_backend_resolved) {
        mla_backend_resolved = true;
        vllm::platforms::AttnSelectorConfig mla_cfg = cfg;
        mla_cfg.use_mla = true;
        try {
          mla_backend = vllm::v1::SelectAttentionBackendName(
              vllm::platforms::GetPlatform(queue_.device.type), "", mla_cfg);
        } catch (const std::exception&) {
          // Op-driven MLA (no registered MLA backend for this device) —
          // recorded, not an error; see attn_backend_names_ in runner.h.
        }
      }
      name = mla_backend;
    } else {
      if (!dense_backend_resolved) {
        dense_backend_resolved = true;
        dense_backend = vllm::v1::SelectAttentionBackendName(
            vllm::platforms::GetPlatform(queue_.device.type), "", cfg);
      }
      name = dense_backend;
    }
    attn_backend_names_.push_back(name);
    if (const char* dbg = std::getenv("VT_ATTN_SELECT_LOG");
        dbg != nullptr && dbg[0] == '1') {
      if (is_mla) {
        std::fprintf(stderr,
                     "[attn-select] kind=mla backend=%s device=%d "
                     "shape=[%lld,%lld,%lld]\n",
                     name.empty() ? "(op-driven)" : name.c_str(),
                     static_cast<int>(queue_.device.type),
                     static_cast<long long>(num_blocks_),
                     static_cast<long long>(fa_block_size),
                     static_cast<long long>(fa_dims[i].head_size));
      } else {
        std::fprintf(stderr,
                     "[attn-select] kind=dense backend=%s device=%d "
                     "shape=[%lld,2,%lld,%lld,%lld]\n",
                     name.c_str(), static_cast<int>(queue_.device.type),
                     static_cast<long long>(num_blocks_),
                     static_cast<long long>(fa_block_size),
                     static_cast<long long>(fa_dims[i].num_kv_heads),
                     static_cast<long long>(fa_dims[i].head_size));
      }
    }
    if (!name.empty()) {
      vllm::v1::CheckKvCacheShape(queue_.device.type, name, num_blocks_,
                                  fa_block_size, fa_dims[i].num_kv_heads,
                                  fa_dims[i].head_size, is_mla);
    }
    attn_kv_.push_back(kv);
  }

  // SPEC-MTP I5d: allocate the MTP draft's own paged KV layer (the `fa_draft`
  // group). It is sized exactly like a target full-attn layer and the propose
  // forward reuses the target's block table / slot mapping over it
  // (speculator.py:222-234). Allocated ONLY when speculation is on and the ctor
  // did not already supply a draft KV (tests may). num_spec==0 has no fa_draft
  // group, so this is never entered on the production default path.
  draft_attn_buf_.clear();
  if (spec_on() && draft_attn_kv_.empty() && full_attn_group_id_ >= 0 &&
      fa_page_bytes > 0) {
    for (int g = 0;
         g < static_cast<int>(kv_cache_config.kv_cache_groups.size()); ++g) {
      if (g == full_attn_group_id_) continue;
      const auto& group = kv_cache_config.kv_cache_groups[static_cast<size_t>(g)];
      if (group.kv_cache_spec->kind() != KVCacheSpecKind::kFullAttention) {
        continue;  // the GDN group and any non-attn group are not the draft.
      }
      draft_attn_buf_.push_back(std::make_unique<CacheBuffer>(
          dev, queue_,
          static_cast<size_t>(num_blocks_) * static_cast<size_t>(fa_page_bytes),
          kv_cache_backend_resident_));
      PagedKvCache dkv;
      dkv.data = draft_attn_buf_.back()->data();
      dkv.dtype = kv_dtype;
      dkv.num_blocks = num_blocks_;
      dkv.block_size = fa_block_size;
      dkv.num_kv_heads = Hkv;
      dkv.head_size = Dh;
      draft_attn_kv_.push_back(dkv);
      break;  // exactly one fa_draft group at k=1.
    }
  }

  // #810: the recurrent VIEWS carry the spec's own per-slot shape with the slot
  // dim prepended — the mirror of `mamba/abstract.py:38-43`'s
  // `state.view(-1, *shape)`. Rank-general up to vt::kMaxRank (checked above),
  // so a 2-D conv state and a 3-D temporal state are both expressible without
  // the runner knowing what either MEANS.
  const auto slot_major_view = [&](void* data, vt::DType dtype,
                                   const std::vector<int64_t>& shape) {
    switch (shape.size()) {
      case 1:
        return vt::Tensor::Contiguous(data, dtype, dev,
                                      {gdn_state_slots_, shape[0]});
      case 2:
        return vt::Tensor::Contiguous(data, dtype, dev,
                                      {gdn_state_slots_, shape[0], shape[1]});
      default:
        return vt::Tensor::Contiguous(
            data, dtype, dev,
            {gdn_state_slots_, shape[0], shape[1], shape[2]});
    }
  };
  gdn_state_.clear();
  for (size_t g = 0; g < ssm_buf_.size(); ++g) {
    GdnStateCache gs;
    gs.ssm_state = slot_major_view(ssm_buf_[g]->data(), gdn_ssm_cache_dtype_,
                                   ssm_state_shape);
    gs.conv_state = slot_major_view(conv_buf_[g]->data(), gdn_conv_cache_dtype_,
                                    conv_state_shape);
    gdn_state_.push_back(gs);
  }
}

std::vector<int32_t> GPUModelRunner::gather_block_table(int group_id,
                                                        int num_reqs,
                                                        int* num_cols) const {
  const BlockTable& bt = input_batch_.block_table[group_id];
  const int cols = bt.max_num_blocks_per_req;
  *num_cols = cols;
  const std::vector<int32_t>& dev = bt.get_device_tensor();  // committed rows
  const size_t n = static_cast<size_t>(num_reqs) * static_cast<size_t>(cols);
  return std::vector<int32_t>(dev.begin(),
                              dev.begin() + static_cast<std::ptrdiff_t>(n));
}

void GPUModelRunner::remap_gdn_state_slots(
    std::vector<int32_t>& gdn_bt, int gdn_cols, int num_reqs,
    const std::vector<std::optional<std::string>>& req_ids) {
  if (gdn_cols <= 0 || num_reqs <= 0) return;
  // The persistent batch holds EVERY live sequence in its [0, num_reqs) prefix.
  // Key the compact state slot on the sequence IDENTITY (req_id), never the
  // block-table col-0 block-id: once a sequence exceeds one mamba block that
  // column collapses to the shared null block-id 0 (MambaManager skips all but
  // the last block), so block-id keying maps every long concurrent sequence to
  // ONE slot — the captured c16 "duplicate live GDN state index" fatal and,
  // pre-validator, silent cross-request recurrent-state corruption.
  // Reused member scratch (buckets persist across steps) — no per-step set
  // allocation. Cleared then refilled with this step's live request ids.
  std::unordered_set<std::string>& alive = gdn_alive_scratch_;
  alive.clear();
  alive.reserve(static_cast<size_t>(num_reqs));
  for (int r = 0; r < num_reqs; ++r) {
    // req_ids[r] is populated for every active [0, num_reqs) row after condense.
    VT_CHECK(req_ids[static_cast<size_t>(r)].has_value(),
             "GDN remap: active batch row is missing its request id");
    alive.insert(*req_ids[static_cast<size_t>(r)]);
  }
  // Reclaim slots of sequences no longer in the batch (finished / preempted):
  // a slot is released only after its owning request leaves.
  for (auto it = gdn_slot_of_req_.begin(); it != gdn_slot_of_req_.end();) {
    if (alive.find(it->first) == alive.end()) {
      gdn_free_slots_.push_back(it->second);
      it = gdn_slot_of_req_.erase(it);
    } else {
      ++it;
    }
  }
  // Assign/reuse a compact BASE slot per live sequence. Without speculation the
  // base is written into col 0 (the only column the GDN metadata builder reads).
  // SPEC-MTP I5d: under speculation each sequence owns num_spec+1 CONSECUTIVE
  // slots [base, base+num_spec]; write them into cols 0..num_spec so the spec
  // GDN builder reads the k+1 draft-timestep state slots (gdn_attn.py:266-269).
  const int spec_cols = spec_on() ? num_spec() + 1 : 1;
  for (int r = 0; r < num_reqs; ++r) {
    const std::string& rid = *req_ids[static_cast<size_t>(r)];
    const size_t off = static_cast<size_t>(r) * static_cast<size_t>(gdn_cols);
    auto it = gdn_slot_of_req_.find(rid);
    int32_t base;
    if (it != gdn_slot_of_req_.end()) {
      base = it->second;
    } else {
      VT_CHECK(!gdn_free_slots_.empty(),
               "GDN state slots exhausted: live sequences exceed max_num_reqs");
      base = gdn_free_slots_.back();
      gdn_free_slots_.pop_back();
      gdn_slot_of_req_.emplace(rid, base);
    }
    for (int c = 0; c < spec_cols && c < gdn_cols; ++c) {
      gdn_bt[off + static_cast<size_t>(c)] = base + c;
    }
  }
}

std::optional<ModelRunnerOutput> GPUModelRunner::execute_model(
    const SchedulerOutput& scheduler_output) {
  // ENG-ASYNC-SCHED depth-2 LIFETIME GUARD. Under async scheduling the previous
  // step went through sample_tokens_async, which DEFERS the main queue's D2H to
  // the consuming step's get_output() (one step_with_batch_queue call later). So
  // on entry here the previous step's forward / sample / scatter kernels may
  // still be IN FLIGHT, and they still reference the state we are about to
  // mutate: exec_state_ (the device logits + the StepInputs host arrays the
  // forward/sampler read) and input_batch_.last_sampled_tokens (the scatter
  // writes it; update_states' condense/swap read+move it). Freeing/mutating that
  // state now — while those kernels run on GB10's unified memory — is a
  // use-after-free that corrupts the host heap. It only manifests with REAL GPU
  // overlap: the CPU eager backend and compute-sanitizer both serialize the
  // queue and never see it, which is exactly why the CPU gates stayed green while
  // the served 35B aborted.
  //
  // Two dispositions of that drain, selected by async_device_mirror():
  //  - DEFAULT (mirror OFF, byte-identical production): last_sampled_tokens is a
  //    HOST array the scatter writes and update_states' condense reorders, so the
  //    drain MUST sit here, at the TOP, before update_states touches it.
  //  - MIRROR ON (VT_ASYNC_DEVICE_MIRROR, device-resident sampled tokens): the
  //    scatter writes and the combine reads the DEVICE mirror (dev->last_sampled),
  //    and condense only reorders a now-stale host array + records ops replayed
  //    on-queue (input_batch_.last_sampled_ops). The host condense<->scatter
  //    read-after-write is gone, so the drain no longer has to PRECEDE
  //    update_states. It MOVES to just before the forward (below): update_states +
  //    prepare_inputs + the attention/GDN metadata build (the bulk host prep, all
  //    of which read input_batch_ only, never exec_state_) then overlap the
  //    previous step's GPU tail, while exec_state_ and the persistent decode-graph
  //    buffers stay untouched — and therefore UAF-safe — until that later drain.
  //    This is a drain MOVE, not a drain REMOVAL: removal would additionally need
  //    double-buffered exec_state_ + block-table device buffers, which the moved
  //    drain makes unnecessary because nothing is freed/overwritten before it.
  //    Mirrors vLLM keeping input_batch.prev_sampled_token_ids device-resident
  //    (states.py:64) with the _prepare_input_ids GPU gather
  //    (gpu_model_runner.py:1786-1881) so _update_states never reads a
  //    device-written buffer and the sync only guards the input staging.
  const bool mirror = async_device_mirror();
  if (async_forward_in_flight_ && !mirror) {
    vt::GetBackend(queue_.device.type).Synchronize(queue_);
    async_forward_in_flight_ = false;
  }

  // update_states: admit new reqs (incl. prefill_token_ids) + apply cached diffs
  // + remove finished/unscheduled + condense (M1.5). Reads/mutates input_batch_
  // only (never exec_state_), so on the mirror path it is safe to run while the
  // previous step's kernels still read exec_state_.
  update_states(input_batch_, scheduler_output);

  // Reset the stash. A 0-token step (e.g. an aborted-request flush) runs no
  // forward — mark num_reqs == 0 so sample_tokens returns an empty output
  // (mirrors upstream execute_model's total==0 early return). MIRROR ON DEFERS
  // this reset (and the drain) to just before the forward, because exec_state_
  // still holds the previous step's device logits the in-flight sampler reads;
  // freeing it here would be the very UAF the guard exists to prevent.
  if (!mirror) exec_state_ = ExecuteModelState{};
  if (scheduler_output.total_num_scheduled_tokens == 0 ||
      input_batch_.num_reqs() == 0) {
    // Mirror path deferred the drain + reset; do them now before returning. The
    // recorded structural ops (this step's condense) survive to the next step
    // that runs a combine — see the last_sampled_ops note below.
    if (async_forward_in_flight_) {
      vt::GetBackend(queue_.device.type).Synchronize(queue_);
      async_forward_in_flight_ = false;
    }
    if (mirror) exec_state_ = ExecuteModelState{};
    return std::nullopt;
  }

  // DECODE-FIRST REORDER (four-way ordering contract) — before any metadata.
  reorder_batch_to_split_decodes_and_prefills(input_batch_, scheduler_output);

  // SPEC-MTP I5d: splice the scheduler's drafts for THIS verify step into
  // token_ids_cpu after each request's committed prefix (gpu_input_batch.py:
  // 484-509 update_req_spec_token_ids) so prepare_inputs reads the k draft tokens
  // at the verify positions. No-op on the default path (empty map / no speculator).
  if (spec_on()) {
    const int nr = input_batch_.num_reqs();
    for (int i = 0; i < nr; ++i) {
      const std::string& req_id = *input_batch_.req_ids[static_cast<size_t>(i)];
      input_batch_.update_req_spec_token_ids(
          i, req_id, scheduler_output.scheduled_spec_decode_tokens);
    }
  }

  // Build the flattened dense-order step inputs (M1.5).
  StepInputs step = prepare_inputs(input_batch_, scheduler_output);
  const int num_reqs = input_batch_.num_reqs();

  // Async-scheduling device-input path (ENG-ASYNC-SCHED W3 runner leaf). When
  // engaged, overwrite each decode row's input token id with the GPU-resident-
  // analog last_sampled_tokens (combine_sampled_and_draft_tokens) instead of the
  // host token_ids_cpu value prepare_inputs read — so step N+1 need not wait on
  // step N's sampled token crossing to the host. idx_mapping is identity: our
  // persistent batch is already condensed dense (batch row == req_state slot).
  // Runs on the HOST side of input prep, BEFORE the forward and OUTSIDE any
  // CUDA-graph capture (input prep always precedes the decode graph replay), so
  // it is capture-safe. The mirror is DEFAULT ON (ROW-SERVE-ASYNC-LLM P0 fix);
  // VT_ASYNC_DEVICE_MIRROR=0 rolls back to the host-array combine, whose host
  // decode-graph read races this combine's device write (the bug the flip fixes).
  // Non-null on the W4 mirror path below (the default; integrated OR discrete):
  // the device input-id buffer the combine patched, handed to the forward so it
  // embeds the spliced ids instead of the (deliberately stale) host vector.
  const int32_t* device_input_ids = nullptr;
  if (async_input_combine_ && num_reqs > 0) {
#ifdef VLLM_CPP_CUDA
    // W4 device-resident sampled tokens. Preferred whenever engaged
    // (async_device_mirror(): CUDA + VT_ASYNC_DEVICE_MIRROR, INTEGRATED OR
    // DISCRETE). `last_sampled` is already on the device (the previous step's
    // scatter wrote it there and nothing read it back), so the three host-known
    // inputs plus the freshly built input_ids are uploaded through the staging
    // copy, the recorded structural edits are replayed first so the mirror's row
    // order matches this step's batch, and the combine patches the DEVICE
    // input_ids the forward will embed. On the INTEGRATED path this is what keeps
    // the sampled tokens OFF the host array update_states' condense reorders,
    // which is what lets the depth-2 drain move past update_states (execute_model
    // top). Mirrors vLLM's prev_sampled_token_ids device tensor + _prepare_input_
    // ids GPU gather (gpu_model_runner.py:1786-1881).
    //
    // Ordering, all on the MAIN queue and therefore exact: replay -> uploads ->
    // combine -> forward. The forward is handed `device_input_ids` below, so the
    // host copy of step.input_token_ids is deliberately left stale for decode
    // rows; nothing on this path reads it (the rejection-sampler path that does is
    // spec-only, and spec forces the sync runner).
    if (AsyncDeviceInputs* dev = get_or_create_async_device_inputs();
        dev != nullptr) {
      replay_last_sampled_ops(*dev);
      const int64_t num_tokens =
          static_cast<int64_t>(step.input_token_ids.size());
      VT_CHECK(num_tokens <= dev->input_ids_capacity,
               "async device mirror: step tokens exceed max_num_batched_tokens");
      VT_CHECK(num_reqs <= dev->max_reqs,
               "async device mirror: step requests exceed max_num_reqs");
      stage_upload(*dev, dev->input_ids, step.input_token_ids.data(), num_tokens);
      stage_upload(*dev, dev->query_start_loc, step.query_start_loc.data(),
                   static_cast<int64_t>(num_reqs) + 1);
      stage_upload(*dev, dev->seq_lens, step.seq_lens.data(), num_reqs);
      stage_upload(*dev, dev->prefill_len, input_batch_.prefill_len.data(),
                   num_reqs);
      vt::cuda::LaunchCombineSampledAndDraftTokens(
          queue_, dev->input_ids, /*idx_mapping=*/nullptr, dev->last_sampled,
          dev->query_start_loc, dev->seq_lens, dev->prefill_len, num_reqs,
          /*num_new_sampled_tokens=*/1);
      device_input_ids = dev->input_ids;
    } else if (vllm::platforms::GetPlatform(queue_.device.type)
                   .is_integrated_gpu()) {
      // DEVICE combine, HOST-ARRAY residence (W3 DGX leaf; mirror OFF = the GB10
      // production default). Splice each decode row's input id from the device-
      // ADDRESSABLE host last_sampled_tokens (pageable on GB10's UMA) on the MAIN
      // queue, BEFORE the forward (which embeds input_token_ids on the same queue →
      // sees the patch) and OUTSIDE any decode-graph capture. idx_mapping identity
      // (condensed-dense batch), nullptr. num_new_sampled_tokens == 1 (T0
      // non-spec). All inputs live in `step` / persistent InputBatch members, so
      // they outlive the async launch. Its scatter counterpart writes the SAME
      // host array update_states' condense reorders, which is why the drain must
      // sit at execute_model's top on this path (the mirror path lifts that).
      // is_integrated_gpu() decouples a future discrete GPU (answers false → host
      // combine below, the right path there since its host arrays are not
      // device-addressable).
      vt::cuda::LaunchCombineSampledAndDraftTokens(
          queue_, step.input_token_ids.data(), /*idx_mapping=*/nullptr,
          input_batch_.last_sampled_tokens.data(), step.query_start_loc.data(),
          step.seq_lens.data(), input_batch_.prefill_len.data(), num_reqs,
          /*num_new_sampled_tokens=*/1);
    } else
#endif
    {
      std::vector<int32_t> idx_mapping(static_cast<size_t>(num_reqs));
      std::iota(idx_mapping.begin(), idx_mapping.end(), 0);
      combine_sampled_and_draft_tokens(
          step.input_token_ids, idx_mapping, input_batch_.last_sampled_tokens,
          step.query_start_loc, step.seq_lens, input_batch_.prefill_len,
          /*num_new_sampled_tokens=*/1);
    }
  }

  // W4: the structural-op log has exactly one consumer, the device mirror's
  // replay above. On every other configuration (host-array integrated path, CPU
  // backend, the VT_ASYNC_DEVICE_MIRROR=0 rollback, async off) nothing drains it,
  // so drop it here rather than let it grow for the life of a serving process.
  // Deliberately NOT cleared when the mirror is on: a step with no requests
  // replays nothing, and its ops must survive to the next step that does.
  if (!mirror) input_batch_.last_sampled_ops.clear();

  // Full-attention KV group metadata (M1.6 MakeCommonAttentionMetadata).
  int fa_cols = 0;
  const std::vector<int32_t> fa_bt =
      gather_block_table(full_attn_group_id_, num_reqs, &fa_cols);
  CommonAttentionMetadata attn_meta = MakeCommonAttentionMetadata(
      step, fa_bt, fa_cols, /*causal=*/true, full_attn_group_id_);

  // GDN KV group metadata: the same step over the GDN group's block table,
  // segmented decode-first by the GDN builder (M1.6 Task 4). GATED on the model
  // HAVING a linear-attention (GDN/Mamba) KV group: a full-attention-only model
  // (dense Qwen3ForCausalLM) has gdn_group_id_ < 0 and no GDN block-table group,
  // so gather_block_table(gdn_group_id_) / remap / the metadata build must be
  // skipped — gdn_meta stays default-empty (num_prefill_tokens == 0, all state
  // arrays nullopt) and no GDN state is wired. The hybrid gate models keep a GDN
  // group (gdn_group_id_ >= 0), so this block runs exactly as before —
  // behavior-preserving / byte-identical for the hybrid path.
  GDNAttentionMetadata gdn_meta;
  if (gdn_group_id_ >= 0) {
    int gdn_cols = 0;
    std::vector<int32_t> gdn_bt =
        gather_block_table(gdn_group_id_, num_reqs, &gdn_cols);
    // SPEC-MTP I5d: the spec GDN builder reads cols 0..num_spec of the block
    // table as the k+1 draft-timestep state slots. Our compact per-sequence pool
    // OWNS the slot ids (the physical mamba block-id is irrelevant, see the remap
    // rationale), so widen the block-table view to num_spec+1 columns when the
    // real GDN block table is narrower, and let remap fill all k+1 slots.
    if (spec_on() && gdn_cols < num_spec() + 1) {
      const int wide = num_spec() + 1;
      std::vector<int32_t> widened(
          static_cast<size_t>(num_reqs) * static_cast<size_t>(wide), 0);
      for (int r = 0; r < num_reqs; ++r) {
        widened[static_cast<size_t>(r) * static_cast<size_t>(wide)] =
            gdn_bt[static_cast<size_t>(r) * static_cast<size_t>(gdn_cols)];
      }
      gdn_bt = std::move(widened);
      gdn_cols = wide;
    }
    // Remap col 0 to a compact per-sequence state slot in [0, gdn_state_slots_),
    // keyed on the request identity so the GDN state cache is sized by
    // max_num_reqs (one recurrent state per sequence) rather than the attention
    // num_blocks, and no two live sequences ever collide on one slot. Only col 0
    // (state indices) is read downstream.
    remap_gdn_state_slots(gdn_bt, gdn_cols, num_reqs, input_batch_.req_ids);
    if (GdnDiagStepLogEnabled()) {
      std::cerr << "[VT_GDN_DIAG] step num_reqs=" << num_reqs
                << " gdn_free_slots=" << gdn_free_slots_.size()
                << " gdn_live_slots=" << gdn_slot_of_req_.size() << "\n";
    }
    const CommonAttentionMetadata gdn_cam = MakeCommonAttentionMetadata(
        step, gdn_bt, gdn_cols, /*causal=*/true, gdn_group_id_);
    if (spec_on()) {
      // SPEC-MTP I5d: feed the GDN builder's spec overload (gdn_attn.py:189-326).
      // num_decode_draft_tokens_cpu[i] = -1 for a non-spec row, else the number of
      // drafts this verify step scheduled for req i (mamba_hybrid.py:247-264);
      // num_accepted_tokens[i] is the PREVIOUS step's accepted count (seeded 1,
      // overwritten by the rejection sampler). The remapped gdn_bt cols 0..k hold
      // this request's k+1 GDN state slots (see remap_gdn_state_slots spec branch).
      std::vector<int32_t> num_decode_draft_tokens(static_cast<size_t>(num_reqs), -1);
      for (int i = 0; i < num_reqs; ++i) {
        const std::string& req_id = *input_batch_.req_ids[static_cast<size_t>(i)];
        const auto it = scheduler_output.scheduled_spec_decode_tokens.find(req_id);
        if (it != scheduler_output.scheduled_spec_decode_tokens.end() &&
            !it->second.empty()) {
          num_decode_draft_tokens[static_cast<size_t>(i)] =
              static_cast<int32_t>(it->second.size());
        }
      }
      std::vector<int32_t> num_accepted(
          input_batch_.num_accepted_tokens.begin(),
          input_batch_.num_accepted_tokens.begin() + num_reqs);
      GDNAttentionMetadataBuilder gdn_builder(num_spec());
      gdn_meta = gdn_builder.build(/*common_prefix_len=*/0, gdn_cam, &num_accepted,
                                   &num_decode_draft_tokens);
    } else {
      GDNAttentionMetadataBuilder gdn_builder;
      gdn_meta = gdn_builder.build(/*common_prefix_len=*/0, gdn_cam);
    }
  }

  // Flattened dense-order forward inputs (positions int64 -> int32 for RoPE).
  const std::vector<int32_t>& token_ids = step.input_token_ids;
  std::vector<int32_t> positions;
  positions.reserve(step.positions.size());
  for (const int64_t position : step.positions) {
    positions.push_back(static_cast<int32_t>(position));
  }

  // THE FORWARD (Task 3, over the persistent KV caches). Returns f32 logits
  // (lm_head already applied): [num_reqs, vocab] when the gather-before-lm_head
  // path is on (prefill/mixed) or pure-decode, else the full
  // [num_actual_tokens, vocab] (VT_LOGITS_GATHER=0).
  //
  // DECODE CUDA-GRAPH path (M2.5): the registered model forward routes a
  // PURE-DECODE fp4/CUDA step through its model-specific graph, whose state now
  // lives behind LoadedModel. It captures once per padded batch size and
  // replays per step; real-row output remains bit-identical to eager Forward.
  //
  // BATCHED (num_reqs>1) — the gate-#1 lever: at conc-64 kernel-launch overhead
  // is ~24% of the decode wall (~1.4k cudaLaunchKernel/step -> 1 cudaGraphLaunch).
  // The batch is padded up to the nearest captured size {1,2,4,8,16,32,64}
  // (mirrors vLLM cudagraph_capture_sizes + pad-to-nearest,
  // compilation/cuda_graph.py); the padded rows are inert (BuildPaddedDecode).
  // The decode forward is row-independent, so padding cannot perturb the real
  // rows. Beyond 64 (kMaxDecodeGraphBatch) / prefill / mixed / bf16 / CPU stay
  // eager. The DENSE (27B) registration owns its sibling decode graph under the
  // same gate.
  const bool pure_decode = attn_meta.num_actual_tokens == num_reqs &&
                           gdn_meta.num_prefill_tokens == 0;
  // ENG-CUDAGRAPH-BREAK W6 (#1374): THE GRAPH-ELIGIBILITY PREDICATE, and this is
  // the line the row exists to move. Until here the runner shipped ONE boolean
  // that means "query length is 1", and a model wanting anything wider had to
  // re-derive the whole test for itself -- which two of them did, in twenty
  // duplicated lines each (`qwen3_5_moe.cpp`, `qwen3_5_dense.cpp` @ #442). The
  // runner now names the step's ACTUAL uniform query length once and every model
  // reads the answer.
  //
  // IT IS ALSO THE FIX FOR [#1020]. Those two copies compared the uniform length
  // against the CONFIGURED `num_spec()`, so a step the scheduler clamped to a
  // shorter -- but still perfectly uniform -- draft depth missed the predicate
  // and ran its verify eager, with no log and no counter.
  // `ActualUniformDecodeQueryLen` reads the length the step HAS, bounded above
  // by `1 + num_spec()` because nothing in this tree captures a longer one.
  //
  // The GDN prefill conjunct is `pure_decode`'s and stays: a step with GDN
  // prefill tokens carries recurrent-prefill segmentation no decode capture was
  // built for, and the two model copies each tested it separately.
  //
  // THE ARM ABOVE 1 IS A SPECULATIVE VERIFY AND NOTHING ELSE, and the shape
  // alone does not say so. A single request prefilling three tokens is uniform
  // at query length 3 by every arithmetic test upstream applies, and at k >= 2
  // it would pass a bare `q <= 1 + k` bound straight into a DECODE capture --
  // measured on this tree's own CPU spec fixture, where a 20-token run reported
  // 19 "uniform spec" steps before this conjunct existed. So the widened arm
  // additionally requires that EVERY request in the step is verifying at exactly
  // `q - 1` drafts, read off the scheduler's own per-request draft counts. That
  // is narrower than the shape test, never wider, and it is what makes
  // `uniform_query_len > 1` mean what its comment says it means.
  const std::optional<int64_t> uniform_qlen =
      gdn_meta.num_prefill_tokens == 0
          ? v1::GraphEligibleQueryLen(num_reqs, attn_meta.num_actual_tokens,
                                      attn_meta.max_query_len, num_spec(),
                                      step.num_draft_tokens_per_req)
          : std::nullopt;
  // #1020 is titled on the word SILENTLY. A step that finds no captured shape
  // now moves a counter, on the shared path every registered model reaches.
  v1::NoteGraphDispatch(uniform_qlen.value_or(0),
                        v1::UniformDecodeQueryLen(num_spec()));
  // Gather-before-lm_head indices (the SAME last-token rows sample_tokens uses).
  // Empty when the toggle is off → old full-logits path. The eager forwards skip
  // the gather when it is a no-op (pure decode: len == num_actual_tokens).
  // SAMPLE-PROMPT-LOGPROBS: a request that asked for prompt logprobs needs an
  // lm_head row at each of its prompt positions, not just at the one row the
  // sampler consumes. The gather seam cannot express that: every model's
  // gather-before-lm_head is guarded on `logits_indices.size() < T` — it is an
  // optimization for "far fewer rows than tokens", and scoring a whole prompt
  // is the opposite. So a step that owes prompt logits takes the full-logits
  // path instead, which is the same shape upstream computes anyway
  // (gpu_model_runner.py:5680-5682 runs compute_logits over the prompt slice).
  //
  // step.prompt_logprob_indices is EMPTY on every step where no request asked,
  // and then this is exactly the expression it has always been.
  const bool gather =
      LogitsGatherEnabled() && step.prompt_logprob_indices.empty();
  const std::vector<int32_t> kNoGather;
  const std::vector<int32_t>& gather_li = gather ? step.logits_indices : kNoGather;

  // THE FORWARD. DEFAULT (gather ON): the *Device entry points keep the
  // [num_reqs,vocab] logits ON DEVICE and hand them to the sampler with no
  // full-logits D2H. VT_LOGITS_GATHER=0: the host Forward returns full
  // [T,vocab] logits and sample_tokens re-gathers on host (unchanged path).
  ModelForwardInput forward_input{
      .token_ids = token_ids,
      .positions = positions,
      .attn_meta = attn_meta,
      .gdn_meta = gdn_meta,
      .attn_kv = attn_kv_,
      .gdn_state = gdn_state_,
      .config = config_,
      .queue = queue_,
      .logits_indices = gather_li,
      .num_reqs = num_reqs,
      .gdn_state_slots = gdn_state_slots_,
      .pure_decode = pure_decode,
      // SPEC-DSPARK W8 (#442): the decode-graph gate mirrors vLLM's UNIFORM
      // decode predicate, whose captured length is 1 + num_speculative_tokens
      // (cudagraph_dispatcher.py:37). 0 when speculation is off, which makes
      // the predicate reduce to today's pure-decode shape.
      .num_speculative_tokens = num_spec(),
      // W6 (#1374): the eligibility answer itself. 0 == no captured decode graph
      // in this tree serves this step.
      .uniform_query_len = uniform_qlen.value_or(0),
      .gather_logits = gather,
      // SPEC-MTP I5d: capture the target's post-final-norm [T,H] hidden for the
      // MTP drafter. Non-null only when a speculator is configured — the Qwen3.5
      // dense/MoE forward then routes to ForwardDeviceTap (byte-identical logits).
      // Null on every default step, so the forward path is byte-identical.
      // SPEC-DFLASH D5: the DFlash drafter conditions on the MULTI-tap instead,
      // so it uses aux_tap (below) and leaves hidden_tap null. SPEC-NGRAM: the
      // draft-free proposer needs no hidden state at all, so leave it null too.
      .hidden_tap = (spec_on() && !use_dflash() && !use_ngram())
                        ? &exec_state_.spec_hidden
                        : nullptr,
  };
  // SPEC-DFLASH D5: on the verify forward capture the D1 multi-tap (the residual
  // stream at the draft's target_layer_ids) as [T, H×taps]. Mutually exclusive
  // with hidden_tap; routes the dense/MoE forward to ForwardDeviceMultiTap
  // (byte-identical logits). Null on every non-dflash step.
  if (use_dflash()) {
    exec_state_.spec_aux.layer_ids = dflash_tap_layer_ids_;
    forward_input.aux_tap = &exec_state_.spec_aux;
  }
  // W4: non-null only on the discrete-CUDA async path, where the combine above
  // patched the DEVICE ids and `token_ids` is deliberately stale for decode rows.
  // Set after construction because the field sits at the END of the struct, where
  // it cannot shift the positional aggregate initializers other callers use.
  forward_input.device_token_ids = device_input_ids;

  // ENG-ASYNC-SCHED depth-2 LIFETIME GUARD — the MOVED drain (mirror path only).
  // The bulk host prep above (update_states + prepare_inputs + attention/GDN
  // metadata + forward_input assembly) has overlapped the previous step's GPU
  // tail. Drain it NOW, before we (1) free the previous step's exec_state_ — the
  // device logits the in-flight sampler still reads — and (2) let the forward
  // stage this step's inputs / replay the decode graph into the persistent buffers
  // the previous step's kernels read. `device_input_ids` (dev->input_ids) and the
  // combine/replay this step already enqueued live in the mirror's own persistent
  // buffers, not exec_state_, so the reset does not disturb them; the forward's
  // CopyInPlace of dev->input_ids into the graph inputs is main-queue-ordered after
  // this step's combine. On the DEFAULT (mirror OFF) path async_forward_in_flight_
  // was already cleared at the top and exec_state_ already reset, so this block is
  // skipped (mirror==false) and the path is byte-identical.
  if (mirror) {
    // VT_ASYNC_EXECUTOR (decode-graph slot double-buffer): if the previous step's
    // stashed logits are a NON-OWNING decode-graph slot view, the exec_state_ reset
    // frees nothing the in-flight async sampler reads (hazard-A — the eager owning
    // WrapDeviceLogits pool block — is absent for a view), and the model's 2-slot
    // parity ring + per-slot reuse event already guard the persistent decode-graph
    // host inputs against this step's replay (hazard-C). Both hazards handled, so
    // SKIP the main-queue Synchronize and let this step's host prep + replay overlap
    // the previous step's GPU tail — the c16/c32 unlock. An eager/mixed previous
    // step still owns its logits pool block, whose reset WOULD UAF the async
    // sampler, so it drains exactly as before. The reset runs on BOTH paths: on the
    // skip path it releases the no-op-deleter view and clears req_ids/discard for
    // this step's stash WITHOUT touching the still-live, still-referenced slot
    // buffer. Inert unless VT_ASYNC_EXECUTOR=1 (async_executor() default OFF), where
    // skip_drain is always false and the block is byte-identical to the drain.
    const bool skip_drain = async_executor() && exec_state_.logits.on_device() &&
                            exec_state_.logits.non_owning_view;
    if (skip_drain && std::getenv("VT_ASYNC_EXECUTOR_TRACE") != nullptr) {
      // Diagnostic (test-only): confirm the overlap path is actually engaged — a
      // skip count of 0 would mean async_executor()/non_owning_view never resolved
      // and the lever is inert. Single engine thread, so a plain static suffices.
      static long long kSkips = 0;
      ++kSkips;
      if (kSkips == 1 || kSkips % 200 == 0)
        std::fprintf(stderr, "[VT_ASYNC_EXECUTOR] drain skipped x%lld\n", kSkips);
    }
    if (!skip_drain && async_forward_in_flight_) {
      vt::GetBackend(queue_.device.type).Synchronize(queue_);
      async_forward_in_flight_ = false;
    }
    exec_state_ = ExecuteModelState{};
  }

  // KV-EXTERNAL-CACHE (LMCache): apply any external-prefix loads recorded by the
  // scheduler's connector for THIS step into the freshly-allocated KV blocks
  // BEFORE the forward reads them (load-before-compute, base.py:293). Inert when
  // no worker-capable connector is set — the KV cache is byte-identical to
  // production and no sync is issued.
  if (kv_connector_ != nullptr) {
    ConnectorLoadExternalKv();
  }

  ForwardLogits logits = ModelRegistry::Forward(*model_, forward_input);

  // KV-EXTERNAL-CACHE (LMCache): after the forward has written this step's KV,
  // STORE every newly-complete prompt block to the external cache (the worker
  // half of LMCacheConnectorV1's synchronous save). Inert (no sync, no I/O)
  // unless a worker-capable connector is set.
  if (kv_connector_ != nullptr) {
    ConnectorStorePromptKv(scheduler_output);
  }

  // Stash for sample_tokens (upstream ExecuteModelState).
  exec_state_.num_actual_tokens = scheduler_output.total_num_scheduled_tokens;
  exec_state_.num_reqs = num_reqs;
  exec_state_.logits = std::move(logits);
  exec_state_.step = std::move(step);
  exec_state_.attn_meta = std::move(attn_meta);
  exec_state_.gdn_meta = std::move(gdn_meta);
  exec_state_.req_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    exec_state_.req_ids.push_back(*input_batch_.req_ids[static_cast<size_t>(i)]);
  }

  // discard_request_mask (gpu_model_runner.py:2029-2051): a scheduled request
  // whose optimistic seq_len (num_computed + num_scheduled == step.seq_lens[i])
  // has not yet reached its total known token count (num_tokens_no_spec[i] ==
  // request.num_tokens) is still consuming prefill and must NOT sample this step.
  // NOTE: under async scheduling step.seq_lens[i] == num_tokens + placeholders
  // for a decode request (the reserved in-flight token), so `< num_tokens`
  // strictly selects prefill chunks only. sample_tokens / get_output() clear the
  // sampled token for these rows, honoring the scheduler contract that a
  // still-prefilling request returns EMPTY token ids (scheduler.py:1888-1890).
  exec_state_.discard.assign(static_cast<size_t>(num_reqs), 0);
  for (int i = 0; i < num_reqs; ++i) {
    const int32_t seq_len = exec_state_.step.seq_lens[static_cast<size_t>(i)];
    const int32_t num_tokens =
        input_batch_.num_tokens_no_spec[static_cast<size_t>(i)];
    exec_state_.discard[static_cast<size_t>(i)] =
        (seq_len < num_tokens) ? 1 : 0;
  }

  return std::nullopt;  // MRV2 split: forward done, sample separately.
}

// Assemble the [num_reqs, vocab] logits + apply the grammar bitmask. Extracted
// verbatim from sample_tokens (same ops, same order) so sample_tokens and
// sample_tokens_async share ONE assembly path (byte-identical sync behavior).
vt::Tensor GPUModelRunner::assemble_sample_logits(
    const std::optional<GrammarOutput>& grammar_output,
    std::vector<float>& sampled_logits) {
  // The number of LOGIT ROWS the sampler runs on. Non-speculative (the
  // production default) this is exactly num_reqs — prepare_inputs sets
  // cu_num_logits = arange(num_reqs+1), so cu_num_logits.back() == num_reqs and
  // every expression below is byte-identical to the pre-SPEC-REJECTION code.
  // Under spec decode it is Σ(1 + k_i) (spec §2.4 / StepInputs::cu_num_logits).
  const int num_logits = step_num_logits();
  const int64_t vocab = config_.vocab_size;

  // Assemble the [num_logits, vocab] logits the sampler runs on. Three cases:
  //
  //  (A) DEVICE path (default, gather ON): the forward already produced exactly
  //      num_logits rows in request order, ON DEVICE — hand the sampler the device
  //      tensor directly (its argmax / temperature / top-k/top-p kernels run
  //      on-device; only the sampled token ids come back). No full-logits D2H.
  //  (A') VT_GPU_SAMPLE=0 A/B: Download the device [num_logits,vocab] logits to a
  //      host buffer and sample from it (the OLD download-then-sample path).
  //  (B) HOST path (VT_LOGITS_GATHER=0): full [num_actual_tokens,vocab] host
  //      logits — re-gather the per-request rows on host via logits_indices,
  //      exactly as before.
  vt::Tensor logits;
  ForwardLogits& fl = exec_state_.logits;
  if (fl.on_device()) {
    VT_CHECK(fl.rows == num_logits,
             "sample_tokens: device logits rows must equal the expanded logits rows");
    if (GpuSampleEnabled()) {
      logits = fl.device_tensor;  // (A) sample straight off device
    } else {
      // (A') download then sample (A/B: reproduce the pre-change host path).
      sampled_logits.resize(static_cast<size_t>(num_logits) *
                            static_cast<size_t>(vocab));
      vt::Backend& b = vt::GetBackend(queue_.device.type);
      b.Copy(queue_, sampled_logits.data(), fl.device_tensor.data,
             sampled_logits.size() * sizeof(float));
      b.Synchronize(queue_);
      logits = vt::Tensor::Contiguous(
          sampled_logits.data(), vt::DType::kF32, queue_.device,
          {static_cast<int64_t>(num_logits), vocab});
    }
  } else if (fl.rows == num_logits) {
    logits = vt::Tensor::Contiguous(
        fl.host.data(), vt::DType::kF32, queue_.device,
        {static_cast<int64_t>(num_logits), vocab});
  } else {
    // (B) VT_LOGITS_GATHER=0: re-gather the logits rows from full [T,vocab] host.
    sampled_logits.resize(static_cast<size_t>(num_logits) *
                          static_cast<size_t>(vocab));
    for (int i = 0; i < num_logits; ++i) {
      const int row = exec_state_.step.logits_indices[static_cast<size_t>(i)];
      std::memcpy(
          sampled_logits.data() + static_cast<size_t>(i) * static_cast<size_t>(vocab),
          fl.host.data() +
              static_cast<size_t>(row) * static_cast<size_t>(vocab),
          static_cast<size_t>(vocab) * sizeof(float));
    }
    logits = vt::Tensor::Contiguous(
        sampled_logits.data(), vt::DType::kF32, queue_.device,
        {static_cast<int64_t>(num_logits), vocab});
  }

  // Apply the structured-output grammar bitmask (utils.py apply_grammar_bitmask)
  // to the gathered [num_logits, vocab] logits BEFORE sampling, when a structured
  // request is scheduled this step (gpu_model_runner.py:4462-4466). The grammar
  // bitmask over the EXPANDED spec rows (a bitmask row per draft position) is
  // DEFERRED with SPEC-MTP (spec §Protocol-compliance "Grammar bitmask under
  // spec decode: OUT of scope"), so the spec-token map stays empty (per-req
  // offset 0) — correct while num_draft_tokens == 0, which is the only state
  // the runner can reach today.
  if (grammar_output.has_value()) {
    apply_grammar_bitmask(*grammar_output, exec_state_.req_ids, {}, queue_,
                          logits);
  }
  return logits;
}

// step_num_logits: the number of expanded logit rows this step (spec §2.4).
// cu_num_logits is arange(num_reqs+1) on the non-speculative path, so this is
// exactly num_reqs there; the `.empty()` guard covers a StepInputs built by an
// older direct-runner test that never went through prepare_inputs.
int GPUModelRunner::step_num_logits() const {
  const std::vector<int32_t>& cu = exec_state_.step.cu_num_logits;
  return cu.empty() ? exec_state_.num_reqs : cu.back();
}

// collect_prompt_logprobs — 1:1 vllm/v1/worker/gpu_model_runner.py:5612-5719
// (`_get_prompt_logprobs_dict`), minus the row selection, which prepare_inputs
// did (see StepInputs::PromptLogprobRows). What is left is upstream's scoring
// (:5688-5697), its slice-by-slice accumulation (:5698-5706) and its
// emit-and-drop on the final chunk (:5665-5667, :5709-5712).
void GPUModelRunner::collect_prompt_logprobs(
    std::map<std::string, LogprobsTensors>& prompt_logprobs_dict) {
  const std::vector<StepInputs::PromptLogprobRows>& rows =
      exec_state_.step.prompt_logprob_rows;
  // An in-progress entry can outlive its request: an ABORT mid-prompt drops the
  // request from the input batch, and upstream frees the same state with the
  // request object at :1199. Swept on every call, including the early return
  // below, so a stale tensor cannot survive to the next request with that id.
  if (!in_progress_prompt_logprobs_.empty()) drop_stale_prompt_logprobs();
  if (rows.empty()) return;

  const int64_t vocab = config_.vocab_size;
  ForwardLogits& fl = exec_state_.logits;

  for (const StepInputs::PromptLogprobRows& r : rows) {
    // The accumulated tensor covers num_prompt_tokens - 1 positions: the first
    // prompt token has no logprob because nothing precedes it (:5646-5651, and
    // logprobs.py:162-167 which fills that leading None). Created on first
    // sight and filled slice by slice across chunks.
    auto [it, inserted] = in_progress_prompt_logprobs_.try_emplace(r.req_id);
    if (inserted) {
      // The height comes from the REQUEST, not from this chunk: prompt length
      // less one. dst_start + num_rows reaches exactly that on the final chunk,
      // and prepare_inputs tiles [0, num_prompt_tokens - 1) in order.
      it->second.num_positions = prompt_logprob_positions(r.req_id);
      it->second.num_tokens_per_position = r.num_prompt_logprobs + 1;
      it->second.logprob_token_ids.assign(
          static_cast<size_t>(it->second.num_positions) *
              static_cast<size_t>(it->second.num_tokens_per_position),
          0);
      it->second.logprobs.assign(it->second.logprob_token_ids.size(), 0.0f);
      it->second.selected_token_ranks.assign(
          static_cast<size_t>(it->second.num_positions), 0);
    }
    LogprobsTensors& acc = it->second;

    if (r.num_rows > 0) {
      // The step that OWES rows took the full-logits path in execute_model, so
      // there is one row per scheduled token and a prompt row is found by its
      // token-stream index. This belongs HERE, at the slice, not at the top of
      // the function: prepare_inputs deliberately keeps a final-chunk entry with
      // num_rows == 0 on the exact-prefill edge (:5668-5673), and that entry
      // contributes no gather index, so the step correctly kept the gathered
      // lm_head and has num_reqs rows, not num_actual_tokens. Asserting on
      // `rows` non-empty instead of on the slice threw whenever any OTHER
      // request in the same step contributed more than one token — and VT_CHECK
      // throws out of engine.step(), taking the whole batch down with it, not
      // just the asking request. Regression: test_llm_engine §9(h).
      VT_CHECK(fl.rows == exec_state_.num_actual_tokens,
               "collect_prompt_logprobs: a step that scores prompt rows must "
               "carry full logits");
      // A request's prompt rows are the CONTIGUOUS run starting at its first
      // scheduled token (prepare_inputs built them as query_start + [0,
      // num_rows)), so this is one slice, on device or on host.
      const int64_t first = exec_state_.step.prompt_logprob_indices
                                [static_cast<size_t>(r.src_start)];
      vt::Tensor view;
      if (fl.on_device()) {
        view = fl.device_tensor.Slice(0, first, first + r.num_rows);
      } else {
        view = vt::Tensor::Contiguous(
            fl.host.data() +
                static_cast<size_t>(first) * static_cast<size_t>(vocab),
            vt::DType::kF32, queue_.device,
            {static_cast<int64_t>(r.num_rows), vocab});
      }

      const LogprobsTensors chunk = sampler_.compute_prompt_logprobs(
          queue_, view, r.num_prompt_logprobs, r.target_token_ids);

      // Copy into slice(start_idx, start_idx + num_logits) of the accumulated
      // tensor (:5698-5706).
      const size_t width =
          static_cast<size_t>(acc.num_tokens_per_position);
      VT_CHECK(chunk.num_tokens_per_position == acc.num_tokens_per_position,
               "collect_prompt_logprobs: chunk width must match the request's");
      VT_CHECK(r.dst_start + r.num_rows <= acc.num_positions,
               "collect_prompt_logprobs: chunk overruns the prompt tensor");
      const size_t dst = static_cast<size_t>(r.dst_start) * width;
      std::copy(chunk.logprob_token_ids.begin(), chunk.logprob_token_ids.end(),
                acc.logprob_token_ids.begin() +
                    static_cast<std::ptrdiff_t>(dst));
      std::copy(chunk.logprobs.begin(), chunk.logprobs.end(),
                acc.logprobs.begin() + static_cast<std::ptrdiff_t>(dst));
      std::copy(chunk.selected_token_ranks.begin(),
                chunk.selected_token_ranks.end(),
                acc.selected_token_ranks.begin() +
                    static_cast<std::ptrdiff_t>(r.dst_start));
    }

    if (r.final_chunk) {
      // The prompt is fully scored: hand the tensor to the output and forget
      // the request on BOTH sides (:5709-5712).
      prompt_logprobs_dict[r.req_id] = std::move(acc);
      in_progress_prompt_logprobs_.erase(it);
      input_batch_.num_prompt_logprobs.erase(r.req_id);
    }
  }
}

// The height of a request's prompt-logprob tensor: num_prompt_tokens - 1
// (gpu_model_runner.py:5648-5650). Read off the input batch, which is where the
// request's prompt length lives for us.
int GPUModelRunner::prompt_logprob_positions(const std::string& req_id) const {
  const auto it = input_batch_.req_id_to_index.find(req_id);
  VT_CHECK(it != input_batch_.req_id_to_index.end(),
           "prompt_logprob_positions: request is not in the batch");
  return input_batch_.num_prompt_tokens[static_cast<size_t>(it->second)] - 1;
}

// Drop in-progress prompt logprobs for requests the input batch no longer
// carries — an abort mid-prompt. Upstream frees the same state with the request
// object itself (gpu_model_runner.py:1199).
void GPUModelRunner::drop_stale_prompt_logprobs() {
  for (auto it = in_progress_prompt_logprobs_.begin();
       it != in_progress_prompt_logprobs_.end();) {
    if (input_batch_.num_prompt_logprobs.count(it->first) == 0) {
      it = in_progress_prompt_logprobs_.erase(it);
    } else {
      ++it;
    }
  }
}

// The SPEC-DECODE VERIFY half (SPEC-REJECTION I3). Ported from
// gpu/model_runner.py:1069-1077 (the rejection_sampler call) +
// rejection_sampler.py:111 (draft_sampled = input_ids[logits_indices]) +
// postprocess_sampled's num_accepted scatter (model_states/mamba_hybrid.py:
// 290-310, `max(num_sampled, 1)`).
//
// UNREACHABLE ON THE PRODUCTION DEFAULT: the only caller gates on
// exec_state_.step.num_draft_tokens > 0, which requires the scheduler to have
// populated scheduled_spec_decode_tokens, which requires a configured
// SpeculativeConfig. See the greedy accept rule in
// include/vllm/v1/spec_decode/rejection_sampler.h.
ModelRunnerOutput GPUModelRunner::sample_tokens_with_rejection(vt::Tensor& logits) {
  const int num_reqs = exec_state_.num_reqs;
  const StepInputs& step = exec_state_.step;
  const int num_logits = step_num_logits();

  // draft_sampled = input_ids[logits_indices] (rejection_sampler.py:111): the
  // scheduled input token at each expanded row. Row cu[r]+i+1 is request r's
  // i-th draft token (the +1 the accept walk reads).
  std::vector<int32_t> draft_sampled(static_cast<size_t>(num_logits));
  for (int j = 0; j < num_logits; ++j) {
    draft_sampled[static_cast<size_t>(j)] =
        step.input_token_ids[static_cast<size_t>(step.logits_indices[static_cast<size_t>(j)])];
  }
  // A row still consuming its prefill chunk samples and rejects nothing
  // (_get_num_sampled_and_rejected_kernel; our discard mask is the same predicate).
  std::vector<char> chunked_prefilling(static_cast<size_t>(num_reqs), 0);
  for (int i = 0; i < num_reqs && i < static_cast<int>(exec_state_.discard.size()); ++i) {
    chunked_prefilling[static_cast<size_t>(i)] =
        exec_state_.discard[static_cast<size_t>(i)] != 0 ? 1 : 0;
  }

  int max_k = 0;
  for (const int32_t k : step.num_draft_tokens_per_req) max_k = k > max_k ? k : max_k;
  const RejectionSampler rejection_sampler(max_k);
  const RejectionSamplerOutput rs =
      rejection_sampler.forward(queue_, logits, draft_sampled, step.cu_num_logits,
                                chunked_prefilling);

  ModelRunnerOutput out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.sampled_token_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
    out.req_ids.push_back(req_id);
    out.req_id_to_index[req_id] = i;

    // num_accepted_tokens[i] = max(num_sampled, 1) — the GDN recurrent-state
    // slot select reads it next step (spec §3 step 3 / §2.4).
    const int32_t ns = rs.num_sampled[static_cast<size_t>(i)];
    input_batch_.num_accepted_tokens[static_cast<size_t>(i)] = ns > 1 ? ns : 1;
    // Acceptance telemetry: k drafts verified this row, (num_sampled-1) accepted
    // (the trailing token is the bonus/replacement, not a draft). Chunked-prefill
    // rows carry no drafts.
    const int32_t kr = step.num_draft_tokens_per_req.empty()
                           ? 0
                           : step.num_draft_tokens_per_req[static_cast<size_t>(i)];
    if (kr > 0 && !chunked_prefilling[static_cast<size_t>(i)]) {
      spec_drafts_proposed_ += kr;
      spec_drafts_accepted_ += (ns > 1 ? ns - 1 : 0);
      // SPEC-MTP-K-GT-1 (#81): the same accounting split by DEPTH. The rejection
      // sampler accepts a PREFIX of the draft, so with `ns - 1` drafts accepted
      // the draft at 0-based depth d was accepted exactly when `d < ns - 1`.
      // #81's M1 asks for per-depth acceptance, and it is the only signal an
      // acceptance-driven depth policy could read: the aggregate ratio cannot
      // tell "every request accepted its first draft" from "one request accepted
      // three". The vectors stay EMPTY when nothing speculates.
      if (static_cast<int32_t>(spec_drafts_proposed_by_depth_.size()) < kr) {
        spec_drafts_proposed_by_depth_.resize(static_cast<size_t>(kr), 0);
        spec_drafts_accepted_by_depth_.resize(static_cast<size_t>(kr), 0);
      }
      const int32_t accepted_drafts = ns > 1 ? ns - 1 : 0;
      for (int32_t d = 0; d < kr; ++d) {
        spec_drafts_proposed_by_depth_[static_cast<size_t>(d)] += 1;
        if (d < accepted_drafts) {
          spec_drafts_accepted_by_depth_[static_cast<size_t>(d)] += 1;
        }
      }
      // PER-BLOCK acceptance trace (VT_SPEC_TRACE=1), off by default. The
      // aggregate proposed/accepted totals cannot distinguish a diffuse
      // per-block difference from one displaced block, which is exactly the
      // question a cross-draft acceptance delta raises. This prints the whole
      // proposal and what the target did with it, so the answer is read off the
      // log rather than argued. `pos` is the request's token count BEFORE this
      // step's write-back, so blocks from two runs line up by position.
      static const bool spec_trace = [] {
        const char* v = std::getenv("VT_SPEC_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
      }();
      if (spec_trace) {
        const int32_t base = step.cu_num_logits[static_cast<size_t>(i)];
        std::string drafts;
        for (int32_t j = 0; j < kr; ++j) {
          drafts += std::to_string(
              draft_sampled[static_cast<size_t>(base) + static_cast<size_t>(j) + 1]);
          drafts += ' ';
        }
        std::string emitted;
        for (const int32_t t : rs.sampled_token_ids[static_cast<size_t>(i)]) {
          emitted += std::to_string(t);
          emitted += ' ';
        }
        std::fprintf(
            stderr,
            "[SPECTRACE] req=%s pos=%d k=%d ns=%d acc=%d draft=[ %s] emit=[ %s]\n",
            req_id.c_str(),
            static_cast<int>(input_batch_.num_tokens_no_spec[static_cast<size_t>(i)]),
            static_cast<int>(kr), static_cast<int>(ns),
            static_cast<int>(ns > 1 ? ns - 1 : 0), drafts.c_str(), emitted.c_str());
      }
    }

    if (chunked_prefilling[static_cast<size_t>(i)]) {
      out.sampled_token_ids.push_back({});
      continue;
    }
    const std::vector<int32_t>& toks = rs.sampled_token_ids[static_cast<size_t>(i)];
    out.sampled_token_ids.push_back(toks);

    // Write-back: append every emitted token (accepted drafts + the bonus /
    // replacement) to slot i's token row, exactly as the non-spec path does for
    // its single token.
    for (const int32_t tok : toks) {
      const int n = input_batch_.num_tokens_no_spec[static_cast<size_t>(i)];
      const size_t idx = static_cast<size_t>(i) *
                             static_cast<size_t>(input_batch_.max_model_len) +
                         static_cast<size_t>(n);
      if (idx < input_batch_.token_ids_cpu.size()) {
        input_batch_.token_ids_cpu[idx] = tok;
      }
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] = n + 1;
      auto& out_ids = input_batch_.req_output_token_ids[static_cast<size_t>(i)];
      if (out_ids.has_value()) {
        out_ids->push_back(tok);
      }
    }
    if (!toks.empty()) {
      input_batch_.last_sampled_tokens[static_cast<size_t>(i)] = toks.back();
    }
  }
  return out;
}

// ARCH-ONE-SURFACE ROW 6: the pooling counterpart of sample_tokens. Mirror of
// gpu/model_runner.py:1586-1607 (pool instead of sample) over the landed
// PoolingRunner (pool/pooling_runner.py:29-42). The stashed forward result of
// the pooling arch is the [rows, hidden] POST-FINAL-NORM HIDDEN (the model has
// no lm_head — adapters.py:135-151), already gathered at logits_indices on the
// default path, which for LAST pooling IS upstream's
// `hidden_states[input_batch.logits_indices]` (pooling_runner.py:36).
ModelRunnerOutput GPUModelRunner::pool_tokens() {
  const int num_reqs = exec_state_.num_reqs;
  const int64_t hidden = exec_state_.logits.vocab;  // == hidden_size here
  ForwardLogits& fl = exec_state_.logits;
  VT_CHECK(!fl.on_device(),
           "pool_tokens: the pooling forward returns a HOST hidden carrier");

  // One hidden row per request. Default (gather ON): the forward already
  // gathered the per-request last-token rows. VT_LOGITS_GATHER=0: re-gather on
  // host from the full [num_actual_tokens, hidden] rows, exactly as the text
  // host path re-gathers logits.
  std::vector<float> gathered;
  const float* rows = nullptr;
  if (fl.rows == num_reqs) {
    rows = fl.host.data();
  } else {
    gathered.resize(static_cast<size_t>(num_reqs) * static_cast<size_t>(hidden));
    for (int i = 0; i < num_reqs; ++i) {
      const int row = exec_state_.step.logits_indices[static_cast<size_t>(i)];
      std::memcpy(gathered.data() + static_cast<size_t>(i) *
                                        static_cast<size_t>(hidden),
                  fl.host.data() + static_cast<size_t>(row) *
                                       static_cast<size_t>(hidden),
                  static_cast<size_t>(hidden) * sizeof(float));
    }
    rows = gathered.data();
  }
  vt::Tensor hidden_rows = vt::Tensor::Contiguous(
      const_cast<float*>(rows), vt::DType::kF32, vt::Device{vt::DeviceType::kCPU, 0},
      {static_cast<int64_t>(num_reqs), hidden});

  // PoolingMetadata over the GATHERED buffer: one row per sequence (first ==
  // last == i), task embed, activation ON — the unconditional L2 normalize of
  // pooling_runner.py:38 (a per-request use_activation knob is the matryoshka/
  // dimensions residual, named in the row spec).
  vllm::PoolingMetadata md;
  for (int i = 0; i < num_reqs; ++i) {
    md.pooling_cursor.first_token_indices.push_back(i);
    md.pooling_cursor.last_token_indices.push_back(i);
    md.pooling_cursor.prompt_lens.push_back(1);
    md.pooling_cursor.seq_lens.push_back(1);
    md.pooling_cursor.num_scheduled_tokens.push_back(1);
    vllm::PoolingParams pp;
    pp.task = vllm::PoolingTask::kEmbed;
    pp.use_activation = true;
    md.pooling_params.push_back(pp);
    md.tasks.push_back(vllm::PoolingTask::kEmbed);
  }
  vllm::PoolerOutput pooled = pooling_runner_->Pool(hidden_rows, md);
  VT_CHECK(static_cast<int>(pooled.size()) == num_reqs,
           "pool_tokens: pooler must return one vector per request");

  // Validity = the request's whole prompt has been seen (seq_lens == prompt_len,
  // pooling_runner.py:40-41). Our discard mask is the SAME predicate
  // (step.seq_lens[i] < num_tokens_no_spec[i] == still consuming prefill), so a
  // chunked-prefill row reports nullopt and the request keeps running.
  ModelRunnerOutput out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.sampled_token_ids.reserve(static_cast<size_t>(num_reqs));
  out.pooler_output.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
    out.req_ids.push_back(req_id);
    out.req_id_to_index[req_id] = i;
    out.sampled_token_ids.push_back({});  // a pooling step samples NOTHING
    if (exec_state_.discard[static_cast<size_t>(i)] != 0) {
      out.pooler_output.push_back(std::nullopt);
    } else {
      out.pooler_output.push_back(std::move(pooled[static_cast<size_t>(i)]));
    }
  }
  return out;
}

ModelRunnerOutput GPUModelRunner::sample_tokens(
    const std::optional<GrammarOutput>& grammar_output) {
  ModelRunnerOutput out;
  const int num_reqs = exec_state_.num_reqs;
  if (num_reqs == 0) {
    return out;  // 0-token flush step (nothing sampled).
  }

  // POOLING ROUTING (ARCH-ONE-SURFACE ROW 6), mirroring the model-level task
  // split of gpu/model_runner.py:1586-1607: on a POOLING model the step's
  // output is the POOLED DATA, never a sampled token. pooling_runner_ is set
  // iff the registration declares is_pooling_model (ctor), so every text arch
  // takes the sampler path below byte-identically.
  if (pooling_runner_ != nullptr) {
    return pool_tokens();
  }

  std::vector<float> sampled_logits;  // host buffer; outlives the sampler when used
  vt::Tensor logits = assemble_sample_logits(grammar_output, sampled_logits);

  // SAMPLE-PROMPT-LOGPROBS (gpu_model_runner.py:3841-3845): score the prompt
  // rows off the SAME forward result. Done before sampling because sampling
  // mutates `logits` in place; the prompt rows are outside that view, but
  // reading them first keeps the two independent of each other. Inert unless a
  // request asked for prompt logprobs.
  collect_prompt_logprobs(out.prompt_logprobs_dict);

  // SPEC-DECODE VERIFY ROUTING (SPEC-REJECTION I3), mirroring
  // gpu/model_runner.py:1065-1077 (`if input_batch.num_draft_tokens == 0 or
  // self.rejection_sampler is None: sampler(...) else rejection_sampler(...)`).
  // exec_state_.step.num_draft_tokens is > 0 ONLY when the scheduler populated
  // scheduled_spec_decode_tokens, which needs an actually-configured
  // SpeculativeConfig. On the production default it is 0 on every step, so this
  // branch is never taken and the sampler path below is byte-identical.
  if (exec_state_.step.num_draft_tokens > 0) {
    ModelRunnerOutput out_rej = sample_tokens_with_rejection(logits);
    out_rej.prompt_logprobs_dict = std::move(out.prompt_logprobs_dict);
    // SPEC-MTP I5d: propose the next verify step's drafts after committing this
    // step's accepted tokens. The accept accounting lives in num_accepted_tokens
    // (num_sampled = accepted, seeded/overwritten there); num_rejected is derived.
    if (spec_on()) {
      std::vector<int32_t> num_sampled(static_cast<size_t>(num_reqs), 1);
      std::vector<int32_t> num_rejected(static_cast<size_t>(num_reqs), 0);
      for (int i = 0; i < num_reqs; ++i) {
        const int32_t acc = input_batch_.num_accepted_tokens[static_cast<size_t>(i)];
        const int32_t k = exec_state_.step.num_draft_tokens_per_req.empty()
                              ? 0
                              : exec_state_.step.num_draft_tokens_per_req[static_cast<size_t>(i)];
        num_sampled[static_cast<size_t>(i)] = acc;              // == accepted count
        num_rejected[static_cast<size_t>(i)] = k - (acc - 1);   // (1+k) - num_sampled
      }
      propose_drafts(num_sampled, num_rejected);
    }
    return out_rej;
  }

  // SamplingMetadata in the SAME dense [0, num_reqs) order (M1.7; CLOSES the
  // make_sampling_metadata wiring dep). Then Sampler.forward.
  const SamplingMetadata sm = input_batch_.make_sampling_metadata();
  const SamplerOutput sampler_output = sampler_.forward(queue_, logits, sm);

  // ModelRunnerOutput.logprobs (gpu_model_runner.py:3842-3851 / outputs.py):
  // the sampler's batch-wide gather_logprobs result (one row per num_logits
  // position, req order). None unless a request asked for logprobs; the
  // scheduler slices it per request (slice_request). ROAD-V1-C7 SAMPLE-LOGPROBS.
  out.logprobs = sampler_output.logprobs_tensors;

  // Build the ModelRunnerOutput + WRITE the sampled tokens BACK into the
  // InputBatch (upstream postprocess_sampled / post_update) so the next step's
  // prepare_inputs reads them at the decode position.
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.sampled_token_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
    out.req_ids.push_back(req_id);
    out.req_id_to_index[req_id] = i;

    // discard_request_mask (vllm/v1/outputs.py:303 valid_sampled_token_ids[i]
    // .clear()): a request still consuming prefill tokens samples a garbage
    // token at its prefill position — emit EMPTY and skip the write-back, so the
    // scheduler appends no output token for it (scheduler.py:1888-1890).
    if (i < static_cast<int>(exec_state_.discard.size()) &&
        exec_state_.discard[static_cast<size_t>(i)]) {
      out.sampled_token_ids.push_back({});
      continue;
    }

    const std::vector<int32_t>& toks = sampler_output.sampled_token_ids[
        static_cast<size_t>(i)];
    out.sampled_token_ids.push_back(toks);

    // Lab debug: VT_DEBUG_SAMPLED=1 prints every greedy token id. Read ONCE at
    // static-init — never a per-token getenv in the sampling hot loop (matches
    // the "read once, never per-step getenv" discipline stated above).
    static const bool kDebugSampled = [] {
      const char* e = std::getenv("VT_DEBUG_SAMPLED");
      return e != nullptr && e[0] == '1';
    }();
    if (kDebugSampled && !toks.empty()) {
      std::fprintf(stderr, "vt-debug sampled req=%d tok=%d\n", i, toks.front());
    }

    // Write-back: append each sampled token to slot i's token row so it becomes
    // the input at its position next step. num_tokens_no_spec is the next free
    // column (== num_prompt + generated-so-far). T0 non-spec decode = 1 token.
    for (const int32_t tok : toks) {
      const int n = input_batch_.num_tokens_no_spec[static_cast<size_t>(i)];
      const size_t idx = static_cast<size_t>(i) *
                             static_cast<size_t>(input_batch_.max_model_len) +
                         static_cast<size_t>(n);
      if (idx < input_batch_.token_ids_cpu.size()) {
        input_batch_.token_ids_cpu[idx] = tok;
      }
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] = n + 1;
      auto& out_ids = input_batch_.req_output_token_ids[static_cast<size_t>(i)];
      if (out_ids.has_value()) {
        out_ids->push_back(tok);
      }
    }
    // Async-scheduling: record this step's last sampled id per req_state so the
    // next step's combine_sampled_and_draft_tokens can build its decode input id
    // without a sampled-id host round-trip (post_update / states.py, T0 non-spec:
    // one token). Inert on the sync path; the write is cheap and keeps the async
    // buffer coherent for the DGX A/B. (On CUDA this becomes the on-GPU
    // last_sampled_tokens scatter — no D2H.)
    if (!toks.empty()) {
      input_batch_.last_sampled_tokens[static_cast<size_t>(i)] = toks.back();
    }
  }
  // ENG-ASYNC-SCHED W4: keep the DEVICE mirror coherent when the SYNCHRONOUS
  // sampler ran.
  //
  // The mirror exists so the async sampler can leave the ids on the device. But
  // whether the engine drives sample_tokens() or sample_tokens_async() is the
  // ENGINE's choice, not the runner's: LLMEngine::step() (which vllm-bench and
  // every synchronous embedding of the library use) calls this function, while
  // only AsyncLLM's depth-2 step_with_batch_queue calls the async one. Feeding
  // the mirror from just one of them leaves the device combine reading a buffer
  // nobody wrote — every decode row splices a zero, and the output stream is
  // garbage from the first generated token. So whichever sampler ran, the mirror
  // is fed here from the values this loop just wrote.
  //
  // This is an upload rather than a device scatter on purpose: the synchronous
  // path ALREADY has the ids on the host (it downloaded them to build `out`), so
  // there is no round-trip left to remove, and a few hundred bytes of H2D is
  // cheaper and simpler than a second kernel. It also makes the mirror exactly
  // the host array on this path, which is what makes the ON/OFF token-identity
  // gate meaningful.
  if (AsyncDeviceInputs* dinp = get_or_create_async_device_inputs();
      dinp != nullptr && num_reqs > 0) {
    stage_upload(*dinp, dinp->last_sampled,
                 input_batch_.last_sampled_tokens.data(), num_reqs);
  }
  // SPEC-MTP I5d: propose drafts after a plain (no-draft, e.g. first) decode step
  // so the next step verifies them. Each generating row sampled exactly one token
  // (num_sampled=1, num_rejected=0); discarded prefill-chunk rows are skipped
  // inside propose_drafts. No-op unless a speculator is configured.
  if (spec_on()) {
    std::vector<int32_t> num_sampled(static_cast<size_t>(num_reqs), 1);
    std::vector<int32_t> num_rejected(static_cast<size_t>(num_reqs), 0);
    propose_drafts(num_sampled, num_rejected);
  }
  return out;
}

// SPEC-MTP I5d: the k=1 MTP propose, run after each step's sampling
// (gpu/model_runner.py:1455-1489 speculator.propose + set_draft_tokens). Uses the
// stashed target hidden tap + this step's verify attention metadata; the draft
// reuses the target's metadata over its OWN paged KV layer (speculator.py:222-234).
// Stashes the resulting per-request drafts in pending_drafts_ for the engine
// core's out-of-band pull. UNREACHABLE unless a speculator is configured.
void GPUModelRunner::propose_drafts(const std::vector<int32_t>& num_sampled_in,
                                    const std::vector<int32_t>& num_rejected_in) {
  const int num_reqs = exec_state_.num_reqs;
  // SPEC-NGRAM (ROAD-V1-D3): the draft-FREE proposer has no draft model, no hidden
  // tap and no draft KV — it matches the sequence's own suffix. Route first.
  if (use_ngram()) {
    propose_drafts_ngram(num_sampled_in, num_rejected_in);
    return;
  }
  // SPEC-DFLASH D5: the block-diffusion drafter has no MTP draft_model_/
  // draft_attn_kv_ (it recomputes context K/V inline); route to its own propose.
  // SPEC-DSPARK W5: a DSpark draft sets BOTH predicates (dflash_weights_ points
  // at its inherited backbone, so the shared machinery runs unchanged), so it is
  // checked first and only the propose tail differs.
  if (use_dspark()) {
    propose_drafts_dspark(num_sampled_in, num_rejected_in);
    return;
  }
  if (use_dflash()) {
    propose_drafts_dflash(num_sampled_in, num_rejected_in);
    return;
  }
  if (num_reqs == 0 || draft_model_ == nullptr || draft_attn_kv_.empty()) {
    pending_drafts_.reset();
    return;
  }
  VT_CHECK(exec_state_.spec_hidden.tensor.data != nullptr,
           "propose_drafts: missing target hidden tap (hidden_tap not captured)");

  // idx_mapping is identity: our persistent batch is condensed-dense (batch row
  // == req_state slot), so last_sampled/next_prefill index directly by row.
  std::vector<int32_t> idx_mapping(static_cast<size_t>(num_reqs));
  std::iota(idx_mapping.begin(), idx_mapping.end(), 0);

  // num_sampled == 0 tells prepare_prefill_inputs to splice next_prefill_tokens
  // instead of last_sampled (a still-prefilling chunk). Force it for discarded
  // rows (which sampled no committed token this step). next_prefill_tokens[i] is
  // the token at the request's next prefill position; unused for generating rows.
  std::vector<int32_t> num_sampled = num_sampled_in;
  std::vector<int32_t> num_rejected = num_rejected_in;
  std::vector<int32_t> next_prefill(static_cast<size_t>(num_reqs), 0);
  for (int i = 0; i < num_reqs; ++i) {
    if (i < static_cast<int>(exec_state_.discard.size()) &&
        exec_state_.discard[static_cast<size_t>(i)]) {
      num_sampled[static_cast<size_t>(i)] = 0;
      num_rejected[static_cast<size_t>(i)] = 0;
    }
    const int32_t seq_len = exec_state_.step.seq_lens[static_cast<size_t>(i)];
    next_prefill[static_cast<size_t>(i)] =
        input_batch_.token_id(i, seq_len);
  }

  // SPEC-MTP-K-GT-1 (#81): the depth THIS step drafts at. Resolved here, once,
  // and passed down as a value rather than read from spec_config_ inside the
  // propose loop, so a scheduler-supplied depth (upstream decides it at
  // scheduler.py:1122-1126) has one place to come from. Today it is the
  // configured k, and no depth policy exists.
  const int k = num_spec();
  VT_CHECK(k >= 1, "propose_drafts: a configured speculator must draft >= 1");
  VT_CHECK(!draft_attn_kv_.empty() && draft_attn_kv_[0].block_size > 0,
           "propose_drafts: the draft KV group has no block geometry");

  const MtpDraftProposal proposal = MtpProposeDrafts(
      *draft_model_, exec_state_.attn_meta, draft_attn_kv_[0],
      exec_state_.spec_hidden.tensor, exec_state_.step.input_token_ids,
      exec_state_.step.positions, idx_mapping,
      input_batch_.last_sampled_tokens, next_prefill, num_sampled, num_rejected,
      /*max_num_reqs=*/num_reqs, /*num_speculative_tokens=*/k,
      /*max_model_len=*/input_batch_.max_model_len,
      /*block_size=*/static_cast<int>(draft_attn_kv_[0].block_size), queue_);
  const std::vector<int32_t>& drafts = proposal.draft_tokens;
  VT_CHECK(drafts.size() ==
               static_cast<size_t>(num_reqs) * static_cast<size_t>(k),
           "propose_drafts: the MTP propose must return k drafts per request");
  // SPEC-MTP-K-GT-1 (#81): the WORK witness, recorded here because this is the
  // only place that knows both the configured k and the forwards the propose
  // actually ran. The check above is a SHAPE check and cannot stand in for it:
  // a propose that ran one forward and padded all k columns passes it, emits the
  // same greedy tokens, and grows the per-depth counters to size k. The exact
  // relation these two counters carry is
  // `spec_mtp_draft_decode_forwards_ == spec_mtp_propose_calls_ * (k - 1)`,
  // which no draft-list shape can produce. Counted on every reaching call,
  // including the discarded-row case below, because the forwards ran either way.
  ++spec_mtp_propose_calls_;
  spec_mtp_draft_decode_forwards_ += proposal.num_draft_decode_forwards;

  // SPEC-MTP-K-GT-1 (#81): the RESULT witness, and the reason it is computed
  // HERE rather than inside the propose. The equality above counts the forwards
  // a propose RAN, and says nothing about whether their results reached this
  // array. A propose that runs all k-1 forwards, throws the sampled tokens away
  // and writes its step-0 draft into every column satisfies that equality
  // exactly, and was measured green on the whole depth suite. Read at the
  // CONSUMER, on the bytes the proposer handed over, such a row is a pure
  // function of its own first column and this counter stays 0 at every k. The
  // accessor in runner.h states what it does NOT prove: per-column provenance,
  // which the DGX gate owes. A non-zero acceptance count at depth does not show
  // it either, because a padded row is accepted at column 1 whenever the
  // target's greedy continuation repeats its own token. Only an acceptance-RATE
  // comparison against a padded control separates the two.
  for (int i = 0; i < num_reqs; ++i) {
    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k);
    bool varied = false;
    for (int j = 1; j < k && !varied; ++j) {
      varied = drafts[base + static_cast<size_t>(j)] != drafts[base];
    }
    if (varied) {
      ++spec_mtp_proposals_with_varied_drafts_;
      break;
    }
  }

  // Stash each request's k drafts, in draft order, for the out-of-band pull. The
  // DraftTokenIds seam already carries variable-length drafts (the n-gram
  // proposer returns 0..k), so nothing downstream changes shape with depth.
  // A discarded (still-prefilling) row gets no draft — an empty list clears its
  // spec tokens (scheduler.update_draft_token_ids skips prefill-chunk requests).
  DraftTokenIds out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.draft_token_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    out.req_ids.push_back(exec_state_.req_ids[static_cast<size_t>(i)]);
    if (i < static_cast<int>(exec_state_.discard.size()) &&
        exec_state_.discard[static_cast<size_t>(i)]) {
      out.draft_token_ids.push_back({});
    } else {
      const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k);
      out.draft_token_ids.emplace_back(
          drafts.begin() + static_cast<std::ptrdiff_t>(base),
          drafts.begin() + static_cast<std::ptrdiff_t>(base) + k);
    }
  }
  pending_drafts_ = std::move(out);
}

// SPEC-DFLASH D5: the DFlash branch — the shared block body with the 1 + k
// fill-in layout and the parallel per-row argmax (dflash/speculator.py:242-273).
void GPUModelRunner::propose_drafts_dflash(
    const std::vector<int32_t>& num_sampled_in,
    const std::vector<int32_t>& num_rejected_in) {
  // num_sampled_in is derivable from (T_req - num_rejected) per request, so only
  // num_rejected_in drives the append.
  (void)num_sampled_in;
  if (dflash_weights_ == nullptr) {
    pending_drafts_.reset();
    return;
  }
  const int k = dflash_k_;
  const int64_t draft_vocab = dflash_weights_->draft_vocab_size;
  propose_drafts_block(
      num_rejected_in, *dflash_weights_, *dflash_config_,
      /*num_query_per_req=*/1 + k,
      [k, draft_vocab](const std::vector<float>& block_logits, int P,
                       const std::vector<int32_t>& anchors) {
        (void)anchors;  // DFlash's anchor is a bonus token, never a prediction.
        return SampleDflashBlockDrafts(block_logits, P, k, draft_vocab);
      });
}

// SPEC-DSPARK W5: the DSpark branch — the SAME shared block body (context
// accumulation, device KV store, block forward are inherited unchanged) with the
// anchor-aware query layout and the sequential Markov sampler
// (dspark/speculator.py:100-169).
void GPUModelRunner::propose_drafts_dspark(
    const std::vector<int32_t>& num_sampled_in,
    const std::vector<int32_t>& num_rejected_in) {
  (void)num_sampled_in;
  if (dspark_weights_ == nullptr) {
    pending_drafts_.reset();
    return;
  }
  vllm::v1::DsparkBlockLayout layout;
  layout.num_speculative_steps = dflash_k_;
  layout.sample_from_anchor = dspark_sample_from_anchor_;
  const vllm::Qwen3DSparkWeights* weights = dspark_weights_;
  propose_drafts_block(
      num_rejected_in, weights->backbone, *dflash_config_,
      layout.num_query_per_req(),
      [this, layout, weights](const std::vector<float>& block_logits, int P,
                              const std::vector<int32_t>& anchors) {
        VT_CHECK(static_cast<int>(anchors.size()) == P,
                 "propose_drafts_dspark: one anchor token per proposing row");
        return vllm::v1::SampleDsparkBlockDrafts(block_logits, anchors, layout,
                                                 *weights, queue_);
      });
}

// SPEC-NGRAM (ROAD-V1-D3): the draft-FREE propose. Ported from
// vllm/v1/spec_decode/ngram_proposer.py NgramProposer.propose (via NgramPropose).
// Runs the host-side suffix-ngram matcher over each generating request's OWN
// committed context and stashes the variable-length (0..k) drafts for the engine
// core's out-of-band pull. No draft model / hidden tap / draft KV. The drafts are
// verified next step by the target model through the SAME rejection/verify loop as
// MTP/DFlash. UNREACHABLE unless method=="ngram".
void GPUModelRunner::propose_drafts_ngram(
    const std::vector<int32_t>& num_sampled_in,
    const std::vector<int32_t>& /*num_rejected_in*/) {
  const int num_reqs = exec_state_.num_reqs;
  if (num_reqs == 0) {
    pending_drafts_.reset();
    return;
  }
  // valid_ngram_requests filter: a row proposes only if it committed >= 1 token
  // this step AND is not a discarded (still-prefilling) chunk (ngram_proposer.py:
  // 157-172; sampled_token_ids empty -> skip). num_tokens_no_spec / token_ids_cpu
  // are already updated with this step's committed tokens by the caller.
  vllm::v1::spec_decode::NgramConfig cfg;
  cfg.min_n = spec_config_->prompt_lookup_min.value_or(0);
  cfg.max_n = spec_config_->prompt_lookup_max.value_or(0);
  cfg.max_model_len = input_batch_.max_model_len;
  const int k = num_spec();

  std::vector<bool> has_sampled(static_cast<size_t>(num_reqs), false);
  std::vector<int32_t> num_tokens_no_spec(static_cast<size_t>(num_reqs), 0);
  std::vector<const int32_t*> token_rows(static_cast<size_t>(num_reqs), nullptr);
  for (int i = 0; i < num_reqs; ++i) {
    const bool discarded = i < static_cast<int>(exec_state_.discard.size()) &&
                           exec_state_.discard[static_cast<size_t>(i)];
    has_sampled[static_cast<size_t>(i)] =
        !discarded && num_sampled_in[static_cast<size_t>(i)] > 0;
    num_tokens_no_spec[static_cast<size_t>(i)] =
        input_batch_.num_tokens_no_spec[static_cast<size_t>(i)];
    token_rows[static_cast<size_t>(i)] =
        input_batch_.token_ids_cpu.data() +
        static_cast<size_t>(i) * static_cast<size_t>(input_batch_.max_model_len);
  }

  std::vector<std::vector<int32_t>> drafts = vllm::v1::spec_decode::NgramPropose(
      cfg, k, has_sampled, num_tokens_no_spec, token_rows);

  // Stash the per-request drafts (variable length 0..k) for the out-of-band pull.
  // An empty list clears a request's spec tokens (a plain decode next step).
  DraftTokenIds out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.draft_token_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    out.req_ids.push_back(exec_state_.req_ids[static_cast<size_t>(i)]);
    out.draft_token_ids.push_back(std::move(drafts[static_cast<size_t>(i)]));
  }
  pending_drafts_ = std::move(out);
}

// SPEC-DFLASH D5 (DF-ENGINE-INTEGRATION): wire the separately-loaded z-lab DFlash
// draft into the runner. Resolves the D1 multi-tap capture indices
// (target_layer_ids) from the draft config once and sizes the per-request
// combined-feature context accumulation.
void GPUModelRunner::set_dflash_draft(const vllm::Qwen3DFlashWeights* weights,
                                      const vllm::HfConfig* config, int k) {
  dflash_weights_ = weights;
  dflash_config_ = config;
  dflash_k_ = k;
  dflash_tap_layer_ids_.clear();
  if (config != nullptr && config->raw.is_object() &&
      config->raw.contains("dflash_config") &&
      config->raw.at("dflash_config").contains("target_layer_ids")) {
    for (const auto& id : config->raw.at("dflash_config").at("target_layer_ids")) {
      dflash_tap_layer_ids_.push_back(id.get<int32_t>());
    }
  }
  dflash_kv_store_.clear();
  dflash_ctx_len_.clear();
  dflash_ctx_reqid_.clear();
}

// SPEC-DSPARK W5: wire a DSpark draft. The inherited backbone goes through
// set_dflash_draft, so the aux multi-tap capture, the per-request device KV
// store and the context-aware block forward are the SAME code the landed DFlash
// lane runs; only the propose tail (use_dspark()) differs.
void GPUModelRunner::set_dspark_draft(const vllm::Qwen3DSparkWeights* weights,
                                      const vllm::HfConfig* config, int k,
                                      bool sample_from_anchor) {
  dspark_weights_ = weights;
  dspark_sample_from_anchor_ = sample_from_anchor;
  set_dflash_draft(weights == nullptr ? nullptr : &weights->backbone, config, k);
}

// SPEC-DFLASH D5: the DFlash branch of propose_drafts. Ported from
// dflash/speculator.py::propose (:300-413). Where vLLM writes each step's
// combined target features into the draft's incremental KV cache
// (precompute_and_store_context_kv) and excludes rejected positions via
// valid_ctx_end = ctx_end - num_rejected (:518), the inline D3/D4 path instead
// ACCUMULATES the per-request combined-feature context on the host and honors the
// rollback by appending only the num_sampled accepted-prefix features each step
// (the rejected drafts' features are simply never appended). Then it runs the
// non-autoregressive (1+k) block forward over that context (DflashProposeBlock)
// and stashes the k drafts/request. Reachable only when use_dflash().
void GPUModelRunner::propose_drafts_block(
    const std::vector<int32_t>& num_rejected_in,
    const vllm::Qwen3DFlashWeights& backbone, const vllm::HfConfig& config,
    int num_query_per_req,
    const std::function<std::vector<std::vector<int32_t>>(
        const std::vector<float>&, int, const std::vector<int32_t>&)>& sample) {
  const int num_reqs = exec_state_.num_reqs;
  if (num_reqs == 0) {
    pending_drafts_.reset();
    return;
  }
  VT_CHECK(exec_state_.spec_aux.tensor.data != nullptr,
           "propose_drafts_block: missing target aux multi-tap (aux_tap not "
           "captured on the verify forward)");
  const int64_t H = config.hidden_size;
  const int taps = static_cast<int>(dflash_tap_layer_ids_.size());
  const int32_t mask_id = backbone.mask_token_id;
  // The number of NOISE/mask rows after the anchor row. DFlash always emits
  // 1 + k rows (anchor + k masks). DSpark's anchor-as-first-prediction layout
  // emits k rows total, so it carries k - 1 masks.
  const int num_mask_rows = num_query_per_req - 1;
  const StepInputs& step = exec_state_.step;

  if (static_cast<int>(dflash_ctx_len_.size()) < num_reqs) {
    dflash_kv_store_.resize(static_cast<size_t>(num_reqs));
    dflash_ctx_len_.resize(static_cast<size_t>(num_reqs), 0);
    dflash_ctx_reqid_.resize(static_cast<size_t>(num_reqs));
  }

  // 1. Download the [T_total, H×taps] bf16 aux tap to host and cast to f32.
  const int64_t T_total = exec_state_.num_actual_tokens;
  const vt::Tensor& aux = exec_state_.spec_aux.tensor;
  VT_CHECK(aux.shape[0] == T_total && aux.shape[1] == H * taps,
           "propose_drafts_block: aux tap shape mismatch");
  std::vector<uint16_t> aux_bf16(static_cast<size_t>(T_total) *
                                 static_cast<size_t>(H) * static_cast<size_t>(taps));
  vt::Backend& b = vt::GetBackend(queue_.device.type);
  b.Copy(queue_, aux_bf16.data(), aux.data, aux_bf16.size() * sizeof(uint16_t));
  b.Synchronize(queue_);
  std::vector<float> aux_f32(aux_bf16.size());
  for (size_t j = 0; j < aux_bf16.size(); ++j) {
    const uint32_t bits = static_cast<uint32_t>(aux_bf16[j]) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    aux_f32[j] = f;
  }

  // 2. fc(cat(aux)) -> [T_total, H] combined features (combine_hidden_states).
  const std::vector<float> combined = Qwen3DFlashModel::CombineAuxFeatures(
      aux_f32, T_total, backbone, config, queue_);

  // 3. Per request: reset a reused slot, PROJECT+APPEND only the newly-accepted
  //    rows to the persistent per-request KV store (D9 — no full recompute), and
  //    (for a generating row) build the next (1+k) mask block.
  std::vector<int32_t> blk_ids;                // [P*num_query_per_req]
  std::vector<int32_t> blk_pos;                // [P*num_query_per_req]
  std::vector<int32_t> blk_cu = {0};           // [P+1]
  std::vector<int> propose_rows;               // batch rows in the propose batch
  std::vector<int32_t> anchors;                // [P] each proposing row's anchor token

  for (int i = 0; i < num_reqs; ++i) {
    // Reset a reused dense slot (a new request now occupies this row).
    if (dflash_ctx_reqid_[static_cast<size_t>(i)] !=
        exec_state_.req_ids[static_cast<size_t>(i)]) {
      dflash_kv_store_[static_cast<size_t>(i)] =
          Qwen3DFlashModel::MakeDeviceKVStore(config, queue_);
      dflash_ctx_len_[static_cast<size_t>(i)] = 0;
      dflash_ctx_reqid_[static_cast<size_t>(i)] =
          exec_state_.req_ids[static_cast<size_t>(i)];
    }
    const int seg0 = step.query_start_loc[static_cast<size_t>(i)];
    const int seg1 = step.query_start_loc[static_cast<size_t>(i + 1)];
    const int T_req = seg1 - seg0;
    if (T_req <= 0) continue;
    // Rows in ascending absolute position (== committed sequence order).
    std::vector<int> rows(static_cast<size_t>(T_req));
    std::iota(rows.begin(), rows.end(), seg0);
    std::sort(rows.begin(), rows.end(), [&](int a, int c) {
      return step.positions[static_cast<size_t>(a)] <
             step.positions[static_cast<size_t>(c)];
    });
    const int nrej = (i < static_cast<int>(num_rejected_in.size()))
                         ? num_rejected_in[static_cast<size_t>(i)]
                         : 0;
    int append = T_req - nrej;
    if (append < 0) append = 0;
    // Invariant: this step's first committed token sits at absolute position L
    // (== current context length). A violation means the accumulation lost sync
    // (the I5e async-input-combine bug class) — assert rather than corrupt.
    const int64_t L = dflash_ctx_len_[static_cast<size_t>(i)];
    VT_CHECK(step.positions[static_cast<size_t>(rows[0])] == L,
             "propose_drafts_block: context position discontinuity (accumulation "
             "out of sync with the target's committed positions)");
    // Gather this step's `append` accepted-prefix combined features (in ascending
    // position order) + their absolute positions [L, L+append), then project+append
    // to the persistent KV store. This projects ONLY the new rows (D9) — bit-identical
    // to the D5/D7 full recompute of the whole context by per-row projection independence.
    std::vector<float> new_feats;
    new_feats.reserve(static_cast<size_t>(append) * static_cast<size_t>(H));
    std::vector<int32_t> new_pos(static_cast<size_t>(append));
    for (int j = 0; j < append; ++j) {
      const float* src =
          combined.data() + static_cast<size_t>(rows[j]) * static_cast<size_t>(H);
      new_feats.insert(new_feats.end(), src, src + H);
      new_pos[static_cast<size_t>(j)] = static_cast<int32_t>(L + j);
    }
    Qwen3DFlashModel::AppendContextKVDevice(*dflash_kv_store_[static_cast<size_t>(i)], new_feats,
                                            new_pos, backbone, config, queue_);
    dflash_ctx_len_[static_cast<size_t>(i)] = static_cast<int32_t>(L + append);

    // A discarded (still-prefilling chunk) row commits its chunk's features but
    // proposes no draft — it has no valid last_sampled anchor yet.
    const bool discarded = i < static_cast<int>(exec_state_.discard.size()) &&
                           exec_state_.discard[static_cast<size_t>(i)];
    if (discarded) continue;

    // Block: anchor = last_sampled (the bonus/last committed token, re-embedded),
    // then k mask tokens; positions L' .. L'+k (L' = the new context length).
    const int64_t Lp = dflash_ctx_len_[static_cast<size_t>(i)];
    const int32_t anchor = input_batch_.last_sampled_tokens[static_cast<size_t>(i)];
    blk_ids.push_back(anchor);
    blk_pos.push_back(static_cast<int32_t>(Lp));
    for (int j = 1; j <= num_mask_rows; ++j) {
      blk_ids.push_back(mask_id);
      blk_pos.push_back(static_cast<int32_t>(Lp + j));
    }
    blk_cu.push_back(static_cast<int32_t>(blk_ids.size()));
    propose_rows.push_back(i);
    anchors.push_back(anchor);
  }

  // 4. Non-autoregressive (1+k) block propose over the PERSISTENT context KV store.
  //    Concatenate the propose rows' per-layer stores (ctx_cu order) into one combined
  //    PrecomputedContextKV and run the context-aware block forward with NO re-projection.
  DraftTokenIds out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.draft_token_ids.assign(static_cast<size_t>(num_reqs), {});
  for (int i = 0; i < num_reqs; ++i)
    out.req_ids.push_back(exec_state_.req_ids[static_cast<size_t>(i)]);

  // VT_SPEC_TRACE=1: how many rows actually proposed this step, and what the
  // first row's drafts were. The aggregate acceptance trace on the VERIFY side
  // cannot distinguish "proposed nothing" from "proposed and everything was
  // rejected", which is exactly the question the first DSpark e2e raised.
  static const bool propose_trace = [] {
    const char* v = std::getenv("VT_SPEC_TRACE");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  if (!propose_rows.empty()) {
    const int P = static_cast<int>(propose_rows.size());
    // D11 A-wire: run the block forward straight off the per-request DEVICE stores
    // (ForwardBlockLogitsWithDeviceKV) — no host concat, no D<->H round-trip. The
    // device-side IndexCopy concat reproduces the exact ascending-position layout
    // the D9 host path built, so tokens+acceptance are bit-identical. ctx_cu =
    // running sums of each store's on-device context length (DeviceKVNumCtx).
    std::vector<vllm::DflashDeviceKVStore*> stores;
    stores.reserve(static_cast<size_t>(P));
    std::vector<int32_t> ctx_cu = {0};
    int64_t total_ctx = 0;
    for (int r = 0; r < P; ++r) {
      vllm::DflashDeviceKVStore* st =
          dflash_kv_store_[static_cast<size_t>(propose_rows[static_cast<size_t>(r)])].get();
      stores.push_back(st);
      total_ctx += Qwen3DFlashModel::DeviceKVNumCtx(*st);
      ctx_cu.push_back(static_cast<int32_t>(total_ctx));
    }
    const auto t_fwd0 = std::chrono::steady_clock::now();
    // SPEC-DFLASH2 W3 (#1314): a DFlash2 draft ALSO captures `final_out` off
    // this forward -- the post-final-norm hidden the candidate selector's
    // `hidden_projection` reads. Upstream's `_generate_draft` takes both from
    // one forward, and it must: the selector projects the SAME hidden states
    // these logits came from. A DFlash1 draft passes nullptr and this call is
    // byte-for-byte what it was.
    //
    // COST, named rather than discovered: asking for `final_out` takes this
    // forward off the single-request PAGED fast path, which is guarded on
    // `final_out == nullptr` (ForwardBlockLogitsWithDeviceKV). That costs a
    // DFlash2 draft the CUDA-graph draft step until W4 computes the candidates
    // inside the forward instead of after it. It costs a DFlash1 draft nothing,
    // and this row claims no throughput number.
    std::vector<float> block_hidden;
    const std::vector<float> block_logits =
        Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
            stores, ctx_cu, blk_ids, blk_pos, blk_cu, backbone, config, queue_,
            nullptr, backbone.IsDflash2() ? &block_hidden : nullptr);
    const auto t_fwd1 = std::chrono::steady_clock::now();
    // SPEC-DFLASH2 W3 (#1314): the PRODUCTION boundary of the DFlash2 port, one
    // step further out than W2 left it. The block forward above ran the draft's
    // grouped dynamic convolution; `Dflash2SelectCandidates` then runs the
    // target head's top-K, the codebook lattice and the edge scores -- the SAME
    // function `DflashProposeBlock` calls, so this path and the one a gate can
    // drive are one implementation rather than two. What is still missing is the
    // PATH WALK, refused BY NAME rather than silently replaced by the DFlash1
    // per-slot argmax `sample` is about to apply. See RefuseDflash2PathWalk for
    // why a fallback is not admissible.
    if (backbone.IsDflash2()) {
      const vllm::v1::Dflash2ProposeState selected = vllm::v1::Dflash2SelectCandidates(
          block_logits, block_hidden, anchors, P, num_query_per_req - 1, backbone,
          config, queue_);
      vllm::v1::RefuseDflash2PathWalk(backbone, selected);
    }
    const std::vector<std::vector<int32_t>> drafts = sample(block_logits, P, anchors);
    const auto t_smp1 = std::chrono::steady_clock::now();
    if (propose_trace) {
      // Splits the draft step into the parallel backbone forward and the
      // sampler. For DSpark the sampler is a k-iteration host loop, each
      // iteration a Markov GEMV plus a device->host download plus a host argmax
      // over the draft vocab; upstream captures the WHOLE draft step in one CUDA
      // graph instead (dspark/speculator.py:22-24).
      std::fprintf(stderr,
                   "[spec-phase] backbone=%.2fms sample=%.2fms logits=%zu\n",
                   std::chrono::duration<double, std::milli>(t_fwd1 - t_fwd0).count(),
                   std::chrono::duration<double, std::milli>(t_smp1 - t_fwd1).count(),
                   block_logits.size());
    }
    for (int r = 0; r < P; ++r) {
      const int row = propose_rows[static_cast<size_t>(r)];
      out.draft_token_ids[static_cast<size_t>(row)] = drafts[static_cast<size_t>(r)];
    }
    if (propose_trace) {
      std::string first;
      for (int32_t id : drafts[0]) {
        first += std::to_string(id);
        first += ' ';
      }
      std::fprintf(stderr, "[spec-propose] rows=%d nqpr=%d drafts/row=%zu first=[%s]\n",
                   P, num_query_per_req, drafts[0].size(), first.c_str());
    }
  }
  if (propose_trace && propose_rows.empty()) {
    std::fprintf(stderr, "[spec-propose] NO proposing rows this step (num_reqs=%d)\n",
                 num_reqs);
  }
  pending_drafts_ = std::move(out);
}

GPUModelRunner::~GPUModelRunner() {
  // Release the lazily-created async-output copy queue (gpu_model_runner.py has
  // no explicit teardown; our vt::Queue owns a CUDA stream that must be freed).
  if (async_copy_queue_.id != 0) {
    vt::DestroyQueue(async_copy_queue_);
  }
  // W4 device mirror. Freed here rather than leaked like the scratch pool: these
  // are per-runner, and a serving process can construct more than one runner.
  if (async_device_inputs_ != nullptr) {
    vt::Backend& b = vt::GetBackend(queue_.device.type);
    for (int32_t* p : {async_device_inputs_->last_sampled,
                       async_device_inputs_->prefill_len,
                       async_device_inputs_->query_start_loc,
                       async_device_inputs_->seq_lens,
                       async_device_inputs_->input_ids,
                       async_device_inputs_->ops}) {
      if (p != nullptr) b.Free(p);
    }
  }
}

vt::Queue& GPUModelRunner::get_or_create_async_copy_queue() {
  // _get_or_create_async_output_copy_stream (gpu_model_runner.py:1137-1141): a
  // single dedicated copy stream on the runner's device, created on first use so
  // the synchronous production path never allocates it.
  if (async_copy_queue_.id == 0) {
    async_copy_queue_ = vt::CreateQueue(queue_.device);
  }
  return async_copy_queue_;
}

AsyncOutputPool& GPUModelRunner::get_or_create_async_output_pool() {
  // Persistent per-step overlap buffers, sized to the batch bound (max_num_reqs)
  // and pre-seeded with a few slots (depth-2 in flight + grammar-deferral
  // headroom), created on first async use. This removes ALL per-step
  // cudaMalloc/cudaHostAlloc/cudaEventCreate from sample_tokens_async — the raw
  // device-syncing allocator calls that serialized the depth-2 overlap.
  if (async_output_pool_ == nullptr) {
    const int cap = max_num_reqs_ > 0 ? max_num_reqs_ : 1;
    async_output_pool_ =
        std::make_unique<AsyncOutputPool>(queue_.device, cap, /*initial_slots=*/4);
  }
  return *async_output_pool_;
}

// ─── ENG-ASYNC-SCHED W4: CUDA device-resident async inputs ───────────────────

// W4, DEFAULT ON since the 2026-08-06 CORRECTNESS flip (ROW-SERVE-ASYNC-LLM P0).
// It landed default OFF (W3's A/B precedent: prove the speed win before flipping),
// and the c16 A/B came back NEUTRAL (0.999x — the drain MOVE overlaps only host
// prep). But the token-exactness probe that A/B ran uncovered a latent, SHIPPING
// correctness bug the OFF path CANNOT avoid: on the async serving loop
// (AsyncLLM -> step_with_batch_queue -> sample_tokens_async) the sampled token is
// not written to token_ids_cpu synchronously (see sample_tokens_async), so the
// next step's prepare_inputs reads a stale/zero decode-row placeholder and RELIES
// on the device combine. On the OFF integrated path the combine patches
// step.input_token_ids on the MAIN QUEUE while the Qwen3.5 decode graph reads that
// same host vector on the CPU (BuildPaddedDecode -> CopyInPlace -> EmbedInto's
// host->device upload) with NO intervening sync — an unsynchronized
// device-write/host-read race that nondeterministically embeds the zero
// placeholder and degenerates batch-1 greedy decode into repeated token-0 garbage.
// Making step.input_token_ids device-resident (this mirror) routes the combine's
// output into the embed via ApplyDeviceTokenIdsOverride, main-queue-ordered after
// the combine, so the embed never does the racing host read. The only OFF-path
// alternative is a per-step Synchronize after the combine — which reintroduces the
// host sync the async path exists to remove (a c16 REGRESSION) and is a GB10-only
// band-aid, not parity. This is exactly what upstream does on every platform
// (states.py:64 keeps prev_sampled_token_ids a GPU tensor; gpu_model_runner.py's
// _prepare_input_ids gathers on the GPU), so flipping the default ON is MIRROR-vLLM
// parity, speed-neutral, and gated: VT_ASYNC_DEVICE_MIRROR=0 is the rollback to the
// (racy) host-array path for an A/B. The async-serving token-exact gate
// (test_qwen36_async_serving.cpp) is RED on =0 and GREEN on the default.
//
// Distinct from VT_ASYNC_RUNNER, which would also turn off async scheduling
// itself; keeping them separate is what makes an honest A/B of W4 alone possible —
// same binary, same scheduler, one mechanism.
#ifdef VLLM_CPP_CUDA
// Guarded with its only use below: on a CPU build the mirror cannot exist, and
// an unused static function is a -Werror=unused-function break there. DEFAULT ON:
// on unless VT_ASYNC_DEVICE_MIRROR is explicitly "0" (the rollback), mirroring the
// AsyncRunnerFlagIsOn convention.
static bool AsyncDeviceMirrorEnvDefault() {
  const char* value = std::getenv("VT_ASYNC_DEVICE_MIRROR");
  return value == nullptr || value[0] != '0';
}
#endif

bool GPUModelRunner::async_device_mirror() const {
  if (async_device_mirror_cached_ >= 0) return async_device_mirror_cached_ != 0;
  bool on = false;
#ifdef VLLM_CPP_CUDA
  // Engage on any real CUDA GPU, integrated OR discrete — NOT the CPU backend.
  //  - DISCRETE (separate memory, !UnifiedMemory): the mirror is REQUIRED, because
  //    the host fallback would main-stream Synchronize to read the sampled ids.
  //  - INTEGRATED (GB10, UnifiedMemory AND is_integrated_gpu): the mirror is not
  //    required for addressability (host arrays ARE device-addressable), but
  //    making last_sampled_tokens device-resident removes the host condense<->
  //    scatter read-after-write that pins the async drain to the TOP of
  //    execute_model, letting the bulk host prep overlap the GPU tail (a drain
  //    MOVE) AND — the reason it is now DEFAULT ON — removes the unsynchronized
  //    combine-write/host-read race the OFF path has with the decode-graph embed
  //    (ROW-SERVE-ASYNC-LLM P0). VT_ASYNC_DEVICE_MIRROR=0 rolls back to it.
  //  - The CPU backend (UnifiedMemory AND not an integrated GPU) keeps the in-place
  //    host path — mirroring there would only add copies with no GPU tail to hide.
  // is_integrated_gpu() is already read at the combine/scatter sites in this file,
  // so it introduces no new device-type token to the leakage ratchet.
  on = async_input_combine_ && AsyncDeviceMirrorEnvDefault() &&
       (!vt::GetBackend(queue_.device.type).UnifiedMemory() ||
        vllm::platforms::GetPlatform(queue_.device.type).is_integrated_gpu());
#endif
  async_device_mirror_cached_ = on ? 1 : 0;
  return on;
}

// VT_ASYNC_EXECUTOR (decode-graph slot double-buffer, ENG-ASYNC-SCHED c16/c32
// overlap unlock). DEFAULT OFF — an opt-in speed lever. Engages only where the
// depth-2 moved drain and the decode graph both exist: the device mirror path on
// a real CUDA GPU. On the CPU backend / mirror-OFF path the drain does not move
// and there is no decode graph, so the lever is inert. Memoized. Reading a plain
// "not 0" env keeps a value of "1" the canonical enable; anything else (unset,
// "0") leaves the drain in place, which is byte-identical production.
bool GPUModelRunner::async_executor() const {
  if (async_executor_cached_ >= 0) return async_executor_cached_ != 0;
  bool on = false;
#ifdef VLLM_CPP_CUDA
  const char* value = std::getenv("VT_ASYNC_EXECUTOR");
  on = value != nullptr && value[0] == '1' && value[1] == '\0' &&
       async_device_mirror();
#endif
  async_executor_cached_ = on ? 1 : 0;
  return on;
}

GPUModelRunner::AsyncDeviceInputs*
GPUModelRunner::get_or_create_async_device_inputs() {
  if (!async_device_mirror()) return nullptr;
  if (async_device_inputs_ != nullptr) return async_device_inputs_.get();

  const int reqs = max_num_reqs_ > 0 ? max_num_reqs_ : 1;
  const int toks = max_num_batched_tokens_ > 0 ? max_num_batched_tokens_ : 1;
  vt::Backend& b = vt::GetBackend(queue_.device.type);
  auto dev = std::make_unique<AsyncDeviceInputs>();
  dev->max_reqs = reqs;
  dev->input_ids_capacity = toks;

  auto alloc_i32 = [&](int64_t count) {
    void* p = b.Alloc(static_cast<size_t>(count) * sizeof(int32_t));
    b.Memset(queue_, p, 0, static_cast<size_t>(count) * sizeof(int32_t));
    return static_cast<int32_t*>(p);
  };
  dev->last_sampled = alloc_i32(reqs);
  dev->prefill_len = alloc_i32(reqs);
  dev->query_start_loc = alloc_i32(static_cast<int64_t>(reqs) + 1);
  dev->seq_lens = alloc_i32(reqs);
  dev->input_ids = alloc_i32(toks);
  dev->ops = alloc_i32(4LL * reqs);

  // The mirror starts from whatever the host array already holds. In production
  // that is all zeros (no request has been admitted yet), but seeding from the
  // host makes the mirror correct even if a runner is switched on mid-flight,
  // and it costs one copy for the process.
  b.Copy(queue_, dev->last_sampled, input_batch_.last_sampled_tokens.data(),
         static_cast<size_t>(reqs) * sizeof(int32_t));
  // The seeds recorded so far are already reflected by that copy; dropping them
  // here keeps the log from replaying them a second time.
  input_batch_.last_sampled_ops.clear();

  async_device_inputs_ = std::move(dev);
  return async_device_inputs_.get();
}

void GPUModelRunner::stage_upload(AsyncDeviceInputs& dev, int32_t* dst,
                                  const int32_t* src, int64_t count) {
  (void)dev;
  if (count <= 0) return;
  // Copied straight from the caller's PAGEABLE host buffer, deliberately: for a
  // pageable source the driver stages the bytes before cudaMemcpyAsync returns,
  // so the caller may reuse or destroy that buffer immediately, while the copy
  // itself stays ordered on the queue.
  //
  // A shared PINNED staging buffer would be the usual optimization and is WRONG
  // here: pinned copies are truly asynchronous, so the next upload's memcpy into
  // the shared buffer could overwrite bytes an in-flight DMA had not yet read,
  // and with a depth-2 scheduler that window spans steps. Making that safe needs
  // per-upload regions plus an event per step; these arrays are a few kilobytes
  // on the front of a step, so the staged pageable copy is the better trade.
  vt::GetBackend(queue_.device.type)
      .Copy(queue_, dst, src, static_cast<size_t>(count) * sizeof(int32_t));
}

void GPUModelRunner::replay_last_sampled_ops(AsyncDeviceInputs& dev) {
#ifdef VLLM_CPP_CUDA
  std::vector<InputBatch::LastSampledOp>& ops = input_batch_.last_sampled_ops;
  if (ops.empty()) return;
  // Flatten to (kind, a, b, value) quads. The log is bounded by the number of
  // admissions/removals in one step, so it fits the [4 * max_num_reqs] buffer;
  // if a step ever exceeded that, replaying a truncated prefix would silently
  // corrupt the mirror, so drain it in whole chunks instead.
  const int64_t cap_ops = 4LL * dev.max_reqs;
  size_t done = 0;
  std::vector<int32_t> flat;
  while (done < ops.size()) {
    const size_t chunk =
        std::min(ops.size() - done, static_cast<size_t>(dev.max_reqs));
    flat.clear();
    flat.reserve(chunk * 4);
    for (size_t i = 0; i < chunk; ++i) {
      const InputBatch::LastSampledOp& op = ops[done + i];
      flat.push_back(op.kind);
      flat.push_back(op.a);
      flat.push_back(op.b);
      flat.push_back(op.value);
    }
    VT_CHECK(static_cast<int64_t>(flat.size()) <= cap_ops,
             "async device mirror: structural-op chunk exceeds its buffer");
    stage_upload(dev, dev.ops, flat.data(), static_cast<int64_t>(flat.size()));
    vt::cuda::LaunchApplyLastSampledOps(queue_, dev.last_sampled, dev.ops,
                                        static_cast<int>(chunk));
    done += chunk;
  }
  ops.clear();
#else
  (void)dev;
#endif
}

std::unique_ptr<AsyncModelRunnerOutput> GPUModelRunner::sample_tokens_async(
    const std::optional<GrammarOutput>& grammar_output) {
  // ARCH-ONE-SURFACE ROW 6: pooling models resolve async scheduling OFF
  // (config/vllm.py:1068-1073 mirror in LoadedEngine::ResolveAsyncEnabled), so
  // the depth-2 async sampler must never see one — refuse loudly rather than
  // run the device sampler over hidden states.
  VT_CHECK(pooling_runner_ == nullptr,
           "sample_tokens_async: pooling models use the synchronous scheduler "
           "(async scheduling is disabled for pooling, config/vllm.py:1068)");
  // When async is NOT engaged (production default), degenerate to the byte-
  // identical synchronous path wrapped as a ready output — so a caller in the
  // depth-2 loop can always call sample_tokens_async without branching, yet the
  // sync behavior is unchanged. The device-resident overlap path below runs only
  // under VT_ASYNC_RUNNER / set_async_input_combine.
  if (!async_input_combine_) {
    return std::make_unique<ReadyModelRunnerOutput>(sample_tokens(grammar_output));
  }

  const vt::Device dev = queue_.device;
  const int num_reqs = exec_state_.num_reqs;

  ModelRunnerOutput skeleton;
  AsyncOutputPool& pool = get_or_create_async_output_pool();
  vt::Queue& copy_q = get_or_create_async_copy_queue();
  if (num_reqs == 0) {
    // 0-token flush step: nothing sampled. Return an already-complete async
    // output (zero rows) so the seam is uniform. A pool slot is borrowed and
    // released with no copy — no per-step allocation.
    AsyncOutputSlot* slot = pool.Acquire();
    return std::make_unique<AsyncGPUModelRunnerOutput>(
        std::move(skeleton), dev, pool, slot, /*num_reqs=*/0, queue_, copy_q);
  }

  std::vector<float> sampled_logits;
  vt::Tensor logits = assemble_sample_logits(grammar_output, sampled_logits);
  // SAMPLE-PROMPT-LOGPROBS: same seam as the synchronous path. The prompt rows
  // are read off this step's forward result HERE, synchronously, because that
  // is where the logits still are — the async output only defers the sampled
  // IDS. Inert unless a request asked.
  collect_prompt_logprobs(skeleton.prompt_logprobs_dict);
  const SamplingMetadata sm = input_batch_.make_sampling_metadata();

  // Sample DEVICE-RESIDENT: the sampler writes the ids into the pool slot's
  // PERSISTENT device buffer (the pool guarantees the slot is not reused until
  // this step's async output releases it, so a depth-2 next step cannot tear this
  // snapshot). The sampler writes it device-resident (all-greedy: no host
  // download inside the sampler), and the async output issues the single
  // sampled-id copy to host on the copy queue. Mirrors sampler_output.
  // sampled_token_ids staying GPU-side (async_utils.py:31). No per-step
  // cudaMalloc here — the slot buffer is reused across steps.
  AsyncOutputSlot* slot = pool.Acquire();
  void* dev_ids = slot->device_sampled_ids;
  vt::Tensor dev_ids_t = vt::Tensor::Contiguous(
      dev_ids, vt::DType::kI64, dev, {static_cast<int64_t>(num_reqs)});
  (void)sampler_.forward(queue_, logits, sm, &dev_ids_t);

  // post_update (input_batch.py:457-543 post_update / states.py): record this
  // step's last sampled id per req_state so the NEXT step's
  // combine_sampled_and_draft_tokens builds its decode input id, and advance the
  // write-back column counter. The token VALUE append to token_ids_cpu /
  // req_output_token_ids is DELETED on the async path (the runner does the
  // post_update only; host token bookkeeping for detok/penalties is fed by the
  // scheduler's update_from_output when get_output() materializes).
  //
  skeleton.req_ids.reserve(static_cast<size_t>(num_reqs));
#ifdef VLLM_CPP_CUDA
  // W4 device-resident scatter. Preferred whenever the mirror is engaged
  // (async_device_mirror(): CUDA + VT_ASYNC_DEVICE_MIRROR, INTEGRATED OR DISCRETE):
  // write each row's sampled id into the DEVICE mirror (dinp->last_sampled) on the
  // MAIN queue, main-stream-ordered with the next step's combine, so the sampled
  // ids never round-trip the host. On the INTEGRATED path this is what keeps the
  // sampled tokens OFF the host array update_states' condense reorders (the drain-
  // move lever). The host `last_sampled_tokens` is deliberately NOT updated here —
  // reading the ids back is the cost being removed; its rows still move with
  // condense/swap and those moves are replayed onto the mirror, but its VALUES are
  // stale and nothing reads them (the sampled ids reach the engine through the
  // async output's own copy, as upstream does). Runs OUTSIDE any CUDA-graph capture.
  if (AsyncDeviceInputs* dinp = get_or_create_async_device_inputs();
      dinp != nullptr) {
    vt::cuda::LaunchScatterLastSampled(queue_, dinp->last_sampled,
                                       static_cast<const int64_t*>(dev_ids),
                                       /*idx_mapping=*/nullptr, num_reqs);
    for (int i = 0; i < num_reqs; ++i) {
      const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
      skeleton.req_ids.push_back(req_id);
      skeleton.req_id_to_index[req_id] = i;
      if (i < static_cast<int>(exec_state_.discard.size()) &&
          exec_state_.discard[static_cast<size_t>(i)]) {
        continue;
      }
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] += 1;
    }
  } else if (vllm::platforms::GetPlatform(dev.type).is_integrated_gpu()) {
    // DEVICE scatter, HOST-ARRAY residence (W3 DGX leaf; mirror OFF = GB10 default):
    // write each row's sampled id into the device-ADDRESSABLE host
    // last_sampled_tokens (pageable on GB10's UMA) on the MAIN queue, main-stream-
    // ordered with the next step's device combine, so the ids never round-trip the
    // host. DELETES the pre-scatter `Synchronize` the host-fallback needed (its ONLY
    // purpose was making the host read of dev_ids well-defined). idx_mapping
    // identity (nullptr); the host bookkeeping below needs no device read. Because
    // this is the array condense reorders, its scatter pins the drain to
    // execute_model's top (the mirror path lifts that). is_integrated_gpu()
    // decouples a future discrete GPU (false -> host bookkeeping below).
    vt::cuda::LaunchScatterLastSampled(
        queue_, input_batch_.last_sampled_tokens.data(),
        static_cast<const int64_t*>(dev_ids), /*idx_mapping=*/nullptr, num_reqs);
    for (int i = 0; i < num_reqs; ++i) {
      const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
      skeleton.req_ids.push_back(req_id);
      skeleton.req_id_to_index[req_id] = i;
      // discard_request_mask: a still-prefilling request produced no output
      // token this step, so do NOT advance its write-back column (bumping it
      // would desync num_tokens from the scheduler). The stale last_sampled row
      // the scatter wrote is never read while the request is prefilling (combine
      // skips seq_len <= prefill_len) and is overwritten on its first decode.
      if (i < static_cast<int>(exec_state_.discard.size()) &&
          exec_state_.discard[static_cast<size_t>(i)]) {
        continue;
      }
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] += 1;
    }
  } else
#endif
  {
    // HOST path (CPU backend): the sampler already ran synchronously, but on any
    // non-CUDA async-degeneration backend `Synchronize` keeps the host read of
    // `dev_ids` well-defined (no read-before-write race). This is the ONE ordering
    // cost the host-array combine pays; the CUDA device-kernel branch above removes
    // it. Runs OUTSIDE any CUDA-graph capture (post-sample host-side).
    vt::GetBackend(dev.type).Synchronize(queue_);
    const int64_t* ids = static_cast<const int64_t*>(dev_ids);
    for (int i = 0; i < num_reqs; ++i) {
      const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
      skeleton.req_ids.push_back(req_id);
      skeleton.req_id_to_index[req_id] = i;
      // discard_request_mask: skip the write-back for a still-prefilling request
      // (no output token generated this step).
      if (i < static_cast<int>(exec_state_.discard.size()) &&
          exec_state_.discard[static_cast<size_t>(i)]) {
        continue;
      }
      input_batch_.last_sampled_tokens[static_cast<size_t>(i)] =
          static_cast<int32_t>(ids[i]);
      input_batch_.num_tokens_no_spec[static_cast<size_t>(i)] += 1;
    }
  }

  // discard_request_mask (gpu_model_runner.py:3625-3628 -> outputs.py:303): the
  // rows get_output() must clear to empty. Prefill-chunk requests sampled a
  // garbage token at their prefill position; the scheduler must see EMPTY ids.
  std::vector<int32_t> invalid_req_indices;
  for (int i = 0; i < num_reqs; ++i) {
    if (i < static_cast<int>(exec_state_.discard.size()) &&
        exec_state_.discard[static_cast<size_t>(i)]) {
      invalid_req_indices.push_back(i);
    }
  }

  // ENG-ASYNC-SCHED depth-2 LIFETIME GUARD: this path issued the sampled-id D2H
  // on the COPY queue and left the forward / sample / scatter on the MAIN queue
  // UNSYNCED (that overlap is the whole point). Those kernels still reference
  // exec_state_ and write input_batch_.last_sampled_tokens, so the NEXT
  // execute_model() must drain the main queue before it resets exec_state_ or
  // mutates input_batch_. Flag it. (The host-fallback branch above already
  // Synchronized, so its flag is a no-op next step — harmless.)
  async_forward_in_flight_ = true;

  // Issue the non-blocking sampled-id D2H on the COPY queue + record the event.
  // The async output BORROWS the pool slot (device buffer already holds the
  // argmax ids) and releases it on consume — no per-step free.
  return std::make_unique<AsyncGPUModelRunnerOutput>(
      std::move(skeleton), dev, pool, slot, num_reqs, queue_, copy_q,
      std::move(invalid_req_indices));
}

// ─── KV-EXTERNAL-CACHE (LMCache) worker-side store/load ──────────────────────
//
// The full-attention page is [2, block_size, num_kv_heads, head_size] per block
// (backend.h:241), so within a block the K plane is the first half and the V
// plane the second — each exactly [block_size, hidden_dim] contiguous bytes,
// which is byte-for-byte the LMCache KV_2LTD per-layer plane the connector's
// PutKv2ltd/GetKv2ltd expect. We store/load each half at its own page offset, so
// an our-store -> our-load cycle restores the block bit-for-bit (the whole point
// of the output-invariance gate). Device<->host movement goes through the
// backend Copy (a UMA memcpy on GB10), synchronized before the bytes are read.

namespace {
// The full-attention page geometry a store/load needs. num_layers == the number
// of full-attention KV buffers, hidden_dim == num_kv_heads*head_size, plane_bytes
// == one K (or V) plane, page_bytes == the whole [K|V] block.
struct FaGeom {
  int num_layers = 0;
  int block_size = 0;
  int hidden_dim = 0;
  std::size_t elem_size = 0;
  std::size_t plane_bytes = 0;
  std::size_t page_bytes = 0;
};
}  // namespace

void GPUModelRunner::ConnectorLoadExternalKv() {
  using kv_offload::lmcache::LMCacheConnector;
  auto* lm = dynamic_cast<LMCacheConnector*>(kv_connector_);
  if (lm == nullptr) return;  // a base/non-worker connector: nothing to load
  std::vector<kv_offload::lmcache::LmcacheLoadJob> loads =
      lm->TakeConnectorLoads();
  if (loads.empty()) return;
  VT_CHECK(!attn_kv_.empty(),
           "LMCache load: model has no full-attention KV group");

  FaGeom g;
  g.num_layers = static_cast<int>(full_attn_buf_.size());
  g.block_size = static_cast<int>(attn_kv_[0].block_size);
  g.hidden_dim = static_cast<int>(attn_kv_[0].num_kv_heads * attn_kv_[0].head_size);
  g.elem_size = vt::SizeOf(attn_kv_[0].dtype);
  g.plane_bytes = static_cast<std::size_t>(g.block_size) *
                  static_cast<std::size_t>(g.hidden_dim) * g.elem_size;
  g.page_bytes = 2 * g.plane_bytes;
  VT_CHECK(g.page_bytes == static_cast<std::size_t>(fa_page_size_bytes_),
           "LMCache load: full-attention page is padded/asymmetric — refusing "
           "to move KV under an unexpected layout");
  const auto& c = lm->config();
  VT_CHECK(c.num_layers == g.num_layers && c.hidden_dim == g.hidden_dim &&
               c.chunk_tokens == g.block_size && c.elem_size == g.elem_size,
           "LMCache load: connector geometry disagrees with the runner's KV "
           "cache — refusing to load (identity safety)");

  vt::Backend& backend = vt::GetBackend(queue_.device.type);
  std::vector<std::string> kp, vp;
  for (const auto& job : loads) {
    VT_CHECK(job.keys.size() == job.gpu_block_ids.size(),
             "LMCache load: key/block-id count mismatch");
    for (std::size_t j = 0; j < job.keys.size(); ++j) {
      kp.clear();
      vp.clear();
      // A chunk the scheduler recorded as a HIT that is now absent is a real
      // bug (torn store, evicted mid-decode), never silently recomputed here.
      const bool got = lm->LoadChunk(job.keys[j], &kp, &vp);
      VT_CHECK(got,
               "LMCache load: a chunk recorded as an external-prefix hit is "
               "absent on the server");
      VT_CHECK(static_cast<int>(kp.size()) == g.num_layers &&
                   static_cast<int>(vp.size()) == g.num_layers,
               "LMCache load: plane count != num_layers");
      const int block_id = job.gpu_block_ids[j];
      for (int l = 0; l < g.num_layers; ++l) {
        VT_CHECK(kp[static_cast<std::size_t>(l)].size() == g.plane_bytes &&
                     vp[static_cast<std::size_t>(l)].size() == g.plane_bytes,
                 "LMCache load: plane byte count != expected");
        auto* base = static_cast<uint8_t*>(full_attn_buf_[static_cast<std::size_t>(l)]->data()) +
                     static_cast<std::size_t>(block_id) * g.page_bytes;
        backend.Copy(queue_, base, kp[static_cast<std::size_t>(l)].data(),
                     g.plane_bytes);
        backend.Copy(queue_, base + g.plane_bytes,
                     vp[static_cast<std::size_t>(l)].data(), g.plane_bytes);
      }
    }
  }
  backend.Synchronize(queue_);
}

void GPUModelRunner::ConnectorStorePromptKv(
    const SchedulerOutput& scheduler_output) {
  using kv_offload::lmcache::LMCacheConnector;
  auto* lm = dynamic_cast<LMCacheConnector*>(kv_connector_);
  if (lm == nullptr) return;
  if (full_attn_group_id_ < 0 || attn_kv_.empty()) return;
  const int num_reqs = input_batch_.num_reqs();
  if (num_reqs == 0) return;

  FaGeom g;
  g.num_layers = static_cast<int>(full_attn_buf_.size());
  g.block_size = static_cast<int>(attn_kv_[0].block_size);
  g.hidden_dim = static_cast<int>(attn_kv_[0].num_kv_heads * attn_kv_[0].head_size);
  g.elem_size = vt::SizeOf(attn_kv_[0].dtype);
  g.plane_bytes = static_cast<std::size_t>(g.block_size) *
                  static_cast<std::size_t>(g.hidden_dim) * g.elem_size;
  g.page_bytes = 2 * g.plane_bytes;
  VT_CHECK(g.page_bytes == static_cast<std::size_t>(fa_page_size_bytes_),
           "LMCache store: full-attention page is padded/asymmetric");
  const auto& cfg = lm->config();
  VT_CHECK(cfg.num_layers == g.num_layers && cfg.hidden_dim == g.hidden_dim &&
               cfg.chunk_tokens == g.block_size && cfg.elem_size == g.elem_size,
           "LMCache store: connector geometry disagrees with the runner's KV "
           "cache");

  int fa_cols = 0;
  const std::vector<int32_t> fa_bt =
      gather_block_table(full_attn_group_id_, num_reqs, &fa_cols);

  vt::Backend& backend = vt::GetBackend(queue_.device.type);
  // The forward's ReshapeAndCache writes may still be in flight on the queue;
  // the device reads below must see the committed KV.
  backend.Synchronize(queue_);

  std::vector<std::string> kp(static_cast<std::size_t>(g.num_layers));
  std::vector<std::string> vp(static_cast<std::size_t>(g.num_layers));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = *input_batch_.req_ids[static_cast<std::size_t>(i)];
    const int num_prompt =
        input_batch_.num_prompt_tokens[static_cast<std::size_t>(i)];
    const int computed_before =
        input_batch_.num_computed_tokens_cpu[static_cast<std::size_t>(i)];
    const int num_scheduled = scheduler_output.num_scheduled_tokens.at(req_id);
    // Post-step computed count, clamped to the prompt (offload_prompt_only).
    const int prompt_done = std::min(computed_before + num_scheduled, num_prompt);
    const int complete_blocks = prompt_done / g.block_size;
    int already = connector_stored_blocks_[req_id];  // 0 if unseen
    if (complete_blocks <= already) continue;

    const int need = complete_blocks * g.block_size;
    std::vector<int32_t> toks;
    toks.reserve(static_cast<std::size_t>(need));
    for (int col = 0; col < need; ++col) {
      toks.push_back(input_batch_.token_id(i, col));
    }
    const std::vector<uint64_t> folds = lm->ChunkFolds(toks);
    for (int b = already; b < complete_blocks &&
                          b < static_cast<int>(folds.size());
         ++b) {
      const int block_id = fa_bt[static_cast<std::size_t>(i) *
                                     static_cast<std::size_t>(fa_cols) +
                                 static_cast<std::size_t>(b)];
      for (int l = 0; l < g.num_layers; ++l) {
        kp[static_cast<std::size_t>(l)].resize(g.plane_bytes);
        vp[static_cast<std::size_t>(l)].resize(g.plane_bytes);
        const auto* base =
            static_cast<const uint8_t*>(full_attn_buf_[static_cast<std::size_t>(l)]->data()) +
            static_cast<std::size_t>(block_id) * g.page_bytes;
        backend.Copy(queue_, kp[static_cast<std::size_t>(l)].data(), base,
                     g.plane_bytes);
        backend.Copy(queue_, vp[static_cast<std::size_t>(l)].data(),
                     base + g.plane_bytes, g.plane_bytes);
      }
      backend.Synchronize(queue_);  // planes host-valid before the wire PUT
      lm->StoreChunk(lm->ChunkKey(folds[static_cast<std::size_t>(b)]), kp, vp);
    }
    connector_stored_blocks_[req_id] = complete_blocks;
  }
}

}  // namespace vllm::v1
