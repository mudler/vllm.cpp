// Ported from: vllm/v1/worker/gpu/model_runner.py @ e24d1b24
// (initialize_kv_cache / execute_model / sample_tokens / sample /
// postprocess_sampled — the T0 slice) + the decode-first reorder from
// vllm/v1/attention/backends/utils.py::reorder_batch_to_split_decodes_and_prefills.
// See include/vllm/v1/worker/gpu/runner.h for scope, the V1-algorithm / MRV2-
// contract composition, the four-way ordering contract, and the deferred paths.
#include "vllm/v1/worker/gpu/runner.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "vllm/v1/sample/ops/bad_words.h"  // apply_allowed_token_ids (-inf mask)
#include "vt/backend.h"  // vt::Backend / GetBackend (VT_GPU_SAMPLE=0 download)
#include "vt/dtype.h"  // VT_CHECK
#include "vt/ops.h"  // vt::GatherTokens (async-decode on-GPU next-input build)
#include "vt/tensor.h"

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

// Async-decode A/B toggle (perf). Default OFF. When ON (VT_ASYNC_DECODE=1) an
// EAGER pure-decode all-greedy step keeps the sampled token ON DEVICE (the
// sampler's greedy fast path — no blocking full-token D2H), builds the NEXT
// step's input_ids ON-GPU via vt::GatherTokens (mirroring vLLM's
// prev_sampled_token_ids + _prepare_input_ids scatter), and reads the tokens back
// on a SEPARATE copy stream, event-synced (AsyncGPUModelRunnerOutput). This lifts
// the blocking token D2H (sampler.cpp DeviceBuffer::download's Synchronize) off
// the decode critical path. Byte-identical to OFF for every non-eligible step and
// TOKEN-IDENTICAL for eligible steps (same GreedyArgmax kernel + tie-break; only
// the token's residence differs). SCOPE: the decode CUDA-graph fast path stays on
// the current (blocking) path — when ON, a pure-decode fp4/CUDA step that would
// use the graph is instead routed through the eager device path so async engages;
// with the graph forced off (VLLM_CPP_CUDAGRAPH=0) the A/B is eager-vs-eager.
// AsyncDecodeEnabled() is defined once in engine/types.cpp (shared with the
// EngineCore's depth-2 batch queue) so the runner and engine never disagree.

// 27B DENSE decode CUDA-graph toggle (the dense sibling of the 35B MoE decode
// graph). Default ON: a pure-decode step on the fp4/CUDA dense (27B) model is
// routed through Qwen3_5DenseDecodeGraph (capture-once/replay-per-token), which
// removes the per-step host tax (the ~62k synchronous cudaMalloc + serial kernel
// launches the eager dense decode paid EVERY step). VLLM_CPP_DENSE_DECODE_GRAPH=0
// restores the eager Qwen3_5DenseModel::ForwardDevice/Forward path (the fallback,
// and the A/B baseline). Inner capture is additionally gated by VLLM_CPP_CUDAGRAPH
// (shared with the MoE graph); with it off the driver runs its eager fallback.
static bool DenseDecodeGraphEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VLLM_CPP_DENSE_DECODE_GRAPH");
    return e == nullptr || e[0] != '0';
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
}  // namespace

GPUModelRunner::GPUModelRunner(const HfConfig& config,
                               const Qwen3_5MoeWeights& weights,
                               const KVCacheConfig& kv_cache_config,
                               vt::Queue queue, int max_num_reqs,
                               int max_model_len, int max_num_batched_tokens)
    : config_(config),
      moe_weights_(&weights),
      queue_(queue),
      input_batch_(max_num_reqs, max_model_len, max_num_batched_tokens,
                   static_cast<int>(config.vocab_size),
                   group_block_sizes(kv_cache_config),
                   group_block_sizes(kv_cache_config)) {
  max_num_reqs_ = max_num_reqs;
  initialize_kv_cache(kv_cache_config);
  // Eager Marlin NVFP4 repack (VT_NVFP4_MARLIN=1): repack all experts + dense
  // shared/lm_head weights at LOAD time so the first request pays no first-touch
  // repack. No-op on CPU / bf16 / when the gate is off.
  Qwen3_5Model::PrepareMarlinResident(weights, config, queue_);
}

GPUModelRunner::GPUModelRunner(const HfConfig& config,
                               const Qwen3_5DenseWeights& weights,
                               const KVCacheConfig& kv_cache_config,
                               vt::Queue queue, int max_num_reqs,
                               int max_model_len, int max_num_batched_tokens)
    : config_(config),
      dense_weights_(&weights),
      queue_(queue),
      input_batch_(max_num_reqs, max_model_len, max_num_batched_tokens,
                   static_cast<int>(config.vocab_size),
                   group_block_sizes(kv_cache_config),
                   group_block_sizes(kv_cache_config)) {
  max_num_reqs_ = max_num_reqs;
  initialize_kv_cache(kv_cache_config);
}

GPUModelRunner::~GPUModelRunner() {
  if (!async_ready_) return;
  vt::Backend& b = vt::GetBackend(queue_.device.type);
  for (int s = 0; s < kAsyncSlots; ++s) {
    if (async_prev_sampled_[s] != nullptr) b.Free(async_prev_sampled_[s]);
    if (async_next_input_[s] != nullptr) b.Free(async_next_input_[s]);
    if (async_remap_[s] != nullptr) b.Free(async_remap_[s]);
    b.DestroyEvent(async_sample_done_[s]);
    b.DestroyEvent(async_copy_done_[s]);
  }
  b.DestroyQueue(async_copy_queue_);
}

// Lazily allocate the async-decode device buffers + side copy stream + events on
// the first async-eligible step (VT_ASYNC_DECODE). Sized by max_num_reqs (the
// pure-decode batch is one token per request), DOUBLE-BUFFERED (two slots) so the
// depth-2 pipeline's two in-flight steps never alias a buffer. No-op after the
// first call.
void GPUModelRunner::ensure_async_resources() {
  if (async_ready_) return;
  vt::Backend& b = vt::GetBackend(queue_.device.type);
  const size_t nmax = static_cast<size_t>(max_num_reqs_ > 0 ? max_num_reqs_ : 1);
  for (int s = 0; s < kAsyncSlots; ++s) {
    async_prev_sampled_[s] = b.Alloc(nmax * sizeof(int64_t));
    async_next_input_[s] = b.Alloc(nmax * sizeof(int32_t));
    async_remap_[s] = b.Alloc(nmax * sizeof(int32_t));
    async_host_tokens_[s].assign(nmax, 0);
    async_sample_done_[s] = b.CreateEvent();
    async_copy_done_[s] = b.CreateEvent();
  }
  async_copy_queue_ = b.CreateQueue();
  async_ready_ = true;
}

void GPUModelRunner::initialize_kv_cache(const KVCacheConfig& kv_cache_config) {
  num_blocks_ = kv_cache_config.num_blocks;
  // GDN mamba-state slots = max concurrent sequences (one recurrent state per
  // sequence), decoupled from the attention num_blocks. Guard against a 0 (e.g.
  // a test path that skipped the ctor arg) by falling back to num_blocks.
  gdn_state_slots_ = max_num_reqs_ > 0 ? max_num_reqs_ : num_blocks_;
  gdn_slot_of_block_.clear();
  gdn_free_slots_.clear();
  gdn_free_slots_.reserve(static_cast<size_t>(gdn_state_slots_));
  for (int64_t s = gdn_state_slots_ - 1; s >= 0; --s)
    gdn_free_slots_.push_back(static_cast<int32_t>(s));

  // Resolve the full-attn + GDN(mamba) KV group ids (T0 gate models: exactly one
  // of each). The block-table group order == kv_cache_groups order.
  for (int g = 0; g < static_cast<int>(kv_cache_config.kv_cache_groups.size());
       ++g) {
    const KVCacheSpecKind kind =
        kv_cache_config.kv_cache_groups[static_cast<size_t>(g)]
            .kv_cache_spec->kind();
    if (kind == KVCacheSpecKind::kFullAttention) {
      full_attn_group_id_ = g;
    } else if (kind == KVCacheSpecKind::kMamba) {
      gdn_group_id_ = g;
    }
  }

  // Per-layer cache dims (source of truth = HfConfig; the MambaSpec.shapes /
  // FullAttentionSpec dims are consistent with these). Allocate one PagedKvCache
  // per full-attn layer and one GdnStateCache per GDN layer, in LAYER ORDER
  // (matches Qwen3_5Model::Forward's per-layer fa_idx / gdn_idx indexing).
  const int64_t Hkv = config_.num_key_value_heads;
  const int64_t Dh = config_.head_dim;
  // Paged KV-cache dtype: bf16 (vLLM's bf16 flash_attn KV store — halves KV
  // memory) unless VT_KV_CACHE_F32 is set (same-binary f32-vs-bf16 A/B).
  const char* kv_f32_env = std::getenv("VT_KV_CACHE_F32");
  const vt::DType kv_dtype =
      (kv_f32_env != nullptr && kv_f32_env[0] == '1') ? vt::DType::kF32
                                                      : vt::DType::kBF16;
  const int64_t Hk = config_.linear_num_key_heads;
  const int64_t Hv = config_.linear_num_value_heads;
  const int64_t Dk = config_.linear_key_head_dim;
  const int64_t Dv = config_.linear_value_head_dim;
  const int64_t Kw = config_.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk;
  const int64_t value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;

  // Full-attn block size from the full-attn group's spec (its paged layout).
  int64_t fa_block_size = 0;
  if (full_attn_group_id_ >= 0) {
    fa_block_size =
        kv_cache_config.kv_cache_groups[static_cast<size_t>(full_attn_group_id_)]
            .kv_cache_spec->block_size;
  }

  const vt::Device dev = queue_.device;
  // bf16 GDN state caches on CUDA (vLLM default mamba_cache_dtype auto → model
  // dtype), f32 on CPU (the CPU GDN ops are f32-only; the exact-value CPU tests
  // assume f32). The CUDA GDN decode/conv kernels read bf16 → f32 → write bf16.
  // A/B: VT_GDN_STATE_BF16=0 forces the f32 cache (same-binary before/after of
  // the bf16 state-traffic lever), mirroring VT_GDN_FUSED_DECODE / VT_GDN_CHUNKED.
  const char* bf16env = std::getenv("VT_GDN_STATE_BF16");
  const bool force_f32 = bf16env != nullptr && bf16env[0] == '0';
  gdn_cache_dtype_ = (dev.type == vt::DeviceType::kCUDA && !force_f32)
                         ? vt::DType::kBF16
                         : vt::DType::kF32;
  for (int64_t l = 0; l < config_.num_hidden_layers; ++l) {
    const bool is_gdn =
        config_.layer_types[static_cast<size_t>(l)] == "linear_attention";
    if (is_gdn) {
      // State caches: bf16 on CUDA (vLLM default), f32 on CPU (f32-only CPU ops).
      // Raw bytes sized by gdn_cache_dtype_; 0 bytes == +0.0f in both dtypes.
      const size_t es = vt::SizeOf(gdn_cache_dtype_);
      ssm_buf_.emplace_back(
          static_cast<size_t>(gdn_state_slots_ * Hv * Dv * Dk) * es, uint8_t{0});
      conv_buf_.emplace_back(
          static_cast<size_t>(gdn_state_slots_ * conv_dim * (Kw - 1)) * es, uint8_t{0});
    } else {
      // KV cache stored in kv_dtype (bf16 default; f32 if VT_KV_CACHE_F32). 0
      // bytes == 0.0 in both bf16 and f32. bf16 halves KV memory vs f32.
      full_attn_buf_.emplace_back(
          static_cast<size_t>(num_blocks_ * 2 * fa_block_size * Hkv * Dh) *
              static_cast<size_t>(vt::SizeOf(kv_dtype)),
          static_cast<uint8_t>(0));
    }
  }

  // Build the views over the (now stable) backing storage.
  attn_kv_.clear();
  for (auto& b : full_attn_buf_) {
    PagedKvCache kv;
    kv.data = b.data();
    kv.dtype = kv_dtype;
    kv.num_blocks = num_blocks_;
    kv.block_size = fa_block_size;
    kv.num_kv_heads = Hkv;
    kv.head_size = Dh;
    attn_kv_.push_back(kv);
  }
  gdn_state_.clear();
  for (size_t g = 0; g < ssm_buf_.size(); ++g) {
    GdnStateCache gs;
    gs.ssm_state = vt::Tensor::Contiguous(ssm_buf_[g].data(), gdn_cache_dtype_,
                                          dev, {gdn_state_slots_, Hv, Dv, Dk});
    gs.conv_state = vt::Tensor::Contiguous(conv_buf_[g].data(), gdn_cache_dtype_,
                                           dev,
                                           {gdn_state_slots_, conv_dim, Kw - 1});
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

void GPUModelRunner::remap_gdn_state_slots(std::vector<int32_t>& gdn_bt,
                                           int gdn_cols, int num_reqs) {
  if (gdn_cols <= 0 || num_reqs <= 0) return;
  // The persistent batch holds EVERY live sequence, so col 0 over [0, num_reqs)
  // is exactly the set of alive mamba-state pool block-ids this step.
  std::unordered_set<int32_t> alive;
  alive.reserve(static_cast<size_t>(num_reqs));
  for (int r = 0; r < num_reqs; ++r)
    alive.insert(gdn_bt[static_cast<size_t>(r) * static_cast<size_t>(gdn_cols)]);
  // Reclaim slots of sequences that have finished (their block-id is gone).
  for (auto it = gdn_slot_of_block_.begin(); it != gdn_slot_of_block_.end();) {
    if (alive.find(it->first) == alive.end()) {
      gdn_free_slots_.push_back(it->second);
      it = gdn_slot_of_block_.erase(it);
    } else {
      ++it;
    }
  }
  // Assign/reuse a compact slot per live sequence and rewrite col 0 in place.
  for (int r = 0; r < num_reqs; ++r) {
    const size_t off = static_cast<size_t>(r) * static_cast<size_t>(gdn_cols);
    const int32_t blk = gdn_bt[off];
    auto it = gdn_slot_of_block_.find(blk);
    int32_t slot;
    if (it != gdn_slot_of_block_.end()) {
      slot = it->second;
    } else {
      VT_CHECK(!gdn_free_slots_.empty(),
               "GDN state slots exhausted: live sequences exceed max_num_reqs");
      slot = gdn_free_slots_.back();
      gdn_free_slots_.pop_back();
      gdn_slot_of_block_.emplace(blk, slot);
      // New sequence -> zero its (possibly reused) recurrent-state slot so the
      // async speculative step of a prior request can't leak into it. Gated on
      // the async toggle to keep the synchronous path byte-identical (it relies
      // on the construction-zeroed slots + coincidence-free reuse, unchanged).
      if (AsyncDecodeEnabled()) {
        zero_gdn_state_slot(slot);
      }
    }
    gdn_bt[off] = slot;
  }
}

void GPUModelRunner::zero_gdn_state_slot(int32_t slot) {
  if (gdn_state_slots_ <= 0 || slot < 0) return;
  const size_t s = static_cast<size_t>(slot);
  for (std::vector<uint8_t>& buf : ssm_buf_) {
    const size_t row = buf.size() / static_cast<size_t>(gdn_state_slots_);
    if (row > 0) std::memset(buf.data() + s * row, 0, row);
  }
  for (std::vector<uint8_t>& buf : conv_buf_) {
    const size_t row = buf.size() / static_cast<size_t>(gdn_state_slots_);
    if (row > 0) std::memset(buf.data() + s * row, 0, row);
  }
}

std::optional<ModelRunnerOutput> GPUModelRunner::execute_model(
    const SchedulerOutput& scheduler_output) {
  // update_states: admit new reqs (incl. prefill_token_ids) + apply cached diffs
  // + remove finished/unscheduled + condense (M1.5).
  update_states(input_batch_, scheduler_output);

  // Reset the stash. A 0-token step (e.g. an aborted-request flush) runs no
  // forward — mark num_reqs == 0 so sample_tokens returns an empty output
  // (mirrors upstream execute_model's total==0 early return).
  exec_state_ = ExecuteModelState{};
  if (scheduler_output.total_num_scheduled_tokens == 0 ||
      input_batch_.num_reqs() == 0) {
    return std::nullopt;
  }

  // DECODE-FIRST REORDER (four-way ordering contract) — before any metadata.
  reorder_batch_to_split_decodes_and_prefills(input_batch_, scheduler_output);

  // Build the flattened dense-order step inputs (M1.5).
  StepInputs step = prepare_inputs(input_batch_, scheduler_output);
  const int num_reqs = input_batch_.num_reqs();

  // Full-attention KV group metadata (M1.6 MakeCommonAttentionMetadata).
  int fa_cols = 0;
  const std::vector<int32_t> fa_bt =
      gather_block_table(full_attn_group_id_, num_reqs, &fa_cols);
  CommonAttentionMetadata attn_meta = MakeCommonAttentionMetadata(
      step, fa_bt, fa_cols, /*causal=*/true, full_attn_group_id_);

  // GDN KV group metadata: the same step over the GDN group's block table,
  // segmented decode-first by the GDN builder (M1.6 Task 4).
  int gdn_cols = 0;
  std::vector<int32_t> gdn_bt =
      gather_block_table(gdn_group_id_, num_reqs, &gdn_cols);
  // Remap the scattered mamba-state pool block-ids (col 0) to compact per-
  // sequence state slots in [0, gdn_state_slots_), so the GDN state cache is
  // sized by max_num_reqs (one recurrent state per sequence) rather than the
  // attention num_blocks. Only col 0 (state indices) is read downstream.
  remap_gdn_state_slots(gdn_bt, gdn_cols, num_reqs);
  const CommonAttentionMetadata gdn_cam = MakeCommonAttentionMetadata(
      step, gdn_bt, gdn_cols, /*causal=*/true, gdn_group_id_);
  GDNAttentionMetadataBuilder gdn_builder;
  GDNAttentionMetadata gdn_meta =
      gdn_builder.build(/*common_prefix_len=*/0, gdn_cam);

  // Flattened dense-order forward inputs (positions int64 -> int32 for RoPE).
  const std::vector<int32_t>& token_ids = step.input_token_ids;
  std::vector<int32_t> positions(step.positions.begin(), step.positions.end());

  // THE FORWARD (Task 3, over the persistent KV caches). Returns f32 logits
  // (lm_head already applied): [num_reqs, vocab] when the gather-before-lm_head
  // path is on (prefill/mixed) or pure-decode, else the full
  // [num_actual_tokens, vocab] (VT_LOGITS_GATHER=0).
  //
  // DECODE CUDA-GRAPH path (M2.5): a PURE-DECODE step (every request one token,
  // no prefill) on an fp4/CUDA model is routed through the Qwen3_5DecodeGraph,
  // which captures the forward once per PADDED batch size and replays it per
  // step. The graph's real-row output is bit-identical to Forward.
  //
  // BATCHED (num_reqs>1) — the gate-#1 lever: at conc-64 kernel-launch overhead
  // is ~24% of the decode wall (~1.4k cudaLaunchKernel/step -> 1 cudaGraphLaunch).
  // The batch is padded up to the nearest captured size {1,2,4,8,16,32,64}
  // (mirrors vLLM cudagraph_capture_sizes + pad-to-nearest,
  // compilation/cuda_graph.py); the padded rows are inert (BuildPaddedDecode).
  // The decode forward is row-independent, so padding cannot perturb the real
  // rows. Beyond 64 (kMaxDecodeGraphBatch) / prefill / mixed / bf16 / CPU stay
  // eager. The DENSE (27B) fp4 model has its OWN decode graph (dense_decode_
  // graph_, the sibling of decode_graph_) under the same gate.
  const bool pure_decode = attn_meta.num_actual_tokens == num_reqs &&
                           gdn_meta.num_prefill_tokens == 0;
  const bool fp4_cuda = queue_.device.type == vt::DeviceType::kCUDA &&
                        moe_weights_ != nullptr &&
                        !moe_weights_->layers.empty() &&
                        !moe_weights_->layers.front().moe.expert_gate_fp4.empty();
  // Dense (27B) fp4/CUDA gate — the sibling of fp4_cuda for the dense decode
  // graph. True when the dense MLP projections are fp4-resident (the real 27B
  // CUDA load); the synthetic CPU tests leave them empty and stay eager.
  const bool fp4_cuda_dense = queue_.device.type == vt::DeviceType::kCUDA &&
                              dense_weights_ != nullptr &&
                              !dense_weights_->layers.empty() &&
                              !dense_weights_->layers.front().mlp.gate_proj_fp4.Empty();
  constexpr int kMaxDecodeGraphBatch = 64;  // largest captured size
  // Gather-before-lm_head indices (the SAME last-token rows sample_tokens uses).
  // Empty when the toggle is off → old full-logits path. The eager forwards skip
  // the gather when it is a no-op (pure decode: len == num_actual_tokens).
  const bool gather = LogitsGatherEnabled();
  const std::vector<int32_t> kNoGather;
  const std::vector<int32_t>& gather_li = gather ? step.logits_indices : kNoGather;

  // Wrap a full [T,vocab] HOST logits vector (the VT_LOGITS_GATHER=0 opt-out) as a
  // ForwardLogits::host — sample_tokens re-gathers the per-request rows on host.
  auto host_logits = [&](std::vector<float>&& h) {
    ForwardLogits fl;
    fl.vocab = config_.vocab_size;
    fl.rows = fl.vocab > 0 ? static_cast<int64_t>(h.size()) / fl.vocab : 0;
    fl.host = std::move(h);
    return fl;
  };

  // ── Async-decode (VT_ASYNC_DECODE): an EAGER pure-decode device-logits step is
  // the async-eligible case (the sampler keeps the token on device + side-stream
  // readback in sample_tokens). When the PREVIOUS step retained its sampled tokens
  // on device (prev_sampled_valid) AND every request this step continues from it,
  // build the NEXT input_ids ON-GPU here — mirror of vLLM _prepare_input_ids
  // (input_ids.gpu.copy_(prev_sampled_token_ids[idx])) — so the sampled token
  // never round-trips through host on the decode critical path. Any new/moved-in
  // request disqualifies the step (fall back to the host token path, which is
  // valid because the prev async step synced its readback into token_ids_cpu).
  const bool async_sample =
      AsyncDecodeEnabled() && pure_decode && gather && GpuSampleEnabled();
  const int32_t* async_dev_tokens = nullptr;  // device i32 [num_reqs] next input_ids
  // THIS step's ping-pong slot: opposite the previous async step's (async_prev_
  // slot_), so an un-finalized step N (buffers still live) and step N+1's forward
  // never alias. sample_tokens reads exec_state_.async_slot to match.
  const int cur_slot =
      (async_prev_slot_ >= 0) ? (1 - async_prev_slot_) : 0;
  if (async_sample) {
    ensure_async_resources();
    exec_state_.async_slot = cur_slot;
    // Build step N+1's input_ids ON-GPU from the PREVIOUS async step's retained
    // device tokens (async_prev_sampled_[async_prev_slot_]) into THIS step's
    // next_input slot. Requires the previous step to have kept its tokens on
    // device (prev_sampled_valid) and every current req to have been in it.
    if (input_batch_.prev_sampled_valid && async_prev_slot_ >= 0 &&
        num_reqs <= max_num_reqs_) {
      std::vector<int32_t> remap(static_cast<size_t>(num_reqs));
      bool ok = true;
      for (int i = 0; i < num_reqs; ++i) {
        const std::string& rid = *input_batch_.req_ids[static_cast<size_t>(i)];
        const auto it = input_batch_.prev_req_id_to_index.find(rid);
        if (it == input_batch_.prev_req_id_to_index.end()) { ok = false; break; }
        remap[static_cast<size_t>(i)] = it->second;
      }
      if (ok) {
        vt::Backend& b = vt::GetBackend(queue_.device.type);
        b.Copy(queue_, async_remap_[cur_slot], remap.data(),
               static_cast<size_t>(num_reqs) * sizeof(int32_t));
        vt::Tensor dst = vt::Tensor::Contiguous(async_next_input_[cur_slot], vt::DType::kI32,
                                                queue_.device, {static_cast<int64_t>(num_reqs)});
        vt::Tensor src = vt::Tensor::Contiguous(async_prev_sampled_[async_prev_slot_],
                                                vt::DType::kI64, queue_.device,
                                                {static_cast<int64_t>(max_num_reqs_)});
        vt::Tensor idx = vt::Tensor::Contiguous(async_remap_[cur_slot], vt::DType::kI32,
                                                queue_.device, {static_cast<int64_t>(num_reqs)});
        vt::GatherTokens(queue_, dst, src, idx);
        async_dev_tokens = static_cast<const int32_t*>(async_next_input_[cur_slot]);
      }
    }
  }

  // THE FORWARD. DEFAULT (gather ON): the *Device entry points keep the
  // [num_reqs,vocab] logits ON DEVICE and hand them to the sampler with no
  // full-logits D2H. VT_LOGITS_GATHER=0: the host Forward returns full
  // [T,vocab] logits and sample_tokens re-gathers on host (unchanged path).
  ForwardLogits logits;
  if (dense_weights_ != nullptr) {
    // DENSE arch (27B). A PURE-DECODE fp4/CUDA step is routed through the dense
    // decode CUDA-graph (capture-once/replay-per-token) under the SAME gate as
    // the 35B MoE graph — removing the per-step host tax (the ~62k synchronous
    // cudaMalloc + serial kernel launches the eager dense decode paid EVERY
    // step). VLLM_CPP_DENSE_DECODE_GRAPH=0 restores the eager path (fallback +
    // A/B baseline). Prefill / mixed / bf16 / CPU / >64 stay eager.
    // async_sample bypasses the graph so the eager device path (which reads the
    // on-GPU-built input_ids) can engage — the graph itself stays on the current
    // blocking path (SCOPE note above).
    if (DenseDecodeGraphEnabled() && pure_decode && fp4_cuda_dense &&
        num_reqs <= kMaxDecodeGraphBatch && !async_sample) {
      if (!dense_decode_graph_) {
        dense_decode_graph_ = std::make_unique<Qwen3_5DenseDecodeGraph>(
            *dense_weights_, config_, queue_, gdn_state_slots_);
      }
      // Device-resident [num_reqs,vocab] (one row/request in request order); the
      // gather toggle is a no-op for pure decode (T==num_reqs).
      logits = dense_decode_graph_->Step(token_ids, positions, attn_meta, gdn_meta,
                                         attn_kv_, gdn_state_);
    } else {
      // Eager paged dense forward. Same paged KV/GDN-state machinery, dense
      // SwiGLU MLP in place of the MoE block. `async_dev_tokens` (non-null only on
      // an async-eligible step with the input built on-GPU) feeds the embedding
      // from the device buffer instead of the host token_ids.
      logits = gather
                   ? Qwen3_5DenseModel::ForwardDevice(
                         token_ids, positions, attn_meta, gdn_meta, attn_kv_,
                         gdn_state_, *dense_weights_, config_, queue_, gather_li,
                         async_dev_tokens)
                   : host_logits(Qwen3_5DenseModel::Forward(
                         token_ids, positions, attn_meta, gdn_meta, attn_kv_,
                         gdn_state_, *dense_weights_, config_, queue_, gather_li));
    }
  } else if (pure_decode && fp4_cuda && num_reqs <= kMaxDecodeGraphBatch &&
             !async_sample) {
    if (!decode_graph_) {
      decode_graph_ = std::make_unique<Qwen3_5DecodeGraph>(
          *moe_weights_, config_, queue_, gdn_state_slots_);
    }
    // Decode graph output is device-resident [num_reqs,vocab] (one row/request in
    // request order); the gather toggle is a no-op for pure decode (T==num_reqs).
    logits = decode_graph_->Step(token_ids, positions, attn_meta, gdn_meta,
                                 attn_kv_, gdn_state_);
  } else {
    logits = gather
                 ? Qwen3_5Model::ForwardDevice(
                       token_ids, positions, attn_meta, gdn_meta, attn_kv_,
                       gdn_state_, *moe_weights_, config_, queue_, gather_li,
                       async_dev_tokens)
                 : host_logits(Qwen3_5Model::Forward(
                       token_ids, positions, attn_meta, gdn_meta, attn_kv_,
                       gdn_state_, *moe_weights_, config_, queue_, gather_li));
  }

  // Stash for sample_tokens (upstream ExecuteModelState).
  exec_state_.num_actual_tokens = scheduler_output.total_num_scheduled_tokens;
  exec_state_.num_reqs = num_reqs;
  exec_state_.async_sample = async_sample;
  exec_state_.logits = std::move(logits);
  exec_state_.step = std::move(step);
  exec_state_.attn_meta = std::move(attn_meta);
  exec_state_.gdn_meta = std::move(gdn_meta);
  exec_state_.req_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    exec_state_.req_ids.push_back(*input_batch_.req_ids[static_cast<size_t>(i)]);
  }

  return std::nullopt;  // MRV2 split: forward done, sample separately.
}

ModelRunnerOutput GPUModelRunner::sample_tokens(
    const std::optional<GrammarOutput>& grammar_output) {
  // The synchronous seam (sync EngineCore, tests): sample + finalize in line.
  PendingModelOutput pending = sample_tokens_impl(grammar_output, /*defer=*/false);
  return std::move(pending.output);
}

PendingModelOutput GPUModelRunner::sample_tokens_impl(
    const std::optional<GrammarOutput>& grammar_output, bool defer) {
  ModelRunnerOutput out;
  // Async-decode invariant: prev_sampled_valid is TRUE only if THIS step actually
  // retains its sampled tokens in async_prev_sampled_ (the greedy success path
  // below sets it). Clear it up front so a non-async / flush step can never leave
  // the next execute_model reading a stale device buffer.
  input_batch_.prev_sampled_valid = false;
  const int num_reqs = exec_state_.num_reqs;
  if (num_reqs == 0) {
    PendingModelOutput p;
    p.ready = true;
    p.output = std::move(out);  // 0-token flush step (nothing sampled).
    return p;
  }

  const int64_t vocab = config_.vocab_size;

  // Assemble the [num_reqs, vocab] logits the sampler runs on. Three cases:
  //
  //  (A) DEVICE path (default, gather ON): the forward already produced exactly
  //      num_reqs rows in request order, ON DEVICE — hand the sampler the device
  //      tensor directly (its argmax / temperature / top-k/top-p kernels run
  //      on-device; only the sampled token ids come back). No full-logits D2H.
  //  (A') VT_GPU_SAMPLE=0 A/B: Download the device [num_reqs,vocab] logits to a
  //      host buffer and sample from it (the OLD download-then-sample path).
  //  (B) HOST path (VT_LOGITS_GATHER=0): full [num_actual_tokens,vocab] host
  //      logits — re-gather the per-request rows on host via logits_indices,
  //      exactly as before.
  std::vector<float> sampled_logits;  // host buffer; outlives the sampler when used
  vt::Tensor logits;
  ForwardLogits& fl = exec_state_.logits;
  if (fl.on_device()) {
    VT_CHECK(fl.rows == num_reqs,
             "sample_tokens: device logits rows must equal num_reqs");
    if (GpuSampleEnabled()) {
      logits = fl.device_tensor;  // (A) sample straight off device
    } else {
      // (A') download then sample (A/B: reproduce the pre-change host path).
      sampled_logits.resize(static_cast<size_t>(num_reqs) *
                            static_cast<size_t>(vocab));
      vt::Backend& b = vt::GetBackend(queue_.device.type);
      b.Copy(queue_, sampled_logits.data(), fl.device_tensor.data,
             sampled_logits.size() * sizeof(float));
      b.Synchronize(queue_);
      logits = vt::Tensor::Contiguous(
          sampled_logits.data(), vt::DType::kF32, queue_.device,
          {static_cast<int64_t>(num_reqs), vocab});
    }
  } else if (fl.rows == num_reqs) {
    logits = vt::Tensor::Contiguous(
        fl.host.data(), vt::DType::kF32, queue_.device,
        {static_cast<int64_t>(num_reqs), vocab});
  } else {
    // (B) VT_LOGITS_GATHER=0: re-gather num_reqs rows from full [T,vocab] host.
    sampled_logits.resize(static_cast<size_t>(num_reqs) *
                          static_cast<size_t>(vocab));
    for (int i = 0; i < num_reqs; ++i) {
      const int row = exec_state_.step.logits_indices[static_cast<size_t>(i)];
      std::memcpy(
          sampled_logits.data() + static_cast<size_t>(i) * static_cast<size_t>(vocab),
          fl.host.data() +
              static_cast<size_t>(row) * static_cast<size_t>(vocab),
          static_cast<size_t>(vocab) * sizeof(float));
    }
    logits = vt::Tensor::Contiguous(
        sampled_logits.data(), vt::DType::kF32, queue_.device,
        {static_cast<int64_t>(num_reqs), vocab});
  }

  // Apply the structured-output grammar bitmask (utils.py apply_grammar_bitmask)
  // to the gathered [num_reqs, vocab] logits BEFORE sampling, when a structured
  // request is scheduled this step (gpu_model_runner.py:4462-4466). Spec-decode
  // is deferred at T0, so pass an empty spec-token map (per-req offset 0).
  if (grammar_output.has_value()) {
    apply_grammar_bitmask(*grammar_output, exec_state_.req_ids, {}, queue_,
                          logits);
  }

  // SamplingMetadata in the SAME dense [0, num_reqs) order (M1.7; CLOSES the
  // make_sampling_metadata wiring dep). Then Sampler.forward.
  const SamplingMetadata sm = input_batch_.make_sampling_metadata();

  // ── Async-decode greedy fast path (VT_ASYNC_DECODE). When execute_model marked
  // this an eager pure-decode device-logits step and the batch is all-greedy with
  // no active logits processor, keep the sampled token ON DEVICE (sample_greedy_
  // device runs the SAME GreedyArgmax kernel — bit-identical to sampler_.forward's
  // greedy path — into async_prev_sampled_[slot]) and issue a SIDE-stream readback
  // (event recorded, NOT synced). It records prev_req_id_to_index /
  // prev_sampled_valid / async_prev_slot_ for the NEXT step's on-GPU input build
  // RIGHT HERE (before this call returns) so step N+1's forward can start without
  // waiting on this step's host readback — the whole point of the pipeline.
  //   * defer == true  (depth-2 engine): return a DEFERRED pending; the host
  //     readback + token write-back happen later in finalize_output, overlapping
  //     step N+1's GPU forward.
  //   * defer == false (sync seam): finalize the slot IN LINE (readback + write-
  //     back now) — token-identical, one blocking readback, no overlap.
  // Any non-eligible condition falls through to the synchronous sampler below.
  if (exec_state_.async_sample && exec_state_.logits.on_device() && async_ready_) {
    const int slot = exec_state_.async_slot;
    vt::Tensor prev = vt::Tensor::Contiguous(async_prev_sampled_[slot], vt::DType::kI64,
                                             queue_.device,
                                             {static_cast<int64_t>(num_reqs)});
    if (sampler_.sample_greedy_device(queue_, logits, prev, sm)) {
      vt::Backend& b = vt::GetBackend(queue_.device.type);
      // Side-stream, non-blocking token D2H: the copy stream waits for the main
      // stream's argmax (sample_done[slot]), copies into host staging[slot], and
      // records copy_done[slot]. NO SyncEvent here — finalize_async_slot(slot)
      // blocks on it, either now (defer=false) or after the next forward is
      // issued (defer=true, the overlap win).
      b.RecordEvent(queue_, async_sample_done_[slot]);
      b.StreamWaitEvent(async_copy_queue_, async_sample_done_[slot]);
      b.Copy(async_copy_queue_, async_host_tokens_[slot].data(),
             async_prev_sampled_[slot],
             static_cast<size_t>(num_reqs) * sizeof(int64_t));
      b.RecordEvent(async_copy_queue_, async_copy_done_[slot]);

      // Record THIS step's mapping for step N+1's on-GPU input build. The device
      // tokens in async_prev_sampled_[slot] stay live (this slot is not reused
      // until step N+2) so step N+1 can gather from them without the host readback.
      input_batch_.prev_req_id_to_index.clear();
      for (int i = 0; i < num_reqs; ++i) {
        input_batch_.prev_req_id_to_index[exec_state_.req_ids[static_cast<size_t>(i)]] = i;
      }
      input_batch_.prev_sampled_valid = true;
      async_prev_slot_ = slot;

      PendingModelOutput p;
      p.async_slot = slot;
      p.num_reqs = num_reqs;
      p.req_ids = exec_state_.req_ids;  // dense (post-reorder) order at sample time
      if (defer) {
        p.ready = false;  // finalize_output syncs + writes back later.
        return p;
      }
      // Synchronous seam: finalize the slot now (block on the readback).
      p.ready = true;
      p.output = finalize_async_slot(slot, p.req_ids, num_reqs);
      return p;
    }
  }

  const SamplerOutput sampler_output = sampler_.forward(queue_, logits, sm);

  // Build the ModelRunnerOutput + WRITE the sampled tokens BACK into the
  // InputBatch (upstream postprocess_sampled / post_update) so the next step's
  // prepare_inputs reads them at the decode position.
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.sampled_token_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = exec_state_.req_ids[static_cast<size_t>(i)];
    out.req_ids.push_back(req_id);
    out.req_id_to_index[req_id] = i;
    const std::vector<int32_t>& toks = sampler_output.sampled_token_ids[
        static_cast<size_t>(i)];
    out.sampled_token_ids.push_back(toks);

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
  }
  PendingModelOutput p;
  p.ready = true;  // synchronous sampler: output already host-ready + written back.
  p.output = std::move(out);
  return p;
}

// finalize_async_slot: host-sync the slot's readback copy, materialize the
// ModelRunnerOutput from the host staging buffer, and apply the DEFERRED token
// write-back into the InputBatch. Mirrors AsyncGPUModelRunnerOutput.get_output()
// (gpu_model_runner.py:290-332). Shared by the in-line (defer=false) and depth-2
// deferred paths. The write-back resolves each request's slot by req_id at THIS
// moment (not the captured sample-time slot): a request removed from the batch
// since sampling (finished / aborted) is SKIPPED — its speculative token is
// discarded, exactly as update_from_output skips it on the scheduler side.
ModelRunnerOutput GPUModelRunner::finalize_async_slot(
    int slot, const std::vector<std::string>& req_ids, int num_reqs) {
  vt::Backend& b = vt::GetBackend(queue_.device.type);
  // Block on the side-stream readback (a no-op on CPU; the real host<->GPU wait
  // on CUDA). After this the tokens are host-ready in async_host_tokens_[slot].
  b.SyncEvent(async_copy_done_[slot]);

  ModelRunnerOutput out;
  out.req_ids.reserve(static_cast<size_t>(num_reqs));
  out.sampled_token_ids.reserve(static_cast<size_t>(num_reqs));
  for (int i = 0; i < num_reqs; ++i) {
    const std::string& req_id = req_ids[static_cast<size_t>(i)];
    out.req_ids.push_back(req_id);
    out.req_id_to_index[req_id] = i;
    const int32_t tok =
        static_cast<int32_t>(async_host_tokens_[slot][static_cast<size_t>(i)]);
    out.sampled_token_ids.push_back({tok});
    // Deferred write-back at the request's CURRENT slot (looked up by id; a
    // finished/removed request is absent -> skip, discarding its speculative
    // token). token_ids_cpu / num_tokens_no_spec grow exactly as the synchronous
    // path, just one step later (the next forward already consumed the ON-DEVICE
    // token, not this host value).
    const auto it = input_batch_.req_id_to_index.find(req_id);
    if (it == input_batch_.req_id_to_index.end()) {
      continue;
    }
    const int r = it->second;
    const int n = input_batch_.num_tokens_no_spec[static_cast<size_t>(r)];
    const size_t idx = static_cast<size_t>(r) *
                           static_cast<size_t>(input_batch_.max_model_len) +
                       static_cast<size_t>(n);
    if (idx < input_batch_.token_ids_cpu.size()) {
      input_batch_.token_ids_cpu[idx] = tok;
    }
    input_batch_.num_tokens_no_spec[static_cast<size_t>(r)] = n + 1;
    auto& out_ids = input_batch_.req_output_token_ids[static_cast<size_t>(r)];
    if (out_ids.has_value()) out_ids->push_back(tok);
  }
  return out;
}

// execute_model_async: run the forward + sample for this step. When async is ON
// and the step is eager pure-decode all-greedy AND allow_defer is set, returns a
// DEFERRED PendingModelOutput (device tokens + recorded-not-synced readback);
// otherwise a host-ready pending. Mirrors execute_model(non_block=True) building
// an AsyncGPUModelRunnerOutput (gpu_model_runner.py:242-296).
PendingModelOutput GPUModelRunner::execute_model_async(
    const SchedulerOutput& scheduler_output,
    const std::optional<GrammarOutput>& grammar_output, bool allow_defer) {
  std::optional<ModelRunnerOutput> forward = execute_model(scheduler_output);
  if (forward.has_value()) {
    // A non-MRV2 / 0-token flush that produced the output directly.
    PendingModelOutput p;
    p.ready = true;
    p.output = std::move(*forward);
    return p;
  }
  return sample_tokens_impl(grammar_output, allow_defer);
}

// finalize_output: block on the pending step's readback and materialize its host
// ModelRunnerOutput (with the deferred write-back). Mirrors
// AsyncGPUModelRunnerOutput.get_output() (gpu_model_runner.py:290-332). A ready
// pending (synchronous / non-async step) is handed back with no blocking.
ModelRunnerOutput GPUModelRunner::finalize_output(PendingModelOutput& pending) {
  if (pending.ready) {
    return std::move(pending.output);
  }
  return finalize_async_slot(pending.async_slot, pending.req_ids,
                             pending.num_reqs);
}

}  // namespace vllm::v1
