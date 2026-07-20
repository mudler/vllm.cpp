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
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/platforms/interface.h"  // GetPlatform(device.type) per-tensor memory-model seam
#include "vllm/v1/sample/ops/bad_words.h"  // apply_allowed_token_ids (-inf mask)
#include "vllm/v1/worker/gpu/async_runner_flag.h"  // VT_ASYNC_RUNNER predicate
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
}  // namespace

GPUModelRunner::GPUModelRunner(const HfConfig& config,
                               LoadedModel& model,
                               const KVCacheConfig& kv_cache_config,
                               vt::Queue queue, int max_num_reqs,
                               int max_model_len, int max_num_batched_tokens)
    : config_(config),
      model_(&model),
      queue_(queue),
      input_batch_(max_num_reqs, max_model_len, max_num_batched_tokens,
                   static_cast<int>(config.vocab_size),
                   group_block_sizes(kv_cache_config),
                   group_block_sizes(kv_cache_config)) {
  max_num_reqs_ = max_num_reqs;
  async_input_combine_ = AsyncRunnerEnvDefault();
  initialize_kv_cache(kv_cache_config);
  ModelRegistry::Prepare(*model_, config_, queue_);
}

GPUModelRunner::GPUModelRunner(const HfConfig& config,
                               std::unique_ptr<LoadedModel> owned_model,
                               const KVCacheConfig& kv_cache_config,
                               vt::Queue queue, int max_num_reqs,
                               int max_model_len, int max_num_batched_tokens)
    : config_(config),
      owned_model_(std::move(owned_model)),
      model_(owned_model_.get()),
      queue_(queue),
      input_batch_(max_num_reqs, max_model_len, max_num_batched_tokens,
                   static_cast<int>(config.vocab_size),
                   group_block_sizes(kv_cache_config),
                   group_block_sizes(kv_cache_config)) {
  max_num_reqs_ = max_num_reqs;
  async_input_combine_ = AsyncRunnerEnvDefault();
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
  // GDN mamba-state slots = max concurrent sequences (one recurrent state per
  // sequence), decoupled from the attention num_blocks. Guard against a 0 (e.g.
  // a test path that skipped the ctor arg) by falling back to num_blocks.
  gdn_state_slots_ = max_num_reqs_ > 0 ? max_num_reqs_ : num_blocks_;
  gdn_slot_of_req_.clear();
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

  // Allocate one PagedKvCache per full-attn layer and one GdnStateCache per GDN
  // layer, in LAYER ORDER (matches Qwen3_5Model::Forward's per-layer fa_idx /
  // gdn_idx indexing). As in upstream, MambaSpec is the source of truth for the
  // recurrent tensors' order, shapes, dtypes, and page bytes.
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

  const MambaSpec* mamba_spec = nullptr;
  if (gdn_group_id_ >= 0) {
    mamba_spec = dynamic_cast<const MambaSpec*>(
        kv_cache_config.kv_cache_groups[static_cast<size_t>(gdn_group_id_)]
            .kv_cache_spec.get());
    VT_CHECK(mamba_spec != nullptr,
             "runner: GDN cache group must carry a MambaSpec");
    VT_CHECK(mamba_spec->shapes.size() == 2 &&
                 mamba_spec->dtypes.size() == 2,
             "runner: Qwen3.5 MambaSpec must contain conv then temporal state");
    const std::vector<int64_t> expected_conv_shape{conv_dim, Kw - 1};
    const std::vector<int64_t> expected_ssm_shape{Hv, Dv, Dk};
    VT_CHECK(mamba_spec->shapes[0] == expected_conv_shape &&
                 mamba_spec->shapes[1] == expected_ssm_shape,
             "runner: Qwen3.5 MambaSpec shapes disagree with model config");
    gdn_conv_cache_dtype_ = mamba_spec->dtypes[0];
    gdn_ssm_cache_dtype_ = mamba_spec->dtypes[1];
    const auto supported_state_dtype = [](vt::DType dtype) {
      return dtype == vt::DType::kF16 || dtype == vt::DType::kBF16 ||
             dtype == vt::DType::kF32;
    };
    VT_CHECK(supported_state_dtype(gdn_conv_cache_dtype_) &&
                 supported_state_dtype(gdn_ssm_cache_dtype_),
             "runner: Qwen3.5 MambaSpec state dtypes must be floating");
  }

  // Full-attn block size from the full-attn group's spec (its paged layout).
  int64_t fa_block_size = 0;
  if (full_attn_group_id_ >= 0) {
    fa_block_size =
        kv_cache_config.kv_cache_groups[static_cast<size_t>(full_attn_group_id_)]
            .kv_cache_spec->block_size;
  }

  const vt::Device dev = queue_.device;
  const char* device_cache_env = std::getenv("VT_DEVICE_KV_CACHE");
  kv_cache_backend_resident_ =
      vllm::platforms::GetPlatform(dev.type).is_cuda() &&
      (device_cache_env == nullptr || device_cache_env[0] != '0');
  full_attn_buf_.clear();
  ssm_buf_.clear();
  conv_buf_.clear();
  for (int64_t l = 0; l < config_.num_hidden_layers; ++l) {
    const bool is_gdn =
        config_.layer_types[static_cast<size_t>(l)] == "linear_attention";
    if (is_gdn) {
      VT_CHECK(mamba_spec != nullptr,
               "runner: linear-attention layer has no MambaSpec");
      // Raw buffers use their independent cache dtypes. Zero bytes are +0.0f
      // for every supported floating storage type.
      const size_t ssm_es = vt::SizeOf(gdn_ssm_cache_dtype_);
      const size_t conv_es = vt::SizeOf(gdn_conv_cache_dtype_);
      const int64_t conv_row_elems = conv_dim * (Kw - 1);
      const int64_t ssm_row_elems = Hv * Dv * Dk;
      ssm_buf_.push_back(std::make_unique<CacheBuffer>(
          dev, queue_,
          static_cast<size_t>(gdn_state_slots_ * ssm_row_elems) * ssm_es,
          kv_cache_backend_resident_));
      conv_buf_.push_back(std::make_unique<CacheBuffer>(
          dev, queue_,
          static_cast<size_t>(gdn_state_slots_ * conv_row_elems) * conv_es,
          kv_cache_backend_resident_));
    } else {
      // KV cache stored in kv_dtype (bf16 default; f32 if VT_KV_CACHE_F32). 0
      // bytes == 0.0 in both bf16 and f32. bf16 halves KV memory vs f32.
      full_attn_buf_.push_back(std::make_unique<CacheBuffer>(
          dev, queue_,
          static_cast<size_t>(num_blocks_ * 2 * fa_block_size * Hkv * Dh) *
              static_cast<size_t>(vt::SizeOf(kv_dtype)),
          kv_cache_backend_resident_));
    }
  }

  // Build the views over the (now stable) backing storage.
  attn_kv_.clear();
  for (auto& b : full_attn_buf_) {
    PagedKvCache kv;
    kv.data = b->data();
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
    gs.ssm_state = vt::Tensor::Contiguous(ssm_buf_[g]->data(),
                                          gdn_ssm_cache_dtype_,
                                          dev, {gdn_state_slots_, Hv, Dv, Dk});
    gs.conv_state = vt::Tensor::Contiguous(conv_buf_[g]->data(),
                                           gdn_conv_cache_dtype_,
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
  // Assign/reuse a compact slot per live sequence and write it into col 0 (the
  // only column the GDN metadata builder reads as the state index). Distinct
  // req_ids get distinct slots, so no two live sequences ever share a slot.
  for (int r = 0; r < num_reqs; ++r) {
    const std::string& rid = *req_ids[static_cast<size_t>(r)];
    const size_t off = static_cast<size_t>(r) * static_cast<size_t>(gdn_cols);
    auto it = gdn_slot_of_req_.find(rid);
    int32_t slot;
    if (it != gdn_slot_of_req_.end()) {
      slot = it->second;
    } else {
      VT_CHECK(!gdn_free_slots_.empty(),
               "GDN state slots exhausted: live sequences exceed max_num_reqs");
      slot = gdn_free_slots_.back();
      gdn_free_slots_.pop_back();
      gdn_slot_of_req_.emplace(rid, slot);
    }
    gdn_bt[off] = slot;
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

  // Async-scheduling device-input path (ENG-ASYNC-SCHED W3 runner leaf). When
  // engaged, overwrite each decode row's input token id with the GPU-resident-
  // analog last_sampled_tokens (combine_sampled_and_draft_tokens) instead of the
  // host token_ids_cpu value prepare_inputs read — so step N+1 need not wait on
  // step N's sampled token crossing to the host. idx_mapping is identity: our
  // persistent batch is already condensed dense (batch row == req_state slot).
  // Runs on the HOST side of input prep, BEFORE the forward and OUTSIDE any
  // CUDA-graph capture (input prep always precedes the decode graph replay), so
  // it is capture-safe. Default OFF: production keeps the byte-identical sync
  // host path (both give the same id, since sample_tokens writes the same token
  // to token_ids_cpu and last_sampled_tokens).
  if (async_input_combine_ && num_reqs > 0) {
#ifdef VLLM_CPP_CUDA
    if (vllm::platforms::GetPlatform(queue_.device.type).is_cuda()) {
      // DEVICE combine (W3 DGX leaf): splice each decode row's input id from the
      // device-resident-analog last_sampled_tokens on the MAIN queue, BEFORE the
      // forward (which embeds input_token_ids on the same queue → sees the patch)
      // and OUTSIDE any decode-graph capture. idx_mapping is identity (condensed-
      // dense batch), passed as nullptr. All inputs live in `step` (moved to
      // exec_state_, buffer preserved) / persistent InputBatch members, so they
      // outlive the async launch. On GB10's pageable memory the host arrays are
      // device-addressable; this removes the sample_tokens_async pre-scatter
      // Synchronize (see below). num_new_sampled_tokens == 1 (T0 non-spec).
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
  // Gather-before-lm_head indices (the SAME last-token rows sample_tokens uses).
  // Empty when the toggle is off → old full-logits path. The eager forwards skip
  // the gather when it is a no-op (pure decode: len == num_actual_tokens).
  const bool gather = LogitsGatherEnabled();
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
      .gather_logits = gather,
  };
  ForwardLogits logits = ModelRegistry::Forward(*model_, forward_input);

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
  const int num_reqs = exec_state_.num_reqs;
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
  return logits;
}

ModelRunnerOutput GPUModelRunner::sample_tokens(
    const std::optional<GrammarOutput>& grammar_output) {
  ModelRunnerOutput out;
  const int num_reqs = exec_state_.num_reqs;
  if (num_reqs == 0) {
    return out;  // 0-token flush step (nothing sampled).
  }

  std::vector<float> sampled_logits;  // host buffer; outlives the sampler when used
  vt::Tensor logits = assemble_sample_logits(grammar_output, sampled_logits);

  // SamplingMetadata in the SAME dense [0, num_reqs) order (M1.7; CLOSES the
  // make_sampling_metadata wiring dep). Then Sampler.forward.
  const SamplingMetadata sm = input_batch_.make_sampling_metadata();
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
  return out;
}

GPUModelRunner::~GPUModelRunner() {
  // Release the lazily-created async-output copy queue (gpu_model_runner.py has
  // no explicit teardown; our vt::Queue owns a CUDA stream that must be freed).
  if (async_copy_queue_.id != 0) {
    vt::DestroyQueue(async_copy_queue_);
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

std::unique_ptr<AsyncModelRunnerOutput> GPUModelRunner::sample_tokens_async(
    const std::optional<GrammarOutput>& grammar_output) {
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
  if (vllm::platforms::GetPlatform(dev.type).is_cuda()) {
    // DEVICE scatter (W3 DGX leaf): write each row's sampled id into
    // last_sampled_tokens on the MAIN queue — main-stream-ordered with the next
    // step's device combine, so the sampled ids NEVER round-trip the host. This
    // DELETES the pre-scatter `Synchronize` the host-array path needed (its ONLY
    // purpose was making the host read of dev_ids well-defined). On GB10's
    // pageable memory last_sampled_tokens.data() is device-addressable; idx_mapping
    // is identity (nullptr). Runs OUTSIDE any CUDA-graph capture (post-sample). The
    // host-side bookkeeping below needs NO device read.
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

  // Issue the non-blocking sampled-id D2H on the COPY queue + record the event.
  // The async output BORROWS the pool slot (device buffer already holds the
  // argmax ids) and releases it on consume — no per-step free.
  return std::make_unique<AsyncGPUModelRunnerOutput>(
      std::move(skeleton), dev, pool, slot, num_reqs, queue_, copy_q,
      std::move(invalid_req_indices));
}

}  // namespace vllm::v1
