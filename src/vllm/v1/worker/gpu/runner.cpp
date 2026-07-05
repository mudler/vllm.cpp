// Ported from: vllm/v1/worker/gpu/model_runner.py @ e24d1b24
// (initialize_kv_cache / execute_model / sample_tokens / sample /
// postprocess_sampled — the T0 slice) + the decode-first reorder from
// vllm/v1/attention/backends/utils.py::reorder_batch_to_split_decodes_and_prefills.
// See include/vllm/v1/worker/gpu/runner.h for scope, the V1-algorithm / MRV2-
// contract composition, the four-way ordering contract, and the deferred paths.
#include "vllm/v1/worker/gpu/runner.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "vllm/v1/sample/ops/bad_words.h"  // apply_allowed_token_ids (-inf mask)
#include "vt/tensor.h"

namespace vllm::v1 {

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
  initialize_kv_cache(kv_cache_config);
}

void GPUModelRunner::initialize_kv_cache(const KVCacheConfig& kv_cache_config) {
  num_blocks_ = kv_cache_config.num_blocks;

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
  for (int64_t l = 0; l < config_.num_hidden_layers; ++l) {
    const bool is_gdn =
        config_.layer_types[static_cast<size_t>(l)] == "linear_attention";
    if (is_gdn) {
      ssm_buf_.emplace_back(
          static_cast<size_t>(num_blocks_ * Hv * Dv * Dk), 0.0f);
      conv_buf_.emplace_back(
          static_cast<size_t>(num_blocks_ * conv_dim * (Kw - 1)), 0.0f);
    } else {
      full_attn_buf_.emplace_back(
          static_cast<size_t>(num_blocks_ * 2 * fa_block_size * Hkv * Dh), 0.0f);
    }
  }

  // Build the views over the (now stable) backing storage.
  attn_kv_.clear();
  for (auto& b : full_attn_buf_) {
    PagedKvCache kv;
    kv.data = b.data();
    kv.dtype = vt::DType::kF32;
    kv.num_blocks = num_blocks_;
    kv.block_size = fa_block_size;
    kv.num_kv_heads = Hkv;
    kv.head_size = Dh;
    attn_kv_.push_back(kv);
  }
  gdn_state_.clear();
  for (size_t g = 0; g < ssm_buf_.size(); ++g) {
    GdnStateCache gs;
    gs.ssm_state = vt::Tensor::Contiguous(ssm_buf_[g].data(), vt::DType::kF32,
                                          dev, {num_blocks_, Hv, Dv, Dk});
    gs.conv_state = vt::Tensor::Contiguous(conv_buf_[g].data(), vt::DType::kF32,
                                           dev, {num_blocks_, conv_dim, Kw - 1});
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
  const std::vector<int32_t> gdn_bt =
      gather_block_table(gdn_group_id_, num_reqs, &gdn_cols);
  const CommonAttentionMetadata gdn_cam = MakeCommonAttentionMetadata(
      step, gdn_bt, gdn_cols, /*causal=*/true, gdn_group_id_);
  GDNAttentionMetadataBuilder gdn_builder;
  GDNAttentionMetadata gdn_meta =
      gdn_builder.build(/*common_prefix_len=*/0, gdn_cam);

  // Flattened dense-order forward inputs (positions int64 -> int32 for RoPE).
  const std::vector<int32_t>& token_ids = step.input_token_ids;
  std::vector<int32_t> positions(step.positions.begin(), step.positions.end());

  // THE FORWARD (Task 3, over the persistent KV caches). Returns the full
  // [num_actual_tokens, vocab] f32 logits (lm_head already applied).
  //
  // DECODE CUDA-GRAPH path (M2.5 Phase 2): a single-stream PURE-DECODE step
  // (num_reqs==1, one token, no prefill) on an fp4/CUDA model is routed through
  // the Qwen3_5DecodeGraph, which captures the forward once per batch shape and
  // replays it per token. The graph's output is bit-identical to Forward.
  //
  // WHY num_reqs==1 ONLY (measured, GB10 free-box A/B): after M2.5 Phase 1 made
  // the decode forward async-on-stream, the host launch overhead is largely
  // hidden behind the GPU, so graph capture recovers only ~2.6% at num_reqs==1
  // (TPOT 67.2 -> 65.5 ms) and is neutral-to-slightly-negative at num_reqs==8
  // (larger per-kernel work + more captured GDN-gather nodes). Restricting to
  // num_reqs==1 keeps the single-stream latency win without regressing batched
  // decode throughput (the gate-#1 workload). Batched decode stays eager; a
  // captured SET of small batch sizes (vLLM's cudagraph_capture_sizes) is future
  // work but the measured headroom is small. Prefill / mixed / bf16 / CPU always
  // stay eager.
  const bool pure_decode = attn_meta.num_actual_tokens == num_reqs &&
                           gdn_meta.num_prefill_tokens == 0;
  const bool fp4_cuda = queue_.device.type == vt::DeviceType::kCUDA &&
                        moe_weights_ != nullptr &&
                        !moe_weights_->layers.empty() &&
                        !moe_weights_->layers.front().moe.expert_gate_fp4.empty();
  std::vector<float> logits;
  if (dense_weights_ != nullptr) {
    // DENSE arch (27B): the eager paged dense forward. Same paged KV/GDN-state
    // machinery, dense SwiGLU MLP in place of the MoE block. The MoE-only fp4
    // decode-graph fast path does not apply here.
    logits = Qwen3_5DenseModel::Forward(token_ids, positions, attn_meta, gdn_meta,
                                        attn_kv_, gdn_state_, *dense_weights_,
                                        config_, queue_);
  } else if (pure_decode && num_reqs == 1 && fp4_cuda) {
    if (!decode_graph_) {
      decode_graph_ = std::make_unique<Qwen3_5DecodeGraph>(*moe_weights_, config_,
                                                           queue_);
    }
    logits = decode_graph_->Step(token_ids, positions, attn_meta, gdn_meta,
                                 attn_kv_, gdn_state_);
  } else {
    logits =
        Qwen3_5Model::Forward(token_ids, positions, attn_meta, gdn_meta, attn_kv_,
                              gdn_state_, *moe_weights_, config_, queue_);
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

  return std::nullopt;  // MRV2 split: forward done, sample separately.
}

ModelRunnerOutput GPUModelRunner::sample_tokens(
    const std::optional<GrammarOutput>& grammar_output) {
  ModelRunnerOutput out;
  const int num_reqs = exec_state_.num_reqs;
  if (num_reqs == 0) {
    return out;  // 0-token flush step (nothing sampled).
  }

  const int64_t vocab = config_.vocab_size;

  // Gather the logits_indices rows: [num_actual_tokens, vocab] -> [num_reqs,
  // vocab] (upstream hidden_states[input_batch.logits_indices] -> compute_logits;
  // our Forward already applied lm_head, so we gather the logit rows directly).
  std::vector<float> sampled_logits(static_cast<size_t>(num_reqs) *
                                    static_cast<size_t>(vocab));
  for (int i = 0; i < num_reqs; ++i) {
    const int row = exec_state_.step.logits_indices[static_cast<size_t>(i)];
    std::memcpy(
        sampled_logits.data() + static_cast<size_t>(i) * static_cast<size_t>(vocab),
        exec_state_.logits.data() +
            static_cast<size_t>(row) * static_cast<size_t>(vocab),
        static_cast<size_t>(vocab) * sizeof(float));
  }
  vt::Tensor logits = vt::Tensor::Contiguous(
      sampled_logits.data(), vt::DType::kF32, queue_.device,
      {static_cast<int64_t>(num_reqs), vocab});

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
  return out;
}

}  // namespace vllm::v1
