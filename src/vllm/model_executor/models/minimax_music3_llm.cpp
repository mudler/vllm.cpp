// See minimax_music3_llm.h for why the language model needed an `inputs_embeds`
// entry, why the draw is a parameter, and why the condition mix's eight "layers"
// are not transformer layers.
#include "vllm/model_executor/models/minimax_music3_llm.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/backend.h"
#include "vt/dtype.h"

#include "vllm/model_executor/models/music3_profile.h"

namespace vllm {
namespace models {
namespace music3 {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void Fail(const std::string& message) { throw std::runtime_error(message); }

// The KV page size. 16 is what the engine's own dense caches use; the value is
// invisible to the result (paged attention is exact) and only sizes the
// allocation.
constexpr int64_t kBlockSize = 16;

// One checkpoint tensor at the AR half's RUNTIME dtype, in a float32 carrier.
//
// The AR half runs bf16 (spec §2.1, `MiniMaxMusic3ResolveRuntimeDtypes`), so an
// F32 file is ROUNDED rather than widened — the same rule the reduced-dimension
// and full-scale W2/W3 gates apply, and the one that made an fp32 depth forward
// miss 448450 of 716800 golden values.
std::vector<float> AtRuntimeDtype(const StTensor& tensor, const std::string& name) {
  if (tensor.dtype == "F32") {
    std::vector<float> out(tensor.nbytes / sizeof(float));
    std::memcpy(out.data(), tensor.data, tensor.nbytes);
    for (float& value : out) value = vt::BF16ToF32(vt::F32ToBF16(value));
    return out;
  }
  if (tensor.dtype == "BF16") {
    const size_t count = tensor.nbytes / sizeof(uint16_t);
    std::vector<float> out(count);
    const auto* raw = reinterpret_cast<const uint16_t*>(tensor.data);
    for (size_t i = 0; i < count; ++i) out[i] = vt::BF16ToF32(raw[i]);
    return out;
  }
  Fail("MiniMax-Music3: tensor '" + name + "' has dtype " + tensor.dtype +
       ", which the AR half cannot run (F32 or BF16 expected, spec §2.1)");
}

std::vector<uint16_t> ToBf16(const std::vector<float>& values) {
  std::vector<uint16_t> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) out[i] = vt::F32ToBF16(values[i]);
  return out;
}

// `language_model.lm_head(last_hidden).float()` (encoders.py:312). The shared
// dense forward emits f32; upstream's bf16 `nn.Linear` emits bf16 and only then
// widens. Restoring that rounding here is the header's recorded mirror — a
// NARROWING, applied at the one place the difference is observable.
void RoundToBf16(std::vector<float>& logits) {
  for (float& value : logits) value = vt::BF16ToF32(vt::F32ToBF16(value));
}

}  // namespace

// ---------------------------------------------------------------------------
// The draw
// ---------------------------------------------------------------------------

Music3CodeSampler Music3SeededSampler(int64_t seed) {
  return [seed](const std::vector<float>& probs, const Music3Draw& draw) -> int64_t {
    if (probs.empty()) Fail("MiniMax-Music3: the sampler was handed an empty distribution");
    // The draw index enters the seed so two draws of one request never share a
    // stream, which upstream's single running generator also avoids.
    std::mt19937_64 engine(static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull +
                           static_cast<uint64_t>(draw.frame_index) * 131ull +
                           static_cast<uint64_t>(draw.codebook));
    std::discrete_distribution<int64_t> categorical(probs.begin(), probs.end());
    return categorical(engine);
  };
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

std::vector<int32_t> Music3ArWeights::Encode(const std::string& text) const {
  // `components.tokenizer(text, return_tensors="pt")` (encoders.py:213). The
  // shipped `Qwen2Tokenizer` carries a ByteLevel post-processor with neither BOS
  // nor EOS, so `Encode` and `EncodeWithSpecialTokens` are the same bytes here;
  // the plain form is used because it is what upstream's call means.
  if (!tokenizer.has_value()) {
    Fail("MiniMax-Music3: the tokenizer was never loaded, so the prompt cannot be encoded");
  }
  return tokenizer->Encode(text);
}

std::vector<float> Music3ArWeights::EmbedRow(int64_t token_id) const {
  const int64_t H = lm_config.hidden_size;
  if (token_id < 0 || token_id >= lm_config.vocab_size) {
    Fail("MiniMax-Music3: embedding row " + std::to_string(token_id) +
         " is outside the language model's " + std::to_string(lm_config.vocab_size) +
         "-entry vocabulary");
  }
  const uint16_t* rows = embed_rows();
  std::vector<float> out(static_cast<size_t>(H));
  for (int64_t j = 0; j < H; ++j) {
    out[static_cast<size_t>(j)] = vt::BF16ToF32(rows[token_id * H + j]);
  }
  return out;
}

Music3ArWeights Music3LoadArWeights(const MiniMaxMusic3Paths& paths,
                                    const MiniMaxMusic3Config& config) {
  Music3ArWeights out;

  // ── the language model, through the SHARED dense loader ──────────────────
  out.lm_config = LoadHfConfig((fs::path(paths.language_model_dir) / "config.json").string());
  if (out.lm_config.vocab_size != config.language_model.vocab_size ||
      out.lm_config.hidden_size != config.language_model.hidden_size ||
      out.lm_config.num_hidden_layers != config.language_model.num_hidden_layers) {
    Fail("MiniMax-Music3: language_model/config.json disagrees with the loader's own read of "
         "it (vocab " +
         std::to_string(out.lm_config.vocab_size) + " vs " +
         std::to_string(config.language_model.vocab_size) + ", hidden " +
         std::to_string(out.lm_config.hidden_size) + " vs " +
         std::to_string(config.language_model.hidden_size) + ", layers " +
         std::to_string(out.lm_config.num_hidden_layers) + " vs " +
         std::to_string(config.language_model.num_hidden_layers) + ")");
  }
  // BREAKDOWN ROWS. These are SPANS, not leaves: they sit INSIDE the
  // `load.ar_weights` leaf that `minimax_music3_speech.cpp` brackets, so summing
  // them would double-count the load. They exist to answer one question that no
  // wall clock can — `SafetensorsFile` is an mmap and every tensor is COPIED out
  // of it, so page-fault I/O and the copy are interleaved inside one call and
  // "is the load I/O or CPU?" cannot be inferred from the total.
  std::vector<SafetensorsFile> shards;
  {
    profile::Timer open_timer("load.ar.open_shards", /*span=*/true);
    shards.reserve(paths.language_model_shards.size());
    for (const std::string& shard : paths.language_model_shards) {
      shards.push_back(SafetensorsFile::Open(shard));
    }
  }
  if (shards.empty()) Fail("MiniMax-Music3: language_model has no safetensors shards");
  {
    profile::Timer lm_timer("load.ar.lm_weights", /*span=*/true);
    out.lm = LoadQwen3ForCausalLMWeights(shards, out.lm_config);
  }
  if (out.lm.embed_tokens.bytes.size() !=
      static_cast<size_t>(out.lm_config.vocab_size) *
          static_cast<size_t>(out.lm_config.hidden_size) * sizeof(uint16_t)) {
    Fail("MiniMax-Music3: the language model's embedding table is not [vocab, hidden] bf16");
  }
  // `tie_word_embeddings: false` is a fact about THIS checkpoint (spec §1) and
  // the untied lm_head is 1.6 GB of real weights. A TIED read would silently
  // substitute the embedding table for the output projection — same shapes, same
  // finite logits, a different model.
  if (out.lm.tie_word_embeddings || out.lm.lm_head.Empty()) {
    Fail("MiniMax-Music3: the language model's lm_head is TIED or absent; this checkpoint's "
         "head is UNTIED (language_model/config.json tie_word_embeddings=false, spec §1) and a "
         "tied read would use the embedding table as the output projection");
  }

  // ── the RVQ depth decoder ────────────────────────────────────────────────
  out.depth_shipped = config.rvq_depth_decoder;
  out.depth_config.hidden_size = config.rvq_depth_decoder.hidden_size;
  out.depth_config.num_layers = config.rvq_depth_decoder.num_layers;
  out.depth_config.num_attention_heads = config.rvq_depth_decoder.num_attention_heads;
  out.depth_config.intermediate_size = config.rvq_depth_decoder.intermediate_size;
  out.depth_config.audio_vocab_size = config.rvq_depth_decoder.audio_vocab_size;
  out.depth_config.num_codebooks = config.rvq_depth_decoder.num_codebooks;
  out.depth_config.max_position_embeddings = config.rvq_depth_decoder.max_position_embeddings;

  const SafetensorsFile depth_file = SafetensorsFile::Open(
      (fs::path(paths.rvq_depth_decoder_dir) / "diffusion_pytorch_model.safetensors").string());
  const auto depth_t0 = profile::Now();
  const auto depth_get = [&depth_file](const std::string& name) {
    return AtRuntimeDtype(depth_file.Get(name), name);
  };
  out.depth.audio_embeddings = depth_get("audio_embeddings.weight");
  out.depth.projection = depth_get("projection.weight");
  out.depth.pos_embedding = depth_get("pos_embedding.weight");
  out.depth.norm = depth_get("norm.weight");
  for (int64_t layer = 0; layer < out.depth_config.num_layers; ++layer) {
    const std::string base = "layers." + std::to_string(layer) + ".";
    DepthDecoderLayerWeights entry;
    entry.input_layernorm = depth_get(base + "input_layernorm.weight");
    entry.post_attention_layernorm = depth_get(base + "post_attention_layernorm.weight");
    entry.to_q = depth_get(base + "attn.to_q.weight");
    entry.to_k = depth_get(base + "attn.to_k.weight");
    entry.to_v = depth_get(base + "attn.to_v.weight");
    entry.to_out = depth_get(base + "attn.to_out.weight");
    entry.gate_proj = depth_get(base + "gate_proj.weight");
    entry.up_proj = depth_get(base + "up_proj.weight");
    entry.down_proj = depth_get(base + "down_proj.weight");
    out.depth.layers.push_back(std::move(entry));
  }
  for (int64_t head = 0; head < out.depth_config.residual_codebooks(); ++head) {
    out.depth.audio_heads.push_back(
        depth_get("audio_heads." + std::to_string(head) + ".weight"));
  }

  profile::AddSince("load.ar.depth_weights", depth_t0, /*span=*/true);

  // ── the tokenizer ────────────────────────────────────────────────────────
  const auto tok_t0 = profile::Now();
  out.tokenizer =
      tok::Tokenizer::FromHfJson((fs::path(paths.tokenizer_dir) / "tokenizer.json").string());
  profile::AddSince("load.ar.tokenizer", tok_t0, /*span=*/true);
  return out;
}

// ---------------------------------------------------------------------------
// The decode session
// ---------------------------------------------------------------------------

Music3LmSession::Music3LmSession(const Music3ArWeights& weights, vt::Queue& queue,
                                 int64_t max_positions)
    : weights_(weights), queue_(queue), block_size_(kBlockSize) {
  if (max_positions <= 0) {
    Fail("MiniMax-Music3: the language-model session needs at least one position");
  }
  const int64_t limit = weights_.lm_config.max_position_embeddings;
  if (limit > 0 && max_positions > limit) {
    // spec §7: the checkpoint's context limit is ours too, and it is ENFORCED
    // rather than discovered as a garbage RoPE extrapolation.
    Fail("MiniMax-Music3: the request needs " + std::to_string(max_positions) +
         " language-model positions, past the checkpoint's " + std::to_string(limit) +
         "-position context (language_model/config.json max_position_embeddings)");
  }
  blocks_per_row_ = (max_positions + block_size_ - 1) / block_size_;

  const HfConfig& c = weights_.lm_config;
  const int64_t per_block =
      2 * block_size_ * c.num_key_value_heads * c.head_dim;  // k and v
  const size_t elems = static_cast<size_t>(kRows * blocks_per_row_ * per_block);
  // WHERE the cache lands is the queue's decision and nothing else's.
  // `dense_attn::KvSlice` labels this pointer with `d.q.device`
  // (dense_attn_block.h:233), so a host `std::vector` handed to a CUDA forward
  // is a host pointer wearing a device tensor's label: finite, correctly shaped,
  // and read by a kernel that cannot dereference it.
  const bool on_device = queue_.device.type != vt::DeviceType::kCPU;
  if (on_device) {
    device_kv_.assign(static_cast<size_t>(c.num_hidden_layers), nullptr);
  } else {
    kv_storage_.resize(static_cast<size_t>(c.num_hidden_layers));
  }
  attn_kv_.reserve(static_cast<size_t>(c.num_hidden_layers));
  for (int64_t layer = 0; layer < c.num_hidden_layers; ++layer) {
    // BF16, which is what the dense forward WRITES (qwen3_5.h: the paged store
    // down-casts K/V to bf16). An f32 cache here would be a wider memory format
    // than the path that fills it.
    PagedKvCache kv;
    if (on_device) {
      // ZEROED, and the zero matters as much as the allocation: `KvSlice` hands
      // the whole [num_blocks, block_size, Hkv, Dh] view to the kernel, so the
      // unwritten tail of the last block is real memory. Uninitialised device
      // bytes there are the classic NaN nobody can localize; the CPU arm's
      // `assign(elems, 0)` has always zeroed for the same reason.
      vt::Backend& backend = vt::GetBackend(queue_.device.type);
      void* p = backend.Alloc(elems * sizeof(uint16_t));
      device_kv_[static_cast<size_t>(layer)] = p;
      backend.Memset(queue_, p, 0, elems * sizeof(uint16_t));
      kv.data = p;
    } else {
      kv_storage_[static_cast<size_t>(layer)].assign(elems, 0);
      kv.data = kv_storage_[static_cast<size_t>(layer)].data();
    }
    kv.dtype = vt::DType::kBF16;
    kv.num_blocks = kRows * blocks_per_row_;
    kv.block_size = block_size_;
    kv.num_kv_heads = c.num_key_value_heads;
    kv.head_size = c.head_dim;
    attn_kv_.push_back(kv);
  }
}

Music3LmSession::~Music3LmSession() {
  if (device_kv_.empty()) return;  // the CPU arm owns nothing to free
  vt::Backend& backend = vt::GetBackend(queue_.device.type);
  // Synchronize before freeing: the last step's kernels can still be reading the
  // cache when the request returns, and a free racing a live kernel surfaces as
  // a fault attributed to whatever allocates next.
  backend.Synchronize(queue_);
  for (void* p : device_kv_) {
    if (p != nullptr) backend.Free(p);
  }
}

std::vector<float> Music3LmSession::RunOne(const std::vector<uint16_t>& embeds_bf16,
                                           int64_t tokens_per_row,
                                           std::vector<float>* out_hidden) {
  const HfConfig& c = weights_.lm_config;
  const int64_t H = c.hidden_size;
  if (static_cast<int64_t>(embeds_bf16.size()) != kRows * tokens_per_row * H) {
    Fail("MiniMax-Music3: the language-model step was handed " +
         std::to_string(embeds_bf16.size()) + " values, " + std::to_string(kRows) + " rows x " +
         std::to_string(tokens_per_row) + " tokens needs " +
         std::to_string(kRows * tokens_per_row * H));
  }
  if (position_ + tokens_per_row > blocks_per_row_ * block_size_) {
    Fail("MiniMax-Music3: the language-model session is full at " + std::to_string(position_) +
         " positions and cannot take " + std::to_string(tokens_per_row) + " more");
  }

  v1::CommonAttentionMetadata meta;
  meta.num_reqs = static_cast<int>(kRows);
  meta.num_actual_tokens = static_cast<int>(kRows * tokens_per_row);
  meta.max_query_len = static_cast<int>(tokens_per_row);
  meta.max_seq_len = static_cast<int>(position_ + tokens_per_row);
  meta.causal = true;
  meta.block_table_num_cols = static_cast<int>(blocks_per_row_);
  meta.query_start_loc.resize(static_cast<size_t>(kRows + 1));
  for (int64_t r = 0; r <= kRows; ++r) {
    meta.query_start_loc[static_cast<size_t>(r)] = static_cast<int32_t>(r * tokens_per_row);
  }
  meta.query_start_loc_cpu = meta.query_start_loc;
  meta.seq_lens.assign(static_cast<size_t>(kRows),
                       static_cast<int32_t>(position_ + tokens_per_row));
  meta.seq_lens_cpu = meta.seq_lens;
  meta.num_computed_tokens_cpu.assign(static_cast<size_t>(kRows),
                                      static_cast<int32_t>(position_));
  // Each row owns a CONTIGUOUS run of blocks, so row 1 can never read row 0's
  // pages: the two prompts differ and a shared page would silently guide against
  // the wrong context.
  meta.block_table_tensor.resize(static_cast<size_t>(kRows * blocks_per_row_));
  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t b = 0; b < blocks_per_row_; ++b) {
      meta.block_table_tensor[static_cast<size_t>(r * blocks_per_row_ + b)] =
          static_cast<int32_t>(r * blocks_per_row_ + b);
    }
  }

  std::vector<int32_t> positions(static_cast<size_t>(kRows * tokens_per_row));
  std::vector<int32_t> logits_indices(static_cast<size_t>(kRows));
  meta.slot_mapping.resize(static_cast<size_t>(kRows * tokens_per_row));
  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t t = 0; t < tokens_per_row; ++t) {
      const int64_t p = position_ + t;
      const size_t flat = static_cast<size_t>(r * tokens_per_row + t);
      positions[flat] = static_cast<int32_t>(p);
      const int64_t block = r * blocks_per_row_ + p / block_size_;
      meta.slot_mapping[flat] = block * block_size_ + p % block_size_;
    }
    // `last_hidden_state[:, -1]` (encoders.py:312): the LAST position of each
    // row, gathered before lm_head so only two vocabulary rows are produced.
    logits_indices[static_cast<size_t>(r)] =
        static_cast<int32_t>(r * tokens_per_row + tokens_per_row - 1);
  }

  std::vector<float> logits = Qwen3DenseModel::ForwardEmbeds(
      embeds_bf16, positions, meta, attn_kv_, weights_.lm, c, queue_, logits_indices,
      out_hidden);
  position_ += tokens_per_row;
  return logits;
}

void Music3LmSession::Prefill(const std::vector<int32_t>& conditional_ids,
                              const std::vector<int32_t>& unconditional_ids,
                              std::vector<float>* out_hidden,
                              std::vector<float>* out_logits) {
  if (conditional_ids.empty()) Fail("MiniMax-Music3: the prompt has no tokens");
  if (conditional_ids.size() != unconditional_ids.size()) {
    Fail("MiniMax-Music3: the unconditional prompt row has " +
         std::to_string(unconditional_ids.size()) + " tokens and the conditional row has " +
         std::to_string(conditional_ids.size()) + "; they are a token-for-token rewrite");
  }
  const int64_t H = weights_.lm_config.hidden_size;
  const int64_t T = static_cast<int64_t>(conditional_ids.size());
  const uint16_t* table = weights_.embed_rows();
  // `embed_tokens(text_ids)` (encoders.py:310), on the host: the two rows are
  // built here and handed to the SAME `inputs_embeds` entry the feedback steps
  // use, so the prompt and the feedback take one code path rather than two.
  std::vector<uint16_t> embeds(static_cast<size_t>(kRows * T * H));
  for (int64_t r = 0; r < kRows; ++r) {
    const std::vector<int32_t>& ids = r == 0 ? conditional_ids : unconditional_ids;
    for (int64_t t = 0; t < T; ++t) {
      const int32_t id = ids[static_cast<size_t>(t)];
      if (id < 0 || id >= weights_.lm_config.vocab_size) {
        Fail("MiniMax-Music3: prompt token " + std::to_string(id) +
             " is outside the language model's vocabulary");
      }
      std::memcpy(&embeds[static_cast<size_t>((r * T + t) * H)],
                  &table[static_cast<int64_t>(id) * H],
                  static_cast<size_t>(H) * sizeof(uint16_t));
    }
  }
  *out_logits = RunOne(embeds, T, out_hidden);
}

void Music3LmSession::Step(const std::vector<float>& embeds, std::vector<float>* out_hidden,
                           std::vector<float>* out_logits) {
  const int64_t H = weights_.lm_config.hidden_size;
  if (static_cast<int64_t>(embeds.size()) != kRows * H) {
    Fail("MiniMax-Music3: a feedback step needs " + std::to_string(kRows * H) +
         " values, got " + std::to_string(embeds.size()));
  }
  *out_logits = RunOne(ToBf16(embeds), /*tokens_per_row=*/1, out_hidden);
}

// ---------------------------------------------------------------------------
// The depth stage
// ---------------------------------------------------------------------------

std::vector<float> Music3DepthStage(const std::vector<float>& last_hidden_conditional,
                                    const std::vector<float>& last_hidden_unconditional,
                                    int64_t frame_index, const Music3ArWeights& weights,
                                    const Music3CodeSampler& sampler,
                                    std::vector<int32_t>* out_frame_codes) {
  // `_generate_depth_codes` (encoders.py:117-142). The two prompt rows share
  // every sequence entry EXCEPT position 0, which is each row's own
  // `projection(last_hidden)` — that is the whole of what the unconditional
  // branch contributes, and dropping it collapses CFG to a plain sample.
  const DepthDecoderConfig& config = weights.depth_config;
  const int64_t H = config.hidden_size;
  if (out_frame_codes == nullptr) Fail("MiniMax-Music3: the depth stage needs a codes pointer");
  if (static_cast<int64_t>(last_hidden_conditional.size()) != H ||
      static_cast<int64_t>(last_hidden_unconditional.size()) != H) {
    Fail("MiniMax-Music3: the depth stage needs two [hidden_size] language-model states");
  }
  if (out_frame_codes->size() != 1) {
    Fail("MiniMax-Music3: the depth stage starts from the frame's semantic code alone");
  }
  profile::Timer depth_timer("ar.depth_stage", /*span=*/true);
  const int32_t semantic = (*out_frame_codes)[0];
  const std::vector<float> semantic_embed =
      weights.EmbedRow(static_cast<int64_t>(semantic) + kAudioCodeOffset);

  std::vector<float> depth_hidden;  // [ (num_codebooks-1) * H ], conditional row
  depth_hidden.reserve(static_cast<size_t>(config.residual_codebooks() * H));

  // THE SCHEDULE, AND WHY IT IS INCREMENTAL (#672). Upstream re-runs the whole
  // growing depth sequence at every step and keeps `hidden[:, -1]`
  // (encoders.py:125-141); so did this loop, through `DepthSequenceEmbeds` +
  // `DepthDecoderForward`, which re-projected and re-forwarded 70 rows per frame
  // to read 14 of them. The decoder is causal with a LEARNED position table, so
  // a row's every intermediate is fixed the moment it is appended:
  // `DepthDecoderAppend` computes each row ONCE and is bit-identical to the
  // whole-sequence forward it replaces (minimax_music3_ar.h, and
  // `test_minimax_music3_ar.cpp` asserts it bitwise at every prefix length).
  //
  // The two CFG rows go through as ONE batch-2 call — which is upstream's own
  // shape, `(2, 2..8, 4096)` — so every weight sweep serves both branches.
  DepthDecoderCache cache;

  // The depth sequence's first two rows: each branch's own
  // `projection(last_hidden)` at position 0, and the SHARED
  // `projection(semantic_embed)` at position 1. Three rows, ONE weight sweep,
  // and the projection is never recomputed for a row already in the cache —
  // upstream keeps its projected rows in `sequence` for exactly this reason
  // (encoders.py:125,127,141).
  std::vector<float> prefix_rows;
  prefix_rows.reserve(static_cast<size_t>(3 * H));
  prefix_rows.insert(prefix_rows.end(), last_hidden_conditional.begin(),
                     last_hidden_conditional.end());
  prefix_rows.insert(prefix_rows.end(), last_hidden_unconditional.begin(),
                     last_hidden_unconditional.end());
  prefix_rows.insert(prefix_rows.end(), semantic_embed.begin(), semantic_embed.end());
  std::vector<float> prefix;
  {
    profile::Timer embed_timer("ar.depth_projection");
    prefix = LinearNoBias(prefix_rows, 3, H, weights.depth.projection, H, ArCompute::kBFloat16);
  }

  // Position 0 is fed for its K/V only: `_generate_depth_codes` reads the last
  // row of a length-2 sequence first, so position 0's own output is never used.
  {
    profile::Timer forward_timer("ar.depth_forward");
    DepthDecoderAppend(std::vector<float>(prefix.begin(), prefix.begin() + 2 * H), /*batch=*/2,
                       config, weights.depth, ArCompute::kBFloat16, &cache);
  }

  std::vector<float> next(static_cast<size_t>(2 * H));
  std::copy(prefix.begin() + 2 * H, prefix.begin() + 3 * H, next.begin());
  std::copy(prefix.begin() + 2 * H, prefix.begin() + 3 * H, next.begin() + H);

  for (int64_t index = 1; index < config.num_codebooks; ++index) {
    std::vector<float> states;
    {
      profile::Timer forward_timer("ar.depth_forward");
      states = DepthDecoderAppend(next, /*batch=*/2, config, weights.depth,
                                  ArCompute::kBFloat16, &cache);
    }
    std::vector<float> hidden_rows[2];
    hidden_rows[0].assign(states.begin(), states.begin() + H);
    hidden_rows[1].assign(states.begin() + H, states.end());
    // `hidden_parts.append(hidden[:1])` — the CONDITIONAL row only, which is why
    // `frame_hiddens` can be reproduced from a golden that stores no other.
    depth_hidden.insert(depth_hidden.end(), hidden_rows[0].begin(), hidden_rows[0].end());

    int64_t drawn = 0;
    {
      profile::Timer head_timer("ar.depth_head_and_draw");
      const std::vector<float> conditional = AudioHeadLogits(
          hidden_rows[0], index - 1, config, weights.depth, ArCompute::kBFloat16);
      const std::vector<float> unconditional = AudioHeadLogits(
          hidden_rows[1], index - 1, config, weights.depth, ArCompute::kBFloat16);
      const std::vector<float> guided =
          GuidedDepthLogits(conditional, unconditional, kArCfgScale);
      const std::vector<float> probs = TopKProbabilities(guided, kArSamplingTopK);
      drawn = sampler(probs, Music3Draw{frame_index, index});
    }
    if (drawn < 0 || drawn >= config.audio_vocab_size) {
      Fail("MiniMax-Music3: the sampler returned residual code " + std::to_string(drawn) +
           " for codebook " + std::to_string(index) + ", outside the " +
           std::to_string(config.audio_vocab_size) + "-entry audio vocabulary");
    }
    out_frame_codes->push_back(static_cast<int32_t>(drawn));
    // c7 is only ever PREDICTED (encoders.py:139): feeding it back would need a
    // 17th position the decoder's `pos_embedding` does not have.
    if (index < config.num_codebooks - 1) {
      // encoders.py:140 — `index` there is ONE-based, so the offset is (index-1),
      // the same row `DepthSequenceEmbeds` builds for `fed_back[index-1]`.
      const int64_t row = (index - 1) * config.audio_vocab_size + drawn;
      const size_t at = static_cast<size_t>(row * H);
      if (at + static_cast<size_t>(H) > weights.depth.audio_embeddings.size()) {
        Fail("MiniMax-Music3: audio_embeddings row " + std::to_string(row) +
             " is past the end of a table of " +
             std::to_string(weights.depth.audio_embeddings.size() / static_cast<size_t>(H)) +
             " rows");
      }
      const std::vector<float> embed(
          weights.depth.audio_embeddings.begin() + static_cast<int64_t>(at),
          weights.depth.audio_embeddings.begin() + static_cast<int64_t>(at) + H);
      profile::Timer embed_timer("ar.depth_projection");
      const std::vector<float> projected =
          LinearNoBias(embed, 1, H, weights.depth.projection, H, ArCompute::kBFloat16);
      std::copy(projected.begin(), projected.end(), next.begin());
      std::copy(projected.begin(), projected.end(), next.begin() + H);
    }
  }
  return depth_hidden;
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

Music3ArResult Music3GenerateFrameHiddens(const std::vector<int32_t>& prompt_ids,
                                          int64_t max_frames,
                                          const Music3ArWeights& weights,
                                          const Music3CodeSampler& sampler,
                                          vt::Queue& queue) {
  if (!sampler) Fail("MiniMax-Music3: the autoregressive loop needs a code sampler");
  if (max_frames <= 0) {
    Fail("MiniMax-Music3: the autoregressive loop needs a positive frame budget, got " +
         std::to_string(max_frames));
  }
  if (static_cast<int64_t>(prompt_ids.size()) > kMaxPromptTokens) {
    // The TOKEN-side ceiling upstream raises (encoders.py:214-217). W6's
    // request contract checks the prompt's BYTE length, which is an upper bound
    // on the token count and so cannot catch a prompt that is merely long.
    Fail("MiniMax-Music3: the assembled prompt is " + std::to_string(prompt_ids.size()) +
         " tokens, past the checkpoint's " + std::to_string(kMaxPromptTokens) + "-token ceiling");
  }

  const DepthDecoderConfig& depth = weights.depth_config;
  const int64_t H = weights.lm_config.hidden_size;
  const int64_t vocab = weights.lm_config.vocab_size;
  const std::vector<int32_t> unconditional_ids = UnconditionalPromptIds(prompt_ids);
  const std::vector<bool> blocked =
      SemanticVocabMask(vocab, kAudioCodeOffset, kSemanticVocabSize, kAudioEndTokenId);

  // The prompt plus at most one feedback position per loop iteration.
  Music3LmSession session(weights, queue,
                          static_cast<int64_t>(prompt_ids.size()) + max_frames + 1);

  std::vector<float> hidden;  // [2, H]
  std::vector<float> logits;  // [2, vocab]
  {
    profile::Timer prefill_timer("ar.lm_prefill");
    session.Prefill(prompt_ids, unconditional_ids, &hidden, &logits);
  }
  profile::Count("ar.prompt_tokens", static_cast<int64_t>(prompt_ids.size()));

  Music3ArResult result;
  result.frame_hiddens.reserve(
      static_cast<size_t>(max_frames * depth.num_codebooks * H));
  result.codes.reserve(static_cast<size_t>((max_frames + 1) * depth.num_codebooks));

  for (int64_t frame_index = 0; frame_index <= max_frames; ++frame_index) {
    if (static_cast<int64_t>(logits.size()) != 2 * vocab) {
      Fail("MiniMax-Music3: the language model returned " + std::to_string(logits.size()) +
           " logits, 2 rows x " + std::to_string(vocab) + " expected");
    }
    // encoders.py:312 — the head's output is bf16 before `.float()`.
    int64_t sampled = 0;
    {
      // The guided-logits half of the frame: two 200 000-wide rows, a top-k over
      // each, and the draw. §11.1's table calls this "not the cost"; that was an
      // argument, and this bracket is what turns it into a number.
      profile::Timer semantic_timer("ar.semantic_guide_and_draw");
      RoundToBf16(logits);
      const std::vector<float> conditional(logits.begin(),
                                           logits.begin() + static_cast<ptrdiff_t>(vocab));
      const std::vector<float> unconditional(logits.begin() + static_cast<ptrdiff_t>(vocab),
                                             logits.end());
      const std::vector<float> guided =
          GuidedSemanticLogits(conditional, unconditional, blocked, kArCfgTopK, kArCfgScale);
      const std::vector<float> probs = TopKProbabilities(guided, kArSamplingTopK);
      sampled = sampler(probs, Music3Draw{frame_index, 0});
    }
    if (sampled < 0 || sampled >= vocab) {
      Fail("MiniMax-Music3: the sampler returned token " + std::to_string(sampled) +
           ", outside the language model's " + std::to_string(vocab) + "-entry vocabulary");
    }
    if (sampled == kAudioEndTokenId) {
      result.stopped_on_end_token = true;
      break;
    }
    const int64_t semantic_code = sampled - kAudioCodeOffset;
    if (semantic_code < 0 || semantic_code >= kSemanticVocabSize) {
      // Only the code window and the end token survive `SemanticVocabMask`, so
      // this is unreachable unless a sampler ignored the distribution it was
      // handed — which is exactly what a teacher-forcing gate could do by
      // accident.
      Fail("MiniMax-Music3: the language model's draw " + std::to_string(sampled) +
           " is neither the audio-end token nor inside the semantic code window [" +
           std::to_string(kAudioCodeOffset) + ", " +
           std::to_string(kAudioCodeOffset + kSemanticVocabSize) + ")");
    }

    std::vector<int32_t> frame_codes{static_cast<int32_t>(semantic_code)};
    const std::vector<float> depth_hidden = Music3DepthStage(
        std::vector<float>(hidden.begin(), hidden.begin() + static_cast<ptrdiff_t>(H)),
        std::vector<float>(hidden.begin() + static_cast<ptrdiff_t>(H), hidden.end()),
        frame_index, weights, sampler, &frame_codes);
    result.codes.insert(result.codes.end(), frame_codes.begin(), frame_codes.end());
    ++result.calls;

    // `if frame_index > 0` (encoders.py:342): the first decode only advances the
    // state past `<|audio_start|>` and emits no frame, which is why `rvq_codes`
    // has one MORE row than `frame_hiddens`.
    if (frame_index > 0) {
      // `cat((last_hidden[:1], depth_hidden), dim=-1)` (encoders.py:343). NOT
      // `FrameHiddenRow`: that helper takes the WHOLE-SEQUENCE depth forward's
      // [num_codebooks, H] block and drops position 0, which is the shape the
      // W2/W3 gate has. The generation loop already holds the seven per-step
      // states themselves, so the same row is two inserts rather than a slice of
      // a tensor it would have to rebuild.
      if (static_cast<int64_t>(depth_hidden.size()) != depth.residual_codebooks() * H) {
        Fail("MiniMax-Music3: the depth stage returned " +
             std::to_string(depth_hidden.size()) + " values, " +
             std::to_string(depth.residual_codebooks()) + " x " + std::to_string(H) +
             " expected");
      }
      result.frame_hiddens.insert(result.frame_hiddens.end(), hidden.begin(),
                                  hidden.begin() + static_cast<ptrdiff_t>(H));
      result.frame_hiddens.insert(result.frame_hiddens.end(), depth_hidden.begin(),
                                  depth_hidden.end());
      ++result.frames;
      if (result.frames >= max_frames) break;
    }

    // `_embed_audio_frame` (encoders.py:106-115, :352). Both rows receive the
    // SAME vector: upstream repeats the sampled codes across the batch, so the
    // rows differ only through their KV history.
    const std::vector<int32_t> residual(frame_codes.begin() + 1, frame_codes.end());
    const std::vector<float> feedback = EmbedAudioFrame(
        weights.EmbedRow(sampled), residual, depth, weights.depth, ArCompute::kBFloat16);
    std::vector<float> both(feedback);
    both.insert(both.end(), feedback.begin(), feedback.end());
    {
      profile::Timer step_timer("ar.lm_decode_step");
      session.Step(both, &hidden, &logits);
    }
  }

  if (result.frames == 0) {
    // Upstream's own refusal (encoders.py:349-350), rather than handing the
    // acoustic half an empty tensor that would crop to nothing four stages later.
    Fail("MiniMax-Music3: the language model generated zero audio frames; the prompt ended "
         "generation immediately");
  }
  return result;
}

}  // namespace music3
}  // namespace models
}  // namespace vllm
