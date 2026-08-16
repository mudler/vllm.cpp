// ENG-EXPERT-STREAM (#912) F3: does DECODE actually reach the streamed-expert
// lane, and does the lane stay alive for the whole run?
//
// WHY THIS FILE EXISTS. Every other test of this row constructs the cache, the
// store and the streamer by hand and drives them directly. All of them passed
// while the production wiring was broken in two separate ways, because none of
// them ran a forward. An independent review measured exactly that: replacing the
// production call site with `nullptr`, and forcing streaming unconditionally ON,
// BOTH left the full gate green, and `Qwen35ExpertStream`, `KqExpertSlice` and
// `VT_MOE_EXPERT_STREAM` appeared nowhere under `tests/` at all. A unit test that
// builds the type by hand proves the class works; it never proves anything
// reaches it. See `.agents/reachability.md`.
//
// So this test enters through a PRODUCTION entry point — `Qwen3_5Model::Forward`,
// the paged forward the runner calls — over a synthetic MoE whose routed experts
// are keep-quant stacked towers, which is the shape the streaming seam serves.
// It then asks the lane what happened.
//
// THE TWO NUMBERS THAT MATTER, and why they are the ones asserted:
//
//   fills > 0   decode reached the streamer at all. Zero means the slice seam
//               is no longer wired, which is the mutation that used to pass.
//   steps == N  the step boundary ran once per forward. Zero means nothing
//               calls it, which is the defect that voided this row's published
//               decode number: `Acquire` protects every entry it serves and only
//               `EndStep` clears that protection, so a cache with no step clock
//               refuses every slice after it first fills, silently falls back to
//               the mapping, and never once exercises its own eviction policy.
//   exhausted == 0
//               nothing was refused. This is `steps == 0` seen from the other
//               side, and it is the number a benchmark can read.
//
// THIS BINARY IS DEDICATED TO STREAMING-ON. `VT_MOE_EXPERT_STREAM` is read once
// into a function-local static on first use, so a single process cannot run both
// arms by changing the environment. The comparison against the unstreamed arm is
// therefore made through the cache-exhaustion fallback, which is a REAL
// production state (a budget below one step's working set reaches it) and takes
// the identical `KqResidentSlice` path an OFF build takes.
#include <stdlib.h>

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::Qwen3_5MoeWeights;
using vllm::Qwen3_5Model;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

namespace {

// Turn the lane on BEFORE anything can read the environment. The read happens in
// a function-local static on the first slice, so setting it inside a test body
// would work today and break the moment a case ordering changed.
struct EnableExpertStreaming {
  EnableExpertStreaming() {
    ::setenv("VT_MOE_EXPERT_STREAM", "1", 1);
    // Comfortably more than one step's working set: 4 experts x 3 towers x 4
    // layers = 48 distinct slices per forward. A budget BELOW that would make
    // `exhausted` nonzero for an honest reason and mask the defect under test.
    ::setenv("VT_MOE_EXPERT_STREAM_SLOTS", "64", 1);
    ::setenv("VT_MOE_EXPERT_STREAM_SLOT_BYTES", "8192", 1);
    // Quiet under ctest, but overwrite=0 so an operator who sets this var
    // still gets the line -- which is the only way to SEE the statistics
    // this row added, and a gate that suppresses its own evidence is a
    // smaller version of the defect it was written for.
    ::setenv("VT_MOE_EXPERT_STREAM_STATS_EVERY", "0", 0);  // quiet under ctest
    // The grouped keep-quant path stages the whole tower and cannot stream; the
    // production code already disables it when streaming is requested. Being
    // explicit here keeps the test honest about which path it is measuring.
    ::setenv("VT_QWEN35_GROUPED_MOE", "0", 1);
  }
};
const EnableExpertStreaming kEnableExpertStreaming;

float RandV(uint64_t s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  s ^= s >> 33;
  return (static_cast<float>((s >> 40) & 0xFFFF) / 32768.0f) - 1.0f;
}

vllm::OwnedTensor MakeOwned(DType dt, const std::vector<int64_t>& shape, uint64_t seed) {
  vllm::OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    std::vector<uint8_t> b(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(b.data());
    for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
    t.bytes = vllm::OwnedBytes(std::move(b));
  } else {
    std::vector<uint8_t> b(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(b.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
    t.bytes = vllm::OwnedBytes(std::move(b));
  }
  return t;
}

// A keep-quant STACKED expert tower: [rows, cols] Q8_0, `nk = true`, exactly the
// shape the GGUF keep-quant loader produces and the only shape KqExpertSlice
// slices.
//
// The blocks are BUILT, not filled with noise. A Q8_0 block is an fp16 scale
// followed by 32 int8 weights, and random bytes put random bit patterns in the
// scale — including the fp16 encodings of inf and NaN, which propagate straight
// through the GEMM and make every later comparison vacuous. The values are
// arbitrary but well-formed, which is all this test needs: every arm decodes the
// SAME bytes, so equality between arms is a real comparison.
vllm::OwnedTensor MakeKqTower(int64_t rows, int64_t cols, uint64_t seed) {
  vllm::OwnedTensor t;
  t.dtype = DType::kQ8_0;
  t.nk = true;
  t.rank = 2;
  t.shape[0] = rows;
  t.shape[1] = cols;
  const size_t row_bytes = vt::RowSizeBytes(DType::kQ8_0, cols);
  const int64_t blocks_per_row = cols / 32;
  REQUIRE(cols % 32 == 0);          // Q8_0 is a 32-element block quant
  REQUIRE(row_bytes == static_cast<size_t>(blocks_per_row) * 34);
  std::vector<uint8_t> b(static_cast<size_t>(rows) * row_bytes);
  size_t o = 0;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint16_t d = vt::F32ToF16(0.004f + 0.001f * RandV(seed + static_cast<uint64_t>(r * 131 + blk)));
      std::memcpy(b.data() + o, &d, 2);
      o += 2;
      for (int j = 0; j < 32; ++j) {
        const int8_t q = static_cast<int8_t>(
            static_cast<int>(100.0f * RandV(seed + static_cast<uint64_t>((r * 131 + blk) * 32 + j))));
        std::memcpy(b.data() + o, &q, 1);
        o += 1;
      }
    }
  }
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = 40;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 32;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  return c;
}

vllm::MoeBlockWeights MakeKqMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(DType::kBF16, {H, 1}, s + 2);
  // The routed experts are STACKED keep-quant towers, and the per-expert vectors
  // stay empty — the A3 layout the streaming seam is defined against.
  m.expert_gate_kq = MakeKqTower(E * I, H, s + 100);
  m.expert_up_kq = MakeKqTower(E * I, H, s + 200);
  m.expert_down_kq = MakeKqTower(E * H, I, s + 300);
  m.shared_gate_proj = MakeOwned(DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(DType::kBF16, {Is, H}, s + 5);
  return m;
}

Qwen3_5MoeWeights MakeWeights(const HfConfig& c, uint64_t base_seed = 0) {
  Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = base_seed + 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5MoeLayerWeights lw;
    lw.is_linear_attention = (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.moe = MakeKqMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

struct CachePool {
  const HfConfig& c;
  int64_t num_blocks;
  int64_t block_size;
  std::vector<std::vector<float>> full_attn_buf;
  std::vector<std::vector<float>> gdn_ssm_buf;
  std::vector<std::vector<float>> gdn_conv_buf;
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;

  CachePool(const HfConfig& cfg, int64_t nb, int64_t bs)
      : c(cfg), num_blocks(nb), block_size(bs) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                  Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
    const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      if (c.layer_types[static_cast<size_t>(l)] == "linear_attention") {
        gdn_ssm_buf.emplace_back(static_cast<size_t>(nb * Hv * Dv * Dk), 0.0f);
        gdn_conv_buf.emplace_back(static_cast<size_t>(nb * conv_dim * (Kw - 1)), 0.0f);
      } else {
        full_attn_buf.emplace_back(static_cast<size_t>(nb * 2 * bs * Hkv * Dh), 0.0f);
      }
    }
    Rebind();
  }

  void Rebind() {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                  Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
    const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    attn_kv.clear();
    gdn_state.clear();
    for (auto& b : full_attn_buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
    for (size_t g = 0; g < gdn_ssm_buf.size(); ++g) {
      GdnStateCache gs;
      gs.ssm_state = vt::Tensor::Contiguous(gdn_ssm_buf[g].data(), DType::kF32,
                                            vt::Device{vt::DeviceType::kCPU, 0},
                                            {num_blocks, Hv, Dv, Dk});
      gs.conv_state = vt::Tensor::Contiguous(gdn_conv_buf[g].data(), DType::kF32,
                                             vt::Device{vt::DeviceType::kCPU, 0},
                                             {num_blocks, conv_dim, Kw - 1});
      gdn_state.push_back(gs);
    }
  }
};

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

CommonAttentionMetadata PrefillAttnMeta(int64_t T, const std::vector<int32_t>& blocks,
                                        int64_t block_size, int64_t start_slot) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = static_cast<int>(blocks.size());
  m.block_table_tensor = blocks;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t blk = blocks[static_cast<size_t>(t / block_size)];
    m.slot_mapping.push_back(blk * block_size + (start_slot + t) % block_size);
  }
  m.causal = true;
  return m;
}

GDNAttentionMetadata PrefillGdnMeta(int64_t T, int32_t sidx) {
  GDNAttentionMetadata g;
  g.num_prefills = 1;
  g.num_prefill_tokens = static_cast<int>(T);
  g.num_decodes = 0;
  g.num_decode_tokens = 0;
  g.num_actual_tokens = static_cast<int>(T);
  g.has_initial_state = std::vector<uint8_t>{0};
  g.non_spec_state_indices_tensor = std::vector<int32_t>{sidx};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_state_indices = std::vector<int32_t>{sidx};
  g.prefill_has_initial_state = std::vector<uint8_t>{0};
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}

// One full paged forward through the PRODUCTION entry point, over a fresh cache
// so every call is independent.
std::vector<float> OneForward(const HfConfig& c, const Qwen3_5MoeWeights& w,
                              const std::vector<int32_t>& ids) {
  vt::Queue q = Q();
  const int64_t T = static_cast<int64_t>(ids.size());
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/8);
  const std::vector<int32_t> blocks = {0};
  return Qwen3_5Model::Forward(ids, pos, PrefillAttnMeta(T, blocks, 8, 0),
                               PrefillGdnMeta(T, 0), pool.attn_kv, pool.gdn_state,
                               w, c, q, {});
}

}  // namespace

TEST_CASE("decode REACHES the expert streamer, and the step clock advances") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};

  // Nothing has run: the lane must not have built a store just by existing.
  {
    const vllm::detail::ExpertStreamStats s0 = vllm::detail::ExpertStreamSnapshot();
    CHECK_FALSE(s0.active);
    CHECK(s0.steps == 0);
  }

  const int kSteps = 3;
  for (int i = 0; i < kSteps; ++i) {
    const std::vector<float> logits = OneForward(c, w, ids);
    REQUIRE(logits.size() ==
            static_cast<size_t>(ids.size()) * static_cast<size_t>(c.vocab_size));
    for (float v : logits) REQUIRE(std::isfinite(v));
  }

  const vllm::detail::ExpertStreamStats s = vllm::detail::ExpertStreamSnapshot();

  // REACHABILITY of the slice seam. Zero fills means decode no longer enters
  // KqExpertSlice's streaming branch at all: the mutation that replaced the
  // production call site with `nullptr` used to leave the whole gate green.
  CHECK(s.active);
  CHECK(s.fills > 0);
  CHECK(s.bytes_filled > 0);
  CHECK((s.hits + s.misses) > 0);

  // REACHABILITY of the STEP BOUNDARY, which is the finding that voided this
  // row's published decode number. One step per forward, no more and no fewer:
  // once per layer would give 12 here and once per expert far more, and both are
  // wrong for the cache's protection semantics.
  CHECK(s.steps == kSteps);

  // The consequence of the two above, and the number a benchmark can read. A
  // cache whose step never ends protects every entry forever, refuses every
  // slice once full, and silently serves the mapping instead.
  CHECK(s.exhausted == 0);

#if defined(__unix__)
  // F5: the MADV_WILLNEED hint is ACCEPTED, not merely issued.
  //
  // madvise(2) returns EINVAL on an address that is not page-aligned, and GGUF
  // tensor data is aligned to `general.alignment`, default 32
  // (gguf_reader.cpp:401), so a slice address is essentially never a page
  // boundary. The call was made on the raw address with its return value
  // discarded, which is a hint that never fired and never said so. This counts
  // only the calls the kernel took.
  //
  // NO SPEEDUP IS ASSERTED, here or anywhere. This says the call is well formed.
  CHECK(s.advised > 0);
#endif
}

TEST_CASE("a streamed slice and the tower view produce IDENTICAL logits") {
  // "Streaming is byte-identical" in both directions, inside one process.
  //
  // The unstreamed arm is reached through the cache-exhaustion fallback rather
  // than through the environment, because `VT_MOE_EXPERT_STREAM` is read once
  // into a function-local static and a single process cannot see it change.
  // That fallback is not a test fiction: it is the branch a real budget below
  // one step's working set takes, and it runs the same KqResidentSlice the
  // streaming-OFF build runs.
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const std::vector<int32_t> ids = {7, 1, 22, 4};

  vllm::detail::ExpertStreamSetForceFallback(false);
  const std::vector<float> streamed = OneForward(c, w, ids);
  const vllm::detail::ExpertStreamStats on = vllm::detail::ExpertStreamSnapshot();
  REQUIRE(on.active);
  const int64_t fills_after_streamed = on.fills;
  REQUIRE(fills_after_streamed > 0);  // the streamed arm really streamed

  vllm::detail::ExpertStreamSetForceFallback(true);
  const std::vector<float> tower = OneForward(c, w, ids);
  const vllm::detail::ExpertStreamStats off = vllm::detail::ExpertStreamSnapshot();
  vllm::detail::ExpertStreamSetForceFallback(false);

  // The unstreamed arm really did NOT stream: no new bytes moved, and every
  // slice it asked for was refused into the fallback.
  CHECK(off.fills == fills_after_streamed);
  CHECK(off.exhausted > 0);

  // BIT-EXACT, not close. The slot holds a byte copy of the same tower bytes, so
  // the two arms feed the kernel identical inputs; anything but equality means
  // the copy is wrong, which a tolerance-based check would hide.
  REQUIRE(streamed.size() == tower.size());
  size_t differing = 0;
  for (size_t i = 0; i < streamed.size(); ++i)
    if (!(streamed[i] == tower[i])) ++differing;
  CHECK(differing == 0);
}

TEST_CASE("a SECOND model does not inherit the first model's slots") {
  // Found by the reachability case above, and not by anything before it.
  //
  // The slot store is a process-lifetime singleton, so it outlives any one
  // model. Its tower identity used to be the weight buffer's ADDRESS, whose
  // comment claimed it was "stable for the model's life because the tower is a
  // borrowed view into the mapping". The premise is true and the conclusion does
  // not follow: the CACHE is not scoped to one model's life. Free a model, load
  // another, and the allocator hands the new towers addresses the old ones held,
  // so the new model's expert resolves to an entry filled from a different
  // checkpoint. It comes back as a HIT, and a hit moves no bytes by contract, so
  // there is nothing at all to observe downstream -- just wrong weights.
  //
  // Measured, on exactly this shape: 24 towers occupied 21 distinct addresses,
  // and 20 of 222 slices returned another tower's bytes.
  const HfConfig c = MakeConfig();

  // A first model, run and then DESTROYED, so its buffers go back to the
  // allocator while the cache keeps its entries.
  {
    const Qwen3_5MoeWeights a = MakeWeights(c, /*base_seed=*/0);
    (void)OneForward(c, a, {5, 9, 2, 31});
  }

  // A second model with DIFFERENT weights. Identical weights would make an
  // address collision invisible, which is precisely why this needs its own seed.
  const Qwen3_5MoeWeights b = MakeWeights(c, /*base_seed=*/777000);
  const std::vector<int32_t> ids = {5, 9, 2, 31};

  const std::vector<float> streamed = OneForward(c, b, ids);

  // The ground truth for THIS model, taken through the fallback, which reads the
  // tower directly and so cannot be poisoned by a stale entry.
  vllm::detail::ExpertStreamSetForceFallback(true);
  const std::vector<float> truth = OneForward(c, b, ids);
  vllm::detail::ExpertStreamSetForceFallback(false);

  REQUIRE(streamed.size() == truth.size());
  size_t differing = 0;
  for (size_t i = 0; i < streamed.size(); ++i)
    if (!(streamed[i] == truth[i])) ++differing;
  CHECK(differing == 0);
}
