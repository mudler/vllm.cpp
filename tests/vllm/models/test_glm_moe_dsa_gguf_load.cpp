// GLM-5.3 (`GlmMoeDsaForCausalLM`) — W7's load gate, on a complete synthetic
// `glm-dsa` GGUF.
//
// Spec `.agents/specs/glm-dsa-latest-deepseek.md` §3.7 W7, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// ─── WHY A SYNTHETIC MODEL AND NOT THE ARTIFACT ──────────────────────────────
// The real gate is a 201.83 GiB load under an `rc` lease, and it is not
// something CI can run. What CI can run is the same loader over a model with
// the same STRUCTURE at 1/1000th the size: the same tensor names, the same
// heterogeneous indexer schedule with `full` and `shared` layers interleaved,
// the same leading dense block, the same stacked keep-quant towers, and the
// same multi-token-prediction tail to drop. Every refusal and every accounting
// rule this loader has is reachable here.
//
// This is the case the reachability rule asks for. It enters through
// `LoadedEngine::FromModelDir` — the production entry point a user arrives
// through — and not through `LoadGlmMoeDsaFromGguf`, so deleting the
// `kGgufArchArms` row, the `REGISTER_VLLM_MODEL` line, or the GGUF branch of
// `LoadGlmMoeDsaForCausalLM` reds it. A unit test that called the loader
// directly would prove the loader works and nothing about whether anything
// reaches it.
//
// ─── THE TOWERS ARE Q8_0, AND THAT IS NOT COSMETIC ───────────────────────────
// A routed-expert tower that routes to an EXPAND residency leaves the streaming
// lane, and `gguf_device_fit.cpp`'s admission rule is all-or-nothing across a
// model's `*_exps.weight` set — one expanded tower disqualifies all of them. On
// the real arm that is the difference between 6.375 GiB of blocks and 24.000
// GiB of bf16 for four tensors, and between streaming and not streaming for all
// 228. F32 has no keep-quant residency, so a fixture whose towers were F32
// would exercise the expand path and never the one production takes. The blocks
// below are BUILT rather than filled with noise: random bytes in a Q8_0 fp16
// scale produce inf and NaN, which propagate and make every later comparison
// vacuous.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/config/weight_residency.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/glm_moe_dsa.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vt/dtype.h"

#include "../gguf_builder.h"
#include "glm_moe_dsa_gguf_fixture.h"

namespace {

using namespace glm_dsa_fixture;  // NOLINT: the fixture IS this file's vocabulary

std::string RefusalOf(const std::function<void()>& fn, const char* what) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  FAIL_CHECK("expected a refusal from " << what << ", got none");
  return {};
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// The load, through the production entry point.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W7: a complete `glm-dsa` GGUF loads through LoadedEngine::FromModelDir") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  vllm::entrypoints::EngineParams params;
  // THE REACHABILITY ASSERTION. This is the entry point a user arrives through;
  // deleting the `kGgufArchArms` row, the `REGISTER_VLLM_MODEL` line or the GGUF
  // branch of `LoadGlmMoeDsaForCausalLM` all red this.
  std::unique_ptr<vllm::entrypoints::LoadedEngine> engine;
  REQUIRE_NOTHROW(engine = vllm::entrypoints::LoadedEngine::FromModelDir(
                      f.path(), params));
  REQUIRE(engine != nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// A ROUTED-EXPERT TOWER IS BORROWED AND IS NOT FAULTED IN AT LOAD.
//
// Spec §3.3 and `## Owed` O30 ([#2214](https://github.com/mudler/vllm.cpp/issues/2214)).
// This is the assertion that separates a model that STREAMS its experts from one
// that only says it does, and it was written because the real artifact said the
// second thing. On `thor:gpu0`, 2026-08-31, the `UD-IQ1_S` load's RSS grew
// LINEARLY past 48 GiB against an 18.99 GiB resident class, at the filesystem's
// read rate, with no plateau: `OwnGgufQuantBlocks` prefaulted every borrowed
// span, and 228 of those spans are the 187.312 GiB expert set the slot lane
// exists to page one slice at a time.
//
// WHY A ZERO IS A REAL ASSERTION HERE, AND NOT A MUTE SWITCH. Every non-tower
// tensor in this fixture is F32, and F32 has no keep-quant residency, so it is
// DEQUANTIZED into an owned buffer and never borrowed. The Q8_0 expert towers
// are therefore the ONLY spans in this whole model that reach the borrow arm at
// all — which is exactly the arm that prefaults. A nonzero count can only be a
// tower. The positive control below is what stops the zero from meaning "the
// borrow arm was never taken": it asserts the towers ARE mmap-backed borrows,
// which is the precondition for the prefault this case forbids.
//
// The regression control for the DEFAULT is `test_gguf_keep_quant`'s own L7
// prefault A/B, which is untouched: `prefault` defaults to true, so every weight
// a forward reads in place is still faulted off the timed prefill.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W10: the streamed towers are borrowed and NOT prefaulted") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::GlmMoeDsaHfConfigFromGguf(g);

  vllm::ResetGgufPrefaultedSpanCountForTesting();
  const vllm::GlmMoeDsaWeights w =
      vllm::LoadGlmMoeDsaFromGguf(g, c, /*policy=*/nullptr);

  // ── the positive control: the towers took the BORROW arm ──
  // `mmap_fd` is set only on the borrow path of `OwnGgufQuantBlocks`, so this
  // says the load reached the code the assertion below constrains. Without it a
  // zero could mean `VT_GGUF_MMAP=0` or a copy arm, and the case would pass on a
  // run that measured nothing.
  int64_t borrowed_towers = 0;
  for (int64_t l = 0; l < kBackbone; ++l) {
    const vllm::GlmMoeDsaLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    if (!lw.is_moe) continue;
    CAPTURE(l);
    CHECK(lw.moe.gate_exps.mmap_fd >= 0);
    CHECK(lw.moe.up_exps.mmap_fd >= 0);
    CHECK(lw.moe.down_exps.mmap_fd >= 0);
    borrowed_towers += 3;
  }
  CHECK(borrowed_towers == (kBackbone - kLeadingDense) * 3);

  // ── the assertion ──
  const uint64_t prefaulted = vllm::GgufPrefaultedSpanCount();
  CAPTURE(prefaulted);
  CAPTURE(borrowed_towers);
  CHECK(prefaulted == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// What the loader produced. Driven through the same file, read directly so the
// weights can be inspected.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W7: the loaded weights carry the schedule, the towers and the accounting") {
  gguf_test::TempFile f(BuildCompleteGlmDsa());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::GlmMoeDsaHfConfigFromGguf(g);
  const vllm::GlmMoeDsaWeights w =
      vllm::LoadGlmMoeDsaFromGguf(g, c, /*policy=*/nullptr);

  // The backbone, not the block count.
  REQUIRE(static_cast<int64_t>(w.layers.size()) == kBackbone);
  CHECK(w.params.num_hidden_layers == kBackbone);
  CHECK(w.params.num_nextn_predict_layers == 1);

  // ── the accounting closes, and the MTP tail is counted rather than ignored ──
  CHECK(w.file_tensors == static_cast<int64_t>(g.Tensors().size()));
  CHECK(w.accounted_tensors + w.mtp_block_tensors_dropped == w.file_tensors);
  CHECK(w.mtp_block_tensors_dropped > 0);
  // Spec D3: the broadcast indexer surplus is the `shared` layers x 5.
  CHECK(w.broadcast_indexer_tensors_dropped == kSharedLayers * 5);

  // ── the heterogeneous schedule reached the WEIGHTS, not only the config ──
  // A `full` layer carries its indexer; a `shared` layer carries none, because
  // it runs no indexer at all and attends through the preceding full layer's
  // selection. This is the assertion that would fail if the loader believed the
  // file (which ships indexer weights on every block) instead of the schedule.
  int64_t with_indexer = 0;
  for (int64_t l = 0; l < kBackbone; ++l) {
    const bool full = w.params.indexer_types[static_cast<size_t>(l)] ==
                      vllm::GlmMoeDsaIndexerKind::kFull;
    const vllm::GlmMoeDsaIndexerWeights& ix = w.layers[static_cast<size_t>(l)].attn.indexer;
    CAPTURE(l);
    CHECK(ix.Empty() == !full);
    if (full) {
      ++with_indexer;
      // The bias is what makes this a LayerNorm rather than an RMSNorm.
      CHECK(!ix.k_norm_bias.Empty());
      CHECK(ix.wq_b.shape[0] == kIdxHeads * kIdxHead);
      CHECK(ix.wq_b.shape[1] == kQLora);
      CHECK(ix.weights_proj.shape[0] == kIdxHeads);
      // `weights_proj` is held at the MODEL dtype since W9 (#2214), and the
      // narrowing is upstream rather than a convenience: `wk_weights_proj`
      // carries no `params_dtype` (deepseek_v2.py:700-707), the shared MLA
      // block computes the indexer in the block dtype, and `vt::MatmulBT` needs
      // its two float operands to agree. W2's point that this tensor decides
      // the selection outright is why the loader PROVES the file's F32 round-
      // trips through bf16 exactly and refuses by name when it does not,
      // instead of rounding on trust.
      CHECK(ix.weights_proj.dtype == vt::DType::kBF16);
    }
  }
  CHECK(with_indexer == kFullLayers);

  // ── the dense/MoE layout ──
  for (int64_t l = 0; l < kBackbone; ++l) {
    CAPTURE(l);
    const vllm::GlmMoeDsaLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    CHECK(lw.is_moe == (l >= kLeadingDense));
    CHECK(lw.dense.Empty() == lw.is_moe);
    CHECK(lw.moe.Empty() == !lw.is_moe);
    if (!lw.is_moe) {
      // The leading dense block uses `intermediate_size`, not
      // `moe_intermediate_size`.
      CHECK(lw.dense.gate_proj.shape[0] == kInter);
    } else {
      // ── THE TOWERS STAYED IN BLOCKS ──
      // This is the assertion the whole residency plan rests on. A tower that
      // dequantized to bf16 here would be 4x the bytes and, worse, would take
      // the whole model out of the streaming lane, because
      // `GgufExpertTowersReachSlotLane` is all-or-nothing.
      CHECK(lw.moe.gate_exps.dtype == vt::DType::kQ8_0);
      CHECK(lw.moe.up_exps.dtype == vt::DType::kQ8_0);
      CHECK(lw.moe.down_exps.dtype == vt::DType::kQ8_0);
      // Held flat as [E*out, K], which is the shape the streaming lane's only
      // gated client uses.
      CHECK(lw.moe.gate_exps.shape[0] == kExperts * kMoeInter);
      CHECK(lw.moe.gate_exps.shape[1] == kHidden);
      CHECK(lw.moe.down_exps.shape[0] == kExperts * kHidden);
      CHECK(lw.moe.down_exps.shape[1] == kMoeInter);
      // The router and its noaux_tc bias are f32: the bias is ADDED to the
      // sigmoid scores before a top-k, and a top-k is a discrete outcome that no
      // tolerance bounds.
      CHECK(lw.moe.router.dtype == vt::DType::kF32);
      CHECK(lw.moe.e_score_correction_bias.dtype == vt::DType::kF32);
      // The shared expert is `moe_intermediate_size * n_shared_experts`, NOT
      // `intermediate_size`.
      CHECK(lw.moe.shared.gate_proj.shape[0] == kMoeInter);
    }
  }

  // ── W9: the post-load absorption ran, and it produced the SEAM's shapes ──
  // The file ships `attn_k_b` as [heads, kv_lora, qk_nope] and `attn_v_b` as
  // [heads, v_head, kv_lora] — llama.cpp's orientation, where a row runs along
  // the contraction axis. `vt::BatchedMatmul` needs the transposes, in bf16,
  // because there is no quantized bmm (mla_attention.py:876-878). A shape check
  // is the only thing that separates the two, and getting it wrong yields
  // finite, plausible, wrong attention.
  CHECK(w.absorbed);
  for (int64_t l = 0; l < kBackbone; ++l) {
    CAPTURE(l);
    const vllm::GlmMoeDsaMlaWeights& a = w.layers[static_cast<size_t>(l)].attn;
    REQUIRE(!a.w_uk_t.Empty());
    REQUIRE(!a.w_uv.Empty());
    REQUIRE(!a.kv_b_proj.Empty());
    CHECK(a.w_uk_t.dtype == vt::DType::kBF16);
    CHECK(a.w_uv.dtype == vt::DType::kBF16);
    CHECK(a.kv_b_proj.dtype == vt::DType::kBF16);
    CHECK(a.w_uk_t.rank == 3);
    CHECK(a.w_uk_t.shape[0] == kHeads);
    CHECK(a.w_uk_t.shape[1] == kQkNope);
    CHECK(a.w_uk_t.shape[2] == kKvLora);
    CHECK(a.w_uv.rank == 3);
    CHECK(a.w_uv.shape[0] == kHeads);
    CHECK(a.w_uv.shape[1] == kKvLora);
    CHECK(a.w_uv.shape[2] == kVHead);
    CHECK(a.kv_b_proj.rank == 2);
    CHECK(a.kv_b_proj.shape[0] == kHeads * (kQkNope + kVHead));
    CHECK(a.kv_b_proj.shape[1] == kKvLora);
    // The file's own halves are RETAINED, borrowed and unexpanded: they are
    // what the census counts and what the absorption reads.
    CHECK(a.k_b_proj.shape[0] == kHeads);
    CHECK(a.v_b_proj.shape[0] == kHeads);
  }

  // ── THE ABSORPTION'S ORIENTATION: A DRIFT LOCK, AND NOT A CORRECTNESS GATE ──
  //
  // SAY WHAT THIS IS, because the difference decides how much it is worth. The
  // rule below is a TRANSCRIPTION of the rule `AbsorbMla` implements, so it
  // cannot prove that rule right — it can only fail when someone changes the
  // loader without changing this statement of it. That is worth having and it
  // is not a gate.
  //
  // WHY THERE IS NO GATE. W9 ran the mutation: dropping the per-head transpose
  // entirely — reading `attn_k_b` verbatim into the nope rows — leaves every
  // shape identical, every byte count identical, all 7 forward cases and all 4
  // load cases GREEN, and moves only the logit values, which nothing on this row
  // compares to anything (spec O1: there is no oracle). The orientation is
  // established by READING two conventions against each other: llama.cpp's
  // `ggml_mul_mat` contracts over `ne[0]`, so `attn_k_b`'s rows run along
  // `qk_nope_head_dim`; `vt::BatchedMatmul` is `torch.bmm`, so `w_uk_t`'s rows
  // run along `kv_lora_rank`. Discharged by G4 — llama.cpp `b10451` on the
  // identical artifact — which needs the complete checkpoint (spec O28).
  //
  // The two halves are NOT treated alike, and that asymmetry is the thing most
  // worth locking: `attn_k_b` is TRANSPOSED and `attn_v_b` is VERBATIM, because
  // the file states them with different contraction axes.
  {
    const vllm::GlmMoeDsaMlaWeights& a = w.layers[0].attn;
    const vllm::GgufTensorInfo& tkb = g.Get("blk.0.attn_k_b.weight");
    const vllm::GgufTensorInfo& tvb = g.Get("blk.0.attn_v_b.weight");
    REQUIRE(tkb.ggml_type == 0);  // F32 in this fixture; read directly
    REQUIRE(tvb.ggml_type == 0);
    const auto* kb = reinterpret_cast<const float*>(tkb.data);
    const auto* vb = reinterpret_cast<const float*>(tvb.data);
    const auto* got = reinterpret_cast<const uint16_t*>(a.kv_b_proj.bytes.data());
    const int64_t row = kQkNope + kVHead;
    int64_t nope_mismatch = 0, v_mismatch = 0;
    for (int64_t h = 0; h < kHeads; ++h) {
      for (int64_t i = 0; i < kQkNope; ++i) {
        for (int64_t j = 0; j < kKvLora; ++j) {
          // TRANSPOSED: file `[h][j][i]` -> checkpoint layout `[h][i][j]`.
          const uint16_t want = vt::F32ToBF16(kb[(h * kKvLora + j) * kQkNope + i]);
          if (got[(h * row + i) * kKvLora + j] != want) ++nope_mismatch;
        }
      }
      for (int64_t i = 0; i < kVHead; ++i) {
        for (int64_t j = 0; j < kKvLora; ++j) {
          // VERBATIM: `attn_v_b` is already `[h][i][j]`.
          const uint16_t want = vt::F32ToBF16(vb[(h * kVHead + i) * kKvLora + j]);
          if (got[(h * row + kQkNope + i) * kKvLora + j] != want) ++v_mismatch;
        }
      }
    }
    CHECK(nope_mismatch == 0);
    CHECK(v_mismatch == 0);
  }

  // ── model level ──
  CHECK(w.embed_tokens.shape[0] == kVocab);
  CHECK(w.embed_tokens.shape[1] == kHidden);
  // The tie is read off the FILE. This fixture ships `output.weight`, so the
  // model is untied and `lm_head` is populated.
  CHECK(!w.lm_head.Empty());
  CHECK(!w.rope_cos_sin_cache.Empty());
  CHECK(w.rope_cos_sin_cache.shape[1] == kQkRope);
}

// ─────────────────────────────────────────────────────────────────────────────
// The two refusals, each proven by making the file violate it.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("glm-dsa W7: a tensor this port does not claim refuses the load by name") {
  // A tensor that is neither claimed nor in the MTP tail. Without the
  // accounting this loads clean and the extra weight is silently absent from
  // the model, which no token gate on this row could ever see (spec O1).
  gguf_test::TempFile f(BuildCompleteGlmDsa("blk.2.attn_sinks.weight"));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::GlmMoeDsaHfConfigFromGguf(g);
  const std::string msg = RefusalOf(
      [&] { (void)vllm::LoadGlmMoeDsaFromGguf(g, c, nullptr); },
      "LoadGlmMoeDsaFromGguf");
  CHECK(msg.find("blk.2.attn_sinks.weight") != std::string::npos);
  CHECK(msg.find("silently absent") != std::string::npos);
}

TEST_CASE("glm-dsa W7: an expert tower that would EXPAND refuses the load by name") {
  gguf_test::TempFile f(BuildCompleteGlmDsa("", /*expand_one_tower=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::GlmMoeDsaHfConfigFromGguf(g);
  const std::string msg = RefusalOf(
      [&] { (void)vllm::LoadGlmMoeDsaFromGguf(g, c, nullptr); },
      "LoadGlmMoeDsaFromGguf");
  CHECK(msg.find("ffn_gate_exps.weight") != std::string::npos);
  // The message must say WHY it matters, because the consequence is not local:
  // one expanded tower takes all of them out of the streaming lane.
  CHECK(msg.find("streaming lane") != std::string::npos);
  CHECK(msg.find("all-or-nothing") != std::string::npos);
}
