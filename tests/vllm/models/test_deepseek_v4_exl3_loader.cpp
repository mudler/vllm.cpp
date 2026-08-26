// MODEL-DSV4-EXL3 W1b — the rank-sliced EXL3 loader arm.
//
// The SparkInfer artifact `0xSero/deepseek-v4-flash-0731-spark` stores its 216
// routed experts as EXL3 trellis tensors SLICED ACROSS FOUR TENSOR-PARALLEL
// RANKS and everything else ("carried") as the un-requantized DeepSeek-V4 FP8
// source tensors. Its `config.json` declares
// `quantization_config.quant_method == "exl3"` with
// `version == "rank-sliced-deepseek-v4-v1"` and
// `hybrid_tr3_tail.tensor_schema ==
//  "layers.{L}.ffn.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"`.
//
// The slicing is upstream's own `LinearEXL3.tp_import_split`
// (exllamav3 @ 2398c056, `modules/quant/exl3.py:296-313`): a split on the OUT
// features takes `svh[first:last]` and `trellis[:, first//16:last//16]`, and a
// split on the IN features takes `suh[first:last]` and
// `trellis[first//16:last//16, :]`. w1 and w3 are out-split, w2 is in-split, so
// coalescing back to TP1 is pure concatenation and is LOSSLESS: 16x16 trellis
// tiles are independent and every rank boundary here is a multiple of 128, the
// Hadamard block size (`exl3_lib/quantize.py:15`).
//
// This suite drives the PRODUCTION loader entry
// (`vllm::LoadDeepseekV4ForCausalLMWeights`, what
// `deepseek_v4_registry.cpp:89` calls) over a hermetic four-rank fixture the
// test writes itself, and it checks the coalesced owners BYTE FOR BYTE against
// the rank inputs it wrote. Real-checkpoint residency (this arm copies the
// tower into host owner buffers, which is ~100 GB on the real artifact) is the
// named MODEL-DSV4-EXL3 W2 residual; see the spec's `## Owed`.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/mxfp4_dequant.h"   // E8M0ToF32
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"   // F8E4M3ToF32
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"  // host_available_memory_bytes
#include "vt/dtype.h"

// The fixture is SHARED with tests/vllm/models/test_deepseek_v4_exl3_forward.cpp,
// which drives the same production loader entry and then runs a forward over
// what it produced. One fixture is what makes the reachability claim falsifiable
// (#1923): the forward suite used to build `DeepseekV4Weights` by hand.
#include "dsv4_exl3_fixture.h"

using namespace dsv4_exl3_fixture;  // NOLINT(build/namespaces) — test fixture

// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("dsv4 exl3: the rank-sliced arm is detected and coalesces TP4 -> TP1") {
  auto f = BuildFixture();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);

  REQUIRE(w.has_exl3_weights);
  CHECK(w.exl3.tp == kTp);
  CHECK(w.exl3.bits == kBits);
  CHECK(w.exl3.codebook == "mcg");
  CHECK(w.exl3.version == "rank-sliced-deepseek-v4-v1");
  REQUIRE(w.exl3.layers.size() == static_cast<size_t>(kLayers));
  REQUIRE(w.exl3.layers[0].experts.size() == static_cast<size_t>(kExperts));

  // 35 carried + 1 layer * 2 experts * 3 projections * 4 ranks * 4 tensors.
  CHECK(w.accounted_tensors == 35 + kLayers * kExperts * 3 * kTp * 4);
  // The MTP tail is skipped exactly as vLLM's own DeepSeek-V4 loader skips it
  // (`AutoWeightsLoader(skip_substrs=["mtp."])`, nvidia/model.py:1474) — but
  // COUNTED, so the skip is visible rather than silent. On the real artifact
  // these are 3985 NVFP4 draft-head tensors.
  CHECK(w.exl3.skipped_mtp_tensors == 2);

  const vllm::DeepseekV4Exl3Expert& e0 = w.exl3.layers[0].experts[0];
  for (const vllm::DeepseekV4Exl3Linear* lin : {&e0.w1, &e0.w3}) {
    CHECK(lin->in_features == kHidden);
    CHECK(lin->out_features == kInter);
    CHECK(lin->bits == kBits);
    CHECK(lin->suh.size() == static_cast<size_t>(kHidden));
    CHECK(lin->svh.size() == static_cast<size_t>(kInter));
    CHECK(lin->trellis.size() == static_cast<size_t>(TrellisElems(kHidden, kInter)));
  }
  CHECK(e0.w2.in_features == kInter);
  CHECK(e0.w2.out_features == kHidden);
  CHECK(e0.w2.suh.size() == static_cast<size_t>(kInter));
  CHECK(e0.w2.svh.size() == static_cast<size_t>(kHidden));
  CHECK(e0.w2.trellis.size() == static_cast<size_t>(TrellisElems(kInter, kHidden)));
  CHECK(e0.w1.mcg == -877912083);
}

TEST_CASE("dsv4 exl3: the coalesced owners equal the rank inputs BYTE FOR BYTE") {
  auto f = BuildFixture();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);

  int trellis_mismatch = 0, sign_mismatch = 0;
  for (int x = 0; x < kExperts; ++x) {
    const vllm::DeepseekV4Exl3Expert& ex = w.exl3.layers[0].experts[x];
    int proj_index = 0;
    for (const char* proj : {"w1", "w2", "w3"}) {
      const vllm::DeepseekV4Exl3Linear& lin =
          std::strcmp(proj, "w1") == 0 ? ex.w1
                                       : (std::strcmp(proj, "w2") == 0 ? ex.w2 : ex.w3);
      const bool out_split = SplitsOut(proj);
      const int64_t k = lin.in_features, n = lin.out_features;
      const int64_t tn = n / 16;
      const int64_t words = 16 * kBits;
      for (int r = 0; r < kTp; ++r) {
        // The per-rank slice this rank file was written with.
        const int64_t rk = out_split ? k : k / kTp;
        const int64_t rn = out_split ? n / kTp : n;
        const int64_t rtk = rk / 16, rtn = rn / 16;
        for (int64_t i = 0; i < rtk; ++i) {
          for (int64_t j = 0; j < rtn; ++j) {
            for (int64_t t = 0; t < words; ++t) {
              const uint16_t want =
                  TrellisWord(x, proj_index, r, (i * rtn + j) * words + t);
              // OUT split concatenates along trellis dim 1, IN split along dim 0.
              const int64_t gi = out_split ? i : (r * rtk + i);
              const int64_t gj = out_split ? (r * rtn + j) : j;
              if (lin.trellis[static_cast<size_t>((gi * tn + gj) * words + t)] != want)
                ++trellis_mismatch;
            }
          }
        }
        // The SPLIT sign vector is concatenated; the INVARIANT one came whole.
        if (out_split) {
          for (int64_t j = 0; j < rn; ++j)
            if (lin.svh[static_cast<size_t>(r * rn + j)] !=
                SignWord(x, proj_index, r, j, 1))
              ++sign_mismatch;
        } else {
          for (int64_t i = 0; i < rk; ++i)
            if (lin.suh[static_cast<size_t>(r * rk + i)] !=
                SignWord(x, proj_index, r, i, 0))
              ++sign_mismatch;
        }
      }
      for (int64_t i = 0; i < (out_split ? k : 0); ++i)
        if (lin.suh[static_cast<size_t>(i)] != SignWord(x, proj_index, 0, i, 0))
          ++sign_mismatch;
      for (int64_t j = 0; j < (out_split ? 0 : n); ++j)
        if (lin.svh[static_cast<size_t>(j)] != SignWord(x, proj_index, 0, j, 1))
          ++sign_mismatch;
      ++proj_index;
    }
  }
  CHECK(trellis_mismatch == 0);
  CHECK(sign_mismatch == 0);

  // The split inputs are NOT retained: the tower holds exactly the coalesced
  // bytes, never the four rank copies as well.
  CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == ExpectedTowerBytes());
}

TEST_CASE("dsv4 exl3: the LOAD reports the tower's residency and refuses one that does not fit") {
  // WHY THIS CASE EXISTS. `DeepseekV4Exl3ResidentBytes` had no production caller
  // at all: only the case above called it, which measures a function rather than
  // a capability ("Nothing lands dead", AGENTS.md). It is also exactly the
  // instrument the row's residency RISK needs — the real artifact's trellis alone
  // is ~83.5 GiB (43 layers x 216 experts x 3 projections) on a box whose unified
  // memory OOM-reboots. So the load itself now reports the figure and refuses a
  // projection the host cannot hold, and this case gates both halves.
  auto f = BuildFixture();
  vllm::DeepseekV4Weights w;
  const std::string log = CaptureStderr([&] {
    w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  });
  REQUIRE(w.has_exl3_weights);

  // REACHABILITY: this line is emitted BY THE LOAD, so deleting the production
  // call site takes this case red.
  CAPTURE(log);
  CHECK(log.find("[vt load] dsv4-exl3:") != std::string::npos);
  CHECK(log.find("resident_bytes=" + std::to_string(ExpectedTowerBytes())) !=
        std::string::npos);

  // ...AND IT MUST CARRY THE REAL BUDGET. The two assertions above match BOTH
  // branches of the report, which is how the fresh review (2026-08-24, MINOR-1)
  // wired `host_available` to a literal 0 at the production call site — symbol
  // still referenced, so it compiled (ninja rc=0, 3 steps) — and watched this
  // suite stay 6/6 66/66 SUCCESS while the refusal went silently inert and the
  // load printed `/proc/meminfo unreadable` on a host where it reads fine. The
  // budget the load ACTUALLY used is therefore asserted, branched on whether
  // THIS host can read the pool at all.
  const int64_t budget_now = vllm::v1::host_available_memory_bytes();
  CAPTURE(budget_now);
  if (budget_now > 0) {
    CHECK(log.find("host MemAvailable=") != std::string::npos);
    CHECK(log.find("MemAvailable unknown") == std::string::npos);
    // Same POOL, not merely some non-zero constant. The window is deliberately
    // wide (1/8x .. 8x) because MemAvailable moves under other work on the box
    // between the load and this second read; it is still far tighter than the
    // 1 MiB and 1 TiB brackets the refusal cases below inject, so a call site
    // rewired to either of those constants fails here.
    const double reported = ReportedMemAvailableGiB(log);
    const double now_gib = static_cast<double>(budget_now) / (1024.0 * 1024.0 * 1024.0);
    CAPTURE(reported);
    CAPTURE(now_gib);
    CHECK(reported > 0.0);
    CHECK(reported >= now_gib / 8.0);
    CHECK(reported <= now_gib * 8.0);
  } else {
    // /proc/meminfo is genuinely unreadable here, so the unknown branch is the
    // CORRECT report and the refusal is correctly inert.
    CHECK(log.find("MemAvailable unknown") != std::string::npos);
  }

  // The REFUSAL, driven through the same production function the load calls,
  // with the budget INJECTED. `check_enough_state_memory`
  // (`vllm/v1/core/kv_cache_utils.h`) is parameterised for exactly this reason:
  // a refusal observable only on a box of a chosen size is not gateable.
  const std::string refusal = ThrowMessage([&] {
    (void)vllm::ReportDeepseekV4Exl3Residency(w, /*layers_done=*/1,
                                              /*layers_total=*/43,
                                              /*host_available_bytes=*/1 << 20);
  });
  CAPTURE(refusal);
  CHECK(Mentions(refusal, "MODEL-DSV4-EXL3"));
  CHECK(Mentions(refusal, "MemAvailable"));
  // An UNKNOWN budget never refuses — `host_available_memory_bytes()` returns 0
  // when /proc/meminfo is unreadable, and `VT_DSV4_EXL3_HOST_BUDGET=0` hands the
  // reporter the same 0 on purpose. An unknown budget must not become a false
  // refusal (`host_available_memory_bytes`, `kv_cache_utils.cpp`, keeps the same
  // polarity).
  CHECK(ThrowMessage([&] {
          (void)vllm::ReportDeepseekV4Exl3Residency(w, 1, 43, 0);
        }).empty());
  // A budget that holds the projection does not refuse either.
  CHECK(ThrowMessage([&] {
          (void)vllm::ReportDeepseekV4Exl3Residency(w, 1, 43, int64_t{1} << 40);
        }).empty());
  // THE INCLUSIVE EDGE (fresh review, NIT-1). The two cases above bracket the
  // threshold at 1 MiB and 1 TiB against a ~304 KiB tower, which catches a
  // direction flip but NOT `projected <= budget` narrowing to `projected <
  // budget`. A projection that EQUALS the budget must load: with layers_done=1
  // the projection is exactly `tower * layers_total` PLUS the materialized
  // carried tower, which W1c made non-zero and which the refusal now prices
  // (a refusal that measured only one of the two towers would let the other
  // take the box down).
  const int64_t host_bytes = vllm::DeepseekV4HostResidentBytes(w);
  CHECK(host_bytes > 0);
  CHECK(ThrowMessage([&] {
          (void)vllm::ReportDeepseekV4Exl3Residency(
              w, 1, 43, ExpectedTowerBytes() * 43 + host_bytes);
        }).empty());
  // ...and ONE byte under it refuses. This is what makes the host term
  // load-bearing rather than decorative: dropping it from the sum leaves the
  // edge above green and takes this one red.
  CHECK(!ThrowMessage([&] {
           (void)vllm::ReportDeepseekV4Exl3Residency(
               w, 1, 43, ExpectedTowerBytes() * 43 + host_bytes - 1);
         }).empty());
  // The report names the carried tower too, so a reader sees both numbers.
  CHECK(log.find("host_bytes=" + std::to_string(host_bytes)) != std::string::npos);
}

TEST_CASE(
    "dsv4 exl3: VT_DSV4_EXL3_HOST_BUDGET defaults ON; only a '0'-leading value "
    "disables the refusal") {
  // The refusal's budget is a HEURISTIC (`/proc/meminfo` MemAvailable ignores
  // swap, and in a container reports the HOST's pool rather than the cgroup's
  // — see the caveats at the read site in `LoadDeepseekV4Exl3`), so it ships
  // with a same-binary escape hatch. This pins the PARSE, which is factored into
  // the header precisely so it is gateable without mutating the environment
  // (house shape: `AsyncRunnerFlagIsOn`, tests/vllm/v1/worker/
  // test_async_runner_flag.cpp).
  using vllm::Dsv4Exl3HostBudgetFlagIsOn;
  // Default (unset) is ON: the refusal ships enabled, because on the
  // unified-memory box this arm targets an over-commit reboots the machine.
  CHECK(Dsv4Exl3HostBudgetFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON, including the explicit opt-in and junk.
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("1"));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn(""));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("on"));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("true"));
  CHECK(Dsv4Exl3HostBudgetFlagIsOn(" 0"));  // leading space, not a '0' first char
  CHECK(Dsv4Exl3HostBudgetFlagIsOn("10"));
  // Disabled: FIRST character '0'.
  CHECK_FALSE(Dsv4Exl3HostBudgetFlagIsOn("0"));
  CHECK_FALSE(Dsv4Exl3HostBudgetFlagIsOn("00"));
  CHECK_FALSE(Dsv4Exl3HostBudgetFlagIsOn("0abc"));
}

TEST_CASE("dsv4 exl3: a NON-exl3 quantization_config takes the DENSE arm") {
  // The detection predicate was gated only in the POSITIVE direction: every case
  // above hands it an EXL3 checkpoint, so widening `quant_method == "exl3"` to an
  // always-true test left four dsv4 suites green (fresh review, 2026-08-24).
  // That matters because the plain vehicle is not "no quantization_config": the
  // `deepseek_v4_fp8` artifact carries one, and this very checkpoint stores it
  // verbatim as `quantization_config.base_quantization_config`. A regression that
  // widened the predicate would route FP8 into the trellis arm undetected.
  SUBCASE("quant_method fp8 — the deepseek_v4_fp8 vehicle") {
    FixtureOptions opt;
    opt.quant_method = "fp8";
    opt.dense_routed_experts = true;
    auto f = BuildFixture(opt);
    vllm::DeepseekV4Weights w;
    // Through ThrowMessage rather than bare: a widened predicate sends this
    // fixture into the EXL3 arm, which THROWS on the absent `version`, and an
    // uncaught throw is a failed CASE with `assertions: N | N passed` — a red
    // that reads as a pass in the summary line. Captured, it is a named
    // assertion that prints the refusal it got instead.
    const std::string msg = ThrowMessage(
        [&] { w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    REQUIRE(msg.empty());
    CHECK_FALSE(w.has_exl3_weights);
    CHECK(w.exl3.layers.empty());
    CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == 0);
    // 35 carried + 1 layer * 2 experts * 3 projections * 4 NVFP4 suffixes. The
    // dense arm walks its own name-map, which the EXL3 arm does not.
    CHECK(w.accounted_tensors == 35 + kLayers * kExperts * 3 * 4);
  }
  SUBCASE("no quantization_config at all") {
    // This one does NOT discriminate the `== "exl3"` widening, because the null
    // guard above it already returns false — it guards the OTHER widening, a
    // predicate that drops that guard.
    //
    // HOW IT DISCRIMINATES, AND WHY THAT IS NOT AN ASSERTION (fresh review,
    // NIT-2). Dropping `qc == nullptr` from `IsExl3Checkpoint` is not a
    // behavioural widening this subcase can observe by value: it is a null
    // dereference inside the predicate itself, so the process dies with SIGSEGV
    // before any CHECK below runs. MEASURED, not assumed: the binary exits 139
    // and doctest prints `5 | 4 passed | 1 failed | 2 skipped` with
    // `assertions: 63 | 63 passed | 0 failed` and `Status: FAILURE!` — a real
    // ctest red under a CLEAN assertion counter, with two later cases never run.
    // DO NOT grep that counter for this case's verdict; read the exit status.
    //
    // It was left this way on purpose. Making it fail by assertion instead needs
    // the deref replaced with a checked accessor, which means deleting the very
    // guard under test; and the sibling `!qc->is_object()` half cannot be
    // discriminated at all, because `nlohmann::json::contains` is already safe
    // on a non-object, so a fixture with a string-valued `quantization_config`
    // would take the dense arm either way and would assert nothing. A
    // memory-safety defect's discriminator is a crash or a sanitizer, not a
    // CHECK.
    FixtureOptions opt;
    opt.omit_quant_config = true;
    opt.dense_routed_experts = true;
    auto f = BuildFixture(opt);
    vllm::DeepseekV4Weights w;
    const std::string msg = ThrowMessage(
        [&] { w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    REQUIRE(msg.empty());
    CHECK_FALSE(w.has_exl3_weights);
    CHECK(vllm::DeepseekV4Exl3ResidentBytes(w) == 0);
    CHECK(w.accounted_tensors == 35 + kLayers * kExperts * 3 * 4);
  }
}

TEST_CASE("dsv4 exl3: the arm is REACHED from the registry's production load") {
  auto f = BuildFixture();
  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(f->shards);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = vllm::ModelRegistry::Load(f->config, source));
  CHECK(model != nullptr);
}

// ───────────────────────────────────────────────────────────────────────────
// MODEL-DSV4-EXL3 W1c — the CARRIED tower is MATERIALIZED, not merely counted.
// ───────────────────────────────────────────────────────────────────────────

TEST_CASE("dsv4 exl3 W1c: the load materializes the carried tower and sets the flag") {
  // W1b accounted for these tensors with a presence check and wrote NOTHING, so
  // `has_host_weights` stayed false and every forward entry point refused a
  // checkpoint that had just loaded (#1923: zero tokens from `vllm-server`).
  // Both halves are asserted here — the flag, and the tower the flag claims.
  FixtureOptions opt;
  opt.layers = 2;
  opt.num_hash_layers = 1;
  opt.topk = 2;
  opt.compress_ratios = {0, 4};
  opt.index_n_heads = 2;
  opt.index_head_dim = 4;
  opt.index_topk = 3;
  auto f = BuildFixture(opt);
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);

  REQUIRE(w.has_exl3_weights);
  REQUIRE(w.has_host_weights);
  REQUIRE(w.host.layers.size() == 2);
  const int64_t H = kHidden;
  const int64_t hc = kHcMult;
  const int64_t hc3 = (2 + hc) * hc;
  const int64_t in_per_group = kHeads * kHeadDim / kOGroups;

  CHECK(w.host.embed.size() == static_cast<size_t>(kVocab * H));
  CHECK(w.host.lm_head.size() == static_cast<size_t>(kVocab * H));
  CHECK(w.host.final_norm_weight.size() == static_cast<size_t>(H));
  CHECK(w.host.hc_head_fn.size() == static_cast<size_t>(hc * hc * H));
  CHECK(w.host.hc_head_base.size() == static_cast<size_t>(hc));

  const vllm::DeepseekV4LayerHostWeights& L0 = w.host.layers[0];
  CHECK(L0.wq_a.size() == static_cast<size_t>(kQLora * H));
  CHECK(L0.wq_b.size() == static_cast<size_t>(kHeads * kHeadDim * kQLora));
  CHECK(L0.wkv.size() == static_cast<size_t>(kHeadDim * H));
  CHECK(L0.wo_a.size() == static_cast<size_t>(kOGroups * kOLora * in_per_group));
  CHECK(L0.wo_b.size() == static_cast<size_t>(H * kOGroups * kOLora));
  CHECK(L0.attn_sink.size() == static_cast<size_t>(kHeads));
  CHECK(L0.hc_attn_fn.size() == static_cast<size_t>(hc3 * hc * H));
  CHECK(L0.shared_w1.size() == static_cast<size_t>(kInter * H));
  CHECK(L0.shared_w2.size() == static_cast<size_t>(H * kInter));
  CHECK(L0.shared_w3.size() == static_cast<size_t>(kInter * H));
  CHECK(L0.gate_weight.size() == static_cast<size_t>(kExperts * H));
  // Layer 0 is the HASH layer: the I64 `tid2eid` narrows to int32, exactly as
  // the GGUF arm narrows it, and the noaux_tc bias is absent.
  CHECK(L0.tid2eid.size() == static_cast<size_t>(kVocab * opt.topk));
  CHECK(L0.gate_bias.empty());

  // Layer 1 carries the DSA compressor + Lightning-Indexer at the COLLAPSED
  // geometry the host forward indexes.
  const vllm::DeepseekV4LayerHostWeights& L1 = w.host.layers[1];
  CHECK(L1.tid2eid.empty());
  CHECK(L1.gate_bias.size() == static_cast<size_t>(kExperts));
  CHECK(L1.comp_wgate.size() == static_cast<size_t>(kHeadDim * H));
  CHECK(L1.comp_ape.size() == static_cast<size_t>(4 * kHeadDim));
  CHECK(L1.comp_norm_weight.size() == static_cast<size_t>(kHeadDim));
  CHECK(L1.idx_wq.size() ==
        static_cast<size_t>(opt.index_n_heads * opt.index_head_dim * H));
  CHECK(L1.idx_wk.size() == static_cast<size_t>(opt.index_head_dim * H));
  CHECK(L1.idx_wproj.size() == static_cast<size_t>(opt.index_n_heads * H));

  // The routed experts are the TRELLIS tower; a second host copy of them would
  // be an unreachable duplicate of the thing this row exists to run.
  CHECK(L0.exp_w1.empty());
  CHECK(L0.exp_w2.empty());
  CHECK(L0.exp_w3.empty());
}

TEST_CASE("dsv4 exl3 W1c: the materialized VALUES are the checkpoint's, decoded") {
  // Sizes alone would pass for a tower filled with zeros, or for one whose FP8
  // block scale was ignored. Each family is recomputed from the fixture's own
  // generator and compared elementwise, so a decode that dropped the scale, read
  // the wrong scale block, or mistook BF16 for F16 fails here.
  auto f = BuildFixture();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_host_weights);
  const int64_t H = kHidden;

  // BF16 -> f32, exactly (bf16 widening is lossless).
  int64_t embed_mismatch = 0;
  for (int64_t i = 0; i < kVocab * H; ++i) {
    const float want =
        vt::BF16ToF32(vt::F32ToBF16(CarriedValue("embed.weight", i, 0.8f, 0.0f)));
    if (w.host.embed[static_cast<size_t>(i)] != want) ++embed_mismatch;
  }
  CHECK(embed_mismatch == 0);

  // F32 straight through.
  int64_t hc_mismatch = 0;
  for (int64_t i = 0; i < kHcMult * kHcMult * H; ++i)
    if (w.host.hc_head_fn[static_cast<size_t>(i)] !=
        CarriedValue("hc_head_fn", i, 0.2f, 0.0f))
      ++hc_mismatch;
  CHECK(hc_mismatch == 0);

  // Block-wise FP8: E4M3 byte times the UE8M0 scale of ITS 128x128 block. The
  // expectation is built from the two decode helpers directly, not from the
  // loader's own path, so the two cannot agree by sharing a bug.
  const std::string base = "layers.0.attn.wq_a";
  const int64_t N = kQLora, K = H;
  const int64_t kb = (K + kBlockK - 1) / kBlockK;
  // THE INSTRUMENT'S OWN PRECONDITION. This case can only see a decode that
  // reads the wrong scale BLOCK if the blocks carry different scales. They did
  // not when the generator had a three-value alphabet, and the mutation that
  // pins the scale to block [0] passed. Assert it rather than assume it.
  REQUIRE(kb > 1);
  CHECK(CarriedScaleByte(base + ".scale", 0) != CarriedScaleByte(base + ".scale", 1));
  int64_t fp8_mismatch = 0;
  int64_t nonzero = 0;
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      const uint8_t wb = CarriedFp8Byte(base + ".weight", n * K + k);
      const uint8_t sb =
          CarriedScaleByte(base + ".scale", (n / kBlockN) * kb + (k / kBlockK));
      const float want = vllm::F8E4M3ToF32(wb) * vllm::E8M0ToF32(sb);
      if (w.host.layers[0].wq_a[static_cast<size_t>(n * K + k)] != want) ++fp8_mismatch;
      if (want != 0.0f) ++nonzero;
    }
  }
  CHECK(fp8_mismatch == 0);
  // A tower of zeros would satisfy an equality check against a zero
  // expectation; it cannot satisfy this.
  CHECK(nonzero == N * K);
  // And the SCALE is load-bearing: the same weight bytes read without their
  // block scale differ from what landed.
  int64_t unscaled_agrees = 0;
  for (int64_t i = 0; i < N * K; ++i)
    if (w.host.layers[0].wq_a[static_cast<size_t>(i)] ==
        vllm::F8E4M3ToF32(CarriedFp8Byte(base + ".weight", i)))
      ++unscaled_agrees;
  CHECK(unscaled_agrees < N * K);
}

TEST_CASE("dsv4 exl3 W1c: the carried tower is read from MISALIGNED payloads") {
  // THE ALIGNMENT CONTRACT (#1923 follow-up). A safetensors payload starts at
  // `8 + header_bytes` and each tensor at whatever `data_offsets` names, so the
  // mmap'd address of a tensor satisfies NO alignment above 1. The first W1c
  // materialization formed `const uint16_t*` and `const int64_t*` straight into
  // that mapping and indexed them, which is undefined behaviour. x86 executes
  // the misaligned load and returns the right answer, so every local suite was
  // green; only CI's `sanitize-cpu (address,undefined)` lane reported it, on
  // the BF16 arm of `Exl3CarriedReader::Float` and the I64 arm of its
  // `HashTable`. The readers now go through `vt::LoadUnaligned`, the seam issue
  // #627 established for exactly this.
  //
  // This case is the durable half of that fix, and it pins TWO things.
  //
  // (1) THE FIXTURE'S OWN PRECONDITION. A fixture whose payload happened to land
  //     aligned would exercise nothing, on any lane, and the sanitizer would go
  //     quiet without the bug being gone. `WriteSafetensors` therefore pads its
  //     header until the payload base is ODD, and the REQUIREs below assert that
  //     the three tensors the widening readers actually consume are each
  //     misaligned for their own element type. A change that re-aligns the
  //     fixture reds HERE rather than silently muting the sanitizer lane.
  //
  // (2) THAT THE LOADER READS THEM CORRECTLY ANYWAY. The values are recomputed
  //     from the fixture's generators, so a reader that mis-assembles the bytes
  //     of an unaligned scalar fails on the numbers, not only on a report.
  //
  // What this case CANNOT do on x86 is red on the raw cast by itself: the
  // hardware performs that load. Under the sanitizer build it does, and that is
  // the lane this pin is aimed at. `Float`'s F16 arm and `HashTable`'s I32 arm
  // are not covered: the artifact carries neither (the F16 `suh`/`svh` belong to
  // the trellis tower, read by a different path), and the I32 arm is a bulk
  // `memcpy`, which has no alignment precondition at all.
  //
  // The hash-layer options, so `layers.0.ffn.gate.tid2eid` — the ONLY I64 the
  // carried tower has, and the second site UBSan reported — actually exists.
  // The default fixture has `num_hash_layers = 0` and leaves that arm unread.
  FixtureOptions opt;
  opt.num_hash_layers = 1;
  opt.topk = 2;
  auto f = BuildFixture(opt);

  const vllm::StTensor* embed = nullptr;
  const vllm::StTensor* hc = nullptr;
  const vllm::StTensor* tid = nullptr;
  for (const vllm::SafetensorsFile& shard : f->shards) {
    for (const std::string& name : shard.Names()) {
      if (name == "embed.weight") embed = &shard.Get(name);
      if (name == "hc_head_fn") hc = &shard.Get(name);
      if (name == "layers.0.ffn.gate.tid2eid") tid = &shard.Get(name);
    }
  }
  REQUIRE(embed != nullptr);
  REQUIRE(hc != nullptr);
  REQUIRE(tid != nullptr);
  CHECK(embed->dtype == "BF16");
  CHECK(hc->dtype == "F32");
  CHECK(tid->dtype == "I64");

  auto misaligned_for = [](const vllm::StTensor* t, size_t align) {
    return (reinterpret_cast<uintptr_t>(t->data) % align) != 0;
  };
  // BF16 is what UBSan reported at :431, I64 what it reported at :458. F32 is
  // read by a bulk memcpy today; it is asserted so that converting that arm to a
  // typed load can never be done onto an accidentally-aligned fixture.
  REQUIRE(misaligned_for(embed, alignof(uint16_t)));
  REQUIRE(misaligned_for(hc, alignof(float)));
  REQUIRE(misaligned_for(tid, alignof(int64_t)));

  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_host_weights);

  // And the misaligned bytes decoded to the right numbers.
  int64_t bf16_mismatch = 0;
  for (int64_t i = 0; i < kVocab * kHidden; ++i)
    if (w.host.embed[static_cast<size_t>(i)] !=
        vt::BF16ToF32(vt::F32ToBF16(CarriedValue("embed.weight", i, 0.8f, 0.0f))))
      ++bf16_mismatch;
  CHECK(bf16_mismatch == 0);

  int64_t f32_mismatch = 0;
  for (int64_t i = 0; i < kHcMult * kHcMult * kHidden; ++i)
    if (w.host.hc_head_fn[static_cast<size_t>(i)] !=
        CarriedValue("hc_head_fn", i, 0.2f, 0.0f))
      ++f32_mismatch;
  CHECK(f32_mismatch == 0);

  // The I64 hash table, narrowed to int32 exactly as the GGUF arm narrows it.
  const int64_t topk = static_cast<int64_t>(w.host.layers[0].tid2eid.size()) / kVocab;
  REQUIRE(topk >= 1);
  REQUIRE(w.host.layers[0].tid2eid.size() ==
          static_cast<size_t>(kVocab * topk));
  int64_t i64_mismatch = 0;
  for (int64_t i = 0; i < kVocab * topk; ++i)
    if (w.host.layers[0].tid2eid[static_cast<size_t>(i)] !=
        static_cast<int32_t>(NameHash("layers.0.ffn.gate.tid2eid", i) % kExperts))
      ++i64_mismatch;
  CHECK(i64_mismatch == 0);
}

TEST_CASE("dsv4 exl3 W1c: a carried tensor this arm cannot route REFUSES BY NAME") {
  SUBCASE("the REAL artifact's 2*head_dim compressor") {
    // MEASURED on `0xSero/deepseek-v4-flash-0731-spark` @ `22f28d32`
    // (2026-08-25): `layers.N.attn.compressor.wgate.weight` is BF16
    // [1024, 4096] = [2*head_dim, H], the ds4 `coff = 2` width, while the host
    // forward's compressor indexes [head_dim, H]. `Gemm`'s host arm is a
    // `MatVec` with no length check, so materializing the wide tensor into that
    // slot is a SILENTLY WRONG number rather than a crash — the refusal is what
    // keeps it from being one. 41 of the real artifact's 43 layers carry a
    // compressor, so this is the shape that stops the real checkpoint, and the
    // spec's `## Owed` names what would close it.
    FixtureOptions opt;
    opt.layers = 1;
    opt.compress_ratios = {128};
    opt.real_compressor_width = true;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "compressor"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
    CHECK(Mentions(msg, "W1c"));
    // The REASON, not just the fact of a throw: both shapes in the message.
    // `compressor.ape` is the first of the family the loader reaches, so it is
    // the one that names the width — [compress_ratio, head_dim] wanted against
    // the artifact's [compress_ratio, 2*head_dim].
    CHECK(Mentions(msg, "[128,512]"));
    CHECK(Mentions(msg, "[128,1024]"));
  }
  SUBCASE("no recipe for the carried FP8 half") {
    // The carried MLA linears are block-wise FP8 and the block size comes from
    // the checkpoint, not from a constant in this loader. Without it there is
    // nothing to assume.
    FixtureOptions opt;
    opt.omit_base_quant_config = true;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "base_quantization_config"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
    CHECK(Mentions(msg, "W1c"));
  }
}

TEST_CASE("dsv4 exl3: unrepresentable inputs REFUSE BY NAME") {
  SUBCASE("an unknown rank-sliced schema version") {
    FixtureOptions opt;
    opt.version = "rank-sliced-deepseek-v4-v2";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "rank-sliced-deepseek-v4-v2"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("a codebook other than mcg") {
    FixtureOptions opt;
    opt.codebook = "mul1";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "mul1"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("a missing rank tensor") {
    FixtureOptions opt;
    opt.drop_tensor = "layers.0.ffn.experts.1.w2.rank2.suh";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "layers.0.ffn.experts.1.w2.rank2.suh"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("a whole missing rank") {
    FixtureOptions opt;
    opt.ranks_written = 3;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "rank3"));
  }
  SUBCASE("a carried tensor no arm routes") {
    FixtureOptions opt;
    opt.extra_carried = "layers.0.attn.wq_c.weight";
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "layers.0.attn.wq_c.weight"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
  }
  SUBCASE("w2 sliced on the wrong axis") {
    FixtureOptions opt;
    opt.swap_w2_slice_axis = true;
    auto f = BuildFixture(opt);
    const std::string msg = ThrowMessage(
        [&] { vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
    CAPTURE(msg);
    CHECK(Mentions(msg, "w2"));
    CHECK(Mentions(msg, "MODEL-DSV4-EXL3"));
    // Pin the REASON, not just the fact of a throw: a refusal test that passes
    // because some earlier check happened to fire gates nothing. This must be
    // the replicated-side comparison noticing that w2's svh differs per rank.
    CHECK(Mentions(msg, "IN-split"));
    CHECK(Mentions(msg, "svh"));
  }
}
