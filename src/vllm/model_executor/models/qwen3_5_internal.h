// Internal Qwen3.5/3.6 model-selection contracts. Not installed.
// Packed-decode selection mirrors vLLM v0.25.0
// qwen_gdn_linear_attn.py:1286-1298 @ 702f4814, with the first local consumer
// deliberately scoped to the 27B CUDA packed-BA/BF16-activation path. The SSM
// cache dtype is independent (FP32 in the gate checkpoint).
#pragma once

#include <cstdint>
#include <vector>

#include "vt/dtype.h"

namespace vt {
struct Queue;
}  // namespace vt

namespace vllm {

struct GdnStateCache;
struct HfConfig;
struct GdnLayerWeights;

namespace v1 {
struct GDNAttentionMetadata;
}  // namespace v1

// Test-only entry point (SPEC-MTP I5a): run one GDN layer's paged forward over a
// batched step (spec or non-spec, per `meta`) and return the [T*H] output on host
// (f32). Drives the real per-step upload (BuildStepDevInputs) + layer assembly
// (GdnBlockPaged), so the synthetic spec-branch test exercises the exact
// production routing. Defined in qwen3_5.cpp.
// Test-only entry point (PERF-27B-GDN-FP8-QKVZ): run ONE GDN layer's FP8 input
// projection over `h_host` ([T*H] f32, rounded to bf16 on upload exactly like
// the real forward's embed target) through EITHER the merged single-GEMM arm
// (`merged=true`) or the exact two legacy fp8 GEMMs, and return the
// concatenation [mixed_qkv | z] as [T*(conv_dim+value_dim)] f32. `z` is
// produced at BF16 when `z_bf16` (the 27B dense default `GdnOutDType`) and
// upcast on the way out, so the two arms are directly byte-comparable. The
// merged arm is byte-identical to the split arm by construction; this proves it
// on the real fp8 GEMM. CUDA-only (the fp8 W8A8 path is). Defined in
// qwen3_5.cpp.
std::vector<float> ProjectGdnFp8QkvzForTest(vt::Queue queue,
                                            const GdnLayerWeights& w,
                                            const std::vector<float>& h_host,
                                            int64_t T, int64_t conv_dim,
                                            int64_t value_dim, bool merged,
                                            bool z_bf16);

std::vector<float> GdnBlockPagedForTest(vt::Queue queue, const GdnLayerWeights& w,
                                        const HfConfig& cfg,
                                        const std::vector<float>& h_host,
                                        const v1::GDNAttentionMetadata& meta,
                                        std::vector<float>& ssm_host,
                                        std::vector<float>& conv_host,
                                        int64_t num_slots, int64_t conv_len,
                                        int64_t T);

}  // namespace vllm

namespace vllm::detail {

struct GdnPackedDecodeEligibility {
  bool runtime_enabled = false;
  bool cuda = false;
  bool dense_model = false;
  bool has_packed_ba = false;
  bool merged_ba_enabled = false;
  bool dtype_compatible = false;
  bool has_state_indices = false;
  int64_t num_prefills = 0;
  int64_t num_prefill_tokens = 0;
  int64_t num_decodes = 0;
  int64_t num_decode_tokens = 0;
  int64_t num_spec_decodes = 0;
  int64_t num_spec_decode_tokens = 0;
  int64_t num_actual_tokens = 0;
};

bool ShouldUsePackedGdnDecode(const GdnPackedDecodeEligibility& eligibility);

// Process-level ENV resolution of the packed pure-decode arm for the real 27B
// dense CUDA gate (owner resident, pure non-spec one-token decode assumed).
// Mirrors, field for field, the exact process-cached env couplings the model
// wires into GdnPackedDecodeEligibility (qwen3_5.cpp GdnBlockPaged):
// PackedGdnDecodeRuntimeEnabled (VT_GDN_PACKED_DECODE), MergedGdnBaEnabled
// (master VT_GDN_MERGED_PROJ + leaf VT_GDN_MERGED_BA), and dtype_compatible —
// GdnInDType (VT_GDN_IN_BF16) == BF16, GdnOutDType dense default
// (VT_GDN_OUT_BF16) == BF16, MergedGdnBaOutputDType(packed)
// (VT_GDN_BA_OUT_BF16) == BF16. Splitting BA (master OR leaf) or reverting any
// coupled dtype therefore deselects packed and runs the decomposed recurrence:
// the gate's dispatch-count contract must expect ZERO packed launches on those
// arms, exactly like VT_GDN_PACKED_DECODE=0. Fields carry the raw getenv
// values (nullptr = unset) so the CPU tier can pin the truth table.
struct GdnPackedDecodeEnvConfig {
  const char* packed_decode = nullptr;  // VT_GDN_PACKED_DECODE
  const char* merged_proj = nullptr;    // VT_GDN_MERGED_PROJ (master)
  const char* merged_ba = nullptr;      // VT_GDN_MERGED_BA (leaf)
  const char* in_bf16 = nullptr;        // VT_GDN_IN_BF16
  const char* out_bf16 = nullptr;       // VT_GDN_OUT_BF16
  const char* ba_out_bf16 = nullptr;    // VT_GDN_BA_OUT_BF16
};

bool PackedGdnDecodeEnvSelected(const GdnPackedDecodeEnvConfig& env);

// PERF-27B-GDN-PACKED-REACHABLE (#365) — what `dtype_compatible` above is
// ACTUALLY about.
//
// Packed-decode selection happens BEFORE `ProjectGdnQkvz` runs, because the
// decision changes how BA is projected (`ProjectGdnBA(..., packed_decode)`), so
// the eligibility cannot read `mixed.dtype`; it has to PREDICT it. This mirrors
// `ProjectGdnQkvz`'s branch order exactly — the merged BF16 `in_proj_qkvz`
// owner is checked first, then the native-FP8 owner, then the split BF16 owner
// — and `fp8_out_dtype` is sourced from the same single helper the projection
// itself reads, so the prediction cannot drift from what the GEMM emits.
struct GdnMixedQkvDTypeInputs {
  bool has_bf16_qkvz_owner = false;  // w.in_proj_qkvz populated (27B bf16, 4B)
  bool has_fp8_qkv_owner = false;    // w.in_proj_qkv_fp8 populated (modelopt)
  // PERF-GDN-PACKED-BRIDGE (#365). ShouldUseMergedGdnFp8Qkvz held for this
  // layer, i.e. ProjectGdnQkvz takes the MERGED fp8 arm rather than the split
  // one. Required because `VT_GDN_FP8_IN_BF16` narrows the merged arm ONLY --
  // `fp8_indt` reaches MergedFp8QkvzD and nothing else, while the split arm
  // still hardcodes F32. A prediction that ignored the arm would claim BF16 on
  // a split-arm checkpoint, which is the UNSAFE direction: vt::GdnPackedDecode
  // would then be handed an f32 mixed_qkv against bf16 a/b/out and THROW.
  bool fp8_merged_arm = false;
  vt::DType in_dtype = vt::DType::kF32;       // GdnInDType()
  vt::DType fp8_out_dtype = vt::DType::kF32;  // the fp8 in_proj epilogue's dtype
};

vt::DType GdnProjectedMixedQkvDType(const GdnMixedQkvDTypeInputs& in);

// PERF-GDN-PACKED-BRIDGE (#365) -- the SINGLE decision for the dtype the
// native-FP8 MERGED GDN in_proj emits, shared by the two parties that must
// agree about it: the PRODUCER (`ProjectGdnQkvz`'s `fp8_indt`, handed to
// `MergedFp8QkvzD`, which allocates the buffer) and the PREDICTOR
// (`GdnFp8MixedQkvDType`, read by the packed-decode eligibility BEFORE the
// projection has run, because the decision feeds `ProjectGdnBA`). They cannot
// drift because they are the same call.
//
// Terms are PERF-FP8-ALPHA-FOLD's own, verbatim: the opt-in toggle
// (`VT_GDN_FP8_IN_BF16`, default OFF), `indt == BF16` so `VT_GDN_IN_BF16`'s
// documented rollback still restores f32 on this arm too, and `outdt == BF16`,
// which confines the narrowing to the dense 27B (the 35B is MoE, so
// `GdnOutDType` is f32 there and this stays inert).
vt::DType GdnFp8MergedMixedQkvDType(bool fp8_in_bf16_enabled, vt::DType in_dtype,
                                    vt::DType out_dtype);

// The packed-decode dtype rule, keyed on the ACTIVATION dtypes the op needs and
// on NOTHING about how the weights are stored. `vt::GdnPackedDecode`'s own
// contract (src/vt/ops.cpp, "gdn_packed_decode: mixed_qkv/a/b/out must share
// FP16/BF16/F32 dtype") is a uniformity rule over the four activation tensors;
// the SSM state is explicitly INDEPENDENT of them ("state must use an
// independent FP16/BF16/F32 dtype"), matching upstream FLA, which requires only
// last-dim contiguity per tensor and casts each operand to f32 on load
// (fla/ops/fused_recurrent.py:256-336 @ 702f4814). The model's packed leg pins
// the shared dtype to BF16 — the dtype both the hand `GdnPackedDecodeKernel`
// and the vendored FLA AOT cubin (`gdn_decode_h48`) are exercised at.
//
// This deliberately replaces the previous `in_proj_qkv_fp8.Empty() &&
// in_proj_z_fp8.Empty()` term. That term was a PROXY: it excluded an fp8 GDN
// tower because the fp8 GEMM emits f32 while a/b/out are bf16, which is a
// statement about the ACTIVATION dtype, not about the weight format. Upstream
// conditions the packed decode on nothing about the quantization method.
struct GdnPackedDecodeDTypes {
  vt::DType mixed_qkv = vt::DType::kF32;  // post-conv activation (== mixed.dtype)
  vt::DType ba_out = vt::DType::kF32;     // MergedGdnBaOutputDType(packed)
  vt::DType core_out = vt::DType::kF32;   // dcore, GdnOutDType(dense)
  vt::DType ssm_state = vt::DType::kF32;  // the recurrent cache, independent
};

bool GdnPackedDecodeDTypesCompatible(const GdnPackedDecodeDTypes& dtypes);

// VT_GDN_PACKED_DECODE_FP8_TOWER, DEFAULT OFF — the same-binary rollback of the
// rule above. OFF reproduces the legacy exclusion exactly (an fp8 GDN tower is
// never packed-decode eligible, whatever dtypes it produces), so the production
// default is byte-identical to the pre-#365 selection on every checkpoint. ON
// drops the weight-format term and lets `GdnPackedDecodeDTypesCompatible` decide
// alone. Parsed here rather than at the getenv so the CPU tier can pin it;
// '1'-leading is ON and everything else (including unset) is OFF, the house
// default-OFF convention (vt::cuda::GdnPackedRegTileFlagIsOn).
//
// NOTE this can only ever PERMIT, never DESELECT: the call-site clause is
// `!fp8_tower || allowed`, so on a checkpoint with no fp8 GDN shards it is
// unconditionally true and `PackedGdnDecodeEnvSelected`'s truth table above is
// unaffected.
bool PackedGdnDecodeFp8TowerFlagIsOn(const char* env_value);

// W2 merged-qkvz dispatch. vLLM always issues one in_proj_qkvz GEMM
// (qwen_gdn_linear_attn.py:923-936 @ 702f4814); locally the single GEMM is
// selected only on CUDA with the packed 27B owner resident, the runtime
// toggles on (VT_GDN_MERGED_PROJ master, VT_GDN_MERGED_QKVZ leaf) and one
// uniform output dtype (mixed_qkv and z leave one GEMM, so GdnInDType must
// equal GdnOutDType — the 27B default is BF16/BF16). Every other combination
// issues the exact two split GEMMs sliced from the same resident owner.
struct GdnMergedQkvzEligibility {
  bool runtime_enabled = false;
  bool cuda = false;
  bool has_packed_qkvz = false;
  bool uniform_dtype = false;
};

bool ShouldUseMergedGdnQkvz(const GdnMergedQkvzEligibility& eligibility);

// PERF-27B-GDN-FP8-QKVZ — the FP8 (W8A8) leaf of the same merge. The BF16 leaf
// above owns a merged `in_proj_qkvz` BF16 parameter; a ModelOpt FP8 tower
// (`nvidia/Qwen3.6-27B-NVFP4` is `modelopt_mixed`, and the 35B shares the
// tower) keeps the two shards NATIVE and so cannot use that owner. vLLM still
// runs ONE merged qkvz GEMM per GDN layer (qwen_gdn_linear_attn.py:923-936 @
// 702f4814; MergedColumnParallelLinear packs qkv+z along N), which is what this
// leaf mirrors: the two RAW fp8 [N,K] shards are N-concatenated ONCE into one
// resident operand and one fp8 GEMM replaces the two.
//
// The merge only reproduces the split arithmetic when the two shards agree on
// the ACTIVATION scale, because the merged GEMM quantizes the activation once:
// ModelOpt fp8 here is per-tensor, so `in_proj_qkv.input_scale` must equal
// `in_proj_z.input_scale` BITWISE. That is exactly the predicate
// `GdnFp8SharedInputScale` (the same one `Fp8SharedInputScale` already applies
// to this pair for the fused RmsNorm+quant), evaluated ONCE when the resident is
// built, never per step. The per-shard WEIGHT scales need no agreement: each
// shard's folded alpha is applied per output column (folded into the GEMM when
// both are equal, else through the resident alpha vector), exactly as the
// merged FP8 QKV path does.
struct GdnMergedFp8QkvzEligibility {
  bool runtime_enabled = false;  // VT_GDN_MERGED_QKVZ_FP8 (+ the BF16 leaf's
                                 // master VT_GDN_MERGED_PROJ / leaf
                                 // VT_GDN_MERGED_QKVZ rollbacks)
  bool fp8_platform = false;     // supports_fp8() + the fp8 GEMM registered
  bool has_fp8_shards = false;   // both in_proj_qkv_fp8 and in_proj_z_fp8 live
  bool shared_k = false;         // both shards read the same [M,K] activation
  bool shared_input_scale = false;  // bitwise-equal per-tensor input_scale
  bool shard_widths_match = false;  // shard N == conv_dim / value_dim
};

bool ShouldUseMergedGdnFp8Qkvz(const GdnMergedFp8QkvzEligibility& eligibility);

// Process-level ENV resolution of the FP8 leaf, mirroring
// PackedGdnDecodeEnvSelected: fields carry the raw getenv values (nullptr =
// unset) so the CPU tier can pin the truth table that `MergedGdnFp8QkvzEnabled`
// caches. VT_GDN_MERGED_QKVZ_FP8 is the leaf switch and defaults ON; the BF16
// leaf's master (VT_GDN_MERGED_PROJ) and leaf (VT_GDN_MERGED_QKVZ) rollbacks
// also disable it, so one switch retires the whole merged-input-projection
// topology.
struct GdnMergedFp8QkvzEnvConfig {
  const char* merged_proj = nullptr;      // VT_GDN_MERGED_PROJ (master)
  const char* merged_qkvz = nullptr;      // VT_GDN_MERGED_QKVZ (BF16 leaf)
  const char* merged_qkvz_fp8 = nullptr;  // VT_GDN_MERGED_QKVZ_FP8 (FP8 leaf)
};

bool MergedGdnFp8QkvzEnvSelected(const GdnMergedFp8QkvzEnvConfig& env);

// The load-time scale-compatibility predicate itself, exposed so the CPU tier
// can pin it: true (and *scale filled) only when BOTH fp8 GDN input shards are
// populated and carry the SAME per-tensor activation scale, compared with exact
// float equality. `Fp8SharedInputScale`'s linear-attention branch delegates
// here, so the fused-quant guard and the merge guard can never drift apart.
bool GdnFp8SharedInputScale(const GdnLayerWeights& gdn, float* scale);

// Model-selection instrumentation for the FP8 GDN input projections, mirroring
// vt::cuda::testing::GdnPackedDecodeDebugStats: counts the host GEMM dispatches
// (`merged` = one per GDN layer, `split` = two per GDN layer) so a real-model
// step can assert the structural 96 -> 48 collapse without a profiler. Graph
// replay has no host dispatch, so this is read on an eager step.
struct GdnFp8InProjDebugStats {
  uint64_t merged_launches = 0;
  uint64_t split_launches = 0;
  uint64_t Total() const { return merged_launches + split_launches; }
};

void ResetGdnFp8InProjDebugStats();
GdnFp8InProjDebugStats GetGdnFp8InProjDebugStats();
void DisableGdnFp8InProjDebugStats();

// Validate the exact prefix that will be uploaded. Negative rows are inert
// padding; every live slot must be unique and in range. This runs on host
// metadata before the device buffer is constructed, keeping CUDA capture free
// of a validation synchronization. Uniqueness is an O(n) seen-set pass (a live
// slot is drawn from a free-list of distinct slots by construction, so a single
// pass fails closed on any duplicate/out-of-range/negative slot); it is bounded
// by state_slots (== max_num_reqs). `force_full_uniqueness` (driven globally by
// VT_GDN_VALIDATE=1) additionally runs the exhaustive O(n^2) pairwise
// cross-verification — a redundant paranoid check, never needed for
// correctness.
void ValidateGdnStateIndices(const std::vector<int32_t>& indices,
                             int64_t required, int64_t state_slots,
                             bool force_full_uniqueness = false);

// Validate the complete eager/graph metadata contract before any state index
// is uploaded or consumed. Prefill-only vectors must be exact suffix/rebased
// views of the full non-spec vectors. Only CUDA-graph padding may contain the
// inert -1 state-slot sentinel.
void ValidateGdnAttentionMetadata(
    const v1::GDNAttentionMetadata& metadata, int64_t state_slots,
    bool allow_inert_padding);

// Row-copy state I/O cannot consume inert padded rows. Exact-size graphs are
// safe under either state-I/O mode; larger padded graph sizes require indexed
// state I/O, whose kernels define the -1 sentinel.
bool CanUseGdnDecodeGraphSize(int64_t real_batch, int64_t capture_batch,
                              bool indexed_state_io);

// Validate that every per-layer GDN cache exposes the same slot domain. Eager
// and graph paths both upload one shared state-index vector, so accepting a
// smaller later-layer cache would turn an index valid for layer zero into an
// out-of-bounds access. Returns the common slot count (zero for no caches).
int64_t ValidateGdnStateCacheLayout(
    const std::vector<GdnStateCache>& state_caches);

// Upstream MambaStateDtypeCalculator::_mamba_state_dtype. The temporal/SSM
// cache dtype is independent from the convolution cache dtype and accepts the
// exact torch dtype aliases used by raw HF configs.
vt::DType ResolveMambaSsmCacheDType(const HfConfig& config,
                                    vt::DType conv_dtype);

// Host preflight run at the entrance of every CUDA-graph Step, before padding,
// refresh, capture, or replay. It validates the real live request prefix
// against both state tensors in every GDN layer; the lower-level upload check
// remains defense in depth for eager/mixed paths.
void ValidateGdnDecodeGraphState(
    const v1::GDNAttentionMetadata& metadata,
    const std::vector<GdnStateCache>& state_caches, int64_t real_batch);

// ─── ENG-ASYNC-SCHED W4: device-resident input ids for the embed ─────────────
//
// `ModelForwardInput::device_token_ids` says "the input ids for this step are
// ALREADY on the device; the host vector is stale for decode rows". Only the
// embed at the very top of the forward cares, and it sits under five layers of
// entry points (eager / gathered / tap / multi-tap / decode-graph replay), each
// of which takes `const std::vector<int32_t>& token_ids` and passes it down.
//
// Rather than add a defaulted pointer parameter to every one of those and to the
// decode-graph class, the two Qwen3.5 registry forwards establish this SCOPED
// override for the duration of one forward and the embed consults it. The
// trade-off is deliberate and bounded: it is thread-local (a forward runs on one
// host thread), strictly RAII so it cannot leak past the call that set it, and
// set ONLY from the registry entry points that receive the ModelForwardInput —
// so its lifetime is exactly the forward's, not process state. It is null on
// every path except the discrete-CUDA async runner.
// The COUNT travels with the pointer so the embed can prove the buffer is the
// one meant for it. A forward can reach a second, unrelated embed over different
// ids (the multimodal generate helper embeds a prompt and then single tokens);
// an override that matched on "non-null" alone would silently feed that embed
// the wrong row count. Length disagreement means "not mine" and falls back to
// the host upload, which is always correct.
struct DeviceTokenIds {
  const int32_t* ids = nullptr;
  int64_t count = 0;
};

DeviceTokenIds& DeviceTokenIdsOverride();

struct DeviceTokenIdsScope {
  DeviceTokenIdsScope(const int32_t* ids, int64_t count)
      : prev(DeviceTokenIdsOverride()) {
    DeviceTokenIdsOverride() = DeviceTokenIds{ids, count};
  }
  ~DeviceTokenIdsScope() { DeviceTokenIdsOverride() = prev; }
  DeviceTokenIdsScope(const DeviceTokenIdsScope&) = delete;
  DeviceTokenIdsScope& operator=(const DeviceTokenIdsScope&) = delete;
  DeviceTokenIds prev;
};

// ─── ENG-EXPERT-STREAM (#912): the streamed-expert lane, seen from outside ───
//
// The lane lives in the anonymous namespace of qwen3_5.cpp because nothing
// outside the forward may construct it. Two things still have to reach it.
//
// A BENCHMARK has to be able to prove the lane stayed live. The row's published
// "streaming ON: no decode gain" number was measured on a cache that had
// switched itself off partway through the third token, and nothing in the run
// could have said so: the process printed one line at startup and none
// afterwards. `steps` and `exhausted` are the two fields that make that state
// visible — steps==0 means the step clock never advanced, exhausted>0 means
// slices were refused and served from the mapping instead.
//
// A GATE has to be able to prove decode still REACHES the lane. Counters that
// stay at zero when the production call site is deleted are what turns "the
// class works" into "the capability is wired", which is the distinction this
// row previously failed.
struct ExpertStreamStats {
  // A slot store exists. False when streaming was never requested, and also
  // when it was requested but no expert slice was ever taken — which is itself
  // the reachability failure worth catching.
  bool active = false;
  int64_t steps = 0;
  int64_t hits = 0;
  int64_t misses = 0;
  int64_t evictions = 0;
  int64_t fills = 0;
  int64_t bytes_filled = 0;
  // Slices the cache could not serve, which fell back to the mapping. Nonzero
  // means the budget is smaller than one step's working set, OR that the step
  // boundary is not being called at all.
  int64_t exhausted = 0;
  // madvise(MADV_WILLNEED) calls the kernel ACCEPTED. Zero while slices are
  // being filled from a mapping means the hint is being rejected, which is what
  // an unaligned address does silently.
  int64_t advised = 0;
};

ExpertStreamStats ExpertStreamSnapshot();

// Force every slice to take the cache-exhaustion fallback, i.e. the resident
// tower view. A real production state (a budget below one step's working set
// reaches it), exposed so one process can compare the streamed and unstreamed
// arms and prove they produce identical bytes.
void ExpertStreamSetForceFallback(bool on);

// End one decode step for the streamed-expert cache. The Qwen3.5 MoE forward
// runs this from its own layer driver; a SECOND full-attention MoE model
// (qwen3_moe.cpp) composes the same block from another translation unit and
// calls it from its layer driver for the same reason.
void EndExpertStreamStep();

}  // namespace vllm::detail
