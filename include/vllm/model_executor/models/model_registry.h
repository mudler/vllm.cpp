// Ported from: vllm/model_executor/models/registry.py
//               @ e24d1b24fe96a56ba8b0d653efa076d03eb95d6c
// (_ModelInfo:746-796, _ModelRegistry:998-1082,
//  resolve_model_cls:1244-1296, global registry:1396-1404).
//
// Python-only lazy imports, subprocess inspection/cache, dynamic Transformers,
// terratorch, and out-of-tree runtime loading intentionally have no C++ runtime
// analogue. The ordered lookup, capability metadata, implemented-model factory
// hooks, and unsupported-architecture messages mirror the pinned registry.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/config/multimodal.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class GgufFile;
class SafetensorsFile;
struct ForwardLogits;
// BACKEND-DISTRIBUTED-TP TP-W1: the per-rank tensor-parallel handle (comm +
// rank), defined in tensor_parallel.h. Forward-declared so the shared registry
// seam does not pull the TP/communicator headers; a LoadedModel carries an
// optional BORROWED pointer to it, set only when tp_size>1 (null = single-GPU,
// byte-identical).
struct TensorParallel;
struct GdnStateCache;
struct PagedKvCache;
struct Qwen3_5DenseWeights;
struct Qwen3_5MoeWeights;
// SPEC-MTP I5d-pre: the checkpoint-owned MTP draft weights, the drafter's
// hidden-state carrier, and the draft model wrapper (all defined in
// qwen3_5_mtp.h). Forward-declared here so the type-erased LoadedModel can
// expose a typed path to the draft (BuildMtpDraft) and a hidden-state tap
// out-field WITHOUT pulling the MTP header — or any concrete weight type — into
// this shared registry seam. Inert unless speculative decoding is configured.
struct Qwen3_5MTPWeights;
struct Qwen3_5MTPHiddenStates;
struct Qwen3_5AuxTaps;
class Qwen3_5MTPModel;

namespace v1 {
struct CommonAttentionMetadata;
struct GDNAttentionMetadata;
}  // namespace v1

// Consumed subset of upstream _ModelInfo. Extend this POD as task rows land;
// these fields are enough to choose generation vs pooling/scoring and hybrid vs
// full-attention construction without importing a Python model class.
struct ModelInfo {
  bool is_text_generation_model = false;
  bool is_pooling_model = false;
  bool is_hybrid = false;
  bool has_inner_state = false;
  bool supports_multimodal = false;
  // Mirror of the SupportsTranscription protocol
  // (vllm/model_executor/models/interfaces.py:1110-1118):
  // `supports_transcription` marks an ASR-capable arch;
  // `supports_transcription_only` marks one with NO text-generation path
  // (interfaces.py:1118 `supports_transcription_only: ClassVar[bool]`), which
  // the entrypoints use to refuse-by-task: LoadedEngine::FromModelDir rejects
  // such an arch with a message pointing at the transcription entry points
  // (vllm_transcribe / /v1/audio/transcriptions), mirroring how vLLM excludes
  // "generate" from supported_tasks for them.
  bool supports_transcription = false;
  bool supports_transcription_only = false;
  std::string_view score_type = "bi-encoder";
};

// Type-erased checkpoint source passed to a registration's weight loader.
struct ModelSource {
  enum class Kind { kSafetensors, kGguf };

  static ModelSource FromSafetensors(
      const std::vector<SafetensorsFile>& shards,
      vt::Queue* load_queue = nullptr);
  // Shares ownership of the mmap'd shards so a loader may keep them alive past
  // the load (e.g. the Qwen3.6-35B MoE deferred-expert streaming that
  // materializes routed experts per layer during PrepareMarlinResident). The
  // borrowing `safetensors` pointer is set to the owned vector.
  static ModelSource FromSafetensorsOwned(
      std::shared_ptr<const std::vector<SafetensorsFile>> shards,
      vt::Queue* load_queue = nullptr);
  static ModelSource FromGguf(const GgufFile& gguf);

  Kind kind = Kind::kSafetensors;
  const std::vector<SafetensorsFile>* safetensors = nullptr;
  // Non-null only for FromSafetensorsOwned: shared ownership a loader can retain.
  std::shared_ptr<const std::vector<SafetensorsFile>> safetensors_owned;
  const GgufFile* gguf = nullptr;
  // Non-owning queue selected by the entrypoint. Dense plain-BF16 loaders may
  // stage completed layers here and the runner then reuses the same queue.
  vt::Queue* load_queue = nullptr;
  // ENG-MM-INPUT-PIPELINE wave L3 (#607): the engine's multimodal input limits,
  // BORROWED for the duration of one load. A loader that owns a tower asks
  // `SkipTowerForModalities` (models/interfaces.h, the mirror of
  // interfaces.py:293) whether to read that tower's tensors at all.
  //
  // Upstream reaches the same value through `vllm_config.model_config
  // .multimodal_config` inside the model's own `__init__`. Our loader seam
  // (ModelWeightLoader below) carries no vllm_config, and ModelSource is
  // already the per-load CONTEXT rather than only the checkpoint — `load_queue`
  // beside it is an engine-selected execution resource, not a property of the
  // file — so this is where the borrow belongs. The two alternatives are
  // recorded in specs/multimodal-track.md §1.5 L3 with the reason each was
  // rejected: a third ModelWeightLoader parameter rewrites every architecture's
  // signature to thread a value nearly all of them ignore, and a process-global
  // (WeightOffloader's shape) has no upstream analogue for this config.
  //
  // NULL means "no limits configured for this load": load everything, which is
  // byte-identical to pre-L3 and is what every non-engine caller gets.
  const MultiModalConfig* multimodal = nullptr;
};

struct ModelFactory;
struct ModelRegistration;

// Opaque, lifetime-owning (or test-only borrowing) concrete model. The runner
// sees only this base and dispatches through its registration's function table.
class LoadedModel {
 public:
  virtual ~LoadedModel();

  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
  LoadedModel(LoadedModel&&) = delete;
  LoadedModel& operator=(LoadedModel&&) = delete;

  const ModelRegistration& registration() const { return registration_; }
  // Runtime capability rather than architecture metadata: GGUF/synthetic
  // instances of a W4A4-capable family may contain only BF16 weights.
  virtual bool uses_nvfp4_w4a4() const { return false; }

  // ENG-MM-INPUT-PIPELINE wave L3 (#607): the mirror of
  // `SupportsMultiModal._tower_model_names` (interfaces.py:141,298) together
  // with the `stage_name` each skipped stage carries (`:279-282`). The stage
  // names of the towers this model CONSTRUCTED WITHOUT LOADING because every
  // modality they serve had limit 0 — upstream's
  // `isinstance(model.visual, StageMissingLayer)`, expressed on a type-erased
  // base.
  //
  // EMPTY on every text model, and on every multimodal model loaded with a
  // non-zero limit. It exists so the skip is observable from a PRODUCTION entry
  // point (LoadedEngine::skipped_towers) rather than only by downcasting to a
  // concrete model class inside a test.
  virtual std::vector<std::string> skipped_towers() const { return {}; }

  // ARCH-ONE-SURFACE ROW 6: the model-owned Pooler of a POOLING model — the
  // mirror of upstream `VllmModelForPooling.pooler` (as_embedding_model wires
  // `self.pooler = DispatchPooler.for_embedding(...)`, adapters.py:248-257).
  // Non-null iff the registration's info.is_pooling_model; the GPU runner
  // builds its PoolingRunner over exactly this pooler (the mirror of
  // gpu/model_runner.py:368-369 `PoolingRunner(self.model)`). Default null:
  // every text-generation model is byte-identical.
  virtual const class Pooler* pooler() const { return nullptr; }

  // ── SPEC-MTP I5d-pre: typed access to the MTP draft, without breaking the
  //    type-erasure of this base. Only the concrete Qwen3.5 dense/MoE
  //    LoadedModel (which owns the target Qwen3_5DenseWeights/Qwen3_5MoeWeights)
  //    can retain the loaded `mtp.*` weights and construct the draft that shares
  //    the target's embed_tokens/lm_head (qwen3_5_mtp.h:61-66). Every default
  //    below is the non-MTP behavior: no support, and BuildMtpDraft yields null.
  //    All of this is unreachable on the production default path (no
  //    SpeculativeConfig ⇒ FromModelDir never attaches MTP weights), so the
  //    engine is byte-identical when speculative decoding is off. ──────────────

  // Whether this concrete model can hold MTP draft weights and build the draft.
  virtual bool supports_mtp_draft() const { return false; }

  // Whether this concrete model can capture the DFlash/DSpark AUX MULTI-TAP —
  // the residual stream at the draft's `target_layer_ids`, written into
  // `ForwardInput::aux_tap` as [T, H x taps]. Both block drafters CONDITION on
  // that tap, so a target without it produces a speculative engine that dies
  // mid-run ("missing target aux multi-tap") rather than at load. Found exactly
  // that way: the first DSpark e2e ran against classic-dense `Qwen3ForCausalLM`,
  // which has no tap, and the engine threw on the first propose. Default false;
  // the Qwen3.5/3.6 dense + MoE forwards override it.
  virtual bool supports_aux_multi_tap() const { return false; }

  // Retain the checkpoint's loaded `mtp.*` draft weights (the 15/19 BF16 tensors
  // LoadQwen3_5MTP produced) inside the concrete model so the draft can be built
  // on demand with a stable lifetime. Default: unsupported (throws), because a
  // non-MTP architecture has nowhere valid to put them.
  virtual void AttachMtpDraftWeights(Qwen3_5MTPWeights weights);

  // Construct the Qwen3_5MTPModel draft from the retained MTP weights + this
  // model's own target weights. Returns null when no MTP weights were attached
  // or the architecture is not an MTP target. The returned draft borrows this
  // model's weights/config, so it must not outlive this LoadedModel.
  virtual std::unique_ptr<Qwen3_5MTPModel> BuildMtpDraft(
      const HfConfig& config) const;

  // ── BACKEND-DISTRIBUTED-TP TP-W1: per-rank tensor-parallel handle ──────────
  // The optional TP group (comm + rank) this model instance runs under. BORROWED
  // (owned by the executor/runner for this instance's lifetime), set only when
  // tp_size>1; null on the single-GPU path so tp-aware layers take the
  // whole-tensor, byte-identical branch (tensor_parallel.h: null ⇒ tp_size()==1
  // ⇒ every collective a no-op, parallel_state.py:638 bypass). Consumed by
  // TP-W2's forward/loader plumbing; here it is the carrier only.
  const TensorParallel* tensor_parallel() const { return tensor_parallel_; }
  void set_tensor_parallel(const TensorParallel* tp) { tensor_parallel_ = tp; }

 protected:
  explicit LoadedModel(const ModelRegistration& registration)
      : registration_(registration) {}

 private:
  const ModelRegistration& registration_;
  const TensorParallel* tensor_parallel_ = nullptr;
};

// ── The type-erasure seam's OTHER half (#775) ───────────────────────────────
// Every registered `prepare`/`forward` is handed a type-erased `LoadedModel&`
// and has to open it as its own concrete model. Doing that with a bare
// `static_cast` down the hierarchy is a PROMISE, not a check: on an object
// whose dynamic type is not that model, every member call through the resulting
// reference is undefined behaviour, and the compiler is entitled to act on the
// promise. UBSan reports it as "member call on address ... which does not point
// to an object of type 'X'". It is invisible without a sanitizer whenever the
// entry point was going to refuse anyway, which is precisely the state a model
// still missing its weight loader is in.
//
// `ModelAs` is the checked form: it establishes the dynamic type first and
// refuses BY NAME when the caller paired one architecture's registration with
// another architecture's model. The cost is one `dynamic_cast` per forward
// step (the seam is entered once per `ModelRegistry::Forward`, not per layer),
// against a step that is milliseconds of GEMMs.
//
// `architecture` is the caller's OWN architecture name, so the refusal says
// which entry point refused rather than only that something did.
[[noreturn]] void RaiseModelTypeMismatch(std::string_view architecture,
                                         const LoadedModel& model);

template <typename Model>
Model& ModelAs(LoadedModel& model, std::string_view architecture) {
  Model* typed = dynamic_cast<Model*>(&model);
  if (typed == nullptr) {
    RaiseModelTypeMismatch(architecture, model);
  }
  return *typed;
}

template <typename Model>
const Model& ModelAs(const LoadedModel& model, std::string_view architecture) {
  const Model* typed = dynamic_cast<const Model*>(&model);
  if (typed == nullptr) {
    RaiseModelTypeMismatch(architecture, model);
  }
  return *typed;
}

// MM-ENGINE-FORWARD: one multimodal (vision/audio-language) forward step's
// vision-conditioned inputs, carried as an OPTIONAL sub-field of ModelForwardInput
// (below). Set (non-nullopt) ONLY by the runner mm-path when the request carries
// Request.mm_features; nullopt on every text step, so the registered TEXT forwards
// never read it and the shared runner path stays byte-identical (the field is
// additive and default-nullopt — text inertness is by construction).
//
// The handles are BORROWED (owned by the runner/driver for the forward's
// duration). The tower + `_merge_multimodal_embeddings` + MRoPE index math already
// ran on the host side (or the encoder runner) to fill these, so the registered
// forward is a pure per-step function of them:
//   * inputs_embeds_bf16 — the ALREADY-MERGED [num_tokens*hidden] host bf16 input
//     embeddings: embed(token_ids) with the projected vision features
//     masked-scattered into the placeholder rows. When set, the forward consumes
//     THESE instead of embedding token_ids.
//   * positions3 — the 3-D MRoPE positions, row-major [3, num_tokens]
//     (3*num_tokens int32), replacing the 1-D ModelForwardInput::positions for the
//     vision-language backbone.
//   * deepstack_bf16 — the [levels*num_tokens*hidden] host bf16 DeepStack
//     multiscale tensor added after decoder layers 0..levels-1 (EMPTY on decode
//     steps and on models without DeepStack, e.g. the 27B GDN-hybrid VL path).
//   * deepstack_levels — `levels` (0 ⇒ no DeepStack inject).
//   * ple_token_ids — CLAIM-GEMMA4-MM-E2E: the Gemma-4 Per-Layer-Embedding (PLE)
//     token ids [num_tokens], with the multimodal (image/audio) rows masked to 0
//     and the vocab_size_per_layer_input range mask applied — mirror of
//     gemma4_mm.py embed_input_ids (`is_multimodal → 0`, :1962-1969) +
//     gemma4.py get_per_layer_inputs (`id < vocab_size_per_layer_input ? id : 0`,
//     :857-863). The Gemma-4 registered mm forward looks up embed_tokens_per_layer
//     from THESE ids (NOT the merged embeds). nullptr for non-Gemma mm models
//     (Qwen3-VL never sets it) — additive + default-null. Gemma-4 also uses the
//     1-D ModelForwardInput::positions (NOT positions3) and no DeepStack.
struct MultiModalForwardInput {
  const std::vector<uint16_t>* inputs_embeds_bf16 = nullptr;
  const std::vector<int32_t>* positions3 = nullptr;
  const std::vector<uint16_t>* deepstack_bf16 = nullptr;
  int64_t deepstack_levels = 0;
  const std::vector<int32_t>* ple_token_ids = nullptr;
};

// One MRV2 forward invocation. References stay valid for the duration of the
// registered forward hook; model-specific decode-graph state lives in the
// concrete LoadedModel rather than leaking concrete weight types into runner.
struct ModelForwardInput {
  const std::vector<int32_t>& token_ids;
  const std::vector<int32_t>& positions;
  const v1::CommonAttentionMetadata& attn_meta;
  const v1::GDNAttentionMetadata& gdn_meta;
  std::vector<PagedKvCache>& attn_kv;
  std::vector<GdnStateCache>& gdn_state;
  const HfConfig& config;
  vt::Queue& queue;
  const std::vector<int32_t>& logits_indices;
  int num_reqs = 0;
  int64_t gdn_state_slots = 0;
  bool pure_decode = false;
  // SPEC-DSPARK W8 (#442): the configured speculation width, so the decode-graph
  // gate can mirror vLLM's UNIFORM-decode predicate instead of "query_len == 1".
  // Upstream's captured decode length is `1 + num_speculative_tokens`
  // (cudagraph_dispatcher.py:37), which is why its T=1+k speculative VERIFY is
  // graph-captured and ours was not. DEFAULT 0 => the predicate reduces exactly
  // to today's pure-decode shape, so every non-spec caller is byte-identical.
  int64_t num_speculative_tokens = 0;
  // ENG-CUDAGRAPH-BREAK W6 (#1374): THE GRAPH-ELIGIBILITY PREDICATE, moved off
  // `pure_decode` and onto the step's ACTUAL uniform query length.
  //
  // The runner computes it once per step through
  // `v1::ActualUniformDecodeQueryLen` (`v1/worker/gpu/cudagraph_dispatch.h`) and
  // every model reads the answer. 0 means "no captured decode graph in this tree
  // can serve this step" -- prefill, mixed, ragged, or uniform at a length above
  // the configured `1 + num_speculative_tokens`. 1 is exactly `pure_decode`.
  // A value ABOVE 1 is a speculative VERIFY step at its actual draft depth,
  // which is the population [#1020] named and which the two Qwen3.5 drivers
  // serve.
  //
  // WHY `pure_decode` SURVIVES BESIDE IT. Seven of the nine decode drivers
  // capture a query_len == 1 shape and nothing else, and widening them here
  // would admit steps no driver can serve -- the exact failure the spec's
  // `## Work breakdown` W6 says to avoid by ordering this stage last. They keep
  // reading `pure_decode`, which is provably NARROWER than this field, so the
  // widening is opt-in per driver rather than imposed on all nine at once.
  int64_t uniform_query_len = 0;
  bool gather_logits = true;
  // SPEC-MTP I5d-pre hidden-state tap. When non-null (only the spec verify
  // forward sets it, I5d), the Qwen3.5 dense/MoE forward routes to
  // Qwen3_5{,Dense}Model::ForwardDeviceTap, which returns byte-identical logits
  // AND moves the full [num_actual_tokens, H] post-final-norm hidden into
  // `*hidden_tap` for the MTP drafter. When null (every spec-off run) the
  // forward takes the current ModelRegistry::Forward path and is byte-identical.
  // Models other than Qwen3.5 ignore this field.
  Qwen3_5MTPHiddenStates* hidden_tap = nullptr;
  // SPEC-DFLASH D1 (DF-AUX-TAPS) multi-layer aux hidden taps. When non-null (only
  // the DFlash spec verify forward sets it, D4), the Qwen3.5 dense/MoE forward
  // routes to Qwen3_5{,Dense}Model::ForwardDeviceMultiTap, which returns byte-
  // identical logits AND captures (hidden + residual) at each configured
  // target_layer_id boundary into `aux_tap->tensor` = [T, H×taps] for the DFlash
  // drafter. When null (every non-DFlash run) the forward takes the current path
  // and is byte-identical. Mutually exclusive with hidden_tap. Models other than
  // Qwen3.5 ignore this field.
  Qwen3_5AuxTaps* aux_tap = nullptr;
  // MM-ENGINE-FORWARD: the optional vision-language per-step inputs. nullopt on
  // every TEXT step (text models never read it ⇒ byte-identical shared-path). Set
  // only by the runner mm-path when the request carries Request.mm_features, and
  // consumed by the registered multimodal forward (Qwen3-VL) which routes the
  // tower/merge/MRoPE/DeepStack-conditioned decode. Default-nullopt keeps every
  // existing (text) call site byte-identical by construction.
  std::optional<MultiModalForwardInput> mm = std::nullopt;
  // ENG-ASYNC-SCHED W4: when non-null, the [token_ids.size()] input ids already
  // live in THIS device buffer and `token_ids` is stale for decode rows. The
  // async runner's device combine spliced each decode row's sampled token here
  // on the main queue, so the host copy never saw it — which is the whole point,
  // since materializing it on the host is the synchronize W4 removes.
  //
  // A model that honors this embeds from the device pointer instead of uploading
  // `token_ids`; a model that ignores it is simply never given one (the runner
  // only sets it on the discrete-CUDA async path, which the Qwen3.5 gate vehicle
  // owns). Null on every other path, so every other forward is byte-identical.
  const int32_t* device_token_ids = nullptr;
};

using ModelConfigHook = void (*)(const HfConfig& config);
using ModelWeightLoader = std::unique_ptr<LoadedModel> (*)(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source);
using ModelPrepareFn = void (*)(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue);
using ModelForwardFn = ForwardLogits (*)(LoadedModel& model,
                                         const ModelForwardInput& input);
using KVCacheSpecBuilder = v1::KVCacheConfig (*)(const HfConfig& config,
                                                 int block_size,
                                                 int num_blocks);

// Per-family plug-in seam. load_weights contains that family's on-disk name map
// and construction path; forward remains type-erased over LoadedModel.
struct ModelFactory {
  ModelConfigHook parse_config = nullptr;
  ModelWeightLoader load_weights = nullptr;
  ModelPrepareFn prepare = nullptr;
  ModelForwardFn forward = nullptr;
  KVCacheSpecBuilder make_kv_cache = nullptr;
  // Preserves the already-gated per-arch scheduler default. This is execution
  // policy, not an upstream _ModelInfo capability.
  bool is_dense_model = false;
  // ROW 7 (kimi-linear.md §20.3): the loader wants a `ModelSource::load_queue`
  // selected BEFORE the weights load — the GB10 recipe (CUDA context first, then
  // per-tensor stage-and-release; Kimi-Linear's 91.5 GiB bf16-resident loader).
  // Default false: every existing arch's engine load path is byte-identical.
  bool stage_on_load = false;
  // ENG-WEIGHT-OFFLOAD: whether THIS model's loader asks
  // `WeightOffloader::ConsiderWeight` for each weight and honours the answer.
  //
  // THE DEFAULT IS FALSE, AND THAT IS THE MECHANISM. There is no single upload
  // helper in this tree: each model allocates its own device buffers, and
  // `load_stats::AddDeviceUpload` reaches only 9 call sites in 6 files, so
  // neither is a chokepoint that could enforce this. A model whose loader was
  // never wired would therefore accept a `cpu_offload_gb` and silently keep
  // every weight on the device, which is a memory bug with no error anywhere.
  //
  // Declaring the capability makes that case LOUD instead: the engine refuses a
  // configured offload against a model that does not claim support, naming the
  // architecture. A new model inherits false and is refused until someone wires
  // it, which is the same polarity as the `-Werror=switch` totality proof in
  // `gguf_keep_quant.cpp`, expressed at run time because there is no enum to
  // switch over.
  bool supports_weight_offload = false;
  // ENG-EXPERT-STREAM-DEVICE W0d (issue #1124): whether THIS model's forward
  // reads its routed-expert weights through the expert-stream slot seam
  // (`KqExpertSlice`), so the stacked `*_exps.weight` towers are served a slice
  // at a time out of the host slot store instead of being staged.
  //
  // THE DEFAULT IS FALSE FOR THE SAME REASON `supports_weight_offload`'s is, and
  // this one is load-bearing in the UNSAFE direction. The load-time fit bound
  // (`gguf_device_fit.h`) can drop a tensor set from what it charges the device,
  // and the loader identifies that set by NAME — `_exps.weight`, which is what a
  // llama.cpp MoE export writes for every MoE family it converts, not only the
  // ones this tree streams. `deepseek_v4_weights.cpp` and `laguna_weights.cpp`
  // both write that exact suffix, and neither model composes `RunMoeBlock`
  // (`deepseek_v2.cpp` says so at its head), so neither ever reaches
  // `KqExpertSlice`. Dropping their towers from the bound would remove a REFUSAL
  // THAT WAS CORRECT and put back the failure #1123 exists to prevent: a
  // 26-minute load and then `cudaMalloc: out of memory` on the first forward.
  //
  // WHY THE CAPABILITY AND NOT AN ARCHITECTURE LIST. The fact is a property of
  // the model's forward, and it lives beside the forward: `qwen3_5_moe.cpp` and
  // `qwen3_moe_registry.cpp` are the two translation units that route into
  // `RunMoeBlock`, and they are the two that set this. A list in the loader would
  // be a second description of the same fact, in a file that cannot see when the
  // first one changes — and a model whose forward stopped streaming would leave
  // the list saying it still does. Inheriting false is the safe answer: a new
  // architecture gets the whole bound and the #1123 refusal until somebody wires
  // the seam and says so here.
  bool streams_routed_experts = false;
};

struct ModelRegistration {
  std::string_view architecture;
  const ModelFactory* factory = nullptr;
  ModelInfo info;
};

struct UnsupportedModelInfo {
  std::string_view architecture;
  std::string_view detail;  // previous version or plugin URL
};

// Self-registration seam (mirrors REGISTER_VLLM_MODEL in
// vllm/model_executor/models/registry.py:682-693 assembling `_VLLM_MODELS`).
// Each architecture registers itself from its OWN translation unit via a static
// `ModelRegistrar`, exactly like the RegisterOp/RegisterBackend/RegisterPlatform
// static-init idiom (src/vt/ops.cpp, src/vt/backend.cpp,
// src/vllm/platforms/platform.cpp). Adding a model = a new TU with one
// REGISTER_VLLM_MODEL line — ZERO edit to a shared array.
//
// The passed ModelRegistration is copied into the process-global registry (its
// string_view/factory members point at TU-static data with static lifetime, so
// the copy stays valid). Registration order across TUs is unspecified under C++
// static init, so the registry imposes a stable canonical sort by architecture
// name on first query (see model_registry.cpp) — resolution semantics (first
// config-architecture match) are order-independent and unchanged.
void RegisterModel(const ModelRegistration& registration);

// Internal adapter helper: returns the registry entry for an implemented
// architecture (used by the synthetic in-memory Make/Borrow adapters below).
const ModelRegistration& RegistrationFor(std::string_view architecture);

// Static-init helper whose constructor performs the self-registration; used
// only through the REGISTER_VLLM_MODEL macro.
struct ModelRegistrar {
  explicit ModelRegistrar(const ModelRegistration& registration) {
    RegisterModel(registration);
  }
};

// Registers one architecture's factory from its own TU. Place at namespace
// scope inside `namespace vllm { ... }`; `unique_tag` is any TU-unique token.
#define REGISTER_VLLM_MODEL(unique_tag, architecture_name, factory_ref,   \
                            info_val)                                      \
  namespace {                                                             \
  const ::vllm::ModelRegistrar vllm_model_registrar_##unique_tag(         \
      ::vllm::ModelRegistration{(architecture_name), &(factory_ref),      \
                                (info_val)});                             \
  } /* namespace */

class ModelRegistry {
 public:
  // Mirrors get_supported_archs() and resolve_model_cls(): registrations are
  // declaration-ordered and Resolve returns the first architecture-list match.
  static std::span<const ModelRegistration> Registrations();
  static std::vector<std::string_view> SupportedArchs();
  static const ModelRegistration& Resolve(
      std::span<const std::string> architectures);
  static const ModelRegistration& Resolve(const HfConfig& config);

  // Mirrors _raise_for_unsupported exactly. The explicit-supported overload is
  // used by the subset-oracle parity test and covers the upstream
  // registered-but-inspection-failed branch.
  [[noreturn]] static void RaiseForUnsupported(
      std::span<const std::string> architectures);
  [[noreturn]] static void RaiseForUnsupported(
      std::span<const std::string> architectures,
      std::span<const std::string_view> supported_architectures);

  static std::span<const UnsupportedModelInfo> PreviouslySupportedModels();
  static std::span<const UnsupportedModelInfo> OutOfTreeSupportedModels();

  // Type-erased construction and live runner hooks.
  static std::unique_ptr<LoadedModel> Load(const HfConfig& config,
                                           const ModelSource& source);
  static void Prepare(LoadedModel& model, const HfConfig& config,
                      vt::Queue& queue);
  static ForwardLogits Forward(LoadedModel& model,
                               const ModelForwardInput& input);
  static v1::KVCacheConfig MakeKVCache(const LoadedModel& model,
                                       const HfConfig& config, int block_size,
                                       int num_blocks);
  static bool IsDenseModel(const LoadedModel& model);
};

// Compatibility adapters for synthetic in-memory Qwen tests and callers that
// already own weights. Production disk loading uses ModelRegistry::Load.
std::unique_ptr<LoadedModel> MakeQwen3_5MoeLoadedModel(
    Qwen3_5MoeWeights weights);
std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights);
std::unique_ptr<LoadedModel> BorrowQwen3_5MoeLoadedModel(
    const Qwen3_5MoeWeights& weights);
std::unique_ptr<LoadedModel> BorrowQwen3_5DenseLoadedModel(
    const Qwen3_5DenseWeights& weights);

}  // namespace vllm
