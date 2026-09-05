// dots3-note registry TU — the ADDITIVE self-registration seam (W1). Follows
// the deepseek_v4_registry.cpp / gemma4_registry.cpp seam exactly: a NEW
// translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to any
// shared array.
//
// Upstream registers ONE architecture string onto this package:
//   registry.py:381 (_MULTIMODAL_MODELS)
//     "Dots3NoteForCausalLM": ("vllm.models.dots3_note", "Dots3NoteForCausalLM")
// and its speculative head separately:
//   registry.py:670 (_SPECULATIVE_DECODING_MODELS)
//     "Dots3NoteMTPModel": ("vllm.models.dots3_note", "Dots3NoteMTP")
// Read at vLLM `origin/main` = `c205726108df54bb6fbf15b19e725a4a3add2b18`.
// `dots3_note` does NOT exist at our parity pin `555967922` — see
// `.agents/porting-inventory.md` §9 and `.agents/specs/dots3-note.md` §6.1.
//
// `Dots3NoteMTPModel` is deliberately NOT registered here. It stays INVENTORIED
// on `MODEL-SPEC-dots3-note-dots3-note-mtp` (W10 owns it): registering a
// speculator that cannot propose would make the engine accept a speculative
// config it then dies on mid-run, which is the failure mode #442 already found
// on another row. W1 does record what the checkpoint says about it — the nextn
// tail is EXACTLY ONE layer (`model.layers.46.*`), which is
// `num_nextn_predict_layers = 1` agreeing with `Dots3NoteConfig.__init__`.
//
// SCOPE HONESTY: registering this arch makes it RESOLVE and parse its config.
// It does NOT make it load and it does NOT make it forward — both refuse BY
// NAME, naming the brick that owes the work. That polarity is not incidental
// here: spec §6.4 records that NO oracle for this model runs on any hardware
// this project owns, so there is no downstream token gate that could catch a
// forward returning plausible garbage. Refusing is the only safe default.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"  // IsDots3NoteGguf
#include "vllm/model_executor/models/dense_attn_block.h"  // MakeTensor
#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::ApplyDeviceTokenIds
#include "vllm/model_executor/models/dots3_note_vision.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/multimodal/inputs.h"  // MultiModalFeatureSpec
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for dots3-note: text generation, NOT hybrid in this
// tree's sense (both attention classes are attention over a paged MLA cache;
// the sliding half is a window on the same cache, not a recurrent state).
//
// ─── `supports_multimodal` IS TRUE AGAIN, AND THE TRAIL IS THE RECORD ────────
// W1 set it TRUE because upstream registers this architecture in
// `_MULTIMODAL_MODELS` and `multimodal.py`::Dots3NoteForCausalLM
// .get_placeholder_str handles image, video AND audio. That is a true statement
// about UPSTREAM, and it was harmless while the released config was refused at
// its first MoE layer: nothing could load, so nothing could read the flag.
//
// W5 and W5c made the released config loadable, and at that moment the flag
// became a claim about THIS port that this port could not honour — no vision
// tower, no audio tower, no multimodal front end — so W5 set it FALSE.
//
// W6a ([#2512](https://github.com/mudler/vllm.cpp/issues/2512)) BACKS IT. The
// DENSE vision tower is on a SERVED request: `encode_mm` and `embed_mm` below
// are non-null, which is what makes `ModelRegistry::SupportsMmInputs` true and
// what turns on the runner's whole multimodal arm; and
// `mm_chat_dots3note.cpp` registers this architecture's own chat seam through
// `REGISTER_VLLM_MM_CHAT`, which is what makes the SERVER able to build image
// features for it. A registry entry is a support claim, and this one now has
// both halves behind it.
//
// W6b (#2613) then made the RELEASED `dots-studio/dots3-note-prev` load its
// whole 42-block vision tower, and W7a (#2703) added the AUDIO half: an
// `input_audio` part now reaches the 32-layer `dots` speech encoder through the
// same two hooks. The flag says this ARCHITECTURE has a served multimodal path,
// which is now true for two modalities; it has never meant that every
// checkpoint of it loads.
//
// W7 IS AUDIO AND W8 IS VIDEO, and this comment used to say the reverse. The
// loader has said so since W2 — `Dots3NoteDeferredTowers()` registers
// `audio_encoder.` against brick "W7" — and #2703 is titled "W7: the audio
// tower". VIDEO is refused by name in `EncodeMmDots3NoteForCausalLM` and owed
// to W8, and the chat seam declares a ceiling of one IMAGE and one AUDIO so a
// video part is refused at the ENTRYPOINT with upstream's own message.
//
// MEASURED, and still true: `supports_multimodal` has no production reader
// outside `model_registry.h` — every other occurrence is a registration writing
// it or a test reading it. What gates the capability is the two hooks, not the
// flag, which is why the scaffold gate asserts all three together.
inline constexpr ModelInfo kDots3NoteInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

class Dots3NoteLoadedModel final : public LoadedModel {
 public:
  Dots3NoteLoadedModel(const ModelRegistration& registration,
                       Dots3NoteWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const Dots3NoteWeights& weights() const { return weights_; }

 private:
  Dots3NoteWeights weights_;
};

std::unique_ptr<LoadedModel> LoadDots3NoteForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // The GGUF k-quant arm is OWED, not optional (AGENTS.md, porting-a-model.md
    // §2) — and for this row it is the only arm that could ever fit a host we
    // own (spec §6.2: 576.89 GB bf16 / 298.67 GB fp8, decimal GB, against a
    // 122 GiB ceiling), which is also the only route this row has to an
    // end-to-end run and to a quant-matched llama.cpp denominator.
    //
    // The TEXT lives in `Dots3NoteGgufRefusal` because the entrypoint's GGUF
    // architecture dispatch throws the SAME string strictly EARLIER (W9a,
    // #2882): a real `dots3note` file is refused at the door it actually
    // arrives at, and this guard still covers a direct `Kind::kGguf` caller.
    throw std::runtime_error(Dots3NoteGgufRefusal());
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Dots3NoteLoadedModel>(
      registration, LoadDots3NoteWeights(*source.safetensors, config));
}

void PrepareDots3NoteForCausalLM(LoadedModel& model, const HfConfig& config,
                                 vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardDots3NoteForCausalLM(LoadedModel& model,
                                          const ModelForwardInput& input) {
  // `ModelAs`, never a bare `static_cast`: opening a type-erased handle by
  // promise is undefined behaviour on any object that is not really this type,
  // and UBSan's vptr check reports it (#775, and the NemotronH repeat #730).
  // It matters MORE on a refusing forward than on a working one, because the
  // type confusion happens on the way to a throw that would have happened
  // anyway and is therefore invisible without a sanitizer.
  auto& d3 = ModelAs<Dots3NoteLoadedModel>(model, "Dots3NoteForCausalLM");
  // [[noreturn]] until W3-W10 land. The delegation is deliberate rather than an
  // inline throw: `check-runner-routing-consistency.py` classifies a registered
  // model from the ForwardDevice impl its hook delegates to, and a model with
  // no recognizable producer lands in the silently-exempt NONE bucket. This
  // shape reports REFUSE, and it is the signature W3 fills in.
  // `input.mm` (W6a, #2512). The runner sets it ONLY on a step in which some
  // request carries multimodal features, and `Dots3NoteModel::ForwardDevice`
  // reads `inputs_embeds` off it in place of the embedding lookup. A text step
  // passes nullptr and is byte-identical to every step this row has ever run.
  return Dots3NoteModel::ForwardDevice(
      input.token_ids, input.positions, input.attn_meta, input.attn_kv,
      d3.weights(), input.queue, input.logits_indices,
      input.mm.has_value() ? &input.mm.value() : nullptr);
}

// ─── W6a: the two ModelFactory multimodal hooks (#2512) ──────────────────────
//
// The split between them is upstream's own and is the runner's contract
// (`ENG-MM-INPUT-PIPELINE` P2, #2379). `encode_mm` is
// `_process_image_input` (`nvidia/multimodal.py:144-155` @ `9035151d6`): run
// the tower on ONE item and return its rows. `embed_mm` is
// `get_input_embeddings` + the masked scatter: the token lookup for the whole
// step, with the gathered encoder rows written over the placeholder positions.
// The runner owns the encoder CACHE and the gather between them; it never sees
// a tower.

// ─── W7a (#2703): the AUDIO arm of `encode_mm` ──────────────────────────────
//
// `_process_audio_input` (`nvidia/multimodal.py:156-170` @ `9035151d6`): run
// the tower on ONE item and return its rows. Split out of
// `EncodeMmDots3NoteForCausalLM` below rather than inlined as a branch, because
// the two towers share exactly one thing — the bf16 store at the end — and
// interleaving them would make each harder to read than either.
MmEncoderOutput EncodeAudioDots3Note(Dots3NoteLoadedModel& d3,
                                     const HfConfig& config, vt::Queue& queue,
                                     const multimodal::MultiModalFeatureSpec& item) {
  const Dots3NoteWeights& w = d3.weights();
  // THE REFUSAL CARRIES ITS REASON, same as the vision arm's.
  VT_CHECK(w.audio.present,
           "Dots3NoteForCausalLM encoder: this load carries no audio tower — " +
               (w.audio_refusal.empty()
                    ? std::string("the loader did not materialize one")
                    : w.audio_refusal) +
               ". See .agents/specs/dots3-note.md §4.14 and issue #2703.");
  VT_CHECK(item.audio_data != nullptr && !item.audio_data->empty(),
           "Dots3NoteForCausalLM encoder: the multimodal item carries no "
           "processed audio features (MultiModalFeatureSpec::audio_data).");

  const Dots3NoteAudioParams& a = w.audio_params;
  const int64_t width = a.adapter_out_dim;
  // DEFENCE IN DEPTH, and no longer the FIRST line: `Dots3NoteAudioRefusal`
  // makes this same comparison at INSTALL, because reaching it HERE throws
  // inside the engine's busy loop and stops `AsyncLLM` for the life of the
  // process. Keep the two predicates identical — a check added here and not
  // there re-opens that cascade through a narrower door.
  VT_CHECK(width == config.hidden_size,
           "Dots3NoteForCausalLM encoder: the audio adapter emits " +
               std::to_string(width) + "-wide rows but the text tower is " +
               std::to_string(config.hidden_size) +
               " wide (`whisper_adapter_out_dim`, nvidia/audio.py:33-35 @ "
               "9035151d6)");

  // The STACKED padded mels and the PER-CHUNK lengths, exactly as the processor
  // produced them (W7b, #2797). `chunk_num_samples[i]` and
  // `chunk_num_tokens[i]` are two different numbers and neither is derivable
  // from the other — see `Dots3NoteAudioForward`'s own note — and they are
  // per-chunk because upstream's `encode_waveform` carries them that way
  // (`audio_sample_lens` / `token_lens`, nvidia/audio.py:217-218 @ 9035151d6).
  // A one-chunk item takes exactly W7a's path through the same call.
  const multimodal::AudioKwargs& mel = *item.audio_data;
  VT_CHECK(!mel.chunk_num_tokens.empty(),
           "Dots3NoteForCausalLM encoder: the multimodal item carries mel "
           "features with no per-chunk lengths. `Dots3NoteAudioProcessor::"
           "ProcessWaveform` fills both vectors for every waveform it accepts.");
  const std::vector<float> tower = Dots3NoteAudioForwardChunks(
      mel.input_features, mel.chunk_num_samples, mel.chunk_num_tokens,
      /*hop_length=*/160, w.audio, a, vt::GetBackend(queue.device.type));

  const int64_t rows =
      width > 0 ? static_cast<int64_t>(tower.size()) / width : 0;
  VT_CHECK(rows > 0 && rows * width == static_cast<int64_t>(tower.size()),
           "Dots3NoteForCausalLM encoder: the audio tower produced " +
               std::to_string(tower.size()) +
               " floats, which is not a whole number of " +
               std::to_string(width) + "-wide rows");
  VT_CHECK(rows == static_cast<int64_t>(item.length),
           "Dots3NoteForCausalLM encoder: the audio tower produced " +
               std::to_string(rows) + " embedding rows for a placeholder span "
               "of " + std::to_string(item.length) +
               " tokens. The processor's `ceil(num_samples / token_stride)` and "
               "the tower's row count disagree, and a masked scatter would then "
               "splice the wrong rows into the prompt.");

  // Stored in the MODEL dtype, for the reason the vision arm records: every
  // tensor of this tower is bf16 on disk and every GEMM above ran in it, so an
  // f32 store would double the encoder cache for values that carry no extra
  // information (porting.md's memory-format rule).
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  std::vector<uint16_t> bits(tower.size());
  for (size_t i = 0; i < tower.size(); ++i) bits[i] = vt::F32ToBF16(tower[i]);
  const size_t bytes = bits.size() * vt::SizeOf(vt::DType::kBF16);
  void* p = backend.Alloc(bytes);
  std::shared_ptr<void> storage(p, [&backend](void* qq) { backend.Free(qq); });
  backend.Copy(queue, p, bits.data(), bytes);
  backend.Synchronize(queue);
  MmEncoderOutput out;
  out.storage = std::move(storage);
  out.embeds = dense_attn::MakeTensor(p, vt::DType::kBF16, queue.device,
                                      {rows, width});
  return out;
}

MmEncoderOutput EncodeMmDots3NoteForCausalLM(
    LoadedModel& model, const HfConfig& config, vt::Queue& queue,
    const multimodal::MultiModalFeatureSpec& item) {
  auto& d3 = ModelAs<Dots3NoteLoadedModel>(model, "Dots3NoteForCausalLM");
  const Dots3NoteWeights& w = d3.weights();

  // TWO MODALITIES REACH A TOWER, and the third is refused BY NAME with the
  // brick that owes it.
  //
  // W7 IS AUDIO AND W8 IS VIDEO. This message used to say the opposite, against
  // the loader's own `Dots3NoteDeferredTowers()` table and against #2703's own
  // title, so an operator who sent audio to a checkpoint was told to wait for
  // the wrong brick.
  VT_CHECK(item.modality == "image" || item.modality == "audio",
           "Dots3NoteForCausalLM encoder: modality '" + item.modality +
               "' is not ported. IMAGE is (W6a/W6b/W6c) and AUDIO is (W7a); "
               "VIDEO needs the multi-frame cu_seqlens builder, the frame "
               "sampler and the interleaved image/audio emission order "
               "(nvidia/multimodal.py:172-223 @ 9035151d6) and is W8. Refused "
               "by name rather than served from another modality's path. See "
               "issues #2512 and #2703.");

  if (item.modality == "audio") return EncodeAudioDots3Note(d3, config, queue, item);

  // THE REFUSAL CARRIES ITS REASON. `vision_refusal` is the message
  // `Dots3NoteVisionRefusal` produced at load; for a checkpoint whose tower is
  // owed it names the first unrepresentable feature and the brick that owes it.
  // Reporting only "no tower" would tell an operator nothing they could act on.
  VT_CHECK(w.vision.present,
           "Dots3NoteForCausalLM encoder: this load carries no vision tower — " +
               (w.vision_refusal.empty()
                    ? std::string("the loader did not materialize one")
                    : w.vision_refusal) +
               ". See .agents/specs/dots3-note.md §4.11 and issue #2512.");
  VT_CHECK(item.data != nullptr && !item.data->empty(),
           "Dots3NoteForCausalLM encoder: the multimodal item carries no "
           "processed image features (MultiModalFeatureSpec::data).");

  const Dots3NoteVisionParams& v = w.vision_params;
  const int64_t width = v.adapter_out_dim;
  // The tower lands in the TEXT hidden space, and a checkpoint whose adapter
  // does not is one whose rows cannot be scattered into the prompt at all.
  //
  // DEFENCE IN DEPTH, and no longer the FIRST line. `Dots3NoteVisionRefusal`
  // now makes this same comparison, and the one on the merge sizes below it, at
  // INSTALL — because reaching either of them here throws inside the engine's
  // busy loop, which stops `AsyncLLM` for the life of the process. Keep the two
  // predicates identical: a check added here and not there re-opens that
  // cascade through a narrower door (fresh review of #2523).
  VT_CHECK(width == config.hidden_size,
           "Dots3NoteForCausalLM encoder: the vision adapter emits " +
               std::to_string(width) + "-wide rows but the text tower is " +
               std::to_string(config.hidden_size) +
               " wide (`adapter_out_dim`, vision.py:476, :485 @ 9035151d6)");

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  // THE TOWER. This call is the point of the brick: before it, nothing on this
  // row ran anything on a served request.
  const std::vector<float> tower = Dots3NoteVisionForward(
      item.data->pixel_values_bf16, item.data->image_grid_thw, w.vision, v,
      backend);
  const int64_t rows =
      width > 0 ? static_cast<int64_t>(tower.size()) / width : 0;
  VT_CHECK(rows > 0 && rows * width == static_cast<int64_t>(tower.size()),
           "Dots3NoteForCausalLM encoder: the tower produced " +
               std::to_string(tower.size()) +
               " floats, which is not a whole number of " +
               std::to_string(width) + "-wide rows");
  VT_CHECK(rows == static_cast<int64_t>(item.length),
           "Dots3NoteForCausalLM encoder: the tower produced " +
               std::to_string(rows) + " embedding rows for a placeholder span "
               "of " + std::to_string(item.length) +
               " tokens. The processor's placeholder expansion "
               "(`prod(grid) // merge**2`) and the tower's adapter merge "
               "disagree, and a masked scatter would then splice the wrong rows "
               "into the prompt.");

  // Stored in the MODEL dtype. bf16 is not a compression here: the whole dense
  // vision arm is bf16 on disk and every GEMM above ran in it, so a f32 store
  // would double the encoder cache for values that carry no extra information
  // (porting.md's memory-format rule, pointed the other way).
  std::vector<uint16_t> bits(tower.size());
  for (size_t i = 0; i < tower.size(); ++i) bits[i] = vt::F32ToBF16(tower[i]);
  const size_t bytes = bits.size() * vt::SizeOf(vt::DType::kBF16);
  void* p = backend.Alloc(bytes);
  std::shared_ptr<void> storage(p, [&backend](void* q) { backend.Free(q); });
  backend.Copy(queue, p, bits.data(), bytes);
  backend.Synchronize(queue);
  MmEncoderOutput out;
  out.storage = std::move(storage);
  out.embeds = dense_attn::MakeTensor(p, vt::DType::kBF16, queue.device,
                                      {rows, width});
  return out;
}

MmForwardBuffers EmbedMmDots3NoteForCausalLM(LoadedModel& model,
                                             const HfConfig& config,
                                             vt::Queue& queue,
                                             const MmEmbedInputs& inputs) {
  auto& d3 = ModelAs<Dots3NoteLoadedModel>(model, "Dots3NoteForCausalLM");
  const Dots3NoteWeights& w = d3.weights();
  VT_CHECK(w.materialized && w.device.present,
           "Dots3NoteForCausalLM embed: the language tower was not "
           "materialized, so there is no embedding table to look tokens up in.");
  VT_CHECK(inputs.token_ids != nullptr && inputs.is_mm_embed != nullptr &&
               inputs.mm_embeds != nullptr,
           "Dots3NoteForCausalLM embed: the runner passed a null MmEmbedInputs "
           "channel");
  const std::vector<int32_t>& token_ids = *inputs.token_ids;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  VT_CHECK(T > 0, "Dots3NoteForCausalLM embed: empty step");
  VT_CHECK(static_cast<int64_t>(inputs.is_mm_embed->size()) == T,
           "Dots3NoteForCausalLM embed: is_mm_embed has " +
               std::to_string(inputs.is_mm_embed->size()) + " entries for " +
               std::to_string(T) + " tokens");

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  dense_attn::Dev d{backend, queue};

  // `embed_input_ids`, the token half: the plain lookup, identical to the one
  // the text path runs.
  std::vector<uint16_t> merged(static_cast<size_t>(T * H));
  {
    dense_attn::DBuf ids(d, vt::DType::kI32, {T}, token_ids.data());
    // ENG-MM-EMBED-DEVICE-IDS (#2730): SPLICE the runner's device identifiers
    // over the host upload just made, before the gather reads it. The host
    // vector is deliberately stale for decode rows -- the combine wrote each
    // sampled token into `MmEmbedInputs::device_token_ids` on the main queue and
    // never wrote it back -- so without this the decode rows of an image request
    // embed token id 0. The `Copy` is enqueued on this same queue, which is why
    // it is ordered AFTER that combine rather than racing it, and it is the
    // SHARED splice the text device arm uses rather than a second copy of it.
    // Null on every non-mirror step, where it writes nothing and this hook is
    // byte-identical to its pre-#2730 self.
    detail::ApplyDeviceTokenIds(
        d.b, d.q, ids.ptr(), T,
        detail::DeviceTokenIds{inputs.device_token_ids, T}, "dots3-note mm embed");
    dense_attn::DBuf emb(d, vt::DType::kBF16, {T, H});
    vt::Tensor table = dense_attn::ResidentWeight(
        d, w.device.embed_tokens, {config.vocab_size, H});
    vt::Embedding(d.q, emb.t(), table, ids.t());
    emb.Download(d, merged.data());
  }

  // The gathered encoder rows, concatenated in mask order.
  int64_t n_rows = 0;
  for (const vt::Tensor& slice : *inputs.mm_embeds) {
    VT_CHECK(slice.rank == 2 && slice.shape[1] == H,
             "Dots3NoteForCausalLM embed: a gathered encoder slice is " +
                 std::to_string(slice.shape[1]) + " wide, expected " +
                 std::to_string(H));
    VT_CHECK(slice.dtype == vt::DType::kBF16,
             "Dots3NoteForCausalLM embed: a gathered encoder slice is not "
             "BF16, so the scatter below would write the wrong bytes.");
    n_rows += slice.shape[0];
  }
  int64_t n_masked = 0;
  for (int64_t t = 0; t < T; ++t)
    if ((*inputs.is_mm_embed)[static_cast<size_t>(t)] != 0) ++n_masked;
  VT_CHECK(n_rows == n_masked,
           "Dots3NoteForCausalLM embed: " + std::to_string(n_rows) +
               " gathered encoder rows for " + std::to_string(n_masked) +
               " masked placeholder positions. A masked scatter that does not "
               "balance splices vision features onto text rows.");

  // `merge_multimodal_embeddings`, the masked scatter. It is a pure COPY of
  // bf16 rows over bf16 rows: no arithmetic, so nothing here rounds and the
  // encoder's own output reaches the residual stream bit-for-bit. Qwen3-VL's
  // hook beside this one goes through f32 because its DeepStack split needs to;
  // this model has no DeepStack and pays nothing.
  if (n_rows > 0) {
    std::vector<uint16_t> gathered(static_cast<size_t>(n_rows * H));
    size_t offset = 0;
    for (const vt::Tensor& slice : *inputs.mm_embeds) {
      const size_t n = static_cast<size_t>(slice.shape[0] * H);
      backend.Copy(queue, gathered.data() + offset, slice.data,
                   n * vt::SizeOf(vt::DType::kBF16));
      offset += n;
    }
    backend.Synchronize(queue);
    int64_t r = 0;
    for (int64_t t = 0; t < T; ++t) {
      if ((*inputs.is_mm_embed)[static_cast<size_t>(t)] == 0) continue;
      std::copy(gathered.begin() + static_cast<ptrdiff_t>(r * H),
                gathered.begin() + static_cast<ptrdiff_t>((r + 1) * H),
                merged.begin() + static_cast<ptrdiff_t>(t * H));
      ++r;
    }
  }

  MmForwardBuffers out;
  const size_t bytes = merged.size() * vt::SizeOf(vt::DType::kBF16);
  void* p = backend.Alloc(bytes);
  out.storage.emplace_back(p, [&backend](void* q) { backend.Free(q); });
  backend.Copy(queue, p, merged.data(), bytes);
  out.mm.inputs_embeds =
      dense_attn::MakeTensor(p, vt::DType::kBF16, queue.device, {T, H});
  // NO `positions3`, deliberately. dots3-note is not an M-RoPE model
  // (`nvidia/multimodal.py:49` @ `9035151d6`: `SupportsMultiModal, SupportsPP`
  // and NOT `SupportsMRoPE`), so `mrope_prompt_positions` is null on the
  // factory, the runner leaves `MmEmbedInputs::mrope_positions` empty, and the
  // forward reads the ordinary 1-D `ModelForwardInput::positions`. Filling a
  // 3-D field the forward never reads would be a claim about this model that
  // upstream does not make.
  backend.Synchronize(queue);
  return out;
}

const ModelFactory kDots3NoteFactory{
    .parse_config = &ParseDots3NoteConfig,
    .load_weights = &LoadDots3NoteForCausalLM,
    .prepare = &PrepareDots3NoteForCausalLM,
    .forward = &ForwardDots3NoteForCausalLM,
    .make_kv_cache = &MakeDots3NoteKVCache,
    // W6a (#2512). Setting these two is what makes
    // `ModelRegistry::SupportsMmInputs` true for this architecture, and the
    // runner's whole multimodal arm hangs on that predicate — which is DERIVED
    // from these pointers and never stored. `mrope_prompt_positions` stays
    // NULL: upstream does not declare `SupportsMRoPE` for this model.
    .encode_mm = &EncodeMmDots3NoteForCausalLM,
    .embed_mm = &EmbedMmDots3NoteForCausalLM,
    .is_dense_model = false,
    // ENG-MM-EMBED-DEVICE-IDS (#2730): the HOOK bit, and DELIBERATELY NOT the
    // forward's beside it. `EmbedMmDots3NoteForCausalLM` splices
    // `MmEmbedInputs::device_token_ids` over the buffer it gathers from, so the
    // merged embeds are built from the identifiers the combine actually wrote.
    // `ForwardDots3NoteForCausalLM` does not: it reaches
    // `Dots3NoteModel::ForwardDevice(input.token_ids, ...)`, and the runner takes
    // the multimodal branch when ANY request in the batch carries multimodal
    // items -- so a batch of TEXT-ONLY requests to this architecture takes the
    // text arm and embeds the stale host vector. It is class (b) of
    // `.agents/specs/eng-async-device-ids-refusal.md` and stays refused by
    // `ModelRegistry::Forward` until the row that ports that arm lands
    // ([#2732](https://github.com/mudler/vllm.cpp/issues/2732)). One bit could
    // not express this pair, which is why there are two.
    .embed_mm_consumes_device_token_ids = true,
};

}  // namespace

// The GGUF-side half of the arch entry points. Kept in THIS TU, next to the
// factory guard that throws the same string, so the refusal has one owner and
// the entrypoint's dispatch borrows it instead of restating it.
bool IsDots3NoteGguf(const GgufFile& gguf) {
  const GgufValue* arch = gguf.FindKv("general.architecture");
  if (arch == nullptr || arch->TypeId() != kGgufString) return false;
  return std::get<std::string>(arch->v) == kDots3NoteGgufArch;
}

// WHAT THIS SENTENCE HAD TO STOP SAYING. Until W9a (#2882) this message told
// the operator that llama.cpp has no `dots3_note` architecture and that there
// is therefore no upstream converter to reuse. Both halves are false, and this
// is PRODUCT OUTPUT rather than a record: it is what a user reads on stderr.
// Verified in a `ggml-org/llama.cpp` clone at `origin/master` = `0ef4d560e`
// (2026-09-04): `LLM_ARCH_DOTS3NOTE -> "dots3note"` at `src/llama-arch.cpp:114`,
// merged by `5a32f7b66ef6cfb3e60deea26e3454cc6ad3438c` ("model: add
// dots3-note", #27060, 2026-08-21, +1412/-9 over 20 files, converter
// `conversion/dots3.py`) and `54ee5ee643f29abba6852903ddfdb688c2361b5b`
// ("mtmd: support dots3-note vision+audio", #27524, 2026-08-22), both ancestors
// of `master`.
//
// It also does NOT oversell the new position. Whether this build can READ such
// a file is W9b/W9c's question and the answer today is no — llama.cpp splits
// our fused `kv_b_proj` into `attn_k_b`/`attn_v_b`, among other deltas — so the
// last sentence says exactly that. A refusal that overstates is the same class
// of defect as one that understates.
std::string Dots3NoteGgufRefusal() {
  return
      "Model architecture Dots3NoteForCausalLM does not support GGUF weights "
      "yet: the GGUF k-quant arm is OWED to W9 -- both a loader arm and, for "
      "an artifact that needs one, a converter. Row "
      "MODEL-MM-dots3-note-dots3-note-for-causal-lm, spec "
      ".agents/specs/dots3-note.md section 4.19. llama.cpp DOES define this "
      "architecture -- LLM_ARCH_DOTS3NOTE -> \"dots3note\" in "
      "ggml-org/llama.cpp src/llama-arch.cpp, merged as "
      "5a32f7b66ef6cfb3e60deea26e3454cc6ad3438c (\"model: add dots3-note\", "
      "2026-08-21) and 54ee5ee643f29abba6852903ddfdb688c2361b5b (\"mtmd: "
      "support dots3-note vision+audio\", 2026-08-22), and published dots3note "
      "GGUF artifacts exist -- so a file reaching this refusal is a real one. "
      "This build cannot read it yet.";
}

REGISTER_VLLM_MODEL(dots3_note, "Dots3NoteForCausalLM", kDots3NoteFactory,
                    kDots3NoteInfo)

}  // namespace vllm
