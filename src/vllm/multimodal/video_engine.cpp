// The generalized video seam's registry — implementation. See
// include/vllm/multimodal/video_engine.h for the contract.
//
// This TU knows about NO family. It holds the process-global family table, the
// checkpoint-name reader every detector inspects, and the resolution rules; the
// families themselves live in their own TUs and self-register. That is what
// makes "adding a family is an additive file" true rather than aspirational.
#include "vllm/multimodal/video_engine.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/video_api.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace vllm::multimodal {
namespace {

// Process-global registry, populated at static init by each family's
// REGISTER_VLLM_VIDEO_FAMILY registrar. Meyers singleton: the vector is
// constructed on the first RegisterVideoFamily call, safely before any
// registrar body runs.
//
// TWO INVARIANTS, both established by RegisterVideoFamily and therefore true
// after EVERY registration rather than after the first query: the table is
// sorted by name, and no name appears twice. "After every registration" is the
// load-bearing half. `RegisterVideoFamily` is a public header function, so a
// caller may register long after main started — canonicalizing once, on first
// query, would silently append everything that arrived afterwards in whatever
// order it arrived, and any adjacency-based reasoning downstream would then be
// reasoning about a list that is no longer sorted.
std::vector<VideoFamilyRegistration>& RegistryStorage() {
  static std::vector<VideoFamilyRegistration> storage;
  return storage;
}

// The registry as every reader sees it. C++ does not order static init across
// TUs, but arrival order is invisible here because the table is kept sorted at
// insertion. Ordering can only change how a refusal READS, never which family
// loads: resolution is order-independent by construction, since exactly one
// detector may claim a checkpoint and exactly one entry may carry a name.
const std::vector<VideoFamilyRegistration>& OrderedRegistry() { return RegistryStorage(); }

// A caller's UTF-8 path as a native filesystem path. Byte-for-byte the spelling
// in v1/kv_offload/fs_io.cpp:31, and the same job the loader lane's Utf8Path
// does (gguf_reader.cpp:22, safetensors_reader.cpp:29) — no shared helper
// exists to call, because each is file-local to its own TU. The u8string step
// is not decoration: on Windows a narrow std::string handed to
// std::filesystem::path is interpreted in the ACTIVE CODE PAGE, so every
// non-ASCII checkpoint path silently resolves to the wrong file (or to none).
// Saying char8_t makes the UTF-8 explicit and the conversion to UTF-16 exact.
std::filesystem::path NativePath(const std::string& utf8) {
#if defined(_WIN32)
  const std::u8string value(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size());
  return std::filesystem::path(value);
#else
  return std::filesystem::path(utf8);
#endif
}

// BOTH probes take the std::error_code overloads, and that is load-bearing
// rather than stylistic. The `::stat` calls these replaced reported an
// uninspectable path — ENAMETOOLONG, ELOOP, EACCES on a parent — by returning
// -1, which arrived here as a plain `false` and became the registry's ordinary
// "no such file or directory" refusal. The THROWING
// std::filesystem::exists(p) / is_directory(p) overloads raise
// filesystem_error for exactly those cases instead, which would escape
// ReadVideoCheckpointTensorNames — whose header contract is to return false
// with *why set — and, through DescribeCheckpoint, escape LoadVideoEngine in
// place of the refusal that names the registered families. Returning false on
// error keeps the POSIX behaviour these calls had. Issue #664.
bool IsDir(const std::string& path) {
  std::error_code error;
  const bool result = std::filesystem::is_directory(NativePath(path), error);
  return !error && result;
}

bool Exists(const std::string& path) {
  std::error_code error;
  const bool result = std::filesystem::exists(NativePath(path), error);
  return !error && result;
}

std::string StripTrailingSlash(const std::string& dir) {
  std::string out = dir;
  while (out.size() > 1 && out.back() == '/') out.pop_back();
  return out;
}

// The two index spellings a multi-shard release uses — the SAME pair the H3
// sharded loader accepts (minimax_h3_sharded.cpp kIndexNames), so a directory
// this reader can enumerate is exactly a directory a family loader can open.
const char* const kIndexNames[] = {
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
};

// Tensor names from a shard index's weight_map. NOTHING is discovered by
// scanning the directory: the index is what maps a tensor to a shard, and
// guessing by filename is how a stale or partial shard set gets loaded.
bool ShardIndexNames(const std::string& dir, std::vector<std::string>* out, std::string* why) {
  for (const char* name : kIndexNames) {
    const std::string candidate = StripTrailingSlash(dir) + "/" + name;
    std::ifstream in(candidate);
    if (!in) continue;
    try {
      nlohmann::json j;
      in >> j;
      const auto map = j.find("weight_map");
      if (map == j.end() || !map->is_object()) {
        *why = "'" + candidate + "' holds no weight_map object";
        return false;
      }
      out->clear();
      for (auto it = map->begin(); it != map->end(); ++it) out->push_back(it.key());
      return true;
    } catch (const std::exception& e) {
      *why = std::string("'") + candidate + "' is not readable JSON: " + e.what();
      return false;
    }
  }
  // Inherited from the ROW 2 guard, and generic: a DiT checkpoint set is a GGUF
  // file, a safetensors file, or a shard directory with an index. A directory
  // with neither index is, overwhelmingly, someone pointing a video load at a
  // text-model directory — so say which entry point that belongs to instead of
  // only saying what is missing.
  *why = "the directory holds neither model.safetensors.index.json nor "
         "diffusion_pytorch_model.safetensors.index.json, so it is not a DiT checkpoint set "
         "— a text-generation model directory loads through vllm_engine_load instead";
  return false;
}

// A file's leading 4 bytes, or "" when it cannot be read. The artifact FORMAT is
// decided by this magic, never by the extension: which suffix a repackaged
// checkpoint carries is the repackager's choice, and a detector that trusts it
// is trusting a filename.
std::string LeadingMagic(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::string();
  char buf[4] = {0, 0, 0, 0};
  in.read(buf, sizeof(buf));
  if (!in) return std::string();
  return std::string(buf, sizeof(buf));
}

}  // namespace

std::string VideoExtra(const std::map<std::string, std::string>& extras, const std::string& key,
                       const std::string& fallback) {
  const auto it = extras.find(key);
  return it == extras.end() ? fallback : it->second;
}

bool ReadVideoCheckpointTensorNames(const std::string& path, std::vector<std::string>* out,
                                    std::string* why) {
  std::string ignored;
  if (why == nullptr) why = &ignored;
  out->clear();
  if (path.empty()) {
    *why = "no dit_path was supplied";
    return false;
  }
  if (!Exists(path)) {
    *why = "no such file or directory";
    return false;
  }
  if (IsDir(path)) return ShardIndexNames(path, out, why);
  try {
    if (LeadingMagic(path) == "GGUF") {
      const GgufFile f = GgufFile::Open(path);
      for (const GgufTensorInfo& t : f.Tensors()) out->push_back(t.name);
      return true;
    }
    const SafetensorsFile f = SafetensorsFile::Open(path);
    *out = f.Names();
    return true;
  } catch (const std::exception& e) {
    out->clear();
    *why = std::string("it reads as neither GGUF nor safetensors: ") + e.what();
    return false;
  }
}

void RegisterVideoFamily(VideoFamilyRegistration registration) {
  if (registration.name.empty()) {
    throw std::runtime_error("video engine: a family must register under a non-empty name");
  }
  if (!registration.detect || !registration.load) {
    throw std::runtime_error("video engine: family '" + registration.name +
                             "' must register both a detector and a loader");
  }
  std::vector<VideoFamilyRegistration>& storage = RegistryStorage();
  const auto at = std::lower_bound(
      storage.begin(), storage.end(), registration.name,
      [](const VideoFamilyRegistration& r, const std::string& name) { return r.name < name; });
  // A DUPLICATE NAME IS NOT A LISTING BLEMISH, IT IS A SILENT MIS-LOAD. Two
  // registrations under one name defeat every guard downstream at once: the
  // listing a refusal prints shows one family, DetectVideoFamilies returns the
  // same name twice so the SEVERAL-claimants refusal never fires, and
  // LoadVideoEngine takes the FIRST matching entry — which makes the choice of
  // loader a function of static-init and link order. Refuse here, at the one
  // point where the collision is still nameable.
  //
  // Registrars run at static init, so this throw terminates the process rather
  // than unwinding into anything that could handle it. That is the intended
  // outcome and the same one the empty-name and missing-callback refusals above
  // already produce: two families claiming one name is a BUILD defect, and a
  // process that dies naming it is strictly better than one that renders noise.
  if (at != storage.end() && at->name == registration.name) {
    throw std::runtime_error(
        "video engine: family '" + registration.name +
        "' is already registered — two families cannot share one name, because resolution would "
        "then pick between them by link order and a checkpoint handed to the wrong family does "
        "not fail, it renders noise");
  }
  storage.insert(at, std::move(registration));
}

std::vector<std::string> RegisteredVideoFamilies() {
  std::vector<std::string> names;
  for (const VideoFamilyRegistration& r : OrderedRegistry()) names.push_back(r.name);
  // NOT de-duplicated. RegisterVideoFamily refuses a name already present, so a
  // duplicate cannot reach the table; the std::unique this replaces was not
  // protecting the listing but HIDING that violation — it collapsed the printed
  // names while leaving two entries in the registry for resolution to pick
  // between. The assertion states the invariant instead of papering over it.
  assert(std::is_sorted(names.begin(), names.end()));
  assert(std::adjacent_find(names.begin(), names.end()) == names.end());
  return names;
}

namespace {

// "[minimax-h3, ltx-2.5]" — the listing every refusal ends with, so the reader
// learns what IS available from the same message that says no.
std::string FamilyList() {
  const std::vector<std::string> names = RegisteredVideoFamilies();
  std::string out = "[";
  for (size_t i = 0; i < names.size(); ++i) out += (i == 0 ? "" : ", ") + names[i];
  return out + "]";
}

// What we actually saw at dit_path — the evidence half of a refusal. A caller
// staring at "could not resolve a family" needs to know whether the file was
// unreadable, empty, or simply carried names nobody claims.
std::string DescribeCheckpoint(const std::string& path) {
  std::vector<std::string> names;
  std::string why;
  if (!ReadVideoCheckpointTensorNames(path, &names, &why)) return why;
  std::ostringstream out;
  out << "it declares " << names.size() << " tensors";
  const size_t shown = names.size() < 3 ? names.size() : 3;
  if (shown > 0) {
    out << " (";
    for (size_t i = 0; i < shown; ++i) out << (i == 0 ? "" : ", ") << names[i];
    if (names.size() > shown) out << ", ...";
    out << ")";
  }
  out << ", and no registered family claims them";
  return out.str();
}

}  // namespace

std::vector<std::string> DetectVideoFamilies(const VideoModelParams& params) {
  std::vector<std::string> claimants;
  for (const VideoFamilyRegistration& r : OrderedRegistry()) {
    // A detector is documented as non-throwing, but a registry that ABORTS the
    // whole resolution because one family's detector threw would turn a
    // third-party bug into "no families exist". Treat a throw as "did not
    // claim" and let the refusal below report what the others found.
    bool claimed = false;
    try {
      claimed = r.detect(params);
    } catch (const std::exception&) {
      claimed = false;
    }
    if (claimed) claimants.push_back(r.name);
  }
  // NOT de-duplicated, for the same reason: registry names are unique, so two
  // claimants ARE two families and the caller must see both. A std::unique here
  // would merge a duplicate-name pair back into a single claimant and let the
  // SEVERAL refusal fall through into a load.
  assert(std::adjacent_find(claimants.begin(), claimants.end()) == claimants.end());
  return claimants;
}

std::unique_ptr<VideoEngine> LoadVideoEngine(const VideoModelParams& params) {
  if (!params.family.empty()) {
    for (const VideoFamilyRegistration& r : OrderedRegistry()) {
      if (r.name == params.family) return r.load(params);
    }
    throw std::runtime_error("video engine: no family named '" + params.family +
                             "' is registered. Registered families: " + FamilyList());
  }

  const std::vector<std::string> claimants = DetectVideoFamilies(params);
  if (claimants.size() == 1) {
    for (const VideoFamilyRegistration& r : OrderedRegistry()) {
      if (r.name == claimants[0]) return r.load(params);
    }
  }
  if (claimants.empty()) {
    // "Declare the family explicitly" is the right advice for a checkpoint no
    // detector claimed. It is a DEAD END for a caller who supplied no
    // checkpoint at all: the declared loader would still have nothing to open,
    // so following it produces a second, more confusing refusal. Name the
    // missing input instead, and say what would satisfy it.
    if (params.dit_path.empty()) {
      throw std::runtime_error(
          "video engine: no dit_path was supplied, so there is no checkpoint to determine a "
          "model family from. Supply dit_path — the denoiser artifact: a GGUF file, a "
          "safetensors file, or a multi-shard directory holding its index. Registered "
          "families: " +
          FamilyList());
    }
    throw std::runtime_error(
        "video engine: cannot determine the model family of dit_path '" + params.dit_path + "' — " +
        DescribeCheckpoint(params.dit_path) + ". Registered families: " + FamilyList() +
        ". Declare the family explicitly rather than letting the loader guess — a checkpoint "
        "handed to the wrong family does not fail, it renders noise");
  }
  std::string matched = "[";
  for (size_t i = 0; i < claimants.size(); ++i) matched += (i == 0 ? "" : ", ") + claimants[i];
  matched += "]";
  throw std::runtime_error(
      "video engine: dit_path '" + params.dit_path + "' was claimed by SEVERAL families " +
      matched +
      ", so at least one detector is too loose. Refusing rather than picking one; declare the "
      "family explicitly to load it meanwhile");
}

// The /v1/videos request mapping, lifted VERBATIM out of the H3 seam at L1
// (which now delegates here) because nothing in it was H3-specific: it is the
// OpenAI video request shape, and every family serves the same endpoint.
VideoGenParams VideoGenParamsFromRequest(const ::vllm::openai::VideoRequest& request,
                                         const std::string& output_dir) {
  VideoGenParams gen;
  gen.prompt = request.prompt;
  gen.task = request.task;
  gen.duration_seconds = request.duration_seconds;
  gen.num_frames = request.num_frames;
  gen.height = request.height;
  gen.width = request.width;
  gen.steps = request.num_inference_steps;
  gen.flow_shift = request.flow_shift;
  gen.audio_flow_shift = request.audio_flow_shift;
  gen.seed = static_cast<uint64_t>(request.seed);
  gen.has_seed = request.has_seed;
  // OpenAI `input_reference` -> FIRST-FRAME conditioning: OpenAI documents it as
  // the image the video STARTS FROM (image-to-video), which is exactly what a
  // frame-0 keyframe expresses. The reference modalities OpenAI has no slot for
  // enter through `metadata` instead; ParseVideoRequest has already refused the
  // combinations the pipeline forbids.
  if (!request.input_reference_bytes.empty()) {
    gen.first_frame_ppm.assign(request.input_reference_bytes.begin(),
                               request.input_reference_bytes.end());
  } else {
    gen.first_frame_path = request.input_reference_path;
  }
  gen.noise_aug = 1.0;  // pin the frame exactly (the pre-fold server's choice)
  gen.ref_video_dir = request.input_reference_video_dir;
  if (!request.input_reference_audio_bytes.empty()) {
    gen.ref_audio_wav.assign(request.input_reference_audio_bytes.begin(),
                             request.input_reference_audio_bytes.end());
  } else {
    gen.ref_audio_path = request.input_reference_audio_path;
  }
  gen.output_dir = output_dir;
  return gen;
}

VideoEngine::~VideoEngine() = default;

}  // namespace vllm::multimodal
