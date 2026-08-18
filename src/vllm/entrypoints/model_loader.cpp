// See include/vllm/entrypoints/model_loader.h. ORIGINAL packaging helper — the
// shared model-load + engine-stack wiring behind both the OpenAI server and the
// C ABI. Mirrors the M1.8 LLMEngine __init__ (vllm/v1/engine/llm_engine.py @
// e24d1b24) as exercised by examples/server/main.cpp and the test harness.
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_dflash_gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/weight_offloader.h"
#include "vllm/model_executor/model_loader/gguf_device_fit.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"  // deepseek4 GGUF dispatch arm
#include "vllm/model_executor/models/muse_glimmer_gguf_weights.h"  // muse-glimmer GGUF arm
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"  // SPEC-MTP I5d-pre draft load
#include "vllm/model_executor/models/qwen3_5_common.h"  // SPEC-MTP I5d KV widening
#include "vllm/model_executor/models/qwen3_dflash.h"  // SPEC-DFLASH D5 draft load
#include "vllm/transformers_utils/hf_config.h"  // SPEC-DFLASH D5 draft config
#include "vllm/platforms/interface.h"  // CurrentPlatform() — SelectQueue
#include "vllm/v1/core/kv_cache_utils.h"  // check_enough_kv_cache_memory (M4)
#include "vllm/v1/structured_output/backend_native.h"  // MakeNativeBackendFactory
#include "vllm/v1/structured_output/jump_forward.h"     // JumpForwardEnabled (SW3)
#include "vt/dtype.h"
#include "vt/tensor.h"
#if defined(VLLM_CPP_CUDA) && defined(VT_CUTLASS_NVFP4)
#include "vt/cuda/nvfp4_autotune.h"
#endif

namespace vllm::entrypoints {

namespace fs = std::filesystem;

// `architecture` is the model's registered architecture string. It is what lets
// a PARTIAL backend decline a model whose kernels it has not registered, instead
// of being selected and then failing deep inside a kernel bind. Empty means "no
// model resolved yet", which is treated as no constraint.
//
// ARCH-ONE-SURFACE ROW 8: `device` is the caller's explicit selection
// (EngineParams::device / vllm_model_params.device). kAuto keeps the
// accelerator-first probe below byte-identical; an EXPLICIT selection routes
// through LoadedEngine::ResolveExplicitDeviceType and — unlike the auto arm —
// a failure to serve the named device PROPAGATES instead of falling back to
// CPU (mirror of vLLM never substituting an explicitly named device,
// vllm/config/device.py:61-66).
namespace {

// The auto arm of the resolution below, WITHOUT creating a queue. Extracted so
// the queue selector and the load-time device-fit refusal (issue #1123) read one
// description of "which device will this model run on" rather than two that can
// drift. May throw, exactly as `CurrentPlatform()` can, and every caller keeps
// the try/catch the original code had around it.
vt::DeviceType AutoAcceleratorDeviceType(std::string_view architecture) {
  const vllm::platforms::Platform& plat = vllm::platforms::CurrentPlatform();
  const vt::DeviceType dev = plat.device_type();
  // A PARTIAL backend (Metal today: 15 of 75 ops) must be able to decline a
  // model whose kernels it has not registered. The default answer is `true`,
  // so CUDA and CPU selection is byte-unchanged.
  if (dev != vt::DeviceType::kCPU &&
      (architecture.empty() || plat.supports_model_architecture(architecture))) {
    return dev;
  }
  return vt::DeviceType::kCPU;
}

// The AUTO arm, resolved by ATTEMPTING the queue. One implementation, so
// `ResolveModelDeviceType` and `SelectQueueForModel` cannot answer differently.
//
// Asking `CurrentPlatform()` alone is not enough, and #1136 measured why. This
// arm has always fallen back to CPU when `CreateQueue()` throws — "a platform can
// be registered while CreateQueue still fails, and CPU must remain reachable" —
// so on such a box a platform query answers `kCUDA` while the load runs on the
// CPU queue. The load-time device-fit refusal reads the query, and it therefore
// refused a checkpoint by naming a device nothing was going to run on, removing a
// load that previously served on CPU. Whether `CreateQueue()` fails is knowable
// only by calling it, so it is called here, once, and the queue goes to whichever
// caller wants one.
struct AutoDeviceResolution {
  vt::DeviceType device = vt::DeviceType::kCPU;
  // Set exactly when `device != kCPU`: the queue whose creation PROVED it.
  std::optional<vt::Queue> queue;
};

AutoDeviceResolution ResolveAutoDevice(std::string_view architecture) {
  AutoDeviceResolution out;
  try {
    const vt::DeviceType dev = AutoAcceleratorDeviceType(architecture);
    if (dev != vt::DeviceType::kCPU) {
      // Order matters: `device` is set only AFTER the queue exists, so a throw
      // leaves the CPU answer rather than a device nothing can serve.
      vt::Queue q = vt::GetBackend(dev).CreateQueue();
      out.queue = q;
      out.device = dev;
    }
  } catch (const std::exception&) {
    // No usable accelerator; CPU, which is what this arm has always returned.
  }
  return out;
}

}  // namespace

vt::DeviceType ResolveModelDeviceType(std::string_view architecture,
                                      vllm::Device device) {
  if (device != vllm::Device::kAuto) {
    const vllm::platforms::Platform* named_platform =
        vllm::platforms::FindPlatformByName(vllm::DeviceName(device));
    // Propagates for an explicitly named absent device, which is the refusal
    // vllm/config/device.py:61-66 mirrors and must not be swallowed here.
    return LoadedEngine::ResolveExplicitDeviceType(
        device, named_platform == nullptr
                    ? std::nullopt
                    : std::optional{named_platform->device_type()});
  }
  AutoDeviceResolution resolved = ResolveAutoDevice(architecture);
  // The queue was created only to learn whether it CAN be created. `vt::Queue` is
  // a NON-OWNING handle (a raw `cudaStream_t`) with no destructor, so dropping the
  // value would leak the stream.
  //
  // Through the FREE `vt::DestroyQueue`, not `Backend::DestroyQueue`: that is what
  // this file's only other queue teardown does (`load_queue`, below), it is what
  // `vt/backend.h` asks of new code so device index and queue cleanup are never
  // ambient, and it adds the `Synchronize` and the handle/id clearing the method
  // does not. The CREATE side deliberately stays `GetBackend(...).CreateQueue()`,
  // because that is the call this arm has always made and switching it would move
  // the production queue-selection path onto the drop-in resource ABI — a
  // behaviour change, which this repair is not.
  if (resolved.queue.has_value()) vt::DestroyQueue(*resolved.queue);
  return resolved.device;
}

vt::Queue SelectQueueForModel(std::string_view architecture,
                              vllm::Device device) {
  if (device != vllm::Device::kAuto) {
    const vt::DeviceType resolved =
        ResolveModelDeviceType(architecture, device);
    if (resolved == vt::DeviceType::kCPU) {
      return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    }
    // No try/catch here on purpose: an explicit accelerator whose queue cannot
    // be created must FAIL the load loudly, never silently serve on CPU.
    return vt::GetBackend(resolved).CreateQueue();
  }
  // M2.2b: run the engine forward on the ACCELERATOR when one is available, so
  // (on CUDA/GB10) the fp4-resident MoE/lm_head weights hit vt::MatmulNvfp4
  // on-device instead of the CPU dequant reference.
  //
  // W0b-1 item 1 (.agents/specs/metal-mlx-reuse-study.md §3.3), closed by work
  // row M3a: this hardcoded `GetBackend(kCUDA)`, which made the engine CPU-only
  // on every non-NVIDIA accelerator no matter how complete that backend was —
  // the single line that stood between the Metal backend and running a model.
  // It now asks the PLATFORM seam, which is the tree's own answer to "which
  // device is this process running on": CurrentPlatform() walks
  // {kCUDA, kROCM, kXPU, kVULKAN, kMETAL, kTENSTORRENT, kCPU} and returns the
  // first whose backend actually probed a device
  // (src/vllm/platforms/platform.cpp:91-98), so on a CUDA box this selects
  // EXACTLY the queue the old code did, byte for byte, and on the M4 it selects
  // Metal. The try/catch stays, now inside `ResolveAutoDevice`: a platform can be
  // registered while CreateQueue still fails, and CPU must remain reachable.
  AutoDeviceResolution resolved = ResolveAutoDevice(architecture);
  if (resolved.queue.has_value()) return *resolved.queue;
  return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
}

namespace {

// --- Issue #150 load-time instrumentation -----------------------------------
// "Measure it properly, then cut it": `VT_LOAD_STATS=1` prints the wall time of
// each load phase and the bytes the load actually MOVED, so the cost of the
// weight path is a measured number instead of an inferred one. Off by default
// and read once; when off this costs two clock reads per load.
bool LoadStatsEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VT_LOAD_STATS");
    return e != nullptr && e[0] != '0';
  }();
  return enabled;
}

double SecondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

void ReportLoadPhase(const char* phase, double seconds) {
  if (!LoadStatsEnabled()) return;
  std::fprintf(stderr, "[vt load] %-14s %8.3f s\n", phase, seconds);
}

void PrintLoadBytes(const char* when) {
  const vllm::load_stats::Counters c = vllm::load_stats::Snapshot();
  const double gib = 1024.0 * 1024.0 * 1024.0;
  std::fprintf(stderr,
               "[vt load] bytes@%-9s host_copy=%.3f GiB borrowed=%.3f GiB "
               "device_upload=%.3f GiB\n",
               when, static_cast<double>(c.host_copy_bytes) / gib,
               static_cast<double>(c.borrowed_bytes) / gib,
               static_cast<double>(c.device_upload_bytes) / gib);
}

void ReportLoadBytes() {
  if (!LoadStatsEnabled()) return;
  PrintLoadBytes("load-end");
  // The device uploads are LAZY -- ResidentWeight runs at first forward use,
  // after this function returns -- so the load-end snapshot always reads
  // device_upload=0. Print the final totals at exit as well, which is where the
  // "bytes moved by this process" question is actually answered. Registered
  // once; std::atexit handlers cannot take an argument, hence the wrapper.
  static const bool once = [] {
    std::atexit([] { PrintLoadBytes("exit"); });
    return true;
  }();
  (void)once;
}

bool DirectDeviceLoadRequested() {
  const char* release = std::getenv("VT_RELEASE_HOST_WEIGHTS");
  if (release != nullptr && release[0] == '0') return false;
  const char* direct = std::getenv("VT_DIRECT_DEVICE_LOAD");
  return direct == nullptr || direct[0] != '0';
}

std::vector<vllm::SafetensorsFile> LoadShards(const std::string& model_dir) {
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(model_dir)) {
    if (e.is_regular_file() && e.path().extension() == ".safetensors") {
      paths.push_back(e.path().string());
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("no *.safetensors shards found in " + model_dir);
  }
  std::sort(paths.begin(), paths.end());
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(paths.size());
  for (const std::string& p : paths) {
    shards.push_back(vllm::SafetensorsFile::Open(p));
  }
  return shards;
}

#if defined(VLLM_CPP_CUDA) && defined(VT_CUTLASS_NVFP4)
bool EnvironmentEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr || value[0] != '0';
}
#endif

// ── SPEC-DFLASH D5: separate DFlash draft-checkpoint load ────────────────────
// Resolve the DFlash draft path: a local directory (with config.json) is used
// as-is; an HF repo id ("z-lab/Qwen3.6-27B-DFlash") resolves to the newest
// ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/ dir.
// True when `path` names a DFlash draft packaged as a single `dflash`-arch GGUF
// rather than a safetensors directory (SPEC-DFLASH-GGUF GD3). Checked before the
// config.json probe because a .gguf file has no config.json and would otherwise
// fall through to the HF-cache search and be reported as "not found".
bool IsDflashGgufDraft(const std::string& path) {
  std::error_code ec;
  return fs::is_regular_file(path, ec) &&
         fs::path(path).extension() == ".gguf";
}

std::string ResolveDflashDraftDir(const std::string& path) {
  std::error_code ec;
  if (IsDflashGgufDraft(path)) return path;
  if (fs::exists(fs::path(path) / "config.json", ec)) return path;
  // HF repo id -> local cache snapshot.
  std::string slug = "models--";
  for (char c : path) slug.push_back(c == '/' ? '-' : c);
  // "z-lab/Qwen3.6-27B-DFlash" -> "models--z-lab--Qwen3.6-27B-DFlash" (each '/'
  // becomes '--'): the single replace above turned '/' into one '-', so redo it
  // as the HF two-dash convention.
  slug.clear();
  slug = "models--";
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] == '/') slug += "--";
    else slug.push_back(path[i]);
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) return path;
  fs::path snaps = fs::path(home) / ".cache/huggingface/hub" / slug / "snapshots";
  if (!fs::is_directory(snaps, ec)) return path;
  std::string best;
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "config.json", ec)) best = e.path().string();
  return best.empty() ? path : best;
}

// Read a named BF16 tensor from safetensors shards into a host OwnedTensor
// (mirrors the D2/D3 parity harness LoadTargetBf16). `nk` marks the torch
// [N=out,K=in] Linear orientation for vt::MatmulBT (lm_head); false for the
// embed lookup table.
vllm::OwnedTensor LoadNamedBf16(const std::vector<vllm::SafetensorsFile>& shards,
                               const std::string& name, bool nk) {
  for (const vllm::SafetensorsFile& s : shards) {
    for (const std::string& n : s.Names()) {
      if (n != name) continue;
      const vllm::StTensor& t = s.Get(name);
      if (t.dtype != "BF16") {
        throw std::runtime_error("dflash: target tensor " + name +
                                 " is not BF16 (got " + t.dtype + ")");
      }
      vllm::OwnedTensor out;
      out.dtype = vt::DType::kBF16;
      out.rank = static_cast<int>(t.shape.size());
      out.nk = nk;
      for (int i = 0; i < out.rank; ++i)
        out.shape[i] = t.shape[static_cast<size_t>(i)];
      out.bytes.resize(t.nbytes);
      std::memcpy(out.bytes.data(), t.data, t.nbytes);
      return out;
    }
  }
  return vllm::OwnedTensor{};
}

// Build the DFlash draft HfConfig from the draft config.json (the real nested
// {block_size, dflash_config:{mask_token_id,target_layer_ids}, layer_types,...}).
// Mirrors the D3 parity harness MakeConfig; kept manual (not LoadHfConfig) so the
// DFlashDraftModel architecture / nested dflash_config parse deterministically.
vllm::HfConfig MakeDflashDraftConfig(const nlohmann::json& c) {
  vllm::HfConfig cfg;
  cfg.hidden_size = c.at("hidden_size").get<int64_t>();
  cfg.num_attention_heads = c.at("num_attention_heads").get<int64_t>();
  cfg.num_key_value_heads = c.at("num_key_value_heads").get<int64_t>();
  cfg.head_dim = c.at("head_dim").get<int64_t>();
  cfg.rotary_dim = cfg.head_dim;
  cfg.rope_theta = c.at("rope_theta").get<double>();
  cfg.intermediate_size = c.at("intermediate_size").get<int64_t>();
  cfg.vocab_size = c.at("vocab_size").get<int64_t>();
  cfg.num_hidden_layers = c.at("num_hidden_layers").get<int64_t>();
  cfg.rms_norm_eps = c.at("rms_norm_eps").get<double>();
  cfg.sliding_window = c.at("sliding_window").get<int64_t>();
  cfg.layer_types = c.at("layer_types").get<std::vector<std::string>>();
  cfg.raw = nlohmann::json::object();
  cfg.raw["dflash_config"] = c.at("dflash_config");
  cfg.raw["block_size"] = c.at("block_size");
  return cfg;
}

// SPEC-DFLASH-GGUF B1: WHERE the draft's SHARED bf16 embed_tokens + lm_head
// come from.
//
// A DFlash draft owns neither. It runs the TARGET's embedding table and the
// TARGET's lm_head over its own hidden states, which is why the z-lab
// safetensors checkpoint and llama.cpp's DFLASH GGUF arch both omit them. Until
// axis B that sharing was expressed as a `const std::vector<SafetensorsFile>&`
// parameter on LoadDflashDraft, and THAT TYPE was the axis-B blocker: a GGUF
// target has no shards to point at, so the whole feature was refused in
// FromModelDir's GGUF branch. This is the same seam re-expressed as a SOURCE,
// which makes every line of draft-side loading identical for both containers.
//
// It holds a non-owning pointer to whichever the caller has: both live inside
// FromModelDir for the duration of the load.
class SharedHeadSource {
 public:
  explicit SharedHeadSource(const std::vector<vllm::SafetensorsFile>* shards)
      : shards_(shards) {}
  explicit SharedHeadSource(const vllm::GgufFile* gguf) : gguf_(gguf) {}

  // Fill the draft's two shared tensors. Both arms produce the SAME thing: bf16
  // `[vocab, H]` with nk=false for the gather table and the same `[vocab, H]`
  // with nk=true for the MatmulBT head. Throws naming the source on absence.
  void LoadInto(vllm::OwnedTensor* embed, vllm::OwnedTensor* head) const {
    if (gguf_ != nullptr) {
      vllm::LoadGgufSharedEmbedAndHeadBf16(*gguf_, embed, head);
    } else {
      *embed = LoadNamedBf16(
          *shards_, "model.language_model.embed_tokens.weight", false);
      *head = LoadNamedBf16(*shards_, "lm_head.weight", true);
    }
    if (embed->Empty() || head->Empty()) {
      throw std::runtime_error(
          "dflash: the target's bf16 embed_tokens + lm_head (which the draft "
          "SHARES) were not found in " +
          Describe());
    }
  }

  std::string Describe() const {
    return gguf_ != nullptr ? std::string("the GGUF target file")
                            : std::string("the target safetensors shards");
  }

 private:
  const std::vector<vllm::SafetensorsFile>* shards_ = nullptr;
  const vllm::GgufFile* gguf_ = nullptr;
};

// Build the DSpark draft HfConfig from its config.json (SPEC-DSPARK W5).
//
// A DSpark config differs from the DFlash one in three ways that all bite:
//   * `mask_token_id` / `target_layer_ids` sit at the TOP LEVEL, not inside a
//     nested `dflash_config`. The inherited backbone helpers read
//     `dflash_config`, exactly as upstream's _dflash_layer_causal does on a
//     DSpark config (getattr -> {} -> fall back to layer_types), so we synthesize
//     that sub-object here rather than fork the helpers.
//   * `sliding_window` may be JSON null (the 4B/8B drafts are all full-attention).
//   * `rope_theta` lives under `rope_parameters`, not at the top level.
// Speculators-format configs are translated to this shape first
// (Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig).
vllm::HfConfig MakeDsparkDraftConfig(const nlohmann::json& c) {
  vllm::HfConfig cfg;
  cfg.hidden_size = c.at("hidden_size").get<int64_t>();
  cfg.num_attention_heads = c.at("num_attention_heads").get<int64_t>();
  cfg.num_key_value_heads = c.at("num_key_value_heads").get<int64_t>();
  cfg.head_dim = c.at("head_dim").get<int64_t>();
  cfg.rotary_dim = cfg.head_dim;
  cfg.rope_theta = 10000.0;
  if (c.contains("rope_theta") && c.at("rope_theta").is_number()) {
    cfg.rope_theta = c.at("rope_theta").get<double>();
  } else if (c.contains("rope_parameters") && c.at("rope_parameters").is_object() &&
             c.at("rope_parameters").contains("rope_theta")) {
    cfg.rope_theta = c.at("rope_parameters").at("rope_theta").get<double>();
  }
  cfg.intermediate_size = c.at("intermediate_size").get<int64_t>();
  cfg.vocab_size = c.at("vocab_size").get<int64_t>();
  cfg.num_hidden_layers = c.at("num_hidden_layers").get<int64_t>();
  cfg.rms_norm_eps = c.at("rms_norm_eps").get<double>();
  if (c.contains("sliding_window") && c.at("sliding_window").is_number_integer()) {
    cfg.sliding_window = c.at("sliding_window").get<int64_t>();
  }
  if (c.contains("layer_types") && c.at("layer_types").is_array()) {
    cfg.layer_types = c.at("layer_types").get<std::vector<std::string>>();
  }
  cfg.raw = c;
  // Synthesize the nested dflash_config the inherited backbone reads.
  nlohmann::json dflash_config = nlohmann::json::object();
  if (c.contains("mask_token_id")) dflash_config["mask_token_id"] = c.at("mask_token_id");
  if (c.contains("target_layer_ids")) {
    dflash_config["target_layer_ids"] = c.at("target_layer_ids");
  }
  cfg.raw["dflash_config"] = dflash_config;
  return cfg;
}

// SPEC-DSPARK-QWEN3-ROUTING (#1193): the two keys upstream classifies a DSpark
// draft by — `architectures` and `model_type` (speculative.py:882-887 and
// :934-944 @ 555967922, plus vllm-project/vllm#52197). Nothing else is read.
struct DsparkDraftIdentity {
  std::vector<std::string> architectures;
  std::string model_type;
};

// Read them off the draft's config.json, or nullopt when there is no config.json
// to read. A GGUF draft (ResolveDflashDraftDir hands back the .gguf file itself)
// and an HF repo id that is not in the local cache both land there, and both
// already have their own precise error further down the load; refusing them HERE
// would replace "draft checkpoint not found" with a classification failure.
std::optional<DsparkDraftIdentity> ReadDsparkDraftIdentity(const std::string& path) {
  std::error_code ec;
  const fs::path cfg = fs::path(ResolveDflashDraftDir(path)) / "config.json";
  if (!fs::exists(cfg, ec)) return std::nullopt;
  std::ifstream f(cfg.string());
  nlohmann::json doc;
  try {
    f >> doc;
  } catch (const nlohmann::json::exception&) {
    return std::nullopt;  // LoadDsparkDraft parses it again and reports this
  }
  if (!doc.is_object()) return std::nullopt;
  // Classify the SAME document LoadDsparkDraft will load: the speculators layout
  // carries no top-level `architectures`, and its translation writes
  // ["Qwen3DSparkModel"] (qwen3_dspark.cpp, update_dspark).
  if (vllm::Qwen3DSparkModel::IsSpeculatorsDsparkConfig(doc)) {
    doc = vllm::Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(doc);
  }
  DsparkDraftIdentity id;
  if (doc.contains("architectures") && doc.at("architectures").is_array()) {
    for (const nlohmann::json& a : doc.at("architectures")) {
      if (a.is_string()) id.architectures.push_back(a.get<std::string>());
    }
  }
  // A config that DECLARES no architecture is not classified at all. Upstream
  // reads the key off a HuggingFace `ModelConfig`, where an absent key is `[]`,
  // and its catch-all would send that empty list to DeepSeek-V4. Refusing on it
  // here would refuse a draft on the ABSENCE of evidence, and the native
  // `deepseek-ai/dspark_qwen3_*_block7` layouts have not been read on this host
  // to confirm they declare it. The narrowing is deliberate and is recorded
  // under `## Owed` in .agents/specs/dspark-qwen3-routing.md.
  if (id.architectures.empty()) return std::nullopt;
  if (doc.contains("model_type") && doc.at("model_type").is_string()) {
    id.model_type = doc.at("model_type").get<std::string>();
  }
  return id;
}

// The two DSpark resolution keys, read off the draft checkpoint's own
// config.json (SPEC-DSPARK-BLOCK-SIZE-GUARD, #1225).
struct DsparkDraftKeys {
  std::optional<int> n_predict = std::nullopt;
  std::optional<int> block_floor = std::nullopt;
  // The key `block_floor` was actually read from, so the refusal can name it.
  // Upstream's `dspark_block_size` unless the fallback below supplied it, which
  // on both published Qwen3 drafts is always.
  const char* block_floor_key = "dspark_block_size";
};

// Mirror of the getattr() reads upstream performs on the draft's hf_config
// before it resolves k, all @ 555967922:
//
//   * n_predict                                                :973-975
//   * the Gemma4 normalization n_predict = block_size          :957-961
//     (guarded by "Gemma4DSparkModel" in architectures, exactly as upstream
//     guards it -- it does NOT apply to a Qwen3 DSpark draft)
//   * dspark_block_size, the block floor                       :1011-1015
//
// ONE DIVERGENCE, argued in .agents/specs/dspark-block-size-guard.md section 2:
// when `dspark_block_size` is absent the floor falls back to `block_size`.
// Upstream reads only `dspark_block_size`, and that identifier occurs in no file
// of the pinned checkout except speculative.py, so it can only arrive from a
// draft config.json -- and NEITHER published Qwen3 draft carries it.
// deepseek-ai/dspark_qwen3_4b_block7 and RadixArk/Qwen3.8-27B-DSpark @ 85ef153b
// both ship `block_size: 7` with no n_predict, and the :957-961 normalization is
// Gemma4-only, so upstream accepts k=6 against a block-7 Qwen3 draft. A literal
// port would key the floor on a field no checkpoint we support sets. Our draft
// block is sized by k alone (spec_decode/dspark/speculator.h:56) and no weight
// is block-shaped, so a short k raises no shape error: it drafts a structurally
// wrong block in silence. The explicit key still wins when a checkpoint does
// carry it, so a later pin that adds it changes nothing here.
//
// Both values stay std::nullopt when the draft checkpoint is not on disk. That
// keeps ResolveSpecConfig resolving a path it cannot read exactly as it did
// before this change; LoadDsparkDraft owns the "not found" message and names the
// directory it looked in.
DsparkDraftKeys ReadDsparkDraftKeys(const std::optional<std::string>& draft_model_path) {
  DsparkDraftKeys keys;
  if (!draft_model_path.has_value()) return keys;
  const std::string draft_dir = ResolveDflashDraftDir(*draft_model_path);
  std::error_code ec;
  const fs::path config_path = fs::path(draft_dir) / "config.json";
  if (!fs::exists(config_path, ec)) return keys;

  nlohmann::json cj;
  try {
    std::ifstream cf(config_path.string());
    cf >> cj;
  } catch (const std::exception&) {
    return keys;  // LoadDsparkDraft re-reads it and reports the parse failure.
  }
  if (!cj.is_object()) return keys;
  // Read through the SAME normalized shape LoadDsparkDraft loads from, so both
  // published config layouts resolve identically.
  if (vllm::Qwen3DSparkModel::IsSpeculatorsDsparkConfig(cj)) {
    cj = vllm::Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(cj);
  }

  const auto read_int = [&cj](const char* key) -> std::optional<int> {
    if (cj.contains(key) && cj.at(key).is_number_integer()) {
      return cj.at(key).get<int>();
    }
    return std::nullopt;
  };

  keys.n_predict = read_int("n_predict");
  if (!keys.n_predict.has_value() && cj.contains("architectures") &&
      cj.at("architectures").is_array()) {
    for (const auto& arch : cj.at("architectures")) {
      if (arch.is_string() && arch.get<std::string>() == "Gemma4DSparkModel") {
        keys.n_predict = read_int("block_size");  // speculative.py:957-961
        break;
      }
    }
  }

  keys.block_floor = read_int("dspark_block_size");
  if (!keys.block_floor.has_value()) {
    keys.block_floor = read_int("block_size");  // the divergence, above
    keys.block_floor_key = "block_size";
  }
  return keys;
}

// Load a DSpark draft: the DFlash backbone plus the Markov head plus, for a
// reduced draft vocab, the d2t map (SPEC-DSPARK W5). Both published config
// layouts are accepted; the tensor layout is identical between them.
std::unique_ptr<DflashDraft> LoadDsparkDraft(const vllm::SpeculativeConfig& spec,
                                             const SharedHeadSource& shared) {
  if (spec.method != "dspark") return nullptr;
  if (!spec.draft_model_path.has_value()) {
    throw std::runtime_error("dspark: resolved config missing draft_model_path");
  }
  const std::string draft_dir = ResolveDflashDraftDir(*spec.draft_model_path);
  std::error_code ec;
  if (!fs::exists(fs::path(draft_dir) / "config.json", ec)) {
    throw std::runtime_error("dspark: draft checkpoint not found at " + draft_dir +
                             " (from \"" + *spec.draft_model_path + "\")");
  }
  std::ifstream cf((fs::path(draft_dir) / "config.json").string());
  nlohmann::json cj;
  cf >> cj;
  // Speculators format -> the native shape the rest of the path expects.
  if (vllm::Qwen3DSparkModel::IsSpeculatorsDsparkConfig(cj)) {
    cj = vllm::Qwen3DSparkModel::TranslateSpeculatorsDsparkConfig(cj);
  }

  auto draft = std::make_unique<DflashDraft>();
  draft->k = spec.ResolvedNumSpeculativeTokens();
  draft->config = MakeDsparkDraftConfig(cj);
  // Native Qwen3 DSpark configs default to sampling from the anchor
  // (dspark/speculator.py:50-52 getattr(..., True)); the Speculators translation
  // has already written the key explicitly with its own FALSE default.
  draft->sample_from_anchor =
      !cj.contains("sample_from_anchor") || cj.at("sample_from_anchor").get<bool>();

  const nlohmann::json& dcfg = draft->config.raw.at("dflash_config");
  if (!dcfg.contains("target_layer_ids") || !dcfg.contains("mask_token_id")) {
    throw std::runtime_error(
        "dspark: the draft config must carry target_layer_ids and mask_token_id");
  }
  const int64_t num_taps = static_cast<int64_t>(dcfg.at("target_layer_ids").size());
  const int32_t mask_id = dcfg.at("mask_token_id").get<int32_t>();

  std::vector<vllm::SafetensorsFile> dshards = LoadShards(draft_dir);
  draft->dspark = std::make_unique<vllm::Qwen3DSparkWeights>(
      vllm::LoadQwen3DSpark(dshards, draft->config, num_taps, mask_id));

  // A DSpark checkpoint usually SHIPS embed_tokens + lm_head (both published
  // families do), unlike the z-lab DFlash draft. Share the target's ONLY when the
  // draft omits them, which is what load_dspark_model's _should_share decides
  // (dspark/utils.py:56-73) -- overwriting a shipped head would silently swap the
  // draft's own (possibly reduced-vocab) output layer for the target's.
  if (draft->dspark->backbone.embed_tokens.Empty() ||
      draft->dspark->backbone.lm_head.Empty()) {
    vllm::OwnedTensor shared_embed;
    vllm::OwnedTensor shared_lm_head;
    shared.LoadInto(&shared_embed, &shared_lm_head);
    if (draft->dspark->backbone.embed_tokens.Empty()) {
      draft->dspark->backbone.embed_tokens = std::move(shared_embed);
    }
    if (draft->dspark->backbone.lm_head.Empty()) {
      draft->dspark->backbone.lm_head = std::move(shared_lm_head);
      draft->dspark->backbone.draft_vocab_size =
          draft->dspark->backbone.lm_head.shape[0];
      draft->dspark->draft_vocab_size = draft->dspark->backbone.draft_vocab_size;
    }
  }
  return draft;
}

// Load the whole DFlash draft (layer weights + fc + norms from the draft
// checkpoint, safetensors dir or `dflash`-arch GGUF; embed_tokens + lm_head
// SHARED bf16 from the TARGET via `shared`) plus the resolved draft config + k.
// The source `shared` points at must still be alive (both are inside
// FromModelDir). Returns null when the config carries no dflash draft path.
std::unique_ptr<DflashDraft> LoadDflashDraft(
    const vllm::SpeculativeConfig& spec, const SharedHeadSource& shared) {
  if (spec.method != "dflash") return nullptr;
  if (!spec.draft_model_path.has_value()) {
    throw std::runtime_error("dflash: resolved config missing draft_model_path");
  }
  const std::string draft_dir = ResolveDflashDraftDir(*spec.draft_model_path);
  std::error_code ec;
  // GGUF draft (SPEC-DFLASH-GGUF axis A): config + weights both come out of the
  // single file. The target-shared embed/lm_head still come from *target_shards*
  // below, exactly as for a safetensors draft - the GGUF DFLASH arch omits
  // token_embd/output precisely because the draft shares the target's.
  auto draft = std::make_unique<DflashDraft>();
  draft->k = spec.ResolvedNumSpeculativeTokens();
  int64_t num_taps = 0;
  int32_t mask_id = -1;
  const char* source_kind = nullptr;
  if (IsDflashGgufDraft(draft_dir)) {
    vllm::GgufFile dg = vllm::GgufFile::Open(draft_dir);
    draft->config = vllm::MakeDflashGgufConfig(dg);
    const nlohmann::json& dcfg = draft->config.raw.at("dflash_config");
    num_taps = static_cast<int64_t>(dcfg.at("target_layer_ids").size());
    mask_id = dcfg.at("mask_token_id").get<int32_t>();
    draft->weights =
        vllm::LoadQwen3DFlashFromGguf(dg, draft->config, num_taps, mask_id);
    source_kind = "GGUF";
  } else {
    if (!fs::exists(fs::path(draft_dir) / "config.json", ec)) {
      throw std::runtime_error("dflash: draft checkpoint not found at " +
                               draft_dir + " (from \"" +
                               *spec.draft_model_path + "\")");
    }
    std::ifstream cf((fs::path(draft_dir) / "config.json").string());
    nlohmann::json cj;
    cf >> cj;
    draft->config = MakeDflashDraftConfig(cj);
    num_taps = static_cast<int64_t>(
        cj.at("dflash_config").at("target_layer_ids").size());
    mask_id = cj.at("dflash_config").at("mask_token_id").get<int32_t>();
    std::vector<vllm::SafetensorsFile> dshards = LoadShards(draft_dir);
    draft->weights =
        vllm::LoadQwen3DFlash(dshards, draft->config, num_taps, mask_id);
    source_kind = "safetensors";
  }

  // The draft SHARES the target's embed_tokens + lm_head (bf16 in both
  // containers of the NVFP4 27B: the safetensors leaves them unquantized, and
  // the GGUF stores token_embd/output as ggml BF16 next to its NVFP4 body),
  // exactly as vLLM's skip_substrs(embed_tokens)/tie handling. Common to BOTH
  // draft sources and BOTH target containers since B1 - the source abstraction
  // is what lets the four combinations share one code path.
  shared.LoadInto(&draft->weights.embed_tokens, &draft->weights.lm_head);
  draft->weights.draft_vocab_size = draft->weights.lm_head.shape[0];
  // A DFLASH GGUF draft carries NO vocab KV and no embedding tensor (it SHARES
  // the target's), so MakeDflashGgufConfig leaves vocab_size 0 - right for the
  // config, fatal for the forward: the draft sizes its embedding lookup as
  // `{config.vocab_size, H}` (qwen3_dflash.cpp:245,477,1008,1043), so 0 is an
  // EMPTY table and the first propose throws "cuda embedding: empty table
  // (vocab 0) with nonempty ids". A safetensors draft never hit this because
  // its config.json declares vocab_size, which is why the condition is on the
  // VALUE and not on the draft source. Take the row count from the TARGET
  // tensor actually being indexed rather than from any declared number, so the
  // view and the buffer cannot disagree. Found by SPEC-DFLASH-GGUF GD4, the
  // first run that ever GENERATED through a GGUF-sourced DFlash draft; still
  // the rule at B1, where the rows now come from a GGUF target's token_embd.
  if (draft->config.vocab_size == 0) {
    draft->config.vocab_size = draft->weights.embed_tokens.shape[0];
  }
  std::cerr << "vllm.cpp: DFlash draft loaded from " << source_kind << " "
            << draft_dir << " (k=" << draft->k << ", taps=" << num_taps
            << ", mask=" << mask_id << ", vocab=" << draft->config.vocab_size
            << ", shared head from " << shared.Describe() << ")\n";
  return draft;
}

// KV-EXTERNAL-CACHE (LMCache): build the external KV connector selected by
// EngineParams::kv_transfer_config, injecting the runner's resolved
// full-attention KV geometry into its extra_config so the connector's KV_2LTD
// chunk layout matches the physical KV page exactly. Returns nullptr when no
// connector is configured (the default-off inert path). Mirrors vLLM building
// the connector from --kv-transfer-config after the KV caches exist.
std::unique_ptr<vllm::v1::kv_offload::KVConnector> BuildKvConnector(
    const EngineParams& params, const vllm::v1::GPUModelRunner& runner) {
  using namespace vllm::v1::kv_offload;
  if (!params.kv_transfer_config.has_value()) return nullptr;
  vllm::KVTransferConfig cfg = *params.kv_transfer_config;
  if (!cfg.kv_connector.has_value() || cfg.kv_connector->empty()) return nullptr;
  // kv_role is required whenever kv_connector is set; default to kBoth so a
  // caller that only names the connector still yields a valid config.
  if (!cfg.kv_role.has_value()) cfg.kv_role = vllm::KVRole::kBoth;

  // D1 SAFETY GUARD, before anything is constructed. A connector's scheduler
  // half shortcuts prefill for externally matched blocks; if its worker half
  // cannot write those bytes into THIS device's KV pages, the model would
  // attend over never-written KV and emit plausible, wrong output with no error
  // anywhere. Refuse instead. The predicate is registered per connector
  // (KVConnectorWorkerTransferFn), so admitting a future worker half is one
  // override plus one registration argument — no ladder to extend here.
  // Checked BEFORE Create() on purpose: a connector ctor can fail first for an
  // unrelated precondition, and a refusal that surfaces as somebody else's
  // error message is not actionable.
  EnsureWorkerTransferSupported(*cfg.kv_connector, runner.device().type);

  const std::vector<vllm::PagedKvCache>& kv = runner.attn_kv();
  if (kv.empty()) {
    throw std::runtime_error(
        "LMCache connector requires a full-attention KV group, but this model "
        "has none");
  }
  const int num_layers = static_cast<int>(kv.size());
  const int hidden_dim =
      static_cast<int>(kv[0].num_kv_heads * kv[0].head_size);
  const int fa_block = static_cast<int>(kv[0].block_size);
  // Inject the geometry (extra_config overrides win in CreateFromConfig). The
  // connector keys block-aligned (chunk_tokens == the full-attention block).
  cfg.kv_connector_extra_config["num_layers"] = std::to_string(num_layers);
  cfg.kv_connector_extra_config["hidden_dim"] = std::to_string(hidden_dim);
  cfg.kv_connector_extra_config["chunk_tokens"] = std::to_string(fa_block);

  KVConnectorContext ctx;
  ctx.config = &cfg;
  ctx.role = KVConnectorRole::kScheduler;
  ctx.block_size = fa_block;
  return KVConnectorFactory::Create(ctx);
}

// Top-level GGUF architecture dispatch: `general.architecture` selects the
// family's HfConfig builder. The qwen35/qwen35moe/qwen3next keys go to
// HfConfigFromGguf; a `deepseek4` file goes to DeepseekV4HfConfigFromGguf (which
// maps it onto the registered DeepseekV4ForCausalLM). Additive by construction —
// a new GGUF-loadable arch adds ONE arm here and owns its config builder in its
// own TU. Everything downstream (Resolve -> tokenizer -> Load) is arch-agnostic.
HfConfig HfConfigFromGgufDispatch(const vllm::GgufFile& gguf) {
  const vllm::GgufValue* arch = gguf.FindKv("general.architecture");
  if (arch != nullptr && arch->TypeId() == vllm::kGgufString &&
      std::get<std::string>(arch->v) == "deepseek4") {
    return vllm::DeepseekV4HfConfigFromGguf(gguf);
  }
  // The Muse Glimmer k-quant arm; its config builder recovers the query
  // pre-scale from the folded attn_q_norm and the iRoPE mask from
  // sliding_window_pattern (muse_glimmer_gguf_weights.h).
  if (vllm::IsMuseGlimmerGguf(gguf)) return vllm::MuseGlimmerHfConfigFromGguf(gguf);
  return vllm::HfConfigFromGguf(gguf);
}

}  // namespace

// Resolve the per-step token budget (max_num_batched_tokens) for chunked
// prefill. An explicit EngineParams override wins; otherwise a PER-ARCH bounded
// default that does NOT scale with max_num_seqs, so a long/many-request prefill
// is split across steps and the per-step GDN chunked-scan activation stays
// bounded regardless of concurrency (the 27B 8x1024 conc-8 OOM fix — the old
// max_model_len*max_num_seqs product ran the whole 8192-token prefill in one
// step).
//
//  * DENSE arch (27B W4A4): 2048 FLAT — mirrors vLLM's own scheduler default
//    (DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048, vllm/config/scheduler.py:42 @
//    e24d1b24). The dense prefill is expensive per token: at mnbt=8192 one
//    giant mixed step runs several full prompts' prefill eagerly and every
//    decode stream stalls behind it (TTFT ~2x, decode starved). MEASURED (27B
//    NVFP4, GB10, in1024/out128): conc32/np96 mnbt=2048 999.16 tok/s vs 8192
//    895.90 (+11.5%); the conc16/conc32 default-vs-8192 A/Bs are in
//    .agents/parity-ledger.md (2026-07-10).
//  * MoE arch (35B A3B W4A16): keep the GB10-tuned CONCURRENCY-AWARE budget —
//    8192 at high concurrency (>=32), else 4096. Its cheap A3B expert prefill
//    wants the bigger chunk: at the 35B gate (conc-64, in1024/out128)
//    mnbt=8192 is +2.7% over 4096 (itself +8.2% over 2048) — bigger prefill
//    chunks amortize per-token GEMM/attention work across the many running
//    seqs. At LOW concurrency 8192 loses pipelining, so those keep 4096.
//    Memory-safe on GB10's 119GB (35B conc-64 peak 54GB; 16384 OOMs).
//
// Invariants mirrored from SchedulerConfig.verify_max_model_len
// (vllm/config/scheduler.py:87): the budget must be >= max_num_seqs (every
// running seq needs at least one token/step). For tiny models whose whole
// workload is smaller than the default we keep the old
// (max_model_len*max_num_seqs) ceiling so no behavior changes there.
int LoadedEngine::ResolveMaxNumBatchedTokens(const EngineParams& params,
                                             int max_model_len,
                                             bool is_dense_arch) {
  const int seqs = params.max_num_seqs > 0 ? params.max_num_seqs : 8;
  if (params.max_num_batched_tokens > 0) {
    // Explicit override; still honor the >= max_num_seqs invariant.
    return std::max(params.max_num_batched_tokens, seqs);
  }
  int budget = is_dense_arch ? 2048 : (seqs >= 32 ? 8192 : 4096);
  // Never exceed the whole workload's ceiling (tiny-model no-op preservation).
  const long ceiling = static_cast<long>(max_model_len) * seqs;
  if (ceiling > 0 && static_cast<long>(budget) > ceiling) {
    budget = static_cast<int>(ceiling);
  }
  return std::max(budget, seqs);
}

bool LoadedEngine::ResolveEnablePrefixCaching(const EngineParams& params,
                                               const ModelInfo& model_info) {
  if (params.enable_prefix_caching.has_value()) {
    return *params.enable_prefix_caching;
  }
  // ModelConfig.is_prefix_caching_supported at the parity pin: generative
  // hybrid and attention-free models default OFF while ordinary decoder-only
  // models default ON. ModelInfo.has_inner_state covers the attention-free
  // family in the native registry; is_hybrid covers Qwen3.5/3.6 GDN.
  return !model_info.is_hybrid && !model_info.has_inner_state;
}

// ARCH-ONE-SURFACE ROW 8: the explicit arms of the device-selection policy —
// see the contract in model_loader.h. Ported semantics:
// vllm/config/device.py:61-66 @ 555967922 (an explicit device string is
// assigned VERBATIM — never substituted), with the loud failure upstream
// raises when the named device cannot serve (torch/worker init on an absent
// CUDA device; our analogue is the unregistered kCUDA platform,
// src/vllm/platforms/cuda.cpp Registrar — kCUDA registers only when a usable
// GPU probed).
vt::DeviceType LoadedEngine::ResolveExplicitDeviceType(
    vllm::Device requested,
    std::optional<vt::DeviceType> named_platform_type) {
  switch (requested) {
    case vllm::Device::kCPU:
      // Explicit CPU never consults the accelerator probe: even on a
      // CUDA-capable build/process this selects the CPU queue.
      return vt::DeviceType::kCPU;
    case vllm::Device::kNamedPlatform:
      if (!named_platform_type.has_value()) {
        throw std::runtime_error(
            "device 'cuda' was requested but no CUDA platform is available in "
            "this build/process (an explicitly named device is never silently "
            "replaced — mirror of vllm/config/device.py:61-66; use device=auto "
            "or device=cpu, or run a CUDA build on a machine with a usable "
            "GPU)");
      }
      return *named_platform_type;
    case vllm::Device::kAuto:
      break;  // auto resolves through the probe in SelectQueue, not here.
  }
  throw std::invalid_argument(
      "ResolveExplicitDeviceType resolves only explicit device selections "
      "(cpu/cuda); auto resolves through the accelerator-first probe");
}

bool LoadedEngine::EnsureNoneHash() {
  // Idempotent: init_none_hash just (re)assigns the NONE_HASH global.
  //
  // The seed is resolved INSIDE init_none_hash (explicit >
  // $VLLM_PREFIX_CACHING_HASH_SEED > $PYTHONHASHSEED > the fixed built-in
  // default), so block hashes are DETERMINISTIC ACROSS PROCESSES by default.
  // The previous comment here claimed the unseeded value "does not affect
  // determinism" because prefix caching is inert below one block; that is true
  // only for the sub-block case and does not generalise — every full block
  // hash chains from NONE_HASH, so a per-process-random seed makes any
  // content-addressed persisted cache score 0% on restart.
  vllm::v1::init_none_hash(vllm::v1::sha256_cbor);
  return true;
}

vllm::SchedulerConfig LoadedEngine::MakeSchedulerConfig(
    int max_model_len, int max_num_seqs, int max_num_batched_tokens,
    vllm::SchedulerPolicy policy) {
  vllm::SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  // Bounded per-step budget (chunked prefill). See ResolveMaxNumBatchedTokens.
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;
  // Scheduling policy (fcfs default; kPriority selects the priority queue).
  cfg.policy = policy;
  return cfg;
}

// The construction-time async-scheduling resolution (W3 enable-flip). Mirrors
// vLLM: async_scheduling=None resolves to True when compatible
// (vllm/config/vllm.py:990-1038) gated on the runner advertising the async
// device path, then the house VT_ASYNC_SCHED=0 rollback env force-disables it in
// the same binary. `async_scheduling` stays nullopt on MakeSchedulerConfig, so
// ResolveAsyncScheduling(runner_supports_async) yields runner_supports_async
// (when otherwise compatible).
bool LoadedEngine::ResolveAsyncEnabled(
    const vllm::SchedulerConfig& scheduler_config, bool runner_supports_async,
    bool is_pooling_model) {
  // Pooling models resolve async scheduling OFF (the mirror of vLLM disabling
  // it by default for pooling models, vllm/config/vllm.py:1068-1073) — the
  // landed is_pooling_model arm of ResolveAsyncScheduling, wired here since
  // ARCH-ONE-SURFACE ROW 6. false (every text arch) is byte-identical.
  return vllm::AsyncSchedulingEnabled(scheduler_config.ResolveAsyncScheduling(
      runner_supports_async, is_pooling_model));
}

std::unique_ptr<vllm::v1::Scheduler> LoadedEngine::MakeScheduler(
    bool async_enabled, vllm::SchedulerConfig scheduler_config,
    vllm::v1::KVCacheConfig kv_cache_config, int block_size,
    bool enable_caching,
    vllm::v1::StructuredOutputManager* structured_output_manager,
    std::optional<vllm::SpeculativeConfig> speculative_config) {
  if (async_enabled) {
    // get_scheduler_cls -> AsyncScheduler (scheduler.py:180-189). SPEC-MTP: the
    // async-scheduling draft-in-output path is deferred, so speculation forces
    // the synchronous Scheduler below — async_enabled is never true when a
    // speculative_config is present.
    return std::make_unique<vllm::v1::AsyncScheduler>(
        std::move(scheduler_config), std::move(kv_cache_config), block_size,
        enable_caching, structured_output_manager);
  }
  return std::make_unique<vllm::v1::Scheduler>(
      std::move(scheduler_config), std::move(kv_cache_config), block_size,
      enable_caching, structured_output_manager, std::move(speculative_config));
}

std::optional<vllm::SpeculativeConfig> LoadedEngine::ResolveSpecConfig(
    const EngineParams& params, const HfConfig& config) {
  if (!params.speculative_config.has_value()) {
    return std::nullopt;  // production default: no speculation.
  }
  const vllm::SpeculativeConfig& cli = *params.speculative_config;
  // SPEC-DFLASH D5: the block-diffusion drafter. num_speculative_tokens is REQUIRED
  // (= the draft block_size, e.g. 16; speculative.py raises if None) and there is no
  // n_predict-module divisibility (the drafter is non-autoregressive). The concrete
  // draft checkpoint is loaded separately (LoadDflashDraft); this only finalizes the
  // scheduler-facing config and carries the draft path forward. The extra scheduler
  // lookahead slot comes from NumLookaheadTokens() = k+1 (already coded).
  if (cli.method == "dflash") {
    if (!cli.num_speculative_tokens.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"dflash\" requires num_speculative_tokens "
          "(the draft block_size, e.g. 16)");
    }
    vllm::SpeculativeConfig cfg =
        vllm::SpeculativeConfig::ResolveDflash(*cli.num_speculative_tokens);
    cfg.draft_model_path = cli.draft_model_path;
    return cfg;
  }
  // SPEC-NGRAM (ROAD-V1-D3): the draft-FREE n-gram proposer. num_speculative_tokens
  // (k) is REQUIRED (speculative.py:1224-1234); the prompt_lookup window defaults to
  // 5/5. There is NO draft checkpoint to load and NO n_predict-module constraint —
  // the drafts come from matching the sequence's own suffix. The GDN spec verify
  // machinery (widened conv + k+1 state slots) is still needed and is reused via
  // MakeQwen3_5KVCacheSpec(num_spec>0) exactly like MTP; the fa_draft draft-KV group
  // it allocates is simply unused (ngram has no draft-model forward).
  if (cli.method == "ngram") {
    if (!cli.num_speculative_tokens.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"ngram\" requires num_speculative_tokens");
    }
    return vllm::SpeculativeConfig::ResolveNgram(*cli.num_speculative_tokens,
                                                 cli.prompt_lookup_min,
                                                 cli.prompt_lookup_max);
  }
  // SPEC-DSPARK W5: the semi-autoregressive block drafter. Like DFlash it names a
  // SEPARATE draft checkpoint and takes k from the CLI (a native Qwen3 DSpark
  // config carries no n_predict, speculative.py:973-994).
  //
  // SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225): read the draft's own n_predict and
  // block floor and PASS them. Both arguments were std::nullopt here, so
  // ResolveDspark's k >= block floor (speculative.h:179-185, from
  // speculative.py:1003-1027) reached no user and a k below the checkpoint's
  // block was accepted in silence.
  if (cli.method == "dspark") {
    const DsparkDraftKeys keys = ReadDsparkDraftKeys(cli.draft_model_path);
    // speculative.py:990-994. Kept ahead of ResolveDspark only for the case it
    // was written for -- a native Qwen3 draft, which carries no n_predict to
    // default from. With one present, :973-979 defaults k and this must not
    // pre-empt it, or the n_predict threaded above would be unreachable.
    if (!cli.num_speculative_tokens.has_value() && !keys.n_predict.has_value()) {
      throw std::invalid_argument(
          "speculative-config: method \"dspark\" requires num_speculative_tokens "
          "(a DSpark draft config carries no n_predict)");
    }
    // SPEC-DSPARK-QWEN3-ROUTING (#1193): classify the draft by its OWN config
    // before resolving anything else. Upstream picks the DSpark lane from the
    // draft's architectures and model_type (speculative.py:882-887 and :934-944
    // @ 555967922, plus vllm#52197); this engine picked it from the CLI method
    // string alone, so a `DSparkDraftModel` checkpoint loaded as a Qwen3 draft by
    // OMISSION rather than by decision, and a DeepSeek-V4 one loaded far enough
    // to fail on a missing key. This is the production caller
    // `SpeculativeConfig::IsDsparkDraft` lacked.
    if (cli.draft_model_path.has_value()) {
      const std::optional<DsparkDraftIdentity> ident =
          ReadDsparkDraftIdentity(*cli.draft_model_path);
      if (ident.has_value()) {
        std::string listed;
        for (const std::string& arch : ident->architectures) {
          if (!listed.empty()) listed += ", ";
          listed += "\"" + arch + "\"";
        }
        // Upstream's detection (speculative.py:882-887 + #52197 hunk 1). A draft
        // that fails it is precisely the set upstream's fallback rewrites into
        // the DeepSeek-V4 lane, so it is refused with that lane named.
        if (!vllm::SpeculativeConfig::IsDsparkDraft(
                *cli.draft_model_path, ident->architectures, ident->model_type)) {
          throw std::invalid_argument(
              "speculative-config: the draft checkpoint at \"" +
              *cli.draft_model_path +
              "\" does not identify as a Qwen3 or Gemma4 DSpark draft: its "
              "model id carries no \"dspark\", and its architectures [" +
              listed + "] with model_type \"" + ident->model_type +
              "\" name none of \"Qwen3DSparkModel\", \"Gemma4DSparkModel\" or "
              "the \"DSparkDraftModel\" + \"qwen3\" pair "
              "(vllm/config/speculative.py:882-887 @ 555967922 + "
              "vllm-project/vllm#52197). Upstream routes exactly this set into "
              "the DeepSeek-V4 DSpark lane (:934-944), and that lane is not "
              "implemented here: DeepseekV4Model is a stub and it needs two "
              "Sparks. Owed by row SPEC-DSPARK-QWEN3-ROUTING "
              "(.agents/specs/dspark-qwen3-routing.md).");
        }
        // Upstream's normalization (#52197 hunk 2), called for its REFUSAL. At
        // this pin `SpeculativeConfig::ResolveDsparkArchitecture` is TOTAL over
        // its two outcomes: it answers "Qwen3DSparkModel" or it throws the
        // DeepSeek-V4 refusal by name. So the returned lane is always the one
        // `LoadDsparkDraft` implements, and a `lane != "Qwen3DSparkModel"` guard
        // here would be a branch nothing can enter — dead code, which the earlier
        // shape of this call site carried and a mutation caught. When a further
        // upstream lane arrives (#52197's own context already carries a
        // `K3DSparkModel` arm absent from this pin), the returned value becomes a
        // decision and the dispatch lands WITH the lane that needs it; the
        // pin-advance item under `## Owed` in
        // .agents/specs/dspark-qwen3-routing.md carries that.
        vllm::SpeculativeConfig::ResolveDsparkArchitecture(ident->architectures,
                                                           ident->model_type);
      }
    }
    vllm::SpeculativeConfig resolved = vllm::SpeculativeConfig::ResolveDspark(
        keys.n_predict, keys.block_floor, cli.num_speculative_tokens,
        keys.block_floor_key);
    resolved.draft_model_path = cli.draft_model_path;
    return resolved;
  }
  if (cli.method != "mtp") {
    throw std::invalid_argument(
        "speculative-config: only methods \"mtp\", \"dflash\" and \"ngram\" are "
        "supported (got \"" +
        cli.method + "\")");
  }
  // mtp_num_hidden_layers from the checkpoint (text_config, default 1 — both gate
  // checkpoints). Mirrors qwen3_5_mtp.cpp NumMtpLayers.
  int64_t mtp_layers = 1;
  if (config.raw.is_object()) {
    const nlohmann::json* text = &config.raw;
    if (config.raw.contains("text_config") &&
        config.raw.at("text_config").is_object()) {
      text = &config.raw.at("text_config");
    }
    if (text->is_object()) {
      mtp_layers = text->value("mtp_num_hidden_layers", int64_t{1});
    }
  }
  // SPEC-MTP-K-GT-1 (#81): the resolved k is SERVED, at any depth. Depth above 1
  // was refused on this line between commits, because the propose path carried
  // only upstream's k=1 early exit (autoregressive/speculator.py:236-238) and
  // would have billed the user for a depth it drafted one token for. The
  // multi-step propose (MtpProposeDrafts) landed the loop behind that early
  // exit, so the refusal is GONE rather than widened.
  return vllm::SpeculativeConfig::ResolveMtp(static_cast<int>(mtp_layers),
                                             cli.num_speculative_tokens);
}

vllm::v1::KVCacheConfig LoadedEngine::MakeKVCacheMaybeSpec(
    const LoadedModel& model, const HfConfig& config, int block_size,
    int num_blocks, const std::optional<vllm::SpeculativeConfig>& spec) {
  if (spec.has_value()) {
    // Speculation is Qwen3.5/3.6-only at this pin (both gate checkpoints); build
    // the widened spec KV directly (extra GDN k+1 state slots + widened conv row
    // + the `fa_draft` full-attn group). MakeQwen3_5KVCacheSpec(num_spec>0).
    return vllm::MakeQwen3_5KVCacheSpec(config, block_size, num_blocks,
                                        spec->ResolvedNumSpeculativeTokens());
  }
  return ModelRegistry::MakeKVCache(model, config, block_size, num_blocks);
}

int LoadedEngine::ResolveNumBlocks(const EngineParams& params,
                                   const vllm::v1::KVCacheConfig& probe) {
  // 1. Explicit override wins (vLLM num_gpu_blocks_override).
  if (params.num_blocks > 0) {
    return params.num_blocks;
  }
  // 2. Absolute KV-pool budget. IGNORES gpu_memory_utilization, exactly like
  //    vLLM CacheConfig (cache.py:189). num_blocks = budget / bytes-per-block.
  if (params.kv_cache_memory_bytes > 0) {
    const int64_t bytes_per_block = vllm::v1::KVBytesPerBlock(probe);
    if (bytes_per_block <= 0) {
      throw std::runtime_error(
          "ResolveNumBlocks: model reports zero KV bytes per block; cannot size "
          "the pool from --kv-cache-memory");
    }
    const int64_t n = params.kv_cache_memory_bytes / bytes_per_block;
    if (n <= 0) {
      throw std::invalid_argument(
          "kv_cache_memory_bytes (" +
          std::to_string(params.kv_cache_memory_bytes) +
          ") is smaller than a single KV block (" +
          std::to_string(bytes_per_block) +
          " bytes); raise --kv-cache-memory or set an explicit --num-blocks");
    }
    return static_cast<int>(n);
  }
  // 3. gpu_memory_utilization profile path (ROAD-V1-MEM M3): needs a real device
  //    profile run to measure the non-KV footprint before the free-memory
  //    fraction can be turned into a block count. Until that lands, fall back to
  //    the historical default so the default path is byte-identical.
  // TODO(ROAD-V1-MEM M3): profile run -> available_kv = free*util - non_kv.
  constexpr int kFallbackNumBlocks = 256;
  // FIX-GPU-MEM-UTIL-INERT (#1165): this line is where an explicitly chosen
  // fraction gets discarded, so this is where the engine has to say so. The
  // flag is NOT refused: roadmap_v1.md:71 records the intent that it keeps
  // vLLM's exact name and fraction semantics so a published vLLM launch line
  // ports unchanged. What was wrong was accepting the value in silence, which
  // left a user believing they had sized the KV pool when they had sized
  // nothing.
  //
  // Only an EXPLICIT value warns. A default nobody set has nothing to report,
  // and a line on every start is noise rather than a warning.
  if (params.gpu_memory_utilization.has_value()) {
    std::cerr
        << "vllm.cpp: WARNING --gpu-memory-utilization "
        << *params.gpu_memory_utilization
        << " was accepted but did NOT size the KV cache.\n"
           "vllm.cpp:   The profile run that turns a free-memory fraction into "
           "a block count is not\n"
           "vllm.cpp:   implemented yet (ROAD-V1-MEM M3, "
           "https://github.com/mudler/vllm.cpp/issues/83).\n"
           "vllm.cpp:   The pool fell back to "
        << kFallbackNumBlocks
        << " blocks. To size it today, pass\n"
           "vllm.cpp:   --kv-cache-memory <bytes> for an absolute KV budget, or "
           "--num-blocks <n> for an\n"
           "vllm.cpp:   exact block count.\n";
    // Unbuffered by the time the loader's next line lands, so the notice cannot
    // be separated from the load it belongs to (same reason as the auto-fit
    // INFO line in ResolveMaxModelLen).
    std::cerr.flush();
  }
  return kFallbackNumBlocks;
}

vllm::v1::KVCacheConfig LoadedEngine::MakeKVCacheResolved(
    const LoadedModel& model, const HfConfig& config, int block_size,
    const EngineParams& params,
    const std::optional<vllm::SpeculativeConfig>& spec) {
  // The per-block byte geometry is independent of the block count, so build a
  // probe at the override-or-256 count, read its geometry to resolve the real
  // count, and only rebuild when the resolved count differs.
  const int probe_blocks = params.num_blocks > 0 ? params.num_blocks : 256;
  vllm::v1::KVCacheConfig probe =
      MakeKVCacheMaybeSpec(model, config, block_size, probe_blocks, spec);
  const int resolved = ResolveNumBlocks(params, probe);
  if (resolved == probe_blocks) {
    return probe;
  }
  return MakeKVCacheMaybeSpec(model, config, block_size, resolved, spec);
}

int LoadedEngine::ResolveMaxModelLen(const EngineParams& params,
                                     const HfConfig& config,
                                     const vllm::v1::KVCacheConfig& kv_cfg,
                                     int block_size) {
  // kv_cache_utils.py:2160-2174 @ 555967922. See model_loader.h for the two
  // arms and why this post-condition matters.
  const int64_t bytes_per_block = vllm::v1::KVBytesPerBlock(kv_cfg);
  const int64_t available =
      static_cast<int64_t>(kv_cfg.num_blocks) * bytes_per_block;

  if (params.max_model_len > 0) {
    // The caller pinned a length. Refuse if the pool cannot serve it — UNLESS
    // there is no paged KV to size at all. kv_cache_utils.py:872-878 guards the
    // whole check with `if kv_cache_spec:` for exactly this: an attention-free
    // model (and, here, a pure Mamba/GDN one, whose state is sized per sequence
    // slot rather than per block, so KVBytesPerBlock is 0) has nothing to run
    // out of, and checking it would refuse a configuration that works.
    if (bytes_per_block > 0) {
      const int64_t needed = vllm::v1::kv_memory_needed_bytes(
          params.max_model_len, block_size, bytes_per_block);
      vllm::v1::check_enough_kv_cache_memory(
          available, needed, params.max_model_len,
          vllm::v1::estimate_max_model_len(available, bytes_per_block,
                                           block_size));
    }
    return params.max_model_len;
  }

  // Unpinned: serve the checkpoint's own context, auto-fitted down to the pool.
  const int64_t derived = config.max_position_embeddings;
  if (derived <= 0) {
    // No context length in the config at all. There is nothing to fit against,
    // and it is not this function's job to invent one.
    return static_cast<int>(derived);
  }
  const int64_t fitted = vllm::v1::auto_fit_max_model_len(
      derived, available, bytes_per_block, block_size);
  if (fitted < derived) {
    // kv_cache_utils.py:2021-2027 logs the reduction. Silence here would make a
    // shortened context look like a model-config surprise later.
    std::cerr << "INFO auto-fit max_model_len: reduced from " << derived
              << " to " << fitted << " to fit the KV cache ("
              << kv_cfg.num_blocks << " blocks x " << block_size
              << " tokens). Raise --num-blocks / --kv-cache-memory for a longer"
                 " context.\n";
    std::cerr.flush();
  }
  return static_cast<int>(fitted);
}

// SPEC-MTP-K-GT-1 (#81): the in-memory mirror of FromModelDir's
// `maybe_attach_mtp`. A caller holding weights in memory had no way to supply
// the `mtp.*` draft head, so an in-memory speculative engine could only run with
// a NULL drafter -- which a depth gate must never mistake for working
// speculation. Attaching before the LoadedEngine body runs is what matters: the
// constructor asks `model_->supports_mtp_draft()` and calls BuildMtpDraft in its
// member-initialiser list, so a later attach would be too late.
std::unique_ptr<LoadedModel> AttachMtp(std::unique_ptr<LoadedModel> model,
                                       std::optional<Qwen3_5MTPWeights> mtp) {
  if (mtp.has_value()) model->AttachMtpDraftWeights(std::move(*mtp));
  return model;
}

LoadedEngine::LoadedEngine(HfConfig config, Qwen3_5MoeWeights weights,
                           tok::Tokenizer tokenizer, const EngineParams& params,
                           std::optional<Qwen3_5MTPWeights> mtp_weights)
    : LoadedEngine(std::move(config),
                   AttachMtp(MakeQwen3_5MoeLoadedModel(std::move(weights)),
                             std::move(mtp_weights)),
                   std::move(tokenizer), params) {}

LoadedEngine::LoadedEngine(HfConfig config, Qwen3_5DenseWeights weights,
                           tok::Tokenizer tokenizer, const EngineParams& params,
                           std::optional<Qwen3_5MTPWeights> mtp_weights)
    : LoadedEngine(std::move(config),
                   AttachMtp(MakeQwen3_5DenseLoadedModel(std::move(weights)),
                             std::move(mtp_weights)),
                   std::move(tokenizer), params) {}

LoadedEngine::LoadedEngine(HfConfig config,
                           std::unique_ptr<LoadedModel> model,
                           tok::Tokenizer tokenizer,
                           const EngineParams& params,
                           vt::Queue* preselected_queue,
                           std::unique_ptr<DflashDraft> dflash_draft)
    : hash_ready_(EnsureNoneHash()),
      config_(std::move(config)),
      // SPEC-MTP I5d: finalize the speculative config against the checkpoint
      // (n_predict + resolved k). nullopt on the production default path.
      resolved_spec_config_(ResolveSpecConfig(params, config_)),
      // SPEC-DFLASH D5: the separately-loaded DFlash draft (null for mtp/non-spec).
      dflash_draft_(std::move(dflash_draft)),
      model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      // #607 L2: carry the multimodal input limits onto the engine, so the ONE
      // config object every consumer asks (GetLimitPerPrompt) is the one the
      // server flags / the C ABI set. Default-constructed == the pre-L2 999.
      mm_config_(params.multimodal),
      // ROAD-V1-MEM M1: resolve the block count from the sizing knobs
      // (num_blocks override > kv_cache_memory_bytes > util fallback) against the
      // model's own per-block byte geometry. FIRST, because max_model_len_ is
      // resolved against this pool.
      kv_cfg_(MakeKVCacheResolved(
          *model_, config_, params.block_size > 0 ? params.block_size : 32,
          params, resolved_spec_config_)),
      // The serving length, checked (pinned) or auto-fitted (unpinned) against
      // kv_cfg_. See ResolveMaxModelLen.
      max_model_len_(ResolveMaxModelLen(
          params, config_, kv_cfg_,
          params.block_size > 0 ? params.block_size : 32)),
      max_num_batched_tokens_(ResolveMaxNumBatchedTokens(
          params, max_model_len_, ModelRegistry::IsDenseModel(*model_))),
      prefix_caching_enabled_(ResolveEnablePrefixCaching(
          params, model_->registration().info)),
      // ENG-SGLANG-BEHAVIOR-FLAG SW3: resolve jump-forward once (config field +
      // VT_ENABLE_JUMP_FORWARD env override). Default nullopt+no-env => false =>
      // the byte-identical decode path (jump-forward is inert until enabled).
      jump_forward_enabled_(
          vllm::v1::JumpForwardEnabled(params.enable_jump_forward)),
      // runner_ FIRST (W3): the async-scheduling flip reads
      // runner_.runner_supports_async(). SPEC-MTP I5d: when speculation is on,
      // pass the resolved config + the MTP draft (built from the retained mtp.*
      // weights, sharing the target embed/lm_head). The draft KV `fa_draft` group
      // is allocated by the runner from kv_cfg_ (empty vector here), so the loop
      // reaches it via runner-owned storage. nullopt/null on the default path.
      runner_(config_, *model_, kv_cfg_,
              preselected_queue != nullptr
                  ? *preselected_queue
                  : SelectQueueForModel(model_->registration().architecture,
                                        params.device),
              /*max_num_reqs=*/params.max_num_seqs > 0 ? params.max_num_seqs : 8,
              max_model_len_,
              /*max_num_batched_tokens=*/max_num_batched_tokens_,
              resolved_spec_config_,
              // Only the MTP method builds an in-target MTP draft; DFlash (D5)
              // loads a SEPARATE draft, wired via set_dflash_draft in the body.
              resolved_spec_config_.has_value() &&
                      resolved_spec_config_->method == "mtp" &&
                      model_->supports_mtp_draft()
                  ? model_->BuildMtpDraft(config_)
                  : nullptr,
              /*draft_kv=*/{}),
      // Resolve the enable-flip from the now-constructed runner + VT_ASYNC_SCHED,
      // then size the batch-queue depth (2 under async scheduling → depth-2
      // step_with_batch_queue; 1 otherwise). Since the 2026-07-17 flip the default
      // (no env) resolves ON (VT_ASYNC_RUNNER default ON), mirroring vLLM;
      // VT_ASYNC_RUNNER=0 / VT_ASYNC_SCHED=0 roll back to the synchronous path.
      // SPEC-MTP I5d: speculation uses the out-of-band take_draft_token_ids /
      // post_step path, which is the SYNCHRONOUS scheduler's contract; the
      // async-scheduling draft-in-output variant is deferred (spec §2.5), so a
      // configured speculator forces sync scheduling here.
      async_scheduling_enabled_(!resolved_spec_config_.has_value() &&
          ResolveAsyncEnabled(
          MakeSchedulerConfig(
              max_model_len_,
              params.max_num_seqs > 0 ? params.max_num_seqs : 8,
              max_num_batched_tokens_, params.policy),
          runner_.runner_supports_async(),
          model_->registration().info.is_pooling_model)),
      max_concurrent_batches_(MakeSchedulerConfig(
                                  max_model_len_,
                                  params.max_num_seqs > 0 ? params.max_num_seqs
                                                          : 8,
                                  max_num_batched_tokens_, params.policy)
                                  .MaxConcurrentBatches(async_scheduling_enabled_)),
      // The engine-wide structured-output manager, native backend over the
      // tokenizer (upstream EngineCore constructs one unconditionally,
      // core.py:134). Wired into the scheduler + engine cores below so
      // response_format / C-ABI structured constraints gate decoding.
      structured_output_manager_(
          params.max_num_seqs > 0 ? params.max_num_seqs : 8,
          vllm::v1::MakeNativeBackendFactory(
              tokenizer_, static_cast<int>(config_.vocab_size))),
      // AsyncScheduler when the flip resolved ON, else the synchronous Scheduler.
      scheduler_(MakeScheduler(
          async_scheduling_enabled_,
          MakeSchedulerConfig(
              max_model_len_, params.max_num_seqs > 0 ? params.max_num_seqs : 8,
              max_num_batched_tokens_, params.policy),
          kv_cfg_, params.block_size > 0 ? params.block_size : 32,
          /*enable_caching=*/prefix_caching_enabled_,
          &structured_output_manager_, resolved_spec_config_)),
      executor_(runner_),
      // SPEC-MTP I5d: with a speculator configured, EngineCore pulls the runner's
      // out-of-band drafts each step (take_draft_token_ids -> update_draft_token_ids)
      // so the next verify step schedules them. Default false (no-op post_step).
      engine_core_(*scheduler_, executor_, &structured_output_manager_,
                   /*check_for_draft_tokens=*/resolved_spec_config_.has_value()),
      // The admission-time prompt-length check validates against the RESOLVED
      // serving length, which is what upstream's model_config.max_model_len is
      // (input_processor.py:399-401). Passing config_ alone would check against
      // the raw checkpoint context and let through prompts the pool cannot hold.
      input_processor_(tokenizer_, config_, max_model_len_),
      output_processor_(&tokenizer_),
      block_hasher_(prefix_caching_enabled_
                        ? vllm::v1::get_request_block_hasher(
                              params.block_size > 0 ? params.block_size : 32,
                              vllm::v1::sha256_cbor)
                        : nullptr),
      engine_(input_processor_, engine_core_, output_processor_, block_hasher_) {
  (void)hash_ready_;
  // issue #371: REFUSE an unservable recurrent-state budget instead of
  // allocating it. Speculation widens the Mamba/GDN state to k+1 snapshot slots
  // per sequence (runner.cpp:449-451), so a k=15 draft costs SIXTEEN times the
  // spec-off state; on a unified-memory box the resulting allocation takes the
  // machine down rather than failing, which is exactly what it did four times on
  // 2026-08-11. Upstream checks the equivalent budget up front and raises
  // (kv_cache_utils.py:751-787, with MambaSpec counting num_speculative_blocks at
  // kv_cache_interface.py:713-718); this is that check for the state term.
  //
  // An UNKNOWN budget (MemAvailable unreadable) never refuses.
  {
    const int seqs = params.max_num_seqs > 0 ? params.max_num_seqs : 8;
    const int64_t state_needed =
        vllm::v1::recurrent_state_bytes(kv_cfg_, seqs);
    const int64_t host_available = vllm::v1::host_available_memory_bytes();
    if (state_needed > 0 && host_available > 0) {
      vllm::v1::check_enough_state_memory(
          host_available, state_needed, seqs,
          resolved_spec_config_.has_value()
              ? resolved_spec_config_->ResolvedNumSpeculativeTokens()
              : 0);
    }
  }
  // SPEC-DFLASH D5: wire the separately-loaded DFlash draft into the runner's
  // verify/propose loop. Done here (after runner_ is fully constructed, before
  // WarmupKernels) so the runner holds a stable borrow of dflash_draft_ (which
  // outlives it). Null for mtp/non-spec, so this is inert on every other path.
  if (dflash_draft_ != nullptr) {
    // Both block drafters CONDITION on the target's aux multi-tap. A target
    // architecture whose forward cannot produce it yields an engine that dies on
    // the first propose with "missing target aux multi-tap" -- which is exactly
    // how the first DSpark e2e failed, against classic-dense Qwen3ForCausalLM.
    // Refuse at LOAD, by name, with the reason.
    if (!model_->supports_aux_multi_tap()) {
      const std::string method =
          dflash_draft_->dspark != nullptr ? "dspark" : "dflash";
      const std::string arch = config_.architectures.empty()
                                   ? std::string("this model")
                                   : config_.architectures.front();
      throw std::runtime_error(
          "speculative-config: method \"" + method +
          "\" needs a target architecture that captures the aux multi-tap (the "
          "residual stream at the draft's target_layer_ids); " + arch +
          " does not. Supported targets today are the Qwen3.5/3.6 dense and MoE "
          "families.");
    }
    if (dflash_draft_->dspark != nullptr) {
      // SPEC-DSPARK W5: wires the inherited backbone through set_dflash_draft
      // internally, so the shared machinery is byte-identical to the DFlash lane.
      runner_.set_dspark_draft(dflash_draft_->dspark.get(), &dflash_draft_->config,
                               dflash_draft_->k,
                               dflash_draft_->sample_from_anchor);
    } else {
      runner_.set_dflash_draft(&dflash_draft_->weights, &dflash_draft_->config,
                               dflash_draft_->k);
    }
  }
  // KV-EXTERNAL-CACHE (LMCache): build + wire the external KV connector when the
  // caller selected one via EngineParams::kv_transfer_config. Default (unset)
  // leaves kv_connector_ null and BOTH the scheduler and the runner unchanged —
  // byte-identical to the production engine. Built here (ctor body), after
  // runner_ and scheduler_ are fully constructed, so the runner's KV geometry is
  // available for the connector's chunk layout.
  kv_connector_ = BuildKvConnector(params, runner_);
  if (kv_connector_ != nullptr) {
    scheduler_->set_kv_connector(kv_connector_.get());
    runner_.set_kv_connector(kv_connector_.get());
    std::cerr << "vllm.cpp: KV external cache connector '"
              << *params.kv_transfer_config->kv_connector
              << "' wired ON (scheduler + worker)\n";
  }
  // Mirror vLLM's "Asynchronous scheduling is enabled/disabled" resolution log
  // (vllm/config/vllm.py:990-1038) so the DGX A/B can audit which arm is live.
  std::cerr << "vllm.cpp: Asynchronous scheduling is "
            << (async_scheduling_enabled_ ? "enabled" : "disabled")
            << " (max_concurrent_batches=" << max_concurrent_batches_ << ")\n";
  WarmupKernels();
}

void LoadedEngine::WarmupKernels() {
#if defined(VLLM_CPP_CUDA) && defined(VT_CUTLASS_NVFP4)
  if (!model_->uses_nvfp4_w4a4() ||
      !vllm::platforms::GetPlatform(runner_.device().type).cutlass_fp4_supported() ||
      !EnvironmentEnabled("VT_FP4_PRE_SERVE_WARMUP") ||
      !EnvironmentEnabled("VT_FP4_AUTOTUNE") ||
      !EnvironmentEnabled("VT_FP4_PLAN_CACHE")) {
    return;
  }

  int32_t dummy_token = -1;
  for (int32_t token = 0; token < tokenizer_.VocabSize(); ++token) {
    if (tokenizer_.HasToken(token) && !tokenizer_.IsSpecial(token)) {
      dummy_token = token;
      break;
    }
  }
  if (dummy_token < 0) {
    throw std::runtime_error(
        "NVFP4 pre-serve warmup could not find a non-special tokenizer token");
  }

  SamplingParams sampling;
  sampling.max_tokens = 1;
  sampling.temperature = 0.0;
  sampling.ignore_eos = true;
  sampling.PostInit();
  std::vector<int32_t> prompt(
      static_cast<size_t>(max_num_batched_tokens_), dummy_token);

  std::cerr << "vllm.cpp: warming FlashInfer-parity NVFP4 profiles at "
            << max_num_batched_tokens_ << " tokens before serving\n";
  vt::cuda::Nvfp4AutotuneWarmupScope warmup(
      static_cast<uint32_t>(max_num_batched_tokens_), runner_.device().index);
  engine_core_.add_request(std::make_unique<vllm::v1::Request>(
      "_vllm_cpp_nvfp4_warmup", std::move(prompt), std::move(sampling),
      /*arrival_time=*/0.0, /*block_hasher=*/nullptr));
  while (scheduler_->get_num_unfinished_requests() > 0) {
    (void)engine_core_.step();
  }
  if (scheduler_->has_finished_requests()) {
    (void)engine_core_.step();
  }
  if (scheduler_->get_num_unfinished_requests() != 0 ||
      scheduler_->has_finished_requests()) {
    throw std::runtime_error(
        "NVFP4 pre-serve warmup left scheduler state behind");
  }
  warmup.Complete();
#endif
}

vllm::v1::AsyncLLM& LoadedEngine::async_engine() {
  std::lock_guard<std::mutex> lock(async_engine_mutex_);
  if (async_engine_ == nullptr) {
    // Thread the resolved depth-2 batch-queue size so the async frontend runs
    // step_with_batch_queue under async scheduling (W3 enable-flip); the sync
    // default keeps max_concurrent_batches_ == 1 (depth-1 step()).
    async_engine_ = std::make_unique<vllm::v1::AsyncLLM>(
        input_processor_, *scheduler_, executor_, output_processor_,
        block_hasher_, /*shutdown_timeout_s=*/0, max_concurrent_batches_,
        &structured_output_manager_,
        // The speculative-decode flag EngineCoreProc needs to run post_step.
        // Without it every speculator's drafts were proposed and dropped on
        // this (the production CLI/server) path.
        /*check_for_draft_tokens=*/resolved_spec_config_.has_value());
  }
  return *async_engine_;
}

std::unique_ptr<LoadedEngine> LoadedEngine::FromModelDir(
    const std::string& model_dir, const EngineParams& params) {
  // ENG-RESIDENCY-CONFIG (#1110): install the host-RAM -> DISK residency config
  // FIRST — before the offloader below, before the device resolution, before any
  // path or weight operation.
  //
  // The ordering is the whole requirement, not tidiness. Each knob this config
  // feeds (`VT_GGUF_MMAP`, `VT_GGUF_PREFAULT`, `VT_MOE_EXPERT_STREAM` and its two
  // sizes) is read during weight load, and two of them cannot be taken back: the
  // expert-stream decision is cached in a function-local static, and the slot
  // store's geometry is fixed when the store is built. A config that arrived after
  // either would be silently ignored, which is why `SetWeightResidencyConfig`
  // throws when a document would CHANGE one of those two decisions rather than
  // accepting it. It throws on nothing else: a second engine in one process is legal,
  // so a late `mmap` or `prefault` (both resolved per load), a document that omits a
  // decided field, and a document that asks for what was decided are all installed,
  // and the install MERGES field by field so a partial document does not drop the
  // first engine's. It is placed ahead of `CreateWeightOffloader`
  // deliberately:
  // that call can THROW for a configured-but-unwired backend, and a document
  // carrying both tiers must still have installed its residency half first.
  //
  // Absent (the default, and every caller that predates this row) installs
  // nothing, so every knob resolves exactly as it did before.
  if (params.weight_residency.has_value() &&
      !params.weight_residency->empty()) {
    vllm::SetWeightResidencyConfig(*params.weight_residency);
    // ONE line, naming THE DOCUMENT THAT WAS INSTALLED — the fields the operator
    // set, not the values the engine will resolve. The two differ exactly when a
    // variable overrides the document, which is why the second line exists: it
    // names every variable that would WIN over a field of it, by variable and by
    // field. The environment deliberately wins (those variables exist so a
    // benchmark arm is switchable without a restart), and a document silently
    // overridden by something exported weeks ago is the one way that precedence
    // hurts.
    //
    // It does not print RESOLVED values, and ONE of the five is the reason:
    // `expert_stream` is cached on first read, so resolving it here would move that
    // decision ahead of the load — the exact ordering this block exists to hold. The
    // other four could be resolved at this point (`prefault` and `slots` outright;
    // `mmap` and `slot_bytes` need a built-in default only their caller has), so
    // printing the document rather than a mixture of asked-for and resolved values is
    // a consistency decision on top of that one constraint. An operator reading
    // `expert_stream=on` beside `VT_MOE_EXPERT_STREAM (...) OVERRIDES` is being told
    // the document said on and the variable decides.
    //
    // IT READS BACK THE INSTALLED GLOBAL, not `params`, and that is what makes the
    // line evidence that the install RAN. Measured: with the line printing from
    // `params`, the reachability mutation — deleting the `SetWeightResidencyConfig`
    // call above — left the server-level suite GREEN, because the log and the
    // install were independent statements.
    const vllm::WeightResidencyConfig installed =
        vllm::ActiveWeightResidencyConfig();
    if (!installed.empty()) {
      std::cerr << "engine: weight residency (offload_config vllm_cpp): "
                << installed.Describe() << std::endl;
      const std::string shadowed = installed.DescribeEnvOverrides();
      if (!shadowed.empty()) {
        std::cerr << "engine: weight residency: the environment OVERRIDES the "
                     "config for "
                  << shadowed << std::endl;
      }
    }
  }
  // ENG-WEIGHT-OFFLOAD W1: install the weight offloader BEFORE any weight I/O,
  // mirroring vLLM setting the process-global at
  // v1/worker/gpu_model_runner.py:939. `ModelRegistry::Prepare` reads it back
  // (our analogue of models/utils.py:824). An absent config installs the no-op,
  // which is the current engine path, so this is inert by default.
  //
  // W1 has no backend, so a config that asks for one gets the no-op and ONE
  // line saying so. A configured `cpu_offload_gb` that silently frees nothing
  // is a memory bug the operator cannot see, and an honest line is cheaper than
  // the bug report it prevents.
  {
    vllm::WeightOffloaderChoice choice = vllm::CreateWeightOffloader(
        params.offload_config.value_or(vllm::OffloadConfig{}));
    if (!choice.selected_backend_pending.empty()) {
      // A backend this build cannot construct at all.
      std::cerr << "engine: offload_config selected backend \""
                << choice.selected_backend_pending
                << "\" but this build has no offloader for it "
                   "(ENG-WEIGHT-OFFLOAD W5); NO weight is offloaded"
                << std::endl;
    } else if (choice.offloader != nullptr && choice.offloader->moves_weights()) {
      // A backend that IS constructed but that no loader consults yet. The
      // distinction matters to whoever set a budget: the first case is "not
      // built", the second is "built and not wired". Both offload nothing, and
      // saying so differently is what lets a bug report name the right one.
      std::cerr << "engine: offload_config installed " << choice.offloader->name()
                << " but no loader consults it yet "
                   "(ENG-WEIGHT-OFFLOAD W2c); NO weight is offloaded"
                << std::endl;
    }
    vllm::SetWeightOffloader(std::move(choice.offloader));
  }
  // ARCH-ONE-SURFACE ROW 8: resolve an EXPLICIT device selection up front,
  // BEFORE any path/config/weight I/O — the mirror of vLLM resolving
  // DeviceConfig at config-creation time, ahead of the model load
  // (vllm/engine/arg_utils.py:1878 builds DeviceConfig first;
  // device.py __post_init__ resolves immediately). An explicitly named absent
  // device therefore fails HERE, loudly, and is never masked by a later
  // path/tokenizer error. The result is discarded: SelectQueueForModel re-runs
  // the SAME ResolveExplicitDeviceType when it actually creates the queue, so
  // the policy has exactly one owner.
  if (params.device != vllm::Device::kAuto) {
    const vllm::platforms::Platform* named_platform =
        vllm::platforms::FindPlatformByName(vllm::DeviceName(params.device));
    (void)ResolveExplicitDeviceType(
        params.device, named_platform == nullptr
                           ? std::nullopt
                           : std::optional{named_platform->device_type()});
  }
  const fs::path dir(model_dir);

  // A single `.gguf` file: config + weights + tokenizer all come from the
  // GGUF (M0.10). The engine stack below is unchanged.
  if (fs::is_regular_file(dir) && dir.extension() == ".gguf") {
    vllm::GgufFile gguf = vllm::GgufFile::Open(model_dir);
    HfConfig config = HfConfigFromGgufDispatch(gguf);
    // Resolve before tokenizer/weight work so unsupported architecture errors
    // are deterministic and match registry.py rather than being masked by a
    // later source-specific missing-tensor/tokenizer error.
    const ModelRegistration& gguf_arch = ModelRegistry::Resolve(config);
    // Issue #1123: refuse a GGUF whose weights cannot be STAGED onto the target
    // device, here, before any weight I/O and before the tokenizer.
    //
    // `Qwen3.8-2.4T-A95B UD-Q1_0` (369.96 GiB) reached a serving state on
    // `--device cuda` on a 119.631 GiB GB10 after 26 minutes and then died on
    // the FIRST forward with `vt cuda: cudaMalloc: out of memory`. The load
    // succeeds because a keep-quant expert tower is BORROWED from this mapping
    // and costs zero anonymous bytes; the forward dies because a
    // weight-staging device copies each tower into device memory
    // (`ResidentWeight`, qwen3_5.cpp:1011 -- 276 towers of 1,275,068,416
    // bytes plus 3 of 2,818,572,288, so 335.62 GiB). Loading for 26 minutes and dying
    // mid-stream is the worst of the available behaviours.
    //
    // Placed AFTER Resolve so an unsupported-architecture error keeps its
    // priority and the error ordering this branch documents is unchanged, and
    // BEFORE the tokenizer and the weights because everything after this point
    // is the cost the refusal exists to avoid paying. The predicate lives in
    // `gguf_device_fit.h`; it decides nothing on a platform that does not stage
    // weights (every CPU load) and nothing when no budget is known.
    {
      const platforms::Platform& target = platforms::GetPlatform(
          ResolveModelDeviceType(gguf_arch.architecture, params.device));
      const DeviceWeightFit fit = CheckDeviceWeightFit(
          gguf, vt::DeviceTypeName(target.device_type()),
          target.needs_weight_staging(),
          DeviceWeightBudgetBytes(
              target.residency_policy().device_memory_total_bytes));
      if (fit.refuse) throw std::runtime_error(fit.message);
    }
    tok::Tokenizer tokenizer = tok::Tokenizer::FromGguf(gguf);
    // Dense-vs-MoE GGUF dispatch now happens through the registry: the bench
    // branch's inline `IsDenseArch` split is superseded by
    // `HfConfigFromGguf` mapping the GGUF `general.architecture` key
    // (`qwen35` dense / `qwen35moe` / `qwen3next`) onto the registered
    // architecture ID, which resolves to the owning arch TU's GGUF loader.
    // SPEC-DFLASH-GGUF B2: dflash used to be REFUSED here, because the draft
    // shares the target's bf16 embed_tokens + lm_head and LoadDflashDraft read
    // them through a safetensors-TYPED seam. B1 replaced that parameter with
    // SharedHeadSource, which serves the same two tensors out of a GGUF, so the
    // refusal has no premise left and is gone. MTP left this ladder earlier for
    // its own reason - see below.
    //
    // SPEC-MTP-GGUF: MTP over GGUF used to be refused alongside dflash, on the
    // premise that GGUF exports carry no `mtp.*` tensors. They can: llama.cpp's
    // Qwen3.5 converter emits the head under layer-indexed `nextn` names and
    // announces it with `<arch>.nextn_predict_layers`, which HfConfigFromGguf
    // reads. A GGUF that genuinely lacks the head is still refused, but now for
    // the true reason and with the fix in the message.
    if (params.speculative_config.has_value() &&
        params.speculative_config->method == "mtp" &&
        !config.raw.contains("mtp_num_hidden_layers")) {
      throw std::runtime_error(
          "mtp speculative decoding needs a GGUF exported WITH the MTP head: "
          "this file declares no <arch>.nextn_predict_layers (it was converted "
          "with --no-mtp, or predates llama.cpp's Qwen3.5 MTP support)");
    }
    std::unique_ptr<LoadedModel> model =
        ModelRegistry::Load(config, ModelSource::FromGguf(gguf));
    // SPEC-MTP-GGUF: attach the head from the SAME file, mirroring the
    // safetensors branch's maybe_attach_mtp. The GGUF is still mapped here; the
    // loader owns its dequantized copies, so nothing borrows past this scope.
    if (params.speculative_config.has_value() &&
        params.speculative_config->method == "mtp") {
      const ModelRegistration& gguf_reg = ModelRegistry::Resolve(config);
      const Qwen3_5MTPKind kind = gguf_reg.factory->is_dense_model
                                      ? Qwen3_5MTPKind::kDense
                                      : Qwen3_5MTPKind::kMoe;
      model->AttachMtpDraftWeights(vllm::LoadQwen3_5MTPFromGguf(
          gguf, config, kind, GgufLoadPolicy::FromEnv()));
    }
    // SPEC-DFLASH-GGUF B3: the axis-B wiring. Structurally the same three lines
    // as the safetensors branch's maybe_load_dflash - ResolveSpecConfig re-runs
    // on the target config inside the LoadedEngine ctor, so the draft path + k
    // are resolved here from the CLI config directly - and the ONE difference is
    // the shared-head source. The draft is loaded while `gguf` is still mapped,
    // and it copies out (its resolver owns its dequantized bf16), so nothing
    // borrows past this scope.
    // SPEC-DSPARK W5: a DSpark draft is a safetensors checkpoint at this pin.
    // A GGUF TARGET would otherwise silently produce a spec-ON engine with NO
    // draft (the propose path finds no weights and yields nothing), so refuse by
    // name. The GGUF draft axis is the DSpark analogue of SPEC-DFLASH-GGUF and is
    // tracked separately.
    if (params.speculative_config.has_value() &&
        params.speculative_config->method == "dspark") {
      throw std::invalid_argument(
          "speculative-config: method \"dspark\" needs a safetensors target at "
          "this pin (a GGUF DSpark draft/target axis is not ported yet)");
    }
    std::unique_ptr<DflashDraft> dflash;
    if (params.speculative_config.has_value() &&
        params.speculative_config->method == "dflash") {
      vllm::SpeculativeConfig resolved = vllm::SpeculativeConfig::ResolveDflash(
          params.speculative_config->ResolvedNumSpeculativeTokens());
      resolved.draft_model_path = params.speculative_config->draft_model_path;
      dflash = LoadDflashDraft(resolved, SharedHeadSource(&gguf));
    }
    return std::unique_ptr<LoadedEngine>(new LoadedEngine(
        std::move(config), std::move(model), std::move(tokenizer), params,
        /*preselected_queue=*/nullptr, std::move(dflash)));
  }

  // SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225): resolve the DSpark speculative config
  // ONCE, here, and hand the result to the draft load further down.
  //
  // The draft load used to resolve it a second time, with its own argument list,
  // and it runs BEFORE the LoadedEngine constructor reaches ResolveSpecConfig —
  // so that second copy, not this function, was the first resolution a DSpark run
  // ever met. It passed `ResolvedNumSpeculativeTokens()`
  // (`include/vllm/config/speculative.h::ResolvedNumSpeculativeTokens`), which is
  // `num_speculative_tokens.value_or(n_predict)` and therefore ZERO when the user
  // named no k, because nothing fills `n_predict` on the CLI-side config. Once the
  // block floor became reachable that refused an absent k against a k of 0, naming
  // a key the checkpoint does not carry and a number nobody typed, on the native
  // Qwen3 lane that works today. The `n_predict` default and the "requires
  // num_speculative_tokens" message were both unreachable in production for the
  // same reason, while this file's tests asserted them through ResolveSpecConfig.
  //
  // Delegating deletes the second implementation instead of repairing it: one
  // resolution, one set of messages, one place the floor is applied. The dspark
  // branch of `ResolveSpecConfig` reads nothing off the target `HfConfig` — it
  // resolves from the CLI config and the draft's own config.json — so the empty
  // config here yields exactly what the constructor's re-resolution against the
  // real one will yield.
  //
  // Placed AFTER the `.gguf` branch above, which keeps its own named refusal for
  // a GGUF target, and BEFORE every path, config, tokenizer and weight operation
  // below. A speculative length the draft cannot serve is then refused before the
  // loader spends twenty minutes mapping a target it will not get to use, which is
  // the same ordering the device resolution above exists to give.
  std::optional<vllm::SpeculativeConfig> dspark_spec;
  if (params.speculative_config.has_value() &&
      params.speculative_config->method == "dspark") {
    dspark_spec = LoadedEngine::ResolveSpecConfig(params, vllm::HfConfig{});
  }

  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    throw std::runtime_error("model path is not a directory: " + model_dir);
  }
  const std::string config_path = (dir / "config.json").string();
  const std::string tokenizer_path = (dir / "tokenizer.json").string();

  // Refuse-by-task (ARCH-ONE-SURFACE ROW 1), BEFORE the full HfConfig parse: a
  // SupportsTranscription-ONLY architecture (Parakeet CTC/RNNT/TDT) has no
  // text-generation path, so the text engine must not be built around it —
  // mirror of vLLM excluding "generate" from supported_tasks for
  // supports_transcription_only models (interfaces.py:1118). The peek is
  // deliberately narrow: only a config whose architectures RESOLVE to a
  // transcription-only registration takes this exit (its config shape — e.g.
  // hidden_size nested under encoder_config — would otherwise fail the text
  // HfConfig parse below with a misleading message); every other model, known
  // or unknown, falls through with error ordering unchanged. The C ABI routes
  // such a directory to the transcription stack before reaching here
  // (vllm_c.cpp), so this fires only for a text-only consumer (server --task
  // generate, vllm-cli, bench).
  if (const std::vector<std::string> archs =
          vllm::PeekHfArchitectures(config_path);
      !archs.empty()) {
    const ModelRegistration* peek = nullptr;
    try {
      peek = &ModelRegistry::Resolve(std::span<const std::string>(archs));
    } catch (const std::exception&) {
      peek = nullptr;  // unknown arch: the existing path owns the diagnosis
    }
    if (peek != nullptr && peek->info.supports_transcription_only) {
      throw std::runtime_error(
          "Model architecture " + std::string(peek->architecture) +
          " supports transcription only (no text generation). Use "
          "vllm_transcribe on the C ABI or the server's "
          "/v1/audio/transcriptions instead of the text-generation entry "
          "points.");
    }
  }
  HfConfig config = vllm::LoadHfConfig(config_path);
  const ModelRegistration& registration = ModelRegistry::Resolve(config);
  // ENG-WEIGHT-OFFLOAD totality guard. Refuse a configured offload against a
  // model whose loader does not consult the offloader, BEFORE any weight I/O.
  // Without this the budget would be accepted and free nothing, with no error
  // anywhere, because there is no single upload seam that could enforce the
  // obligation structurally.
  vllm::RefuseUnsupportedWeightOffload(
      params.offload_config.value_or(vllm::OffloadConfig{}),
      registration.architecture,
      registration.factory != nullptr &&
          registration.factory->supports_weight_offload);
  tok::Tokenizer tokenizer = tok::Tokenizer::FromHfJson(tokenizer_path);

  // Shared ownership so a loader may retain the mmap'd shards past the load: the
  // Qwen3.6-35B MoE loader defers its routed-expert host copies and streams them
  // per layer during PrepareMarlinResident (bounds load-phase peak PSS). The
  // deferred-expert closure holds the last reference and releases the shards once
  // the device Marlin resident is built; loaders that don't retain it drop the
  // shards when this local `shards` and the model's ModelSource go out of scope.
  const auto t_open = std::chrono::steady_clock::now();
  auto shards = std::make_shared<const std::vector<vllm::SafetensorsFile>>(
      LoadShards(model_dir));
  ReportLoadPhase("mmap+header", SecondsSince(t_open));

  // SPEC-MTP I5d-pre: when a speculative (MTP) config is set, load the `mtp.*`
  // draft weights from the SAME shards and retain them on the loaded target
  // model, so the runner has a typed path to build the draft
  // (LoadedModel::BuildMtpDraft). This runs INSIDE FromModelDir, while the local
  // `shards` shared_ptr is still alive — the dense direct-device-load path
  // releases the shards once the target is on device, so the draft tensors must
  // be materialized here before this function returns. Inert (never called) on
  // the production default where speculative_config is unset. The verify/propose
  // runner loop is I5d; this only retains the weights.
  const auto maybe_attach_mtp = [&](LoadedModel& loaded) {
    if (!params.speculative_config.has_value() ||
        params.speculative_config->method != "mtp") {
      return;
    }
    const Qwen3_5MTPKind kind = registration.factory->is_dense_model
                                    ? Qwen3_5MTPKind::kDense
                                    : Qwen3_5MTPKind::kMoe;
    loaded.AttachMtpDraftWeights(vllm::LoadQwen3_5MTP(*shards, config, kind));
  };

  // SPEC-DFLASH D5: when a dflash config is set, load the SEPARATE z-lab draft
  // checkpoint (host bf16 weights + the target-SHARED embed/lm_head read from
  // *shards, which are still alive here) into a DflashDraft bundle the engine
  // owns and wires into the runner. Null (never built) on every other path.
  const auto maybe_load_dflash = [&]() -> std::unique_ptr<DflashDraft> {
    if (!params.speculative_config.has_value()) return nullptr;
    // SPEC-DSPARK W5: the DSpark draft rides the same seam and the same bundle.
    if (params.speculative_config->method == "dspark") {
      // SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225): use the config resolved at the top
      // of this function, which is where the block floor is applied. Resolving
      // again here is what put a SECOND, differently-argued copy of the
      // resolution ahead of the constructor's; `LoadDsparkDraft` sizes the draft
      // block from this k alone, so the k it gets must be the refused-or-accepted
      // one and not a second opinion.
      return LoadDsparkDraft(*dspark_spec, SharedHeadSource(shards.get()));
    }
    if (params.speculative_config->method != "dflash") {
      return nullptr;
    }
    // ResolveSpecConfig re-runs on the target config in the LoadedEngine ctor, so
    // resolve the draft here from the CLI config (path + k) directly.
    vllm::SpeculativeConfig resolved = vllm::SpeculativeConfig::ResolveDflash(
        params.speculative_config->ResolvedNumSpeculativeTokens());
    resolved.draft_model_path = params.speculative_config->draft_model_path;
    return LoadDflashDraft(resolved, SharedHeadSource(shards.get()));
  };

  // Live architecture dispatch: consume config.architectures in order and let
  // the matched registration own the weight-name map/loader. Unknown dense
  // configs now reject instead of falling through num_experts == 0.
  //
  // ROW 7 (kimi-linear.md §20.3): a factory with `stage_on_load` (Kimi-Linear's
  // 91.5 GiB bf16-resident loader) takes the queue-selected branch below so the
  // CUDA context exists BEFORE the weights load and each tensor stages then
  // releases its host mirror (the §13 GB10 recipe). Every other arch resolves
  // this condition exactly as before — byte-identical.
  const bool queue_load =
      (registration.factory->is_dense_model && DirectDeviceLoadRequested()) ||
      registration.factory->stage_on_load;
  if (!queue_load) {
    const auto t_weights = std::chrono::steady_clock::now();
    std::unique_ptr<LoadedModel> model = ModelRegistry::Load(
        config, ModelSource::FromSafetensorsOwned(shards));
    ReportLoadPhase("weights", SecondsSince(t_weights));
    ReportLoadBytes();
    maybe_attach_mtp(*model);
    std::unique_ptr<DflashDraft> dflash = maybe_load_dflash();
    return std::unique_ptr<LoadedEngine>(new LoadedEngine(
        std::move(config), std::move(model), std::move(tokenizer), params,
        /*preselected_queue=*/nullptr, std::move(dflash)));
  }

  // Select before loading so an eligible discrete-CUDA dense loader stages each
  // completed layer to the exact queue the runner will use. If construction
  // fails before the runner takes over, destroy the selected native stream.
  vt::Queue load_queue =
      SelectQueueForModel(registration.architecture, params.device);
  try {
    const auto t_weights = std::chrono::steady_clock::now();
    std::unique_ptr<LoadedModel> model = ModelRegistry::Load(
        config, ModelSource::FromSafetensorsOwned(shards, &load_queue));
    ReportLoadPhase("weights", SecondsSince(t_weights));
    ReportLoadBytes();
    maybe_attach_mtp(*model);
    std::unique_ptr<DflashDraft> dflash = maybe_load_dflash();
    return std::unique_ptr<LoadedEngine>(new LoadedEngine(
        std::move(config), std::move(model), std::move(tokenizer), params,
        &load_queue, std::move(dflash)));
  } catch (...) {
    vt::DestroyQueue(load_queue);
    throw;
  }
}

}  // namespace vllm::entrypoints
