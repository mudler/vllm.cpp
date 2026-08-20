// vllm.cpp original: the `--model` grammar, and the one entry point behind it.
//
// vLLM is the primary oracle and defines the `org/repo` form at pin
// `5559679229`:
//   - local-or-remote decision, `weight_utils.py:345`
//   - two-phase fetch, config JSON before weights, `weight_utils.py:349-357`
//   - index-driven file selection, `weight_utils.py:472-490`
//   - first matching pattern wins, `weight_utils.py:493-496`
//   - weight format preference, `default_loader.py:167-184`
//   - cross-process download lock, `weight_utils.py:506`
//   - `--revision` and `--download-dir` as their own flags,
//     `arg_utils.py:839`, `config/model.py:183`
//
// The `org/repo:QUANT` form is the one thing vLLM does not implement, and it
// comes from the secondary oracle llama.cpp at stock tag `b10451`:
//   - repository and tag split, `common/download.h:39-42`
//
// ENG-HF-MODEL-DOWNLOAD W4, issue #1280.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vllm {
namespace transformers_utils {

// What the `--model` value turned out to be. `first match wins`, and the two
// local shapes are probed BEFORE anything opens a socket, so a network call can
// never shadow a path that exists on disk.
enum class ModelReference {
  // An existing directory. Handed back unchanged.
  kLocalDirectory,
  // An existing `.gguf` file. Handed back unchanged.
  kLocalGgufFile,
  // `org/repo`. A vLLM-shaped snapshot: config JSON first, then weights.
  kHubSnapshot,
  // `org/repo:Q4_K_M`. One GGUF file, llama.cpp's form.
  kHubGgufFile,
  // Anything else. Handed back unchanged so the existing error still fires.
  kUnrecognized,
};

// The parsed form of a `--model` value, decided WITHOUT a network call.
struct ParsedModelReference {
  ModelReference kind = ModelReference::kUnrecognized;
  std::string repo_id;  // empty unless the kind is a hub form
  std::string tag;      // the quantization tag, `kHubGgufFile` only
};

// Decide which of the five shapes `model` is. Opens no socket and reads only
// the local file system.
//
// The tag is split on the LAST colon, as llama.cpp does at
// `common/download.h:39-42 @ b10451`. A Windows path such as `C:\models\qwen`
// therefore splits into `C` and `\models\qwen`, and `C` is not a repository
// identifier, so the value falls through to `kUnrecognized` and keeps today's
// behavior. Splitting on the FIRST colon would have made the same path look
// like a repository with a tag.
ParsedModelReference ParseModelReference(const std::string& model);

struct ModelResolveOptions {
  // vLLM's own `--revision`. Applies to both hub forms. Empty means the
  // repository's default branch.
  std::string revision;
  // vLLM's own `--download-dir`. It IS the directory that holds the
  // `models--org--repo` folders, which is how `snapshot_download(cache_dir=...)`
  // reads it. Empty means the value `HfHubCacheDir()` resolves.
  std::filesystem::path download_dir;
  // Progress to standard error.
  bool verbose = false;
};

// Resolve a `--model` value to a local path the loader can open, fetching the
// checkpoint when the value names a repository.
//
// A `kHubSnapshot` returns the snapshot directory. A `kHubGgufFile` returns the
// one `.gguf` file. Every other kind returns `model` unchanged.
//
// TWO vLLM BEHAVIORS ARE MIRRORED RATHER THAN SIMPLIFIED, because skipping
// either costs real bandwidth on this project's checkpoints:
//
//  1. TWO PHASES. The configuration JSON is fetched first and the weights after
//     it (`weight_utils.py:349-357`), so a repository that is not a model at
//     all fails after a few hundred kilobytes instead of after sixty
//     gigabytes.
//  2. INDEX-DRIVEN SELECTION. When the repository holds a
//     `model.safetensors.index.json`, the exact names in its `weight_map` are
//     fetched (`weight_utils.py:472-490`) rather than everything that matches
//     `*.safetensors`. Without it the fetch also pulls duplicate-format
//     subdirectories such as `original/`, which on a published checkpoint is a
//     second complete copy of the weights.
//
// Format preference falls out of the same mechanism: the patterns
// `["*.safetensors", "*.bin"]` are tried in order and the FIRST one that
// matches the remote listing wins (`default_loader.py:167-184`,
// `weight_utils.py:493-496`).
//
// Throws std::runtime_error on any refusal, with a message that names the
// missing part.
std::string ResolveModelPath(const std::string& model,
                             const ModelResolveOptions& opts);

// The file selection, exposed so it can be gated on a listing without a socket.
// `paths` are the repository-relative paths a tree listing reported, and
// `index_weight_map` holds the values of `model.safetensors.index.json`'s
// `weight_map`, or is empty when the repository has no index.
//
// Returns the weight files to fetch, in listing order.
std::vector<std::string> SelectWeightFiles(
    const std::vector<std::string>& paths,
    const std::vector<std::string>& index_weight_map);

// True when `path` is a file at the ROOT of the repository, which is the only
// place either phase looks. vLLM lists the root non-recursively
// (`weight_utils.py:349`), so a duplicate-format subdirectory never reaches its
// pattern matching either.
bool IsRepoRootFile(const std::string& path);

}  // namespace transformers_utils
}  // namespace vllm
