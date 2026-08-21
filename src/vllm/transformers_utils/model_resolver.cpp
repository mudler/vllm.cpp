// See include/vllm/transformers_utils/model_resolver.h for the grammar this
// implements and for the vLLM `5559679229` anchors it mirrors.
#include "vllm/transformers_utils/model_resolver.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/transformers_utils/downloader.h"
#include "vllm/transformers_utils/hf_cache.h"
#include "vllm/transformers_utils/hf_hub.h"

namespace vllm {
namespace transformers_utils {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

// PHASE ONE. Everything a repository needs before anybody looks at a weight:
// the model configuration, the tokenizer, and the shard index. A repository
// that is not a model at all fails here, after a few hundred kilobytes.
// Mirrors vLLM fetching the configuration through `transformers` before
// `download_weights_from_hf` runs (`weight_utils.py:349-357`).
bool IsConfigPhaseFile(const std::string& path) {
  const fs::path p(path);
  const std::string extension = p.extension().string();
  return extension == ".json" || extension == ".txt" || extension == ".model" ||
         extension == ".jinja";
}

// PHASE TWO, in preference order, first matching pattern wins
// (`default_loader.py:167-184`, `weight_utils.py:493-496`).
const char* const kWeightExtensions[] = {".safetensors", ".bin"};

std::string ToLower(std::string text) {
  for (char& c : text) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return text;
}

// A blob file name that names one file on a case-sensitive file system and on a
// case-insensitive one alike, and that can never escape the blobs directory.
std::string FlattenPath(const std::string& path) {
  std::string flat;
  flat.reserve(path.size());
  for (const char c : path) {
    const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_';
    flat.push_back(safe ? c : '_');
  }
  return flat;
}

// WHICH NAME A BLOB GETS, and it is decided WITHOUT a request.
//
// The object identifier wins when the listing carried one that survived the
// integrity rules, because it is a content hash and two repositories that hold
// one byte sequence then share one blob.
//
// Otherwise the name is the COMMIT plus the path. The commit is what every call
// after the reference resolution names, so a commit and a repository-relative
// path already identify exactly one byte sequence, which is the property a
// content hash was wanted for. It is deliberately NOT the entity tag: a tag is
// a transport artifact that a mirror may respell without the bytes changing,
// and naming a cache file after one would have cost a `HEAD` request per file
// on a WARM cache, where the correct number of requests is zero.
std::string BlobNameFor(const HfFile& file, const std::string& commit) {
  if (!file.oid.empty()) return file.oid;
  return commit + "--" + FlattenPath(file.path);
}

// The quantization tag a GGUF file name carries: `Model-Q4_K_M.gguf` -> `q4_k_m`,
// and `Model-Q4_K_M-00001-of-00003.gguf` -> `q4_k_m`. Empty when the name
// carries none. Mirrors llama.cpp matching a tag against the file names a
// repository holds rather than against a manifest.
std::string GgufTagOf(const std::string& path) {
  const fs::path p(path);
  if (p.extension().string() != ".gguf") return std::string();
  std::string stem = p.stem().string();
  // Drop a `-00001-of-00003` shard suffix so every shard of one quantization
  // reports the same tag.
  const size_t of = ToLower(stem).rfind("-of-");
  if (of != std::string::npos) {
    const size_t dash = stem.rfind('-', of - 1);
    if (dash != std::string::npos) stem = stem.substr(0, dash);
  }
  const size_t dash = stem.rfind('-');
  if (dash == std::string::npos) return std::string();
  return ToLower(stem.substr(dash + 1));
}

std::string ReadFileToString(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::string();
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// The `weight_map` VALUES of a `model.safetensors.index.json`, de-duplicated in
// first-seen order. Empty when the file is absent or carries no map, which is
// how a single-shard repository reads.
std::vector<std::string> ReadIndexWeightMap(const fs::path& index_file) {
  std::vector<std::string> names;
  std::error_code ec;
  if (!fs::is_regular_file(index_file, ec)) return names;
  json doc;
  try {
    doc = json::parse(ReadFileToString(index_file));
  } catch (const json::exception&) {
    return names;
  }
  if (!doc.is_object() || !doc.contains("weight_map") ||
      !doc["weight_map"].is_object()) {
    return names;
  }
  std::set<std::string> seen;
  for (const auto& [tensor, shard] : doc["weight_map"].items()) {
    (void)tensor;
    if (!shard.is_string()) continue;
    const std::string name = shard.get<std::string>();
    if (seen.insert(name).second) names.push_back(name);
  }
  return names;
}

// Fetch one listed file into the cache and place its snapshot entry. Returns
// the snapshot path.
fs::path FetchOne(const HfFile& file, const std::string& commit,
                  const fs::path& repo_path, const HfDownloadOptions& dopts) {
  const fs::path blob = HfBlobPath(repo_path, BlobNameFor(file, commit));
  const fs::path final_path = HfSnapshotPath(repo_path, commit) / file.path;
  const HfFileShape shape = HfShapeForPath(file.path);

  std::error_code ec;
  const HfDownloadResult result =
      HubDownloadFile(file.url, blob, file.size, shape, dopts);
  if (dopts.verbose) {
    std::cerr << "download: " << file.path << ' '
              << (result.already_present
                      ? "already in the cache"
                      : (result.resumed ? "resumed" : "fetched"))
              << ", blob " << blob.filename().string()
              << (file.oid.empty() ? " (named by commit and path)"
                                   : " (named by object identifier)")
              << std::endl;
  }

  fs::create_directories(final_path.parent_path(), ec);
  if (!HfFinalizeSnapshotEntry(blob, final_path)) {
    throw std::runtime_error("vllm.cpp: cannot place the snapshot entry " +
                             final_path.string() + " for the cached blob " +
                             blob.string());
  }
  return final_path;
}

const HfFile* FindByPath(const std::vector<HfFile>& files,
                         const std::string& path) {
  for (const HfFile& file : files) {
    if (file.path == path) return &file;
  }
  return nullptr;
}

HfHubOptions HubOptionsFor(const ModelResolveOptions& opts) {
  HfHubOptions hub = HfHubOptionsFromEnv();
  if (!opts.download_dir.empty()) {
    // vLLM's `--download-dir` IS the directory that holds the
    // `models--org--repo` folders: it is handed to
    // `snapshot_download(cache_dir=...)` (`config/model.py:183`).
    hub.hub_dir = opts.download_dir;
  }
  return hub;
}

// `org/repo` -> the snapshot directory, fetching what the cache lacks.
std::string ResolveSnapshot(const std::string& repo_id,
                            const ModelResolveOptions& opts) {
  const HfHubOptions hub = HubOptionsFor(opts);
  const std::string commit = HubResolveCommitCached(repo_id, opts.revision, hub);
  const fs::path repo_path = HfRepoPath(hub.hub_dir, repo_id);
  if (repo_path.empty()) {
    throw std::runtime_error(
        "vllm.cpp: repository '" + repo_id +
        "' cannot be fetched because this host has no HuggingFace cache "
        "directory. Set HF_HOME, or pass --download-dir.");
  }
  const fs::path snapshot = HfSnapshotPath(repo_path, commit);

  std::error_code ec;
  if (hub.offline) {
    if (fs::is_regular_file(snapshot / "config.json", ec)) return snapshot.string();
    throw std::runtime_error(
        "vllm.cpp: HF_HUB_OFFLINE is set and the cache under " +
        snapshot.string() +
        " holds no config.json for repository '" + repo_id +
        "'. Fetch it once with HF_HUB_OFFLINE unset, or point HF_HOME at a "
        "cache that already holds it.");
  }

  // ONE lock per repository, across processes, mirroring vLLM
  // `weight_utils.py:506`. Two servers started at once against one cache must
  // not write one blob twice.
  const HfRepoLock lock(repo_path);
  HfInstallDownloadInterruptHandler();

  const std::vector<HfFile> files = HubListRepoFiles(repo_id, commit, hub);

  HfDownloadOptions dopts;
  dopts.hub = hub;
  dopts.verbose = opts.verbose;

  // PHASE ONE.
  std::vector<std::string> root_paths;
  for (const HfFile& file : files) {
    if (!IsRepoRootFile(file.path)) continue;
    root_paths.push_back(file.path);
    if (IsConfigPhaseFile(file.path)) FetchOne(file, commit, repo_path, dopts);
  }
  if (!fs::is_regular_file(snapshot / "config.json", ec)) {
    throw std::runtime_error(
        "vllm.cpp: repository '" + repo_id + "' at revision " + commit +
        " has no config.json at its root, so it is not a model checkpoint this "
        "server can load. Nothing beyond its configuration was fetched.");
  }

  // PHASE TWO.
  const std::vector<std::string> index_weight_map =
      ReadIndexWeightMap(snapshot / "model.safetensors.index.json");
  const std::vector<std::string> weights =
      SelectWeightFiles(root_paths, index_weight_map);
  if (weights.empty()) {
    throw std::runtime_error(
        "vllm.cpp: repository '" + repo_id + "' at revision " + commit +
        " holds no *.safetensors and no *.bin weight file at its root.");
  }
  for (const std::string& path : weights) {
    const HfFile* file = FindByPath(files, path);
    if (file == nullptr) {
      throw std::runtime_error(
          "vllm.cpp: repository '" + repo_id +
          "' has a model.safetensors.index.json whose weight_map names '" +
          path +
          "', and the tree listing does not hold that file. The index and the "
          "listing disagree, so the checkpoint is not usable.");
    }
    FetchOne(*file, commit, repo_path, dopts);
  }
  return snapshot.string();
}

// `org/repo:Q4_K_M` -> the one GGUF file.
std::string ResolveGgufFile(const std::string& repo_id, const std::string& tag,
                            const ModelResolveOptions& opts) {
  const HfHubOptions hub = HubOptionsFor(opts);
  const std::string commit = HubResolveCommitCached(repo_id, opts.revision, hub);
  const fs::path repo_path = HfRepoPath(hub.hub_dir, repo_id);
  if (repo_path.empty()) {
    throw std::runtime_error(
        "vllm.cpp: repository '" + repo_id +
        "' cannot be fetched because this host has no HuggingFace cache "
        "directory. Set HF_HOME, or pass --download-dir.");
  }
  const std::string wanted = ToLower(tag);

  std::error_code ec;
  if (hub.offline) {
    const fs::path snapshot = HfSnapshotPath(repo_path, commit);
    for (const fs::directory_entry& entry :
         fs::directory_iterator(snapshot, ec)) {
      if (entry.is_regular_file(ec) &&
          GgufTagOf(entry.path().filename().string()) == wanted) {
        return entry.path().string();
      }
    }
    throw std::runtime_error(
        "vllm.cpp: HF_HUB_OFFLINE is set and the cache under " +
        snapshot.string() + " holds no '" + tag + "' GGUF file for repository '" +
        repo_id + "'.");
  }

  const HfRepoLock lock(repo_path);
  HfInstallDownloadInterruptHandler();

  const std::vector<HfFile> files = HubListRepoFiles(repo_id, commit, hub);

  std::vector<const HfFile*> matches;
  std::set<std::string> offered;
  for (const HfFile& file : files) {
    const std::string file_tag = GgufTagOf(file.path);
    if (file_tag.empty()) continue;
    offered.insert(file_tag);
    if (file_tag == wanted) matches.push_back(&file);
  }
  if (matches.empty()) {
    std::string tags;
    for (const std::string& name : offered) {
      if (!tags.empty()) tags += ", ";
      tags += name;
    }
    throw std::runtime_error(
        "vllm.cpp: repository '" + repo_id + "' at revision " + commit +
        " holds no GGUF file for the tag '" + tag + "'. It holds " +
        (tags.empty() ? std::string("no GGUF file at all") : tags) + ".");
  }

  HfDownloadOptions dopts;
  dopts.hub = hub;
  dopts.verbose = opts.verbose;

  // A quantization split across shards is fetched WHOLE and the FIRST shard is
  // returned, because that is the name the GGUF reader opens and it pulls its
  // siblings in from the same directory.
  std::sort(matches.begin(), matches.end(),
            [](const HfFile* a, const HfFile* b) { return a->path < b->path; });
  fs::path first;
  for (const HfFile* file : matches) {
    const fs::path placed = FetchOne(*file, commit, repo_path, dopts);
    if (first.empty()) first = placed;
  }
  return first.string();
}

}  // namespace

bool IsRepoRootFile(const std::string& path) {
  return !path.empty() && path.find('/') == std::string::npos;
}

std::vector<std::string> SelectWeightFiles(
    const std::vector<std::string>& paths,
    const std::vector<std::string>& index_weight_map) {
  // INDEX-DRIVEN SELECTION (`weight_utils.py:472-490`). When the repository
  // ships a shard index, the exact names in its `weight_map` are the answer.
  // Pattern matching would additionally pull every duplicate-format copy of the
  // weights the repository happens to carry, which on a published checkpoint is
  // a second complete set of shards.
  if (!index_weight_map.empty()) return index_weight_map;

  for (const char* extension : kWeightExtensions) {
    std::vector<std::string> matches;
    for (const std::string& path : paths) {
      if (!IsRepoRootFile(path)) continue;
      if (fs::path(path).extension().string() == extension) {
        matches.push_back(path);
      }
    }
    // FIRST MATCHING PATTERN WINS (`weight_utils.py:493-496`). A repository
    // that ships both formats is fetched once, in safetensors.
    if (!matches.empty()) return matches;
  }
  return {};
}

ParsedModelReference ParseModelReference(const std::string& model) {
  ParsedModelReference parsed;
  if (model.empty()) return parsed;

  std::error_code ec;
  // THE LOCAL PROBES COME FIRST, and that order is the requirement rather than
  // an optimization: a network call must never shadow a path that exists on
  // disk.
  if (fs::is_directory(model, ec)) {
    parsed.kind = ModelReference::kLocalDirectory;
    return parsed;
  }
  if (fs::is_regular_file(model, ec) && fs::path(model).extension() == ".gguf") {
    parsed.kind = ModelReference::kLocalGgufFile;
    return parsed;
  }

  // Split on the LAST colon, as llama.cpp does at `common/download.h:39-42 @
  // b10451`. See the header for the Windows path this protects.
  const size_t colon = model.rfind(':');
  if (colon != std::string::npos && colon + 1 < model.size()) {
    const std::string repo = model.substr(0, colon);
    const std::string tag = model.substr(colon + 1);
    if (IsValidHfRepoId(repo)) {
      parsed.kind = ModelReference::kHubGgufFile;
      parsed.repo_id = repo;
      parsed.tag = tag;
      return parsed;
    }
  }

  if (IsValidHfRepoId(model)) {
    parsed.kind = ModelReference::kHubSnapshot;
    parsed.repo_id = model;
    return parsed;
  }
  return parsed;
}

std::string ResolveModelPath(const std::string& model,
                             const ModelResolveOptions& opts) {
  const ParsedModelReference parsed = ParseModelReference(model);
  switch (parsed.kind) {
    case ModelReference::kLocalDirectory:
    case ModelReference::kLocalGgufFile:
    case ModelReference::kUnrecognized:
      // Unchanged, and no socket. An unrecognized value reaches the loader
      // exactly as it did before this row, so its existing error still fires.
      return model;
    case ModelReference::kHubSnapshot:
      return ResolveSnapshot(parsed.repo_id, opts);
    case ModelReference::kHubGgufFile:
      return ResolveGgufFile(parsed.repo_id, parsed.tag, opts);
  }
  return model;
}

}  // namespace transformers_utils
}  // namespace vllm
